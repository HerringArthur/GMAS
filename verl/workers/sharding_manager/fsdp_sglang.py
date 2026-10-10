# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

import torch
import torch.distributed as dist
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.model_executor.model_runner import LocalSerializedTensor
from sglang.srt.utils import MultiprocessingSerializer
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.fsdp_utils import fsdp_version, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.utils.torch_functional import check_cuda_is_available

from .base import BaseShardingManager

# from vllm.distributed import parallel_state as sglang_ps
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _preprocess_tensor_for_update_weights(tensor: torch.Tensor):
    if isinstance(tensor, DTensor):
        return tensor.full_tensor()
    return tensor


class FSDPSGLangShardingManager(BaseShardingManager):
    @check_cuda_is_available()
    def __init__(
        self,
        module: FSDP,
        inference_engine: Engine,
        model_config,
        full_params: bool = False,
        device_mesh: DeviceMesh = None,
        offload_param: bool = False,
        weight_sync_bucket_mb: int = 0,
        cache_cpu_weights: bool = False,
    ):
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.device_mesh = device_mesh
        self.offload_param = offload_param
        if weight_sync_bucket_mb < 0:
            raise ValueError("weight_sync_bucket_mb must be nonnegative")
        self.weight_sync_bucket_bytes = int(weight_sync_bucket_mb * 1024**2)
        self.cache_cpu_weights = cache_cpu_weights
        self._cached_cpu_weights = None

        # Full params
        self.full_params = full_params
        if full_params and fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(self.module, state_dict_type=StateDictType.FULL_STATE_DICT, state_dict_config=FullStateDictConfig())
        elif fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        self.tp_size = self.device_mesh["infer_tp"].size()
        self.tp_rank = self.device_mesh["infer_tp"].get_local_rank()

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    @GPUMemoryLogger(role="FSDPSGLangShardingManager enter", logger=logger)
    def __enter__(self):
        torch.cuda.empty_cache()
        log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
        if self._cached_cpu_weights is not None:
            params = self._cached_cpu_weights
        else:
            if self.offload_param:
                load_fsdp_model_to_gpu(self.module)
            params = self.module.state_dict()
            if self.cache_cpu_weights:
                # Independent storage: FSDP state tensors can alias live actor
                # parameters. SGLang sleep discards its weights, so every wake
                # still restores them, using this snapshot until invalidation.
                params = {name: _preprocess_tensor_for_update_weights(value).detach().to("cpu", copy=True)
                          for name, value in params.items()}
                self._cached_cpu_weights = params
        log_gpu_memory_usage("After state_dict() in sharding manager memory", logger=logger)
        # Copy, not share memory
        self.update_weights(params)
        log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)

        del params
        if self.offload_param:
            offload_fsdp_model_to_cpu(self.module)
        torch.cuda.empty_cache()
        log_gpu_memory_usage("After del state_dict and empty_cache in sharding manager", logger=logger)

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="FSDPSGLangShardingManager exit", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        log_gpu_memory_usage("Before SGLang offload in sharding manager", logger=logger)
        self.release_memory()
        log_gpu_memory_usage("After SGLang offload in sharding manager", logger=logger)

        self.module.train()

        # add empty cache after each compute
        torch.cuda.empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def update_weights(self, params):
        if self.tp_rank == 0:
            self.inference_engine.resume_memory_occupation()

        device = torch.cuda.current_device()
        bucket, bucket_bytes = [], 0

        def send(flush_cache):
            # Keep GPU storage alive until the synchronous engine RPC returns.
            serialized = [MultiprocessingSerializer.serialize(tensor) for _, tensor in bucket]
            if self.tp_size == 1:
                gathered = [serialized]
            else:
                gathered = [None] * self.tp_size if self.tp_rank == 0 else None
                dist.gather_object(serialized, gathered, dst=self.device_mesh["infer_tp"].mesh.tolist()[0],
                                   group=self.device_mesh["infer_tp"].get_group())
            if self.tp_rank == 0:
                self.inference_engine.update_weights_from_tensor(
                    named_tensors=[(name, LocalSerializedTensor(values=[part[i] for part in gathered]))
                                   for i, (name, _) in enumerate(bucket)],
                    load_format=None, flush_cache=flush_cache,
                )
            # TP peers must retain their IPC tensor storage until the owner has
            # finished loading it. gather_object alone does not provide this.
            if self.tp_size > 1:
                dist.barrier(group=self.device_mesh["infer_tp"].get_group())

        for name, value in params.items():
            tensor = _preprocess_tensor_for_update_weights(value)
            size = tensor.numel() * tensor.element_size()
            if bucket and (not self.weight_sync_bucket_bytes or bucket_bytes + size > self.weight_sync_bucket_bytes):
                send(False)
                bucket, bucket_bytes = [], 0
            bucket.append((name, tensor.to(device, non_blocking=True)))
            bucket_bytes += size
        if bucket:
            send(True)

    def invalidate_weights(self):
        """Called before actor updates/checkpoint loads, including failed ones."""
        self._cached_cpu_weights = None

    def release_memory(self):
        if self.device_mesh["infer_tp"].get_local_rank() == 0:
            self.inference_engine.release_memory_occupation()

    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        if self.tp_size == 1:
            return data

        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        group = self.device_mesh["infer_tp"].get_group()

        all_gather_data_proto(data=data, process_group=group)
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]
