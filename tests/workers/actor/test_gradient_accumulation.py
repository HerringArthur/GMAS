"""CPU autograd regressions for the production actor update loop.

Load the real method and losses without importing Ray, FlashAttention or CUDA.
Only the model, DataProto wrapper and distributed collective are substituted.
Run: python -m unittest discover -s tests/workers/actor -v
"""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from tensordict import TensorDict


ROOT = Path(__file__).resolve().parents[3]
MODES = ("token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm")


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class Batch:
    def __init__(self, batch, non_tensor_batch=None, meta_info=None):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch or {}
        self.meta_info = meta_info or {}

    def __len__(self):
        return self.batch.batch_size[0]

    def __getitem__(self, selection):
        return Batch(self.batch[selection], {k: v[selection] for k, v in self.non_tensor_batch.items()}, self.meta_info)

    def select(self, batch_keys, non_tensor_batch_keys=None):
        return Batch(self.batch.select(*batch_keys),
                     {k: self.non_tensor_batch[k] for k in non_tensor_batch_keys or []}, self.meta_info)


def load_production():
    namespace = {"torch": torch, "DataProto": Batch,
                 "get_torch_device": lambda: SimpleNamespace(current_device=lambda: "cpu"),
                 "append_to_dict": lambda target, values: [target.setdefault(k, []).append(v) for k, v in values.items()]}
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for path, names in (
        ("verl/utils/torch_functional.py", {"masked_mean"}),
        ("verl/trainer/ppo/core_algos.py", {"agg_loss", "compute_policy_loss", "kl_penalty"}),
        ("verl/workers/actor/dp_actor.py", {"_loss_normalizer", "_micro_batch_loss_scale"}),
    ):
        tree = ast.parse((ROOT / path).read_text())
        nodes.extend(n for n in tree.body if getattr(n, "name", "") in names)
        if path.endswith("dp_actor.py"):
            actor = next(n for n in tree.body if getattr(n, "name", "") == "DataParallelPPOActor")
            update = next(n for n in actor.body if getattr(n, "name", "") == "update_policy")
            update.decorator_list = []
            nodes.append(update)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "<production-actor>", "exec"), namespace)
    namespace["verl_F"] = SimpleNamespace(masked_mean=namespace["masked_mean"])
    # Uneven token-based splits exercise the dynamic path, including a tail.
    namespace["rearrange_micro_batches"] = lambda batch, **kwargs: (batch.split(3), None)
    return namespace


PRODUCTION = load_production()


class TinyActor:
    def update_policy(self, batch):
        backward = torch.Tensor.backward

        def record_loss(loss, *args, **kwargs):
            self.pending_losses.append(loss.detach())
            return backward(loss, *args, **kwargs)

        with patch.object(torch.Tensor, "backward", record_loss):
            return PRODUCTION["update_policy"](self, batch)

    def __init__(self, mode, mini, micro, *, dynamic=False, multi_modal=False):
        self.config = Config(
            use_adaptive_ppo_mini_batch_size=True, ppo_mini_batch_size=999,
            ppo_micro_batch_size_per_gpu=micro, use_dynamic_bsz=dynamic,
            ppo_max_token_len_per_gpu=100, use_kl_loss=True, kl_loss_type="kl", kl_loss_coef=0.2,
            ppo_epochs=1, entropy_coeff=0.03, clip_ratio=0.2, clip_ratio_low=None,
            clip_ratio_high=None, clip_ratio_c=3.0, loss_agg_mode=mode,
        )
        self.actor_module = torch.nn.Linear(1, 2, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.actor_module.weight.copy_(torch.tensor([[0.12], [-0.08]], dtype=torch.float64))
        self.actor_optimizer = torch.optim.SGD(self.actor_module.parameters(), lr=0.03)
        self.ulysses_sequence_parallel_size = 1
        self.gradients, self.micro_sizes, self.objectives = [], [], []
        self.pending_losses = []
        self.mini = mini
        self.multi_modal = multi_modal

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy):
        self.micro_sizes.append(micro_batch["responses"].shape[0])
        length = micro_batch["responses"].shape[1]
        features = micro_batch["input_ids"][:, -length:].to(torch.float64) / 20
        if self.multi_modal:
            for row, metadata in zip(micro_batch["input_ids"], micro_batch["multi_modal_inputs"]):
                assert row[0].item() == metadata["row_id"]
        logits = self.actor_module(features.unsqueeze(-1)) / temperature
        log_distribution = logits.log_softmax(-1)
        log_prob = log_distribution.gather(-1, micro_batch["responses"].unsqueeze(-1)).squeeze(-1)
        entropy = -(log_distribution.exp() * log_distribution).sum(-1)
        return entropy, log_prob

    def _optimizer_step(self):
        grad = self.actor_module.weight.grad.detach().clone()
        if not torch.isfinite(grad).all():
            raise AssertionError(f"Nonfinite gradient: {grad}")
        self.gradients.append(grad)
        if self.pending_losses:
            self.objectives.append(torch.stack(self.pending_losses).sum())
            self.pending_losses.clear()
        self.actor_optimizer.step()
        return grad.norm()


def make_batch(rows, mini, *, multi_turn=False, multi_modal=False):
    length = 5
    positions = torch.arange(rows * (length + 2)).reshape(rows, length + 2)
    response_mask = torch.arange(length)[None, :] < (torch.arange(rows) % length + 1)[:, None]
    tensors = TensorDict({
        "input_ids": positions,
        "responses": positions[:, -length:] % 2,
        "position_ids": positions,
        "attention_mask": torch.cat([torch.ones(rows, 2, dtype=torch.bool), response_mask], dim=1),
        "old_log_probs": torch.full((rows, length), -0.69, dtype=torch.float64),
        "ref_log_prob": torch.full((rows, length), -0.72, dtype=torch.float64),
        "advantages": torch.linspace(-0.7, 0.9, rows * length, dtype=torch.float64).reshape(rows, length),
    }, batch_size=[rows])
    if multi_turn:
        tensors["loss_mask"] = tensors["attention_mask"].clone()
        tensors["loss_mask"][:, -1] = False
    metadata = {"wg_id": np.array(["role"] * rows)}
    if multi_modal:
        metadata["multi_modal_inputs"] = np.array([{"row_id": row[0].item()} for row in positions], dtype=object)
    return Batch(tensors, metadata, {"temperature": 0.9, "role/ppo_mini_batch_size": mini, "multi_turn": multi_turn})


def full_batch_reference(actor, batch):
    """One unsplit loss per optimizer step, without the new scaling helpers."""
    for start in range(0, len(batch), actor.mini):
        part = batch[start:start + actor.mini]
        tensors = dict(part.batch.items())
        tensors.update(part.non_tensor_batch)
        mask_key = "loss_mask" if batch.meta_info["multi_turn"] else "attention_mask"
        mask = tensors[mask_key][:, -tensors["responses"].shape[1]:]
        entropy, log_prob = actor._forward_micro_batch(tensors, 0.9, True)
        pg_loss, *_ = PRODUCTION["compute_policy_loss"](
            tensors["old_log_probs"], log_prob, tensors["advantages"], mask,
            cliprange=0.2, cliprange_low=0.2, cliprange_high=0.2, clip_ratio_c=3.0,
            loss_agg_mode=actor.config.loss_agg_mode,
        )
        aggregate = lambda values: PRODUCTION["agg_loss"](values, mask, actor.config.loss_agg_mode)
        loss = pg_loss - 0.03 * aggregate(entropy) + 0.2 * aggregate(log_prob - tensors["ref_log_prob"])
        actor.actor_optimizer.zero_grad()
        loss.backward()
        actor.objectives.append(loss.detach())
        actor._optimizer_step()


class GradientAccumulationTests(unittest.TestCase):
    def assert_updates_match(self, actual, expected):
        self.assertEqual(len(actual.gradients), len(expected.gradients))
        for left, right in zip(actual.gradients, expected.gradients):
            torch.testing.assert_close(left, right, rtol=2e-6, atol=2e-8)
        torch.testing.assert_close(torch.stack(actual.objectives), torch.stack(expected.objectives), rtol=2e-6, atol=2e-8)
        torch.testing.assert_close(actual.actor_module.weight, expected.actor_module.weight, rtol=2e-6, atol=2e-8)

    def test_static_splits_match_unsplit_gradients_including_tail(self):
        for mode in MODES:
            for rows, mini, micro in ((7, 7, 8), (7, 7, 4), (16, 8, 4), (18, 7, 4)):
                with self.subTest(mode=mode, rows=rows, mini=mini, micro=micro):
                    batch = make_batch(rows, mini)
                    actual, expected = TinyActor(mode, mini, micro), TinyActor(mode, mini, micro)
                    actual.update_policy(batch)
                    full_batch_reference(expected, batch)
                    self.assert_updates_match(actual, expected)
                    self.assertEqual(sum(actual.micro_sizes), rows)
                    self.assertLessEqual(max(actual.micro_sizes), micro)

    def test_dynamic_and_multimodal_splits_use_actual_masks_and_sizes(self):
        for mode in MODES:
            for dynamic, multi_modal in ((True, False), (False, True)):
                with self.subTest(mode=mode, dynamic=dynamic, multi_modal=multi_modal):
                    batch = make_batch(18, 7, multi_turn=True, multi_modal=multi_modal)
                    actual = TinyActor(mode, 7, 4, dynamic=dynamic, multi_modal=multi_modal)
                    expected = TinyActor(mode, 7, 4, multi_modal=multi_modal)
                    actual.update_policy(batch)
                    full_batch_reference(expected, batch)
                    self.assert_updates_match(actual, expected)

    def test_distributed_gradients_match_global_batch_with_unequal_token_counts(self):
        for mode in MODES[:3]:
            for sp_replicas in (1, 2):
                with self.subTest(mode=mode, sp_replicas=sp_replicas):
                    batch = make_batch(8, 8)
                    # Also exercise a rank with zero valid tokens in token-mean.
                    if mode == "token-mean":
                        batch.batch["attention_mask"][:4, -5:] = False
                    denominator = (batch.batch["attention_mask"][:, -5:].sum().item()
                                   if mode == "token-mean" else 8)
                    grads = []
                    for rank in range(2):
                        part = batch[rank * 4:(rank + 1) * 4]
                        part.meta_info = {**part.meta_info, "role/ppo_mini_batch_size": 4}
                        actual = TinyActor(mode, 4, 3)
                        with patch.object(torch.distributed, "is_initialized", return_value=True), \
                             patch.object(torch.distributed, "get_world_size", return_value=2 * sp_replicas), \
                             patch.object(torch.distributed, "all_reduce", side_effect=lambda count, **kw: count.fill_(denominator * sp_replicas)):
                            actual.update_policy(part)
                        grads.append(actual.gradients[0])
                    expected = TinyActor(mode, 8, 8)
                    full_batch_reference(expected, batch)
                    torch.testing.assert_close(torch.stack(grads).mean(0), expected.gradients[0], rtol=2e-6, atol=2e-8)

    def test_all_masked_token_batch_has_finite_zero_gradient(self):
        batch = make_batch(7, 7)
        batch.batch["attention_mask"][:, -5:] = False
        actor = TinyActor("token-mean", 7, 4)
        actor.update_policy(batch)
        torch.testing.assert_close(actor.gradients[0], torch.zeros_like(actor.gradients[0]))

    def test_invalid_sizes_fail_before_forward(self):
        for mini, micro in ((0, 4), (7, 0), (7, -1)):
            with self.subTest(mini=mini, micro=micro):
                actor = TinyActor("token-mean", mini, micro)
                with self.assertRaises(ValueError):
                    actor.update_policy(make_batch(7, mini))
                self.assertEqual(actor.micro_sizes, [])

    def test_worker_accepts_micro_batch_larger_than_normalized_mini_batch(self):
        tree = ast.parse((ROOT / "verl/workers/fsdp_workers.py").read_text())
        worker = next(n for n in tree.body if getattr(n, "name", "") == "ActorRolloutRefWorker")
        init = next(n for n in worker.body if getattr(n, "name", "") == "__init__")
        # Run the normalization block with the device-mesh boundary substituted.
        block = next(n for n in init.body if isinstance(n, ast.If) and ast.unparse(n.test) == "self._is_actor"
                     and any(isinstance(child, ast.AugAssign) for child in n.body))
        config = Config(actor=Config(ppo_mini_batch_size=56, ppo_micro_batch_size=None,
                                     ppo_micro_batch_size_per_gpu=8), rollout=Config(n=1))
        worker = SimpleNamespace(config=config, _is_actor=True, device_mesh=SimpleNamespace(size=lambda: 8),
                                 ulysses_sequence_parallel_size=1)
        exec(compile(ast.Module(body=[block], type_ignores=[]), "<worker-normalization>", "exec"), {"self": worker})
        self.assertEqual(config.actor.ppo_mini_batch_size, 7)
        self.assertEqual(config.actor.ppo_micro_batch_size_per_gpu, 8)


if __name__ == "__main__":
    unittest.main()
