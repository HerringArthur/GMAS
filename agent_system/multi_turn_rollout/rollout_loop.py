# Copyright 2025 Nanyang Technological University (NTU), Singapore
# Copyright 2025 verl-agent (GiGPO) Team
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

from contextlib import contextmanager
from collections.abc import Mapping

import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from transformers import PreTrainedTokenizer
import uuid
from agent_system.multi_turn_rollout.utils import to_list_of_dict, torch_to_numpy, filter_group_data, preprocess_batch
from agent_system.environments import EnvironmentManagerBase
from agent_system.agent import BaseOrchestra
from agent_system.agent.orchestra.router.router_orchestra import update_text_action
from typing import List, Dict, Any, Optional
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto


def _unique_worker_groups(actor_rollout_wg):
    """Return each actor rollout worker group once.

    GMAS passes a mapping from model/agent ids to worker groups.  Keeping this
    helper tolerant of a single worker group preserves the collector's use in
    non-GMAS configurations, while the identity check avoids beginning and
    ending a shared group more than once.
    """
    groups = actor_rollout_wg.values() if isinstance(actor_rollout_wg, Mapping) else (actor_rollout_wg,)
    unique_groups = []
    seen = set()
    for group in groups:
        if group is None or id(group) in seen:
            continue
        seen.add(id(group))
        if callable(getattr(group, "begin_rollout", None)) and callable(getattr(group, "end_rollout", None)):
            unique_groups.append(group)
    return unique_groups


@contextmanager
def rollout_residency_scope(actor_rollout_wg):
    """Keep selected SGLang engines resident for one complete rollout.

    Activation is lazy: begin_rollout only arms a worker, and that worker
    synchronizes weights when its first generation request arrives.  This
    prevents router branches that are never selected from consuming memory or
    synchronization time.  The finally block releases every worker that was
    armed, including when environment interaction or generation raises.
    """
    groups = _unique_worker_groups(actor_rollout_wg)
    started = []
    try:
        for group in groups:
            started.append(group)
            group.begin_rollout()
        yield
    finally:
        first_error = None
        for group in reversed(started):
            try:
                group.end_rollout()
            except BaseException as exc:  # release all groups before surfacing an error
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment

        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        # success_rate = {}
        # for key, value in success.items():
        #     success_rate[key] = np.mean(value)
        
        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    # tool_callings
                    data['tool_callings'] = tool_callings[bs]
                    # success_rate
                    # for key, value in success_rate.items():
                    #     data[key] = value
                    # pass
                    data['pass'] = success['success_rate'][bs]

                    effective_batch.append(data)
            
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """

        batch_size = len(gen_batch.batch)
        
        # Initial observations from the environment
        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))

        if effective_rollout_n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % effective_rollout_n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)

            batch = preprocess_batch(gen_batch=gen_batch, 
                                     obs=obs, 
                                     config=self.config, 
                                     tokenizer=self.tokenizer, 
                                     processor=self.processor,
                                     )

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info

            # pad to be divisible by dp_size
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            # # unpad
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid

            batch = batch.union(batch_output)
            
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            
            next_obs, rewards, dones, infos = envs.step(text_actions)

            
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            # Create reward tensor, only assign rewards for active environments
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            
            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # Update done states
            is_done = np.logical_or(is_done, dones)
                
            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break
        
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
    
    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            effective_rollout_n: int,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                effective_rollout_n=effective_rollout_n,
            )
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(batch_list=batch_list,
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                tool_callings=tool_callings, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings

    def multi_turn_loop(
            self,
            gen_batch: DataProto,
            actor_rollout_wg,
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """Collect one complete rollout while keeping used policies resident."""
        with rollout_residency_scope(actor_rollout_wg):
            return self._multi_turn_loop_impl(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                is_train=is_train,
            )

    def _multi_turn_loop_impl(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        # Determine the effective rollout_n based on training/validation mode
        if is_train:
            effective_rollout_n = self.config.env.rollout.n
            gen_batch = gen_batch.repeat(repeat_times=effective_rollout_n, interleave=True)
        else:
            # For validation, use val_n for pass@k and avg@k computation
            val_rollout_n = getattr(self.config.env.rollout, 'val_n', None)
            if val_rollout_n is not None and val_rollout_n > 1:
                effective_rollout_n = val_rollout_n
                gen_batch = gen_batch.repeat(repeat_times=effective_rollout_n, interleave=True)
            else:
                effective_rollout_n = 1

        if self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                effective_rollout_n=effective_rollout_n,
            )
        else:
            # Vanilla Sampling   
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                effective_rollout_n=effective_rollout_n,
            )
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(totoal_tool_callings)
        

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
        )
        
        return gen_batch_output

# =============================================================================
# Multi‑Agent collector orchestrating a *team* of agents
# =============================================================================
class MultiAgentTrajectoryCollector(TrajectoryCollector):
    """Trajectory collector that *delegates* action generation to a
    user‑configurable :class:`BaseOrchestra` (chain, hierarchy, etc.)."""

    # ------------------------------------------------------------------
    def __init__(
        self,
        config: Any,
        wg_to_agents_mapping: Dict[str, List[Dict[str, str]]],
        tokenizers: Dict[str, PreTrainedTokenizer],
        processors: Dict[str, Any] = None,
    ):
        super().__init__(config=config, tokenizer=tokenizers, processor=processors)

        agent_ids = config.agent.agent_ids
        model_ids = config.agent.model_ids
        orchestra_type = config.agent.orchestra_type
        print("agent_ids: ", agent_ids)
        print("orchestra_type: ", orchestra_type)

        agents_to_wg_mapping = {}
        for wg_id, agents in wg_to_agents_mapping.items():
            for a in agents:
                agent_id = a['agent_id']
                agents_to_wg_mapping[agent_id] = wg_id

        orchestra_kwargs = {}
        if orchestra_type == "search":
            from agent_system.agent.orchestra.search import SearchMultiAgentOrchestra as orchestra
        elif orchestra_type == "math":
            from agent_system.agent.orchestra.math import MathMultiAgentOrchestra as orchestra
        elif orchestra_type == "router":
            from agent_system.agent.orchestra.router import RouterMultiAgentOrchestra as orchestra
            orchestra_kwargs["role_ids"] = config.agent.role_ids
        else:
            raise ValueError(f"Unknown orchestra_type '{orchestra_type}'.")

        self.multiagent_orchestra: BaseOrchestra = orchestra(
            agent_ids=agent_ids,
            model_ids=model_ids,
            agents_to_wg_mapping=agents_to_wg_mapping,
            tokenizers=tokenizers,
            processors=processors,
            config=config,
            **orchestra_kwargs,
        )

    # ------------------------------------------------------------------
    def vanilla_multi_turn_loop(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        envs: EnvironmentManagerBase,
        effective_rollout_n: int,
    ):

        batch_size = len(gen_batch.batch)

        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))
        self.multiagent_orchestra.reset()
        
        if effective_rollout_n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % effective_rollout_n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        router_orchestra = self.config.agent.orchestra_type == "router"
        max_rounds = self.config.agent.orchestra.router.max_loop_num if router_orchestra else 1
        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)
            ###############################
            if router_orchestra:
                # Router.run performs one collaboration round. Keep its context
                # and latest text actions until samples finish.
                self.multiagent_orchestra.initialize_context(obs)
            multiagent_batch_buffer = []
            text_actions = [""] * batch_size
            for _round in range(max_rounds):
                text_actions, round_buffer = self.multiagent_orchestra.run(
                    gen_batch=gen_batch,
                    env_obs=obs,
                    actor_rollout_wgs=actor_rollout_wg,
                    active_masks=active_masks,
                    step=_step+1
                )
                multiagent_batch_buffer.extend(round_buffer)
                if not router_orchestra or np.all(self.multiagent_orchestra.return_reasons[active_masks] != ""):
                    break

            if router_orchestra:
                # On budget exhaustion submit the saved candidate answer, without
                # inventing a Router FINISH. Label only each sample's last call.
                reasons = self.multiagent_orchestra.return_reasons
                limit_mask = active_masks & (reasons == "")
                self.multiagent_orchestra.text_actions = update_text_action(
                    self.multiagent_orchestra.text_actions,
                    self.multiagent_orchestra.candidate_answers,
                    limit_mask,
                )
                text_actions = self.multiagent_orchestra.text_actions.copy()
                reasons[limit_mask] = "loop_limit"
                remaining = active_masks.copy()
                for entry in reversed(multiagent_batch_buffer):
                    metadata = entry['batch'].non_tensor_batch
                    metadata['return_reason'] = np.full(batch_size, "", dtype=object)
                    last_call = remaining & metadata['agent_active_mask']
                    metadata['return_reason'][last_call] = reasons[last_call]
                    remaining[last_call] = False

            # Submit the latest text actions after collaboration ends.
            next_obs, rewards, dones, infos = envs.step(text_actions)
            if router_orchestra:
                # Attach execution observations to the actual last decision only.
                # These strings are metadata, never sampled response tokens.
                observations = next_obs.get("anchor")
                if observations is None:
                    observations = next_obs.get("text") or [""] * batch_size
                for entry in multiagent_batch_buffer:
                    metadata = entry['batch'].non_tensor_batch
                    for index in np.flatnonzero(metadata["return_reason"] != ""):
                        metadata["environment_action"][index] = text_actions[index]
                        metadata["environment_result"][index] = observations[index] or ""
                        if metadata["return_reason"][index] == "search":
                            status = infos[index].get("tool_status", "executed")
                            metadata["execution_status"][index] = status
            ###############################
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            # Create reward tensor, only assign rewards for active environments
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"

            for data in multiagent_batch_buffer:
                agent_id, agent_batch = data['agent_id'], data['batch']
                agent_batch.non_tensor_batch['agent_id'] = np.array([agent_id for _ in range(batch_size)], dtype=object)
                agent_batch.non_tensor_batch['uid'] = uid_batch
                agent_batch.non_tensor_batch['traj_uid'] = traj_uid
                agent_batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
                agent_batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
                agent_batch_list: list[dict] = to_list_of_dict(agent_batch)
                for i in range(batch_size):
                    if agent_batch_list[i]['agent_active_mask']:
                        total_batch_list[i].append(agent_batch_list[i])
                        total_infos[i].append(infos[i])

            # Update done states
            is_done = np.logical_or(is_done, dones)
                
            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break
        
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
