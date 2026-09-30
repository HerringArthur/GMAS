# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import math
from collections import defaultdict, Counter

import numpy as np
import torch

import verl.utils.torch_functional as verl_F

from verl import DataProto
import uuid

from difflib import SequenceMatcher
from typing import Sequence, List, Dict, Any, Tuple


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError

# ---------------------------------------------------------- #
# --------------- General Functions of GiGPO --------------- #
# ---------------------------------------------------------- #
def to_hashable(x):
    """Convert an object into a hashable type (used for clustering/grouping)."""
    if isinstance(x, (int, float, str, bool)):
        return x
    elif isinstance(x, (np.integer, np.floating)):
        return x.item()
    elif isinstance(x, np.ndarray):
        return tuple(x.flatten())
    elif isinstance(x, (list, tuple)):
        return tuple(to_hashable(e) for e in x)
    elif isinstance(x, dict):
        return tuple(sorted((k, to_hashable(v)) for k, v in x.items()))
    else:
        raise TypeError(f"Unsupported type: {type(x)}")


def _normalize_group_scalar(value: Any) -> Any:
    """Normalize a scalar metadata value before using it as a dictionary key."""
    if isinstance(value, np.generic):
        value = value.item()
    return to_hashable(value)


def _normalize_prefix_key(value: Any, field_name: str) -> Tuple[Any, ...]:
    """Normalize a serialized agent path into a hashable tuple.

    Router metadata is carried through ``DataProto.non_tensor_batch`` and can be
    represented as a tuple, list, or object-array element depending on where the
    batch was collated.  Keeping the path as a tuple prevents list identity and
    NumPy broadcasting from changing the grouping key.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must contain a list or tuple path, got {type(value)!r}.")
    return tuple(_normalize_group_scalar(item) for item in value)


def build_router_subagent_group_index(
    *,
    agent_ids: Sequence[Any],
    uids: Sequence[Any],
    traj_uids: Sequence[Any],
    prefix_keys: Sequence[Any],
    parent_prefix_keys: Sequence[Any],
    selected_agents: Sequence[Any],
    is_action_valid: Sequence[Any],
    router_agent_id: str = "Router Agent",
) -> Dict[str, Any]:
    """Build statistical grouping and trajectory-deduplication indices.

    This helper deliberately does not inspect rewards and does not remove rows
    from the training batch.  Each deduplication unit stores all of its original
    row indices; a later credit stage can assign one reward/credit to the unit
    and broadcast that value back to every stored row.

    SubAgent rows are grouped by ``prefix_key`` and deduplicated by
    ``(prefix_key, traj_uid)``.  This allows the same prefix from different task
    ``uid`` values to share a baseline while keeping different occurrences of an
    agent on different prefixes separate.

    Router rows are grouped by ``(uid, parent_prefix_key)`` and deduplicated by
    ``(uid, parent_prefix_key, traj_uid)``.  Invalid Router rows are excluded
    from Router statistics but their row indices are returned so the caller can
    retain existing invalid-action handling.  The selected action is stored on
    each deduplicated unit; it is intentionally not part of the deduplication
    key, so one trajectory cannot count as multiple Router choices.

    Args:
        agent_ids: Agent name for each response row.
        uids: Task/group uid for each response row.
        traj_uids: Unique trajectory uid for each response row.
        prefix_keys: Full worker path for each response row.
        parent_prefix_keys: Path immediately before the Router choice/current
            worker segment for each response row.
        selected_agents: Parsed Router choice, including ``FINISH``. Values on
            SubAgent rows are ignored.
        is_action_valid: Validity flag for each response row. Only Router rows
            with a true flag enter Router statistics.
        router_agent_id: Name used to identify Router rows.

    Returns:
        A dictionary with the following fields:

        ``subagent_groups``:
            ``{prefix_key: {traj_uid: [row_indices]}}``.
        ``router_groups``:
            ``{(uid, parent_prefix_key): {traj_uid: {row_indices,
            selected_action}}}``.
        ``row_to_subagent_unit`` and ``row_to_router_unit``:
            Reverse mappings used to broadcast a later credit to every original
            response row.
        ``invalid_router_rows``:
            Row indices for invalid Router responses.
    """
    fields = {
        "agent_ids": agent_ids,
        "uids": uids,
        "traj_uids": traj_uids,
        "prefix_keys": prefix_keys,
        "parent_prefix_keys": parent_prefix_keys,
        "selected_agents": selected_agents,
        "is_action_valid": is_action_valid,
    }
    lengths = {name: len(values) for name, values in fields.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Router/SubAgent metadata arrays must have equal lengths, got {lengths}.")

    subagent_groups: Dict[Tuple[Any, ...], Dict[Any, List[int]]] = {}
    router_groups: Dict[Tuple[Any, Tuple[Any, ...]], Dict[Any, Dict[str, Any]]] = {}
    row_to_subagent_unit: Dict[int, Tuple[Tuple[Any, ...], Any]] = {}
    row_to_router_unit: Dict[int, Tuple[Any, Tuple[Any, ...], Any]] = {}
    invalid_router_rows: List[int] = []

    for row_index, (agent_id, uid, traj_uid, prefix_key, parent_prefix_key, selected_agent, valid) in enumerate(
        zip(
            agent_ids,
            uids,
            traj_uids,
            prefix_keys,
            parent_prefix_keys,
            selected_agents,
            is_action_valid,
        )
    ):
        normalized_agent_id = _normalize_group_scalar(agent_id)
        normalized_uid = _normalize_group_scalar(uid)
        normalized_traj_uid = _normalize_group_scalar(traj_uid)
        normalized_prefix = _normalize_prefix_key(prefix_key, "prefix_key")
        normalized_parent_prefix = _normalize_prefix_key(parent_prefix_key, "parent_prefix_key")

        if normalized_agent_id != router_agent_id:
            # The outer prefix key and inner trajectory key together implement
            # the (traj_uid, prefix_key) statistical deduplication rule.
            prefix_group = subagent_groups.setdefault(normalized_prefix, {})
            prefix_group.setdefault(normalized_traj_uid, []).append(row_index)
            row_to_subagent_unit[row_index] = (normalized_prefix, normalized_traj_uid)
            continue

        if not bool(valid):
            invalid_router_rows.append(row_index)
            continue

        router_group_key = (normalized_uid, normalized_parent_prefix)
        router_group = router_groups.setdefault(router_group_key, {})
        normalized_action = _normalize_group_scalar(selected_agent)
        unit = router_group.get(normalized_traj_uid)
        if unit is None:
            unit = {
                "row_indices": [],
                "selected_action": normalized_action,
            }
            router_group[normalized_traj_uid] = unit
        elif unit["selected_action"] != normalized_action:
            raise ValueError(
                "A Router deduplication unit contains conflicting selected actions: "
                f"uid={normalized_uid!r}, parent_prefix_key={normalized_parent_prefix!r}, "
                f"traj_uid={normalized_traj_uid!r}."
            )
        unit["row_indices"].append(row_index)
        row_to_router_unit[row_index] = (normalized_uid, normalized_parent_prefix, normalized_traj_uid)

    return {
        "subagent_groups": subagent_groups,
        "router_groups": router_groups,
        "row_to_subagent_unit": row_to_subagent_unit,
        "row_to_router_unit": row_to_router_unit,
        "invalid_router_rows": invalid_router_rows,
    }


def attach_episode_rewards_to_group_index(
    group_index: Dict[str, Any],
    traj_uids: Sequence[Any],
    episode_rewards: Sequence[Any],
) -> Dict[str, Any]:
    """Attach one rollout-level reward to every deduplicated unit.

    ``episode_rewards`` is the cumulative environment reward produced by the
    rollout collector.  It is copied onto every response row belonging to the
    same trajectory, so this function first creates one value per ``traj_uid``
    and then exposes reward lookups for the SubAgent and Router deduplication
    units.  It does not calculate credit or modify token-level rewards.
    """
    if len(traj_uids) != len(episode_rewards):
        raise ValueError(
            "traj_uids and episode_rewards must have equal lengths, "
            f"got {len(traj_uids)} and {len(episode_rewards)}."
        )

    trajectory_rewards: Dict[Any, Any] = {}
    for traj_uid, reward in zip(traj_uids, episode_rewards):
        normalized_traj_uid = _normalize_group_scalar(traj_uid)
        if isinstance(reward, np.generic):
            reward = reward.item()
        trajectory_rewards.setdefault(normalized_traj_uid, reward)

    subagent_unit_rewards: Dict[Tuple[Tuple[Any, ...], Any], Any] = {}
    for prefix_key, trajectory_units in group_index["subagent_groups"].items():
        for traj_uid in trajectory_units:
            subagent_unit_rewards[(prefix_key, traj_uid)] = trajectory_rewards[traj_uid]

    router_unit_rewards: Dict[Tuple[Any, Tuple[Any, ...], Any], Any] = {}
    for (uid, parent_prefix_key), trajectory_units in group_index["router_groups"].items():
        for traj_uid in trajectory_units:
            router_unit_rewards[(uid, parent_prefix_key, traj_uid)] = trajectory_rewards[traj_uid]

    return {
        **group_index,
        "trajectory_rewards": trajectory_rewards,
        "subagent_unit_rewards": subagent_unit_rewards,
        "router_unit_rewards": router_unit_rewards,
    }


def compute_subagent_credit(group_index: Dict[str, Any]) -> Dict[str, Any]:
    """Compute prefix-group-relative credit for deduplicated SubAgent units.

    The group mean is computed across unique ``traj_uid`` values in each
    ``prefix_key`` group.  All response rows belonging to one unit receive the
    same scalar credit; this function does not broadcast credit to tokens.
    """
    unit_rewards = group_index["subagent_unit_rewards"]
    unit_credits: Dict[Tuple[Tuple[Any, ...], Any], Any] = {}
    group_baselines: Dict[Tuple[Any, ...], Any] = {}
    row_credits: Dict[int, Any] = {}

    for prefix_key, trajectory_units in group_index["subagent_groups"].items():
        rewards = [unit_rewards[(prefix_key, traj_uid)] for traj_uid in trajectory_units]
        baseline = sum(rewards) / len(rewards)
        group_baselines[prefix_key] = baseline
        for traj_uid, row_indices in trajectory_units.items():
            credit = unit_rewards[(prefix_key, traj_uid)] - baseline
            unit_credits[(prefix_key, traj_uid)] = credit
            for row_index in row_indices:
                row_credits[row_index] = credit

    return {
        "subagent_unit_credits": unit_credits,
        "subagent_group_baselines": group_baselines,
        "subagent_row_credits": row_credits,
    }


def compute_router_credit(group_index: Dict[str, Any]) -> Dict[str, Any]:
    """Compute empirical action-relative credit for deduplicated Router units.

    For each ``(uid, parent_prefix_key)`` group, every trajectory contributes
    once to the empirical action frequency and action mean reward.  The
    baseline is the frequency-weighted action mean.  ``FINISH`` is treated as a
    regular action because it is already stored in ``selected_action``.
    """
    unit_rewards = group_index["router_unit_rewards"]
    unit_credits: Dict[Tuple[Any, Tuple[Any, ...], Any], Any] = {}
    group_baselines: Dict[Tuple[Any, Tuple[Any, ...]], Any] = {}
    action_statistics: Dict[Tuple[Any, Tuple[Any, ...]], Dict[Any, Dict[str, Any]]] = {}
    row_credits: Dict[int, Any] = {}

    for group_key, trajectory_units in group_index["router_groups"].items():
        uid, parent_prefix_key = group_key
        action_rewards: Dict[Any, List[Any]] = defaultdict(list)
        for traj_uid, unit in trajectory_units.items():
            action = unit["selected_action"]
            action_rewards[action].append(unit_rewards[(uid, parent_prefix_key, traj_uid)])

        total_trajectories = sum(len(rewards) for rewards in action_rewards.values())
        action_stats: Dict[Any, Dict[str, Any]] = {}
        baseline = 0.0
        for action, rewards in action_rewards.items():
            count = len(rewards)
            probability = count / total_trajectories
            mean_reward = sum(rewards) / count
            action_stats[action] = {
                "count": count,
                "probability": probability,
                "mean_reward": mean_reward,
            }
            baseline += probability * mean_reward

        group_baselines[group_key] = baseline
        action_statistics[group_key] = action_stats

        for traj_uid, unit in trajectory_units.items():
            unit_key = (uid, parent_prefix_key, traj_uid)
            # Router credit is action-level: every trajectory choosing the
            # same action receives S_j - baseline, rather than its own
            # trajectory reward minus the baseline.
            action = unit["selected_action"]
            credit = action_stats[action]["mean_reward"] - baseline
            unit_credits[unit_key] = credit
            for row_index in unit["row_indices"]:
                row_credits[row_index] = credit

    return {
        "router_unit_credits": unit_credits,
        "router_group_baselines": group_baselines,
        "router_action_statistics": action_statistics,
        "router_row_credits": row_credits,
    }


def compute_router_subagent_credit(group_index: Dict[str, Any]) -> Dict[str, Any]:
    """Compute both credit tables without mutating the training batch."""
    return {
        **compute_subagent_credit(group_index),
        **compute_router_credit(group_index),
    }


def normalize_router_subagent_credit(
    group_index: Dict[str, Any],
    credit_result: Dict[str, Any],
    norm_adv_by_std_in_grpo: bool = True,
    epsilon: float = 1e-6,
) -> Dict[str, Any]:
    """Normalize SubAgent and Router credit within their own statistical groups.

    SubAgent standard deviations use unique trajectory credits within each
    ``prefix_key``. Router standard deviations use the empirical action
    probabilities within each ``(uid, parent_prefix_key)`` group. No response
    row duplication is used when calculating either scale.
    """
    subagent_unit_credits = credit_result["subagent_unit_credits"]
    router_unit_credits = credit_result["router_unit_credits"]

    subagent_unit_advantages: Dict[Tuple[Tuple[Any, ...], Any], Any] = {}
    subagent_group_stds: Dict[Tuple[Any, ...], Any] = {}
    subagent_row_advantages: Dict[int, Any] = {}
    for prefix_key, trajectory_units in group_index["subagent_groups"].items():
        credits = [subagent_unit_credits[(prefix_key, traj_uid)] for traj_uid in trajectory_units]
        variance = sum(credit * credit for credit in credits) / len(credits)
        std = math.sqrt(variance)
        subagent_group_stds[prefix_key] = std
        scale = std + epsilon if norm_adv_by_std_in_grpo else 1.0
        for traj_uid, row_indices in trajectory_units.items():
            unit_key = (prefix_key, traj_uid)
            advantage = subagent_unit_credits[unit_key] / scale
            subagent_unit_advantages[unit_key] = advantage
            for row_index in row_indices:
                subagent_row_advantages[row_index] = advantage

    router_unit_advantages: Dict[Tuple[Any, Tuple[Any, ...], Any], Any] = {}
    router_group_stds: Dict[Tuple[Any, Tuple[Any, ...]], Any] = {}
    router_row_advantages: Dict[int, Any] = {}
    for group_key, trajectory_units in group_index["router_groups"].items():
        action_stats = credit_result["router_action_statistics"][group_key]
        variance = sum(
            stats["probability"] * (stats["mean_reward"] - credit_result["router_group_baselines"][group_key]) ** 2
            for stats in action_stats.values()
        )
        std = math.sqrt(variance)
        router_group_stds[group_key] = std
        scale = std + epsilon if norm_adv_by_std_in_grpo else 1.0
        for traj_uid, unit in trajectory_units.items():
            unit_key = (group_key[0], group_key[1], traj_uid)
            action = unit["selected_action"]
            advantage = (
                action_stats[action]["mean_reward"]
                - credit_result["router_group_baselines"][group_key]
            ) / scale
            router_unit_advantages[unit_key] = advantage
            for row_index in unit["row_indices"]:
                router_row_advantages[row_index] = advantage

    return {
        "subagent_unit_advantages": subagent_unit_advantages,
        "subagent_group_stds": subagent_group_stds,
        "subagent_row_advantages": subagent_row_advantages,
        "router_unit_advantages": router_unit_advantages,
        "router_group_stds": router_group_stds,
        "router_row_advantages": router_row_advantages,
    }


def build_token_level_credit_advantage(
    normalized_credit: Dict[str, Any],
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Broadcast normalized row credit to the valid response tokens."""
    token_advantages = torch.zeros_like(response_mask, dtype=torch.float32)
    row_advantages = {
        **normalized_credit["subagent_row_advantages"],
        **normalized_credit["router_row_advantages"],
    }
    for row_index, advantage in row_advantages.items():
        token_advantages[row_index] = advantage
    return token_advantages * response_mask


def build_router_subagent_group_index_from_batch(
    data: DataProto,
    router_agent_id: str = "Router Agent",
) -> Dict[str, Any]:
    """Build Router/SubAgent grouping indices directly from a ``DataProto``.

    This adapter is intentionally separate from reward extraction.  It is the
    entry point that the future custom GRPO branch can call after batch padding
    and before computing group statistics.
    """
    required_fields = (
        "agent_id",
        "uid",
        "traj_uid",
        "prefix_key",
        "parent_prefix_key",
        "selected_agent",
        "is_action_valid",
        "episode_rewards",
    )
    missing_fields = [field for field in required_fields if field not in data.non_tensor_batch]
    if missing_fields:
        raise KeyError(
            "Router/SubAgent grouping requires non-tensor metadata fields: "
            + ", ".join(missing_fields)
        )
    non_tensor_batch = data.non_tensor_batch
    group_index = build_router_subagent_group_index(
        agent_ids=non_tensor_batch["agent_id"],
        uids=non_tensor_batch["uid"],
        traj_uids=non_tensor_batch["traj_uid"],
        prefix_keys=non_tensor_batch["prefix_key"],
        parent_prefix_keys=non_tensor_batch["parent_prefix_key"],
        selected_agents=non_tensor_batch["selected_agent"],
        is_action_valid=non_tensor_batch["is_action_valid"],
        router_agent_id=router_agent_id,
    )
    return attach_episode_rewards_to_group_index(
        group_index,
        traj_uids=non_tensor_batch["traj_uid"],
        episode_rewards=non_tensor_batch["episode_rewards"],
    )


def summarize_group_size(group_size: list):
    """
    Summarize the dynamics of step-level group.
    Args:
        group_size : List[int]
    """
    counts = Counter(group_size)
    total = sum(counts.values())
    max_size = max(counts)

    summary = {}
    for size in range(1, max_size + 1):
        cnt = counts.get(size, 0)
        prop = cnt / total if total > 0 else 0
        summary[size] = (cnt, prop)

    print("Summary of step-level group sizes:")
    print("Size | Count | Proportion")
    print("-------------------------")
    for size, (cnt, prop) in summary.items():
        if prop:
            print(f"{size:>4} | {cnt:>5} | {prop:>9.2%}")
            
def are_similar(a: str, b: str, threshold: float = 0.95) -> bool:
    """
    Check whether two text observations are similar enough.
    
    Args:
        a, b (str): Input strings to compare.
        threshold (float): Minimum similarity ratio.
    
    Returns:
        bool: True if similarity >= threshold.
    """
    if not isinstance(a, str) or not isinstance(b, str):
        raise ValueError("Only text-based observations are supported for similarity-based GiGPO in this version.")
    return SequenceMatcher(None, a, b).ratio() >= threshold

def compute_step_discounted_returns(batch: DataProto, gamma: float):
    """
    Compute discounted returns for each trajectory. (Eq. 5 in the paper)
    
    Args:
        batch (DataProto): Input batch.
        gamma (float): Discount factor.
    
    Returns:
        torch.Tensor: Discounted returns.
    """
    rewards = batch.non_tensor_batch['rewards'].astype(np.float32)
    traj_uids = batch.non_tensor_batch['traj_uid']
    active_masks = batch.non_tensor_batch['active_masks'].astype(np.float32)
    env_step = batch.non_tensor_batch['env_step'].astype(np.int32)
    # returns_by_traj_ = {}
    returns_by_traj = {}
    unique_traj_uids = np.unique(traj_uids)
    for uid in unique_traj_uids:
        # Get indices for this trajectory
        traj_indices = np.where(traj_uids == uid)[0]
        
        # Extract rewards and masks for this trajectory
        traj_rewards = rewards[traj_indices]
        traj_active_masks = active_masks[traj_indices]
        traj_env_step = env_step[traj_indices]
        assert traj_active_masks.all(), "active_masks should be all 1s for the same trajectory"
        
        first_of_group = np.r_[True, traj_env_step[1:] != traj_env_step[:-1]]
        step_starts = np.flatnonzero(first_of_group)
        step_ends = np.r_[step_starts[1:], len(traj_env_step)]

        step_rewards = traj_rewards[step_starts].astype(np.float32)

        # Calculate returns
        step_returns = np.zeros_like(step_rewards, dtype=np.float32)
        running_return = 0.0
        
        # Calculate returns from the end to the start
        for k in reversed(range(len(step_rewards))):
            running_return = step_rewards[k] + gamma * running_return
            step_returns[k] = running_return
        
        traj_returns = np.zeros_like(traj_rewards, dtype=np.float32)
        for sr, s, e in zip(step_returns, step_starts, step_ends):
            traj_returns[s:e] = sr
            
        # Store the results
        # returns_by_traj_[uid] = traj_returns
        returns_by_traj[uid] = (traj_indices, traj_returns)
    
    # Recombine the returns into the original batch order
    # all_returns_ = np.zeros_like(rewards)
    # for i, uid in enumerate(traj_uids):
    #     traj_indices = np.where(traj_uids == uid)[0]
    #     idx_in_traj = np.where(traj_indices == i)[0][0]  # Find position of i in its trajectory
    #     all_returns_[i] = returns_by_traj_[uid][idx_in_traj]

    all_returns = np.zeros_like(rewards, dtype=np.float32)
    for uid in unique_traj_uids:
        traj_indices, traj_returns = returns_by_traj[uid]
        all_returns[traj_indices] = traj_returns

    # assert (all_returns==all_returns_).all()
    
    all_returns = torch.tensor(all_returns, dtype=torch.float32, device=batch.batch['input_ids'].device)
    return all_returns

# ---------------------------------------------------------- #
# ---------------- Core Functions of GiGPO ----------------- #
# ---------------------------------------------------------- #

def compute_gigpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   step_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   anchor_obs: np.array,
                                   index: np.array,
                                   traj_index: np.array,
                                   epsilon: float = 1e-6,
                                   step_advantage_w: float = 1.0,
                                   mode: str = "mean_norm",
                                   enable_similarity: bool = False,
                                   similarity_thresh: float = 0.95,
                                   group_by_agent_id: bool = False
                                   ):
    """
    Compute the advantages for GiGPO (https://arxiv.org/abs/2505.10978).
    """
    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Compute episode relative advantages (Eq. 3 in the paper).
    episode_advantages = episode_norm_reward(token_level_rewards, response_mask, index, traj_index, epsilon, remove_std, group_by_agent_id)
    
    # Anchor state grouping (Eq. 6 in the paper).
    step_group_uids = build_step_group(anchor_obs, index, enable_similarity, similarity_thresh)

    # Compute step relative advantages (Eq. 7 in the paper).
    step_advantages = step_norm_reward(step_rewards, response_mask, step_group_uids, epsilon, remove_std)

    # Compute joint advantages (Eq. 8 in the paper).
    scores = episode_advantages + step_advantage_w * step_advantages
    return scores, scores


def episode_norm_reward(token_level_rewards: torch.Tensor,
                        response_mask: torch.Tensor,
                        index: np.array,
                        traj_index: np.array,
                        epsilon: float = 1e-6,
                        remove_std: bool = True,
                        group_by_agent_id: bool = False,
                        ):
    """
    Compute episode-level advantage using mean-std normalization for GiGPO.
    (with only one scalar reward for each episode).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.array)`
            shape: (bs,)
        traj_index: `(np.array)`
            shape: (bs,)
        epsilon: float
            A small value to avoid division by zero.
        remove_std: bool
            If True, the standard deviation is removed from the normalization.
        group_by_agent_id: bool
            If True, the mean and std are computed across agent group.
            If False (i.e., standard trajectory-level GRPO), the mean and std are computed across trajectories within one group.
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not group_by_agent_id:
                seen_pairs.add((index[i], traj_index[i]))

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        episode_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask

    return episode_advantages


def build_step_group(anchor_obs: np.array, index: np.array, enable_similarity: bool = False, similarity_thresh: float = 0.95, summarize: bool = False):
    """
    Group observations by index and then cluster identical observations within each index group.
    Assigns a unique step_group_uid (UUID) to each cluster.
    
    Parameters:
    -----------
    anchor_obs : np.array
        Array of observation strings
    index : np.array
        Array of episode_group_uid
    summarize : bool
        Whether to summarize the group sizes (default: True)
    enable_similarity : bool
        Whether to enable similarity-based step-level grouping (default: False)
    similarity_thresh : float
        Threshold for similarity to consider two observations as identical (default: 1.0, meaning exact match)
    
    Returns:
    --------
    np.array
        Array of step_group_uid values corresponding to the original anchor_obs array
    """
    if enable_similarity:
        assert similarity_thresh > 0.0 and similarity_thresh < 1.0, "When enabling similarity-based step-level group, similarity_thresh should be in (0, 1)"

    # Initialize the result array with placeholder values
    step_group_uids = np.empty(len(anchor_obs), dtype=object)
    
    # Get unique indices
    unique_indices = np.unique(index)

    group_size: List[int] = []
    # Process each unique index
    for idx in unique_indices:
        if not enable_similarity:
            # Get all observations for this index using np.where
            indices = np.where(index == idx)[0]
            obs_group = anchor_obs[indices]
            
            # Create clusters for identical observations
            clusters = defaultdict(list)
            for i, obs in enumerate(obs_group):
                clusters[to_hashable(obs)].append(indices[i])  # Store the original index position
            
            # Assign unique step_group_uid to each cluster
            for obs, original_indices in clusters.items():
                # Generate a UUID for this cluster
                uid = str(uuid.uuid4())
                
                # Assign the same step_group_uid to all elements in this cluster
                group_size.append(len(original_indices))
                for original_idx in original_indices:
                    step_group_uids[original_idx] = uid
        else:
            locs = np.where(index == idx)[0]
            obs_group = anchor_obs[locs]

            # Dynamically maintain clusters: [{rep: str, locs: List[int]} ...]
            clusters: List[Dict[str, Any]] = []

            for obs, loc in zip(obs_group, locs):
                 # Try to place into an existing cluster
                placed = False
                for cluster in clusters:
                    if are_similar(obs, cluster["rep"], similarity_thresh):
                        cluster["locs"].append(loc)
                        placed = True
                        break
                # If no matching cluster, create a new one
                if not placed:
                    clusters.append({"rep": obs, "locs": [loc]})

            # Assign a UUID to each cluster
            for cluster in clusters:
                uid = str(uuid.uuid4())
                group_size.append(len(cluster["locs"]))
                for loc in cluster["locs"]:
                    step_group_uids[loc] = uid

        # Validate that all elements have been assigned a uid
    if None in step_group_uids or np.any(step_group_uids == None):
        missing_indices = np.where(step_group_uids == None)[0]
        raise ValueError(f"Failed to assign UIDs to all observations. Missing at indices: {missing_indices}")

    if summarize:
        summarize_group_size(group_size)
    print(f"Avg size of step-level group: {np.mean(group_size)}")
    return step_group_uids


def step_norm_reward(step_rewards: torch.Tensor,
                      response_mask: torch.Tensor,
                      index: np.array,
                      epsilon: float = 1e-6,
                      remove_std: bool = True,
                      ):
    """
    Compute step-level advantage using mean-std normalization for GiGPO.
    Args:
        step_rewards: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.shape[-1]
    scores = step_rewards.clone()

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                print(f"id2score: {id2score}")
                print(f"len(id2score[idx]): {len(id2score[idx])}")
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        step_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
    
    return step_advantages



def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    group_by_agent_id: bool = False,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).
        group_by_agent_id: bool
            If True, the mean and std are computed across agent group.
            If False (i.e., standard episode-level adv), the mean and std are computed across trajectories within one group.

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    print("group_by_agent_id: ", group_by_agent_id)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    traj_accumulator = defaultdict(list)
    traj2avg = {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            traj_accumulator[(index[i], traj_index[i])].append(scores[i])
        
        for (idx, t_idx), reward_list in traj_accumulator.items():
            if group_by_agent_id:
                id2score[idx].extend(reward_list)
            else:
                avg_score = torch.stack(reward_list).mean()
                traj2avg[(idx, t_idx)] = avg_score
                id2score[idx].append(avg_score)
        if not group_by_agent_id:
            for i in range(bsz):
                scores[i] = traj2avg[(index[i], traj_index[i])]

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    group_by_agent_id: bool = False,
):
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        norm_adv_by_std_in_grpo: if True, normalize advantage by std within group
        group_by_agent_id: bool
            If True, the mean and std are computed across agent group.
            If False (i.e., standard episode-level adv), the mean and std are computed across trajectories within one group.

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)
            if not group_by_agent_id:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}.")
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, traj_index: np.ndarray, epsilon: float = 1e-6, group_by_agent_id: bool = False):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not group_by_agent_id:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, traj_index: np.ndarray, epsilon: float = 1e-6, group_by_agent_id: bool = False):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not group_by_agent_id:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(vpreds: torch.Tensor, returns: torch.Tensor, values: torch.Tensor, response_mask: torch.Tensor, cliprange_value: float, loss_agg_mode: str = "token-mean"):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data
