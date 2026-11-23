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

"""Trajectory-level reward shaping for the Router orchestra.

The search environment scores only the submitted answer and returns zero for
every intermediate step.  Two incentives follow from that and neither is
intended: a candidate submitted through ``finish()`` is worth exactly as much
as the same candidate recovered by the collaboration budget, so looping weakly
dominates stopping; and a Router decision costs nothing, so redundant
``verify`` calls are free even though they consume the round budget and add no
evidence.

Both terms are computed here as trajectory-level scalars.  The caller applies
the shaped reward to the Router rows only and keeps the raw reward on role
rows, so SubAgent credit remains a pure answer-quality comparison while the
Router alone carries the termination signal.

Nothing is applied unless a caller passes a nonzero cost, so scenes that never
set ``return_reason`` (Math, sequential orchestras) keep their exact behaviour.
"""

from typing import Any, List, Optional, Sequence

import numpy as np

ROUTER_AGENT = "Router Agent"
FINISH_REASON = "finish"


def router_decision_counts(
    total_batch_list: Sequence[Sequence[dict]],
    router_agent_id: str = ROUTER_AGENT,
) -> np.ndarray:
    """Count active Router decisions in each collected trajectory.

    ``total_batch_list[i]`` holds only the rows that were active for sample
    ``i``, so the count is the number of decisions that sample actually paid
    for.
    """
    counts = np.zeros(len(total_batch_list), dtype=np.float32)
    for index, rows in enumerate(total_batch_list):
        counts[index] = sum(
            1
            for row in rows
            if bool(row.get("active_masks", True))
            and str(row.get("agent_id", "")) == router_agent_id
        )
    return counts


def truncation_flags(
    total_batch_list: Sequence[Sequence[dict]],
) -> np.ndarray:
    """Mark trajectories that ended without an explicit ``finish``.

    The collector labels only a sample's last call with a return reason, and
    ``finish`` there means the Router submitted its candidate.  ``loop_limit``
    means the collaboration budget ran out and the candidate was submitted on
    the sample's behalf.  A trajectory that carries no reason at all belongs to
    another scene, so it is left unflagged rather than penalised.
    """
    flags = np.zeros(len(total_batch_list), dtype=np.float32)
    for index, rows in enumerate(total_batch_list):
        reason: Optional[str] = None
        for row in reversed(rows):
            candidate = str(row.get("return_reason", "") or "")
            if candidate:
                reason = candidate
                break
        if reason is not None and reason != FINISH_REASON:
            flags[index] = 1.0
    return flags


def shape_episode_rewards(
    episode_rewards: Sequence[float],
    decision_counts: Sequence[float],
    truncation_flags: Sequence[float],
    decision_cost: float = 0.0,
    truncation_penalty: float = 0.0,
) -> np.ndarray:
    """Subtract the shaping terms from each trajectory's total reward.

    ``decision_cost`` prices every Router decision, which makes a redundant
    ``verify`` loop cost something and makes an early submission cheaper than
    one that burns the round budget.  ``truncation_penalty`` applies once to a
    trajectory that never finishes.  With both at zero the rewards are
    returned unchanged.
    """
    rewards = np.asarray(episode_rewards, dtype=np.float32).copy()
    if not decision_cost and not truncation_penalty:
        return rewards
    decisions = np.asarray(decision_counts, dtype=np.float32)
    truncated = np.asarray(truncation_flags, dtype=np.float32)
    if decisions.shape != rewards.shape or truncated.shape != rewards.shape:
        raise ValueError(
            "reward shaping expects matching lengths, got "
            f"{rewards.shape}, {decisions.shape} and {truncated.shape}."
        )
    return (
        rewards
        - float(decision_cost) * decisions
        - float(truncation_penalty) * truncated
    )


def shaped_reward_for_row(agent_id: Any, shaped_reward: float, raw_reward: float) -> float:
    """Return the reward a response row should carry.

    The termination shaping is a Router-only signal: Router rows receive the
    shaped value, every role row keeps the raw reward so SubAgent credit stays
    a pure answer-quality comparison.
    """
    return float(shaped_reward) if str(agent_id) == ROUTER_AGENT else float(raw_reward)


def _read(node: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a mapping node or an attribute node, whichever it is."""
    if node is None:
        return default
    try:
        if key in node:
            return node.get(key, default)
    except TypeError:
        pass
    return getattr(node, key, default)


def read_shaping_settings(config: Any) -> tuple:
    """Return ``(decision_cost, truncation_penalty)`` from ``env.reward_shaping``.

    Missing keys default to zero so an unconfigured run is unchanged.
    """
    shaping = _read(_read(config, "env"), "reward_shaping")
    if shaping is None:
        return 0.0, 0.0
    return (
        float(_read(shaping, "decision_cost", 0.0) or 0.0),
        float(_read(shaping, "truncation_penalty", 0.0) or 0.0),
    )
