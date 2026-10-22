# Copyright 2026 Nanyang Technological University (NTU), Singapore
# Copyright 2026 Dr. MAS Team
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

"""Role-based routing using the shared Orchestra call/update/buffer interface.

Agent contract:

* The router agent package registers Router Agent and each configured role with
  AgentRegistry so BaseOrchestra can create them by name.
* All agents expose the usual ``call(...) -> (DataProto, List[str])`` interface
  and return a full-size batch with ``is_action_valid``, ``agent_active_mask``,
  ``wg_id`` and ``env_step`` for each input row. Inactive rows are invalid.
* RouterAgent additionally returns ``non_tensor_batch['selected_role']``: an
  exact configured role name or ``FINISH``, and marks other selections invalid
  through ``is_action_valid``. Its text contains any assignment
  instructions, which are passed to the selected role through team_context.
* SubAgent uses its role name/configuration to select its prompt. SEARCH validates
  a single nonempty search request; other roles check for nonempty responses.
  It supplies an empty ``selected_role`` field. This module does not parse
  model-generated tags. All valid role responses update team_context. Only
  SOLVE/REVISE in math and SEARCH/ANSWER in search update text_actions. A valid
  SEARCH response in the search scene pauses that sample for environment execution.

Example configuration, alongside a matching agent-to-worker-group mapping::

    agent:
      agent_ids: ["Router Agent", "SOLVE", "VERIFY", "REVISE"]
      role_ids: ["SOLVE", "VERIFY", "REVISE"]
      model_sharing: false
      orchestra:
        router:
          max_loop_num: 4

Role prompts belong to the agents. Further role-output validation and
credit-assignment metadata are deferred.
"""

import importlib
from typing import Any, Dict, List, Tuple

import numpy as np
from transformers import PreTrainedTokenizer

from verl import DataProto

from agent_system.agent.orchestra.base import BaseOrchestra, update_team_context, update_text_action


class RouterMultiAgentOrchestra(BaseOrchestra):
    """Let each active sample choose its next role independently.

    Each run() performs one Router call and at most one role call per sample.
    The rollout collector repeats run() and owns the round budget. Team context
    persists across environment steps until reset() starts a new rollout.
    initialize_context() clears pending actions and return reasons for each
    environment step. FINISH and valid SEARCH responses mark samples ready for
    the next environment interaction.
    """

    ROUTER_AGENT = "Router Agent"
    FINISH = "FINISH"

    def __init__(
        self,
        agent_ids: List[str],
        role_ids: List[str],
        model_ids: List[str],
        agents_to_wg_mapping: Dict[str, str],
        tokenizers: Dict[str, PreTrainedTokenizer] = None,
        processors: Dict[str, Any] = None,
        config: Any = None,
    ):
        importlib.import_module("agent_system.agent.agents.router")
        super().__init__(
            agent_ids=agent_ids,
            model_ids=model_ids,
            agents_to_wg_mapping=agents_to_wg_mapping,
            tokenizers=tokenizers,
            processors=processors,
            config=config,
        )
        self.role_ids = role_ids
        self.team_context = None
        self.text_actions = None
        self.return_reasons = None

    def reset(self):
        super().reset()
        self.team_context = None
        self.text_actions = None
        self.return_reasons = None

    def initialize_context(self, env_obs):
        """Start an environment step while preserving this rollout's team context."""
        text_actions, team_context, env_obs = super().initialize_context(env_obs)
        if self.team_context is None:
            self.team_context = team_context
        self.text_actions = text_actions
        self.return_reasons = np.full(len(self.text_actions), "", dtype=object)
        return self.text_actions, self.team_context, env_obs

    def run(
        self,
        gen_batch: DataProto,
        env_obs: Dict[str, Any],
        actor_rollout_wgs,
        active_masks: np.ndarray,
        step: int,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        self.reset_buffer()
        if self.team_context is None:
            self.initialize_context(env_obs)
        pending_mask = active_masks & (self.return_reasons == "")
        if not pending_mask.any():
            return self.text_actions.copy(), self.multiagent_batch_buffer

        # 1. Route each pending sample once, using work from preceding calls.
        actor_rollout_wg = actor_rollout_wgs[self.agents_to_wg_mapping[self.ROUTER_AGENT]]
        router_batch, router_responses = self.agents[self.ROUTER_AGENT].call(
            gen_batch=gen_batch,
            env_obs=env_obs,
            team_context=self.team_context,
            actor_rollout_wg=actor_rollout_wg,
            agent_active_mask=pending_mask,
            step=step,
        )
        selections = router_batch.non_tensor_batch["selected_role"]
        finish_mask = selections == self.FINISH
        route_mask = pending_mask & router_batch.non_tensor_batch["is_action_valid"]
        self.team_context = update_team_context(self.ROUTER_AGENT, self.team_context, router_responses, route_mask)
        self.save_to_buffer(self.ROUTER_AGENT, router_batch)
        self.return_reasons[route_mask & finish_mask] = "finish"

        # 2. Dispatch each sample to its selected role. This traversal groups
        # batch rows by model; it does not run additional collaboration rounds.
        scene = self.config.env.env_name
        for role in self.role_ids:
            role_mask = route_mask & (selections == role)
            if not role_mask.any():
                continue
            actor_rollout_wg = actor_rollout_wgs[self.agents_to_wg_mapping[role]]
            batch, responses = self.agents[role].call(
                gen_batch=gen_batch,
                env_obs=env_obs,
                team_context=self.team_context,
                actor_rollout_wg=actor_rollout_wg,
                agent_active_mask=role_mask,
                step=step,
            )
            valid_mask = role_mask & batch.non_tensor_batch["is_action_valid"]
            self.team_context = update_team_context(role, self.team_context, responses, valid_mask)
            update_action = (
                scene == "math" and role in ("SOLVE", "REVISE")
            ) or (
                scene == "search" and role in ("SEARCH", "ANSWER")
            )
            if update_action:
                self.text_actions = update_text_action(self.text_actions, responses, valid_mask)
            if scene == "search" and role == "SEARCH":
                self.return_reasons[valid_mask] = "search"
            self.save_to_buffer(role, batch)

        return self.text_actions.copy(), self.multiagent_batch_buffer
