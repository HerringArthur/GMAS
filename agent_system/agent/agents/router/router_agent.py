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

import re
from typing import Any, Dict, List, Tuple

import numpy as np
from transformers import PreTrainedTokenizer
from verl import DataProto

from agent_system.agent.agents.base import BaseAgent
from agent_system.agent.registry import AgentRegistry
from agent_system.multi_turn_rollout.utils import preprocess_batch


ROUTER_PROMPT = """
# Task Introduction
{env_prompt}

# Shared Progress at Step {step}
{team_context}

# Your Role
Select exactly one agent from the available agents listed below to perform the
next part of the work. These agents have no fixed roles. Use their previous work
and the remaining work to decide who should act next. You may select the same
agent again if appropriate.

When the task is complete and a worker has already produced a result suitable for
submission, output <route>FINISH</route> to submit that worker result. FINISH does
not generate a new answer. A search request is not a final result. If more work
is needed, choose a worker instead. A worker's HANDOFF alone does not imply that
the task is complete.

You only choose a worker or FINISH; do not solve the problem or issue tool requests.
You may reason within <think> </think> tags, then output
<route>exact agent name</route> or <route>FINISH</route>.
Use an available name exactly as written.
Do not output anything after </route>.
"""


@AgentRegistry.register("Router Agent")
class RouterAgent(BaseAgent):
    """Select a configured worker or FINISH independently for each active sample."""

    FINISH = "FINISH"

    def __init__(self, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any):
        super().__init__("Router Agent", ROUTER_PROMPT, wg_id, tokenizer, processor, config)
        self.worker_ids = [name for name in config.agent.agent_ids if name != self.name]
        if not self.worker_ids or len(set(self.worker_ids)) != len(self.worker_ids):
            raise ValueError("Router Agent requires at least one worker with a unique name.")
        if self.FINISH in self.worker_ids:
            raise ValueError("FINISH is reserved for submitting the worker result.")

    def build_prompt(self, env_obs: Dict[str, Any], team_context: List[str], step: int) -> Dict[str, Any]:
        obs = super().build_prompt(env_obs, team_context, step)
        available_agents = "\n# Available Agents\n" + "\n".join(f"- {name}" for name in self.worker_ids)
        obs['text'] = [text + available_agents for text in obs['text']]
        return obs

    def parse_response(self, response: str) -> str:
        """Return an allowed worker name, FINISH, or an empty string for invalid text."""
        match = re.fullmatch(r"\s*(?:<think>.*?</think>\s*)?<route>([^<>]+)</route>\s*", response, flags=re.DOTALL)
        if match is None:
            return ""
        selected_agent = match.group(1).strip()
        return selected_agent if selected_agent in self.worker_ids or selected_agent == self.FINISH else ""

    def call(
        self,
        gen_batch: DataProto,
        env_obs: Dict[str, Any],
        team_context: List[str],
        actor_rollout_wg,
        agent_active_mask,
        step: int,
    ) -> Tuple[DataProto, List[str]]:
        """Return full router text and the selected agent in non-tensor metadata."""
        obs = self.build_prompt(env_obs, team_context, step)
        batch = preprocess_batch(
            gen_batch=gen_batch,
            obs=obs,
            config=self.config,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
        batch, text_responses = self._generate_with_llm(batch, actor_rollout_wg, agent_active_mask, gen_batch.meta_info)
        selected_agents = [
            self.parse_response(response) if active else ""
            for response, active in zip(text_responses, agent_active_mask)
        ]
        batch.non_tensor_batch['selected_agent'] = np.array(selected_agents, dtype=object)
        batch.non_tensor_batch['is_action_valid'] = np.array([bool(name) for name in selected_agents], dtype=bool)
        batch.non_tensor_batch['env_step'] = np.array([step] * len(text_responses), dtype=object)
        return batch, text_responses
