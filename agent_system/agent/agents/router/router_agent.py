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
from agent_system.agent.agents.router.search_context import build_search_observation
from agent_system.agent.agents.router.context import build_context_observation, ensure_prompt_fits


ROUTER_TASK_PROMPT = """
# Your Role
You are the router. Review the original task and shared work, then select a worker
or FINISH. Workers have no fixed roles. You may select the same worker again.

Decide using the current progress:
- If only intermediate work exists, identify what remains and select a worker.
- If a candidate answer needs correction, identify its specific defect and the
  work needed to correct it, then select a worker.
- If a complete candidate answers the original task and is supported by the
  reasoning or available evidence, output <route>FINISH</route>. Avoid further
  worker calls or searches once the answer is ready.

When selecting a worker, describe the next task or correction in a brief
<think> </think> block before the route so the worker can act on it. If previous
calls have not resolved the issue, use that history to make the next task specific.
Reselecting a worker is allowed; a repeated answer alone does not require a switch.

HANDOFF and intermediate work do not imply completion. FINISH submits an existing
candidate; it does not create or rewrite an answer. Do not produce the answer or
issue a tool request yourself.

Output exactly one <route>exact agent name</route> or <route>FINISH</route>.
Use an available name exactly as written. Any reasoning or task instruction must
appear before the route inside a single complete <think> </think> block. Output
nothing after </route>.
"""


ROUTER_PROMPT = ROUTER_TASK_PROMPT


ROUTER_MATH_PROMPT = ROUTER_PROMPT + r"""
# Candidate Format
A Math candidate must contain a complete, nonempty \boxed{...} outside <think>.
Intermediate derivations are shared work, not candidate answers. Check that the
candidate answers the original problem and is supported by the derivation.
"""


ROUTER_SEARCH_PROMPT = ROUTER_TASK_PROMPT + """
# Candidate Format
A Search candidate must contain a complete, nonempty <answer> </answer> block
outside <think>. Check that it answers the original question and is supported by
sufficient evidence. A search request or a claim of completion inside <think> is
not a candidate answer. If no candidate exists, select a worker to produce one.
"""


@AgentRegistry.register("Router Agent")
class RouterAgent(BaseAgent):
    """Select a configured worker or FINISH independently for each active sample."""

    FINISH = "FINISH"
    RESPONSE_PATTERN = re.compile(
        r"\s*(?:<think>((?:(?!</?think>).)*)</think>\s*)?<route>([^<>]+)</route>\s*",
        flags=re.DOTALL,
    )

    def __init__(self, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any):
        env_name = config.env.env_name.lower()
        if 'search' in env_name:
            prompt = ROUTER_SEARCH_PROMPT
        elif 'math' in env_name:
            prompt = ROUTER_MATH_PROMPT
        else:
            prompt = ROUTER_PROMPT
        super().__init__("Router Agent", prompt, wg_id, tokenizer, processor, config)
        # The bounded Search builder reads this field; Math uses it below too.
        self.search_feedback = None
        self.worker_ids = [name for name in config.agent.agent_ids if name != self.name]
        if not self.worker_ids or len(set(self.worker_ids)) != len(self.worker_ids):
            raise ValueError("Router Agent requires at least one worker with a unique name.")
        if self.FINISH in self.worker_ids:
            raise ValueError("FINISH is reserved for submitting the worker result.")

    def reset(self):
        self.search_feedback = None
        self._context_token_cache = None

    def build_prompt(self, env_obs: Dict[str, Any], team_context: List[str], step: int,
                     agent_active_mask=None, collaboration_context=None) -> Dict[str, Any]:
        if 'search' in self.config.env.env_name.lower():
            available_agents = "\n# Valid Routing Outputs\n" + "\n".join(
                f"- <route>{name}</route>" for name in self.worker_ids
            ) + "\n- <route>FINISH</route>\n\nChoose exactly one routing output."
            return build_search_observation(self, env_obs, team_context, step, self.prompt, available_agents,
                                            agent_active_mask=agent_active_mask,
                                            collaboration_context=collaboration_context)
        available_agents = "\n# Available Agents\n" + "\n".join(f"- {name}" for name in self.worker_ids)
        return build_context_observation(self, env_obs, team_context, step, self.prompt, available_agents,
                                         agent_active_mask=agent_active_mask,
                                         collaboration_context=collaboration_context)

    def parse_response(self, response: str) -> str:
        """Return an allowed worker name, FINISH, or an empty string for invalid text."""
        match = self.RESPONSE_PATTERN.fullmatch(response)
        if match is None:
            return ""
        selected_agent = match.group(2).strip()
        return selected_agent if selected_agent in self.worker_ids or selected_agent == self.FINISH else ""

    def routing_instruction(self, response: str) -> str:
        """Keep the instruction verbatim for a valid worker selection, if present."""
        if self.parse_response(response) not in self.worker_ids:
            return ''
        return (self.RESPONSE_PATTERN.fullmatch(response).group(1) or '').strip()

    def call(
        self,
        gen_batch: DataProto,
        env_obs: Dict[str, Any],
        team_context: List[str],
        actor_rollout_wg,
        agent_active_mask,
        step: int,
        collaboration_context=None,
    ) -> Tuple[DataProto, List[str]]:
        """Return full router text and the selected agent in non-tensor metadata."""
        active_kwargs = {'agent_active_mask': agent_active_mask}
        obs = self.build_prompt(env_obs, team_context, step, collaboration_context=collaboration_context, **active_kwargs)
        ensure_prompt_fits(self, gen_batch, obs, agent_active_mask)
        batch = preprocess_batch(
            gen_batch=gen_batch,
            obs=obs,
            config=self.config,
            tokenizer=self.tokenizer,
            processor=self.processor,
            **active_kwargs,
        )
        batch, text_responses = self._generate_with_llm(batch, actor_rollout_wg, agent_active_mask, gen_batch.meta_info)
        selected_agents = [
            self.parse_response(response) if active else ""
            for response, active in zip(text_responses, agent_active_mask)
        ]
        if self.search_feedback is None or len(self.search_feedback) != len(text_responses):
            self.search_feedback = [''] * len(text_responses)
        for i, active in enumerate(agent_active_mask):
            if active:
                self.search_feedback[i] = '' if selected_agents[i] else (
                    'Output one complete <route>configured worker name</route> or '
                    '<route>FINISH</route>, optionally preceded by one complete <think> block. '
                    'Use a worker name from the available list and output nothing after </route>.'
                )
        batch.non_tensor_batch['selected_agent'] = np.array(selected_agents, dtype=object)
        batch.non_tensor_batch['routing_instruction'] = np.array([
            self.routing_instruction(response) if active else ''
            for response, active in zip(text_responses, agent_active_mask)
        ], dtype=object)
        batch.non_tensor_batch['router_feedback'] = np.array([
            feedback if active else '' for feedback, active in zip(self.search_feedback, agent_active_mask)
        ], dtype=object)
        batch.non_tensor_batch['is_action_valid'] = np.array([bool(name) for name in selected_agents], dtype=bool)
        batch.non_tensor_batch['env_step'] = np.array([step] * len(text_responses), dtype=object)
        return batch, text_responses
