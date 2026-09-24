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


SUBAGENT_PROMPT = """
# Task Introduction
{env_prompt}

# Shared Progress at Step {step}
{team_context}

# Your Task
Continue working on the problem using the shared progress and any available tool results.
Choose a useful part of the remaining work to complete, regardless of the task type.
You do not have a fixed role. Focus on this part of the work rather than trying to
complete the entire problem in one turn.

You may reason within <think> </think> tags. Describe the work you completed, your
findings, and any remaining issues before the final control tag. Include enough
detail for yourself or another agent to continue the work.

End every response with exactly one of these control tags:
- <handoff>CONTINUE</handoff> to let the current agent continue generating.
- <handoff>HANDOFF</handoff> to return control to the router so it can select the
  next agent. This does not mean that the entire problem is solved.

The <search>query</search> and <answer>answer</answer> tags may be used in the work
above when required for compatibility with the task's existing tool or answer
format. They do not replace the final control tag or determine whether to continue
with the current agent or return control to the router.

Place the final control tag outside any other tags. Its content must be exactly
CONTINUE or HANDOFF. Do not output anything after </handoff>.
"""


@AgentRegistry.register("Sub Agent")
class SubAgent(BaseAgent):
    """A role-free worker that completes partial work and chooses to continue or hand off."""

    def __init__(self, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any, name: str = "Sub Agent"):
        super().__init__(name, SUBAGENT_PROMPT, wg_id=wg_id, tokenizer=tokenizer, processor=processor, config=config)

    @staticmethod
    def parse_response(response: str) -> str:
        """Read the final handoff decision independently of task-specific output tags."""
        match = re.fullmatch(
            r"(.*?)<handoff>\s*(CONTINUE|HANDOFF)\s*</handoff>\s*",
            response,
            flags=re.DOTALL,
        )
        if match is None:
            return "invalid"

        progress, decision = match.groups()
        # Quoted decisions inside complete reasoning blocks are not control tags.
        progress = re.sub(r"<think>.*?</think>", "", progress, flags=re.DOTALL)
        if re.search(r"</?(?:think|handoff)>", progress):
            return "invalid"
        return decision

    def call(
        self,
        gen_batch: DataProto,
        env_obs: Dict[str, Any],
        team_context: List[str],
        actor_rollout_wg,
        agent_active_mask,
        step: int,
    ) -> Tuple[DataProto, List[str]]:
        """Generate partial work; the orchestra handles continuation and handoffs.

        Return full responses, including compatibility tags, to preserve progress.
        Store CONTINUE, HANDOFF, or invalid in ``handoff_decision`` in the batch's
        non-tensor metadata. Search/answer tags do not determine this decision.
        """
        obs = self.build_prompt(env_obs, team_context, step)
        batch = preprocess_batch(
            gen_batch=gen_batch,
            obs=obs,
            config=self.config,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
        batch, text_responses = self._generate_with_llm(batch, actor_rollout_wg, agent_active_mask, gen_batch.meta_info)

        decisions = [
            self.parse_response(response) if active else "invalid"
            for response, active in zip(text_responses, agent_active_mask)
        ]
        batch.non_tensor_batch['handoff_decision'] = np.array(decisions, dtype=object)
        batch.non_tensor_batch['is_action_valid'] = np.array([decision != "invalid" for decision in decisions], dtype=bool)
        batch.non_tensor_batch['env_step'] = np.array([step] * len(text_responses), dtype=object)
        return batch, text_responses
