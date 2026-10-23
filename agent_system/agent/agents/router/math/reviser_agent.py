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

from typing import Any, Dict, List, Tuple

import numpy as np
from transformers import PreTrainedTokenizer
from verl import DataProto

from agent_system.agent.agents.base import BaseAgent
from agent_system.agent.registry import AgentRegistry
from agent_system.agent.utils import math_projection
from agent_system.multi_turn_rollout.utils import preprocess_batch


REVISER_PROMPT = r"""
# Task Introduction
{env_prompt}

# Current Candidate Answer
The latest saved candidate is provided below for evaluation and improvement.
Being saved does not imply that it is correct or has been verified.
{candidate_answer}

# Your Teammates' Outputs at Step {step}
{team_context}

# Your Assignment
Use the original task, current candidate answer, shared work, and the latest
Router assignment to perform your role. Return your work so the Router can
choose the next step.

# Your Role
You are REVISE. Your responsibility is to correct and improve the current
candidate solution using the verifier's feedback and the Router's assignment.

Compare the candidate with the original problem and identify the specific
reasoning steps, assumptions, calculations, or missing cases that need repair.
Check each criticism before applying it. Preserve reasoning that remains valid,
correct confirmed errors, and explain any feedback you find mistaken. If no
review is available, inspect the candidate yourself to identify needed changes.

After making corrections, check all affected steps and the final conclusion
against the original problem. Fill in missing reasoning so the revised solution
can be understood on its own. If no candidate is available, use the problem and
shared work to construct a complete solution.

Briefly explain what you changed and why, then provide the complete revised solution.
You should give the final answer within \boxed{{}}.
"""


@AgentRegistry.register("REVISE")
class ReviserAgent(BaseAgent):
    """Revise the mathematical solution using the current candidate and feedback."""

    def __init__(self, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any):
        super().__init__(
            "REVISE", REVISER_PROMPT,
            wg_id=wg_id,
            tokenizer=tokenizer,
            processor=processor,
            config=config,
        )

    def call(
        self,
        gen_batch: DataProto,
        env_obs: Dict[str, Any],
        team_context: List[str],
        actor_rollout_wg,
        agent_active_mask,
        step: int,
        candidate_answers: List[str],
    ) -> Tuple[DataProto, List[str]]:
        obs = self.build_prompt(env_obs, team_context, step, candidate_answers=candidate_answers)
        batch = preprocess_batch(
            gen_batch=gen_batch,
            obs=obs,
            config=self.config,
            tokenizer=self.tokenizer,
            processor=self.processor,
            agent_active_mask=agent_active_mask,
        )
        batch, text_responses = self._generate_with_llm(batch, actor_rollout_wg, agent_active_mask, gen_batch.meta_info)

        text_responses, valids = math_projection(text_responses, check_think_tag=False)
        batch.non_tensor_batch["is_action_valid"] = valids & agent_active_mask
        batch.non_tensor_batch["selected_role"] = np.full(len(text_responses), "", dtype=object)
        batch.non_tensor_batch["env_step"] = np.full(len(text_responses), step, dtype=object)
        return batch, text_responses
