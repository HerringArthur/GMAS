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

from transformers import PreTrainedTokenizer
from verl import DataProto

from agent_system.agent.agents.base import BaseAgent
from agent_system.code_actions import action_instructions, project_code_actions
from agent_system.multi_turn_rollout.utils import preprocess_batch


VERIFIER_PROMPT = r"""
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
You are VERIFY. Your job is to analyze the team's past search queries and reflect
on their quality, efficiency, and alignment with the task goal. Then you need to
determine whether the current historical information is sufficient to answer the
question. Report your assessment to the Router Agent.

Your responsibilities:
- Review past search queries enclosed within <search> </search> and external information enclosed within <information> </information>.
- Evaluate whether previous queries were reasonable and aligned with the task objective.
- Identify potential issues (if any), including repeated or redundant queries; imprecise queries that are too broad, vague, or missing critical constraints/entities; misaligned queries that drift away from the actual task goal.
- Assess whether the available information is complete and sufficient to generate a high-quality answer.
- If a candidate answer is available, check whether it is supported by the retrieved evidence.

Call record_review(sufficient=False, feedback="Your assessment and evidence").
Set sufficient=True only if the supplied evidence is sufficient to answer the
question; otherwise use False. Distinguish missing evidence from a search that
returned no results, and check the candidate against the actual evidence.
Your review updates shared state; it does not replace the candidate or submit it.
"""


class VerifierAgent(BaseAgent):
    """Review the evidence supporting the current candidate answer."""

    def __init__(self, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any):
        super().__init__(
            "VERIFY", VERIFIER_PROMPT,
            wg_id=wg_id,
            tokenizer=tokenizer,
            processor=processor,
            config=config,
        )
        self.prompt += action_instructions("VERIFY", "search")

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

        return project_code_actions(
            batch, text_responses, agent_active_mask,
            role=self.name, scene="search", step=step,
        )
