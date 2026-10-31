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
from agent_system.agent.registry import AgentRegistry
from agent_system.multi_turn_rollout.utils import preprocess_batch


SEARCH_PROMPT = r"""
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
You are SEARCH. Generate one targeted query using search(query="...").
Use the original question and actual retrieved evidence to preserve the task's
entities and relationships. Search results arrive from the environment after
this call; do not claim a search has run before its result is available.
"""


@AgentRegistry.register("SEARCH")
class SearchAgent(BaseAgent):
    """Generate a search request for the environment to execute."""

    def __init__(self, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any):
        super().__init__(
            "SEARCH", SEARCH_PROMPT,
            wg_id=wg_id,
            tokenizer=tokenizer,
            processor=processor,
            config=config,
        )
        self.prompt += action_instructions("SEARCH", "search")

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
