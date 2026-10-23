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


ROLE_DESCRIPTIONS = {
    "math": {
        "SOLVE": "Analyze the problem, develop a solution, and provide an answer.",
        "VERIFY": "Check the mathematical derivation and answer; identify errors or confirm correctness.",
        "REVISE": "Use the existing solution and review feedback to correct or prepare the answer.",
    },
    "search": {
        "SEARCH": "Formulate a targeted search request. Valid requests are executed by the environment before your next decision; no FINISH is needed.",
        "VERIFY": "Check whether the retrieved evidence is reliable and supports the current conclusion.",
        "ANSWER": "Use the available evidence to compose a complete answer to the question.",
    },
}


ROUTER_PROMPT = """
# Task Introduction
{env_prompt}

# Current Candidate Answer
The latest saved candidate is provided below for evaluation and improvement.
Being saved does not imply that it is correct or has been verified.
{candidate_answer}

# Your Teammates' Outputs
{team_context}

# Your Role
You are the Router Agent. Read the task, current candidate answer, and shared
work, then select the role needed next from the available roles below. Choose
the order according to the current progress. You may select the same role again.

FINISH submits the latest saved candidate answer to the environment. Reviews
update the shared context without replacing that answer. Choose FINISH when the
candidate answer is ready for submission. If no candidate answer exists or it
needs correction, select a role to prepare it.

# Output Format
Output exactly one <route>ROLE_NAME</route> or <route>FINISH</route>.
You may precede it with one brief <think>...</think> block explaining the choice
and giving the selected role its next task. This text is shared with that role.
Use an exact available role name and output nothing after </route>.
"""


@AgentRegistry.register("Router Agent")
class RouterAgent(BaseAgent):
    """Select the next role or FINISH for each active sample."""

    def __init__(self, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any):
        super().__init__("Router Agent", ROUTER_PROMPT, wg_id, tokenizer, processor, config)
        self.role_ids = config.agent.role_ids

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
        last_search_step = self.config.env.env_name == "search" and step == self.config.env.max_steps
        available_roles = [role for role in self.role_ids if not (last_search_step and role == "SEARCH")]
        role_descriptions = ROLE_DESCRIPTIONS[self.config.env.env_name]
        role_prompt = "\n# Available Roles\n" + "\n".join(
            f"- {role}: {role_descriptions[role]}" for role in available_roles
        ) + "\n- FINISH: Submit the latest saved candidate answer to the environment.\n"
        if last_search_step:
            role_prompt += (
                "\nThis is the final environment step. SEARCH is unavailable. "
                "Use the available evidence to prepare or check an answer, "
                "then select FINISH to submit the saved candidate.\n"
            )
        obs = self.build_prompt(env_obs, team_context, step, candidate_answers=candidate_answers)
        obs["text"] = [prompt + role_prompt for prompt in obs["text"]]
        batch = preprocess_batch(
            gen_batch=gen_batch,
            obs=obs,
            config=self.config,
            tokenizer=self.tokenizer,
            processor=self.processor,
            agent_active_mask=agent_active_mask,
        )
        batch, text_responses = self._generate_with_llm(batch, actor_rollout_wg, agent_active_mask, gen_batch.meta_info)

        selected_roles = np.full(len(text_responses), "", dtype=object)
        for i, (response, active) in enumerate(zip(text_responses, agent_active_mask)):
            if not active:
                continue
            match = re.fullmatch(
                r"\s*(?:<think>(?:(?!</?(?:think|route)>).)*</think>\s*)?<route>([^<>]+)</route>\s*",
                response,
                flags=re.DOTALL,
            )
            if match is not None:
                role = match.group(1).strip()
                if role in self.role_ids or role == "FINISH":
                    selected_roles[i] = role

        batch.non_tensor_batch["selected_role"] = selected_roles
        batch.non_tensor_batch["is_action_valid"] = selected_roles != ""
        batch.non_tensor_batch["env_step"] = np.full(len(text_responses), step, dtype=object)
        return batch, text_responses
