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
from agent_system.multi_turn_rollout.utils import preprocess_batch


SUBAGENT_PROMPT = """
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
"""


ROLE_PROMPTS = {
    "math": {
        "SOLVE": r"""
# Your Role
You are SOLVE. Reason through the math problem step by step and develop a
complete solution, using relevant shared work. Explain the derivation and put
the final answer to the original problem in \boxed{{...}}.
""",
        "VERIFY": """
# Your Role
You are VERIFY. Check the latest mathematical solution against the original
problem. Review each reasoning step, formula, calculation, and final answer.
Explain whether the solution is correct, identify any errors, and describe
what needs correction.
""",
        "REVISE": r"""
# Your Role
You are REVISE. Use the existing mathematical solution, review feedback, and
Router assignment to correct or prepare the answer. Check the feedback before
applying it. Provide the complete revised solution and put the final answer
to the original problem in \boxed{{...}}.
""",
    },
    "search": {
        "SEARCH": """
# Your Role
You are SEARCH. Read the question and retrieved information, identify what is
missing, and formulate a targeted search query. Account for previous searches
when choosing the query. Put any reasoning in <think>...</think>, followed by
one <search>your query</search> block as the final part of the response.
""",
        "VERIFY": """
# Your Role
You are VERIFY. Check whether the retrieved evidence is reliable, relevant,
and sufficient to support the current conclusion. Identify conflicting facts,
unsupported claims, and missing information. Provide a review explaining what
the evidence establishes and what still needs checking.
""",
        "ANSWER": """
# Your Role
You are ANSWER. Use the question, retrieved evidence, and shared review feedback
to compose a complete, accurate answer. Preserve essential qualifiers and make
the answer consistent with the evidence. Put any reasoning in <think>...</think>,
followed by one <answer>final answer</answer> block as the final part of the response.
""",
    },
}


class SubAgent(BaseAgent):
    """Execute the configured role using its prompt and assigned worker group."""

    def __init__(self, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any, name: str):
        super().__init__(
            name,
            SUBAGENT_PROMPT + ROLE_PROMPTS[config.env.env_name][name],
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

        batch.non_tensor_batch["is_action_valid"] = np.array(
            [active and bool(response.strip()) for response, active in zip(text_responses, agent_active_mask)],
            dtype=bool,
        )
        if self.name == "SEARCH":
            for i, (response, active) in enumerate(zip(text_responses, agent_active_mask)):
                if not active:
                    continue
                batch.non_tensor_batch["is_action_valid"][i] = False
                content = response.strip()

                # Remove the optional thought block before checking the request.
                if content.startswith("<think>"):
                    thought, closing_tag, content = content[len("<think>"):].partition("</think>")
                    # The environment also scans the thought text for action tags.
                    if not closing_tag or re.search(r"</?(think|search|answer)>", thought):
                        continue
                    content = content.strip()

                match = re.fullmatch(r"<search>([^<>]+)</search>", content)
                batch.non_tensor_batch["is_action_valid"][i] = match is not None and bool(match.group(1).strip())
        batch.non_tensor_batch["selected_role"] = np.full(len(text_responses), "", dtype=object)
        batch.non_tensor_batch["env_step"] = np.full(len(text_responses), step, dtype=object)
        return batch, text_responses
