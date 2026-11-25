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
        "SOLVE": "Solve the problem and provide an answer.",
        "VERIFY": "Check the mathematical derivation and answer.",
        "REVISE": "Use review feedback to correct the solution and provide a complete revised solution.",
    },
    "search": {
        "SEARCH": "Write a search query. The environment executes it automatically before your next decision.",
        "VERIFY": "Review retrieved evidence for ANSWER. Read the correction if one is given. A no supplies a query for SEARCH.",
        "ANSWER": "Use the available evidence to write or correct an answer.",
    },
}


ROUTER_PROMPT = """
# Task Introduction
{env_prompt}

# Current Candidate Answer
Being saved does not imply that it is correct or has been verified.
{candidate_answer}

# Your Teammates' Outputs
{team_context}

# Your Role
You are the Router Agent. Choose the next role using the task, candidate answer,
and team outputs. You may select a role again.
VERIFY adds feedback without replacing the candidate answer.
Use FINISH  when a candidate exists and is ready to submit. Otherwise,
choose a role to prepare or correct it.
"""


@AgentRegistry.register("Router Agent")
class RouterAgent(BaseAgent):
    """Select the next role or FINISH for each active sample."""

    def __init__(self, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any):
        super().__init__("Router Agent", ROUTER_PROMPT, wg_id, tokenizer, processor, config)
        self.role_ids = config.agent.role_ids
        self.max_steps = int(getattr(config.env, "max_steps", 1) or 1)
        router_cfg = getattr(getattr(config.agent, "orchestra", None), "router", None)
        self.max_rounds = int(getattr(router_cfg, "max_loop_num", 1) or 1)

    def call(
        self,
        gen_batch: DataProto,
        env_obs: Dict[str, Any],
        team_context: List[str],
        actor_rollout_wg,
        agent_active_mask,
        step: int,
        candidate_answers: List[str],
        round_index: int = 1,
    ) -> Tuple[DataProto, List[str]]:
        last_search_step = self.config.env.env_name == "search" and step == self.config.env.max_steps
        available_roles = [role for role in self.role_ids if not (last_search_step and role == "SEARCH")]
        role_descriptions = ROLE_DESCRIPTIONS[self.config.env.env_name]
        budget_line = (
            f"\n# Budget\n"
            f"Environment step {step} of {self.max_steps}; "
            f"collaboration round {round_index} of {self.max_rounds}.\n"
        )
        role_prompt = budget_line + "\n# Available Roles\n" + "\n".join(
            f"<route>{role}</route>: {role_descriptions[role]}" for role in available_roles
        ) + "\n<route>FINISH</route>: Submit the saved candidate answer.\n"
        if last_search_step:
            role_prompt += (
                "\nThis is the final environment step. SEARCH is unavailable. "
                "Prepare or check an answer, then submit it with FINISH.\n"
            )
        role_prompt += (
            "\n# Output Format\n"
            "Return exactly one of the route blocks listed above, including its tags.\n"
            "Output nothing else and end immediately after </route>.\n"
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
