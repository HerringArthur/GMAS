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


VERIFIER_PROMPT = r"""
# Task Introduction
{env_prompt}

# Current Candidate Answer
Treat this candidate as unverified.
{candidate_answer}

# Your Teammates' Outputs at Step {step}
{team_context}

# Your Role
You are VERIFY. Assess whether the retrieved evidence supports answering the
original question. Use the Router assignment to focus your review.
Point to supporting facts or documents actually present in the search history,
or identify specific missing or conflicting evidence. Queries, candidate answers,
and teammates' claims are not retrieved evidence. Do not invent sources or results.
Without a candidate, judge whether the evidence is sufficient to form an answer.
With a candidate, also check whether the evidence supports that answer.

# Output Format
Give a brief assessment, then end with exactly one verdict:
<verify>yes</verify>: The evidence is sufficient and supports the candidate, if present.
<verify>no</verify>: The evidence is missing, insufficient, conflicting, or contradicts the candidate.
Use the verdict tag only once, at the end. Output nothing after </verify>.
"""


def _has_final_verdict(response: str) -> bool:
    """Require one terminal yes/no verdict, preserving the complete report."""
    verdict = re.search(r"<verify>\s*(?:yes|no)\s*</verify>\s*\Z", response)
    return verdict is not None and re.search(
        r"</?verify\b", response[:verdict.start()], flags=re.IGNORECASE,
    ) is None


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
            [active and _has_final_verdict(response) for response, active in zip(text_responses, agent_active_mask)],
            dtype=bool,
        )
        batch.non_tensor_batch["selected_role"] = np.full(len(text_responses), "", dtype=object)
        batch.non_tensor_batch["env_step"] = np.full(len(text_responses), step, dtype=object)
        return batch, text_responses
