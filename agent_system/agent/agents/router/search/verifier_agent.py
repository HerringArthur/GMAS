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
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

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
You are VERIFY. Review retrieved evidence for ANSWER and SEARCH. Identify exactly what the
question asks for (person, work, place, date, etc.). Check each fact needed to
connect that target to the answer. Cite a passage by its title or a short quote.
Use only passages in the search history as sources.

When the passages settle the question, tell ANSWER the supported fact and how
it answers the question. If the saved candidate names the wrong thing, identify
the mismatch and give the supported correction. A candidate is optional.
When evidence is missing or conflicting, name the specific unresolved link and
give SEARCH one query targeting it. Use entity names or clues from the passages
to narrow the query. If no passage has been retrieved, base the query on the question.

# Output Format
Write a brief review in plain text. Explain what the passages establish or
what is missing. If more evidence is needed, include your suggested search
query in that explanation. End with exactly one verdict:
<verify>yes</verify> if ANSWER can write or correct the answer from the passages,
or <verify>no</verify> if a necessary fact still requires retrieval.
"""


class VerificationReport(NamedTuple):
    assessment: str
    verdict: str


def parse_report(response: Optional[str]) -> Optional[VerificationReport]:
    """Check the handoff format before sharing the full text with the team."""
    if not response:
        return None
    match = re.search(r"<verify>\s*(yes|no)\s*</verify>\s*\Z", response)
    if not match:
        return None
    assessment = response[:match.start()].strip()
    if re.search(r"</?verify\b", assessment, re.IGNORECASE):
        return None
    # The team needs a visible review, even if the model also emits reasoning tags.
    prose = re.sub(r"<(think|next)>.*?</\1>", "", assessment, flags=re.DOTALL | re.IGNORECASE)
    if re.search(r"</?(?:think|next|answer|route|search)\b", prose, flags=re.IGNORECASE):
        return None
    if not any(char.isalnum() for char in prose):
        return None
    return VerificationReport(assessment, match.group(1))


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
            [active and parse_report(response) is not None for response, active in zip(text_responses, agent_active_mask)],
            dtype=bool,
        )
        batch.non_tensor_batch["selected_role"] = np.full(len(text_responses), "", dtype=object)
        batch.non_tensor_batch["env_step"] = np.full(len(text_responses), step, dtype=object)
        return batch, text_responses
