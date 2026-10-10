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


SUBAGENT_SEARCH_PROMPT = """
# Your Task
You are a worker in a team answering this question. You have no fixed role.
At every turn, first read the question, shared progress, and retrieved evidence.
Briefly assess in <think> </think> what the evidence establishes and what specific
information, if any, is still missing. Then output exactly one action outside
<think>, followed by exactly one handoff control tag.

Choose the action using the evidence:
1. If evidence is still missing, output <search>targeted query</search>. Search for
   the missing fact; do not repeat a query whose results already answer the
   question. Finish with <handoff>CONTINUE</handoff> to use the returned evidence
   yourself, or <handoff>HANDOFF</handoff> for the router to select the next worker.
2. As soon as the evidence is sufficient, output <answer>final answer only</answer>
   and finish with <handoff>HANDOFF</handoff> for the router to review and submit.
   The answer must be the shortest value that fully answers the question: a name,
   date, number, occupation, or yes/no. Include essential qualifiers. Explanations
   belong in <think>; do not put an explanatory sentence, repeat the question, or
   add a prefix such as "The answer is" inside <answer>.

Saying that the task is complete inside <think> is not an answer. A response with
only reasoning and a handoff tag is incomplete: it must also contain a search or
answer action. Do not output both actions, and do not keep choosing CONTINUE after
producing a final answer. Place the control tag outside all other tags, and output
nothing after </handoff>.
"""


SUBAGENT_SEARCH_FINAL_PROMPT = """
# Final answer required
This is the last environment step. Answer the original question using the
retrieved evidence and your knowledge. Give the best-supported answer available,
even if some uncertainty remains. Another search cannot return any evidence.
You are still the current worker; return the answer to the router for review.

Output a brief <think>reasoning</think> if needed, followed by exactly:
<answer>short final answer</answer><handoff>HANDOFF</handoff>

Inside <answer>, put only the requested name, date, number, occupation, or yes/no,
with essential qualifiers. Put explanations in <think>. Output nothing after
</handoff>. Do not issue a search or CONTINUE.
"""


@AgentRegistry.register("Sub Agent")
class SubAgent(BaseAgent):
    """A role-free worker that completes partial work and chooses to continue or hand off."""

    def __init__(self, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any, name: str = "Sub Agent"):
        prompt = SUBAGENT_SEARCH_PROMPT if 'search' in config.env.env_name.lower() else SUBAGENT_PROMPT
        super().__init__(name, prompt, wg_id=wg_id, tokenizer=tokenizer, processor=processor, config=config)
        self.search_feedback = None

    def reset(self):
        self.search_feedback = None
        self._search_token_cache = None

    def build_prompt(self, env_obs: Dict[str, Any], team_context: List[str], step: int, agent_active_mask=None) -> Dict[str, Any]:
        if 'search' in self.config.env.env_name.lower():
            max_steps = getattr(self.config.env, 'max_steps', None)
            prompt = SUBAGENT_SEARCH_FINAL_PROMPT if max_steps is not None and step >= max_steps else self.prompt
            return build_search_observation(self, env_obs, team_context, step, prompt,
                                            agent_active_mask=agent_active_mask)
        return super().build_prompt(env_obs, team_context, step)

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

    @staticmethod
    def search_action(response: str) -> Tuple[str, bool, str]:
        """Extract one complete external action, with an error suitable for retry."""
        work = re.sub(r'<handoff>\s*(CONTINUE|HANDOFF)\s*</handoff>\s*$', '', response).strip()
        visible = re.sub(r'<think>.*?</think>', '', work, flags=re.DOTALL)
        blocks = list(re.finditer(r'<(search|answer)>(.*?)</\1>', visible, flags=re.DOTALL))
        if len(blocks) != 1:
            return '', False, ('Output exactly one complete <search>query</search> OR '
                               '<answer>short value</answer> outside <think>, followed by one handoff tag. '
                               'Reasoning and a handoff tag alone are not an action; do not output both actions.')
        block = blocks[0]
        tag, content = block.groups()
        remainder = visible[:block.start()] + visible[block.end():]
        if re.search(r'</?(?:search|answer)\b', remainder + content):
            return '', False, 'Remove extra, nested, or incomplete action tags. Output only one complete search or answer block.'
        if not content.strip():
            return '', False, 'The action is empty. Supply a search query or a short final answer inside its tags.'
        return f'<{tag}>{content.strip()}</{tag}>', tag == 'search', ''

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
        active_kwargs = {'agent_active_mask': agent_active_mask} if 'search' in self.config.env.env_name.lower() else {}
        obs = self.build_prompt(env_obs, team_context, step, **active_kwargs)
        batch = preprocess_batch(
            gen_batch=gen_batch,
            obs=obs,
            config=self.config,
            tokenizer=self.tokenizer,
            processor=self.processor,
            **active_kwargs,
        )
        batch, text_responses = self._generate_with_llm(batch, actor_rollout_wg, agent_active_mask, gen_batch.meta_info)

        decisions = [
            self.parse_response(response) if active else "invalid"
            for response, active in zip(text_responses, agent_active_mask)
        ]
        valids = [decision != 'invalid' for decision in decisions]
        if 'search' in self.config.env.env_name.lower():
            if self.search_feedback is None or len(self.search_feedback) != len(text_responses):
                self.search_feedback = [''] * len(text_responses)
            for i, active in enumerate(agent_active_mask):
                if not active:
                    continue
                if not valids[i]:
                    error = ('End the response with exactly one complete <handoff>CONTINUE</handoff> '
                             'or <handoff>HANDOFF</handoff> outside <think>. Nothing may follow it.')
                else:
                    _, _, error = self.search_action(text_responses[i])
                max_steps = getattr(self.config.env, 'max_steps', None)
                if error and max_steps is not None and step >= max_steps:
                    error += (' This is the final step: provide <answer>short final value</answer>'
                              '<handoff>HANDOFF</handoff>; do not issue a search.')
                self.search_feedback[i] = error
                valids[i] = not bool(error)
        batch.non_tensor_batch['handoff_decision'] = np.array(decisions, dtype=object)
        batch.non_tensor_batch['is_action_valid'] = np.array(valids, dtype=bool)
        batch.non_tensor_batch['env_step'] = np.array([step] * len(text_responses), dtype=object)
        return batch, text_responses
