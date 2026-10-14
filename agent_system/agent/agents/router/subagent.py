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
from agent_system.agent.agents.router.context import build_context_observation, ensure_prompt_fits


SUBAGENT_TASK_PROMPT = """
# Your Task
You are a worker in a team. You have no fixed role. Read the original task, shared
progress, and available tool results, then advance the remaining work.

You may provide intermediate findings and identify what remains to be done, with
enough detail for yourself or another worker to continue. When the task is solved,
provide the final answer in the environment's required format and return control
to the router for review. The final answer must directly and fully answer the
original task, preserve essential qualifiers, and agree with your reasoning.
Put reasoning in <think> </think>. Follow the environment-specific rules below.

Include nonempty work and end every response with exactly one control tag. Both
control values return control to the Router for review:
- <handoff>CONTINUE</handoff> to indicate that more work may be useful.
- <handoff>HANDOFF</handoff> to request a new assignment or final review. This can
  accompany partial work or a final answer; HANDOFF alone does not mean the task is solved.

After a final answer, use HANDOFF for review. Place the control tag outside all
other tags and output nothing after </handoff>.

Protocol examples (copy the tag spelling and order exactly):
  <search>one query</search><handoff>CONTINUE</handoff>
  <answer>short final answer</answer><handoff>HANDOFF</handoff>
  <think>intermediate reasoning</think><handoff>CONTINUE</handoff>
Never use Markdown bullets around protocol tags, change tag capitalization, emit
more than one search/answer block, or write anything after </handoff>.
"""


SUBAGENT_PROMPT = SUBAGENT_TASK_PROMPT + """
# Output Format
Use the tool and final-answer formats specified by the task. Intermediate work
does not need a final-answer marker. A claim of completion inside <think> is not
a final answer; put the final answer outside <think> in the required format.
"""


SUBAGENT_MATH_PROMPT = SUBAGENT_PROMPT + r"""
# Math Output
Put the final answer in \boxed{...} outside <think>, followed by
<handoff>HANDOFF</handoff>. Reserve \boxed{...} for the final answer to the
original problem; leave intermediate results unboxed.
"""


SUBAGENT_SEARCH_PROMPT = SUBAGENT_TASK_PROMPT + """
# Search Output
Assess what the evidence establishes and what work remains.
- If more reasoning or coordination is needed, provide intermediate work and a
  control tag. No search or answer block is required for intermediate work.
- If information is missing, output <search>targeted query</search> outside
  <think>. Do not repeat a query whose results already answer the question.
  Use either control tag to return to the Router after the search; the Router
  decides whether to reselect a worker or finish.
- As soon as the evidence and reasoning support a complete answer, output
  <answer>final answer</answer> outside <think>, followed by HANDOFF. Answer the
  original question concisely and completely. Put explanations in <think>.

Use at most one search or answer block per response. Do not output both. A claim
of completion or an answer inside <think> does not replace a final <answer> block.
"""


SUBAGENT_SEARCH_FINAL_PROMPT = SUBAGENT_TASK_PROMPT + """
# Final answer required
This is the last environment step. Answer the original question using the
retrieved evidence and your knowledge. Give the best-supported answer available,
even if some uncertainty remains. Another search cannot return any evidence.
You are still the current worker; return the answer to the router for review.

Output a brief <think>reasoning</think> if needed, followed by exactly:
<answer>final answer</answer><handoff>HANDOFF</handoff>

Inside <answer>, answer the original question concisely and completely, preserving
essential qualifiers and agreeing with your reasoning. Put explanations in
<think>. Intermediate work alone is insufficient on this step. Do not issue a
search or CONTINUE. Output nothing after </handoff>.
"""


@AgentRegistry.register("Sub Agent")
class SubAgent(BaseAgent):
    """A role-free worker that completes partial work and chooses to continue or hand off."""

    def __init__(self, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any, name: str = "Sub Agent"):
        env_name = config.env.env_name.lower()
        if 'search' in env_name:
            prompt = SUBAGENT_SEARCH_PROMPT
        elif 'math' in env_name:
            prompt = SUBAGENT_MATH_PROMPT
        else:
            prompt = SUBAGENT_PROMPT
        super().__init__(name, prompt, wg_id=wg_id, tokenizer=tokenizer, processor=processor, config=config)
        self.search_feedback = None

    def reset(self):
        self.search_feedback = None
        self._context_token_cache = None

    def build_prompt(self, env_obs: Dict[str, Any], team_context: List[str], step: int,
                     agent_active_mask=None, collaboration_context=None) -> Dict[str, Any]:
        if 'search' in self.config.env.env_name.lower():
            max_steps = getattr(self.config.env, 'max_steps', None)
            prompt = SUBAGENT_SEARCH_FINAL_PROMPT if max_steps is not None and step >= max_steps else self.prompt
            return build_search_observation(self, env_obs, team_context, step, prompt,
                                            agent_active_mask=agent_active_mask,
                                            collaboration_context=collaboration_context)
        return build_context_observation(self, env_obs, team_context, step, self.prompt,
                                         agent_active_mask=agent_active_mask,
                                         collaboration_context=collaboration_context)

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
        reasoning = re.findall(r"<think>(.*?)</think>", progress, flags=re.DOTALL)
        if any(re.search(r"</?think>", block) for block in reasoning):
            return "invalid"
        progress = re.sub(r"<think>.*?</think>", "", progress, flags=re.DOTALL)
        if re.search(r"</?(?:think|handoff)>", progress):
            return "invalid"
        if not progress.strip() and not any(block.strip() for block in reasoning):
            return "invalid"
        return decision

    @staticmethod
    def _canonicalize_tags(response: str) -> str:
        """Canonicalize tag capitalization without changing tag contents."""
        for tag in ('think', 'search', 'answer', 'handoff'):
            response = re.sub(
                rf'</?{tag}\s*>',
                lambda match, tag=tag: (
                    f'</{tag}>' if match.group(0).startswith('</') else f'<{tag}>'
                ),
                response,
                flags=re.IGNORECASE,
            )
        return response

    @staticmethod
    def normalize_response(response: str, search_enabled: bool = False) -> Tuple[str, bool]:
        """Repair only unambiguous worker outputs at the orchestration boundary.

        A model often stops immediately after a complete search or answer block,
        even though the prompt asks for a trailing handoff tag. Such a response
        has enough structure to execute a search and continue with the same
        worker. Answers and intermediate work still require an explicit handoff
        because silently submitting a candidate changes the training target.
        Incomplete or ambiguous actions are
        left untouched so they still receive correction feedback and remain
        visible as invalid training rows.
        """
        canonical = SubAgent._canonicalize_tags(response)
        if SubAgent.parse_response(canonical) != 'invalid':
            if search_enabled:
                _, _, formatting_error = SubAgent.search_action(canonical)
                if formatting_error:
                    # Keep malformed actions invalid so correction feedback is
                    # shown and the raw form cannot enter shared history.
                    return canonical, False
            return canonical, canonical != response

        # A foreign control tag must not be reinterpreted as worker progress.
        if re.search(r'</?route\b', canonical, flags=re.IGNORECASE):
            return response, False
        if re.search(r'</?handoff\b', canonical, flags=re.IGNORECASE):
            # A handoff that is not a complete terminal tag is ambiguous.
            return response, False

        action = ''
        if search_enabled:
            # A missing handoff can be repaired, including a leading Markdown
            # list marker, but the repaired text must itself be canonical.
            repair_source = re.sub(
                r'(?m)^(\s*)(?:[-*•]|\d+[.)])\s+(?=<(?:search|answer)\b)',
                r'\1',
                canonical,
            )
            action, is_search, error = SubAgent.search_action(repair_source)
            if error:
                return response, False
            if not action or not is_search:
                return response, False
            decision = 'CONTINUE'
        else:
            return response, False

        repaired = repair_source.rstrip() + f'<handoff>{decision}</handoff>'
        if SubAgent.parse_response(repaired) == 'invalid':
            return response, False
        return repaired, True

    @staticmethod
    def search_action(response: str) -> Tuple[str, bool, str]:
        """Extract a Search action or accept intermediate work without an action.

        An empty action with no error is valid intermediate work, not a final
        candidate. Control-tag validity is checked separately by parse_response.
        """
        work = re.sub(r'<handoff>\s*(CONTINUE|HANDOFF)\s*</handoff>\s*$', '', response,
                      flags=re.IGNORECASE).strip()
        visible = re.sub(r'<think>.*?</think>', '', work, flags=re.DOTALL | re.IGNORECASE)
        blocks = list(re.finditer(r'<(search|answer)>(.*?)</\1>', visible,
                                  flags=re.DOTALL | re.IGNORECASE))
        if not blocks:
            if re.search(r'</?(?:search|answer)\b', visible, flags=re.IGNORECASE):
                return '', False, 'Complete the search or answer tags, or remove them when sharing intermediate work.'
            if not re.sub(r'</?think>', '', work, flags=re.IGNORECASE).strip():
                return '', False, 'Provide nonempty intermediate work, a search query, or a final answer before the control tag.'
            return '', False, ''
        if len(blocks) > 1:
            return '', False, ('Output exactly one complete <search>query</search> OR '
                               '<answer>final answer</answer> outside <think> when issuing an action. '
                               'Do not output multiple actions in one response.')
        block = blocks[0]
        tag, content = block.groups()
        remainder = visible[:block.start()] + visible[block.end():]
        if re.search(r'</?(?:search|answer)\b', remainder + content, flags=re.IGNORECASE):
            return '', False, 'Remove extra, nested, or incomplete action tags. Output only one complete search or answer block.'
        prefix = visible[:block.start()]
        suffix = visible[block.end():]
        if re.match(r'^\s*(?:[-*•]|\d+[.)])\s*', prefix):
            return '', False, 'Remove list markers around the action; emit one canonical action tag.'
        if suffix.strip():
            return '', False, 'Put explanations inside <think> and output nothing after the action block.'
        if not content.strip():
            return '', False, 'The action is empty. Supply a search query or a short final answer inside its tags.'
        tag = tag.lower()
        return f'<{tag}>{content.strip()}</{tag}>', tag == 'search', ''

    def call(
        self,
        gen_batch: DataProto,
        env_obs: Dict[str, Any],
        team_context: List[str],
        actor_rollout_wg,
        agent_active_mask,
        step: int,
        collaboration_context=None,
    ) -> Tuple[DataProto, List[str]]:
        """Generate partial work; the orchestra handles continuation and handoffs.

        Return full responses, including compatibility tags, to preserve progress.
        Store CONTINUE, HANDOFF, or invalid in ``handoff_decision`` in the batch's
        non-tensor metadata. Search/answer tags do not determine this decision.
        """
        active_kwargs = {'agent_active_mask': agent_active_mask}
        obs = self.build_prompt(env_obs, team_context, step, collaboration_context=collaboration_context, **active_kwargs)
        ensure_prompt_fits(self, gen_batch, obs, agent_active_mask)
        batch = preprocess_batch(
            gen_batch=gen_batch,
            obs=obs,
            config=self.config,
            tokenizer=self.tokenizer,
            processor=self.processor,
            **active_kwargs,
        )
        batch, text_responses = self._generate_with_llm(batch, actor_rollout_wg, agent_active_mask, gen_batch.meta_info)
        normalized_responses = []
        protocol_repaired = []
        for response in text_responses:
            normalized, repaired = self.normalize_response(
                response,
                search_enabled='search' in self.config.env.env_name.lower(),
            )
            normalized_responses.append(normalized)
            protocol_repaired.append(repaired)
        text_responses = normalized_responses

        decisions = [
            self.parse_response(response) if active else "invalid"
            for response, active in zip(text_responses, agent_active_mask)
        ]
        protocol_valids = [decision != 'invalid' for decision in decisions]
        valids = protocol_valids.copy()
        if 'search' in self.config.env.env_name.lower():
            if self.search_feedback is None or len(self.search_feedback) != len(text_responses):
                self.search_feedback = [''] * len(text_responses)
            max_steps = getattr(self.config.env, 'max_steps', None)
            final_step = max_steps is not None and step >= max_steps
            for i, active in enumerate(agent_active_mask):
                if not active:
                    continue
                if not protocol_valids[i]:
                    error = ('Provide nonempty work with complete, non-nested <think> tags if used, '
                             'then exactly one <handoff>CONTINUE</handoff> or '
                             '<handoff>HANDOFF</handoff> outside <think>. Nothing may follow it.')
                else:
                    action, is_search, error = self.search_action(text_responses[i])
                    if final_step and not error and (not action or is_search or decisions[i] != 'HANDOFF'):
                        error = 'This response does not satisfy the final-step answer requirement.'
                if error and final_step:
                    error += (' This is the final step: provide <answer>final answer</answer>'
                              '<handoff>HANDOFF</handoff>; intermediate work alone, searches, '
                              'and CONTINUE are not allowed.')
                # A syntactic repair is executable only when the resulting
                # response also satisfies the environment-step semantics.
                if error:
                    protocol_repaired[i] = False
                self.search_feedback[i] = error or (
                    'Your previous response was executable but had a protocol mismatch. '
                    'Use exactly one canonical action followed by <handoff>CONTINUE</handoff> '
                    'or <handoff>HANDOFF</handoff>; do not use Markdown bullets or change tag capitalization.'
                    if protocol_repaired[i] else ''
                )
                valids[i] = not bool(error)
        else:
            # Math and other non-search workers also need explicit correction
            # feedback now that invalid text is hidden from shared history.
            if self.search_feedback is None or len(self.search_feedback) != len(text_responses):
                self.search_feedback = [''] * len(text_responses)
            for i, active in enumerate(agent_active_mask):
                if not active:
                    continue
                if not protocol_valids[i]:
                    self.search_feedback[i] = (
                        'Your previous response was invalid and has been omitted from shared history. '
                        'Provide nonempty work, then exactly one <handoff>CONTINUE</handoff> or '
                        '<handoff>HANDOFF</handoff> outside any <think> block, with nothing after it.'
                    )
                elif protocol_repaired[i]:
                    self.search_feedback[i] = (
                        'Use the canonical handoff protocol exactly; the repaired response was omitted '
                        'from shared history and will not receive protocol credit.'
                    )
                else:
                    self.search_feedback[i] = ''
        batch.non_tensor_batch['handoff_decision'] = np.array(decisions, dtype=object)
        batch.non_tensor_batch['protocol_repaired'] = np.array(protocol_repaired, dtype=bool)
        # Repaired responses can be executed to preserve the trajectory, but are
        # kept invalid for PPO so the model is trained toward the canonical
        # protocol rather than receiving credit for a synthetic handoff.
        batch.non_tensor_batch['is_action_valid'] = np.array(
            [valid and not repaired for valid, repaired in zip(valids, protocol_repaired)],
            dtype=bool,
        )
        batch.non_tensor_batch['env_step'] = np.array([step] * len(text_responses), dtype=object)
        return batch, text_responses
