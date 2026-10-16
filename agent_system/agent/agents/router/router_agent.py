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


ROUTER_TASK_PROMPT = """
# Your Role
You are the router. Review the original task and shared work, then select a worker
or FINISH. Workers have no fixed roles. You may select the same worker again.

Every valid SubAgent response, including CONTINUE and HANDOFF, returns control to
you. Decide using the current progress:
- If only intermediate work exists, identify what remains and select a worker.
- If a candidate answer exists, inspect the attached evidence and supporting
  work. If it directly answers the question and is sufficiently supported, use
  <route>FINISH</route>.
- If support is missing, weak, or needs independent verification, state the defect
  and select a worker. After HANDOFF with a candidate, prefer a different worker
  for independent verification when possible.
- Avoid further worker calls or searches once the candidate is supported and ready.

When selecting a worker, put its next task or correction in one brief <think>
</think> block before the route. Repeated candidate text requires an explicit
evidence review; reselect a worker when no alternative exists.

HANDOFF and intermediate work do not imply completion. FINISH submits an existing
candidate; it does not create or rewrite an answer. Do not produce the answer or
issue a tool request yourself.

Output exactly one <route>exact agent name</route> or <route>FINISH</route>, with
an optional single <think> block before it and nothing after </route>. Use exact
configured names and tag capitalization; do not use Markdown.
"""


ROUTER_PROMPT = ROUTER_TASK_PROMPT


ROUTER_MATH_PROMPT = ROUTER_PROMPT + r"""
# Candidate Format
A Math candidate must contain a complete, nonempty \boxed{...} outside <think>.
Intermediate derivations are shared work, not candidate answers. Check that the
candidate answers the original problem and is supported by the derivation.
"""


ROUTER_SEARCH_PROMPT = ROUTER_TASK_PROMPT + """
# Candidate Format
A Search candidate must contain a complete, nonempty <answer> </answer> block
outside <think>. Check that it answers the original question and is supported by
sufficient evidence. A search request or a claim of completion inside <think> is
not a candidate answer. If no candidate exists, select a worker to produce one.
"""


@AgentRegistry.register("Router Agent")
class RouterAgent(BaseAgent):
    """Select a configured worker or FINISH independently for each active sample."""

    FINISH = "FINISH"
    RESPONSE_PATTERN = re.compile(
        r"\s*(?:<think>((?:(?!</?think>).)*)</think>\s*)?<route>([^<>]+)</route>\s*",
        flags=re.DOTALL,
    )
    # Models occasionally capitalize XML-like tags or append a natural-language
    # explanation after a complete route.  Those outputs still contain one
    # unambiguous route and can be reduced to the canonical protocol safely.
    COMPAT_RESPONSE_PATTERN = re.compile(
        r"\s*(?:<think>((?:(?!</?think>).)*)</think>\s*)?"
        r"<route>\s*([^<>]+?)\s*</route>(.*)",
        flags=re.DOTALL | re.IGNORECASE,
    )

    def __init__(self, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any):
        env_name = config.env.env_name.lower()
        if 'search' in env_name:
            prompt = ROUTER_SEARCH_PROMPT
        elif 'math' in env_name:
            prompt = ROUTER_MATH_PROMPT
        else:
            prompt = ROUTER_PROMPT
        super().__init__("Router Agent", prompt, wg_id, tokenizer, processor, config)
        # The bounded Search builder reads this field; Math uses it below too.
        self.search_feedback = None
        self.worker_ids = [name for name in config.agent.agent_ids if name != self.name]
        if not self.worker_ids or len(set(self.worker_ids)) != len(self.worker_ids):
            raise ValueError("Router Agent requires at least one worker with a unique name.")
        
    def reset(self):
        self.search_feedback = None
        self._context_token_cache = None

    def build_prompt(self, env_obs: Dict[str, Any], team_context: List[str], step: int,
                     agent_active_mask=None, collaboration_context=None) -> Dict[str, Any]:
        available_agents = "\n# Valid Routing Outputs\n" + "\n".join(
            f"- <route>{name}</route>" for name in self.worker_ids
        ) + "\n- <route>FINISH</route>\n\nChoose exactly one routing output."
        if 'search' in self.config.env.env_name.lower():
            return build_search_observation(self, env_obs, team_context, step, self.prompt, available_agents,
                                            agent_active_mask=agent_active_mask,
                                            collaboration_context=collaboration_context)
        return build_context_observation(self, env_obs, team_context, step, self.prompt, available_agents,
                                         agent_active_mask=agent_active_mask,
                                         collaboration_context=collaboration_context)

    def parse_response(self, response: str) -> str:
        """Return an allowed worker name, FINISH, or an empty string for invalid text."""
        match = self.RESPONSE_PATTERN.fullmatch(response)
        if match is None:
            return ""
        selected_agent = match.group(2).strip()
        return selected_agent if selected_agent in self.worker_ids or selected_agent == self.FINISH else ""

    def normalize_response(self, response: str) -> Tuple[str, bool]:
        """Reduce recoverable route variants to the canonical route protocol.

        The generated text is still kept in the rollout tokens.  The normalized
        text is used for orchestration and shared context so a capitalization
        difference or trailing explanation does not turn an otherwise
        unambiguous route into an expensive retry. Ambiguous output
        (multiple routes, a route inside reasoning, or an unknown name) remains
        invalid and is returned unchanged.
        """
        if self.parse_response(response):
            return response, False

        match = self.COMPAT_RESPONSE_PATTERN.fullmatch(response)
        if match is None:
            return response, False
        reasoning, selected_agent, trailing = match.groups()
        # A second route in the discarded suffix makes the response ambiguous.
        if re.search(r"</?route\b", trailing, flags=re.IGNORECASE):
            return response, False
        selected_agent = selected_agent.strip()
        if selected_agent not in self.worker_ids and selected_agent != self.FINISH:
            return response, False

        canonical = ''
        if reasoning is not None:
            canonical = f'<think>{reasoning.strip()}</think>'
        canonical += f'<route>{selected_agent}</route>'
        # A canonical response is always accepted by the strict parser.  The
        # tail is intentionally dropped because it is outside the route grammar
        # and often contains an accidental answer or handoff marker.
        return canonical, True

    def routing_instruction(self, response: str) -> str:
        """Keep the instruction verbatim for a valid worker selection, if present."""
        if self.parse_response(response) not in self.worker_ids:
            return ''
        return (self.RESPONSE_PATTERN.fullmatch(response).group(1) or '').strip()

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
        """Return full router text and the selected agent in non-tensor metadata."""
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
            normalized, repaired = self.normalize_response(response)
            normalized_responses.append(normalized)
            protocol_repaired.append(repaired)
        text_responses = normalized_responses
        selected_agents = [
            self.parse_response(response) if active else ""
            for response, active in zip(text_responses, agent_active_mask)
        ]
        if self.search_feedback is None or len(self.search_feedback) != len(text_responses):
            self.search_feedback = [''] * len(text_responses)
        for i, active in enumerate(agent_active_mask):
            if active:
                if not selected_agents[i]:
                    self.search_feedback[i] = (
                        'Output one complete <route>configured worker name</route> or '
                        '<route>FINISH</route>, optionally preceded by one complete <think> block. '
                        'Use a worker name from the available list and output nothing after </route>.'
                    )
                elif protocol_repaired[i]:
                    self.search_feedback[i] = (
                        'Your previous route was executable but had a protocol mismatch. '
                        'Use exactly one canonical <route>...</route> as the final text, '
                        'with no explanation or Markdown after it.'
                    )
                else:
                    self.search_feedback[i] = ''
        batch.non_tensor_batch['selected_agent'] = np.array(selected_agents, dtype=object)
        batch.non_tensor_batch['routing_instruction'] = np.array([
            self.routing_instruction(response) if active else ''
            for response, active in zip(text_responses, agent_active_mask)
        ], dtype=object)
        batch.non_tensor_batch['router_feedback'] = np.array([
            feedback if active else '' for feedback, active in zip(self.search_feedback, agent_active_mask)
        ], dtype=object)
        batch.non_tensor_batch['protocol_repaired'] = np.array(protocol_repaired, dtype=bool)
        # A repaired route is executable for this rollout, but remains marked
        # protocol-invalid so training can teach the model the canonical form.
        batch.non_tensor_batch['is_action_valid'] = np.array(
            [bool(name) and not repaired for name, repaired in zip(selected_agents, protocol_repaired)],
            dtype=bool,
        )
        batch.non_tensor_batch['env_step'] = np.array([step] * len(text_responses), dtype=object)
        return batch, text_responses
