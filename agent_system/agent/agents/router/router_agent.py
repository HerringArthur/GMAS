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

import logging
import re
from typing import Any, Dict, List, Tuple

import numpy as np
from transformers import PreTrainedTokenizer
from verl import DataProto

from agent_system.agent.agents.base import BaseAgent
from agent_system.agent.registry import AgentRegistry
from agent_system.multi_turn_rollout.utils import preprocess_batch


ROUTER_PROMPT = """
# Task Introduction
{env_prompt}

# Shared Progress at Step {step}
{team_context}

# Your Role
Select exactly one agent from the available agents listed below to perform the
next part of the work. These agents have no fixed roles. Use their previous work
and the remaining work to decide who should act next. You may select the same
agent again if appropriate.

When the task is complete and a worker has already produced a result suitable for
submission, output <route>FINISH</route> to submit that worker result. FINISH does
not generate a new answer. A search request is not a final result. If more work
is needed, choose a worker instead. A worker's HANDOFF alone does not imply that
the task is complete.

You only choose a worker or FINISH; do not solve the problem or issue tool requests.
You may reason within <think> </think> tags, then output
<route>exact agent name</route> or <route>FINISH</route>.
Use an available name exactly as written.
Do not output anything after </route>.
"""

logger = logging.getLogger(__name__)


@AgentRegistry.register("Router Agent")
class RouterAgent(BaseAgent):
    """Select a configured worker or FINISH independently for each active sample."""

    FINISH = "FINISH"

    def __init__(self, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any):
        super().__init__("Router Agent", ROUTER_PROMPT, wg_id, tokenizer, processor, config)
        self.worker_ids = [name for name in config.agent.agent_ids if name != self.name]
        if not self.worker_ids or len(set(self.worker_ids)) != len(self.worker_ids):
            raise ValueError("Router Agent requires at least one worker with a unique name.")
        if self.FINISH in self.worker_ids:
            raise ValueError("FINISH is reserved for submitting the worker result.")

    def build_prompt(self, env_obs: Dict[str, Any], team_context: List[str], step: int) -> Dict[str, Any]:
        obs = super().build_prompt(env_obs, team_context, step)
        available_agents = "\n# Available Agents\n" + "\n".join(f"- {name}" for name in self.worker_ids)
        obs['text'] = [text + available_agents for text in obs['text']]
        return obs

    def parse_response(self, response: str) -> str:
        """Return an allowed worker name, FINISH, or an empty string for invalid text."""
        match = re.fullmatch(r"\s*(?:<think>.*?</think>\s*)?<route>([^<>]+)</route>\s*", response, flags=re.DOTALL)
        if match is None:
            return ""
        selected_agent = match.group(1).strip()
        return selected_agent if selected_agent in self.worker_ids or selected_agent == self.FINISH else ""

    @staticmethod
    def _object_array(values: List[Any]) -> np.ndarray:
        """Build a one-dimensional object array without expanding mappings."""
        result = np.empty(len(values), dtype=object)
        for index, value in enumerate(values):
            result[index] = value
        return result

    def _score_route_candidates_from_log_prob(
        self,
        batch: DataProto,
        text_responses: List[str],
        actor_rollout_wg,
        agent_active_mask: np.ndarray,
    ) -> List[Dict[str, float]]:
        """Score every route completion with the current policy.

        Generation only exposes the log probability of the sampled completion.
        Router credit needs the probability of every candidate, so this helper
        builds one fixed-prompt completion for each candidate and asks the actor
        worker to evaluate their log likelihoods. Backends without
        ``compute_log_prob`` leave the metadata empty instead of fabricating a
        one-hot distribution.
        """
        empty = [{} for _ in text_responses]
        try:
            import torch
        except ImportError:  # pragma: no cover - lightweight test/runtime environments
            return empty
        if not hasattr(actor_rollout_wg, 'compute_log_prob'):
            return empty
        if not hasattr(batch, 'batch') or batch.batch is None:
            return empty
        if 'position_ids' not in batch.batch:
            return empty

        # vLLM/ HF rollouts expose the prompt under ``prompts``.  The naive
        # rollout backend calls the same tensor ``input_ids`` and does not emit
        # a separate ``prompts`` key; in that backend input_ids is prompt-only.
        prompts = batch.batch.get('prompts', batch.batch.get('input_ids'))
        if prompts is None:
            return empty
        prompt_width = prompts.shape[1]
        attention_mask = batch.batch['attention_mask'][:, :prompt_width]
        position_ids = batch.batch['position_ids'][..., :prompt_width]
        candidates = self.worker_ids + [self.FINISH]
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0

        row_specs = []
        for sample_index, (response, active) in enumerate(zip(text_responses, agent_active_mask)):
            if not active:
                continue
            route_start = response.find('<route>')
            if route_start < 0:
                continue
            response_prefix = response[:route_start]
            candidate_ids = []
            for candidate in candidates:
                candidate_text = f'{response_prefix}<route>{candidate}</route>'
                token_ids = self.tokenizer.encode(candidate_text, add_special_tokens=False)
                if not token_ids:
                    candidate_ids = []
                    break
                candidate_ids.append(token_ids)
            if candidate_ids:
                row_specs.append((sample_index, candidate_ids))

        if not row_specs:
            return empty

        response_width = max(len(token_ids) for _, candidate_ids in row_specs for token_ids in candidate_ids)
        input_rows = []
        response_rows = []
        attention_rows = []
        position_rows = []
        source_indices = []
        candidate_indices = []
        token_counts = []

        for sample_index, candidate_ids in row_specs:
            prompt_ids = prompts[sample_index]
            prompt_attention = attention_mask[sample_index]
            prompt_position = position_ids[sample_index]
            for candidate_index, token_ids in enumerate(candidate_ids):
                response_row = torch.full(
                    (response_width,), pad_token_id, dtype=prompt_ids.dtype, device=prompt_ids.device
                )
                response_row[:len(token_ids)] = torch.tensor(
                    token_ids, dtype=prompt_ids.dtype, device=prompt_ids.device
                )
                response_attention = torch.zeros(
                    response_width, dtype=prompt_attention.dtype, device=prompt_attention.device
                )
                response_attention[:len(token_ids)] = 1

                if prompt_position.dim() == 1:
                    prompt_valid = prompt_attention.bool()
                    last_position = (
                        prompt_position[prompt_valid][-1]
                        if prompt_valid.any()
                        else torch.tensor(-1, dtype=prompt_position.dtype, device=prompt_position.device)
                    )
                    response_position = last_position + torch.arange(
                        1, response_width + 1, dtype=prompt_position.dtype, device=prompt_position.device
                    )
                    full_position = torch.cat([prompt_position, response_position], dim=0)
                else:
                    # Multimodal position ids can have multiple rotary rows.
                    prompt_valid = prompt_attention.bool()
                    if prompt_valid.any():
                        last_position = prompt_position[:, prompt_valid][:, -1]
                    else:
                        last_position = torch.zeros(
                            prompt_position.shape[0], dtype=prompt_position.dtype, device=prompt_position.device
                        )
                    delta = torch.arange(
                        1, response_width + 1, dtype=prompt_position.dtype, device=prompt_position.device
                    ).unsqueeze(0)
                    response_position = last_position.unsqueeze(1) + delta
                    full_position = torch.cat([prompt_position, response_position], dim=1)

                input_rows.append(torch.cat([prompt_ids, response_row], dim=0))
                response_rows.append(response_row)
                attention_rows.append(torch.cat([prompt_attention, response_attention], dim=0))
                position_rows.append(full_position)
                source_indices.append(sample_index)
                candidate_indices.append(candidate_index)
                token_counts.append(len(token_ids))

        candidate_non_tensors = {}
        for key in ('multi_modal_inputs', 'multi_modal_data'):
            if key in batch.non_tensor_batch:
                candidate_non_tensors[key] = self._object_array(
                    [batch.non_tensor_batch[key][sample_index] for sample_index in source_indices]
                )
        candidate_batch = DataProto.from_dict(
            tensors={
                'prompts': torch.stack([prompts[i] for i in source_indices]),
                'responses': torch.stack(response_rows),
                'input_ids': torch.stack(input_rows),
                'attention_mask': torch.stack(attention_rows),
                'position_ids': torch.stack(position_rows),
            },
            non_tensors=candidate_non_tensors,
            meta_info=dict(batch.meta_info),
        )
        scored = actor_rollout_wg.compute_log_prob(candidate_batch)
        log_probs = scored.batch.get('old_log_probs', scored.batch.get('log_probs'))
        if log_probs is None:
            return empty

        score_table = [[None for _ in candidates] for _ in text_responses]
        for row_index, (sample_index, candidate_index) in enumerate(zip(source_indices, candidate_indices)):
            token_count = token_counts[row_index]
            score = float(log_probs[row_index, :token_count].sum().detach().cpu().item())
            score_table[sample_index][candidate_index] = score

        probabilities = empty
        for sample_index, scores in enumerate(score_table):
            if any(score is None for score in scores):
                continue
            score_tensor = torch.tensor(scores, dtype=torch.float64)
            probs = torch.softmax(score_tensor, dim=0).tolist()
            probabilities[sample_index] = {
                candidate: float(probability)
                for candidate, probability in zip(candidates, probs)
            }
        return probabilities

    def _route_probabilities(
        self,
        batch: DataProto,
        text_responses: List[str],
        actor_rollout_wg,
        agent_active_mask: np.ndarray,
    ) -> List[Dict[str, float]]:
        """Normalize candidate log-likelihoods for workers and FINISH.

        Use compute_log_prob to score candidates. Empty mappings indicate
        inactive samples or unavailable scores.
        """
        try:
            return self._score_route_candidates_from_log_prob(
                batch, text_responses, actor_rollout_wg, agent_active_mask
            )
        except Exception:  # pragma: no cover - backend-specific failure
            logger.exception('Router log-probability scoring failed; leaving route probabilities empty.')
            return [{} for _ in text_responses]

    def call(
        self,
        gen_batch: DataProto,
        env_obs: Dict[str, Any],
        team_context: List[str],
        actor_rollout_wg,
        agent_active_mask,
        step: int,
    ) -> Tuple[DataProto, List[str]]:
        """Return full router text and the selected agent in non-tensor metadata."""
        obs = self.build_prompt(env_obs, team_context, step)
        batch = preprocess_batch(
            gen_batch=gen_batch,
            obs=obs,
            config=self.config,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
        batch, text_responses = self._generate_with_llm(batch, actor_rollout_wg, agent_active_mask, gen_batch.meta_info)
        selected_agents = [
            self.parse_response(response) if active else ""
            for response, active in zip(text_responses, agent_active_mask)
        ]
        batch.non_tensor_batch['selected_agent'] = np.array(selected_agents, dtype=object)
        batch.non_tensor_batch['route_probabilities'] = self._object_array(
            self._route_probabilities(batch, text_responses, actor_rollout_wg, agent_active_mask)
        )
        batch.non_tensor_batch['is_action_valid'] = np.array([bool(name) for name in selected_agents], dtype=bool)
        batch.non_tensor_batch['env_step'] = np.array([step] * len(text_responses), dtype=object)
        return batch, text_responses
