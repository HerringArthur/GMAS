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

"""Role-based routing using the shared Orchestra call/update/buffer interface.

Agent contract:

* The router agent package registers Router Agent and each configured role with
  AgentRegistry so BaseOrchestra can create them by name.
* All agents expose the usual ``call(...) -> (DataProto, List[str])`` interface
  and return a full-size batch with ``is_action_valid``, ``agent_active_mask``,
  ``wg_id`` and ``env_step`` for each input row. Inactive rows are invalid.
  Each call receives the latest ``candidate_answers`` separately from history,
  so agents can read a candidate even when its original history entry is absent.
* Agents parse one model-generated function call and attach action_name,
  action_args and action_error without altering sampled response tokens.
  RouterAgent maps call_agent/finish to selected_role for existing credit.
  Role responses are rendered for context and the existing environment API.
  Router task arguments are passed explicitly through the role context.
  Execution and state checks happen here, after parsing, with errors fed back
  to the next Router decision. Reviews retain their candidate/environment version.
  Orchestra attaches ``parent_prefix_key`` to each batch for Router credit.
  Role credit uses the task and role identity, independently of this prefix.
  Valid role responses and executable SEARCH queries update team_context. Only
  SOLVE/REVISE in math and SEARCH/ANSWER in search update text_actions. Answer
  roles also save candidate_answers, which FINISH copies into text_actions.
  An executable SEARCH query pauses that sample for environment execution.

Example configuration, alongside a matching agent-to-worker-group mapping::

    agent:
      agent_ids: ["Router Agent", "SOLVE", "VERIFY", "REVISE"]
      role_ids: ["SOLVE", "VERIFY", "REVISE"]
      model_sharing: false
      orchestra:
        router:
          max_loop_num: 4
          max_context_runs: 2

Role prompts and output validation belong to the agents. Credit is computed
in the trainer from collected trajectory rewards and routing metadata.
"""

import importlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from transformers import PreTrainedTokenizer

from verl import DataProto

from agent_system.agent.orchestra.base import BaseOrchestra


def update_team_context(
    agent_id: str,
    team_context: List[List[str]],
    text_response: List[str],
    agent_active_mask: Optional[np.ndarray] = None,
    *,
    step: int,
    max_context_runs: int,
) -> List[List[str]]:
    """Keep the latest runs directly in each sample's team context.

    Router starts a run; its selected role appends to the same entry. Invalid
    actions contribute execution feedback, and attempts still count.
    """
    if agent_active_mask is None:
        agent_active_mask = np.ones(len(team_context), dtype=bool)

    for i in range(len(team_context)):
        if agent_active_mask[i]:
            if agent_id == "Router Agent":
                team_context[i].append(f'\n[Environment step {step}]\n' if text_response[i] else "")
            if text_response[i]:
                label = "Execution feedback for" if text_response[i].startswith("ActionError:") else "The output of"
                team_context[i][-1] += f'\n{label} "{agent_id}": {text_response[i]}\n'
            team_context[i] = team_context[i][-max_context_runs:]
    return team_context


def update_text_action(
    text_actions: List[str],
    text_response: List[str],
    agent_active_mask: Optional[np.ndarray] = None,
) -> List[str]:
    """Replace selected entries in an action or candidate-answer list in place.

    The caller chooses which roles may update the list. Sources can be fresh
    responses or saved candidates submitted by FINISH or the round limit.
    """
    if agent_active_mask is None:
        agent_active_mask = np.ones(len(text_actions), dtype=bool)

    for i in range(len(text_actions)):
        if agent_active_mask[i]:
            text_actions[i] = text_response[i]
    return text_actions


class RouterMultiAgentOrchestra(BaseOrchestra):
    """Let each active sample choose its next role independently.

    Each run() performs one Router call and at most one role call per sample.
    The rollout collector repeats run() and owns the round budget. Each sample
    retains its latest max_context_runs completed runs across environment steps.
    During a run, the selected role also sees the current Router assignment.
    Candidate answers persist independently until reset() starts a new rollout.
    Role paths also persist for Router credit, independently of history eviction.
    initialize_context() clears pending actions and return reasons for each
    environment step. FINISH and executable SEARCH queries mark samples ready for
    the next environment interaction.
    """

    ROUTER_AGENT = "Router Agent"
    FINISH = "FINISH"

    def __init__(
        self,
        agent_ids: List[str],
        role_ids: List[str],
        model_ids: List[str],
        agents_to_wg_mapping: Dict[str, str],
        tokenizers: Dict[str, PreTrainedTokenizer] = None,
        processors: Dict[str, Any] = None,
        config: Any = None,
    ):
        importlib.import_module("agent_system.agent.agents.router")
        super().__init__(
            agent_ids=agent_ids,
            model_ids=model_ids,
            agents_to_wg_mapping=agents_to_wg_mapping,
            tokenizers=tokenizers,
            processors=processors,
            config=config,
        )
        self.role_ids = role_ids
        self.max_context_runs = config.agent.orchestra.router.max_context_runs
        self.team_context = None
        self.text_actions = None
        self.candidate_answers = None
        self.return_reasons = None
        self.role_paths = None
        self.candidate_versions = None
        self.reviews = None

    def reset(self):
        super().reset()
        self.team_context = None
        self.text_actions = None
        self.candidate_answers = None
        self.return_reasons = None
        self.role_paths = None
        self.candidate_versions = None
        self.reviews = None

    def save_to_buffer(self, name: str, batch: DataProto):
        # All role rows need the same metadata columns when the collector
        # collates them. Non-SEARCH calls never carry an executable query.
        batch.non_tensor_batch.setdefault(
            "search_executable",
            np.zeros_like(batch.non_tensor_batch["is_action_valid"], dtype=bool),
        )
        batch.non_tensor_batch["candidate_version"] = np.array(self.candidate_versions, dtype=object)
        batch.non_tensor_batch["candidate_review"] = np.fromiter(
            (dict(review) if review is not None else None for review in self.reviews),
            dtype=object, count=len(self.reviews),
        )
        super().save_to_buffer(name, batch)

    def initialize_context(self, env_obs):
        """Start an environment step, retaining team context and candidate answers."""
        text_actions, _, env_obs = super().initialize_context(env_obs)
        if self.team_context is None:
            self.team_context = [[] for _ in text_actions]
        if self.candidate_answers is None:
            self.candidate_answers = [""] * len(text_actions)
        if self.role_paths is None:
            self.role_paths = [()] * len(text_actions)
        if self.candidate_versions is None:
            self.candidate_versions = [0] * len(text_actions)
            self.reviews = [None] * len(text_actions)
        self.text_actions = text_actions
        self.return_reasons = np.full(len(self.text_actions), "", dtype=object)
        return self.text_actions, ["".join(runs) for runs in self.team_context], env_obs

    @staticmethod
    def _feedback(batch, responses, mask):
        """Keep execution errors visible without treating them as model output."""
        metadata = batch.non_tensor_batch
        return [
            (responses[i] if metadata["is_action_valid"][i]
             else "ActionError: " + metadata["action_error"][i]) if active else ""
            for i, active in enumerate(mask)
        ]

    def run(
        self,
        gen_batch: DataProto,
        env_obs: Dict[str, Any],
        actor_rollout_wgs,
        active_masks: np.ndarray,
        step: int,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        self.reset_buffer()
        if self.team_context is None:
            self.initialize_context(env_obs)
        pending_mask = active_masks & (self.return_reasons == "")
        if not pending_mask.any():
            return self.text_actions.copy(), self.multiagent_batch_buffer

        # 1. Route each pending sample once, using work from preceding calls.
        prompt_context = ["".join(runs) for runs in self.team_context]
        for i, review in enumerate(self.reviews):
            if review is not None:
                prompt_context[i] += (
                    "\n# Latest Recorded Review\n"
                    f"Reviewed candidate version: {review['candidate_version']}; "
                    f"current candidate version: {self.candidate_versions[i]}. "
                    f"Reviewed environment step: {review['env_step']}; current step: {step}.\n"
                    f"Assessment: {review['assessment']}\n"
                    "A review of an earlier candidate or evidence state must be reassessed.\n"
                )
        actor_rollout_wg = actor_rollout_wgs[self.agents_to_wg_mapping[self.ROUTER_AGENT]]
        router_batch, router_responses = self.agents[self.ROUTER_AGENT].call(
            gen_batch=gen_batch,
            env_obs=env_obs,
            team_context=prompt_context,
            candidate_answers=self.candidate_answers,
            actor_rollout_wg=actor_rollout_wg,
            agent_active_mask=pending_mask,
            step=step,
        )
        selections = router_batch.non_tensor_batch["selected_role"]
        finish_mask = selections == self.FINISH
        route_mask = pending_mask & router_batch.non_tensor_batch["is_action_valid"]
        scene = self.config.env.env_name
        metadata = router_batch.non_tensor_batch
        for i in np.flatnonzero(route_mask):
            error = ""
            if selections[i] == self.FINISH and not self.candidate_answers[i].strip():
                error = "finish() requires a saved candidate answer; call an answer-producing role first."
            elif scene == "search" and step == self.config.env.max_steps and selections[i] == "SEARCH":
                error = "SEARCH is unavailable on the final environment step."
            if error:
                route_mask[i] = False
                metadata["action_error"][i] = error
                metadata["execution_status"][i] = "state_error"
            else:
                metadata["execution_status"][i] = "submitted" if finish_mask[i] else "dispatched"
        router_batch.non_tensor_batch["is_action_valid"] = route_mask
        parent_prefix_keys = np.fromiter(self.role_paths, dtype=object, count=len(self.role_paths))
        router_batch.non_tensor_batch["parent_prefix_key"] = parent_prefix_keys
        self.team_context = update_team_context(
            self.ROUTER_AGENT, self.team_context,
            self._feedback(router_batch, router_responses, pending_mask),
            agent_active_mask=pending_mask, step=step, max_context_runs=self.max_context_runs,
        )
        # The role sees the same prior runs as the Router plus its new assignment,
        # even though stored context has already evicted the oldest run.
        prompt_context = [
            context + self.team_context[i][-1] if pending_mask[i] else context
            for i, context in enumerate(prompt_context)
        ]
        for i in np.flatnonzero(route_mask & ~finish_mask):
            task = metadata["routing_instruction"][i]
            prompt_context[i] += f"\n# Router Assignment\nSelected role: {selections[i]}\nTask: {task or 'Perform your role for the current task.'}\n"
        self.save_to_buffer(self.ROUTER_AGENT, router_batch)
        submit_mask = route_mask & finish_mask
        self.text_actions = update_text_action(self.text_actions, self.candidate_answers, submit_mask)
        self.return_reasons[submit_mask] = "finish"

        # 2. Dispatch each sample to its selected role. This traversal groups
        # batch rows by model; it does not run additional collaboration rounds.
        for role in self.role_ids:
            role_mask = route_mask & (selections == role)
            if not role_mask.any():
                continue
            actor_rollout_wg = actor_rollout_wgs[self.agents_to_wg_mapping[role]]
            batch, responses = self.agents[role].call(
                gen_batch=gen_batch,
                env_obs=env_obs,
                team_context=prompt_context,
                candidate_answers=self.candidate_answers,
                actor_rollout_wg=actor_rollout_wg,
                agent_active_mask=role_mask,
                step=step,
            )
            # Every actual role call advances the path, even if its output is
            # invalid. FINISH and invalid Router attempts do not append a role.
            for i in np.flatnonzero(role_mask):
                self.role_paths[i] += (role,)
            batch.non_tensor_batch["parent_prefix_key"] = parent_prefix_keys.copy()
            valid_mask = role_mask & batch.non_tensor_batch["is_action_valid"]
            is_search = scene == "search" and role == "SEARCH"
            usable_mask = valid_mask
            if is_search:
                usable_mask = valid_mask & batch.non_tensor_batch["search_executable"]
            role_metadata = batch.non_tensor_batch
            for i in np.flatnonzero(usable_mask):
                role_metadata["execution_status"][i] = "queued" if is_search else "applied"
                if role_metadata["action_name"][i] == "record_review":
                    self.reviews[i] = {
                        "candidate_version": self.candidate_versions[i],
                        "env_step": step,
                        "assessment": dict(role_metadata["action_args"][i]),
                    }
            self.team_context = update_team_context(
                role, self.team_context, self._feedback(batch, responses, role_mask), role_mask,
                step=step, max_context_runs=self.max_context_runs,
            )
            update_answer = (
                scene == "math" and role in ("SOLVE", "REVISE")
            ) or (
                scene == "search" and role == "ANSWER"
            )
            if update_answer:
                self.candidate_answers = update_text_action(self.candidate_answers, responses, valid_mask)
                for i in np.flatnonzero(valid_mask):
                    self.candidate_versions[i] += 1
                    explanation = role_metadata["action_args"][i].get("explanation", "")
                    if explanation:
                        self.team_context[i][-1] += f"\nAnswer explanation: {explanation}\n"
            if update_answer or is_search:
                self.text_actions = update_text_action(self.text_actions, responses, usable_mask)
            if is_search:
                self.return_reasons[usable_mask] = "search"
            self.save_to_buffer(role, batch)

        return self.text_actions.copy(), self.multiagent_batch_buffer
