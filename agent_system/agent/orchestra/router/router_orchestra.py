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

"""Route partial work using the call/update/buffer flow shared by other orchestras."""

import re
from typing import Any, Dict, List, Tuple

import numpy as np
from transformers import PreTrainedTokenizer
from verl import DataProto
from verl.utils.reward_score.math_dapo import last_boxed_only_string

from agent_system.agent.agents.router import RouterAgent, SubAgent
from agent_system.agent.agents.router.context import CollaborationContext, WorkRecord
from agent_system.agent.orchestra.base import BaseOrchestra, update_team_context, update_text_action


class RouterMultiAgentOrchestra(BaseOrchestra):
    """One router selects SubAgents or submits their latest result.

    The router preset uses three named SubAgents. Full agent outputs accumulate
    across run() calls; search history is provided separately by env_obs.
    """

    ROUTER_AGENT = "Router Agent"

    def __init__(
        self,
        agent_ids: List[str],
        model_ids: List[str],
        agents_to_wg_mapping: Dict[str, str],
        tokenizers: Dict[str, PreTrainedTokenizer] = None,
        processors: Dict[str, Any] = None,
        config: Any = None,
    ):
        if any(not isinstance(name, str) or not name.strip() or name != name.strip()
               or any(char in name for char in '<>\n\r') for name in agent_ids):
            raise ValueError("Agent names must be nonempty, single-line strings without angle brackets or outer whitespace.")
        if len(agent_ids) != len(set(agent_ids)) or self.ROUTER_AGENT not in agent_ids:
            raise ValueError("agent_ids must contain unique names, including 'Router Agent'.")
        self.worker_ids = [name for name in agent_ids if name != self.ROUTER_AGENT]
        if not self.worker_ids:
            raise ValueError("Router orchestra requires at least one SubAgent.")
        super().__init__(
            agent_ids=agent_ids,
            model_ids=model_ids,
            agents_to_wg_mapping=agents_to_wg_mapping,
            tokenizers=tokenizers,
            processors=processors,
            config=config,
        )
        self.agent_order = self.worker_ids
        router_config = getattr(self.config.agent.orchestra, 'router', None)
        self.max_loop_num = getattr(router_config, 'max_loop_num', 8)
        if not isinstance(self.max_loop_num, int) or isinstance(self.max_loop_num, bool) or self.max_loop_num < 1:
            raise ValueError("agent.orchestra.router.max_loop_num must be a positive integer.")
        self.search_enabled = 'search' in self.config.env.env_name.lower()
        self.math_enabled = 'math' in self.config.env.env_name.lower() and not self.search_enabled
        self.team_context = None
        self.current_agents = None
        self.submission_actions = None
        self.candidate_answers = None
        self.candidate_authors = None
        self.candidate_responses = None
        self.candidate_repeat_counts = None
        self.routed_agents = None
        self.routing_instructions = None
        self.return_reasons = None
        self.work_records = None
        self.candidate_records = None
        self.assignment_turns = None
        # One path is kept for each environment item.  A path is extended only
        # when the router chooses a worker; worker CONTINUE/HANDOFF responses do
        # not add another node by themselves.
        self.agent_paths = None
        # The prefix that preceded the current worker segment.  This is needed
        # to assign the same parent prefix to every CONTINUE response in that
        # segment.
        self.segment_parent_paths = None

    def create_agent(self, name: str, **kwargs):
        if name == self.ROUTER_AGENT:
            return RouterAgent(**kwargs)
        return SubAgent(name=name, **kwargs)

    def reset(self):
        super().reset()
        self.team_context = None
        self.current_agents = None
        self.submission_actions = None
        self.candidate_answers = None
        self.candidate_authors = None
        self.candidate_responses = None
        self.candidate_repeat_counts = None
        self.routed_agents = None
        self.routing_instructions = None
        self.return_reasons = None
        self.work_records = None
        self.candidate_records = None
        self.assignment_turns = None
        self.agent_paths = None
        self.segment_parent_paths = None

    def initialize_context(self, env_obs):
        """Reuse agent work from this rollout while reading fresh environment history."""
        text_actions, initial_context, env_obs = super().initialize_context(env_obs)
        batch_size = len(text_actions)
        if self.team_context is None:
            self.team_context = initial_context
            self.current_agents = np.full(batch_size, '', dtype=object)
            self.submission_actions = [''] * batch_size
            self.candidate_answers = [''] * batch_size
            self.candidate_authors = [''] * batch_size
            self.candidate_responses = [''] * batch_size
            self.candidate_repeat_counts = [0] * batch_size
            self.routed_agents = [''] * batch_size
            self.routing_instructions = [''] * batch_size
            self.work_records = [[] for _ in range(batch_size)]
            self.candidate_records = [None] * batch_size
            self.assignment_turns = [0] * batch_size
            self.agent_paths = [[] for _ in range(batch_size)]
            self.segment_parent_paths = [tuple() for _ in range(batch_size)]
        elif len(self.team_context) != batch_size:
            raise ValueError("Batch size changed; call reset() before starting a new rollout.")
        elif len(self.agent_paths) != batch_size or len(self.segment_parent_paths) != batch_size:
            raise ValueError("Rollout path state is inconsistent; call reset() before starting a new rollout.")
        return text_actions, self.team_context, env_obs

    @staticmethod
    def _object_array(values: List[Any]) -> np.ndarray:
        """Create a one-dimensional object array without tuple broadcasting."""
        result = np.empty(len(values), dtype=object)
        for index, value in enumerate(values):
            result[index] = value
        return result

    def _set_path_metadata(
        self,
        batch: DataProto,
        paths: List[Tuple[str, ...]],
        parent_paths: List[Tuple[str, ...]],
    ) -> None:
        """Attach path metadata to every row in an agent output batch."""
        batch.non_tensor_batch['agent_path'] = self._object_array(paths)
        batch.non_tensor_batch['prefix_key'] = self._object_array(paths)
        batch.non_tensor_batch['parent_prefix_key'] = self._object_array(parent_paths)

    def save_to_buffer(self, name: str, batch: DataProto):
        # Every row must have the same metadata fields when the rollout collector
        # collates router and worker samples into a single training batch.
        defaults = {
            'selected_agent': '',
            'handoff_decision': '',
            'agent_path': tuple(),
            'prefix_key': tuple(),
            'parent_prefix_key': tuple(),
            'router_feedback': '',
            'return_reason': '',
        }
        for key, default in defaults.items():
            if key not in batch.non_tensor_batch:
                values = [default for _ in range(len(batch))]
                batch.non_tensor_batch[key] = self._object_array(values)
        # Snapshot state after this call. Later candidates/routes must not mutate
        # earlier training rows. Inactive rows do not inherit another sample's work.
        active = batch.non_tensor_batch['agent_active_mask']
        state = {
            'routed_agent': (self.routed_agents, ''),
            'routing_instruction': (self.routing_instructions, ''),
            'candidate_answer': (self.candidate_answers, ''),
            'candidate_author': (self.candidate_authors, ''),
            'candidate_repeat_count': (self.candidate_repeat_counts, 0),
        }
        for key, (values, empty) in state.items():
            if key not in batch.non_tensor_batch:
                batch.non_tensor_batch[key] = self._object_array([
                    values[i] if active[i] else empty for i in range(len(batch))
                ])
        super().save_to_buffer(name, batch)

    def _environment_action(self, response: str) -> Tuple[str, bool]:
        """Return a tool request or formatted candidate, never Math/Search partial work."""
        work = re.sub(r'<handoff>\s*(CONTINUE|HANDOFF)\s*</handoff>\s*$', '', response).strip()
        if self.search_enabled:
            action, is_search, error = SubAgent.search_action(response)
            return ('', False) if error else (action, is_search)
        if self.math_enabled:
            visible = re.sub(r'<think>.*?</think>', '', work, flags=re.DOTALL)
            start = visible.rfind(r'\boxed')
            # Use the existing brace-aware extractor on the last command. An
            # unfinished last command must not resurrect an earlier boxed result.
            boxed = last_boxed_only_string(visible[start:]) if start >= 0 else None
            if boxed and boxed[len(r'\boxed{'):-1].strip():
                return boxed, False
            return '', False
        return work, False

    def _record_work(self, index: int, step: int, author: str, kind: str, valid: bool, response: str) -> WorkRecord:
        record = WorkRecord(len(self.work_records[index]) + 1, step, author, kind, bool(valid), response)
        self.work_records[index].append(record)
        return record

    def _context_snapshot(self, active_mask: np.ndarray, loop_round: int):
        """Copy immutable per-sample records before a model call; never infer candidate state."""
        return [CollaborationContext(
            candidate_action=self.submission_actions[i],
            candidate_record=self.candidate_records[i],
            candidate_repeat_count=self.candidate_repeat_counts[i],
            assigned_worker=self.routed_agents[i],
            routing_instruction=self.routing_instructions[i],
            assignment_turn=self.assignment_turns[i],
            records=tuple(self.work_records[i]),
            loop_round=loop_round,
            loop_limit=self.max_loop_num,
        ) if active else None for i, active in enumerate(active_mask)]

    def _record_candidate(self, index: int, action: str, author: str, response: str, record: WorkRecord) -> None:
        """Count consecutive candidate outputs by extracted text, without grading them."""
        if self.search_enabled:
            answer = action[len('<answer>'):-len('</answer>')].strip()
        elif self.math_enabled:
            answer = action[len(r'\boxed{'):-1].strip()
        else:
            answer = action.strip()
        # First occurrence counts as one. Partial work and tool calls do not
        # emit candidates and therefore do not change this count.
        self.candidate_repeat_counts[index] = (
            self.candidate_repeat_counts[index] + 1 if answer == self.candidate_answers[index] else 1
        )
        self.submission_actions[index] = action
        self.candidate_answers[index] = answer
        self.candidate_authors[index] = author
        self.candidate_responses[index] = response
        self.candidate_records[index] = record

    def _record_return_reasons(self, active_masks: np.ndarray) -> None:
        """Mark only the last executed row for each sample in this run()."""
        remaining = active_masks.copy()
        for entry in reversed(self.multiagent_batch_buffer):
            batch = entry['batch']
            last = remaining & batch.non_tensor_batch['agent_active_mask']
            batch.non_tensor_batch['return_reason'][last] = self.return_reasons[last]
            remaining[last] = False
            if not remaining.any():
                break

    def run(
        self,
        gen_batch: DataProto,
        env_obs: Dict[str, Any],
        actor_rollout_wgs,
        active_masks: np.ndarray,
        step: int,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Route, execute, and update samples until FINISH, search, or the loop limit.

        CONTINUE keeps the current worker; HANDOFF schedules another router call.
        Router FINISH submits the latest available worker result for that sample.
        An answer tag is an environment format, not an internal stop signal.
        Search yields to envs.step(); the next env_obs supplies search history.
        Invalid routes/decisions are retried within the same finite loop budget.
        """
        self.reset_buffer()
        batch_size = len(gen_batch)
        active_masks = np.asarray(active_masks, dtype=bool)
        if active_masks.shape != (batch_size,):
            raise ValueError("active_masks must have one entry per batch item.")
        self.return_reasons = np.full(batch_size, '', dtype=object)
        if not active_masks.any():
            return [''] * batch_size, self.multiagent_batch_buffer
        text_actions, team_context, env_obs = self.initialize_context(env_obs)
        self.current_agents[~active_masks] = ''
        text_actions = update_text_action(text_actions, self.submission_actions, active_masks)
        pending_mask = active_masks.copy()

        for loop_index in range(self.max_loop_num):
            if not pending_mask.any():
                break

            # Step 1: Route new samples and samples whose worker handed off.
            router_mask = pending_mask & (self.current_agents == '')
            if router_mask.any():
                actor_rollout_wg = actor_rollout_wgs[self.agents_to_wg_mapping[self.ROUTER_AGENT]]
                parent_paths = [tuple(path) for path in self.agent_paths]
                router = self.agents[self.ROUTER_AGENT]
                batch, text_responses = router.call(
                    gen_batch=gen_batch,
                    env_obs=env_obs,
                    team_context=team_context,
                    actor_rollout_wg=actor_rollout_wg,
                    agent_active_mask=router_mask,
                    step=step,
                    collaboration_context=self._context_snapshot(router_mask, loop_index + 1),
                )
                team_context = update_team_context(self.ROUTER_AGENT, team_context, text_responses, router_mask)

                selections = batch.non_tensor_batch['selected_agent']
                finish_requests = router_mask & (selections == RouterAgent.FINISH)
                can_submit = np.array([bool(action) for action in self.submission_actions], dtype=bool)
                # Finishing before any result exists is invalid, so retry the router
                # rather than silently ending the task or selecting a worker for it.
                missing_candidate = finish_requests & ~can_submit
                batch.non_tensor_batch['is_action_valid'][missing_candidate] = False
                if self.search_enabled:
                    candidate_format = 'a complete, nonempty <answer> block outside <think>'
                elif self.math_enabled:
                    candidate_format = r'a complete, nonempty \boxed{...} outside <think>'
                else:
                    candidate_format = 'a final answer in the task\'s required format'
                for i in np.flatnonzero(missing_candidate):
                    feedback = ('FINISH cannot submit an answer because no candidate exists. '
                                f'Select a worker to produce {candidate_format} from the available work.')
                    router.search_feedback[i] = feedback
                    batch.non_tensor_batch['router_feedback'][i] = feedback
                finished = finish_requests & can_submit & batch.non_tensor_batch['is_action_valid']
                pending_mask[finished] = False
                self.return_reasons[finished] = 'finish'
                valid_routes = router_mask & np.isin(selections, self.worker_ids) & batch.non_tensor_batch['is_action_valid']
                router_paths = [tuple(path) for path in self.agent_paths]
                for i in np.flatnonzero(router_mask):
                    record = self._record_work(
                        i, step, self.ROUTER_AGENT,
                        'finish' if selections[i] == RouterAgent.FINISH else 'route',
                        batch.non_tensor_batch['is_action_valid'][i], text_responses[i],
                    )
                    if valid_routes[i]:
                        self.assignment_turns[i] = record.turn
                for i in np.flatnonzero(valid_routes):
                    self.routed_agents[i] = str(selections[i])
                    # A valid route without an instruction clears the old one.
                    self.routing_instructions[i] = batch.non_tensor_batch['routing_instruction'][i]
                    child_path = parent_paths[i] + (str(selections[i]),)
                    self.agent_paths[i] = list(child_path)
                    self.segment_parent_paths[i] = parent_paths[i]
                    router_paths[i] = child_path
                self._set_path_metadata(batch, router_paths, parent_paths)
                self.current_agents[valid_routes] = selections[valid_routes]
                self.save_to_buffer(self.ROUTER_AGENT, batch)

            # Step 2: Each selected SubAgent runs once for its assigned samples.
            for agent_id in self.agent_order:
                agent_mask = pending_mask & (self.current_agents == agent_id)
                if not agent_mask.any():
                    continue
                actor_rollout_wg = actor_rollout_wgs[self.agents_to_wg_mapping[agent_id]]
                batch, text_responses = self.agents[agent_id].call(
                    gen_batch=gen_batch,
                    env_obs=env_obs,
                    team_context=team_context,
                    actor_rollout_wg=actor_rollout_wg,
                    agent_active_mask=agent_mask,
                    step=step,
                    collaboration_context=self._context_snapshot(agent_mask, loop_index + 1),
                )
                team_context = update_team_context(agent_id, team_context, text_responses, agent_mask)
                worker_paths = [tuple(path) if agent_mask[i] else tuple() for i, path in enumerate(self.agent_paths)]
                worker_parent_paths = [
                    self.segment_parent_paths[i] if agent_mask[i] else tuple()
                    for i in range(batch_size)
                ]
                self._set_path_metadata(
                    batch,
                    worker_paths,
                    worker_parent_paths,
                )
                decisions = batch.non_tensor_batch['handoff_decision']
                valid_mask = agent_mask & np.isin(decisions, ['CONTINUE', 'HANDOFF']) & batch.non_tensor_batch['is_action_valid']
                agent_actions = [''] * batch_size
                search_mask = np.zeros(batch_size, dtype=bool)
                for i in np.flatnonzero(valid_mask):
                    agent_actions[i], search_mask[i] = self._environment_action(text_responses[i])
                action_mask = valid_mask & np.array([bool(action) for action in agent_actions], dtype=bool)
                text_actions = update_text_action(text_actions, agent_actions, action_mask)
                for i in np.flatnonzero(agent_mask):
                    kind = ('invalid' if not valid_mask[i] else 'search' if search_mask[i]
                            else 'answer' if action_mask[i] else 'intermediate')
                    record = self._record_work(i, step, agent_id, kind, valid_mask[i], text_responses[i])
                    if action_mask[i] and not search_mask[i]:
                        self._record_candidate(i, agent_actions[i], agent_id, text_responses[i], record)

                # Step 3: CONTINUE keeps the worker; HANDOFF requests new routing.
                self.current_agents[valid_mask & (decisions == 'HANDOFF')] = ''
                # Return searches to the environment before generating again.
                pending_mask[search_mask] = False
                self.return_reasons[search_mask] = 'search'
                self.save_to_buffer(agent_id, batch)

        # Keep the existing fallback to the saved candidate when the loop budget
        # expires, but do not label this as an approved FINISH.
        self.return_reasons[pending_mask] = 'loop_limit'
        self._record_return_reasons(active_masks)
        self.team_context = team_context
        return text_actions, self.multiagent_batch_buffer
