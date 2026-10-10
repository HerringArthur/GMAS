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

from agent_system.agent.agents.router import RouterAgent, SubAgent
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
        self.team_context = None
        self.current_agents = None
        self.submission_actions = None
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
        }
        for key, default in defaults.items():
            if key not in batch.non_tensor_batch:
                values = [default for _ in range(len(batch))]
                batch.non_tensor_batch[key] = self._object_array(values)
        super().save_to_buffer(name, batch)

    def _environment_action(self, response: str) -> Tuple[str, bool]:
        """Remove orchestration control and adapt text to the existing environment."""
        work = re.sub(r'<handoff>\s*(CONTINUE|HANDOFF)\s*</handoff>\s*$', '', response).strip()
        if not self.search_enabled:
            return work, False
        action, is_search, _ = SubAgent.search_action(response)
        return action, is_search

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
        if not active_masks.any():
            return [''] * batch_size, self.multiagent_batch_buffer
        text_actions, team_context, env_obs = self.initialize_context(env_obs)
        self.current_agents[~active_masks] = ''
        text_actions = update_text_action(text_actions, self.submission_actions, active_masks)
        pending_mask = active_masks.copy()

        for _ in range(self.max_loop_num):
            if not pending_mask.any():
                break

            # Step 1: Route new samples and samples whose worker handed off.
            router_mask = pending_mask & (self.current_agents == '')
            if router_mask.any():
                actor_rollout_wg = actor_rollout_wgs[self.agents_to_wg_mapping[self.ROUTER_AGENT]]
                parent_paths = [tuple(path) for path in self.agent_paths]
                batch, text_responses = self.agents[self.ROUTER_AGENT].call(
                    gen_batch=gen_batch,
                    env_obs=env_obs,
                    team_context=team_context,
                    actor_rollout_wg=actor_rollout_wg,
                    agent_active_mask=router_mask,
                    step=step,
                )
                team_context = update_team_context(self.ROUTER_AGENT, team_context, text_responses, router_mask)

                selections = batch.non_tensor_batch['selected_agent']
                finish_requests = router_mask & (selections == RouterAgent.FINISH)
                can_submit = np.array([bool(action) for action in self.submission_actions], dtype=bool)
                # Finishing before any result exists is invalid, so retry the router
                # rather than silently ending the task or selecting a worker for it.
                batch.non_tensor_batch['is_action_valid'][finish_requests & ~can_submit] = False
                finished = finish_requests & can_submit & batch.non_tensor_batch['is_action_valid']
                pending_mask[finished] = False
                valid_routes = router_mask & np.isin(selections, self.worker_ids) & batch.non_tensor_batch['is_action_valid']
                router_paths = [tuple(path) for path in self.agent_paths]
                for i in np.flatnonzero(valid_routes):
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
                self.save_to_buffer(agent_id, batch)

                decisions = batch.non_tensor_batch['handoff_decision']
                valid_mask = agent_mask & np.isin(decisions, ['CONTINUE', 'HANDOFF']) & batch.non_tensor_batch['is_action_valid']
                agent_actions = [''] * batch_size
                search_mask = np.zeros(batch_size, dtype=bool)
                for i in np.flatnonzero(valid_mask):
                    agent_actions[i], search_mask[i] = self._environment_action(text_responses[i])
                action_mask = valid_mask & np.array([bool(action) for action in agent_actions], dtype=bool)
                text_actions = update_text_action(text_actions, agent_actions, action_mask)
                self.submission_actions = update_text_action(self.submission_actions, agent_actions, action_mask & ~search_mask)

                # Step 3: CONTINUE keeps the worker; HANDOFF requests new routing.
                self.current_agents[valid_mask & (decisions == 'HANDOFF')] = ''
                # Return searches to the environment before generating again.
                pending_mask[search_mask] = False

        self.team_context = team_context
        return text_actions, self.multiagent_batch_buffer
