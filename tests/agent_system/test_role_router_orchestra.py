"""Exercise the new Orchestra contract with scripted agents, without model loading.

Real RouterAgent and role-agent integration is covered by test_role_router_agent.py.
Run: python3 -m unittest discover -s tests/agent_system -p 'test_role_router_orchestra.py' -v
"""

import ast
import importlib.util
import sys
import unittest
import uuid
from collections import deque
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class FakeBatch:
    def __init__(self, size):
        self.size = size
        self.batch = {}
        self.non_tensor_batch = {}

    def __len__(self):
        return self.size


class ScriptedAgent:
    def __init__(self, name="Router Agent", **kwargs):
        self.name = name
        self.wg_id = kwargs["wg_id"]
        self.script = deque()
        self.calls = []
        self.reset_count = 0

    def enqueue(self, responses, selections=None, valid=None, executable=None):
        batch = FakeBatch(len(responses))
        batch.batch["responses"] = tuple(responses)
        batch.non_tensor_batch["is_action_valid"] = np.array(
            valid if valid is not None else [True] * len(responses), dtype=bool,
        )
        batch.non_tensor_batch["selected_role"] = np.array(
            selections if selections is not None else [""] * len(responses), dtype=object,
        )
        if self.name == "SEARCH":
            batch.non_tensor_batch["search_executable"] = np.array(
                executable if executable is not None else batch.non_tensor_batch["is_action_valid"], dtype=bool,
            )
        size = len(responses)
        for field in ("action_error", "routing_instruction", "execution_status", "environment_action", "environment_result"):
            batch.non_tensor_batch[field] = np.full(size, "", dtype=object)
        batch.non_tensor_batch["action_error"][~batch.non_tensor_batch["is_action_valid"]] = "Invalid scripted call."
        name = {"Router Agent": "call_agent", "SEARCH": "search", "VERIFY": "record_review"}.get(self.name, "save_candidate")
        batch.non_tensor_batch["action_name"] = np.full(size, name, dtype=object)
        batch.non_tensor_batch["action_args"] = np.fromiter(
            ({"feedback": response} if self.name == "VERIFY" else {} for response in responses),
            dtype=object, count=size,
        )
        if self.name == "Router Agent":
            batch.non_tensor_batch["action_name"][batch.non_tensor_batch["selected_role"] == "FINISH"] = "finish"
        self.script.append((batch, responses))
        return batch

    def call(self, **kwargs):
        self.calls.append({
            **kwargs,
            "team_context": list(kwargs["team_context"]),
            "candidate_answers": list(kwargs["candidate_answers"]),
            "agent_active_mask": kwargs["agent_active_mask"].copy(),
        })
        if not self.script:
            raise AssertionError(f"Unexpected call to {self.name}")
        batch, responses = self.script.popleft()
        mask = kwargs["agent_active_mask"]
        metadata = batch.non_tensor_batch
        metadata["is_action_valid"] &= mask
        if "search_executable" in metadata:
            metadata["search_executable"] &= mask
        metadata["agent_active_mask"] = mask.copy()
        metadata["wg_id"] = np.full(len(batch), self.wg_id, dtype=object)
        metadata["env_step"] = np.full(len(batch), kwargs["step"], dtype=object)
        metadata["selected_role"] = np.where(mask, metadata["selected_role"], "")
        return batch, responses

    def reset(self):
        self.reset_count += 1


def load_orchestra():
    def module(name, **attributes):
        result = ModuleType(name)
        result.__dict__.update(attributes)
        return result

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, ROOT / path)
        result = importlib.util.module_from_spec(spec)
        sys.modules[name] = result
        spec.loader.exec_module(result)
        return result

    stubs = {
        "transformers": module("transformers", PreTrainedTokenizer=object),
        "verl": module("verl", DataProto=FakeBatch),
        "agent_system.agent.agents.base": module("agent_system.agent.agents.base", BaseAgent=ScriptedAgent),
    }
    with patch.dict(sys.modules, stubs):
        registry = load("agent_system.agent.registry", "agent_system/agent/registry.py")
        load("agent_system.agent.orchestra.base", "agent_system/agent/orchestra/base.py")
        result = load("role_router_under_test", "agent_system/agent/orchestra/router/router_orchestra.py")
    return result.RouterMultiAgentOrchestra, registry.AgentRegistry, result.update_text_action


RouterOrchestra, AgentRegistry, update_text_action = load_orchestra()


def load_collector_loop():
    """Run the real collector method with only model/environment boundaries mocked."""
    path = ROOT / "agent_system/multi_turn_rollout/rollout_loop.py"
    tree = ast.parse(path.read_text())
    collector = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                     and node.name == "MultiAgentTrajectoryCollector")
    method = next(node for node in collector.body if isinstance(node, ast.FunctionDef)
                  and node.name == "vanilla_multi_turn_loop")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
    scope = {
        "np": np, "uuid": uuid,
        "update_text_action": update_text_action,
        "torch_to_numpy": lambda value, is_object=False: np.asarray(value, dtype=object if is_object else None),
        "to_list_of_dict": lambda batch: [
            {key: value[i] for key, value in batch.non_tensor_batch.items()} for i in range(len(batch))
        ],
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope["vanilla_multi_turn_loop"]


collect = load_collector_loop()


class ScriptedEnvironment:
    def __init__(self, observations, transitions, on_step=None):
        self.observations = observations
        self.transitions = deque(transitions)
        self.actions = []
        self.on_step = on_step

    def reset(self, kwargs=None):
        return {"text": self.observations}, [{} for _ in self.observations]

    def step(self, actions):
        self.actions.append(list(actions))
        if self.on_step:
            self.on_step()
        observations, rewards, dones = self.transitions.popleft()
        return {"text": observations}, np.array(rewards), np.array(dones), [{} for _ in actions]

    def success_evaluator(self, **kwargs):
        return {"success_rate": [1.0] * len(self.observations)}


class RoleRouterTests(unittest.TestCase):
    def make_orchestra(self, loops=4, roles=("SOLVE", "VERIFY", "REVISE"), env_name="math", context_runs=2):
        names = ["Router Agent", *roles]
        self.config = SimpleNamespace(
            agent=SimpleNamespace(orchestra_type="router", orchestra=SimpleNamespace(
                router=SimpleNamespace(max_loop_num=loops, max_context_runs=context_runs))),
            env=SimpleNamespace(env_name=env_name, max_steps=2),
        )
        mapping = {name: f"policy:{name}" for name in names}
        self.groups = {group: object() for group in mapping.values()}
        AgentRegistry._REGISTRY.clear()
        for name in names:
            AgentRegistry.register(name)(partial(ScriptedAgent, name=name))
        package_name = "agent_system.agent.agents.router"
        with patch.dict(sys.modules, {package_name: ModuleType(package_name)}):
            self.orchestra = RouterOrchestra(
                names, list(roles), ["same-initial-checkpoint"] * len(names), mapping,
                {group: None for group in self.groups}, {group: None for group in self.groups}, self.config,
            )
        return self.orchestra

    def route(self, selections, valid=None):
        return self.orchestra.agents["Router Agent"].enqueue(
            [f"Assignment: {role}" for role in selections], selections, valid,
        )

    def respond(self, role, responses, valid=None, executable=None):
        return self.orchestra.agents[role].enqueue(responses, valid=valid, executable=executable)

    def run_round(self, active=(True,), step=1, observation=None):
        actions, buffer = self.orchestra.run(
            FakeBatch(len(active)), {"text": observation or ["Problem"] * len(active)},
            self.groups, np.array(active), step,
        )
        return actions, list(buffer)

    def test_credit_paths_track_actual_calls_and_survive_history_eviction(self):
        orchestra = self.make_orchestra(context_runs=1)
        self.route(["SOLVE", "VERIFY", "SOLVE"])
        self.respond("SOLVE", ["solution", "ignored", "inactive"])
        self.respond("VERIFY", ["ignored", "invalid review", "inactive"], valid=[True, False, True])
        _, first = self.run_round(active=(True, True, False))
        self.assertEqual(orchestra.role_paths, [("SOLVE",), ("VERIFY",), ()])
        for entry in first:
            paths = entry["batch"].non_tensor_batch["parent_prefix_key"]
            self.assertEqual(paths.shape, (3,))
            self.assertEqual(paths.tolist(), [(), (), ()])

        self.route(["UNKNOWN", "SOLVE", "SOLVE"], valid=[False, True, True])
        self.respond("SOLVE", ["ignored", "solution", "inactive"])
        _, second = self.run_round(active=(True, True, False))
        self.assertEqual(orchestra.role_paths, [("SOLVE",), ("VERIFY", "SOLVE"), ()])
        self.assertEqual(second[0]["batch"].non_tensor_batch["parent_prefix_key"].tolist(),
                         [("SOLVE",), ("VERIFY",), ()])

        orchestra.initialize_context({"text": ["next step"] * 3})
        self.route(["FINISH", "SOLVE", "SOLVE"])
        self.respond("SOLVE", ["ignored", "revised", "inactive"])
        _, third = self.run_round(active=(True, True, False), step=2)
        self.assertEqual(third[0]["batch"].non_tensor_batch["parent_prefix_key"].tolist(),
                         [("SOLVE",), ("VERIFY", "SOLVE"), ()])
        self.assertEqual(orchestra.role_paths, [("SOLVE",), ("VERIFY", "SOLVE", "SOLVE"), ()])
        self.assertEqual(len(orchestra.team_context[1]), 1)
        self.assertEqual(first[0]["batch"].non_tensor_batch["parent_prefix_key"].tolist(), [(), (), ()])
        orchestra.reset()
        self.assertIsNone(orchestra.role_paths)
        orchestra.initialize_context({"text": ["new rollout"] * 3})
        self.assertEqual(orchestra.role_paths, [(), (), ()])

    def collect(self, environment):
        collector = SimpleNamespace(config=self.config, multiagent_orchestra=self.orchestra)
        batch = FakeBatch(len(environment.observations))
        batch.batch = np.zeros((len(batch), 1))
        return collect(collector, batch, self.groups, environment, effective_rollout_n=1)

    def test_run_executes_one_round_and_next_router_receives_previous_work(self):
        orchestra = self.make_orchestra()
        self.route(["SOLVE"])
        self.respond("SOLVE", ["Initial answer"])
        actions, first_buffer = self.run_round()
        self.assertEqual(actions, ["Initial answer"])
        self.assertEqual([row["agent_id"] for row in first_buffer], ["Router Agent", "SOLVE"])
        self.assertEqual(orchestra.return_reasons.tolist(), [""])
        self.assertEqual(len(orchestra.agents["Router Agent"].calls), 1)
        self.assertEqual(orchestra.agents["Router Agent"].calls[0]["candidate_answers"], [""])
        self.assertEqual(orchestra.agents["SOLVE"].calls[0]["candidate_answers"], [""])

        self.route(["VERIFY"])
        self.respond("VERIFY", ["The derivation contains an error"])
        self.assertEqual(self.run_round()[0], ["Initial answer"])
        self.assertIn("Initial answer", orchestra.agents["Router Agent"].calls[-1]["team_context"][0])
        self.assertEqual(orchestra.agents["Router Agent"].calls[-1]["candidate_answers"], ["Initial answer"])
        self.assertEqual(orchestra.agents["VERIFY"].calls[-1]["candidate_answers"], ["Initial answer"])
        self.route(["REVISE"])
        revised_batch = self.respond("REVISE", ["Corrected answer"])
        self.assertEqual(self.run_round()[0], ["Corrected answer"])
        revision = orchestra.agents["REVISE"].calls[0]
        self.assertEqual(revision["candidate_answers"], ["Initial answer"])
        self.assertIn('The output of "VERIFY": The derivation contains an error', revision["team_context"][0])
        self.assertIn("Assignment: REVISE", revision["team_context"][0])
        self.assertIs(revision["actor_rollout_wg"], self.groups["policy:REVISE"])
        self.assertEqual(revised_batch.batch["responses"], ("Corrected answer",))
        self.assertEqual(actions, ["Initial answer"])
        self.assertEqual(len(first_buffer), 2)

        self.route(["FINISH"])
        actions, final_buffer = self.run_round()
        self.assertEqual(actions, ["Corrected answer"])
        self.assertEqual(orchestra.agents["Router Agent"].calls[-1]["candidate_answers"], ["Corrected answer"])
        self.assertEqual([row["agent_id"] for row in final_buffer], ["Router Agent"])
        self.assertEqual(orchestra.return_reasons.tolist(), ["finish"])
        self.assertEqual(self.run_round()[1], [])

    def test_batch_dispatch_and_finish_are_independent_per_sample(self):
        orchestra = self.make_orchestra()
        self.route(["SOLVE", "REVISE", "SOLVE"])
        self.respond("SOLVE", ["A", "must ignore", "inactive"])
        self.respond("REVISE", ["must ignore", "B", "inactive"])
        self.assertEqual(self.run_round(active=(True, True, False))[0], ["A", "B", ""])
        self.route(["FINISH", "SOLVE", "REVISE"])
        self.respond("SOLVE", ["must ignore", "B revised", "inactive"])
        self.run_round(active=(True, True, False))
        self.route(["SOLVE", "FINISH", "SOLVE"])
        self.assertEqual(self.run_round(active=(True, True, False))[0], ["A", "B revised", ""])
        self.assertEqual(orchestra.agents["Router Agent"].calls[-1]["agent_active_mask"].tolist(), [False, True, False])
        solver_calls = orchestra.agents["SOLVE"].calls
        self.assertEqual(solver_calls[0]["agent_active_mask"].tolist(), [True, False, False])
        self.assertEqual(solver_calls[1]["agent_active_mask"].tolist(), [False, True, False])
        self.assertIs(solver_calls[0]["actor_rollout_wg"], solver_calls[1]["actor_rollout_wg"])
        self.assertEqual(orchestra.team_context[2], [])
        self.assertNotIn("must ignore", "".join("".join(runs) for runs in orchestra.team_context))

    def test_context_window_keeps_complete_runs_and_candidate_independently(self):
        for window in (1, 2, 3):
            with self.subTest(window=window):
                orchestra = self.make_orchestra(loops=8, context_runs=window)
                for turn in range(1, 6):
                    role = "SOLVE" if turn == 1 else "VERIFY"
                    decision = f"Decision for run {turn}."
                    response = f"Output for run {turn}."
                    orchestra.agents["Router Agent"].enqueue([decision], [role])
                    self.respond(role, [response])
                    _, buffer = self.run_round(step=1)
                    if turn == 1:
                        first_buffer = buffer

                    router_context = orchestra.agents["Router Agent"].calls[-1]["team_context"][0]
                    role_context = orchestra.agents[role].calls[-1]["team_context"][0]
                    self.assertIn(decision, role_context)
                    for previous in range(1, turn):
                        old_decision = f"Decision for run {previous}."
                        old_response = f"Output for run {previous}."
                        for context in (router_context, role_context):
                            if previous >= turn - window:
                                self.assertIn(old_decision, context)
                                self.assertIn(old_response, context)
                            else:
                                self.assertNotIn(old_decision, context)
                                self.assertNotIn(old_response, context)

                    self.assertEqual(len(orchestra.team_context[0]), min(turn, window))
                    for previous in range(1, turn + 1):
                        for text in (f"Decision for run {previous}.", f"Output for run {previous}."):
                            if previous >= turn - window + 1:
                                self.assertIn(text, "".join(orchestra.team_context[0]))
                            else:
                                self.assertNotIn(text, "".join(orchestra.team_context[0]))

                self.assertEqual(orchestra.candidate_answers, ["Output for run 1."])
                self.assertEqual(orchestra.agents["VERIFY"].calls[-1]["candidate_answers"], ["Output for run 1."])
                self.assertNotIn("Output for run 1.", "".join(orchestra.team_context[0]))
                self.assertEqual(first_buffer[1]["batch"].batch["responses"], ("Output for run 1.",))

    def test_context_window_does_not_age_paused_or_inactive_samples(self):
        orchestra = self.make_orchestra(roles=("SEARCH", "VERIFY"), env_name="search")
        query = "<search>paused sample query</search>"
        self.route(["SEARCH", "VERIFY", "VERIFY"])
        self.respond("SEARCH", [query, "ignored", "inactive"])
        self.respond("VERIFY", ["ignored", "Review 1.", "inactive"])
        self.run_round(active=(True, True, False))
        paused_context = orchestra.team_context[0].copy()

        for turn in range(2, 6):
            self.route(["VERIFY", "VERIFY", "VERIFY"])
            self.respond("VERIFY", ["must ignore", f"Review {turn}.", "inactive"])
            self.run_round(active=(True, True, False))
            self.assertEqual(orchestra.team_context[0], paused_context)
            self.assertEqual(orchestra.team_context[2], [])
        self.assertEqual(len(orchestra.team_context[1]), 2)
        self.assertIn("Review 4.", "".join(orchestra.team_context[1]))
        self.assertIn("Review 5.", "".join(orchestra.team_context[1]))
        self.assertNotIn("Review 3.", "".join(orchestra.team_context[1]))

        orchestra.initialize_context({"text": ["Evidence A", "Evidence B", "Inactive"]})
        self.route(["VERIFY", "FINISH", "VERIFY"])
        self.respond("VERIFY", ["Review after retrieval.", "ignored", "inactive"])
        self.run_round(active=(True, True, False), step=2)
        self.assertEqual(orchestra.agents["Router Agent"].calls[-1]["team_context"][0], "".join(paused_context))
        self.assertIn(query, "".join(orchestra.team_context[0]))
        self.assertIn("Review after retrieval.", "".join(orchestra.team_context[0]))
        self.assertEqual(len(orchestra.team_context[0]), 2)
        self.assertIn("[Environment step 1]", orchestra.team_context[0][0])
        self.assertIn("[Environment step 2]", orchestra.team_context[0][1])

    def test_invalid_runs_count_toward_window_without_recording_invalid_text(self):
        orchestra = self.make_orchestra()
        self.route(["SOLVE"])
        self.respond("SOLVE", ["Saved candidate"])
        self.run_round()
        for _ in range(2):
            invalid_batch = self.route(["UNKNOWN"], valid=[False])
            self.run_round()
            self.assertFalse(invalid_batch.non_tensor_batch["is_action_valid"][0])
        self.assertEqual(len(orchestra.team_context[0]), 2)
        self.assertTrue(all("Execution feedback" in run and "ActionError" in run for run in orchestra.team_context[0]))
        self.assertEqual(orchestra.candidate_answers, ["Saved candidate"])
        self.route(["FINISH"])
        self.assertEqual(self.run_round()[0], ["Saved candidate"])
        router_call = orchestra.agents["Router Agent"].calls[-1]
        self.assertIn("ActionError", router_call["team_context"][0])
        self.assertEqual(router_call["candidate_answers"], ["Saved candidate"])

    def test_invalid_outputs_wait_for_next_router_call_and_preserve_answer(self):
        orchestra = self.make_orchestra()
        invalid = self.route(["UNKNOWN"], valid=[False])
        self.assertEqual(self.run_round()[0], [""])
        self.assertFalse(invalid.non_tensor_batch["is_action_valid"][0])
        self.route(["SOLVE"])
        self.respond("SOLVE", ["saved answer"])
        self.run_round()
        self.route(["REVISE"])
        invalid = self.respond("REVISE", ["broken output"], valid=[False])
        self.assertEqual(self.run_round()[0], ["saved answer"])
        self.assertEqual(orchestra.candidate_answers, ["saved answer"])
        self.assertEqual(orchestra.return_reasons.tolist(), [""])
        self.route(["FINISH"])
        self.run_round()
        context = orchestra.agents["Router Agent"].calls[-1]["team_context"][0]
        self.assertNotIn("broken output", context)
        self.assertNotIn("UNKNOWN", context)

    def test_reset_clears_context_answers_and_stopping_state(self):
        orchestra = self.make_orchestra()
        self.route(["SOLVE"])
        self.respond("SOLVE", ["old answer"])
        self.run_round()
        self.assertEqual(orchestra.candidate_answers, ["old answer"])
        self.route(["FINISH"])
        self.run_round()
        orchestra.reset()
        self.assertIsNone(orchestra.candidate_answers)
        self.assertIsNone(orchestra.team_context)
        self.route(["VERIFY"])
        self.respond("VERIFY", ["new report"])
        self.assertEqual(self.run_round()[0], [""])
        self.assertEqual(orchestra.candidate_answers, [""])
        self.assertEqual(orchestra.agents["Router Agent"].calls[-1]["candidate_answers"], [""])
        self.assertEqual(orchestra.agents["VERIFY"].calls[-1]["candidate_answers"], [""])
        self.assertIn("new report", "".join(orchestra.team_context[0]))
        self.assertNotIn("old answer", "".join(orchestra.team_context[0]))
        self.assertEqual(orchestra.return_reasons.tolist(), [""])
        self.assertTrue(all(agent.reset_count == 1 for agent in orchestra.agents.values()))

    def test_no_active_samples_make_no_model_calls(self):
        orchestra = self.make_orchestra()
        self.assertEqual(self.run_round(active=(False, False)), (["", ""], []))
        self.assertTrue(all(not agent.calls for agent in orchestra.agents.values()))
        self.assertEqual(orchestra.team_context, [[], []])

    def test_environment_step_preserves_work_and_resets_pending_actions(self):
        orchestra = self.make_orchestra(roles=("ANSWER", "VERIFY", "SEARCH"), env_name="search")
        for role, response in (
            ("ANSWER", "Candidate answer"),
            ("VERIFY", "The candidate needs supporting evidence"),
            ("SEARCH", "<search>supporting evidence</search>"),
        ):
            self.route([role])
            self.respond(role, [response])
            self.run_round()
        previous_context = ["".join(runs) for runs in orchestra.team_context]
        self.assertNotIn("Candidate answer", previous_context[0])
        self.assertEqual(orchestra.agents["SEARCH"].calls[-1]["candidate_answers"], ["Candidate answer"])
        self.assertEqual(orchestra.candidate_answers, ["Candidate answer"])
        self.assertEqual(orchestra.text_actions, ["<search>supporting evidence</search>"])
        self.assertEqual(orchestra.return_reasons.tolist(), ["search"])

        next_obs = {"text": ["Newly retrieved evidence"]}
        actions, context, obs = orchestra.initialize_context(next_obs)
        self.assertEqual(actions, [""])
        self.assertEqual(orchestra.candidate_answers, ["Candidate answer"])
        self.assertEqual(context, previous_context)
        self.assertEqual(orchestra.return_reasons.tolist(), [""])
        self.assertIs(obs, next_obs)

        self.route(["VERIFY"])
        self.respond("VERIFY", ["The new evidence supports the candidate"])
        self.assertEqual(self.run_round(step=2, observation=next_obs["text"])[0], [""])
        router_call = orchestra.agents["Router Agent"].calls[-1]
        self.assertTrue(router_call["team_context"][0].startswith(previous_context[0]))
        self.assertIn("Reviewed environment step: 1; current step: 2", router_call["team_context"][0])
        self.assertEqual(router_call["candidate_answers"], ["Candidate answer"])
        self.assertEqual(orchestra.agents["VERIFY"].calls[-1]["candidate_answers"], ["Candidate answer"])
        self.assertEqual(router_call["env_obs"], next_obs)
        self.assertNotIn("The candidate needs supporting evidence", "".join(orchestra.team_context[0]))
        self.assertIn("<search>supporting evidence</search>", "".join(orchestra.team_context[0]))
        self.assertIn("[Environment step 1]", "".join(orchestra.team_context[0]))
        self.assertIn("[Environment step 2]", "".join(orchestra.team_context[0]))
        self.assertIn("The new evidence supports the candidate", "".join(orchestra.team_context[0]))
        self.route(["FINISH"])
        self.assertEqual(self.run_round(step=2, observation=next_obs["text"])[0], ["Candidate answer"])

    def test_collector_finishes_collaboration_before_submitting_and_keeps_all_rows(self):
        orchestra = self.make_orchestra()
        for role in ("SOLVE", "VERIFY", "REVISE", "FINISH"):
            self.route([role])
        self.respond("SOLVE", ["Initial answer"])
        self.respond("VERIFY", ["Incorrect derivation"])
        self.respond("REVISE", ["Corrected answer"])
        env = ScriptedEnvironment(["Problem"], [(None, [1.0], [True])], on_step=lambda: self.assertEqual(
            len(orchestra.agents["Router Agent"].calls), 4))
        rows, rewards, lengths, _, _, _ = self.collect(env)
        self.assertEqual(env.actions, [["Corrected answer"]])
        self.assertEqual([row["agent_id"] for row in rows[0]], [
            "Router Agent", "SOLVE", "Router Agent", "VERIFY", "Router Agent", "REVISE", "Router Agent",
        ])
        self.assertEqual([row["return_reason"] for row in rows[0]], ["", "", "", "", "", "", "finish"])
        self.assertTrue(all(row["env_step"] == 1 and row["rewards"] == 1.0 for row in rows[0]))
        self.assertEqual(rewards.tolist(), [1.0])
        self.assertEqual(lengths.tolist(), [1.0])
        self.assertTrue(all(set(row) == set(rows[0][0]) for row in rows[0]))

    def test_collector_search_result_reaches_next_environment_step_and_completed_rows_stay_inactive(self):
        orchestra = self.make_orchestra(roles=("SEARCH", "ANSWER"), env_name="search")
        self.route(["SEARCH", "ANSWER"])
        query = "<search>search query</search>"
        self.respond("SEARCH", [query, "ignored"])
        self.respond("ANSWER", ["ignored", "B answer"])
        self.route(["ANSWER", "FINISH"])
        self.route(["UNKNOWN", "ANSWER"], valid=[False, True])
        self.route(["ANSWER", "ANSWER"])
        self.respond("ANSWER", ["A answer from evidence", "ignored"])
        self.route(["FINISH", "ANSWER"])
        env = ScriptedEnvironment(["A problem", "B problem"], [
            (["A retrieved evidence", "B done"], [0.0, 1.0], [False, True]),
            (None, [1.0, 0.0], [True, True]),
        ])
        rows, rewards, lengths, _, _, _ = self.collect(env)
        self.assertEqual(env.actions, [[query, "B answer"], ["A answer from evidence", ""]])
        self.assertEqual([row["return_reason"] for row in rows[0]], ["", "search", "", "", "", "finish"])
        self.assertEqual(rows[1][-1]["return_reason"], "finish")
        calls = orchestra.agents["Router Agent"].calls
        self.assertEqual(calls[1]["agent_active_mask"].tolist(), [False, True])
        self.assertTrue(all(call["agent_active_mask"].tolist() == [True, False] for call in calls[2:]))
        self.assertEqual(calls[2]["team_context"][0], calls[1]["team_context"][0])
        self.assertIn(query, calls[2]["team_context"][0])
        self.assertIn("B answer", calls[2]["team_context"][1])
        self.assertNotIn("B answer", calls[2]["team_context"][0])
        self.assertNotIn(query, calls[2]["team_context"][1])
        self.assertEqual("".join(orchestra.team_context[1]), calls[2]["team_context"][1])
        self.assertIn("retrieved evidence", calls[2]["env_obs"]["text"][0])
        self.assertEqual(lengths.tolist(), [2.0, 1.0])
        self.assertEqual(rewards.tolist(), [1.0, 1.0])
        self.assertFalse(rows[0][2]["is_action_valid"])
        self.assertEqual([row["parent_prefix_key"] for row in rows[0]],
                         [(), (), ("SEARCH",), ("SEARCH",), ("SEARCH",), ("SEARCH", "ANSWER")])
        self.assertEqual([row["parent_prefix_key"] for row in rows[1]], [(), (), ("ANSWER",)])

    def test_search_pauses_only_valid_executable_active_rows(self):
        orchestra = self.make_orchestra(roles=("SEARCH",), env_name="search")
        self.route(["SEARCH"] * 3)
        query = "<search>first query</search>"
        search_batch = self.respond(
            "SEARCH", [query, "broken request", "<search>inactive</search>"],
            valid=[True, False, False], executable=[True, True, True],
        )
        actions, buffer = self.run_round(active=(True, True, False))
        self.assertEqual(actions, [query, "", ""])
        self.assertEqual(orchestra.return_reasons.tolist(), ["search", "", ""])
        self.assertEqual(search_batch.non_tensor_batch["is_action_valid"].tolist(), [True, False, False])
        self.assertIn(query, "".join(orchestra.team_context[0]))
        self.assertNotIn("broken request", "".join(orchestra.team_context[1]))
        self.assertEqual(orchestra.candidate_answers, ["", "", ""])
        self.assertEqual(buffer[0]["batch"].non_tensor_batch["search_executable"].tolist(), [False] * 3)

        self.route(["SEARCH"] * 3)
        second_query = "<search>second query</search>"
        self.respond("SEARCH", ["must not overwrite", second_query, "ignored"])
        actions, _ = self.run_round(active=(True, True, False))
        self.assertEqual(actions, [query, second_query, ""])
        self.assertEqual(orchestra.return_reasons.tolist(), ["search", "search", ""])
        for name in ("Router Agent", "SEARCH"):
            self.assertEqual(orchestra.agents[name].calls[-1]["agent_active_mask"].tolist(), [False, True, False])
        self.assertEqual(self.run_round(active=(True, True, False))[1], [])

    def test_collector_executes_search_without_waiting_for_finish(self):
        orchestra = self.make_orchestra(roles=("SEARCH", "ANSWER"), env_name="search")
        self.route(["SEARCH"])
        query = "<search>query</search>"
        self.respond("SEARCH", [query])
        self.route(["ANSWER"])
        self.respond("ANSWER", ["<answer>A</answer>"])
        self.route(["FINISH"])
        env = ScriptedEnvironment(["Problem"], [
            (["Retrieved evidence"], [0.0], [False]),
            (None, [1.0], [True]),
        ], on_step=lambda: self.assertEqual(
            len(orchestra.agents["Router Agent"].calls), 1 if len(env.actions) == 1 else 3))
        rows, *_ = self.collect(env)
        self.assertEqual(env.actions, [[query], ["<answer>A</answer>"]])
        self.assertEqual([row["return_reason"] for row in rows[0]][:2], ["", "search"])

    def test_final_search_step_blocks_search_and_preserves_other_choices(self):
        orchestra = self.make_orchestra(roles=("SEARCH", "VERIFY", "ANSWER"), env_name="search")
        router_batch = self.route(["SEARCH", "VERIFY", "ANSWER", "FINISH", "SEARCH"])
        self.respond("VERIFY", ["ignored", "review", "ignored", "ignored", "inactive"])
        self.respond("ANSWER", ["ignored", "ignored", "<answer>A</answer>", "ignored", "inactive"])
        actions, buffer = self.run_round(active=(True, True, True, True, False), step=2)
        self.assertEqual(actions, ["", "", "<answer>A</answer>", "", ""])
        self.assertEqual(router_batch.non_tensor_batch["is_action_valid"].tolist(),
                         [False, True, True, False, False])
        self.assertEqual(router_batch.non_tensor_batch["selected_role"][0], "SEARCH")
        self.assertEqual(router_batch.batch["responses"][0], "Assignment: SEARCH")
        self.assertFalse(orchestra.agents["SEARCH"].calls)
        self.assertEqual(orchestra.role_paths, [(), ("VERIFY",), ("ANSWER",), (), ()])
        self.assertIn("ActionError: SEARCH is unavailable", orchestra.team_context[0][0])
        self.assertEqual(orchestra.return_reasons.tolist(), ["", "", "", "", ""])
        self.assertEqual([entry["agent_id"] for entry in buffer], ["Router Agent", "VERIFY", "ANSWER"])

    def test_final_search_step_retries_blocked_search_then_submits_candidates_at_limit(self):
        orchestra = self.make_orchestra(loops=2, roles=("SEARCH", "VERIFY", "ANSWER"), env_name="search")
        saved = "<answer>Saved A</answer>"
        fresh = "<answer>New B</answer>"
        queries = ["<search>evidence for A</search>", "<search>evidence for B</search>"]
        self.route(["ANSWER", "SEARCH"])
        self.respond("ANSWER", [saved, "ignored"])
        self.respond("SEARCH", ["ignored", queries[1]])
        self.route(["SEARCH", "FINISH"])
        self.respond("SEARCH", [queries[0], "ignored"])
        self.route(["SEARCH", "SEARCH"])
        self.route(["VERIFY", "ANSWER"])
        self.respond("VERIFY", ["A is supported", "ignored"])
        self.respond("ANSWER", ["ignored", fresh])
        env = ScriptedEnvironment(["A problem", "B problem"], [
            (["A evidence", "B evidence"], [0.0, 0.0], [False, False]),
            (None, [1.0, 1.0], [True, True]),
        ])
        rows, *_ = self.collect(env)
        self.assertEqual(env.actions, [queries, [saved, fresh]])
        self.assertEqual(orchestra.candidate_answers, [saved, fresh])
        self.assertEqual([row[-1]["return_reason"] for row in rows], ["loop_limit", "loop_limit"])
        for sample in rows:
            final_rows = [row for row in sample if row["env_step"] == 2]
            self.assertEqual(final_rows[0]["selected_role"], "SEARCH")
            self.assertFalse(final_rows[0]["is_action_valid"])
            self.assertFalse(any(row["agent_id"] == "SEARCH" for row in final_rows))
        self.assertEqual(orchestra.role_paths, [("ANSWER", "SEARCH", "VERIFY"), ("SEARCH", "ANSWER")])

    def test_collector_budget_fallback_submits_saved_answer_and_labels_last_call(self):
        self.make_orchestra(loops=2)
        self.route(["SOLVE"])
        self.respond("SOLVE", ["answer"])
        self.route(["VERIFY"])
        self.respond("VERIFY", ["review"])
        env = ScriptedEnvironment(["Problem"], [(None, [1.0], [True])])
        rows, *_ = self.collect(env)
        self.assertEqual(env.actions, [["answer"]])
        self.assertEqual([row["return_reason"] for row in rows[0]], ["", "", "", "loop_limit"])
        self.assertEqual([row["selected_role"] for row in rows[0]], ["SOLVE", "", "VERIFY", ""])

    def test_collector_finishes_with_prior_answer_or_replaces_it_after_search(self):
        orchestra = self.make_orchestra(roles=("SEARCH", "VERIFY", "ANSWER"), env_name="search")
        answers = ["<answer>A</answer>", "<answer>B</answer>"]
        queries = ["<search>evidence for A</search>", "<search>evidence for B</search>"]
        self.route(["ANSWER", "ANSWER"])
        self.respond("ANSWER", answers)
        self.route(["SEARCH", "SEARCH"])
        self.respond("SEARCH", queries)
        self.route(["VERIFY", "VERIFY"])
        self.respond("VERIFY", ["A is supported", "B needs correction"])
        self.route(["FINISH", "ANSWER"])
        self.respond("ANSWER", ["must ignore", "<answer>B revised</answer>"])
        self.route(["ANSWER", "FINISH"])
        env = ScriptedEnvironment(["A problem", "B problem"], [
            (["A evidence", "B evidence"], [0.0, 0.0], [False, False]),
            (None, [1.0, 1.0], [True, True]),
        ])
        rows, *_ = self.collect(env)
        self.assertEqual(env.actions, [queries, [answers[0], "<answer>B revised</answer>"]])
        self.assertEqual(orchestra.candidate_answers, env.actions[-1])
        self.assertEqual([row["agent_id"] for row in rows[0]].count("ANSWER"), 1)
        self.assertEqual([row["agent_id"] for row in rows[1]].count("ANSWER"), 2)
        self.assertEqual(orchestra.agents["ANSWER"].calls[-1]["agent_active_mask"].tolist(), [False, True])
        self.assertEqual([sample[-1]["return_reason"] for sample in rows], ["finish", "finish"])
        self.assertNotIn("must ignore", "".join(orchestra.team_context[0]))

    def test_budget_fallback_uses_prior_candidate_and_preserves_search_and_finish_actions(self):
        orchestra = self.make_orchestra(loops=2, roles=("SEARCH", "VERIFY", "ANSWER"), env_name="search")
        self.config.env.max_steps = 3
        answers = ["<answer>A</answer>", "<answer>B</answer>", "<answer>C</answer>"]
        queries = [f"<search>evidence for {name}</search>" for name in ("A", "B", "C")]
        self.route(["ANSWER"] * 3)
        self.respond("ANSWER", answers)
        self.route(["SEARCH"] * 3)
        self.respond("SEARCH", queries)
        self.route(["FINISH", "SEARCH", "VERIFY"])
        next_query = "<search>more evidence for B</search>"
        self.respond("SEARCH", ["ignored", next_query, "ignored"])
        self.respond("VERIFY", ["ignored", "ignored", "C needs another check"])
        self.route(["ANSWER"] * 3)
        self.respond("ANSWER", ["ignored", "ignored", "invalid answer"], valid=[True, True, False])
        self.route(["FINISH"] * 3)
        env = ScriptedEnvironment(["A problem", "B problem", "C problem"], [
            (["A evidence", "B evidence", "C evidence"], [0.0] * 3, [False] * 3),
            (["A done", "More B evidence", "C done"], [1.0, 0.0, 1.0], [True, False, True]),
            (None, [0.0, 1.0, 0.0], [True, True, True]),
        ])
        rows, *_ = self.collect(env)
        self.assertEqual(env.actions, [queries, [answers[0], next_query, answers[2]], ["", answers[1], ""]])
        self.assertEqual(orchestra.text_actions, env.actions[-1])
        self.assertEqual(orchestra.candidate_answers, answers)
        self.assertEqual([sample[-1]["return_reason"] for sample in rows], ["finish", "finish", "loop_limit"])
        self.assertEqual([row["return_reason"] for row in rows[1] if row["env_step"] == 2][-1], "search")
        self.assertEqual(orchestra.agents["ANSWER"].calls[-1]["agent_active_mask"].tolist(), [False, False, True])
        self.assertFalse(rows[2][-1]["is_action_valid"])
        self.assertNotIn("invalid answer", "".join(orchestra.team_context[2]))

    def test_collector_invalid_routes_are_bounded_without_a_candidate(self):
        self.make_orchestra(loops=2)
        self.route(["FINISH"], valid=[False])
        self.route(["UNKNOWN"], valid=[False])
        env = ScriptedEnvironment(["Problem"], [(None, [0.0], [True])])
        rows, *_ = self.collect(env)
        self.assertEqual(env.actions, [[""]])
        self.assertEqual(len(rows[0]), 2)
        self.assertTrue(all(not row["is_action_valid"] for row in rows[0]))
        self.assertEqual(rows[0][-1]["return_reason"], "loop_limit")

    def test_collector_rejects_finish_without_a_candidate(self):
        orchestra = self.make_orchestra(loops=1)
        self.route(["FINISH"])
        env = ScriptedEnvironment(["Problem"], [(None, [0.0], [True])])
        rows, *_ = self.collect(env)
        self.assertEqual(env.actions, [[""]])
        self.assertEqual(len(rows[0]), 1)
        self.assertFalse(rows[0][0]["is_action_valid"])
        self.assertEqual(rows[0][0]["return_reason"], "loop_limit")
        self.assertEqual(rows[0][0]["execution_status"], "state_error")
        self.assertIn("requires a saved candidate", rows[0][0]["action_error"])
        self.assertTrue(all(not orchestra.agents[role].calls for role in orchestra.role_ids))

    def test_collector_still_calls_other_orchestras_once_per_environment_step(self):
        self.make_orchestra()
        self.config.agent.orchestra_type = "math"
        calls = []

        def run(**kwargs):
            calls.append(kwargs)
            batch = FakeBatch(1)
            batch.non_tensor_batch["agent_active_mask"] = np.array([True])
            return ["answer"], [{"agent_id": "Solver Agent", "batch": batch}]

        self.orchestra = SimpleNamespace(reset=lambda: None, run=run)
        env = ScriptedEnvironment(["Problem"], [(None, [1.0], [True])])
        self.collect(env)
        self.assertEqual(len(calls), 1)
        self.assertEqual(env.actions, [["answer"]])

    def test_same_role_updates_latest_output_per_sample_until_finish(self):
        orchestra = self.make_orchestra(roles=("SOLVE",))
        self.route(["SOLVE"] * 3)
        self.respond("SOLVE", ["A answer", "B answer", "C answer"])
        self.assertEqual(self.run_round(active=(True, True, True))[0], ["A answer", "B answer", "C answer"])
        self.assertEqual(orchestra.return_reasons.tolist(), ["", "", ""])
        self.assertIn("A answer", "".join(orchestra.team_context[0]))

        self.route(["SOLVE", "SOLVE", "FINISH"])
        self.respond("SOLVE", ["A revised", "B revised", "ignored"])
        self.assertEqual(self.run_round(active=(True, True, True))[0], ["A revised", "B revised", "C answer"])
        calls = orchestra.agents["SOLVE"].calls
        self.assertEqual(calls[-1]["agent_active_mask"].tolist(), [True, True, False])
        self.assertIs(calls[0]["actor_rollout_wg"], calls[1]["actor_rollout_wg"])
        self.route(["FINISH"] * 3)
        self.run_round(active=(True, True, True))
        self.assertEqual(orchestra.return_reasons.tolist(), ["finish", "finish", "finish"])

    def test_invalid_role_outputs_do_not_update_context_or_actions(self):
        orchestra = self.make_orchestra(roles=("SOLVE",))
        self.route(["SOLVE", "SOLVE"])
        self.respond(
            "SOLVE", ["valid answer", "invalid answer"], valid=[True, False],
        )
        self.assertEqual(self.run_round(active=(True, True))[0], ["valid answer", ""])
        self.assertIn("valid answer", "".join(orchestra.team_context[0]))
        self.assertNotIn("invalid answer", "".join(orchestra.team_context[1]))
        self.assertEqual(orchestra.return_reasons.tolist(), ["", ""])
        self.route(["FINISH", "FINISH"])
        actions, buffer = self.run_round(active=(True, True))
        self.assertEqual(actions, ["valid answer", ""])
        self.assertEqual(buffer[0]["batch"].non_tensor_batch["is_action_valid"].tolist(), [True, False])
        self.assertEqual(orchestra.return_reasons.tolist(), ["finish", ""])

    def test_reviews_and_other_roles_preserve_candidate_answers_and_training_rows(self):
        for scene, answer_role, other_scene_role in (("math", "SOLVE", "ANSWER"),
                                                      ("search", "ANSWER", "REVISE")):
            with self.subTest(scene=scene):
                orchestra = self.make_orchestra(
                    loops=5, roles=(answer_role, "VERIFY", "PLANNER", other_scene_role), env_name=scene,
                )
                self.route([answer_role])
                self.respond(answer_role, ["candidate answer"])
                for role in ("VERIFY", "PLANNER", other_scene_role):
                    self.route([role])
                    self.respond(role, [f"feedback from {role}"])
                self.route(["FINISH"])
                env = ScriptedEnvironment(["Problem"], [(None, [1.0], [True])])
                rows, *_ = self.collect(env)
                self.assertEqual(env.actions, [["candidate answer"]])
                self.assertEqual(orchestra.return_reasons.tolist(), ["finish"])
                final_context = orchestra.agents["Router Agent"].calls[-1]["team_context"][0]
                self.assertIn("feedback from VERIFY", final_context)
                self.assertIn("# Latest Recorded Review", final_context)
                self.assertNotIn("candidate answer", final_context)
                self.assertEqual(orchestra.agents["Router Agent"].calls[-1]["candidate_answers"], ["candidate answer"])
                for role in ("PLANNER", other_scene_role):
                    self.assertIn(f"feedback from {role}", final_context)
                for role in ("VERIFY", "PLANNER", other_scene_role):
                    self.assertEqual(sum(row["agent_id"] == role for row in rows[0]), 1)


if __name__ == "__main__":
    unittest.main()
