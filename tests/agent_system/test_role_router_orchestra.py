"""Exercise the new Orchestra contract with scripted agents, without model loading.

Real RouterAgent/SubAgent integration is covered by test_role_router_agent.py.
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

    def enqueue(self, responses, selections=None, valid=None):
        batch = FakeBatch(len(responses))
        batch.batch["responses"] = tuple(responses)
        batch.non_tensor_batch["is_action_valid"] = np.array(
            valid if valid is not None else [True] * len(responses), dtype=bool,
        )
        batch.non_tensor_batch["selected_role"] = np.array(
            selections if selections is not None else [""] * len(responses), dtype=object,
        )
        self.script.append((batch, responses))
        return batch

    def call(self, **kwargs):
        self.calls.append({
            **kwargs,
            "team_context": list(kwargs["team_context"]),
            "agent_active_mask": kwargs["agent_active_mask"].copy(),
        })
        if not self.script:
            raise AssertionError(f"Unexpected call to {self.name}")
        batch, responses = self.script.popleft()
        mask = kwargs["agent_active_mask"]
        metadata = batch.non_tensor_batch
        metadata["is_action_valid"] &= mask
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
    return result.RouterMultiAgentOrchestra, registry.AgentRegistry


RouterOrchestra, AgentRegistry = load_orchestra()


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
    def make_orchestra(self, loops=4, roles=("SOLVE", "VERIFY", "REVISE")):
        names = ["Router Agent", *roles]
        self.config = SimpleNamespace(
            agent=SimpleNamespace(orchestra_type="router", orchestra=SimpleNamespace(
                router=SimpleNamespace(max_loop_num=loops))),
            env=SimpleNamespace(max_steps=2),
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

    def respond(self, role, responses, valid=None):
        return self.orchestra.agents[role].enqueue(responses, valid=valid)

    def run_round(self, active=(True,), step=1, observation=None):
        actions, buffer = self.orchestra.run(
            FakeBatch(len(active)), {"text": observation or ["Problem"] * len(active)},
            self.groups, np.array(active), step,
        )
        return actions, list(buffer)

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

        self.route(["VERIFY"])
        self.respond("VERIFY", ["The derivation contains an error"])
        self.assertEqual(self.run_round()[0], ["The derivation contains an error"])
        self.assertIn("Initial answer", orchestra.agents["Router Agent"].calls[-1]["team_context"][0])
        self.route(["REVISE"])
        revised_batch = self.respond("REVISE", ["Corrected answer"])
        self.assertEqual(self.run_round()[0], ["Corrected answer"])
        revision = orchestra.agents["REVISE"].calls[0]
        self.assertIn('The output of "VERIFY": The derivation contains an error', revision["team_context"][0])
        self.assertIn("Assignment: REVISE", revision["team_context"][0])
        self.assertIs(revision["actor_rollout_wg"], self.groups["policy:REVISE"])
        self.assertEqual(revised_batch.batch["responses"], ("Corrected answer",))
        self.assertEqual(actions, ["Initial answer"])
        self.assertEqual(len(first_buffer), 2)

        self.route(["FINISH"])
        actions, final_buffer = self.run_round()
        self.assertEqual(actions, ["Corrected answer"])
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
        self.assertEqual(orchestra.team_context[2], "")
        self.assertNotIn("must ignore", "".join(orchestra.team_context))

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
        self.route(["FINISH"])
        self.run_round()
        orchestra.reset()
        self.route(["VERIFY"])
        self.respond("VERIFY", ["new report"])
        self.assertEqual(self.run_round()[0], ["new report"])
        self.assertNotIn("old answer", orchestra.team_context[0])
        self.assertEqual(orchestra.return_reasons.tolist(), [""])
        self.assertTrue(all(agent.reset_count == 1 for agent in orchestra.agents.values()))

    def test_no_active_samples_make_no_model_calls(self):
        orchestra = self.make_orchestra()
        self.assertEqual(self.run_round(active=(False, False)), (["", ""], []))
        self.assertTrue(all(not agent.calls for agent in orchestra.agents.values()))

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
        orchestra = self.make_orchestra(roles=("SEARCH", "ANSWER"))
        self.route(["SEARCH", "ANSWER"])
        self.respond("SEARCH", ["search query", "ignored"])
        self.respond("ANSWER", ["ignored", "B answer"])
        self.route(["FINISH", "FINISH"])
        self.route(["UNKNOWN", "ANSWER"], valid=[False, True])
        self.route(["ANSWER", "ANSWER"])
        self.respond("ANSWER", ["A answer from evidence", "ignored"])
        self.route(["FINISH", "ANSWER"])
        env = ScriptedEnvironment(["A problem", "B problem"], [
            (["A retrieved evidence", "B done"], [0.0, 1.0], [False, True]),
            (None, [1.0, 0.0], [True, True]),
        ])
        rows, rewards, lengths, _, _, _ = self.collect(env)
        self.assertEqual(env.actions, [["search query", "B answer"], ["A answer from evidence", ""]])
        self.assertEqual([row["return_reason"] for row in rows[0]], ["", "", "finish", "", "", "", "finish"])
        self.assertEqual(rows[1][-1]["return_reason"], "finish")
        calls = orchestra.agents["Router Agent"].calls
        self.assertEqual(calls[1]["agent_active_mask"].tolist(), [True, True])
        self.assertTrue(all(call["agent_active_mask"].tolist() == [True, False] for call in calls[2:]))
        self.assertEqual(calls[2]["team_context"], ["", ""])
        self.assertIn("retrieved evidence", calls[2]["env_obs"]["text"][0])
        self.assertEqual(lengths.tolist(), [2.0, 1.0])
        self.assertEqual(rewards.tolist(), [1.0, 1.0])
        self.assertFalse(rows[0][3]["is_action_valid"])

    def test_collector_budget_fallback_submits_latest_output_and_labels_last_call(self):
        self.make_orchestra(loops=2)
        self.route(["SOLVE"])
        self.respond("SOLVE", ["answer"])
        self.route(["VERIFY"])
        self.respond("VERIFY", ["review"])
        env = ScriptedEnvironment(["Problem"], [(None, [1.0], [True])])
        rows, *_ = self.collect(env)
        self.assertEqual(env.actions, [["review"]])
        self.assertEqual([row["return_reason"] for row in rows[0]], ["", "", "", "loop_limit"])
        self.assertEqual([row["selected_role"] for row in rows[0]], ["SOLVE", "", "VERIFY", ""])

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

    def test_collector_accepts_finish_without_a_candidate(self):
        orchestra = self.make_orchestra()
        self.route(["FINISH"])
        env = ScriptedEnvironment(["Problem"], [(None, [0.0], [True])])
        rows, *_ = self.collect(env)
        self.assertEqual(env.actions, [[""]])
        self.assertEqual(len(rows[0]), 1)
        self.assertTrue(rows[0][0]["is_action_valid"])
        self.assertEqual(rows[0][0]["return_reason"], "finish")
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
        orchestra = self.make_orchestra(roles=("Researcher",))
        self.route(["Researcher"] * 3)
        self.respond("Researcher", ["partial work", "B answer", "C query"])
        self.assertEqual(self.run_round(active=(True, True, True))[0], ["partial work", "B answer", "C query"])
        self.assertEqual(orchestra.return_reasons.tolist(), ["", "", ""])
        self.assertIn("partial work", orchestra.team_context[0])

        self.route(["Researcher", "Researcher", "FINISH"])
        self.respond("Researcher", ["A answer", "review of B", "ignored"])
        self.assertEqual(self.run_round(active=(True, True, True))[0], ["A answer", "review of B", "C query"])
        calls = orchestra.agents["Researcher"].calls
        self.assertEqual(calls[-1]["agent_active_mask"].tolist(), [True, True, False])
        self.assertIs(calls[0]["actor_rollout_wg"], calls[1]["actor_rollout_wg"])
        self.route(["FINISH"] * 3)
        self.run_round(active=(True, True, True))
        self.assertEqual(orchestra.return_reasons.tolist(), ["finish", "finish", "finish"])

    def test_invalid_role_outputs_do_not_update_context_or_actions(self):
        orchestra = self.make_orchestra(roles=("Researcher",))
        self.route(["Researcher", "Researcher"])
        self.respond(
            "Researcher", ["only a report", "invalid query"], valid=[True, False],
        )
        self.assertEqual(self.run_round(active=(True, True))[0], ["only a report", ""])
        self.assertIn("only a report", orchestra.team_context[0])
        self.assertNotIn("invalid query", orchestra.team_context[1])
        self.assertEqual(orchestra.return_reasons.tolist(), ["", ""])
        self.route(["FINISH", "FINISH"])
        actions, buffer = self.run_round(active=(True, True))
        self.assertEqual(actions, ["only a report", ""])
        self.assertEqual(buffer[0]["batch"].non_tensor_batch["is_action_valid"].tolist(), [True, True])
        self.assertEqual(orchestra.return_reasons.tolist(), ["finish", "finish"])


if __name__ == "__main__":
    unittest.main()
