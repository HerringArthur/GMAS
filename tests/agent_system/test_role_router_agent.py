"""Test role agents and routing with mocked preprocessing and model generation."""

import ast
import importlib.util
import re
import sys
import unittest
from collections import deque
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load_agents():
    def module(name, **attributes):
        result = ModuleType(name)
        result.__dict__.update(attributes)
        return result

    def unused(*args, **kwargs):
        raise AssertionError("Model execution must be mocked.")

    def load(name, relative_path):
        spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
        result = importlib.util.module_from_spec(spec)
        sys.modules[name] = result
        spec.loader.exec_module(result)
        return result

    # Execute the real projections and answer extraction without importing
    # unrelated grading dependencies.
    projection_module = module("agent_system.agent.utils", np=np, re=re, List=list, Tuple=tuple)
    for relative_path, function_names in (
        ("agent_system/environments/env_package/math/utils.py",
         {"last_boxed_only_string", "remove_boxed", "extract_boxed_answer", "extract_answer"}),
        ("agent_system/agent/utils.py", {"general_projection", "math_projection"}),
    ):
        projection_path = ROOT / relative_path
        functions = [
            node for node in ast.parse(projection_path.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name in function_names
        ]
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(projection_path), "exec"),
             projection_module.__dict__)

    stubs = {
        "transformers": module("transformers", PreTrainedTokenizer=object),
        "verl": module("verl", DataProto=object),
        "verl.protocol": module("verl.protocol", **{
            name: unused for name in (
                "pad_dataproto_to_divisor", "unpad_dataproto",
                "extract_dataproto_via_active_mask", "restore_dataproto_via_active_mask",
            )
        }),
        "agent_system.multi_turn_rollout.utils": module(
            "agent_system.multi_turn_rollout.utils", preprocess_batch=unused,
        ),
        "agent_system.agent.utils": projection_module,
    }
    with patch.dict(sys.modules, stubs):
        registry = load("agent_system.agent.registry", "agent_system/agent/registry.py")
        load("agent_system.agent.agents.base", "agent_system/agent/agents/base.py")
        router = load("agent_system.agent.agents.router.router_agent", "agent_system/agent/agents/router/router_agent.py")
        role_modules = {}
        for scene, role_files in {
            "math": {"SOLVE": "solver_agent", "VERIFY": "verifier_agent", "REVISE": "reviser_agent"},
            "search": {"SEARCH": "search_agent", "VERIFY": "verifier_agent", "ANSWER": "answer_agent"},
        }.items():
            role_modules[scene] = {
                role: load(f"agent_system.agent.agents.router.{scene}.{filename}",
                           f"agent_system/agent/agents/router/{scene}/{filename}.py")
                for role, filename in role_files.items()
            }
        package = load("agent_system.agent.agents.router", "agent_system/agent/agents/router/__init__.py")
        load("agent_system.agent.orchestra.base", "agent_system/agent/orchestra/base.py")
        orchestra = load("role_agents_orchestra_under_test", "agent_system/agent/orchestra/router/router_orchestra.py")
    return router, role_modules, package, registry.AgentRegistry, orchestra.RouterMultiAgentOrchestra


router_module, role_modules, router_package, AgentRegistry, RouterOrchestra = load_agents()


class RoleRouterAgentTests(unittest.TestCase):
    def config(self, scene):
        roles = ["SEARCH", "VERIFY", "ANSWER"] if scene == "search" else ["SOLVE", "VERIFY", "REVISE"]
        return SimpleNamespace(
            agent=SimpleNamespace(role_ids=roles, orchestra_type="router", orchestra=SimpleNamespace(
                router=SimpleNamespace(max_loop_num=4, max_context_runs=1))),
            env=SimpleNamespace(env_name=scene, max_steps=2),
        )

    def generate(self, role, scene, responses, active=None):
        config = self.config(scene)
        module = router_module if role == "Router Agent" else role_modules[scene][role]
        cls = {"Router Agent": "RouterAgent", "SEARCH": "SearchAgent", "ANSWER": "AnswerAgent",
               "VERIFY": "VerifierAgent", "SOLVE": "SolverAgent", "REVISE": "ReviserAgent"}[role]
        agent = getattr(module, cls)("policy", None, None, config)
        size = len(responses)
        mask = np.array(active if active is not None else [True] * size)
        raw_tokens = tuple(responses)
        batch = SimpleNamespace(batch={"responses": raw_tokens}, non_tensor_batch={})
        original = list(responses)
        with patch.object(module, "preprocess_batch", return_value=batch) as preprocess:
            with patch.object(agent, "_generate_with_llm", return_value=(batch, responses)):
                returned, rendered = agent.call(
                    SimpleNamespace(meta_info={}), {"text": [f"Question {i}" for i in range(size)]},
                    [f"Evidence {i}" for i in range(size)], object(), mask, step=1,
                    candidate_answers=[f"Candidate {i}" for i in range(size)],
                )
        self.assertIs(returned, batch)
        self.assertIs(batch.batch["responses"], raw_tokens)
        self.assertEqual(responses, original)
        prompts = preprocess.call_args.kwargs["obs"]["text"]
        for i, prompt in enumerate(prompts):
            self.assertIn(f"Question {i}", prompt)
            self.assertIn(f"Candidate {i}", prompt)
            self.assertIn(f"Evidence {i}", prompt)
            self.assertIn("# Function Call Output", prompt)
            if size > 1:
                self.assertNotIn(f"Candidate {(i+1) % size}", prompt)
        return batch.non_tensor_batch, rendered, prompts

    def test_router_choices_tasks_inactive_rows_and_code_token_preservation(self):
        metadata, rendered, _ = self.generate("Router Agent", "search", [
            'call_agent(role="SEARCH", task="Find date")', 'finish()',
            '<route>SEARCH</route>', 'call_agent(role="SOLVE")', 'call_agent(role="VERIFY")',
        ], [True, True, True, True, False])
        self.assertEqual(metadata["selected_role"].tolist(), ["SEARCH", "FINISH", "", "", ""])
        self.assertEqual(metadata["routing_instruction"].tolist(), ["Find date", "", "", "", ""])
        self.assertEqual(metadata["is_action_valid"].tolist(), [True, True, False, False, False])
        self.assertEqual(rendered[:2], ['call_agent(role="SEARCH", task="Find date")', 'finish()'])

    def test_search_and_answer_render_only_environment_payload(self):
        meta, actions, _ = self.generate("SEARCH", "search", ['search(query="北京")', 'search(query="")', 'finish()'])
        self.assertEqual(actions, ['<search>北京</search>', '', ''])
        self.assertEqual(meta["search_executable"].tolist(), [True, False, False])
        meta, actions, _ = self.generate("ANSWER", "search", [
            'save_candidate(answer="Paris", explanation="Evidence supports it")', 'call_agent(role="SEARCH")',
        ])
        self.assertEqual(actions, ['<answer>Paris</answer>', ''])
        self.assertEqual(meta["action_args"][0]["explanation"], "Evidence supports it")
        self.assertIn("unavailable", meta["action_error"][1])

    def test_math_solution_and_revision_keep_complete_derivation(self):
        for role in ("SOLVE", "REVISE"):
            meta, actions, _ = self.generate(role, "math", [
                r'save_candidate(answer=r"\frac{1}{2}", solution="Complete derivation")',
                'save_candidate(answer="2")',
            ])
            self.assertEqual(actions, ['Complete derivation\n\\boxed{\\frac{1}{2}}', ''])
            self.assertEqual(meta["is_action_valid"].tolist(), [True, False])

    def test_verifiers_use_distinct_boolean_semantics(self):
        for scene, verdict, wrong in (("search", "sufficient", "approved"), ("math", "approved", "sufficient")):
            meta, actions, _ = self.generate("VERIFY", scene, [
                f'record_review({verdict}=False, feedback="Missing support")',
                f'record_review({wrong}=True, feedback="Wrong schema")', 'free text',
            ])
            self.assertEqual(meta["is_action_valid"].tolist(), [True, False, False])
            self.assertIn(f"{verdict}=False", actions[0])
            self.assertEqual(actions[1:], ['', ''])

    def test_scene_specific_registration_keeps_worker_binding(self):
        for scene in ("search", "math"):
            agent = router_package.create_verifier_agent("verify-policy", None, None, self.config(scene))
            self.assertIsInstance(agent, role_modules[scene]["VERIFY"].VerifierAgent)
            self.assertEqual(agent.wg_id, "verify-policy")

    def run_real_agents(self, scene, scripts, transitions):
        from tests.agent_system.test_role_router_orchestra import FakeBatch, ScriptedEnvironment, collect
        config = self.config(scene)
        roles = config.agent.role_ids
        names = ["Router Agent", *roles]
        mapping = {name: f"policy:{name}" for name in names}
        groups = {group: object() for group in mapping.values()}
        # Other fixtures also register scripted agents; restore this registry explicitly.
        AgentRegistry._REGISTRY["Router Agent"] = router_module.RouterAgent
        for role, module in role_modules[scene].items():
            cls = {"SEARCH": "SearchAgent", "ANSWER": "AnswerAgent", "VERIFY": "VerifierAgent",
                   "SOLVE": "SolverAgent", "REVISE": "ReviserAgent"}[role]
            AgentRegistry._REGISTRY[role] = getattr(module, cls)
        with patch.dict(sys.modules, {"agent_system.agent.agents.router": router_package}):
            orchestra = RouterOrchestra(names, roles, ["checkpoint"] * len(names), mapping,
                                        dict.fromkeys(groups), dict.fromkeys(groups), config)
        scripts = {name: deque(items) for name, items in scripts.items()}
        prompts, raw_batches = [], []
        size = len(transitions[0][1])

        def preprocess(**kwargs):
            prompts.extend(kwargs["obs"]["text"])
            return FakeBatch(size)

        def generate(name, batch, group, mask, meta):
            self.assertIs(group, groups[mapping[name]])
            responses = scripts[name].popleft()
            batch.batch["responses"] = tuple(responses)
            batch.non_tensor_batch["wg_id"] = np.full(size, mapping[name], dtype=object)
            batch.non_tensor_batch["agent_active_mask"] = mask.copy()
            raw_batches.append((name, batch, tuple(responses)))
            return batch, responses

        env = ScriptedEnvironment([f"Question {i}" for i in range(size)], transitions)
        gen = FakeBatch(size)
        gen.batch = np.zeros((size, 1))
        gen.meta_info = {}
        with ExitStack() as stack:
            stack.enter_context(patch.object(router_module, "preprocess_batch", side_effect=preprocess))
            for module in role_modules[scene].values():
                stack.enter_context(patch.object(module, "preprocess_batch", side_effect=preprocess))
            for name, agent in orchestra.agents.items():
                stack.enter_context(patch.object(agent, "_generate_with_llm", side_effect=partial(generate, name)))
            rows, rewards, *_ = collect(SimpleNamespace(config=config, multiagent_orchestra=orchestra),
                                        gen, groups, env, effective_rollout_n=1)
        self.assertTrue(all(not queue for queue in scripts.values()))
        for _, batch, source in raw_batches:
            self.assertEqual(batch.batch["responses"], source)
        self.assertTrue(all(set(row) == set(rows[0][0]) for trajectory in rows for row in trajectory))
        return orchestra, rows, rewards, env, prompts

    def test_search_execution_feedback_save_review_finish_and_batch_isolation(self):
        scripts = {
            "Router Agent": [
                ['finish()', 'call_agent(role="ANSWER")'],
                ['call_agent(role="SEARCH", task="Find newborn term")', 'finish()'],
                ['call_agent(role="ANSWER")', 'ignored'],
                ['call_agent(role="VERIFY")', 'ignored'],
                ['finish()', 'ignored'],
            ],
            "ANSWER": [['ignored', 'save_candidate(answer="B")'], ['save_candidate(answer="neonate")', 'ignored']],
            "SEARCH": [['search(query="newborn infant first weeks")', 'ignored']],
            "VERIFY": [['record_review(sufficient=True, feedback="Evidence supports neonate")', 'ignored']],
        }
        orchestra, rows, rewards, env, prompts = self.run_real_agents("search", scripts, [
            (["Evidence: newborn means neonate", ""], [0., 1.], [False, True]),
            (None, [1., 0.], [True, True]),
        ])
        self.assertEqual(env.actions, [['<search>newborn infant first weeks</search>', '<answer>B</answer>'],
                                       ['<answer>neonate</answer>', '']])
        self.assertFalse(rows[0][0]["is_action_valid"])
        self.assertEqual(rows[0][0]["execution_status"], "state_error")
        self.assertTrue(any("ActionError: finish() requires" in p for p in prompts))
        self.assertTrue(any("# Router Assignment\nSelected role: SEARCH\nTask: Find newborn term" in p for p in prompts))
        search_row = next(row for row in rows[0] if row["agent_id"] == "SEARCH")
        self.assertEqual(search_row["environment_result"], "Evidence: newborn means neonate")
        self.assertEqual(search_row["environment_action"], env.actions[0][0])
        self.assertEqual(orchestra.reviews[0]["candidate_version"], 1)
        self.assertEqual(orchestra.reviews[0]["env_step"], 2)
        self.assertEqual(orchestra.role_paths, [("SEARCH", "ANSWER", "VERIFY"), ("ANSWER",)])
        self.assertEqual([row["agent_id"] for row in rows[1]], ["Router Agent", "ANSWER", "Router Agent"])
        # Actual invalid calls still receive the existing training penalty.
        from tests.trainer.ppo import test_valid_action_training as support
        batch = support.rollout([r["wg_id"] for r in rows[0]], [r["is_action_valid"] for r in rows[0]])
        batch.batch["token_level_scores"] = support.tensor([[0., rewards[0]]] * len(rows[0]))
        support.load_driver()["apply_invalid_action_penalty"](batch, invalid_action_penalty_coef=0.1)
        np.testing.assert_allclose(batch.batch["token_level_scores"].sum(dim=-1), [0.9] + [1.] * (len(rows[0])-1))

    def test_math_solve_review_revise_finish_retains_stale_review_version(self):
        scripts = {
            "Router Agent": [[f'call_agent(role="{role}")'] for role in ('SOLVE', 'VERIFY', 'REVISE')] + [['finish()']],
            "SOLVE": [['save_candidate(answer="3", solution="Initial derivation")']],
            "VERIFY": [['record_review(approved=False, feedback="Arithmetic error")']],
            "REVISE": [['save_candidate(answer="2", solution="Corrected derivation")']],
        }
        orchestra, rows, _, env, prompts = self.run_real_agents("math", scripts, [(None, [1.], [True])])
        self.assertEqual(env.actions, [['Corrected derivation\n\\boxed{2}']])
        self.assertEqual(orchestra.candidate_versions, [2])
        self.assertEqual(orchestra.reviews[0]["candidate_version"], 1)
        self.assertTrue(any('Reviewed candidate version: 1; current candidate version: 2' in p for p in prompts))
        self.assertEqual(orchestra.role_paths, [("SOLVE", "VERIFY", "REVISE")])
        self.assertEqual(rows[0][-1]["selected_role"], "FINISH")

    def test_bad_worker_call_reports_error_then_router_retries_without_losing_candidate(self):
        scripts = {
            "Router Agent": [['call_agent(role="ANSWER")'], ['call_agent(role="ANSWER")'],
                             ['call_agent(role="ANSWER")'], ['finish()']],
            "ANSWER": [['save_candidate(answer="Original")'], ['call_agent(role="SEARCH")'],
                       ['save_candidate(answer="Corrected")']],
        }
        orchestra, rows, _, env, prompts = self.run_real_agents("search", scripts, [(None, [1.], [True])])
        self.assertEqual(env.actions, [['<answer>Corrected</answer>']])
        self.assertTrue(any('ActionError: ANSWER can call save_candidate' in p and '<answer>Original</answer>' in p for p in prompts))
        self.assertFalse(rows[0][3]["is_action_valid"])
        self.assertEqual(orchestra.role_paths, [("ANSWER", "ANSWER", "ANSWER")])
        self.assertEqual(orchestra.candidate_versions, [2])


if __name__ == "__main__":
    unittest.main()
