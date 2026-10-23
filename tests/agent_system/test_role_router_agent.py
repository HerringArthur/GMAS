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
    def setUp(self):
        self.config = SimpleNamespace(
            agent=SimpleNamespace(role_ids=["SOLVE", "VERIFY", "REVISE"]),
            env=SimpleNamespace(env_name="math", max_steps=4),
        )
        self.router = AgentRegistry.create(
            "Router Agent", wg_id="router-policy", tokenizer=None, processor=None, config=self.config,
        )

    def call_router(self, responses, active=None, env_obs=None, team_context=None, candidate_answers=None, step=2):
        size = len(responses)
        mask = np.array(active if active is not None else [True] * size, dtype=bool)
        raw_response_tokens = object()
        batch = SimpleNamespace(
            batch={"responses": raw_response_tokens},
            non_tensor_batch={
                "wg_id": np.full(size, "router-policy", dtype=object),
                "agent_active_mask": mask,
            },
        )
        gen_batch = SimpleNamespace(meta_info={"temperature": 0.7})
        group = object()
        with patch.object(router_module, "preprocess_batch", return_value=batch) as preprocess:
            with patch.object(self.router, "_generate_with_llm", return_value=(batch, responses)) as generate:
                result, returned_responses = self.router.call(
                    gen_batch=gen_batch,
                    env_obs=env_obs if env_obs is not None else {"text": ["Problem"] * size},
                    team_context=team_context if team_context is not None else [""] * size,
                    actor_rollout_wg=group,
                    agent_active_mask=mask,
                    step=step,
                    candidate_answers=candidate_answers if candidate_answers is not None else [""] * size,
                )
        generate.assert_called_once_with(batch, group, mask, gen_batch.meta_info)
        self.assertIs(preprocess.call_args.kwargs["agent_active_mask"], mask)
        self.assertIs(result, batch)
        self.assertIs(returned_responses, responses)
        self.assertIs(result.batch["responses"], raw_response_tokens)
        self.assertIs(result.non_tensor_batch["agent_active_mask"], mask)
        self.assertEqual(result.non_tensor_batch["wg_id"].tolist(), ["router-policy"] * size)
        self.assertEqual(result.non_tensor_batch["env_step"].tolist(), [step] * size)
        return result, preprocess.call_args.kwargs["obs"]

    def test_candidates_are_visible_without_history_and_stay_aligned_per_sample(self):
        candidates = [
            r"Full derivation A with {x}, ending in \boxed{42}",
            "<think>Reasoning B</think><answer>Paris</answer>",
            "",
        ]
        _, obs = self.call_router(["<route>VERIFY</route>"] * 3, candidate_answers=candidates)
        for i, prompt in enumerate(obs["text"]):
            self.assertIn("# Current Candidate Answer", prompt)
            self.assertIn("Being saved does not imply that it is correct or has been verified.", prompt)
            if candidates[i]:
                self.assertIn(candidates[i], prompt)
                self.assertNotIn("No candidate answer has been generated yet.", prompt)
            else:
                self.assertIn("No candidate answer has been generated yet.", prompt)
            for j, candidate in enumerate(candidates):
                if i != j and candidate:
                    self.assertNotIn(candidate, prompt)

    def test_role_choices_finish_and_inactive_rows_preserve_batch_alignment(self):
        responses = [
            "<route>SOLVE</route>",
            "<think>Check the second step.\nExplain any error.</think>\n<route> VERIFY </route>",
            "<route>REVISE</route>",
            "<route>FINISH</route>",
            "<route>UNKNOWN</route>",
            "<route>Router Agent</route>",
            "<route>SOLVE</route>",
            "<route>FINISH</route>",
        ]
        batch, _ = self.call_router(responses, active=[True] * 6 + [False, False])
        self.assertEqual(
            batch.non_tensor_batch["selected_role"].tolist(),
            ["SOLVE", "VERIFY", "REVISE", "FINISH", "", "", "", ""],
        )
        self.assertEqual(
            batch.non_tensor_batch["is_action_valid"].tolist(),
            [True, True, True, True, False, False, False, False],
        )

    def test_malformed_or_ambiguous_routes_are_invalid(self):
        responses = [
            "",
            "<route></route>",
            "<route>SOLVE",
            "<route>SOLVE</route> extra",
            "<route>SOLVE</route><route>VERIFY</route>",
            "<think><route>SOLVE</route></think>",
            "<think><route>SOLVE</route></think><route>VERIFY</route>",
            "<think><think>nested</think></think><route>SOLVE</route>",
            "<think>one</think><think>two</think><route>SOLVE</route>",
            "<ROUTE>SOLVE</ROUTE>",
        ]
        batch, _ = self.call_router(responses)
        self.assertEqual(batch.non_tensor_batch["selected_role"].tolist(), [""] * len(responses))
        self.assertFalse(batch.non_tensor_batch["is_action_valid"].any())

    def test_prompt_and_validation_use_configured_roles(self):
        self.config.agent.role_ids = ["REVISE", "SOLVE"]
        self.router = router_module.RouterAgent("router-policy", None, None, self.config)
        env_obs = {"text": ["Find {x}", "Second problem"], "image": ["image A", "image B"]}
        batch, obs = self.call_router(
            ["<route>VERIFY</route>", "<route>SOLVE</route>"],
            env_obs=env_obs,
            team_context=["Earlier answer: 3", "Review: check the sign"],
        )
        self.assertIn("Find {x}", obs["text"][0])
        self.assertIn("Earlier answer: 3", obs["text"][0])
        self.assertNotIn("Earlier answer: 3", obs["text"][1])
        self.assertIn("Review: check the sign", obs["text"][1])
        self.assertIn("- REVISE: Use the existing solution", obs["text"][0])
        self.assertIn("- SOLVE: Analyze the problem", obs["text"][0])
        self.assertNotIn("- VERIFY:", obs["text"][0])
        self.assertIn("- FINISH:", obs["text"][0])
        self.assertEqual(obs["image"], env_obs["image"])
        self.assertEqual(env_obs["text"], ["Find {x}", "Second problem"])
        self.assertEqual(batch.non_tensor_batch["selected_role"].tolist(), ["", "SOLVE"])
        self.assertEqual(batch.non_tensor_batch["is_action_valid"].tolist(), [False, True])

    def test_final_search_step_prompt_removes_search_only_for_that_step(self):
        self.config.env.env_name = "search"
        self.config.agent.role_ids = ["SEARCH", "VERIFY", "ANSWER"]
        self.router = router_module.RouterAgent("router-policy", None, None, self.config)
        choices = ["<route>VERIFY</route>", "<route>ANSWER</route>", "<route>FINISH</route>"]
        batch, obs = self.call_router(choices, step=4)
        for prompt in obs["text"]:
            self.assertNotIn("- SEARCH:", prompt)
            self.assertIn("final environment step", prompt)
            self.assertIn("SEARCH is unavailable", prompt)
            for role in ("VERIFY", "ANSWER", "FINISH"):
                self.assertIn(f"- {role}:", prompt)
        self.assertTrue(batch.non_tensor_batch["is_action_valid"].all())

        self.router.reset()
        batch, obs = self.call_router(["<route>SEARCH</route>"], step=1)
        self.assertIn("- SEARCH:", obs["text"][0])
        self.assertNotIn("final environment step", obs["text"][0])
        self.assertTrue(batch.non_tensor_batch["is_action_valid"][0])
        self.assertEqual(self.config.agent.role_ids, ["SEARCH", "VERIFY", "ANSWER"])

    def test_scenarios_expose_only_their_roles_and_use_distinct_verification_instructions(self):
        scenarios = [
            ("math", ["SOLVE", "VERIFY", "REVISE"], ["SEARCH", "ANSWER"], "mathematical derivation"),
            ("search", ["SEARCH", "VERIFY", "ANSWER"], ["SOLVE", "REVISE"], "retrieved evidence"),
        ]
        for env_name, roles, other_roles, verification_text in scenarios:
            with self.subTest(env_name=env_name):
                self.config.env.env_name = env_name
                self.config.agent.role_ids = roles
                self.router = router_module.RouterAgent("router-policy", None, None, self.config)
                choices = roles + ["FINISH"] + other_roles
                batch, obs = self.call_router([f"<route>{role}</route>" for role in choices])
                for role in roles:
                    self.assertIn(f"- {role}:", obs["text"][0])
                for role in other_roles:
                    self.assertNotIn(f"- {role}:", obs["text"][0])
                self.assertIn(verification_text, obs["text"][0])
                self.assertEqual(
                    batch.non_tensor_batch["selected_role"].tolist(), roles + ["FINISH", "", ""],
                )
                self.assertEqual(
                    batch.non_tensor_batch["is_action_valid"].tolist(), [True] * 4 + [False, False],
                )


class RoleWorkerAgentTests(unittest.TestCase):
    def test_scene_specific_verifiers_keep_independent_classes_and_worker_bindings(self):
        agents = []
        for scene, wg_id in (("math", "math-verify"), ("search", "search-verify"), ("math", "second-math-verify")):
            agent = AgentRegistry.create(
                "VERIFY", wg_id=wg_id, tokenizer=None, processor=None,
                config=SimpleNamespace(env=SimpleNamespace(env_name=scene)),
            )
            self.assertIsInstance(agent, role_modules[scene]["VERIFY"].VerifierAgent)
            self.assertEqual(agent.name, "VERIFY")
            self.assertEqual(agent.wg_id, wg_id)
            agents.append(agent)
        self.assertIsNot(type(agents[0]), type(agents[1]))
        self.assertIs(type(agents[0]), type(agents[2]))
        self.assertIn("mathematical solution", agents[0].prompt)
        self.assertIn("retrieved evidence", agents[1].prompt)
        self.assertEqual(agents[0].prompt, agents[2].prompt)

    def test_registered_roles_receive_scene_prompts_and_shared_assignment(self):
        scenarios = {
            "math": {
                "SOLVE": "complete solution",
                "VERIFY": "mathematical solution",
                "REVISE": "complete revised solution",
            },
            "search": {
                "SEARCH": "<search>your query</search>",
                "VERIFY": "retrieved evidence",
                "ANSWER": "<answer>final answer</answer>",
            },
        }
        for env_name, roles in scenarios.items():
            config = SimpleNamespace(env=SimpleNamespace(env_name=env_name))
            for role, expected_instruction in roles.items():
                with self.subTest(env_name=env_name, role=role):
                    agent = AgentRegistry.create(
                        role, wg_id=f"policy:{role}", tokenizer=None, processor=None, config=config,
                    )
                    observation = {"text": ["Original task {x}"], "image": ["image"]}
                    context = ['The output of "Router Agent": <think>Check the evidence.</think>']
                    candidate = r"Candidate with full reasoning and \boxed{42}"
                    batch = SimpleNamespace(non_tensor_batch={})
                    response = {
                        "SOLVE": r"Full solution: \boxed{42}",
                        "REVISE": r"Revised solution: \boxed{42}",
                        "VERIFY": "Review complete. <verify>approve</verify>" if env_name == "math" else "Role output",
                        "SEARCH": "<think>Find supporting evidence.</think><search>supporting evidence</search>",
                        "ANSWER": "<think>Use the evidence.</think><answer>Final answer</answer>",
                    }.get(role, "Role output")
                    agent_module = role_modules[env_name][role]
                    with patch.object(agent_module, "preprocess_batch", return_value=batch) as preprocess:
                        with patch.object(agent, "_generate_with_llm", return_value=(batch, [response])):
                            agent.call(
                                SimpleNamespace(meta_info={}), observation, context,
                                object(), np.array([True]), step=2, candidate_answers=[candidate],
                            )
                    prompt = preprocess.call_args.kwargs["obs"]
                    self.assertEqual(type(agent).__module__, agent_module.__name__)
                    self.assertEqual(agent.name, role)
                    self.assertEqual(agent.wg_id, f"policy:{role}")
                    self.assertIn(f"You are {role}.", prompt["text"][0])
                    self.assertIn(expected_instruction, prompt["text"][0])
                    self.assertIn(context[0], prompt["text"][0])
                    self.assertIn("# Current Candidate Answer", prompt["text"][0])
                    self.assertIn(candidate, prompt["text"][0])
                    self.assertIn("Original task {x}", prompt["text"][0])
                    self.assertIn("Step 2", prompt["text"][0])
                    if env_name == "math" and role in ("SOLVE", "REVISE"):
                        self.assertIn(r"\boxed{...}", prompt["text"][0])
                    if env_name == "math" and role == "VERIFY":
                        self.assertIn("<verify>approve</verify>", prompt["text"][0])
                        self.assertIn("<verify>reject</verify>", prompt["text"][0])
                    self.assertEqual(observation["text"], ["Original task {x}"])
                    self.assertEqual(prompt["image"], observation["image"])

    def test_math_answer_roles_check_boxed_answers_and_preserve_full_solutions(self):
        for role in ("SOLVE", "REVISE"):
            with self.subTest(role=role):
                wg_id = f"math:{role}"
                agent = AgentRegistry.create(
                    role, wg_id=wg_id, tokenizer=None, processor=None,
                    config=SimpleNamespace(env=SimpleNamespace(env_name="math")),
                )
                responses = [
                    r"Full derivation ending in \boxed{\frac{1}{2}}",
                    " \n ",
                    r"Inactive solution: \boxed{3}",
                    "Partial derivation without a boxed answer",
                    r"Unfinished answer: \boxed{\frac{1}{2}",
                    r"<think>Full derivation.</think>\boxed{4}",
                    r"Empty answer: \boxed{}",
                    r"First \boxed{1}, then an unfinished correction \boxed{2",
                ]
                size = len(responses)
                mask = np.array([True, True, False, True, True, True, True, True])
                raw_generation = tuple(responses)
                batch = SimpleNamespace(
                    batch={"responses": raw_generation},
                    non_tensor_batch={"wg_id": np.full(size, wg_id, dtype=object), "agent_active_mask": mask},
                )
                gen_batch = SimpleNamespace(meta_info={})
                group = object()
                with patch.object(role_modules["math"][role], "preprocess_batch", return_value=batch) as preprocess:
                    with patch.object(agent, "_generate_with_llm", return_value=(batch, responses)) as generate:
                        returned_batch, returned_responses = agent.call(
                            gen_batch, {"text": ["Problem"] * size}, [""] * size, group, mask,
                            step=2, candidate_answers=[""] * size,
                        )
                generate.assert_called_once_with(batch, group, mask, gen_batch.meta_info)
                self.assertIs(preprocess.call_args.kwargs["agent_active_mask"], mask)
                self.assertIs(returned_batch, batch)
                self.assertIs(batch.batch["responses"], raw_generation)
                self.assertEqual(returned_responses, list(raw_generation))
                self.assertIs(batch.non_tensor_batch["agent_active_mask"], mask)
                self.assertEqual(batch.non_tensor_batch["wg_id"].tolist(), [wg_id] * size)
                self.assertEqual(batch.non_tensor_batch["is_action_valid"].tolist(),
                                 [True, False, False, False, False, True, True, False])
                self.assertEqual(batch.non_tensor_batch["selected_role"].tolist(), [""] * size)
                self.assertEqual(batch.non_tensor_batch["env_step"].tolist(), [2] * size)

    def test_math_verifier_checks_tags_and_preserves_full_reports(self):
        agent = AgentRegistry.create(
            "VERIFY", wg_id="math-verify-policy", tokenizer=None, processor=None,
            config=SimpleNamespace(env=SimpleNamespace(env_name="math")),
        )
        responses = [
            "All steps are correct. <verify> approve </verify>",
            "<think>A root is missing.</think><verify>reject</verify>",
            "A root is missing, but there is no verdict tag.",
            "Unclosed verdict: <verify>reject",
            "答案正确。<verify>approve</verify>",
            "Unspecified verdict. <verify>uncertain</verify>",
            "Inactive report. <verify>approve</verify>",
        ]
        expected = responses.copy()
        expected[0] = "All steps are correct. <verify>approve</verify>"
        size = len(responses)
        mask = np.array([True] * (size - 1) + [False])
        raw_generation = tuple(responses)
        batch = SimpleNamespace(batch={"responses": raw_generation}, non_tensor_batch={})
        with patch.object(role_modules["math"]["VERIFY"], "preprocess_batch", return_value=batch):
            with patch.object(agent, "_generate_with_llm", return_value=(batch, responses)):
                returned_batch, returned_responses = agent.call(
                    SimpleNamespace(meta_info={}), {"text": ["Problem"] * size}, [""] * size,
                    object(), mask, step=1, candidate_answers=[""] * size,
                )
        self.assertIs(returned_batch, batch)
        self.assertIs(batch.batch["responses"], raw_generation)
        self.assertEqual(returned_responses, expected)
        self.assertEqual(batch.non_tensor_batch["is_action_valid"].tolist(),
                         [True, True, False, False, False, True, False])

    def test_call_preserves_responses_and_excludes_empty_and_inactive_rows(self):
        for scene, role in (("math", "VERIFY"), ("math", "REVISE"), ("search", "VERIFY")):
            with self.subTest(scene=scene, role=role):
                wg_id = f"{scene}:{role}"
                agent = AgentRegistry.create(
                    role, wg_id=wg_id, tokenizer=None, processor=None,
                    config=SimpleNamespace(env=SimpleNamespace(env_name=scene)),
                )
                response = {
                    ("math", "VERIFY"): "The second step has an error. <verify>reject</verify>",
                    ("math", "REVISE"): r"Correcting the second step gives \boxed{2}",
                }.get((scene, role), "The second step has an error.")
                responses = [response, " \n ", response, ""]
                mask = np.array([True, True, False, True])
                raw_tokens = object()
                batch = SimpleNamespace(
                    batch={"responses": raw_tokens},
                    non_tensor_batch={"wg_id": np.full(4, wg_id, dtype=object), "agent_active_mask": mask},
                )
                gen_batch = SimpleNamespace(meta_info={})
                group = object()
                candidates = [r"First derivation: \boxed{1}", "Second candidate", "Inactive candidate", ""]
                with patch.object(role_modules[scene][role], "preprocess_batch", return_value=batch) as preprocess:
                    with patch.object(agent, "_generate_with_llm", return_value=(batch, responses)) as generate:
                        returned_batch, returned_responses = agent.call(
                            gen_batch, {"text": ["Problem"] * 4}, ["shared work"] * 4, group, mask, step=1,
                            candidate_answers=candidates,
                        )
                generate.assert_called_once_with(batch, group, mask, gen_batch.meta_info)
                self.assertIs(preprocess.call_args.kwargs["agent_active_mask"], mask)
                self.assertIs(returned_batch, batch)
                self.assertIs(returned_responses, responses)
                self.assertIs(batch.batch["responses"], raw_tokens)
                self.assertEqual(batch.non_tensor_batch["is_action_valid"].tolist(), [True, False, False, False])
                self.assertEqual(batch.non_tensor_batch["selected_role"].tolist(), [""] * 4)
                self.assertEqual(batch.non_tensor_batch["env_step"].tolist(), [1] * 4)
                prompts = preprocess.call_args.kwargs["obs"]["text"]
                self.assertIn(candidates[0], prompts[0])
                self.assertNotIn(candidates[1], prompts[0])
                self.assertIn(candidates[1], prompts[1])
                self.assertNotIn(candidates[0], prompts[1])
                self.assertIn("No candidate answer has been generated yet.", prompts[3])

    def test_answer_uses_drmas_projection_and_preserves_raw_generation(self):
        agent = AgentRegistry.create(
            "ANSWER", wg_id="answer-policy", tokenizer=None, processor=None,
            config=SimpleNamespace(env=SimpleNamespace(env_name="search")),
        )
        responses = [
            "<think>Use the evidence.</think><answer> Beijing </answer>",
            "<answer>Beijing</answer>",
            "<think>Use the evidence.</think>Beijing",
            "<think>Use the evidence.</think><answer>北京</answer>",
            "<think>Use the evidence.</think><answer></answer>",
            "<think>Use the evidence.</think><answer>First</answer><answer>Second</answer>",
            "<think><answer>Inside thought</answer></think>",
            "<think>Use the evidence.</think></answer><answer>Reversed",
            "<think>One</think><think>Two</think><answer>Beijing</answer>",
            "<think>Inactive</think><answer>Beijing</answer>",
        ]
        expected = [
            "<answer>Beijing</answer>", "<answer>Beijing</answer>", "", "<answer>北京</answer>",
            "<answer></answer>", "<answer>First</answer>", "<answer>Inside thought</answer>",
            "<answer></answer>", "<answer>Beijing</answer>", "<answer>Beijing</answer>",
        ]
        size = len(responses)
        mask = np.array([True] * (size - 1) + [False])
        raw_generation = tuple(responses)
        batch = SimpleNamespace(batch={"responses": raw_generation}, non_tensor_batch={})
        with patch.object(role_modules["search"]["ANSWER"], "preprocess_batch", return_value=batch):
            with patch.object(agent, "_generate_with_llm", return_value=(batch, responses)):
                returned_batch, returned_responses = agent.call(
                    SimpleNamespace(meta_info={}), {"text": ["Question"] * size}, [""] * size,
                    object(), mask, step=1, candidate_answers=[""] * size,
                )
        self.assertIs(returned_batch, batch)
        self.assertIs(batch.batch["responses"], raw_generation)
        self.assertEqual(returned_responses, expected)
        self.assertEqual(batch.non_tensor_batch["is_action_valid"].tolist(),
                         [True, False, False, False, True, True, True, True, False, False])

    def test_search_uses_drmas_projection_and_preserves_raw_generation(self):
        agent = AgentRegistry.create(
            "SEARCH", wg_id="search-policy", tokenizer=None, processor=None,
            config=SimpleNamespace(env=SimpleNamespace(env_name="search")),
        )
        responses = [
            " \n<think>Find supporting evidence.</think>\n<search> second query </search>\n",
            "<search>first query</search>",
            "plain text without a search request",
            "<think>Find supporting evidence.</think><search>北京</search>",
            "<think>Find supporting evidence.</think><search> \n </search>",
            "<think>Find supporting evidence.</think><search>unfinished",
            "<think>Find supporting evidence.</think><search>first</search><search>second</search>",
            "<think><search>quoted query</search></think><search>actual query</search>",
            "<think><answer>premature answer</answer></think><search>query</search>",
            "<think>One</think><think>Two</think><search>query</search>",
            "<think>Inactive</think><search>inactive query</search>",
        ]
        expected = [
            "<search>second query</search>", "<search>first query</search>", "",
            "<search>北京</search>", "<search></search>", "", "<search>first</search>",
            "<search>quoted query</search>", "<search>query</search>",
            "<search>query</search>", "<search>inactive query</search>",
        ]
        size = len(responses)
        mask = np.array([True] * (size - 1) + [False])
        raw_generation = tuple(responses)
        batch = SimpleNamespace(batch={"responses": raw_generation}, non_tensor_batch={})
        with patch.object(role_modules["search"]["SEARCH"], "preprocess_batch", return_value=batch):
            with patch.object(agent, "_generate_with_llm", return_value=(batch, responses)):
                returned_batch, returned_responses = agent.call(
                    SimpleNamespace(meta_info={}), {"text": ["Question"] * size},
                    [""] * size, object(), mask, step=1,
                    candidate_answers=[""] * size,
                )
        self.assertIs(returned_batch, batch)
        self.assertIs(batch.batch["responses"], raw_generation)
        self.assertEqual(returned_responses, expected)
        self.assertEqual(batch.non_tensor_batch["is_action_valid"].tolist(),
                         [True, False, False, False, True, False, True, True, True, False, False])

    def test_orchestra_dispatches_real_agents_with_role_prompts_and_model_groups(self):
        for env_name, roles in (("math", ["SOLVE", "VERIFY", "REVISE"]),
                                ("search", ["SEARCH", "VERIFY", "ANSWER"])):
            with self.subTest(env_name=env_name):
                config = SimpleNamespace(
                    agent=SimpleNamespace(role_ids=roles, orchestra=SimpleNamespace(
                        router=SimpleNamespace(max_context_runs=1))),
                    env=SimpleNamespace(env_name=env_name, max_steps=2),
                )
                names = ["Router Agent", *roles]
                mapping = {name: f"policy:{name}" for name in names}
                groups = {group: object() for group in mapping.values()}
                with patch.dict(sys.modules, {"agent_system.agent.agents.router": router_package}):
                    orchestra = RouterOrchestra(
                        names, roles, ["checkpoint"] * 4, mapping,
                        dict.fromkeys(groups), dict.fromkeys(groups), config,
                    )
                self.assertEqual(list(orchestra.agents), names)
                instruction = f"<think>Perform the assigned task.</think><route>{roles[0]}</route>"
                first_output = "<search>first query</search>" if env_name == "search" else r"first output: full derivation, \boxed{1}"
                first_response = "<think>Search-only reasoning.</think>" + first_output if env_name == "search" else first_output
                final_action = "<answer>final output</answer>" if env_name == "search" else r"final output: corrected derivation, \boxed{2}"
                final_response = "<think>Answer-only reasoning.</think>" + final_action if env_name == "search" else final_action
                review = "review <verify>reject</verify>" if env_name == "math" else "review"
                final_review = "final review <verify>approve</verify>" if env_name == "math" else "final review"
                scripts = {
                    "Router Agent": deque([
                        [instruction, "<route>VERIFY</route>"],
                        [f"<route>{roles[2]}</route>", "<route>FINISH</route>"],
                        ["<route>VERIFY</route>", f"<route>{roles[0]}</route>"],
                        ["<route>FINISH</route>", f"<route>{roles[0]}</route>"],
                    ]),
                    roles[0]: deque([[first_response, "ignored"]]),
                    "VERIFY": deque([["ignored", review], [final_review, "ignored"]]),
                    roles[2]: deque([[final_response, "ignored"]]),
                }
                prompts = []

                def preprocess(**kwargs):
                    prompts.append(kwargs["obs"]["text"])
                    return SimpleNamespace(batch={}, non_tensor_batch={})

                def generate(name, batch, group, mask, meta_info):
                    self.assertIs(group, groups[mapping[name]])
                    responses = scripts[name].popleft()
                    batch.batch["responses"] = tuple(responses)
                    batch.non_tensor_batch["wg_id"] = np.full(2, mapping[name], dtype=object)
                    batch.non_tensor_batch["agent_active_mask"] = mask.copy()
                    return batch, responses

                with ExitStack() as stack:
                    stack.enter_context(patch.object(router_module, "preprocess_batch", side_effect=preprocess))
                    for agent_module in role_modules[env_name].values():
                        stack.enter_context(patch.object(agent_module, "preprocess_batch", side_effect=preprocess))
                    for name, agent in orchestra.agents.items():
                        stack.enter_context(patch.object(agent, "_generate_with_llm", side_effect=partial(generate, name)))
                    inputs = (SimpleNamespace(meta_info={}), {"text": ["A task", "B task"]}, groups, np.ones(2, dtype=bool), 1)
                    actions, buffer = orchestra.run(*inputs)
                    self.assertEqual(actions, [first_output, ""])
                    self.assertEqual(buffer[1]["batch"].batch["responses"], (first_response, "ignored"))
                    self.assertIn(first_output, "".join(orchestra.team_context[0]))
                    self.assertNotIn("Search-only reasoning.", "".join(orchestra.team_context[0]))
                    self.assertIn("review", "".join(orchestra.team_context[1]))
                    self.assertIn(f"You are {roles[0]}.", prompts[1][0])
                    self.assertIn(instruction, prompts[1][0])
                    self.assertIn("You are VERIFY.", prompts[2][1])
                    self.assertEqual(buffer[1]["batch"].non_tensor_batch["agent_active_mask"].tolist(), [True, False])
                    self.assertEqual(buffer[2]["batch"].non_tensor_batch["is_action_valid"].tolist(), [False, True])
                    self.assertTrue(all(set(entry["batch"].non_tensor_batch) == set(buffer[0]["batch"].non_tensor_batch)
                                        for entry in buffer))
                    other_action, other_reason = "", "finish"
                    if env_name == "search":
                        self.assertEqual(orchestra.return_reasons.tolist(), ["search", ""])
                        actions, buffer = orchestra.run(*inputs)
                        self.assertEqual(actions, [first_output, ""])
                        self.assertEqual(buffer[0]["batch"].non_tensor_batch["agent_active_mask"].tolist(), [False, True])
                        self.assertEqual(orchestra.return_reasons.tolist(), ["search", "finish"])
                        self.assertEqual(orchestra.run(*inputs)[1], [])
                        next_obs = {"text": ["A task with retrieved evidence", "B done"]}
                        orchestra.initialize_context(next_obs)
                        inputs = (inputs[0], next_obs, groups, np.array([True, False]), 2)
                        scripts["Router Agent"] = deque([
                            ["<route>ANSWER</route>", "<route>FINISH</route>"],
                            ["<route>VERIFY</route>", "<route>SEARCH</route>"],
                            ["<route>FINISH</route>", "<route>SEARCH</route>"],
                        ])
                        other_action, other_reason = "", ""
                    prompt_start = len(prompts)
                    actions, answer_buffer = orchestra.run(*inputs)
                    self.assertEqual(actions, [final_action, other_action])
                    self.assertEqual(orchestra.candidate_answers, [final_action, ""])
                    self.assertEqual(answer_buffer[-1]["batch"].batch["responses"], (final_response, "ignored"))
                    self.assertIn(final_action, "".join(orchestra.team_context[0]))
                    self.assertNotIn("Answer-only reasoning.", "".join(orchestra.team_context[0]))
                    self.assertIn("retrieved evidence" if env_name == "search" else first_output, prompts[prompt_start][0])
                    self.assertIn(first_output, prompts[prompt_start][0])
                    self.assertIn(instruction, prompts[prompt_start][0])
                    self.assertIn(f"You are {roles[2]}.", prompts[prompt_start + 1][0])
                    self.assertIn(first_output, prompts[prompt_start + 1][0])
                    # The review evicts the answer's original run. The next
                    # Router must still see it through the candidate section.
                    actions, _ = orchestra.run(*inputs)
                    self.assertEqual(actions, [final_action, other_action])
                    self.assertNotIn("final output", "".join(orchestra.team_context[0]))
                    self.assertIn("final review", "".join(orchestra.team_context[0]))
                    prompt_start = len(prompts)
                    actions, _ = orchestra.run(*inputs)
                    self.assertEqual(actions, [final_action, other_action])
                    self.assertIn("# Current Candidate Answer", prompts[prompt_start][0])
                    self.assertIn(final_action, prompts[prompt_start][0])
                    self.assertNotIn("Answer-only reasoning.", prompts[prompt_start][0])
                    self.assertNotIn("final output", prompts[prompt_start][1])
                    self.assertEqual(orchestra.return_reasons.tolist(), ["finish", other_reason])
                    self.assertTrue(all(not script for script in scripts.values()))
                    self.assertEqual(orchestra.run(*inputs)[1], [])
                    orchestra.reset()
                    self.assertIsNone(orchestra.team_context)
                    self.assertIsNone(orchestra.candidate_answers)


if __name__ == "__main__":
    unittest.main()
