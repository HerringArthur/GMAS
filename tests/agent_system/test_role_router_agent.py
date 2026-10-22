"""Test role agents and routing with mocked preprocessing and model generation."""

import importlib.util
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
    }
    with patch.dict(sys.modules, stubs):
        registry = load("agent_system.agent.registry", "agent_system/agent/registry.py")
        load("agent_system.agent.agents.base", "agent_system/agent/agents/base.py")
        router = load("agent_system.agent.agents.router.router_agent", "agent_system/agent/agents/router/router_agent.py")
        subagent = load("agent_system.agent.agents.router.subagent", "agent_system/agent/agents/router/subagent.py")
        package = load("agent_system.agent.agents.router", "agent_system/agent/agents/router/__init__.py")
        load("agent_system.agent.orchestra.base", "agent_system/agent/orchestra/base.py")
        orchestra = load("role_agents_orchestra_under_test", "agent_system/agent/orchestra/router/router_orchestra.py")
    return router, subagent, package, registry.AgentRegistry, orchestra.RouterMultiAgentOrchestra


router_module, subagent_module, router_package, AgentRegistry, RouterOrchestra = load_agents()


class RoleRouterAgentTests(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            agent=SimpleNamespace(role_ids=["SOLVE", "VERIFY", "REVISE"]),
            env=SimpleNamespace(env_name="math"),
        )
        self.router = AgentRegistry.create(
            "Router Agent", wg_id="router-policy", tokenizer=None, processor=None, config=self.config,
        )

    def call_router(self, responses, active=None, env_obs=None, team_context=None):
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
                    step=2,
                )
        generate.assert_called_once_with(batch, group, mask, gen_batch.meta_info)
        self.assertIs(preprocess.call_args.kwargs["agent_active_mask"], mask)
        self.assertIs(result, batch)
        self.assertIs(returned_responses, responses)
        self.assertIs(result.batch["responses"], raw_response_tokens)
        self.assertIs(result.non_tensor_batch["agent_active_mask"], mask)
        self.assertEqual(result.non_tensor_batch["wg_id"].tolist(), ["router-policy"] * size)
        self.assertEqual(result.non_tensor_batch["env_step"].tolist(), [2] * size)
        return result, preprocess.call_args.kwargs["obs"]

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


class RoleSubAgentTests(unittest.TestCase):
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
                    prompt = agent.build_prompt(observation, context, step=2)
                    self.assertIsInstance(agent, subagent_module.SubAgent)
                    self.assertEqual(agent.name, role)
                    self.assertEqual(agent.wg_id, f"policy:{role}")
                    self.assertIn(f"You are {role}.", prompt["text"][0])
                    self.assertIn(expected_instruction, prompt["text"][0])
                    self.assertIn(context[0], prompt["text"][0])
                    self.assertIn("Original task {x}", prompt["text"][0])
                    self.assertIn("Step 2", prompt["text"][0])
                    if env_name == "math" and role in ("SOLVE", "REVISE"):
                        self.assertIn(r"\boxed{...}", prompt["text"][0])
                    self.assertEqual(observation["text"], ["Original task {x}"])
                    self.assertEqual(prompt["image"], observation["image"])

    def test_call_preserves_responses_and_excludes_empty_and_inactive_rows(self):
        agent = AgentRegistry.create(
            "VERIFY", wg_id="verify-policy", tokenizer=None, processor=None,
            config=SimpleNamespace(env=SimpleNamespace(env_name="math")),
        )
        responses = ["The second step has an error.", " \n ", "inactive report", ""]
        mask = np.array([True, True, False, True])
        raw_tokens = object()
        batch = SimpleNamespace(
            batch={"responses": raw_tokens},
            non_tensor_batch={"wg_id": np.full(4, "verify-policy", dtype=object), "agent_active_mask": mask},
        )
        gen_batch = SimpleNamespace(meta_info={})
        group = object()
        with patch.object(subagent_module, "preprocess_batch", return_value=batch) as preprocess:
            with patch.object(agent, "_generate_with_llm", return_value=(batch, responses)) as generate:
                returned_batch, returned_responses = agent.call(
                    gen_batch, {"text": ["Problem"] * 4}, ["shared work"] * 4, group, mask, step=1,
                )
        generate.assert_called_once_with(batch, group, mask, gen_batch.meta_info)
        self.assertIs(preprocess.call_args.kwargs["agent_active_mask"], mask)
        self.assertIs(returned_batch, batch)
        self.assertIs(returned_responses, responses)
        self.assertIs(batch.batch["responses"], raw_tokens)
        self.assertEqual(batch.non_tensor_batch["is_action_valid"].tolist(), [True, False, False, False])
        self.assertEqual(batch.non_tensor_batch["selected_role"].tolist(), [""] * 4)
        self.assertEqual(batch.non_tensor_batch["env_step"].tolist(), [1] * 4)

    def test_orchestra_dispatches_real_agents_with_role_prompts_and_model_groups(self):
        for env_name, roles in (("math", ["SOLVE", "VERIFY", "REVISE"]),
                                ("search", ["SEARCH", "VERIFY", "ANSWER"])):
            with self.subTest(env_name=env_name):
                config = SimpleNamespace(
                    agent=SimpleNamespace(role_ids=roles), env=SimpleNamespace(env_name=env_name),
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
                scripts = {
                    "Router Agent": deque([
                        [instruction, "<route>VERIFY</route>"],
                        [f"<route>{roles[2]}</route>", "<route>FINISH</route>"],
                        ["<route>FINISH</route>", f"<route>{roles[0]}</route>"],
                    ]),
                    roles[0]: deque([["first output", "ignored"]]),
                    "VERIFY": deque([["ignored", "review"]]),
                    roles[2]: deque([["final output", "ignored"]]),
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
                    stack.enter_context(patch.object(subagent_module, "preprocess_batch", side_effect=preprocess))
                    for name, agent in orchestra.agents.items():
                        stack.enter_context(patch.object(agent, "_generate_with_llm", side_effect=partial(generate, name)))
                    inputs = (SimpleNamespace(meta_info={}), {"text": ["A task", "B task"]}, groups, np.ones(2, dtype=bool), 1)
                    actions, buffer = orchestra.run(*inputs)
                    self.assertEqual(actions, ["first output", "review"])
                    self.assertIn(f"You are {roles[0]}.", prompts[1][0])
                    self.assertIn(instruction, prompts[1][0])
                    self.assertIn("You are VERIFY.", prompts[2][1])
                    self.assertEqual(buffer[1]["batch"].non_tensor_batch["agent_active_mask"].tolist(), [True, False])
                    self.assertEqual(buffer[2]["batch"].non_tensor_batch["is_action_valid"].tolist(), [False, True])
                    self.assertTrue(all(set(entry["batch"].non_tensor_batch) == set(buffer[0]["batch"].non_tensor_batch)
                                        for entry in buffer))
                    actions, _ = orchestra.run(*inputs)
                    self.assertEqual(actions, ["final output", "review"])
                    self.assertIn("first output", prompts[3][0])
                    self.assertIn(f"You are {roles[2]}.", prompts[4][0])
                    actions, _ = orchestra.run(*inputs)
                    self.assertEqual(actions, ["final output", "review"])
                    self.assertEqual(orchestra.return_reasons.tolist(), ["finish", "finish"])
                    self.assertTrue(all(not script for script in scripts.values()))
                    self.assertEqual(orchestra.run(*inputs)[1], [])
                    orchestra.reset()
                    self.assertIsNone(orchestra.team_context)


if __name__ == "__main__":
    unittest.main()
