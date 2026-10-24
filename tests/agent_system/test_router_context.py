"""Context regressions for the current role-based Router and preprocessing.

The obsolete SubAgent/handoff/parser tests are superseded by test_role_router_*.
These tests retain candidate/history isolation and token-boundary coverage.
The deterministic byte tokenizer tests plumbing, not Qwen token counts or a
protected-context budget (which the current Router does not implement).
"""

import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from omegaconf import OmegaConf
import torch

import test_role_router_agent as agent_fixture
import test_role_router_orchestra as orchestra_fixture


ROOT = Path(__file__).resolve().parents[2]


class ByteTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))

    def apply_chat_template(self, messages, **kwargs):
        return "".join(f'<{m["role"]}>{m["content"]}</{m["role"]}>' for m in messages) + "<assistant>"

    def __call__(self, prompt, **kwargs):
        ids = torch.tensor([self.encode(prompt)])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def load_preprocess():
    namespace = {"torch": torch, "np": np, "F": torch.nn.functional}
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for path, names in (
        ("verl/utils/torch_functional.py", {"pad_sequence_to_length", "postprocess_data", "tokenize_and_postprocess_data"}),
        ("verl/utils/model.py", {"compute_position_id_with_mask"}),
        ("agent_system/multi_turn_rollout/utils.py", {"preprocess_fn"}),
    ):
        nodes.extend(n for n in ast.parse((ROOT / path).read_text()).body if getattr(n, "name", "") in names)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "<production-preprocess>", "exec"), namespace)
    namespace["verl_F"] = SimpleNamespace(tokenize_and_postprocess_data=namespace["tokenize_and_postprocess_data"])
    return namespace["preprocess_fn"]


preprocess = load_preprocess()


class RouterContextTests(unittest.TestCase):
    make_orchestra = orchestra_fixture.RoleRouterTests.make_orchestra
    route = orchestra_fixture.RoleRouterTests.route
    respond = orchestra_fixture.RoleRouterTests.respond
    run_round = orchestra_fixture.RoleRouterTests.run_round

    def test_candidate_and_training_snapshot_survive_invalid_answer_and_history_eviction(self):
        orchestra = self.make_orchestra(roles=("SEARCH", "VERIFY", "ANSWER"), env_name="search", context_runs=1)
        candidate = "<answer>Alpha</answer>"
        self.route(["ANSWER"])
        self.respond("ANSWER", [candidate])
        _, first_buffer = self.run_round()
        first_call = deepcopy(orchestra.agents["ANSWER"].calls[-1])

        self.route(["ANSWER"])
        self.respond("ANSWER", ["INVALID_CANDIDATE"], valid=[False])
        _, invalid_buffer = self.run_round()
        self.assertFalse(invalid_buffer[1]["batch"].non_tensor_batch["is_action_valid"][0])
        self.assertEqual(invalid_buffer[1]["batch"].batch["responses"], ("INVALID_CANDIDATE",))

        self.route(["VERIFY"])
        self.respond("VERIFY", ["Source supports Alpha."])
        self.run_round()
        self.assertNotIn(candidate, "".join(orchestra.team_context[0]))
        self.assertNotIn("INVALID_CANDIDATE", "".join(orchestra.team_context[0]))
        self.assertEqual(orchestra.agents["VERIFY"].calls[-1]["candidate_answers"], [candidate])
        self.assertEqual(first_call["candidate_answers"], [""])
        self.assertEqual(first_buffer[1]["batch"].batch["responses"], (candidate,))

        self.route(["FINISH"])
        actions, _ = self.run_round()
        self.assertEqual(actions, [candidate])
        orchestra.reset()
        self.assertIsNone(orchestra.team_context)
        self.assertIsNone(orchestra.candidate_answers)

    def test_prompt_assembly_keeps_batched_question_candidate_and_history_isolated(self):
        cfg = SimpleNamespace(agent=SimpleNamespace(role_ids=["SEARCH", "VERIFY", "ANSWER"]),
                              env=SimpleNamespace(env_name="search", max_steps=4))
        router = agent_fixture.router_module.RouterAgent("router", None, None, cfg)
        obs = {"text": ["QUESTION_A {literal}", "QUESTION_B"]}
        original = deepcopy(obs)
        prompts = router.build_prompt(obs, ["EVIDENCE_A", "EVIDENCE_B"], 2, ["CANDIDATE_A", "CANDIDATE_B"])
        self.assertEqual(obs, original)
        for index, label in enumerate(("A", "B")):
            other = "B" if label == "A" else "A"
            for field in ("QUESTION", "EVIDENCE", "CANDIDATE"):
                self.assertIn(f"{field}_{label}", prompts["text"][index])
                self.assertNotIn(f"{field}_{other}", prompts["text"][index])
        self.assertIn("{literal}", prompts["text"][0])


class RouterTokenBoundaryTests(unittest.TestCase):
    def process(self, full_length, truncation="left"):
        tokenizer = ByteTokenizer()
        system = "System instruction"
        overhead = len(tokenizer.encode(tokenizer.apply_chat_template([
            {"role": "system", "content": system}, {"role": "user", "content": ""},
        ])))
        body = "Q" + "x" * (full_length - overhead - 2) + "Z"
        cfg = OmegaConf.create({"data": {"max_prompt_length": 4096, "truncation": truncation, "return_raw_chat": True}})
        source = SimpleNamespace(non_tensor_batch={"raw_prompt": [[{"role": "system", "content": system}]],
                                                   "data_source": ["test"]})
        result = preprocess(0, source, {"text": [body]}, cfg, tokenizer)
        complete = tokenizer.encode(tokenizer.apply_chat_template(result["raw_prompt"]))
        self.assertEqual(len(complete), full_length)
        return result, complete

    def test_chat_template_and_padding_at_4096_boundary(self):
        for length in (4095, 4096, 4097, 6000):
            with self.subTest(length=length):
                row, complete = self.process(length)
                self.assertEqual(row["input_ids"].shape[0], 4096)
                self.assertEqual(row["attention_mask"].sum().item(), min(length, 4096))
                self.assertEqual(row["raw_prompt_ids"], complete[-4096:])
                self.assertEqual(row["input_ids"][row["attention_mask"].bool()].tolist(), row["raw_prompt_ids"])

    def test_supported_truncation_modes_keep_tensor_and_engine_inputs_identical(self):
        for truncation in ("left", "right", "middle"):
            with self.subTest(truncation=truncation):
                row, complete = self.process(5000, truncation)
                expected = {"left": complete[-4096:], "right": complete[:4096],
                            "middle": complete[:2048] + complete[-2048:]}[truncation]
                self.assertEqual(row["raw_prompt_ids"], expected)
                self.assertEqual(row["input_ids"].tolist(), expected)

    def test_error_mode_rejects_overflow_after_chat_template(self):
        self.process(4096, "error")
        with self.assertRaisesRegex(NotImplementedError, "larger than"):
            self.process(4097, "error")


if __name__ == "__main__":
    unittest.main()
