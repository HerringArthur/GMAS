"""CPU-only recorder tests; runnable without importing the GPU training stack."""

import csv
import importlib.util
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).resolve().parents[3] / "verl/trainer/ppo/benchmark.py"
SPEC = importlib.util.spec_from_file_location("gmas_benchmark", MODULE_PATH)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class Array:
    """Only the tensor read operations used by the recorder."""

    def __init__(self, data):
        self.data = data

    @property
    def shape(self):
        return len(self.data), len(self.data[0])

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return Array([row[key[1]] for row in self.data[key[0]]])
        return Array(self.data[key])

    def tolist(self):
        return self.data

    def sum(self):
        return SimpleNamespace(item=lambda: sum(map(sum, self.data)))


class Batch:
    def __init__(self, training=False):
        # Deliberately interleave trajectories. A sample must recover all turns
        # in their original order, not just one Router/Sub Agent row.
        self.non_tensor_batch = {
            "traj_uid": ["a", "b", "a"], "uid": ["q1", "q2", "q1"],
            "agent_id": ["Router", "Router", "Worker"],
            "wg_id": ["router", "router", "worker"],
            "env_step": [1, 1, 2], "episode_rewards": [1.0, 0.0, 1.0],
            "agent_path": [("Worker",), (), ("Worker",)],
            "return_reason": ["search", "finish", "finish"],
            "action_name": ["call_agent", "finish", "search"],
            "action_args": [{"role": "SEARCH", "task": "查证"}, {}, {"query": "newborn"}],
            "environment_action": ["", "<answer>B</answer>", "<search>newborn</search>"],
            "execution_status": ["dispatched", "submitted", "success"],
            "candidate_review": [None, None, {"candidate_version": 1, "env_step": 1,
                                               "assessment": {"sufficient": False, "feedback": "Need evidence"}}],
        }
        self.batch = {
            "prompts": Array([[0, 11, 12], [0, 21, 22], [31, 32, 33]]),
            "responses": Array([[13, 14, 0], [23, 0, 0], [34, 0, 0]]),
            "attention_mask": Array([[0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0]]),
        }
        if training:
            self.non_tensor_batch["wg_id"] += ["router", "worker"]

    def __len__(self):
        return len(self.non_tensor_batch["wg_id"])


class Tokenizer:
    def __init__(self, prefix):
        self.prefix = prefix

    def decode(self, tokens, skip_special_tokens):
        assert not skip_special_tokens
        return self.prefix + ":" + ",".join(map(str, tokens))


class BenchmarkTests(unittest.TestCase):
    def config(self, folder, samples=3):
        return {"trainer": {"resume_mode": "disable", "benchmark": {
            "output_dir": folder, "samples_per_step": samples, "warmup_steps": 1, "seed": 42,
        }}}

    def test_whole_trajectories_masks_tokenizers_and_rng(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = benchmark.TrainingBenchmark(self.config(folder), 3)
            rng_before = random.getstate()
            counts = recorder.record_rollouts(
                Batch(), {"router": Tokenizer("R"), "worker": Tokenizer("W")}, 1, 2
            )
            self.assertEqual(random.getstate(), rng_before)
            self.assertEqual(counts["benchmark/response_tokens"], 4)
            self.assertEqual(counts["benchmark/trajectories"], 2)
            records = [json.loads(line) for line in (Path(folder) / "rollout_samples/step_000001.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 2)  # requested three, only two available
            sample = next(record for record in records if record["traj_uid"] == "a")
            self.assertEqual([turn["agent_id"] for turn in sample["turns"]], ["Router", "Worker"])
            self.assertEqual(sample["turns"][0]["input"], "R:11,12")
            self.assertEqual(sample["turns"][0]["output"], "R:13,14")
            self.assertEqual(sample["turns"][1]["output"], "W:34")
            self.assertEqual(sample["turns"][0]["action_args"], {"role": "SEARCH", "task": "查证"})
            self.assertEqual(sample["turns"][1]["environment_action"], "<search>newborn</search>")
            self.assertFalse(sample["turns"][1]["candidate_review"]["assessment"]["sufficient"])
            self.assertEqual(sample["episode_rewards"], 1.0)
            self.assertIn("[Input]", (Path(folder) / "rollout_samples/step_000001.txt").read_text())

    def test_warmup_exclusion_weighted_throughput_and_padding_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = benchmark.TrainingBenchmark(self.config(folder, samples=0), 3)
            for step, seconds in enumerate((100.0, 10.0, 30.0), 1):
                metrics = recorder.record_rollouts(Batch(), {}, step, 512)
                metrics.update({"timing_s/step": seconds, "timing_s/gen": seconds / 2,
                                "timing_s/dump_rollout_samples": 2.0, "loss": float("nan")})
                recorder.record_step(metrics, step, Batch(training=True))
                summary = json.loads((Path(folder) / "summary.json").read_text())
                if step == 1:
                    self.assertIsNone(summary["step_s"])
                    self.assertFalse(summary["complete"])
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["measured_steps"], 2)
            self.assertEqual(summary["step_s"], {"mean": 20.0, "median": 20.0, "min": 10.0, "max": 30.0})
            self.assertEqual(summary["step_excluding_samples_s"]["mean"], 18.0)
            self.assertAlmostEqual(summary["questions_per_s"], 1024 / 40)
            self.assertAlmostEqual(summary["generated_response_tokens_per_gen_s"], 8 / 20)
            with (Path(folder) / "timing.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["copied_rows"], "2")
            self.assertEqual(rows[0]["is_warmup"], "True")
            self.assertEqual(rows[1]["is_warmup"], "False")
            saved = json.loads((Path(folder) / "metrics.jsonl").read_text().splitlines()[0])
            self.assertIsNone(saved["metrics"]["loss"])
            self.assertEqual(saved["metrics"]["benchmark/copied_rows/worker"], 1)
            self.assertFalse((Path(folder) / "rollout_samples").exists())
            with self.assertRaises(FileExistsError):
                benchmark.TrainingBenchmark(self.config(folder), 3)

    def test_retained_invalid_rollout_rows_are_not_counted_as_padding(self):
        for valid_flags in ([True, False, True], [False, False, False]):
            for padded in (False, True):
                with self.subTest(valid_flags=valid_flags, padded=padded), tempfile.TemporaryDirectory() as folder:
                    recorder = benchmark.TrainingBenchmark(self.config(folder, samples=0), 2)
                    rollout = Batch()
                    rollout.non_tensor_batch["is_action_valid"] = valid_flags
                    metrics = recorder.record_rollouts(rollout, {}, 1, 2)
                    metrics.update({
                        "timing_s/step": 10.0,
                        "timing_s/dump_rollout_samples": 0.0,
                    })
                    recorder.record_step(metrics, 1, Batch(training=padded))
                    self.assertEqual(metrics["benchmark/valid_rollout_rows"], sum(valid_flags))
                    self.assertEqual(metrics["benchmark/rollout_rows"], 3)
                    self.assertEqual(metrics["benchmark/training_rows"], 5 if padded else 3)
                    self.assertEqual(metrics["benchmark/copied_rows"], 2 if padded else 0)
                    self.assertEqual(metrics["benchmark/copied_rows/router"], 1 if padded else 0)
                    self.assertEqual(metrics["benchmark/copied_rows/worker"], 1 if padded else 0)

    def test_invalid_warmup_and_resume_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                benchmark.TrainingBenchmark(self.config(folder), 1)
            config = self.config(folder)
            config["trainer"]["resume_mode"] = "auto"
            with self.assertRaises(ValueError):
                benchmark.TrainingBenchmark(config, 3)


if __name__ == "__main__":
    unittest.main()
