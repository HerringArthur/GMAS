"""Check credit statistics and the rollout-to-training batch contract on CPU.

Production function bodies run unchanged; NumPy substitutes only for tensor
allocation and model/distributed boundaries, as in the adjacent trainer tests.
"""

import ast
from collections import defaultdict
from itertools import count
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from tests.trainer.ppo import test_valid_action_training as training_support


ROOT = Path(__file__).resolve().parents[3]


def load_credit():
    names = {
        "to_hashable", "_normalize_group_scalar", "_normalize_prefix_key",
        "build_router_subagent_group_index", "attach_training_rewards_to_group_index",
        "compute_subagent_credit", "compute_router_credit", "compute_router_subagent_credit",
        "normalize_router_subagent_credit", "build_token_level_credit_advantage",
        "build_router_subagent_group_index_from_batch",
    }
    tree = ast.parse((ROOT / "verl/trainer/ppo/core_algos.py").read_text())
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes.extend(node for node in tree.body if getattr(node, "name", None) in names)
    torch = SimpleNamespace(zeros_like=np.zeros_like, float32=np.float32)
    namespace = {"np": np, "math": math, "defaultdict": defaultdict, "torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 "<production-credit>", "exec"), namespace)
    core = SimpleNamespace(**{name: namespace[name] for name in names})

    tree = ast.parse((ROOT / "verl/trainer/ppo/ray_trainer.py").read_text())
    compute = next(node for node in tree.body if getattr(node, "name", None) == "compute_advantage")
    nodes = [nodes[0], compute]
    namespace.update(core_algos=core, AdvantageEstimator=SimpleNamespace(GAE="gae", GRPO="grpo"))
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 "<production-advantage>", "exec"), namespace)
    return core, namespace["compute_advantage"]


core, compute_advantage = load_credit()
ROW_IDS = count()


def row(role, trajectory, reward, *, uid="Q1", prefix=(), selected="", step=1, valid=True):
    return dict(agent_id=role, uid=uid, traj_uid=trajectory, episode_rewards=reward,
                parent_prefix_key=prefix, selected_role=selected, env_step=step, is_action_valid=valid,
                rollout_row_id=next(ROW_IDS))


def metadata(rows):
    return {key: np.fromiter((item[key] for item in rows), dtype=object, count=len(rows))
            for key in rows[0]}


def statistics(rows, normalize=False, rewards=None):
    if rewards is None:
        rewards = [item["episode_rewards"] for item in rows]
    batch = SimpleNamespace(
        non_tensor_batch=metadata(rows),
        batch={"token_level_scores": training_support.tensor([[0.0, reward] for reward in rewards])},
    )
    groups = core.build_router_subagent_group_index_from_batch(batch)
    credits = core.compute_router_subagent_credit(groups)
    normalized = core.normalize_router_subagent_credit(groups, credits, norm_adv_by_std_in_grpo=normalize)
    return groups, credits, normalized


class RoleCreditTests(unittest.TestCase):
    def test_role_statistics_weight_each_real_call_equally_and_ignore_padding(self):
        rows = [
            row("SOLVE", "T1", 1),
            row("SOLVE", "T1", 1, prefix=("SOLVE", "VERIFY"), step=2, valid=False),
            row("SOLVE", "T2", 0),
            row("VERIFY", "T1", 1, prefix=("SOLVE",)),
            row("SOLVE", "T3", 1, uid="Q2"),
            row("SOLVE", "T4", 1, uid="Q2"),
        ]
        groups, credits, normalized = statistics(
            rows + [rows[0], rows[1]], normalize=True, rewards=[1.0, 0.9, 0.0, 1.0, 1.0, 1.0, 1.0, 0.9],
        )
        key = ("Q1", "SOLVE")
        self.assertEqual(groups["subagent_groups"][key], {
            ("T1", rows[0]["rollout_row_id"]): [0, 6],
            ("T1", rows[1]["rollout_row_id"]): [1, 7],
            ("T2", rows[2]["rollout_row_id"]): [2],
        })
        rewards = np.array([1.0, 0.9, 0.0])
        expected = rewards - rewards.mean()
        self.assertAlmostEqual(credits["subagent_group_baselines"][key], 1.9 / 3)
        self.assertAlmostEqual(normalized["subagent_group_stds"][key], rewards.std(ddof=0))
        np.testing.assert_allclose([credits["subagent_row_credits"][i] for i in range(3)], expected)
        np.testing.assert_allclose([normalized["subagent_row_advantages"][i] for i in range(3)],
                                   expected / (rewards.std(ddof=0) + 1e-6))
        self.assertGreater(normalized["subagent_row_advantages"][0], normalized["subagent_row_advantages"][1])
        self.assertGreater(normalized["subagent_row_advantages"][1], 0)
        for original, copy in [(0, 6), (1, 7)]:
            self.assertEqual(normalized["subagent_row_advantages"][original],
                             normalized["subagent_row_advantages"][copy])
        for i in [3, 4, 5]:
            self.assertEqual(normalized["subagent_row_advantages"][i], 0)

    def test_distinct_calls_in_one_trajectory_can_have_different_advantages(self):
        rows = [row("SEARCH", "T1", 1), row("SEARCH", "T1", 1, valid=False)]
        _, credits, normalized = statistics(rows, normalize=True, rewards=[1.0, 0.9])
        self.assertAlmostEqual(credits["subagent_group_baselines"][("Q1", "SEARCH")], 0.95)
        np.testing.assert_allclose(list(normalized["subagent_row_advantages"].values()),
                                   np.array([0.05, -0.05]) / (0.05 + 1e-6))

    def test_all_invalid_role_calls_still_participate(self):
        rows = [row("SEARCH", "T1", 1, valid=False), row("SEARCH", "T2", 0, valid=False)]
        _, credits, normalized = statistics(rows, normalize=True, rewards=[0.9, -0.1])
        self.assertAlmostEqual(credits["subagent_group_baselines"][("Q1", "SEARCH")], 0.4)
        np.testing.assert_allclose(list(normalized["subagent_row_advantages"].values()),
                                   np.array([0.5, -0.5]) / (0.5 + 1e-6))

    def test_router_credit_uses_action_means_and_ignores_padding_copies(self):
        rows = [
            row("Router Agent", "T1", 1, prefix=("SOLVE",), selected="VERIFY"),
            row("Router Agent", "T2", 0, prefix=("SOLVE",), selected="VERIFY"),
            row("Router Agent", "T3", 0, prefix=("SOLVE",), selected="REVISE"),
            row("Router Agent", "T4", 0, prefix=("SOLVE",), selected="REVISE"),
        ]
        groups, credits, normalized = statistics(rows + [rows[0], rows[0]], normalize=True)
        key = ("Q1", ("SOLVE",))
        self.assertEqual(len(groups["router_groups"][key]), 4)
        self.assertEqual(credits["router_group_baselines"][key], 0.25)
        self.assertEqual(credits["router_action_statistics"][key]["VERIFY"],
                         {"count": 2, "probability": 0.5, "mean_reward": 0.5})
        self.assertEqual(credits["router_row_credits"], {0: 0.25, 1: 0.25, 2: -0.25, 3: -0.25, 4: 0.25, 5: 0.25})
        self.assertEqual(normalized["router_group_stds"][key], 0.25)
        self.assertAlmostEqual(normalized["router_row_advantages"][1], 0.25 / (0.25 + 1e-6))

    def test_router_groups_use_task_and_prefix_but_not_environment_step(self):
        rows = [
            row("Router Agent", "T1", 1, prefix=("SEARCH",), selected="ANSWER", step=2),
            row("Router Agent", "T2", 0, prefix=["SEARCH"], selected="VERIFY", step=3),
            row("Router Agent", "T1", 1, prefix=("SEARCH", "ANSWER"), selected="FINISH", step=2),
            row("Router Agent", "T3", 0, uid="Q2", prefix=("SEARCH",), selected="ANSWER", step=2),
        ]
        groups, credits, _ = statistics(rows)
        self.assertEqual(len(groups["router_groups"]), 3)
        self.assertEqual(credits["router_row_credits"], {0: 0.5, 1: -0.5, 2: 0.0, 3: 0.0})

    def test_finish_and_real_cross_step_retry_are_distinct_from_padding(self):
        rows = [
            row("Router Agent", "T1", 1, selected="FINISH"),
            row("Router Agent", "T1", 1, selected="SEARCH", step=2),
            row("Router Agent", "T2", 0, selected="FINISH"),
            row("Router Agent", "invalid", 0, selected="", valid=False),
        ]
        groups, credits, _ = statistics(rows + [rows[0]])
        self.assertEqual(len(groups["router_groups"][("Q1", ())]), 4)
        self.assertEqual(groups["invalid_router_rows"], [3])
        self.assertEqual(credits["router_row_credits"][3], -0.5)
        stats = credits["router_action_statistics"][("Q1", ())]
        self.assertEqual(stats["FINISH"]["count"], 2)
        self.assertEqual(stats["__INVALID__"]["count"], 1)
        self.assertEqual(credits["router_row_credits"][1], 0.5)
        self.assertEqual(credits["router_row_credits"][0], 0.0)
        self.assertEqual(credits["router_row_credits"][0], credits["router_row_credits"][4])

    def test_invalid_retries_and_valid_route_in_same_step_are_distinct_calls(self):
        rows = [
            row("Router Agent", "T1", 1, valid=False),
            row("Router Agent", "T1", 1, selected="SEARCH", valid=False),
            row("Router Agent", "T1", 1, selected="SEARCH"),
            row("Router Agent", "T2", 0, selected="ANSWER"),
        ]
        groups, credits, _ = statistics(rows + [rows[0], rows[2]], rewards=[0.9, 0.9, 1.0, 0.0, 0.9, 1.0])
        key = ("Q1", ())
        self.assertEqual(len(groups["router_groups"][key]), 4)
        stats = credits["router_action_statistics"][key]
        self.assertEqual(stats["__INVALID__"]["count"], 2)
        self.assertAlmostEqual(stats["__INVALID__"]["mean_reward"], 0.9)
        self.assertEqual(stats["SEARCH"]["count"], 1)
        self.assertAlmostEqual(credits["router_group_baselines"][key], 0.7)
        np.testing.assert_allclose([credits["router_row_credits"][i] for i in range(6)],
                                   [0.2, 0.2, 0.3, -0.7, 0.2, 0.3])
        self.assertEqual(len(credits["router_row_credits"]), 6)
        self.assertEqual(credits["router_row_credits"][0], credits["router_row_credits"][4])

    def test_single_observed_router_choice_has_zero_credit(self):
        rows = [row("Router Agent", "T1", 1, selected="SOLVE"),
                row("Router Agent", "T2", 0, selected="SOLVE")]
        _, credits, normalized = statistics(rows, normalize=True)
        self.assertEqual(credits["router_row_credits"], {0: 0.0, 1: 0.0})
        self.assertEqual(normalized["router_row_advantages"], {0: 0.0, 1: 0.0})

    def test_reordering_and_duplication_do_not_change_credit_or_token_masks(self):
        rows = [row("SOLVE", "T1", 1), row("SOLVE", "T1", 1, prefix=("SOLVE", "VERIFY")),
                row("SOLVE", "T2", 0), row("Router Agent", "T1", 1, selected="FINISH"),
                row("Router Agent", "T2", 0, selected="VERIFY")]
        _, _, original = statistics(rows, normalize=True)
        order = [4, 0, 2, 3, 1, 0, 3]
        _, _, reordered = statistics([rows[i] for i in order], normalize=True)
        mask = np.array([[1, 1, 0]] * len(order))
        advantages = core.build_token_level_credit_advantage(reordered, mask)
        expected = core.build_token_level_credit_advantage(original, np.ones((len(rows), 3)))[order] * mask
        np.testing.assert_array_equal(advantages, expected)

    def test_custom_advantage_uses_actual_format_penalty_and_adds_kl_only_once(self):
        for role in ["SEARCH", "Router Agent"]:
            for normalize in [False, True]:
                with self.subTest(role=role, normalize=normalize):
                    ns = training_support.load_driver()
                    batch = training_support.FormatRewardTests().make_batch()
                    batch.non_tensor_batch.update(metadata([
                        row(role, "T1", 1, selected="SEARCH"),
                        row(role, "T1", 1, valid=False),
                    ]))
                    batch.non_tensor_batch["is_action_valid"] = np.array([True, False])
                    ns["apply_invalid_action_penalty"](batch, invalid_action_penalty_coef=0.1)
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                    compute_advantage(batch, adv_estimator="grpo", credit_mode="router_subagent",
                                      norm_adv_by_std_in_grpo=normalize)
                    credits = np.array([0.05, -0.05])
                    if normalize:
                        credits /= 0.05 + 1e-6
                    expected = credits[:, None] * np.array([[1, 1], [1, 0]])
                    np.testing.assert_allclose(batch.batch["advantages"], expected, rtol=1e-6)

                    batch.batch["old_log_probs"] = training_support.tensor([[0.1, 0.3], [0.5, 0.7]])
                    batch.batch["ref_log_prob"] = training_support.tensor(np.zeros((2, 2)))
                    controller = SimpleNamespace(value=0.2, update=lambda **kwargs: None)
                    ns["apply_kl_penalty"](batch, controller)
                    compute_advantage(batch, adv_estimator="grpo", credit_mode="router_subagent",
                                      norm_adv_by_std_in_grpo=normalize)
                    np.testing.assert_allclose(batch.batch["advantages"],
                                               expected + np.array([[-0.02, -0.06], [-0.1, 0]]), rtol=1e-6)
                    np.testing.assert_array_equal(batch.batch["returns"], batch.batch["advantages"])
                    np.testing.assert_array_equal(batch.non_tensor_batch["episode_rewards"], [1.0, 1.0])

                    batch.batch["loss_mask"] = training_support.tensor([[0, 0, 0, 1], [0, 0, 1, 0]])
                    compute_advantage(batch, adv_estimator="grpo", credit_mode="router_subagent",
                                      norm_adv_by_std_in_grpo=normalize, multi_turn=True)
                    np.testing.assert_allclose(batch.batch["advantages"],
                                               (expected + np.array([[-0.02, -0.06], [-0.1, 0]]))
                                               * np.array([[0, 1], [1, 0]]), rtol=1e-6)

    def test_collector_metadata_survives_padding_and_routes_credit_to_role_models(self):
        from tests.agent_system import test_role_router_orchestra as rollout_support

        scenario = rollout_support.RoleRouterTests()
        orchestra = scenario.make_orchestra(context_runs=1)
        scenario.route(["SOLVE", "SOLVE"])
        scenario.respond("SOLVE", ["First solution", "Other solution"])
        scenario.route(["VERIFY", "FINISH"])
        scenario.respond("VERIFY", ["Review", "ignored"])
        scenario.route(["SOLVE", "FINISH"])
        scenario.respond("SOLVE", ["Improved solution", "ignored"])
        scenario.route(["FINISH", "FINISH"])
        environment = rollout_support.ScriptedEnvironment(["Same question"] * 2, [(None, [1.0, 0.0], [True, True])])
        gen_batch = rollout_support.FakeBatch(2)
        gen_batch.batch = np.zeros((2, 1))
        collector = SimpleNamespace(config=scenario.config, multiagent_orchestra=orchestra)
        trajectories, rewards, *_ = rollout_support.collect(
            collector, gen_batch, scenario.groups, environment, effective_rollout_n=2,
        )
        rows = []
        for trajectory, reward in zip(trajectories, rewards):
            for item in trajectory:
                rows.append({**item, "episode_rewards": reward, "row_id": len(rows),
                             "rollout_row_id": len(rows)})
        self.assertEqual(rows[4]["parent_prefix_key"], ("SOLVE", "VERIFY"))
        self.assertEqual(orchestra.role_paths, [("SOLVE", "VERIFY", "SOLVE"), ("SOLVE",)])
        self.assertEqual(len(orchestra.team_context[0]), 1)
        self.assertNotIn("selected_agent", rows[0])
        self.assertNotIn("prefix_key", rows[0])
        self.assertNotIn("decision_id", rows[0])

        mask = np.array([[1, 1, 0]] * len(rows))
        batch = training_support.Batch(
            batch={
                "input_ids": mask.copy(), "response_mask": mask.copy(),
                "token_level_scores": training_support.tensor([[0.0, item["episode_rewards"], 0.0] for item in rows]),
            },
            non_tensor_batch=metadata(rows),
        )
        utils = training_support.load_driver()
        config = training_support.Config(
            actor_rollout_ref=training_support.Config(
                actor=training_support.Config(use_adaptive_ppo_mini_batch_size=False, ppo_mini_update_num=1,
                                             ppo_micro_batch_size_per_gpu=4, use_kl_loss=False),
                rollout=training_support.Config(log_prob_micro_batch_size_per_gpu=4),
            ),
            trainer=training_support.Config(n_gpus_per_node=1, nnodes=1),
            algorithm=training_support.Config(use_kl_in_reward=False),
        )
        worker_groups = list(scenario.groups)
        split = utils["split_batch_by_wg_ids"](batch, worker_groups)
        for wg_id, sub_batch in split.items():
            split[wg_id] = utils["adjust_batch"](config, sub_batch, wg_id)
            split[wg_id].reorder(np.arange(len(split[wg_id]))[::-1])
        padded = utils["combine_batches"](split)
        self.assertGreater(len(padded), len(rows))
        compute_advantage(padded, adv_estimator="grpo", credit_mode="router_subagent", norm_adv_by_std_in_grpo=False)
        expected = np.array([0, 1 / 3, 0.5, 0, 0, 1 / 3, 0, 0, -2 / 3, -0.5])
        np.testing.assert_allclose(padded.batch["advantages"][:, 0], expected[padded.non_tensor_batch["row_id"].astype(int)])
        np.testing.assert_array_equal(padded.batch["advantages"][:, 2], 0)
        updates = utils["split_batch_by_wg_ids"](padded, worker_groups)
        self.assertNotIn("policy:REVISE", updates)
        for wg_id, sub_batch in updates.items():
            self.assertTrue(all(wg == wg_id for wg in sub_batch.non_tensor_batch["wg_id"]))
            np.testing.assert_allclose(sub_batch.batch["advantages"][:, 0],
                                       expected[sub_batch.non_tensor_batch["row_id"].astype(int)])


if __name__ == "__main__":
    unittest.main()
