"""Check Router reward shaping on plain dictionaries, without model loading.

Run: python3 -m unittest discover -s tests/agent_system -p 'test_reward_shaping.py' -v
"""

import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_module():
    path = ROOT / "agent_system/multi_turn_rollout/reward_shaping.py"
    spec = importlib.util.spec_from_file_location("reward_shaping_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shaping = load_module()


class StrSubclass(str):
    """Stand in for object-dtype batch entries holding a str subclass."""


def row(agent_id, *, reason="", active=True):
    return {"agent_id": agent_id, "return_reason": reason, "active_masks": active}


class DecisionCountTests(unittest.TestCase):
    def test_counts_only_router_rows(self):
        trajectories = [
            [row("Router Agent"), row("SEARCH"), row("Router Agent"), row("ANSWER")],
            [row("Router Agent")],
        ]
        np.testing.assert_array_equal(shaping.router_decision_counts(trajectories), [2, 1])

    def test_inactive_rows_are_not_counted(self):
        trajectories = [[row("Router Agent"), row("Router Agent", active=False)]]
        np.testing.assert_array_equal(shaping.router_decision_counts(trajectories), [1])

    def test_agent_id_subclass_is_normalized(self):
        trajectories = [[row(StrSubclass("Router Agent"))]]
        np.testing.assert_array_equal(shaping.router_decision_counts(trajectories), [1])

    def test_empty_trajectory_counts_zero(self):
        np.testing.assert_array_equal(shaping.router_decision_counts([[]]), [0])

    def test_returns_float_array(self):
        counts = shaping.router_decision_counts([[row("Router Agent")]])
        self.assertEqual(counts.dtype, np.float32)


class TruncationTests(unittest.TestCase):
    def test_trajectory_ending_in_finish_is_not_truncated(self):
        trajectories = [
            [row("Router Agent", reason="finish"), row("ANSWER", reason="finish")],
        ]
        np.testing.assert_array_equal(shaping.truncation_flags(trajectories), [0.0])

    def test_last_non_empty_reason_decides(self):
        trajectories = [
            [row("Router Agent"), row("Router Agent"), row("ANSWER", reason="loop_limit")],
            [row("Router Agent"), row("SEARCH", reason="search")],
        ]
        np.testing.assert_array_equal(shaping.truncation_flags(trajectories), [1.0, 1.0])

    def test_trajectory_without_any_reason_is_left_alone(self):
        # Math scenes and other orchestras never set return_reason.
        np.testing.assert_array_equal(shaping.truncation_flags([[row("Router Agent")]]), [0.0])
        np.testing.assert_array_equal(shaping.truncation_flags([[]]), [0.0])


class ShapingTests(unittest.TestCase):
    def test_zero_costs_leave_rewards_untouched(self):
        rewards = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        shaped = shaping.shape_episode_rewards(
            rewards,
            decision_counts=[3.0, 1.0, 16.0],
            truncation_flags=[1.0, 0.0, 1.0],
            decision_cost=0.0,
            truncation_penalty=0.0,
        )
        np.testing.assert_array_equal(shaped, rewards)

    def test_decision_cost_subtracts_per_router_decision(self):
        shaped = shaping.shape_episode_rewards(
            [1.0, 1.0],
            decision_counts=[1.0, 8.0],
            truncation_flags=[0.0, 0.0],
            decision_cost=0.05,
            truncation_penalty=0.0,
        )
        np.testing.assert_allclose(shaped, [0.95, 0.6], rtol=1e-6)

    def test_truncation_penalty_subtracts_once(self):
        shaped = shaping.shape_episode_rewards(
            [1.0, 0.0],
            decision_counts=[0.0, 0.0],
            truncation_flags=[0.0, 1.0],
            decision_cost=0.0,
            truncation_penalty=0.2,
        )
        np.testing.assert_allclose(shaped, [1.0, -0.2], rtol=1e-6)

    def test_terms_combine(self):
        shaped = shaping.shape_episode_rewards(
            [1.0, 0.0],
            decision_counts=[2.0, 16.0],
            truncation_flags=[0.0, 1.0],
            decision_cost=0.02,
            truncation_penalty=0.2,
        )
        np.testing.assert_allclose(shaped, [0.96, -0.52], rtol=1e-6)

    def test_finishing_wrong_beats_burning_the_budget(self):
        # Same wrong answer, one submission versus a full truncated loop.
        shaped = shaping.shape_episode_rewards(
            [0.0, 0.0],
            decision_counts=[1.0, 16.0],
            truncation_flags=[0.0, 1.0],
            decision_cost=0.02,
            truncation_penalty=0.2,
        )
        self.assertGreater(shaped[0], shaped[1])

    def test_empty_batch(self):
        shaped = shaping.shape_episode_rewards(
            [], decision_counts=[], truncation_flags=[],
            decision_cost=0.02, truncation_penalty=0.2,
        )
        self.assertEqual(len(shaped), 0)

    def test_inputs_are_not_mutated(self):
        rewards = np.array([1.0], dtype=np.float32)
        shaping.shape_episode_rewards(
            rewards, decision_counts=[4.0], truncation_flags=[1.0],
            decision_cost=0.02, truncation_penalty=0.2,
        )
        np.testing.assert_array_equal(rewards, [1.0])


class SettingsTests(unittest.TestCase):
    class Config(dict):
        """Stand in for the OmegaConf nodes; only ``get`` is used."""

    def test_missing_section_disables_shaping(self):
        self.assertEqual(shaping.read_shaping_settings(self.Config()), (0.0, 0.0))

    def test_missing_keys_default_to_zero(self):
        config = self.Config(env=self.Config(reward_shaping=self.Config()))
        self.assertEqual(shaping.read_shaping_settings(config), (0.0, 0.0))

    def test_reads_configured_costs(self):
        config = self.Config(env=self.Config(
            reward_shaping=self.Config(decision_cost=0.02, truncation_penalty=0.2),
        ))
        self.assertEqual(shaping.read_shaping_settings(config), (0.02, 0.2))

    def test_none_values_fall_back_to_zero(self):
        config = self.Config(env=self.Config(
            reward_shaping=self.Config(decision_cost=None, truncation_penalty=None),
        ))
        self.assertEqual(shaping.read_shaping_settings(config), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
