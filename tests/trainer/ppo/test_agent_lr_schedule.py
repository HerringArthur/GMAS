"""Exercise worker configuration and real PyTorch schedules without Ray/CUDA."""

import ast
from collections import defaultdict
from enum import Enum
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from omegaconf import DictConfig, ListConfig, OmegaConf
import torch


ROOT = Path(__file__).resolve().parents[3]


def load_production():
    namespace = dict(OmegaConf=OmegaConf, DictConfig=DictConfig, ListConfig=ListConfig,
                     defaultdict=defaultdict, Enum=Enum, math=math,
                     LambdaLR=torch.optim.lr_scheduler.LambdaLR)
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for path, names in (
        ("agent_system/agent/utils.py", {"normalize_agent_id", "normalize_model_id", "omega_equal_resolved", "_set_specific_parameter", "build_wg_ids"}),
        ("verl/utils/torch_functional.py", {"get_constant_schedule_with_warmup", "get_cosine_schedule_with_warmup"}),
        ("verl/trainer/ppo/ray_trainer.py", {"Role"}),
    ):
        tree = ast.parse((ROOT / path).read_text())
        nodes.extend(n for n in tree.body if getattr(n, "name", "") in names)
        if path.endswith("ray_trainer.py"):
            trainer = next(n for n in tree.body if getattr(n, "name", "") == "RayPPOTrainer")
            nodes.append(next(n for n in trainer.body if getattr(n, "name", "") == "init_workers"))

    # Execute the unchanged schedule-construction block with a tiny optimizer.
    tree = ast.parse((ROOT / "verl/workers/fsdp_workers.py").read_text())
    total_assignment = lambda n: isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "total_steps" for t in n.targets)
    block = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                 and any(total_assignment(child) for child in n.body)
                 and "role" in ast.unparse(n.test))
    start = next(i for i, n in enumerate(block.body) if total_assignment(n))
    end = next(i for i, n in enumerate(block.body) if isinstance(n, ast.If) and ast.unparse(n.test) == "warmup_style == 'constant'")
    wrapper = ast.parse("def build_schedule(optim_config, actor_optimizer):\n    pass").body[0]
    wrapper.body = block.body[start:end + 1] + [ast.Return(value=ast.Name(id="actor_lr_scheduler", ctx=ast.Load()))]
    nodes.append(wrapper)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "<production-agent-schedule>", "exec"), namespace)
    return namespace


PRODUCTION = load_production()


class AgentScheduleTests(unittest.TestCase):
    def initialize(self, steps=12, sharing=False):
        config = OmegaConf.create({
            "agent": {"agent_ids": ["Router Agent", "SEARCH", "VERIFY", "ANSWER"],
                      "model_ids": ["model/qwen"] * 4, "model_sharing": sharing,
                      "agent_specific_parameters": None if sharing else {"actor": {"optim": {"lr": [1e-6, 2e-6, 3e-6, 4e-6]}}}},
            "actor_rollout_ref": {"actor": {"ppo_mini_update_num": 5, "optim": {"total_training_steps": -1, "lr": 1e-6}},
                                  "model": {"path": None}, "rollout": {}},
            "trainer": {},
        })
        mapping = PRODUCTION["build_wg_ids"](config)
        # Reproduce startup order: per-agent configs exist before the horizon.
        config.actor_rollout_ref.actor.optim.total_training_steps = steps
        self.assertTrue(all(a["config_actor_rollout_ref"].actor.optim.total_training_steps == -1
                            for agents in mapping.values() for a in agents))
        captured = []

        def construct(**kwargs):
            captured.append(kwargs["config"])
            return kwargs

        class WorkerGroup:
            def __init__(self, **kwargs):
                pass

            def spawn(self, prefix_set):
                return {name: SimpleNamespace(init_model=Mock()) for name in prefix_set}

        PRODUCTION.update(RayClassWithInitArgs=construct, create_colocated_worker_cls=lambda **kwargs: kwargs)
        trainer = SimpleNamespace(
            config=config, total_training_steps=steps, wg_to_agents_mapping=mapping,
            resource_pool_manager=SimpleNamespace(create_resource_pool=Mock(), resource_pool_dict={"global": "pool"},
                                                  get_resource_pool=lambda role: "pool"),
            hybrid_engine=True, use_critic=False, use_rm=False, use_reference_policy=False,
            role_worker_mapping={PRODUCTION["Role"].ActorRollout: object},
            ray_worker_group_cls=WorkerGroup, device_name="cpu",
        )
        PRODUCTION["init_workers"](trainer)
        return trainer, captured

    def test_all_agent_copies_receive_actual_steps_and_keep_learning_rates(self):
        for steps in (12, 37):
            for sharing in (False, True):
                with self.subTest(steps=steps, sharing=sharing):
                    trainer, captured = self.initialize(steps, sharing)
                    self.assertEqual(len(captured), 1 if sharing else 4)
                    for agents in trainer.wg_to_agents_mapping.values():
                        for agent in agents:
                            self.assertEqual(agent["config_actor_rollout_ref"].actor.optim.total_training_steps, steps)
                    self.assertEqual([cfg.actor.optim.lr for cfg in captured], [1e-6] if sharing else [1e-6, 2e-6, 3e-6, 4e-6])
                    self.assertTrue(all(cfg.actor.optim.total_training_steps == steps for cfg in captured))
                    self.assertTrue(all(cfg.model.path == "model/qwen" for cfg in captured))

    def test_invalid_horizon_fails_before_worker_construction(self):
        for steps in (-1, 0):
            with self.subTest(steps=steps), self.assertRaisesRegex(ValueError, "total_training_steps must be positive"):
                self.initialize(steps)

    def schedule(self, cfg):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.tensor(1.0))], lr=1.0)
        return optimizer, PRODUCTION["build_schedule"](cfg, optimizer)

    def test_warmup_ratio_and_cosine_follow_worker_horizon(self):
        _, captured = self.initialize()
        cfg = captured[0].actor.optim
        cfg.warmup_style = "cosine"
        cfg.lr_warmup_steps_ratio = 0.25
        optimizer, scheduler = self.schedule(cfg)
        values = [scheduler.get_last_lr()[0]]
        for _ in range(12):
            optimizer.step()
            scheduler.step()
            values.append(scheduler.get_last_lr()[0])
        for actual, expected in zip(values[:4], (0, 1 / 3, 2 / 3, 1)):
            self.assertAlmostEqual(actual, expected)
        self.assertTrue(all(a >= b for a, b in zip(values[3:], values[4:])))
        self.assertAlmostEqual(values[-1], 0)

    def test_constant_and_explicit_warmup_are_preserved(self):
        for explicit in (0, 2):
            with self.subTest(warmup=explicit):
                cfg = OmegaConf.create({"total_training_steps": 12, "warmup_style": "constant",
                                        "lr_warmup_steps": explicit, "lr_warmup_steps_ratio": 0.9})
                optimizer, scheduler = self.schedule(cfg)
                values = [scheduler.get_last_lr()[0]]
                for _ in range(3):
                    optimizer.step()
                    scheduler.step()
                    values.append(scheduler.get_last_lr()[0])
                self.assertEqual(values, [1, 1, 1, 1] if explicit == 0 else [0, 0.5, 1, 1])

    def test_worker_rejects_uninitialized_horizon(self):
        with self.assertRaisesRegex(ValueError, "actor optim.total_training_steps"):
            self.schedule(OmegaConf.create({"total_training_steps": -1}))


if __name__ == "__main__":
    unittest.main()
