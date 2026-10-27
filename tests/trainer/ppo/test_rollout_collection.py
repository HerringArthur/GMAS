"""Run collection control flow and recording on CPU with mocked GPU boundaries."""

import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[3]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixtures = load_file("collection_fixtures", "tests/trainer/ppo/test_benchmark.py")
with patch.dict(sys.modules, {"verl.trainer.ppo.benchmark": fixtures.benchmark}):
    collection = load_file("collection", "verl/trainer/ppo/rollout_collection.py")


class Config(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name, value):
        self[name] = value


def config(folder, scene="search"):
    return Config(
        trainer=Config(rollout_only=True, total_epochs=1, total_training_steps=2,
                       resume_mode="auto", resume_from_path="old-checkpoint", save_freq=1,
                       val_only=True, val_before_train=True, test_freq=1,
                       benchmark=Config(output_dir=folder, samples_per_step=128, warmup_steps=0)),
        actor_rollout_ref=Config(actor=Config(strategy="fsdp", use_kl_loss=True),
                                rollout=Config(mode="sync")),
        reward_model=Config(enable=True),
        algorithm=Config(use_kl_in_reward=True, filter_groups=Config(enable=True)),
        env=Config(env_name=scene, seed=42, rollout=Config(n=2, val_n=16)),
        data=Config(train_batch_size=2, val_batch_size=10),
        agent=Config(model_ids=["fixed/router", "fixed/worker"]),
    )


def production_methods(path, class_name, method_names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(node for node in tree.body if getattr(node, "name", "") == class_name)
    methods = [node for node in cls.body if getattr(node, "name", "") in method_names]
    for method in methods:
        method.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future] + methods, type_ignores=[])), str(path), "exec"), namespace)


class PromptBatch:
    def __init__(self, question_count=2):
        self.question_count = question_count
        self.batch = {key: [1, 2] for key in ("input_ids", "attention_mask", "position_ids")}
        self.non_tensor_batch = {key: ["original"] for key in (
            "raw_prompt_ids", "data_source", "env_kwargs", "raw_prompt", "tools_kwargs", "multi_modal_data",
        )}

    @classmethod
    def from_single_dict(cls, data):
        return deepcopy(data)

    def __len__(self):
        return self.question_count

    def pop(self, batch_keys, non_tensor_batch_keys):
        result = PromptBatch(self.question_count)
        result.batch = {key: self.batch.pop(key) for key in batch_keys}
        result.non_tensor_batch = {key: self.non_tensor_batch.pop(key) for key in non_tensor_batch_keys}
        return result


class RolloutCollectionTests(unittest.TestCase):
    def test_mode_disables_training_services_but_keeps_model_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = config(folder)
            sources = deepcopy(cfg.agent.model_ids)
            collection.configure_rollout_only(cfg)
            self.assertEqual(cfg.agent.model_ids, sources)
            self.assertEqual(cfg.trainer.resume_mode, "disable")
            self.assertIsNone(cfg.trainer.resume_from_path)
            self.assertEqual(cfg.trainer.save_freq, -1)
            self.assertFalse(cfg.trainer.val_only or cfg.trainer.val_before_train)
            self.assertEqual(cfg.trainer.test_freq, -1)
            self.assertFalse(cfg.reward_model.enable or cfg.algorithm.use_kl_in_reward)
            self.assertFalse(cfg.actor_rollout_ref.actor.use_kl_loss)
            self.assertFalse(cfg.algorithm.filter_groups.enable)
            cfg.trainer.rollout_only = False
            before = deepcopy(cfg)
            collection.configure_rollout_only(cfg)
            self.assertEqual(cfg, before)

    def test_invalid_collection_settings_fail_before_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            for setting, value in (("samples_per_step", 0), ("output_dir", None)):
                cfg = config(folder)
                cfg.trainer.benchmark[setting] = value
                with self.subTest(setting=setting), self.assertRaises(ValueError):
                    collection.configure_rollout_only(cfg)
            for component, setting, value in (("actor", "strategy", "megatron"), ("rollout", "mode", "async")):
                cfg = config(folder)
                cfg.actor_rollout_ref[component][setting] = value
                with self.subTest(setting=setting), self.assertRaises(ValueError):
                    collection.configure_rollout_only(cfg)
            (Path(folder) / "rollout_samples").mkdir()
            with self.assertRaises(FileExistsError):
                collection.configure_rollout_only(config(folder))

    def test_fit_collects_both_scenes_without_updates_resume_validation_or_saving(self):
        for scene, worker_role in (("search", "SEARCH"), ("math", "SOLVE")):
            with self.subTest(scene=scene), tempfile.TemporaryDirectory() as folder:
                cfg = config(folder, scene)
                collection.configure_rollout_only(cfg)
                # Even stale training settings cannot divert fit into training.
                cfg.trainer.save_freq = 1
                cfg.trainer.val_before_train = True
                cfg.trainer.resume_mode = "disable"
                forbidden = Mock(side_effect=AssertionError("training service invoked"))
                rollout = fixtures.Batch()
                rollout.non_tensor_batch.update({
                    "agent_id": ["Router Agent", "Router Agent", worker_role],
                    "is_action_valid": [False, True, True], "pass": [1.0, 0.0, 1.0],
                    "data_source": [scene] * 3,
                })
                collector = Mock(return_value=rollout)
                trainer = SimpleNamespace(
                    config=cfg, train_dataloader=[PromptBatch()] * 3,
                    tokenizers={"router": fixtures.Tokenizer("R"), "worker": fixtures.Tokenizer("W")},
                    traj_collector=SimpleNamespace(multi_turn_loop=collector),
                    actor_rollout_wg={"router": SimpleNamespace(update_actor=forbidden, compute_log_prob=forbidden)},
                    envs=object(), _load_checkpoint=forbidden, _save_checkpoint=forbidden, _validate=forbidden,
                )
                ns = {"DataProto": PromptBatch}
                production_methods("verl/trainer/ppo/ray_trainer.py", "RayPPOTrainer",
                                   {"fit", "_collect_rollout_batch"}, ns)
                trainer._collect_rollout_batch = lambda data: ns["_collect_rollout_batch"](trainer, data)
                omega = ModuleType("omegaconf")
                omega.OmegaConf = SimpleNamespace(to_container=lambda value, **kw: deepcopy(value))
                with patch.dict(sys.modules, {"omegaconf": omega, "verl.trainer.ppo.rollout_collection": collection}):
                    ns["fit"](trainer)
                forbidden.assert_not_called()
                self.assertEqual(collector.call_count, 2)
                for call in collector.call_args_list:
                    self.assertTrue(call.kwargs["is_train"])
                    self.assertIs(call.kwargs["envs"], trainer.envs)
                    self.assertEqual(call.kwargs["gen_batch"].non_tensor_batch["env_kwargs"], ["original"])
                paths = sorted(Path(folder).glob("rollout_samples/*.jsonl"))
                self.assertEqual(len(paths), 2)
                for path in paths:
                    episodes = [json.loads(line) for line in path.read_text().splitlines()]
                    self.assertEqual({episode["pass"] for episode in episodes}, {0.0, 1.0})
                    self.assertEqual(sum(len(episode["turns"]) for episode in episodes), 3)
                    self.assertEqual(sum(turn["is_action_valid"] is False for episode in episodes for turn in episode["turns"]), 1)
                sft = load_file("collection_sft", "examples/data_preprocess/router_sft.py")
                json_records = list(sft.parse_log(paths[0]))
                text_records = list(sft.parse_log(paths[0].with_suffix(".txt")))
                self.assertEqual(len(json_records), 2)
                for left, right in zip(json_records, text_records):
                    left.pop("source_file")
                    right.pop("source_file")
                    self.assertEqual(left, right)
                metrics = [json.loads(line)["metrics"] for line in (Path(folder) / "metrics.jsonl").read_text().splitlines()]
                for row in metrics:
                    self.assertEqual(row["collection/parameter_updates"], 0)
                    self.assertEqual(row["benchmark/training_rows"], 0)
                    self.assertEqual(row["benchmark/copied_rows"], 0)
                self.assertTrue(json.loads((Path(folder) / "summary.json").read_text())["complete"])
                self.assertFalse(list(Path(folder).rglob("global_step_*")))

    def test_collection_horizon_and_warmup_are_checked_before_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            for steps, warmup in ((0, 0), (3, 0), (2, 2)):
                cfg = config(folder)
                cfg.trainer.total_training_steps = steps
                cfg.trainer.benchmark.warmup_steps = warmup
                collection.configure_rollout_only(cfg)
                generate = Mock()
                with self.subTest(steps=steps, warmup=warmup), self.assertRaises(ValueError):
                    collection.collect_rollouts(cfg, [object()] * 2, {}, generate)
                generate.assert_not_called()

    def test_workers_use_rollout_role_and_training_keeps_actor_role(self):
        for rollout_only in (True, False):
            with self.subTest(rollout_only=rollout_only):
                captured = []
                cfg = config("unused")
                cfg.trainer.rollout_only = rollout_only
                worker_cfg = Config(model=Config(path=None), actor=Config(optim=Config(total_training_steps=-1)))
                mapping = {"router": [{"agent_id": "Router Agent", "model_id": "fixed/router", "config_actor_rollout_ref": worker_cfg}]}
                class WorkerGroup:
                    def __init__(self, **kwargs):
                        pass

                    def spawn(self, prefix_set):
                        return {name: SimpleNamespace(init_model=Mock()) for name in prefix_set}

                ns = {
                    "normalize_agent_id": lambda name: name.replace(" ", ""),
                    "OmegaConf": SimpleNamespace(select=lambda *args: None),
                    "RayClassWithInitArgs": lambda **kwargs: captured.append(kwargs) or kwargs,
                    "create_colocated_worker_cls": lambda **kwargs: kwargs,
                    "Role": SimpleNamespace(ActorRollout="actor_rollout"),
                }
                production_methods("verl/trainer/ppo/ray_trainer.py", "RayPPOTrainer", {"init_workers"}, ns)
                trainer = SimpleNamespace(
                    config=cfg, total_training_steps=2, wg_to_agents_mapping=mapping,
                    resource_pool_manager=SimpleNamespace(create_resource_pool=Mock(), resource_pool_dict={"global": "pool"},
                                                          get_resource_pool=lambda role: "pool"),
                    hybrid_engine=True, use_critic=False, use_rm=False, use_reference_policy=False,
                    role_worker_mapping={"actor_rollout": object}, ray_worker_group_cls=WorkerGroup, device_name="cpu",
                )
                ns["init_workers"](trainer)
                self.assertEqual(captured[0]["role"], ("rollout" if rollout_only else "actor_rollout") + "/RouterAgent/")
                self.assertEqual(worker_cfg.model.path, "fixed/router")
                self.assertEqual(worker_cfg.actor.optim.total_training_steps, -1 if rollout_only else 2)
                trainer.actor_rollout_wg["router"].init_model.assert_called_once()

    def test_rollout_worker_has_no_optimizer_and_freezes_loaded_model(self):
        actor_module = ModuleType("verl.workers.actor")
        actor_module.DataParallelPPOActor = Mock(side_effect=AssertionError("training actor created"))
        omega = ModuleType("omegaconf")
        omega.OmegaConf = SimpleNamespace(to_container=lambda value: value, create=Config)
        model = SimpleNamespace(requires_grad_=Mock(), eval=Mock(), _fsdp_wrapped_module=object())
        builder = Mock(return_value=(model, None, None, object()))
        ns = {
            "import_external_libs": lambda value: None, "copy_to_local": lambda path, **kwargs: path,
            "fsdp_version": lambda value: 1,
        }
        production_methods("verl/workers/fsdp_workers.py", "ActorRolloutRefWorker", {"init_model"}, ns)
        cfg = Config(model=Config(path="fixed/router"), actor=Config(fsdp_config=Config(reshard_after_forward=True)))
        worker = SimpleNamespace(
            config=cfg, _is_actor=False, _is_rollout=True, _is_ref=False,
            _is_offload_param=False, _is_offload_optimizer=False,
            _build_model_optimizer=builder, _build_rollout=Mock(return_value=(object(), object())),
        )
        with patch.dict(sys.modules, {"omegaconf": omega, "verl.workers.actor": actor_module}):
            ns["init_model"](worker)
        self.assertIsNone(builder.call_args.kwargs["optim_config"])
        self.assertIs(builder.call_args.kwargs["fsdp_config"], cfg.actor.fsdp_config)
        self.assertIsNone(worker.actor_optimizer)
        model.requires_grad_.assert_called_once_with(False)
        model.eval.assert_called_once()
        actor_module.DataParallelPPOActor.assert_not_called()
        self.assertFalse(hasattr(worker, "checkpoint_manager"))

    def test_collection_dataloader_does_not_require_validation_files(self):
        cfg = config("unused")
        cfg.data.val_files = "/missing/validation.parquet"
        forbidden = Mock(side_effect=AssertionError("validation dataset opened"))
        main = ModuleType("verl.trainer.main_ppo")
        main.create_rl_dataset = forbidden
        main.create_rl_sampler = forbidden
        loader = Mock(return_value=[PromptBatch()] * 3)
        ns = {"StatefulDataLoader": loader}
        production_methods("verl/trainer/ppo/ray_trainer.py", "RayPPOTrainer", {"_create_dataloader"}, ns)
        trainer = SimpleNamespace(config=cfg, rollout_only=True)
        with patch.dict(sys.modules, {"verl.trainer.main_ppo": main}):
            ns["_create_dataloader"](trainer, object(), None, object(), object())
        forbidden.assert_not_called()
        loader.assert_called_once()
        self.assertIsNone(trainer.val_dataloader)
        self.assertEqual(trainer.total_training_steps, 2)

    def test_collection_builds_only_sampling_environments_in_both_scenes(self):
        tree = ast.parse((ROOT / "agent_system/environments/env_manager.py").read_text())
        function = next(node for node in tree.body if getattr(node, "name", "") == "make_envs")
        for scene in ("math", "search"):
            cfg = config("unused", scene)
            factory = Mock(return_value=object())
            package = ModuleType(f"agent_system.environments.env_package.{scene}")
            setattr(package, f"build_{scene}_envs", factory)
            setattr(package, f"{scene}_projection", object())
            manager = Mock(return_value=object())
            ns = {"partial": lambda value: value, "MathEnvironmentManager": manager, "SearchEnvironmentManager": manager}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "<environments>", "exec"), ns)
            with patch.dict(sys.modules, {package.__name__: package}):
                sampled, validation = ns["make_envs"](cfg)
            self.assertIs(sampled, manager.return_value)
            self.assertIsNone(validation)
            factory.assert_called_once()
            self.assertTrue(factory.call_args.kwargs["is_train"])
            self.assertEqual(factory.call_args.kwargs["env_num"], cfg.data.train_batch_size)


if __name__ == "__main__":
    unittest.main()
