"""Exercise the real fit() control flow without importing Ray or CUDA dependencies.

The driver and batch utility function bodies are loaded from source unchanged.
NumPy-backed batches and mocked model boundaries test row routing, padding,
reward alignment, and skipped steps; they do not test model numerics.
Run: python3 -m unittest discover -s tests/trainer/ppo -p 'test_valid_action_training.py' -v
"""

import ast
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from enum import Enum
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[3]


class Tensor(np.ndarray):
    def size(self, dim):
        return self.shape[dim]

    def cpu(self):
        return self

    def detach(self):
        return self

    @property
    def device(self):
        return 'cpu'

    def sum(self, dim=None, **kwargs):
        return tensor(np.asarray(self).sum(axis=kwargs.pop('axis', dim), **kwargs))

    def unsqueeze(self, dim):
        return tensor(np.expand_dims(np.asarray(self), axis=dim))


def tensor(value):
    return np.asarray(value).view(Tensor)


class Batch:
    def __init__(self, batch=None, non_tensor_batch=None, meta_info=None):
        self.batch = batch or {}
        self.non_tensor_batch = non_tensor_batch or {}
        self.meta_info = meta_info if meta_info is not None else {}

    def __len__(self):
        return next(iter(self.batch.values())).shape[0]

    def __getitem__(self, index):
        return SimpleNamespace(
            batch={key: value[index] for key, value in self.batch.items()},
            non_tensor_batch={key: value[index] for key, value in self.non_tensor_batch.items()},
        )

    @classmethod
    def from_single_dict(cls, data):
        return deepcopy(data)

    def pop(self, batch_keys, non_tensor_batch_keys):
        return Batch(
            {key: self.batch.pop(key) for key in batch_keys},
            {key: self.non_tensor_batch.pop(key) for key in non_tensor_batch_keys},
            self.meta_info,
        )

    def select_idxs(self, indices):
        return Batch(
            {key: value[indices] for key, value in self.batch.items()},
            {key: value[indices] for key, value in self.non_tensor_batch.items()},
            self.meta_info,
        )

    @classmethod
    def concat(cls, batches):
        return Batch(
            {key: tensor(np.concatenate([b.batch[key] for b in batches])) for key in batches[0].batch},
            {key: np.concatenate([b.non_tensor_batch[key] for b in batches]) for key in batches[0].non_tensor_batch},
            batches[0].meta_info,
        )

    def union(self, other):
        self.batch.update(other.batch)
        self.non_tensor_batch.update(other.non_tensor_batch)
        self.meta_info.update(other.meta_info)
        return self

    def reorder(self, indices):
        selected = self.select_idxs(indices)
        self.batch, self.non_tensor_batch = selected.batch, selected.non_tensor_batch


class Config(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error

    def __setattr__(self, key, value):
        self[key] = value


@contextmanager
def timer(name, timings):
    try:
        yield
    finally:
        timings[name] = 1.0


def load_driver():
    """Load production bodies, avoiding unrelated import-time GPU dependencies."""
    namespace = {
        'np': np, 'DataProto': Batch, 'deepcopy': deepcopy, 'Enum': Enum,
        '_timer': timer, 'pprint': lambda *args: None, 'defaultdict': defaultdict,
        'torch': SimpleNamespace(
            sum=lambda value, dim: value.sum(dim), zeros_like=np.zeros_like,
            tensor=lambda value, **kw: tensor(np.atleast_1d(value).astype(kw.get('dtype', float))),
            float32=np.float32, no_grad=nullcontext,
            stack=lambda values: tensor(np.stack([np.asarray(value) for value in values])),
            mean=lambda value, dim=None: tensor(np.asarray(value).mean(axis=dim)),
            std=lambda value: tensor(np.asarray(value).std(ddof=1)),
        ),
        'masked_mean': lambda value, mask, axis: tensor(
            (np.asarray(value) * mask).sum(axis=axis) / np.asarray(mask).sum(axis=axis)
        ),
    }
    specs = {
        'agent_system/multi_turn_rollout/utils.py': {
            'get_valid_action_mask', 'adjust_batch',
            'split_batch_by_wg_ids', 'combine_batches',
        },
        'verl/trainer/ppo/ray_trainer.py': {
            'AdvantageEstimator', 'compute_response_mask', 'apply_invalid_action_penalty',
            'apply_kl_penalty', 'compute_advantage',
        },
        'verl/trainer/ppo/core_algos.py': {'compute_grpo_outcome_advantage'},
    }
    nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    for path, names in specs.items():
        tree = ast.parse((ROOT / path).read_text())
        nodes.extend(node for node in tree.body if getattr(node, 'name', None) in names)
        if path.endswith('ray_trainer.py'):
            trainer = next(node for node in tree.body if getattr(node, 'name', '') == 'RayPPOTrainer')
            nodes.append(next(node for node in trainer.body if getattr(node, 'name', '') == 'fit'))
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), '<trainer-test>', 'exec'), namespace)
    namespace['core_algos'] = SimpleNamespace(
        compute_grpo_outcome_advantage=namespace['compute_grpo_outcome_advantage'],
        kl_penalty=lambda logprob, ref_logprob, kl_penalty: logprob - ref_logprob,
    )
    return namespace


def rollout(groups, valid=None):
    count = len(groups)
    row_ids = np.arange(count) + 10
    batch = Batch(
        batch={
            'input_ids': tensor(np.column_stack([row_ids] * 4)),
            'prompts': tensor(np.column_stack([row_ids] * 2)),
            'responses': tensor(np.column_stack([row_ids] * 2)),
            'position_ids': tensor(np.tile(np.arange(4), (count, 1))),
            'attention_mask': tensor(np.ones((count, 4), dtype=int)),
        },
        non_tensor_batch={
            'row_id': row_ids, 'wg_id': np.array(groups), 'agent_id': np.array(groups),
            'raw_prompt_ids': np.array(['prompt'] * count), 'data_source': np.array(['test'] * count),
        },
    )
    if valid is not None:
        batch.non_tensor_batch['is_action_valid'] = np.array(valid, dtype=bool)
        batch.non_tensor_batch['protocol_repaired'] = ~np.array(valid, dtype=bool)
    return batch


class ValidActionTrainingTests(unittest.TestCase):
    def run_fit(self, batches, async_reward=False, balance=True, use_penalty=True):
        ns = load_driver()
        self.calls, self.logs, self.adjustments = [], [], []
        groups = ['router', 'worker', 'unused']

        def inspect_batch(stage, batch, wg=None):
            self.assertGreater(len(batch), 0, stage)
            if wg is not None:
                self.assertTrue((batch.non_tensor_batch['wg_id'] == wg).all())
            self.calls.append((stage, wg, batch.non_tensor_batch['row_id'].tolist()))

        def scores(batch):
            inspect_batch('reward', batch)
            ids = batch.non_tensor_batch['row_id']
            return tensor(np.column_stack([np.zeros(len(batch)), ids])), {'reward_row': ids.tolist()}

        def assert_reward_alignment(batch):
            flags = batch.non_tensor_batch.get('is_action_valid', np.ones(len(batch), dtype=bool))
            expected = batch.non_tensor_batch['row_id'] - 0.1 * (~flags) * use_penalty
            np.testing.assert_allclose(batch.batch['token_level_scores'][:, -1], expected)
            np.testing.assert_array_equal(batch.non_tensor_batch['reward_row'], batch.non_tensor_batch['row_id'])

        def worker(wg):
            def log_prob(batch):
                inspect_batch('log_prob', batch, wg)
                assert_reward_alignment(batch)
                values = tensor(np.ones_like(batch.batch['responses']))
                return Batch({'old_log_probs': values, 'entropys': values})

            def ref(batch):
                inspect_batch('reference', batch, wg)
                return Batch({'ref_log_prob': tensor(np.zeros_like(batch.batch['responses']))})

            def update(batch):
                inspect_batch('actor', batch, wg)
                assert_reward_alignment(batch)
                self.assertIn('advantages', batch.batch)
                return Batch(meta_info={f'{wg}/metrics': {}})

            return SimpleNamespace(compute_log_prob=log_prob, compute_ref_log_prob=ref, update_actor=update)

        def values(batch):
            inspect_batch('values', batch)
            return Batch({'values': tensor(np.zeros_like(batch.batch['responses']))})

        def critic_update(batch):
            inspect_batch('critic', batch)
            return Batch(meta_info={'metrics': {}})

        def advantage(batch, **kwargs):
            inspect_batch('advantage', batch)
            assert_reward_alignment(batch)
            self.assertEqual(kwargs['credit_mode'], 'router_subagent')
            np.testing.assert_array_equal(batch.batch['token_level_rewards'], batch.batch['token_level_scores'])
            batch.batch['advantages'] = batch.batch['token_level_scores']
            batch.batch['returns'] = batch.batch['token_level_scores']
            return batch

        original_adjust = ns['adjust_batch']

        def adjust(config, data, wg_id):
            inspect_batch('adjust', data, wg_id)
            self.adjustments.append((wg_id, len(data)))
            return original_adjust(config, data, wg_id)

        original_penalty = ns['apply_invalid_action_penalty']

        def penalty(batch, **kwargs):
            inspect_batch('penalty', batch)
            return original_penalty(batch, **kwargs)

        def balance_batch(batch, wg_id, **kwargs):
            inspect_batch('balance', batch, wg_id)
            batch.reorder(np.arange(len(batch))[::-1])

        actor_config = Config(
            actor=Config(use_adaptive_ppo_mini_batch_size=True, ppo_mini_update_num=1,
                         ppo_micro_batch_size_per_gpu=2, use_kl_loss=True, loss_agg_mode='token-mean',
                         use_invalid_action_penalty=use_penalty, invalid_action_penalty_coef=0.1),
            rollout=Config(log_prob_micro_batch_size_per_gpu=2, n=1, multi_turn=Config(enable=False)),
            ref=Config(log_prob_micro_batch_size_per_gpu=2),
        )
        config = Config(
            trainer=Config(project_name='test', experiment_name='filter', logger=[], total_epochs=1,
                           val_before_train=False, balance_batch=balance, n_gpus_per_node=1, nnodes=1,
                           critic_warmup=0, test_freq=1, save_freq=1, rollout_data_dir='fake-dump',
                           benchmark=Config(output_dir='fake-benchmark')),
            actor_rollout_ref=actor_config, critic=Config(ulysses_sequence_parallel_size=1),
            reward_model=Config(launch_reward_fn_async=async_reward), env=Config(rollout=Config(n=1)),
            algorithm=Config(adv_estimator='grpo', gamma=1.0, lam=1.0, use_kl_in_reward=False,
                             credit_mode='router_subagent', use_pf_ppo=False,
                             pf_ppo=Config(reweight_method='none', weight_pow=2),
                             gigpo=Config(step_advantage_w=1.0, mode='mean_std_norm', enable_similarity=False,
                                          similarity_thresh=0.95)),
        )
        queue = iter(batches)
        progress = Mock()
        self.benchmark = Mock()
        self.benchmark.record_rollouts.return_value = {}
        training_logger = SimpleNamespace(log=lambda data, step: self.logs.append((step, dict(data))))
        stubs = {}
        for name, attrs in {
            'omegaconf': {'OmegaConf': SimpleNamespace(to_container=lambda value, **kw: deepcopy(value), create=deepcopy)},
            'verl.utils.tracking': {'Tracking': lambda **kw: training_logger},
            'verl.trainer.ppo.benchmark': {'TrainingBenchmark': lambda *args: self.benchmark},
        }.items():
            stubs[name] = ModuleType(name)
            stubs[name].__dict__.update(attrs)

        ns.update(
            OmegaConf=stubs['omegaconf'].OmegaConf, tqdm=lambda **kw: progress, adjust_batch=adjust,
            compute_reward=lambda batch, fn: scores(batch),
            compute_reward_async=SimpleNamespace(remote=lambda batch, *args: scores(batch)),
            ray=SimpleNamespace(get=lambda result: result), apply_invalid_action_penalty=penalty,
            compute_advantage=advantage, agg_loss=lambda **kw: tensor(1.0), reduce_metrics=lambda x: x,
            compute_data_metrics=lambda **kw: {},
            compute_timing_metrics=lambda **kw: {
                f'timing_s/{name}': value for name, value in kw['timing_raw'].items()
            },
            compute_throughout_metrics=lambda **kw: {},
        )
        self.trainer = SimpleNamespace(
            config=config, total_training_steps=len(batches), train_dataloader=[batches[0]] * len(batches),
            _load_checkpoint=lambda: None, _save_checkpoint=Mock(), _validate=Mock(return_value={'val/test': 1}),
            val_reward_fn=object(), reward_fn=None, envs=None, tokenizers={
                wg: SimpleNamespace(batch_decode=lambda rows, **kw: rows[:, 0].tolist()) for wg in groups
            },
            traj_collector=SimpleNamespace(multi_turn_loop=lambda **kw: next(queue)),
            wg_to_agents_mapping={wg: [{'config_actor_rollout_ref': actor_config}] for wg in groups},
            actor_rollout_wg={wg: worker(wg) for wg in groups}, ref_in_actor=True,
            use_reference_policy=True, use_critic=True, use_rm=True,
            rm_wg=SimpleNamespace(compute_rm_score=lambda batch: (
                inspect_batch('reward_model', batch) or Batch({'rm_scores': tensor(np.zeros_like(batch.batch['responses']))})
            )),
            critic_wg=SimpleNamespace(compute_values=values, update_critic=critic_update),
            _balance_batch=balance_batch, _dump_generations=Mock(),
            resource_pool_manager=SimpleNamespace(get_n_gpus=lambda: 1),
        )
        with patch.dict(sys.modules, stubs):
            ns['fit'](self.trainer)
        self.assertEqual(progress.update.call_count, len(batches))
        self.assertEqual(self.trainer._validate.call_count, len(batches))
        self.assertEqual(self.trainer._save_checkpoint.call_count, len(batches))
        self.assertEqual(self.benchmark.record_step.call_count, len(batches))

    def test_keeps_invalid_rows_through_padding_and_updates_with_penalized_rewards(self):
        for asynchronous in [False, True]:
            with self.subTest(asynchronous=asynchronous):
                source = rollout(['router', 'worker', 'router', 'worker', 'unused', 'worker', 'worker', 'router'],
                                 [True, True, False, False, False, True, True, False])
                self.run_fit([source], async_reward=asynchronous)
                self.assertEqual(self.adjustments, [('router', 3), ('worker', 4), ('unused', 1)])
                self.assertEqual(len(source), 8)
                self.assertEqual(self.logs[0][1]['training/invalid_action_rows'], 4)
                self.assertEqual(self.logs[0][1]['episode/valid_action_ratio'], 0.5)
                actors = {wg: ids for stage, wg, ids in self.calls if stage == 'actor'}
                self.assertEqual(set(actors), {'router', 'worker', 'unused'})
                self.assertEqual(len(actors['router']), 4)
                self.assertEqual(len(actors['worker']), 4)
                self.assertEqual(len(actors['unused']), 2)
                self.assertEqual(set(sum(actors.values(), [])), set(source.non_tensor_batch['row_id']))
                self.assertEqual(len(self.benchmark.record_rollouts.call_args.args[0]), 8)
                trained = self.benchmark.record_step.call_args.args[2]
                self.assertEqual(len(trained), 10)
                np.testing.assert_array_equal(
                    trained.non_tensor_batch['rollout_row_id'], trained.non_tensor_batch['row_id'] - 10,
                )
                np.testing.assert_array_equal(
                    trained.non_tensor_batch['is_action_valid'],
                    source.non_tensor_batch['is_action_valid'][trained.non_tensor_batch['row_id'] - 10],
                )
                self.assertEqual(trained.meta_info['router/ppo_mini_batch_size'], 4)
                self.assertEqual(trained.meta_info['worker/ppo_mini_batch_size'], 4)
                dumped = self.trainer._dump_generations.call_args.kwargs
                np.testing.assert_allclose(
                    dumped['scores'],
                    np.asarray(dumped['outputs']) - 0.1 * (~trained.non_tensor_batch['is_action_valid']),
                )
                self.assertEqual(dumped['outputs'], dumped['reward_extra_infos_dict']['reward_row'])

    def test_all_invalid_and_following_valid_batch_both_train(self):
        source = rollout(['router', 'worker'], [False, False])
        self.run_fit([source, rollout(['worker'], [True])])
        self.assertEqual(self.logs[0][1]['training/skipped_empty_batch'], 0)
        self.assertEqual(self.logs[1][1]['training/skipped_empty_batch'], 0)
        self.assertEqual(self.logs[0][1]['training/valid_action_rows'], 0)
        self.assertIn('timing_s/step', self.logs[0][1])
        self.assertEqual(self.adjustments, [('router', 1), ('worker', 1), ('worker', 1)])
        self.assertEqual(len(self.benchmark.record_step.call_args_list[0].args[2]), 4)
        self.assertEqual(len(source), 2)

    def test_all_invalid_last_step_still_updates_models(self):
        self.run_fit([rollout(['router', 'worker'], [False, False])])
        self.assertEqual({wg for stage, wg, _ in self.calls if stage == 'actor'}, {'router', 'worker'})
        self.assertEqual(self.trainer.global_steps, 2)
        self.trainer._dump_generations.assert_called_once()

    def test_disabled_penalty_still_keeps_invalid_rows(self):
        self.run_fit([rollout(['router', 'worker'], [False, True])], use_penalty=False)
        self.assertFalse(any(stage == 'penalty' for stage, _, _ in self.calls))
        self.assertEqual({wg for stage, wg, _ in self.calls if stage == 'actor'}, {'router', 'worker'})

    def test_legacy_batch_without_validity_flags_is_fully_trainable(self):
        self.run_fit([rollout(['router', 'worker'])], balance=False)
        self.assertEqual(self.adjustments, [('router', 1), ('worker', 1)])
        self.assertEqual(self.logs[0][1]['episode/valid_action_ratio'], 1.0)
        self.assertFalse(any(stage == 'penalty' for stage, _, _ in self.calls))

    def test_empty_rollout_is_skipped(self):
        self.run_fit([rollout([], [])])
        self.assertEqual(self.calls, [])
        self.assertEqual(self.logs[0][1]['training/skipped_empty_batch'], 1)


class FormatRewardTests(unittest.TestCase):
    def make_batch(self):
        class TensorDict(dict):
            @property
            def batch_size(self):
                return next(iter(self.values())).shape[:1]

        batch = rollout(['worker', 'worker'], [True, False])
        batch.batch = TensorDict(batch.batch)
        batch.batch['attention_mask'][1, -1] = 0
        batch.batch['response_mask'] = tensor([[1, 1], [1, 0]])
        batch.batch['token_level_scores'] = tensor([[0.0, 1.0], [1.0, 0.0]])
        batch.non_tensor_batch.update(
            episode_rewards=np.array([1.0, 1.0]), uid=np.array(['Q', 'Q']),
            traj_uid=np.array(['T1', 'T2']),
        )
        return batch

    def test_penalty_changes_terminal_score_and_step_reward_but_not_episode_reward(self):
        ns, batch = load_driver(), self.make_batch()
        batch.batch['step_rewards'] = tensor([0.5, 0.5])
        _, metrics = ns['apply_invalid_action_penalty'](batch, invalid_action_penalty_coef=0.1)
        np.testing.assert_allclose(batch.batch['token_level_scores'], [[0.0, 1.0], [0.9, 0.0]])
        np.testing.assert_allclose(batch.batch['step_rewards'], [0.5, 0.4])
        np.testing.assert_array_equal(batch.non_tensor_batch['episode_rewards'], [1.0, 1.0])
        self.assertNotIn('invalid_action_penalty', batch.batch)
        self.assertEqual(metrics['episode/valid_action_ratio'], 0.5)

    def test_kl_combines_with_already_penalized_scores_once(self):
        ns, batch = load_driver(), self.make_batch()
        ns['apply_invalid_action_penalty'](batch, invalid_action_penalty_coef=0.1)
        batch.batch['old_log_probs'] = tensor(np.ones((2, 2)))
        batch.batch['ref_log_prob'] = tensor(np.zeros((2, 2)))
        controller = SimpleNamespace(value=0.2, update=Mock())
        ns['apply_kl_penalty'](batch, controller)
        np.testing.assert_allclose(batch.batch['token_level_rewards'], [[-0.2, 0.8], [0.7, 0.0]])
        np.testing.assert_allclose(batch.batch['token_level_scores'], [[0.0, 1.0], [0.9, 0.0]])
        controller.update.assert_called_once_with(current_kl=1.0, n_steps=2)

    def test_standard_grpo_normalizes_penalized_rewards_and_broadcasts_to_response(self):
        ns, batch = load_driver(), self.make_batch()
        ns['apply_invalid_action_penalty'](batch, invalid_action_penalty_coef=0.1)
        batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
        ns['compute_advantage'](batch, adv_estimator=ns['AdvantageEstimator'].GRPO,
                                credit_mode='standard', group_by_agent_id=True)
        scores = np.array([1.0, 0.9])
        expected = (scores - scores.mean()) / (scores.std(ddof=1) + 1e-6)
        np.testing.assert_allclose(
            batch.batch['advantages'], expected[:, None] * np.array([[1, 1], [1, 0]]),
        )


if __name__ == '__main__':
    unittest.main()
