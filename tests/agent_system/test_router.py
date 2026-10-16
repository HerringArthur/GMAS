"""CPU orchestration tests with model generation and heavyweight dependencies mocked.

Run: python3 -m unittest discover -s tests/agent_system -p 'test_router.py' -v
"""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


class FakeBatch:
    def __init__(self, size):
        self.size = size
        self.non_tensor_batch = {}
        self.meta_info = {}

    def __len__(self):
        return self.size


def load_classes():
    """Load the real agent/orchestra modules without requiring a GPU training stack."""
    def module(name, **attributes):
        result = ModuleType(name)
        result.__dict__.update(attributes)
        return result

    def unused(*args, **kwargs):
        raise AssertionError('The model execution boundary must be mocked in these tests.')

    stubs = {
        'transformers': module('transformers', PreTrainedTokenizer=object),
        'verl': module('verl', DataProto=FakeBatch),
        'verl.protocol': module('verl.protocol', **{
            name: unused for name in ('pad_dataproto_to_divisor', 'unpad_dataproto',
                                     'extract_dataproto_via_active_mask', 'restore_dataproto_via_active_mask')
        }),
        'agent_system.multi_turn_rollout.utils': module(
            'agent_system.multi_turn_rollout.utils',
            preprocess_batch=lambda **kwargs: FakeBatch(len(kwargs['gen_batch'])),
        ),
        'agent_system.agent.agents.router': module('agent_system.agent.agents.router'),
    }

    def load(name, relative_path):
        spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[name] = loaded
        spec.loader.exec_module(loaded)
        return loaded

    with patch.dict(sys.modules, stubs):
        load('agent_system.agent.registry', 'agent_system/agent/registry.py')
        load('agent_system.agent.agents.base', 'agent_system/agent/agents/base.py')
        load('agent_system.agent.orchestra.base', 'agent_system/agent/orchestra/base.py')
        load('verl.utils.reward_score.math_dapo', 'verl/utils/reward_score/math_dapo.py')
        context = load('agent_system.agent.agents.router.context', 'agent_system/agent/agents/router/context.py')
        load('agent_system.agent.agents.router.search_context', 'agent_system/agent/agents/router/search_context.py')
        router = load('agent_system.agent.agents.router.router_agent', 'agent_system/agent/agents/router/router_agent.py')
        worker = load('agent_system.agent.agents.router.subagent', 'agent_system/agent/agents/router/subagent.py')
        stubs['agent_system.agent.agents.router'].RouterAgent = router.RouterAgent
        stubs['agent_system.agent.agents.router'].SubAgent = worker.SubAgent
        orchestra = load('agent_system.agent.orchestra.router.router_orchestra', 'agent_system/agent/orchestra/router/router_orchestra.py')
    return router.RouterAgent, worker.SubAgent, orchestra.RouterMultiAgentOrchestra, context


RouterAgent, SubAgent, RouterOrchestra, Context = load_classes()


def config(loops=3, env_name='math', max_steps=None):
    return SimpleNamespace(
        agent=SimpleNamespace(
            agent_ids=['Router Agent', 'Sub Agent 1', 'Sub Agent 2', 'Sub Agent 3'],
            orchestra=SimpleNamespace(router=SimpleNamespace(max_loop_num=loops)),
        ),
        env=SimpleNamespace(env_name=env_name, max_steps=max_steps),
    )


class RouterTests(unittest.TestCase):
    def make_orchestra(self, loops=3, env_name='math', max_steps=None):
        cfg = config(loops, env_name, max_steps)
        names = cfg.agent.agent_ids
        self.groups = {name: object() for name in names}
        self.events = []
        self.prompts = []
        self.contexts = []
        self.orchestra = RouterOrchestra(
            names, ['model'] * len(names), {name: name for name in names},
            {name: None for name in names}, {name: None for name in names}, cfg,
        )
        return self.orchestra

    def script(self, name, responses):
        agent = self.orchestra.agents[name]
        queue = iter(responses)
        build_prompt = agent.build_prompt

        def capture(env_obs, team_context, step, **kwargs):
            self.events.append((name, list(team_context), step))
            self.contexts.append((name, kwargs.get('collaboration_context')))
            obs = build_prompt(env_obs, team_context, step, **kwargs)
            self.prompts.append((name, list(obs['text'])))
            return obs

        def generate(batch, group, mask, meta_info):
            self.assertIs(group, self.groups[name])
            texts = next(queue)
            self.assertEqual(len(texts), len(batch))
            batch.non_tensor_batch['agent_active_mask'] = mask.copy()
            return batch, texts

        agent.build_prompt = capture
        agent._generate_with_llm = generate

    def run_orchestra(self, active=(True,), step=1, observation='task', anchor=None):
        return self.orchestra.run(
            FakeBatch(len(active)), {'text': [observation] * len(active), 'anchor': anchor},
            self.groups, np.array(active), step,
        )

    def test_router_parser_only_accepts_configured_worker(self):
        router = RouterAgent('wg', None, None, config())
        cases = [
            ('<route>Sub Agent 1</route>', 'Sub Agent 1'),
            ('<route>FINISH</route>', 'FINISH'),
            ('<think>Choose carefully.</think>\n<route> Sub Agent 2 </route>', 'Sub Agent 2'),
            ('<route>Router Agent</route>', ''),
            ('<route>Unknown</route>', ''),
            ('<route>Sub Agent 1</route><route>Sub Agent 2</route>', ''),
            ('<route>Sub Agent 1', ''),
            ('<route>Sub Agent 1</route>extra', ''),
            ('<think><route>Sub Agent 1</route></think>', ''),
            ('<think><think>nested</think><route>Sub Agent 1</route>', ''),
            ('<think>one</think><think>two</think><route>Sub Agent 1</route>', ''),
        ]
        for response, expected in cases:
            with self.subTest(response=response):
                self.assertEqual(router.parse_response(response), expected)
        obs = {'text': ['task']}
        prompt = router.build_prompt(obs, ['shared work'], 2)['text'][0]
        self.assertIn('- <route>Sub Agent 1</route>', prompt)
        self.assertIn('- <route>Sub Agent 2</route>', prompt)
        self.assertIn('- <route>Sub Agent 3</route>', prompt)
        self.assertIn('shared work', prompt)
        self.assertEqual(obs, {'text': ['task']})

    def test_named_workers_and_independent_batch_routing(self):
        orchestra = self.make_orchestra(loops=2)
        self.assertIsInstance(orchestra.agents['Sub Agent 1'], SubAgent)
        self.assertEqual(orchestra.agents['Sub Agent 1'].name, 'Sub Agent 1')
        self.script('Router Agent', [
            ['<route>Sub Agent 1</route>', '<route>Sub Agent 2</route>', '<route>Sub Agent 1</route>'],
            ['<route>Sub Agent 1</route>', '<route>Sub Agent 1</route>', ''],
        ])
        self.script('Sub Agent 1', [
            ['<think>A reasoning</think>A1<handoff>CONTINUE</handoff>', '', ''],
            [r'A2\boxed{2}<handoff>HANDOFF</handoff>', 'A for sample 2<handoff>CONTINUE</handoff>', ''],
        ])
        self.script('Sub Agent 2', [['', 'B1<handoff>HANDOFF</handoff>', '']])
        actions, buffer = self.run_orchestra(active=(True, True, False))
        self.assertEqual(actions, [r'\boxed{2}', '', ''])
        self.assertEqual([entry['agent_id'] for entry in buffer],
                         ['Router Agent', 'Sub Agent 1', 'Sub Agent 2', 'Router Agent', 'Sub Agent 1'])
        self.assertEqual(buffer[3]['batch'].non_tensor_batch['agent_active_mask'].tolist(), [True, True, False])
        self.assertEqual(buffer[0]['batch'].non_tensor_batch['selected_agent'].tolist(), ['Sub Agent 1', 'Sub Agent 2', ''])
        self.assertIn('<think>A reasoning</think>', orchestra.team_context[0])
        self.assertIn('<handoff>CONTINUE</handoff>', orchestra.team_context[0])
        self.assertNotIn('B1', orchestra.team_context[0])
        self.assertIn('B1', self.events[-1][1][1])
        self.assertEqual(orchestra.team_context[2], '')
        schemas = [set(entry['batch'].non_tensor_batch) for entry in buffer]
        self.assertTrue(all(keys == schemas[0] for keys in schemas))

    def test_search_yields_and_continue_returns_to_router(self):
        orchestra = self.make_orchestra(loops=3, env_name='search')
        self.script('Router Agent', [
            ['<route>Sub Agent 1</route>'],
            ['<route>Sub Agent 1</route>'],
            ['<route>FINISH</route>'],
        ])
        self.script('Sub Agent 1', [
            ['<think>Need a tool.</think><search>query</search><handoff>CONTINUE</handoff>'],
            ['<answer>revised answer</answer><handoff>HANDOFF</handoff>'],
        ])
        actions, buffer = self.run_orchestra()
        self.assertEqual(actions, ['<search>query</search>'])
        self.assertEqual(len(buffer), 2)
        self.assertEqual(orchestra.current_agents.tolist(), [''])
        actions, buffer = self.run_orchestra(step=2, observation='task and tool result')
        self.assertEqual(actions, ['<answer>revised answer</answer>'])
        self.assertEqual([entry['agent_id'] for entry in buffer], ['Router Agent', 'Sub Agent 1', 'Router Agent'])
        self.assertIn('<search>query</search>', self.events[-1][1][0])
        self.assertEqual(orchestra.current_agents.tolist(), [''])

    def test_search_handoff_routes_after_tool_result(self):
        self.make_orchestra(loops=1, env_name='search')
        self.script('Router Agent', [['<route>Sub Agent 1</route>'], ['<route>Sub Agent 2</route>']])
        self.script('Sub Agent 1', [['<search>query</search><handoff>HANDOFF</handoff>']])
        self.script('Sub Agent 2', [['<answer>42</answer><handoff>CONTINUE</handoff>']])
        self.run_orchestra()
        actions, buffer = self.run_orchestra(step=2, observation='tool result')
        self.assertEqual(actions, ['<answer>42</answer>'])
        self.assertEqual([entry['agent_id'] for entry in buffer], ['Router Agent', 'Sub Agent 2'])

    def test_invalid_routes_retry_without_inventing_worker(self):
        self.make_orchestra(loops=2)
        self.script('Router Agent', [['<route>Unknown</route>'], ['<route>Sub Agent 2</route>']])
        self.script('Sub Agent 2', [['work<handoff>CONTINUE</handoff>']])
        actions, buffer = self.run_orchestra()
        self.assertEqual(actions, [''])
        self.assertEqual([entry['agent_id'] for entry in buffer], ['Router Agent', 'Router Agent', 'Sub Agent 2'])
        self.assertFalse(buffer[0]['batch'].non_tensor_batch['is_action_valid'][0])

    def test_invalid_worker_output_keeps_worker_and_is_not_submitted(self):
        self.make_orchestra(loops=2)
        self.script('Router Agent', [['<route>Sub Agent 1</route>']])
        self.script('Sub Agent 1', [['<answer>unfinished'], ['valid work<handoff>CONTINUE</handoff>']])
        actions, buffer = self.run_orchestra()
        self.assertEqual(actions, [''])
        self.assertEqual([entry['agent_id'] for entry in buffer], ['Router Agent', 'Sub Agent 1', 'Sub Agent 1'])
        self.assertTrue(buffer[-1]['batch'].non_tensor_batch['is_action_valid'][0])

    def test_invalid_routes_are_bounded(self):
        self.make_orchestra(loops=2)
        self.script('Router Agent', [['bad route'], ['bad route']])
        actions, buffer = self.run_orchestra()
        self.assertEqual(actions, [''])
        self.assertEqual(len(buffer), 2)

    def test_reset_clears_progress_and_selection(self):
        orchestra = self.make_orchestra(loops=1)
        self.script('Router Agent', [['<route>Sub Agent 1</route>']])
        self.script('Sub Agent 1', [['work<handoff>CONTINUE</handoff>']])
        self.run_orchestra()
        orchestra.reset()
        self.assertIsNone(orchestra.team_context)
        self.assertIsNone(orchestra.current_agents)
        self.assertIsNone(orchestra.memory)
        self.assertIsNone(orchestra.submission_actions)
        self.assertIsNone(orchestra.candidate_answers)
        self.assertIsNone(orchestra.candidate_authors)
        self.assertIsNone(orchestra.candidate_responses)
        self.assertIsNone(orchestra.candidate_repeat_counts)
        self.assertIsNone(orchestra.routed_agents)
        self.assertIsNone(orchestra.routing_instructions)
        self.assertIsNone(orchestra.return_reasons)
        self.assertIsNone(orchestra.work_records)
        self.assertIsNone(orchestra.candidate_records)
        self.assertIsNone(orchestra.assignment_turns)
        self.assertIsNone(orchestra.agents['Router Agent'].search_feedback)
        self.assertEqual(orchestra.multiagent_batch_buffer, [])

    def test_finish_submits_worker_result_without_another_worker_call(self):
        self.make_orchestra(loops=8)
        self.script('Router Agent', [['<route>Sub Agent 1</route>'], ['<route>FINISH</route>']])
        self.script('Sub Agent 1', [[r'<think>Derivation</think>\boxed{42}<handoff>HANDOFF</handoff>']])
        actions, buffer = self.run_orchestra()
        self.assertEqual(actions, [r'\boxed{42}'])
        self.assertEqual([entry['agent_id'] for entry in buffer], ['Router Agent', 'Sub Agent 1', 'Router Agent'])
        self.assertEqual(buffer[-1]['batch'].non_tensor_batch['return_reason'].tolist(), ['finish'])

    def test_finish_and_continue_are_independent_per_sample(self):
        self.make_orchestra(loops=3)
        self.script('Router Agent', [
            ['<route>Sub Agent 1</route>', '<route>Sub Agent 2</route>'],
            ['<route>FINISH</route>', '<route>Sub Agent 2</route>'],
            ['', '<route>FINISH</route>'],
        ])
        self.script('Sub Agent 1', [[r'\boxed{1}<handoff>HANDOFF</handoff>', '']])
        self.script('Sub Agent 2', [
            ['', 'B1<handoff>CONTINUE</handoff>'],
            ['', r'\boxed{3}<handoff>CONTINUE</handoff>'],
        ])
        actions, _ = self.run_orchestra(active=(True, True))
        self.assertEqual(actions, [r'\boxed{1}', r'\boxed{3}'])

    def test_finish_without_result_is_invalid_and_retried(self):
        self.make_orchestra(loops=2)
        self.script('Router Agent', [['<route>FINISH</route>'], ['<route>Sub Agent 1</route>']])
        self.script('Sub Agent 1', [[r'\boxed{42}<handoff>CONTINUE</handoff>']])
        actions, buffer = self.run_orchestra()
        self.assertEqual(actions, [r'\boxed{42}'])
        self.assertFalse(buffer[0]['batch'].non_tensor_batch['is_action_valid'][0])
        self.assertIn('no candidate exists', self.prompts[1][1][0])

    def test_search_request_cannot_be_submitted_as_final_result(self):
        self.make_orchestra(loops=1, env_name='search')
        self.script('Router Agent', [['<route>Sub Agent 1</route>'], ['<route>FINISH</route>']])
        self.script('Sub Agent 1', [['<search>query</search><handoff>HANDOFF</handoff>']])
        self.run_orchestra()
        actions, buffer = self.run_orchestra(step=2, observation='tool result')
        self.assertEqual(actions, [''])
        self.assertFalse(buffer[0]['batch'].non_tensor_batch['is_action_valid'][0])

    def test_inactive_batch_does_not_call_agents(self):
        self.make_orchestra()
        actions, buffer = self.run_orchestra(active=(False, False))
        self.assertEqual(actions, ['', ''])
        self.assertEqual(buffer, [])

    def test_agent_work_is_retained_and_tool_history_comes_from_env_obs(self):
        orchestra = self.make_orchestra(loops=1, env_name='search')
        self.script('Router Agent', [['<route>Sub Agent 1</route>'], ['<route>Sub Agent 2</route>']])
        self.script('Sub Agent 1', [['Intermediate conclusion.<search>query</search><handoff>HANDOFF</handoff>']])
        self.script('Sub Agent 2', [['<answer>42</answer><handoff>HANDOFF</handoff>']])
        self.run_orchestra()
        actions, buffer = self.run_orchestra(
            step=2,
            observation='task\n<search>query</search>\n<information>tool evidence</information>',
            anchor=['raw tool result must not be appended'],
        )
        self.assertEqual(actions, ['<answer>42</answer>'])
        self.assertEqual([entry['agent_id'] for entry in buffer], ['Router Agent', 'Sub Agent 2'])
        for _, prompts in self.prompts[-2:]:
            self.assertIn('Intermediate conclusion.', prompts[0])
            self.assertEqual(prompts[0].count('tool evidence'), 1)
        self.assertIn('Intermediate conclusion.', orchestra.team_context[0])
        self.assertNotIn('tool evidence', orchestra.team_context[0])
        self.assertNotIn('raw tool result', orchestra.team_context[0])

    def test_three_subagents_are_distinct_and_all_can_be_selected(self):
        orchestra = self.make_orchestra(loops=2)
        self.assertEqual(len(orchestra.agents), 4)
        self.assertIsInstance(orchestra.agents['Router Agent'], RouterAgent)
        for name in ('Sub Agent 1', 'Sub Agent 2', 'Sub Agent 3'):
            self.assertIsInstance(orchestra.agents[name], SubAgent)
            self.assertEqual(orchestra.agents[name].name, name)
        self.assertIsNot(orchestra.agents['Sub Agent 1'], orchestra.agents['Sub Agent 3'])
        self.script('Router Agent', [
            ['<route>Sub Agent 1</route>', '<route>Sub Agent 2</route>', '<route>Sub Agent 3</route>'],
            ['<route>FINISH</route>'] * 3,
        ])
        self.script('Sub Agent 1', [[r'\boxed{1}<handoff>HANDOFF</handoff>', '', '']])
        self.script('Sub Agent 2', [['', r'\boxed{2}<handoff>HANDOFF</handoff>', '']])
        self.script('Sub Agent 3', [['', '', r'\boxed{3}<handoff>HANDOFF</handoff>']])
        actions, buffer = self.run_orchestra(active=(True, True, True))
        self.assertEqual(actions, [r'\boxed{1}', r'\boxed{2}', r'\boxed{3}'])
        self.assertEqual([entry['agent_id'] for entry in buffer],
                         ['Router Agent', 'Sub Agent 1', 'Sub Agent 2', 'Sub Agent 3', 'Router Agent'])

    def test_reset_starts_next_rollout_without_previous_work(self):
        orchestra = self.make_orchestra(loops=1)
        self.script('Router Agent', [['<route>Sub Agent 1</route>'], ['<route>Sub Agent 3</route>']])
        self.script('Sub Agent 1', [['old result<handoff>CONTINUE</handoff>']])
        self.script('Sub Agent 3', [[r'new result\boxed{42}<handoff>HANDOFF</handoff>']])
        self.run_orchestra()
        orchestra.reset()
        actions, _ = self.run_orchestra()
        self.assertEqual(self.events[-2][1], [''])
        self.assertEqual(actions, [r'\boxed{42}'])
        self.assertNotIn('old result', orchestra.team_context[0])

    def test_finish_can_submit_preserved_work_in_the_next_run(self):
        orchestra = self.make_orchestra(loops=1, env_name='search')
        self.script('Router Agent', [['<route>Sub Agent 3</route>'], ['<route>FINISH</route>']])
        self.script('Sub Agent 3', [['<answer>42</answer><handoff>HANDOFF</handoff>']])
        self.run_orchestra()
        actions, buffer = self.run_orchestra(step=2)
        self.assertEqual(actions, ['<answer>42</answer>'])
        self.assertEqual([entry['agent_id'] for entry in buffer], ['Router Agent'])
        self.assertIn('<answer>42</answer>', self.events[-1][1][0])

    def test_search_partial_handoff_routes_without_creating_candidate(self):
        orchestra = self.make_orchestra(loops=3, env_name='search')
        self.script('Router Agent', [
            ['<route>Sub Agent 1</route>'],
            ['<route>FINISH</route>'],
            ['<route>Sub Agent 2</route>'],
        ])
        partial = '<think>The sources agree; the final answer remains to be written.</think><handoff>HANDOFF</handoff>'
        self.script('Sub Agent 1', [[partial]])
        self.script('Sub Agent 2', [['<answer>toward the heart</answer><handoff>HANDOFF</handoff>']])
        actions, buffer = self.run_orchestra()
        self.assertEqual(actions, ['<answer>toward the heart</answer>'])
        self.assertEqual([entry['agent_id'] for entry in buffer],
                         ['Router Agent', 'Sub Agent 1', 'Router Agent', 'Router Agent', 'Sub Agent 2'])
        self.assertTrue(buffer[1]['batch'].non_tensor_batch['is_action_valid'][0])
        self.assertFalse(buffer[2]['batch'].non_tensor_batch['is_action_valid'][0])
        self.assertIn('no candidate exists', self.prompts[3][1][0])
        self.assertIn('<answer> block outside <think>', self.prompts[3][1][0])
        self.assertIn(partial, orchestra.team_context[0])
        self.assertIn('The sources agree', self.prompts[-1][1][0])
        self.assertEqual(orchestra.agents['Sub Agent 1'].search_feedback, [''])

    def test_search_partial_continue_returns_to_router_before_answer(self):
        self.make_orchestra(loops=5, env_name='search')
        self.script('Router Agent', [
            ['<route>Sub Agent 1</route>'],
            ['<route>Sub Agent 1</route>'],
            ['<route>FINISH</route>'],
        ])
        self.script('Sub Agent 1', [
            ['<think>Compare the sources before concluding.</think><handoff>CONTINUE</handoff>'],
            ['<answer>toward the heart</answer><handoff>HANDOFF</handoff>'],
        ])
        actions, buffer = self.run_orchestra()
        self.assertEqual(actions, ['<answer>toward the heart</answer>'])
        self.assertEqual([entry['agent_id'] for entry in buffer],
                         ['Router Agent', 'Sub Agent 1', 'Router Agent', 'Sub Agent 1', 'Router Agent'])
        self.assertTrue(all(entry['batch'].non_tensor_batch['is_action_valid'][0] for entry in buffer))

    def test_search_partial_work_preserves_existing_candidate(self):
        orchestra = self.make_orchestra(loops=3, env_name='search')
        self.script('Router Agent', [
            ['<route>Sub Agent 1</route>'], ['<route>Sub Agent 2</route>'], ['<route>FINISH</route>'],
        ])
        self.script('Sub Agent 1', [['<answer>42</answer><handoff>HANDOFF</handoff>']])
        partial = 'Checked the supporting source.<handoff>HANDOFF</handoff>'
        self.script('Sub Agent 2', [[partial]])
        actions, buffer = self.run_orchestra()
        self.assertEqual(actions, ['<answer>42</answer>'])
        self.assertEqual(orchestra.submission_actions, ['<answer>42</answer>'])
        self.assertIn(partial, orchestra.team_context[0])
        self.assertTrue(all(entry['batch'].non_tensor_batch['is_action_valid'][0] for entry in buffer))

    def test_search_malformed_action_can_be_repaired_with_partial_work(self):
        orchestra = self.make_orchestra(loops=4, env_name='search')
        self.script('Router Agent', [
            ['<route>Sub Agent 1</route>'], ['<route>Sub Agent 2</route>'], ['<route>FINISH</route>'],
        ])
        self.script('Sub Agent 1', [
            ['<answer>unfinished<handoff>HANDOFF</handoff>'],
            ['<think>Partial result; one inference remains.</think><handoff>HANDOFF</handoff>'],
        ])
        self.script('Sub Agent 2', [['<answer>42</answer><handoff>HANDOFF</handoff>']])
        actions, buffer = self.run_orchestra()
        self.assertEqual(actions, ['<answer>42</answer>'])
        self.assertEqual([entry['agent_id'] for entry in buffer],
                         ['Router Agent', 'Sub Agent 1', 'Sub Agent 1', 'Router Agent', 'Sub Agent 2', 'Router Agent'])
        self.assertFalse(buffer[1]['batch'].non_tensor_batch['is_action_valid'][0])
        self.assertTrue(buffer[2]['batch'].non_tensor_batch['is_action_valid'][0])
        retry_prompt = self.prompts[2][1][0]
        self.assertIn('# Correction to your previous response', retry_prompt)
        self.assertIn('Complete the search or answer tags', retry_prompt)
        self.assertEqual(orchestra.agents['Sub Agent 1'].search_feedback, [''])

    def test_search_final_step_retries_until_answer_and_handoff(self):
        invalid_responses = [
            '<think>Partial work.</think><handoff>HANDOFF</handoff>',
            'Partial work.<handoff>CONTINUE</handoff>',
            '<search>query</search><handoff>HANDOFF</handoff>',
            '<search>query</search><handoff>CONTINUE</handoff>',
            '<answer>42</answer><handoff>CONTINUE</handoff>',
            '<answer> </answer><handoff>HANDOFF</handoff>',
            '<answer>unfinished<handoff>HANDOFF</handoff>',
        ]
        for response in invalid_responses:
            with self.subTest(response=response):
                orchestra = self.make_orchestra(loops=2, env_name='search', max_steps=4)
                self.script('Router Agent', [['<route>Sub Agent 1</route>']])
                self.script('Sub Agent 1', [[response], ['<answer>42</answer><handoff>HANDOFF</handoff>']])
                actions, buffer = self.run_orchestra(step=4)
                self.assertEqual(actions, ['<answer>42</answer>'])
                self.assertEqual([entry['agent_id'] for entry in buffer],
                                 ['Router Agent', 'Sub Agent 1', 'Sub Agent 1'])
                self.assertFalse(buffer[1]['batch'].non_tensor_batch['is_action_valid'][0])
                self.assertTrue(buffer[2]['batch'].non_tensor_batch['is_action_valid'][0])
                self.assertIn('# Final answer required', self.prompts[1][1][0])
                self.assertIn('This is the final step: provide', self.prompts[2][1][0])
                self.assertEqual(orchestra.agents['Sub Agent 1'].search_feedback, [''])

    def test_search_protocol_repairs_missing_handoff_and_preserves_invalid_training_flag(self):
        orchestra = self.make_orchestra(loops=1, env_name='search')
        self.script('Router Agent', [['<Route>Sub Agent 1</Route>']])
        self.script('Sub Agent 1', [['- <search>number of species</search>']])
        actions, buffer = self.run_orchestra()
        self.assertEqual(actions, ['<search>number of species</search>'])
        worker_batch = buffer[-1]['batch']
        self.assertEqual(worker_batch.non_tensor_batch['handoff_decision'].tolist(), ['CONTINUE'])
        self.assertEqual(worker_batch.non_tensor_batch['protocol_repaired'].tolist(), [True])
        self.assertEqual(worker_batch.non_tensor_batch['is_action_valid'].tolist(), [False])
        self.assertNotIn('- <search>number of species</search>', orchestra.team_context[0])

    def test_router_protocol_repairs_trailing_explanation(self):
        orchestra = self.make_orchestra(loops=1, env_name='search')
        self.script('Router Agent', [['<route>Sub Agent 1</route>Let the worker search.']])
        self.script('Sub Agent 1', [['<answer>42</answer><handoff>HANDOFF</handoff>']])
        actions, buffer = self.run_orchestra()
        self.assertEqual(actions, ['<answer>42</answer>'])
        router_batch = buffer[0]['batch']
        self.assertEqual(router_batch.non_tensor_batch['selected_agent'].tolist(), ['Sub Agent 1'])
        self.assertEqual(router_batch.non_tensor_batch['protocol_repaired'].tolist(), [True])
        self.assertEqual(router_batch.non_tensor_batch['is_action_valid'].tolist(), [False])

    def test_math_partial_work_can_continue_and_handoff_before_boxed_answer(self):
        self.make_orchestra(loops=4, env_name='math', max_steps=1)
        self.script('Router Agent', [
            ['<route>Sub Agent 1</route>'], ['<route>Sub Agent 2</route>'], ['<route>FINISH</route>'],
        ])
        self.script('Sub Agent 1', [
            ['<think>First simplify the expression.</think><handoff>CONTINUE</handoff>'],
            ['Intermediate result: 2x = 1.<handoff>HANDOFF</handoff>'],
        ])
        final = r'\boxed{\frac{1}{2}}'
        self.script('Sub Agent 2', [[final + '<handoff>HANDOFF</handoff>']])
        actions, buffer = self.run_orchestra()
        self.assertEqual(actions, [final])
        self.assertEqual([entry['agent_id'] for entry in buffer],
                         ['Router Agent', 'Sub Agent 1', 'Router Agent', 'Sub Agent 2', 'Router Agent'])
        self.assertTrue(all(entry['batch'].non_tensor_batch['is_action_valid'][0] for entry in buffer))
        worker_prompt = self.prompts[1][1][0]
        self.assertIn(r'\boxed{...}', worker_prompt)
        self.assertNotIn(r'\boxed{{...}}', worker_prompt)
        self.assertNotIn('# Final answer required', worker_prompt)

    def test_math_candidates_require_a_complete_external_box(self):
        orchestra = self.make_orchestra()
        cases = [
            ('Intermediate result: 42.', ''),
            (r'<think>Try \boxed{42}.</think>Still checking.', ''),
            (r'\boxed{}', ''),
            (r'\boxed{   }', ''),
            (r'\boxed{\frac{1}{2}', ''),
            (r'\boxed{41} then \boxed{', ''),
            (r'\boxed{41} then \boxed', ''),
            (r'\boxed{\frac{1}{2}}', r'\boxed{\frac{1}{2}}'),
            (r'<think>Try \boxed{41}.</think>Final: \boxed{42}', r'\boxed{42}'),
            (r'\boxed{42}<think>Compare with \boxed{41}.</think>', r'\boxed{42}'),
            (r'\boxed{41} then \boxed{42}', r'\boxed{42}'),
        ]
        for response, expected in cases:
            with self.subTest(response=response):
                self.assertEqual(orchestra._environment_action(response + '<handoff>HANDOFF</handoff>'),
                                 (expected, False))

    def test_math_partial_work_cannot_finish_or_replace_a_candidate(self):
        orchestra = self.make_orchestra(loops=5)
        self.script('Router Agent', [
            ['<route>Sub Agent 1</route>'], ['<route>FINISH</route>'],
            ['<route>Sub Agent 2</route>'], ['<route>Sub Agent 1</route>'], ['<route>FINISH</route>'],
        ])
        self.script('Sub Agent 1', [
            [r'<think>Possible result: \boxed{41}.</think>Need verification.<handoff>HANDOFF</handoff>'],
            ['Checked the derivation.<handoff>HANDOFF</handoff>'],
        ])
        candidate = r'<think>Verified the equation.</think>\boxed{42}<handoff>HANDOFF</handoff>'
        self.script('Sub Agent 2', [[candidate]])
        actions, buffer = self.run_orchestra()
        self.assertEqual(actions, [r'\boxed{42}'])
        self.assertFalse(buffer[2]['batch'].non_tensor_batch['is_action_valid'][0])
        self.assertIn(r'\boxed{...} outside <think>', self.prompts[3][1][0])
        self.assertEqual(orchestra.candidate_authors, ['Sub Agent 2'])
        self.assertEqual(orchestra.candidate_responses, [candidate])
        self.assertEqual(orchestra.candidate_repeat_counts, [1])
        self.assertEqual(orchestra.return_reasons.tolist(), ['finish'])

    def test_candidate_counts_compare_text_and_preserve_source_across_partial_work(self):
        for env_name, first, second in [('math', r'\frac{1}{2}', r'\frac{2}{4}'), ('search', 'Alpha', 'Beta')]:
            with self.subTest(env_name=env_name):
                def answer(value):
                    return r'\boxed{' + value + '}' if env_name == 'math' else '<answer>' + value + '</answer>'

                orchestra = self.make_orchestra(loops=1, env_name=env_name)
                self.script('Router Agent', [
                    [f'<route>Sub Agent {i}</route>'] for i in (1, 2, 2, 2, 3, 1)
                ])
                first_response = answer(first) + '<handoff>HANDOFF</handoff>'
                repeated_response = '<think>Rechecked.</think>' + answer(' ' + first + ' ') + '<handoff>HANDOFF</handoff>'
                self.script('Sub Agent 1', [[first_response], [first_response]])
                self.script('Sub Agent 2', [
                    [repeated_response], ['Additional check.<handoff>HANDOFF</handoff>'],
                    [answer(second) + '<handoff>HANDOFF</handoff>'],
                ])
                self.script('Sub Agent 3', [[answer(second) + '<handoff>HANDOFF</handoff>']])
                snapshots = []
                for step, (value, count, author) in enumerate([
                    (first, 1, 'Sub Agent 1'), (first, 2, 'Sub Agent 2'), (first, 2, 'Sub Agent 2'),
                    (second, 1, 'Sub Agent 2'), (second, 2, 'Sub Agent 3'), (first, 1, 'Sub Agent 1'),
                ], 1):
                    _, buffer = self.run_orchestra(step=step)
                    self.assertEqual(orchestra.candidate_answers, [value])
                    self.assertEqual(orchestra.candidate_repeat_counts, [count])
                    self.assertEqual(orchestra.candidate_authors, [author])
                    snapshots.append(buffer[-1]['batch'])
                    if step == 3:
                        self.assertEqual(orchestra.candidate_responses, [repeated_response])
                self.assertEqual([row.non_tensor_batch['candidate_repeat_count'][0] for row in snapshots],
                                 [1, 2, 2, 1, 2, 1])
                self.assertEqual(snapshots[0].non_tensor_batch['candidate_author'].tolist(), ['Sub Agent 1'])

    def test_reselecting_worker_updates_instruction_and_bare_route_clears_it(self):
        for env_name, candidate in [('math', r'\boxed{42}'), ('search', '<answer>42</answer>')]:
            with self.subTest(env_name=env_name):
                orchestra = self.make_orchestra(loops=4, env_name=env_name)
                self.script('Router Agent', [
                    ['<think>Complete the remaining work.</think><route>Sub Agent 1</route>'],
                    ['<think>Check the missing qualifier.</think><route>Sub Agent 1</route>'],
                    ['<route>Sub Agent 1</route>'], ['<route>FINISH</route>'],
                ])
                self.script('Sub Agent 1', [[candidate + '<handoff>HANDOFF</handoff>']] * 3)
                actions, buffer = self.run_orchestra()
                self.assertEqual(actions, [candidate])
                workers = [entry['batch'] for entry in buffer if entry['agent_id'] == 'Sub Agent 1']
                self.assertEqual([row.non_tensor_batch['routing_instruction'][0] for row in workers],
                                 ['Complete the remaining work.', 'Check the missing qualifier.', ''])
                self.assertIn('Check the missing qualifier.', self.prompts[3][1][0])
                self.assertEqual(orchestra.routed_agents, ['Sub Agent 1'])
                self.assertEqual(orchestra.routing_instructions, [''])
                self.assertEqual(orchestra.candidate_repeat_counts, [3])
                self.assertTrue(all(entry['batch'].non_tensor_batch['is_action_valid'][0] for entry in buffer))

    def test_invalid_route_does_not_replace_last_valid_assignment(self):
        orchestra = self.make_orchestra(loops=1)
        self.script('Router Agent', [
            ['<think>Check the derivation.</think><route>Sub Agent 1</route>'],
            ['<think>Discard this invalid assignment.</think><route>Unknown</route>'],
            ['<route>Sub Agent 2</route>'],
        ])
        self.script('Sub Agent 1', [['Partial work.<handoff>HANDOFF</handoff>']])
        self.script('Sub Agent 2', [['Checked one step.<handoff>HANDOFF</handoff>']])
        self.run_orchestra()
        _, buffer = self.run_orchestra(step=2)
        self.assertEqual(orchestra.routed_agents, ['Sub Agent 1'])
        self.assertEqual(orchestra.routing_instructions, ['Check the derivation.'])
        self.assertEqual(buffer[0]['batch'].non_tensor_batch['routing_instruction'].tolist(), [''])
        self.run_orchestra(step=3)
        self.assertIn('Use a worker name from the available list', self.prompts[-2][1][0])
        self.assertEqual(orchestra.routed_agents, ['Sub Agent 2'])
        self.assertEqual(orchestra.routing_instructions, [''])
        self.assertEqual(orchestra.agents['Router Agent'].search_feedback, [''])

    def test_assignment_survives_search_and_worker_continue(self):
        orchestra = self.make_orchestra(loops=3, env_name='search')
        self.script('Router Agent', [
            ['<think>Find and verify the source.</think><route>Sub Agent 1</route>'],
            ['<think>Find and verify the source.</think><route>Sub Agent 1</route>'],
            ['<route>FINISH</route>'],
        ])
        self.script('Sub Agent 1', [
            ['<search>query</search><handoff>CONTINUE</handoff>'],
            ['<answer>Alpha</answer><handoff>HANDOFF</handoff>'],
        ])
        _, buffer = self.run_orchestra()
        first_worker = buffer[-1]['batch']
        self.assertEqual(first_worker.non_tensor_batch['return_reason'].tolist(), ['search'])
        actions, buffer = self.run_orchestra(step=2)
        self.assertEqual(actions, ['<answer>Alpha</answer>'])
        self.assertEqual([entry['agent_id'] for entry in buffer], ['Router Agent', 'Sub Agent 1', 'Router Agent'])
        worker_entries = [entry for entry in buffer if entry['agent_id'] == 'Sub Agent 1']
        for entry in worker_entries:
            self.assertEqual(entry['batch'].non_tensor_batch['routing_instruction'].tolist(), ['Find and verify the source.'])
        self.assertEqual(first_worker.non_tensor_batch['candidate_repeat_count'].tolist(), [0])
        self.assertEqual(orchestra.candidate_repeat_counts, [1])

    def test_return_reasons_and_state_are_independent_per_sample(self):
        orchestra = self.make_orchestra(loops=2, env_name='search')
        self.script('Router Agent', [
            [f'<think>Task {i}.</think><route>Sub Agent {i}</route>' for i in (1, 2, 3)] + ['<route>Sub Agent 1</route>'],
            ['<route>FINISH</route>', '', '', ''],
        ])
        self.script('Sub Agent 1', [['<answer>Alpha</answer><handoff>HANDOFF</handoff>', '', '', '']])
        self.script('Sub Agent 2', [['', '<search>query</search><handoff>HANDOFF</handoff>', '', '']])
        self.script('Sub Agent 3', [
            ['', '', 'Partial work.<handoff>CONTINUE</handoff>', ''],
            ['', '', 'Further work.<handoff>CONTINUE</handoff>', ''],
        ])
        actions, buffer = self.run_orchestra(active=(True, True, True, False))
        self.assertEqual(actions, ['<answer>Alpha</answer>', '<search>query</search>', '', ''])
        self.assertEqual(orchestra.return_reasons.tolist(), ['finish', 'search', 'loop_limit', ''])
        self.assertEqual(orchestra.candidate_authors, ['Sub Agent 1', '', '', ''])
        self.assertEqual(orchestra.candidate_repeat_counts, [1, 0, 0, 0])
        self.assertEqual(orchestra.routing_instructions, ['Task 1.', 'Task 2.', 'Task 3.', ''])
        schemas = [set(entry['batch'].non_tensor_batch) for entry in buffer]
        self.assertTrue(all(keys == schemas[0] for keys in schemas))
        for i, expected in enumerate(['finish', 'search', 'loop_limit', '']):
            reasons = [entry['batch'].non_tensor_batch['return_reason'][i] for entry in buffer
                       if entry['batch'].non_tensor_batch['return_reason'][i]]
            self.assertEqual(reasons, [expected] if expected else [])
        for entry in buffer:
            row = entry['batch'].non_tensor_batch
            for i, active in enumerate(row['agent_active_mask']):
                if not active:
                    self.assertEqual(row['candidate_author'][i], '')
                    self.assertEqual(row['routing_instruction'][i], '')
                    self.assertEqual(row['return_reason'][i], '')

    def test_loop_limit_fallback_is_distinct_from_finish_and_inactive_run(self):
        for env_name, candidate in [('math', r'\boxed{42}'), ('search', '<answer>42</answer>')]:
            with self.subTest(env_name=env_name):
                orchestra = self.make_orchestra(loops=1, env_name=env_name)
                self.script('Router Agent', [['<route>Sub Agent 1</route>'], ['<route>FINISH</route>']])
                self.script('Sub Agent 1', [[candidate + '<handoff>HANDOFF</handoff>']])
                actions, buffer = self.run_orchestra()
                self.assertEqual(actions, [candidate])
                first_worker = buffer[-1]['batch']
                self.assertEqual(first_worker.non_tensor_batch['return_reason'].tolist(), ['loop_limit'])
                actions, buffer = self.run_orchestra(step=2)
                self.assertEqual(actions, [candidate])
                self.assertEqual([entry['agent_id'] for entry in buffer], ['Router Agent'])
                self.assertEqual(buffer[0]['batch'].non_tensor_batch['return_reason'].tolist(), ['finish'])
                self.assertEqual(first_worker.non_tensor_batch['return_reason'].tolist(), ['loop_limit'])
                self.assertEqual(orchestra.candidate_repeat_counts, [1])
                actions, buffer = self.run_orchestra(active=(False,), step=3)
                self.assertEqual(actions, [''])
                self.assertEqual(buffer, [])
                self.assertEqual(orchestra.return_reasons.tolist(), [''])

    def test_search_action_separates_intermediate_work_from_external_actions(self):
        cases = [
            ('Intermediate findings.', ('', False, '')),
            ('<think>More work remains.</think>', ('', False, '')),
            ('<think>Consider <answer>42</answer> after <search>query</search>.</think>', ('', False, '')),
            ('<think>Need a fact.</think><search> query </search>', ('<search>query</search>', True, '')),
            ('<think>Conclusion reached.</think><answer> toward the heart </answer>',
             ('<answer>toward the heart</answer>', False, '')),
            ('<answer>yes</answer>', ('<answer>yes</answer>', False, '')),
            ('<answer>no</answer>', ('<answer>no</answer>', False, '')),
        ]
        for work, expected in cases:
            for decision in ('CONTINUE', 'HANDOFF'):
                with self.subTest(work=work, decision=decision):
                    self.assertEqual(SubAgent.search_action(work + f'<handoff>{decision}</handoff>'), expected)

    def test_search_action_rejects_empty_malformed_and_multiple_actions(self):
        invalid_work = [
            '', '   ', '<think> </think>',
            '<search> </search>', '<answer></answer>',
            '<answer>unfinished', '<search', '</answer>', '<answer>mismatched</search>',
            '<search>query</search><answer>42</answer>',
            '<answer>41</answer><answer>42</answer>',
            '<answer><answer>42</answer></answer>',
            '<answer><search>query</search></answer>',
            '<answer>42</answer><search>unfinished',
            '- <search>query</search><handoff>CONTINUE</handoff>',
            '<search>query</search> extra explanation<handoff>CONTINUE</handoff>',
        ]
        for work in invalid_work:
            with self.subTest(work=work):
                action, is_search, error = SubAgent.search_action(work + '<handoff>HANDOFF</handoff>')
                self.assertEqual(action, '')
                self.assertFalse(is_search)
                self.assertTrue(error)

    def test_control_parser_is_independent_of_compatibility_tags(self):
        for tag in ('search', 'answer'):
            for decision in ('CONTINUE', 'HANDOFF'):
                text = f'<{tag}>content</{tag}><handoff>{decision}</handoff>'
                self.assertEqual(SubAgent.parse_response(text), decision)
        for text in ('', '<handoff>HANDOFF</handoff>', '<think> </think><handoff>CONTINUE</handoff>',
                     '<answer>42</answer>', '<handoff>HANDOFF',
                     '<handoff>CONTINUE</handoff><handoff>HANDOFF</handoff>',
                     '<think><think>nested</think><handoff>HANDOFF</handoff>',
                     '<think>unfinished<handoff>HANDOFF</handoff>'):
            self.assertEqual(SubAgent.parse_response(text), 'invalid')
        self.assertEqual(SubAgent.parse_response(
            '<think>Quoted control: <handoff>HANDOFF</handoff></think><handoff>CONTINUE</handoff>'
        ), 'CONTINUE')


if __name__ == '__main__':
    unittest.main()
