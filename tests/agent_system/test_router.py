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
        router = load('agent_system.agent.agents.router.router_agent', 'agent_system/agent/agents/router/router_agent.py')
        worker = load('agent_system.agent.agents.router.subagent', 'agent_system/agent/agents/router/subagent.py')
        stubs['agent_system.agent.agents.router'].RouterAgent = router.RouterAgent
        stubs['agent_system.agent.agents.router'].SubAgent = worker.SubAgent
        orchestra = load('agent_system.agent.orchestra.router.router_orchestra', 'agent_system/agent/orchestra/router/router_orchestra.py')
    return router.RouterAgent, worker.SubAgent, orchestra.RouterMultiAgentOrchestra


RouterAgent, SubAgent, RouterOrchestra = load_classes()


def config(loops=3, env_name='math'):
    return SimpleNamespace(
        agent=SimpleNamespace(
            agent_ids=['Router Agent', 'Sub Agent 1', 'Sub Agent 2', 'Sub Agent 3'],
            orchestra=SimpleNamespace(router=SimpleNamespace(max_loop_num=loops)),
        ),
        env=SimpleNamespace(env_name=env_name),
    )


class RouterTests(unittest.TestCase):
    def make_orchestra(self, loops=3, env_name='math'):
        cfg = config(loops, env_name)
        names = cfg.agent.agent_ids
        self.groups = {name: object() for name in names}
        self.events = []
        self.prompts = []
        self.orchestra = RouterOrchestra(
            names, ['model'] * len(names), {name: name for name in names},
            {name: None for name in names}, {name: None for name in names}, cfg,
        )
        return self.orchestra

    def script(self, name, responses):
        agent = self.orchestra.agents[name]
        queue = iter(responses)
        build_prompt = agent.build_prompt

        def capture(env_obs, team_context, step):
            self.events.append((name, list(team_context), step))
            obs = build_prompt(env_obs, team_context, step)
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
        ]
        for response, expected in cases:
            with self.subTest(response=response):
                self.assertEqual(router.parse_response(response), expected)
        obs = {'text': ['task']}
        prompt = router.build_prompt(obs, ['shared work'], 2)['text'][0]
        self.assertIn('- Sub Agent 1', prompt)
        self.assertIn('- Sub Agent 2', prompt)
        self.assertIn('- Sub Agent 3', prompt)
        self.assertIn('shared work', prompt)
        self.assertEqual(obs, {'text': ['task']})

    def test_named_workers_and_independent_batch_routing(self):
        orchestra = self.make_orchestra(loops=2)
        self.assertIsInstance(orchestra.agents['Sub Agent 1'], SubAgent)
        self.assertEqual(orchestra.agents['Sub Agent 1'].name, 'Sub Agent 1')
        self.script('Router Agent', [
            ['<route>Sub Agent 1</route>', '<route>Sub Agent 2</route>', '<route>Sub Agent 1</route>'],
            ['', '<route>Sub Agent 1</route>', ''],
        ])
        self.script('Sub Agent 1', [
            ['<think>A reasoning</think>A1<handoff>CONTINUE</handoff>', '', ''],
            ['A2<handoff>HANDOFF</handoff>', 'A for sample 2<handoff>CONTINUE</handoff>', ''],
        ])
        self.script('Sub Agent 2', [['', 'B1<handoff>HANDOFF</handoff>', '']])
        actions, buffer = self.run_orchestra(active=(True, True, False))
        self.assertEqual(actions, ['A2', 'A for sample 2', ''])
        self.assertEqual([entry['agent_id'] for entry in buffer],
                         ['Router Agent', 'Sub Agent 1', 'Sub Agent 2', 'Router Agent', 'Sub Agent 1'])
        self.assertEqual(buffer[3]['batch'].non_tensor_batch['agent_active_mask'].tolist(), [False, True, False])
        self.assertEqual(buffer[0]['batch'].non_tensor_batch['selected_agent'].tolist(), ['Sub Agent 1', 'Sub Agent 2', ''])
        self.assertIn('<think>A reasoning</think>', orchestra.team_context[0])
        self.assertIn('<handoff>CONTINUE</handoff>', orchestra.team_context[0])
        self.assertNotIn('B1', orchestra.team_context[0])
        self.assertIn('B1', self.events[-1][1][1])
        self.assertEqual(orchestra.team_context[2], '')

    def test_search_yields_and_continue_resumes_same_worker(self):
        orchestra = self.make_orchestra(loops=2, env_name='search')
        self.script('Router Agent', [['<route>Sub Agent 1</route>']])
        self.script('Sub Agent 1', [
            ['<think>Need a tool.</think><search>query</search><handoff>CONTINUE</handoff>'],
            ['<answer>first answer</answer><handoff>CONTINUE</handoff>'],
            ['<answer>revised answer</answer><handoff>HANDOFF</handoff>'],
        ])
        actions, buffer = self.run_orchestra()
        self.assertEqual(actions, ['<search>query</search>'])
        self.assertEqual(len(buffer), 2)
        self.assertEqual(orchestra.current_agents.tolist(), ['Sub Agent 1'])
        actions, buffer = self.run_orchestra(step=2, observation='task and tool result')
        self.assertEqual(actions, ['<answer>revised answer</answer>'])
        self.assertEqual([entry['agent_id'] for entry in buffer], ['Sub Agent 1', 'Sub Agent 1'])
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
        self.assertEqual(actions, ['work'])
        self.assertEqual([entry['agent_id'] for entry in buffer], ['Router Agent', 'Router Agent', 'Sub Agent 2'])
        self.assertFalse(buffer[0]['batch'].non_tensor_batch['is_action_valid'][0])

    def test_invalid_worker_output_keeps_worker_and_is_not_submitted(self):
        self.make_orchestra(loops=2)
        self.script('Router Agent', [['<route>Sub Agent 1</route>']])
        self.script('Sub Agent 1', [['<answer>unfinished'], ['valid work<handoff>CONTINUE</handoff>']])
        actions, buffer = self.run_orchestra()
        self.assertEqual(actions, ['valid work'])
        self.assertEqual([entry['agent_id'] for entry in buffer], ['Router Agent', 'Sub Agent 1', 'Sub Agent 1'])

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
        self.assertEqual(orchestra.multiagent_batch_buffer, [])

    def test_finish_submits_worker_result_without_another_worker_call(self):
        self.make_orchestra(loops=8)
        self.script('Router Agent', [['<route>Sub Agent 1</route>'], ['<route>FINISH</route>']])
        self.script('Sub Agent 1', [['<think>Derivation</think>Result: 42<handoff>HANDOFF</handoff>']])
        actions, buffer = self.run_orchestra()
        self.assertEqual(actions, ['<think>Derivation</think>Result: 42'])
        self.assertEqual([entry['agent_id'] for entry in buffer], ['Router Agent', 'Sub Agent 1', 'Router Agent'])

    def test_finish_and_continue_are_independent_per_sample(self):
        self.make_orchestra(loops=3)
        self.script('Router Agent', [
            ['<route>Sub Agent 1</route>', '<route>Sub Agent 2</route>'],
            ['<route>FINISH</route>', ''],
        ])
        self.script('Sub Agent 1', [['A result<handoff>HANDOFF</handoff>', '']])
        self.script('Sub Agent 2', [
            ['', 'B1<handoff>CONTINUE</handoff>'],
            ['', 'B2<handoff>CONTINUE</handoff>'],
            ['', 'B3<handoff>CONTINUE</handoff>'],
        ])
        actions, _ = self.run_orchestra(active=(True, True))
        self.assertEqual(actions, ['A result', 'B3'])

    def test_finish_without_result_is_invalid_and_retried(self):
        self.make_orchestra(loops=2)
        self.script('Router Agent', [['<route>FINISH</route>'], ['<route>Sub Agent 1</route>']])
        self.script('Sub Agent 1', [['result<handoff>CONTINUE</handoff>']])
        actions, buffer = self.run_orchestra()
        self.assertEqual(actions, ['result'])
        self.assertFalse(buffer[0]['batch'].non_tensor_batch['is_action_valid'][0])

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
        self.script('Sub Agent 1', [['result 1<handoff>HANDOFF</handoff>', '', '']])
        self.script('Sub Agent 2', [['', 'result 2<handoff>HANDOFF</handoff>', '']])
        self.script('Sub Agent 3', [['', '', 'result 3<handoff>HANDOFF</handoff>']])
        actions, buffer = self.run_orchestra(active=(True, True, True))
        self.assertEqual(actions, ['result 1', 'result 2', 'result 3'])
        self.assertEqual([entry['agent_id'] for entry in buffer],
                         ['Router Agent', 'Sub Agent 1', 'Sub Agent 2', 'Sub Agent 3', 'Router Agent'])

    def test_reset_starts_next_rollout_without_previous_work(self):
        orchestra = self.make_orchestra(loops=1)
        self.script('Router Agent', [['<route>Sub Agent 1</route>'], ['<route>Sub Agent 3</route>']])
        self.script('Sub Agent 1', [['old result<handoff>CONTINUE</handoff>']])
        self.script('Sub Agent 3', [['new result<handoff>HANDOFF</handoff>']])
        self.run_orchestra()
        orchestra.reset()
        actions, _ = self.run_orchestra()
        self.assertEqual(self.events[-2][1], [''])
        self.assertEqual(actions, ['new result'])
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

    def test_control_parser_is_independent_of_compatibility_tags(self):
        for tag in ('search', 'answer'):
            for decision in ('CONTINUE', 'HANDOFF'):
                text = f'<{tag}>content</{tag}><handoff>{decision}</handoff>'
                self.assertEqual(SubAgent.parse_response(text), decision)
        for text in ('<answer>42</answer>', '<handoff>HANDOFF',
                     '<handoff>CONTINUE</handoff><handoff>HANDOFF</handoff>',
                     '<think>unfinished<handoff>HANDOFF</handoff>'):
            self.assertEqual(SubAgent.parse_response(text), 'invalid')


if __name__ == '__main__':
    unittest.main()
