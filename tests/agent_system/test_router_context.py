"""Context integration tests using recorded work and a deterministic tokenizer."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

import test_router as router_fixture
from test_router import Context, FakeBatch, RouterAgent, SubAgent, config


class ByteTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode('utf-8'))

    def decode(self, tokens, skip_special_tokens=False):
        return bytes(tokens).decode('utf-8', errors='replace')

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
        return ''.join(f'<{message["role"]}>{message["content"]}</{message["role"]}>' for message in messages) + '<assistant>'


def section(prompt, title):
    return prompt.split(title + '\n', 1)[1].split('\n\n# ', 1)[0]


class RouterContextTests(unittest.TestCase):
    make_orchestra = router_fixture.RouterTests.make_orchestra
    script = router_fixture.RouterTests.script
    run_orchestra = router_fixture.RouterTests.run_orchestra

    def test_saved_candidate_and_source_survive_invalid_answer_and_partial_work(self):
        for env_name, good, bad in [('math', r'\boxed{42}', r'\boxed{99}'),
                                    ('search', '<answer>Alpha</answer>', '<answer>Wrong</answer>')]:
            with self.subTest(env_name=env_name):
                orchestra = self.make_orchestra(loops=4, env_name=env_name)
                self.script('Router Agent', [
                    ['<route>Sub Agent 1</route>'], ['<route>Sub Agent 2</route>'], ['<route>FINISH</route>'],
                ])
                self.script('Sub Agent 1', [['<think>SOURCE_REASONING supports the result.</think>'
                                           + good + '<handoff>HANDOFF</handoff>']])
                self.script('Sub Agent 2', [[bad], ['Further checking remains.<handoff>HANDOFF</handoff>']])
                actions, _ = self.run_orchestra()
                self.assertEqual(actions, [good])
                review_prompt = self.prompts[-1][1][0]
                candidate = section(review_prompt, '# Candidate answer (awaiting review)')
                support = section(review_prompt, '# Work accompanying the candidate (source excerpts, unverified)')
                self.assertIn(good, candidate)
                self.assertNotIn(bad, candidate)
                self.assertIn('Author: Sub Agent 1; turn: 2; step: 1', candidate)
                self.assertIn('Sub Agent 1', support)
                self.assertIn('SOURCE_REASONING', support)
                self.assertNotIn('protocol invalid', review_prompt)
                self.assertNotIn(bad, orchestra.team_context[0])
                self.assertEqual(orchestra.candidate_records[0].turn, 2)
                self.assertFalse(orchestra.work_records[0][3].valid)

    def test_invalid_answer_without_a_saved_candidate_is_only_historical_work(self):
        self.make_orchestra(loops=3, env_name='search')
        self.script('Router Agent', [['<route>Sub Agent 1</route>'], ['<route>Sub Agent 2</route>']])
        self.script('Sub Agent 1', [
            ['<answer>INVALID_CANDIDATE</answer>'], ['Partial progress.<handoff>HANDOFF</handoff>'],
        ])
        self.script('Sub Agent 2', [['<answer>Alpha</answer><handoff>HANDOFF</handoff>']])
        self.run_orchestra()
        review_prompt = self.prompts[3][1][0]
        candidate = section(review_prompt, '# Candidate answer (awaiting review)')
        self.assertEqual(candidate, 'No candidate answer has been saved.')
        self.assertNotIn('protocol invalid', review_prompt)
        self.assertNotIn('INVALID_CANDIDATE', candidate)

    def test_invalid_worker_output_is_kept_for_labels_but_hidden_from_next_prompt(self):
        orchestra = self.make_orchestra(loops=2, env_name='search')
        malformed = '<think>Need a result.</think>\n- <search>query</search><answer>42</answer><handoff>HANDOFF</handoff>'
        self.script('Router Agent', [['<route>Sub Agent 1</route>']])
        self.script('Sub Agent 1', [[malformed], ['Checked the derivation.<handoff>CONTINUE</handoff>']])

        actions, _ = self.run_orchestra()

        self.assertEqual(actions, [''])
        self.assertIn(malformed, orchestra.work_records[0][1].response)
        self.assertFalse(orchestra.work_records[0][1].valid)
        retry_prompt = self.prompts[2][1][0]
        self.assertIn('# Correction to your previous response', retry_prompt)
        self.assertNotIn(malformed, retry_prompt)
        self.assertNotIn(malformed, orchestra.team_context[0])

    def test_current_assignment_is_replaced_and_missing_instruction_is_explicit(self):
        for env_name, answer in [('math', r'\boxed{42}'), ('search', '<answer>42</answer>')]:
            with self.subTest(env_name=env_name):
                self.make_orchestra(loops=4, env_name=env_name)
                self.script('Router Agent', [
                    ['<think>OLD_TASK.</think><route>Sub Agent 1</route>'],
                    ['<think>NEW_TASK.</think><route>Sub Agent 1</route>'],
                    ['<route>Sub Agent 1</route>'], ['<route>FINISH</route>'],
                ])
                self.script('Sub Agent 1', [[answer + '<handoff>HANDOFF</handoff>']] * 3)
                self.run_orchestra()
                worker_prompts = [prompts[0] for name, prompts in self.prompts if name == 'Sub Agent 1']
                current = section(worker_prompts[1], '# Current assignment')
                self.assertIn('NEW_TASK', current)
                self.assertNotIn('OLD_TASK', current)
                current = section(worker_prompts[2], '# Current assignment')
                self.assertIn('No specific task instruction was provided', current)
                self.assertNotIn('OLD_TASK', current)
                self.assertNotIn('NEW_TASK', current)
                router_prompt = self.prompts[2][1][0]
                self.assertIn('OLD_TASK', section(router_prompt, '# Previous assignment (historical)'))

    def make_long_context(self, env_name):
        action = r'\boxed{\frac{1}{2}}' if env_name == 'math' else '<answer>toward the heart</answer>'
        records = tuple(Context.WorkRecord(i, 1, 'Sub Agent 2', 'intermediate', True,
                                          f'WORK_START_{i} ' + 'x ' * 2000 + f' WORK_END_{i}') for i in range(1, 21))
        support = 'SUPPORT_START ' + 'reasoning ' * 1000 + ' SUPPORT_END'
        candidate_record = Context.WorkRecord(21, 1, 'Sub Agent 1', 'answer', True,
                                             '<think>' + support + '</think>' + action + '<handoff>HANDOFF</handoff>')
        assignment = 'VERIFY_THE_RESULT ' + 'with the recorded information. ' * 8
        route = Context.WorkRecord(22, 1, 'Router Agent', 'route', True,
                                   '<think>' + assignment + '</think><route>Sub Agent 2</route>')
        state = Context.CollaborationContext(action, candidate_record, 3, 'Sub Agent 2', assignment, 22,
                                             records + (candidate_record, route), 3, 8)
        return state

    def test_long_math_and_search_context_preserves_core_and_source_excerpts(self):
        for env_name in ('math', 'search'):
            with self.subTest(env_name=env_name):
                cfg = config(env_name=env_name, max_steps=4)
                cfg.data = SimpleNamespace(max_prompt_length=6500)
                tokenizer = ByteTokenizer()
                worker = SubAgent('wg', tokenizer, None, cfg, name='Sub Agent 2')
                state = self.make_long_context(env_name)
                question = 'ORIGINAL_QUESTION with {braces} and 中文.'
                observation = question
                if env_name == 'search':
                    observation = ('The question is: ' + question + '\nYour team can access search.\nHistory:\n'
                                   '<search>recorded query</search><information>EVIDENCE_START '
                                   + 'document text ' * 1500 + ' EVIDENCE_END</information>')
                obs = {'text': [observation], 'anchor': ['untouched']}
                prompt = worker.build_prompt(obs, ['RAW_HISTORY_SENTINEL'], 2,
                                             collaboration_context=[state])['text'][0]
                self.assertLessEqual(len(tokenizer.encode(prompt)), cfg.data.max_prompt_length - 256)
                self.assertIn(question, prompt)
                self.assertIn(state.routing_instruction, section(prompt, '# Current assignment'))
                self.assertIn(state.candidate_action, section(prompt, '# Candidate answer (awaiting review)'))
                self.assertIn('Author: Sub Agent 1; turn: 21; step: 1', prompt)
                self.assertIn('Consecutive occurrences of this answer text: 3', prompt)
                self.assertIn('SUPPORT_START', prompt)
                self.assertIn('SUPPORT_END', prompt)
                self.assertIn('WORK_START_20', prompt)
                self.assertIn('WORK_END_20', prompt)
                self.assertIn('omitted', prompt)
                self.assertNotIn('RAW_HISTORY_SENTINEL', prompt)
                self.assertEqual(obs['text'], [observation])
                self.assertEqual(obs['anchor'], ['untouched'])
                if env_name == 'search':
                    self.assertIn('Query 1', prompt)
                    self.assertIn('EVIDENCE_START', prompt)
                    self.assertIn('EVIDENCE_END', prompt)
                else:
                    self.assertNotIn('# Retrieved evidence', prompt)

    def test_oversized_core_is_rejected_instead_of_truncated(self):
        for env_name in ('math', 'search'):
            for oversized in ('question', 'candidate', 'assignment', 'feedback'):
                with self.subTest(env_name=env_name, oversized=oversized):
                    cfg = config(env_name=env_name)
                    cfg.data = SimpleNamespace(max_prompt_length=4096)
                    worker = SubAgent('wg', ByteTokenizer(), None, cfg, name='Sub Agent 2')
                    state = self.make_long_context(env_name)
                    question = 'task'
                    if oversized == 'question':
                        question = 'Q' * 6000
                    elif oversized == 'candidate':
                        state = replace(state, candidate_action='<answer>' + 'A' * 6000 + '</answer>')
                    elif oversized == 'assignment':
                        state = replace(state, routing_instruction='T' * 6000)
                    else:
                        worker.search_feedback = ['F' * 6000]
                    with self.assertRaisesRegex(ValueError, 'Core collaboration context exceeds'):
                        worker.build_prompt({'text': [question]}, [''], 1, collaboration_context=[state])

    def test_prompt_and_snapshot_isolation_for_batched_samples(self):
        orchestra = self.make_orchestra(loops=2, env_name='search')
        self.script('Router Agent', [
            ['<think>ALPHA_TASK.</think><route>Sub Agent 1</route>',
             '<think>BETA_TASK.</think><route>Sub Agent 2</route>', ''],
            ['<route>FINISH</route>', '<route>FINISH</route>', ''],
        ])
        self.script('Sub Agent 1', [[
            '<think>ALPHA_REASON.</think><answer>ALPHA_ANSWER</answer><handoff>HANDOFF</handoff>', '', '',
        ]])
        self.script('Sub Agent 2', [['',
            '<think>BETA_REASON.</think><answer>BETA_ANSWER</answer><handoff>HANDOFF</handoff>', '',
        ]])
        self.run_orchestra(active=(True, True, False))
        for _, prompts in self.prompts:
            self.assertNotIn('BETA_', prompts[0])
            self.assertNotIn('ALPHA_', prompts[1])
            self.assertEqual(prompts[2], '')
        initial = self.contexts[0][1]
        self.assertEqual(initial[0].records, ())
        self.assertEqual(initial[0].candidate_action, '')
        self.assertIsNone(initial[2])
        final = self.contexts[-1][1]
        self.assertEqual(final[0].candidate_action, '<answer>ALPHA_ANSWER</answer>')
        self.assertEqual(final[1].candidate_action, '<answer>BETA_ANSWER</answer>')
        self.assertEqual([record.turn for record in orchestra.work_records[0]], [1, 2, 3])
        self.assertEqual(len(final[0].records), 2)
        self.assertEqual(orchestra.work_records[2], [])

    def test_environment_and_orchestration_budgets_are_separate_across_search(self):
        self.make_orchestra(loops=2, env_name='search', max_steps=4)
        self.script('Router Agent', [
            ['<think>Find and verify the source.</think><route>Sub Agent 1</route>'],
            ['<think>Review the retrieved source.</think><route>Sub Agent 1</route>'],
            ['<route>FINISH</route>'],
        ])
        self.script('Sub Agent 1', [
            ['<search>query</search><handoff>CONTINUE</handoff>'],
            ['<answer>Alpha</answer><handoff>HANDOFF</handoff>'],
        ])
        self.run_orchestra()
        first = self.prompts[-1][1][0]
        self.run_orchestra(step=2, observation='The question is: task\nYour team can access search.\nHistory:\n'
                                                 '<search>query</search><information>source</information>')
        resumed = self.prompts[-2][1][0]
        reviewed = self.prompts[-1][1][0]
        self.assertIn('Environment step 1 of 4', first)
        self.assertIn('Environment step 2 of 4', resumed)
        self.assertIn('Round 1 of 2', first)
        self.assertIn('Round 1 of 2', resumed)
        self.assertIn('Round 2 of 2', reviewed)
        self.assertIn('Review the retrieved source.', section(resumed, '# Current assignment'))
        self.assertIn('Query 1', resumed)

    def test_reset_clears_context_sources_assignments_and_token_cache(self):
        orchestra = self.make_orchestra(loops=1, env_name='search')
        self.script('Router Agent', [
            ['<think>OLD_TASK.</think><route>Sub Agent 1</route>'], ['<route>Sub Agent 1</route>'],
        ])
        self.script('Sub Agent 1', [
            ['<answer>OLD_ANSWER</answer><handoff>HANDOFF</handoff>'],
            ['New partial work.<handoff>HANDOFF</handoff>'],
        ])
        self.run_orchestra()
        self.assertIsNotNone(orchestra.agents['Router Agent']._context_token_cache)
        orchestra.reset()
        self.assertIsNone(orchestra.work_records)
        self.assertIsNone(orchestra.candidate_records)
        self.assertIsNone(orchestra.agents['Router Agent']._context_token_cache)
        self.assertIsNone(orchestra.agents['Sub Agent 1']._context_token_cache)
        self.run_orchestra()
        for _, prompts in self.prompts[-2:]:
            self.assertNotIn('OLD_', prompts[0])
        self.assertEqual(orchestra.candidate_records, [None])
        self.assertEqual([record.turn for record in orchestra.work_records[0]], [1, 2])

    def test_chat_template_overflow_cannot_silently_truncate_protected_context(self):
        cfg = config(env_name='math')
        cfg.data = SimpleNamespace(max_prompt_length=4096, apply_chat_template_kwargs={})
        agent = RouterAgent('wg', ByteTokenizer(), None, cfg)
        batch = FakeBatch(2)
        batch.non_tensor_batch['raw_prompt'] = [[], [{'role': 'system', 'content': 'S' * 5000}]]
        obs = {'text': ['task', 'task']}
        Context.ensure_prompt_fits(agent, batch, obs, np.array([True, False]))
        with self.assertRaisesRegex(ValueError, 'after applying the chat template'):
            Context.ensure_prompt_fits(agent, batch, obs, np.array([True, True]))

    def test_unicode_source_excerpts_stay_within_budget_and_preserve_source_text(self):
        tokenizer = ByteTokenizer()
        cache = Context.ContextTokenCache(tokenizer)
        source = '前缀🙂' + '中间文字' * 100 + '🙂后缀'
        for budget in (0, 3, 19, 32, 64, 128):
            with self.subTest(budget=budget):
                result = Context.excerpt(source, budget, cache)
                self.assertLessEqual(len(tokenizer.encode(result)), budget)
                self.assertNotIn('\ufffd', result)
                if '[... omitted ...]' in result:
                    prefix, suffix = result.split('\n[... omitted ...]\n')
                    self.assertTrue(source.startswith(prefix))
                    self.assertTrue(source.endswith(suffix))


if __name__ == '__main__':
    unittest.main()
