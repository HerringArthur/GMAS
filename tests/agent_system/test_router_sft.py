"""Real-rollout Router format SFT checks, requiring only the standard library."""

import ast
import io
import json
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from examples.data_preprocess import router_sft as sft

ROOT = Path(__file__).resolve().parents[2]
CONSTANTS = {}
for node in ast.parse((ROOT / 'agent_system/agent/agents/router/router_agent.py').read_text()).body:
    if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') in ('ROUTER_PROMPT', 'ROLE_DESCRIPTIONS'):
        CONSTANTS[node.targets[0].id] = ast.literal_eval(node.value)


def record(output='SEARCH', scene='search', question='What is the capital of France?', candidate='', final=False):
    env = (f'The question is: {question}\nYour team can access an external search engine.\n' if scene == 'search'
           else f"The team's math problem is: {question}\n")
    prompt = CONSTANTS['ROUTER_PROMPT'].format(
        env_prompt=env, candidate_answer=candidate or sft.NO_CANDIDATE,
        team_context='Prior real work mentioning VERIFY and SEARCH.',
    )
    roles = [role for role in sft.ROLES[scene] if not (final and role == 'SEARCH')]
    prompt += '\n# Available Roles\n' + '\n'.join(f'- {role}: role description' for role in roles) + '\n'
    return {
        'record_type': sft.RECORD_TYPE, 'source_file': 'rollout.jsonl', 'sample': 1,
        'traj_uid': 'trajectory-a', 'uid': 'question-a', 'step': 1, 'turn': 1,
        'env_step': 4 if final else 1, 'data_source': scene, 'episode_pass': 0.0,
        'episode_rewards': 0.0, 'original_valid': False, 'selected_role': '',
        'input': '<|im_start|>system\nOriginal system.\n<|im_end|>\n<|im_start|>user\n' + prompt
                 + '<|im_end|>\n<|im_start|>assistant\n',
        'original_output': output + '<|im_end|>',
    }


class RouterSFTTests(unittest.TestCase):
    def test_bare_roles_case_and_wrappers(self):
        for raw, expected in [('SEARCH', 'SEARCH'), ('verify', 'VERIFY'), ('**ANSWER**', 'ANSWER'),
                              ('`SOLVE`', 'SOLVE'), ('(revise)', 'REVISE'), ('/search', 'SEARCH'),
                              ('FINISH', 'FINISH')]:
            with self.subTest(raw=raw):
                fixed = sft.repair_response(raw + '<|im_end|>')
                self.assertEqual(fixed['response'], f'<route>{expected}</route>')
                self.assertTrue(fixed['is_repaired'])

    def test_explicit_route_forms_and_missing_open_angle(self):
        for raw in ('<Route>SEARCH</Route>', 'ROUTE: SEARCH', 'ROUTES: SEARCH',
                    'ROUTE>SEARCH</route>', '#route>SEARCH</route>', '```xml\n<route>SEARCH</route>\n```'):
            with self.subTest(raw=raw):
                self.assertEqual(sft.repair_response(raw)['response'], '<route>SEARCH</route>')

    def test_reasoning_is_preserved_and_not_used_as_a_role_label(self):
        text = '<think>SEARCH found evidence; ANSWER has not composed a candidate.</think>\nVERIFY'
        fixed = sft.repair_response(text)
        self.assertEqual(fixed['route'], 'VERIFY')
        self.assertIn('SEARCH found evidence; ANSWER has not composed a candidate.', fixed['response'])
        self.assertEqual(sft.repair_response(text, route_only=True)['response'], '<route>VERIFY</route>')
        raw = 'THINK: Need evidence for this question.\nSEARCH\nROUTE: SEARCH\nLOAD: Use a precise query.'
        fixed = sft.repair_response(raw)
        self.assertEqual(fixed['response'], '<think>Need evidence for this question.\nUse a precise query.</think>\n<route>SEARCH</route>')

    def test_valid_response_and_duplicate_identical_decisions(self):
        valid = '<think>Check the evidence.</think>\n<route>VERIFY</route>'
        self.assertEqual(sft.repair_response(valid)['response'], valid)
        self.assertFalse(sft.repair_response(valid)['is_repaired'])
        repeated = 'SEARCH\n<think>Find evidence.</think>\n<route>SEARCH</route>'
        self.assertEqual(sft.repair_response(repeated)['response'], '<think>Find evidence.</think>\n<route>SEARCH</route>')

    def test_ambiguous_unknown_worker_or_truncated_outputs_are_rejected(self):
        invalid = [
            'SEARCH\nVERIFY', '<route>SEARCH</route><route>ANSWER</route>', 'ROUTE: SEARCH or VERIFY',
            '<search>Some query</search>', '<answer>Paris</answer>', 'I might search and then answer.',
            '<route>UNKNOWN</route>', 'VERFYYY', '<route>SEARCH', '<think>unfinished\nSEARCH',
            '<think><route>SEARCH</route></think><route>VERIFY</route>',
            '<think>a</think><think>b</think><route>SEARCH</route>',
            'SEARCH<|im_end|>\n<|im_start|>assistant\nANSWER', '<route>SEARCH</route> Ignore this suffix.',
        ]
        for typo in ('VERFIY', 'VERIFIY', 'VEHIFY'):
            invalid.extend([
                typo, f'<route>{typo}</route>', f'ROUTE: {typo}',
                f'{typo}\n<route>VERIFY</route>',
                f'{typo}: Check the evidence.\n<route>ANSWER</route>',
            ])
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(sft.Rejected):
                sft.repair_response(raw)

    def test_input_is_verbatim_and_assistant_target_is_only_router_output(self):
        source = record('SEARCH')
        messages, _ = sft.decode_input(source['input'])
        example = sft.convert_record(source)
        self.assertEqual(example['messages'][:-1], messages)
        self.assertEqual(example['messages'][-1], {'role': 'assistant', 'content': '<route>SEARCH</route>'})
        self.assertEqual(example['episode_pass'], 0)
        source['selected_role'] = 'ANSWER'
        self.assertEqual(sft.convert_record(source)['route'], 'SEARCH')
        source['original_output'] = 'unclear'
        with self.assertRaisesRegex(sft.Rejected, 'no_explicit_role'):
            sft.convert_record(source)

    def test_success_filter_is_optional_and_never_uses_positive_reward_as_pass(self):
        source = record()
        self.assertEqual(sft.convert_record(source)['route'], 'SEARCH')
        source['episode_rewards'] = 0.5
        with self.assertRaisesRegex(sft.Rejected, 'episode_not_successful'):
            sft.convert_record(source, success_only=True)
        source['episode_pass'] = 1
        self.assertEqual(sft.convert_record(source, success_only=True)['route'], 'SEARCH')

    def test_state_guards_filter_without_relabelling(self):
        with self.assertRaisesRegex(sft.Rejected, 'finish_without_candidate'):
            sft.convert_record(record('FINISH'))
        with self.assertRaisesRegex(sft.Rejected, 'role_unavailable'):
            sft.convert_record(record('SEARCH', final=True))
        self.assertEqual(sft.convert_record(record('VERIFY', final=True))['route'], 'VERIFY')
        self.assertEqual(sft.convert_record(record('FINISH', candidate='<answer>Paris</answer>'))['route'], 'FINISH')
        with self.assertRaisesRegex(sft.Rejected, 'math_review_without_candidate'):
            sft.convert_record(record('REVISE', scene='math'))
        for role in sft.ROLES['math']:
            candidate = r'Real derivation ending in \boxed{2}.' if role != 'SOLVE' else ''
            self.assertEqual(sft.convert_record(record(role, scene='math', candidate=candidate))['route'], role)

    def test_qwen_prefills_are_restored_and_unknown_templates_rejected(self):
        source = record('We need evidence.</think>\nSEARCH')
        source['input'] += '<think>\n'
        self.assertEqual(sft.convert_record(source)['messages'][-1]['content'], '<think>We need evidence.</think>\n<route>SEARCH</route>')
        source = record('SOLVE', scene='math')
        source['input'] += '<think>\n\n</think>\n\n'
        self.assertEqual(sft.convert_record(source)['messages'][-1]['content'], '<think></think>\n<route>SOLVE</route>')
        source['input'] = 'truncated input'
        with self.assertRaisesRegex(sft.Rejected, 'chat_template'):
            sft.convert_record(source)

    def test_txt_and_jsonl_readers_preserve_episode_and_ignore_worker(self):
        source = record()
        turns = [{'turn': 1, 'agent_id': 'Router Agent', 'env_step': 1, 'is_action_valid': False,
                  'input': source['input'], 'output': source['original_output']},
                 {'turn': 2, 'agent_id': 'SEARCH', 'input': 'worker prompt', 'output': '<search>query</search>'}]
        episode = {'step': 1, 'traj_uid': 'trajectory-a', 'uid': 'question-a', 'pass': 1.0, 'data_source': 'nq', 'turns': turns}
        readable = ['===== Step 1 / Sample 1 =====', json.dumps({k: v for k, v in episode.items() if k != 'turns'}, indent=2)]
        for turn in turns:
            readable.extend([f"--- Turn {turn['turn']}: {turn['agent_id']} ---",
                             json.dumps({k: v for k, v in turn.items() if k not in ('input', 'output')}),
                             '[Input]', turn['input'], '[Output]', turn['output'], ''])
        with tempfile.TemporaryDirectory() as directory:
            txt, js = Path(directory) / 'step.txt', Path(directory) / 'step.jsonl'
            txt.write_text('\n'.join(readable) + '\n')
            sft.write_records(js, [episode])
            a, = sft.parse_log(txt)
            b, = sft.parse_log(js)
        a.pop('source_file')
        b.pop('source_file')
        self.assertEqual(a, b)
        self.assertEqual(a['input'], source['input'])
        self.assertEqual(a['episode_pass'], 1.0)

    def test_duplicates_conflicting_labels_and_cap_are_auditable(self):
        inputs = [record('SEARCH'), record('<route>SEARCH</route>'), record('ANSWER', question='Conflicting question'),
                  record('VERIFY', question='Conflicting question')]
        audits = [dict(row, status='pending') for row in inputs]
        pairs = [(sft.convert_record(row), audit) for row, audit in zip(inputs, audits)]
        kept = sft.select_examples(pairs, 4)
        self.assertEqual(len(kept), 1)
        self.assertFalse(kept[0]['is_repaired'])
        self.assertEqual(kept[0]['occurrences'], 2)
        self.assertEqual([r['reason'] for r in audits], ['duplicate_input_and_route', 'accepted',
                                                       'conflicting_labels_for_input', 'conflicting_labels_for_input'])
        self.assertNotIn('pending', [a['status'] for a in audits])

    def test_question_split_keeps_all_states_together(self):
        rows = []
        for i in range(40):
            item = sft.convert_record(record('SEARCH', question=f'Question {i}'))
            rows.extend([item, dict(item, route='VERIFY')])
        splits = sft.split_rows(rows, 0.2, 42)
        train = {r['question_id'] for r in splits['train']}
        val = {r['question_id'] for r in splits['val']}
        self.assertTrue(train and val)
        self.assertFalse(train & val)
        self.assertEqual(sft.question_id(' A   question\n'), sft.question_id('a question'))

    def test_end_to_end_jsonl_export_from_rollout_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = [record('SEARCH', question=f'Question {i}') for i in range(40)]
            sft.write_records(root / 'records.jsonl', sources)
            args = SimpleNamespace(logs=[], records=[root / 'records.jsonl'],
                                   tokenizer=None, success_only=False, route_only=False,
                                   max_per_question_route=4, val_fraction=0.2, seed=42, max_length=8192,
                                   parquet=False, output_dir=root / 'dataset')
            with redirect_stdout(io.StringIO()):
                sft.build(args)
            report = json.loads((args.output_dir / 'report.json').read_text())
            self.assertEqual(report['accepted_examples'], 40)
            self.assertEqual(report['filter_counts'], {})
            self.assertFalse(report['ready_for_training'])
            audit = list(sft.read_records(args.output_dir / 'audit.jsonl'))
            self.assertEqual(len(audit), 40)
            self.assertEqual(sum(r['status'] == 'accepted' for r in audit), 40)
            for split in ('train', 'val'):
                self.assertTrue(list(sft.read_records(args.output_dir / f'{split}.jsonl')))
            with self.assertRaisesRegex(ValueError, 'already exists'):
                sft.build(args)

    def test_old_synthetic_or_hand_labelled_data_cannot_enter_the_pipeline(self):
        with self.assertRaisesRegex(ValueError, 'real rollout'):
            sft.convert_record({'scene': 'search', 'route': 'ANSWER', 'messages': []})

    def test_outputs_match_the_production_router_parser(self):
        # Use the actual parser pattern without importing GPU/training modules.
        tree = ast.parse((ROOT / 'agent_system/agent/agents/router/router_agent.py').read_text())
        pattern = next(ast.literal_eval(node.args[0]) for node in ast.walk(tree)
                       if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'fullmatch')
        for raw in ('SEARCH', '<Route>VERIFY</Route>', 'THINK: Fix the calculation.\nREVISE', 'ROUTES: FINISH'):
            fixed = sft.repair_response(raw)
            self.assertEqual(re.fullmatch(pattern, fixed['response'], re.S)[1], fixed['route'])


if __name__ == '__main__':
    unittest.main()
