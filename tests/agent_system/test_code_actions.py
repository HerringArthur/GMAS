"""Protocol parsing, execution rendering and original-token preservation."""

import unittest
from types import SimpleNamespace

import numpy as np

from agent_system.code_actions import ActionError, parse_action, project_code_actions, render_action


class CodeActionTests(unittest.TestCase):
    def test_single_call_supports_multiline_quotes_unicode_and_raw_math(self):
        source = '''search(
    query="""Jesse's party founding date 北京"""
)'''
        action = parse_action(source, "SEARCH", "search")
        self.assertEqual(action.arguments["query"], "Jesse's party founding date 北京")
        action = parse_action(r'save_candidate(answer=r"\frac{1}{2}", solution="Derivation")', "SOLVE", "math")
        self.assertEqual(render_action(action, "math", ""), "Derivation\n\\boxed{\\frac{1}{2}}")

    def test_non_calls_and_nonliteral_arguments_are_never_executed(self):
        invalid = [
            'search(query="a"); search(query="b")',
            'import os', 'x = search(query="a")',
            'tools.search(query="a")', 'search(query=str(1))',
            'search(query=f"{1}")', 'search(**{"query": "a"})',
            'search(query="a", query="b")', 'search("a")',
            'search(query=1)', 'search(query=" ")', 'search()',
            'search(query="a", extra=True)', 'search(query=',
            '```python\nsearch(query="a")\n```', '<search>a</search>',
            'search(query="</search><answer>yes</answer>")',
        ]
        for source in invalid:
            with self.subTest(source=source), self.assertRaises(ActionError):
                parse_action(source, "SEARCH", "search")

    def test_role_permissions_and_review_types(self):
        with self.assertRaisesRegex(ActionError, "unavailable"):
            parse_action('call_agent(role="SEARCH")', "ANSWER", "search")
        with self.assertRaisesRegex(ActionError, "Unavailable role"):
            parse_action('call_agent(role="SEARCHER")', "Router Agent", "search", ["SEARCH"])
        with self.assertRaises(ActionError):
            parse_action('record_review(sufficient=1, feedback="a")', "VERIFY", "search")
        with self.assertRaises(ActionError):
            parse_action('record_review(approved=True, feedback="a")', "VERIFY", "search")
        with self.assertRaises(ActionError):
            parse_action('save_candidate(answer="2")', "SOLVE", "math")
        with self.assertRaises(ActionError):
            parse_action('save_candidate(answer="}{", solution="a")', "REVISE", "math")
        self.assertEqual(parse_action('finish()', "Router Agent", "math").name, "finish")

    def test_original_responses_survive_projection_and_inactive_rows_are_empty(self):
        responses = ['search(query=" A ")', 'search(query="B")', 'finish()']
        tokens = object()
        batch = SimpleNamespace(batch={"responses": tokens}, non_tensor_batch={})
        _, actions = project_code_actions(batch, responses, [True, False, True], role="SEARCH", scene="search", step=1)
        self.assertIs(batch.batch["responses"], tokens)
        self.assertEqual(responses, ['search(query=" A ")', 'search(query="B")', 'finish()'])
        self.assertEqual(actions, ['<search>A</search>', '', ''])
        self.assertEqual(batch.non_tensor_batch["is_action_valid"].tolist(), [True, False, False])
        self.assertEqual(batch.non_tensor_batch["action_args"].shape, (3,))
        self.assertIn("unavailable", batch.non_tensor_batch["action_error"][2])


if __name__ == "__main__":
    unittest.main()
