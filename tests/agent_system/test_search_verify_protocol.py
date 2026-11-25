"""Regression cases for visible evidence reviews in search rollouts."""

import unittest

from tests.agent_system.test_role_router_agent import role_modules


class SearchVerifyProtocolTests(unittest.TestCase):
    def test_invalid_prefix_cannot_stand_in_for_a_review(self):
        parse = role_modules["search"]["VERIFY"].parse_report
        for prefix in ("...", "<think>unfinished", "<THINK>hidden</THINK>", "<answer>Paris</answer>"):
            with self.subTest(prefix=prefix):
                self.assertIsNone(parse(prefix + " <verify>yes</verify>"))

    def test_plain_text_can_include_comparisons_and_non_english_evidence(self):
        parse = role_modules["search"]["VERIFY"].parse_report
        review = "文档给出的温度 < 20°C，但没有说明测量日期。"
        self.assertEqual(parse(review + "\n<verify>no</verify>").assessment, review)
