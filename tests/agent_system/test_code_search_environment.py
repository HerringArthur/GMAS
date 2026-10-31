"""Run the real search tool and environment bodies with HTTP/grading mocked."""

import ast
import json
import logging
from pathlib import Path
import re
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.parse import urlparse

from agent_system.code_actions import parse_action, render_action


ROOT = Path(__file__).resolve().parents[2]
PREFIX = "agent_system/environments/env_package/search/third_party/skyrl_gym/"


def load_environment():
    class Env:
        def __class_getitem__(cls, key):
            return cls

    scope = dict(Env=Env, BaseTextEnvStepOutput=dict, re=re, json=json, threading=threading,
                 urlparse=urlparse, DEFAULT_TIMEOUT=30, logger=logging.getLogger("test-search"),
                 compute_score=Mock(return_value=1.0), call_search_api=Mock())
    for filename, names in (
        ("tools/core.py", {"tool", "ToolGroup"}),
        ("tools/search.py", {"SearchToolGroup", "_passages2string"}),
        ("envs/base_text_env.py", {"BaseTextEnv"}),
        ("envs/search/env.py", {"SearchEnv"}),
    ):
        path = ROOT / PREFIX / filename
        tree = ast.parse(path.read_text())
        nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
        nodes.extend(node for node in tree.body if getattr(node, "name", None) in names)
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), scope)
    return scope


class CodeSearchEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.scope = load_environment()
        tool = self.scope["SearchToolGroup"]
        with patch.object(tool, "_get_shared_session", return_value=object()):
            self.env = self.scope["SearchEnv"](SimpleNamespace(
                search_url="http://unused/retrieve", topk=3, timeout=30, log_requests=False,
            ))
        self.env.reset(dict(ground_truth="neonate", max_turns=4, data_source="test"))

    def query(self):
        source = 'search(query="newborn term")'
        return render_action(parse_action(source, "SEARCH", "search"), "search", source)

    def test_query_reaches_tool_once_and_candidate_terminates_without_another_search(self):
        api = self.scope["call_search_api"]
        api.return_value = ({"result": [[{"document": {"contents": "A newborn is a neonate."}}]]}, None)
        out = self.env.step(self.query())
        self.assertEqual(api.call_count, 1)
        self.assertEqual(api.call_args.kwargs["query"], "newborn term")
        self.assertTrue(out["metadata"]["tool_calling"])
        self.assertEqual(out["metadata"]["tool_status"], "success")
        self.assertIn("<information>", out["observations"][0]["content"])
        self.assertIn("neonate", out["observations"][0]["content"])
        source = 'save_candidate(answer="neonate")'
        answer = render_action(parse_action(source, "ANSWER", "search"), "search", source)
        final = self.env.step(answer)
        self.assertTrue(final["done"])
        self.assertFalse(final["metadata"]["tool_calling"])
        self.assertEqual(api.call_count, 1)
        self.assertIn(answer, self.scope["compute_score"].call_args.args[0])

    def test_api_failure_is_execution_status_and_empty_steps_do_not_count_as_search(self):
        self.scope["call_search_api"].return_value = (None, "timeout")
        with self.assertLogs("test-search", level="ERROR"):
            out = self.env.step(self.query())
        self.assertEqual(out["metadata"]["tool_status"], "api_error")
        self.assertTrue(out["metadata"]["tool_calling"])
        self.assertIn("timeout", out["observations"][0]["content"])
        empty = self.env.step("")
        self.assertFalse(empty["metadata"]["tool_calling"])
        self.assertEqual(empty["metadata"]["tool_status"], "not_called")
        self.assertEqual(self.scope["call_search_api"].call_count, 1)

    def test_prefetched_result_is_consumed_without_duplicate_request(self):
        self.env.tool_group._prefetched_search = ("newborn term", {"result": []}, None)
        out = self.env.step(self.query())
        self.assertEqual(out["metadata"]["tool_status"], "no_results")
        self.assertTrue(out["metadata"]["tool_calling"])
        self.scope["call_search_api"].assert_not_called()
        self.assertIsNone(self.env.tool_group._prefetched_search)


if __name__ == "__main__":
    unittest.main()
