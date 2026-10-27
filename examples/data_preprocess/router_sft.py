"""Build Router format SFT from real GMAS rollout logs, using explicit rules only.

The input is preserved. The target retains the role explicitly emitted by the
Router, with normalized tags and optional original reasoning. No model calls,
reference answers, synthetic states, or policy relabelling are involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROLES = {"search": ("SEARCH", "VERIFY", "ANSWER", "FINISH"),
         "math": ("SOLVE", "VERIFY", "REVISE", "FINISH")}
CANDIDATE_INTRO = (
    "The latest saved candidate is provided below for evaluation and improvement.\n"
    "Being saved does not imply that it is correct or has been verified.\n"
)
NO_CANDIDATE = "No candidate answer has been generated yet."
STRICT_ROUTE = re.compile(
    r"\s*(?:<think>(?:(?!</?(?:think|route)>).)*</think>\s*)?<route>([^<>]+)</route>\s*", re.S,
)
RECORD_TYPE = "router_rollout_v1"


class Rejected(ValueError):
    """A logged row that cannot be safely converted by the explicit rules."""


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def question_id(question):
    return digest(" ".join(question.split()).casefold())


def read_records(path):
    """Read trajectory JSONL or Router records extracted from real trajectories."""
    path = Path(path).expanduser()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_records(path, rows):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def text_episodes(path):
    """Read the existing TrainingBenchmark human-readable export."""
    text = Path(path).read_text(encoding="utf-8")
    sections = re.split(r"^===== Step \d+ / Sample \d+ =====\n", text, flags=re.M)
    if len(sections) == 1 or sections[0].strip():
        raise ValueError(f"{path}: expected TrainingBenchmark sample headers")
    for section in sections[1:]:
        chunks = re.split(r"^--- Turn \d+: [^\n]+\n", section, flags=re.M)
        episode = json.loads(chunks[0])
        episode["turns"] = []
        for chunk in chunks[1:]:
            metadata, separator, body = chunk.partition("\n")
            if not separator or not body.startswith("[Input]\n"):
                raise ValueError(f"{path}: malformed turn header")
            turn = json.loads(metadata)
            prompt, separator, response = body[len("[Input]\n"):].partition("\n[Output]\n")
            if not separator:
                raise ValueError(f"{path}: turn has no output delimiter")
            turn.update(input=prompt, output=response.rstrip("\n"))
            episode["turns"].append(turn)
        yield episode


def parse_log(path):
    """Keep every Router attempt, including invalid ones and episode outcomes."""
    path = Path(path).expanduser()
    episodes = text_episodes(path) if path.suffix == ".txt" else read_records(path)
    for number, episode in enumerate(episodes, 1):
        if not isinstance(episode.get("turns"), list):
            raise ValueError(f"{path}: expected benchmark trajectory JSONL with a turns list")
        for turn in episode["turns"]:
            if turn.get("agent_id") != "Router Agent":
                continue
            yield {
                "record_type": RECORD_TYPE, "source_file": str(path), "sample": number,
                "traj_uid": episode.get("traj_uid"), "uid": episode.get("uid"),
                "step": episode.get("step"), "turn": turn["turn"], "env_step": turn.get("env_step"),
                "data_source": episode.get("data_source", "unknown"),
                "episode_pass": episode.get("pass"), "episode_rewards": episode.get("episode_rewards"),
                "original_valid": turn.get("is_action_valid"), "selected_role": turn.get("selected_role"),
                "input": turn["input"], "original_output": turn["output"],
            }


def decode_input(raw):
    """Restore system/user verbatim and retain Qwen assistant prefill separately."""
    match = re.fullmatch(
        r"<\|im_start\|>system\n(.*?)<\|im_end\|>\n<\|im_start\|>user\n(.*?)"
        r"<\|im_end\|>\n<\|im_start\|>assistant\n(.*)", raw, re.S,
    )
    if match is None:
        raise Rejected("unsupported_or_truncated_chat_template")
    system, prompt, prefix = match.groups()
    if re.search(r"<\|(?:im_start|im_end)\|>", system + prompt):
        raise Rejected("ambiguous_chat_boundaries")
    if prefix.strip() not in ("", "<think>", "<think>\n\n</think>"):
        raise Rejected("unsupported_assistant_prefill")
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}], prefix


def prompt_state(prompt):
    """Read visible constraints, without reconstructing or rewriting the prompt."""
    try:
        task, candidate_part = prompt.split("\n# Current Candidate Answer\n", 1)
        candidate, rest = candidate_part.split("\n# Your Teammates' Outputs", 1)
        _, options = rest.rsplit("\n# Available Roles\n", 1)
    except ValueError as error:
        raise Rejected("missing_prompt_sections") from error
    candidate = candidate.removeprefix(CANDIDATE_INTRO).strip()
    if not candidate_part.startswith(CANDIDATE_INTRO):
        raise Rejected("unsupported_candidate_section")
    if candidate == NO_CANDIDATE:
        candidate = ""
    if "The team's math problem is: " in task:
        scene = "math"
        question = task.split("The team's math problem is: ", 1)[1].strip()
    elif "The question is: " in task and "\nYour team can access" in task:
        scene = "search"
        question = task.split("The question is: ", 1)[1].split("\nYour team can access", 1)[0].strip()
    else:
        raise Rejected("unknown_scene_or_question")
    roles = re.findall(r"^- ([A-Z]+):", options, re.M)
    if not question or not roles or any(role not in ROLES[scene] for role in roles):
        raise Rejected("unsupported_available_roles")
    return scene, question, candidate, roles


def clean_response(response):
    response = response.strip()
    # Remove only terminal transport tokens, never arbitrary tokens in the body.
    response = re.sub(r"(?:\s*(?:<\|im_end\|>|<\|endoftext\|>))+$", "", response).strip()
    if re.search(r"<\|[^\n]+?\|>", response):
        raise Rejected("embedded_special_tokens")
    return response


def normalize_role(token):
    """Normalize case only; reject misspelled and unknown role names."""
    token = token.upper()
    if token not in set().union(*ROLES.values()):
        raise Rejected("unknown_role")
    return token


def bare_role(line):
    """Match an entire decision line; never infer a role from prose keywords."""
    line = line.strip()
    for opening, closing in (("**", "**"), ("`", "`"), ('"', '"'), ("'", "'"), ("(", ")")):
        if line.startswith(opening) and line.endswith(closing) and len(line) > len(opening) + len(closing):
            line = line[len(opening):-len(closing)].strip()
    token = line.removeprefix("/")
    if token.upper() in set().union(*ROLES.values()):
        return normalize_role(token)
    return None


def repair_response(raw, route_only=False):
    """Return a legal response with the same explicit role, or reject ambiguity."""
    original = clean_response(raw)
    text = original
    rules, selections, reasoning = [], [], []
    fence = re.fullmatch(r"```(?:xml|text)?\s*\n(.*?)\n```", text, re.S | re.I)
    if fence:
        text = fence[1].strip()
        rules.append("remove_code_fence")

    # Reasoning may mention any roles: it is never searched for a decision.
    thinks = list(re.finditer(r"<think>(.*?)</think>", text, re.S | re.I))
    if len(thinks) > 1:
        raise Rejected("multiple_think_blocks")
    if thinks:
        content = thinks[0][1].strip()
        if re.search(r"</?(?:think|route)\b", content, re.I):
            raise Rejected("nested_control_tags")
        reasoning.append(content)
        text = text[:thinks[0].start()] + "\n" + text[thinks[0].end():]
    if re.search(r"</?think\b", text, re.I):
        raise Rejected("incomplete_think_block")

    # Also recognize the two explicit missing-'<' spellings present in the logs.
    route_tag = re.compile(r"(?<![\w<])(?:<route|#?route)>\s*([A-Za-z_]+)\s*</route>", re.I)

    def consume_tag(match):
        selections.append(normalize_role(match[1]))
        rules.append("route_tag")
        return "\n"

    text = route_tag.sub(consume_tag, text)
    if re.search(r"</?(?:search|answer)\b", text, re.I):
        raise Rejected("worker_action_in_router_output")
    if "<" in text or ">" in text or "```" in text:
        raise Rejected("unsupported_or_incomplete_markup")

    remainder = []
    for line in text.splitlines():
        marker = re.fullmatch(r"\s*ROUTES?\s*:\s*([A-Za-z_]+)\s*", line, re.I)
        if marker:
            selections.append(normalize_role(marker[1]))
            rules.append("route_field")
        else:
            role = bare_role(line)
            if role:
                selections.append(role)
                rules.append("bare_role")
            else:
                if re.match(r"\s*ROUTES?\s*:", line, re.I):
                    raise Rejected("ambiguous_route_field")
                remainder.append(line)
    if not selections:
        raise Rejected("no_explicit_role")
    if len(set(selections)) != 1:
        raise Rejected("conflicting_roles")
    route = selections[0]
    leftover = "\n".join(remainder).strip()
    if leftover:
        # Keep explicitly labelled original explanation/task text. Free prose is
        # not guessed into either a decision or a rationale.
        if not re.match(r"(?:THINK|LOAD)\s*:", leftover, re.I):
            raise Rejected("unrecognized_extra_text")
        if re.search(r"^\s*(?:#|[-*]\s|(?:SEARCH|VERIFY|ANSWER|SOLVE|REVISE|FINISH)\b)", leftover, re.M | re.I):
            raise Rejected("ambiguous_extra_text")
        leftover = re.sub(r"^\s*(?:THINK|LOAD)\s*:\s*", "", leftover, flags=re.M | re.I).strip()
        reasoning.insert(0, leftover)
        rules.append("wrap_labelled_reasoning")
    content = "\n\n".join(part for part in reasoning if part)
    # Preserve even an empty think block/prefill unless route-only was requested.
    response = (f"<think>{content}</think>\n" if (reasoning or thinks) and not route_only else "") + f"<route>{route}</route>"
    if not STRICT_ROUTE.fullmatch(response):
        raise Rejected("canonical_output_failed_validation")
    strict_original = STRICT_ROUTE.fullmatch(original)
    if strict_original and strict_original[1].strip() == route:
        rule = "already_valid"
    else:
        rule = "+".join(sorted(set(rules)))
    return {"response": response, "route": route, "repair_rule": rule,
            "is_repaired": rule != "already_valid"}


def convert_record(record, route_only=False, success_only=False):
    if record.get("record_type") != RECORD_TYPE:
        raise ValueError("Expected extracted real rollout records; synthetic/annotated records are not accepted")
    if success_only and record.get("episode_pass") != 1:
        raise Rejected("episode_not_successful_or_unknown")
    messages, prefix = decode_input(record["input"])
    scene, question, candidate, roles = prompt_state(messages[1]["content"])
    fixed = repair_response(prefix + record["original_output"], route_only)
    route = fixed["route"]
    if route not in roles:
        raise Rejected("role_unavailable_in_prompt")
    if route == "FINISH" and not candidate:
        raise Rejected("finish_without_candidate")
    if scene == "math" and route in ("VERIFY", "REVISE") and not candidate:
        raise Rejected("math_review_without_candidate")
    # No change to question/evidence/candidate/history; the existing role choice
    # is the only label source. selected_role metadata is not a fallback label.
    return {
        "messages": messages + [{"role": "assistant", "content": fixed["response"]}],
        "question_id": question_id(question), "scene": scene, "route": route,
        "repair_rule": fixed["repair_rule"], "is_repaired": fixed["is_repaired"],
        "source_file": record["source_file"], "traj_uid": record.get("traj_uid"),
        "turn": record["turn"], "data_source": record.get("data_source", "unknown"),
        "episode_pass": record.get("episode_pass"), "original_valid": record.get("original_valid"),
    }


def input_id(row):
    return digest(json.dumps(row["messages"][:-1], sort_keys=True, ensure_ascii=False))


def select_examples(pairs, max_per_question_route):
    """Deduplicate repeated retries and drop inputs with conflicting role labels."""
    groups = defaultdict(list)
    for example, audit in pairs:
        groups[input_id(example)].append((example, audit))
    unique = []
    for key in sorted(groups):
        entries = groups[key]
        if len({example["route"] for example, _ in entries}) > 1:
            for _, audit in entries:
                audit.update(status="rejected", reason="conflicting_labels_for_input")
            continue
        # Prefer an originally well-formed output, then a shorter original
        # explanation. This selects a real sample; it does not invent a label.
        entries.sort(key=lambda pair: (pair[0]["is_repaired"], len(pair[0]["messages"][-1]["content"]),
                                       pair[0]["messages"][-1]["content"], pair[1]["source_file"], pair[1]["turn"]))
        chosen, audit = entries[0]
        chosen["occurrences"] = len(entries)
        unique.append((chosen, audit))
        for _, duplicate in entries[1:]:
            duplicate.update(status="rejected", reason="duplicate_input_and_route")
    counts = Counter()
    kept = []
    for example, audit in unique:
        key = (example["question_id"], example["scene"], example["route"])
        if max_per_question_route and counts[key] >= max_per_question_route:
            audit.update(status="rejected", reason="per_question_route_cap")
            continue
        counts[key] += 1
        audit.update(status="accepted", reason="accepted")
        kept.append(example)
    return kept


def split_rows(rows, val_fraction, seed):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    result = {"train": [], "val": []}
    for row in rows:
        fraction = int(digest(f"{seed}:{row['question_id']}")[:16], 16) / 2**64
        result["val" if fraction < val_fraction else "train"].append(row)
    return result


def source_records(args):
    for path in args.logs:
        yield from parse_log(path)
    for path in getattr(args, "records", []):
        yield from read_records(path)


def extract_log(args):
    rows = list(source_records(args))
    write_records(args.output, rows)
    print(json.dumps({"router_calls": len(rows), "invalid_calls": sum(r["original_valid"] is False for r in rows)}))


def summarize_report(report):
    """Render saved dataset statistics without re-reading large rollout logs."""
    lines = [
        "Router SFT 数据构造统计",
        f"输入文件数: {len(report['source_files'])}",
        f"Router 调用数: {report['router_calls']}",
        f"日志标记为无效的 Router 调用数: {report['original_invalid_calls']}",
        f"最终保留: {report['accepted_examples']} ({report['acceptance_rate']:.2%})",
        f"过滤/去重: {report['rejected_calls']}",
        f"保留样本中经过格式修复: {report['accepted_repaired_examples']}",
        f"保留样本中原本格式合法: {report['accepted_examples'] - report['accepted_repaired_examples']}",
        f"保留样本覆盖的独立题目数: {report['accepted_questions']}",
        f"只保留成功轨迹: {report['success_only']}",
        f"已检查 token 长度: {report['length_checked']}，最大长度: {report['max_length']}",
        f"Tokenizer: {report['tokenizer']}",
        f"已满足长度检查及非空 train/val 条件: {report['ready_for_training']}",
    ]
    for name, split in report['splits'].items():
        lines.append(f"\n{name}: {split['examples']} 条样本，{split['questions']} 道题目")
        for key, count in sorted(split['scene_route'].items()):
            lines.append(f"  {key}: {count}")
        lengths = split['token_lengths']
        if lengths:
            lines.append(f"  完整 messages token 长度: min={lengths['min']}, "
                         f"median={lengths['median']}, mean={lengths['mean']:.1f}, max={lengths['max']}")
    for title, key in (("保留样本的数据来源", "accepted_data_sources"),
                       ("保留样本所属轨迹的结果（按样本计数）", "accepted_episode_outcomes"),
                       ("保留样本的修复规则", "accepted_rules"), ("过滤/去重原因", "filter_counts")):
        lines.append(f"\n{title}:")
        lines.extend(f"  {name}: {count}" for name, count in sorted(report[key].items()))
        if not report[key]:
            lines.append("  无")
    lines.append("\n已识别场景中训练集缺失的角色: " + (", ".join(report['missing_routes']) or "无"))
    return "\n".join(lines) + "\n"


def build(args):
    output = Path(args.output_dir).expanduser()
    outputs = ("train.jsonl", "val.jsonl", "train.parquet", "val.parquet", "report.json", "audit.jsonl", "summary.txt")
    if any((output / name).exists() for name in outputs):
        raise ValueError("Output already exists; use a new output directory")
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    pairs, audit_rows, observed_scenes = [], [], set()
    for record in source_records(args):
        audit = dict(record, status="pending")
        audit_rows.append(audit)
        try:
            row = convert_record(record, args.route_only, args.success_only)
            observed_scenes.add(row["scene"])
            audit.update(question_id=row["question_id"], route=row["route"],
                         repair_rule=row["repair_rule"], response=row["messages"][-1]["content"])
            if tokenizer is not None:
                tokens = tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=False)
                row["token_length"] = len(tokens)
                audit["token_length"] = len(tokens)
                if len(tokens) > args.max_length:
                    raise Rejected("overlong_example")
            pairs.append((row, audit))
        except Rejected as error:
            audit.update(status="rejected", reason=str(error))
    if not audit_rows:
        raise ValueError("No Router turns found in the input files")
    kept = select_examples(pairs, args.max_per_question_route)
    splits = split_rows(kept, args.val_fraction, args.seed)
    report = {
        "source": "real_rollout_format_repair", "label_policy": "preserve_explicit_role",
        "success_only": args.success_only, "route_only": args.route_only,
        "max_per_question_route": args.max_per_question_route,
        "length_checked": tokenizer is not None, "max_length": args.max_length,
        "tokenizer": args.tokenizer, "seed": args.seed, "val_fraction": args.val_fraction,
        "source_files": sorted({r["source_file"] for r in audit_rows}),
        "router_calls": len(audit_rows), "original_invalid_calls": sum(r.get("original_valid") is False for r in audit_rows),
        "accepted_examples": len(kept), "accepted_repaired_examples": sum(r["is_repaired"] for r in kept),
        "rejected_calls": len(audit_rows) - len(kept), "acceptance_rate": len(kept) / len(audit_rows),
        "accepted_questions": len({r["question_id"] for r in kept}),
        "accepted_data_sources": dict(Counter(r["data_source"] for r in kept)),
        "accepted_episode_outcomes": dict(Counter(
            "success" if r["episode_pass"] == 1 else "failure" if r["episode_pass"] == 0 else "unknown" for r in kept
        )),
        "filter_counts": dict(Counter(r["reason"] for r in audit_rows if r["status"] == "rejected")),
        "accepted_rules": dict(Counter(r["repair_rule"] for r in kept)), "splits": {},
    }
    for name, rows in splits.items():
        lengths = [r["token_length"] for r in rows if "token_length" in r]
        report["splits"][name] = {"examples": len(rows), "questions": len({r["question_id"] for r in rows}),
                                 "scene_route": dict(Counter(r["scene"] + ":" + r["route"] for r in rows)),
                                 "token_lengths": {"min": min(lengths), "max": max(lengths),
                                                   "mean": statistics.mean(lengths), "median": statistics.median(lengths)}
                                 if lengths else None}
    report["missing_routes"] = [f"{scene}:{role}" for scene in sorted(observed_scenes) for role in ROLES[scene]
                                if not any(r["scene"] == scene and r["route"] == role for r in splits["train"])]
    # Missing roles are reported, not fabricated or required for format warmup.
    report["ready_for_training"] = bool(splits["train"] and splits["val"] and tokenizer)
    if args.parquet:
        import pyarrow as pa
        import pyarrow.parquet as pq

    output.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        write_records(output / f"{name}.jsonl", rows)
        if args.parquet and rows:
            pq.write_table(pa.Table.from_pylist(rows), output / f"{name}.parquet")
    write_records(output / "audit.jsonl", audit_rows)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "summary.txt").write_text(summarize_report(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract-log", help="Export every real Router attempt with original inputs/outputs")
    extract.add_argument("--logs", nargs="+", required=True, help="TrainingBenchmark step_*.txt or step_*.jsonl")
    extract.add_argument("--output", required=True)
    extract.set_defaults(func=extract_log)
    p = sub.add_parser("build", help="Repair explicit Router formats and export SFT messages")
    p.add_argument("--logs", nargs="+", default=[])
    p.add_argument("--records", nargs="+", default=[], help="Records previously exported by extract-log")
    p.add_argument("--success-only", action="store_true", help="Require episode pass=1 (default includes failed episodes)")
    p.add_argument("--route-only", action="store_true", help="Omit original reasoning; target only the route tag")
    p.add_argument("--max-per-question-route", type=int, default=4, help="Cap repeated states per question/scene/role; 0 disables")
    p.add_argument("--tokenizer", help="SFT model/tokenizer for rejecting overlong examples")
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--parquet", action="store_true")
    p.add_argument("--output-dir", required=True)
    p.set_defaults(func=build)
    args = parser.parse_args()
    if args.command == "build":
        if not args.logs and not args.records:
            parser.error("build requires --logs or --records")
        if args.max_per_question_route < 0 or args.max_length < 1 or not 0 < args.val_fraction < 1:
            parser.error("Invalid cap, maximum length, or validation fraction")
    args.func(args)


if __name__ == "__main__":
    main()
