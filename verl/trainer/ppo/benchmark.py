"""Opt-in driver-side timing and sampled trajectory recording (no GPU imports)."""

import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def _plain(value):
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json(value, **kwargs):
    return json.dumps(_plain(value), ensure_ascii=False, allow_nan=False, **kwargs)


def _atomic_text(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _stats(values):
    if not values:
        return None
    return {
        "mean": statistics.mean(values), "median": statistics.median(values),
        "min": min(values), "max": max(values),
    }


class TrainingBenchmark:
    """Record pre-padding rollouts and completed steps without changing training RNG."""

    STAGES = ("gen", "old_log_prob", "reward", "adv", "update_actor", "dump_rollout_samples")
    TURN_FIELDS = (
        "agent_id", "wg_id", "env_step", "selected_agent", "handoff_decision",
        "is_action_valid", "agent_path", "prefix_key", "parent_prefix_key",
        "routed_agent", "routing_instruction", "router_feedback", "return_reason",
        "candidate_answer", "candidate_author", "candidate_repeat_count", "candidate_evidence",
        "candidate_quality", "candidate_review", "last_worker_decision", "protocol_repaired", "rewards",
    )
    EPISODE_FIELDS = ("uid", "episode_rewards", "episode_lengths", "tool_callings", "pass", "data_source")

    def __init__(self, config, total_steps):
        settings = config["trainer"]["benchmark"]
        self.output_dir = Path(settings["output_dir"])
        self.samples_per_step = int(settings.get("samples_per_step", 3))
        self.warmup_steps = int(settings.get("warmup_steps", 2))
        self.seed = int(settings.get("seed", 42))
        self.total_steps = int(total_steps)
        if self.samples_per_step < 0 or not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError("Require samples_per_step >= 0 and 0 <= warmup_steps < total_steps")
        if config["trainer"].get("resume_mode") != "disable":
            raise ValueError("Benchmark requires trainer.resume_mode=disable to keep step statistics unambiguous")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if (self.output_dir / "metrics.jsonl").exists():
            raise FileExistsError(f"Benchmark metrics already exist in {self.output_dir}; use a fresh output_dir")
        self.rows = []
        _atomic_text(self.output_dir / "effective_config.json", _json(config, indent=2) + "\n")

    def record_rollouts(self, batch, tokenizers, step, input_questions):
        """Select whole trajectories in collector order, before copying training rows."""
        nt = batch.non_tensor_batch
        trajectories = defaultdict(list)
        for index, trajectory in enumerate(nt["traj_uid"]):
            trajectories[str(trajectory)].append(index)
        response_width = batch.batch["responses"].shape[-1]
        # Count all generated response positions once, before adjust_batch copies rows.
        response_tokens = int(batch.batch["attention_mask"][:, -response_width:].sum().item())
        counts = Counter(str(wg) for wg in nt["wg_id"])
        metrics = {
            "benchmark/questions": input_questions,
            "benchmark/trajectories": len(trajectories),
            "benchmark/rollout_rows": len(batch),
            "benchmark/response_tokens": response_tokens,
            **{f"benchmark/rollout_rows/{wg}": count for wg, count in counts.items()},
        }
        valid_flags = nt.get("is_action_valid")
        if valid_flags is None:
            valid_mask = [True] * len(batch)
        else:
            valid_mask = [bool(flag) for flag in valid_flags]
        metrics["benchmark/valid_rollout_rows"] = sum(valid_mask)
        valid_counts = Counter(
            str(wg) for wg, valid in zip(nt["wg_id"], valid_mask) if valid
        )
        metrics.update({
            f"benchmark/valid_rollout_rows/{wg}": count
            for wg, count in valid_counts.items()
        })
        if "protocol_repaired" in nt:
            metrics["benchmark/protocol_repaired"] = sum(bool(item) for item in nt["protocol_repaired"])
        if not self.samples_per_step:
            return metrics

        # A private RNG avoids changing data shuffling or rollout sampling state.
        selected = random.Random(self.seed + step).sample(
            list(trajectories), min(self.samples_per_step, len(trajectories))
        )
        samples = []
        for trajectory in selected:
            indices = trajectories[trajectory]
            first = indices[0]
            sample = {"step": step, "traj_uid": trajectory, "turns": []}
            for key in self.EPISODE_FIELDS:
                if key in nt:
                    sample[key] = _plain(nt[key][first])
            for order, index in enumerate(indices, 1):
                tokenizer = tokenizers[nt["wg_id"][index]]
                prompt = batch.batch["prompts"][index].tolist()
                response = batch.batch["responses"][index].tolist()
                mask = batch.batch["attention_mask"][index].tolist()
                split = len(prompt)
                prompt = [token for token, keep in zip(prompt, mask[:split]) if keep]
                response = [token for token, keep in zip(response, mask[split:]) if keep]
                turn = {"turn": order}
                for key in self.TURN_FIELDS:
                    if key in nt:
                        turn[key] = _plain(nt[key][index])
                turn.update(
                    input=tokenizer.decode(prompt, skip_special_tokens=False),
                    output=tokenizer.decode(response, skip_special_tokens=False),
                    response_tokens=len(response),
                )
                sample["turns"].append(turn)
            samples.append(sample)

        folder = self.output_dir / "rollout_samples"
        folder.mkdir(exist_ok=True)
        path = folder / f"step_{step:06d}"
        _atomic_text(path.with_suffix(".jsonl"), "".join(_json(sample) + "\n" for sample in samples))
        readable = []
        for number, sample in enumerate(samples, 1):
            episode = {key: value for key, value in sample.items() if key != "turns"}
            readable.extend([f"===== Step {step} / Sample {number} =====", _json(episode, indent=2)])
            for turn in sample["turns"]:
                metadata = {key: value for key, value in turn.items() if key not in ("input", "output")}
                readable.extend([
                    f"--- Turn {turn['turn']}: {turn.get('agent_id', '')} ---",
                    _json(metadata), "[Input]", turn["input"], "[Output]", turn["output"], "",
                ])
        _atomic_text(path.with_suffix(".txt"), "\n".join(readable) + "\n")
        print(f"[benchmark] step={step}: saved {len(samples)} trajectories to {path}.txt", flush=True)
        return metrics

    def record_step(self, metrics, step, training_batch):
        """Save standard step timing, explicit observation cost, and padding counts."""
        step_s = float(metrics["timing_s/step"])
        dump_s = float(metrics.get("timing_s/dump_rollout_samples", 0.0))
        if not math.isfinite(step_s) or step_s <= 0 or not 0 <= dump_s < step_s:
            raise ValueError(f"Invalid benchmark timing: step={step_s}, sample_dump={dump_s}")
        training_counts = Counter(str(wg) for wg in training_batch.non_tensor_batch["wg_id"])
        for wg, count in training_counts.items():
            metrics[f"benchmark/training_rows/{wg}"] = count
            source_count = metrics.get(
                f"benchmark/valid_rollout_rows/{wg}",
                metrics[f"benchmark/rollout_rows/{wg}"],
            )
            metrics[f"benchmark/copied_rows/{wg}"] = count - source_count
        metrics["benchmark/training_rows"] = len(training_batch)
        source_rows = metrics.get("benchmark/valid_rollout_rows", metrics["benchmark/rollout_rows"])
        metrics["benchmark/copied_rows"] = len(training_batch) - source_rows
        metrics["benchmark/step_excluding_samples_s"] = step_s - dump_s
        metrics["benchmark/questions_per_s"] = metrics["benchmark/questions"] / step_s
        row = {
            "step": step, "is_warmup": len(self.rows) < self.warmup_steps,
            "step_s": step_s, "step_excluding_samples_s": step_s - dump_s,
            **{f"{stage}_s": metrics.get(f"timing_s/{stage}", 0.0) for stage in self.STAGES},
            **{key: metrics[f"benchmark/{key}"] for key in (
                "questions", "trajectories", "response_tokens", "rollout_rows",
                "training_rows", "copied_rows", "questions_per_s",
            )},
        }
        with (self.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(_json({"step": step, "metrics": metrics}) + "\n")
        with (self.output_dir / "timing.csv").open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not self.rows:
                writer.writeheader()
            writer.writerow(row)
        self.rows.append(row)
        self._write_summary()
        print(f"[benchmark] step={step}: {step_s:.2f}s, samples={dump_s:.2f}s, "
              f"questions/s={row['questions_per_s']:.3f}, warmup={row['is_warmup']}", flush=True)

    def _write_summary(self):
        measured = self.rows[self.warmup_steps:]
        seconds = sum(row["step_s"] for row in measured)
        gen_seconds = sum(row["gen_s"] for row in measured)
        questions = sum(row["questions"] for row in measured)
        summary = {
            "requested_steps": self.total_steps,
            "completed_steps": len(self.rows),
            "complete": len(self.rows) >= self.total_steps,
            "warmup_steps": self.warmup_steps,
            "measured_steps": len(measured),
            "step_s": _stats([row["step_s"] for row in measured]),
            "step_excluding_samples_s": _stats([row["step_excluding_samples_s"] for row in measured]),
            "stage_mean_s": {
                stage: statistics.mean(row[f"{stage}_s"] for row in measured)
                for stage in self.STAGES
            } if measured else {},
            "questions_per_s": questions / seconds if seconds else None,
            "trajectories_per_s": sum(row["trajectories"] for row in measured) / seconds if seconds else None,
            "generated_response_tokens_per_gen_s": (
                sum(row["response_tokens"] for row in measured) / gen_seconds if gen_seconds else None
            ),
            "timing_scope": (
                "timing_s/step includes rollout sample selection, token counting, decoding and file writes. "
                "step_excluding_samples_s subtracts that measured cost; it is an estimate, not a run with sampling disabled. "
                "Step timers exclude setup, data loading before the timer, metric computation/logging and summary writes. "
                "Counts use collector output before training padding; tokens include all agent responses."
            ),
        }
        _atomic_text(self.output_dir / "summary.json", _json(summary, indent=2) + "\n")
        lines = [
            f"Completed steps: {len(self.rows)}/{self.total_steps}",
            f"Warmup steps excluded: {self.warmup_steps}; measured steps: {len(measured)}",
        ]
        if measured:
            for name in ("step_s", "step_excluding_samples_s"):
                stats = summary[name]
                lines.append(f"{name}: mean={stats['mean']:.3f}, median={stats['median']:.3f}, "
                             f"min={stats['min']:.3f}, max={stats['max']:.3f}")
            lines.append(f"Questions/s: {summary['questions_per_s']:.4f}")
            lines.extend(f"Mean {stage}: {value:.3f}s" for stage, value in summary["stage_mean_s"].items())
        else:
            lines.append("Waiting for completed steps after warmup.")
        lines.extend(["", summary["timing_scope"]])
        _atomic_text(self.output_dir / "summary.txt", "\n".join(lines) + "\n")
