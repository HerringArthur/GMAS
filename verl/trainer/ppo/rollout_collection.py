"""Collect complete collaboration trajectories with fixed model weights."""

from pathlib import Path
from time import perf_counter

from verl.trainer.ppo.benchmark import TrainingBenchmark


def configure_rollout_only(config):
    """Disable training services before models, environments and datasets load."""
    trainer = config["trainer"]
    if not trainer.get("rollout_only", False):
        return
    settings = trainer["benchmark"]
    if not settings.get("output_dir") or int(settings.get("samples_per_step", 0)) <= 0:
        raise ValueError("rollout_only requires benchmark.output_dir and samples_per_step > 0")
    output = Path(settings["output_dir"])
    if (output / "metrics.jsonl").exists() or (output / "rollout_samples").exists():
        raise FileExistsError(f"Collection already exists in {output}; use a fresh output_dir")
    actor_rollout = config["actor_rollout_ref"]
    if actor_rollout["actor"]["strategy"] not in ("fsdp", "fsdp2"):
        raise ValueError("rollout_only currently requires FSDP workers")
    if actor_rollout["rollout"].get("mode", "sync") != "sync":
        raise ValueError("rollout_only currently requires synchronous rollout workers")

    # Always load the explicitly configured agent.model_ids, never auto-resume
    # training state. fit() takes a separate path even if save_freq was supplied.
    trainer["resume_mode"] = "disable"
    trainer["resume_from_path"] = None
    trainer["save_freq"] = -1
    trainer["val_only"] = False
    trainer["val_before_train"] = False
    trainer["test_freq"] = -1
    config["reward_model"]["enable"] = False
    config["algorithm"]["use_kl_in_reward"] = False
    config["algorithm"]["filter_groups"]["enable"] = False
    actor_rollout["actor"]["use_kl_loss"] = False


def collect_rollouts(config, dataloader, tokenizers, generate):
    """Run generation and recording only; generate returns (rollout, question_count).

    The callback uses the normal collaboration collector and environment outcome
    evaluator. No optimizer, reward-training, validation or checkpoint callbacks
    are part of this loop.
    """
    trainer = config["trainer"]
    epochs = int(trainer["total_epochs"])
    available_steps = len(dataloader) * epochs
    requested = trainer.get("total_training_steps")
    steps = available_steps if requested is None else int(requested)
    if not 0 < steps <= available_steps:
        raise ValueError(f"Collection requests {steps} batches but dataset/epochs provide {available_steps}")
    if int(trainer["benchmark"]["samples_per_step"]) <= 0:
        raise ValueError("Collection requires samples_per_step > 0")
    recorder = TrainingBenchmark(config, steps)
    print(f"Collecting {steps} batches with fixed models; no parameter updates or checkpoints.", flush=True)
    step = 0
    for epoch in range(epochs):
        for batch_dict in dataloader:
            step += 1
            start = perf_counter()
            rollout, question_count = generate(batch_dict)
            generation_end = perf_counter()
            metrics = recorder.record_rollouts(rollout, tokenizers, step, question_count)
            dump_end = perf_counter()
            metrics.update({
                "collection/step": step, "collection/epoch": epoch,
                "collection/parameter_updates": 0,
                "timing_s/gen": generation_end - start,
                "timing_s/dump_rollout_samples": dump_end - generation_end,
                "timing_s/step": dump_end - start,
            })
            recorder.record_step(metrics, step, training_batch=None)
            if step == steps:
                return
