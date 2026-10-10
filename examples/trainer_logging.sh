#!/usr/bin/env bash
# Shared console logging for the DrMAS and GMAS launchers.
# Usage: run_with_trainer_log drmas|gmas train|eval command [arguments...]
run_with_trainer_log() (
    set -o pipefail
    local trainer=$1
    local mode=$2
    shift 2
    local repo_root log_file
    repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || return
    log_file=$(python - "$repo_root" "$trainer" "$mode" "$@" <<'PY'
import hashlib
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from hydra.core.override_parser.overrides_parser import OverridesParser

repo_root, trainer, mode = sys.argv[1:4]
if trainer not in ("drmas", "gmas"):
    raise ValueError(f"Unknown trainer: {trainer}")
mode = "eval" if mode in ("eval", "evaluation") else "train"
parser = OverridesParser.create()
model_ids = None
for argument in sys.argv[4:]:
    key = argument.lstrip("+").partition("=")[0]
    if key == "agent.model_ids":
        model_ids = parser.parse_override(argument).value()
    elif key == "trainer.val_only":
        mode = "eval" if parser.parse_override(argument).value() else "train"

# DrMAS Slurm wrappers append their effective model IDs inside a Python shell
# function, so the outer launcher receives these IDs through the environment.
if os.environ.get("TRAINER_LOG_MODEL_IDS"):
    model_ids = parser.parse_override("agent.model_ids=" + os.environ["TRAINER_LOG_MODEL_IDS"]).value()
if not isinstance(model_ids, list) or not model_ids:
    raise ValueError("Logging requires a non-empty agent.model_ids list")

names = []
for model_id in model_ids:
    name = str(model_id).rstrip("/").rsplit("/", 1)[-1]
    name = re.sub(r"[^A-Za-z0-9._+-]", "_", name)
    if not name:
        raise ValueError(f"Cannot derive a log name from model ID: {model_id!r}")
    if name not in names:
        names.append(name)
model_tag = "+".join(names)
if len(model_tag) > 180:
    digest = hashlib.sha256(model_tag.encode()).hexdigest()[:8]
    model_tag = model_tag[:160] + "_" + digest

log_dir = Path(repo_root) / "logs" / f"{trainer}_trainer"
log_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
log_file = log_dir / f"{mode}_{model_tag}_{timestamp}.log"
log_file.touch(exist_ok=False)
print(log_file)
PY
    ) || return

    export PYTHONUNBUFFERED=1
    {
        printf 'Log file: %s\n' "$log_file"
        "$@"
    } 2>&1 | tee -a "$log_file"
)
