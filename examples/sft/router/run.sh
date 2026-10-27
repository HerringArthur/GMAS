#!/usr/bin/env bash
# Run from the project root: bash examples/sft/router/run.sh DATA_DIR MODEL_DIR OUTPUT_DIR [Hydra overrides...]
set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 DATA_DIR MODEL_DIR OUTPUT_DIR [Hydra overrides...]" >&2
    exit 2
fi

router_data_dir=$1
router_model_dir=$2
router_output_dir=$3
shift 3
router_nproc=${NPROC_PER_NODE:-4}
router_batch_size=${TRAIN_BATCH_SIZE:-32}

python3 - "$router_data_dir" "$router_batch_size" "$router_nproc" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
report = json.loads((root / "report.json").read_text())
if not report["ready_for_training"]:
    raise SystemExit("Check report.json: need token length checks and nonempty train/val splits.")
if report["splits"]["train"]["examples"] < int(sys.argv[2]):
    raise SystemExit("Training dataset is smaller than one global batch; reduce TRAIN_BATCH_SIZE.")
if report["splits"]["val"]["examples"] < int(sys.argv[3]):
    raise SystemExit("Validation dataset is smaller than the GPU count.")
for split in ("train", "val"):
    if not (root / f"{split}.parquet").is_file():
        raise SystemExit("Build the dataset with --parquet first.")
if report["missing_routes"]:
    print("Roles not covered in training:", ", ".join(report["missing_routes"]))
print("Real rollout format-repair dataset checks passed.")
PY

torchrun --standalone --nnodes=1 --nproc_per_node="$router_nproc" \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$router_data_dir/train.parquet" \
    data.val_files="$router_data_dir/val.parquet" \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.train_batch_size="$router_batch_size" \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=8192 \
    data.truncation=error \
    model.partial_pretrain="$router_model_dir" \
    model.enable_gradient_checkpointing=true \
    optim.lr=5e-6 \
    optim.warmup_steps_ratio=0.05 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="$router_output_dir" \
    trainer.default_hdfs_dir=null \
    trainer.project_name=router-sft \
    trainer.experiment_name=search-math-router \
    'trainer.logger=[console]' \
    "$@"
