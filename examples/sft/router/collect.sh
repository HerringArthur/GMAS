#!/usr/bin/env bash
# Run in an allocated GPU job with the GMAS environment already activated.
# Usage: bash examples/sft/router/collect.sh search|math OUTPUT_DIR [Hydra overrides...]
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 search|math OUTPUT_DIR [Hydra overrides...]" >&2
    exit 2
fi
router_scene=$1
router_collect_dir=$2
shift 2
case "$router_scene" in
    search|math) ;;
    *) echo "Expected search or math, got: $router_scene" >&2; exit 2 ;;
esac

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
exec bash "examples/gmas_trainer/run_${router_scene}.sh" train \
    trainer.total_training_steps="${COLLECT_STEPS:-12}" \
    trainer.benchmark.output_dir="$router_collect_dir" \
    trainer.benchmark.samples_per_step="${SAMPLES_PER_STEP:-128}" \
    trainer.benchmark.warmup_steps=0 \
    'trainer.logger=[console]' \
    "$@" \
    trainer.rollout_only=True
