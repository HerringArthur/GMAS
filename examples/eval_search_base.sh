#!/usr/bin/env bash
# Compare the initial model on the same 210 search questions, one attempt each.
# Usage: CUDA_VISIBLE_DEVICES=2,3,4,5 bash examples/eval_search_base.sh [both|drmas|gmas] [Hydra overrides...]
source /data/workspace/xuhaobin/miniconda3/etc/profile.d/conda.sh
conda activate DrMAS
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
method=${1:-both}
if [ "$#" -gt 0 ]; then
    shift
fi
case "$method" in
    both) methods=(drmas gmas) ;;
    drmas|gmas) methods=("$method") ;;
    *) echo "Usage: $0 [both|drmas|gmas] [Hydra overrides...]" >&2; exit 2 ;;
esac

: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the GPUs selected for evaluation}"
IFS=',' read -r -a eval_gpus <<< "$CUDA_VISIBLE_DEVICES"
model_path=${BASE_MODEL_PATH:-/data/workspace/xuhaobin/xzy/models/Qwen2.5-3B-Instruct}
data_dir=${SEARCH_DATA_DIR:-/data/workspace/xuhaobin/xzy/data/MASsearch/drmas_search}

# Fail before loading model workers if the retriever is unavailable.
config_only=False
for argument in "$@"; do
    case "$argument" in
        --cfg|--help|--hydra-help|--info) config_only=True ;;
    esac
done
if [ "$config_only" == False ]; then
    python - <<'PY'
import requests

response = requests.post(
    "http://127.0.0.1:8001/retrieve",
    json={"query": "Who developed the theory of relativity?", "topk": 3},
    timeout=60,
)
response.raise_for_status()
if len(response.json()["result"][0]) != 3:
    raise RuntimeError("Retriever preflight did not return three documents")
print("Retriever preflight passed", flush=True)
PY
fi

for trainer in "${methods[@]}"; do
    agent_count=3
    if [ "$trainer" == gmas ]; then
        agent_count=4
    fi
    model_ids=$(python -c 'import json, sys; print(json.dumps([sys.argv[1]] * int(sys.argv[2])))' "$model_path" "$agent_count")
    bash "examples/${trainer}_trainer/run_search.sh" eval \
        "agent.model_ids=$model_ids" \
        agent.model_sharing=False \
        "data.val_files=$data_dir/test_sampled.parquet" \
        data.val_batch_size=32 \
        env.rollout.val_n=1 \
        env.seed=0 \
        env.search.search_url=http://127.0.0.1:8001/retrieve \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
        actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
        trainer.logger='["console"]' \
        "trainer.n_gpus_per_node=${#eval_gpus[@]}" \
        "trainer.experiment_name=${trainer}_base_search_210" \
        trainer.val_only=True \
        trainer.val_before_train=True \
        trainer.resume_mode=disable \
        "$@"
done
