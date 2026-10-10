#!/bin/bash
# Submit one integrated DrMAS search training job.
# With START_RETRIEVER=1, two GPUs serve retrieval and six GPUs train.
# With START_RETRIEVER=0, all eight GPUs train against an existing retriever.
#
# Usage:
#   sbatch examples/drmas_trainer/sbatch_search.sh
#   sbatch examples/drmas_trainer/sbatch_search.sh 3B
#   sbatch examples/drmas_trainer/sbatch_search.sh 7B eval
#   sbatch examples/drmas_trainer/sbatch_search.sh eval
#   MODEL_SHARING=False MODEL_IDS='["Qwen3-4B","Qwen3-4B","Qwen3-4B"]' sbatch --export=ALL examples/drmas_trainer/sbatch_search.sh
#
# Optional environment variables:
#   PROJECT_ROOT=/seu_share2/home/fenglei/213243847/SRPO_experiments
#   MODEL_SIZE=7B                     # 3B or 7B; used for shared/default MODEL_PATH
#   MODEL_TAG=custom                  # log tag when MODEL_SIZE does not describe MODEL_IDS
#   MODEL_3B=/path/to/Qwen2.5-3B-Instruct
#   MODEL_7B=/path/to/Qwen2.5-7B-Instruct
#   SEARCH_TRAIN_DATA=/path/to/train.parquet
#   SEARCH_VAL_DATA=/path/to/test_sampled.parquet
#   SEARCH_EVAL_DATA=/path/to/test.parquet
#   SEARCH_DATA=/path/to/searchR1     # e5_Flat.index and wiki-18.jsonl
#   RETRIEVER_MODEL=/path/to/e5-base-v2-real
#   RETRIEVER_SCRIPT=/path/to/retrieval_server.py
#   SEARCH_PORT=8000
#   RETRIEVER_N_GPUS=2                # default: 2 when starting, otherwise 0
#   N_GPUS=6                          # default: 6 when starting, otherwise 8
#   TRAIN_BATCH_SIZE=120              # default: 120 for 6 GPUs, 128 for 8 GPUs
#   GROUP_SIZE=5
#   ACTOR_PPO_MINI_BATCH_SIZE=120
#   ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU=4
#   MODEL_SHARING=True                 # True/False; overrides run_search.sh
#   MODEL_IDS='["Qwen3-4B","/path/to/search_model","Qwen/Qwen2.5-7B"]'
#                                      # required when MODEL_SHARING=False; order is Verifier/Search/Answer
#                                      # bare names resolve to $PROJECT_ROOT/model/<name>
#   ACTOR_OPTIM_LR=1e-6
#   ACTOR_OPTIM_LR_LIST='[1e-6,1e-6,1e-6]'
#   ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU_LIST='[4,4,4]'
#   CHECKPOINT_ROOT=/path/to/checkpoints
#   TRAINER_PROJECT_NAME=DrMAS_search
#   TRAINER_EXPERIMENT_NAME=custom_experiment
#   TRAINER_DEFAULT_LOCAL_DIR=/path/to/checkpoints/DrMAS_search/custom_experiment
#   USE_WANDB=False                   # set True to use W&B
#   WANDB_API_KEY_IN_SCRIPT=your_wandb_api_key
#   WANDB_ENTITY_IN_SCRIPT=your_wandb_entity
#   WANDB_BASE_URL_IN_SCRIPT=https://api.wandb.ai
#   WANDB_MODE_IN_SCRIPT=online
#   CONDA_BASE=/path/to/miniconda3
#   CONDA_ENV_NAME=DrMAS
#   RETRIEVER_CONDA_ENV=retriever
#   CACHE_ROOT=/path/to/cache_root     # HuggingFace/datasets/tmp cache root
#   DRMAS_RAY_TMPDIR=/path/to/ray_tmp
#   RAY_SHORT_TMPDIR=/tmp/dm${SLURM_JOB_ID}
#   RAY_TMPDIR=/path/to/ray_tmp
#   START_RETRIEVER=0                 # use an existing retrieval server

#SBATCH --job-name=drmas_search
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/drmas_search_%j.out
#SBATCH --error=logs/drmas_search_%j.err
#SBATCH --partition=gpuB

set -euo pipefail

# Set USE_WANDB_IN_SCRIPT=True and fill these once if you want W&B with
# plain `sbatch ...` and no exported W&B variables.
USE_WANDB_IN_SCRIPT="False"
WANDB_API_KEY_IN_SCRIPT=""
WANDB_ENTITY_IN_SCRIPT=""
WANDB_BASE_URL_IN_SCRIPT="https://api.wandb.ai"
WANDB_MODE_IN_SCRIPT="online"
WANDB_SILENT_IN_SCRIPT="true"

set -x

MODEL_SIZE=${MODEL_SIZE:-}
MODE=${MODE:-train}
case "${1:-}" in
    3B|3b|7B|7b)
        MODEL_SIZE=$1
        MODE=${2:-$MODE}
        ;;
    "")
        ;;
    *)
        MODE=$1
        ;;
esac
MODEL_TAG=${MODEL_TAG:-}
case "$MODE" in
    train)
        ;;
    eval|evaluation)
        MODE=eval
        ;;
    *)
        echo "MODE must be train, eval, or evaluation; got: $MODE" >&2
        exit 1
        ;;
esac

PROJECT_ROOT=${PROJECT_ROOT:-/seu_share2/home/fenglei/213243847/SRPO_experiments}
CONDA_BASE=${CONDA_BASE:-/data/workspace/xuhaobin/miniconda3}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-DrMAS}
RETRIEVER_CONDA_ENV=${RETRIEVER_CONDA_ENV:-retriever}
START_RETRIEVER=${START_RETRIEVER:-1}
if [[ -n "${SEARCH_URL:-}" && -z "${SEARCH_PORT:-}" && "$SEARCH_URL" =~ :([0-9]+)/retrieve ]]; then
    SEARCH_PORT="${BASH_REMATCH[1]}"
fi
SEARCH_PORT=${SEARCH_PORT:-8000}
SEARCH_URL=${SEARCH_URL:-http://127.0.0.1:${SEARCH_PORT}/retrieve}
case "$START_RETRIEVER" in
    1)
        RETRIEVER_N_GPUS=${RETRIEVER_N_GPUS:-2}
        N_GPUS=${N_GPUS:-6}
        TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-120}
        ;;
    0)
        RETRIEVER_N_GPUS=0
        N_GPUS=${N_GPUS:-8}
        TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
        ;;
    *)
        echo "START_RETRIEVER must be 0 or 1; got: $START_RETRIEVER" >&2
        exit 1
        ;;
esac
RETRIEVER_WAIT_SECONDS=${RETRIEVER_WAIT_SECONDS:-300}
GROUP_SIZE=${GROUP_SIZE:-5}
ACTOR_ROLLOUT_N=1
ACTOR_PPO_MINI_BATCH_SIZE=${ACTOR_PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU=${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-4}
MODEL_SHARING=${MODEL_SHARING:-True}
DEFAULT_AGENT_IDS='["Verifier Agent","Search Agent","Answer Agent"]'
case "$MODEL_SHARING" in
    True|true)
        MODEL_SHARING=True
        ;;
    False|false)
        MODEL_SHARING=False
        ;;
    *)
        echo "MODEL_SHARING must be True or False; got: $MODEL_SHARING" >&2
        exit 1
        ;;
esac

compact_string_array() {
    local name=$1
    local value=$2
    jq -cer --arg name "$name" '
        if type == "array"
           and length > 0
           and ([.[] | select(type != "string" or length == 0)] | length == 0)
        then .
        else error($name + " must be a non-empty JSON array of strings")
        end
    ' <<< "$value"
}

compact_array_with_length() {
    local name=$1
    local value=$2
    local expected_len=$3
    local compacted
    local actual_len
    compacted=$(jq -cer --arg name "$name" '
        if type == "array" and length > 0
        then .
        else error($name + " must be a non-empty JSON array")
        end
    ' <<< "$value")
    actual_len=$(jq -r 'length' <<< "$compacted")
    if (( actual_len != expected_len )); then
        echo "$name length must match fixed search-agent count; got $actual_len vs $expected_len" >&2
        exit 1
    fi
    printf '%s\n' "$compacted"
}

resolve_model_id() {
    local model_id=$1
    local candidate

    case "$model_id" in
        /*|./*|../*)
            if [[ ! -e "$model_id" ]]; then
                echo "Model path from MODEL_IDS does not exist: $model_id" >&2
                return 1
            fi
            printf '%s\n' "$model_id"
            ;;
        */*)
            printf '%s\n' "$model_id"
            ;;
        *)
            candidate="$PROJECT_ROOT/model/$model_id"
            if [[ ! -e "$candidate" ]]; then
                echo "Model alias from MODEL_IDS does not exist under PROJECT_ROOT/model: $model_id" >&2
                echo "Expected path: $candidate" >&2
                return 1
            fi
            printf '%s\n' "$candidate"
            ;;
    esac
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." >/dev/null 2>&1 && pwd)
if [[ ! -f "$REPO_ROOT/examples/drmas_trainer/run_search.sh" ]]; then
    if [[ -n "${DRMAS_ROOT:-}" && -f "$DRMAS_ROOT/examples/drmas_trainer/run_search.sh" ]]; then
        REPO_ROOT="$DRMAS_ROOT"
    elif [[ -f "$PROJECT_ROOT/DrMAS/examples/drmas_trainer/run_search.sh" ]]; then
        REPO_ROOT="$PROJECT_ROOT/DrMAS"
    elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/examples/drmas_trainer/run_search.sh" ]]; then
        REPO_ROOT="$SLURM_SUBMIT_DIR"
    elif [[ -f "$PWD/examples/drmas_trainer/run_search.sh" ]]; then
        REPO_ROOT="$PWD"
    else
        echo "Cannot locate repo root. Set DRMAS_ROOT=/path/to/DrMAS before sbatch." >&2
        exit 1
    fi
fi
cd "$REPO_ROOT"
mkdir -p logs

CACHE_ROOT=${CACHE_ROOT:-/seu_share2/home/fenglei/213243847/DrMAS/cache}
HF_HOME=${HF_HOME:-$CACHE_ROOT/huggingface}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}
if [[ -z "${TMPDIR:-}" || "$TMPDIR" == "/tmp" || "$TMPDIR" == /tmp/* ]]; then
    TMPDIR="$CACHE_ROOT/tmp"
fi
DRMAS_RAY_TMPDIR=${DRMAS_RAY_TMPDIR:-$CACHE_ROOT/ray/${SLURM_JOB_ID:-$$}}
RAY_SHORT_TMPDIR=${RAY_SHORT_TMPDIR:-/tmp/dm${SLURM_JOB_ID:-$$}}
if [[ -z "${RAY_TMPDIR:-}" || "$RAY_TMPDIR" == "/tmp" || "$RAY_TMPDIR" == /tmp/* ]]; then
    RAY_TMPDIR="$RAY_SHORT_TMPDIR"
fi
export CACHE_ROOT HF_HOME HF_DATASETS_CACHE HF_HUB_CACHE TRANSFORMERS_CACHE XDG_CACHE_HOME TMPDIR DRMAS_RAY_TMPDIR RAY_SHORT_TMPDIR RAY_TMPDIR
mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$TMPDIR" "$DRMAS_RAY_TMPDIR"

set +x
USE_WANDB=${USE_WANDB:-$USE_WANDB_IN_SCRIPT}
case "$USE_WANDB" in
    True|true|1|yes|YES)
        USE_WANDB=True
        TRAINER_LOGGER="[console,wandb]"
        WANDB_API_KEY=${WANDB_API_KEY:-$WANDB_API_KEY_IN_SCRIPT}
        WANDB_ENTITY=${WANDB_ENTITY:-$WANDB_ENTITY_IN_SCRIPT}
        WANDB_BASE_URL=${WANDB_BASE_URL:-$WANDB_BASE_URL_IN_SCRIPT}
        WANDB_MODE=${WANDB_MODE:-$WANDB_MODE_IN_SCRIPT}
        WANDB_SILENT=${WANDB_SILENT:-$WANDB_SILENT_IN_SCRIPT}
        if [[ -z "$WANDB_API_KEY" ]]; then
            echo "WANDB_API_KEY is not configured. Fill WANDB_API_KEY_IN_SCRIPT in sbatch_search.sh or export WANDB_API_KEY." >&2
            exit 1
        fi
        export WANDB_API_KEY WANDB_BASE_URL WANDB_MODE WANDB_SILENT
        if [[ -n "$WANDB_ENTITY" ]]; then
            export WANDB_ENTITY
        fi
        ;;
    False|false|0|no|NO)
        USE_WANDB=False
        TRAINER_LOGGER="[console]"
        ;;
    *)
        echo "USE_WANDB must be True or False; got: $USE_WANDB" >&2
        exit 1
        ;;
esac
set -x
if [[ "$USE_WANDB" == "True" ]]; then
    WANDB_DIR=${WANDB_DIR:-$CACHE_ROOT/wandb}
    WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-$WANDB_DIR/cache}
    WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR:-$WANDB_DIR/config}
    export WANDB_DIR WANDB_CACHE_DIR WANDB_CONFIG_DIR
    mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
fi

if [[ "$RAY_TMPDIR" == "$RAY_SHORT_TMPDIR" ]]; then
    if [[ -L "$RAY_SHORT_TMPDIR" ]]; then
        if [[ "$(readlink "$RAY_SHORT_TMPDIR")" != "$DRMAS_RAY_TMPDIR" ]]; then
            echo "RAY_SHORT_TMPDIR already points elsewhere: $RAY_SHORT_TMPDIR -> $(readlink "$RAY_SHORT_TMPDIR")" >&2
            exit 1
        fi
    elif [[ -e "$RAY_SHORT_TMPDIR" ]]; then
        echo "RAY_SHORT_TMPDIR exists and is not a symlink: $RAY_SHORT_TMPDIR" >&2
        exit 1
    else
        ln -s "$DRMAS_RAY_TMPDIR" "$RAY_SHORT_TMPDIR"
    fi
else
    mkdir -p "$RAY_TMPDIR"
fi

if [[ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE=$(conda info --base)
    else
        echo "Cannot locate conda. Set CONDA_BASE=/path/to/miniconda3 before sbatch." >&2
        exit 1
    fi
fi

# Conda hooks can read unset backup variables while activating environments.
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"
set -u

if [[ "$MODEL_SHARING" == "False" && -n "${MODEL_IDS:-}" ]]; then
    MODEL_SIZE=${MODEL_SIZE:-custom}
    MODEL_TAG=${MODEL_TAG:-$MODEL_SIZE}
    MODEL_PATH=""
else
    MODEL_SIZE=${MODEL_SIZE:-7B}
    MODEL_TAG=${MODEL_TAG:-$MODEL_SIZE}
    case "$MODEL_SIZE" in
        3B|3b)
            MODEL_PATH=${MODEL_3B:-$PROJECT_ROOT/model/Qwen2.5-3B-Instruct}
            ;;
        7B|7b)
            MODEL_PATH=${MODEL_7B:-$PROJECT_ROOT/model/Qwen2.5-7B-Instruct}
            ;;
        *)
            echo "MODEL_SIZE must be 3B or 7B when MODEL_IDS is not provided; got: $MODEL_SIZE" >&2
            exit 1
            ;;
    esac
fi

if ! command -v jq >/dev/null 2>&1; then
    echo "jq is required to validate MODEL_IDS and generated agent-specific lists." >&2
    exit 1
fi
if [[ "$MODEL_SHARING" == "False" && -z "${MODEL_IDS:-}" ]]; then
    echo "When MODEL_SHARING=False, set MODEL_IDS as a JSON array in fixed search-agent order: Verifier, Search, Answer." >&2
    echo "Example: MODEL_IDS='[\"Qwen3-4B\",\"Qwen3-4B\",\"Qwen3-4B\"]'" >&2
    exit 1
fi
EFFECTIVE_AGENT_IDS=$DEFAULT_AGENT_IDS
AGENT_COUNT=$(jq -r 'length' <<< "$EFFECTIVE_AGENT_IDS")
if [[ -n "${MODEL_IDS:-}" ]]; then
    MODEL_IDS_INPUT=$(compact_string_array "MODEL_IDS" "$MODEL_IDS")
else
    MODEL_IDS_INPUT=$(jq -cn --arg model "$MODEL_PATH" --argjson agents "$EFFECTIVE_AGENT_IDS" '$agents | map($model)')
fi
MODEL_COUNT=$(jq -r 'length' <<< "$MODEL_IDS_INPUT")
if (( AGENT_COUNT != MODEL_COUNT )); then
    echo "MODEL_IDS must contain exactly $AGENT_COUNT entries in fixed search-agent order: Verifier, Search, Answer; got $MODEL_COUNT" >&2
    exit 1
fi
RESOLVED_MODEL_ID_ITEMS=()
while IFS= read -r model_id; do
    resolved_model_id=$(resolve_model_id "$model_id")
    RESOLVED_MODEL_ID_ITEMS+=("$resolved_model_id")
done < <(jq -r '.[]' <<< "$MODEL_IDS_INPUT")
EFFECTIVE_MODEL_IDS=$(printf '%s\n' "${RESOLVED_MODEL_ID_ITEMS[@]}" | jq -Rsc 'split("\n")[:-1]')
ACTOR_OPTIM_LR=${ACTOR_OPTIM_LR:-1e-6}
ACTOR_OPTIM_LR_LIST=${ACTOR_OPTIM_LR_LIST:-$(jq -cn --argjson n "$AGENT_COUNT" --argjson lr "$ACTOR_OPTIM_LR" '[range(0; $n) | $lr]')}
ACTOR_OPTIM_LR_LIST=$(compact_array_with_length "ACTOR_OPTIM_LR_LIST" "$ACTOR_OPTIM_LR_LIST" "$AGENT_COUNT")
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU_LIST=${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU_LIST:-$(jq -cn --argjson n "$AGENT_COUNT" --argjson micro "$ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU" '[range(0; $n) | $micro]')}
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU_LIST=$(compact_array_with_length "ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU_LIST" "$ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU_LIST" "$AGENT_COUNT")
MODEL_NAME_TAG=$(jq -r '[.[] | split("/")[-1] | ascii_downcase | gsub("-"; "_")] | join("_")' <<< "$EFFECTIVE_MODEL_IDS")
TRAINER_PROJECT_NAME=${TRAINER_PROJECT_NAME:-DrMAS_search}
if [[ -n "${EXPERIMENT_NAME:-}" && -z "${TRAINER_EXPERIMENT_NAME:-}" ]]; then
    TRAINER_EXPERIMENT_NAME=$EXPERIMENT_NAME
fi
TRAINER_EXPERIMENT_NAME=${TRAINER_EXPERIMENT_NAME:-drmas_share${MODEL_SHARING}_${MODEL_NAME_TAG}}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/seu_share2/home/fenglei/213243847/DrMAS/checkpoints}
TRAINER_DEFAULT_LOCAL_DIR=${TRAINER_DEFAULT_LOCAL_DIR:-$CHECKPOINT_ROOT/$TRAINER_PROJECT_NAME/$TRAINER_EXPERIMENT_NAME}
mkdir -p "$TRAINER_DEFAULT_LOCAL_DIR"

SEARCH_TRAIN_DATA=${SEARCH_TRAIN_DATA:-$PROJECT_ROOT/data/drmas_search/train.parquet}
SEARCH_VAL_DATA=${SEARCH_VAL_DATA:-$PROJECT_ROOT/data/drmas_search/test_sampled.parquet}
SEARCH_EVAL_DATA=${SEARCH_EVAL_DATA:-$PROJECT_ROOT/data/drmas_search/test.parquet}
if [[ "$MODE" == "eval" ]]; then
    RUN_VAL_DATA="$SEARCH_EVAL_DATA"
else
    RUN_VAL_DATA="$SEARCH_VAL_DATA"
fi

REQUIRED_TRAINING_PATHS=("$SEARCH_TRAIN_DATA" "$RUN_VAL_DATA")
while IFS= read -r model_id; do
    if [[ "$model_id" == /* ]]; then
        REQUIRED_TRAINING_PATHS+=("$model_id")
    fi
done < <(jq -r '.[]' <<< "$EFFECTIVE_MODEL_IDS")
for path in "${REQUIRED_TRAINING_PATHS[@]}"; do
    if [[ ! -e "$path" ]]; then
        echo "Required training path does not exist: $path" >&2
        exit 1
    fi
done

ALLOCATED_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
IFS=',' read -r -a CUDA_DEVICES <<< "$ALLOCATED_CUDA_VISIBLE_DEVICES"
if (( RETRIEVER_N_GPUS < 0 || N_GPUS < 1 )); then
    echo "RETRIEVER_N_GPUS must be non-negative and N_GPUS must be positive" >&2
    exit 1
fi
if (( TRAIN_BATCH_SIZE < 1 || TRAIN_BATCH_SIZE % N_GPUS != 0 )); then
    echo "TRAIN_BATCH_SIZE must be positive and divisible by N_GPUS; got: $TRAIN_BATCH_SIZE" >&2
    exit 1
fi
NORMALIZED_ACTOR_PPO_MINI_BATCH_SIZE=$(( ACTOR_PPO_MINI_BATCH_SIZE * ACTOR_ROLLOUT_N / N_GPUS ))
if (( NORMALIZED_ACTOR_PPO_MINI_BATCH_SIZE < ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU || NORMALIZED_ACTOR_PPO_MINI_BATCH_SIZE % ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU != 0 )); then
    echo "ACTOR_PPO_MINI_BATCH_SIZE * ACTOR_ROLLOUT_N / N_GPUS must be divisible by ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU after integer normalization; got normalized mini batch $NORMALIZED_ACTOR_PPO_MINI_BATCH_SIZE and micro batch $ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU" >&2
    exit 1
fi
if (( ${#CUDA_DEVICES[@]} < RETRIEVER_N_GPUS + N_GPUS )); then
    echo "Need $((RETRIEVER_N_GPUS + N_GPUS)) visible GPUs, got CUDA_VISIBLE_DEVICES=$ALLOCATED_CUDA_VISIBLE_DEVICES" >&2
    exit 1
fi
if (( RETRIEVER_N_GPUS == 0 )); then
    RETRIEVER_CUDA_VISIBLE_DEVICES=""
elif [[ -z "${RETRIEVER_CUDA_VISIBLE_DEVICES:-}" ]]; then
    RETRIEVER_CUDA_VISIBLE_DEVICES=$(IFS=','; echo "${CUDA_DEVICES[*]:0:RETRIEVER_N_GPUS}")
fi
if [[ -z "${TRAIN_CUDA_VISIBLE_DEVICES:-}" ]]; then
    TRAIN_CUDA_VISIBLE_DEVICES=$(IFS=','; echo "${CUDA_DEVICES[*]:RETRIEVER_N_GPUS:N_GPUS}")
fi

check_retriever() {
    python3 - "$SEARCH_URL" <<'PY'
import json
import sys
import urllib.request

url = sys.argv[1]
payload = json.dumps({"query": "test", "topk": 1}).encode("utf-8")
request = urllib.request.Request(
    url,
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(request, timeout=10) as response:
        response.read(256)
except Exception as exc:
    raise SystemExit(f"Retriever check failed for {url}: {exc}")
PY
}

RETRIEVER_PID=""
cleanup() {
    if [[ -n "$RETRIEVER_PID" ]] && kill -0 "$RETRIEVER_PID" 2>/dev/null; then
        kill "$RETRIEVER_PID" 2>/dev/null || true
        wait "$RETRIEVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ "$START_RETRIEVER" == "1" ]]; then
    RETRIEVER_SCRIPT=${RETRIEVER_SCRIPT:-$REPO_ROOT/examples/search/retriever/retrieval_server.py}
    SEARCH_DATA=${SEARCH_DATA:-$PROJECT_ROOT/data/searchR1}
    RETRIEVER_INDEX=${RETRIEVER_INDEX:-$SEARCH_DATA/e5_Flat.index}
    RETRIEVER_CORPUS=${RETRIEVER_CORPUS:-$SEARCH_DATA/wiki-18.jsonl}
    if [[ -z "${RETRIEVER_MODEL:-}" ]]; then
        if [[ -n "${MODELS:-}" ]]; then
            RETRIEVER_MODEL="$MODELS/e5-base-v2-real"
        else
            RETRIEVER_MODEL="$PROJECT_ROOT/model/e5-base-v2-real"
        fi
    fi
    RETRIEVER_LOG=${RETRIEVER_LOG:-logs/retrieval_server_${SLURM_JOB_ID:-manual}_${MODEL_TAG}.log}

    for path in "$RETRIEVER_SCRIPT" "$RETRIEVER_INDEX" "$RETRIEVER_CORPUS" "$RETRIEVER_MODEL"; do
        if [[ ! -e "$path" ]]; then
            echo "Required retriever path does not exist: $path" >&2
            exit 1
        fi
    done

    CUDA_VISIBLE_DEVICES="$RETRIEVER_CUDA_VISIBLE_DEVICES" \
    conda run -n "$RETRIEVER_CONDA_ENV" python "$RETRIEVER_SCRIPT" \
        --index_path "$RETRIEVER_INDEX" \
        --corpus_path "$RETRIEVER_CORPUS" \
        --topk 3 \
        --retriever_name e5 \
        --retriever_model "$RETRIEVER_MODEL" \
        --faiss_gpu \
        --port "$SEARCH_PORT" > "$RETRIEVER_LOG" 2>&1 &
    RETRIEVER_PID=$!

    RETRIEVER_READY=0
    for _ in $(seq 1 "$RETRIEVER_WAIT_SECONDS"); do
        if check_retriever >/dev/null 2>&1; then
            RETRIEVER_READY=1
            break
        fi
        if ! kill -0 "$RETRIEVER_PID" 2>/dev/null; then
            echo "Retriever exited before becoming ready. Last log lines:" >&2
            tail -n 80 "$RETRIEVER_LOG" >&2 || true
            exit 1
        fi
        sleep 1
    done
    if [[ "$RETRIEVER_READY" != "1" ]]; then
        echo "Retriever did not become ready within ${RETRIEVER_WAIT_SECONDS}s. Last log lines:" >&2
        tail -n 80 "$RETRIEVER_LOG" >&2 || true
        exit 1
    fi
fi
check_retriever

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "ALLOCATED_CUDA_VISIBLE_DEVICES=$ALLOCATED_CUDA_VISIBLE_DEVICES"
echo "RETRIEVER_N_GPUS=$RETRIEVER_N_GPUS"
echo "RETRIEVER_CUDA_VISIBLE_DEVICES=$RETRIEVER_CUDA_VISIBLE_DEVICES"
echo "TRAIN_CUDA_VISIBLE_DEVICES=$TRAIN_CUDA_VISIBLE_DEVICES"
echo "MODEL_SIZE=$MODEL_SIZE"
echo "MODEL_TAG=$MODEL_TAG"
if [[ -n "$MODEL_PATH" ]]; then
    echo "MODEL_PATH=$MODEL_PATH"
fi
echo "TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE"
echo "GROUP_SIZE=$GROUP_SIZE"
echo "ACTOR_ROLLOUT_N=$ACTOR_ROLLOUT_N"
echo "ACTOR_PPO_MINI_BATCH_SIZE=$ACTOR_PPO_MINI_BATCH_SIZE"
echo "ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU=$ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU"
echo "NORMALIZED_ACTOR_PPO_MINI_BATCH_SIZE=$NORMALIZED_ACTOR_PPO_MINI_BATCH_SIZE"
echo "MODEL_SHARING=$MODEL_SHARING"
echo "AGENT_COUNT=$AGENT_COUNT"
echo "EFFECTIVE_AGENT_IDS=$EFFECTIVE_AGENT_IDS"
echo "MODEL_IDS_INPUT=$MODEL_IDS_INPUT"
echo "EFFECTIVE_MODEL_IDS=$EFFECTIVE_MODEL_IDS"
echo "ACTOR_OPTIM_LR_LIST=$ACTOR_OPTIM_LR_LIST"
echo "ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU_LIST=$ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU_LIST"
echo "TRAINER_PROJECT_NAME=$TRAINER_PROJECT_NAME"
echo "TRAINER_EXPERIMENT_NAME=$TRAINER_EXPERIMENT_NAME"
echo "TRAINER_DEFAULT_LOCAL_DIR=$TRAINER_DEFAULT_LOCAL_DIR"
echo "SEARCH_TRAIN_DATA=$SEARCH_TRAIN_DATA"
echo "RUN_VAL_DATA=$RUN_VAL_DATA"
echo "SEARCH_URL=$SEARCH_URL"
echo "CACHE_ROOT=$CACHE_ROOT"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "DRMAS_RAY_TMPDIR=$DRMAS_RAY_TMPDIR"
echo "RAY_SHORT_TMPDIR=$RAY_SHORT_TMPDIR"
echo "RAY_TMPDIR=$RAY_TMPDIR"
echo "TMPDIR=$TMPDIR"
echo "USE_WANDB=$USE_WANDB"
echo "TRAINER_LOGGER=$TRAINER_LOGGER"
if [[ "$USE_WANDB" == "True" ]]; then
    echo "WANDB_API_KEY=configured"
    echo "WANDB_ENTITY=${WANDB_ENTITY:-}"
    echo "WANDB_BASE_URL=$WANDB_BASE_URL"
    echo "WANDB_MODE=$WANDB_MODE"
    echo "WANDB_DIR=$WANDB_DIR"
fi

# run_search.sh fixes these values internally. Intercept only its trainer launch
# and append Hydra overrides so the original script remains unchanged.
DRMAS_PYTHON_BIN=$(command -v python3)
export DRMAS_PYTHON_BIN TRAIN_BATCH_SIZE SEARCH_TRAIN_DATA RUN_VAL_DATA SEARCH_URL N_GPUS GROUP_SIZE ACTOR_ROLLOUT_N ACTOR_PPO_MINI_BATCH_SIZE MODEL_SHARING EFFECTIVE_AGENT_IDS EFFECTIVE_MODEL_IDS ACTOR_OPTIM_LR_LIST ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU_LIST TRAINER_PROJECT_NAME TRAINER_EXPERIMENT_NAME TRAINER_DEFAULT_LOCAL_DIR TRAINER_LOGGER
python3() {
    if [[ "${1:-}" == "-m" && "${2:-}" == "verl.trainer.main_ppo" ]]; then
        "$DRMAS_PYTHON_BIN" "$@" \
            "data.train_files=$SEARCH_TRAIN_DATA" \
            "data.val_files=$RUN_VAL_DATA" \
            "data.train_batch_size=$TRAIN_BATCH_SIZE" \
            "agent.agent_ids=$EFFECTIVE_AGENT_IDS" \
            "agent.model_ids=$EFFECTIVE_MODEL_IDS" \
            "agent.model_sharing=$MODEL_SHARING" \
            "agent.agent_specific_parameters.actor.optim.lr=$ACTOR_OPTIM_LR_LIST" \
            "env.search.search_url=$SEARCH_URL" \
            "env.rollout.n=$GROUP_SIZE" \
            "actor_rollout_ref.rollout.n=$ACTOR_ROLLOUT_N" \
            "actor_rollout_ref.actor.ppo_mini_batch_size=$ACTOR_PPO_MINI_BATCH_SIZE" \
            "agent.agent_specific_parameters.actor.ppo_micro_batch_size_per_gpu=$ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU_LIST" \
            "trainer.logger=$TRAINER_LOGGER" \
            "trainer.project_name=$TRAINER_PROJECT_NAME" \
            "trainer.experiment_name=$TRAINER_EXPERIMENT_NAME" \
            "trainer.default_local_dir=$TRAINER_DEFAULT_LOCAL_DIR" \
            "trainer.n_gpus_per_node=$N_GPUS"
    else
        "$DRMAS_PYTHON_BIN" "$@"
    fi
}
export -f python3

TRAINER_LOG_MODEL_IDS="$EFFECTIVE_MODEL_IDS" \
CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" \
bash examples/drmas_trainer/run_search.sh "$MODE"
