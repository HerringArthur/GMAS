#!/bin/bash
# Submit one integrated GMAS math training job.
# By default, allocate eight GPUs and initialize four independent agents from Qwen3-4B.
#
# Usage (run mkdir before sbatch so Slurm can open its log files):
#   mkdir -p logs
#   sbatch examples/gmas_trainer/sbatch_math.sh
#   sbatch examples/gmas_trainer/sbatch_math.sh 4B
#   sbatch examples/gmas_trainer/sbatch_math.sh 8B eval
#   sbatch examples/gmas_trainer/sbatch_math.sh eval
#   sbatch examples/gmas_trainer/sbatch_math.sh 4B train trainer.total_epochs=5
#   MODEL_IDS=Qwen3-8B,Qwen3-4B,Qwen3-4B,Qwen3-4B sbatch examples/gmas_trainer/sbatch_math.sh
#   N_GPUS=4 sbatch --gres=gpu:4 examples/gmas_trainer/sbatch_math.sh
#
# Optional environment variables:
#   PROJECT_ROOT=/seu_share2/home/fenglei/213243847/SRPO_experiments
#   GMAS_ROOT=/path/to/OrchestRL        # repository path; DRMAS_ROOT is also accepted
#   MODEL_SIZE=4B                      # 4B or 8B; used only when MODEL_IDS is unset
#   MODEL_PATH=/path/to/model          # one checkpoint initializes all four agents
#   MODEL_4B=/path/to/Qwen3-4B
#   MODEL_8B=/path/to/Qwen3-8B
#   MODEL_IDS='["/path/to/router","/path/to/sub1","/path/to/sub2","/path/to/sub3"]'
#                                      # JSON array or comma-separated list of exactly four models
#                                      # order: Router Agent, Sub Agent 1, Sub Agent 2, Sub Agent 3
#                                      # short names resolve to $PROJECT_ROOT/model/<name>
#                                      # org/model Hugging Face IDs pass through unchanged
#   MODEL_SHARING=False                # match run_math.sh; identical paths still train independently
#   MATH_TRAIN_DATA=/path/to/train.parquet
#   MATH_VAL_DATA=/path/to/test_sampled.parquet
#   MATH_EVAL_DATA=/path/to/test.parquet
#   N_GPUS=8                          # change --gres separately when requesting fewer GPUs
#   TRAIN_BATCH_SIZE=32
#   USE_WANDB=1                       # default: 0
#   TRAINER_LOGGER='["console","wandb"]'
#   EXPERIMENT_NAME=my_gmas_run        # default reflects resolved models and MODEL_SHARING
#   GMAS_RUNTIME_ROOT=/seu_share2/home/fenglei/213243847/GMAS
#   GMAS_TMPDIR=/path/to/tmp
#   GMAS_RAY_TMPDIR=/path/to/ray_root  # backing dir; Ray appends ray/
#   RAY_SHORT_TMPDIR=/tmp/gm12345      # short symlink path for Ray sockets
#   CONDA_BASE=/path/to/miniconda3
#   CONDA_ENV_NAME=DrMAS               # reuse the same environment as the DrMAS job
#
# Extra Hydra overrides are passed last and take precedence over these defaults.
# Use the environment variables above for paths/GPU counts to keep preflight checks aligned.

#SBATCH --job-name=gmas_math
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/gmas_math_%j.out
#SBATCH --error=logs/gmas_math_%j.err
#SBATCH --partition=gpuB

set -euo pipefail

# W&B settings embedded in this script. Fill SCRIPT_WANDB_API_KEY on the
# cluster copy if you want sbatch jobs to authenticate without wandb login.
# Do not commit a real API key to a shared repository.
SCRIPT_USE_WANDB=0
SCRIPT_WANDB_API_KEY=""
SCRIPT_WANDB_ENTITY=""
SCRIPT_WANDB_BASE_URL=""
SCRIPT_WANDB_MODE="online"

set -x

configure_wandb() {
    local xtrace_enabled=0
    case "$-" in
        *x*)
            xtrace_enabled=1
            set +x
            ;;
    esac

    if [[ "$USE_WANDB" == "1" && -n "$SCRIPT_WANDB_API_KEY" ]]; then
        export WANDB_API_KEY="$SCRIPT_WANDB_API_KEY"
    fi
    if [[ "$USE_WANDB" == "1" && -n "$SCRIPT_WANDB_ENTITY" ]]; then
        export WANDB_ENTITY="$SCRIPT_WANDB_ENTITY"
    fi
    if [[ "$USE_WANDB" == "1" && -n "$SCRIPT_WANDB_BASE_URL" ]]; then
        export WANDB_BASE_URL="$SCRIPT_WANDB_BASE_URL"
    fi
    if [[ "$USE_WANDB" == "1" && -z "${WANDB_MODE:-}" ]]; then
        export WANDB_MODE="$SCRIPT_WANDB_MODE"
    fi

    if (( xtrace_enabled )); then
        set -x
    fi
}

MODEL_SIZE=${MODEL_SIZE:-}
MODE=${MODE:-train}
MODEL_SHARING=${MODEL_SHARING:-False}
MODEL_IDS=${MODEL_IDS:-}
USE_WANDB=${USE_WANDB:-$SCRIPT_USE_WANDB}
MATH_AGENT_IDS='["Router Agent","Sub Agent 1","Sub Agent 2","Sub Agent 3"]'

case "${1:-}" in
    4B|4b|8B|8b)
        MODEL_SIZE=$1
        shift
        ;;
esac
case "${1:-}" in
    train|eval|evaluation)
        MODE=$1
        shift
        ;;
    ""|*=*)
        ;;
    *)
        echo "Usage: $0 [4B|8B] [train|eval|evaluation] [Hydra overrides...]" >&2
        exit 2
        ;;
esac
case "$MODE" in
    train) ;;
    eval|evaluation) MODE=eval ;;
    *)
        echo "MODE must be train, eval, or evaluation; got: $MODE" >&2
        exit 1
        ;;
esac
case "$MODEL_SHARING" in
    True|true) MODEL_SHARING=True ;;
    False|false) MODEL_SHARING=False ;;
    *)
        echo "MODEL_SHARING must be True or False; got: $MODEL_SHARING" >&2
        exit 1
        ;;
esac
case "$USE_WANDB" in
    1|true|True|yes|YES) USE_WANDB=1 ;;
    0|false|False|no|NO) USE_WANDB=0 ;;
    *)
        echo "USE_WANDB must be 0 or 1; got: $USE_WANDB" >&2
        exit 1
        ;;
esac
if [[ -z "${TRAINER_LOGGER:-}" ]]; then
    if [[ "$USE_WANDB" == "1" ]]; then
        TRAINER_LOGGER='["console","wandb"]'
    else
        TRAINER_LOGGER='["console"]'
    fi
fi

PROJECT_ROOT=${PROJECT_ROOT:-/seu_share2/home/fenglei/213243847/SRPO_experiments}
CONDA_BASE=${CONDA_BASE:-/data/workspace/xuhaobin/miniconda3}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-DrMAS}
N_GPUS=${N_GPUS:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
GMAS_RUNTIME_ROOT=${GMAS_RUNTIME_ROOT:-/seu_share2/home/fenglei/213243847/GMAS}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-$GMAS_RUNTIME_ROOT/cache}
HF_HOME=${HF_HOME:-$GMAS_RUNTIME_ROOT/cache/huggingface}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
GMAS_TMPDIR=${GMAS_TMPDIR:-$GMAS_RUNTIME_ROOT/tmp}
GMAS_RAY_TMPDIR=${GMAS_RAY_TMPDIR:-$GMAS_TMPDIR}
RAY_SHORT_TMPDIR=${RAY_SHORT_TMPDIR:-/tmp/gm${SLURM_JOB_ID:-$$}}
TMPDIR=$GMAS_TMPDIR
RAY_TMPDIR=$RAY_SHORT_TMPDIR
WANDB_DIR=${WANDB_DIR:-$GMAS_RUNTIME_ROOT/wandb}
WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-$WANDB_DIR/cache}
WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR:-$WANDB_DIR/config}
export XDG_CACHE_HOME HF_HOME HF_DATASETS_CACHE HF_HUB_CACHE GMAS_TMPDIR GMAS_RAY_TMPDIR RAY_SHORT_TMPDIR TMPDIR RAY_TMPDIR WANDB_DIR WANDB_CACHE_DIR WANDB_CONFIG_DIR
configure_wandb

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
GMAS_ROOT=${GMAS_ROOT:-${DRMAS_ROOT:-}}
if [[ -n "$GMAS_ROOT" ]]; then
    REPO_ROOT="$GMAS_ROOT"
else
    REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." >/dev/null 2>&1 && pwd)
    if [[ ! -f "$REPO_ROOT/examples/gmas_trainer/run_math.sh" ]]; then
        for candidate in "${SLURM_SUBMIT_DIR:-$PWD}" "$PROJECT_ROOT/DrMAS" "$PROJECT_ROOT/OrchestRL" "$PWD"; do
            if [[ -f "$candidate/examples/gmas_trainer/run_math.sh" ]]; then
                REPO_ROOT="$candidate"
                break
            fi
        done
    fi
fi
if [[ ! -f "$REPO_ROOT/examples/gmas_trainer/run_math.sh" ]]; then
    echo "Cannot locate repo root. Set GMAS_ROOT=/path/to/OrchestRL before sbatch." >&2
    exit 1
fi
cd "$REPO_ROOT"
mkdir -p logs
mkdir -p "$XDG_CACHE_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$GMAS_TMPDIR" "$GMAS_RAY_TMPDIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
if [[ -e "$RAY_TMPDIR" && ! -L "$RAY_TMPDIR" ]]; then
    echo "RAY_TMPDIR must be a short symlink path, but it already exists and is not a symlink: $RAY_TMPDIR" >&2
    exit 1
fi
if [[ -L "$RAY_TMPDIR" ]]; then
    unlink "$RAY_TMPDIR"
fi
ln -s "$GMAS_RAY_TMPDIR" "$RAY_TMPDIR"
mkdir -p "$RAY_TMPDIR/ray"

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
if ! command -v jq >/dev/null 2>&1; then
    echo "jq is required by GMAS run_math.sh and for validating MODEL_IDS." >&2
    exit 1
fi

MATH_TRAIN_DATA=${MATH_TRAIN_DATA:-$PROJECT_ROOT/data/drmas_math/train.parquet}
MATH_VAL_DATA=${MATH_VAL_DATA:-$PROJECT_ROOT/data/drmas_math/test_sampled.parquet}
MATH_EVAL_DATA=${MATH_EVAL_DATA:-$PROJECT_ROOT/data/drmas_math/test.parquet}
if [[ "$MODE" == "eval" ]]; then
    RUN_VAL_DATA="$MATH_EVAL_DATA"
else
    RUN_VAL_DATA="$MATH_VAL_DATA"
fi

trim_model_id() {
    local value=$1
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s\n' "$value"
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
                echo "Model alias from MODEL_IDS does not exist: $candidate" >&2
                return 1
            fi
            printf '%s\n' "$candidate"
            ;;
    esac
}

if [[ -z "$MODEL_IDS" ]]; then
    MODEL_SIZE=${MODEL_SIZE:-4B}
    case "$MODEL_SIZE" in
        4B|4b) MODEL_PATH=${MODEL_PATH:-${MODEL_4B:-$PROJECT_ROOT/model/Qwen3-4B}} ;;
        8B|8b) MODEL_PATH=${MODEL_PATH:-${MODEL_8B:-$PROJECT_ROOT/model/Qwen3-8B}} ;;
        *)
            echo "MODEL_SIZE must be 4B or 8B when MODEL_IDS is unset; got: $MODEL_SIZE" >&2
            exit 1
            ;;
    esac
    if [[ ! -e "$MODEL_PATH" ]]; then
        echo "Model path does not exist: $MODEL_PATH" >&2
        exit 1
    fi
    RESOLVED_MODEL_IDS=$(jq -cn --arg model "$MODEL_PATH" '[$model, $model, $model, $model]')
else
    MODEL_IDS_TRIMMED=$(trim_model_id "$MODEL_IDS")
    if [[ "$MODEL_IDS_TRIMMED" == \[* ]]; then
        PARSED_MODEL_IDS="$MODEL_IDS_TRIMMED"
    else
        PARSED_MODEL_IDS=$(jq -cn --arg models "$MODEL_IDS_TRIMMED" '$models | split(",")')
    fi
    jq -er 'if type == "array" and length == 4 and all(.[]; type == "string" and length > 0) then true else error("MODEL_IDS must contain exactly four non-empty strings: [router, sub1, sub2, sub3]") end' <<< "$PARSED_MODEL_IDS" >/dev/null
    RESOLVED_MODEL_ID_ITEMS=()
    while IFS= read -r model_id; do
        model_id=$(trim_model_id "$model_id")
        if [[ -z "$model_id" ]]; then
            echo "MODEL_IDS entries must not be empty." >&2
            exit 1
        fi
        resolved_model_id=$(resolve_model_id "$model_id")
        RESOLVED_MODEL_ID_ITEMS+=("$resolved_model_id")
    done < <(jq -r '.[]' <<< "$PARSED_MODEL_IDS")
    RESOLVED_MODEL_IDS=$(printf '%s\n' "${RESOLVED_MODEL_ID_ITEMS[@]}" | jq -Rsc 'split("\n")[:-1]')
fi

for path in "$MATH_TRAIN_DATA" "$RUN_VAL_DATA"; do
    if [[ ! -f "$path" ]]; then
        echo "Required dataset does not exist: $path" >&2
        exit 1
    fi
done

if [[ ! "$N_GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "N_GPUS must be a positive integer; got: $N_GPUS" >&2
    exit 1
fi
if [[ ! "$TRAIN_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || (( TRAIN_BATCH_SIZE % N_GPUS != 0 )); then
    echo "TRAIN_BATCH_SIZE must be positive and divisible by N_GPUS; got: $TRAIN_BATCH_SIZE" >&2
    exit 1
fi
ALLOCATED_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
IFS=',' read -r -a CUDA_DEVICES <<< "$ALLOCATED_CUDA_VISIBLE_DEVICES"
if (( ${#CUDA_DEVICES[@]} < N_GPUS )); then
    echo "Need $N_GPUS visible GPUs, got CUDA_VISIBLE_DEVICES=$ALLOCATED_CUDA_VISIBLE_DEVICES" >&2
    exit 1
fi
if [[ -z "${TRAIN_CUDA_VISIBLE_DEVICES:-}" ]]; then
    TRAIN_CUDA_VISIBLE_DEVICES=$(IFS=','; echo "${CUDA_DEVICES[*]:0:N_GPUS}")
fi

MODEL_NAME_TAG=$(jq -r '.[] | split("/") | map(select(length > 0)) | last' <<< "$RESOLVED_MODEL_IDS" | tr '[:upper:]' '[:lower:]' | tr '-' '_' | paste -sd_ -)
EXPERIMENT_NAME=${EXPERIMENT_NAME:-gmas_share${MODEL_SHARING}_${MODEL_NAME_TAG}}

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "MODE=$MODE"
echo "ALLOCATED_CUDA_VISIBLE_DEVICES=$ALLOCATED_CUDA_VISIBLE_DEVICES"
echo "TRAIN_CUDA_VISIBLE_DEVICES=$TRAIN_CUDA_VISIBLE_DEVICES"
echo "N_GPUS=$N_GPUS"
echo "TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE"
echo "MODEL_SHARING=$MODEL_SHARING"
echo "AGENT_IDS=$MATH_AGENT_IDS"
echo "MODEL_IDS_RESOLVED=$RESOLVED_MODEL_IDS"
echo "MATH_TRAIN_DATA=$MATH_TRAIN_DATA"
echo "RUN_VAL_DATA=$RUN_VAL_DATA"
echo "EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "TRAINER_LOGGER=$TRAINER_LOGGER"
echo "XDG_CACHE_HOME=$XDG_CACHE_HOME"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "RAY_TMPDIR=$RAY_TMPDIR"
echo "GMAS_RAY_TMPDIR=$GMAS_RAY_TMPDIR"
echo "TMPDIR=$TMPDIR"
echo "WANDB_DIR=$WANDB_DIR"
echo "USE_WANDB=$USE_WANDB"
if [[ "$USE_WANDB" == "1" ]]; then
    echo "WANDB_MODE=${WANDB_MODE:-<unset>}"
    echo "WANDB_ENTITY=${WANDB_ENTITY:-<unset>}"
    if [[ -n "${WANDB_API_KEY:-}" ]]; then
        echo "WANDB_API_KEY=<set>"
    else
        echo "WANDB_API_KEY=<unset>"
    fi
fi

# GMAS run_math.sh already accepts trailing Hydra overrides.
hydra_overrides=(
    "data.train_files=$MATH_TRAIN_DATA"
    "data.val_files=$RUN_VAL_DATA"
    "data.train_batch_size=$TRAIN_BATCH_SIZE"
    "agent.agent_ids=$MATH_AGENT_IDS"
    "agent.model_ids=$RESOLVED_MODEL_IDS"
    "agent.model_sharing=$MODEL_SHARING"
    "trainer.logger=$TRAINER_LOGGER"
    "trainer.experiment_name=$EXPERIMENT_NAME"
    "trainer.n_gpus_per_node=$N_GPUS"
)
CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" \
exec bash examples/gmas_trainer/run_math.sh "$MODE" "${hydra_overrides[@]}" "$@"
