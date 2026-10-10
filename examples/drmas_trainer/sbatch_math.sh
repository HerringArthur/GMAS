#!/bin/bash
# Submit one integrated DrMAS math training job.
# By default, the job allocates eight GPUs and trains on all eight.
#
# Usage:
#   sbatch examples/drmas_trainer/sbatch_math.sh
#   sbatch examples/drmas_trainer/sbatch_math.sh 4B
#   sbatch examples/drmas_trainer/sbatch_math.sh 8B eval
#   sbatch examples/drmas_trainer/sbatch_math.sh eval
#   MODEL_SHARING=False MODEL_IDS='["Qwen2.5-3B-Instruct","Qwen3-4B"]' sbatch examples/drmas_trainer/sbatch_math.sh
#   MODEL_SHARING=False MODEL_IDS=Qwen2.5-3B-Instruct,Qwen3-4B sbatch examples/drmas_trainer/sbatch_math.sh
#
# Optional environment variables:
#   PROJECT_ROOT=/seu_share2/home/fenglei/213243847/SRPO_experiments
#   MODEL_SIZE=8B                     # 4B or 8B; used only when MODEL_IDS is unset
#   MODEL_PATH=/path/to/model          # overrides MODEL_SIZE defaults only when MODEL_IDS is unset
#   MODEL_4B=/path/to/Qwen3-4B
#   MODEL_8B=/path/to/Qwen3-8B
#   MATH_TRAIN_DATA=/path/to/train.parquet
#   MATH_VAL_DATA=/path/to/test_sampled.parquet
#   MATH_EVAL_DATA=/path/to/test.parquet
#   N_GPUS=8                          # default: 8
#   TRAIN_BATCH_SIZE=32
#   MODEL_SHARING=False               # True or False; defaults to run_math.sh
#   MODEL_IDS='["Qwen2.5-3B-Instruct","/path/to/verifier_model"]'
#   MODEL_IDS=Qwen2.5-3B-Instruct,Qwen3-4B
#                                      # first model is Solver Agent, second is Verifier Agent
#                                      # short names resolve to $PROJECT_ROOT/model/<name>
#   USE_WANDB=1                       # default: 0
#   DRMAS_RUNTIME_ROOT=/seu_share2/home/fenglei/213243847/DrMAS
#   DRMAS_TMPDIR=/path/to/tmp          # defaults under DRMAS_RUNTIME_ROOT
#   DRMAS_RAY_TMPDIR=/path/to/ray_root # backing dir; Ray appends ray/
#   RAY_SHORT_TMPDIR=/tmp/dm12345      # short symlink path for Ray sockets
#   CONDA_BASE=/path/to/miniconda3
#   CONDA_ENV_NAME=DrMAS

#SBATCH --job-name=drmas_math
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/drmas_math_%j.out
#SBATCH --error=logs/drmas_math_%j.err
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
MODEL_SHARING=${MODEL_SHARING:-}
MODEL_IDS=${MODEL_IDS:-}
USE_WANDB=${USE_WANDB:-$SCRIPT_USE_WANDB}
RESOLVED_MODEL_IDS=""
MATH_AGENT_IDS='["Solver Agent","Verifier Agent"]'
case "$USE_WANDB" in
    1|true|True|yes|YES)
        USE_WANDB=1
        ;;
    0|false|False|no|NO|"")
        USE_WANDB=0
        ;;
    *)
        echo "USE_WANDB must be 0 or 1; got: $USE_WANDB" >&2
        exit 1
        ;;
esac
if [[ -z "${TRAINER_LOGGER:-}" ]]; then
    if [[ "$USE_WANDB" == "1" ]]; then
        TRAINER_LOGGER="['console','wandb']"
    else
        TRAINER_LOGGER="['console']"
    fi
fi
case "${1:-}" in
    4B|4b|8B|8b)
        MODEL_SIZE=$1
        MODE=${2:-$MODE}
        ;;
    "")
        ;;
    *)
        MODE=$1
        ;;
esac
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
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
DRMAS_RUNTIME_ROOT=${DRMAS_RUNTIME_ROOT:-/seu_share2/home/fenglei/213243847/DrMAS}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-$DRMAS_RUNTIME_ROOT/cache}
HF_HOME=${HF_HOME:-$DRMAS_RUNTIME_ROOT/cache/huggingface}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
DRMAS_TMPDIR=${DRMAS_TMPDIR:-$DRMAS_RUNTIME_ROOT/tmp}
DRMAS_RAY_TMPDIR=${DRMAS_RAY_TMPDIR:-$DRMAS_TMPDIR}
RAY_SHORT_TMPDIR=${RAY_SHORT_TMPDIR:-/tmp/dm${SLURM_JOB_ID:-$$}}
TMPDIR=$DRMAS_TMPDIR
RAY_TMPDIR=$RAY_SHORT_TMPDIR
WANDB_DIR=${WANDB_DIR:-$DRMAS_RUNTIME_ROOT/wandb}
WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-$WANDB_DIR/cache}
WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR:-$WANDB_DIR/config}
export XDG_CACHE_HOME HF_HOME HF_DATASETS_CACHE HF_HUB_CACHE DRMAS_TMPDIR DRMAS_RAY_TMPDIR RAY_SHORT_TMPDIR TMPDIR RAY_TMPDIR WANDB_DIR WANDB_CACHE_DIR WANDB_CONFIG_DIR
configure_wandb

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." >/dev/null 2>&1 && pwd)
if [[ ! -f "$REPO_ROOT/examples/drmas_trainer/run_math.sh" ]]; then
    if [[ -n "${DRMAS_ROOT:-}" && -f "$DRMAS_ROOT/examples/drmas_trainer/run_math.sh" ]]; then
        REPO_ROOT="$DRMAS_ROOT"
    elif [[ -f "$PROJECT_ROOT/DrMAS/examples/drmas_trainer/run_math.sh" ]]; then
        REPO_ROOT="$PROJECT_ROOT/DrMAS"
    elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/examples/drmas_trainer/run_math.sh" ]]; then
        REPO_ROOT="$SLURM_SUBMIT_DIR"
    elif [[ -f "$PWD/examples/drmas_trainer/run_math.sh" ]]; then
        REPO_ROOT="$PWD"
    else
        echo "Cannot locate repo root. Set DRMAS_ROOT=/path/to/DrMAS before sbatch." >&2
        exit 1
    fi
fi
cd "$REPO_ROOT"
mkdir -p logs
mkdir -p "$XDG_CACHE_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$DRMAS_TMPDIR" "$DRMAS_RAY_TMPDIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
if [[ -e "$RAY_TMPDIR" && ! -L "$RAY_TMPDIR" ]]; then
    echo "RAY_TMPDIR must be a short symlink path, but it already exists and is not a symlink: $RAY_TMPDIR" >&2
    exit 1
fi
if [[ -L "$RAY_TMPDIR" ]]; then
    unlink "$RAY_TMPDIR"
fi
ln -s "$DRMAS_RAY_TMPDIR" "$RAY_TMPDIR"
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

if [[ -z "$MODEL_IDS" ]]; then
    MODEL_SIZE=${MODEL_SIZE:-8B}
    case "$MODEL_SIZE" in
        4B|4b)
            MODEL_PATH=${MODEL_PATH:-${MODEL_4B:-$PROJECT_ROOT/model/Qwen3-4B}}
            N_GPUS=${N_GPUS:-8}
            ;;
        8B|8b)
            MODEL_PATH=${MODEL_PATH:-${MODEL_8B:-$PROJECT_ROOT/model/Qwen3-8B}}
            N_GPUS=${N_GPUS:-8}
            ;;
        *)
            echo "MODEL_SIZE must be 4B or 8B when MODEL_IDS is unset; got: $MODEL_SIZE" >&2
            exit 1
            ;;
    esac
else
    MODEL_PATH=${MODEL_PATH:-}
    N_GPUS=${N_GPUS:-8}
fi

MATH_TRAIN_DATA=${MATH_TRAIN_DATA:-$PROJECT_ROOT/data/drmas_math/train.parquet}
MATH_VAL_DATA=${MATH_VAL_DATA:-$PROJECT_ROOT/data/drmas_math/test_sampled.parquet}
MATH_EVAL_DATA=${MATH_EVAL_DATA:-$PROJECT_ROOT/data/drmas_math/test.parquet}
if [[ "$MODE" == "eval" ]]; then
    RUN_VAL_DATA="$MATH_EVAL_DATA"
else
    RUN_VAL_DATA="$MATH_VAL_DATA"
fi

case "$MODEL_SHARING" in
    ""|True|False|true|false)
        ;;
    *)
        echo "MODEL_SHARING must be True or False when set; got: $MODEL_SHARING" >&2
        exit 1
        ;;
esac
if [[ "$MODEL_SHARING" == "False" || "$MODEL_SHARING" == "false" ]]; then
    if [[ -z "$MODEL_IDS" ]]; then
        echo "MODEL_SHARING=False requires explicit MODEL_IDS." >&2
        echo "Example: MODEL_IDS='[\"/path/to/solver\",\"/path/to/verifier\"]'" >&2
        exit 1
    fi
fi
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
trim_model_id() {
    local value=$1

    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s\n' "$value"
}
if [[ -n "$MODEL_IDS" ]]; then
    if ! command -v jq >/dev/null 2>&1; then
        echo "jq is required to validate MODEL_IDS." >&2
        exit 1
    fi
    MODEL_ID_ITEMS=()
    MODEL_IDS_TRIMMED=$(trim_model_id "$MODEL_IDS")
    if [[ "$MODEL_IDS_TRIMMED" == \[* ]]; then
        jq -er 'if type == "array" and length == 2 and all(.[]; type == "string" and length > 0) then true else error("MODEL_IDS must be a JSON string array with exactly two entries: [solver_model, verifier_model]") end' <<< "$MODEL_IDS" >/dev/null
        while IFS= read -r model_id; do
            MODEL_ID_ITEMS+=("$model_id")
        done < <(jq -r '.[]' <<< "$MODEL_IDS")
    else
        IFS=',' read -r -a MODEL_ID_ITEMS <<< "$MODEL_IDS"
        if (( ${#MODEL_ID_ITEMS[@]} != 2 )); then
            echo "MODEL_IDS must contain exactly two entries: solver_model,verifier_model" >&2
            exit 1
        fi
    fi
    RESOLVED_MODEL_ID_ITEMS=()
    for model_id in "${MODEL_ID_ITEMS[@]}"; do
        model_id=$(trim_model_id "$model_id")
        if [[ -z "$model_id" ]]; then
            echo "MODEL_IDS entries must not be empty." >&2
            exit 1
        fi
        resolved_model_id=$(resolve_model_id "$model_id")
        RESOLVED_MODEL_ID_ITEMS+=("$resolved_model_id")
    done
    RESOLVED_MODEL_IDS=$(printf '%s\n' "${RESOLVED_MODEL_ID_ITEMS[@]}" | jq -Rsc 'split("\n")[:-1]')
fi

REQUIRED_PATHS=("$MATH_TRAIN_DATA" "$RUN_VAL_DATA")
if [[ -z "$MODEL_IDS" ]]; then
    REQUIRED_PATHS=("$MODEL_PATH" "${REQUIRED_PATHS[@]}")
fi
for path in "${REQUIRED_PATHS[@]}"; do
    if [[ ! -e "$path" ]]; then
        echo "Required training path does not exist: $path" >&2
        exit 1
    fi
done

ALLOCATED_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
IFS=',' read -r -a CUDA_DEVICES <<< "$ALLOCATED_CUDA_VISIBLE_DEVICES"
if (( N_GPUS < 1 )); then
    echo "N_GPUS must be positive; got: $N_GPUS" >&2
    exit 1
fi
if (( TRAIN_BATCH_SIZE < 1 || TRAIN_BATCH_SIZE % N_GPUS != 0 )); then
    echo "TRAIN_BATCH_SIZE must be positive and divisible by N_GPUS; got: $TRAIN_BATCH_SIZE" >&2
    exit 1
fi
if (( ${#CUDA_DEVICES[@]} < N_GPUS )); then
    echo "Need $N_GPUS visible GPUs, got CUDA_VISIBLE_DEVICES=$ALLOCATED_CUDA_VISIBLE_DEVICES" >&2
    exit 1
fi
if [[ -z "${TRAIN_CUDA_VISIBLE_DEVICES:-}" ]]; then
    TRAIN_CUDA_VISIBLE_DEVICES=$(IFS=','; echo "${CUDA_DEVICES[*]:0:N_GPUS}")
fi

if [[ -n "$MODEL_IDS" ]]; then
    MODEL_SIZE_DISPLAY=${MODEL_SIZE:-"<unused; MODEL_IDS set>"}
    MODEL_PATH_DISPLAY=${MODEL_PATH:-"<unused; MODEL_IDS set>"}
else
    MODEL_SIZE_DISPLAY=$MODEL_SIZE
    MODEL_PATH_DISPLAY=$MODEL_PATH
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "ALLOCATED_CUDA_VISIBLE_DEVICES=$ALLOCATED_CUDA_VISIBLE_DEVICES"
echo "TRAIN_CUDA_VISIBLE_DEVICES=$TRAIN_CUDA_VISIBLE_DEVICES"
echo "MODEL_SIZE=$MODEL_SIZE_DISPLAY"
echo "MODEL_PATH=$MODEL_PATH_DISPLAY"
echo "N_GPUS=$N_GPUS"
echo "TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE"
echo "MODEL_SHARING=${MODEL_SHARING:-<run_math.sh default>}"
echo "TRAINER_LOGGER=$TRAINER_LOGGER"
echo "AGENT_IDS=$MATH_AGENT_IDS"
echo "MODEL_IDS_INPUT=${MODEL_IDS:-<unset>}"
echo "MODEL_IDS_RESOLVED=${RESOLVED_MODEL_IDS:-[$MODEL_PATH,$MODEL_PATH]}"
echo "MATH_TRAIN_DATA=$MATH_TRAIN_DATA"
echo "RUN_VAL_DATA=$RUN_VAL_DATA"
echo "XDG_CACHE_HOME=$XDG_CACHE_HOME"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "RAY_TMPDIR=$RAY_TMPDIR"
echo "DRMAS_RAY_TMPDIR=$DRMAS_RAY_TMPDIR"
echo "TMPDIR=$TMPDIR"
echo "WANDB_DIR=$WANDB_DIR"
echo "WANDB_CACHE_DIR=$WANDB_CACHE_DIR"
echo "WANDB_CONFIG_DIR=$WANDB_CONFIG_DIR"
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

# run_math.sh fixes these values internally. Intercept only its trainer launch
# and append Hydra overrides so the original script remains unchanged.
DRMAS_PYTHON_BIN=$(command -v python3)
export DRMAS_PYTHON_BIN MODEL_PATH TRAIN_BATCH_SIZE MODEL_SHARING MODEL_IDS RESOLVED_MODEL_IDS MATH_AGENT_IDS MATH_TRAIN_DATA RUN_VAL_DATA N_GPUS TRAINER_LOGGER
python3() {
    if [[ "${1:-}" == "-m" && "${2:-}" == "verl.trainer.main_ppo" ]]; then
        local -a hydra_overrides=(
            "data.train_files=$MATH_TRAIN_DATA" \
            "data.val_files=$RUN_VAL_DATA" \
            "data.train_batch_size=$TRAIN_BATCH_SIZE" \
            "trainer.logger=$TRAINER_LOGGER" \
            "agent.agent_ids=$MATH_AGENT_IDS" \
            "trainer.n_gpus_per_node=$N_GPUS"
        )
        if [[ -n "$RESOLVED_MODEL_IDS" ]]; then
            hydra_overrides+=("agent.model_ids=$RESOLVED_MODEL_IDS")
        else
            hydra_overrides+=("agent.model_ids=[$MODEL_PATH,$MODEL_PATH]")
        fi
        if [[ -n "$MODEL_SHARING" ]]; then
            hydra_overrides+=("agent.model_sharing=$MODEL_SHARING")
        fi
        "$DRMAS_PYTHON_BIN" "$@" "${hydra_overrides[@]}"
    else
        "$DRMAS_PYTHON_BIN" "$@"
    fi
}
export -f python3

TRAINER_LOG_MODEL_IDS="${RESOLVED_MODEL_IDS:-[\"$MODEL_PATH\",\"$MODEL_PATH\"]}" \
CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" \
bash examples/drmas_trainer/run_math.sh "$MODE"
