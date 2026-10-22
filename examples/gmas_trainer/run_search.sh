#!/usr/bin/env bash
# Usage: bash examples/gmas_trainer/run_search.sh [train|eval|evaluation] [Hydra overrides...]
# Activate the Miniforge-based training environment in the sbatch job first.
# WORK_DIR, DATA_DIR and MODEL_DIR can override the local project paths.

set -euo pipefail
set -x

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source examples/trainer_logging.sh

data_dir="${DATA_DIR:-${WORK_DIR:-$PWD}/data/drmas_search}"
search_url="${SEARCH_URL:-http://127.0.0.1:8001/retrieve}"

MODE=${1:-train}
if [ "$#" -gt 0 ]; then
    shift
fi
case "$MODE" in
    train|eval|evaluation) ;;
    *)
        echo "Usage: $0 [train|eval|evaluation] [Hydra overrides...]" >&2
        exit 2
        ;;
esac
if [ "$MODE" == "eval" ] || [ "$MODE" == "evaluation" ]; then
    echo "Running in evaluation mode"
    VAL_ONLY=True
    TRAIN_DATA="$data_dir/train.parquet"
    VAL_DATA="$data_dir/test.parquet" # Full test dataset
    train_data_size=128
    val_data_size=64
    val_group_size=16  # For pass@16 and avg@16 computation during evaluation
else
    echo "Running in training mode"
    VAL_ONLY=False
    TRAIN_DATA="$data_dir/train.parquet"
    VAL_DATA="$data_dir/test_sampled.parquet" # For fast validation during training (sample 30 entries per data source, total 210 samples)
    train_data_size=128
    val_data_size=256
    val_group_size=1
fi

###################### Algorithm Configurations #################
algorithm=grpo
group_size=5
group_by_agent_id=False # GMAS groups by worker prefix / Router node instead.
credit_mode=router_subagent
norm_adv_by_std_in_grpo=True
# Match the DrMAS default; when enabled, KL is a separate token-level penalty.
use_kl_in_reward=False

##################### Agent Configurations #####################
env_name=search
role_ids='["SEARCH","VERIFY","ANSWER"]'
agent_ids='["Router Agent","SEARCH","VERIFY","ANSWER"]'
model_path="${MODEL_DIR:-${WORK_DIR:-$PWD}/model/Qwen2.5-3B-Instruct}"
model_ids="[\"$model_path\",\"$model_path\",\"$model_path\",\"$model_path\"]"
model_sharing=False

orchestra_type=router
max_turn=4 # Environment interactions, including searches.
max_loop_num=4 # Router/role collaboration rounds per environment step.

# Per-agent overrides, ordered as Router, SEARCH, VERIFY, ANSWER.
actor_optim_lr='[1e-6,1e-6,1e-6,1e-6]'
actor_ppo_micro_batch_size_per_gpu='[4,4,4,4]'

model_name_tag=$(python -c 'import json, sys; print("_".join(model.rsplit("/", 1)[-1].lower().replace("-", "_") for model in json.loads(sys.argv[1])))' "$model_ids")

experiment_name="gmas_share${model_sharing}_${model_name_tag}"

# Speed-first starting point: retain optimizer states on GPU. Compare CPU weight
# caching with a Hydra override if repeated FSDP gathers dominate rollout time.
# SGLang-specific aliases below are passed to Engine and logged at startup.
run_with_trainer_log gmas "$MODE" python3 -m verl.trainer.main_ppo --config-name router_trainer \
    algorithm.adv_estimator=$algorithm \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=800 \
    data.filter_overlong_prompts=False \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=null \
    actor_rollout_ref.actor.optim.lr=null \
    +agent.agent_specific_parameters.actor.optim.lr=$actor_optim_lr \
    actor_rollout_ref.model.use_remove_padding=True \
    ++actor_rollout_ref.actor.response_only_logits=True \
    actor_rollout_ref.actor.use_adaptive_ppo_mini_batch_size=True \
    actor_rollout_ref.actor.ppo_mini_update_num=5 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=null \
    +agent.agent_specific_parameters.actor.ppo_micro_batch_size_per_gpu=$actor_ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    ++actor_rollout_ref.rollout.engine_kwargs.sglang.disable_cuda_graph=False \
    ++actor_rollout_ref.rollout.engine_kwargs.sglang.max_running_requests=1024 \
    ++actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size=-1 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    ++actor_rollout_ref.rollout.weight_sync_bucket_mb=128 \
    ++actor_rollout_ref.rollout.cache_cpu_weights=False \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.group_by_agent_id=$group_by_agent_id \
    algorithm.credit_mode=$credit_mode \
    algorithm.norm_adv_by_std_in_grpo=$norm_adv_by_std_in_grpo \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    env.env_name="$env_name" \
    env.seed=0 \
    env.search.search_url="$search_url" \
    ++env.search.batch_size=64 \
    env.max_steps=$max_turn \
    env.rollout.n=$group_size \
    env.rollout.val_n=$val_group_size \
    env.history_length=$max_turn \
    agent.agent_ids="$agent_ids" \
    agent.role_ids="$role_ids" \
    agent.model_ids="$model_ids" \
    agent.model_sharing=$model_sharing \
    agent.use_agent_memory=False \
    agent.orchestra_type=$orchestra_type \
    agent.orchestra.router.max_loop_num=$max_loop_num \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='GMAS_search' \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.val_only=$VAL_ONLY \
    trainer.val_before_train=True \
    "$@"
