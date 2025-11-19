set -x

PROJECT_NAME="activevln"
EXPERIMENT_NAME="r2r"

export SAVE_CHECKPOINT_DIR="verl_checkpoints"

DATASET_TRAIN=data/r2r_4000_train.parquet
DATASET_VAL=data/r2r_val_tiny.parquet

REF_MODEL_PATH=Arvil/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn 

# export RAY_DEBUG="1"
# export RAY_DEBUG_POST_MORTEM="1"

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/vlnce"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-path $CONFIG_PATH --config-name train_vlnce_4gpus.yaml \
    data.train_files=[${DATASET_TRAIN}] \
    data.val_files=[${DATASET_VAL}] \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.rollout.agent.base_url="http://127.0.0.1:5001" \
    actor_rollout_ref.rollout.agent.timeout=180 \
    actor_rollout_ref.rollout.agent.gen_length_tolerance=2.0 \
    actor_rollout_ref.rollout.agent.enable_dynamic_sampling=true \
    actor_rollout_ref.rollout.agent.experiment_name=${EXPERIMENT_NAME} \
    actor_rollout_ref.rollout.agent.reward.reward_type=weighted_success_ndtw \
    actor_rollout_ref.rollout.agent.reward.success_reward_base=15 \
    actor_rollout_ref.rollout.agent.reward.ndtw_reward_base=0 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}

