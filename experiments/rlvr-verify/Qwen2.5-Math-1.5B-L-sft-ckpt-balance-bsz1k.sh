#!/bin/bash
set -x

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
export WANDB_API_KEY="..."
export VLLM_ATTENTION_BACKEND=XFORMERS
export MODEL_PATH="./checkpoints/sft-critique/Qwen2.5-Math-1.5B-L-full/final_checkpoint"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Train over a single node, 1 A100-80GB GPUs.
python3 -m verl.trainer.main_ppo_verify \
    algorithm.adv_estimator=grpo \
    data.train_files=./data/rlvr-critique/Qwen2.5-Math-1.5B-L-openr1-f3/rlvr_critique_balance.parquet \
    data.val_files=./data/rlvr-critique/Qwen2.5-Math-1.5B-L-openr1-f3/test_n4_full.parquet \
    data.train_batch_size=1024 \
    data.val_batch_size=512 \
    data.max_prompt_length=5096 \
    data.max_response_length=3072 \
    data.shuffle=False \
    +data.use_template=True \
    +data.reward_impl_version=4 \
    +actor_rollout_ref.ref.use_ref=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.000 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.n_val=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.000 \
    +algorithm.grpo_use_std=False \
    trainer.critic_warmup=0 \
    +trainer.del_last_ckpt=False \
    +trainer.log_train=False \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rlvr-verify' \
    trainer.experiment_name='Qwen2.5-Math-1.5B-L-sft-ckpt-balance-bsz1k' \
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=40 \
    trainer.default_hdfs_dir=null \
    trainer.total_training_steps=403 \
    trainer.total_epochs=30 "${@:1}"