#!/bin/bash
set -x

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
export WANDB_API_KEY="..."
export VLLM_ATTENTION_BACKEND=XFORMERS

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Train over a single node
torchrun --standalone --nnodes=1 --nproc_per_node=8 -m verl.trainer.fsdp_sft_trainer \
    data.train_files=./data/sft_critique/Qwen2.5-Math-1.5B-L-openr1-f3/sft_critique_full.parquet \
    data.val_files=./data/sft_critique/Qwen2.5-Math-1.5B-L-openr1-f3/sft_critique_full_val.parquet \
    data.prompt_key=input \
    data.response_key=output \
    +data.use_template=True \
    data.max_length=12288 \
    data.train_batch_size=1024 \
    data.micro_batch_size_per_gpu=2 \
    optim.lr=1e-6 \
    model.partial_pretrain="yangzhch6/Qwen2.5-Math-1.5B-L" \
    trainer.default_local_dir=./checkpoints/sft-critique/Qwen2.5-Math-1.5B-L-full \
    trainer.project_name=sft-gt-critique \
    trainer.experiment_name=sft-gt-critique-Qwen2.5-Math-1.5B-L-full \
    trainer.total_epochs=3 \
    trainer.logger='["console","wandb"]' \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true