#!/bin/bash

# Check if an argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_name>"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=1

# retrieve experiment name and set the other parameters accordingly
EXP_NAME="$1"
OUTPUT_DIR="./checkpoints/gpro_multicontrol${EXP_NAME}/"
LOG_FILE="gpro_multicontrol${EXP_NAME}.log"

# run GRPO experiment
#
nohup python3 grpo_exps.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --reward_model_name "" \
    --dataset_name "" \
    --max_length 512 \
    --output_max_length 50 \
    --mini_batch_size 1 \
    --batch_size 8 \
    --ppo_epochs 4 \
    --early_stopping False \
    --target_kl 0.1 \
    --reward_baseline 0.0 \
    --batched_gen False \
    --output_dir "$OUTPUT_DIR" \
    --experiment_name "$EXP_NAME" \
    > "$LOG_FILE" 2>&1 &