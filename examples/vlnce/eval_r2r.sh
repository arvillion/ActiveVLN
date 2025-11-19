#!/bin/bash
SAVE_PATH="eval_results/r2r/Qwen2.5-VL-3B_rl_r2r_4000"
CHUNKS=32
CONFIG_PATH="vlnce_server/VLN_CE/vlnce_baselines/config/r2r_baselines/activevln_r2r_test.yaml"

export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://127.0.0.1:8003/v1"
export PYTHONPATH=$PYTHONPATH:"$(pwd)/vlnce_server"

python3 eval/vlnce/eval_vlnce.py \
    --exp-config $CONFIG_PATH \
    --split-num $CHUNKS \
    --result-path $SAVE_PATH

# Score calculation
python3 eval/vlnce/analyze_results.py \
    --path $SAVE_PATH