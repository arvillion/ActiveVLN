#!/bin/bash
SAVE_PATH="eval_results/rxr/Qwen2.5-VL-3B_rl_rxr_4000_step350"
CHUNKS=32
CONFIG_PATH="vlnce_server/VLN_CE/vlnce_baselines/config/rxr_baselines/activevln_rxr_test.yaml"

export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://127.0.0.1:8003/v1"
export PYTHONPATH=$PYTHONPATH:"$(pwd)/vlnce_server"

python3 eval/vlnce/eval_vlnce.py \
    --exp-config $CONFIG_PATH \
    --split-num $CHUNKS \
    --result-path $SAVE_PATH \
    --max-turns 120 


# Score calculation
python3 eval/vlnce/analyze_results.py \
    --path $SAVE_PATH
