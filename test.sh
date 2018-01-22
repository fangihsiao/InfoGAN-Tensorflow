#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES='0' python main.py \
    --model_dir ./exp/InfoGan \
    --is_training=False \
    --fix_var=False
