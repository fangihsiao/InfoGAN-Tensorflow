#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES='0' python main.py \
    --model_dir ./exp/InfoGan \
    --is_training=True \
    --epoch 100 \
    --batch_size 64 \
    --fix_var=False
