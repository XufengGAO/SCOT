#!/usr/bin/env bash

set -x

benchmark="pfpascal"
backbone="resnet101"

python test.py \
    --benchmark $benchmark \
    --backbone $backbone \
    --weight_thres 0.10 \
    --alpha 0.1 \
    --exp1 1.0 \
    --exp2 0.5 \
    --classmap 1 \
    --split "test" \
    --pretrained_path "./backbone/eooch_100.pt"
