#!/usr/bin/env bash

set -x

benchmark="pfpascal"
backbone="resnet50"

python test.py \
    --benchmark $benchmark \
    --backbone $backbone \
    --alpha 0.1 \
    --exp1 1.0 \
    --exp2 0.5 \
    --classmap 1 \
    --split "test" \
    --img_side '(200,300)' \
    --pretrained_path "./backbone/r50_a01.pt"