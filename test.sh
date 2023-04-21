#!/usr/bin/env bash

set -x

benchmark="pfpascal"
backbone="resnet50"

python test2.py \
    --benchmark $benchmark \
    --backbone $backbone \
    --alpha 0.1 \
    --exp1 1.0 \
    --exp2 0.5 \
    --classmap 1 \
    --split "test" \
    --img_side '(200,300)' \
    --selfsup 'dino' \
    --pretrained_path "./backbone/dino_r50_a010.pt"