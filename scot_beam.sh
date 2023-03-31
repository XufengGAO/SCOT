#!/usr/bin/env bash

set -x

benchmark="pfpascal"
backbone="resnet101"

python train.py \
    --benchmark $benchmark \
    --backbone $backbone \
    # --use_pretrained True \
    # --pretrained_path "/scratch/students/2023-spring-sp-xugao/SCOT/logs/_0330_151508.log/best_model.pt"\
    # --run_id "5h6kghje" \
    # --start_epoch 18 \
    --weight_thres 0.15 \
    --lr 0.001 \
    --epochs 28 \
    --batch_size 6 \
    --optimizer 'sgd' \
    --exp1 1.0 \
    --exp2 0.5 \
    --classmap 1 \
    --use_wandb True \
    --use_xavier False \
    --use_scheduler False \
    --use_grad_clip False \
    --loss_stage "votes"\
    --supervision 'flow'
    
    