#!/usr/bin/env bash

set -x

benchmark="pfpascal"
backbone="resnet101"

python -m torch.utils.bottleneck train.py \
    --benchmark $benchmark \
    --backbone $backbone \
    --weight_thres 0.10 \
    --select_all 0.90 \
    --supervision 'flow'\
    --lr 0.0001 \
    --momentum 0.9 \
    --epochs 1 \
    --batch_size 6 \
    --optimizer 'sgd' \
    --exp1 1.0 \
    --exp2 0.5 \
    --classmap 1 \
    --use_wandb True \
    --use_xavier False \
    --use_scheduler False \
    --use_grad_clip False \
    --loss_stage "votes"    
    
    # --use_pretrained True \
    # --pretrained_path "/scratch/students/2023-spring-sp-xugao/SCOT/logs/_0330_151508.log/best_model.pt"\
    # --run_id "5h6kghje" \
    # --start_epoch 18 \