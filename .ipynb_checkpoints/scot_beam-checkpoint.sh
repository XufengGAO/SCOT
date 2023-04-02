#!/usr/bin/env bash

set -x

benchmark="pfpascal"
backbone="resnet101"
# -m torch.utils.bottleneck
python train.py \
    --benchmark $benchmark \
    --backbone $backbone \
    --weight_thres 0.10 \
    --select_all 0.90 \
    --supervision 'flow'\
    --lr 0.0003 \
    --momentum 0.9 \
    --epochs 100 \
    --batch_size 8 \
    --optimizer 'sgd' \
    --exp1 1.0 \
    --exp2 0.5 \
    --classmap 1 \
    --use_wandb True \
    --use_xavier False \
    --use_scheduler False \
    --use_grad_clip False \
    --loss_stage "votes" \
    --split "trn"
    #\
    # --use_pretrained True \
    # --pretrained_path "/scratch/students/2023-spring-sp-xugao/SCOT/logs/_0331_232809.log/best_model.pt"\
    # --run_id "xxduk35l" \
    # --logpath "logs/_0331_232809.log" \
    # --start_epoch 26