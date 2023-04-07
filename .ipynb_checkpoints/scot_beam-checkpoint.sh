#!/usr/bin/env bash

set -x

benchmark="pfpascal"
backbone="resnet50"
# -m torch.utils.bottleneck
python train.py \
    --benchmark $benchmark \
    --backbone $backbone \
    --weight_thres 0.10 \
    --select_all 0.90 \
    --supervision 'strong'\
    --alpha 0.1 \
    --lr 0.005 \
    --momentum 0.9 \
    --epochs 100 \
    --batch_size 14 \
    --optimizer 'sgd' \
    --exp1 1.0 \
    --exp2 0.5 \
    --classmap 1 \
    --use_wandb False \
    --use_xavier False \
    --use_scheduler False \
    --use_grad_clip False \
    --loss_stage "votes" \
    --split "trn" 
    # --selfsup "dino" 
    # --use_pretrained True \
    # --pretrained_path "/scratch/students/2023-spring-sp-xugao/SCOT/backbone/ckp.pt"\
    # --run_id "uau60b8q" \
    # --logpath "logs/_0402_080338.log" \
    # --start_epoch 101