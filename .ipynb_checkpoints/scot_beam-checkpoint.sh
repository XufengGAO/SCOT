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
    --supervision 'strong'\
    --alpha 0.1 \
    --lr 0.005 \
    --momentum 0.9 \
    --epochs 100 \
    --batch_size 10 \
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
    # --use_pretrained True \
    # --pretrained_path "/scratch/students/2023-spring-sp-xugao/SCOT/backbone/ckp_r50.pt"
    # #     # --selfsup "dino" 
    # --run_id "uau60b8q" \
    # --logpath "logs/_0402_080338.log" \
    # --start_epoch 101