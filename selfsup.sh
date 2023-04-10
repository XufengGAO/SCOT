#!/usr/bin/env bash

set -x

benchmark="pfpascal"
backbone="resnet50"
# -m torch.utils.bottleneck
python train.py \
    --benchmark $benchmark \
    --backbone $backbone \
    --wandb_proj "new SCOT" \
    --selfsup "sup" \
    --supervision 'warp' \
    --weight_thres 0.10 \
    --select_all 0.90 \
    --alpha 0.1 \
    --lr 0.005 \
    --momentum 0.9 \
    --epochs 150 \
    --batch_size 2 \
    --optimizer 'sgd' \
    --exp1 1.0 \
    --exp2 0.5 \
    --classmap 1 \
    --use_wandb False \
    --use_xavier False \
    --use_scheduler False \
    --use_grad_clip False \
    --loss_stage "votes" \
    --split "trn" \
    --cam "mask/resnet50/200_300" \
    --img_side '(200,300)'
    # --use_pretrained True \
    # --pretrained_path "./backbone/ckp_r50.pt"
    # --run_id '1j2qjgkr' \
    # --start_epoch 48 \
    # --logpath "./logs/3e-03_votes_strong_sgd_m0.90_supervised_resnet101.log"
    # #     # --selfsup "dino" 
    # --run_id "uau60b8q" \
    # --logpath "logs/_0402_080338.log" \
    # --start_epoch 101
        # --use_scheduler True \
    # --scheduler "step"