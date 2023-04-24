#!/usr/bin/env bash

nnodes=2
master_addr="10.233.66.8"
master_port=12356

# CUDA_VISIBLE_DEVICES=0 \
python3 -m torch.distributed.launch --master_port=${master_port} --nproc_per_node=1 \
                                    --nnodes=${nnodes} --node_rank=$1 \
                                    --master_addr=${master_addr} \
                                    train.py \
                                    --benchmark $benchmark \
                                    --backbone $backbone \
                                    --weight_thres 0.10 \
                                    --select_all 0.90 \
                                    --loss 'strong_ce'\
                                    --alpha 0.1 \
                                    --lr 0.001 \
                                    --momentum 0.9 \
                                    --epochs 100 \
                                    --batch_size 2 \
                                    --optimizer 'sgd' \
                                    --exp1 1.0 \
                                    --exp2 0.5 \
                                    --classmap 1 \
                                    --use_wandb False \
                                    --loss_stage "sim" \
                                    --split "trn" \
                                    --cam "mask/resnet50/200_300" \
                                    --img_side '(200,300)' 