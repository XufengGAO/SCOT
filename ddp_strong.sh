#!/usr/bin/env bash


benchmark="pfpascal"
backbone="resnet50"
nnodes=4
master_addr="10.233.87.67"
master_port=12361

# CUDA_VISIBLE_DEVICES=0 \
python3 -m torch.distributed.launch --master_port=${master_port} --nproc_per_node=1 \
                                    --nnodes=${nnodes} --node_rank=$1 \
                                    --master_addr=${master_addr} \
                                    ddp_train.py \
                                    --benchmark $benchmark \
                                    --backbone $backbone \
                                    --weight_thres 0.10 \
                                    --select_all 0.90 \
                                    --criterion 'strong_ce'\
                                    --temp 1.0 \
                                    --weak_lambda 0.5 \
                                    --alpha 0.1 \
                                    --lr 0.003 \
                                    --momentum 0.99 \
                                    --epochs 50 \
                                    --batch_size 16 \
                                    --optimizer 'sgd' \
                                    --exp1 1.0 \
                                    --exp2 0.5 \
                                    --classmap 1 \
                                    --use_wandb True \
                                    --wandb_proj 'ddp_scot' \
                                    --loss_stage "votes" \
                                    --cam "mask/resnet50/200_300" \
                                    --img_side '(200,300)' 