#!/usr/bin/env bash


benchmark="pfpascal"
backbone="resnet101"
nnodes=2
master_addr="10.233.114.222"
master_port=12362

# CUDA_VISIBLE_DEVICES=0 \
python3 -m torch.distributed.launch --master_port=${master_port} --nproc_per_node=1 \
                                    --nnodes=${nnodes} --node_rank=$1 \
                                    --master_addr=${master_addr} \
                                    ddp_train.py \
                                    --benchmark $benchmark \
                                    --backbone $backbone \
                                    --weight_thres 0.10 \
                                    --select_all 0.90 \
                                    --criterion 'weak'\
                                    --temp 0.05 \
                                    --weak_lambda '[1.0, 1.0, 1.0]' \
                                    --weak_mode 'custom_lambda' \
                                    --match_norm_type 'l1' \
                                    --alpha 0.1 \
                                    --lr 0.003 \
                                    --momentum 0.9 \
                                    --epochs 50 \
                                    --batch_size 2 \
                                    --optimizer 'sgd' \
                                    --exp2 0.5 \
                                    --use_wandb False \
                                    --wandb_proj 'ddp_scot' \
                                    --loss_stage "votes" \
                                    --cam "mask/resnet101/200_300" \
                                    --img_side '(200,300)' 