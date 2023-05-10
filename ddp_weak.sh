#!/usr/bin/env bash


benchmark="pfpascal"
backbone="resnet101"
nnodes=4
master_addr="10.233.87.208"
master_port=12364

# CUDA_VISIBLE_DEVICES=0 \
python3 -m torch.distributed.launch --master_port=${master_port} --nproc_per_node=1 \
                                    --nnodes=${nnodes} --node_rank=$1 \
                                    --master_addr=${master_addr} \
                                    ddp_train.py \
                                    --benchmark $benchmark \
                                    --backbone $backbone \
                                    --weight_thres 0.10 \
                                    --select_all 0.9 \
                                    --criterion 'weak'\
                                    --temp 0.05 \
                                    --weak_lambda '[0.0, 1.0, 0.0]' \
                                    --collect_grad False \
                                    --entropy_func 'info_entropy_sq'\
                                    --match_norm_type 'l1' \
                                    --alpha 0.1 \
                                    --lr 0.01 \
                                    --momentum 0.9 \
                                    --epochs 100 \
                                    --batch_size 4 \
                                    --optimizer 'sgd' \
                                    --exp2 0.5 \
                                    --use_wandb True \
                                    --wandb_proj 'ddp_scot' \
                                    --loss_stage "sim" \
                                    --cam "mask/resnet101/200_300" \
                                    --img_side '(240,240)' 