#!/usr/bin/env bash


benchmark="pfpascal"
backbone="resnet101"

python3 ddp_test.py \
                    --benchmark $benchmark \
                    --backbone $backbone \
                    --alpha 0.1 \
                    --split 'val' \
                    --batch_size 8 \
                    --exp2 0.5 \
                    --use_wandb True \
                    --wandb_proj 'ddp_scot' \
                    --resume './backbone/selectAll_1.pt'\
                    --wandb_name 'SCOT_selectAll_w=1' \
                    --run_id 'o7mx3vp7' \
                    --cam "mask/resnet101/200_300" \
                    --img_side '(200,300)' 