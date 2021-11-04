#!/bin/bash
export PYTHONWARNINGS="ignore"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u /jianyu-fast-vol/reprojection/train_psmnet_sim_ir_reprojection.py \
--logdir='/jianyu-fast-vol/eval/reprojection_p1' \
--config-file '/jianyu-fast-vol/reprojection/configs/remote_train_primitive_steps.yaml' \
--loss-ratio 0 \
