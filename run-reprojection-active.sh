#!/bin/bash
export PYTHONWARNINGS="ignore"

python -m torch.distributed.launch /jianyu-fast-vol/reprojection/train_psmnet_reprojection.py \
--logdir='/jianyu-fast-vol/eval/reprojection_active_train' \
--config-file '/jianyu-fast-vol/reprojection/configs/remote_train_primitive_steps.yaml' 