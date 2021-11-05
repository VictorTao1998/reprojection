#!/bin/bash
export PYTHONWARNINGS="ignore"

python -m torch.distributed.launch /jianyu-fast-vol/reprojection/train_psmnet_real_ir_reprojection.py \
--logdir='/jianyu-fast-vol/eval/reprojection_real_p2_ps35' \
--config-file '/jianyu-fast-vol/reprojection/configs/remote_train_primitive_steps.yaml' \
--loss-ratio 0 \
--pattern2 \
--patch_size 35