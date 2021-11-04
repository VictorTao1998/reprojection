#!/bin/bash
export PYTHONWARNINGS="ignore"

python -m torch.distributed.launch /jianyu-fast-vol/reprojection/train_psmnet_sim_ir_reprojection.py \
--logdir='/jianyu-fast-vol/eval/reprojection_p2' \
--config-file '/jianyu-fast-vol/reprojection/configs/remote_train_primitive_steps.yaml' \
--loss-ratio 0 \
--pattern2
