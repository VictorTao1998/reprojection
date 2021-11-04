#!/bin/bash
export PYTHONWARNINGS="ignore"

python -m torch.distributed.launch /code/train_psmnet_real_ir_reprojection.py \
--logdir='/data/eval/reprojection_p2' \
--config-file '/code/configs/remote_train_primitive_steps_local.yaml' \
--loss-ratio 0 
