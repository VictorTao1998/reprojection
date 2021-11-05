#!/bin/bash
export PYTHONWARNINGS="ignore"

python -m torch.distributed.launch /code/test_ir_reprojection_converge.py \
--logdir='/data/eval/reproj_conv' \
--config-file '/code/configs/remote_train_primitive_steps_local.yaml' \
--kernal_size 41 
