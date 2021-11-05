#!/bin/bash
export PYTHONWARNINGS="ignore"

python -m torch.distributed.launch /jianyu-fast-vol/reprojection/test_psmnet_reprojection.py \
--config-file '/jianyu-fast-vol/reprojection/configs/remote_test.yaml' \
--model '/jianyu-fast-vol/eval/reprojection_real_p1/models/model_48000.pth'
--output '/jianyu-fast-vol/eval/reprojection_test_real_p1' \
--onreal \
--exclude-bg \
--exclude-zeros