"""
Author: Isabella Liu 11/1/21
Feature: Train PSMNet only using IR reprojection
"""
import gc
import os
import argparse
import numpy as np
import tensorboardX
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import random

from datasets.messytable import MessytableDataset
from nets.psmnet import PSMNet
from nets.cycle_gan import CycleGANModel
from nets.transformer import Transformer
from utils.cascade_metrics import compute_err_metric
from utils.warp_ops import apply_disparity_cu
from utils.reprojection import get_reprojection_error, get_reprojection_error_old
from utils.config import cfg
from utils.reduce import set_random_seed, synchronize, AverageMeterDict, \
    tensor2float, tensor2numpy, reduce_scalar_outputs, make_nograd_func
from utils.util import setup_logger, weights_init, \
    adjust_learning_rate, save_scalars, save_scalars_graph, save_images, save_images_grid, disp_error_img


cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Reprojection with Pyramid Stereo Network (PSMNet)')
parser.add_argument('--config-file', type=str, default='./configs/local_train_steps.yaml',
                    metavar='FILE', help='Config files')
parser.add_argument('--logdir', required=True, help='Directory to save logs and checkpoints')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed (default: 1)')
parser.add_argument("--local_rank", type=int, default=0, help='Rank of device in distributed training')
parser.add_argument('--debug', action='store_true', help='Whether run in debug mode (will load less data)')
parser.add_argument('--warp-op', action='store_true',default=True, help='whether use warp_op function to get disparity')
parser.add_argument('--gaussian-blur', action='store_true',default=False, help='whether apply gaussian blur')
parser.add_argument('--color-jitter', action='store_true',default=False, help='whether apply color jitter')
parser.add_argument('--kernal_size', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=1)

args = parser.parse_args()
cfg.merge_from_file(args.config_file)

# Set random seed to make sure networks in different processes are same
set_random_seed(args.seed)

# Set up distributed training
num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
is_distributed = num_gpus > 1
args.is_distributed = is_distributed
if is_distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group( backend="nccl", init_method="env://")
    synchronize()
cuda_device = torch.device("cuda:{}".format(args.local_rank))

# Set up tensorboard and logger
os.makedirs(args.logdir, exist_ok=True)
os.makedirs(os.path.join(args.logdir, 'models'), exist_ok=True)
summary_writer = tensorboardX.SummaryWriter(logdir=args.logdir)
logger = setup_logger("Reprojection PSMNet", distributed_rank=args.local_rank, save_dir=args.logdir)
logger.info(f'Loaded config file: \'{args.config_file}\'')
logger.info(f'Running with configs:\n{cfg}')
logger.info(f'Running with {num_gpus} GPUs')

# python -m torch.distributed.launch train_psmnet_ir_reprojection.py --summary-freq 1 --save-freq 1 --logdir ../train_10_14_psmnet_ir_reprojection/debug --debug
# python -m torch.distributed.launch train_psmnet_ir_reprojection.py --config-file configs/remote_train_steps.yaml --summary-freq 10 --save-freq 100 --logdir ../train_10_21_psmnet_smooth_ir_reproj/debug --debug


def train(TrainImgLoader, ValImgLoader):


    for batch_idx, sample in enumerate(TrainImgLoader):

        global_step = batch_idx
        # Train one sample
        train_sample(sample, summary_writer, global_step, isTrain=True)
        print(' Iter ' + str(batch_idx))
        break

    gc.collect()


def train_sample(sample, summary_writer, global_step, isTrain=True):

    # Load data
    img_L = sample['img_L'].to(cuda_device)  # [bs, 3, H, W]
    img_R = sample['img_R'].to(cuda_device)
    img_L_ir_pattern1 = sample['img_L_ir_pattern1'].to(cuda_device)  # [bs, 1, H, W]
    img_R_ir_pattern1 = sample['img_R_ir_pattern1'].to(cuda_device)
    img_L_ir_pattern2 = sample['img_L_ir_pattern2'].to(cuda_device)  # [bs, 1, H, W]
    img_R_ir_pattern2 = sample['img_R_ir_pattern2'].to(cuda_device)
    img_real_L = sample['img_real_L'].to(cuda_device)  # [bs, 3, 2H, 2W]
    img_real_R = sample['img_real_R'].to(cuda_device)  # [bs, 3, 2H, 2W]
    img_real_L_ir_pattern1 = sample['img_real_L_ir_pattern1'].to(cuda_device)
    img_real_R_ir_pattern1 = sample['img_real_R_ir_pattern1'].to(cuda_device)
    img_real_L_ir_pattern2 = sample['img_real_L_ir_pattern2'].to(cuda_device)
    img_real_R_ir_pattern2 = sample['img_real_R_ir_pattern2'].to(cuda_device)
    #print(img_real_L_ir_pattern1.shape)

    bs, c, h, w = img_real_L_ir_pattern1.shape
    hp, wp = random.choice(list(range(h-args.kernal_size))), random.choice(list(range(w-args.kernal_size)))
    for s in range(1,args.kernal_size):
        mask = torch.zeros_like(img_real_L_ir_pattern1)
        mask = (mask != 0)
        mask[:,:,hp:hp+s, wp:wp+s] = True
    # Get reprojection loss on real

        for d in range(192):
            disp = torch.ones_like(img_real_L_ir_pattern1)*d
            real_ir_reproj_loss1, _, _ = get_reprojection_error_old(img_real_L_ir_pattern1, img_real_R_ir_pattern1, disp, mask)
            real_ir_reproj_loss2, _, _ = get_reprojection_error_old(img_real_L_ir_pattern2, img_real_R_ir_pattern2, disp, mask)
            summary_writer.add_scalars('./pattern1/reprojection_loss1_size_' + str(s), {
                                            'pattern1': real_ir_reproj_loss1
                                        }, d)
            summary_writer.add_scalars('./pattern2/reprojection_loss1_size_' + str(s), {
                                            'pattern2': real_ir_reproj_loss2
                                        }, d)
    save_images_grid(summary_writer, 'train_reproj', {'real reprojection' :{'pattern1': img_real_L_ir_pattern1,
                                                'pattern2': img_real_L_ir_pattern2}}, global_step)



if __name__ == '__main__':
    # Obtain dataloader
    train_dataset = MessytableDataset(cfg.SPLIT.TRAIN, gaussian_blur=args.gaussian_blur, color_jitter=args.color_jitter,
                                    debug=args.debug, sub=100)
    val_dataset = MessytableDataset(cfg.SPLIT.VAL, gaussian_blur=args.gaussian_blur, color_jitter=args.color_jitter,
                                    debug=args.debug, sub=10)
    if is_distributed:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(),
                                                            rank=dist.get_rank())
        val_sampler = torch.utils.data.DistributedSampler(val_dataset, num_replicas=dist.get_world_size(),
                                                          rank=dist.get_rank())

        TrainImgLoader = torch.utils.data.DataLoader(train_dataset, args.batch_size, sampler=train_sampler,
                                                     num_workers=cfg.SOLVER.NUM_WORKER, drop_last=True, pin_memory=True)
        ValImgLoader = torch.utils.data.DataLoader(val_dataset, args.batch_size, sampler=val_sampler,
                                                   num_workers=cfg.SOLVER.NUM_WORKER, drop_last=False, pin_memory=True)
    else:
        TrainImgLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                     shuffle=True, num_workers=cfg.SOLVER.NUM_WORKER, drop_last=True)

        ValImgLoader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                                   shuffle=False, num_workers=cfg.SOLVER.NUM_WORKER, drop_last=False)



    # Start training
    train(TrainImgLoader, ValImgLoader)