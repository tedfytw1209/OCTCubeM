# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Revised by Zixuan Zucks Liu @University of Washington

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import os
import time
import math
import json
import argparse
import datetime
import numpy as np

from pathlib import Path

import timm
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import torch
import torch.backends.cudnn as cudnn
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
import wandb


import util.misc as misc
import util.lr_decay as lrd
from util.datasets import build_dataset
from util.datasets import build_transform, load_patient_list

from util.misc import NativeScalerWithGradNormCount as NativeScaler

from util.pos_embed import interpolate_pos_embed, interpolate_temporal_pos_embed
from util.WeightedLabelSmoothingCrossEntropy import WeightedLabelSmoothingCrossEntropy

from util.PatientDataset import TransformableSubset, PatientDataset3D, PatientDatasetCenter2D
from util.PatientDataset_inhouse import create_3d_transforms
from util.datasets import build_transform

from engine_finetune import train_one_epoch, evaluate, init_csv_writer

# RETFound-center
import models_vit
import models_vit_flash_attn

# RETFound-all
import models_vit_3dhead
import models_vit_3dhead_flash_attn

# OCTCube
import models_vit_st
import models_vit_st_joint
import models_vit_st_flash_attn
import models_vit_st_joint_flash_attn
import models_vit_st_flash_attn_nodrop
import model_slivit_baseline



home_directory = os.getenv('HOME') + '/'

dataset_path = '/ext_oph_datasets/DUKE_14_Srin/duke14_processed/'
dataset_name = 'duke14'


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--not_dataset_random_reshuffle_patient', default=False, action='store_true', help='randomly reshuffle the patient dataset, only used (set to be True) for duke14 and hcms fewshot. For standard 5 fold, need to set to be False')

    # Slivit parameters
    parser.add_argument('--slivit_fe_path', default=home_directory + 'OCTCubeM/assets/SLIViT/checkpoints/kermany/feature_extractor.pth'
    , type=str, help='feature extractor for SLIViT')
    parser.add_argument('--slivit_num_of_patches', default=24, type=int, help='number of patches for SLIViT')
    # new necessary functional parameters
    parser.add_argument('--variable_joint', default=False, action='store_true', help='use variable joint attention')
    parser.set_defaults(variable_joint=False)

    # New parameters
    parser.add_argument('--load_non_flash_attn_to_flash_attn', default=False, action='store_true', help='use focal loss')
    parser.add_argument('--always_test', default=False, action='store_true', help='always run test if specified')
    parser.add_argument('--not_use_2d_aug', default=False, action='store_true', help='not use 2D augmentation')
    parser.add_argument('--not_print_logits', default=False, action='store_true', help='not print logits')
    parser.add_argument('--not_save_figs', default=False, action='store_true', help='not save figures')
    parser.add_argument('--same_3_frames', default=False, action='store_true', help='use the same 3 frames to mock 1 frame for 3D spatio-temporal model')
    parser.add_argument('--return_bal_acc', default=False, action='store_true', help='return balanced accuracy')

    # mae_st parameters
    parser.add_argument("--t_patch_size", default=3, type=int)
    parser.add_argument("--num_frames", default=48, type=int)
    parser.add_argument('--max_frames', default=48, type=int, help='maximum number of frames for each patient, equivalent to --pad_to_num_frames in inhouse dataset')
    parser.add_argument("--sep_pos_embed", action="store_true")
    parser.set_defaults(sep_pos_embed=True)
    parser.add_argument("--cls_embed", action="store_true")
    parser.set_defaults(cls_embed=True)
    parser.add_argument("--transform_type", default="frame_2D", type=str, choices=["frame_2D", "monai_3D"]) # only glaucoma has volume_3D transform
    parser.add_argument("--color_mode", default="rgb", type=str, choices=["rgb", "gray"])
    parser.add_argument("--smaller_temporal_crop", default='interp', type=str, choices=['interp', 'crop'], help='interpolation type for temporal position embedding')

    # Patient dataset parameters
    # Dataset parameters
    parser.add_argument('--data_path', default=home_directory + dataset_path, type=str, help='dataset path')
    parser.add_argument('--patient_dataset', default=True, action='store_true', help='Use patient dataset')
    parser.add_argument('--patient_dataset_type', default='Center2D', type=str, choices=['3D', 'Center2D', 'Center2D_flash_attn',  '3D_flash_attn', '3D_st', '3D_st_joint', '3D_st_flash_attn', '3D_st_joint_flash_attn', '3D_st_flash_attn_nodrop', 'convnext_slivit'], help='patient dataset type')
    parser.add_argument('--dataset_mode', default='frame', type=str, choices=['frame', 'volume'], help='dataset mode for the patient dataset')
    parser.add_argument('--iterate_mode', default='patient', type=str, choices=['visit', 'patient'], help='iterate mode for the patient dataset, duke14 uses patient')
    parser.add_argument('--name_split_char', default='_', type=str, help='split character for the image filename')
    parser.add_argument('--patient_idx_loc', default=5, type=int, help='patient index location in the image filename, e.g., 2 for amd_oct_2_1.png')
    parser.add_argument('--visit_idx_loc', default=None, type=int, help='[optional] visit index location in the image filename')
    parser.add_argument('--cls_unique', default=False, action='store_true', help='use unique class labels for the dataset')

    # Task parameters
    parser.add_argument('--task_mode', default='multi_cls', type=str, choices=['binary_cls', 'multi_cls'],
                        help='Task mode for the dataset (no multi_label here)')
    parser.add_argument('--val_metric', default='AUPRC', type=str, choices=['AUC', 'ACC', 'AUPRC', 'BalAcc'], help='Validation metric for early stopping, newly added BalAcc (only used in AI-READI)')

    parser.add_argument('--save_model', default=False, action='store_true', help='save model')
    parser.add_argument('--enable_early_stop', default=False, action='store_true', help='enable early stop, currently not used in this script')
    parser.add_argument('--early_stop_patience', default=10, type=int, help='early stop patience, currently not used in this script')

    # K_fold cross validation
    parser.add_argument('--k_fold', default=True, action='store_true', help='Use K-fold cross validation')
    parser.add_argument('--k_folds', default=5, type=int, help='number of folds for K-fold cross validation')

    # Original parameters
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--val_batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--few_shot', default=False, action='store_true',
                        help='finetune from checkpoint')
    parser.add_argument('--task', default=f'./finetune_{dataset_name}/',type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    parser.add_argument('--nb_classes', default=3, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./outputs_ft/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--rank', default=-1, type=int)

    # wandb parameters
    parser.add_argument('--use_wandb', default=True, action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--no_wandb', dest='use_wandb', action='store_false',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--wandb_project', default='OCTCubeM', type=str,
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', default=None, type=str,
                        help='Weights & Biases entity (username or team name)')
    parser.add_argument('--wandb_run_name', default=None, type=str,
                        help='Weights & Biases run name')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    # Save args to a json file
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    if not args.return_bal_acc:
        val_bal_acc = None
        test_bal_acc = None

    cudnn.benchmark = True
    if not args.patient_dataset:
        dataset_train = build_dataset(is_train='train', args=args)
        dataset_val = build_dataset(is_train='val', args=args)
        dataset_test = build_dataset(is_train='test', args=args)
        assert args.k_fold is False
    else:
        if args.transform_type == 'frame_2D':
            train_transform = build_transform(is_train='train', args=args)
            val_transform = build_transform(is_train='val', args=args)
        elif args.transform_type == 'monai_3D':
            train_transform, val_transform = create_3d_transforms(**vars(args))

        if args.patient_dataset_type == '3D' or args.patient_dataset_type == '3D_st' or args.patient_dataset_type.startswith('3D') or args.patient_dataset_type == 'convnext_slivit':
            dataset_for_Kfold = PatientDataset3D(root_dir=args.data_path, patient_idx_loc=args.patient_idx_loc, transform=None, dataset_mode=args.dataset_mode, name_split_char=args.name_split_char, cls_unique=args.cls_unique, iterate_mode=args.iterate_mode, max_frames=args.max_frames, mode=args.color_mode, transform_type=args.transform_type, same_3_frames=args.same_3_frames, downsample_width=True, random_shuffle_patient=not args.not_dataset_random_reshuffle_patient)
        elif args.patient_dataset_type == 'Center2D' or args.patient_dataset_type == 'Center2D_flash_attn':
            dataset_for_Kfold = PatientDatasetCenter2D(root_dir=args.data_path, patient_idx_loc=args.patient_idx_loc, transform=None, dataset_mode=args.dataset_mode, name_split_char=args.name_split_char, cls_unique=args.cls_unique, iterate_mode=args.iterate_mode, downsample_width=True, random_shuffle_patient=not args.not_dataset_random_reshuffle_patient)

        if args.k_fold:
            # Assuming KFold setup is external, and args.fold indicates the current fold
            kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
            if args.iterate_mode == 'patient':
                patient_indices = range(len(dataset_for_Kfold))
                # folds = list(kf.split(patient_indices))
                folds = misc.generate_sublists(patient_indices, N=15, k=3, iterations=args.k_folds, random_state=0)
                print(patient_indices)
            elif args.iterate_mode == 'visit':
                patient_mapping_visit_indices = list(dataset_for_Kfold.mapping_patient2visit.keys())
                folds = list(kf.split(patient_mapping_visit_indices))

            print([dataset_for_Kfold.patients[idx]['class_idx'] for idx in dataset_for_Kfold.patients.keys()])
            print(folds)

        # skip elif args.single_fold, as no need for small dataset

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # Initialize wandb
    if args.use_wandb and global_rank == 0:
        project_name = args.wandb_project
        args.task = args.task[:120]  # Wandb group name max length is 128
        group_name = args.task.replace('.','').replace('/', '_')
        wandb_task_name = args.wandb_run_name if args.wandb_run_name else (args.task.replace('.','').replace('/', '_') + datetime.datetime.now().strftime("_%Y%m%d_%H%M%S"))

        wandb.init(
            project=project_name,
            entity=args.wandb_entity,
            name=wandb_task_name,
            group=group_name,
            config=vars(args),
            dir=os.path.join(args.log_dir, wandb_task_name) if args.log_dir else None,
            reinit=True
        )

    if args.k_fold and args.patient_dataset:
        fold_results = []
        fold_results_test = []
        perfect_patience = 5
        print(f"Start K-fold cross validation for {args.k_folds} folds")
        for fold in range(args.k_folds):
            print(f"Fold {fold}")

            fold_perfect_patience_cnt = 0
            if args.iterate_mode == 'patient':
                train_indices, val_indices = folds[fold]
            elif args.iterate_mode == 'visit':
                idx_train_pat_id, idx_val_pat_id = folds[fold]
                train_pat_id, val_pat_id = [patient_mapping_visit_indices[idx] for idx in idx_train_pat_id], [patient_mapping_visit_indices[idx] for idx in idx_val_pat_id]
                train_indices = dataset_for_Kfold.get_visit_idx(train_pat_id)
                val_indices = dataset_for_Kfold.get_visit_idx(val_pat_id)

            if args.patient_dataset_type == '3D' or args.patient_dataset_type == '3D_st' or args.patient_dataset_type.startswith('3D') or args.patient_dataset_type == 'convnext_slivit':
                if args.few_shot:
                    dataset_train = TransformableSubset(dataset_for_Kfold, val_indices)
                    dataset_val = TransformableSubset(dataset_for_Kfold, train_indices)
                else:
                    dataset_train = TransformableSubset(dataset_for_Kfold, train_indices)
                    dataset_val = TransformableSubset(dataset_for_Kfold, val_indices)
                dataset_train.update_dataset_transform(train_transform)

            elif args.patient_dataset_type == 'Center2D' or args.patient_dataset_type == 'Center2D_flash_attn':
                if args.few_shot:
                    dataset_train = TransformableSubset(dataset_for_Kfold, val_indices, transform=train_transform)
                    dataset_val = TransformableSubset(dataset_for_Kfold, train_indices, transform=val_transform)
                else:
                    dataset_train = TransformableSubset(dataset_for_Kfold, train_indices, transform=train_transform)
                    dataset_val = TransformableSubset(dataset_for_Kfold, val_indices, transform=val_transform)

            dataset_test = dataset_val
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            print(f"Start train val test for {len(dataset_train)} train, {len(dataset_val)} val, {len(dataset_test)} test")

            print("Sampler_train = %s" % str(sampler_train))
            if args.dist_eval:
                if len(dataset_val) % num_tasks != 0:
                    print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                        'This will slightly alter validation results as extra duplicate entries are added to achieve '
                        'equal num of samples per-process.')
                sampler_val = torch.utils.data.DistributedSampler(
                    dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
                sampler_test = sampler_val
            else:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
                sampler_test = sampler_val

            if global_rank == 0 and args.log_dir is not None and not args.eval:
                os.makedirs(args.log_dir, exist_ok=True)
                log_writer = SummaryWriter(log_dir=args.log_dir+args.task)
            else:
                log_writer = None

            data_loader_train = torch.utils.data.DataLoader(
                dataset_train, sampler=sampler_train,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=True,
            )

            data_loader_val = torch.utils.data.DataLoader(
                dataset_val, sampler=sampler_val,
                batch_size=args.val_batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )

            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, sampler=sampler_test,
                batch_size=args.val_batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )

            mixup_fn = None
            mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
            if mixup_active:
                print("Mixup is activated!")
                mixup_fn = Mixup(
                    mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                    prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                    label_smoothing=args.smoothing, num_classes=args.nb_classes)
            if args.patient_dataset_type == '3D':
                model = models_vit_3dhead.__dict__[args.model](
                    img_size=args.input_size,
                    num_classes=args.nb_classes,
                    drop_path_rate=args.drop_path,
                    global_pool=args.global_pool,
                )
            elif args.patient_dataset_type == 'Center2D':
                model = models_vit.__dict__[args.model](
                    img_size=args.input_size,
                    num_classes=args.nb_classes,
                    drop_path_rate=args.drop_path,
                    global_pool=args.global_pool,
                )
            elif args.patient_dataset_type == 'Center2D_flash_attn':
                model = models_vit_flash_attn.__dict__[args.model](
                    img_size=args.input_size,
                    num_classes=args.nb_classes,
                    drop_path_rate=args.drop_path,
                    global_pool=args.global_pool,
                )
            elif args.patient_dataset_type == '3D_flash_attn':
                print('Use 3D flash attn model')
                model = models_vit_3dhead_flash_attn.__dict__[args.model](
                    img_size=args.input_size,
                    num_classes=args.nb_classes,
                    drop_path_rate=args.drop_path,
                    global_pool=args.global_pool,
                )
            elif args.patient_dataset_type == '3D_st':
                print('Use 3D spatio-temporal model')
                model = models_vit_st.__dict__[args.model](
                    img_size=args.input_size,
                    num_classes=args.nb_classes,
                    drop_path_rate=args.drop_path,
                    global_pool=args.global_pool,
                    t_patch_size=args.t_patch_size,
                    num_frames=args.num_frames,
                    sep_pos_embed=args.sep_pos_embed,
                    cls_embed=args.cls_embed,
                )
            elif args.patient_dataset_type == '3D_st_joint':
                model = models_vit_st_joint_flash_attn.__dict__[args.model](
                    img_size=args.input_size,
                    num_classes=args.nb_classes,
                    drop_path_rate=args.drop_path,
                    global_pool=args.global_pool,
                    t_patch_size=args.t_patch_size,
                    num_frames=args.num_frames,
                    sep_pos_embed=args.sep_pos_embed,
                    cls_embed=args.cls_embed,
                    transform_type=args.transform_type,
                    color_mode=args.color_mode,
                    smaller_temporal_crop=args.smaller_temporal_crop,
                    use_high_res_patch_embed=args.use_high_res_patch_embed,
                )
            elif args.patient_dataset_type == '3D_st_flash_attn':
                print('Use 3D spatio-temporal model w/ flash attention')
                model = models_vit_st_flash_attn.__dict__[args.model](
                    num_frames=args.num_frames,
                    t_patch_size=args.t_patch_size,
                    img_size=args.input_size,
                    num_classes=args.nb_classes,
                    drop_path_rate=args.drop_path,
                    global_pool=args.global_pool,
                    sep_pos_embed=args.sep_pos_embed,
                    cls_embed=args.cls_embed,
                    use_flash_attention=True,
                )
            elif args.patient_dataset_type == '3D_st_flash_attn_nodrop':
                print('Use 3D spatio-temporal model w/ flash attention and no dropout')
                model = models_vit_st_flash_attn_nodrop.__dict__[args.model](
                        num_frames=args.num_frames,
                        t_patch_size=args.t_patch_size,
                        image_size=args.input_size,
                        num_classes=args.nb_classes,
                        drop_path_rate=args.drop_path,
                        global_pool=args.global_pool,
                        sep_pos_embed=args.sep_pos_embed,
                        cls_embed=args.cls_embed,
                        use_flash_attention=True
                    )

            elif args.patient_dataset_type == 'convnext_slivit':
                model = model_slivit_baseline.get_slivit_model(args)
            if args.finetune and not args.eval and args.patient_dataset_type != 'convnext_slivit':
                checkpoint = torch.load(args.finetune, map_location='cpu')

                print("Load pre-trained checkpoint from: %s" % args.finetune)
                checkpoint_model = checkpoint['model']
                state_dict = model.state_dict()
                for k in ['head.weight', 'head.bias']:
                    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del checkpoint_model[k]

                # interpolate position embedding
                if args.sep_pos_embed and (args.patient_dataset_type == '3D_st' or args.patient_dataset_type == '3D_st_flash_attn' or args.patient_dataset_type == '3D_st_flash_attn_nodrop' or args.patient_dataset_type.startswith('3D_st')):
                    interpolate_pos_embed(model, checkpoint_model)
                    interpolate_temporal_pos_embed(model, checkpoint_model, smaller_interpolate_type=args.smaller_temporal_crop)

                else:
                    interpolate_pos_embed(model, checkpoint_model)


                # load pre-trained model
                if args.load_non_flash_attn_to_flash_attn:
                    msg = model.load_state_dict_to_backbone(checkpoint["model"])
                else:
                    msg = model.load_state_dict(checkpoint_model, strict=False)
                print(msg)
                print(msg.missing_keys)
                if args.global_pool:
                    if args.patient_dataset_type == '3D' or args.patient_dataset_type == '3D_flash_attn':
                        assert set(msg.missing_keys) == {'fc_aggregate_cls.weight', 'fc_aggregate_cls.bias',
                        'aggregate_cls_norm.weight', 'aggregate_cls_norm.bias',
                        'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
                    elif args.patient_dataset_type == '3D_st_flash_attn_nodrop':
                        print('Goin right way')
                        assert set(msg.missing_keys) == {'fc_aggregate_cls.weight', 'fc_aggregate_cls.bias',
                        'aggregate_cls_norm.weight', 'aggregate_cls_norm.bias',
                        'head.weight', 'head.bias'}
                    elif args.patient_dataset_type == 'Center2D' or args.patient_dataset_type == 'Center2D_flash_attn':
                        assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
                    elif args.patient_dataset_type == '3D_st' or args.patient_dataset_type == '3D_st_joint' or args.patient_dataset_type == '3D_st_flash_attn' or args.patient_dataset_type == '3D_st_joint_flash_attn':
                        assert set(msg.missing_keys) == {'head.weight', 'head.bias'}
                else:
                    assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

                # manually initialize fc layer
                trunc_normal_(model.head.weight, std=2e-5)
            model.to(device)

            model_without_ddp = model
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print("Model = %s" % str(model_without_ddp))
            print('number of params (M): %.2f' % (n_parameters / 1.e6))

            eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

            if args.lr is None:  # only base_lr is specified
                args.lr = args.blr * eff_batch_size / 256

            print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
            print("actual lr: %.2e" % args.lr)

            print("accumulate grad iterations: %d" % args.accum_iter)
            print("effective batch size: %d" % eff_batch_size)
            if args.distributed:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
                model_without_ddp = model.module

            if args.patient_dataset_type == 'convnext_slivit':
                # don't use layer-wise lr decay for convnext_slivit
                param_groups = model_without_ddp.parameters()
            else:
                # build optimizer with layer-wise lr decay (lrd)
                param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                    no_weight_decay_list=model_without_ddp.no_weight_decay(),
                    layer_decay=args.layer_decay
                )

            optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
            loss_scaler = NativeScaler()

            if mixup_fn is not None:
                # smoothing is handled with mixup label transform
                criterion = SoftTargetCrossEntropy()
            elif args.smoothing > 0.:
                criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
            else:
                criterion = torch.nn.CrossEntropyLoss()

            print("criterion = %s" % str(criterion))

            #misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

            if args.eval:
                test_mode = f'test_fold_{fold}'
                init_csv_writer(args.task, mode=test_mode)
                test_stats, auc_roc, auc_pr = evaluate(data_loader_test, model, device, args.task, epoch=0, mode=test_mode, num_class=args.nb_classes, criterion=criterion, task_mode=args.task_mode, disease_list=None, return_bal_acc=args.return_bal_acc, args=args)
                if args.return_bal_acc:
                    test_auc_pr, test_bal_acc = auc_pr
                exit(0)

            print(f"Start training for {args.epochs} epochs")
            start_time = time.time()
            max_accuracy = 0.0
            max_auc = 0.0
            max_auc_pr = 0.0
            max_epoch = 0
            max_accuracy_test = 0.0
            max_auc_test = 0.0
            max_auc_pr_test = 0.0
            max_epoch_test = 0
            val_mode = f'val_fold_{fold}'

            # add balance accuracy tracker
            max_bal_acc = 0.0
            max_bal_acc_test = 0.0

            if args.task_mode == 'binary_cls' or args.task_mode == 'multi_cls':
                init_csv_writer(args.task, mode=val_mode)

            for epoch in range(args.start_epoch, args.epochs):
                if args.distributed:
                    data_loader_train.sampler.set_epoch(epoch)
                train_stats = train_one_epoch(
                    model, criterion, data_loader_train,
                    optimizer, device, epoch, loss_scaler,
                    args.clip_grad, mixup_fn,
                    log_writer=log_writer,
                    args=args
                )
                if train_stats is None:
                    # downscale the learning rate by 2
                    for param_group in optimizer.param_groups:
                        param_group['lr'] /= 2
                    print(f"Downscale the learning rate to {param_group['lr']}")

                if args.patient_dataset_type == '3D' or args.patient_dataset_type == '3D_st' or args.patient_dataset_type.startswith('3D') or args.patient_dataset_type == 'convnext_slivit':
                    dataset_train.remove_dataset_transform()
                    dataset_val.update_dataset_transform(val_transform)
                    # print('go to val:', dataset_val.dataset.transform)
                disease_list = None
                try:
                    val_stats, val_auc_roc, val_auc_pr = evaluate(data_loader_val, model, device, args.task, epoch, mode=val_mode, num_class=args.nb_classes, criterion=criterion, task_mode=args.task_mode, disease_list=disease_list, return_bal_acc=args.return_bal_acc, args=args)
                    if args.return_bal_acc:
                        val_auc_pr, val_bal_acc = val_auc_pr
                except ValueError as e:
                    print(e)
                    print('break')
                    print(f'break at {epoch}', file=open(os.path.join(args.output_dir, f"auc_fold_{fold}.txt"), mode="a"))
                    break

                max_flag = False
                if args.val_metric == 'AUC':
                    print('Use AUC as the validation metric')
                    if max_auc <= val_auc_roc:
                        max_auc = val_auc_roc
                        if max_auc < val_auc_roc:
                            max_epoch = epoch
                            max_flag = True
                        elif max_accuracy <= val_stats['acc1']:
                            max_accuracy = val_stats['acc1']
                            max_epoch = epoch
                            max_flag = True
                        elif max_auc_pr <= val_auc_pr:
                            max_auc_pr = val_auc_pr
                            max_epoch = epoch
                            max_flag = True
                elif args.val_metric == 'AUPRC':
                    print('Use AUPRC as the validation metric')
                    if max_auc_pr <= val_auc_pr:

                        if max_auc_pr < val_auc_pr:
                            max_epoch = epoch
                            max_auc = val_auc_roc
                            max_accuracy = val_stats['acc1']
                            max_flag = True
                        max_auc_pr = val_auc_pr
                        if max_accuracy <= val_stats['acc1']:
                            max_accuracy = val_stats['acc1']
                            max_auc = val_auc_roc
                            max_epoch = epoch
                            max_flag = True
                        elif max_auc <= val_auc_roc:
                            max_auc = val_auc_roc
                            max_accuracy = val_stats['acc1']
                            max_epoch = epoch
                            max_flag = True
                        if val_bal_acc is not None and val_bal_acc > max_bal_acc:
                            max_bal_acc = val_bal_acc
                            max_flag = True
                    elif args.val_metric == 'BalAcc':
                        print('Use BalAcc as the validation metric')
                        if max_bal_acc <= val_bal_acc:
                            if max_bal_acc < val_bal_acc:
                                max_epoch = epoch
                                max_auc = val_auc_roc
                                max_accuracy = val_stats['acc1']
                                max_auc_pr = val_auc_pr
                                max_flag = True
                            max_bal_acc = val_bal_acc
                            if max_auc < val_auc_roc:
                                max_auc = val_auc_roc
                                max_accuracy = val_stats['acc1']
                                max_auc_pr = val_auc_pr
                                max_epoch = epoch
                                max_flag = True
                            if max_auc_pr < val_auc_pr:
                                max_auc_pr = val_auc_pr
                                max_accuracy = val_stats['acc1']
                                max_auc = val_auc_roc
                                max_epoch = epoch
                                max_flag = True
                            if max_accuracy < val_stats['acc1']:
                                max_accuracy = val_stats['acc1']
                                max_auc = val_auc_roc
                                max_auc_pr = val_auc_pr
                                max_epoch = epoch
                                max_flag = True

                if max_flag is True:
                    print(f"Max AUC: {max_auc}, Max ACC: {max_accuracy}, Max AUCPR: {max_auc_pr}, Max Bal Acc: {max_bal_acc}, at epoch {epoch}")
                    print(f"Max AUC: {max_auc}, Max ACC: {max_accuracy}, Max AUCPR: {max_auc_pr}, Max Bal Acc: {max_bal_acc}, at epoch {epoch}", file=open(os.path.join(args.output_dir, f"auc_fold_{fold}.txt"), mode="a"))
                    if args.output_dir and args.save_model:
                        misc.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)
                if max_flag or epoch == (args.epochs - 1):
                    test_mode = f'test_fold_{fold}'
                    init_csv_writer(args.task, mode=test_mode)
                    try:
                        test_stats, test_auc_roc, test_auc_pr = evaluate(data_loader_test, model, device, args.task, epoch, mode=test_mode, num_class=args.nb_classes, criterion=criterion, task_mode=args.task_mode, disease_list=disease_list, return_bal_acc=args.return_bal_acc, args=args)
                        if args.return_bal_acc:
                            test_auc_pr, test_bal_acc = test_auc_pr
                    except ValueError as e:
                        print(e)
                        print('break')
                        break

                    # Log test stats to wandb
                    if args.use_wandb and global_rank == 0:
                        wandb_test_log = {f'fold_{fold}/epoch': epoch}
                        wandb_test_log.update({f'fold_{fold}/test_{k}': v for k, v in test_stats.items()})
                        wandb_test_log.update({
                            f'fold_{fold}/test_auc': test_auc_roc,
                            f'fold_{fold}/test_auc_pr': test_auc_pr,
                        })
                        if args.return_bal_acc and test_bal_acc is not None:
                            wandb_test_log[f'fold_{fold}/test_bal_acc'] = test_bal_acc
                        wandb.log(wandb_test_log, step=epoch + fold * args.epochs)

                    max_flag_test = False
                    if args.val_metric == 'AUC':
                        print('Use AUC as the validation metric')
                        if max_auc_test <= test_auc_roc:
                            max_auc_test = test_auc_roc
                            if max_auc_test < test_auc_roc:
                                max_epoch_test = epoch
                                max_flag_test = True
                            elif max_accuracy_test <= test_stats['acc1']:
                                max_accuracy_test = test_stats['acc1']
                                max_epoch_test = epoch
                                max_flag_test = True
                            elif max_auc_pr_test <= test_auc_pr:
                                max_auc_pr_test = test_auc_pr
                                max_epoch_test = epoch
                                max_flag_test = True
                    elif args.val_metric == 'AUPRC':
                        print('Use AUPRC as the validation metric')
                        if max_auc_pr_test <= test_auc_pr:
                            if max_auc_pr_test < test_auc_pr:
                                max_epoch_test = epoch
                                max_auc_test = test_auc_roc
                                max_accuracy_test = test_stats['acc1']
                                max_flag_test = True
                            max_auc_pr_test = test_auc_pr
                            if max_accuracy_test <= test_stats['acc1']:
                                max_accuracy_test = test_stats['acc1']
                                max_auc_test = test_auc_roc
                                max_epoch_test = epoch
                                max_flag_test = True
                            elif max_auc_test <= test_auc_roc:
                                max_auc_test = test_auc_roc
                                max_accuracy_test = test_stats['acc1']
                                max_epoch_test = epoch
                                max_flag_test = True
                            if args.return_bal_acc:
                                max_bal_acc_test = test_bal_acc
                                max_flag_test = True
                    elif args.val_metric == 'BalAcc':
                        print('Use BalAcc as the validation metric')
                        if max_bal_acc <= val_bal_acc:
                            if max_bal_acc < val_bal_acc:
                                max_epoch = epoch
                                max_auc = val_auc_roc
                                max_accuracy = val_stats['acc1']
                                max_auc_pr = val_auc_pr
                                max_flag = True
                            max_bal_acc = val_bal_acc
                            if max_auc < val_auc_roc:
                                max_auc = val_auc_roc
                                max_accuracy = val_stats['acc1']
                                max_auc_pr = val_auc_pr
                                max_epoch = epoch
                                max_flag = True
                            if max_auc_pr < val_auc_pr:
                                max_auc_pr = val_auc_pr
                                max_accuracy = val_stats['acc1']
                                max_auc = val_auc_roc
                                max_epoch = epoch
                                max_flag = True
                            if max_accuracy < val_stats['acc1']:
                                max_accuracy = val_stats['acc1']
                                max_auc = val_auc_roc
                                max_auc_pr = val_auc_pr
                                max_epoch = epoch
                                max_flag = True
                    if max_flag_test is True:
                        print(f"Max AUC: {max_auc_test}, Max ACC: {max_accuracy_test}, Max AUCPR: {max_auc_pr_test}, Max Bal Acc: {max_bal_acc_test}, at epoch {epoch}")
                        print(f"Max AUC: {max_auc_test}, Max ACC: {max_accuracy_test}, Max AUCPR: {max_auc_pr_test}, Max Bal Acc: {max_bal_acc_test}, at epoch {epoch}", file=open(os.path.join(args.output_dir, f"auc_test_fold_{fold}.txt"), mode="a"))


                if log_writer is not None:
                    log_writer.add_scalar('perf/val_acc1', val_stats['acc1'], epoch)
                    log_writer.add_scalar('perf/val_auc', val_auc_roc, epoch)
                    log_writer.add_scalar('perf/val_auc_pr', val_auc_pr, epoch)
                    log_writer.add_scalar('perf/val_loss', val_stats['loss'], epoch)
                    if args.return_bal_acc and val_bal_acc is not None:
                        log_writer.add_scalar('perf/val_bal_acc', val_bal_acc, epoch)

                # Log to wandb
                if args.use_wandb and global_rank == 0 and train_stats is not None:
                    wandb_log = {f'fold_{fold}/epoch': epoch}
                    wandb_log.update({f'fold_{fold}/train_{k}': v for k, v in train_stats.items()})
                    wandb_log.update({f'fold_{fold}/val_{k}': v for k, v in val_stats.items()})
                    wandb_log.update({
                        f'fold_{fold}/val_auc': val_auc_roc,
                        f'fold_{fold}/val_auc_pr': val_auc_pr,
                        f'fold_{fold}/max_val_auc': max_auc,
                        f'fold_{fold}/max_val_acc': max_accuracy,
                        f'fold_{fold}/max_val_auc_pr': max_auc_pr,
                    })
                    if args.return_bal_acc and val_bal_acc is not None:
                        wandb_log[f'fold_{fold}/val_bal_acc'] = val_bal_acc
                        wandb_log[f'fold_{fold}/max_val_bal_acc'] = max_bal_acc
                    wandb.log(wandb_log, step=epoch + fold * args.epochs)

                if train_stats is not None:
                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                    'epoch': epoch,
                                    'n_parameters': n_parameters,
                                    'max_val_acc': max_accuracy,
                                    'max_val_auc': max_auc,
                                    'max_val_auc_pr': max_auc_pr,
                                    'max_val_epoch': max_epoch,
                                    'max_val_bal_acc': max_bal_acc}

                if args.output_dir and misc.is_main_process():
                    if log_writer is not None:
                        log_writer.flush()
                    with open(os.path.join(args.output_dir, f"log_fold_{fold}.txt"), mode="a") as f:
                        f.write(json.dumps(log_stats) + "\n")

                if args.patient_dataset_type == '3D' or args.patient_dataset_type == '3D_st' or args.patient_dataset_type.startswith('3D'):
                    dataset_val.remove_dataset_transform()
                    dataset_train.update_dataset_transform(train_transform)

                if max_auc > 0.99999 and max_accuracy > 0.9999999 * 100 and max_auc_pr > 0.99999 and max_auc_test > 0.99999 and max_auc_pr_test > 0.99999 and max_accuracy_test > 0.9999999 * 100:
                    fold_perfect_patience_cnt += 1
                    print(f"Fold perfect patience count: {fold_perfect_patience_cnt} at epoch {epoch}")
                    print(f"Fold perfect patience count: {fold_perfect_patience_cnt} at epoch {epoch}", file=open(os.path.join(args.output_dir, f"auc_fold_{fold}.txt"), mode="a"))
                    print(f"Fold perfect patience count: {fold_perfect_patience_cnt} at epoch {epoch}", file=open(os.path.join(args.output_dir, f"auc_test_fold_{fold}.txt"), mode="a"))
                    if fold_perfect_patience_cnt >= perfect_patience:
                        print(f"Early stopping at epoch {epoch}, as the model is already perfect. max_auc: {max_auc}, max_auc_pr_test: {max_auc_pr}, max_accuracy: {max_accuracy}")
                        print(f"Early stopping at epoch {epoch}, as the model is already perfect. max_auc: {max_auc}, max_auc_pr_test: {max_auc_pr}, max_accuracy: {max_accuracy}", file=open(os.path.join(args.output_dir, f"auc_fold_{fold}.txt"), mode="a"))

                        print(f"Early stopping at epoch {epoch}, as the model is already perfect. max_auc_test: {max_auc_test}, max_auc_pr_test: {max_auc_pr_test}, max_accuracy_test: {max_accuracy_test}")
                        print(f"Early stopping at epoch {epoch}, as the model is already perfect. max_auc_test: {max_auc_test}, max_auc_pr_test: {max_auc_pr_test}, max_accuracy_test: {max_accuracy_test}", file=open(os.path.join(args.output_dir, f"auc_test_fold_{fold}.txt"), mode="a"))
                        break

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str))
            print('Training time {}'.format(total_time_str), file=open(os.path.join(args.output_dir, f"time_fold_{fold}.txt"), mode="a"))
            if args.return_bal_acc:
                fold_results.append((max_auc, max_accuracy, max_auc_pr, max_bal_acc))
                fold_results_test.append((max_auc_test, max_accuracy_test, max_auc_pr_test, max_bal_acc_test))
            else:
                fold_results.append((max_auc, max_accuracy, max_auc_pr))
                fold_results_test.append((max_auc_test, max_accuracy_test, max_auc_pr_test))

        # Calculate average AUC and accuracy and std
        fold_results = np.array(fold_results)
        fold_results_mean = np.mean(fold_results, axis=0)
        fold_results_std = np.std(fold_results, axis=0)

        print(f"Fold results: {fold_results}\nMean: {fold_results_mean}\nStd: {fold_results_std}")
        print(f"Fold results: {fold_results}\nMean: {fold_results_mean}\nStd: {fold_results_std}",
            file=open(os.path.join(args.output_dir, "fold_results.txt"), mode="a"))

        # Calculate average AUC and accuracy and std
        fold_results_test = np.array(fold_results_test)
        fold_results_mean_test = np.mean(fold_results_test, axis=0)
        fold_results_std_test = np.std(fold_results_test, axis=0)

        print(f"Fold results: {fold_results_test}\nMean: {fold_results_mean_test}\nStd: {fold_results_std_test}")
        print(f"Fold results: {fold_results_test}\nMean: {fold_results_mean_test}\nStd: {fold_results_std_test}",
            file=open(os.path.join(args.output_dir, "fold_results_test.txt"), mode="a"))

        # Log final results to wandb
        if args.use_wandb and global_rank == 0:
            wandb_summary = {
                'final/mean_val_auc': fold_results_mean[0],
                'final/mean_val_acc': fold_results_mean[1],
                'final/mean_val_auc_pr': fold_results_mean[2],
                'final/std_val_auc': fold_results_std[0],
                'final/std_val_acc': fold_results_std[1],
                'final/std_val_auc_pr': fold_results_std[2],
                'final/mean_test_auc': fold_results_mean_test[0],
                'final/mean_test_acc': fold_results_mean_test[1],
                'final/mean_test_auc_pr': fold_results_mean_test[2],
                'final/std_test_auc': fold_results_std_test[0],
                'final/std_test_acc': fold_results_std_test[1],
                'final/std_test_auc_pr': fold_results_std_test[2],
            }
            if args.return_bal_acc:
                wandb_summary['final/mean_val_bal_acc'] = fold_results_mean[3]
                wandb_summary['final/std_val_bal_acc'] = fold_results_std[3]
                wandb_summary['final/mean_test_bal_acc'] = fold_results_mean_test[3]
                wandb_summary['final/std_test_bal_acc'] = fold_results_std_test[3]
            wandb.log(wandb_summary)
            wandb.finish()

    else:  # args.distributed:
        assert args.patient_dataset is False

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias

        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        if args.dist_eval:
            if len(dataset_test) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)


        if global_rank == 0 and args.log_dir is not None and not args.eval:
            os.makedirs(args.log_dir, exist_ok=True)
            log_writer = SummaryWriter(log_dir=args.log_dir+args.task)
        else:
            log_writer = None

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )


        mixup_fn = None
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            print("Mixup is activated!")
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=args.nb_classes)

        model = models_vit.__dict__[args.model](
            img_size=args.input_size,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )

        if args.finetune and not args.eval:
            checkpoint = torch.load(args.finetune, map_location='cpu')

            print("Load pre-trained checkpoint from: %s" % args.finetune)
            checkpoint_model = checkpoint['model']
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            # interpolate position embedding
            interpolate_pos_embed(model, checkpoint_model)

            # load pre-trained model
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(msg)

            if args.global_pool:
                assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
            else:
                assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

            # manually initialize fc layer
            trunc_normal_(model.head.weight, std=2e-5)

        model.to(device)

        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("Model = %s" % str(model_without_ddp))
        print('number of params (M): %.2f' % (n_parameters / 1.e6))

        eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

        if args.lr is None:  # only base_lr is specified
            args.lr = args.blr * eff_batch_size / 256

        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)

        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module

        # build optimizer with layer-wise lr decay (lrd)
        param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
            no_weight_decay_list=model_without_ddp.no_weight_decay(),
            layer_decay=args.layer_decay
        )
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
        loss_scaler = NativeScaler()

        if mixup_fn is not None:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing > 0.:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        print("criterion = %s" % str(criterion))

        #misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

        if args.eval:
            test_stats,auc_roc, auc_pr = evaluate(data_loader_test, model, device, args.task, epoch=0, mode='test', num_class=args.nb_classes)
            exit(0)

        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        max_accuracy = 0.0
        max_auc = 0.0
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            train_stats = train_one_epoch(
                model, criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, mixup_fn,
                log_writer=log_writer,
                args=args
            )

            val_stats,val_auc_roc, auc_pr = evaluate(data_loader_val, model, device, args.task,epoch, mode='val', num_class=args.nb_classes)
            if max_auc <= val_auc_roc:
                max_auc = val_auc_roc



                if args.output_dir:
                    misc.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch)


            if epoch==(args.epochs-1):
                test_stats,auc_roc, auc_pr = evaluate(data_loader_test, model, device, args.task,epoch, mode='test', num_class=args.nb_classes)


            if log_writer is not None:
                log_writer.add_scalar('perf/val_acc1', val_stats['acc1'], epoch)
                log_writer.add_scalar('perf/val_auc', val_auc_roc, epoch)
                log_writer.add_scalar('perf/val_loss', val_stats['loss'], epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}

            if args.output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a") as f:
                    f.write(json.dumps(log_stats) + "\n")


        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
