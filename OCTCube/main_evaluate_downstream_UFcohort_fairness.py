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
import pandas as pd

from pathlib import Path

import timm
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import torch
import torch.backends.cudnn as cudnn
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter


import util.misc as misc
import util.lr_decay as lrd
from util.datasets import build_dataset
from util.datasets import build_transform, load_patient_list

from util.misc import NativeScalerWithGradNormCount as NativeScaler

from util.pos_embed import interpolate_pos_embed, interpolate_temporal_pos_embed
from util.WeightedLabelSmoothingCrossEntropy import WeightedLabelSmoothingCrossEntropy

from util.PatientDataset import TransformableSubset, PatientDataset3D, PatientDatasetCenter2D, PatientDataset2D
from util.PatientDataset_inhouse import create_3d_transforms
from util.datasets import build_transform

from engine_finetune import train_one_epoch, evaluate, init_csv_writer
import wandb

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

import util.transforms.video_transforms as video_transforms


home_directory = os.getenv('HOME') + '/'

dataset_path = 'OCTCubeM/assets/ext_oph_datasets/glaucoma_processed/'
dataset_name = 'glaucoma'

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    # Slivit parameters
    parser.add_argument('--slivit_fe_path', default=home_directory + 'OCTCubeM/assets/SLIViT/checkpoints/kermany/feature_extractor.pth'
    , type=str, help='feature extractor for SLIViT')
    parser.add_argument('--slivit_num_of_patches', default=60, type=int, help='number of patches for SLIViT')
    # new necessary functional parameters
    parser.add_argument('--variable_joint', default=False, action='store_true', help='use variable joint attention')
    parser.set_defaults(variable_joint=False)

    # New parameters
    parser.add_argument('--load_non_flash_attn_to_flash_attn', default=False, action='store_true', help='use focal loss')
    parser.add_argument('--not_use_2d_aug', default=False, action='store_true', help='not use 2D augmentation')
    parser.add_argument('--not_print_logits', default=False, action='store_true', help='not print logits')
    parser.add_argument('--not_save_figs', default=False, action='store_true', help='not save figures')
    parser.add_argument('--same_3_frames', default=False, action='store_true', help='use the same 3 frames to mock 1 frame for 3D spatio-temporal model')
    parser.add_argument('--return_bal_acc', default=False, action='store_true', help='return balanced accuracy')

    # mae_st parameters
    parser.add_argument("--t_patch_size", default=3, type=int)
    parser.add_argument("--num_frames", default=64, type=int)
    parser.add_argument('--max_frames', default=64, type=int, help='maximum number of frames for each patient')
    parser.add_argument("--sep_pos_embed", action="store_true")
    parser.set_defaults(sep_pos_embed=True)
    parser.add_argument("--cls_embed", action="store_true")
    parser.set_defaults(cls_embed=True)
    parser.add_argument("--transform_type", default="volume_3D", type=str, choices=["frame_2D", "monai_3D", "volume_3D"]) # only glaucoma has volume_3D transform
    parser.add_argument("--color_mode", default="rgb", type=str, choices=["rgb", "gray"])
    parser.add_argument("--smaller_temporal_crop", default='interp', type=str, choices=['interp', 'crop'], help='interpolation type for temporal position embedding')

    # Patient dataset parameters
    # Dataset parameters
    parser.add_argument('--data_path', default=home_directory + dataset_path, type=str, help='dataset path')
    parser.add_argument('--csv_path', default=home_directory + dataset_path, type=str, help='csv path')
    parser.add_argument('--patient_dataset', default='', type=str, help='Use patient dataset')
    parser.add_argument('--patient_dataset_type', default='Center2D', type=str, choices=['3D', 'Center2D', 'Center2D_flash_attn', '2D', '2D_flash_attn',  '3D_flash_attn', '3D_st', '3D_st_joint', '3D_st_flash_attn', '3D_st_joint_flash_attn', '3D_st_flash_attn_nodrop', 'convnext_slivit'], help='patient dataset type')
    parser.add_argument('--dataset_mode', default='volume', type=str, choices=['frame', 'volume'], help='dataset mode for the patient dataset')
    parser.add_argument('--iterate_mode', default='visit', type=str, choices=['visit', 'patient'], help='iterate mode for the patient dataset, glaucome uses visit')
    parser.add_argument('--name_split_char', default='-', type=str, help='split character for the image filename')
    parser.add_argument('--patient_idx_loc', default=1, type=int, help='patient index location in the image filename, e.g., 2 for amd_oct_2_1.png')
    parser.add_argument('--visit_idx_loc', default=None, type=int, help='[optional] visit index location in the image filename')
    parser.add_argument('--cls_unique', default=False, action='store_true', help='use unique class labels for the dataset')

    # Task parameters
    parser.add_argument('--task_mode', default='binary_cls', type=str, choices=['binary_cls','multi_cls'], help='Task mode for the dataset (no multi_label here)')
    parser.add_argument('--val_metric', default='AUPRC', type=str, help='Validation metric for early stopping, newly added BalAcc (only used in AI-READI)')

    parser.add_argument('--save_model', default=False, action='store_true', help='save model')
    parser.add_argument('--enable_early_stop', default=False, action='store_true', help='enable early stop, currently not used in this script')
    parser.add_argument('--early_stop_patience', default=10, type=int, help='early stop patience, currently not used in this script')

    # K_fold cross validation
    parser.add_argument('--k_fold', default=False, action='store_true', help='Use K-fold cross validation')
    parser.add_argument('--k_folds', default=5, type=int, help='number of folds for K-fold cross validation')

    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--val_batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
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

    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./outputs_ft/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
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
    #06/22 add
    parser.add_argument('--testval', action='store_true', default=False,
                        help='Use test set for validation, otherwise use val set')
    
    # Subset sampling parameters
    parser.add_argument('--subset_num', default=0, type=int,
                        help='Create subset with absolute number of samples (old method)')
    parser.add_argument('--new_subset_num', default=0, type=int,
                        help='Create subset with absolute number of samples (new method with separate train/val)')
    parser.add_argument('--droplast', action='store_true', default=False,
                        help='Drop the last incomplete batch, if the dataset size is not divisible by the batch size')
    parser.add_argument('--subsetseed', default=42, type=int)
    parser.add_argument('--bootstrap_runs', action='store_true', default=False, help="Doing bootstrap sampling for training dataset")
    
    #subgroup settings
    parser.add_argument('--subgroup_path', default='', type=str, help='Subgroup for training')
    parser.add_argument('--subgroup_col', default='', type=str, help='Subgroup column for training')
    parser.add_argument('--protect_value', default='', type=str, help='Protected attribute value for training')
    parser.add_argument('--prevalent_value', default='', type=str, help='Prevalent attribute value for training')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    project_name = "OCTCubeM_fairness"
    wandb_task_name = args.task.replace('.','').replace('/', '_') + datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
    group_name = None
    model_add_dir = ""
    
    wandb.init(
        project=project_name,
        name=wandb_task_name,
        group=group_name,
        config=args,
        dir=os.path.join(args.log_dir,wandb_task_name, model_add_dir),
    )

    # Save args to a json file
    if args.output_dir:
        os.makedirs(os.path.join(args.output_dir, model_add_dir), exist_ok=True)
        with open(os.path.join(args.output_dir, model_add_dir, 'args.json'), 'w') as f:
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
    assert args.patient_dataset=='UFcohort'
    #transform
    if args.transform_type == 'volume_3D':
        train_transform = video_transforms.Compose([
            video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        val_transform = train_transform
    elif args.transform_type == 'monai_3D':
        train_transform, val_transform = create_3d_transforms(**vars(args))
    elif args.transform_type == 'frame_2D':
        train_transform = build_transform(is_train='train', args=args)
        val_transform = build_transform(is_train='val', args=args)
        if args.not_use_2d_aug:
            train_transform = build_transform(is_train='val', args=args)
            val_transform = build_transform(is_train='val', args=args)
    #dataset
    if not args.testval:
        tr_istrain = 'train'
        val_istrain = 'val'
    else:
        tr_istrain = ['train', 'val']
        val_istrain = 'test'
    if args.patient_dataset_type == '3D' or args.patient_dataset_type == '3D_st' or args.patient_dataset_type.startswith('3D') or args.patient_dataset_type == 'convnext_slivit':
        #[B, C, T, H, W]
        dataset_train = PatientDataset3D(root_dir=args.data_path, patient_idx_loc=args.patient_idx_loc, transform=None, dataset_mode=args.dataset_mode, name_split_char=args.name_split_char, cls_unique=args.cls_unique, iterate_mode=args.iterate_mode, max_frames=args.max_frames, mode=args.color_mode, transform_type=args.transform_type, volume_resize=args.input_size, same_3_frames=args.same_3_frames, csv_path=args.csv_path, is_train=tr_istrain)
        dataset_val = PatientDataset3D(root_dir=args.data_path, patient_idx_loc=args.patient_idx_loc, transform=None, dataset_mode=args.dataset_mode, name_split_char=args.name_split_char, cls_unique=args.cls_unique, iterate_mode=args.iterate_mode, max_frames=args.max_frames, mode=args.color_mode, transform_type=args.transform_type, volume_resize=args.input_size, same_3_frames=args.same_3_frames, csv_path=args.csv_path, is_train=val_istrain)
        dataset_test = PatientDataset3D(root_dir=args.data_path, patient_idx_loc=args.patient_idx_loc, transform=None, dataset_mode=args.dataset_mode, name_split_char=args.name_split_char, cls_unique=args.cls_unique, iterate_mode=args.iterate_mode, max_frames=args.max_frames, mode=args.color_mode, transform_type=args.transform_type, volume_resize=args.input_size, same_3_frames=args.same_3_frames, csv_path=args.csv_path, is_train='test')
    elif args.patient_dataset_type == 'Center2D' or args.patient_dataset_type == 'Center2D_flash_attn':
        dataset_train = PatientDatasetCenter2D(root_dir=args.data_path, patient_idx_loc=args.patient_idx_loc, transform=None, dataset_mode=args.dataset_mode, name_split_char=args.name_split_char, cls_unique=args.cls_unique, iterate_mode=args.iterate_mode, csv_path=args.csv_path, is_train=tr_istrain)
        dataset_val = PatientDatasetCenter2D(root_dir=args.data_path, patient_idx_loc=args.patient_idx_loc, transform=None, dataset_mode=args.dataset_mode, name_split_char=args.name_split_char, cls_unique=args.cls_unique, iterate_mode=args.iterate_mode, csv_path=args.csv_path, is_train=val_istrain)
        dataset_test = PatientDatasetCenter2D(root_dir=args.data_path, patient_idx_loc=args.patient_idx_loc, transform=None, dataset_mode=args.dataset_mode, name_split_char=args.name_split_char, cls_unique=args.cls_unique, iterate_mode=args.iterate_mode, csv_path=args.csv_path, is_train='test')
    elif args.patient_dataset_type == '2D' or args.patient_dataset_type == '2D_flash_attn':
        dataset_train = PatientDataset2D(root_dir=args.data_path, patient_idx_loc=args.patient_idx_loc, transform=None, dataset_mode=args.dataset_mode, name_split_char=args.name_split_char, cls_unique=args.cls_unique, iterate_mode=args.iterate_mode, csv_path=args.csv_path, is_train=tr_istrain)
        dataset_val = PatientDataset2D(root_dir=args.data_path, patient_idx_loc=args.patient_idx_loc, transform=None, dataset_mode=args.dataset_mode, name_split_char=args.name_split_char, cls_unique=args.cls_unique, iterate_mode=args.iterate_mode, csv_path=args.csv_path, is_train=val_istrain)
        dataset_test = PatientDataset2D(root_dir=args.data_path, patient_idx_loc=args.patient_idx_loc, transform=None, dataset_mode=args.dataset_mode, name_split_char=args.name_split_char, cls_unique=args.cls_unique, iterate_mode=args.iterate_mode, csv_path=args.csv_path, is_train='test')

    dataset_train.update_transform(train_transform)
    dataset_val.update_transform(val_transform)
    dataset_test.update_transform(val_transform)
    
    # Apply subset sampling by absolute number if subset_num > 0
    args.droplast = True  # Default to True
    if hasattr(args, 'subset_num') and args.subset_num > 0:
        print(f'Old subset method for absolute number {args.subset_num}')
        
        def create_subset_by_num(dataset, split_name, subset_num):
            """Create a subset of the dataset with specified absolute number"""
            targets = np.array(dataset.targets)
            unique_classes, class_counts = np.unique(targets, return_counts=True)
            n_classes = len(unique_classes)
            
            print(f'{split_name} - Original size: {len(dataset)}, Classes: {n_classes}, Target subset size: {subset_num}')
            
            if subset_num < n_classes:
                # Too small to guarantee at least one per class → fall back to plain random sample
                print(f'Warning: subset_num ({subset_num}) < number of classes ({n_classes}), using random sampling')
                rng = np.random.RandomState(args.subsetseed)
                subset_indices = rng.choice(len(dataset), min(subset_num, len(dataset)), replace=False)
            else:
                # Use stratified sampling to maintain class distribution
                if subset_num >= len(dataset):
                    print(f'Warning: subset_num ({subset_num}) >= dataset size ({len(dataset)}), using full dataset')
                    subset_indices = list(range(len(dataset)))
                else:
                    sss = StratifiedShuffleSplit(n_splits=1, train_size=subset_num, random_state=args.subsetseed)
                    subset_indices = next(sss.split(range(len(dataset)), targets))[0]
            
            # Create subset dataset
            subset_dataset = Subset(dataset, subset_indices)

            # Add targets attribute to subset for compatibility
            subset_dataset.targets = [dataset.targets[i] for i in subset_indices]
            subset_dataset.classes = dataset.classes
            subset_dataset.class_to_idx = dataset.class_to_idx
            
            print(f'{split_name} - Final subset size: {len(subset_dataset)}')
            return subset_dataset

        dataset_train = create_subset_by_num(dataset_train, 'Train', int(args.subset_num))
        args.droplast = False  # Disable droplast when using old subset method

    # Apply subset sampling by absolute number if new_subset_num > 0
    if hasattr(args, 'new_subset_num') and args.new_subset_num > 0:
        print(f'New subset method for absolute number {args.new_subset_num}')
        def create_separate_class_based_subsets(train_dataset, val_dataset, total_subset_num):
            """Create separate subsets from train and validation datasets based on class ratios"""
            
            def create_class_balanced_subset(dataset, split_name, target_size):
                """Create a class-balanced subset from a single dataset"""
                targets = np.array(dataset.targets)
                unique_classes, class_counts = np.unique(targets, return_counts=True)
                n_classes = len(unique_classes)
                
                # Calculate class ratios within this dataset
                class_ratios = class_counts / len(targets)
                
                print(f'\n{split_name} dataset - Original size: {len(dataset)}, Classes: {n_classes}')
                print(f'{split_name} class counts: {dict(zip(unique_classes, class_counts))}')
                print(f'{split_name} class ratios: {dict(zip(unique_classes, class_ratios))}')
                print(f'{split_name} target subset size: {target_size}')
                
                # Separate samples by class and permute
                rng = np.random.RandomState(args.subsetseed)
                selected_indices = []
                
                for class_idx in unique_classes:
                    # Get all samples for this class
                    class_mask = targets == class_idx
                    class_samples = np.where(class_mask)[0]
                    
                    # Permute samples within this class
                    class_samples_copy = class_samples.copy()
                    rng.shuffle(class_samples_copy)
                    
                    # Calculate how many samples to select for this class
                    class_target_samples = int((target_size-n_classes) * class_ratios[class_idx]) + 1
                    
                    # Ensure we don't exceed available samples
                    available_samples = len(class_samples_copy)
                    if class_target_samples > available_samples:
                        print(f'Warning: {split_name} Class {class_idx} needs {class_target_samples} samples but only {available_samples} available')
                        class_target_samples = available_samples
                    
                    # Select samples for this class
                    selected_class_samples = class_samples_copy[:class_target_samples]
                    selected_indices.extend(selected_class_samples)
                    
                    print(f'{split_name} Class {class_idx}: ratio={class_ratios[class_idx]:.3f}, target={class_target_samples}, selected={len(selected_class_samples)}')
                
                print('Selected indices:', selected_indices)
                subset_dataset = Subset(dataset, selected_indices)
                
                # Add targets attribute to subset for compatibility
                
                
                print(f'{split_name} final subset size: {len(subset_dataset)}')
                return subset_dataset
            
            # Calculate target sizes for train and validation (80/20 split)
            train_target_size = int(total_subset_num * 0.8)
            val_target_size = int(total_subset_num * 0.2)
            
            print(f'Total target subset size: {total_subset_num}')
            print(f'Train target size: {train_target_size} (80%)')
            print(f'Validation target size: {val_target_size} (20%)')
            
            # Create subsets separately
            train_subset = create_class_balanced_subset(train_dataset, 'Train', train_target_size)
            val_subset = create_class_balanced_subset(val_dataset, 'Validation', val_target_size)
            
            return train_subset, val_subset

        dataset_train, dataset_val = create_separate_class_based_subsets(dataset_train, dataset_val, int(args.new_subset_num))
        args.droplast = False  # Disable droplast when using old subset method
    
    assert args.k_fold is False
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    val_mode = 'val'
    test_mode = 'test'
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
        os.makedirs(os.path.join(args.log_dir,args.task), exist_ok=True)
        log_writer = SummaryWriter(log_dir=os.path.join(args.log_dir,args.task))
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

    if args.patient_dataset_type == '3D':
        model = models_vit_3dhead.__dict__[args.model](
            img_size=args.input_size,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )
    elif args.patient_dataset_type == 'Center2D' or args.patient_dataset_type == '2D':
        model = models_vit.__dict__[args.model](
            img_size=args.input_size,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )
    elif args.patient_dataset_type == 'Center2D_flash_attn' or args.patient_dataset_type == '2D_flash_attn':
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
    else:
        model = models_vit.__dict__[args.model](
            img_size=args.input_size,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )

    if args.finetune and not args.eval and args.patient_dataset_type != 'convnext_slivit':
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        print("checkpoint keys: ", list(checkpoint.keys()))
        if 'model' in list(checkpoint.keys()):
            checkpoint_model = checkpoint['model']
        #TMP FOR mm_octcube_ir.pt
        elif 'state_dict' in list(checkpoint.keys()):
            checkpoint_model = checkpoint['state_dict']
            checkpoint_model = {k.replace('module.', '', 1): v for k, v in checkpoint_model.items()}
            checkpoint_model = {k.replace('text.', '', 1): v for k, v in checkpoint_model.items() if k.startswith('text.')}
            for drop_k in ['head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias']:
                checkpoint_model.pop(drop_k, None)
        else:
            checkpoint_model = checkpoint
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
            elif args.patient_dataset_type == 'Center2D' or args.patient_dataset_type == 'Center2D_flash_attn' or args.patient_dataset_type == '2D' or args.patient_dataset_type == '2D_flash_attn':
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

    # build optimizer with layer-wise lr decay (lrd)
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

    #if resume training
    if args.resume:
        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, model_add_dir=model_add_dir)

    if args.eval:
        test_stats, auc_roc, auc_pr = evaluate(data_loader_test, model, device, os.path.join(args.log_dir,args.task), epoch=0, mode=test_mode, num_class=args.nb_classes, criterion=criterion, task_mode=args.task_mode, disease_list=None, return_bal_acc=args.return_bal_acc, args=args)
        if args.return_bal_acc:
            test_auc_pr, test_bal_acc = auc_pr
        wandb_dict={f'test_{k}': v for k, v in test_stats.items()}
        wandb.log(wandb_dict)
        wandb.finish()
        exit(0)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
