# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This script provides a simple MAE3D implementation with incorporating both 3D and 2D OCT datasets into pre-training. The self-paced learning (SPL) strategy is also implemented for the 2D dataset. The 2d dataset is Kermany dataset. The 3d dataset is in-house dataset. The 2d dataset is used for pre-training the model, while the 3d dataset is used for fine-tuning the model.

# One interesting setting is that the 2D dataset can have a different higher resolution compared to the 3D dataset. The 2D dataset is 512x512, while the 3D dataset is 256x256. The 2D dataset is used for pre-training, while the 3D dataset is used for fine-tuning. Despite we found add the 2D dataset will not improve the overall performance, it is still interesting to see how the model can be trained with different resolutions.

# Currently, the pre-training for 2D-only dataset is not yet supported. But we 'd love to support it in the future if the community is interested in it.

import argparse
import datetime
import json
import os
import time
import pickle as pkl

import custom_util.misc as misc

import numpy as np
import timm
import torch
import torch.backends.cudnn as cudnn
from iopath.common.file_io import g_pathmgr as pathmgr

import models_mae_joint_res_flash_attn as models_mae
from engine_pretrain import train_one_epoch_joint, eval_one_epoch

from custom_util.misc import NativeScalerWithGradNormCount as NativeScaler
from custom_util.misc import convert_spatial_pos_embed
from custom_util.PatientDataset import TransformableSubset
from custom_util.PatientDataset_inhouse import PatientDatasetCenter2D_inhouse, PatientDataset3D_inhouse, create_3d_transforms, load_patient_list
from custom_util.PatientDataset_pretrain import PatientDatasetCenter2D_inhouse_pretrain, Inhouse_and_Kermany_Dataset
from tensorboard.compat.tensorflow_stub.io.gfile import register_filesystem
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as transforms
import torchvision.datasets as datasets


home_directory = os.getenv('HOME') + '/'
split_path = 'Oph_cls_task/scr_train_val_test_split_622/'

def K_scheduler(epoch, K_max=0.7, K_min=0.3, all_epoch=100, warmup_epochs=10, epoch_offset=0):
    num_epochs = epoch - epoch_offset
    # start from K_max and decrease to K_min
    if num_epochs <= warmup_epochs:
        return K_max
    else:
        return K_max - (num_epochs - warmup_epochs) * (K_max - K_min) / (all_epoch - warmup_epochs - epoch_offset)

def mask_ratio_2d_scheduler(epoch, mask_ratio_max=0.85, mask_ratio_min=0.75, all_epoch=100, warmup_epochs=10, epoch_offset=0):
    num_epochs = epoch - epoch_offset
    # start from mask_ratio_min and increase to mask_ratio_max
    if num_epochs <= warmup_epochs:
        return mask_ratio_min
    else:
        return mask_ratio_min + (num_epochs - warmup_epochs) * (mask_ratio_max - mask_ratio_min) / (all_epoch - warmup_epochs - epoch_offset)


def parse_input_size(value):
    # Attempt to evaluate the input if it's given as tuple
    try:
        # Convert input to a tuple if it's in form "x,y"
        if ',' in value:
            size = tuple(map(int, value.split(',')))
            if len(size) == 2:
                return size
            else:
                raise argparse.ArgumentTypeError("Tuple must have exactly two integers.")
        # Otherwise, handle it as an integer
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid input size value: {value}")


def get_args_parser():
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    parser.add_argument('--kermany_data_dir', default=home_directory + '/ext_oph_datasets/Kermany/CellData/OCT/', type=str,)
    parser.add_argument('--split_path', default=home_directory + split_path, type=str, help='split path storing the train/val/test split of patient files')
    parser.add_argument('--data_path', default=home_directory + '/Ophthal/', type=str, help='dataset path')
    parser.add_argument('--patient_id_list_dir', default='multi_label_expr_all_0319/', type=str, help='patient id list dir')
    parser.add_argument('--metadata_dir', default='Oph_cls_task/', type=str, help='metadata dir')
    parser.add_argument('--eval_only', action='store_true', help='perform evaluation only')
    parser.add_argument('--eval_only_epoch', default=0, type=int, help='perform evaluation only epoch')
    parser.add_argument('--resume_type', default='retfound', type=str, choices=['training_latest', 'training_new', 'retfound', 'training_continue_reset_optim', 'retfound_2_flash_attn', 'imagenet_2_flash_attn', 'imagenet_ft_2_flash_attn'] , help='resume type')
    parser.add_argument("--batch_size_2d",default=16, type=int, help="2d Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",)
    parser.add_argument("--mask_ratio_2d_min", default=0.75, type=float, help="Masking ratio (percentage of removed patches).",)
    parser.add_argument("--mask_ratio_2d_max", default=0.85, type=float, help="Masking ratio (percentage of removed patches).",)
    parser.add_argument("--epoch_offset", default=0, type=int, help="epoch offset",)
    parser.add_argument("--K_min", default=0.3, type=float, help="number of classes")
    parser.add_argument("--K_max", default=0.9, type=float, help="number of classes")
    parser.add_argument("--high_res_input_size", default=512, type=int, help="images input size")
    parser.add_argument("--epoch_load_spl", default=-1, type=int, help="epoch offset",)
    parser.add_argument("--load_spl_dir", default='', type=str, help="load spl dir",)
    parser.add_argument("--init_ckpt", default="", help="Initialize from non-flash-attn checkpoint")

    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="mae_vit_large_patch16",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )

    parser.add_argument("--input_size", default=256, type=parse_input_size, help="images input size")
    parser.add_argument("--in_chans", default=1, type=int, help="images input size")

    parser.add_argument(
        "--mask_ratio",
        default=0.75,
        type=float,
        help="Masking ratio (percentage of removed patches).",
    )

    parser.add_argument(
        "--norm_pix_loss",
        default=False,
        action="store_true",
        help="Use (per-patch) normalized pixels as targets for computing loss",
    )
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=40, metavar="N", help="epochs to warmup LR"
    )
    parser.add_argument(
        '--train_data',
        default='/datasets01/imagenet_full_size/061417/',
        type=str,
        help='training dataset path'
    )
    parser.add_argument(
        '--val_data',
        default='/datasets01/imagenet_full_size/061417/',
        type=str,
        help='evaluate dataset path'
    )
    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--log_dir",
        default="",
        help="path where to tensorboard log",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--no_env", action="store_true")

    # Video related configs
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    parser.add_argument("--decoder_embed_dim", default=512, type=int)
    parser.add_argument("--decoder_depth", default=8, type=int)
    parser.add_argument("--decoder_num_heads", default=16, type=int)
    parser.add_argument("--t_patch_size", default=3, type=int)
    parser.add_argument("--num_frames", default=60, type=int)
    parser.add_argument("--checkpoint_period", default=1, type=int)
    parser.add_argument("--sampling_rate", default=4, type=int)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--repeat_aug", default=4, type=int)
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
    )
    parser.add_argument("--no_qkv_bias", action="store_true")
    parser.add_argument("--bias_wd", action="store_true")
    parser.add_argument("--num_checkpoint_del", default=20, type=int)
    parser.add_argument("--sep_pos_embed", action="store_true")
    parser.set_defaults(sep_pos_embed=True)
    parser.add_argument(
        "--trunc_init",
        action="store_true",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
    )
    parser.set_defaults(fp32=False)
    parser.add_argument(
        "--fp16",
        action="store_true",
    )
    parser.set_defaults(fp16=True)
    parser.add_argument(
        "--jitter_scales_relative",
        default=[0.5, 1.0],
        type=float,
        nargs="+",
    )
    parser.add_argument(
        "--jitter_aspect_relative",
        default=[0.75, 1.3333],
        type=float,
        nargs="+",
    )
    parser.add_argument(
        "--beta",
        default=None,
        type=float,
        nargs="+",
    )
    parser.add_argument(
        "--pred_t_dim",
        type=int,
        default=60,
    )
    parser.add_argument("--cls_embed", action="store_true")
    parser.set_defaults(cls_embed=True)
    return parser


def main(args):
    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    if args.output_dir and misc.is_main_process():
        with open(os.path.join(args.output_dir, "args.txt"), "a") as f:
            f.write("{}".format(args).replace(', ', ',\n'))


    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # 2d_dataset_transform, we use Kermany as an example
    transform_2d_train = transforms.Compose([
            transforms.Resize((args.high_res_input_size, args.high_res_input_size), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    dataset_train_2d = PatientDatasetCenter2D_inhouse_pretrain(root_dir=args.data_path, task_mode='multi_label', disease='AMD', disease_name_list=None, metadata_fname=None, dataset_mode='frame', transform=transform_2d_train, iterate_mode='visit', downsample_width=True, patient_id_list_dir=args.patient_id_list_dir, enable_spl=True, mask_transform=transform_2d_train, return_mask=False, metadata_dir=args.metadata_dir)

    test_pat_id = load_patient_list(args.split_path, split='test', name_suffix='_pat_list.txt')
    included_patient = list(dataset_train_2d.patients.keys())
    filtered_test_pat_id = sorted(list(set(test_pat_id) & set(included_patient)))

    dataset_train_2d.all_image_list = dataset_train_2d.get_all_image_list(filtered_test_pat_id)

    dataset_train_2d.update_len_dataset_list()
    dataset_train_2d.init_spl(K=0.2)
    dataset_train_2d_kermany = datasets.ImageFolder(os.path.join(args.kermany_data_dir, 'train'), transform=transform_2d_train)
    dataset_train_2d_all = Inhouse_and_Kermany_Dataset(dataset_train_2d, dataset_train_2d_kermany)


    # 3d dataset
    transform_train, transform_eval = create_3d_transforms(**vars(args))

    dataset = PatientDataset3D_inhouse(root_dir=args.data_path, task_mode='multi_label', disease='AMD', disease_name_list=None, metadata_fname=None, dataset_mode='frame', mode='gray', transform=None, iterate_mode='visit', downsample_width=True, patient_id_list_dir=args.patient_id_list_dir, pad_to_num_frames=True, padding_num_frames=args.num_frames, transform_type='monai_3D', return_img_w_patient_and_visit_name=True, return_data_dict=True, metadata_dir=args.metadata_dir)

    train_pat_id = load_patient_list(args.split_path, split='train', name_suffix='_pat_list.txt')
    val_pat_id = load_patient_list(args.split_path, split='val', name_suffix='_pat_list.txt')
    test_pat_id = load_patient_list(args.split_path, split='test', name_suffix='_pat_list.txt')
    included_patient = list(dataset.patients.keys())

    filtered_train_pat_id = sorted(list(set(train_pat_id) & set(included_patient)))
    filtered_val_pat_id = sorted(list(set(val_pat_id) & set(included_patient)))
    filtered_test_pat_id = sorted(list(set(test_pat_id) & set(included_patient)))

    train_pat_indices = dataset.get_visit_idx(filtered_train_pat_id)
    val_pat_indices = dataset.get_visit_idx(filtered_val_pat_id)
    test_pat_indices = dataset.get_visit_idx(filtered_test_pat_id)

    final_train_pat_indices = train_pat_indices + val_pat_indices
    final_val_pat_indices = test_pat_indices
    dataset_train = TransformableSubset(dataset, final_train_pat_indices)
    dataset_val = TransformableSubset(dataset, final_val_pat_indices)
    dataset_train.update_dataset_transform(transform_train)

    all_image_list = dataset_train_2d.all_image_list
    all_image_dict = dataset_train_2d.all_image_dict
    print('Number of train 2d images:', len(all_image_list), len(all_image_dict))

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        # newly added: for 2d dataset
        print("Sampler_train = %s" % str(sampler_train))
        sampler_train_2d = torch.utils.data.DistributedSampler(
            dataset_train_2d_all, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        data_loader_train_2d = torch.utils.data.DataLoader(
            dataset_train_2d_all, sampler=sampler_train_2d,
            batch_size=args.batch_size_2d,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        print("Data_loader_train_2d = %s" % len(data_loader_train_2d), len(data_loader_train_2d) * args.batch_size_2d)

    else:
        num_tasks = 1
        global_rank = 0
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        try:
            pathmgr.mkdirs(args.log_dir)
        except Exception as _:
            pass
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    # newly added for validation
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # define the model
    model = models_mae.__dict__[args.model](
        **vars(args),
    )

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
        )
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.add_weight_decay(
        model_without_ddp,
        args.weight_decay,
        bias_wd=args.bias_wd,
    )
    if args.beta is None:
        beta = (0.9, 0.95)
    else:
        beta = args.beta
    optimizer = torch.optim._multi_tensor.AdamW(
        param_groups,
        lr=args.lr,
        betas=beta,
    )
    loss_scaler = NativeScaler(fp32=args.fp32)
    if args.resume or args.init_ckpt or args.resume_type == 'imagenet_2_flash_attn' or args.resume_type == 'imagenet_ft_2_flash_attn':
        print("Resuming from checkpoint")
        if args.resume_type == 'training_latest':
            print('training latest')
            misc.load_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                convert_pos_embed=True,
            )
            print(args.start_epoch)
            if os.path.exists(args.output_dir + f'/all_image_dict-{args.start_epoch:02d}.pkl'):
                with pathmgr.open(args.output_dir + f'/all_image_dict-{args.start_epoch:02d}.pkl', "rb") as f:
                    all_image_dict = pkl.load(f)
                dataset_train_2d.all_image_dict = all_image_dict
                K_for_spl = K_scheduler(args.start_epoch, K_max=args.K_max, K_min=args.K_min, all_epoch=args.epochs, warmup_epochs=args.warmup_epochs, epoch_offset=args.epoch_offset)
                # update spl for dataset_train_2d
                dataset_train_2d.update_spl(K=K_for_spl)

                # newly added: for 2d dataset
                print("Update: Sampler_train = %s" % str(sampler_train))
                sampler_train_2d = torch.utils.data.DistributedSampler(
                    dataset_train_2d_all, num_replicas=num_tasks, rank=global_rank, shuffle=True
                )
                data_loader_train_2d = torch.utils.data.DataLoader(
                    dataset_train_2d_all, sampler=sampler_train_2d,
                    batch_size=args.batch_size_2d,
                    num_workers=args.num_workers,
                    pin_memory=args.pin_mem,
                    drop_last=True,
                )
            print('len of dataset_train_2d:', len(dataset_train_2d), len(dataset_train_2d_all))
            print('len of data_loader_train_2d:', len(data_loader_train_2d))

        elif args.resume_type == 'training_new':
            print('training new here')
            misc.load_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                not_load_optim=True,
                convert_pos_embed=True,
                high_res_model=True,
            )
        elif args.resume_type == 'retfound':
            misc.load_model_retfound(
                args=args,
                model_without_ddp=model_without_ddp,
            )

        elif args.resume_type == 'retfound_2_flash_attn':
            misc.load_model_retfound_flash_attn(
                args=args,
                model_without_ddp=model_without_ddp,
                encoder_only=True,
            )

        elif args.resume_type == 'imagenet_2_flash_attn':
            imagenet_model = timm.create_model('vit_large_patch16_224.mae', pretrained=True)
            misc.load_model_retfound_flash_attn(
                args=args,
                model_without_ddp=model_without_ddp,
                encoder_only=True,
                preload_model=imagenet_model,
            )

        elif args.resume_type == 'imagenet_ft_2_flash_attn':
            imagenet_model = timm.create_model('vit_large_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
            misc.load_model_retfound_flash_attn(
                args=args,
                model_without_ddp=model_without_ddp,
                encoder_only=True,
                preload_model=imagenet_model,
            )


        elif args.resume_type == 'training_continue_reset_optim':
            misc.load_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                not_load_optim=True,
                convert_pos_embed=True,
                high_res_model=True,
            )
            print('epoch load spl:', args.epoch_load_spl)

            if args.epoch_load_spl >= 0 and os.path.exists(args.load_spl_dir + f'/all_image_dict-{args.epoch_load_spl:02d}.pkl'):
                with pathmgr.open(args.load_spl_dir + f'/all_image_dict-{args.epoch_load_spl:02d}.pkl', "rb") as f:
                    all_image_dict = pkl.load(f)
                dataset_train_2d.all_image_dict = all_image_dict
                K_for_spl = K_scheduler(args.start_epoch, K_max=args.K_max, K_min=args.K_min, all_epoch=args.epochs, warmup_epochs=args.warmup_epochs, epoch_offset=args.epoch_offset)
                # update spl for dataset_train_2d
                dataset_train_2d.update_spl(K=K_for_spl)

                # Set up the dataloader for 2d dataset
                print("Update: Sampler_train = %s" % str(sampler_train))
                sampler_train_2d = torch.utils.data.DistributedSampler(
                    dataset_train_2d_all, num_replicas=num_tasks, rank=global_rank, shuffle=True
                )
                data_loader_train_2d = torch.utils.data.DataLoader(
                    dataset_train_2d_all, sampler=sampler_train_2d,
                    batch_size=args.batch_size_2d,
                    num_workers=args.num_workers,
                    pin_memory=args.pin_mem,
                    drop_last=True,
                )
                print('continue training update spl: len of dataset_train_2d:', len(dataset_train_2d), len(dataset_train_2d_all))
                print('continue training update spl: len of data_loader_train_2d:', len(data_loader_train_2d))

        else:
            raise ValueError("Invalid resume type")

        if args.eval_only:
            # test the evaluation function
            val_stats = eval_one_epoch(
                    model,
                    data_loader_val,
                    device,
                    args.eval_only_epoch if args.eval_only_epoch > 0 else 0,
                    log_writer=log_writer,
                    args=args,
                    fp32=args.fp32,
                    fp16=args.fp16,
                    joint=True,
                    visible_frame_freq=5,
                    data_loader_2d=data_loader_train_2d,
                    mask_ratio_2d=0.75,
                )

            print('Validation stats:', val_stats)
            print('End of evaluation, exiting...')
            exit()

    checkpoint_path = ""
    print(f"Start training for {args.epochs} epochs, currently at epoch {args.start_epoch}")
    start_time = time.time()


    for epoch in range(args.start_epoch, args.epochs):
        K_for_spl = K_scheduler(epoch, K_max=args.K_max, K_min=args.K_min, all_epoch=args.epochs, warmup_epochs=args.warmup_epochs, epoch_offset=args.epoch_offset)
        mask_ratio_2d = mask_ratio_2d_scheduler(epoch, mask_ratio_max=args.mask_ratio_2d_max, mask_ratio_min=args.mask_ratio_2d_min, all_epoch=args.epochs, warmup_epochs=args.warmup_epochs, epoch_offset=args.epoch_offset)
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch_joint(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
            fp32=args.fp32,
            fp16=args.fp16,
            data_loader_2d=data_loader_train_2d,
            dataset_2d_all_image_dict=all_image_dict,
            mask_ratio_2d=mask_ratio_2d,
        )

        dataset_train.remove_dataset_transform()
        dataset_val.update_dataset_transform(transform_eval)

        # run the evaluation
        val_stats = eval_one_epoch(
            model,
            data_loader_val,
            device,
            epoch,
            log_writer=log_writer,
            args=args,
            fp32=args.fp32,
            fp16=args.fp16,
            joint=True,
            visible_frame_freq=20,
            data_loader_2d=data_loader_train_2d,
            mask_ratio_2d=0.75,
        )

        dataset_val.remove_dataset_transform()
        dataset_train.update_dataset_transform(transform_train)

        if args.output_dir and (
            epoch % args.checkpoint_period == 0 or epoch + 1 == args.epochs
        ):
            checkpoint_path = misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
            "epoch": epoch,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with pathmgr.open(
                f"{args.output_dir}/log.txt",
                "a",
            ) as f:
                f.write(json.dumps(log_stats) + "\n")
            filename = f"{args.output_dir}/all_image_dict-{epoch+1:02d}.pkl"
            with pathmgr.open(filename, "wb") as f:
                pkl.dump(all_image_dict, f)

        # update spl for dataset_train_2d
        dataset_train_2d.update_spl(K=K_for_spl)

        # reset dataloader for 2d dataset
        print("Update: Sampler_train = %s" % str(sampler_train))
        sampler_train_2d = torch.utils.data.DistributedSampler(
            dataset_train_2d_all, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        data_loader_train_2d = torch.utils.data.DataLoader(
            dataset_train_2d_all, sampler=sampler_train_2d,
            batch_size=args.batch_size_2d,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        print("Update: Data_loader_train_2d = %s" % len(data_loader_train_2d))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    print(torch.cuda.memory_allocated())
    return [checkpoint_path]


def launch_one_thread(
    local_rank,
    shard_rank,
    num_gpus_per_node,
    num_shards,
    init_method,
    output_path,
    opts,
    stats_queue,
):
    print(opts)
    args = get_args_parser()
    args = args.parse_args(opts)
    args.rank = shard_rank * num_gpus_per_node + local_rank
    args.world_size = num_shards * num_gpus_per_node
    args.gpu = local_rank
    args.dist_url = init_method
    args.output_dir = output_path
    output = main(args)
    stats_queue.put(output)
