# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# --------------------------------------------------------

# Public fundus-photo benchmark fine-tuning / evaluation of OCTCube-IR's 2D
# en-face ("text") tower, on the same 7-dataset OCTRFF benchmark MIRAGE's
# run_cls_tuning_fundus.py / run_fundus_all_tasks_l4.sh use (pre-split
# train/val/test/Class_x/ image folders, loaded with torchvision's
# ImageFolder; num_classes auto-inferred from the train/ folder, same as
# MIRAGE's process_args()):
#   Glaucoma_fundus:3 IDRiD_data:5 JSIEC:39 MESSIDOR2:5 PAPILA:3 Retina:4 APTOS2019:5
#
# OCTCube-IR has no dedicated color-fundus input domain -- its 2D en-face
# tower (models_vit_flash_attn.flash_attn_vit_large_patch16, saved under a
# `text.` prefix inside the jointly-pretrained mm_octcube_ir.pt checkpoint --
# see main_finetune_downstream_UFcohort_OCTCubeIR.py for the dual-tower
# checkpoint format) was trained on grayscale-source en-face IR images. This
# is the same cross-domain-transfer setup MIRAGE itself uses for this exact
# benchmark (its own comment: "MIRAGE has no dedicated 'fundus' input domain
# ... routed through 'slo'"); the tower's input adapter is 3-channel by
# architecture regardless, so RGB fundus photos load directly.
#
# Sibling to main_finetune_downstream_public2D_OCTCubeIR.py (same
# Center2D_flash_attn model + the same OCTCube-IR checkpoint loader), which
# instead targets this repo's own patient-ID-parsed OCT-B-scan public
# datasets (Duke14/OIMHS/UMN/GLAUCOMA) via k-fold CV. This script has no
# k-fold: these 7 datasets ship a fixed train/val/test split.

import os
import sys
import time
import json
import hashlib
import argparse
import datetime
import numpy as np

from pathlib import Path

from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import torch
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
import wandb

import util.misc as misc
import util.lr_decay as lrd
from util.datasets import build_transform
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.pos_embed import interpolate_pos_embed

from engine_finetune import train_one_epoch, evaluate, init_csv_writer

import models_vit_flash_attn

# Same checkpoint-splitting logic as the k-fold sibling script; only the en-face
# ("text.") tower of OCTCube-IR's joint checkpoint is kept.
from main_finetune_downstream_public2D_OCTCubeIR import load_octcubeir_2d_tower_checkpoint


# All 7 datasets are pre-split train/val/test/Class_x/ image folders directly
# under {data_root}/{data_set}/ (per OphFoundation's reference benchmark
# layout); num_classes is auto-inferred from the train/ folder, not listed
# here. All have >=3 classes, so task_mode is always multi_cls.
FUNDUS_DATASETS = ['Glaucoma_fundus', 'IDRiD_data', 'JSIEC', 'MESSIDOR2', 'PAPILA', 'Retina', 'APTOS2019']


def get_args_parser():
    parser = argparse.ArgumentParser('OCTCube-IR 2D en-face tower fine-tuning on the public fundus benchmark', add_help=False)

    required_parser = parser.add_argument_group('required arguments')
    required_parser.add_argument(
        '--data_set', type=str, required=True, choices=FUNDUS_DATASETS,
        help='Which public fundus dataset to fine-tune/evaluate on (required).')
    required_parser.add_argument(
        '--data_root', type=str, required=True,
        help='Root directory containing the per-dataset train/val/test/Class_x/ '
             'folders (e.g. .../OCTRFF_Data/benchmark/). (required)')
    required_parser.add_argument(
        '--finetune', type=str, required=True,
        help="Path to OCTCube-IR's joint checkpoint (ckpt/mm_octcube_ir.pt); only its "
             "en-face ('text.') tower is loaded. (required)")

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--val_metric', default='AUPRC', type=str, choices=['AUC', 'ACC', 'AUPRC', 'BalAcc'],
                        help='Validation metric used to pick the best epoch. (default: %(default)s)')
    parser.add_argument('--return_bal_acc', default=False, action='store_true')
    parser.add_argument('--not_print_logits', default=False, action='store_true')
    parser.add_argument('--not_save_figs', default=False, action='store_true')
    parser.add_argument('--variable_joint', default=False, action='store_true', help=argparse.SUPPRESS)
    parser.set_defaults(variable_joint=False)
    parser.add_argument('--save_model', default=False, action='store_true')

    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--val_batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--model', default='flash_attn_vit_large_patch16', type=str, metavar='MODEL')
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--drop_path', type=float, default=0.2, metavar='PCT')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM')
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None, metavar='LR')
    parser.add_argument('--blr', type=float, default=5e-3, metavar='LR')
    parser.add_argument('--layer_decay', type=float, default=0.65)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N')

    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME')
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT')
    parser.add_argument('--remode', type=str, default='pixel')
    parser.add_argument('--recount', type=int, default=1)
    parser.add_argument('--resplit', action='store_true', default=False)
    parser.add_argument('--mixup', type=float, default=0)
    parser.add_argument('--cutmix', type=float, default=0)
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None)
    parser.add_argument('--mixup_prob', type=float, default=1.0)
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5)
    parser.add_argument('--mixup_mode', type=str, default='batch')

    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool')

    parser.add_argument('--output_dir', default='./outputs_ft_public2D_octcubeir_fundus/',
                        help='Base output directory (a --data_set/args-checksum subdir is appended)')
    parser.add_argument('--log_dir', default='./output_dir')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume a previously fine-tuned checkpoint (also used for --eval)')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--eval', action='store_true', help='Evaluate the test split only, then exit (see --resume)')
    parser.add_argument('--dist_eval', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-run even if results.txt already exists in the resolved output dir')

    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    parser.add_argument('--use_wandb', default=True, action='store_true')
    parser.add_argument('--no_wandb', dest='use_wandb', action='store_false')
    parser.add_argument('--wandb_project', default='OCTCubeM', type=str)
    parser.add_argument('--wandb_entity', default=None, type=str)
    parser.add_argument('--wandb_run_name', default=None, type=str)

    return parser


def process_args(args):
    """Resolve --data_root/--data_set into a concrete --data_path and
    auto-infer --nb_classes from the train/ folder's subdirectories --
    mirroring MIRAGE's process_args()/run_cls_tuning_fundus.py exactly."""
    if args.data_root[-1] != '/':
        args.data_root += '/'
    args.data_path = args.data_root + args.data_set

    train_data_path = os.path.join(args.data_path, 'train')
    num_classes = sum(1 for d in Path(train_data_path).iterdir() if d.is_dir())
    num_samples = sum(len(list(d.iterdir())) for d in Path(train_data_path).iterdir() if d.is_dir())
    args.nb_classes = num_classes
    print(f'Number of classes: {num_classes}')
    print(f'Number of training samples: {num_samples}')

    # All 7 datasets here have >=3 classes.
    args.task_mode = 'multi_cls'
    # fixed for this script -- see engine_finetune.py, which branches on this string
    args.patient_dataset_type = 'Center2D_flash_attn'
    return args


def get_output_dir(args):
    """Base output dir + --data_set + a checksum of the hyperparameters that
    change run behavior -- same convention as the k-fold sibling script /
    MIRAGE's args-checksum output dirs."""
    checksum_keys = [
        'data_set', 'model', 'blr', 'layer_decay', 'weight_decay', 'drop_path',
        'epochs', 'warmup_epochs', 'batch_size', 'seed', 'input_size',
    ]
    args_vars = {k: getattr(args, k) for k in checksum_keys}
    args_str = json.dumps(args_vars, indent=2, sort_keys=True)
    args_checksum = hashlib.md5(args_str.encode('utf-8')).hexdigest()[:8]

    output_dir = args.output_dir
    if output_dir[-1] != '/':
        output_dir += '/'
    output_dir += f'{args.data_set}/{args.model}_finetune_w_{args_checksum}/'
    return output_dir


def build_fundus_dataset(subset, args, transform):
    root = os.path.join(args.data_path, subset)
    return datasets.ImageFolder(root, transform=transform)


def main(args):
    misc.init_distributed_mode(args)
    args = process_args(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    args.output_dir = get_output_dir(args)
    args.task = os.path.join(args.output_dir, '')  # trailing slash: engine_finetune concatenates args.task + filename

    if os.path.exists(os.path.join(args.output_dir, 'results.txt')) and not args.overwrite:
        print(f'Experiment already run at {args.output_dir}. Exiting (pass --overwrite to re-run).')
        sys.exit(0)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    val_bal_acc = None
    test_bal_acc = None

    cudnn.benchmark = True

    train_transform = build_transform(is_train='train', args=args)
    val_transform = build_transform(is_train='val', args=args)

    dataset_train = build_fundus_dataset('train', args, train_transform)
    dataset_val = build_fundus_dataset('val', args, val_transform)
    dataset_test = build_fundus_dataset('test', args, val_transform)
    print(f"Train/val/test sizes: {len(dataset_train)}/{len(dataset_val)}/{len(dataset_test)}")

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if args.use_wandb and global_rank == 0:
        wandb_task_name = args.wandb_run_name if args.wandb_run_name else (
            f'public2D-OCTCubeIR-fundus-{args.data_set}' + datetime.datetime.now().strftime("_%Y%m%d_%H%M%S"))
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=wandb_task_name,
            group=f'public2D-OCTCubeIR-fundus-{args.data_set}',
            tags=[args.data_set, 'OCTCubeIR-2D', 'fundus', 'finetune'],
            config=vars(args),
            dir=args.log_dir,
            reinit=True,
        )

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    print("Sampler_train = %s" % str(sampler_train))

    if args.dist_eval:
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test) if not args.dist_eval else \
        torch.utils.data.DistributedSampler(dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir + args.task)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)

    debug_samples, debug_targets = next(iter(data_loader_train))
    print("=" * 80)
    print("Dataset:", args.data_set, "| Input batch shape:", debug_samples.shape, "| Target shape:", debug_targets.shape)
    print("=" * 80)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val, batch_size=args.val_batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test, batch_size=args.val_batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    model = models_vit_flash_attn.__dict__[args.model](
        img_size=args.input_size,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    if args.finetune and not args.eval:
        load_octcubeir_2d_tower_checkpoint(model, args.finetune)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    param_groups = lrd.param_groups_lrd(
        model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    print("criterion = %s" % str(criterion))

    # A previously fine-tuned checkpoint (e.g. from a prior run of this script
    # with --save_model), for resuming training or for --eval.
    if args.resume:
        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        init_csv_writer(args.task, mode='test')
        test_stats, auc_roc, auc_pr = evaluate(
            data_loader_test, model, device, args.task, epoch=0, mode='test',
            num_class=args.nb_classes, criterion=criterion, task_mode=args.task_mode,
            disease_list=None, return_bal_acc=args.return_bal_acc, args=args)
        if args.return_bal_acc:
            test_auc_pr, test_bal_acc = auc_pr
        sys.exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_auc = 0.0
    max_auc_pr = 0.0
    max_epoch = 0
    max_accuracy_test = 0.0
    max_auc_test = 0.0
    max_auc_pr_test = 0.0
    max_bal_acc = 0.0
    max_bal_acc_test = 0.0
    max_f1 = 0.0
    max_f1_test = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            loss_scaler, args.clip_grad, mixup_fn, log_writer=log_writer, args=args)
        if train_stats is None:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 2
            print(f"Downscale the learning rate to {param_group['lr']}")

        try:
            val_stats, val_auc_roc, val_auc_pr = evaluate(
                data_loader_val, model, device, args.task, epoch, mode='val',
                num_class=args.nb_classes, criterion=criterion, task_mode=args.task_mode,
                disease_list=None, return_bal_acc=args.return_bal_acc, args=args)
            if args.return_bal_acc:
                val_auc_pr, val_bal_acc = val_auc_pr
        except ValueError as e:
            print(e)
            print('break')
            print(f'break at {epoch}', file=open(os.path.join(args.output_dir, "auc.txt"), mode="a"))
            break

        max_flag = False
        if args.val_metric == 'AUC':
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
                if args.return_bal_acc and val_bal_acc is not None and val_bal_acc > max_bal_acc:
                    max_bal_acc = val_bal_acc
                    max_flag = True
        elif args.val_metric == 'BalAcc':
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

        val_f1 = val_stats.get('f1', 0.0)
        if max_flag is True:
            max_f1 = val_f1
            print(f"Max AUC: {max_auc}, Max ACC: {max_accuracy}, Max AUCPR: {max_auc_pr}, "
                  f"Max Bal Acc: {max_bal_acc}, Max F1: {max_f1}, at epoch {epoch}")
            print(f"Max AUC: {max_auc}, Max ACC: {max_accuracy}, Max AUCPR: {max_auc_pr}, "
                  f"Max Bal Acc: {max_bal_acc}, Max F1: {max_f1}, at epoch {epoch}",
                  file=open(os.path.join(args.output_dir, "auc.txt"), mode="a"))
            if args.output_dir and args.save_model:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        if max_flag or epoch == (args.epochs - 1):
            init_csv_writer(args.task, mode='test')
            try:
                test_stats, test_auc_roc, test_auc_pr = evaluate(
                    data_loader_test, model, device, args.task, epoch, mode='test',
                    num_class=args.nb_classes, criterion=criterion, task_mode=args.task_mode,
                    disease_list=None, return_bal_acc=args.return_bal_acc, args=args)
                if args.return_bal_acc:
                    test_auc_pr, test_bal_acc = test_auc_pr
            except ValueError as e:
                print(e)
                print('break')
                break

            if args.use_wandb and global_rank == 0:
                wandb_test_log = {'epoch': epoch}
                wandb_test_log.update({f'test_{k}': v for k, v in test_stats.items()})
                wandb_test_log.update({'test_auc': test_auc_roc, 'test_auc_pr': test_auc_pr})
                if args.return_bal_acc and test_bal_acc is not None:
                    wandb_test_log['test_bal_acc'] = test_bal_acc
                wandb.log(wandb_test_log, step=epoch)

            max_flag_test = False
            if args.val_metric == 'AUC':
                if max_auc_test <= test_auc_roc:
                    max_auc_test = test_auc_roc
                    if max_auc_test < test_auc_roc:
                        max_flag_test = True
                    elif max_accuracy_test <= test_stats['acc1']:
                        max_accuracy_test = test_stats['acc1']
                        max_flag_test = True
                    elif max_auc_pr_test <= test_auc_pr:
                        max_auc_pr_test = test_auc_pr
                        max_flag_test = True
            elif args.val_metric == 'AUPRC':
                if max_auc_pr_test <= test_auc_pr:
                    if max_auc_pr_test < test_auc_pr:
                        max_auc_test = test_auc_roc
                        max_accuracy_test = test_stats['acc1']
                        max_flag_test = True
                    max_auc_pr_test = test_auc_pr
                    if max_accuracy_test <= test_stats['acc1']:
                        max_accuracy_test = test_stats['acc1']
                        max_auc_test = test_auc_roc
                        max_flag_test = True
                    elif max_auc_test <= test_auc_roc:
                        max_auc_test = test_auc_roc
                        max_accuracy_test = test_stats['acc1']
                        max_flag_test = True
                    if args.return_bal_acc:
                        max_bal_acc_test = test_bal_acc
                        max_flag_test = True
            elif args.val_metric == 'BalAcc':
                if max_bal_acc_test <= test_bal_acc:
                    max_bal_acc_test = test_bal_acc
                    max_auc_test = test_auc_roc
                    max_accuracy_test = test_stats['acc1']
                    max_auc_pr_test = test_auc_pr
                    max_flag_test = True

            if max_flag_test is True:
                max_f1_test = test_stats.get('f1', max_f1_test)
                print(f"Max AUC: {max_auc_test}, Max ACC: {max_accuracy_test}, Max AUCPR: {max_auc_pr_test}, "
                      f"Max Bal Acc: {max_bal_acc_test}, Max F1: {max_f1_test}, at epoch {epoch}")
                print(f"Max AUC: {max_auc_test}, Max ACC: {max_accuracy_test}, Max AUCPR: {max_auc_pr_test}, "
                      f"Max Bal Acc: {max_bal_acc_test}, Max F1: {max_f1_test}, at epoch {epoch}",
                      file=open(os.path.join(args.output_dir, "auc_test.txt"), mode="a"))

        if log_writer is not None:
            log_writer.add_scalar('perf/val_acc1', val_stats['acc1'], epoch)
            log_writer.add_scalar('perf/val_auc', val_auc_roc, epoch)
            log_writer.add_scalar('perf/val_auc_pr', val_auc_pr, epoch)
            log_writer.add_scalar('perf/val_loss', val_stats['loss'], epoch)
            if args.return_bal_acc and val_bal_acc is not None:
                log_writer.add_scalar('perf/val_bal_acc', val_bal_acc, epoch)

        if args.use_wandb and global_rank == 0 and train_stats is not None:
            wandb_log = {'epoch': epoch}
            wandb_log.update({f'train_{k}': v for k, v in train_stats.items()})
            wandb_log.update({f'val_{k}': v for k, v in val_stats.items()})
            wandb_log.update({
                'val_auc': val_auc_roc, 'val_auc_pr': val_auc_pr,
                'max_val_auc': max_auc, 'max_val_acc': max_accuracy,
                'max_val_auc_pr': max_auc_pr, 'max_val_f1': max_f1,
            })
            if args.return_bal_acc and val_bal_acc is not None:
                wandb_log['val_bal_acc'] = val_bal_acc
                wandb_log['max_val_bal_acc'] = max_bal_acc
            wandb.log(wandb_log, step=epoch)

        if train_stats is not None:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch, 'n_parameters': n_parameters,
                         'max_val_acc': max_accuracy, 'max_val_auc': max_auc,
                         'max_val_auc_pr': max_auc_pr, 'max_val_epoch': max_epoch,
                         'max_val_bal_acc': max_bal_acc, 'max_val_f1': max_f1}
            if args.output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a") as f:
                    f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print('Training time {}'.format(total_time_str), file=open(os.path.join(args.output_dir, "time.txt"), mode="a"))

    if args.return_bal_acc:
        results = (max_auc, max_accuracy, max_auc_pr, max_bal_acc, max_f1)
        results_test = (max_auc_test, max_accuracy_test, max_auc_pr_test, max_bal_acc_test, max_f1_test)
    else:
        results = (max_auc, max_accuracy, max_auc_pr, max_f1)
        results_test = (max_auc_test, max_accuracy_test, max_auc_pr_test, max_f1_test)

    print(f"Val results (AUC, ACC, AUCPR[, BalAcc], F1): {results}")
    print(f"Val results (AUC, ACC, AUCPR[, BalAcc], F1): {results}",
          file=open(os.path.join(args.output_dir, "results.txt"), mode="a"))
    print(f"Test results (AUC, ACC, AUCPR[, BalAcc], F1): {results_test}")
    print(f"Test results (AUC, ACC, AUCPR[, BalAcc], F1): {results_test}",
          file=open(os.path.join(args.output_dir, "results_test.txt"), mode="a"))

    if args.use_wandb and global_rank == 0:
        wandb_summary = {
            'final/val_auc': max_auc, 'final/val_acc': max_accuracy, 'final/val_auc_pr': max_auc_pr,
            'final/test_auc': max_auc_test, 'final/test_acc': max_accuracy_test, 'final/test_auc_pr': max_auc_pr_test,
            'final/val_f1': max_f1, 'final/test_f1': max_f1_test,
        }
        if args.return_bal_acc:
            wandb_summary['final/val_bal_acc'] = max_bal_acc
            wandb_summary['final/test_bal_acc'] = max_bal_acc_test
        wandb.log(wandb_summary)
        wandb.finish()


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
