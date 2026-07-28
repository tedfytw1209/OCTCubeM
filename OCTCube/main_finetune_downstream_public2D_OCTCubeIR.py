# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# --------------------------------------------------------

# Public-benchmark fine-tuning / evaluation of OCTCube-IR's 2D en-face ("text")
# tower on the four public single-B-scan classification datasets that already
# have a `Center2D_flash_attn` pipeline in this repo (Duke14, OIMHS, UMN,
# GLAUCOMA -- see main_finetune_downstream_{duke14,oimhs,umn,glaucoma_correct_visit}.py
# and their scripts/cross-{cohort,device}/RETFound-center/*.sh launchers).
#
# Those 4 existing scripts already load a *single* 2D ViT
# (models_vit_flash_attn.flash_attn_vit_large_patch16) from a plain RETFound
# checkpoint. That architecture is identical to OCTCube-IR's en-face tower
# (retinal-COEM's `ViT_flash_attn`, saved under a `text.` prefix inside the
# jointly-pretrained `mm_octcube_ir.pt`, alongside the 3D OCT tower under
# `visual.` -- see main_finetune_downstream_UFcohort_OCTCubeIR.py for the dual
# checkpoint). This script only changes ONE thing relative to those 4 scripts:
# where the pretrained weights come from -- the `text.*` slice of OCTCube-IR's
# joint checkpoint instead of a plain single-tower RETFound checkpoint. The
# only OCT-B-scan-derived 2D image is fed through the tower; OCTCube-IR's OCT
# encoder is unused here.
#
# Dataset loading, k-fold splitting and the train/eval loop are otherwise
# unchanged from those 4 scripts (here trimmed to only the Center2D_flash_attn
# path -- no 3D/RETFound-all/SLIViT branches), parameterized by --data_set so
# one script covers all four datasets (data path/CLI conventions -- --data_root
# + --data_set split, output-dir args-checksum, wandb tags -- follow
# MIRAGE's run_cls_tuning_fundus.py).

import os
import sys
import time
import json
import hashlib
import argparse
import datetime
import numpy as np
import pandas as pd

from pathlib import Path

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
from util.datasets import build_transform
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.pos_embed import interpolate_pos_embed
from util.PatientDataset import TransformableSubset, PatientDatasetCenter2D

from engine_finetune import train_one_epoch, evaluate, init_csv_writer

import models_vit_flash_attn


# Per-dataset data-loading + task config, taken from the existing
# scripts/cross-{cohort,device}/RETFound-center/*.sh launchers (the only thing
# that changes across datasets here is *where the data lives and what the
# task is* -- not the model or the training loop).
DATASET_CONFIGS = {
    'duke14': dict(
        subdir='DUKE_14_Srin/duke14_processed/',
        nb_classes=3, dataset_mode='frame', iterate_mode='patient',
        name_split_char='_', patient_idx_loc=1, cls_unique=True,
        task_mode='multi_cls', random_shuffle_patient=False,
    ),
    'oimhs': dict(
        subdir='OIMHS_dataset/cls_images/',
        nb_classes=3, dataset_mode='frame', iterate_mode='visit',
        name_split_char='_', patient_idx_loc=3, cls_unique=False,
        task_mode='multi_cls', random_shuffle_patient=True,
    ),
    'umn': dict(
        subdir='UMN/UMN_dataset/image_classification/',
        nb_classes=2, dataset_mode='frame', iterate_mode='patient',
        name_split_char='_', patient_idx_loc=2, cls_unique=True,
        task_mode='binary_cls', random_shuffle_patient=True,
    ),
    'glaucoma': dict(
        subdir='GLAUCOMA/glaucoma_processed/',
        nb_classes=2, dataset_mode='volume', iterate_mode='visit',
        name_split_char='-', patient_idx_loc=1, cls_unique=False,
        task_mode='binary_cls', random_shuffle_patient=True,
    ),
}

home_directory = os.getenv('HOME') + '/'
default_data_root = home_directory + 'OCTCubeM/assets/ext_oph_datasets/'


def save_fold_split_to_csv(dataset, train_indices, val_indices, fold, output_dir, dataset_name, iterate_mode='patient'):
    """Save the CV fold split information to a CSV file with frame-level details."""
    import re
    rows = []

    def extract_image_fmt(frame_path):
        frame_dir = os.path.dirname(frame_path)
        frame_name = os.path.basename(frame_path)
        match = re.search(r'(\d+)(\.\w+)$', frame_name)
        if match:
            num_str = match.group(1)
            ext = match.group(2)
            prefix = frame_name[:match.start(1)]
            fmt_spec = f'%0{len(num_str)}d'
            fmt_name = f'{prefix}{fmt_spec}{ext}'
            return os.path.join(frame_dir, fmt_name)
        return frame_path

    def add_rows_for_indices(indices, split_name):
        for idx in indices:
            if iterate_mode == 'patient':
                patient_id = list(dataset.patients.keys())[idx]
                data_dict = dataset.patients[patient_id]
            else:
                data_dict = dataset.visits_dict[idx]
                patient_id = str(idx)

            label = data_dict['class_idx']

            if 'frames' in data_dict and len(data_dict['frames']) > 0:
                frames = data_dict['frames']
                first_frame = frames[0] if isinstance(frames[0], str) else frames[0][0]
                image_fmt = extract_image_fmt(first_frame)

                for slice_num, frame_path in enumerate(frames):
                    if isinstance(frame_path, list):
                        frame_path = frame_path[0]
                    frame_name = os.path.basename(frame_path)
                    rows.append({
                        'patient_id': patient_id,
                        'image_name': frame_name,
                        'image_fmt': image_fmt,
                        'slice_num': slice_num,
                        'split': split_name,
                        'label': label
                    })
            else:
                rows.append({
                    'patient_id': patient_id,
                    'image_name': patient_id,
                    'image_fmt': '',
                    'slice_num': 0,
                    'split': split_name,
                    'label': label
                })

    add_rows_for_indices(train_indices, 'train')
    add_rows_for_indices(val_indices, 'val')

    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f'{dataset_name}_fold_{fold}_split.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved fold {fold} split to {csv_path} ({len(df)} rows)")
    return csv_path


def load_octcubeir_2d_tower_checkpoint(model, finetune_path):
    """Initialize the 2D en-face classifier from OCTCube-IR's en-face ("text") tower.

    `mm_octcube_ir.pt` is a saved retinal-COEM `CustomTextCLIP` (open_clip/model.py)
    whose state_dict stores the 3D OCT tower under a `visual.` prefix and the 2D
    en-face tower -- a `ViT_flash_attn`, architecturally identical to
    `models_vit_flash_attn` here -- under a `text.` prefix (see
    main_finetune_downstream_UFcohort_OCTCubeIR.py's `load_octcubeir_dual_checkpoint`
    for the dual-tower version this is derived from). We keep only the `text.*`
    keys, then load them exactly the way the existing Center2D_flash_attn scripts
    load a plain RETFound checkpoint: drop `head.*` on shape mismatch, interpolate
    the spatial position embedding, and load with strict=False.
    """
    checkpoint = torch.load(finetune_path, map_location='cpu')
    print("Load OCTCube-IR pre-trained checkpoint from: %s" % finetune_path)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        full_sd = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        full_sd = checkpoint['model']
    else:
        full_sd = checkpoint

    # strip a leading DDP 'module.' prefix if present
    full_sd = {k.replace('module.', '', 1): v for k, v in full_sd.items()}

    # keep only the en-face ("text") tower
    checkpoint_model = {k[len('text.'):]: v for k, v in full_sd.items() if k.startswith('text.')}
    print(f"Found {len(checkpoint_model)} text.* (en-face tower) keys out of {len(full_sd)} total keys")
    assert len(checkpoint_model) > 20, (
        "Expected many 'text.*' (en-face tower) keys in the OCTCube-IR checkpoint but "
        f"found {len(checkpoint_model)}. Inspect the checkpoint prefixes with "
        "inspect_mm_octcube_ir_keys.py before running.")

    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    interpolate_pos_embed(model, checkpoint_model)

    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    print(msg.missing_keys)
    assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}, \
        f"Unexpected missing keys after loading OCTCube-IR's en-face tower: {msg.missing_keys}"

    # manually initialize fc layer, as the other Center2D_flash_attn scripts do
    trunc_normal_(model.head.weight, std=2e-5)
    return model


def get_args_parser():
    parser = argparse.ArgumentParser('OCTCube-IR 2D en-face tower fine-tuning on public datasets', add_help=False)

    required_parser = parser.add_argument_group('required arguments')
    required_parser.add_argument(
        '--data_set', type=str, required=True, choices=sorted(DATASET_CONFIGS.keys()),
        help='Which public dataset to fine-tune/evaluate on (required).')
    required_parser.add_argument(
        '--finetune', type=str, required=True,
        help="Path to OCTCube-IR's joint checkpoint (ckpt/mm_octcube_ir.pt); only its "
             "en-face ('text.') tower is loaded. (required)")

    # Dataset parameters
    parser.add_argument('--data_root', default=default_data_root, type=str,
                        help='Root directory containing the per-dataset processed folders '
                             '(default: %(default)s)')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # K-fold cross validation (all 4 datasets are always run with k-fold CV here)
    parser.add_argument('--k_folds', default=10, type=int, help='number of folds for K-fold cross validation')

    # Task / eval parameters
    parser.add_argument('--val_metric', default='AUPRC', type=str, choices=['AUC', 'ACC', 'AUPRC', 'BalAcc'],
                        help='Validation metric used to pick the best epoch per fold. (default: %(default)s)')
    parser.add_argument('--return_bal_acc', default=False, action='store_true', help='also track balanced accuracy')
    parser.add_argument('--not_print_logits', default=False, action='store_true', help='not print logits')
    parser.add_argument('--not_save_figs', default=False, action='store_true', help='not save figures')
    parser.add_argument('--variable_joint', default=False, action='store_true', help=argparse.SUPPRESS)
    parser.set_defaults(variable_joint=False)
    parser.add_argument('--save_model', default=False, action='store_true', help='save the best checkpoint per fold')

    # Training / optimizer parameters (defaults match the common values across
    # the 4 existing RETFound-center launcher scripts; override per dataset in
    # the launcher .sh, matching MIRAGE's launch()-function pattern)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--val_batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=150, type=int)
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

    # Augmentation parameters
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
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    parser.add_argument('--output_dir', default='./outputs_ft_public2D_octcubeir/',
                        help='Base output directory (a --data_set/args-checksum subdir is appended)')
    parser.add_argument('--log_dir', default='./output_dir')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume a previously fine-tuned checkpoint (also used for --eval)')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--eval', action='store_true', help='Evaluate fold 0 only, then exit (see --resume)')
    parser.add_argument('--dist_eval', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-run even if fold_results_test.txt already exists in the resolved output dir')

    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    # wandb
    parser.add_argument('--use_wandb', default=True, action='store_true')
    parser.add_argument('--no_wandb', dest='use_wandb', action='store_false')
    parser.add_argument('--wandb_project', default='OCTCubeM', type=str)
    parser.add_argument('--wandb_entity', default=None, type=str)
    parser.add_argument('--wandb_run_name', default=None, type=str)

    return parser


def process_args(args):
    """Resolve --data_set into the concrete dataset/task config, and --data_root
    + --data_set into a concrete --data_path, mirroring MIRAGE's process_args()
    (which derives args.data_path / args.num_classes from --data_root/--data_set)."""
    cfg = DATASET_CONFIGS[args.data_set]

    if args.data_root[-1] != '/':
        args.data_root += '/'
    args.data_path = args.data_root + cfg['subdir']

    args.nb_classes = cfg['nb_classes']
    args.dataset_mode = cfg['dataset_mode']
    args.iterate_mode = cfg['iterate_mode']
    args.name_split_char = cfg['name_split_char']
    args.patient_idx_loc = cfg['patient_idx_loc']
    args.cls_unique = cfg['cls_unique']
    args.task_mode = cfg['task_mode']
    args.random_shuffle_patient = cfg['random_shuffle_patient']
    # fixed for this script -- see engine_finetune.py, which branches on this string
    args.patient_dataset_type = 'Center2D_flash_attn'

    print(f"Resolved --data_set {args.data_set}: data_path={args.data_path}, "
          f"nb_classes={args.nb_classes}, dataset_mode={args.dataset_mode}, "
          f"iterate_mode={args.iterate_mode}, task_mode={args.task_mode}")
    return args


def get_output_dir(args):
    """Base output dir + --data_set + a checksum of the hyperparameters that
    change run behavior, so distinct sweeps don't collide -- matching MIRAGE's
    args-checksum output-dir convention."""
    checksum_keys = [
        'data_set', 'model', 'blr', 'layer_decay', 'weight_decay', 'drop_path',
        'epochs', 'warmup_epochs', 'batch_size', 'seed', 'k_folds', 'input_size',
    ]
    args_vars = {k: getattr(args, k) for k in checksum_keys}
    args_str = json.dumps(args_vars, indent=2, sort_keys=True)
    args_checksum = hashlib.md5(args_str.encode('utf-8')).hexdigest()[:8]

    output_dir = args.output_dir
    if output_dir[-1] != '/':
        output_dir += '/'
    output_dir += f'{args.data_set}/{args.model}_finetune_w_{args_checksum}/'
    return output_dir


def main(args):
    misc.init_distributed_mode(args)
    args = process_args(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    args.output_dir = get_output_dir(args)
    args.task = os.path.join(args.output_dir, '')  # trailing slash: engine_finetune concatenates args.task + filename

    if os.path.exists(os.path.join(args.output_dir, 'fold_results_test.txt')) and not args.overwrite:
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

    dataset_for_Kfold = PatientDatasetCenter2D(
        root_dir=args.data_path, patient_idx_loc=args.patient_idx_loc, transform=None,
        dataset_mode=args.dataset_mode, name_split_char=args.name_split_char,
        cls_unique=args.cls_unique, iterate_mode=args.iterate_mode,
        random_shuffle_patient=args.random_shuffle_patient,
    )
    print(f"Dataset for Kfold: {len(dataset_for_Kfold)}")

    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    if args.iterate_mode == 'patient':
        patient_indices = range(len(dataset_for_Kfold))
        folds = list(kf.split(patient_indices))
    elif args.iterate_mode == 'visit':
        patient_mapping_visit_indices = sorted(list(dataset_for_Kfold.mapping_patient2visit.keys()))
        rng = np.random.RandomState(args.seed)
        patient_mapping_visit_indices = rng.permutation(patient_mapping_visit_indices)
        folds = list(kf.split(patient_mapping_visit_indices))

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if args.use_wandb and global_rank == 0:
        wandb_task_name = args.wandb_run_name if args.wandb_run_name else (
            f'public2D-OCTCubeIR-{args.data_set}' + datetime.datetime.now().strftime("_%Y%m%d_%H%M%S"))
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=wandb_task_name,
            group=f'public2D-OCTCubeIR-{args.data_set}',
            tags=[args.data_set, 'OCTCubeIR-2D', 'finetune'],
            config=vars(args),
            dir=args.log_dir,
            reinit=True,
        )

    fold_results = []
    fold_results_test = []
    print(f"Start K-fold cross validation for {args.k_folds} folds")
    for fold in range(args.k_folds):
        print(f"Fold {fold}")

        if args.iterate_mode == 'patient':
            train_indices, val_indices = folds[fold]
        elif args.iterate_mode == 'visit':
            idx_train_pat_id, idx_val_pat_id = folds[fold]
            train_pat_id = [patient_mapping_visit_indices[idx] for idx in idx_train_pat_id]
            val_pat_id = [patient_mapping_visit_indices[idx] for idx in idx_val_pat_id]
            train_indices = dataset_for_Kfold.get_visit_idx(train_pat_id)
            val_indices = dataset_for_Kfold.get_visit_idx(val_pat_id)

        save_fold_split_to_csv(dataset_for_Kfold, train_indices, val_indices, fold, args.output_dir, args.data_set, args.iterate_mode)

        dataset_train = TransformableSubset(dataset_for_Kfold, train_indices, transform=train_transform)
        dataset_val = TransformableSubset(dataset_for_Kfold, val_indices, transform=val_transform)
        # No held-out test split in this k-fold setup (matches the existing
        # duke14/oimhs/umn/glaucoma scripts): the val fold doubles as the test fold.
        dataset_test = dataset_val

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        print("Sampler_train = %s" % str(sampler_train))

        if args.dist_eval:
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = sampler_val

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

        # A previously fine-tuned checkpoint (e.g. from a prior run of this
        # script with --save_model), for resuming training or for --eval.
        if args.resume:
            misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

        if args.eval:
            test_mode = f'test_fold_{fold}'
            init_csv_writer(args.task, mode=test_mode)
            test_stats, auc_roc, auc_pr = evaluate(
                data_loader_test, model, device, args.task, epoch=0, mode=test_mode,
                num_class=args.nb_classes, criterion=criterion, task_mode=args.task_mode,
                disease_list=None, return_bal_acc=args.return_bal_acc, args=args)
            if args.return_bal_acc:
                test_auc_pr, test_bal_acc = auc_pr
            # K-fold CV has no single canonical checkpoint, so --eval evaluates
            # fold 0 against --resume (or --finetune's un-adapted init, if
            # --resume is empty) and stops here.
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
        # Precision/Recall/Kappa/MCC at the best (val_metric-selected) epoch --
        # engine_finetune.evaluate() already computes these every call (see its
        # macro_metrics dict); tracked here the same way max_f1/max_bal_acc are,
        # so they show up in wandb and in fold_results.txt alongside ACC/AUC/PR/F1.
        max_precision = 0.0
        max_recall = 0.0
        max_kappa = 0.0
        max_mcc = 0.0
        max_precision_test = 0.0
        max_recall_test = 0.0
        max_kappa_test = 0.0
        max_mcc_test = 0.0
        val_mode = f'val_fold_{fold}'

        if args.task_mode == 'binary_cls':
            init_csv_writer(args.task, mode=val_mode)

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
                    data_loader_val, model, device, args.task, epoch, mode=val_mode,
                    num_class=args.nb_classes, criterion=criterion, task_mode=args.task_mode,
                    disease_list=None, return_bal_acc=args.return_bal_acc, args=args)
                if args.return_bal_acc:
                    val_auc_pr, val_bal_acc = val_auc_pr
            except ValueError as e:
                print(e)
                print('break')
                print(f'break at {epoch}', file=open(os.path.join(args.output_dir, f"auc_fold_{fold}.txt"), mode="a"))
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
                max_precision = val_stats.get('precision', 0.0)
                max_recall = val_stats.get('recall', 0.0)
                max_kappa = val_stats.get('kappa', 0.0)
                max_mcc = val_stats.get('mcc', 0.0)
                print(f"Max AUC: {max_auc}, Max ACC: {max_accuracy}, Max AUCPR: {max_auc_pr}, "
                      f"Max Bal Acc: {max_bal_acc}, Max F1: {max_f1}, Max Precision: {max_precision}, "
                      f"Max Recall: {max_recall}, Max Kappa: {max_kappa}, Max MCC: {max_mcc}, at epoch {epoch}")
                print(f"Max AUC: {max_auc}, Max ACC: {max_accuracy}, Max AUCPR: {max_auc_pr}, "
                      f"Max Bal Acc: {max_bal_acc}, Max F1: {max_f1}, Max Precision: {max_precision}, "
                      f"Max Recall: {max_recall}, Max Kappa: {max_kappa}, Max MCC: {max_mcc}, at epoch {epoch}",
                      file=open(os.path.join(args.output_dir, f"auc_fold_{fold}.txt"), mode="a"))
                if args.output_dir and args.save_model:
                    misc.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp,
                        optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

            if max_flag or epoch == (args.epochs - 1):
                test_mode = f'test_fold_{fold}'
                init_csv_writer(args.task, mode=test_mode)
                try:
                    test_stats, test_auc_roc, test_auc_pr = evaluate(
                        data_loader_test, model, device, args.task, epoch, mode=test_mode,
                        num_class=args.nb_classes, criterion=criterion, task_mode=args.task_mode,
                        disease_list=None, return_bal_acc=args.return_bal_acc, args=args)
                    if args.return_bal_acc:
                        test_auc_pr, test_bal_acc = test_auc_pr
                except ValueError as e:
                    print(e)
                    print('break')
                    break

                if args.use_wandb and global_rank == 0:
                    wandb_test_log = {f'fold_{fold}/epoch': epoch}
                    wandb_test_log.update({f'fold_{fold}/test_{k}': v for k, v in test_stats.items()})
                    wandb_test_log.update({f'fold_{fold}/test_auc': test_auc_roc, f'fold_{fold}/test_auc_pr': test_auc_pr})
                    if args.return_bal_acc and test_bal_acc is not None:
                        wandb_test_log[f'fold_{fold}/test_bal_acc'] = test_bal_acc
                    wandb.log(wandb_test_log, step=epoch + fold * args.epochs)

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
                    max_precision_test = test_stats.get('precision', max_precision_test)
                    max_recall_test = test_stats.get('recall', max_recall_test)
                    max_kappa_test = test_stats.get('kappa', max_kappa_test)
                    max_mcc_test = test_stats.get('mcc', max_mcc_test)
                    print(f"Max AUC: {max_auc_test}, Max ACC: {max_accuracy_test}, Max AUCPR: {max_auc_pr_test}, "
                          f"Max Bal Acc: {max_bal_acc_test}, Max F1: {max_f1_test}, Max Precision: {max_precision_test}, "
                          f"Max Recall: {max_recall_test}, Max Kappa: {max_kappa_test}, Max MCC: {max_mcc_test}, at epoch {epoch}")
                    print(f"Max AUC: {max_auc_test}, Max ACC: {max_accuracy_test}, Max AUCPR: {max_auc_pr_test}, "
                          f"Max Bal Acc: {max_bal_acc_test}, Max F1: {max_f1_test}, Max Precision: {max_precision_test}, "
                          f"Max Recall: {max_recall_test}, Max Kappa: {max_kappa_test}, Max MCC: {max_mcc_test}, at epoch {epoch}",
                          file=open(os.path.join(args.output_dir, f"auc_test_fold_{fold}.txt"), mode="a"))

            if log_writer is not None:
                log_writer.add_scalar('perf/val_acc1', val_stats['acc1'], epoch)
                log_writer.add_scalar('perf/val_auc', val_auc_roc, epoch)
                log_writer.add_scalar('perf/val_auc_pr', val_auc_pr, epoch)
                log_writer.add_scalar('perf/val_loss', val_stats['loss'], epoch)
                if args.return_bal_acc and val_bal_acc is not None:
                    log_writer.add_scalar('perf/val_bal_acc', val_bal_acc, epoch)

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
                    f'fold_{fold}/max_val_f1': max_f1,
                    f'fold_{fold}/max_val_precision': max_precision,
                    f'fold_{fold}/max_val_recall': max_recall,
                    f'fold_{fold}/max_val_kappa': max_kappa,
                    f'fold_{fold}/max_val_mcc': max_mcc,
                })
                if args.return_bal_acc and val_bal_acc is not None:
                    wandb_log[f'fold_{fold}/val_bal_acc'] = val_bal_acc
                    wandb_log[f'fold_{fold}/max_val_bal_acc'] = max_bal_acc
                wandb.log(wandb_log, step=epoch + fold * args.epochs)

            if train_stats is not None:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'epoch': epoch, 'n_parameters': n_parameters,
                             'max_val_acc': max_accuracy, 'max_val_auc': max_auc,
                             'max_val_auc_pr': max_auc_pr, 'max_val_epoch': max_epoch,
                             'max_val_bal_acc': max_bal_acc, 'max_val_f1': max_f1,
                             'max_val_precision': max_precision, 'max_val_recall': max_recall,
                             'max_val_kappa': max_kappa, 'max_val_mcc': max_mcc}
                if args.output_dir and misc.is_main_process():
                    if log_writer is not None:
                        log_writer.flush()
                    with open(os.path.join(args.output_dir, f"log_fold_{fold}.txt"), mode="a") as f:
                        f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        print('Training time {}'.format(total_time_str), file=open(os.path.join(args.output_dir, f"time_fold_{fold}.txt"), mode="a"))

        # Named dict (not a positional tuple) so ACC/F1/AUC/PR/Precision/Recall/
        # Kappa/MCC[/BalAcc] -- all 8 metrics engine_finetune.evaluate() already
        # computes -- are always explicit in fold_results*.txt and in wandb,
        # matching MIRAGE's EVAL_CSV_COLUMNS convention.
        val_fold_metrics = {
            'auc': max_auc, 'acc': max_accuracy, 'auc_pr': max_auc_pr,
            'precision': max_precision, 'recall': max_recall,
            'kappa': max_kappa, 'mcc': max_mcc, 'f1': max_f1,
        }
        test_fold_metrics = {
            'auc': max_auc_test, 'acc': max_accuracy_test, 'auc_pr': max_auc_pr_test,
            'precision': max_precision_test, 'recall': max_recall_test,
            'kappa': max_kappa_test, 'mcc': max_mcc_test, 'f1': max_f1_test,
        }
        if args.return_bal_acc:
            val_fold_metrics['bal_acc'] = max_bal_acc
            test_fold_metrics['bal_acc'] = max_bal_acc_test
        fold_results.append(val_fold_metrics)
        fold_results_test.append(test_fold_metrics)

    def summarize_fold_results(results, label):
        names = list(results[0].keys())
        arr = np.array([[r[n] for n in names] for r in results])
        mean = dict(zip(names, np.mean(arr, axis=0)))
        std = dict(zip(names, np.std(arr, axis=0)))
        print(f"{label} fold results: {results}\nMean: {mean}\nStd: {std}")
        print(f"{label} fold results: {results}\nMean: {mean}\nStd: {std}",
              file=open(os.path.join(args.output_dir, f"fold_results_{label}.txt"), mode="a"))
        return mean, std

    val_mean, val_std = summarize_fold_results(fold_results, 'val')
    test_mean, test_std = summarize_fold_results(fold_results_test, 'test')

    if args.use_wandb and global_rank == 0:
        wandb_summary = {}
        for name, value in val_mean.items():
            wandb_summary[f'final/mean_val_{name}'] = value
            wandb_summary[f'final/std_val_{name}'] = val_std[name]
        for name, value in test_mean.items():
            wandb_summary[f'final/mean_test_{name}'] = value
            wandb_summary[f'final/std_test_{name}'] = test_std[name]
        wandb.log(wandb_summary)
        wandb.finish()


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
