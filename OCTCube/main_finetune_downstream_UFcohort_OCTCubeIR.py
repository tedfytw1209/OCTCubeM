# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Revised by Zixuan Zucks Liu @University of Washington

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

# OCTCube-IR dual-modality (3D OCT volume + 2D en-face IR) fine-tuning / evaluation
# on the UF cohort.
#
# MODEL + CHECKPOINT LOADING follow retinal-COEM (the OCTCube-IR model). The model
# reproduces retinal-COEM's `CustomTextCLIPClassification`
# (retinal-COEM/src/open_clip/model.py:741): two towers, each projecting to
# `embed_dim` (=512), whose L2-normalized features are concatenated and passed
# through a `ClassificationHead` (LayerNorm -> Linear -> GELU -> Linear). BOTH
# towers are initialized from the single jointly-pretrained OCTCube-IR checkpoint
# (`mm_octcube_ir.pt`), a saved `CustomTextCLIP` whose `state_dict` stores the 3D
# OCT tower under a `visual.` prefix and the 2D en-face tower under a `text.`
# prefix. We cannot `import open_clip` from here (its model.py imports a missing
# `model_backup` package), so we replicate retinal-COEM's tower build + per-tower
# checkpoint init using the architecturally-identical OCTCube tower classes
# (`models_vit_st_flash_attn_nodrop`, `models_vit_flash_attn`) — the same source
# the retinal-COEM copies derive from.
#
# DATASET + TASK follow `main_finetune_downstream_UFcohort_dual.py`: the paired
# `Dual_Dataset` (OCT `PatientDataset3D` + en-face `PatientDataset2D` from the
# same UF-cohort CSV) and the dual train/eval engine (`engine_finetune_dual`).

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
import torch.nn as nn
import torch.nn.functional as F
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

from util.PatientDataset import PatientDataset3D, PatientDataset2D, Dual_Dataset
from util.PatientDataset_inhouse import create_3d_transforms
from util.datasets import build_transform

from engine_finetune_dual import train_one_epoch_dual, evaluate_dual, init_csv_writer
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


class ClassificationHead(nn.Module):
    """Exact reproduction of retinal-COEM's `ClassificationHead`
    (retinal-COEM/src/open_clip/model.py:723): LayerNorm -> Linear -> GELU -> Linear.
    """
    def __init__(self, input_dim, hidden_dim, num_classes, initialization=True):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.input_norm = nn.LayerNorm(input_dim)
        if initialization:
            torch.nn.init.normal_(self.fc1.weight, std=0.02)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class DualViT_OCTCubeIR_Classifier(nn.Module):
    """OCTCube-IR dual-tower classifier — reproduces retinal-COEM's
    `CustomTextCLIPClassification` fusion (model.py:741), arranged like OCTCube's
    `DualViTClassifier` (models_vit.py) so it plugs into `engine_finetune_dual`
    (forward(oct, cfp) -> logits) and `lrd.param_groups_lrd` (which reads
    `self.blocks`).

    Each tower projects to `embed_dim` via its own head (the CLIP projection);
    the two features are L2-normalized, concatenated (2*embed_dim), and passed
    through a fresh `ClassificationHead` to `num_classes` logits.
    """
    def __init__(self, vit_model_1, vit_model_2, embed_dim, num_classes):
        super(DualViT_OCTCubeIR_Classifier, self).__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.vit_model_1 = vit_model_1  # visual / 3D OCT tower  (out_dim = embed_dim)
        self.vit_model_2 = vit_model_2  # text  / 2D en-face tower (out_dim = embed_dim)
        # Combined block list so lrd.param_groups_lrd can count layers, exactly as
        # DualViTClassifier does (shares the same block module objects; no new params).
        self.blocks = nn.ModuleList(list(vit_model_1.blocks) + list(vit_model_2.blocks))
        # retinal-COEM fuses the two embed_dim features -> 2*embed_dim -> ClassificationHead.
        self.classification_head = ClassificationHead(2 * embed_dim, hidden_dim=embed_dim, num_classes=num_classes)

    def forward(self, input_1, input_2):
        # tower features (post-projection, embed_dim), L2-normalized as in
        # retinal-COEM's encode_image/encode_text (normalize=True).
        features_1 = F.normalize(self.vit_model_1(input_1), dim=-1)
        features_2 = F.normalize(self.vit_model_2(input_2), dim=-1)
        concatenated_features = torch.cat((features_1, features_2), dim=-1)
        return self.classification_head(concatenated_features)

    def no_weight_decay(self):
        s1 = {f'vit_model_1.{k}' for k in self.vit_model_1.no_weight_decay()}
        s2 = {f'vit_model_2.{k}' for k in self.vit_model_2.no_weight_decay()}
        return s1 | s2


def get_model(patient_dataset_type, args):
    """Build the OCTCube-IR dual-tower classifier (retinal-COEM CustomTextCLIP-style).

    The towers are the OCTCube copies of retinal-COEM's `ViT_ST_nodrop` (3D OCT)
    and `ViT_flash_attn` (2D en-face), each built with the head projecting to
    `embed_dim` (= mm_embed_dim, the OCTCube-IR CLIP embedding dim, 512) — i.e.,
    `num_classes=args.mm_embed_dim` here plays the role of retinal-COEM's
    `out_dim=embed_dim`, so the loaded `visual.head`/`text.head` (embed_dim x width)
    match and are preserved.
    """
    assert patient_dataset_type == 'Dual', \
        "main_finetune_downstream_UFcohort_OCTCubeIR.py only supports --patient_dataset_type Dual"
    # OCT tower (visual): 3D spatio-temporal ViT, flash attention, no dropout.
    sub_model_oct = models_vit_st_flash_attn_nodrop.__dict__[args.model](
        num_frames=args.num_frames,
        t_patch_size=args.t_patch_size,
        image_size=args.input_size,
        num_classes=args.mm_embed_dim,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
        sep_pos_embed=args.sep_pos_embed,
        cls_embed=args.cls_embed,
        use_flash_attention=True,
    )
    print('OCT sub-model (3D, visual tower):')
    print(sub_model_oct)
    # En-face tower (text): 2D ViT, flash attention.
    sub_model_cfp = models_vit_flash_attn.__dict__[args.model](
        img_size=args.input_size,
        num_classes=args.mm_embed_dim,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )
    print('En-face sub-model (2D, text tower):')
    print(sub_model_cfp)
    model = DualViT_OCTCubeIR_Classifier(
        vit_model_1=sub_model_oct,
        vit_model_2=sub_model_cfp,
        embed_dim=args.mm_embed_dim,
        num_classes=args.nb_classes,
    )
    return model


def _load_tower_from_state_dict(model, checkpoint_model, is_3d, args, tower_name=''):
    """Load one tower's (already prefix-stripped) state dict, exactly as
    retinal-COEM initializes these towers from a checkpoint (model.py: the
    `ViT_ST_nodrop` block ~271-295 and the `ViT_flash_attn` block ~486-509):
    drop `head.{weight,bias}` only on shape mismatch, interpolate the spatial
    position embedding (plus temporal for the 3D OCT tower), then load with
    `strict=False`.

    Because both towers here are built with the head projecting to `embed_dim`
    (matching the checkpoint's CLIP projection), the head shapes match and are
    KEPT — the full pretrained OCTCube-IR tower (including `head` and `fc_norm`)
    is loaded. The classification head lives outside the towers and stays freshly
    initialized.
    """
    state_dict = model.state_dict()

    # retinal-COEM drops head only when its shape mismatches; keep it otherwise.
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"[{tower_name}] Removing shape-mismatched key {k}: "
                  f"{tuple(checkpoint_model[k].shape)} vs {tuple(state_dict[k].shape)}")
            del checkpoint_model[k]

    # Defensively drop any remaining shape-mismatched key (strict=False still
    # raises on a shape conflict for a key present in both).
    for k in list(checkpoint_model.keys()):
        if k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"[{tower_name}] Removing shape-mismatched key {k}: "
                  f"{tuple(checkpoint_model[k].shape)} vs {tuple(state_dict[k].shape)}")
            del checkpoint_model[k]

    # interpolate position embeddings (temporal only applies to the 3D OCT tower)
    if is_3d:
        interpolate_pos_embed(model, checkpoint_model)
        interpolate_temporal_pos_embed(model, checkpoint_model,
                                       smaller_interpolate_type=args.smaller_temporal_crop)
    else:
        interpolate_pos_embed(model, checkpoint_model)

    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(f"[{tower_name}] load_state_dict message: {msg}")
    print(f"[{tower_name}] missing keys: {msg.missing_keys}")
    print(f"[{tower_name}] unexpected keys: {msg.unexpected_keys}")
    return model


def load_octcubeir_dual_checkpoint(oct_model, fundus_model, finetune, args):
    """Initialize BOTH towers from the single jointly-pretrained OCTCube-IR checkpoint.

    `mm_octcube_ir.pt` is a saved `CustomTextCLIP` (retinal-COEM/src/open_clip/
    model.py:635) whose `state_dict` stores the 3D OCT tower (`self.visual`, a
    `ViT_ST_nodrop`) under a `visual.` prefix and the 2D en-face tower
    (`self.text`, a `ViT_flash_attn`) under a `text.` prefix. It is saved as
    `{'epoch', 'name', 'state_dict', 'optimizer', ...}` (see
    training/main_retclip.py), optionally with a DDP `module.` prefix. We strip
    `module.`, split by tower prefix, and load each tower with retinal-COEM's own
    per-tower init recipe (see `_load_tower_from_state_dict`).
    """
    checkpoint = torch.load(finetune, map_location='cpu')
    print("Load OCTCube-IR pre-trained checkpoint from: %s" % finetune)
    print("checkpoint top-level keys: ", list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint))

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        full_sd = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        full_sd = checkpoint['model']
    else:
        full_sd = checkpoint

    # strip a leading DDP 'module.' prefix if present
    full_sd = {k.replace('module.', '', 1): v for k, v in full_sd.items()}

    # split the joint checkpoint into the two towers by prefix
    oct_sd = {k[len('visual.'):]: v for k, v in full_sd.items() if k.startswith('visual.')}
    fundus_sd = {k[len('text.'):]: v for k, v in full_sd.items() if k.startswith('text.')}
    print(f"Found {len(oct_sd)} visual.* (OCT tower) keys and "
          f"{len(fundus_sd)} text.* (en-face tower) keys "
          f"out of {len(full_sd)} total keys")

    assert len(oct_sd) > 20, (
        "Expected many 'visual.*' (OCT tower) keys in the OCTCube-IR checkpoint but "
        f"found {len(oct_sd)}. Inspect the checkpoint prefixes with "
        "inspect_mm_octcube_ir_keys.py before running.")
    assert len(fundus_sd) > 20, (
        "Expected many 'text.*' (en-face tower) keys in the OCTCube-IR checkpoint but "
        f"found {len(fundus_sd)}. Inspect the checkpoint prefixes with "
        "inspect_mm_octcube_ir_keys.py before running.")

    # Fail loudly if the checkpoint's CLIP projection dim disagrees with the towers
    # we built (num_classes=mm_embed_dim). Otherwise the mismatched `head.*` would be
    # silently dropped and the pretrained projection re-initialized at random,
    # quietly weakening the "both towers from OCTCube-IR" initialization.
    for tname, tsd in [('visual', oct_sd), ('text', fundus_sd)]:
        if 'head.weight' in tsd:
            ckpt_out_dim = tsd['head.weight'].shape[0]
            if ckpt_out_dim != args.mm_embed_dim:
                raise ValueError(
                    f"OCTCube-IR checkpoint '{tname}.head' out-dim ({ckpt_out_dim}) != "
                    f"--mm_embed_dim ({args.mm_embed_dim}). Re-run with "
                    f"--mm_embed_dim {ckpt_out_dim} so the pretrained projection head "
                    f"loads instead of being dropped and randomly re-initialized.")

    _load_tower_from_state_dict(oct_model, oct_sd, is_3d=True, args=args,
                                tower_name='OCT/visual')
    _load_tower_from_state_dict(fundus_model, fundus_sd, is_3d=False, args=args,
                                tower_name='en-face/text')
    return oct_model, fundus_model


def get_args_parser():
    parser = argparse.ArgumentParser('OCTCube-IR dual-modality fine-tuning', add_help=False)
    # Slivit parameters (unused here, kept for arg-compatibility with the template)
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

    # OCTCube-IR checkpoint / model parameters
    parser.add_argument('--mm_embed_dim', default=512, type=int,
                        help='OCTCube-IR CLIP embedding dim (retinal-COEM config embed_dim=512); '
                             'each tower projects to this and the two are concatenated (2*embed_dim)')

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
    parser.add_argument('--data_path', default=home_directory + dataset_path, type=str, help='dataset path')
    parser.add_argument('--csv_path', default=home_directory + dataset_path, type=str, help='csv path')
    parser.add_argument('--patient_dataset', default='', type=str, help='Use patient dataset')
    parser.add_argument('--patient_dataset_type', default='Dual', type=str, choices=['Dual'], help='patient dataset type (only Dual is supported by this script)')
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

    # K_fold cross validation (UFcohort always runs with k_fold False)
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
    parser.add_argument('--finetune', default='', type=str,
                        help='OCTCube-IR joint checkpoint (mm_octcube_ir.pt); both towers are '
                             'initialized from it (visual.* -> OCT tower, text.* -> en-face tower)')
    parser.add_argument('--few_shot', default=False, action='store_true',
                        help='(kept for arg-compatibility; no-op on the UFcohort single-split path)')
    parser.add_argument('--task', default=f'./finetune_{dataset_name}/', type=str,
                        help='task name / relative output subdir')
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
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint (also used to load a finetuned model for --eval)')

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
    parser.add_argument('--subsetseed', default=42, type=int,
                        help='RNG seed for stratified subset sampling; vary across bootstrap runs')
    parser.add_argument('--bootstrap_runs', default=False, action='store_true',
                        help='Bootstrap sampling for the training subset; groups runs by seed in wandb '
                             'and writes each seed to an output subdir seed_<subsetseed>/')

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

    # Bootstrap runs are grouped by seed in wandb and isolated per seed on disk
    # (output_dir/seed_<subsetseed>/), following main_finetune_downstream_UFcohort.py.
    if args.bootstrap_runs:
        project_name = "OCTCubeM_bootstrap"
        args.task = args.task[:120]  # wandb group name max length is 128
        group_name = args.task.replace('.', '').replace('/', '_')
        wandb_task_name = "seed_" + str(args.subsetseed)
        model_add_dir = "seed_" + str(args.subsetseed)
    else:
        project_name = "OCTCubeM"
        wandb_task_name = args.task.replace('.', '').replace('/', '_') + datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
        group_name = None
        model_add_dir = ""

    wandb.init(
        project=project_name,
        name=wandb_task_name,
        group=group_name,
        config=args,
        dir=os.path.join(args.log_dir, wandb_task_name, model_add_dir),
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

    # ===== DATASET + TASK follow main_finetune_downstream_UFcohort_dual.py =====
    assert args.patient_dataset == 'UFcohort', \
        "This script only supports --patient_dataset UFcohort"
    assert args.patient_dataset_type == 'Dual', \
        "This script only supports --patient_dataset_type Dual"

    # transforms (OCT: monai_3D/volume_3D ; en-face: 2D)
    if args.transform_type == 'volume_3D':
        oct_train_transform = video_transforms.Compose([
            video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        oct_val_transform = oct_train_transform
    elif args.transform_type == 'monai_3D':
        oct_train_transform, oct_val_transform = create_3d_transforms(**vars(args))

    fundus_train_transform = build_transform(is_train='train', args=args)
    fundus_val_transform = build_transform(is_train='val', args=args)
    if args.not_use_2d_aug:
        fundus_train_transform = build_transform(is_train='val', args=args)
        fundus_val_transform = build_transform(is_train='val', args=args)

    # splits
    if not args.testval:
        tr_istrain = 'train'
        val_istrain = 'val'
    else:
        tr_istrain = ['train', 'val']
        val_istrain = 'test'

    # OCT (3D) datasets  [B, C, T, H, W]
    oct_dataset_train = PatientDataset3D(root_dir=args.data_path, patient_idx_loc=args.patient_idx_loc, transform=None, dataset_mode=args.dataset_mode, name_split_char=args.name_split_char, cls_unique=args.cls_unique, iterate_mode=args.iterate_mode, max_frames=args.max_frames, mode=args.color_mode, transform_type=args.transform_type, volume_resize=args.input_size, same_3_frames=args.same_3_frames, csv_path=args.csv_path, is_train=tr_istrain)
    oct_dataset_val = PatientDataset3D(root_dir=args.data_path, patient_idx_loc=args.patient_idx_loc, transform=None, dataset_mode=args.dataset_mode, name_split_char=args.name_split_char, cls_unique=args.cls_unique, iterate_mode=args.iterate_mode, max_frames=args.max_frames, mode=args.color_mode, transform_type=args.transform_type, volume_resize=args.input_size, same_3_frames=args.same_3_frames, csv_path=args.csv_path, is_train=val_istrain)
    oct_dataset_test = PatientDataset3D(root_dir=args.data_path, patient_idx_loc=args.patient_idx_loc, transform=None, dataset_mode=args.dataset_mode, name_split_char=args.name_split_char, cls_unique=args.cls_unique, iterate_mode=args.iterate_mode, max_frames=args.max_frames, mode=args.color_mode, transform_type=args.transform_type, volume_resize=args.input_size, same_3_frames=args.same_3_frames, csv_path=args.csv_path, is_train='test')

    oct_dataset_train.update_transform(oct_train_transform)
    oct_dataset_val.update_transform(oct_val_transform)
    oct_dataset_test.update_transform(oct_val_transform)

    # en-face (2D) datasets
    fundus_dataset_train = PatientDataset2D(root_dir=args.data_path, patient_idx_loc=args.patient_idx_loc, transform=None, dataset_mode=args.dataset_mode, name_split_char=args.name_split_char, cls_unique=args.cls_unique, iterate_mode=args.iterate_mode, csv_path=args.csv_path, is_train=tr_istrain)
    fundus_dataset_val = PatientDataset2D(root_dir=args.data_path, patient_idx_loc=args.patient_idx_loc, transform=None, dataset_mode=args.dataset_mode, name_split_char=args.name_split_char, cls_unique=args.cls_unique, iterate_mode=args.iterate_mode, csv_path=args.csv_path, is_train=val_istrain)
    fundus_dataset_test = PatientDataset2D(root_dir=args.data_path, patient_idx_loc=args.patient_idx_loc, transform=None, dataset_mode=args.dataset_mode, name_split_char=args.name_split_char, cls_unique=args.cls_unique, iterate_mode=args.iterate_mode, csv_path=args.csv_path, is_train='test')

    fundus_dataset_train.update_transform(fundus_train_transform)
    fundus_dataset_val.update_transform(fundus_val_transform)
    fundus_dataset_test.update_transform(fundus_val_transform)

    # merge OCT (3D) + en-face (2D) into paired dual datasets
    dataset_train = Dual_Dataset(oct_dataset_train, fundus_dataset_train)
    dataset_val = Dual_Dataset(oct_dataset_val, fundus_dataset_val)
    dataset_test = Dual_Dataset(oct_dataset_test, fundus_dataset_test)

    # Forward class metadata from the OCT sub-dataset so the subset path
    # (which reads dataset.targets / .classes / .class_to_idx / .annotations) works.
    for dual_ds, oct_ds in [(dataset_train, oct_dataset_train),
                            (dataset_val, oct_dataset_val),
                            (dataset_test, oct_dataset_test)]:
        for attr in ['targets', 'classes', 'class_to_idx', 'annotations']:
            if hasattr(oct_ds, attr):
                setattr(dual_ds, attr, getattr(oct_ds, attr))

    # Optional subset sampling (single full-training-set run uses neither);
    # mirrors main_finetune_downstream_UFcohort_dual.py.
    if hasattr(args, 'subset_num') and args.subset_num > 0:
        print(f'Old subset method for absolute number {args.subset_num}')

        def create_subset_by_num(dataset, split_name, subset_num):
            """Create a subset of the dataset with specified absolute number"""
            targets = np.array(dataset.targets)
            unique_classes, class_counts = np.unique(targets, return_counts=True)
            n_classes = len(unique_classes)

            print(f'{split_name} - Original size: {len(dataset)}, Classes: {n_classes}, Target subset size: {subset_num}')

            if subset_num < n_classes:
                print(f'Warning: subset_num ({subset_num}) < number of classes ({n_classes}), using random sampling')
                rng = np.random.RandomState(args.subsetseed)
                subset_indices = rng.choice(len(dataset), min(subset_num, len(dataset)), replace=False)
            else:
                if subset_num >= len(dataset):
                    print(f'Warning: subset_num ({subset_num}) >= dataset size ({len(dataset)}), using full dataset')
                    subset_indices = list(range(len(dataset)))
                else:
                    sss = StratifiedShuffleSplit(n_splits=1, train_size=subset_num, random_state=args.subsetseed)
                    subset_indices = next(sss.split(range(len(dataset)), targets))[0]

            subset_dataset = Subset(dataset, subset_indices)
            subset_dataset.targets = [dataset.targets[i] for i in subset_indices]
            # Dual_Dataset exposes .targets (forwarded from the OCT sub-dataset) but
            # not necessarily .classes/.class_to_idx; only copy what exists.
            if hasattr(dataset, 'classes'):
                subset_dataset.classes = dataset.classes
            if hasattr(dataset, 'class_to_idx'):
                subset_dataset.class_to_idx = dataset.class_to_idx

            print(f'{split_name} - Final subset size: {len(subset_dataset)}')
            return subset_dataset

        dataset_train = create_subset_by_num(dataset_train, 'Train', int(args.subset_num))

    if hasattr(args, 'new_subset_num') and args.new_subset_num > 0:
        print(f'New subset method for absolute number {args.new_subset_num}')

        def create_separate_class_based_subsets(train_dataset, val_dataset, total_subset_num):
            """Create separate subsets from train and validation datasets based on class ratios"""

            def create_class_balanced_subset(dataset, split_name, target_size):
                targets = np.array(dataset.targets)
                unique_classes, class_counts = np.unique(targets, return_counts=True)
                n_classes = len(unique_classes)
                class_ratios = class_counts / len(targets)

                print(f'\n{split_name} dataset - Original size: {len(dataset)}, Classes: {n_classes}')
                print(f'{split_name} class counts: {dict(zip(unique_classes, class_counts))}')
                print(f'{split_name} class ratios: {dict(zip(unique_classes, class_ratios))}')
                print(f'{split_name} target subset size: {target_size}')

                rng = np.random.RandomState(args.subsetseed)
                selected_indices = []

                for class_idx in unique_classes:
                    class_mask = targets == class_idx
                    class_samples = np.where(class_mask)[0]
                    class_samples_copy = class_samples.copy()
                    rng.shuffle(class_samples_copy)
                    class_target_samples = int((target_size - n_classes) * class_ratios[class_idx]) + 1
                    available_samples = len(class_samples_copy)
                    if class_target_samples > available_samples:
                        print(f'Warning: {split_name} Class {class_idx} needs {class_target_samples} samples but only {available_samples} available')
                        class_target_samples = available_samples
                    selected_class_samples = class_samples_copy[:class_target_samples]
                    selected_indices.extend(selected_class_samples)
                    print(f'{split_name} Class {class_idx}: ratio={class_ratios[class_idx]:.3f}, target={class_target_samples}, selected={len(selected_class_samples)}')

                print('Selected indices:', selected_indices)
                subset_dataset = Subset(dataset, selected_indices)
                subset_dataset.targets = [dataset.targets[i] for i in selected_indices]
                # Dual_Dataset exposes .targets (forwarded from the OCT sub-dataset) but
                # not necessarily .annotations/.classes/.class_to_idx; only copy what exists.
                if hasattr(dataset, 'annotations'):
                    subset_dataset.annotations = dataset.annotations.iloc[selected_indices].reset_index(drop=True)
                if hasattr(dataset, 'classes'):
                    subset_dataset.classes = dataset.classes
                if hasattr(dataset, 'class_to_idx'):
                    subset_dataset.class_to_idx = dataset.class_to_idx

                print(f'{split_name} final subset size: {len(subset_dataset)}')
                return subset_dataset

            train_target_size = int(total_subset_num * 0.8)
            val_target_size = int(total_subset_num * 0.2)
            print(f'Total target subset size: {total_subset_num}')
            print(f'Train target size: {train_target_size} (80%)')
            print(f'Validation target size: {val_target_size} (20%)')
            train_subset = create_class_balanced_subset(train_dataset, 'Train', train_target_size)
            val_subset = create_class_balanced_subset(val_dataset, 'Validation', val_target_size)
            return train_subset, val_subset

        dataset_train, dataset_val = create_separate_class_based_subsets(dataset_train, dataset_val, int(args.new_subset_num))

    assert args.k_fold is False

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    val_mode = 'val'
    # Train sampler shuffles across epochs (correctness; the paired Dual_Dataset
    # keeps OCT/en-face aligned by index so shuffling stays consistent).
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
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if args.dist_eval:
        if len(dataset_test) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir + args.task)
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

    # ===== MODEL + CHECKPOINT LOADING follow retinal-COEM (OCTCube-IR) =====
    model = get_model(args.patient_dataset_type, args)
    if args.finetune and not args.eval:
        load_octcubeir_dual_checkpoint(model.vit_model_1, model.vit_model_2, args.finetune, args)

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
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    # evaluate_dual writes `<task> + 'metrics_*.csv'` / confusion-matrix figures by
    # string concatenation, so pass the run's output_dir (with a trailing separator)
    # to co-locate metrics with checkpoint-best.pth / log.txt instead of scattering
    # them under a CWD-relative args.task path.
    metrics_dir = os.path.join(args.output_dir, model_add_dir, '') if args.output_dir else args.task

    # resume training / load a finetuned model for --eval (guarded: misc.load_model
    # raises on an empty --resume, so only call it when a checkpoint is given).
    if args.resume:
        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats, auc_roc, auc_pr = evaluate_dual(data_loader_test, model, device, metrics_dir, epoch=0, mode=args.task_mode, num_class=args.nb_classes, criterion=criterion, task_mode=args.task_mode, disease_list=None, return_bal_acc=args.return_bal_acc, args=args)
        if args.return_bal_acc:
            test_auc_pr, test_bal_acc = auc_pr
        wandb_dict = {f'test_{k}': v for k, v in test_stats.items()}
        wandb.log(wandb_dict)
        wandb.finish()
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_score = 0.0
    best_val_stats, test_stats = {}, {}
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch_dual(
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

        val_stats, val_auc_roc, val_auc_pr = evaluate_dual(data_loader_val, model, device, metrics_dir, epoch, mode=val_mode, num_class=args.nb_classes, criterion=criterion, task_mode=args.task_mode, disease_list=None, return_bal_acc=args.return_bal_acc, args=args)
        if args.return_bal_acc:
            val_auc_pr, val_bal_acc = val_auc_pr
        # eval score
        if args.val_metric == 'AUC':
            e_score = val_auc_roc
        elif args.val_metric == 'AUPRC':
            e_score = val_auc_pr
        elif args.val_metric in val_stats:
            e_score = val_stats[args.val_metric]
        else:
            raise ValueError(f"Unknown validation metric: {args.val_metric}")
        # select best; evaluate the test set at the best-val epoch (as in _dual.py)
        if max_score <= e_score:
            max_score = e_score
            if args.output_dir:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_add_dir=model_add_dir, mode='best')
            best_val_stats = val_stats
            test_stats, auc_roc, auc_pr = evaluate_dual(data_loader_test, model, device, metrics_dir, epoch, mode='test', num_class=args.nb_classes, criterion=criterion, task_mode=args.task_mode, disease_list=None, return_bal_acc=args.return_bal_acc, args=args)

        if log_writer is not None:
            log_writer.add_scalar('perf/val_acc1', val_stats['acc1'], epoch)
            log_writer.add_scalar('perf/val_auc', val_auc_roc, epoch)
            log_writer.add_scalar('perf/val_loss', val_stats['loss'], epoch)

        # train_one_epoch_dual returns None when the loss was non-finite (the LR was
        # just downscaled to recover); guard so logging does not deref None.
        if train_stats is not None:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        else:
            log_stats = {'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, model_add_dir, "log.txt"), mode="a") as f:
                f.write(json.dumps(log_stats) + "\n")
            wandb_dict = {"epoch": epoch}
            if train_stats is not None:
                wandb_dict.update({f'train_{k}': v for k, v in train_stats.items()})
            wandb_dict.update({f'val_{k}': v for k, v in val_stats.items()})
            wandb.log(wandb_dict, step=epoch)
    # best valid
    wandb_dict = {}
    wandb_dict.update({f'best_val_{k}': v for k, v in best_val_stats.items()})
    wandb.log(wandb_dict)
    wandb_dict = {}
    wandb_dict.update({f'test_{k}': v for k, v in test_stats.items()})
    wandb.log(wandb_dict)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if log_writer is not None and misc.is_main_process():
        log_writer.close()
        wandb.finish()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
