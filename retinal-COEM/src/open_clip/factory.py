# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.


# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Revised by Zixuan Zucks Liu @University of Washington

import json
import logging
import os
import pathlib
import re
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple, Union

import torch

from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .model import CLIP, CustomenfaceCLIP, convert_weights_to_lp, convert_to_custom_enface_state_dict,\
    resize_pos_embed, get_cast_dtype, CustomenfaceCLIPClassification, CustomenfaceCLIP3Mod, CustomenfaceCLIP3ModClassification, CustomenfaceCLIP3ModClassification_gradcam
from .openai import load_openai_model
from .pretrained import is_pretrained_cfg, get_pretrained_cfg, download_pretrained, list_pretrained_tags_by_model
from .transform import image_transform
from .tokenizer import HFTokenizer, tokenize


_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs

_MODEL_CKPT_PATHS = [Path(__file__).parent / f"model_ckpts/"]
_MODEL_CKPTS = {}  # directory (model_name: ckpt) of model checkpoints


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        with open(cf, 'r') as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'enface_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg
            else:
                logging.warning(f"Invalid model config: {cf}")

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}

def _rescan_model_ckpts():
    global _MODEL_CKPTS

    ckpt_ext = ('.json',)
    ckpt_files = []
    for ckpt_path in _MODEL_CKPT_PATHS:
        if ckpt_path.is_file() and ckpt_path.suffix in ckpt_ext:
            ckpt_files.append(ckpt_path)
        elif ckpt_path.is_dir():
            for ext in ckpt_ext:
                ckpt_files.extend(ckpt_path.glob(f'*{ext}'))

    for cf in ckpt_files:
        with open(cf, 'r') as f:
            model_ckpt = json.load(f)
            print(model_ckpt)
            if 'log_dir' in model_ckpt:
                _MODEL_CKPTS[cf.stem] = model_ckpt


    _MODEL_CKPTS = {k: v for k, v in sorted(_MODEL_CKPTS.items(), key=lambda x: _natural_key(x[0]))}

def get_model_ckpt(model_name):
    if model_name in _MODEL_CKPTS:
        return _MODEL_CKPTS[model_name]
    else:
        return None

def get_model_ckpt_cv_fold_path_list(expr_dir, model_name,):
    print(f"expr_dir: {expr_dir}, model_name: {model_name}")
    all_metric_dict = {}
    model_ckpt = get_model_ckpt(model_name)
    log_dir = expr_dir + model_ckpt['log_dir']
    stored_expr_name = log_dir.split('/')[-2]
    ckpt_dir = log_dir + 'checkpoints/'
    best_val_dir = ckpt_dir + 'best_val/'

    ckpt_cv_fold_path_list = sorted([str(f) for f in pathlib.Path(best_val_dir).iterdir() if f.is_dir()])
    metric_dict = {}
    for ckpt_cv_fold_path in ckpt_cv_fold_path_list:
        metric_idx = ckpt_cv_fold_path.split('.')[0].split('/')[-1]
        metric_dict[metric_idx] = ckpt_cv_fold_path

    for metric_idx, metric_path in metric_dict.items():

        cv_fold_path = sorted([str(f) for f in pathlib.Path(metric_path).iterdir()])

        metric_dict[metric_idx] = cv_fold_path

    all_metric_dict['best_val'] = metric_dict

    best_test_dir = ckpt_dir + 'best_test/'
    ckpt_cv_fold_path_list = sorted([str(f) for f in pathlib.Path(best_test_dir).iterdir() if f.is_dir()])
    metric_dict_list = []

    for ckpt_cv_fold_path in ckpt_cv_fold_path_list:
        metric_dict = {}
        metric_dict_path = sorted([str(f) for f in pathlib.Path(ckpt_cv_fold_path).iterdir()])
        for idx, metric_path in enumerate(metric_dict_path):

            metric_idx = metric_path.split('.')[0].split('/')[-1]
            cv_fold_path = sorted([str(f) for f in pathlib.Path(metric_path).iterdir()])
            metric_dict[metric_idx] = cv_fold_path
        metric_dict_list.append(metric_dict)

    all_metric_dict['best_test'] = metric_dict

    best_independent_test_dir = ckpt_dir + 'best_independent_test/'
    if not pathlib.Path(best_independent_test_dir).exists():
        return all_metric_dict
    ckpt_cv_fold_path_list = sorted([str(f) for f in pathlib.Path(best_independent_test_dir).iterdir() if f.is_dir()])
    metric_dict_list = []

    for ckpt_cv_fold_path in ckpt_cv_fold_path_list:
        metric_dict = {}
        metric_dict_path = sorted([str(f) for f in pathlib.Path(ckpt_cv_fold_path).iterdir()])
        for idx, metric_path in enumerate(metric_dict_path):

            metric_idx = metric_path.split('.')[0].split('/')[-1]
            cv_fold_path = sorted([str(f) for f in pathlib.Path(metric_path).iterdir()])
            metric_dict[metric_idx] = cv_fold_path
        metric_dict_list.append(metric_dict)

    all_metric_dict['best_independent_test'] = metric_dict_list
    return all_metric_dict

def create_loaded_cv_ckpt(args):
    model_ckpt = args.loaded_expr_name
    model_selection_type = args.model_selection_type
    test_idx = args.loaded_test_idx
    metric_idx = args.loaded_metric_idx
    model_path = get_model_ckpt(model_ckpt)

    stored_expr_name = model_path['log_dir'].split('/')[-2]
    all_metric_dict = get_model_ckpt_cv_fold_path_list(args.expr_dir, model_ckpt)
    chosen_metric_dict = all_metric_dict[model_selection_type]
    if model_selection_type == 'best_val':
        metric_idx_name = f"{model_selection_type}_r2_{metric_idx}"
    elif model_selection_type == 'best_independent_test' or model_selection_type == 'best_test':
        chosen_metric_dict = chosen_metric_dict[test_idx]
        metric_idx_name = f"r2_{metric_idx}"

    chosen_path_list = chosen_metric_dict[metric_idx_name]
    return chosen_path_list, stored_expr_name


_rescan_model_ckpts()
_rescan_model_configs()  # initial populate of model config registry
print(f"Model configs: {list(_MODEL_CONFIGS.keys())}, {len(_MODEL_CONFIGS)},")
# print whole config with indent=2
print(json.dumps(_MODEL_CONFIGS, indent=2))


def list_models():
    """ enumerate available model architectures based on config files """
    return list(_MODEL_CONFIGS.keys())


def add_model_config(path):
    """ add model config path or file and update registry """
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()


def get_model_config(model_name):
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None


def get_tokenizer(model_name):
    config = get_model_config(model_name)
    tokenizer = HFTokenizer(config['enface_cfg']['hf_tokenizer_name']) if 'hf_tokenizer_name' in config['enface_cfg'] else tokenize
    return tokenizer


def get_context_length(model_name):
    config = get_model_config(model_name)
    return config['enface_cfg'].get("context_length", 77)

def get_vision_length(model_name):
    config = get_model_config(model_name)
    return config['vision_cfg'].get("vision_max_length", 20000)


def load_state_dict(checkpoint_path: str, map_location='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(model, checkpoint_path, strict=False):
    state_dict = load_state_dict(checkpoint_path)
    # detect old format and make compatible with new format
    if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
        state_dict = convert_to_custom_enface_state_dict(state_dict)
    resize_pos_embed(state_dict, model)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys


def create_model(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_enface: bool = False,
        force_patch_dropout: Optional[float] = None,
        pretrained_image: bool = False,
        pretrained_hf: bool = True,
        cache_dir: Optional[str] = None,
        args: Optional = None,
):
    model_name = model_name.replace('/', '-')  # for callers using old naming with / in ViT names
    if isinstance(device, str):
        device = torch.device(device)

    if pretrained and os.path.basename(pretrained.lower()).startswith('openai'):
        logging.info(f'Loading pretrained {model_name} from OpenAI.')
        model = load_openai_model(
            pretrained if pretrained != "openai" else model_name,
            precision=precision,
            device=device,
            jit=jit,
            cache_dir=cache_dir,
        )
    else:
        model_cfg = get_model_config(model_name)
        if model_cfg is not None:
            logging.info(f'Creating model {model_name} with config: {model_cfg}')
        else:
            logging.error(f'Model config for {model_name} not found; available models {list_models()}.')
            raise RuntimeError(f'Model config for {model_name} not found.')
        # FIXME: Add num_frames to model_cfg
        if args is not None and args.num_frames > 0:
            model_cfg["vision_cfg"]["num_frames"] = args.num_frames
            model_cfg["vision_cfg"]["smaller_temporal_crop"] = args.smaller_temporal_crop
        if force_quick_gelu:
            # override for use of QuickGELU on non-OpenAI transformer models
            model_cfg["quick_gelu"] = True

        if force_patch_dropout is not None:
            # override the default patch dropout value
            model_cfg["vision_cfg"]["patch_dropout"] = force_patch_dropout

        # FIXME: Add tuple input_size passing to model_cfg['vision_cfg']['image_size']
        if args is not None and isinstance(args.input_size, tuple):
            model_cfg["vision_cfg"]["image_size"] = args.input_size
            print(f"Update tuple input_size: {args.input_size}", 'image size: ', model_cfg["vision_cfg"]["image_size"])


        if pretrained_image:
            if 'timm_model_name' in model_cfg.get('vision_cfg', {}):
                # pretrained weight loading for timm models set via vision_cfg
                model_cfg['vision_cfg']['timm_model_pretrained'] = True
            else:
                assert False, 'pretrained image towers currently only supported for timm models'

        cast_dtype = get_cast_dtype(precision)
        custom_enface = model_cfg.pop('custom_enface', False) or force_custom_enface or ('hf_model_name' in model_cfg.get('enface_cfg', {})) or ('vit_model_name' in model_cfg.get('enface_cfg', {}))
        if args is not None and args.load_non_flash_attn_to_flash_attn:
            model_cfg["enface_cfg"]["load_non_flash_attn_to_flash_attn"] = True

        if custom_enface:
            print("Creating CustomenfaceCLIP")
            if 'hf_model_name' in model_cfg.get('enface_cfg', {}):
                model_cfg['enface_cfg']['hf_model_pretrained'] = pretrained_hf
            if args.cls_dataset:
                if args.enable_3mod_training:
                    if args.use_gradcam:
                        model = CustomenfaceCLIP3ModClassification_gradcam(**model_cfg, cast_dtype=cast_dtype, num_classes=args.num_classes)
                    else:
                        model = CustomenfaceCLIP3ModClassification(**model_cfg, cast_dtype=cast_dtype, num_classes=args.num_classes)
                else:
                    model = CustomenfaceCLIPClassification(**model_cfg, cast_dtype=cast_dtype, num_classes=args.num_classes)
            elif args.enable_3mod_training:
                model = CustomenfaceCLIP3Mod(**model_cfg, cast_dtype=cast_dtype)

            else:
                model = CustomenfaceCLIP(**model_cfg, cast_dtype=cast_dtype)


        else:
            model = CLIP(**model_cfg, cast_dtype=cast_dtype)

        pretrained_cfg = {}
        if pretrained:
            checkpoint_path = ''
            pretrained_cfg = get_pretrained_cfg(model_name, pretrained)
            if pretrained_cfg:
                checkpoint_path = download_pretrained(pretrained_cfg, cache_dir=cache_dir)
            elif os.path.exists(pretrained):
                checkpoint_path = pretrained

            if checkpoint_path:
                print(f"checkpoint_path: {checkpoint_path}", 'coming right up')
                logging.info(f'Loading pretrained {model_name} weights ({pretrained}).')
                load_checkpoint(model, checkpoint_path)
            else:
                error_str = (
                    f'Pretrained weights ({pretrained}) not found for model {model_name}.'
                    f'Available pretrained tags ({list_pretrained_tags_by_model(model_name)}.')
                logging.warning(error_str)
                raise RuntimeError(error_str)

        model.to(device=device)

        # FIXME: Usually training precision is amp so we don't need to consider this for training but for inference
        if precision in ("fp16", "bf16"):
            convert_weights_to_lp(model, dtype=torch.bfloat16 if precision == 'bf16' else torch.float16)

        # set image / mean metadata from pretrained_cfg if available, or use default
        model.visual.image_mean = pretrained_cfg.get('mean', None) or OPENAI_DATASET_MEAN
        model.visual.image_std = pretrained_cfg.get('std', None) or OPENAI_DATASET_STD

        if jit:
            model = torch.jit.script(model)

    return model


def create_model_and_transforms(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_enface: bool = False,
        force_patch_dropout: Optional[float] = None,
        pretrained_image: bool = False,
        pretrained_hf: bool = True,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        cache_dir: Optional[str] = None,
        args: Optional = None,
):
    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_enface=force_custom_enface,
        force_patch_dropout=force_patch_dropout,
        pretrained_image=pretrained_image,
        pretrained_hf=pretrained_hf,
        cache_dir=cache_dir,
        args=args,
    )

    image_mean = image_mean or getattr(model.visual, 'image_mean', None)
    image_std = image_std or getattr(model.visual, 'image_std', None)

    preprocess_train = image_transform(
        model.visual.image_size,
        is_train=True,
        mean=image_mean,
        std=image_std
    )
    preprocess_val = image_transform(
        model.visual.image_size,
        is_train=False,
        mean=image_mean,
        std=image_std
    )

    return model, preprocess_train, preprocess_val


def create_model_from_pretrained(
        model_name: str,
        pretrained: str,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_enface: bool = False,
        return_transform: bool = True,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        cache_dir: Optional[str] = None,
):
    if not is_pretrained_cfg(model_name, pretrained) and not os.path.exists(pretrained):
        raise RuntimeError(
            f'{pretrained} is not a valid pretrained cfg or checkpoint for {model_name}.'
            f' Use open_clip.list_pretrained() to find one.')

    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_enface=force_custom_enface,
        cache_dir=cache_dir,
    )

    if not return_transform:
        return model

    image_mean = image_mean or getattr(model.visual, 'image_mean', None)
    image_std = image_std or getattr(model.visual, 'image_std', None)
    preprocess = image_transform(
        model.visual.image_size,
        is_train=False,
        mean=image_mean,
        std=image_std
    )

    return model, preprocess
