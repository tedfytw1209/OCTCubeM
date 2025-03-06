# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.


# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
# Revised by Zixuan Zucks Liu @University of Washington
"""
import argparse
from dataclasses import dataclass
import logging
import math
from types import SimpleNamespace
from typing import Optional, Tuple, Union

### add the LongNet path to the model.py ###
import os
import sys
this_file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_file_dir + '/../')
from model_backup.masked_modelling import mae_model, Masked_Model

import pdb
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .vision_transformer4k import vit4k_xs
from .perceiver import VisionPerceiver
from .hf_model import HFTextEncoder
from .modified_resnet import ModifiedResNet
from .timm_model import TimmModel
from .transformer import LayerNormFp32, LayerNorm, QuickGELU, Attention, VisionTransformer, TextTransformer
from .utils import to_2tuple

from .models_vit_st_flash_attn import VisionTransformer as VisionTransformer_ST
from .models_vit_st_flash_attn_nodrop import VisionTransformer as VisionTransformer_ST_nodrop
from .models_vit_3dhead_flash_attn import VisionTransformerWith3DPoolingHead as VisionTransformer_3DHead
from .models_vit_flash_attn import VisionTransformer as VisionTransformer_Flash
from .models_vit_flash_attn_2mod import VisionTransformer as VisionTransformer_Flash_2mod
from .pos_embed import interpolate_pos_embed, interpolate_temporal_pos_embed
from functools import partial


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    global_average_pool: bool = False  # whether to global average pool the last embedding layer, instead of using CLS token (https://arxiv.org/abs/2205.01580)
    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    HIPT: str = None  # HIPT model name
    model_name: str = None  # HIPT model name
    vit_model_name: str = None  # Vision Transformer flash attn model name
    pretrain_obj: str = None
    model_arch: str = None
    model_ckpt: str = None
    num_latents: int = 512,
    num_image_channels: int = 512,
    num_latent_channels: int = 512,
    vision_max_length: int = 200
    drop_path_rate: float = 0.  # drop path rate for stochastic depth (https://arxiv.org/abs/1603.09382)
    num_heads: int = 24
    t_patch_size: int = 3
    in_chans: int = 3
    layer_decay: float = 0.6
    weight_decay: float = 0.05
    dropout: float = 0.1
    attn_drop_rate: float = 0.0
    drop_rate: float = 0.0
    smaller_temporal_crop: str = 'interp'
    norm_layer_eps: float = 1e-6
    use_flash_attn: bool = False
    num_frames: int = -1
    global_pool: bool = False

    # vision as text (will not be used for vision)
    hf_model_name: str = None
    load_non_flash_attn_to_flash_attn: bool = False





@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    hf_model_name: str = None
    hf_tokenizer_name: str = None
    hf_model_pretrained: bool = True
    proj: str = 'mlp'
    pooler_type: str = 'mean_pooler'


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size
        )
        act_layer = nn.GELU  # so that text transformer doesn't use QuickGELU w/ timm models
    elif vision_cfg.HIPT:
        visual = vit4k_xs(patch_size=vision_cfg.patch_size, drop_path_rate=vision_cfg.drop_path_rate)
        visual.image_size = vision_cfg.image_size
    elif vision_cfg.model_name and 'longnet' in vision_cfg.model_name:
        # make the longnet arguments
        longnet_args = SimpleNamespace()

        # add model configs
        if vision_cfg.pretrain_obj:
            longnet_args.objective = vision_cfg.pretrain_obj
        else:
            longnet_args.objective = 'MAE'
        longnet_args.arch = vision_cfg.model_arch
        longnet_args.num_image_channels = vision_cfg.num_image_channels
        longnet_args.num_latent_channels = vision_cfg.num_latent_channels

        # make the model
        visual = mae_model(longnet_args)
        visual.image_size = vision_cfg.image_size

        # load the model checkpoints
        try:
            if vision_cfg.model_ckpt:
                state_dict = torch.load(vision_cfg.model_ckpt, map_location='cpu')['model']
                state_dict = {k.replace('module.', ""): v for k, v in state_dict.items()}
                state_dict = {k.replace('backbone.', ""): v for k, v in state_dict.items()}
                missing_keys, unexpected_keys = visual.load_state_dict(state_dict, strict=False)
                if len(missing_keys) > 0:
                    for k in missing_keys:
                        print('Miss ', k)
                print("Done!")
            else:
                print('Warnings! Randomizing the model!')
        except:
            print('Warnings! No such checkpoints!')
    elif vision_cfg.model_name and 'perceiver' in vision_cfg.model_name:
        visual = VisionPerceiver(num_latents=vision_cfg.num_latents,
                                num_image_channels=vision_cfg.num_image_channels,
                                num_latent_channels=vision_cfg.num_latent_channels)
        visual.image_size = vision_cfg.image_size
    elif vision_cfg.model_name and 'ViT_ST' == vision_cfg.model_name:
        visual = VisionTransformer_ST(
            out_dim=embed_dim,
            num_frames=vision_cfg.num_frames,
            image_size=vision_cfg.image_size,
            depth=vision_cfg.layers,
            embed_dim=vision_cfg.width,
            patch_size=vision_cfg.patch_size,
            num_heads=vision_cfg.num_heads,
            t_patch_size=vision_cfg.t_patch_size,
            in_chans=vision_cfg.in_chans,
            mlp_ratio=vision_cfg.mlp_ratio,
            norm_layer=partial(nn.LayerNorm, eps=vision_cfg.norm_layer_eps),
            # layer_decay=vision_cfg.layer_decay,
            # weight_decay=vision_cfg.weight_decay,
            drop_path_rate=vision_cfg.drop_path_rate,
            use_flash_attn=True,
            dropout=vision_cfg.dropout,
            attn_drop_rate=vision_cfg.attn_drop_rate,
            drop_rate=vision_cfg.drop_rate,
            sep_pos_embed=True,
            global_pool=True,
            cls_embed=True,
            no_qkv_bias=False,
            qk_scale=None,
        )
        try:
            if vision_cfg.model_ckpt:
                checkpoint = torch.load(vision_cfg.model_ckpt, map_location='cpu')
                checkpoint_model = checkpoint['model']
                state_dict = visual.state_dict()

                for k in ['head.weight', 'head.bias']:
                    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        print(k, checkpoint_model[k].shape, state_dict[k].shape)
                        del checkpoint_model[k]
                interpolate_pos_embed(visual, checkpoint_model)
                interpolate_temporal_pos_embed(visual, checkpoint_model, smaller_interpolate_type=vision_cfg.smaller_temporal_crop)

                missing_keys, unexpected_keys = visual.load_state_dict(checkpoint_model, strict=False)
                print(unexpected_keys)
                if len(missing_keys) > 0:
                    for k in missing_keys:
                        print('Miss ', k)
                print("Done!")

            else:

                print('Warnings! Randomizing the model!')
        except:
            raise ValueError('Warnings! No such checkpoints!')
            print('Warnings! No such checkpoints!')
    elif vision_cfg.model_name and 'ViT_ST_nodrop' in vision_cfg.model_name:
        print('Using ViT_ST_nodrop model!')

        visual = VisionTransformer_ST_nodrop(
            out_dim=embed_dim,
            num_frames=vision_cfg.num_frames,
            image_size=vision_cfg.image_size,
            depth=vision_cfg.layers,
            embed_dim=vision_cfg.width,
            patch_size=vision_cfg.patch_size,
            num_heads=vision_cfg.num_heads,
            t_patch_size=vision_cfg.t_patch_size,
            in_chans=vision_cfg.in_chans,
            mlp_ratio=vision_cfg.mlp_ratio,
            norm_layer=partial(nn.LayerNorm, eps=vision_cfg.norm_layer_eps),
            # layer_decay=vision_cfg.layer_decay,
            # weight_decay=vision_cfg.weight_decay,
            drop_path_rate=vision_cfg.drop_path_rate,
            use_flash_attn=True,
            # dropout=vision_cfg.dropout,
            attn_drop_rate=vision_cfg.attn_drop_rate,
            drop_rate=vision_cfg.drop_rate,
            sep_pos_embed=True,
            global_pool=vision_cfg.global_pool,
            cls_embed=True,
            no_qkv_bias=False,
            qk_scale=None,
        )
        try:
            if vision_cfg.model_ckpt:
                checkpoint = torch.load(vision_cfg.model_ckpt, map_location='cpu')
                checkpoint_model = checkpoint['model']
                state_dict = visual.state_dict()

                for k in ['head.weight', 'head.bias']:
                    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        print(k, checkpoint_model[k].shape, state_dict[k].shape)
                        del checkpoint_model[k]
                interpolate_pos_embed(visual, checkpoint_model)
                interpolate_temporal_pos_embed(visual, checkpoint_model, smaller_interpolate_type=vision_cfg.smaller_temporal_crop)

                missing_keys, unexpected_keys = visual.load_state_dict(checkpoint_model, strict=False)
                print(unexpected_keys)
                if len(missing_keys) > 0:
                    for k in missing_keys:
                        print('Miss ', k)
                print("Done!")
            else:
                print('Warnings! Randomizing the model!')
        except:
            raise ValueError('Warnings! No such checkpoints!')
            print('Warnings! No such checkpoints!')
    elif vision_cfg.model_name and 'ViT_3Dhead' in vision_cfg.model_name:
        print('Using 3D head model')
        visual = VisionTransformer_3DHead(
            out_dim=embed_dim,
            image_size=vision_cfg.image_size,
            depth=vision_cfg.layers,
            embed_dim=vision_cfg.width,
            patch_size=vision_cfg.patch_size,
            num_heads=vision_cfg.num_heads,
            in_chans=vision_cfg.in_chans,
            mlp_ratio=vision_cfg.mlp_ratio,
            norm_layer=partial(nn.LayerNorm, eps=vision_cfg.norm_layer_eps),
            # layer_decay=vision_cfg.layer_decay,
            # weight_decay=vision_cfg.weight_decay,
            drop_path_rate=vision_cfg.drop_path_rate,
            use_flash_attn=True,
            dropout=vision_cfg.dropout,
            attn_drop_rate=vision_cfg.attn_drop_rate,
            drop_rate=vision_cfg.drop_rate,
            # sep_pos_embed=True,
            global_pool=True,
            cls_embed=True,
            no_qkv_bias=False,
            qk_scale=None,
        )
        try:
            if vision_cfg.model_ckpt:

                checkpoint = torch.load(vision_cfg.model_ckpt, map_location='cpu')
                checkpoint_model = checkpoint['model']
                state_dict = visual.state_dict()

                for k in ['head.weight', 'head.bias']:
                    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        print(k, checkpoint_model[k].shape, state_dict[k].shape)
                        del checkpoint_model[k]
                interpolate_pos_embed(visual, checkpoint_model)

                if vision_cfg.load_non_flash_attn_to_flash_attn:
                    missing_keys, unexpected_keys = visual.load_state_dict_to_backbone(checkpoint_model)
                else:
                    missing_keys, unexpected_keys = visual.load_state_dict(checkpoint_model, strict=False)

                print(unexpected_keys)
                if len(missing_keys) > 0:
                    for k in missing_keys:
                        print('Miss ', k)
                print("Done!")

            else:

                print('Warnings! Randomizing the model!')
        except:
            raise ValueError('Warnings! No such checkpoints!')
            print('Warnings! No such checkpoints!')

    elif vision_cfg.model_name and 'ViT_2Dhead' in vision_cfg.model_name:
        print('Using 2D original retFound model')
        visual = VisionTransformer_Flash(
            out_dim=embed_dim,
            image_size=vision_cfg.image_size,
            depth=vision_cfg.layers,
            embed_dim=vision_cfg.width,
            patch_size=vision_cfg.patch_size,
            num_heads=vision_cfg.num_heads,
            in_chans=vision_cfg.in_chans,
            mlp_ratio=vision_cfg.mlp_ratio,
            norm_layer=partial(nn.LayerNorm, eps=vision_cfg.norm_layer_eps),
            # layer_decay=vision_cfg.layer_decay,
            # weight_decay=vision_cfg.weight_decay,
            drop_path_rate=vision_cfg.drop_path_rate,
            use_flash_attn=True,
            dropout=vision_cfg.dropout,
            attn_drop_rate=vision_cfg.attn_drop_rate,
            drop_rate=vision_cfg.drop_rate,
            # sep_pos_embed=True,
            global_pool=True,
            cls_embed=True,
            no_qkv_bias=False,
            qk_scale=None,
        )

        try:
            if vision_cfg.model_ckpt:
                checkpoint = torch.load(vision_cfg.model_ckpt, map_location='cpu')
                checkpoint_model = checkpoint['model']
                state_dict = visual.state_dict()

                for k in ['head.weight', 'head.bias']:
                    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        print(k, checkpoint_model[k].shape, state_dict[k].shape)
                        del checkpoint_model[k]
                interpolate_pos_embed(visual, checkpoint_model)

                if vision_cfg.load_non_flash_attn_to_flash_attn:
                    missing_keys, unexpected_keys = visual.load_state_dict_to_backbone(checkpoint_model)
                else:
                    missing_keys, unexpected_keys = visual.load_state_dict(checkpoint_model, strict=False)

                print(unexpected_keys)
                if len(missing_keys) > 0:
                    for k in missing_keys:
                        print('Miss ', k)
                print("Done!")

            else:

                print('Warnings! Randomizing the model!')
        except:
            raise ValueError('Warnings! No such checkpoints!')
            print('Warnings! No such checkpoints!')

    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            global_average_pool=vision_cfg.global_average_pool,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    return visual


def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg or CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
        # is_vision: bool = False
):
    # print(text_cfg)
    if isinstance(text_cfg, dict):
        if 'vit_model_name' in text_cfg:
            text_cfg = CLIPVisionCfg(**text_cfg)
        else:
            text_cfg = CLIPTextCfg(**text_cfg)

    if text_cfg.hf_model_name:
        text = HFTextEncoder(
            text_cfg.hf_model_name,
            output_dim=embed_dim,
            proj=text_cfg.proj,
            pooler_type=text_cfg.pooler_type,
            pretrained=text_cfg.hf_model_pretrained
        )
    elif text_cfg.vit_model_name and 'ViT_flash_attn' in text_cfg.vit_model_name and 'mod' not in text_cfg.vit_model_name:
        text = VisionTransformer_Flash(
            out_dim=embed_dim,
            image_size=text_cfg.image_size,
            depth=text_cfg.layers,
            embed_dim=text_cfg.width,
            patch_size=text_cfg.patch_size,
            num_heads=text_cfg.num_heads,
            mlp_ratio=text_cfg.mlp_ratio,
            in_chans=text_cfg.in_chans,
            norm_layer=partial(nn.LayerNorm, eps=text_cfg.norm_layer_eps),
            layer_decay=text_cfg.layer_decay,
            weight_decay=text_cfg.weight_decay,
            drop_path_rate=text_cfg.drop_path_rate,
            use_flash_attn=True,
            dropout=text_cfg.dropout,
            attn_drop_rate=text_cfg.attn_drop_rate,
            drop_rate=text_cfg.drop_rate,
            # sep_pos_embed=True,
            global_pool=text_cfg.global_pool,
            cls_embed=True,
            no_qkv_bias=False,
            qk_scale=None,
        )
        try:
            if text_cfg.model_ckpt:
                checkpoint = torch.load(text_cfg.model_ckpt, map_location='cpu')
                checkpoint_model = checkpoint['model']
                state_dict = text.state_dict()
                for k in ['head.weight', 'head.bias']:
                    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        print(k, checkpoint_model[k].shape, state_dict[k].shape)
                        del checkpoint_model[k]
                interpolate_pos_embed(text, checkpoint_model)
                if text_cfg.load_non_flash_attn_to_flash_attn:
                    missing_keys, unexpected_keys = text.load_state_dict_to_backbone(checkpoint_model)
                else:
                    missing_keys, unexpected_keys = text.load_state_dict(checkpoint_model, strict=False)
                if len(missing_keys) > 0:
                    for k in missing_keys:
                        print('Miss ', k)
                print("Done!")
            else:
                print('Warnings! Randomizing the model!')
        except:
            raise ValueError('Warnings! No such checkpoints!')
            print('Warnings! No such checkpoints!')
    elif text_cfg.vit_model_name and 'ViT_flash_attn_2mod' in text_cfg.vit_model_name:
        text = VisionTransformer_Flash_2mod(
            out_dim=embed_dim,
            image_size=text_cfg.image_size,
            depth=text_cfg.layers,
            embed_dim=text_cfg.width,
            patch_size=text_cfg.patch_size,
            num_heads=text_cfg.num_heads,
            mlp_ratio=text_cfg.mlp_ratio,
            in_chans=text_cfg.in_chans,
            norm_layer=partial(nn.LayerNorm, eps=text_cfg.norm_layer_eps),
            layer_decay=text_cfg.layer_decay,
            weight_decay=text_cfg.weight_decay,
            drop_path_rate=text_cfg.drop_path_rate,
            use_flash_attn=True,
            dropout=text_cfg.dropout,
            attn_drop_rate=text_cfg.attn_drop_rate,
            drop_rate=text_cfg.drop_rate,
            # sep_pos_embed=True,
            global_pool=text_cfg.global_pool,
            cls_embed=True,
            no_qkv_bias=False,
            qk_scale=None,
            num_mod_heads=2,
        )
        try:
            if text_cfg.model_ckpt:
                checkpoint = torch.load(text_cfg.model_ckpt, map_location='cpu')
                checkpoint_model = checkpoint['model']
                state_dict = text.state_dict()
                for k in ['head.weight', 'head.bias']:
                    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        print(k, checkpoint_model[k].shape, state_dict[k].shape)
                        del checkpoint_model[k]
                interpolate_pos_embed(text, checkpoint_model)
                if text_cfg.load_non_flash_attn_to_flash_attn:
                    missing_keys, unexpected_keys = text.load_state_dict_to_backbone(checkpoint_model)
                    print('going in correct path')
                else:
                    missing_keys, unexpected_keys = text.load_state_dict(checkpoint_model, strict=False)
                if len(missing_keys) > 0:
                    for k in missing_keys:
                        print('Miss ', k)
                print("Done!")
            else:
                print('Warnings! Randomizing the model!')
        except:
            raise ValueError('Warnings! No such checkpoints!')
            print('Warnings! No such checkpoints!')

    else:
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm

        text = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return text


class CLIP(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)

        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, img_coords=None, img_pad_masks=None, normalize: bool = False):
        features = self.visual(image, img_coords, img_pad_masks)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def forward(self, image, text, img_coords=None, img_pad_masks=None):
        image_features = self.encode_image(image, img_coords, img_pad_masks, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        return image_features, text_features, self.logit_scale.exp()


class CustomTextCLIP(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg or CLIPVisionCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    # def encode_image(self, image, img_coords=None, img_pad_masks=None, normalize: bool = False):
    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(self, image, text, single_modality=None):
        if single_modality is not None:
            assert single_modality in ['image', 'text'], f"single_modality should be either 'image' or 'text', got {single_modality}"
            if single_modality == 'image':
                image_features = self.encode_image(image, normalize=True)
                return image_features, None, self.logit_scale.exp()
            else:
                text_features = self.encode_text(text, normalize=True)
                return None, text_features, self.logit_scale.exp()

        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        return image_features, text_features, self.logit_scale.exp()


class CustomTextCLIP3Mod(CustomTextCLIP):
    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(embed_dim, vision_cfg, text_cfg, quick_gelu, cast_dtype)
        self.logit_scale1 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale2 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_text(self, text, normalize: bool = False, modality=0):
        features = self.text(text, modality=modality)
        return F.normalize(features, dim=-1) if normalize else features


    def forward(self, image, text1, text2, single_modality=None):
        assert single_modality in ['image', 'text1', 'text2', None], f"single_modality should be either 'image', 'text1', 'text2' or None, got {single_modality}"
        # image_weight, text1_weight, text2_weight = weight_modality
        if single_modality is not None:
            if single_modality == 'image':
                image_features = self.encode_image(image, normalize=True)
                return image_features, None, None, self.logit_scale.exp(), self.logit_scale1.exp(), self.logit_scale2.exp()
            elif single_modality == 'text1':
                text1_features = self.encode_text(text1, normalize=True, modality=0)
                return None, text1_features, None, self.logit_scale.exp(), self.logit_scale1.exp(), self.logit_scale2.exp()
            elif single_modality == 'text2':
                text2_features = self.encode_text(text2, normalize=True, modality=1)
                return None, None, text2_features, self.logit_scale.exp(), self.logit_scale1.exp(), self.logit_scale2.exp
        image_features = self.encode_image(image, normalize=True)
        text1_features = self.encode_text(text1, normalize=True, modality=0)
        text2_features = self.encode_text(text2, normalize=True, modality=1)

        return image_features, text1_features, text2_features, self.logit_scale.exp(), self.logit_scale1.exp(), self.logit_scale2.exp()


class ClassificationHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, initialization=True):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.input_norm = nn.LayerNorm(input_dim)
        if initialization:
            torch.nn.init.normal_(self.fc1.weight, std=0.02)
            torch.nn.init.normal_(self.fc1.weight, std=0.02)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x

class CustomTextCLIPClassification(CustomTextCLIP):
    def __init__(
            self,
            embed_dim: int,
            vision_cfg,
            text_cfg,
            # hidden_dim: int,
            num_classes: int,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None
    ):
        super(CustomTextCLIPClassification, self).__init__(
            embed_dim=embed_dim,
            vision_cfg=vision_cfg,
            text_cfg=text_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype
        )
        self.classification_head = ClassificationHead(2 * embed_dim, hidden_dim=embed_dim, num_classes=num_classes)

    def forward(self, image, text, single_modality=None):
        image_features, text_features, logit_scale = super().forward(image, text, single_modality)
        if single_modality is not None:
            dummy_features = torch.zeros_like(image_features) if single_modality == 'image' else torch.zeros_like(text_features)
            dummy_features = dummy_features.to(image_features.device) if single_modality == 'image' else dummy_features.to(text_features.device)
            concatenated_features = torch.cat((image_features, dummy_features), dim=-1) if single_modality == 'image' else torch.cat((dummy_features, text_features), dim=-1)
        else:
            concatenated_features = torch.cat((image_features, text_features), dim=-1)
        return self.classification_head(concatenated_features), logit_scale


class CustomTextCLIP3ModClassification(CustomTextCLIP3Mod):
    def __init__(
            self,
            embed_dim: int,
            vision_cfg,
            text_cfg,
            # hidden_dim: int,
            num_classes: int,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None
    ):
        super(CustomTextCLIP3ModClassification, self).__init__(
            embed_dim=embed_dim,
            vision_cfg=vision_cfg,
            text_cfg=text_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype
        )
        self.classification_head = ClassificationHead(3 * embed_dim, hidden_dim=embed_dim, num_classes=num_classes)

    def forward(self, image, text1, text2, single_modality=None):
        image_features, text1_features, text2_features, logit_scale, logit_scale1, logit_scale2 = super().forward(image, text1, text2, single_modality)
        if single_modality is not None:
            # dummy_features = torch.zeros_like(image_features) if single_modality == 'image' else torch.zeros_like(text1_features) if single_modality == 'text1' else torch.zeros_like(text2_features)
            if single_modality == 'image':
                dummy_features = torch.zeros_like(image_features)
                concatenated_features = torch.cat([image_features, dummy_features, dummy_features], dim=-1)
            elif single_modality == 'text1':
                dummy_features = torch.zeros_like(text1_features)
                concatenated_features = torch.cat([dummy_features, text1_features, dummy_features], dim=-1)
            else:
                assert single_modality == 'text2'
                dummy_features = torch.zeros_like(text2_features)
                concatenated_features = torch.cat([dummy_features, dummy_features, text2_features], dim=-1)

        else:
            concatenated_features = torch.cat((image_features, text1_features, text2_features), dim=-1)
        return self.classification_head(concatenated_features), logit_scale, logit_scale1, logit_scale2


class CustomTextCLIP3Mod_gradcam(CustomTextCLIP):
    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(embed_dim, vision_cfg, text_cfg, quick_gelu, cast_dtype)
        self.logit_scale1 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale2 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.text2 = None

    def encode_text(self, text, normalize: bool = False, modality=0):
        if modality == 1 and self.text2 is not None:
            features = self.text2(text, modality=modality)
        else:
            features = self.text(text, modality=modality)
        return F.normalize(features, dim=-1) if normalize else features

    def _copy_text_tower(self, device):
        self.text2 = copy.deepcopy(self.text)
        self.text2.to(device)


    def forward(self, image, text1, text2, single_modality=None):
        assert single_modality in ['image', 'text1', 'text2', None], f"single_modality should be either 'image', 'text1', 'text2' or None, got {single_modality}"
        # image_weight, text1_weight, text2_weight = weight_modality
        if single_modality is not None:
            if single_modality == 'image':
                image_features = self.encode_image(image, normalize=True)
                return image_features, None, None, self.logit_scale.exp(), self.logit_scale1.exp(), self.logit_scale2.exp()
            elif single_modality == 'text1':
                text1_features = self.encode_text(text1, normalize=True, modality=0)
                return None, text1_features, None, self.logit_scale.exp(), self.logit_scale1.exp(), self.logit_scale2.exp()
            elif single_modality == 'text2':
                text2_features = self.encode_text(text2, normalize=True, modality=1)
                return None, None, text2_features, self.logit_scale.exp(), self.logit_scale1.exp(), self.logit_scale2.exp
        image_features = self.encode_image(image, normalize=True)
        text1_features = self.encode_text(text1, normalize=True, modality=0)
        text2_features = self.encode_text(text2, normalize=True, modality=1)

        return image_features, text1_features, text2_features, self.logit_scale.exp(), self.logit_scale1.exp(), self.logit_scale2.exp()

class CustomTextCLIP3ModClassification_gradcam(CustomTextCLIP3Mod_gradcam):
    def __init__(
            self,
            embed_dim: int,
            vision_cfg,
            text_cfg,
            # hidden_dim: int,
            num_classes: int,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None
    ):
        super(CustomTextCLIP3ModClassification_gradcam, self).__init__(
            embed_dim=embed_dim,
            vision_cfg=vision_cfg,
            text_cfg=text_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype
        )
        self.classification_head = ClassificationHead(3 * embed_dim, hidden_dim=embed_dim, num_classes=num_classes)

    def forward(self, batched_x, single_modality=None):
        image = batched_x['image']
        text1 = batched_x['text1']
        text2 = batched_x['text2']
        image_features, text1_features, text2_features, logit_scale, logit_scale1, logit_scale2 = super().forward(image, text1, text2, single_modality)
        if single_modality is not None:

            if single_modality == 'image':
                dummy_features = torch.zeros_like(image_features)
                concatenated_features = torch.cat([image_features, dummy_features, dummy_features], dim=-1)
            elif single_modality == 'text1':
                dummy_features = torch.zeros_like(text1_features)
                concatenated_features = torch.cat([dummy_features, text1_features, dummy_features], dim=-1)
            else:
                assert single_modality == 'text2'
                dummy_features = torch.zeros_like(text2_features)
                concatenated_features = torch.cat([dummy_features, dummy_features, text2_features], dim=-1)

        else:
            concatenated_features = torch.cat((image_features, text1_features, text2_features), dim=-1)
        return self.classification_head(concatenated_features) #, logit_scale, logit_scale1, logit_scale2

def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""

    def _convert_weights(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp  # backwards compat


# used to maintain checkpoint compatibility
def convert_to_custom_text_state_dict(state_dict: dict):
    if 'text_projection' in state_dict:
        # old format state_dict, move text tower -> .text
        new_state_dict = {}
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in (
                'text_projection',
                'positional_embedding',
                'token_embedding',
                'transformer',
                'ln_final',
            )):
                k = 'text.' + k
            new_state_dict[k] = v
        return new_state_dict
    return state_dict


def build_model_from_openai_state_dict(
        state_dict: dict,
        quick_gelu=True,
        cast_dtype=torch.float16,
):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_size = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers
    )
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,  # OpenAI models were trained with QuickGELU
        cast_dtype=cast_dtype,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)

    convert_weights_to_fp16(model)  # OpenAI state dicts are partially converted to float16
    model.load_state_dict(state_dict)
    return model.eval()


def trace_model(model, batch_size=256, device=torch.device('cpu')):
    model.eval()
    image_size = model.visual.image_size
    example_images = torch.ones((batch_size, 3, image_size, image_size), device=device)
    example_text = torch.zeros((batch_size, model.context_length), dtype=torch.int, device=device)
    model = torch.jit.trace_module(
        model,
        inputs=dict(
            forward=(example_images, example_text),
            encode_text=(example_text,),
            encode_image=(example_images,)
        ))
    model.visual.image_size = image_size
    return model


def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', seq_dim=1):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME: detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        align_corners=True,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed
