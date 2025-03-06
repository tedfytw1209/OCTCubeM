# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.


# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields

# Revised by Zixuan Zucks Liu @University of Washington
# --------------------------------------------------------

from functools import partial
from typing import Callable, Optional, Sequence
import re
import torch
import torch.nn as nn

from einops import rearrange
from collections import OrderedDict

import timm.models.vision_transformer
from flash_attn.models.vit import create_block
from timm.layers import to_2tuple

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding, inherited from timm 0.3.2
    """
    def __init__(self, image_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        image_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.image_size[0] and W == self.image_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, image_size=256, out_dim=400, embed_dim=1024, depth=24, patch_size=16, in_chans=3, global_pool=True, use_flash_attn=True, num_heads=16, mlp_ratio=4.0, no_qkv_bias=False,qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, dropout=0.5, cls_embed=True, **kwargs):
        super().__init__()
        self.image_size = image_size
        self.use_flash_attn = use_flash_attn
        self.global_pool = global_pool
        print("Using global_pool", global_pool)
        if self.use_flash_attn:
            self.patch_embed = PatchEmbed(
                image_size, patch_size, in_chans, embed_dim
            )
            self.out_dim = out_dim
            # self.num_classes = num_classes
            self.cls_embed = cls_embed
            self.embed_dim = embed_dim
            self.depth = depth
            self.pos_drop = nn.Dropout(p=drop_rate)
            if self.cls_embed:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
                self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
            else:
                # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
                self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
            # print(f"num_patches: {self.patch_embed.num_patches}")

            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, depth)
            ]  # stochastic depth decay rule

            self.blocks = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    not no_qkv_bias,
                    drop_rate,
                    attn_drop_rate,
                    drop_path1=dpr[i - 1] if i > 0 else 0.0,
                    drop_path2=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=nn.GELU,
                    use_flash_attn=use_flash_attn,
                    fused_bias_fc=False,
                    fused_mlp=False,
                    fused_dropout_add_ln=False,
                    layer_idx=i,
                    n_layer=depth,
                    last_layer_subset=False,
                )
                for i in range(depth)
            ])


        else:
            num_classes = out_dim
            super(VisionTransformer, self).__init__(image_size=image_size, num_classes=num_classes, embed_dim=embed_dim, depth=depth, patch_size=patch_size, in_chans=in_chans, num_heads=num_heads, mlp_ratio=4.0, no_qkv_bias=False, qk_scale=None, drop_date=drop_rate, attn_drop_rate=attn_drop_rate, norm_layer=norm_layer, drop_path_rate=drop_path_rate, drop_rate=drop_rate, cls_embed=cls_embed,
            **kwargs)


        self.norm = norm_layer(embed_dim)
        if self.global_pool:
            print("Using global_pool")
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        # --------------------------------------------------------------------------

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, out_dim)

        torch.nn.init.normal_(self.head.weight, std=0.02)


    def forward_features(self, x, hidden_states=False):
        # embed patches
        B = x.shape[0]
        x = self.patch_embed(x)

        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        # apply Transformer blocks
        hidden_states_list = []
        if self.use_flash_attn:
            residual = None
            for blk in self.blocks:
                x, residual = blk(x, residual)
                hidden_states_list.append(x)
        else:
            for blk in self.blocks:
                x = blk(x)
                hidden_states_list.append(x)


        if hidden_states:
            return hidden_states_list

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            x = self.fc_norm(x)
        else:
            x = self.norm(x)
            x = x[:, 0]

        return x

    def forward(self, x, hidden_states=False):
        x = self.forward_features(x, hidden_states=hidden_states)
        # classifier
        x = self.head(x)

        return x

    def load_state_dict_to_backbone(self, state_dict, strict=False, filter_keys=[]):
        if "patch_embed.proj.weight" in state_dict:
            patch_embed_weight = state_dict["patch_embed.proj.weight"]

            print("Loading patch_embed.proj.weight", patch_embed_weight.shape)
        else:
            print("Skip loading patch_embed.proj.weight")

        def key_mapping_attn(key):
            key = re.sub(r"blocks.(\d+).attn.proj.", r"blocks.\1.mixer.out_proj.", key)
            return key

        state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())
        n_layer = len(self.blocks)

        # Convert from Wqkv to Wq and Wkv for cross attention (last layer)
        for i in range(n_layer):

            state_dict[f"blocks.{i}.mixer.Wqkv.weight"] = state_dict.pop(f"blocks.{i}.attn.qkv.weight")
            state_dict[f"blocks.{i}.mixer.Wqkv.bias"] = state_dict.pop(f"blocks.{i}.attn.qkv.bias")

        # filter out pos_embed and patch_embed
        state_dict = {k: v for k, v in state_dict.items() if not any([f in k for f in filter_keys])}
        return super().load_state_dict(state_dict, strict=strict)

    #FIXME: add grad_checkpointing
    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.blocks.grad_checkpointing = enable

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        for param in self.parameters():
            param.requires_grad = False

        if unlocked_groups != 0:
            groups = [
                [
                    self.patch_embed,
                    self.cls_token if hasattr(self, "cls_token") else None,
                    self.pos_embed,
                ],
                *self.blocks[:-1],
                [
                    self.blocks[-1],
                    self.fc_norm if hasattr(self, "fc_norm") else self.norm,
                ],
                self.head,
            ]
            print(f"Unlocking {unlocked_groups} groups, len(groups)={len(groups)}")
            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, torch.nn.Parameter):
                        x.requires_grad = True
                    else:
                        for p in x.parameters():
                            p.requires_grad = True

            _unlock(groups[-unlocked_groups:])

def flash_attn_vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

