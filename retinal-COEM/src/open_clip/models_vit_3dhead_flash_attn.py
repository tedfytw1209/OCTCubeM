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

import torch
import torch.nn as nn

import timm.models.vision_transformer
from .models_vit_flash_attn import VisionTransformer as VisionTransformer2DCenterHead
from timm.layers import to_2tuple
from typing import Callable, Optional, Sequence

class VisionTransformerWith3DPoolingHead(VisionTransformer2DCenterHead):
    def __init__(self, image_size=256, out_dim=400, embed_dim=768, depth=12, patch_size=16, in_chans=3, global_pool=False, use_flash_attn=True, num_heads=12, mlp_ratio=4.0, no_qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, dropout=0.5, cls_embed=True, **kwargs):

        super(VisionTransformerWith3DPoolingHead, self).__init__(image_size=image_size, out_dim=out_dim, embed_dim=embed_dim, depth=depth, patch_size=patch_size, in_chans=in_chans, num_heads=num_heads, mlp_ratio=mlp_ratio, no_qkv_bias=no_qkv_bias, qk_scale=qk_scale, drop_date=drop_rate, attn_drop_rate=attn_drop_rate, norm_layer=norm_layer, drop_path_rate=drop_path_rate,  cls_embed=cls_embed, dropout=dropout, use_flash_attn=use_flash_attn, global_pool=global_pool, **kwargs)

        # Assuming the same configuration for norm layer and embedding dimension as in the base Vision Transformer
        # Fully connected layer for the aggregated CLS tokens
        self.fc_aggregate_cls = nn.Linear(embed_dim, embed_dim)

        # Normalization layer after aggregation
        self.aggregate_cls_norm = norm_layer(embed_dim)


    def forward_features(self, x: torch.Tensor, hidden_states=False) -> torch.Tensor:

        B, N, C, H, W = x.shape

        x = x.view(B * N, C, H, W)

        # Process the entire batch of slices through the transformer
        x = super().forward_features(x, hidden_states=hidden_states)

        # Reshape back to separate the batch and slices dimensions, keeping the token dimension
        x = x.view(B, N, -1)

        # 3D average pooling
        x = x.mean(dim=1)  # Mean pooling over the slices dimension

        # Pass through a fully connected layer and normalize

        x = self.fc_aggregate_cls(x)
        x = self.aggregate_cls_norm(x)


        return x

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
                [
                    self.fc_aggregate_cls,
                    self.aggregate_cls_norm,
                    self.head,
                ]

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



def flash_attn_vit_large_patch16_3DSliceHead(**kwargs):
    model = VisionTransformerWith3DPoolingHead(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

