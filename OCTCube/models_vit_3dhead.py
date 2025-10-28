# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Revised by Zixuan Zucks Liu @University of Washington

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.vision_transformer
# from models_vit import VisionTransformer as VisionTransformer2DCenterHead
import sys

try:
    # Case 1: Running from OCTCubeM/
    from OCTCube.models_vit import VisionTransformer as VisionTransformer2DCenterHead
except ModuleNotFoundError:
    try:
        # Case 2: Running from OCTCube/
        from .models_vit import VisionTransformer as VisionTransformer2DCenterHead
    except ImportError:
        # Case 3: Running as a standalone script
        sys.path.append('../')  # Add OCTCubeM/ to path
        from OCTCube.models_vit import VisionTransformer as VisionTransformer2DCenterHead




class VisionTransformerWith3DPoolingHead(VisionTransformer2DCenterHead):
    def __init__(self, global_pool=True, **kwargs):
        super(VisionTransformerWith3DPoolingHead, self).__init__(global_pool, **kwargs)
        self.global_pool = global_pool

        # Assuming the same configuration for norm layer and embedding dimension as in the base Vision Transformer
        embed_dim = kwargs['embed_dim']

        # Fully connected layer for the aggregated CLS tokens
        self.fc_aggregate_cls = nn.Linear(embed_dim, embed_dim)

        # Normalization layer after aggregation
        self.aggregate_cls_norm = kwargs['norm_layer'](embed_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = x.shape
        cls_tokens = []
        x = x.view(B * N, C, H, W)

        # Process the entire batch of slices through the transformer
        x = super().forward_features(x)

        # Reshape back to separate the batch and slices dimensions, keeping the token dimension
        x = x.view(B, N, -1)

        # 3D average pooling
        x = x.mean(dim=1)  # Mean pooling over the slices dimension

        # Pass through a fully connected layer and normalize
        x = self.fc_aggregate_cls(x)
        x = self.aggregate_cls_norm(x)


        return x


def vit_large_patch16_3DSliceHead(**kwargs):
    model = VisionTransformerWith3DPoolingHead(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

