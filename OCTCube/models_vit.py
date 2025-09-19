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

import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

class DualViTClassifier(nn.Module):
    def __init__(self, vit_model_1, vit_model_2, num_classes):
        """
        初始化 DualViTClassifier 模型。

        Args:
            vit_model_1: 第一个 ViT
            vit_model_2: 第二个 ViT
            num_classes (int): 分类的类别数。
        """
        super(DualViTClassifier, self).__init__()
        self.num_classes = num_classes
        self.vit_model_1 = vit_model_1
        self.vit_model_2 = vit_model_2

    def forward(self, input_1, input_2):
        """
        前向传播。

        Args:
            input_1 (torch.Tensor): 输入到第一个 ViT 的数据 (batch_size, 3, H, W)。
            input_2 (torch.Tensor): 输入到第二个 ViT 的数据 (batch_size, 3, D, H, W)。

        Returns:
            torch.Tensor: 分类结果 (batch_size, num_classes)。
        """
        # 第一个 ViT 提取特征 (batch_size, num_classes)
        features_1 = self.vit_model_1(input_1)
        # 第二个 ViT 提取特征 (batch_size, num_classes)
        features_2 = self.vit_model_2(input_2)

        # output mean (batch_size, num_classes)
        combined_features = (features_1 + features_2) / 2

        return combined_features

    def no_weight_decay(self):
        # 不进行权重衰减的参数
        return self.vit_model_1.no_weight_decay() + self.vit_model_2.no_weight_decay()

def DualViT(**kwargs):
    model = DualViTClassifier(**kwargs)
    return model

def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

