# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Revised by Zixuan Zucks Liu @University of Washington

import torch
import os
from torch import nn
from transformers import AutoModelForImageClassification as amfic

from vit_pytorch.vit import ViT
from einops.layers.torch import Rearrange


class SLIViT(ViT):
    def __init__(self, *, feature_extractor, vit_dim, vit_depth, heads, mlp_dim,
                 num_of_patches, dropout=0., emb_dropout=0., patch_height=768,
                 patch_width=64, rnd_pos_emb=False, num_classes=1, dim_head=64):

        super().__init__(image_size=(patch_height * num_of_patches, patch_width),
                         patch_size=(patch_height, patch_width),
                         num_classes=num_classes, dim=vit_dim, depth=vit_depth,
                         heads=heads, mlp_dim=mlp_dim, channels=1,  # Adjust if necessary
                         dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout)

        # SLIViT-specific attributes
        self.feature_extractor = feature_extractor  # Initialize the feature_extractor
        self.num_patches = num_of_patches

        # Override random positional embedding initialization (by default)
        if not rnd_pos_emb:
            self.pos_embedding = nn.Parameter(
                torch.arange(self.num_patches + 1).repeat(vit_dim, 1).t().unsqueeze(0).float()
            )

        # Override the patch embedding layer to handle feature-map patching (rather than the standard image patching)
        self.to_patch_embedding[0] = Rearrange('b c (h p1) (w p2) -> b (c h w) (p1 p2)',
                                               p1=patch_height, p2=patch_width)


    def forward(self, x):
        x = self.feature_extractor(x).last_hidden_state
        x = x.reshape((x.shape[0], self.num_patches, 768, 64))


        return super().forward(x)


class ConvNext(nn.Module):
    def __init__(self, model):
        super(ConvNext, self).__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)[0]
        return x


class CustomHuggingFaceModel(nn.Module):
    def __init__(self, hugging_face_model):
        super().__init__()
        self.model = hugging_face_model

    def forward(self, x):
        # Get logits from the Hugging Face model
        return self.model(x).logits


def get_feature_extractor(num_labels, pretrained_weights=''):
    convnext = amfic.from_pretrained("facebook/convnext-tiny-224", return_dict=False,
                                     num_labels=num_labels, ignore_mismatched_sizes=True)

    # weights from the Hugging Face model cannot be correctly loaded into the fastai model due to mismatched layers
    # and requires wrapping in a custom model (that only returns the logits)
    chf = CustomHuggingFaceModel(convnext)

    if pretrained_weights:
        chf.load_state_dict(torch.load(pretrained_weights, map_location=torch.device("cuda")))

    nested_model = list(chf.model.children())[0]

    return torch.nn.Sequential(*list(nested_model.children())[:2])  # drop last LayerNorm layer


def get_slivit_model(args):

    slivit = SLIViT(feature_extractor=get_feature_extractor(4, args.slivit_fe_path), num_classes=args.nb_classes,
                    vit_dim=256, vit_depth=5, heads=20, mlp_dim=512,
                    num_of_patches=args.slivit_num_of_patches, dropout=0, emb_dropout=0)
    return slivit