# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Revised by Zixuan Zucks Liu @University of Washington


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from functools import partial

import re
import sys
import torch
import torch.nn as nn

from einops import rearrange
from collections import OrderedDict

import timm.vision_transformer
from flash_attn.models.vit import create_block

from timm.vision_transformer import Block
from timm.layers import to_2tuple

try:
    # Case 1: Running from OCTCubeM/
    from util.misc import master_print as print
    from util.video_vit import Attention, Block, PatchEmbed
    from util.pos_embed import get_2d_sincos_pos_embed
except ModuleNotFoundError:
    try:
        # Case 2: Running from OCTCube/
        from .util.misc import master_print as print
        from .util.video_vit import Attention, Block, PatchEmbed
        from .util.pos_embed import get_2d_sincos_pos_embed
    except ImportError:
        # Case 3: Running standalone, fix path
        sys.path.append('../')  # Add OCTCubeM/ to path
        from util.misc import master_print as print
        from util.video_vit import Attention, Block, PatchEmbed
        from util.pos_embed import get_2d_sincos_pos_embed


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding, inherited from timm 0.3.2
    """
    def __init__(self, input_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        input_size = to_2tuple(input_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (input_size[1] // patch_size[1]) * (input_size[0] // patch_size[0])
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.input_size[0] and W == self.input_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.input_size[0]}*{self.input_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
        Currently, don't support cls_embed=False
    """
    def __init__(self, input_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 global_pool=True, cls_embed=True, use_flash_attn=True,
                 no_qkv_bias=False,qk_scale=None, drop_rate=0.0,
                 attn_drop_rate=0.0, drop_path_rate=0.0, **kwargs):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics

        self.use_flash_attn = use_flash_attn
        self.global_pool = global_pool
        self.embed_dim = embed_dim
        self.depth = depth
        self.patch_embed = PatchEmbed(input_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        input_size = self.patch_embed.input_size
        self.input_size = input_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding


        if self.use_flash_attn:

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
                    use_flash_attn=self.use_flash_attn,
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
            self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_norm=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        if self.use_flash_attn:
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, decoder_depth)
            ]
            self.decoder_blocks = nn.ModuleList(
            [
                create_block(
                    decoder_embed_dim,
                    decoder_num_heads,
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
                    n_layer=decoder_depth,
                    last_layer_subset=False,
                )
                for i in range(decoder_depth)
            ])
        else:
            self.decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_norm=None, norm_layer=norm_layer)
                for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        hidden_states_list = []
        # apply Transformer blocks
        if self.use_flash_attn:
            residual = None
            for blk in self.blocks:
                x, residual = blk(x, residual)
                hidden_states_list.append(x)
        else:
            for blk in self.blocks:
                x = blk(x)
                hidden_states_list.append(x)

        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        if self.use_flash_attn:
            residual = None
            for blk in self.decoder_blocks:
                x, residual = blk(x, residual)
        else:
            # apply Transformer blocks
            for blk in self.decoder_blocks:
                x = blk(x)

        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask, return_frame_loss=False):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        if return_frame_loss:
            frame_loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        if return_frame_loss:
            return loss, frame_loss
        return loss

    def forward(self, imgs, mask_ratio=0.75, return_frame_loss=False):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask, return_frame_loss=return_frame_loss)
        if return_frame_loss:
            loss, frame_loss = loss
            return loss, pred, mask, frame_loss
        return loss, pred, mask

    def load_state_dict_to_backbone(self, state_dict, strict=False, filter_keys=[]):
        if "patch_embed.proj.weight" in state_dict:
            patch_embed_weight = state_dict["patch_embed.proj.weight"]
            if patch_embed_weight.dim() == 4:
                # convert from Conv2d to Linear
                state_dict["patch_embed.proj.weight"] = rearrange(
                    patch_embed_weight, "o c h w -> o (c h w)"
                )
        else:
            print("Skip loading patch_embed.proj.weight")

        def key_mapping_attn(key):
            key = re.sub(r"blocks.(\d+).attn.proj.", r"blocks.\1.mixer.out_proj.", key)
            return key

        state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())
        n_layer = len(self.blocks)
        # Convert from Wqkv to Wq and Wkv for cross attention (last layer)
        for i in range(n_layer):
            Wq, Wk, Wv = state_dict.pop(f"blocks.{i}.attn.q.weight"), state_dict.pop(
                f"blocks.{i}.attn.k.weight"
            ), state_dict.pop(f"blocks.{i}.attn.v.weight")
            bq, bk, bv = state_dict.pop(f"blocks.{i}.attn.q.bias"), state_dict.pop(
                f"blocks.{i}.attn.k.bias"
            ), state_dict.pop(f"blocks.{i}.attn.v.bias")
            Wqkv = torch.cat([Wq, Wk, Wv], dim=0)
            bqkv = torch.cat([bq, bk, bv], dim=0)
            state_dict[f"blocks.{i}.mixer.Wqkv.weight"] = Wqkv
            state_dict[f"blocks.{i}.mixer.Wqkv.bias"] = bqkv

        n_layer = len(self.decoder_blocks)
        for i in range(n_layer):
            Wq, Wk, Wv = state_dict.pop(f"decoder_blocks.{i}.attn.q.weight"), state_dict.pop(
                f"decoder_blocks.{i}.attn.k.weight"
            ), state_dict.pop(f"decoder_blocks.{i}.attn.v.weight")
            bq, bk, bv = state_dict.pop(f"decoder_blocks.{i}.attn.q.bias"), state_dict.pop(
                f"decoder_blocks.{i}.attn.k.bias"
            ), state_dict.pop(f"decoder_blocks.{i}.attn.v.bias")
            Wqkv = torch.cat([Wq, Wk, Wv], dim=0)
            bqkv = torch.cat([bq, bk, bv], dim=0)
            state_dict[f"decoder_blocks.{i}.mixer.Wqkv.weight"] = Wqkv
            state_dict[f"decoder_blocks.{i}.mixer.Wqkv.bias"] = bqkv
        # filter out pos_embed and patch_embed
        state_dict = {k: v for k, v in state_dict.items() if not any([f in k for f in filter_keys])}
        return super().load_state_dict(state_dict, strict=strict)

    def load_state_dict_to_backbone_retfound(self, state_dict, strict=False, filter_keys=[], encoder_only=False):
        if "patch_embed.proj.weight" in state_dict:
            patch_embed_weight = state_dict["patch_embed.proj.weight"]

        else:
            print("Skip loading patch_embed.proj.weight")

        if "high_res_patch_embed.proj.weight" in state_dict:
            high_res_patch_embed_weight = state_dict["high_res_patch_embed.proj.weight"]
            if high_res_patch_embed_weight.dim() == 4:

                state_dict["high_res_patch_embed.proj.weight"] = rearrange(
                    high_res_patch_embed_weight, "o c h w -> o (c h w)"
                )
                print(self.patch_embed.proj.weight.shape)
                print('going into the high_res_patch_embed', high_res_patch_embed_weight.shape, self.high_res_patch_embed.proj.weight.shape)

        def key_mapping_attn(key):
            key = re.sub(r"blocks.(\d+).attn.proj.", r"blocks.\1.mixer.out_proj.", key)
            return key

        state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())
        print('state_dict:', state_dict.keys())
        n_layer = len(self.blocks)
        # Convert from Wqkv to Wq and Wkv for cross attention (last layer)
        for i in range(n_layer):

            Wqkv = state_dict.pop(f"blocks.{i}.attn.qkv.weight")
            bqkv = state_dict.pop(f"blocks.{i}.attn.qkv.bias")
            state_dict[f"blocks.{i}.mixer.Wqkv.weight"] = Wqkv
            state_dict[f"blocks.{i}.mixer.Wqkv.bias"] = bqkv
        if not encoder_only:
            n_layer = len(self.decoder_blocks)
            for i in range(n_layer):

                Wqkv = state_dict.pop(f"decoder_blocks.{i}.attn.qkv.weight")
                bqkv = state_dict.pop(f"decoder_blocks.{i}.attn.qkv.bias")
                state_dict[f"decoder_blocks.{i}.mixer.Wqkv.weight"] = Wqkv
                state_dict[f"decoder_blocks.{i}.mixer.Wqkv.bias"] = bqkv

        # filter out pos_embed and patch_embed
        state_dict = {k: v for k, v in state_dict.items() if not any([f in k for f in filter_keys])}
        return super().load_state_dict(state_dict, strict=strict)

def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



# set recommended archs
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
