# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial

import re
import torch
import torch.nn as nn
from einops import rearrange
from collections import OrderedDict
from custom_util import video_vit
from custom_util.loggings import master_print as print
import torch.nn.functional as F
import numpy as np
from flash_attn.models.vit import create_block

class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        num_frames=16,
        t_patch_size=4,
        patch_embed=video_vit.PatchEmbed,
        no_qkv_bias=False,
        sep_pos_embed=False,
        trunc_init=False,
        cls_embed=False,
        pred_t_dim=8,
        high_res_input_size=512,
        use_flash_attn=False,
        # out_chans=None,
        # enable_high_res_patch_embed=False,
        **kwargs,
    ):
        super().__init__()
        self.trunc_init = trunc_init
        self.sep_pos_embed = sep_pos_embed
        self.cls_embed = cls_embed
        self.pred_t_dim = pred_t_dim
        self.in_chans = in_chans
        # self.out_chans = out_chans if out_chans is not None else in_chans
        self.t_pred_patch_size = t_patch_size * pred_t_dim // num_frames
        if isinstance(input_size, int):
            input_size = (input_size, input_size)

        self.patch_embed = patch_embed(
            input_size,
            patch_size,
            in_chans,
            embed_dim,
            num_frames,
            t_patch_size,
        )
        num_patches = self.patch_embed.num_patches
        input_size = self.patch_embed.input_size
        self.input_size = input_size


        # self.enable_high_res_patch_embed = enable_high_res_patch_embed
        self.high_res_patch_embed = video_vit.PatchEmbed(
            high_res_input_size,
            patch_size,
            in_chans,
            embed_dim,
            num_frames,
            t_patch_size,
        )
        self.high_res_input_size = self.high_res_patch_embed.input_size

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.decoder_cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, self.high_res_input_size[1] * self.high_res_input_size[2], embed_dim)
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], embed_dim)
            )
            if self.cls_embed:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, embed_dim),
            )

        self.use_flash_attn = use_flash_attn
        if use_flash_attn:
            # make dpr
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
            ]
        )
        else:
            self.blocks = nn.ModuleList(
                [
                    video_vit.Block(
                        embed_dim,
                        num_heads,
                        mlp_ratio,
                        qkv_bias=not no_qkv_bias,
                        qk_scale=None,
                        norm_layer=norm_layer,
                    )
                    for i in range(depth)
                ]
            )
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if sep_pos_embed:
            self.decoder_pos_embed_spatial = nn.Parameter(
                torch.zeros(1, self.high_res_input_size[1] * self.high_res_input_size[2], decoder_embed_dim)
            )

            self.decoder_pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], decoder_embed_dim)
            )
            if self.cls_embed:
                self.decoder_pos_embed_class = nn.Parameter(
                    torch.zeros(1, 1, decoder_embed_dim)
                )
        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.decoder_pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, decoder_embed_dim),
            )
        if use_flash_attn:
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
            self.decoder_blocks = nn.ModuleList(
                [
                    video_vit.Block(
                        decoder_embed_dim,
                        decoder_num_heads,
                        mlp_ratio,
                        qkv_bias=not no_qkv_bias,
                        qk_scale=None,
                        norm_layer=norm_layer,
                    )
                    for i in range(decoder_depth)
                ]
            )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            self.t_pred_patch_size * patch_size**2 * in_chans,
            bias=True,
        )

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        print("model initialized")

    def initialize_weights(self):
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.sep_pos_embed:
            torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

            torch.nn.init.trunc_normal_(self.decoder_pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed_temporal, std=0.02)

            if self.cls_embed:
                torch.nn.init.trunc_normal_(self.pos_embed_class, std=0.02)
                torch.nn.init.trunc_normal_(self.decoder_pos_embed_class, std=0.02)
        else:
            torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        w = self.patch_embed.proj.weight.data
        if self.trunc_init:
            torch.nn.init.trunc_normal_(w)
            torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
        else:
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs, high_res=False):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        N, _, T, H, W = imgs.shape
        if high_res:
            p = self.high_res_patch_embed.patch_size[0]
        else:
            p = self.patch_embed.patch_size[0]
        u = self.t_pred_patch_size
        assert W % p == 0 and H % p == 0 and T % u == 0
        h = H // p
        w = W // p
        t = T // u

        x = imgs.reshape(shape=(N, self.in_chans, t, u, h, p, w, p))
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(N, t * h * w, p**2 * u * self.in_chans))

        if high_res:
            self.patch_info_high_res = (N, T, H, W, p, u, t, h, w)
        else:
            self.patch_info = (N, T, H, W, p, u, t, h, w)

        return x

    def unpatchify(self, x, high_res=False, actual_t_dim=None):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        if high_res:
            N, T, H, W, p, u, t, h, w = self.patch_info_high_res
        else:
            N, T, H, W, p, u, t, h, w = self.patch_info
        print('T:', T, 'actual_t_dim:', actual_t_dim)
        if actual_t_dim is not None:
            T = actual_t_dim
        print('unpatchify:', x.shape, N, T, H, W, p, u, t, h, w, 'high_res:', high_res)

        x = x.reshape(shape=(N, t, h, w, u, p, p, self.in_chans))

        x = torch.einsum("nthwupqc->nctuhpwq", x)
        imgs = x.reshape(shape=(N, self.in_chans, T, H, W))
        return imgs

    def random_masking(self, x, mask_ratio, pre_mask=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        if pre_mask is not None:
            pre_mask = torch.mean(pre_mask, dim=-1)
            pre_mask = (pre_mask > 0.0).float()
            len_keep = torch.min(1 - torch.sum(pre_mask, dim=-1)).long()
            noise = pre_mask.clone()
        else:
            len_keep = int(L * (1 - mask_ratio))
            noise = torch.rand(N, L, device=x.device) if mask_ratio > 0 else torch.arange(
                L, device=x.device
            ).expand(N, L)


        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)


        return x_masked, mask, ids_restore, ids_keep

    def forward_encoder(self, x, mask_ratio, pre_mask=None):
        # embed patches

        H, W = x.shape[-2:]

        if H == self.high_res_input_size[1] * self.high_res_patch_embed.patch_size[0]:
            high_res = True
            x = self.high_res_patch_embed(x)
        else:
            high_res = False
            x = self.patch_embed(x)

        if pre_mask is not None:
            # pre_mask: [N, 1, T, H, W]
            pre_mask = torch.index_select(
                pre_mask, 2,
                torch.linspace(
                    0,
                    pre_mask.shape[2] - 1,
                    self.pred_t_dim, device=x.device
                ).long()
            )
            pre_mask = self.patchify(pre_mask)
            pre_mask = pre_mask.reshape(N, T * L, C)

        N, T, L, C = x.shape

        x = x.reshape(N, T * L, C)
        temp_pos_emb_type = 'all'
        if T == 1:
            temp_pos_emb_type = 'none'
        # masking: length -> length * mask_ratio
        x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio, pre_mask=pre_mask)
        x = x.view(N, -1, C)
        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # add pos embed w/o cls token
        if self.sep_pos_embed:
            if not high_res:
                # pool pos_embed to match the input size

                pos_embed = F.interpolate(
                    self.pos_embed_spatial.view(1, self.high_res_input_size[1], self.high_res_input_size[2], -1).permute(0, 3, 1, 2), [self.input_size[1], self.input_size[2]],mode='bicubic', align_corners=False
                ).permute(0, 2, 3, 1).view(1, self.input_size[1] * self.input_size[2], -1)

                pos_h, pos_w = self.input_size[1], self.input_size[2]

            else:
                pos_embed = self.pos_embed_spatial
                pos_h, pos_w = self.high_res_input_size[1], self.high_res_input_size[2]

            if temp_pos_emb_type == 'all':
                pos_embed = pos_embed.repeat(
                    1, T, 1
                ) + torch.repeat_interleave(
                    self.pos_embed_temporal,
                    pos_h * pos_w,
                    dim=1,
                )
            elif temp_pos_emb_type == 'none':
                pos_embed = pos_embed.repeat(
                    1, 1, 1
                )

            pos_embed = pos_embed.expand(x.shape[0], -1, -1)

            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )

            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed_class.expand(pos_embed.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )

        else:
            if self.cls_embed:
                cls_ind = 1
            else:
                cls_ind = 0
            pos_embed = self.pos_embed[:, cls_ind:, :].expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed[:, :1, :].expand(x.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        x = x.view([N, -1, C]) + pos_embed

        if self.use_flash_attn:
            residual = None
            for blk in self.blocks:
                x, residual = blk(x, residual)
        else:
            # apply Transformer blocks
            for blk in self.blocks:
                x = blk(x)

        x = self.norm(x)

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore, high_res=False):
        N = x.shape[0]
        # print('decoder x:', x.shape, high_res)

        T = self.patch_embed.t_grid_size
        if high_res:
            H = W = self.high_res_patch_embed.grid_size
        else:
            H = W = self.patch_embed.grid_size
        actual_t_dim = ids_restore.shape[-1] // (H * W)

        # embed tokens
        x = self.decoder_embed(x)
        C = x.shape[-1]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(N, actual_t_dim * H * W + 0 - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        x_ = x_.view([N, actual_t_dim * H * W, C])
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
        )  # unshuffle
        x = x_.view([N, actual_t_dim * H * W, C])
        temp_pos_emb_type = 'all'
        if actual_t_dim == 1:
            temp_pos_emb_type = 'none'

        # append cls token
        if self.cls_embed:
            decoder_cls_token = self.decoder_cls_token
            decoder_cls_tokens = decoder_cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((decoder_cls_tokens, x), dim=1)

        if self.sep_pos_embed:

            if not high_res:
                # pool pos_embed to match the input size

                decoder_pos_embed = F.interpolate(
                    self.decoder_pos_embed_spatial.view(1, self.high_res_input_size[1], self.high_res_input_size[2], -1).permute(0, 3, 1, 2), [self.input_size[1], self.input_size[2]],mode='bicubic', align_corners=False
                ).permute(0, 2, 3, 1).view(1, self.input_size[1] * self.input_size[2], -1)
                pos_h, pos_w = self.input_size[1], self.input_size[2]
            else:

                decoder_pos_embed = self.decoder_pos_embed_spatial
                pos_h, pos_w = self.high_res_input_size[1], self.high_res_input_size[2]
            if temp_pos_emb_type == 'all':

                decoder_pos_embed = decoder_pos_embed.repeat(
                    1, self.input_size[0], 1
                ) + torch.repeat_interleave(
                    self.decoder_pos_embed_temporal,
                    pos_h * pos_w,
                    dim=1,
                )
            elif temp_pos_emb_type == 'none':
                decoder_pos_embed = self.decoder_pos_embed_spatial.repeat(
                    1, 1, 1
                )
            if self.cls_embed:
                decoder_pos_embed = torch.cat(
                    [
                        self.decoder_pos_embed_class.expand(
                            decoder_pos_embed.shape[0], -1, -1
                        ),
                        decoder_pos_embed,
                    ],
                    1,
                )
        else:
            decoder_pos_embed = self.decoder_pos_embed[:, :, :]


        # add pos embed
        x = x + decoder_pos_embed

        # [FIXME:] update for flash attn, not sure if need to change
        if hasattr(self.blocks[0], "attn"):
            attn = self.decoder_blocks[0].attn
            requires_t_shape = hasattr(attn, "requires_t_shape") and attn.requires_t_shape
            if requires_t_shape:
                x = x.view([N, T, H * W, C])
        else:
            requires_t_shape = False

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

        if requires_t_shape:
            x = x.view([N, actual_t_dim * H * W, -1])

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]

        return x

    def forward_encoder_decoder(self, imgs):
        latent, mask, ids_restore = self.forward_encoder(imgs, 0)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        return pred

    def forward_loss(self, imgs, pred, mask, frame_loss=False):
        """
        imgs: [N, 3, T, H, W]
        pred: [N, t*h*w, u*p*p*3]
        mask: [N*t, h*w], 0 is keep, 1 is remove,
        """
        T = imgs.shape[2]
        high_res = False
        H = W = imgs.shape[-1]

        if H == self.high_res_input_size[1] * self.high_res_patch_embed.patch_size[0]:
            high_res = True

        if T == 3:
            target = self.patchify(imgs, high_res)
        else:
            _imgs = torch.index_select(
                imgs,
                2,
                torch.linspace(
                    0,
                    T - 1,
                    self.pred_t_dim,
                )
                .long()
                .to(imgs.device),
            )

            target = self.patchify(_imgs, high_res)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        mask = mask.view(loss.shape)

        input_size = self.patch_embed.input_size
        resized_loss = loss.view(-1, T // self.patch_embed.t_patch_size, input_size[1] // self.patch_embed.patch_size[0], input_size[2] // self.patch_embed.patch_size[1])

        # resize mask to [N, T, H, W]
        resized_mask = mask.view(-1, T // self.patch_embed.t_patch_size, input_size[1] // self.patch_embed.patch_size[0], input_size[2] // self.patch_embed.patch_size[1])

        frame_losses = (resized_loss * resized_mask).sum(dim=(2, 3)) / (resized_mask.sum(dim=(2, 3)) + 1e-6)


        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        if frame_loss:
            return loss, frame_losses

        return loss

    def forward(self, imgs, mask_ratio=0.75, frame_loss=False, pre_mask=None):

        H, W = imgs.shape[-2:]
        if H == self.high_res_input_size[1] * self.high_res_patch_embed.patch_size[0]:
            high_res = True
        else:
            high_res = False

        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore, high_res=high_res)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask, frame_loss=frame_loss)
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
            if patch_embed_weight.dim() == 4:
                # convert from Conv2d to Linear
                state_dict["patch_embed.proj.weight"] = rearrange(
                    patch_embed_weight, "o c h w -> o (c h w)"
                )
                print('going into the patch_embed', patch_embed_weight.shape, self.patch_embed.proj.weight.shape)
        else:
            print("Skip loading patch_embed.proj.weight")

        if "high_res_patch_embed.proj.weight" in state_dict:
            high_res_patch_embed_weight = state_dict["high_res_patch_embed.proj.weight"]
            if high_res_patch_embed_weight.dim() == 4:
                # convert from Conv2d to Linear
                state_dict["high_res_patch_embed.proj.weight"] = rearrange(
                    high_res_patch_embed_weight, "o c h w -> o (c h w)"
                )
                print(self.patch_embed.proj.weight.shape)
                print('going into the high_res_patch_embed', high_res_patch_embed_weight.shape, self.high_res_patch_embed.proj.weight.shape)
            # exit()
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

    def forward_patch_embed(self, imgs):
        H, W = imgs.shape[-2:]
        if H == self.high_res_input_size[1] * self.high_res_patch_embed.patch_size[0]:
            high_res = True
            print('high_res:', high_res, imgs.shape)
            x = self.high_res_patch_embed(imgs)
            print('new x shape:', x.shape)
        else:
            high_res = False
            x = self.patch_embed(imgs)
        N, T, L, C = x.shape
        x = x.reshape(N, T * L, C)

        return x

def flash_attn_mae_vit_large_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_flash_attn=True,
        **kwargs,
    )
    return model



def mae_vit_base_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_large_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_huge_patch14(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
