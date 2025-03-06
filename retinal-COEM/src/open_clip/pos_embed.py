# Revised by Zixuan Zucks Liu @University of Washington

import numpy as np

import torch


def interpolate_pos_embed(model, checkpoint_model):

    interpolate_flag = False
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        pos_embed_name = 'pos_embed'
        interpolate_flag = True
    elif 'pos_embed_spatial' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed_spatial']
        pos_embed_name = 'pos_embed_spatial'
        interpolate_flag = True
    if interpolate_flag:
        embedding_size = pos_embed_checkpoint.shape[-1]
        if pos_embed_name == 'pos_embed':
            num_patches = model.patch_embed.num_patches
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches

        elif pos_embed_name == 'pos_embed_spatial':
            num_patches = model.patch_embed.num_patches // (model.patch_embed.frames // model.patch_embed.t_patch_size)
            num_extra_tokens = model.pos_embed_spatial.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        if isinstance(model.image_size, int):
            new_size = int(num_patches ** 0.5)
            new_size = (new_size, new_size)
        else:
            new_size = (model.image_size[0] // model.patch_embed.patch_size[0], model.image_size[1] // model.patch_embed.patch_size[1])

        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(f"Position interpolate {pos_embed_name}" + " from %dx%d to %dx%d" % (orig_size, orig_size, new_size[0], new_size[1]))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size[0], new_size[1]), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model[pos_embed_name] = new_pos_embed



# Added by zucks
def interpolate_temporal_pos_embed(model, checkpoint_model, smaller_interpolate_type='interp'):
    # assume model is vit for downstream tasks
    # [TODO:] assume no extra tokens, if needed, need to add
    if "pos_embed_temporal" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed_temporal"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        orig_num_temporal_patches = pos_embed_checkpoint.shape[-2]
        new_num_temporal_patches = model.patch_embed.frames // model.patch_embed.t_patch_size

        if orig_num_temporal_patches != new_num_temporal_patches:
            print(
                "Position interpolate from %d to %d"
                % (orig_num_temporal_patches, new_num_temporal_patches)
            )

            pos_tokens = pos_embed_checkpoint.permute(0, 2, 1)

            if orig_num_temporal_patches > new_num_temporal_patches and smaller_interpolate_type == "crop":
                # crop in the middle
                start_idx = (orig_num_temporal_patches - new_num_temporal_patches) // 2
                pos_tokens = pos_tokens[:, :, start_idx:start_idx + new_num_temporal_patches]
                print(f"Crop in the middle, from {start_idx} to {start_idx + new_num_temporal_patches}")
            else:
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens,
                    size=new_num_temporal_patches,
                    mode="linear",
                    align_corners=False,
                )

            pos_tokens = pos_tokens.permute(0, 2, 1)
            new_pos_embed = pos_tokens

            checkpoint_model["pos_embed_temporal"] = new_pos_embed