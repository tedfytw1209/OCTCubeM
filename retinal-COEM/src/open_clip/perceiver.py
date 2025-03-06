from statistics import mode
import torch
from torch import nn as nn
from .perceiver_module import (
    PerceiverEncoder, 
    InputAdapter,
    MultiHeadAttention,
    get_2d_sincos_pos_embed)


def perceiver_base(input_adapter, num_latents, num_latent_channels):
    model = PerceiverEncoder(
                            input_adapter=input_adapter,
                            num_latents=num_latents,
                            num_latent_channels=num_latent_channels,
                            num_cross_attention_heads=4,
                            num_cross_attention_layers=1,
                            num_self_attention_heads=4,
                            num_self_attention_layers_per_block=6
        )
    return model

def perceiver_large(input_adapter, num_latents, num_latent_channels):
    model = PerceiverEncoder(
                            input_adapter=input_adapter,
                            num_latents=num_latents,
                            num_latent_channels=num_latent_channels,
                            num_cross_attention_heads=4,
                            num_cross_attention_layers=1,
                            num_self_attention_heads=4,
                            num_self_attention_layers_per_block=12,
                            num_self_attention_blocks=2
        )
    return model


class CoordInputAdapter(nn.Module):

    def __init__(self, embed_dim=512):
        super().__init__()
        self.grid_size = 1000
        self.g = 256
        self.num_tiles = self.grid_size ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tiles, embed_dim), requires_grad=False).half()
        self.init_pos_emb()

    def init_pos_emb(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_tiles**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).half().unsqueeze(0))

    def make_grids(self, coords):
        # convert the 2d coordinates into flat 1d positions
        # index the positions by rows
        pos = torch.floor(coords / 256.) 
        pos = pos[:, :, 0] * self.grid_size + pos[:, :, 1]
        pos[pos < 0] = 0
        pos[pos >= self.num_tiles] = self.num_tiles - 1
        pos = pos.long()
        return pos

    def forward(self, coords):
        # note that in pad_masks, 1 means padded tensors
        pos = self.make_grids(coords)
        pos_emb = [self.pos_embed[:, p.to(self.pos_embed.device), :] for p in pos]
        pos_emb = torch.cat(pos_emb, dim=0).to(coords.device)
        return pos_emb
    

class ImageInputAdapter(InputAdapter):
    def __init__(self, 
                num_image_channels: int,
                num_latent_channels: int,
    ):
        super().__init__(num_input_channels=num_latent_channels)
        self.proj = nn.Linear(num_image_channels, num_latent_channels)
        self.coord_adapter = CoordInputAdapter(embed_dim=num_latent_channels)

    def forward(self, x, coords):
        x_adapt = self.proj(x)
        x_pos = self.coord_adapter(coords)
        x_out = x_adapt + x_pos
        return x_out


class VisionPerceiver(nn.Module):
    def __init__(
        self,
        num_latents=256,
        num_latent_channels=512,
        num_image_channels=512,
    ):
        super().__init__()

        self.input_adapter = ImageInputAdapter(num_image_channels=num_image_channels, \
                                               num_latent_channels=num_latent_channels)
        self.perceiver = perceiver_base(self.input_adapter, num_latents, num_latent_channels)
        p_num = self.count_parameters(self.perceiver)
        print('Perceiver parameters: ', p_num)

    def set_grad_checkpointing(self, enable=True):
        self.perceiver.set_grad_checkpointing(enable)
    
    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def forward(self, img, img_coords=None, img_pad_mask=None):
        x_latent = self.perceiver(img, img_coords, pad_mask=img_pad_mask)
        x_latent = torch.mean(x_latent, dim=1)
        return x_latent

