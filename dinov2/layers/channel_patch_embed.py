from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_


class PatchEmbedPerChannel(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size) * in_chans
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.in_chans = in_chans

        self.proj = nn.Conv3d(
            1,
            embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )  # CHANGED

        self.channel_embed = nn.parameter.Parameter(
            torch.zeros(1, embed_dim, in_chans, 1, 1)
        )
        trunc_normal_(self.channel_embed, std=0.02)

    def forward(self, x, extra_tokens={}):
        # assume all images in the same batch has the same input channels
        if "channels" in extra_tokens.keys():
            cur_channels = extra_tokens["channels"][0]
        else:
            cur_channels = self.in_chans

        B, Cin, H, W = x.shape
        # Note: The current number of channels (Cin) can be smaller or equal to in_chans

        # shared projection layer across channels
        x = self.proj(x.unsqueeze(1))  # B Cout Cin H W

        # channel specific offsets
        x += self.channel_embed[:, :, cur_channels, :, :]  # B Cout Cin H W

        # preparing the output sequence
        x = x.flatten(2)  # B Cout CinHW
        x = x.transpose(1, 2)  # B CinHW Cout

        return x
