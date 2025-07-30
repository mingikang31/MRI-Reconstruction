"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from typing import List, Tuple
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys 


shared_project_root = '/export1/project/shirin/qmri'
sys.path.append(shared_project_root)
from utils.util import *
from utils.loss_functions import *


class NormAttentionUnet(nn.Module):
    """
    Normalized U-Net model.
    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()
        print("Mingi's NormAttentionUnet")

        self.unet = AttentionUnet_torch(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, c, 1, 1)
        std = x.std(dim=2).view(b, c, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / (std), mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x = torch.stack([x.real, x.imag], dim=-1)
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        x = self.complex_to_chan_dim(x)
        # print(f"Input shape after complex_to_chan_dim: {x.shape}")

        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        x = self.unet(x)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)


        return x



class AttentionUnet_torch(nn.Module):
    """
    A U-Net model with Attention Gates.

    This implementation follows the exact architecture of the provided U-Net,
    but injects an AttentionGate at each level of the decoder to re-weight
    the skip connection features.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        print("Attention U-Net")

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        # --- Encoder Path (Identical to Unet) ---
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        self.attention_gates = nn.ModuleList() 
        
        for _ in range(num_pool_layers - 1):
            self.attention_gates.append(AttentionGate(gate_chans=ch, skip_chans=ch, inter_chans=ch // 2))

            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        # Final decoder level
        self.attention_gates.append(AttentionGate(gate_chans=ch, skip_chans=ch, inter_chans=ch // 2))
        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

        # The lists of attention gates and upsampling layers must have the same length
        assert len(self.attention_gates) == len(self.up_transpose_conv)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image

        # --- Encoder Path (Identical) ---
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # --- Decoder Path (Modified with Attention) ---
        for transpose_conv, conv, attention_gate in zip(self.up_transpose_conv, self.up_conv, self.attention_gates):
            skip_connection = stack.pop()
            gating_signal = transpose_conv(output) # 'g' is the upsampled output

            # Apply Attention Gate
            # The attended_skip has the same shape as the original skip_connection
            attended_skip = attention_gate(g=gating_signal, x=skip_connection)

            # The upsampled 'gating_signal' becomes the new 'output' for concatenation
            output = gating_signal

            # reflect pad on the right/bottom if needed (using attended_skip for shape)
            padding = [0, 0, 0, 0]
            if output.shape[-1] != attended_skip.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != attended_skip.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            # Concatenate the upsampled output with the ATTENDED skip connection
            output = torch.cat([output, attended_skip], dim=1)
            output = conv(output)

        return output

class AttentionGate(nn.Module):
    """
    Attention Gate module.

    Filters encoder features (skip connection) based on decoder features (gating signal).
    """
    def __init__(self, gate_chans: int, skip_chans: int, inter_chans: int):
        """
        Args:
            gate_chans: Number of channels in the gating signal (from the decoder).
            skip_chans: Number of channels in the skip connection (from the encoder).
            inter_chans: Number of channels in the intermediate convolution.
        """
        super().__init__()

        # 1x1 conv for the gating signal
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_chans, inter_chans, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(inter_chans),
        )

        # 1x1 conv for the skip connection
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_chans, inter_chans, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(inter_chans),
        )

        # Final 1x1 conv to get the attention map, followed by sigmoid
        self.psi = nn.Sequential(
            nn.Conv2d(inter_chans, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: Gating signal from the decoder.
            x: Skip connection from the encoder.

        Returns:
            The attended (re-weighted) skip connection.
        """
        g_out = self.W_g(g)
        x_out = self.W_x(x)

        # Combine, activate, and generate the attention map (alpha)
        psi = self.relu(g_out + x_out)
        alpha = self.psi(psi)

        # Multiply the original skip connection with the attention map
        return x * alpha
    
class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), 
            # nn.Conv2d(
            #     in_chans, out_chans, kernel_size=3, stride=1, padding=1, bias=False
            # ),
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)

