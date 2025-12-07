"""
Normalized Attention U-Net for MRI Reconstruction

https://github.com/facebookresearch/fastMRI
"""

import math 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class NormAttentionUnet(nn.Module):
    """Attention U-Net Archiatecture with normalization layers for MRI reconstruction"""
    def __init__(self, 
                 channels, 
                 num_pool_layers,
                 in_channels=2, 
                 out_channels=2,
                 dropout=0.0):
        
        super(NormAttentionUnet, self).__init__()
        self.channels = channels 
        self.num_pool_layers = num_pool_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout



        self.unet = AttentionUnet(
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            channels = self.channels,
            num_pool_layers = self.num_pool_layers,
            dropout = self.dropout
        )
        

    def complex_to_channel_dim(self, x):
        """Convert complex tensor to channel dimension."""
        b, c, h, w, _ = x.shape 
        assert _ == 2, "Last dimension must be of size 2 for complex numbers."
        return x.permute(0, 4, 1, 2, 3).reshape(b, c * 2, h, w)

    def channel_dim_to_complex(self, x):
        """Convert channel dimension back to complex tensor."""
        b, c, h, w = x.shape 
        assert c % 2 == 0, "Channel dimension must be even to convert back to complex."
        return x.view(b, 2, c // 2, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x):
        """Apply normalization to the input tensor."""
        b, c, h, w = x.shape 
        x_stat = x.view(b, 2, c//2, h, w)

        mean = x_stat.mean(dim=2).view(b, c, 1, 1)
        std = x_stat.std(dim=2).view(b, c, 1, 1) 

        return (x - mean) / (std), mean, std

    def unnorm(self, x, mean, std):
        """Revert normalization on the input tensor."""
        return x * std + mean 

    def pad(self, x):
        """Pad input tensor to be multiple of 16 in height and width."""
        b, c, h, w = x.shape 
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1 

        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        x = F.pad(x, w_pad + h_pad)
        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(self, x, h_pad, w_pad, h_mult, w_mult):
        """Remove padding from the input tensor."""
        return x[..., h_pad[0]: h_mult - h_pad[1], w_pad[0]: w_mult - w_pad[1]]

    def forward(self, x):
        """Forward pass of the Normalized U-Net."""
        assert x.shape[-1] == 2, "Input tensor must have last dimension of size 2 for complex numbers."

        ## Normalize and pad
        x = self.complex_to_channel_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        x = self.unet(x) 

        ## Unpad and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.channel_dim_to_complex(x)
        return x


class AttentionUnet(nn.Module):
    """Attention U-Net Architecture."""
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 channels, 
                 num_pool_layers, 
                 dropout):
        super(AttentionUnet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.num_pool_layers = num_pool_layers
        self.dropout = dropout

        # Down Sample Layers
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_channels, channels, dropout)])
        ch = channels 
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, dropout))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, dropout)

        # Up Sample Layers
        self.up_conv = nn.ModuleList() 
        self.up_transpose_conv = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_conv.append(ConvBlock(ch * 2, ch, dropout))
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.attention_gates.append(AttentionGate(gate_channels = ch, skip_channels = ch, inter_channels = ch // 2))
            ch //= 2

        # Final Decoder Level 
        self.attention_gates.append(AttentionGate(gate_channels = ch, skip_channels = ch, inter_channels = ch // 2))
        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, dropout), 
                nn.Conv2d(ch, out_channels, kernel_size=1, stride=1)
            )
        )

        assert len(self.attention_gates) == len(self.up_transpose_conv)

    def forward(self, x):
        stack = [] 
        out = x 

        # Down Sampling Layers 
        for layer in self.down_sample_layers:
            out = layer(out) 
            stack.append(out)
            out = F.avg_pool2d(out, kernel_size=2, stride=2, padding=0)

        out = self.conv(out) 

        # Up Sampling Layers 
        for conv, transpose_conv, attention_gate in zip(self.up_conv, self.up_transpose_conv, self.attention_gates):
            skip_connection = stack.pop() 
            gating_signal = transpose_conv(out) 

            attended_skip_connection = attention_gate(gating_signal, skip_connection)

            out = gating_signal 

            # Check Padding 
            padding = [0, 0, 0, 0] 
            if out.shape[-1] != attended_skip_connection.shape[-1]:
                padding[1] = 1 # Padding Right 
            if out.shape[-2] != attended_skip_connection[-2]:
                padding[3] = 1 
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(out, padding, "reflect")

            out = torch.cat([out, attended_skip_connection], dim=1)
            out = conv(out)

        return out 

class AttentionGate(nn.Module):
    """Attention Gate Module: Filters encoder features (skip connection) based on decoder features (gating signal)"""
    def __init__(self, gate_channels, skip_channels, inter_channels):
        super(AttentionGate, self).__init__()
        self.gate_channels = gate_channels # Channels in the gating signal (from decoder) 
        self.skip_channels = skip_channels # Channels in the skip connection (from encoder)
        self.inter_channels = inter_channels # Channels in the intermediate convolution 

        # Pointwise Convolution for Gating Signal 
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True), 
            nn.InstanceNorm2d(inter_channels)
        )

        # Pointwise Convolution for Skip Connection
        self.W_s = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True), 
            nn.InstanceNorm2d(inter_channels)
        )

        # Pointwise Convolution for Attention Map 
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True), 
            nn.InstanceNorm2d(1), 
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, gating_signal, skip_connection):
        gs_out = self.W_g(gating_signal)
        sc_out = self.W_s(skip_connection)

        psi = self.relu(gs_out + sc_out)
        alpha = self.psi(psi)
        return skip_connection * alpha
        

class ConvBlock(nn.Module):
    """Convolutional Block with two convolutional layers and ReLU activations."""

    def __init__(self, in_channels, out_channels, dropout):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), 
            nn.InstanceNorm2d(out_channels), 
            nn.LeakyReLU(negative_slope=0.2, inplace=True), 
            nn.Dropout2d(dropout), 
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), 
            nn.InstanceNorm2d(out_channels), 
            nn.LeakyReLU(negative_slope=0.2, inplace=True), 
            nn.Dropout2d(dropout)
        )

    def forward(self, x):
        return self.layers(x) 


class TransposeConvBlock(nn.Module):
    """Transpose Convolutional Block with either Upsample/Conv2d or ConvTranspose"""

    def __init__(self, in_channels, out_channels, type="ConvTranspose"):
        super(TransposeConvBlock, self).__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels 

        if type == "ConvTranspose": 
            layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        else: 
            layer = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), 
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
            )

        self.layers = nn.Sequential(
            layer, 
            nn.InstanceNorm2d(out_channels), 
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
    def forward(self, x):
        return self.layers(x)


