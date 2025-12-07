"""
Reconstruction U-Net Model with Coil Sensitivity Estimation Module. 
"""
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from models.unet import NormUnet 
from utils.operator import grad_g_operator, transpose_operator
from utils.fourier import ifft2c 
from utils.util import rss_complex 


class ReconUnet(nn.Module):
    """Reconstruction U-Net model with coil sensitivity estimation for MRI reconstruction."""

    def __init__(self, 
                 sensitivity_channels=8, 
                 sensitivity_pool_layers=4, 
                 channels=32, 
                 pool_layers=4):
        super(ReconUnet, self).__init__()

        self.sensitivity_model = SensitivityModel(channels = sensitivity_channels,
                                                  num_pool_layers = sensitivity_pool_layers)
        self.model = ReconUnetBlock(
            NormUnet(
                channels=channels,
                num_pool_layers=pool_layers
            )
        )

    def forward(self, kspace, mask):
        sensitivity_maps = self.sensitivity_model(kspace, ACS_size=48)
        kspace_pred = self.model(kspace, mask, sensitivity_maps)
        total_image = transpose_operator(kspace_pred, sensitivity_maps) 
        return total_image, sensitivity_maps

class ReconUnetBlock(nn.Module):
    """Single block for ReconUnet model."""
    def __init__(self, model):
        super(ReconUnetBlock, self).__init__()
        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    def forward(self, current_kspace, mask, sensitivity_maps):
        mask = torch.stack([mask, mask], dim=-1)
        current_image = transpose_operator(current_kspace, sensitivity_maps) 
        output_image = self.model(current_image) 
        
        reconstruction = grad_g_operator(output_image, sensitivity_maps) 
        final_kspace = current_kspace * mask + self.dc_weight * reconstruction * (1 - mask)
        return final_kspace

class SensitivityModel(nn.Module):
    """Model for learning coil sensitivity map estimation from k-space data."""
    """Applies IFFT to multichannel k-space data then U-Net to coil images to estimate coil sensitivity maps."""

    def __init__(self,
                 channels, 
                 num_pool_layers, 
                 in_channels=2, 
                 out_channels=2, 
                 dropout=0.0):
        super(SensitivityModel, self).__init__()

        self.norm_unet = NormUnet(
            channels=channels,
            num_pool_layers=num_pool_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout
        )

    def channels_to_batch_dim(self, x):
        """Convert channel dimension to batch dimension."""
        b, c, h, w, _ = x.shape 
        assert _ == 2, "Last dimension must be of size 2 for complex numbers."
        return x.view(b * c, 1, h, 2, _), b

    def batch_channels_to_channel_dim(self, x, batch_size):
        """Convert batch dimension back to channel dimension."""
        bc, _, h, w, _ = x.shape 
        assert _ == 2, "Last dimension must be of size 2 for complex numbers."
        return x.view(batch_size, bc // batch_size, h, w, _)

    def divide_rss(self, x):
        return x / rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def forward(self, masked_kspace, ACS_size):
        # Get ACS Region 
        ACS_center_x = masked_kspace.shape[-3] // 2
        ACS_center_y = masked_kspace.shape[-2] // 2
        ACS = torch.zeros_like(masked_kspace)
        ACS[:, :, 
            ACS_center_x - ACS_size // 2:ACS_center_x + ACS_size // 2 + 1, 
            ACS_center_y - ACS_size // 2:ACS_center_y + ACS_size // 2 + 1, 
        ] = masked_kspace[
            :, :, 
            ACS_center_x - ACS_size // 2:ACS_center_x + ACS_size // 2 + 1, 
            ACS_center_y - ACS_size // 2:ACS_center_y + ACS_size // 2 + 1
        ]

        x = ifft2c(ACS)

        x, batch_size = self.channels_to_batch_dim(x)

        # Estimate Coil Sensitivity Maps 
        x = self.norm_unet(x)
        x = self.batch_channels_to_channel_dim(x, batch_size)
        x = self.divide_rss(x)
        assert torch.isnan(x).sum() == 0, "NaN values found in sensitivity maps."
        return x 
        
