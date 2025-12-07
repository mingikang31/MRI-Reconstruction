"""
Utility functions for MRI reconstruction.

(1) to_complex: Convert real tensor to complex tensor
(2) walsh_sensitivity_maps: Compute Walsh sensitivity maps
(3) complex_abs_squared: Compute squared magnitude of complex tensor
(4) complex_mul: Multiply two complex tensors
(5) complex_conj: Compute complex conjugate of a complex tensor
(6) zero_filled_rss: Compute zero-filled RSS image from k-space data
(7) rss: Root Sum of Squares (RSS) across specified dimension
(8) rss_complex: Root Sum of Squares (RSS) for complex tensors across specified dimension
(9) normalize_batch: Normalize batch of images to [0, 1]
(10) normalize: Normalize single image to [0, 1]
(11) lr_scheduler: Learning rate scheduler that decays LR every 20 epochs
(12) get_mask: load undersampling mask from file 
"""

import torch 
import numpy as np 
from utils.fourier import ifft2c

def to_complex(x, device):
    """Turn real tensor with last dimension of size 2 into complex tensor."""
    return torch.view_as_complex(x).to(device)

def walsh_sensitivity_maps(kspace):
    """Compute Walsh sensitivity maps from k-space data."""
    img_complex = torch.view_as_complex(ifft2c(kspace))
    rss_img = rss(img_complex)
    sensitivity_maps = img_complex / (rss_img + 1e-8)
    img_combined = (img_complex * sensitivity_maps.conj()).sum(dim=1, keepdim=True)
    return torch.view_as_real(sensitivity_maps), torch.view_as_real(img_combined)

def complex_abs_squared(x):
    """Compute squared magnitude of complex tensor."""
    assert x.shape[-1] == 2, "Last dimension must be of size 2 for complex numbers."
    return (x ** 2).sum(dim=-1)

def complex_mul(x, y):
    """Multiply two complex tensors."""
    assert x.shape[-1] == 2 and y.shape[-1] == 2, "Last dimension must be of size 2 for complex numbers."
    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]
    return torch.stack((re, im), dim=-1)

def complex_conj(x):
    """Compute complex conjugate of a complex tensor."""
    assert x.shape[-1] == 2, "Last dimension must be of size 2 for complex numbers."
    return torch.stack((x[..., 0], -x[..., 1]), dim=-1)

def zero_filled_rss(kspace):
    """Compute zero-filled RSS image from k-space data."""
    img_complex = torch.view_as_complex(ifft2c(kspace))
    img_rss = torch.sqrt((img_complex.abs() ** 2).sum(dim=1, keepdim=True))
    return img_rss / (img_rss.max() + 1e-8)

def rss(x, dim=0):
    """Root Sum of Squares (RSS) across specified dimension."""
    return torch.sqrt((x ** 2).sum(dim=dim))

def rss_complex(x, dim=0):
    """Root Sum of Squares (RSS) for complex tensors across specified dimension."""
    return torch.sqrt((complex_abs_squared(x)).sum(dim=dim))

def normalize_batch(x, eps=1e-8):
    """Normalize batch of images to [0, 1]."""
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)

def normalize(x):
    """Normalize single image to [0, 1]."""
    return (x - x.min()) / (x.max() - x.min())

def lr_scheduler(optimizer, epoch):
    """Decay learning rate by a factor of 0.5 every 20 epochs."""
    if epoch % 20 == 0 and epoch > 25:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.9 
    

def get_mask(mask_path="data/mask4x.npy"):
    """Load undersampling mask from file."""
    mask = np.load(mask_path)
    return torch.from_numpy(mask)
    