"""
Metric Functions for MRI Reconstruction. 

(1) compare_mse: Compute Mean Squared Error (MSE)
(2) compare_psnr: Compute Peak Signal-to-Noise Ratio (PSNR)
(3) ssim: Compute Structural Similarity Index Measure (SSIM)
(4) gaussian: Generate 1D Gaussian kernel
(5) create_window: Create 2D Gaussian window for SSIM computation
(6) compare_ssim: Compute SSIM between reconstructed and ground truth images
(7) compare_psnr_batch: Compute PSNR for a batch of images
(8) compare_ssim_batch: Compute SSIM for a batch of images
"""
import torch 
import torch.nn as nn 
import torch.nn.functional as F

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torch.autograd import Variable 
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from math import exp 

def compare_mse(img_recon, img_gt, size_average=True):
    """Compute Mean Squared Error (MSE) between reconstructed and ground truth images."""
    img_diff = img_recon - img_gt
    img_diff = img_diff ** 2 

    if size_average: 
        img_diff = img_diff.mean() 
    else:
        img_diff = img_diff.mean(-1).mean(-1).mean(-1)

    return img_diff

def compare_psnr(img_recon, img_gt, size_average=True, max_value=1.0):
    """Compute Peak Signal-to-Noise Ratio (PSNR) between reconstructed and ground truth images."""
    return 10 * torch.log10((max_value ** 2) / compare_mse(img_recon, img_gt, size_average=size_average))

def ssim(img_recon, img_gt, window, window_size, channel, size_average=True):
    """Compute Structural Similarity Index Measure (SSIM) between reconstructed and ground truth images."""
    mu1 = F.conv2d(img_recon, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img_gt, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img_recon * img_recon, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img_gt * img_gt, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img_recon * img_gt, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(-1).mean(-1).mean(-1)

def gaussian(window_size, sigma):
    """Generate a 1D Gaussian kernel."""
    gauss = torch.Tensor([
        exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)
    ])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    """Create a 2D Gaussian window for SSIM computation."""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def compare_ssim(img_recon, img_gt, size_average=True, window_size=11):
    """Compute SSIM between reconstructed and ground truth images."""
    (_, channel, _, _) = img_recon.size() 
    window = create_window(window_size, channel).to(img_recon.device)

    if img_recon.is_cuda:
        window = window.cuda(img_recon.get_device())
    window = window.type_as(img_recon)

    return ssim(img_recon, img_gt, window, window_size, channel, size_average)

def compare_psnr_batch(img_recon, img_gt, max_value=1.0, eps=1e-8):
    """Computes PSNR for a batch of images."""
    assert img_recon.shape == img_gt.shape, "Input images must have the same dimensions."
    assert isinstance(img_recon, torch.Tensor), "Input images must be PyTorch tensors."
    assert isinstance(img_gt, torch.Tensor), "Input images must be PyTorch tensors."
    assert img_recon.ndim != 4 and img_recon.shape[1] != 1, "Input Images must be in shape [B, C, H, W]"

    mse = torch.mean((img_recon - img_gt) ** 2, dim=(1, 2, 3)) + eps
    psnr = 10 * torch.log10((max_value ** 2) / mse) 
    return psnr 

def compare_ssim_batch(img_recon, img_gt, data_range=1.0):
    """Compute SSIM for a batch of images."""
    assert img_recon.shape == img_gt.shape, "Input images must have the same dimensions."
    assert isinstance(img_recon, torch.Tensor), "Input images must be PyTorch tensors."
    assert isinstance(img_gt, torch.Tensor), "Input images must be PyTorch tensors."
    assert img_recon.ndim == 4 and img_recon.shape[1] != 1, "Input Images must be in shape [B, C, H, W]"

    ssim = StructuralSimilarityIndexMeasure(data_range=data_range, reduction='none').to(img_recon.device)
    ssim = ssim(img_recon, img_gt)
    return ssim 







    