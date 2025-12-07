"""
Loss Functions for MRI Reconstruction

(1) gradient_loss: Computes gradient loss for input tensor
(2) gradient_loss_img: Computes gradient loss between input tensor and ground truth image
(3) gaussian_weight_map: Generate a Gaussian Weight Map of specified size and sigma
"""
import torch 
import torch.nn as nn
import torch.nn.functional as F

def gradient_loss(x, penalty='l2'):
    """Computes gradient loss for input tensor."""

    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])

    if penalty == 'l2':
        dy = dy ** 2
        dx = dx ** 2

    elif penalty == 'l1':
        dy = dy
        dx = dx
    
    return (torch.mean(dx) + torch.mean(dy)) / 2.0

def gradient_loss_img(x, gt):
    """Computes gradient loss between input tensor and ground truth image."""
    sobel_kernel_x = torch.tensor([[1, 0, -1], 
                                   [2, 0, -2], 
                                   [1, 0, -1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)

    sobel_kernel_y = torch.tensor([[1, 2, 1], 
                                   [0, 0, 0], 
                                   [-1, -2, -1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)

    x_grad_x = F.conv2d(x, sobel_kernel_x, padding=1)
    x_grad_y = F.conv2d(x, sobel_kernel_y, padding=1) 
    gt_grad_x = F.conv2d(gt, sobel_kernel_x, padding=1)
    gt_grad_y = F.conv2d(gt, sobel_kernel_y, padding=1)

    grad_loss_x = F.mse_loss(x_grad_x, gt_grad_x)
    grad_loss_y = F.mse_loss(x_grad_y, gt_grad_y)  
    return grad_loss_x + grad_loss_y

def gaussian_weight_map(height = 234, width = 176, sigma = 40):
    """Generate a Gaussian Weight Map of size [height, width] with given sigma."""
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    center_y, center_x = height // 2, width // 2 
    dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2
    weight_map = 1.0 + 10.0 * torch.exp(-dist_sq / (2 * sigma ** 2))
    return weight_map # Shape [height, width]
