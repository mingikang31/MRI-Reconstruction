"""
Main training and validation script for MRI reconstruction using a U-Net based model with coil sensitivity estimation.
"""

import os 
import sys 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np 
from PIL import Image

from dataset import MRIEchoDataset
from models.recon_unet import ReconUnet
from models.recon_attention_unet import ReconAttentionUnet
from utils.util import to_complex, walsh_sensitivity_maps, normalize, normalize_batch, zero_filled_rss, get_mask
from utils.metric import compare_psnr_batch, compare_ssim_batch
from utils.loss import gradient_loss_img, gradient_loss
from utils.operator import grad_g_operator

# Output Directory for images 
output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)

# Device Configuration for Torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Model Initialization ## [Change Model] AttentionUnet/Unet
model = ReconUnet(
    sensitivity_channels=8, 
    sensitivity_pool_layers=4, 
    channels=64, 
    pool_layers=4
).to(device)

print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

# Dataset and DataLoader Setup 
train_subjects = ["S001", "S003", "S004", "S005", "S006", "S007", "S010"]
val_subjects = ["S012"]

train_dataset = MRIEchoDataset(
    h5_dir="./data/h5_files",
    subjects=train_subjects)
val_dataset = MRIEchoDataset(
    h5_dir="./data/h5_files",
    subjects=val_subjects)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# Optimizer, Scheduler, Scaler
epochs = 200
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=25, gamma=0.5)
scaler = torch.cuda.amp.GradScaler()

best_psnr = 0.0 

for epoch in range(epochs):
    #### Training Loop ####
    model.train() 

    iter_ = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training", total=len(train_loader))

    # Train Logging 
    train_psnr = [] 
    train_ssim = [] 
    train_epoch_loss = []

    for samples in iter_:
        # Process Data
        ksp_gt = to_complex(samples["ksp_gt"], device)
        smps_gt, img_gt = walsh_sensitivity_maps(torch.view_as_real(ksp_gt))
        norm_factor_img = img_gt.abs().max() 
        img_gt = img_gt / norm_factor_img

        # Masking
        mask = get_mask().to(device) 

        # K-space Processing 
        ksp_4x = (ksp_gt * mask[None, None, :, :]) / norm_factor_img    
        ksp_gt = ksp_gt / norm_factor_img

        input_img = zero_filled_rss(torch.view_as_real(ksp_4x))

        # Forward Pass with Mixed Precision
        with torch.set_grad_enabled(True):
            with torch.cuda.amp.autocast():
                recon_img, smps = model(torch.view_as_real(ksp_4x), mask) 
                recon_ksp = grad_g_operator(recon_img, smps)

                # Loss Computation 
                loss_ksp = F.mse_loss(recon_ksp, torch.view_as_real(ksp_gt))
                loss_img = F.mse_loss(torch.view_as_complex(recon_img).abs(), torch.view_as_complex(img_gt).abs())
                loss_gradient = gradient_loss_img(torch.view_as_complex(recon_img).abs(), torch.view_as_complex(img_gt).abs())
                loss_smps = gradient_loss(smps) 

                total_loss = 1.5 * loss_img + 0.5 * loss_ksp + 0.001 * loss_smps + 0.1 * loss_gradient

        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        output_magnitude = torch.view_as_complex(recon_img).abs()
        gt_magnitude = torch.view_as_complex(img_gt).abs()

        psnr_vals = compare_psnr_batch(normalize_batch(output_magnitude), normalize_batch(gt_magnitude))
        ssim_vals = compare_ssim_batch(normalize_batch(output_magnitude), normalize_batch(gt_magnitude))

        psnr_vals = psnr_vals.mean().detach().cpu().numpy()
        ssim_vals = ssim_vals.mean().detach().cpu().numpy()

        train_psnr.append(psnr_vals)
        train_ssim.append(ssim_vals)
        train_epoch_loss.append(total_loss.item())

    ave_psnr = np.mean(train_psnr)
    ave_ssim = np.mean(train_ssim)
    ave_loss = np.mean(train_epoch_loss)
    print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {ave_loss:.4f}, PSNR: {ave_psnr:.2f}, SSIM: {ave_ssim:.4f}")

    # Reset Logging Lists
    train_psnr = []
    train_ssim = []
    train_epoch_loss = []

    #### Validation Loop ####
    model.eval()

    iter_ = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation", total=len(val_loader))

    # Validation Logging 
    val_psnr = [] 
    val_ssim = [] 
    val_epoch_loss = []

    for samples in iter_:
        # Process Data
        ksp_gt = to_complex(samples["ksp_gt"], device)
        smps_gt, img_gt = walsh_sensitivity_maps(torch.view_as_real(ksp_gt))
        norm_factor_img = img_gt.abs().max() 
        img_gt = img_gt / norm_factor_img

        # Masking
        mask = get_mask().to(device) 

        # K-space Processing 
        ksp_4x = to_complex(samples["ksp_4x"].float(), device)
        norm_factor_4x = ksp_4x.abs().max()
        ksp_4x = ksp_4x / norm_factor_4x
        ksp_gt = ksp_gt / norm_factor_img

        input_img = zero_filled_rss(torch.view_as_real(ksp_4x))

        # Forward Pass with Mixed Precision
        with torch.set_grad_enabled(False):
            with torch.cuda.amp.autocast():
                recon_img, smps = model(torch.view_as_real(ksp_4x), mask) 
                recon_ksp = grad_g_operator(recon_img, smps)

                # Loss Computation 
                loss_ksp = F.mse_loss(recon_ksp, torch.view_as_real(ksp_gt))
                loss_img = F.mse_loss(torch.view_as_complex(recon_img).abs(), torch.view_as_complex(img_gt).abs())
                loss_gradient = gradient_loss_img(torch.view_as_complex(recon_img).abs(), torch.view_as_complex(img_gt).abs())
                loss_smps = gradient_loss(smps) 

                total_loss = 1.5 * loss_img + 0.5 * loss_ksp + 0.001 * loss_smps + 0.1 * loss_gradient


        output_magnitude = torch.view_as_complex(recon_img).abs()
        gt_magnitude = torch.view_as_complex(img_gt).abs()

        psnr_vals = compare_psnr_batch(normalize_batch(output_magnitude), normalize_batch(gt_magnitude))
        ssim_vals = compare_ssim_batch(normalize_batch(output_magnitude), normalize_batch(gt_magnitude))

        psnr_vals = psnr_vals.mean().detach().cpu().numpy()
        ssim_vals = ssim_vals.mean().detach().cpu().numpy()

        val_psnr.append(psnr_vals)
        val_ssim.append(ssim_vals)
        val_epoch_loss.append(total_loss.item())

    if epoch % 5 == 0: 
        target_show = normalize(torch.abs(torch.view_as_complex(img_gt[0, ...])).to('cpu').detach().numpy())
        output_show = normalize(torch.abs(torch.view_as_complex(recon_img[0, ...])).to('cpu').detach().numpy())
        input_show = normalize(input_img[0, ...].to('cpu').detach().numpy())
        concat_img = np.concatenate([input_show, output_show, target_show], axis=1)
        
        # Save all four images 
        target_img = (target_show * 255).astype(np.uint8)
        output_img = (output_show * 255).astype(np.uint8)
        input_img_ = (input_show * 255).astype(np.uint8)
        concat_img = (concat_img * 255).astype(np.uint8)
        
        Image.fromarray(input_img_).save(os.path.join(output_dir, f'epoch_{epoch+1}_input.png'))
        Image.fromarray(output_img).save(os.path.join(output_dir, f'epoch_{epoch+1}_output.png'))
        Image.fromarray(target_img).save(os.path.join(output_dir, f'epoch_{epoch+1}_target.png'))
        Image.fromarray(concat_img).save(os.path.join(output_dir, f'epoch_{epoch+1}_concat.png'))

    ave_psnr = np.mean(val_psnr)
    ave_ssim = np.mean(val_ssim)
    ave_loss = np.mean(val_epoch_loss)
    print(f"[Epoch {epoch+1}/{epochs}] Validation Loss: {ave_loss:.4f}, PSNR: {ave_psnr:.2f}, SSIM: {ave_ssim:.4f}")

    if ave_psnr > best_psnr:
        best_psnr = ave_psnr
        torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
        print(f"Best model saved with PSNR: {best_psnr:.2f}")

    # Reset Logging Lists
    val_psnr = []
    val_ssim = []
    val_epoch_loss = []