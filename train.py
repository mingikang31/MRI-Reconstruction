"""Improved Training + Validation functions for MRI Reconstruction with better gradient flow."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import time
import os 
import numpy as np
from collections import deque

"""PSNR + SSIM functions for evaluating image quality."""
def psnr(output, target, max_val=1.0):
    mse = nn.functional.mse_loss(output, target)
    if mse == 0:
        return float('inf')  # PSNR is infinite if there is no error
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def average_psnr(outputs, targets, max_val=1.0):
    psnr_values = [psnr(output, target, max_val) for output, target in zip(outputs, targets)]
    return torch.mean(torch.tensor(psnr_values))

def ssim(output, target, max_val=1.0):
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    
    mu_x = F.avg_pool2d(output, kernel_size=11, stride=1, padding=5)
    mu_y = F.avg_pool2d(target, kernel_size=11, stride=1, padding=5)
    
    sigma_x = F.avg_pool2d(output * output, kernel_size=11, stride=1, padding=5) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, kernel_size=11, stride=1, padding=5) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(output * target, kernel_size=11, stride=1, padding=5) - mu_x * mu_y
    
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    
    return torch.mean(ssim_map)

class GradientMonitor:
    """Monitor gradients and detect vanishing/exploding gradient problems"""
    
    def __init__(self, model, window_size=10):
        self.model = model
        self.gradient_history = deque(maxlen=window_size)
        self.layer_gradients = {}
        
    def compute_gradient_norm(self):
        """Compute total gradient norm across all parameters"""
        total_norm = 0.0
        layer_norms = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                
                # Track gradients by layer type
                layer_type = name.split('.')[0]  
                if layer_type not in layer_norms:
                    layer_norms[layer_type] = 0.0
                layer_norms[layer_type] += param_norm.item() ** 2
        
        total_norm = total_norm ** 0.5
        for layer in layer_norms:
            layer_norms[layer] = layer_norms[layer] ** 0.5
            
        self.gradient_history.append(total_norm)
        self.layer_gradients = layer_norms
        
        return total_norm, layer_norms
    
    def is_vanishing(self, threshold=1e-6):
        """Check if gradients are vanishing"""
        if len(self.gradient_history) < 3:
            return False
        
        recent_grads = list(self.gradient_history)[-3:]
        return all(grad < threshold for grad in recent_grads)
    
    def is_exploding(self, threshold=10.0):
        """Check if gradients are exploding"""
        if len(self.gradient_history) == 0:
            return False
        
        return self.gradient_history[-1] > threshold
    
    def get_trend(self):
        """Get gradient trend (increasing, decreasing, stable)"""
        if len(self.gradient_history) < 5:
            return "insufficient_data"
        
        recent = list(self.gradient_history)[-5:]
        if recent[-1] < recent[0] * 0.5:
            return "decreasing"
        elif recent[-1] > recent[0] * 2.0:
            return "increasing" 
        else:
            return "stable"

def improved_train(model, train_loader, test_loader, criterion, optimizer, num_epochs, start_epoch=0, device=None, save_path='output/', warmup_epochs=5):
    """
    Improved training with better gradient management and learning rate strategies
    """
    model.train()
    epoch_times = []
    
    # Store best metrics for tracking
    best_psnr = 0.0
    best_psnr_epoch = 0
    best_ssim = 0.0
    best_ssim_epoch = 0
    
    # Initialize gradient monitor
    grad_monitor = GradientMonitor(model)
    
    # Improved learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )
    
    # Loss tracking for plateau detection
    loss_history = deque(maxlen=20)
    plateau_counter = 0
    
    # Training state tracking
    training_state = {
        'lr_increases': 0,
        'noise_injections': 0,
        'gradient_clips': 0
    }

    if start_epoch > 0: 
        print(f"Resuming training from epoch {start_epoch}...")
        start_epoch -= 1  
    
    try:
        for epoch in range(start_epoch, num_epochs):
            print(f'\n{"="*60}')
            print(f'Epoch {epoch+1}/{num_epochs}')
            print(f'Current LR: {optimizer.param_groups[0]["lr"]:.14f}')
            
            start = time.time()
            model.train()
            running_loss = 0.0
            
            # Epoch-level gradient statistics
            epoch_grad_norms = []
            
            for batch_idx, batch in enumerate(train_loader):
                images = batch['images'].to(device) 
                measurements = batch['measurements'].to(device)
                reconstructions = batch['reconstructions'].to(device)
                
                optimizer.zero_grad()
                outputs = model(reconstructions)
                loss = criterion(outputs, images)
                
                loss.backward()
                
                # Monitor gradients
                grad_norm, layer_grads = grad_monitor.compute_gradient_norm()
                epoch_grad_norms.append(grad_norm)
                
                # Gradient clipping (always apply)
                clip_value = 1.0
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
                if grad_norm > clip_value:
                    training_state['gradient_clips'] += 1
                
                optimizer.step()
                running_loss += loss.item()
            
            # Post-epoch analysis
            end = time.time()
            epoch_time = end - start
            avg_loss = running_loss / len(train_loader)
            avg_grad_norm = np.mean(epoch_grad_norms)
            
            print(f'Time: {epoch_time:.2f}s, Avg Loss: {avg_loss:.10f}, Avg Grad Norm: {avg_grad_norm:.10f}')
            
            epoch_times.append(epoch_time)
            loss_history.append(avg_loss)
            
            # Gradient problem detection and intervention
            if grad_monitor.is_vanishing():
                print("VANISHING GRADIENTS DETECTED!")
                current_lr = optimizer.param_groups[0]['lr']
                
                if current_lr < 1e-2:  # Only increase if not already high
                    new_lr = min(current_lr * 5, 1e-2)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                    print(f"Increased LR: {current_lr:.14f} → {new_lr:.14f}")
                    training_state['lr_increases'] += 1
                
                # Reset scheduler to prevent it from lowering LR immediately
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=10, T_mult=2, eta_min=1e-7
                )
            
            elif grad_monitor.is_exploding():
                print("EXPLODING GRADIENTS DETECTED!")
                current_lr = optimizer.param_groups[0]['lr']
                new_lr = max(current_lr * 0.1, 1e-6)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"Decreased LR: {current_lr:.14f} → {new_lr:.14f}")
            
            # Plateau detection and intervention
            if len(loss_history) >= 10:
                recent_losses = list(loss_history)[-10:]
                loss_std = np.std(recent_losses)
                
                if loss_std < 1e-6:  # Loss plateau
                    plateau_counter += 1
                    print(f"Loss plateau detected (counter: {plateau_counter})")
                    
                    if plateau_counter >= 5:
                        print("INJECTING PARAMETER NOISE to escape plateau")
                        with torch.no_grad():
                            for name, param in model.named_parameters():
                                if 'weight' in name:  # Only add noise to weights, not biases
                                    noise_scale = param.std() * 0.01  # 1% of parameter std
                                    noise = torch.randn_like(param) * noise_scale
                                    param.add_(noise)
                        
                        plateau_counter = 0
                        training_state['noise_injections'] += 1
                else:
                    plateau_counter = 0
            
            # Step scheduler (after potential LR adjustments)
            scheduler.step()
            
            # Evaluation every 5 epochs
            if (epoch + 1) % 5 == 0:
                print(f"\n[Evaluating at epoch {epoch + 1}]\n")
                print(f"Gradient norm: {grad_norm:.10f}")
                
                # Print per-layer gradients
                print("Layer gradient norms:")
                for layer, norm in layer_grads.items():
                    status = ""
                    if norm < 1e-6:
                        status = " ⚠️ VANISHING"
                    elif norm > 5.0:
                        status = " 🔥 LARGE"
                    print(f"  {layer}: {norm:.10f}{status}")

                # Evaluate PSNR and SSIM
                psnr_value = psnr_evaluation(model, test_loader, device)
                ssim_value = ssim_evaluation(model, test_loader, device)
                print(f'\nPSNR: {psnr_value:.4f}, SSIM: {ssim_value:.4f}')
                
                # Track best model
                if psnr_value > best_psnr:
                    best_psnr = psnr_value
                    best_psnr_epoch = epoch + 1
                    print(f'New best PSNR: {epoch+1} epoch!')
                    torch.save(model.state_dict(), f"{save_path}best_psnr_model.pth")

                if ssim_value > best_ssim:
                    best_ssim = ssim_value
                    best_ssim_epoch = epoch + 1
                    print(f'New best SSIM: {epoch+1} epoch!')
                    torch.save(model.state_dict(), f"{save_path}best_ssim_model.pth")
            
            # Save checkpoints every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = f"{save_path}checkpoint_epoch_{epoch+1}.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                    'best_psnr': best_psnr,
                    'training_state': training_state
                }, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
    
    except KeyboardInterrupt:
        print(f"\n{'='*60}")
        print(f"*** TRAINING INTERRUPTED AT EPOCH {epoch + 1} ***")
        print(f"{'='*60}")
        
        # Save interrupted state
        interrupted_path = f"{save_path}interrupted_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), interrupted_path)
        print(f"Model saved: {interrupted_path}")
        
        return model, best_psnr, best_psnr_epoch
    
    # Training completed normally
    print(f'\n{"="*60}')
    print(f'TRAINING COMPLETED!')
    print(f'{"="*60}')
    print(f'Best PSNR: {best_psnr:.4f} at epoch {best_psnr_epoch}')
    print(f'Best SSIM: {best_ssim:.4f} at epoch {best_ssim_epoch}')
    print(f'Average epoch time: {sum(epoch_times)/len(epoch_times):.2f}s')
    print(f'Total training time: {sum(epoch_times)/60:.1f} minutes')
    
    # Print training interventions summary
    print(f'\nTraining Interventions Summary:')
    print(f'  LR increases: {training_state["lr_increases"]}')
    print(f'  Noise injections: {training_state["noise_injections"]}')
    print(f'  Gradient clips: {training_state["gradient_clips"]}')
    
    # Save final model
    final_path = f"{save_path}final_model.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved: {final_path}\n")
    
    return model, best_psnr, best_psnr_epoch

import os
import torch

def continue_training(model, train_loader, test_loader, criterion, optimizer, num_epochs, device=None, save_path='output/', checkpoint_path="./Saved/UNet/"):
    """
    Continue training from a saved model checkpoint
    """
    start_epoch = 0  # default to starting from scratch
    
    if os.path.exists(checkpoint_path):
        print(f"Loading model checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            try:
                model.load_state_dict(checkpoint["model_state_dict"])
                if "optimizer_state_dict" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = checkpoint.get("epoch", 0) + 1
                print(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch}")
            except RuntimeError as e:
                print(f"Error loading model weights: {e}")
                print("Attempting to load with strict=False...")
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            # Legacy: if the checkpoint is just state_dict
            try:
                model.load_state_dict(checkpoint)
                print("Loaded raw state_dict (no optimizer/scheduler info)")
            except RuntimeError as e:
                print(f"Error loading model weights: {e}")
                print("Failed to resume training.")
                return
    else:
        print(f"Warning: Checkpoint path '{checkpoint_path}' not found. Starting training from scratch.")

    return improved_train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        save_path=save_path,
        start_epoch=start_epoch  
    )

def psnr_evaluation(model, dataloader, device=None):
    model.eval() 
    total_psnr = 0.0 
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['images'].to(device)
            reconstructions = batch['reconstructions'].to(device)
            outputs = model(reconstructions)
            psnr_value = psnr(outputs, images)
            total_psnr += psnr_value
        average_psnr = total_psnr / len(dataloader)    
    return average_psnr

def ssim_evaluation(model, dataloader, device=None):
    model.eval() 
    total_ssim = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['images'].to(device)
            reconstructions = batch['reconstructions'].to(device)
            outputs = model(reconstructions)
            ssim_value = ssim(outputs, images)
            total_ssim += ssim_value
        average_ssim = total_ssim / len(dataloader)
    return average_ssim

# Improved optimizer and loss functions
def get_robust_optimizer(model, initial_lr=1e-4):
    """Get a robust optimizer configuration"""
    return torch.optim.AdamW(
        model.parameters(), 
        lr=initial_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
        amsgrad=True  # More stable version of Adam
    )

def get_robust_loss():
    """Get a robust loss function"""
    # Combination of MSE and L1 loss for better training stability
    class CombinedLoss(nn.Module):
        def __init__(self, mse_weight=0.8, l1_weight=0.2):
            super().__init__()
            self.mse_weight = mse_weight
            self.l1_weight = l1_weight
            self.mse = nn.MSELoss()
            self.l1 = nn.L1Loss()
        
        def forward(self, pred, target):
            return (self.mse_weight * self.mse(pred, target) + 
                   self.l1_weight * self.l1(pred, target))
    
    return CombinedLoss()


def get_combined_loss():
    from pytorch_msssim import ssim, ms_ssim # You might need to install this: pip install pytorch-msssim

    class AdvancedCombinedLoss(nn.Module):
        def __init__(self, mse_weight=0.5, l1_weight=0.3, ssim_weight=0.2, ms_ssim_weight=0.0):
            super().__init__()
            self.mse_weight = mse_weight
            self.l1_weight = l1_weight
            self.ssim_weight = ssim_weight
            self.ms_ssim_weight = ms_ssim_weight
            self.mse = nn.MSELoss()
            self.l1 = nn.L1Loss()

        def forward(self, pred, target):
            if pred.ndim == 3:
                pred = pred.unsqueeze(1)
            if target.ndim == 3:
                target = target.unsqueeze(1)

            loss_mse = self.mse(pred, target)
            loss_l1 = self.l1(pred, target)
            # SSIM returns a similarity score (higher is better), so for loss, use 1 - ssim
            loss_ssim = 1 - ssim(pred, target, data_range=target.max()) # Adjust data_range if your data isn't 0-1
            loss_ms_ssim = 1 - ms_ssim(pred, target, data_range=target.max()) 

            return (self.mse_weight * loss_mse +
                    self.l1_weight * loss_l1 +
                    self.ssim_weight * loss_ssim + 
                    self.ms_ssim_weight * loss_ms_ssim)

    return AdvancedCombinedLoss()



"""Visualize output images"""
import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_model_comparison(model, input_image, clean_image, 
                              titles=None, cmap='gray', save_path=None, 
                              figsize=(15, 5), device=None):
    """
    Visualize comparison between input image, model output, and clean target image.
    
    Args:
        model: Trained PyTorch model
        input_image: Input reconstruction/noisy image (tensor or numpy)
        clean_image: Ground truth clean image (tensor or numpy)
        titles: List of 3 titles for [input, model_output, clean]. Default provided.
        cmap: Colormap for visualization
        save_path: Path to save the comparison image
        figsize: Figure size tuple
        device: Device to run model on (auto-detected if None)
    """
    
    # Auto-detect device if not provided
    if device is None:
        device = next(model.parameters()).device
    
    # Set default titles
    if titles is None:
        titles = ["Input Reconstruction", "Model Output", "Ground Truth"]
    
    # Ensure model is in eval mode
    model.eval()
    
    # Convert inputs to tensors and move to device
    if not torch.is_tensor(input_image):
        input_image = torch.from_numpy(input_image).float()
    if not torch.is_tensor(clean_image):
        clean_image = torch.from_numpy(clean_image).float()
    
    # Add batch dimension if needed
    if len(input_image.shape) == 3:
        input_image = input_image.unsqueeze(0)
    if len(clean_image.shape) == 3:
        clean_image = clean_image.unsqueeze(0)
    
    # Move to device
    input_image = input_image.to(device)
    clean_image = clean_image.to(device)
    
    # Get model prediction
    with torch.no_grad():
        model_output = model(input_image)
    
    # Convert all to numpy for visualization
    def tensor_to_numpy(tensor):
        if torch.is_tensor(tensor):
            tensor = tensor.detach().cpu().numpy()
        
        # Remove batch dimension
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)
        # Remove channel dimension if single channel
        if len(tensor.shape) == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        elif len(tensor.shape) == 3:
            # If multiple channels, take the first one
            tensor = tensor[0]
        
        return tensor
    
    input_np = tensor_to_numpy(input_image)
    output_np = tensor_to_numpy(model_output)
    clean_np = tensor_to_numpy(clean_image)
    
    # Create the comparison plot
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot input image
    im1 = axes[0].imshow(input_np, cmap=cmap)
    axes[0].set_title(titles[0], fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Plot model output
    im2 = axes[1].imshow(output_np, cmap=cmap)
    axes[1].set_title(titles[1], fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Plot clean/target image
    im3 = axes[2].imshow(clean_np, cmap=cmap)
    axes[2].set_title(titles[2], fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Add colorbar
    plt.colorbar(im2, ax=axes, orientation='horizontal', pad=0.1, shrink=0.8)
    
    # Calculate and display metrics
    with torch.no_grad():
        # PSNR calculation
        mse_input = torch.mean((input_image - clean_image) ** 2)
        mse_output = torch.mean((model_output - clean_image) ** 2)
        
        psnr_input = 20 * torch.log10(1.0 / torch.sqrt(mse_input + 1e-10))
        psnr_output = 20 * torch.log10(1.0 / torch.sqrt(mse_output + 1e-10))
        
        # Add metrics as text
        fig.suptitle(f'PSNR - Input: {psnr_input:.2f} dB | Model Output: {psnr_output:.2f} dB | Improvement: {psnr_output-psnr_input:.2f} dB', 
                    fontsize=14, fontweight='bold', y=0.02)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison visualization to {save_path}")
    
    plt.show()
    
    return input_np, output_np, clean_np

def visualize_model_comparison_with_difference(model, input_image, clean_image,
                                             titles=None, cmap='gray', save_path=None,
                                             figsize=(20, 5), device=None):
    """
    Extended version that also shows difference maps.
    
    Shows: Input | Model Output | Ground Truth | Input-GT Diff | Output-GT Diff
    """
    
    # Auto-detect device if not provided
    if device is None:
        device = next(model.parameters()).device
    
    # Set default titles
    if titles is None:
        titles = ["Input", "Model Output", "Ground Truth", "Input Error", "Model Error"]
    
    # Ensure model is in eval mode
    model.eval()
    
    # Convert and prepare tensors (same as above)
    if not torch.is_tensor(input_image):
        input_image = torch.from_numpy(input_image).float()
    if not torch.is_tensor(clean_image):
        clean_image = torch.from_numpy(clean_image).float()
    
    if len(input_image.shape) == 3:
        input_image = input_image.unsqueeze(0)
    if len(clean_image.shape) == 3:
        clean_image = clean_image.unsqueeze(0)
    
    input_image = input_image.to(device)
    clean_image = clean_image.to(device)
    
    # Get model prediction
    with torch.no_grad():
        model_output = model(input_image)
    
    # Convert to numpy
    def tensor_to_numpy(tensor):
        if torch.is_tensor(tensor):
            tensor = tensor.detach().cpu().numpy()
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)
        if len(tensor.shape) == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        elif len(tensor.shape) == 3:
            tensor = tensor[0]
        return tensor
    
    input_np = tensor_to_numpy(input_image)
    output_np = tensor_to_numpy(model_output)
    clean_np = tensor_to_numpy(clean_image)
    
    # Calculate difference maps
    input_diff = np.abs(input_np - clean_np)
    output_diff = np.abs(output_np - clean_np)
    
    # Create the extended comparison plot
    fig, axes = plt.subplots(1, 5, figsize=figsize)
    
    # Plot images
    axes[0].imshow(input_np, cmap=cmap)
    axes[0].set_title(titles[0], fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(output_np, cmap=cmap)
    axes[1].set_title(titles[1], fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(clean_np, cmap=cmap)
    axes[2].set_title(titles[2], fontweight='bold')
    axes[2].axis('off')
    
    # Plot difference maps with different colormap
    im4 = axes[3].imshow(input_diff, cmap='hot')
    axes[3].set_title(titles[3], fontweight='bold')
    axes[3].axis('off')
    plt.colorbar(im4, ax=axes[3], shrink=0.6)
    
    im5 = axes[4].imshow(output_diff, cmap='hot')
    axes[4].set_title(titles[4], fontweight='bold')
    axes[4].axis('off')
    plt.colorbar(im5, ax=axes[4], shrink=0.6)
    
    # Calculate metrics
    with torch.no_grad():
        mse_input = torch.mean((input_image - clean_image) ** 2)
        mse_output = torch.mean((model_output - clean_image) ** 2)
        
        psnr_input = 20 * torch.log10(1.0 / torch.sqrt(mse_input + 1e-10))
        psnr_output = 20 * torch.log10(1.0 / torch.sqrt(mse_output + 1e-10))
        
        # SSIM calculation (simplified)
        def simple_ssim(x, y):
            return torch.mean(2 * x * y + 1e-6) / torch.mean(x**2 + y**2 + 1e-6)
        
        ssim_input = simple_ssim(input_image, clean_image)
        ssim_output = simple_ssim(model_output, clean_image)
    
    # Add comprehensive metrics
    metrics_text = (f'PSNR: Input={psnr_input:.2f}dB | Output={psnr_output:.2f}dB | Δ={psnr_output-psnr_input:.2f}dB\n'
                   f'SSIM: Input={ssim_input:.3f} | Output={ssim_output:.3f} | Δ={ssim_output-ssim_input:.3f}')
    
    fig.suptitle(metrics_text, fontsize=12, fontweight='bold', y=0.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved extended comparison to {save_path}")
    
    plt.show()
    
    return input_np, output_np, clean_np, input_diff, output_diff

# Example usage:
"""
# Basic comparison
visualize_model_comparison(
    model=your_model,
    input_image=noisy_reconstruction,
    clean_image=ground_truth,
    save_path="model_comparison.png"
)

# Extended comparison with difference maps
visualize_model_comparison_with_difference(
    model=your_model,
    input_image=noisy_reconstruction,
    clean_image=ground_truth,
    save_path="detailed_comparison.png"
)
"""