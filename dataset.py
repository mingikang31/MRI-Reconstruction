import os 
import glob 
import numpy as np 
import torch 
from PIL import Image 
from tqdm import tqdm 
import random 
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod 
from typing import List, Optional 

from torch.utils.data import Dataset, DataLoader

from operators import * 

# GLOBAL VARIABLES 

MRI_DIR = "/export1/project/mingi/Dataset/brainMRI"
OUT_DIR = "" # directory for saving transformed images # Not in use rn #
SAVE_PLOT_DIR = "" # directory for saving plots # Not in use rn #

GRAPPA_DIR = "./data/GRAPPA_acc2"   


"""Dataset Class for MRI Data"""
def load_img_to_tensor(source_dir, max_images=8000, normalize=True, target_size=None, data_type = "tensor", device=None):
    png_images = glob.glob(os.path.join(source_dir, "**/*.png"), recursive=True) 
    png_images.sort()
    images = []

    for file_name in tqdm(png_images[:max_images], desc="Loading images"):

        try:
            img = Image.open(file_name) 
            
            if target_size is not None: 
                img = img.resize(target_size, Image.LANCZOS)

            # Convert to numpy array 
            img_array = np.array(img)
            
            if normalize: 
                img_array = img_array.astype(np.float64) / 255.0
            else: 
                img_array = img_array.astype(np.float64)
                
            images.append(img_array) 
            
            if (len(images) >= max_images):
                break
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            continue
    
    if not images:
        raise ValueError("No images found in the specified directory.")
        
    if data_type == "tensor":
        images_numpy = np.array(images, dtype=np.float64)
        images_tensor = torch.from_numpy(images_numpy).float().to(device)

        if len(images_tensor.shape) == 3:
            images_tensor = images_tensor.unsqueeze(1)
            
        return images_tensor
    elif data_type == "numpy":
        images_numpy = np.array(images, dtype=np.float64)
        return images_numpy

def save_grappa_processed_data(images, save_dir="./data", acc=4, num_coils=8, acs_lines=32, kernel_size=(3, 3), device='cpu'):
    operator = GRAPPAOperator( acc=acc, device=device, num_coils=num_coils, acs_lines=acs_lines, kernel_size=kernel_size, batch_size=1)

    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
    
    for i, img in enumerate(tqdm(images, desc="Saving images")):
        grappa_path = os.path.join(save_dir, f"image_{i:04d}.png")

        undersampled_kspace_grappa = operator.forward(img)  # Add batch dimension
        reconstructed_image_grappa = operator.transpose(undersampled_kspace_grappa).squeeze()

        write_png(reconstructed_image_grappa, grappa_path)  # Save the reconstructed image



def to_uint8(image) -> np.ndarray:
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()

    if np.iscomplexobj(image):
        image = np.abs(image)

    image = image - image.min()               # shift to 0
    maxv  = image.max()
    if maxv > 0:
        image = image / maxv                  # scale to 0-1

    return (image * 255).astype(np.uint8)


def write_png(img, out_path: str):
    Image.fromarray(to_uint8(img), mode="L").save(out_path)

class ProcessedMRIDataset(Dataset):
    def __init__(self, original_images, reconstructions):
        self.original_images = original_images
        self.reconstructions = reconstructions

    def __len__(self):
        return len(self.original_images)

    def __getitem__(self, idx):
        return {
            'idx': idx,
            'original_image': self.original_images[idx],
            'reconstruction': self.reconstructions[idx]
        }
class ProcessedMRIDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False):
        super(ProcessedMRIDataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn
        )
        
    def collate_fn(self, batch):
        images = torch.stack([item['original_image'] for item in batch])
        measurements = torch.stack([item['reconstruction'] for item in batch])
        reconstructions = torch.stack([item['reconstruction'] for item in batch])

        indices = [item['idx'] for item in batch]
        
        return {
            'images': images,
            'measurements': measurements,
            'reconstructions': reconstructions,
            'indices': indices
        }

class MRIDataset(Dataset):
    def __init__(self, images, acc=4, num_coils=8, acs_lines=32, kernel_size=(3, 3), operator=GRAPPAOperator, device=None, seed=0, split_ratio=0.8, split="train"):
        self.images = images
        self.acc = acc
        self.num_coils = num_coils
        self.acs_lines = acs_lines
        self.kernel_size = kernel_size
        self.operator = operator
        self.device = images.device

        torch.manual_seed(seed)  # Set seed for reproducibility
        
        
        if operator == MRIOperator:
            self.operator = operator(acc=self.acc, device=self.device, batch_size=1)
        if operator == GRAPPAOperator or operator == GRAPPAOperator_v1:
            self.operator = operator(acc=self.acc, device=self.device, num_coils=num_coils, acs_lines=acs_lines, kernel_size=kernel_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        return {
            'image': image,
            "idx": idx
        }
        
class MRIDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False, acc=4, num_coils=8, acs_lines=32, kernel_size=(3, 3), operator=GRAPPAOperator):
        super(MRIDataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn
        )

        self.acc = acc
        self.num_coils = num_coils
        self.acs_lines = acs_lines
        self.kernel_size = kernel_size  
        self.operator = operator
        self.operator_class = operator
        
    def collate_fn(self, batch):
        images = torch.stack([item['image'] for item in batch])
        
        # Get device from images
        device = images.device
        
        # MRI Operator
        if self.operator_class == MRIOperator:
            operator_instance = self.operator_class(acc=self.acc, device=device, batch_size=images.shape[0])
        elif self.operator_class in [GRAPPAOperator, GRAPPAOperator_v1]:
            operator_instance = self.operator_class(acc=self.acc, device=device, num_coils=self.num_coils, acs_lines=self.acs_lines, kernel_size=self.kernel_size, batch_size=images.shape[0])
        
        # Apply operators
        measurements = operator_instance.forward(images)
        reconstructions = operator_instance.transpose(measurements)
        
        indices = [item['idx'] for item in batch]
        
        return {
            'images': images,
            'measurements': measurements,
            'reconstructions': reconstructions,
            'indices': indices
        }

def split_images(images, train_ratio=0.833, seed=42):
    torch.manual_seed(seed)  # Set seed for reproducibility
    indices = torch.randperm(len(images))  # Shuffle indices
    split_idx = int(len(images) * train_ratio)

    return images[indices[:split_idx]], images[indices[split_idx:]]




"""Visualization Functions"""
def visualize_images(images, num_images, figsize=(18, 18), title="MRI Images", cmap='gray', normalize=False, save_path=None):
    
    # convert to numpy if tensor
    if torch.is_tensor(images): 
        images = images.detach().cpu().numpy() 
    
    # remove channel dimension if exists 
    if len(images.shape) == 4: 
        images = images.squeeze(1)
        
    # Grid Size 
    grid_size = int(np.ceil(np.sqrt(num_images)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    axes = axes.flatten()
    
    if grid_size == 1: 
        axes = [axes]
    else: 
        axes = axes.flatten() 
        
    for i in range(num_images):
        img = images[i]
        
        if normalize: 
            img = (img - np.min(img)) / (np.max(img) - np.min(img))

        axes[i].imshow(img, cmap=cmap)
        axes[i].axis('off')
        axes[i].set_title(f"Image {i+1}")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show() 
    
def visualize_reconstruction(original_image, device, acc=4, batch_size=1, num_coils=8, acs_lines=32, kernel_size=(3, 3), operator=GRAPPAOperator, title="Reconstruction", cmap='gray', save_path=None):
    """
    Visualize MRI reconstruction pipeline with original, k-space, and reconstructed images.
    
    Args:
        original_image: Input image tensor
        acc: Acceleration factor
        batch_size: Batch size for operator
        operator: MRIOperator class
        title: Plot title
        cmap: Colormap
        save_path: Path to save figure
    """
    if operator == MRIOperator:
        operator = operator(acc=acc, device=device, batch_size=batch_size)
    elif operator == GRAPPAOperator or operator == GRAPPAOperator_v1:
        operator = operator(acc=acc, device=device, num_coils=num_coils, acs_lines=acs_lines, kernel_size=kernel_size)

    original_image = original_image.to(device)  # Ensure image is on the correct device
    
    measurement = operator.forward(original_image) # Forward operator to k-space (measurement)
    
    reconstructed = operator.transpose(measurement) # Zero-filled reconstruction

    # Convert to numpy for visualization
    if torch.is_tensor(original_image):
        original = original_image.detach().cpu().numpy().squeeze()
    
    if torch.is_tensor(measurement): 
        measurement = measurement.detach().cpu().numpy().squeeze()
        measurement = np.abs(measurement) if np.iscomplexobj(measurement) else measurement
        if len(measurement.shape) == 3:
            measurement = np.sqrt(np.sum(measurement**2, axis=0))
    
    
    if torch.is_tensor(reconstructed):
        reconstructed = reconstructed.detach().cpu().numpy().squeeze()
        reconstructed = np.abs(reconstructed) if np.iscomplexobj(reconstructed) else reconstructed

    # Calculate difference
    difference = np.abs(original - reconstructed)
       
    
    # Calculate vmin/vmax for better contrast
    vmin = np.percentile(difference, 5)   # 5th percentile instead of min
    vmax = np.percentile(difference, 95)  # 95th percentile instead of max

    # Create subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original Image 
    axes[0, 0].imshow(original, cmap=cmap)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    # K-space measurement (magnitude, log scale) 
    axes[0, 1].imshow(np.log(measurement + 1e-8), cmap=cmap)
    axes[0, 1].set_title("K-space Measurement (Log Scale)")
    axes[0, 1].axis('off')

    # K-space measurement (magnitude, natural log scale) 
    axes[0, 2].imshow(np.log1p(measurement + 1e-8), cmap='gray', aspect='auto')
    axes[0, 2].set_title("K-space Measurement (Natural Log Scale)")
    axes[0, 2].axis('off')

    # Zero-filled Reconstructed Image 
    axes[1, 0].imshow(reconstructed, cmap=cmap)
    axes[1, 0].set_title("Zero-Filled Reconstruction")
    axes[1, 0].axis('off')

    # Difference Image with improved colormap and scaling
    im_diff = axes[1, 1].imshow(difference, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
    axes[1, 1].set_title("Reconstruction Error")
    axes[1, 1].axis('off')
    plt.colorbar(im_diff, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Sampling profile (1D visualization of mask)
    mask_np = operator.mask.squeeze().detach().cpu().numpy() 
    profile = np.sum(mask_np, axis=1) if len(mask_np.shape) > 1 else mask_np
    axes[1, 2].plot(profile)
    axes[1, 2].set_title("Sampling Profile")
    axes[1, 2].set_xlabel("Phase Encoding Line")
    axes[1, 2].set_ylabel("Sampled Points")
    axes[1, 2].grid(True)

    # Calculate sampling ratio
    sampling_ratio = operator.mask.sum().item() / operator.mask.numel()
    
    fig.suptitle(f'{title} (Acceleration: {acc}x, Sampling: {sampling_ratio:.2%})', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved reconstruction visualization to {save_path}")
    
    plt.show()

    # Print statistics
    print(f"=== Reconstruction Statistics ===")
    print(f"Original image - Min: {original.min():.4f}, Max: {original.max():.4f}")
    print(f"Reconstructed image - Min: {reconstructed.min():.4f}, Max: {reconstructed.max():.4f}")
    print(f"Mean absolute error: {np.mean(difference):.4f}")
    print(f"Max absolute error: {np.max(difference):.4f}")
    print(f"PSNR: {20 * np.log10(np.max(original) / np.sqrt(np.mean(difference**2))):.2f} dB")
    print(f"Sampling ratio: {sampling_ratio:.2%}")
    
    return original, reconstructed, difference

def compare_acceleration_factors(image, device, acc_factors=[1, 2, 4, 8], batch_size=1, num_coils=8, acs_lines=32, kernel_size=(3, 3), operator=GRAPPAOperator):
    """
    Compare reconstruction quality at different acceleration factors.
    """
    fig, axes = plt.subplots(2, len(acc_factors), figsize=(16, 8))
    
    # Convert image to proper format if needed
    if len(image.shape) == 3:
        image = image.unsqueeze(0)  # Add batch dimension

    # Ensure image is on the correct device
    image = image.to(device)
    orig_np = image.detach().cpu().numpy().squeeze()

    if operator == MRIOperator:
        operator = operator(acc=acc, device=device.device, batch_size=batch_size)
    elif operator == GRAPPAOperator or operator == GRAPPAOperator_v1:
        operator = operator(acc=acc, device=device.device, num_coils=num_coils, acs_lines=acs_lines, kernel_size=kernel_size)


    # First pass: collect all error images to determine global vmin/vmax
    error_maps = []
    recon_maps = []
    for acc in acc_factors:
        measurement = operator.forward(image)
        reconstructed = operator.transpose(measurement)
        recon_np = reconstructed.detach().cpu().numpy().squeeze()
        if np.iscomplexobj(recon_np):
            recon_np = np.abs(recon_np)
            recon_np = recon_np - recon_np.min()
            recon_np = recon_np / recon_np.max()
        diff = np.abs(orig_np - recon_np)
        error_maps.append(diff)
        recon_maps.append(recon_np)
    
    error_maps = np.array(error_maps)
    
    # Use percentile-based scaling for better contrast
    vmin = np.percentile(error_maps, 5)   # 5th percentile instead of min
    vmax = np.percentile(error_maps, 95)  # 95th percentile instead of max
    
    # Second pass: plot with consistent color scale
    for i, (acc, recon_np, diff) in enumerate(zip(acc_factors, recon_maps, error_maps)):
        axes[0, i].imshow(recon_np, cmap='gray')
        axes[0, i].set_title(f'Acceleration {acc}x')
        axes[0, i].axis('off')
        
        # Better colormap options for error visualization
        # im_diff = axes[1, i].imshow(diff, cmap='magma', vmin=vmin, vmax=vmax)  # Dark to bright yellow/white
        # Alternative options:
        # im_diff = axes[1, i].imshow(diff, cmap='inferno', vmin=vmin, vmax=vmax)  # Dark to bright red/yellow
        # im_diff = axes[1, i].imshow(diff, cmap='jet', vmin=vmin, vmax=vmax)     # Classic blue to red
        # im_diff = axes[1, i].imshow(diff, cmap='turbo', vmin=vmin, vmax=vmax)   # Modern rainbow
        im_diff = axes[1, i].imshow(diff, cmap='RdYlBu_r', vmin=vmin, vmax=vmax) # Red-Yellow-Blue reversed
        
        axes[1, i].set_title(f'Error (MAE: {np.mean(diff):.4f})')
        axes[1, i].axis('off')
        plt.colorbar(im_diff, ax=axes[1, i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()
    plt.tight_layout()
    plt.show()

def visualize_operators_reconstruction(original_image, device, acc=4, batch_size=1, num_coils=8, acs_lines=32, kernel_size=(3, 3), operator=GRAPPAOperator, title="MRI vs GRAPPA Reconstruction", cmap='gray', save_path=None):
    """
    Visualize MRI reconstruction pipeline comparing MRI and GRAPPA operators.
    
    Args:
        original_image: Input image tensor
        acc: Acceleration factor
        batch_size: Batch size for operator
        operator: MRIOperator class
        title: Plot title
        cmap: Colormap
        save_path: Path to save figure
    """
    mri_operator = MRIOperator(acc=acc, device=device, batch_size=batch_size)
    grappa_operator = GRAPPAOperator(acc=acc, device=device, num_coils=num_coils, acs_lines=acs_lines, kernel_size=kernel_size)

    original_image = original_image.to(device)

    # Apply operators
    mri_measurement = mri_operator.forward(original_image)
    mri_reconstruction = mri_operator.transpose(mri_measurement)
    grappa_measurement = grappa_operator.forward(original_image)
    grappa_reconstruction = grappa_operator.transpose(grappa_measurement)

    # Convert to numpy for visualization
    original = original_image.detach().cpu().numpy().squeeze()
    
    mri_measurement = mri_measurement.detach().cpu().numpy().squeeze()
    mri_measurement = np.abs(mri_measurement) if np.iscomplexobj(mri_measurement) else mri_measurement
    
    mri_reconstruction = mri_reconstruction.detach().cpu().numpy().squeeze()
    mri_reconstruction = np.abs(mri_reconstruction) if np.iscomplexobj(mri_reconstruction) else mri_reconstruction
    
    grappa_measurement = grappa_measurement.detach().cpu().numpy().squeeze()
    grappa_measurement = np.abs(grappa_measurement) if np.iscomplexobj(grappa_measurement) else grappa_measurement
    # For multi-coil data, take RSS across coils
    if len(grappa_measurement.shape) == 3:
        grappa_measurement = np.sqrt(np.sum(grappa_measurement**2, axis=0))
    
    grappa_reconstruction = grappa_reconstruction.detach().cpu().numpy().squeeze()
    grappa_reconstruction = np.abs(grappa_reconstruction) if np.iscomplexobj(grappa_reconstruction) else grappa_reconstruction

    # Calculate errors
    difference_mri = np.abs(original - mri_reconstruction)
    difference_grappa = np.abs(original - grappa_reconstruction)
    
    # Global error bounds for consistent comparison
    all_errors = np.concatenate([difference_mri.flatten(), difference_grappa.flatten()])
    vmin_global = np.percentile(all_errors, 5)
    vmax_global = np.percentile(all_errors, 95)

    # Create figure with better spacing
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.patch.set_facecolor('white')
    
    # Calculate sampling ratio
    sampling_ratio = grappa_operator.mask.sum().item() / grappa_operator.mask.numel()
    
    # Main title with statistics
    fig.suptitle(f'{title}\nAcceleration: {acc}x | Sampling: {sampling_ratio:.1%} | MAE (MRI): {np.mean(difference_mri):.4f} | MAE (GRAPPA): {np.mean(difference_grappa):.4f}', 
                 fontsize=16, fontweight='bold', y=0.95)

    # Row 1: MRI Operator
    row_title_props = dict(fontsize=14, fontweight='bold', color='darkblue')
    
    # Original Image (shared)
    axes[0, 0].imshow(original, cmap=cmap)
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # MRI K-space
    axes[0, 1].imshow(np.log(mri_measurement + 1e-8), cmap='viridis', aspect='equal')
    axes[0, 1].set_title("MRI K-space\n(Log Scale)", fontsize=12)
    axes[0, 1].axis('off')
    
    # MRI Reconstruction
    axes[0, 2].imshow(mri_reconstruction, cmap=cmap)
    axes[0, 2].set_title("MRI Zero-Filled\nReconstruction", fontsize=12)
    axes[0, 2].axis('off')
    
    # MRI Error
    im_diff_mri = axes[0, 3].imshow(difference_mri, cmap='RdYlBu_r', vmin=vmin_global, vmax=vmax_global)
    axes[0, 3].set_title("MRI Reconstruction\nError", fontsize=12)
    axes[0, 3].axis('off')
    
    # Add colorbar for MRI error
    cbar_mri = plt.colorbar(im_diff_mri, ax=axes[0, 3], fraction=0.046, pad=0.04)
    cbar_mri.set_label('Error Magnitude', fontsize=10)

    # Row 2: GRAPPA Operator
    # Original Image (shared)
    axes[1, 0].imshow(original, cmap=cmap)
    axes[1, 0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # GRAPPA K-space
    axes[1, 1].imshow(np.log(grappa_measurement + 1e-8), cmap='viridis', aspect='equal')
    axes[1, 1].set_title("GRAPPA K-space\n(RSS Combined, Log Scale)", fontsize=12)
    axes[1, 1].axis('off')
    
    # GRAPPA Reconstruction
    axes[1, 2].imshow(grappa_reconstruction, cmap=cmap)
    axes[1, 2].set_title("GRAPPA\nReconstruction", fontsize=12)
    axes[1, 2].axis('off')
    
    # GRAPPA Error
    im_diff_grappa = axes[1, 3].imshow(difference_grappa, cmap='RdYlBu_r', vmin=vmin_global, vmax=vmax_global)
    axes[1, 3].set_title("GRAPPA Reconstruction\nError", fontsize=12)
    axes[1, 3].axis('off')
    
    # Add colorbar for GRAPPA error
    cbar_grappa = plt.colorbar(im_diff_grappa, ax=axes[1, 3], fraction=0.046, pad=0.04)
    cbar_grappa.set_label('Error Magnitude', fontsize=10)

    # Add method labels on the left
    fig.text(0.02, 0.75, 'MRI Operator', rotation=90, fontsize=14, fontweight='bold', 
             color='darkblue', ha='center', va='center')
    fig.text(0.02, 0.25, 'GRAPPA Operator', rotation=90, fontsize=14, fontweight='bold', 
             color='darkred', ha='center', va='center')

    # Adjust layout
    plt.subplots_adjust(left=0.05, right=0.95, top=0.87, bottom=0.05, hspace=0.3, wspace=0.15)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved reconstruction visualization to {save_path}")
    
    plt.show()
    
    # Print detailed statistics
    print(f"\n{'='*60}")
    print(f"RECONSTRUCTION COMPARISON STATISTICS")
    print(f"{'='*60}")
    print(f"Acceleration Factor: {acc}x")
    print(f"Sampling Ratio: {sampling_ratio:.2%}")
    print(f"\nMRI Operator (Zero-Filled):")
    print(f"  - Mean Absolute Error: {np.mean(difference_mri):.6f}")
    print(f"  - Max Absolute Error:  {np.max(difference_mri):.6f}")
    print(f"  - PSNR: {20 * np.log10(np.max(original) / np.sqrt(np.mean(difference_mri**2))):.2f} dB")
    print(f"\nGRAPPA Operator:")
    print(f"  - Mean Absolute Error: {np.mean(difference_grappa):.6f}")
    print(f"  - Max Absolute Error:  {np.max(difference_grappa):.6f}")
    print(f"  - PSNR: {20 * np.log10(np.max(original) / np.sqrt(np.mean(difference_grappa**2))):.2f} dB")
    print(f"\nImprovement (GRAPPA vs MRI):")
    mae_improvement = ((np.mean(difference_mri) - np.mean(difference_grappa)) / np.mean(difference_mri)) * 100
    print(f"  - MAE Reduction: {mae_improvement:.1f}%")
    print(f"{'='*60}")


"""Single Image Visualization Functions"""
def visualize_single_image(image, title="Single Image", cmap='gray', save_path=None):
    # Check if the image is a tensor and convert to numpy
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    
    # Remove channel dimension if exists
    if len(image.shape) == 4:
        image = image.squeeze(0).image.squeeze(0)
    elif len(image.shape) == 3:
        image = image.squeeze(0)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()

def visualize_single_measurement(measurement, title="Single K-space Measurement", cmap='gray', save_path=None):
    """
    Visualize a single k-space measurement.
    
    Args:
        measurement: K-space measurement tensor
        title: Plot title
        cmap: Colormap
        normalize: Whether to normalize the image
        save_path: Path to save figure
    """
    if torch.is_tensor(measurement):
        measurement = measurement.detach().cpu().numpy()
    
    if len(measurement.shape) == 4:
        measurement = measurement.squeeze(0).squeeze(0)
    if len(measurement.shape) == 3:
        measurement = np.sqrt(np.sum(measurement**2, axis=0))

    measurement = np.abs(measurement) if np.iscomplexobj(measurement) else measurement
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))


    # K-space measurement (magnitude, log scale) 
    axes[0].imshow(np.log(measurement + 1e-8), cmap=cmap)
    axes[0].set_title("K-space Measurement (Log Scale)")
    axes[0].axis('off')

    # K-space measurement (magnitude, natural log scale) 
    axes[1].imshow(np.log1p(measurement + 1e-8), cmap='gray', aspect='equal')
    axes[1].set_title("K-space Measurement (Natural Log Scale)")
    axes[1].axis('off')

    fig.suptitle(title, fontsize=16)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()

def visualize_single_reconstructed(reconstructed, title="MRI Reconstruction", cmap='gray', save_path=None):
    if torch.is_tensor(reconstructed):
        reconstructed = reconstructed.detach().cpu().numpy()

    if len(reconstructed.shape) == 4:
        reconstructed = reconstructed.squeeze(0)
    elif len(reconstructed.shape) == 3:
        reconstructed = reconstructed.squeeze(0)

    reconstructed = np.abs(reconstructed) if np.iscomplexobj(reconstructed) else reconstructed

    reconstructed = reconstructed - reconstructed.min()  # Normalize to [0, 1]
    reconstructed = reconstructed / reconstructed.max()

    plt.figure(figsize=(6, 6))
    plt.imshow(reconstructed, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    plt.show()



if __name__ == "__main__":
    # Example Data Load
    source_dir = MRI_DIR 
    max_images = 50
    target_size = (320, 320)  # Resize to 320x320
    data_type = "tensor"  # or "numpy"
    
    images = load_img_to_tensor(source_dir, max_images=max_images, target_size=target_size, data_type=data_type)
    
    print(f"Loaded {len(images)} images of shape {images.shape}")
    
    visualize_images(images, num_images=16, figsize=(12, 12), title="Sample MRI Images", normalize=True)

    # Example MRIOperator Usage
    acc = 4  # Acceleration factor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    operator = MRIOperator(acc=acc, device=device, batch_size=1)
    sample_data = images[:1]  # Take one sample for demonstration
    k_space_data = operator.forward(sample_data)
    reconstructed_data = operator.transpose(k_space_data)
    print(f"K-space data shape: {k_space_data.shape}")
    print(f"Reconstructed data shape: {reconstructed_data.shape}")


     # Visualize reconstruction
    recon = visualize_single_reconstructed(images[0], acc=4, batch_size=1, operator=MRIOperator, title="MRI Reconstruction Example", cmap='gray', save_path=None)