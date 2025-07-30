import torch
from torch.utils.data import Dataset
import numpy as np
import os
import SimpleITK as sitk
from utils.util import *

def normalize_kspace(ksp):
    norm = torch.abs(ksp).max()
    return ksp / norm if norm > 0 else ksp

def center_region_ksp_norm(ksp, region_size=48):
    h, w = ksp.shape[-2:]
    center_h_start = h // 2 - region_size // 2
    center_h_end = h // 2 + region_size // 2
    center_w_start = w // 2 - region_size // 2
    center_w_end = w // 2 + region_size // 2
    
    center_region = ksp[..., center_h_start:center_h_end, center_w_start:center_w_end]
    norm = torch.abs(center_region).max()
    
    return ksp / norm if norm > 0 else ksp

def apply_n4_correction(image_np: np.ndarray):
    """Applies N4 bias field correction using SimpleITK."""
    image_sitk = sitk.GetImageFromArray(image_np)
    mask_sitk = sitk.OtsuThreshold(image_sitk, 0, 1, 200)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_sitk = corrector.Execute(image_sitk, mask_sitk)
    corrected_image_np = sitk.GetArrayFromImage(corrected_sitk)
    
    log_bias_field = corrector.GetLogBiasFieldAsImage(image_sitk)
    bias_field_np = np.exp(sitk.GetArrayFromImage(log_bias_field))
    
    return corrected_image_np, bias_field_np

def get_mask():
    mask = torch.from_numpy(np.load("./data/mask4x.npy"))
    return mask

class MRIEchoDataset(Dataset):
    def __init__(self, h5_dir, subjects, echos = range(1,16,2), train=True, correction=False):
        self.correction = correction
        self.pt_file_paths = []            
        for sub in subjects:
            for echo in echos:
                # For training 
                if train: 
                    for slice_idx in range(90, 110): 
                        path = os.path.join(h5_dir, sub, f"echo{echo}", f"slice_{slice_idx:03d}.pt")
                        self.pt_file_paths.append(path)
                else:
                    # For Validation/Visualization
                    for slice_idx in range(70, 140): 
                        path = os.path.join(h5_dir, sub, f"echo{echo}", f"slice_{slice_idx:03d}.pt")
                        self.pt_file_paths.append(path)

    def __len__(self):
        return len(self.pt_file_paths)

    def __getitem__(self, idx):
        with open(self.pt_file_paths[idx], 'rb') as f:
            data = torch.load(f, weights_only=True)

        data_dict = {} 
        if self.correction:
            ksp_original = data["ksp_gt"]   

            # Bias Correction 
            coil_images = ifft2c(torch.view_as_real(ksp_original))
            rss_image = torch.sqrt(torch.sum(torch.square(torch.abs(coil_images)), dim=0))
            rss_img_np = rss_image.numpy()
            _, bias_field_np = apply_n4_correction(rss_img_np)
            bias_field = torch.from_numpy(bias_field_np).to(coil_images.device) 
            corrected_coil_images = coil_images / (bias_field.unsqueeze(0) + 1e-8) 
            corrected_ksp_gt = fft2c(corrected_coil_images)
            corrected_ksp_gt = torch.view_as_complex(corrected_ksp_gt)

            data_dict["ksp_gt_cor"] = torch.view_as_real(corrected_ksp_gt)

        data_dict["ksp_gt"] = torch.view_as_real(data["ksp_gt"])
        data_dict["ksp_4x"] = torch.view_as_real(data["ksp_4x"])
        data_dict["sens_map"] = torch.view_as_real(data["sens_map"])
        return data_dict



class MRIEchoDatasetBatch(Dataset):
    def __init__(self, h5_dir, subjects, echos=range(1, 16, 2), train=True):
        """
        Dataset where each item is one slice from one subject with all echos stacked.
        
        Organization:
        - Index 0: Subject 0, Slice 70, All echos
        - Index 1: Subject 0, Slice 71, All echos
        - ...
        - Index 69: Subject 0, Slice 139, All echos
        - Index 70: Subject 1, Slice 70, All echos
        - Index 71: Subject 1, Slice 71, All echos
        - etc.
        
        Args:
            h5_dir: Directory containing the data
            subjects: List of subject IDs
            echos: Range/list of echo numbers to include
            slice_range: Tuple of (start_slice, end_slice) - end is exclusive
            
        Returns:
            Each item has tensors of shape [num_echos, coils, H, W, 2]
        """
        self.h5_dir = h5_dir
        self.subjects = subjects
        self.echos = list(echos)
        slice_range = (90, 110) if train else (70, 90)  # Adjust based on train/val
        self.slice_start, self.slice_end = slice_range
        self.slice_indices = list(range(self.slice_start, self.slice_end))
        
        self.num_slices = len(self.slice_indices)
        self.total_length = len(self.subjects) * self.num_slices
        
    def __len__(self):
        return self.total_length
    
    def _get_subject_and_slice(self, idx):
        """Convert linear index to (subject_idx, slice_idx)"""
        subject_idx = idx // self.num_slices
        slice_position = idx % self.num_slices
        slice_idx = self.slice_indices[slice_position]
        return subject_idx, slice_idx
    
    def __getitem__(self, idx):
        """
        Get one slice from one subject with all echos stacked.
        
        Args:
            idx: Linear index
            
        Returns:
            Dictionary with:
            - 'ksp_gt': [num_echos, coils, H, W, 2]
            - 'ksp_4x': [num_echos, coils, H, W, 2] 
            - 'sens_map': [num_echos, coils, H, W, 2]
            - 'subject': subject ID string
            - 'slice_idx': slice number
        """
        subject_idx, slice_idx = self._get_subject_and_slice(idx)
        subject = self.subjects[subject_idx]
        
        echo_data = []
        
        # Collect data for all echos of this slice from this subject
        for echo in self.echos:
            file_path = os.path.join(self.h5_dir, subject, f"echo{echo}", f"slice_{slice_idx:03d}.pt")
            
            try:
                with open(file_path, 'rb') as f:
                    data = torch.load(f, weights_only=True)
                
                echo_data.append({
                    "ksp_gt": torch.view_as_real(data["ksp_gt"]),
                    "ksp_4x": torch.view_as_real(data["ksp_4x"]),
                    "sens_map": torch.view_as_real(data["sens_map"]),
                })
                
            except FileNotFoundError:
                print(f"Warning: Missing file {file_path}")
                # Create zero tensor with expected shape
                if len(echo_data) > 0:
                    # Use shape from previous echo
                    zero_shape = echo_data[0]["ksp_gt"].shape
                else:
                    # Default shape - adjust based on your data
                    zero_shape = (64, 234, 176, 2)  # [coils, H, W, real_imag]
                
                echo_data.append({
                    "ksp_gt": torch.zeros(zero_shape),
                    "ksp_4x": torch.zeros(zero_shape),
                    "sens_map": torch.zeros(zero_shape),
                })
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                # Handle other errors similarly
                if len(echo_data) > 0:
                    zero_shape = echo_data[0]["ksp_gt"].shape
                else:
                    zero_shape = (64, 234, 176, 2)
                
                echo_data.append({
                    "ksp_gt": torch.zeros(zero_shape),
                    "ksp_4x": torch.zeros(zero_shape),
                    "sens_map": torch.zeros(zero_shape),
                })
        
        # Stack all echos for this slice
        ksp_gt = torch.stack([echo["ksp_gt"] for echo in echo_data], dim=0)
        ksp_4x = torch.stack([echo["ksp_4x"] for echo in echo_data], dim=0)
        sens_map = torch.stack([echo["sens_map"] for echo in echo_data], dim=0)
        
        return {
            "ksp_gt": ksp_gt,        # [num_echos, coils, H, W, 2]
            "ksp_4x": ksp_4x,        # [num_echos, coils, H, W, 2]
            "sens_map": sens_map,    # [num_echos, coils, H, W, 2]
            "subject": subject,      # Subject ID string
            "slice_idx": slice_idx,  # Slice number
            "linear_idx": idx        # Original linear index for debugging
        }
    
    def get_subject_slice_info(self, idx):
        """Helper function to see what subject/slice an index corresponds to"""
        subject_idx, slice_idx = self._get_subject_and_slice(idx)
        subject = self.subjects[subject_idx]
        return f"Index {idx}: {subject}, Slice {slice_idx}"
