"""
Dataset Functions for kspace qMRI dataset. 

- K-space data provided by the Washington University in St. Louis Radiology Department and the Yablonskiy Lab.  
"""

import torch 
from torch.utils.data import Dataset 
import numpy as np 
import os 
import SimpleITK as sitk 
from utils.fourier import ifft2c, fft2c

"""
K-space Normalization Functions

(1) normalize_kspace: Normalize k-space by its maximum absolute value.
(2) center_region_ksp_norm: Normalize k-space by the maximum absolute value in the center region.
(3) apply_n4_correction: Apply N4 bias field correction to a numpy image array using SimpleITK.
"""

def normalize_kspace(ksp):
    """Normalize k-space by its maximum absolute value."""
    norm = torch.abs(ksp).max()
    return ksp / norm if norm > 0 else ksp 

def center_region_ksp_norm(ksp, region_size=48):
    h, w = ksp.shape[-2:]
    center_h_start = h // 2 - region_size // 2
    center_w_start = w // 2 - region_size // 2
    center_h_end = h // 2 + region_size // 2
    center_w_end = w // 2 + region_size // 2

    center_region = ksp[..., center_h_start:center_h_end, center_w_start:center_w_end]

    norm = torch.abs(center_region).max()

    return ksp / norm if norm > 0 else ksp 

def apply_n4_correction(image_np):
    """Apply N4 bias field correction to a numpy image array using SimpleITK."""
    image_sitk = sitk.GetImageFromArray(image_np)
    mask_sitk = sitk.OtsuThreshold(image_sitk, 0, 1, 200)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_image_sitk = corrector.Execute(image_sitk, mask_sitk)
    corrected_image_np= sitk.GetArrayFromImage(corrected_image_sitk)

    log_bias_field = corrector.GetLogBiasFieldAsImage(image_sitk)
    bias_field_np = np.exp(sitk.GetArrayFromImage(log_bias_field))

    return corrected_image_np, bias_field_np

"""
Dataset Objects 

(1) MRIEchoDataset: Dataset for multi-echo qMRI data
(2) MRIEchoBatchDataset: Dataset for multi-echo qMRI data with separate batched dim for echoes
"""

class MRIEchoDataset(Dataset):
    """Dataset for multi-echo qMRI data."""
    def __init__(self, h5_dir, subjects, echos=range(1, 16, 2), train=True, correction=False):
        self.h5_dir = h5_dir
        self.subjects = subjects
        self.echos = echos
        self.train = train
        self.correction = correction

        self.pt_file_paths = [] 

        for subject in subjects: 
            for echo in echos: 
                if train: 
                    for slice_idx in range(90, 110): 
                        path = os.path.join(h5_dir, subject, f"echo{echo}", f"slice_{slice_idx:03d}.pt")
                        self.pt_file_paths.append(path)
                else:
                    # For Validation/Visualization
                    for slice_idx in range(70, 140): 
                        path = os.path.join(h5_dir, subject, f"echo{echo}", f"slice_{slice_idx:03d}.pt")
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
            rss_image_np = rss_image.numpy()

            # Bias correction using SimpleITK
            _, bias_field_np = apply_n4_correction(rss_image_np)
            bias_field = torch.from_numpy(bias_field_np).to(ksp_original.device).unsqueeze(0)

            # Corrected k-space 
            corrected_coil_images = coil_images / (bias_field + 1e-8)
            corrected_ksp_gt = fft2c(corrected_coil_images)
            corrected_ksp_gt = torch.view_as_real(corrected_ksp_gt)
            
            data_dict["ksp_gt_corrected"] = corrected_ksp_gt    

        data_dict["ksp_gt"] = torch.view_as_real(data["ksp_gt"])
        data_dict["ksp_4x"] = torch.view_as_real(data["ksp_4x"])
        data_dict["sens_map"] = torch.view_as_real(data["sens_map"])
        return data_dict

class MRIEchoBatchDataset(Dataset):
    """Dataset for multi-echo qMRI data with separate batched dim for echoes."""
    def __init__(self, h5_dir, subjects, echos=range(1, 16, 2), train=True):
        self.h5_dir = h5_dir
        self.subjects = subjects
        self.echos = echos
        self.train = train

        slice_range = (90, 110) if train else (70, 140)
        self.slice_start, self.slice_end = slice_range
        self.slice_indices = list(range(self.slice_start, self.slice_end))

        self.num_slices = len(self.slice_indices)
        self.total_length = len(self.subjects) * self.num_slices

    def __len__(self):
        return self.total_length

    def _get_subject_and_slice(self, idx):
        """Convert linear index to (subject_idx, slice_idx)."""
        subject_idx = idx // self.num_slices 
        slice_position = idx % self.num_slices
        slice_idx = self.slice_indices[slice_position]
        return subject_idx, slice_idx

    def __getitem__(self, idx):
        subject_idx, slice_idx = self._get_subject_and_slice(idx)
        subject = self.subjects[subject_idx]

        echo_data = [] 

        # Collect data for all echos of this slice from this subject 
        for echo in self.echos:
            path = os.path.join(self.h5_dir, subject, f"echo{echo}", f"slice_{slice_idx:03d}.pt")

            try: 
                with open(path, 'rb') as f:
                    data = torch.load(f, weights_only=True)

                echo_data.append({
                    "ksp_gt": torch.view_as_real(data["ksp_gt"]),
                    "ksp_4x": torch.view_as_real(data["ksp_4x"]),
                    "sens_map": torch.view_as_real(data["sens_map"])
                })
    
            except Exception as e:
                print(f"Error loading file {path}: {e}")

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
            "ksp_gt": ksp_gt,
            "ksp_4x": ksp_4x,
            "sens_map": sens_map, 
            "subject": subject,
            "slice_idx": slice_idx,
            "linear_idx": idx 
        }

    def get_subject_slice_info(self, idx):
        """Get subject and slice index for a given dataset index."""
        subject_idx, slice_idx = self._get_subject_and_slice(idx)
        subject = self.subjects[subject_idx]
        return subject, slice_idx
        