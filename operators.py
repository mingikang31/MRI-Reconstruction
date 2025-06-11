import numpy as np 
import torch 
from scipy.linalg import solve

from operators_utils import *

# GLOBAL VARIABLES 

MRI_DIR = "/export1/project/mingi/Dataset/brainMRI"
OUT_DIR = "" # directory for saving transformed images # Not in use rn #
SAVE_PLOT_DIR = ""

"""Transformation Operators for MRI Data"""
### Grappa Operator Classes 

"""Transformation Operators for MRI Data"""
### Grappa Operator Classes 
class GRAPPAOperator(LinearOperator):
    """
    GRAPPAOperator performs multi-coil forward simulation and GRAPPA reconstruction.
    """
    def __init__(self, acc, device, num_coils=8, acs_lines=32, kernel_size=(3, 3), batch_size=1):
        super(GRAPPAOperator, self).__init__()
        self.acc = acc
        self.device = device
        self.num_coils = num_coils
        self.acs_lines = acs_lines
        self.kernel_size = kernel_size
        self.mask = get_mask(batch_size=batch_size, R=self.acc, acs_lines=self.acs_lines).to(device)

    def create_synthetic_csms(self, h, w):
        csms = torch.zeros((self.num_coils, h, w), device=self.device, dtype=torch.complex64)
        for i in range(self.num_coils):
            x_phase = torch.linspace(-np.pi, np.pi, w, device=self.device) * (i - self.num_coils / 2)
            y_phase = torch.linspace(-np.pi, np.pi, h, device=self.device) * (i - self.num_coils / 2)
            y_grid, x_grid = torch.meshgrid(y_phase, x_phase, indexing='ij')
            csms[i, :, :] = torch.exp(1j * (x_grid + y_grid))
        return csms

    def forward(self, data, **kwargs):
        if data.dim() == 3:
            data = data.unsqueeze(0)
        if data.shape[1] == 1:
            data = data.squeeze(1)

        batch_size, height, width = data.shape
        csms = self.create_synthetic_csms(height, width)
        full_k_space = torch.zeros(
            (batch_size, self.num_coils, height, width),
            device=self.device,
            dtype=torch.complex64
        )
        for i in range(batch_size):
            img_complex = torch.complex(data[i], torch.zeros_like(data[i]))
            for c in range(self.num_coils):
                coil_img = img_complex * csms[c]
                full_k_space[i, c, :, :] = fft2_m(coil_img)

        mask_expanded = self.mask.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_coils, -1, width)
        undersampled_kspace = full_k_space * mask_expanded
        return undersampled_kspace

    def transpose(self, undersampled_kspace, **kwargs):
        batch_size, num_coils, height, width = undersampled_kspace.shape
        kh, kw = self.kernel_size

        # 1) Initialize reconstructed_kspace with the undersampled data
        reconstructed_kspace = undersampled_kspace.clone()

        # 2) Extract the ACS region (centered in ky)
        center_h = height // 2
        acs_start = center_h - self.acs_lines // 2
        acs_end = center_h + self.acs_lines // 2
        acs_region = undersampled_kspace[:, :, acs_start:acs_end, :]  # (B, C, num_acs_lines, W)
        num_acs_lines, acs_width = acs_region.shape[2], acs_region.shape[3]

        # 3) Build source/target matrices from ACS
        source_patches = []
        target_patches = []
        for y_acs in range(num_acs_lines - (kh - 1) * self.acc):
            for x_acs in range(acs_width - kw):
                # (a) Source patch: size (B, num_coils, kh, kw)
                src = torch.zeros((batch_size, num_coils, kh, kw),
                                  device=self.device, dtype=torch.complex64)
                for i in range(kh):
                    ky_index = y_acs + i * self.acc
                    src[:, :, i, :] = acs_region[:, :, ky_index, x_acs : x_acs + kw]

                source_patches.append(src.reshape(batch_size, -1))

                # (b) Target: center of the (kh×kw) window
                ky_center = y_acs + (kh // 2) * self.acc
                kx_center = x_acs + (kw // 2)
                tgt = acs_region[:, :, ky_center, kx_center]  # (B, num_coils)
                target_patches.append(tgt)

        Source = torch.stack(source_patches, dim=1)  # (B, #patches, features)
        Target = torch.stack(target_patches, dim=1)  # (B, #patches, num_coils)

        # 4) Solve for W using normal equations + Tikhonov regularization
        S_H = Source.permute(0, 2, 1).conj()                   # (B, features, #patches)
        SHS = torch.matmul(S_H, Source)                        # (B, features, features)
        SHT = torch.matmul(S_H, Target)                        # (B, features, num_coils)

        λ = 1e-3  # small regularization constant
        I = torch.eye(SHS.shape[-1], device=SHS.device).unsqueeze(0)  # (1, features, features)
        W = torch.linalg.solve(SHS + λ * I, SHT)                # (B, features, num_coils)

        # 5) Reconstruct missing k-space lines
        y_min = (kh // 2) * self.acc
        y_max = height - (kh // 2) * self.acc
        for y in range(y_min, y_max):
            # Check a single column to see if that entire ky row was acquired
            if self.mask[0, y] == 1:
                continue  # fully sampled row, no need to interpolate

            for x in range(kw // 2, width - (kw // 2)):
                # Build source patch around (y, x)
                src_patch = torch.zeros((batch_size, num_coils, kh, kw),
                                        device=self.device, dtype=torch.complex64)
                for i in range(kh):
                    ky_i = y + (i - kh // 2) * self.acc
                    src_patch[:, :, i, :] = reconstructed_kspace[:, :, ky_i,
                                                   x - (kw // 2) : x + (kw // 2) + 1]

                src_vec = src_patch.reshape(batch_size, 1, -1)  # (B, 1, features)
                rec_pt = torch.matmul(src_vec, W)               # (B, 1, num_coils)
                reconstructed_kspace[:, :, y, x] = rec_pt.squeeze(1)

        # 6) Inverse FFT + root-sum-of-squares coil combine
        reconstructed_images = torch.zeros((batch_size, height, width),
                                           device=self.device, dtype=torch.float32)
        for i in range(batch_size):
            coil_images = ifft2_m(reconstructed_kspace[i])  # (num_coils, H, W)
            rss = torch.sqrt(torch.sum(torch.abs(coil_images) ** 2, dim=0))  # (H, W)
            reconstructed_images[i] = rss

        # 7) Add channel dimension back → (B, 1, H, W)
        return reconstructed_images.unsqueeze(1)

class GRAPPAOperator_v1(LinearOperator):
    """
    GRAPPAOperator performs multi-coil forward simulation and GRAPPA reconstruction.
    """
    def __init__(self, acc, device, num_coils=8, acs_lines=32, kernel_size=(3, 3)):
        super(GRAPPAOperator_v1, self).__init__()
        self.acc = acc
        self.device = device
        self.num_coils = num_coils
        self.acs_lines = acs_lines
        self.kernel_size = kernel_size
        self.mask = get_mask(batch_size=1, R=self.acc, acs_lines=self.acs_lines).to(device)

    def create_synthetic_csms(self, h, w):
        csms = torch.zeros((self.num_coils, h, w), device=self.device, dtype=torch.complex64)
        for i in range(self.num_coils):
            x_phase = torch.linspace(-np.pi, np.pi, w, device=self.device) * (i - self.num_coils / 2)
            y_phase = torch.linspace(-np.pi, np.pi, h, device=self.device) * (i - self.num_coils / 2)
            y_grid, x_grid = torch.meshgrid(y_phase, x_phase, indexing='ij')
            csms[i, :, :] = torch.exp(1j * (x_grid + y_grid))
        return csms

    def forward(self, data, **kwargs):
        if data.dim() == 3:
            data = data.unsqueeze(0)
        if data.shape[1] == 1:
            data = data.squeeze(1)

        batch_size, height, width = data.shape
        csms = self.create_synthetic_csms(height, width)
        full_k_space = torch.zeros((batch_size, self.num_coils, height, width), device=self.device, dtype=torch.complex64)
        for i in range(batch_size):
            img_complex = torch.complex(data[i], torch.zeros_like(data[i]))
            for c in range(self.num_coils):
                coil_img = img_complex * csms[c]
                full_k_space[i, c, :, :] = fft2_m(coil_img)

        mask_expanded = self.mask.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_coils, -1, width)
        undersampled_kspace = full_k_space * mask_expanded
        return undersampled_kspace

    def transpose(self, undersampled_kspace, **kwargs):
        batch_size, num_coils, height, width = undersampled_kspace.shape
        kh, kw = self.kernel_size
        reconstructed_kspace = torch.clone(undersampled_kspace)
        center_h = height // 2
        acs_start = center_h - self.acs_lines // 2
        acs_end = center_h + self.acs_lines // 2
        acs_region = undersampled_kspace[:, :, acs_start:acs_end, :]
        num_acs_lines, acs_width = acs_region.shape[2], acs_region.shape[3]

        source_patches = []
        target_patches = []

        for y in range(num_acs_lines - (kh - 1) * self.acc):
            for x in range(acs_width - kw):
                source_patch = torch.zeros((batch_size, num_coils, kh, kw), device=self.device, dtype=torch.complex64)
                for i in range(kh):
                    source_patch[:, :, i, :] = acs_region[:, :, y + i * self.acc, x:x + kw]
                
                # FIX 1: Reshape each patch into a vector (B, features) before appending.
                source_patches.append(source_patch.reshape(batch_size, -1))

                target_patch = acs_region[:, :, y + kh // 2 * self.acc, x + kw // 2]
                target_patches.append(target_patch)

        Source = torch.stack(source_patches, dim=1)
        Target = torch.stack(target_patches, dim=1)

        S_H = Source.permute(0, 2, 1).conj()
        W = torch.linalg.solve(torch.matmul(S_H, Source), torch.matmul(S_H, Target))

        for y in range(height):
            is_acquired = self.mask[0, y] == 1
            if not is_acquired: # Simplified this check
                for x in range(kw // 2, width - kw // 2):
                    source_patch = torch.zeros((batch_size, num_coils, kh, kw), device=self.device, dtype=torch.complex64)
                    for i in range(kh):
                        dy = y + (i - kh // 2) * self.acc
                        if 0 <= dy < height:
                            source_patch[:, :, i, :] = reconstructed_kspace[:, :, dy, x - kw // 2:x + kw // 2 + 1]

                    source_vec = source_patch.reshape(batch_size, 1, -1)
                    reconstructed_point = torch.matmul(source_vec, W)
                    reconstructed_kspace[:, :, y, x] = reconstructed_point.squeeze(1)

        reconstructed_images = torch.zeros((batch_size, height, width), device=self.device, dtype=torch.float32)
        for i in range(batch_size):
            coil_images = ifft2_m(reconstructed_kspace[i])
            reconstructed_images[i, :, :] = torch.sqrt(torch.sum(torch.abs(coil_images)**2, dim=0))

        return reconstructed_images.unsqueeze(1)

### MRI Operator Functions y = PFS(x) + e | y = Ax + e
class MRIOperator(LinearOperator): 
    def __init__(self, acc, device, batch_size=1): 
        super(MRIOperator, self)
        self.acc = acc
        self.device = device
        
        mask = get_mask(batch_size=batch_size, R=acc).unsqueeze(1).unsqueeze(1).to(device) # (B, 1, 1, total_lines)
        
        self.mask = mask.expand(-1, 1, 320, -1 )
        
    def forward(self, data, **kwargs): 
        im = torch.complex(data, torch.zeros_like(data))  # Convert to complex tensor
        return fft2_m(im) * self.mask  # Apply mask in k-space
        
    def transpose(self, data, **kwargs):
        return ifft2_m(data)

def shufflerow(tensor, axis):
    row_perm = torch.rand(tensor.shape[:axis+1]).argsort(axis).to(tensor.device) # get permutation indices
    for _ in range(tensor.ndim-axis-1): row_perm.unsqueeze_(-1)
    row_perm = row_perm.repeat(*[1 for _ in range(axis+1)], *(tensor.shape[axis+1:]))  # reformat this for the gather operation
    return tensor.gather(axis, row_perm)

def get_mask(batch_size=1, acs_lines=32, total_lines=320, R=1):
    # Overall sampling budget
    num_sampled_lines = total_lines // R

    # Get locations of ACS lines
    # !!! Assumes k-space is even sized and centered, true for fastMRI
    center_line_idx = torch.arange((total_lines - acs_lines) // 2,
                                (total_lines + acs_lines) // 2)

    # Find remaining candidates
    outer_line_idx = torch.cat([torch.arange(0, (total_lines - acs_lines) // 2), torch.arange((total_lines + acs_lines) // 2, total_lines)])
    random_line_idx = shufflerow(outer_line_idx.unsqueeze(0).repeat([batch_size, 1]), 1)[:, : num_sampled_lines - acs_lines]
    # random_line_idx = outer_line_idx[torch.randperm(outer_line_idx.shape[0])[:num_sampled_lines - acs_lines]]

    # Create a mask and place ones at the right locations
    mask = torch.zeros((batch_size, total_lines))
    mask[:, center_line_idx] = 1.
    mask[torch.arange(batch_size).repeat_interleave(random_line_idx.shape[-1]), random_line_idx.reshape(-1)] = 1.

    return mask

class GRAPPAOperator_np(LinearOperator):
    """
    GRAPPAOperator performs multi-coil forward simulation and GRAPPA reconstruction using NumPy.
    """
    def __init__(self, acc, num_coils=8, acs_lines=32, kernel_size=(3, 3), batch_size=1):
        super(GRAPPAOperator_np, self).__init__()
        self.acc = acc
        self.num_coils = num_coils
        self.acs_lines = acs_lines
        self.kernel_size = kernel_size
        self.mask = get_mask_np(batch_size=batch_size, R=self.acc, acs_lines=self.acs_lines)

    def create_synthetic_csms(self, h, w):
        csms = np.zeros((self.num_coils, h, w), dtype=np.complex64)
        for i in range(self.num_coils):
            x_phase = np.linspace(-np.pi, np.pi, w) * (i - self.num_coils / 2)
            y_phase = np.linspace(-np.pi, np.pi, h) * (i - self.num_coils / 2)
            y_grid, x_grid = np.meshgrid(y_phase, x_phase, indexing='ij')
            csms[i, :, :] = np.exp(1j * (x_grid + y_grid))
        return csms

    def forward(self, data, **kwargs):
        # Convert torch tensor to numpy if needed
        if hasattr(data, 'detach'):
            data = data.detach().cpu().numpy()
        
        if data.ndim == 3:
            data = np.expand_dims(data, axis=0)
        if data.shape[1] == 1:
            data = np.squeeze(data, axis=1)

        batch_size, height, width = data.shape
        csms = self.create_synthetic_csms(height, width)
        full_k_space = np.zeros(
            (batch_size, self.num_coils, height, width),
            dtype=np.complex64
        )
        
        for i in range(batch_size):
            img_complex = data[i].astype(np.complex64)
            for c in range(self.num_coils):
                coil_img = img_complex * csms[c]
                full_k_space[i, c, :, :] = fft2_m_np(coil_img)

        # Expand mask for broadcasting
        mask_expanded = np.expand_dims(self.mask, axis=1)  # (B, 1, H)
        mask_expanded = np.expand_dims(mask_expanded, axis=-1)  # (B, 1, H, 1)
        mask_expanded = np.broadcast_to(mask_expanded, (batch_size, self.num_coils, height, width))
        
        undersampled_kspace = full_k_space * mask_expanded
        return undersampled_kspace

    def transpose(self, undersampled_kspace, **kwargs):
        # Convert torch tensor to numpy if needed
        if hasattr(undersampled_kspace, 'detach'):
            undersampled_kspace = undersampled_kspace.detach().cpu().numpy()
            
        batch_size, num_coils, height, width = undersampled_kspace.shape
        kh, kw = self.kernel_size

        # 1) Initialize reconstructed_kspace with the undersampled data
        reconstructed_kspace = undersampled_kspace.copy()

        # 2) Extract the ACS region (centered in ky)
        center_h = height // 2
        acs_start = center_h - self.acs_lines // 2
        acs_end = center_h + self.acs_lines // 2
        acs_region = undersampled_kspace[:, :, acs_start:acs_end, :]  # (B, C, num_acs_lines, W)
        num_acs_lines, acs_width = acs_region.shape[2], acs_region.shape[3]

        # 3) Build source/target matrices from ACS
        source_patches = []
        target_patches = []
        for y_acs in range(num_acs_lines - (kh - 1) * self.acc):
            for x_acs in range(acs_width - kw):
                # (a) Source patch: size (B, num_coils, kh, kw)
                src = np.zeros((batch_size, num_coils, kh, kw), dtype=np.complex64)
                for i in range(kh):
                    ky_index = y_acs + i * self.acc
                    src[:, :, i, :] = acs_region[:, :, ky_index, x_acs:x_acs + kw]

                source_patches.append(src.reshape(batch_size, -1))

                # (b) Target: center of the (kh×kw) window
                ky_center = y_acs + (kh // 2) * self.acc
                kx_center = x_acs + (kw // 2)
                tgt = acs_region[:, :, ky_center, kx_center]  # (B, num_coils)
                target_patches.append(tgt)

        Source = np.stack(source_patches, axis=1)  # (B, #patches, features)
        Target = np.stack(target_patches, axis=1)  # (B, #patches, num_coils)

        # 4) Solve for W using normal equations + Tikhonov regularization
        S_H = np.conj(np.transpose(Source, (0, 2, 1)))  # (B, features, #patches)
        SHS = np.matmul(S_H, Source)  # (B, features, features)
        SHT = np.matmul(S_H, Target)  # (B, features, num_coils)

        λ = 1e-3  # small regularization constant
        I = np.expand_dims(np.eye(SHS.shape[-1]), axis=0)  # (1, features, features)
        I = np.broadcast_to(I, SHS.shape)  # (B, features, features)
        
        # Solve for each batch
        W = np.zeros_like(SHT)
        for b in range(batch_size):
            W[b] = solve(SHS[b] + λ * I[b], SHT[b])

        # 5) Reconstruct missing k-space lines
        y_min = (kh // 2) * self.acc
        y_max = height - (kh // 2) * self.acc
        for y in range(y_min, y_max):
            # Check a single column to see if that entire ky row was acquired
            if self.mask[0, y] == 1:
                continue  # fully sampled row, no need to interpolate

            for x in range(kw // 2, width - (kw // 2)):
                # Build source patch around (y, x)
                src_patch = np.zeros((batch_size, num_coils, kh, kw), dtype=np.complex64)
                for i in range(kh):
                    ky_i = y + (i - kh // 2) * self.acc
                    src_patch[:, :, i, :] = reconstructed_kspace[:, :, ky_i, 
                                                               x - (kw // 2):x + (kw // 2) + 1]

                src_vec = src_patch.reshape(batch_size, 1, -1)  # (B, 1, features)
                rec_pt = np.matmul(src_vec, W)  # (B, 1, num_coils)
                reconstructed_kspace[:, :, y, x] = np.squeeze(rec_pt, axis=1)

        # 6) Inverse FFT + root-sum-of-squares coil combine
        reconstructed_images = np.zeros((batch_size, height, width), dtype=np.float32)
        for i in range(batch_size):
            coil_images = ifft2_m_np(reconstructed_kspace[i])  # (num_coils, H, W)
            rss = np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=0))  # (H, W)
            reconstructed_images[i] = rss

        # 7) Add channel dimension back → (B, 1, H, W)
        return np.expand_dims(reconstructed_images, axis=1)


def shufflerow_np(tensor, axis):
    """NumPy version of shufflerow function"""
    # Get the shape for the permutation
    perm_shape = tensor.shape[:axis+1]
    row_perm = np.random.rand(*perm_shape).argsort(axis)
    
    # Expand dimensions to match original tensor
    for _ in range(tensor.ndim - axis - 1):
        row_perm = np.expand_dims(row_perm, axis=-1)
    
    # Broadcast to full tensor shape
    row_perm = np.broadcast_to(row_perm, tensor.shape)
    
    # Use advanced indexing instead of gather
    return np.take_along_axis(tensor, row_perm, axis=axis)


def get_mask_np(batch_size=1, acs_lines=32, total_lines=320, R=1):
    """NumPy version of get_mask function"""
    # Overall sampling budget
    num_sampled_lines = total_lines // R

    # Get locations of ACS lines
    center_line_idx = np.arange((total_lines - acs_lines) // 2,
                               (total_lines + acs_lines) // 2)

    # Find remaining candidates
    outer_line_idx = np.concatenate([
        np.arange(0, (total_lines - acs_lines) // 2),
        np.arange((total_lines + acs_lines) // 2, total_lines)
    ])
    
    # Repeat for batch size and shuffle
    outer_expanded = np.tile(outer_line_idx, (batch_size, 1))
    random_line_idx = shufflerow_np(outer_expanded, 1)[:, :num_sampled_lines - acs_lines]

    # Create a mask and place ones at the right locations
    mask = np.zeros((batch_size, total_lines))
    mask[:, center_line_idx] = 1.0
    
    # Set random lines to 1
    batch_indices = np.arange(batch_size).repeat(random_line_idx.shape[-1])
    line_indices = random_line_idx.reshape(-1)
    mask[batch_indices, line_indices] = 1.0

    return mask


# Helper functions for FFT operations (NumPy versions)
def fft2_m_np(data):
    """2D FFT using NumPy"""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(data)))

def ifft2_m_np(data):
    """2D IFFT using NumPy"""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(data)))
