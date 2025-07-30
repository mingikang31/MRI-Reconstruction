"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import List, Optional
import tifffile as tiff
import torch
import imageio
import numpy as np
from packaging import version
import scipy.io as sio
from typing import Dict, Optional, Sequence, Tuple, Union
import os
import cv2
from utils.mask_function import MaskFunc
import matplotlib.pyplot as plt
if version.parse(torch.__version__) >= version.parse("1.7.0"):
    import torch.fft  # type: ignore
from torch.utils.data import DataLoader, SubsetRandomSampler

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

import csv

def log_metrics_csv(epoch, psnr, ssim, loss, phase, logfile="metrics_log.csv"):
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    write_header = not os.path.exists(logfile)
    with open(logfile, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Epoch", "Phase", "PSNR", "SSIM", "Loss"])
        writer.writerow([epoch, phase, psnr, ssim, loss])


def add_complex_gaussian_noise(kspace, sigma):
    """
    Add complex-valued Gaussian noise to k-space.

    Args:
        kspace: torch.Tensor of shape [B, C, H, W], dtype=torch.complex64
        sigma: standard deviation of the noise (scalar or tensor)

    Returns:
        Noisy k-space tensor of same shape.
    """
    noise_real = torch.randn_like(kspace.real) * sigma
    noise_imag = torch.randn_like(kspace.imag) * sigma
    noise = torch.complex(noise_real, noise_imag)
    return kspace + noise

def img_show(img, title = "", cmap = 'gray'):
    tensor = img.detach().cpu()
    if torch.is_complex(tensor):
        tensor = tensor.abs()
    elif tensor.ndim == 3 and tensor.shape[-1] == 2:
            # Real + Imag pair last dimension
        tensor = torch.sqrt(tensor[..., 0]**2 + tensor[..., 1]**2)

        # Remove batch and channel if needed
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim == 3 and tensor.shape[0] <= 3:
        tensor = tensor.squeeze(0)  # assume channel-first
    if tensor.ndim == 3 and tensor.shape[-1] <= 3:
        tensor = tensor.permute(1, 2, 0)  # HWC format

        # Normalize to [0, 1]
    tensor = tensor.float()
    tensor -= tensor.min()
    tensor /= (tensor.max())

    plt.imshow(tensor.numpy(), cmap=cmap)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.colorbar()
    plt.show()

def to_tiff(x, path, is_normalized=True):
    try:
        x = np.squeeze(x)
    except:
        pass

    try:
        x = torch.squeeze(x).numpy()
    except:
        pass

    print(x.shape, path)

    if len(x.shape) == 3:
        n_slice, n_x, n_y = x.shape

        if is_normalized:
            for i in range(n_slice):
                x[:, :, i] -= np.amin(x[:, :, i])
                x[:, :, i] /= np.amax(x[:, :, i])
            #
                x[:, :, i] *= 255

            x = x.astype(np.uint8)
            # x = x.astype(np.float32)
    x = x.astype(np.float32)
    tiff.imwrite(path, x, imagej=True, ijmetadata={'Slice': n_slice})


def mask_square_center(x: torch.Tensor, mask_from: int, mask_to: int) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.

    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """

    # mask = torch.zeros_like(x)
    # mask[:, :, mask_from:mask_to, mask_from:mask_to] = x[:, :, mask_from:mask_to, mask_from:mask_to]
    mask = torch.zeros_like(x)
    mask_region = x[:, :, mask_from:mask_to, mask_from:mask_to]
    mask_region = torch.view_as_complex(mask_region)

    window = torch.hamming_window(mask_to.item()-mask_from.item()).cuda()
    mask_region = mask_region * window
    mask_region = torch.stack((mask_region.real, mask_region.imag), dim=-1)
    mask[:, :, mask_from:mask_to, mask_from:mask_to] = mask_region
    # print(x[:, :, mask_from:mask_to, mask_from:mask_to].shape)

    return mask


def mask_center(x: torch.Tensor, mask_from: int, mask_to: int) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.

    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    mask = torch.zeros_like(x)
    print(mask.shape)
    mask[:, :, :, mask_from:mask_to] = x[:, :, :, mask_from:mask_to]

    return mask


def rss(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS).

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform

    Returns:
        The RSS value.
    """
    return torch.sqrt((data ** 2).sum(dim))


def rss_complex(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS) for complex inputs.

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform

    Returns:
        The RSS value.
    """
    return torch.sqrt(complex_abs_sq(data).sum(dim))


def lr_scheduler(optimizer, epoch):
    """Decay learning rate by a factor of 0.5 every 5000."""
    if epoch % 20 == 0 and epoch > 25:
        for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9
        print('LR is set to {}'.format(param_group['lr']))

def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())


def ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return structural_similarity(gt, pred, data_range=gt.max())

def to_tiff(x, path, is_normalized=True):
    try:
        x = np.squeeze(x)
    except:
        pass

    try:
        x = torch.squeeze(x).numpy()
    except:
        pass

    print(x.shape, path)

    if len(x.shape) == 3:
        n_slice, n_x, n_y = x.shape

        if is_normalized:
            for i in range(n_slice):
                x[:, :, i] -= np.amin(x[:, :, i])
                x[:, :, i] /= np.amax(x[:, :, i])
            #
                x[:, :, i] *= 255

            x = x.astype(np.uint8)
            # x = x.astype(np.float32)
    x = x.astype(np.float32)
    tiff.imwrite(path, x, imagej=True, ijmetadata={'Slice': n_slice})

def divide_root_sum_of_squares(x: torch.Tensor) -> torch.Tensor:
    return x / rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)


def rss_complex(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS) for complex inputs.

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform

    Returns:
        The RSS value.
    """
    return torch.sqrt(complex_abs_sq(data).sum(dim))


def complex_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Complex multiplication.

    This multiplies two complex tensors assuming that they are both stored as
    real arrays with the last dimension being the complex dimension.

    Args:
        x: A PyTorch tensor with the last dimension of size 2.
        y: A PyTorch tensor with the last dimension of size 2.

    Returns:
        A PyTorch tensor with the last dimension of size 2.
    """
    if not x.shape[-1] == y.shape[-1] == 2:
        raise ValueError("Tensors do not have separate complex dim.")

    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]

    return torch.stack((re, im), dim=-1)


def complex_conj(x: torch.Tensor) -> torch.Tensor:
    """
    Complex conjugate.

    This applies the complex conjugate assuming that the input array has the
    last dimension as the complex dimension.

    Args:
        x: A PyTorch tensor with the last dimension of size 2.
        y: A PyTorch tensor with the last dimension of size 2.

    Returns:
        A PyTorch tensor with the last dimension of size 2.
    """
    if not x.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return torch.stack((x[..., 0], -x[..., 1]), dim=-1)

def complex_abs_sq(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the squared absolute value of a complex tensor.

    Args:
        data: A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        Squared absolute value of data.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data ** 2).sum(dim=-1)

def complex_abs(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data: A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        Absolute value of data.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data ** 2).sum(dim=-1).sqrt()

def fft2c(data: torch.Tensor) -> torch.Tensor:
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.

    Returns:
        The FFT of the input.
    """
    data = torch.fft.fftshift(data, dim=(-2, -1))
    data = torch.fft.fft2(data, dim=(-2, -1))
    data =  torch.fft.ifftshift(data, dim=(-2, -1))
    return data


# def ifft2c(data: torch.Tensor) -> torch.Tensor:
#     """
#     Apply centered 2-dimensional Inverse Fast Fourier Transform.
#
#     Args:
#         data: Complex valued input data containing at least 3 dimensions:
#             dimensions -3 & -2 are spatial dimensions and dimension -1 has size
#             2. All other dimensions are assumed to be batch dimensions.
#
#     Returns:
#         The IFFT of the input.
#     """
#     coil_imgs = torch.fft.ifftshift(data, dim=(-2, -1))
#     coil_imgs = torch.fft.ifft2(coil_imgs, dim=(-2, -1))
#     data = torch.fft.fftshift(coil_imgs, dim=(-2, -1))
#
#     return data

def ifft2c(data: torch.Tensor) -> torch.Tensor:
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.

    Returns:
        The IFFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm="ortho"
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data

def fft2c(data: torch.Tensor) -> torch.Tensor:
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.

    Returns:
        The FFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm="ortho"
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data




def roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """
    Similar to roll but for only one dim.

    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.

    Returns:
        Rolled version of x.
    """
    shift = shift % x.size(dim)
    if shift == 0:
        return x

    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)


def roll(
    x: torch.Tensor,
    shift: List[int],
    dim: List[int],
) -> torch.Tensor:
    """
    Similar to np.roll but applies to PyTorch Tensors.

    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.

    Returns:
        Rolled version of x.
    """
    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")

    for (s, d) in zip(shift, dim):
        x = roll_one_dim(x, s, d)

    return x


def fftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors

    Args:
        x: A PyTorch tensor.
        dim: Which dimension to fftshift.

    Returns:
        fftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = x.shape[dim_num] // 2

    return roll(x, shift, dim)


def ifftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors

    Args:
        x: A PyTorch tensor.
        dim: Which dimension to ifftshift.

    Returns:
        ifftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = (x.shape[dim_num] + 1) // 2

    return roll(x, shift, dim)


def to_tiff(x, path, is_normalized=True):
    try:
        x = np.squeeze(x)
    except:
        pass

    try:
        x = torch.squeeze(x).numpy()
    except:
        pass

    print(x.shape, path)

    if len(x.shape) == 3:
        n_slice, n_x, n_y = x.shape

        if is_normalized:
            for i in range(n_slice):
                x[:, :, i] -= np.amin(x[:, :, i])
                x[:, :, i] /= np.amax(x[:, :, i])
            #
                x[:, :, i] *= 255

            x = x.astype(np.uint8)
            # x = x.astype(np.float32)
    x = x.astype(np.float32)
    tiff.imwrite(path, x, imagej=True, ijmetadata={'Slice': n_slice})

def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of data.

    Returns:
        The center cropped image.
    """
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]

def complex_center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data: The complex input tensor to be center cropped. It should have at
            least 3 dimensions and the cropping is applied along dimensions -3
            and -2 and the last dimensions should have a size of 2.
        shape: The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        The center cropped image
    """
    if not (0 < shape[0] <= data.shape[-3] and 0 < shape[1] <= data.shape[-2]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to, :]


def compare_mse(img_test, img_true, size_average=True):
    img_diff = img_test - img_true
    img_diff = img_diff ** 2

    if size_average:
        img_diff = img_diff.mean()
    else:
        img_diff = img_diff.mean(-1).mean(-1).mean(-1)

    return img_diff


def imsave(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)


def compare_psnr(img_test, img_true, size_average=True, max_value=1):
    return 10 * torch.log10((max_value ** 2) / compare_mse(img_test, img_true, size_average))


def compare_psnr_batch(img1, img2, max_value=1.0, eps=1e-8):
    """
    Computes PSNR for a batch of images.

    Args:
        img1: torch.Tensor of shape [B, 1, H, W]
        img2: torch.Tensor of shape [B, 1, H, W]
        max_value: the max value of the image (usually 1.0 after normalization)
        eps: small value to avoid log(0)

    Returns:
        psnr_vals: torch.Tensor of shape [B] with PSNR for each image
    """
    if not isinstance(img1, torch.Tensor) or not isinstance(img2, torch.Tensor):
        raise TypeError("Inputs must be torch tensors.")

    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: img1 {img1.shape}, img2 {img2.shape}")

    if img1.ndim != 4 or img1.shape[1] != 1:
        raise ValueError(f"Expected shape [B, 1, H, W], got {img1.shape}")

    mse = torch.mean((img1 - img2) ** 2, dim=(1, 2, 3)) + eps
    psnr_vals = 10 * torch.log10((max_value ** 2) / mse)
    return psnr_vals


def to_rgb(img):
    """
    Converts the given array into a RGB image. If the number of channels is not
    3 the array is tiled such that it has 3 channels. Finally, the values are
    rescaled to [0,255)

    :param img: the array to convert [nx, ny, channels]

    :returns img: the rgb image [nx, ny, 3]
    """
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)

    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    img *= 255
    return img


def save_img(img, path):
    """
    Writes the image to disk

    :param img: the rgb image to save
    :param path: the target path
    """
    img = to_rgb(img)
    imageio.imwrite(path, img.round().astype(np.uint8))

def save_gray(img, path):
    """
    Writes the image to disk

    :param img: the rgb image to save
    :param path: the target path
    """
    imageio.imwrite(path, img)


def save_mat(img, path):
    """
    Writes the image to disk

    :param img: the rgb image to save
    :param path: the target path
    """

    sio.savemat(path, {'img': img})

def apply_mask(
    data: torch.Tensor,
    mask_func: MaskFunc,
    seed: Optional[Union[int, Tuple[int, ...]]] = None,
    padding: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data: The input k-space data. This should have at least 3 dimensions,
            where dimensions -3 and -2 are the spatial dimensions, and the
            final dimension has size 2 (for complex values).
        mask_func: A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed: Seed for the random number generator.
        padding: Padding value to apply for mask.

    Returns:
        tuple containing:
            masked data: Subsampled k-space data
            mask: The generated mask
    """
    shape = np.array(data.shape)
    shape[:-3] = 1
    mask = mask_func(shape, seed)
    if padding is not None:
        mask[:, :, : padding[0]] = 0
        mask[:, :, padding[1] :] = 0  # padding value inclusive on right of zeros

    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data, mask

def normalize_per_image(x):
    mean = x.mean(dim=[1,2,3], keepdim=True)
    std = x.std(dim=[1,2,3], keepdim=True) + 1e-8
    return (x - mean) / std

def rss(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS).
    RSS is computed assuming that dim is the coil dimension.
    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform
    Returns:
        The RSS value.
    """

    rss_image = torch.sqrt(torch.sum(torch.abs(data) ** 2, dim=1, keepdim=True))  # check the abs
    return rss_image

def to_complex(x, device):
    return torch.view_as_complex(x.to(device, non_blocking=True))

def get_img_from_ksp(data):
    out_img_multi = ifft2c(data)
    h_trans_output = torch.sqrt(torch.sum((out_img_multi.real ** 2 + out_img_multi.imag ** 2), dim =1, keepdim=True))
    # h_trans_output = torch.sum(out_img_multi, dim=1, keepdim=True)
    return h_trans_output

def get_img_from_ksp_smp(data, smp):
    out_img_multi = ifft2c(data)
    recon = (out_img_multi * smp.conj()).sum(dim=1, keepdim=True)
    # h_trans_output = torch.sqrt(torch.sum((out_img_multi.real ** 2 + out_img_multi.imag ** 2), dim =1, keepdim=True))
    # h_trans_output = torch.sum(out_img_multi, dim=1, keepdim=True)
    return recon


def normalize_batch(x, eps=1e-8):
    # Min and max over [H, W] per image (keep dims for broadcasting)
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def normalize_magnitude(x):  # img: magnitude image
    norm = torch.sqrt((x[..., 0] ** 2 + x[..., 1] ** 2).max())
    x_norm = x / (norm + 1e-8)
    return x_norm

def normlize(data):
    """
    0-1 normlization
    Args:
        data: The input tensor
    Returns:
        The 0-1 normlized data.
    """
    return (data-data.min())/(data.max()-data.min())

def save_image_pair(img_pred, img_gt, epoch, phase, output_dir="visual_logs"):
    os.makedirs(output_dir, exist_ok=True)
    plt.imsave(f"{output_dir}/{phase}_recon_epoch{epoch}.png", img_pred, cmap='gray')
    plt.imsave(f"{output_dir}/{phase}_gt_epoch{epoch}.png", img_gt, cmap='gray')


def parse_subject_data(txt_path):
    with open(txt_path, 'r') as f:
        raw = f.read().replace('\\', '')  # remove line-continuation slashes

    subject_dict = {}
    for line in raw.strip().split('\n'):
        if not line.strip():
            continue
        key, value = line.split(':', 1)
        subject_id = key.strip()
        coords = eval(value.strip())  # safe here since it's structured as list of tuples
        subject_dict[subject_id] = coords

    return subject_dict


def get_loader(dataset, subset_size, batch_size, num_workers):
    indices = np.random.choice(len(dataset), subset_size, replace=False)
    sampler = SubsetRandomSampler(indices)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader


def complex_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Complex multiplication.

    This multiplies two complex tensors assuming that they are both stored as
    real arrays with the last dimension being the complex dimension.

    Args:
        x: A PyTorch tensor with the last dimension of size 2.
        y: A PyTorch tensor with the last dimension of size 2.

    Returns:
        A PyTorch tensor with the last dimension of size 2.
    """
    if not x.shape[-1] == y.shape[-1] == 2:
        raise ValueError("Tensors do not have separate complex dim.")

    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]

    return torch.stack((re, im), dim=-1)

def complex_conj(x: torch.Tensor) -> torch.Tensor:
    """
    Complex conjugate.

    This applies the complex conjugate assuming that the input array has the
    last dimension as the complex dimension.

    Args:
        x: A PyTorch tensor with the last dimension of size 2.
        y: A PyTorch tensor with the last dimension of size 2.

    Returns:
        A PyTorch tensor with the last dimension of size 2.
    """
    if not x.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return torch.stack((x[..., 0], -x[..., 1]), dim=-1)


def walsh_sensitivity_maps(kspace):
    img_complex = torch.view_as_complex(ifft2c(kspace))
    rss_img = rss(img_complex)
    sens_maps = img_complex / (rss_img + 1e-8)  # [C, H, W], complex
    img_combined = (img_complex * sens_maps.conj()).sum(dim=1, keepdim=True)  # [H, W], complex
    return torch.view_as_real(sens_maps), torch.view_as_real(img_combined)


def zero_filled_rss(kspace):
    img_c = torch.view_as_complex(ifft2c(kspace))
    img_rss = torch.sqrt((img_c.abs()**2).sum(dim=1, keepdim = True))  # [H, W]
    return img_rss / (img_rss.max() + 1e-8)


import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

def compare_ssim_batch(img1, img2, data_range=1.0):
    """
    Computes SSIM for a batch of images.

    Args:
        img1: torch.Tensor of shape [B, 1, H, W]
        img2: torch.Tensor of shape [B, 1, H, W]
        data_range: the data range of the image (usually 1.0 after normalization)

    Returns:
        ssim_vals: torch.Tensor of shape [B] with SSIM for each image
    """
    if not isinstance(img1, torch.Tensor) or not isinstance(img2, torch.Tensor):
        raise TypeError("Inputs must be torch tensors.")

    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: img1 {img1.shape}, img2 {img2.shape}")

    if img1.ndim != 4 or img1.shape[1] != 1:
        raise ValueError(f"Expected shape [B, 1, H, W], got {img1.shape}")

    # Instantiate the metric on the same device as the images
    # reduction='none' ensures we get a score for each image in the batch
    ssim_metric = StructuralSimilarityIndexMeasure(
        data_range=data_range, reduction='none'
    ).to(img1.device)

    ssim_vals = ssim_metric(img1, img2)
    
    return ssim_vals



def zero_filled_rss(kspace):
    img_c = torch.view_as_complex(ifft2c(kspace))
    img_rss = torch.sqrt((img_c.abs()**2).sum(dim=1, keepdim = True))  # [H, W]
    return img_rss / (img_rss.max() + 1e-8)