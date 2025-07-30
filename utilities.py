import torch 
import torch
from torch.autograd import Variable
from math import exp
import torch.nn.functional as f


def compare_mse(img_test, img_true, size_average=True):
    img_diff = img_test - img_true
    img_diff = img_diff ** 2

    if size_average:
        img_diff = img_diff.mean()
    else:
        img_diff = img_diff.mean(-1).mean(-1).mean(-1)

    return img_diff

def compare_nmse(img_test, img_true):
    return torch.linalg.norm(img_true - img_test) ** 2 / torch.linalg.norm(img_true) ** 2

def compare_psnr(img_test, img_true, size_average=True, max_value=1):
    return 10 * torch.log10((max_value ** 2) / compare_mse(img_test, img_true, size_average))


def compare_snr(img_test, img_true, size_average=True):
    if not size_average:
        raise NotImplementedError('size_average must be True')

    return 20 * torch.log10(torch.norm(img_true.flatten()) / torch.norm(img_true.flatten() - img_test.flatten()))


def compare_rsnr(img_test, img_true, size_average=True):
    if not size_average:
        raise NotImplementedError('size_average must be True')

    img_test = torch.squeeze(img_test)
    img_true = torch.squeeze(img_true)

    if img_test.shape.__len__() != 2 or img_true.shape.__len__() != 2:
        raise NotImplementedError('only 2D images are supported')

    img_true_flatten = torch.flatten(img_true)
    img_test_flatten = torch.flatten(img_test)

    a = torch.zeros((2, 2))
    a[0, 0] = torch.sum(img_true_flatten ** 2)
    a[0, 1] = torch.sum(img_true_flatten)
    a[1, 0] = a[0, 1]
    a[1, 1] = img_test.shape[0] * img_test.shape[1]

    b = torch.zeros((2, 1))
    b[0] = torch.sum(img_test_flatten * img_true_flatten)
    b[1] = torch.sum(img_test_flatten)

    c = torch.matmul(torch.inverse(a), b)
    if img_true.is_cuda:
        c = c.cuda()

    rsnr = compare_snr(img_test, c[0] * img_true + c[1], size_average=True)

    return rsnr


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = f.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = f.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = f.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = f.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = f.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(-1).mean(-1).mean(-1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data_input.type() == img1.data_input.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def compare_ssim(img_test, img_true, size_average=True, window_size=11):
    (_, channel, _, _) = img_test.size()
    window = create_window(window_size, channel)

    if img_test.is_cuda:
        window = window.cuda(img_test.get_device())
    window = window.type_as(img_test)

    return _ssim(img_test, img_true, window, window_size, channel, size_average)


def compare_rpsnr(img_test, img_true, size_average=True):
    if not size_average:
        raise NotImplementedError('size_average must be True')

    img_test = torch.squeeze(img_test)
    img_true = torch.squeeze(img_true)

    if img_test.shape.__len__() != 2 or img_true.shape.__len__() != 2:
        raise NotImplementedError('only 2D images are supported')

    img_true_flatten = torch.flatten(img_true)
    img_test_flatten = torch.flatten(img_test)

    a = torch.zeros((2, 2))
    a[0, 0] = torch.sum(img_true_flatten ** 2)
    a[0, 1] = torch.sum(img_true_flatten)
    a[1, 0] = a[0, 1]
    a[1, 1] = img_test.shape[0] * img_test.shape[1]

    b = torch.zeros((2, 1))
    b[0] = torch.sum(img_test_flatten * img_true_flatten)
    b[1] = torch.sum(img_test_flatten)

    c = torch.matmul(torch.inverse(a), b)
    if img_true.is_cuda:
        c = c.cuda()

    rsnr = compare_psnr(img_test, c[0] * img_true + c[1], size_average=True)

    return rsnr


def compare_rssim(img_test, img_true, size_average=True):
    if not size_average:
        raise NotImplementedError('size_average must be True')

    img_test = torch.squeeze(img_test)
    img_true = torch.squeeze(img_true)

    if img_test.shape.__len__() != 2 or img_true.shape.__len__() != 2:
        raise NotImplementedError('only 2D images are supported')

    img_true_flatten = torch.flatten(img_true)
    img_test_flatten = torch.flatten(img_test)

    a = torch.zeros((2, 2))
    a[0, 0] = torch.sum(img_true_flatten ** 2)
    a[0, 1] = torch.sum(img_true_flatten)
    a[1, 0] = a[0, 1]
    a[1, 1] = img_test.shape[0] * img_test.shape[1]

    b = torch.zeros((2, 1))
    b[0] = torch.sum(img_test_flatten * img_true_flatten)
    b[1] = torch.sum(img_test_flatten)

    c = torch.matmul(torch.inverse(a), b)
    if img_true.is_cuda:
        c = c.cuda()

    img_test = img_test.unsqueeze(0).unsqueeze(0)
    img_true = img_true.unsqueeze(0).unsqueeze(0)

    rsnr = compare_ssim(img_test, c[0] * img_true + c[1], size_average=True)

    return rsnr

def normalize(data):
    """
    0-1 normlization
    Args:
        data: The input tensor
    Returns:
        The 0-1 normlized data.
    """
    return (data-data.min())/(data.max()-data.min())

def ifft2c(data: torch.Tensor) -> torch.Tensor:
    """Centered 2D inverse FFT"""
    coil_imgs = torch.fft.ifftshift(data, dim=(-2, -1))
    coil_imgs = torch.fft.ifft2(coil_imgs, dim=(-2, -1), norm='ortho') 
    data = torch.fft.fftshift(coil_imgs, dim=(-2, -1))
    return data

def transpose_operator(out_kspace, smap):
    out_img_multi = ifft2c(out_kspace)
    h_trans_output = torch.sum(out_img_multi * torch.conj(smap), dim=1, keepdim=True)
    return h_trans_output

def get_img_from_ksp_old(data):
    out_img_multi = ifft2c(data)
    h_trans_output = torch.sum(out_img_multi, dim=1, keepdim=True)
    return h_trans_output
    # return rss(ifft2c(data))


### Code from Shirin ### TODO CHECK LATER 
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
########

def img_to_kspace_multicoil(output_img, smps):
    """
    Proper forward model: single-coil image -> multi-coil k-space
    
    Args:
        output_img: [batch, 1, H, W, 2] or [batch, 1, H, W] - single coil image
        smps: [batch, 64, H, W, 2] or [batch, 64, H, W] - sensitivity maps
    
    Returns:
        output_ksp: [batch, 64, H, W, 2] - multi-coil k-space
    """
    # Convert to complex if needed
    if not output_img.is_complex():
        output_img_complex = torch.view_as_complex(output_img) 
    else:
        output_img_complex = output_img
        
    if not smps.is_complex():
        smps_complex = torch.view_as_complex(smps) 
    else:
        smps_complex = smps
    
    # Remove singleton coil dimension from output
    if output_img_complex.shape[1] == 1:
        output_img_complex = output_img_complex.squeeze(1) 
    
    multicoil_img = output_img_complex.unsqueeze(1) * smps_complex  
    
    # Convert to k-space
    output_ksp_complex = fft2c(multicoil_img)  
    
    return output_ksp_complex


def normalize_batch(x, eps=1e-8):
    # Min and max over [H, W] per image (keep dims for broadcasting)
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def to_complex(x, device):
    return torch.view_as_complex(x.to(device, non_blocking=True))

def rss(data: torch.Tensor, coil_dim: int = 1) -> torch.Tensor:
    rss_image = torch.sqrt(torch.sum(torch.abs(data) ** 2, dim=coil_dim, keepdim=True))
    return rss_image




def fft2c(data: torch.Tensor) -> torch.Tensor:
    """Centered 2D forward FFT"""
    kspace = torch.fft.fftshift(data, dim=(-2, -1))
    kspace = torch.fft.fft2(kspace, dim=(-2, -1), norm='ortho')
    kspace = torch.fft.ifftshift(kspace, dim=(-2, -1))
    return kspace



def grad_g_operator(out_img, smap):
    out_img_multi = out_img * smap
    grad_g_output = fft2c(out_img_multi)
    return grad_g_output

def transpose_operator(out_kspace, smap):
    out_img_multi = ifft2c(out_kspace)
    h_trans_output = torch.sum(out_img_multi * torch.conj(smap), dim=1, keepdim=True)
    return h_trans_output

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

def forward_operator(out_img, smap, mask):
    out_img_multi = complex_mul(out_img, smap)
    out_kspace_multi = fft2c(out_img_multi)
    if mask.size(dim=-1) != 2:
        mask = torch.stack([mask, mask], dim=-1)
    h_output = out_kspace_multi * mask + 0.0
    return h_output


def normalize_for_display(img_tensor):
    """
    Normalize a tensor image for display purposes.
    
    Args:
        img_tensor: A PyTorch tensor image of shape (C, H, W) or (H, W).
    
    Returns:
        Normalized image tensor in the range [0, 1].
    """
    img_tensor = img_tensor.float()  # Ensure float type
    img_min = img_tensor.min()
    img_max = img_tensor.max()
    return (img_tensor - img_min) / (img_max - img_min + 1e-8)  # Avoid division by zero


def apply_window_level(img_tensor, window=None, level=None):
    """
    Apply window/level adjustment like medical imaging viewers
    """
    img = img_tensor.clone()
    
    if window is None:
        window = torch.quantile(img, 0.95) - torch.quantile(img, 0.05)
    if level is None:
        level = torch.quantile(img, 0.5)
    
    img_min = level - window / 2
    img_max = level + window / 2
    
    # Apply windowing
    img = torch.clamp(img, img_min, img_max)
    img = (img - img_min) / (img_max - img_min)
    
    return img