"""
Operator utility functions for MRI reconstruction. 

"""

from utils.fourier import fft2c, ifft2c 
from utils.util import complex_mul, complex_conj    

def transpose_operator(out_kspace, sensitivity_maps, mask=None):
    """Transpose operation: from k-space to image space."""
    # Apply inverse FFT
    out_img = ifft2c(out_kspace)

    # Multiply with conjugate of sensitivity maps
    sensitivity_maps_conj = complex_conj(sensitivity_maps)
    out_img_combined = complex_mul(out_img, sensitivity_maps_conj).sum(dim=1, keepdim=True)
    return out_img_combined

def grad_g_operator(img, sensitivity_maps, mask=None):
    """Gradient of g operator."""
    # Multiply with sensitivity maps
    img_sens = complex_mul(img, sensitivity_maps)

    # Apply FFT
    kspace = fft2c(img_sens)

    # Apply mask if provided
    if mask is not None:
        kspace = kspace * mask
    return kspace