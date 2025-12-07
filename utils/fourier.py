"""
Fourier Transformation Functions

(1) ifft2c: Apply centered 2D inverse FFT
(2) fft2c: Apply centered 2D FFT
(3) fftshift: PyTorch version of np.fft.fftshift
(4) ifftshift: PyTorch version of np.fft.ifftshift
(5) roll: PyTorch roll function similar to np.roll
(6) roll_one_dim: Roll Tensor along a specified dimension
"""

import torch 

def ifft2c(x):
    """Apply centered 2D inverse FFT."""
    # Check Complexity 
    if x.dtype in [torch.complex64, torch.complex128]:
        complex = x 
    else: 
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be of size 2 for complex numbers.")
        complex = torch.view_as_complex(x)

    # Apply IFFT 
    complex = ifftshift(complex, dim=[-2, -1])
    complex = torch.fft.ifftn(complex, dim=(-2, -1), norm='ortho')
    complex = fftshift(complex, dim=[-2, -1])

    # Return as real tensor with last dimension of size 2
    return torch.view_as_real(complex)

def fft2c(x):
    """Centered 2D FFT."""
    # Check Complexity 
    if x.dtype in [torch.complex64, torch.complex128]:
        complex = x 
    else: 
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be of size 2 for complex numbers.")
        complex = torch.view_as_complex(x)

    # Apply FFT 
    complex = ifftshift(complex, dim=[-2, -1])
    complex = torch.fft.fftn(complex, dim=(-2, -1), norm='ortho')
    complex = fftshift(complex, dim=[-2, -1])

    # Return as real tensor with last dimension of size 2
    return torch.view_as_real(complex)

def fftshift(x, dim=None):
    """PyTorch version of np.fft.fftshift."""
    if dim is None:
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    shift = [0] * len(dim)
    for i, d in enumerate(dim):
        shift[i] = x.shape[d] // 2

    return roll(x, shift, dim)

def ifftshift(x, dim=None):
    """PyTorch version of np.fft.ifftshift."""
    if dim is None: 
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i 

    shift = [0] * len(dim)
    for i, d in enumerate(dim):
        shift[i] = (x.shape[d] + 1) // 2

    return roll(x, shift, dim)

def roll(x, shift, dim):
    """PyTorch roll function similar to np.roll."""
    if len(shift) != len(dim):
        raise ValueError("Shift and dim must have the same length.")

    for s, d, in zip(shift, dim):
        x = roll_one_dim(x, s, d)

    return x

def roll_one_dim(x, shift, dim):
    """Roll Tensor along a specified dimension."""
    shift = shift % x.size(dim)
    if shift == 0:
        return x 

    # Left and Right Slices
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)
    