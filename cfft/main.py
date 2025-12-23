import numpy
import torch
from numpy.typing import NDArray
from torch import Tensor

# both ifftshift() and fftshift() here is neccessary to make FFT consist with DFT

def fftnc(x:NDArray|Tensor, axes: tuple[int]|None=None) -> NDArray|Tensor:
    if axes is not None and not isinstance(axes, tuple): raise TypeError("")
    
    if isinstance(x, numpy.ndarray):
        x = numpy.fft.ifftshift(x, axes=axes)
        x = numpy.fft.fftn(x, axes=axes)
        x = numpy.fft.fftshift(x, axes=axes)
        return x

    if isinstance(x, torch.Tensor):
        x = torch.fft.ifftshift(x, dim=axes)
        x = torch.fft.fftn(x, dim=axes)
        x = torch.fft.fftshift(x, dim=axes)
        return x

    raise TypeError(f"Expected numpy.ndarray or torch.Tensor, got {type(x)!r}")

def ifftnc(x:NDArray|Tensor, axes: tuple[int]|None=None) -> NDArray|Tensor:
    if axes is not None and not isinstance(axes, tuple): raise TypeError("")
    
    if isinstance(x, numpy.ndarray):
        x = numpy.fft.ifftshift(x, axes=axes)
        x = numpy.fft.ifftn(x, axes=axes)
        x = numpy.fft.fftshift(x, axes=axes)
        return x

    if isinstance(x, torch.Tensor):
        x = torch.fft.ifftshift(x, dim=axes)
        x = torch.fft.ifftn(x, dim=axes)
        x = torch.fft.fftshift(x, dim=axes)
        return x
    
    raise TypeError(f"Expected numpy.ndarray or torch.Tensor, got {type(x)!r}")