from numpy import *
from numpy.fft import *

def cft(x:ndarray, axes:tuple|None=None) -> ndarray:
    x = ifftshift(x, axes=axes)
    x = fftn(x, axes=axes)
    x = fftshift(x, axes=axes)
    return x # both ifftshift() and fftshift() here is neccessary to make FFT consist with DFT

def ift(x:ndarray, axes:tuple|None=None) -> ndarray:
    x = ifftshift(x, axes=axes)
    x = ifftn(x, axes=axes)
    x = fftshift(x, axes=axes)
    return x # both ifftshift() and fftshift() here is neccessary to make FFT consist with DFT