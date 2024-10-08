from numpy import *
from numpy.fft import *

def cft(im:ndarray, axes:tuple|None=None) -> ndarray:
    return fftshift(fftn(fftshift(im, axes=axes), axes=axes), axes=axes) # both fftshift() here is neccessary to make FFT consist with DFT

def ift(ks:ndarray, axes:tuple|None=None) -> ndarray:
    return fftshift(ifftn(fftshift(ks, axes=axes), axes=axes), axes=axes) # both fftshift() here is neccessary to make FFT consist with DFT