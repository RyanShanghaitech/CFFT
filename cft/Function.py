from numpy import *
from numpy.fft import *

def fft(im:ndarray) -> ndarray:
    return fftshift(fftn(fftshift(im)))/asarray(im.shape).prod() # both fftshift() here is neccessary to make FFT consist with DFT
def ift(ks:ndarray) -> ndarray:
    return fftshift(ifftn(ifftshift(ks)))*asarray(ks.shape).prod()