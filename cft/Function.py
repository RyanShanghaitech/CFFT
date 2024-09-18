from numpy import *
from numpy.fft import *

def fft(im:ndarray) -> ndarray:
    return fftshift(fftn(im))/asarray(im.shape).prod()
def ift(ks:ndarray) -> ndarray:
    return ifftn(ifftshift(ks))*asarray(ks.shape).prod()