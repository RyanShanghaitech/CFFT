from numpy.fft import *

def funFFT(im): return fftshift(fftn(fftshift(im)))
def funIFFT(ks): return fftshift(ifftn(fftshift(ks)))