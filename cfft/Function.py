from numpy.fft import *

def cft(im): return fftshift(fftn(fftshift(im)))
def icft(ks): return fftshift(ifftn(fftshift(ks)))