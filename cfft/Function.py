from numpy import fft as npfft

def fft(im): return npfft.fftshift(npfft.fftn(npfft.fftshift(im)))
def ifft(ks): return npfft.fftshift(npfft.ifftn(npfft.fftshift(ks)))