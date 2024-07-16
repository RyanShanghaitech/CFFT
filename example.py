from numpy import *
from matplotlib.pyplot import *
from skimage import data, transform
import cfft
sizIm = 128

# create phatom
im = transform.resize(data.shepp_logan_phantom(), (sizIm, sizIm))

# generate centered kspace
ksCFFT = cfft.fft(im)
imRecoCFT = cfft.ifft(ksCFFT)

# generate raw kspace using a single fftshift
ksFFT = fft.fftshift(fft.fftn(im))
imRecoFFT = fft.ifftn(ksFFT)

print(f"mean err = {mean(abs(ksCFFT-ksFFT).flatten())}")
print(f"mean err = {mean(abs(imRecoCFT-imRecoFFT).flatten())}")

# compare
figure()
subplot(231)
imshow(abs(im), cmap='gray')
title('img')
subplot(232)
imshow(abs(ksCFFT), cmap='gray', norm="log")
title('img.CFFT.abs')
subplot(233)
imshow(real(imRecoCFT), cmap='gray')
title('img.CFFT.CIFFT.real')
subplot(234)
imshow(abs(im), cmap='gray')
title('img')
subplot(235)
imshow(abs(ksFFT), cmap='gray', norm="log")
title('img.FFT.SHIFT.abs')
subplot(236)
imshow(real(imRecoFFT), cmap='gray')
title('img.FFT.SHIFT.IFFT.real')

show()
