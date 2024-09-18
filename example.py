from numpy import *
from matplotlib.pyplot import *
from skimage import data, transform
import cft
sizIm = 128

# create phatom
im = transform.resize(data.shepp_logan_phantom(), (sizIm, sizIm))

# generate kspace using fft
ksFFT = fft.fftn(im)
imRecoFFT = fft.ifftn(ksFFT)

# generate centralized kspace
ksCFT = cft.fft(im)
imRecoCFT = cft.ift(ksCFT)

# compare
figure()

subplot(2,5,1)
imshow(im, cmap='gray')
title('img'); colorbar()
subplot(2,5,2)
imshow(abs(ksFFT), norm="log")
title('img.FFT.abs'); colorbar()
subplot(2,5,3)
imshow(angle(ksFFT), cmap="hsv", vmin=-pi, vmax=pi)
title('img.FFT.ang'); colorbar()
subplot(2,5,4)
imshow(imRecoFFT.real, cmap='gray')
title('imRecoFFT.real'); colorbar()
subplot(2,5,5)
imshow(imRecoFFT.imag, cmap='gray')
title('imRecoFFT.imag'); colorbar()

subplot(2,5,6)
imshow(im, cmap='gray')
title('img'); colorbar()
subplot(2,5,7)
imshow(abs(ksCFT), norm="log")
title('img.CFT.abs'); colorbar()
subplot(2,5,8)
imshow(angle(ksCFT), cmap="hsv", vmin=-pi, vmax=pi)
title('img.CFT.ang'); colorbar()
subplot(2,5,9)
imshow(imRecoCFT.real, cmap='gray')
title('imRecoCFT.real'); colorbar()
subplot(2,5,10)
imshow(imRecoCFT.imag, cmap='gray')
title('imRecoCFT.imag'); colorbar()

show()
