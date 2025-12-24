from numpy import *
from matplotlib.pyplot import *
import slime
from utility.ndimshow import *

nPix = 256

# 2D
arrPhant = slime.genPhant(nAx=2, nPix=nPix)
arrM0 = slime.Enum2M0(arrPhant)*slime.genPhMap(2, nPix)

ndimshow(arrM0, figure(figsize=(9,3), dpi=120))

# 3D
arrPhant = slime.genPhant(nAx=3, nPix=nPix)
arrM0 = slime.Enum2M0(arrPhant)*slime.genPhMap(3, nPix)

ndimshow(arrM0, figure(figsize=(9,3), dpi=120))

show()