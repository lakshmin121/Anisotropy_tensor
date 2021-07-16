"""
Ex-1: basic implementation of Fourier transform (FT) for detection of fibre orientation.
Objective: Simple detection of the orientation of a unidirectional line.
------------------------------------------------------------------------------------------------
@ Lakshminarayanan Mohana Kumar
11th July 2021
"""
import os
from os import path
from glob import glob
import numpy as np
from matplotlib_settings import *
from matplotlib import pyplot as plt
from scipy.fftpack import fft2, fftshift
from skimage import img_as_float
from skimage.transform import rotate
from skimage.filters import window
# import skimage.io  as skio
# from skimage import img_as_float, img_as_ubyte
# from fiborient import orient_tensor_2D


dataDir = "../data/test_images_1"
fileSelection = "*_theta90_phi*.png"  # regex
outDir = "../data/FTex2_res"
imgFname = "img3_array_of_oblique_lines.tiff"
imgFTFname = "img3_array_of_oblique_lines_FT.tiff"

if not path.exists(outDir):
    os.mkdir(outDir)

# General
m, n = 81, 81
img = np.zeros((m, n))

# Single horizontal line
# img[m//2+1, :] = 1

# array of horizontal lines
# sp = 20
# img[::sp, :] = 1

# array of lines in different orientation
phivals = [0, 30, 60]
img0 = img.copy()
sp = 10
img0[::sp, :] = 1
for phi in phivals:
    img = img + rotate(img0, phi)

# FT
imgFT = np.abs(fftshift(fft2(img)))

fig1 = plt.figure(figsize=(1.5, 1.5))
plt.axis('off')
plt.imshow(img, cmap='gray')
# plt.tight_layout()

fig2 = plt.figure(figsize=(1.5, 1.5))
plt.axis('off')
plt.imshow(np.log(imgFT), cmap='magma')
# plt.tight_layout()

# plt.show()
fig1.savefig(path.join(outDir, imgFname), dpi=150)
fig2.savefig(path.join(outDir, imgFTFname), dpi=150)
