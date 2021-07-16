"""
Studying the influence of the parameter sigma on performance of structure tensor.
------------------------------------------------------------------------------------------------
@ Lakshminarayanan Mohana Kumar
11th July 2021
"""

from os import path
from glob import glob
import numpy as np
from matplotlib_settings import *
from matplotlib import pyplot as plt
import skimage.io  as skio
from skimage import img_as_float
from skimage.transform import EuclideanTransform, warp
import skimage.feature as skfeature
from fiborient import orient_tensor_2D

dataDir = "../data/test_images_1"
phi_choices = np.arange(0, 95, 10)

# # randomly choosing 3 phi values
# nsel = 3
# sel = np.random.randint(0, len(phi_choices), nsel)

sel = [0, 3, 6, 8]
# read and combine images
for itr, s in enumerate(sel):
    fname = "fibres_vf0.01_theta90_phi{}_rescaled.png".format(phi_choices[s])
    if itr == 0:
        img = img_as_float(skio.imread(path.join(dataDir, fname), as_gray=True))
    else:
        img0 = skio.imread(path.join(dataDir, fname), as_gray=True)
        tform = EuclideanTransform(translation=s)
        img = img + warp(img0, tform.inverse)

img = img / len(sel)

# Structure Tensor
# ST = [[fx*fx, fx*fy], [fy*fx, fy*fy]]
ST = skfeature.structure_tensor(img, sigma=1, order='xy')  # estimation of structure tensor at every pixel
STxx, STxy, STyy = ST
fib_mask = img > 0

# Method-1: Orientation Distribution Function (ODF)
phi_px = -np.rad2deg(0.5 * np.arctan2(2*STxy, (STyy - STxx)))  # local (principal) orientation
p_phi, phi_bins = np.histogram(phi_px[fib_mask], phi_choices, density=True)  # ODF

# 2D orientation tensor from ODF
Q_phi, A_phi = orient_tensor_2D(p_phi, phi_bins)
print(Q_phi)
print(A_phi)


# Q = [np.sum(Sij[fib_mask]) for Sij in [STxx, STxy, STxy, STyy]]
# Q = np.array(Q).reshape((2, 2)) / np.sum(fib_mask)
# print('\n', Q)
#
# print('\n', np.divide(Q_phi, Q))

# plt.imshow(img, cmap='gray')
# plt.show()
