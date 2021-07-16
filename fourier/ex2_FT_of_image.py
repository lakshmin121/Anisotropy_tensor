"""
Basic implementation of Fourier Transform of an image with an array of lines.
Objective: The orientation tensors obtained from the ODF (using structure tensor), the Fourier space,
and the theoretical distribution are compared.
------------------------------------------------------------------------------------------------
@ Lakshminarayanan Mohana Kumar
14th July 2021
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
from fiborient import orient_tensor_2D, theo_orient_tensor_2D
from skimage.filters import window
from scipy.fftpack import fft2, fftshift

dataDir = "../data/test_images_1"
phi_choices = np.arange(0, 95, 10)

# choosing phi values
sel = [3, 7, 9]  # combining fibre of different orientations.
Q_theo = np.zeros((2, 2))
# read and combine images
for itr, s in enumerate(sel):
    phi = phi_choices[s]
    Q_theo = Q_theo + theo_orient_tensor_2D(phi)  # orientation tensor theoretically calculated from values of phi
    fname = "fibres_vf0.01_theta90_phi{}_rescaled.png".format(phi)  # image file to be read
    if itr == 0:
        img = img_as_float(skio.imread(path.join(dataDir, fname), as_gray=True))
    else:
        img0 = skio.imread(path.join(dataDir, fname), as_gray=True)
        tform = EuclideanTransform(translation=s//2)  # translating the image by a small amount before combining
        img = img + warp(img0, tform.inverse)  # combining images

img = img / len(sel)

# Theoreticcal orientation tensor of the combined array of fibres
Q_theo = Q_theo / len(sel)  # orientation tensor
A_theo = Q_theo - 0.5*np.eye(2)  # anisotropy tensor

print("Theoretical:")
print("Q: ")
print(Q_theo,  ' trace: ', np.trace(Q_theo))
print("A: ")
print(A_theo,  ' trace: ', np.trace(A_theo))

# Structure Tensor
# ST = [[fx*fx, fx*fy], [fy*fx, fy*fy]]
ST = skfeature.structure_tensor(img, sigma=1, order='xy')  # estimation of structure tensor at every pixel
STxx, STxy, STyy = ST
fib_mask = img > 0

# Method-1: Orientation Distribution Function (ODF)
phi_px = -np.rad2deg(0.5 * np.arctan2(2*STxy, (STyy - STxx)))  # local (principal) orientation
p_phi, phi_bins = np.histogram(phi_px[fib_mask], phi_choices, density=True)  # ODF

# 2D orientation tensor from the discrete ODF obtained from structure tensor
Q_phi, A_phi = orient_tensor_2D(p_phi, phi_bins)  # using custom function.
print("\nFrom ODF: ")
print("Q: ")
print(Q_phi, ' trace: ', np.trace(Q_phi))
print("A: ")
print(A_phi, ' trace: ', np.trace(A_phi))

# Fourier Transform
wimg = img * window('hann', img.shape)  # windowing the image
wimgFT = np.abs(fftshift(fft2(wimg)))  # FT

# Calculation of orientation tensor in Fourier space:
m, n = wimgFT.shape
u = np.arange(0, m) - m//2
v = np.arange(0, n) - n//2
uu, vv = np.meshgrid(u, v)  # all points in Fourier space
r = np.sqrt(uu**2 + vv**2)  # radial distance to each point.
r = np.where(r==0, 1, r)

ku = np.divide(uu, r)  # spatial frequency (unit vector component) in u-direction (x)
kv = np.divide(vv, r)  # spatial frequency (unit vector component) in v-direction (y)
E = np.sum(wimgFT)  # Total energy in Fourier space (sum of values at all points)

# elements of the orientation tensor
Quu = np.sum(ku**2 * wimgFT) / E
Quv = np.sum(ku*kv * wimgFT) / E
Qvv = np.sum(kv**2 * wimgFT) / E

Q = np.array([[Quu, Quv], [Quv, Qvv]])  # orientation tensor in Fspace
A = Q - 0.5 * np.eye(2)
print("\nFrom Fourier space: ")
print("Q': ")
print(Q, ' trace: ', np.trace(Q))
print("A': ")
print(A,  ' trace: ', np.trace(A))

# estimating the actual orientation tensor from the tensor obtained in Fspace
rot = -np.pi/2  # The axes in Fourier space are rotated by 90 degress w.r.t the axes in original image.
R = np.array([[np.cos(rot), np.sin(rot)], [-np.sin(rot), np.cos(rot)]])
print("\nR: ")
print(R)

# transformation of orientation tensor for 90 degrees rotation.
Q2 = R @ Q @ R.T  # Q is from Fourier space. Q2 is in original image.
print("\nRQ'R':")
print(Q2, ' trace: ', np.trace(Q2))

# ------------------------
# Figures

fig1 = plt.figure()
plt.imshow(wimg, cmap='gray')

fig2 = plt.figure()
im2 = plt.imshow(np.log(wimgFT))
plt.colorbar(im2)

plt.show()
