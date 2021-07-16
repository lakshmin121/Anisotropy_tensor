"""
Ex-1: basic implementation of structure tensor for detection of fibre orientation.
Objective:
Study the accuracy of structure tensor based detection of fibre orientation.
Images used have unidirectional array of fibres (single phi value).
    1. estimation of gradient-based structure tensor at every pixel.
    2. estimation of local fibre orientation at every pixel.
    3. masking with fibre regions to select only fibre orientation (separated from background).
    4. probability distribution of orientation.
    5. orientation and anisotropy tensors for the orientation state.
------------------------------------------------------------------------------------------------
@ Lakshminarayanan Mohana Kumar
10th July 2021
"""

from os import path
from glob import glob
import numpy as np
from matplotlib_settings import *
from matplotlib import pyplot as plt
import skimage.io  as skio
import skimage.feature as skfeature
from fiborient import orient_tensor_2D

dataDir = "../data/test_images_1"
fileSelection = "*_theta90_phi*.png"  # regex
outDir = "../data/ex1_res"
outFname = "theta90_s1"

filenames = glob(path.join(dataDir, fileSelection))
numFiles = len(filenames)
print("{} files detected.".format(numFiles))

# --------------------------------------------------------------------------------------------------
# Function Definitions


def fibre_orientation_plot(fibre_img, orientation_dataDeg):
    m, n = fibre_img.shape
    if orientation_dataDeg.ndim < 2:
        orientation_dataDeg = orientation_dataDeg.reshape(fibre_img.shape)
    assert fibre_img.shape == orientation_dataDeg.shape

    fib_indices = np.nonzero(fibre_img)  # extracting locations (indices) of fibres in the image
    orient_vals = orientation_dataDeg[fib_indices]

    f = plt.figure(figsize=(2.25, 1.5), dpi=120, frameon=False)
    ax = f.gca()
    ax.axis('off')
    x = fib_indices[1]
    y = m - fib_indices[0]
    plt.scatter(x, y, c=orient_vals, cmap='viridis', vmin=-90, vmax=90)
    cticks = np.arange(-90, 95, 30)
    cbar = plt.colorbar(ticks=cticks)
    plt.text(0.5, 1, "mean $\phi$= {:2.1f}".format(orient_vals.mean()))

    return f, fib_indices, orient_vals


def fibre_orientation_hist(orient_vals, bins=np.arange(-90, 95, 15)):
    f = plt.figure(figsize=(2.25, 1.5), dpi=120)
    ax = f.gca()
    hist, b, _ = ax.hist(orient_vals, bins=bins, density=True)
    ax.set_xticks(np.arange(-90, 95, 30))
    ax.set_xlabel("$\phi$ [deg]")
    ax.set_ylabel("p($\phi$)")
    return f, hist, bins


def extract_phiID(filename):
    fparts = path.split(filename)
    fNameparts = fparts[1].split('_')
    phiDetect = lambda x: x.startswith('phi')
    phiID = [p for p in filter(phiDetect, fNameparts)]

    return phiID[0]

# --------------------------------------------------------------------------------------------------
# Analysis


with open(path.join(outDir, outFname+'.txt'), 'w+') as fout:
    for fname in filenames:
        phiID = extract_phiID(fname)
        print('\n', phiID)
        # Read file
        img = skio.imread(fname, as_gray=True)  # read image
        print("Image shape: ", img.shape)

        # Structure Tensor
        # ST = [[fx*fx, fx*fy], [fy*fx, fy*fy]]
        ST = skfeature.structure_tensor(img, sigma=1, order='xy')  # estimation of structure tensor at every pixel
        STxx, STxy, STyy = ST
        phi_px = -np.rad2deg(0.5 * np.arctan2(2*STxy, (STyy - STxx)))  # local orientation
        # print(phi_px.min(), phi_px.max())

        phi_plot, fib_indices, phi_vals = fibre_orientation_plot(img, phi_px)
        phi_hist, p_phi, phi_bins = fibre_orientation_hist(phi_vals)

        # 2D orientation tensor
        Q_phi, A_phi = orient_tensor_2D(p_phi, phi_bins)
        print(Q_phi)
        print(A_phi)

        # phi_plot.savefig(path.join(outDir, outFname+'_'+phiID+"_orient_plot.png"))
        # phi_hist.savefig(path.join(outDir, outFname+'_'+phiID+"_hist.png"))

        # fout.write(phiID+'\n')
        # fout.write("Q = ")
        # fout.write(np.array2string(Q_phi, prefix='Q=', precision=3))
        # fout.write("\nA = ")
        # fout.write(np.array2string(A_phi, prefix='A=', precision=3))
        # fout.write('\n\n')
