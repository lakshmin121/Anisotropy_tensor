"""
Code to measure 3D fibre orientation from two mutually orthogonal 2D projections.
_________________________________________________________________________________
@ Lakshminarayanan Mohana Kumar
19th Aug 2021
"""

import os
import sys
sys.path.append("..")
from glob import glob
import numpy as np
import pandas as pd
import skimage.io as skio
from skimage.transform import warp_polar, rotate
from skimage.filters import window
from scipy.fftpack import fft2, fftshift
from matplotlib import pyplot as plt
from matplotlib_settings import *
from fibfourier import imgfft
from fiborient import tensor2odf_2D, orient_tensor_2D
from multiprocessing import cpu_count, Pool
from functools import partial
from scipy.ndimage.filters import gaussian_filter1d


dataDir = "../data/art_images"
outDir = "../data/art3D_FT"
zpad_factor = 10.5
windowName = 'hann'


def confined_energy(imageFT, xx, yy, xc, yc, r):
    mask = (xx - xc) ** 2 + (yy - yc) ** 2 <= r ** 2
    return int(np.sum(imageFT * mask))

def FTbound(imageFT, step=1):
    imageFT = np.nan_to_num(imageFT, 0)
    xc, yc = np.asarray(imageFT.shape) // 2
    s = min(*imageFT.shape)
    xv = np.arange(0, s)
    yv = xv
    xx, yy = np.meshgrid(xv, yv)
    rvals = np.arange(0, s // 2, step)

    apool = Pool(processes=cpu_count() - 1)
    area_vals = apool.map(partial(confined_energy, imageFT, xx, yy, xc, yc), rvals)
    return rvals, np.array(area_vals, dtype=np.float)

# MAIN
if __name__ == '__main__':
    if not os.path.exists(outDir):
        os.mkdir(outDir)

    # List of all 3D images
    filenames = glob(os.path.join(dataDir, '*.nc'))
    nfiles = len(filenames)
    print("{} nc files detected.".format(nfiles))
    print(filenames)

    # Pick a 3D image
    imgfName = os.path.split(filenames[0])[-1]
    imgName = os.path.splitext(imgfName)[0]
    phifName = imgName + '_phiHist.csv'
    thtfName = imgName + '_thetaHist.csv'
    infoName = imgName + '.txt'
    print("Image filename: " + imgfName)
    print("Image: " + imgName)

    # Read two projected images:
    z_projfName = imgName + '_zproj.tiff'  # z - projection
    imgaxes = ['x', 'y']
    a = np.random.choice(1, size=1)[0]
    a = imgaxes[a]  # any one of 'x' or 'y' projections.
    a_projfName = imgName + '_{}proj.tiff'.format(a)

    imgz = skio.imread(os.path.join(dataDir, z_projfName), as_gray=True)
    imga = skio.imread(os.path.join(dataDir, a_projfName), as_gray=True)


    # Fibre properties:
    fibdia, fiblen =0, 0
    with open(os.path.join(dataDir, infoName), 'r') as f:
        lines = f.readlines()
    flag = False
    for line in lines:
        words = line.split()
        if '#Image' in words:
            flag=True
        if flag:
            if 'Diameter:' in words:
                fibdia = int(words[words.index('Diameter:') + 1])
            if 'Length:' in words:
                fiblen = int(words[words.index('Length:') + 1])

    assert fibdia > 0, "incorrect fibre diameter"
    assert fiblen > 0, "incorrect fibre length"
    print("fibre length: {} px, diameter: {} px".format(fiblen, fibdia))



    # Fourier Analysis of Z projection
    imgzFT = imgfft(imgz, windowName=windowName, zpad_factor=zpad_factor)
    imgzFT -= np.mean(imgzFT)
    imgzFTlog = np.nan_to_num(np.log(1+imgzFT), 0)
    figz = plt.figure(figsize=(2, 2), dpi=600)
    axz = figz.gca()
    axz.imshow(imgzFTlog, cmap='magma')


    # Bounds in FT space as a minimization problem
    # print(imgzFTlog.min(), imgzFTlog.max())
    rVals, areaVals = FTbound(imgzFTlog, step=zpad_factor*5)
    dArea = np.gradient(areaVals)
    dArea = gaussian_filter1d(dArea, zpad_factor / 4)  # smoothens the gradient values to a continuous func.
    armax_idx = np.argmax(dArea)
    ridx = armax_idx + np.argmin(np.abs(dArea[armax_idx:]))
    rbound = rVals[ridx]
    print("rbound: ", rbound)

    figar = plt.figure(figsize=(2, 2), dpi=600)
    axar = figar.gca()
    axar.plot(rVals, areaVals / areaVals.max(), color='0.1', lw=0.75, label='Energy')
    axar.plot(rVals, dArea / dArea.max(), color='0.5', lw=0.75, alpha=0.5, label='Gradient')
    axar.scatter(rbound, dArea[ridx] / dArea.max(), marker='o', s=7, c='w', lw=0.5, edgecolors='0.1')
    axar.set_ylim([-0.25, 1.25])
    axar.set_yticks(np.arange(0, 1.1, 0.25))
    axar.legend(ncol=2, handlelength=1, loc='upper center')

    xc, yc = np.asarray(imgzFT.shape) // 2
    angle = np.deg2rad(np.arange(0, 360))
    x, y = xc + rbound * np.cos(angle), yc + rbound * np.sin(angle)
    axz.plot(x, y, color='0.95', lw=0.5)
    # plt.show()
    figar.savefig(os.path.join(outDir, imgName+'_z_energy.tiff'))
    figz.savefig(os.path.join(outDir, imgName+'_z_spectrum.tiff'))

    # Polar warp
    imgzFTpolar = warp_polar(imgzFT, radius=rbound)
    imgzFTpolar = imgzFTpolar.T
    r0 = int(fibdia * zpad_factor)
    phiHistp = np.sum(imgzFTpolar[r0:], axis=0)  # Radial sum
    m = len(phiHistp) // 2
    phiHistp = phiHistp[:m] + phiHistp[m:]
    phiHistsum = np.sum(phiHistp)
    phiHistp = phiHistp / phiHistsum
    phiHistp = np.roll(phiHistp, -90)
    phiBinsp = np.arange(0, 181, 1)
    phiBinspc = 0.5 * (phiBinsp[1:] + phiBinsp[:-1])

    # Correction
    excess = np.sum(imgzFTpolar[:r0 + 1])
    phiHistp = phiHistp * (1 + excess / phiHistsum)  # approximate
    phiHistp = phiHistp / np.sum(phiHistp)

    # Tensor from FT
    Qphi, Aphi = orient_tensor_2D(phiHistp, phiBinspc)
    phiODF2 = tensor2odf_2D(phiBinspc, Aphi) * 2 * np.pi / 180
    dphip = np.mean(phiBinsp[1:] - phiBinsp[:-1])
    print("Check: Total probability from FT ODF = ", np.sum(phiODF2) * dphip)

    # Fibre Orientation (Histogram) used during generation of image.
    phiDF = pd.read_csv(os.path.join(dataDir, phifName))  # read histogram data from CSV file.
    phiBins = phiDF['phiBins'].to_numpy()
    phiMin, phiMax = (round(np.min(phiBins), 0), round(np.max(phiBins), 0))
    phiBinc = 0.5 * (phiBins[1:] + phiBins[:-1])
    phiRange = phiBins[-1] - phiBins[0]

    phiHist = phiDF['phiHist'].to_numpy()
    phiHist = phiHist[:-1]
    dphi = np.mean(phiBins[1:] - phiBins[:-1])
    print("Check: Total probability from theoretical PMF = ", np.sum(phiHist) * dphi)

    phiBinsExtnd = np.arange(0, 361, dphi)
    phiHistExtnd = np.concatenate((phiHist / 2, phiHist / 2))
    msg = "len(phiBinsExtnd) = {0} while len(phiHistExtnd) = {1}".format(len(phiBinsExtnd), len(phiHistExtnd))
    assert len(phiBinsExtnd) == len(phiHistExtnd) + 1, print(msg)

    # Theoretical tensor and ODF from histogram
    Q2_theo, A2_theo = orient_tensor_2D(phiHist, phiBinc)
    phiODF2_theo = tensor2odf_2D(phiBinc, A2_theo) * 2 * np.pi / 180
    print("Check: Total probability from theoretical ODF = ", np.sum(phiODF2_theo) * dphi)

    figb = plt.figure(figsize=(4, 2), dpi=300)
    ax = figb.gca()
    # ax.bar(phiBinsExtnd[:-1], phiHistExtnd, width=dphi, align='edge',linewidth=0, alpha=0.5)
    ax.bar(phiBins[:-1], phiHist, width=dphi, align='edge', linewidth=0, alpha=0.5)
    ax.bar(phiBinsp[:-1], phiHistp, width=1.0, align='edge', linewidth=0, alpha=0.5)
    plt.plot(phiBinc, phiODF2_theo, color=np.asarray([176, 21, 21]) / 255, lw=0.75, linestyle='-', label='Theoretical')
    plt.plot(phiBinspc, phiODF2, color=np.asarray([230, 100, 20]) / 255, lw=0.75, linestyle='--', label='FT')
    ax.set_xticks(phiBinsp[::30])
    ax.set_xticklabels(phiBinsp[::30])
    ax.legend(ncol=2, handlelength=1, loc='upper center')
    figb.savefig(os.path.join(outDir, imgName + '_hist.tiff'))
