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
from skimage.transform import warp_polar
from matplotlib_settings import *
from fibfourier import imgfft
from fiborient import tensor2odf_2D, orient_tensor_2D
from multiprocessing import cpu_count, Pool
from functools import partial
from scipy.ndimage.filters import gaussian_filter1d
from orientation_probabilities import map_rv2cos, map_rv2tan, map_rv2arctan, ratio_distr



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


def energy_boundary(imageFT, step=1):
    rvals, energy = FTbound(imageFT, step=step)
    grad = np.gradient(energy)
    grad = gaussian_filter1d(grad, step / 4)  # smoothens the gradVals values to a continuous func.
    gradmax_idx = np.argmax(grad)  # location of maximum gradVals.
    ridx = gradmax_idx + np.argmin(np.abs(grad[gradmax_idx:]))

    return ridx, rvals, energy, grad


def polarFT_hist(imageFT, rmin=None, rmax=None, correction=True):
    if rmin is None:
        rmin = 0
    if rmax is None:
        rmax = int(np.min(imageFT.shape) // 2)
    imageFTpolar = warp_polar(imageFT, radius=rmax)
    imageFTpolar = imageFTpolar.T
    hist = np.sum(imageFTpolar[rmin:], axis=0)  # Radial sum
    m = len(hist) // 2
    hist = hist[:m] + hist[m:]
    histsum = np.sum(hist)
    hist = hist / histsum

    # Correction
    if correction:
        excess = np.sum(imageFT[:rmin + 1])
        hist = hist * (1 + excess / histsum)  # approximate
        hist = hist / np.sum(hist)

    return hist, imageFTpolar


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
    # imgaxes = ['x', 'y']
    # a = np.random.choice(1, size=1)[0]
    # a = imgaxes[a]  # any one of 'x' or 'y' projections.
    a = 'y'
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


    # # -----------------------------------------------------------------------------------------------------------------
    # # FOURIER ANALYSIS OF Z-PROJECTION
    # # Fourier Transform
    # imgzFT = imgfft(imgz, windowName=windowName, zpad_factor=zpad_factor)
    # imgzFT -= np.mean(imgzFT)  # removing the average (similar to thresholding).
    # imgzFTlog = np.nan_to_num(np.log(1+imgzFT), 0)  # spectrum in log scale + convert any nan generated to zero.
    #
    # # Bounds in FT space as a minimization problem
    # rindx, rVals, energyVals, gradVals = energy_boundary(imgzFTlog, step=zpad_factor*5)
    # rBound = rVals[rindx]
    #
    # # Polar warp
    # phiHistp, imgzFTpolar = polarFT_hist(imgzFT, rmin=int(fibdia * zpad_factor), rmax=rBound)
    # phiHistp = np.roll(phiHistp, -90)
    # phiBinsp = np.arange(0, 181, 1)
    # phiBinspc = 0.5 * (phiBinsp[1:] + phiBinsp[:-1])
    #
    # # Tensor from FT
    # Qphi, Aphi = orient_tensor_2D(phiHistp, phiBinspc)
    # phiODF2 = tensor2odf_2D(phiBinspc, Aphi) * 2 * np.pi / 180
    # # dphip = np.mean(phiBinsp[1:] - phiBinsp[:-1])
    # print("Check: Total probability from FT ODF (phi) = ", np.trapz(phiODF2, phiBinspc))
    #
    # # Fibre Orientation (Histogram) used during generation of image.
    # phiDF = pd.read_csv(os.path.join(dataDir, phifName))  # read histogram data from CSV file.
    # phiBins = phiDF['phiBins'].to_numpy()
    # phiMin, phiMax = (round(np.min(phiBins), 0), round(np.max(phiBins), 0))
    # phiBinc = 0.5 * (phiBins[1:] + phiBins[:-1])
    # phiRange = phiBins[-1] - phiBins[0]
    #
    # phiHist = phiDF['phiHist'].to_numpy()
    # phiHist = phiHist[:-1]
    # dphi = np.mean(phiBins[1:] - phiBins[:-1])
    # print("Check: Total probability from theoretical PMF (phi) = ", np.trapz(phiHist, phiBinc))
    #
    # # Theoretical tensor and ODF from histogram
    # Q2_theo, A2_theo = orient_tensor_2D(phiHist, phiBinc)
    # phiODF2_theo = tensor2odf_2D(phiBinc, A2_theo) * 2 * np.pi / 180
    # print("Check: Total probability from theoretical ODF (phi) = ", np.sum(phiODF2_theo) * dphi)
    #
    # figb = plt.figure(figsize=(4, 2), dpi=300)
    # ax = figb.gca()
    # # ax.bar(phiBinsExtnd[:-1], phiHistExtnd, width=dphi, align='edge',linewidth=0, alpha=0.5)
    # ax.bar(phiBins[:-1], phiHist, width=dphi, align='edge', linewidth=0, alpha=0.5)
    # ax.bar(phiBinsp[:-1], phiHistp, width=1.0, align='edge', linewidth=0, alpha=0.5)
    # plt.plot(phiBinc, phiODF2_theo, color=np.asarray([176, 21, 21]) / 255, lw=0.75, linestyle='-', label='Theoretical')
    # plt.plot(phiBinspc, phiODF2, color=np.asarray([230, 100, 20]) / 255, lw=0.75, linestyle='--', label='FT')
    # ax.set_xticks(phiBinsp[::30])
    # ax.set_xticklabels(phiBinsp[::30])
    # ax.legend(ncol=2, handlelength=1, loc='upper center')
    # figb.savefig(os.path.join(outDir, imgName + '_z_hist.tiff'))
    #
    # np.save(os.path.join(outDir, 'phiODF2'), phiODF2)
    #
    # # -----------------------------------------------------------------------------------------------------------------
    # FOURIER ANALYSIS OF X- or Y- PROJECTION
    # Fourier Transform
    imgaFT = imgfft(imga, windowName=windowName, zpad_factor=zpad_factor)
    imgaFT -= np.mean(imgaFT)  # removing the average (similar to thresholding).
    imgaFTlog = np.nan_to_num(np.log(1 + imgaFT), 0)  # spectrum in log scale + convert any nan generated to zero.

    # Bounds in FT space as a minimization problem
    rindx, rVals, energyVals, gradVals = energy_boundary(imgaFTlog, step=zpad_factor*5)
    rBounda = rVals[rindx]

    # Polar warp
    alphaHistp, imgaFTpolar = polarFT_hist(imgaFT, rmin=int(fibdia * zpad_factor), rmax=rBounda)
    alphaHistp = np.roll(alphaHistp, -90)
    alphaBinsp = np.arange(0, 181, 1)
    alphaBinspc = 0.5 * (alphaBinsp[1:] + alphaBinsp[:-1])

    # Tensor from FT
    Qalpha, Aalpha= orient_tensor_2D(alphaHistp, alphaBinspc)
    alphaODF2 = tensor2odf_2D(np.pi - alphaBinspc, Aalpha) * 2 * np.pi / 180
    # dalphap = np.mean(alphaBinsp[1:] - alphaBinsp[:-1])
    print("Check: Total probability from FT ODF (alpha) = ", np.trapz(alphaODF2, alphaBinspc))

    # TODO: add axis labels to all figures.
    figara = plt.figure(figsize=(2, 2), dpi=600)
    axara = figara.gca()
    axara.plot(rVals, energyVals / energyVals.max(), color='0.1', lw=0.75, label='Energy')
    axara.plot(rVals, gradVals / gradVals.max(), color='0.5', lw=0.75, alpha=0.5, label='Gradient')
    axara.scatter(rBounda, gradVals[rindx] / gradVals.max(), marker='o', s=7, c='w', lw=0.5, edgecolors='0.1')
    axara.set_ylim([-0.25, 1.25])
    axara.set_yticks(np.arange(0, 1.1, 0.25))
    axara.legend(ncol=2, handlelength=1, loc='upper center')


    figa = plt.figure(figsize=(2, 2), dpi=600)
    axa = figa.gca()
    axa.imshow(imgaFTlog, cmap='magma')
    xc, yc = np.asarray(imgaFT.shape) // 2
    angle = np.deg2rad(np.arange(0, 360))
    x, y = xc + rBounda * np.cos(angle), yc + rBounda * np.sin(angle)
    axa.plot(x, y, color='0.95', lw=0.5)
    # plt.show()

    figara.savefig(os.path.join(outDir, imgName + '_{}_energy.tiff'.format(a)))
    figa.savefig(os.path.join(outDir, imgName + '_{}_spectrum.tiff'.format(a)))


    figb = plt.figure(figsize=(4, 2), dpi=300)
    ax = figb.gca()
    ax.bar(alphaBinsp[:-1], alphaHistp, width=1.0, align='edge', linewidth=0, alpha=0.5)
    plt.plot(alphaBinspc, alphaODF2, color=np.asarray([230, 100, 20]) / 255, lw=0.75, linestyle='--', label='FT')
    ax.set_xticks(alphaBinsp[::30])
    ax.set_xticklabels(alphaBinsp[::30])
    ax.legend(ncol=2, handlelength=1, loc='upper center')
    figb.savefig(os.path.join(outDir, imgName + '_y_hist.tiff'))

    np.save(os.path.join(outDir, 'alphaODF2'), alphaODF2)

    # -----------------------------------------------------------------------------------------------------------------
    # ESTIMATION OF TAN THETA
    # Fibre Orientation (Histogram) used during generation of image.
    thetaDF = pd.read_csv(os.path.join(dataDir, thtfName))  # read histogram data from CSV file.
    thetaHist = thetaDF['thetaHist'].to_numpy()
    thetaHist = thetaHist[:-1]

    # Redundant code %%%%%%%%%%%%%%%%%%%%%%%%%
    phiBinsp = np.arange(0, 181, 1)
    phiBinspc = 0.5 * (phiBinsp[1:] + phiBinsp[:-1])
    alphaBinsp = np.arange(0, 181, 1)
    alphaBinspc = 0.5 * (alphaBinsp[1:] + alphaBinsp[:-1])


    phiODF2 = np.load(os.path.join(outDir, 'phiODF2.npy'))
    alphaODF2 = np.load(os.path.join(outDir, 'alphaODF2.npy'))
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    phiBinc = np.deg2rad(phiBinspc)
    cos_phiBinc = np.cos(phiBinc)
    cos_phiPMF = map_rv2cos(phiBinc, phiODF2 * 180 / np.pi, cos_phiBinc)

    alpBinc = np.deg2rad(alphaBinspc - 89)
    print(alpBinc)
    tan_alpBinc = np.tan(alpBinc)
    tan_alpPMF = map_rv2tan(alpBinc, alphaODF2 * 180 / np.pi, tan_alpBinc)

    # Ratio distribution
    thtBinc = alpBinc
    tan_thtBinc = np.tan(thtBinc)
    tan_thtPMF = ratio_distr(tan_alpBinc, cos_phiBinc[::-1], tan_thtBinc, tan_alpPMF, cos_phiPMF[::-1])
    print("Total tan theta probability: ", np.trapz(tan_thtPMF, tan_thtBinc))

    thtPMF = map_rv2arctan(tan_thtBinc, tan_thtPMF, thtBinc)

    fig_cosphi = plt.figure()
    ax = fig_cosphi.gca()
    ax.plot(cos_phiBinc, cos_phiPMF)
    fig_cosphi.savefig(os.path.join(outDir, 'cos_phi.tiff'))

    fig_tanalp = plt.figure()
    ax = fig_tanalp.gca()
    ax.plot(tan_alpBinc, tan_alpPMF)
    fig_tanalp.savefig(os.path.join(outDir, 'tan_alp.tiff'))

    fig_tantht = plt.figure()
    ax= fig_tantht.gca()
    ax.plot(tan_thtBinc, tan_thtPMF)
    fig_tantht.savefig(os.path.join(outDir, 'tan_tht.tiff'))

    fig_tht2Distr = plt.figure(figsize=(2, 2))
    ax = fig_tht2Distr.gca()
    thetaBins = np.deg2rad(thetaDF['thetaBins'].to_numpy()) - np.pi/2
    width = np.mean(thetaBins[1:] - thetaBins[:-1])
    ax.bar(thetaBins, thetaDF['thetaHist'].to_numpy() * 180 / np.pi, width=0.9*width)
    ax.plot(thtBinc, thtPMF, lw=0.75)
    ax.set_xlabel("$\\theta$")
    ax.set_ylabel("$p(\\theta)$")

    fig_tht2Distr.savefig(os.path.join(outDir, 'tht2Distr.tiff'))  # TODO: change filename.
