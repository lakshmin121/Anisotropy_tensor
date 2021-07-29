"""
Testing the accuracy of ODF determined from anisotropy tensor.
______________________________________________
@ Lakshminarayanan Mohana Kumar
23rd July 2021
"""

import os
import sys
import numpy as np
import pandas as pd
sys.path.append("../artificial images/")
import skimage.io as skio
from fiborient import tensor2odf_2D, orient_tensor_2D
from fibfourier import fourier_orient_tensor, localised_fourier_orient_tensor
from matplotlib import pyplot as plt
from matplotlib_settings import *


# SET-UP
# Nf = 500  # number of fibres = number of phi values
dataDir = "../data/test_images_2D/vf20_ar50_tk50"
outDir = "tests_odf_from_anisotensor"
if not os.path.exists(outDir):
    os.mkdir(outDir)

def get_yticks(ymax, s):
    ymaxid = np.ceil(ymax / s)
    return np.arange(0, (ymaxid + 0.2) * s, s)

m = 0
for k in [0.1, 0.25, 0.5, 1, 5]:
    imgName = 'vm_m{0}k{1}'.format(m, k)
    img_fname = imgName + '.tiff'
    prob_fname = imgName + '_prob.csv'

    # Read image
    img = skio.imread(os.path.join(dataDir, img_fname), as_gray=True)

    # Fibre Orientation (Histogram)
    phiDF = pd.read_csv(os.path.join(dataDir, prob_fname))
    phiBins = phiDF['phiBins'].to_numpy()
    phiMin, phiMax = (round(np.min(phiBins), 1), round(np.max(phiBins), 1))

    phiHist = phiDF['phiHist'].to_numpy()
    phiHist = phiHist[:-1]
    dphi = np.mean(phiBins[1:] - phiBins[:-1])
    phiBinc = 0.5 * (phiBins[1:] + phiBins[:-1])
    phiRange = phiBins[-1] - phiBins[0]
    # print("Check: ", np.sum(phiHist) * dphi)


    # Theoretical tensor and ODF from histogram
    Q2_theo, A2_theo = orient_tensor_2D(phiHist, phiBinc)
    Q4_theo, A4_theo = orient_tensor_2D(phiHist, phiBinc, order=4)
    phiODF2_theo = tensor2odf_2D(phiBinc, A2_theo) * 2 * np.pi / 180
    # phiODF4_theo = tensor2odf_2D(phiBinc, (A2_theo, A4_theo)) * 2 * np.pi / 180


    # Tensor and ODF from image
    Q2, A2 = fourier_orient_tensor(img, windowName='hann')
    # Q4, A4 = fourier_orient_tensor(img,  windowName='hann', order=4)
    phiODF2 = tensor2odf_2D(phiBinc, A2) * 2 * np.pi / 180
    # phiODF4 = tensor2odf_2D(phiBinc, (A2, A4)) * 2 * np.pi / 180
    # print("Check2: ", np.sum(phiODF2) * dphi)
    # print("Check2: ", np.sum(phiODF4) * dphi)


    # Tensor and ODF from image (windowed approach)
    Q2w, A2w = localised_fourier_orient_tensor(img, x_width=50, y_width=50,
                                             x_step=20, y_step=20,
                                             windowName='hann', order=2)
    phiODF2w = tensor2odf_2D(phiBinc, A2w*2) * 2 * np.pi / 180
    print("Check2: ", np.sum(phiODF2) * dphi)
    # Q4w, A4w = localised_fourier_orient_tensor(img, x_width=50, y_width=50,
    #                                            x_step=20, y_step=20,
    #                                            windowName='hann', order=4)
    # phiODF4w = tensor2odf_2D(phiBinc, (A2w, A4w)) * 2 * np.pi / 180
    # print("Check2: ", np.sum(phiODF4w) * dphi)

    print("\n\nQ2_theo: \n", Q2_theo)
    print("\nQ2: \n", Q2)
    print("\nQ2w: \n", Q2w)
    print("\nA2_theo: \n", A2_theo)
    print("\nA2: \n", A2)
    print("\nA2w: \n", A2w)

    # # Error estimates
    # err2 = np.linalg.norm(A2 - A2_theo)
    # err4 = np.linalg.norm(A4 - A4_theo)
    # print("error 2nd order: ", err2)
    # print("error 4th order: ", err4)

    fig = plt.figure(figsize=(3.7, 2))
    ax = fig.gca()
    plt.bar(phiBins, np.append(phiHist, np.nan), width=0.9 * dphi, align='edge', lw=0)
    plt.plot(phiBinc, phiODF2_theo, color=np.asarray([176, 21, 21]) / 255, lw=0.75, linestyle='-', label='Theoretical')
    plt.plot(phiBinc, phiODF2, color=np.asarray([230, 100, 20]) / 255, lw=0.75, linestyle='--', label='FT')
    plt.plot(phiBinc, phiODF2w, color='black', lw=0.75, linestyle='dotted', label='Localised FT 2nd')
    # plt.plot(phiBinc, phiODF4w, color='gray', lw=0.75, linestyle='dotted', label='Localised FT 4th')
    legend = plt.legend(loc='upper center', ncol=3, numpoints=5, frameon=True, mode=None, fancybox=False, fontsize=10,
                        columnspacing=0.5, borderaxespad=0.15, edgecolor='gray')
    legend.get_frame().set_linewidth(0.5)

    xticks = phiBins[::3]
    ax.set_xlim([phiBins[0], phiBins[-1]])
    ymax = np.max(phiHist)
    if ymax > 0.01:
        yticks = get_yticks(ymax, s=0.01)
    else:
        yticks = get_yticks(ymax, s=0.005)
    ax.set_ylim([yticks[0], yticks[-1]])
    ax.set_yticks(yticks)
    ax.set_xticks(xticks)
    plt.xticks(rotation=90)
    plt.yticks(rotation=90)
    ax.set_xticklabels(xticks)
    ax.set_xlabel("$\phi$ [degrees]")
    ax.set_ylabel("p($\phi$)")

    fig.savefig(os.path.join(outDir, imgName + '_fit.tiff'), dpi=300)
