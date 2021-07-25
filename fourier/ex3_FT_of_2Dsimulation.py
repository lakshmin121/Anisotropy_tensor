"""
Implementation of the FT analysis of 2D FRC images.
Objective: Compare the ODF obtained from FT orientation tensor with original ODF used for the 2D simulation of the
artificial FRC image.
Challenges:
1. A complete efficient implementation of ODF from FT.
2. Develop a method to compare ODF from 2 similar distributions with slight error.
------------------------------------------------------------------------------------------------
@ Lakshminarayanan Mohana Kumar
22nd July 2021
"""

from os import path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import skimage.io  as skio

from fibfourier import fourier_orient_tensor_2order, fourier_orient_tensor_4order
from fiborient import tensor2odf_2D

dataDir = "../data/test_images_2D/vf20p"

m = 0
for k in [0.1, 0.25, 0.5, 1, 5]:
    imgName = 'vm_m{0}k{1}'.format(m, k)
    img_fname = imgName + '.tiff'
    prob_fname = imgName + '_prob.csv'

    # Read image and histogram data (csv)
    img = skio.imread(path.join(dataDir, img_fname), as_gray=True)
    histDF = pd.read_csv(path.join(dataDir, prob_fname))
    phiBins = histDF['phiBins'].to_numpy()
    phiHist = histDF['phiHist'].to_numpy()
    dphi = np.mean(phiBins[1:] - phiBins[:-1])
    phiRange = phiBins[-1] - phiBins[0]
    print(dphi)

    Q2, A2 = fourier_orient_tensor_2order(img, 'hann')
    Q4, A4 = fourier_orient_tensor_4order(img, 'hann')
    phiVals = np.arange(phiBins[0], phiBins[-1], 2)
    phiODF = tensor2odf_2D(phiVals, (A2, A4))

    fig = plt.figure(figsize=(2.25, 1.25))
    ax = fig.gca()
    plt.bar(phiBins, phiHist, width=0.9*dphi, align='edge', lw=0)
    plt.plot(phiVals, phiODF * dphi / phiRange, color=np.asarray([176, 21, 21])/255, lw=1)
    xticks = histDF['phiBins'][::3]
    ax.set_xlim([phiBins[0], phiBins[-1]])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    ax.set_xlabel("$\phi$ [degrees]")
    ax.set_ylabel("p($\phi$)")
    # plt.show()
    fig.savefig(path.join(dataDir, imgName+'_fit.tiff'), dpi=300)
