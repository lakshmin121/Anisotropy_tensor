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
import orientation_probabilities as op
import skimage.io as skio
from fiborient import tensor2odf_2D, orient_tensor_2D
from fibfourier import fourier_orient_tensor
from matplotlib import pyplot as plt
from matplotlib_settings import *


# SET-UP
Nf = 500  # number of fibres = number of phi values
dataDir = "../data/test_images_2D/vf20_ar50_tk50"
outDir = "tests_odf_from_anisotensor"
if not os.path.exists(outDir):
    os.mkdir(outDir)


def generate_fibre_orientations(odf='vonmises', size=1, **kwargs):
    if odf == 'vonmises':
        try:
            kappa = kwargs['kappa']
        except KeyError as e:
            raise e("Require value of kappa.")
        phi_vals = op.vonmises(size=size, **kwargs)
    elif odf == 'uniform':
        try:
            phidomainDeg = kwargs['phidomainDeg']
            phi_vals = op.uniform(domainDeg=phidomainDeg, size=size)
        except KeyError:
            print("using default domain (0, 180) degrees for uniform distribution.")
            phi_vals = op.uniform(size=size)
    else:
        raise ValueError("Invalid odf: {}".format(odf))
    return phi_vals


m = 0
for k in [0.1, 0.25, 0.5, 1, 5]:
    imgName = 'vm_m{0}k{1}'.format(m, k)
    img_fname = imgName + '.tiff'
    prob_fname = imgName + '_prob.csv'

    odf = 'vonmises'  # Orietation Distribution Function
    parameters = {'muDeg': m,
                  'kappa': k,
                  'spreadDeg': 180}
    phiDomainDeg = (-90, 90)

    # Read image
    img = skio.imread(os.path.join(dataDir, img_fname), as_gray=True)

    # Fibre Orientation
    # phiDF = pd.read_csv(os.path.join(dataDir, prob_fname))
    phivals = generate_fibre_orientations(odf=odf, size=Nf, **parameters)
    # phiBins = phiDF['phiBins'].to_numpy()
    phiMin, phiMax = (round(np.min(phivals), 1), round(np.max(phivals), 1))

    phiBins = np.arange(phiDomainDeg[0], phiDomainDeg[1]+1, 5)
    phiHist, bins = np.histogram(phivals, bins=phiBins, density=True)
    dphi = np.mean(phiBins[1:] - phiBins[:-1])
    phiBinc = 0.5 * (phiBins[1:] + phiBins[:-1])
    phiRange = phiBins[-1] - phiBins[0]
    print("Check: ", np.sum(phiHist) * dphi)

    # Theoretical tensor and ODF from histogram
    Q2_theo, A2_theo = orient_tensor_2D(phiHist, phiBinc)
    Q4_theo, A4_theo = orient_tensor_2D(phiHist, phiBinc, order=4)
    phiODF2_theo = tensor2odf_2D(phiBinc, A2_theo)
    phiODF4_theo = tensor2odf_2D(phiBinc, (A2_theo, A4_theo))

    # Tensor and ODF from image
    # Tensor and ODF from image
    # Q2, A2 = fourier_orient_tensor_2order(img, windowName='hann')
    Q2, A2 = fourier_orient_tensor(img, windowName='hann')
    Q4, A4 = fourier_orient_tensor(img,  windowName='hann', order=4)
    A2 = A2 * 2
    A4 = A4 * 2
    phiODF2 = tensor2odf_2D(phiBinc, A2)
    phiODF4 = tensor2odf_2D(phiBinc, (A2, A4))
    print("Check2: ", np.sum(phiODF2) * dphi)
    print("Check2: ", np.sum(phiODF4) * dphi)

    print("\n\nA2: \n", A2)
    print("\n\nA2_theo: \n", A2_theo)

    # Error estimates
    err2 = np.linalg.norm(A2 - A2_theo)
    err4 = np.linalg.norm(A4 - A4_theo)
    print("error 2nd order: ", err2)
    print("error 4th order: ", err4)

    fig = plt.figure(figsize=(2.25, 1.25))
    ax = fig.gca()
    plt.bar(phiBins, np.append(phiHist, np.nan), width=0.9 * dphi, align='edge', lw=0)
    plt.plot(phiBinc, phiODF2_theo * dphi / phiRange, color=np.asarray([176, 21, 21]) / 255, lw=1, linestyle='--')
    plt.plot(phiBinc, phiODF4_theo * dphi / phiRange, color=np.asarray([176, 21, 21]) / 255, lw=1)
    plt.plot(phiBinc, phiODF2 * dphi / phiRange, color=np.asarray([230, 100, 20]) / 255, lw=1, linestyle='--')
    plt.plot(phiBinc, phiODF4 * dphi / phiRange, color=np.asarray([230, 100, 20]) / 255, lw=1)

    xticks = phiBins[::3]
    ax.set_xlim([phiBins[0], phiBins[-1]])
    ax.set_xticks(xticks)
    plt.xticks(rotation=90)
    plt.yticks(rotation=90)
    ax.set_xticklabels(xticks)
    ax.set_xlabel("$\phi$ [degrees]")
    ax.set_ylabel("p($\phi$)")

    fig.savefig(os.path.join(outDir, imgName + '_rand_fit.tiff'), dpi=300)
