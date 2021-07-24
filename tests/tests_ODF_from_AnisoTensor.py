"""
Testing the accuracy of ODF determined from anisotropy tensor.
______________________________________________
@ Lakshminarayanan Mohana Kumar
23rd July 2021
"""

import os
import sys
import numpy as np
sys.path.append("../artificial images/")
import orientation_probabilities as op
from fiborient import tensor2odf_2D, orient_tensor_2D, orientation_tensor_4order
from matplotlib import pyplot as plt
from matplotlib_settings import *


# SET-UP
Nf = 500  # number of fibres = number of phi values
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
            phi_vals = op.uniform(phidomainDeg=phidomainDeg, size=size)
        except KeyError:
            print("using default domain (0, 180) degrees for uniform distribution.")
            phi_vals = op.uniform(size=size)
    else:
        raise ValueError("Invalid odf: {}".format(odf))
    return phi_vals


m = 0
for k in [0.1]: #, 0.25, 0.5, 1, 5]:
    imgName = 'vm_m{0}k{1}'.format(m, k)
    img_fname = imgName + '.tiff'

    odf = 'vonmises'  # Orietation Distribution Function
    parameters = {'muDeg': m,
                  'kappa': k,
                  'spreadDeg': 180}
    phiDomainDeg = (-90, 90)

    # Fibre Orientation
    phivals = generate_fibre_orientations(odf=odf, size=Nf, **parameters)
    phiMin, phiMax = (round(np.min(phivals), 1), round(np.max(phivals), 1))

    phiBins = np.arange(phiDomainDeg[0], phiDomainDeg[1]+1, 5)
    phiHist, bins = np.histogram(phivals, bins=phiBins, normed=True)
    dphi = np.mean(phiBins[1:] - phiBins[:-1])
    phiBinc = 0.5 * (phiBins[1:] + phiBins[:-1])
    phiRange = phiBins[-1] - phiBins[0]

    Q2, A2 = orient_tensor_2D(phiHist, phiBinc)
    print(A2)
    Q4, A4 = orientation_tensor_4order(phiHist, phiBinc)
    print(A4)
    # phiODF2 = tensor2odf_2D(phiBinc, A2)
    # phiODF4 = tensor2odf_2D(phiBinc, (A2, A4))
    # C = np.sum(phiODF2) * (dphi * np.pi / 180)
    # print(np.min(phiODF2), np.max(phiODF2))
    # print(C)
    #
    # fig = plt.figure(figsize=(2.25, 1.25))
    # ax = fig.gca()
    # plt.bar(phiBins, np.append(phiHist, np.nan), width=0.9 * dphi, align='edge', lw=0)
    # plt.plot(phiBinc, phiODF2 * dphi / phiRange, color=np.asarray([176, 21, 21]) / 255, lw=1, linestyle='--')
    # plt.plot(phiBinc, phiODF4 * dphi / phiRange, color=np.asarray([80, 21, 21]) / 255, lw=1)
    # xticks = phiBins[::3]
    # ax.set_xlim([phiBins[0], phiBins[-1]])
    # ax.set_xticks(xticks)
    # ax.set_xticklabels(xticks)
    # ax.set_xlabel("$\phi$ [degrees]")
    # ax.set_ylabel("p($\phi$)")
    # # plt.show()
    # fig.savefig(os.path.join(outDir, imgName + '_fit.tiff'), dpi=300)
