"""
Code to test transformations of PDFs of RVs.
_________________________________________________________
@ Lakshminarayanna Mohana Kumar
1st Sep 2021
"""

import os
import numpy as np
from matplotlib_settings import *
from fiborient import orient_tensor_2D, tensor2odf_2D
from orientation_probabilities import map_rv2tan, map_rv2cos, map_rv2arctan, ratio_distr

# 3D isotropic distribution
outDir = "tests_RV_trans"

if not os.path.exists(outDir):
    os.mkdir(outDir)


def fit_ODF(distr, bins):
    h, b = np.histogram(distr, bins=bins, density=True)
    Q, A = orient_tensor_2D(h, bins)
    binc = 0.5 * (bins[:-1] + bins[1:])
    odf = tensor2odf_2D(binc, A) * 2
    return odf


# DISTRIBUTION
# 3D isotropic distribution
phiVals = np.deg2rad(np.arange(0.5, 180))
thtVals = phiVals - np.pi/2
alpVals = thtVals

phiBins = np.deg2rad(np.arange(0, 181, 10))
phiBinc = 0.5 * (phiBins[1:] + phiBins[:-1])
thtBins = np.deg2rad(np.arange(-90, 91, 10))
thtBinc = 0.5 * (thtBins[1:] + thtBins[:-1])
alpBins = thtBins
alpBinc = thtBinc

N = 1000
phiDistr = np.random.uniform(0, np.pi, size=N)
u = np.random.uniform(0, 1, size=N)
thtDistr = np.arccos(1 - 2 * u) - np.pi/2
alpDistr = np.arctan(np.tan(thtDistr) * np.cos(phiDistr))

# ODF
phiODF = fit_ODF(np.rad2deg(phiDistr), np.rad2deg(phiBins))
thtODF = fit_ODF(np.rad2deg(thtDistr), np.rad2deg(thtBins))
alpODF = fit_ODF(np.rad2deg(alpDistr), np.rad2deg(alpBins))

fig_phiDistr = plt.figure(figsize=(2, 2))
ax = fig_phiDistr.gca()
h, b, _ = ax.hist(phiDistr, bins=phiBins, density=True)
ax.plot(phiBinc, phiODF, lw=0.75)
ax.set_xlabel("$\phi$")
ax.set_ylabel("$p(\phi)$")

fig_thtDistr = plt.figure(figsize=(2, 2))
ax = fig_thtDistr.gca()
h, b, _ = ax.hist(thtDistr, bins=thtBins, density=True)
ax.plot(thtBinc, thtODF, lw=0.75)
ax.set_xlabel("$\\theta$")
ax.set_ylabel("$p(\\theta)$")

fig_alpDistr = plt.figure(figsize=(2, 2))
ax = fig_alpDistr.gca()
h, b, _ = ax.hist(alpDistr, bins=alpBins, density=True)
ax.plot(alpBinc, alpODF, lw=0.75)
ax.set_xlabel("$\\alpha$")
ax.set_ylabel("$p(\\alpha)$")

fig_phiDistr.savefig(os.path.join(outDir, 'phiDistr.tiff'))
fig_thtDistr.savefig(os.path.join(outDir, 'thtDistr.tiff'))
fig_alpDistr.savefig(os.path.join(outDir, 'alpDistr.tiff'))


# Estimating Distribution of Theta from Phi and Alpha
cos_phiBinc = np.cos(phiBinc)
cos_phiPMF = map_rv2cos(phiBinc, phiODF, cos_phiBinc)
cos_phiDistr = np.cos(phiDistr)

tan_alpBinc = np.tan(alpBinc)
tan_alpPMF = map_rv2tan(alpBinc, alpODF, tan_alpBinc)
tan_alpDistr = np.tan(alpDistr)

fig_cos_phiDistr = plt.figure(figsize=(2, 2))
ax = fig_cos_phiDistr.gca()
h, b, _ = ax.hist(cos_phiDistr, bins=cos_phiBinc[::-1], density=True)
ax.plot(cos_phiBinc, cos_phiPMF, lw=0.75)
ax.set_xlabel("$\cos{\phi}$")
ax.set_ylabel("$p(\cos{\phi})$")

fig_tan_alpDistr = plt.figure(figsize=(2, 2))
ax = fig_tan_alpDistr.gca()
h, b, _ = ax.hist(tan_alpDistr, bins=tan_alpBinc, density=True)
ax.plot(tan_alpBinc, tan_alpPMF, lw=0.75)
ax.set_xlabel("$\\tan{\\alpha}$")
ax.set_ylabel("$p(\\tan{\\alpha})$")

fig_cos_phiDistr.savefig(os.path.join(outDir, 'cos_phiDistr.tiff'))
fig_tan_alpDistr.savefig(os.path.join(outDir, 'tan_alpDistr.tiff'))

# Ratio distribution
tan_thtBinc = np.tan(thtBinc)
tan_thtPMF  = ratio_distr(tan_alpBinc, cos_phiBinc[::-1], tan_alpBinc, tan_alpPMF, cos_phiPMF[::-1])
tan_thtDistr = np.tan(thtDistr)

fig_tan_thtDistr = plt.figure(figsize=(2, 2))
ax = fig_tan_thtDistr.gca()
h, b, _ = ax.hist(tan_thtDistr, bins=tan_thtBinc, density=True)
ax.plot(tan_thtBinc, tan_thtPMF, lw=0.75)
ax.set_xlabel("$\\tan{\\theta}$")
ax.set_ylabel("$p(\\tan{\\theta})$")

fig_tan_thtDistr.savefig(os.path.join(outDir, 'tan_thtDistr.tiff'))

# Theta
thtPMF = map_rv2arctan(tan_thtBinc, tan_thtPMF, thtBinc)

fig_tht2Distr = plt.figure(figsize=(2, 2))
ax = fig_tht2Distr.gca()
h, b, _ = ax.hist(thtDistr, bins=thtBins, density=True)
ax.plot(thtBinc, thtPMF, lw=0.75)
ax.set_xlabel("$\\theta$")
ax.set_ylabel("$p(\\theta)$")

fig_tht2Distr.savefig(os.path.join(outDir, 'tht2Distr.tiff'))
