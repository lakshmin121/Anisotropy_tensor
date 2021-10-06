"""
Code to verify 3D isotropic distribution
"""

import numpy as np
from matplotlib_settings import *
from fiborient import orient_tensor_2D, tensor2odf_2D
from orientation_probabilities import map_rv2tan, map_rv2cos, map_rv2arctan, ratio_distr, ratio_distr2
from scipy.interpolate import interp1d, interp2d
from scipy.ndimage import gaussian_filter
from itertools import product


def fit_ODF(distr, bins):
    binsDeg = np.rad2deg(bins)
    h, b = np.histogram(np.rad2deg(distr), bins=binsDeg, density=True)
    Q, A = orient_tensor_2D(h, binsDeg)
    bincDeg = 0.5 * (binsDeg[:-1] + binsDeg[1:])
    odf = tensor2odf_2D(bincDeg, A) * 2
    return odf


bins = np.deg2rad(np.arange(0.5, 179.5, 5)) - np.pi/2
binc = 0.5 * (bins[1:] + bins[:-1])


# DISTRIBUTION
N = 1000
phiDistr = (np.random.uniform(0, 1, size=N) - 0.5) * np.pi
iud = np.random.uniform(0, 1, size=N)
thtDistr = np.arccos(1 - 2 * iud)
alpDistr = np.arctan(np.tan(thtDistr) * np.cos(phiDistr))

# # ODF
phiODF = fit_ODF(phiDistr, bins)
thtODF = fit_ODF(thtDistr, bins)
alpODF = fit_ODF(alpDistr, bins)


# # -----------
# # Estimation
# cos_phiBinc = np.cos(binc)
# cos_phiPMF = map_rv2cos(binc, phiODF, cos_phiBinc)
# cos_phiDistr = np.cos(phiDistr)
#
# figcos = plt.figure(figsize=(2, 2))
# ax = figcos.gca()
# h, b, _ = ax.hist(cos_phiDistr, bins=np.cos(bins)[::-1], density=True, alpha=0.5, label='$\cos{\phi}$')
# ax.plot(cos_phiBinc, cos_phiPMF, lw=0.75, label='PMF $\cos{\phi}$')
#
# # alpBinc = np.deg2rad(alphaBinspc - 89)
# tan_alpBinc = np.tan(binc - np.pi/2)
# tan_alpPMF = map_rv2tan(binc - np.pi/2, alpODF, tan_alpBinc)
# tan_alpPMF = tan_alpPMF / np.trapz(tan_alpPMF, tan_alpBinc)
# tan_alpDistr = np.tan(alpDistr - np.pi/2)
#
# figtan = plt.figure(figsize=(2, 2))
# ax = figtan.gca()
# h, b, _ = ax.hist(tan_alpDistr, bins=np.tan(bins - np.pi/2)[1:-1], density=True, alpha=0.5, label='$\\tan{\\alpha}$')
# ax.plot(tan_alpBinc[1:-1], tan_alpPMF[1:-1], lw=0.75, label='PMF $\\tan{\\alpha}$')

# # Ratio distribution
# thtBinc = binc
# tan_thtBinc = np.tan(thtBinc)
# tan_thtPMF = ratio_distr2(tan_alpBinc, cos_phiBinc[::-1], tan_thtBinc, tan_alpPMF, cos_phiPMF[::-1], atol=1e-16)
# tan_thtPMF = tan_thtPMF / np.trapz(tan_thtPMF, tan_thtBinc)
# tan_thtDistr = np.tan(thtDistr-np.pi/2)
# print("Total tan theta probability: ", np.trapz(tan_thtPMF, tan_thtBinc))
#
# figtantht = plt.figure(figsize=(2, 2))
# ax = figtantht.gca()
# h, b, _ = ax.hist(tan_thtDistr, bins=np.tan(bins-np.pi/2)[1:-1], density=True, alpha=0.5, label='$\\tan{\\theta}$')
# ax.plot(tan_alpBinc[1:-1], tan_thtPMF[1:-1], lw=0.75, label='PMF $\\tan{\\theta}$')
#
# thtPMF = map_rv2arctan(tan_thtBinc, tan_thtPMF, thtBinc)
# thtPMF = thtPMF / np.trapz(thtPMF, thtBinc)
# print("Total theta probability: ", np.trapz(thtPMF, thtBinc))
#
# figtht = plt.figure(figsize=(2, 2))
# ax = figtht.gca()
# h, b, _ = ax.hist(thtDistr, bins=bins, density=True, alpha=0.5, label='$\\tan{\\theta}$')
# ax.plot(thtBinc, thtPMF, lw=0.75, label='PMF $\\theta$')
# # # ------------

# # -------------------
# # Full Jacobian
# phiObs = binc
# alpVals = binc
#
# # u = np.arctan(np.tan(alpVals) / np.cos(phiObs))
# u = binc
# v = phiObs
# uv = product(u, v)
#
# def jacobian(i, j):
#     cV = np.cos(j)
#     tU = np.tan(i)
#     sU = 1 / np.cos(i)
#     # return (cV * sU**2 / (cV**2 + tU**2))
#     jacb = sU**2 * cV / (1 + (tU * cV)**2)
#     return jacb
#
# fxy = interp2d(alpVals, phiObs, np.outer(alpODF, phiODF), kind='linear', fill_value=0)
# fuv = np.array([fxy(np.arctan(np.tan(a) * np.cos(b)), b) * jacobian(a, b) for a, b in uv])
# fuv = fuv.reshape((len(u), len(v)))
# fu  = np.trapz(fuv, v)
#
# fig1 = plt.figure(figsize=(2, 2))
# ax1 = fig1.gca()
# ax1.plot(u, fu, lw=0.75, label='Estimated PMF')
# ax1.plot(binc, thtODF, lw=0.75, label='ODF $\\theta$')
# ax1.legend(loc='upper right')
#
# # -------------------

# # ----------------------
# # Fourier Approach
# alpVals = binc
# phiObs = binc
# n = len(alpVals)
# A = np.zeros((n, n), dtype=complex)
#
# for k in range(n):
#     a = np.array([np.trapz(phiODF * np.exp(-1j*k*np.arctan(np.tan(alp) / phiObs)), phiObs) for alp in alpVals], dtype=complex)
#     A[:, k] = a.T
#
# A = A / n
#
# c = np.linalg.pinv(A.T @ A) @ (A.T @ phiObs.astype(complex))
# print(c.shape)
#
# thtVals = binc
# F = np.zeros(A.shape)
# f = np.exp(-1j*thtVals)
# for k in range(n):
#     F[k, :] = f**k
#
# F = F / n
#
# thtPDF = np.abs(F @ c)
# print(thtPDF.shape)
#
# fig1 = plt.figure(figsize=(2, 2))
# ax1 = fig1.gca()
# ax1.plot(thtVals, thtPDF, lw=0.75, label='Estimated PMF')
# ax1.plot(binc, thtODF, lw=0.75, label='ODF $\\theta$')
# ax1.legend(loc='upper right')
# # ----------------------------

# ------------------
# New formulation


fig = plt.figure(figsize=(2, 2))
ax = fig.gca()
h, b, _ = ax.hist(alpDistr, bins=bins, density=True, alpha=0.5, label='$\\alpha$')
h, b, _ = ax.hist(thtDistr, bins=bins, density=True, alpha=0.5, label='$\\theta$')
h, b, _ = ax.hist(phiDistr, bins=bins, density=True, alpha=0.5, label='$\phi$')
# ax.plot(binc, alpODF, lw=0.75, linestyle='dotted', label='$p(\\alpha)$')
# ax.plot(binc, thtODF, lw=0.75, linestyle='dashed', label='$p(\\theta)$')
# ax.plot(binc, phiODF, lw=0.75, label='$p(\phi)$')
# ax.plot(binc, thtPMF, lw=0.75, label='PMF $\\theta$')

xticks = np.sort(np.pi * np.array([0, 1/6, 1/3, 1/2, 2/3, 5/6, 1]))
ax.set_xticks(xticks)
ax.set_xticklabels(['0', '$\\frac{\pi}{6}$', '$\\frac{\pi}{3}$', '$\\frac{\pi}{2}$',
                    '$\\frac{2\pi}{3}$' , '$\\frac{5\pi}{6}$', '$\pi$'])
yticks = np.round(np.arange(0, 0.61, 0.2), 1)
ax.set_yticks(yticks)
ax.set_yticklabels(yticks)
ax.set_ylim([0, 0.8])
ax.legend(loc='upper right', ncol=3, handlelength=1)
plt.show()
