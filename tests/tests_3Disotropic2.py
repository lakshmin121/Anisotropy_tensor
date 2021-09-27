"""
Code to verify 3D isotropic distribution
"""

import numpy as np
from matplotlib_settings import *
from fiborient import orient_tensor_2D, tensor2odf_2D
from orientation_probabilities import map_rv2tan, map_rv2cos, map_rv2arctan, ratio_distr, ratio_distr2, product_distr
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


phiBins = np.deg2rad(np.arange(0, 181, 5)) - np.pi/2
phiBinc = 0.5 * (phiBins[1:] + phiBins[:-1])
thtBins = phiBins #- np.pi/2
thtBinc = 0.5 * (thtBins[1:] + thtBins[:-1])


# DISTRIBUTION
N = 1000
phiDistr = (np.random.uniform(0, 1, size=N) - 0.5) * np.pi
iud = np.random.uniform(0, 1, size=N)
thtDistr = np.arcsin(1 - 2 * iud)
alpDistr = np.arctan(np.tan(thtDistr) / np.cos(phiDistr))

# # ODF
phiODF = fit_ODF(phiDistr, phiBins)
thtODF = fit_ODF(thtDistr, thtBins)
alpODF = fit_ODF(alpDistr, thtBins)


# # -----------
# # Estimation
# cos_phiBinc = np.cos(phiBinc)[::-1]
# print(cos_phiBinc)
# cos_phiPMF = map_rv2cos(phiBinc, phiODF, cos_phiBinc)
# cos_phiDistr = np.cos(phiDistr)
#
# figcos = plt.figure(figsize=(2, 2))
# ax = figcos.gca()
# h, b, _ = ax.hist(cos_phiDistr, bins=np.cos(phiBins)[::-1], density=True, alpha=0.5, label='$\cos{\phi}$')
# ax.plot(cos_phiBinc, cos_phiPMF, lw=0.75, label='PMF $\cos{\phi}$')
#
# tan_alpBinc = np.tan(thtBinc)
# tan_alpPMF = map_rv2tan(thtBinc, alpODF, tan_alpBinc)
# tan_alpPMF = tan_alpPMF / np.trapz(tan_alpPMF, tan_alpBinc)
# tan_alpDistr = np.tan(alpDistr)
#
# figtan = plt.figure(figsize=(2, 2))
# ax = figtan.gca()
# h, b, _ = ax.hist(tan_alpDistr, bins=np.tan(thtBins)[1:-1], density=True, alpha=0.5, label='$\\tan{\\alpha}$')
# ax.plot(tan_alpBinc, tan_alpPMF, lw=0.75, label='PMF $\\tan{\\alpha}$')
#
# # Product distribution
# tan_thtBinc = np.tan(thtBinc)
# tan_thtPMF = product_distr(tan_alpBinc, cos_phiBinc, tan_thtBinc, tan_alpPMF, cos_phiPMF, atol=1e-16)
# tan_thtPMF = tan_thtPMF / np.trapz(tan_thtPMF, tan_thtBinc)
# tan_thtDistr = np.tan(thtDistr)
# print("Total tan theta probability: ", np.trapz(tan_thtPMF, tan_thtBinc))
#
# figtantht = plt.figure(figsize=(2, 2))
# ax = figtantht.gca()
# h, b, _ = ax.hist(tan_thtDistr, bins=np.tan(thtBins)[1:-1], density=True, alpha=0.5, label='$\\tan{\\theta}$')
# ax.plot(tan_alpBinc[1:-1], tan_thtPMF[1:-1], lw=0.75, label='PMF $\\tan{\\theta}$')
#
# thtPMF = map_rv2arctan(tan_thtBinc, tan_thtPMF, thtBinc)
# thtPMF = thtPMF / np.trapz(thtPMF, thtBinc)
# print("Total theta probability: ", np.trapz(thtPMF, thtBinc))
#
# figtht = plt.figure(figsize=(2, 2))
# ax = figtht.gca()
# h, b, _ = ax.hist(thtDistr, bins=thtBins, density=True, alpha=0.5, label='$\\tan{\\theta}$')
# # ax.plot(thtBinc, thtPMF, lw=0.75, label='PMF $\\theta$')
# ax.scatter(thtBinc, thtPMF, s=4, label='PMF $\\theta$')
# # # ------------

# -------------------
# Full Jacobian
phiVals = phiBinc
alpVals = thtBinc

# u = np.arctan(np.tan(alpVals) / np.cos(phiObs))
u = thtBinc
v = phiVals
uv = product(u, v)

def jacobian(i, j):
    cV = np.cos(j)
    tU = np.tan(i)
    sU = 1 / np.cos(i)
    return (cV * sU**2 / (cV**2 + tU**2))

fxy = interp2d(alpVals, phiVals, np.outer(alpODF, phiODF), kind='linear', fill_value=0)
fuv = np.array([fxy(np.arctan(np.tan(a) / np.cos(b)), b) * jacobian(a, b) for a, b in uv])
fuv = fuv.reshape((len(u), len(v)))
fu  = np.trapz(fuv, v)

fig1 = plt.figure(figsize=(2, 2))
ax1 = fig1.gca()
ax1.plot(u, fu, lw=0.75, label='Estimated PMF')
ax1.plot(thtBinc, thtODF, lw=0.75, label='ODF $\\theta$')
ax1.legend(loc='upper right')

# -------------------

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

def sample_ODF(x, pdf, size):
    assert len(x) == len(pdf)
    if not np.isclose(np.trapz(pdf, x), 0):
        print("Total probability not close to unity.")
    cdf = np.cumsum(pdf)

    Finv = interp1d(cdf, x, kind='linear', fill_value='extrapolate')
    iud = np.random.uniform(0, 1, size=size)
    return Finv(iud)


fig = plt.figure(figsize=(2, 2))
ax = fig.gca()
h, b, _ = ax.hist(phiDistr, bins=phiBins, density=True, alpha=0.5, label='$\phi$')
h, b, _ = ax.hist(thtDistr, bins=thtBins, density=True, alpha=0.5, label='$\\theta$')
h, b, _ = ax.hist(alpDistr, bins=thtBins, density=True, alpha=0.5, label='$\\alpha$')
# ax.plot(binc, alpODF, lw=0.75, linestyle='dotted', label='$p(\\alpha)$')
# ax.plot(binc, thtODF, lw=0.75, linestyle='dashed', label='$p(\\theta)$')
# ax.plot(binc, phiODF, lw=0.75, label='$p(\phi)$')
# ax.plot(binc, thtPMF, lw=0.75, label='PMF $\\theta$')

# xticks = np.sort(np.pi * np.array([0, 1/6, 1/3, 1/2, 2/3, 5/6, 1]))
xticks = np.sort(np.pi * np.array([-1/2, -1/3, -1/6, 0, 1/6, 1/3, 1/2, 2/3, 5/6, 1]))
ax.set_xticks(xticks)
ax.set_xticklabels(['$-\\frac{\pi}{2}$', '$-\\frac{\pi}{3}$', '$-\\frac{\pi}{6}$','0',
                    '$\\frac{\pi}{6}$', '$\\frac{\pi}{3}$', '$\\frac{\pi}{2}$',
                    '$\\frac{2\pi}{3}$', '$\\frac{5\pi}{6}$', '$\pi$'])
yticks = np.round(np.arange(0, 0.61, 0.2), 1)
ax.set_yticks(yticks)
ax.set_yticklabels(yticks)
ax.set_ylim([0, 0.8])
ax.legend(loc='upper right', ncol=3, handlelength=1)
plt.show()
