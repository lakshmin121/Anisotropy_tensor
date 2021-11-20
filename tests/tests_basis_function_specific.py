"""
Code to test the concept of generating 2D projectional-plane basis functions from 3D basis functions through integration.
Also check the multiplication with anisotropy tensor. Rigorous testing

utility functions to perform tests.
_______________________________________________________________________________________
@ Lakshminarayanan Mohana Kumar
8 Nov 2021
"""

import os
import sys
import numpy as np
sys.path.append('..')
# from glob import glob
from itertools import product
import pandas as pd
from odffit import basisfunc_3D, basisfunc_proj, projdir_rotation_3D
from matplotlib_settings import *
from scipy.stats import mode
from fiborient import orient_tensor_3D
from scipy.interpolate import interp1d


SECONDORDER_SIZE = 3

# Input parameters: muPhiRad, kappaPhi, muThtRad, kappaTht


def plot_hist(angSamples, angDomain, nbins):
    bins = np.linspace(angDomain[0],angDomain[1], nbins+1, endpoint=True)
    fig = plt.figure(figsize=(3.6, 2.5), dpi=600)
    ax = fig.gca()
    hist, b, _ = ax.hist(angSamples, bins=bins, density=True)
    xticks = np.arange(angDomain[0], angDomain[1] + np.pi / 12, np.pi / 6)
    ax.set_xticks(xticks)
    # yticks = np.round(np.arange(0, 1 / np.pi, 1) * (len(bins)-1), 2) * np.pi / (len(bins)-1)
    # ax.set_yticks(yticks)
    ax.set_xticklabels(ANGLE_TICK_LABEL_DICT[np.sign(np.min(xticks[xticks!=0])) * len(xticks)])

    return fig, hist, b


def joint_probability(thetaVals, phiVals, nbins=180,
                      thetaDomainRad=(-np.pi/2, np.pi/2), phiDomainRad=(-np.pi/2, np.pi/2)):
    global ANGLE_TICK_LABEL_DICT
    # Joint Probability: p(theta, phi).
    fig = plt.figure(figsize=(3.5, 3), dpi=600)  # see a plot of the joint distribution
    ax = fig.gca()
    phimin, phimax = phiDomainRad
    phibins = np.linspace(phimin, phimax, nbins+1, endpoint=True)
    thtmin, thtmax = thetaDomainRad
    thtbins = np.linspace(thtmin, thtmax, nbins + 1, endpoint=True)
    jointProb, xEdges, yEdges, histImg = ax.hist2d(phiVals, thetaVals,
                                                   bins=(phibins, thtbins),
                                                   density=True)
    ax.set_aspect('equal')
    ax.set_ylabel("$\\theta$")
    ax.set_xlabel("$\phi$")
    # xticks = np.arange(np.min(xEdges), np.max(xEdges)+np.pi/12, np.pi/6)
    # yticks = np.arange(np.min(yEdges), np.max(yEdges)+np.pi/12, np.pi/6)
    xticks = np.arange(phimin, phimax + np.pi / 12, np.pi / 6)
    yticks = np.arange(thtmin, thtmax + np.pi / 12, np.pi / 6)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ANGLE_TICK_LABEL_DICT[np.sign(np.min(yticks[yticks!=0])) * len(yticks)])
    ax.set_xticklabels(ANGLE_TICK_LABEL_DICT[np.sign(np.min(xticks[xticks!=0])) * len(xticks)])
    cbar = plt.colorbar(histImg, orientation='horizontal')
    cbar.ax.tick_params(labelsize=9)

    return jointProb.T, xEdges, yEdges, fig


def alpha(thetaRad, phiRad, projdir):
    """Estimate the orientation of individual fibres, alpha, on the projection plane from their global orientations
    and rotation matrix corresponding to projection direction."""
    assert len(thetaRad) == len(phiRad)
    cosphi, sinphi = np.cos(phiRad), np.sin(phiRad)
    costht, sintht = np.cos(thetaRad), np.sin(thetaRad)

    uvecs = np.zeros((3, len(thetaRad)))  # Global fibre directions
    uvecs[0, :] = sintht * cosphi
    uvecs[1, :] = sintht * sinphi
    uvecs[2, :] = costht

    R = projdir_rotation_3D(projdir[0], projdir[1])
    print("\nR: \n", R)

    uvecp = np.einsum('ij, jk -> ik', R, uvecs)  # transforming to fib dir on proj plane
    alphaRad = np.arctan2(uvecp[1, :], uvecp[0, :])
    return alphaRad


def lsfit_score(hist, curve):
    assert len(hist.shape) == 1
    assert len(curve.shape) == 1
    lh, lc = len(hist), len(curve)
    if lh > lc:
        vec1 = curve
        x1 = np.arange(len(vec1)) * lh / lc
        x2 = np.arange(len(hist))
        f2 = interp1d(x2, hist)
        ls_score = np.linalg.norm(vec1 - f2(x1))
    elif lh < lc:
        vec1 = hist
        x1 = np.arange(len(vec1))  * lc / lh
        x2 = np.arange(len(curve))
        f2 = interp1d(x2, curve)
        ls_score = np.linalg.norm(vec1 - f2(x1))
    else:
        ls_score = np.linalg.norm(hist - curve)

    return ls_score / np.min((lh, lc))


def test_basis_function(phiSamples, thtSamples, nbins=36,
                        projdir=(0, 0), upsDomainRad=(0, np.pi), psiDomainRad=(0, np.pi),
                        nUps=180, nPsi=180, order=2, refdirRad=(0, 0)
                        ):
    # TODO: write all possible tests for basis function, projection, anisotropy tensor,
    # and ODF generates using their products.
    nPoints = 180
    if np.min(phiSamples) < 0 or np.all(np.isclose(phiSamples, phiSamples[0])):
        phiSamplespos = phiSamples + np.pi/2  # values between [0, pi)
    else:
        phiSamplespos = phiSamples
    if np.min(thtSamples) < 0 or np.all(np.isclose(thtSamples, thtSamples[0])):
        thtSamplespos = thtSamples + np.pi/2  # values between [0, pi)
    else:
        thtSamplespos = thtSamples

    phiFig, phiHist, phiBins = plot_hist(phiSamples, (-np.pi/2, np.pi/2), nbins)
    thtFig, thtHist, thtBins = plot_hist(thtSamples, (-np.pi/2, np.pi/2), nbins)

    phiMode, _ = mode(np.digitize(phiSamples, bins=phiBins))
    thtMode, _ = mode(np.digitize(thtSamples, bins=thtBins))

    print("Phi: min={0}\t max={1}\t mean={2}".format(phiSamples.min(), phiSamples.max(), phiBins[phiMode]))
    print("Theta: min={0}\t max={1}\t mean={2}".format(thtSamples.min(), thtSamples.max(), thtBins[thtMode]))

    # Generating corresponding distribution on the projected plane
    # alphSamples = np.arctan2(np.tan(thtSamples), np.cos(phiSamples))
    alphSamples = alpha(thtSamplespos, phiSamplespos, projdir) - np.pi/2
    print("alpha: min: {0}\t max: {1}".format(np.min(alphSamples), np.max(alphSamples)))
    alphVals = np.linspace(-np.pi/2, np.pi/2, nPoints, endpoint=True)
    alphFig, alphHist, alphBins  = plot_hist(alphSamples, (-np.pi/2, np.pi/2), nbins)


    # 3D DISTRIBUTION
    jointProb, phiEdges, thtEdges, jointProbFig = joint_probability(thtSamplespos, phiSamplespos, nbins=nbins,
                                                                    thetaDomainRad=(0, np.pi), phiDomainRad=(0, np.pi))
    # check jointprob
    print("Total joint prob: ",  np.trapz(np.trapz(jointProb, phiEdges[:-1]), thtEdges[:-1]))
    Q2, A2 = orient_tensor_3D(jointProb, thtEdges, phiEdges)
    Q4, A4 = orient_tensor_3D(jointProb, thtEdges, phiEdges, order=4)
    # print("Q2:\n", Q2)

    # Direction of 3D orientation bias
    eigvals, eigvecs = np.linalg.eig(Q2 / np.trace(Q2))
    indcs = np.argsort(eigvals)
    eigvals = eigvals[indcs]
    eigvecs = eigvecs[:, indcs]
    biasvec = eigvecs[:, -1]
    biasvec = np.sign(biasvec[0]) * biasvec / np.linalg.norm(biasvec)
    biasTht = np.arctan2(np.sign(biasvec[0])*np.sign(biasvec[1])*np.sqrt(biasvec[0]**2 + biasvec[1]**2) / biasvec[-1], 1)
    biasPhi = np.arctan2(biasvec[1] / biasvec[0], 1)
    if biasPhi < 0:
        biasPhi = np.pi + biasPhi
    if biasTht < 0:
        biasTht = np.pi + biasTht
    print("Eigenvals: ", eigvals)
    print("Eigenvecs: \n", eigvecs)
    print("Biasvec: \n", biasvec)
    print("Bias (phi, theta): ", (biasPhi, biasTht))
    print("Bias (phi, theta): ", (biasPhi + np.pi/2, biasTht + np.pi/2))


    # PROJECTION
    Fproj2nd = basisfunc_proj(projdir=projdir, upsDomainRad=upsDomainRad, nPsi=nPsi, nUps=nUps,
                              psiDomainRad=psiDomainRad, refdirRad=refdirRad)
    Fproj4th = basisfunc_proj(projdir=projdir, upsDomainRad=upsDomainRad, nPsi=nPsi, nUps=nUps,
                              psiDomainRad=psiDomainRad, refdirRad=refdirRad, order=4)

    alphProb2 = 1/(4*np.pi) * (2 + (15/2) * np.einsum('ij, ijk -> k', A2, Fproj2nd)) * 2
    alphProb4 = alphProb2 + 1/(4*np.pi) * (315/8) * np.einsum('ij, ijk -> k', A4, Fproj4th) * 2

    # Checks
    totalProb2 = np.trapz(alphProb2, alphVals)  # TODO: Verify division by 2
    totalProb4 = np.trapz(alphProb4, alphVals)
    print("\nTotal probabilities:")
    print("\t2nd order: ", totalProb2)
    print("\t4th order: ", totalProb4)

    # alphProb2 = alphProb2 / totalProb2
    # alphProb4 = alphProb4 / totalProb4

    # ls2 = lsfit_score(alphHist, alphProb2)
    # ls2rev = lsfit_score(alphHist, np.flip(alphProb2))
    # print(ls2, ls2rev)
    # if ls2rev < ls2:
    #     alphProb2 = np.flip(alphProb2)
    #
    # ls4 = lsfit_score(alphHist, alphProb4)
    # ls4rev = lsfit_score(alphHist, np.flip(alphProb4))
    # print(ls4, ls4rev)
    # if ls4rev < ls4:
    #     alphProb4 = np.flip(alphProb4)

    ax = alphFig.gca()
    ax.plot(alphVals, alphProb2 * nbins / np.pi, label='$2^{nd}$ order')
    ax.plot(alphVals, alphProb4 * nbins / np.pi, ls='dashed', label='$4^{th}$ order')
    # ax.legend(loc='upper right')

    return jointProbFig, thtFig, phiFig, alphFig, biasPhi, biasTht
