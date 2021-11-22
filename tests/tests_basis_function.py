"""
Code to test the concept of generating 2D projectional-plane basis functions from 3D basis functions through integration.
Also check the multiplication with anisotropy tensor.
_______________________________________________________________________________________
@ Lakshminarayanan Mohana Kumar
10 Oct 2021
"""

import os
import sys
import numpy as np
sys.path.append('..')
# from glob import glob
# from itertools import product
import pandas as pd
# from odffit import basisfunc_3D, basisfunc_proj
# from matplotlib_settings import *
# from fiborient import orient_tensor_3D
from tests_basis_function_specific import test_basis_function

orig_stdout = sys.stdout

# # ---------------------------------------------------------------------------
# # TEST TYPE - 2
#
# teststring = "TEST-Type-2: Planar isotropic distribution of fibres,\n" +\
# "\t p(theta, phi) = delta(theta - theta_0) / pi \n" +\
# "Then, by projecting to XY-plane (phi-plane), \n" +\
# "        p(phi) = 1/pi \n " +\
# "Several tests to be performed by varying: theta_0."
#
# outDir = os.path.join("tests_basis_function", "Type-2")
# if not os.path.exists(outDir):
#         os.mkdir(outDir)
#
# f = open(os.path.join(outDir, 'test_summary.txt'), 'w+')
# sys.stdout = f
# print(teststring)
#
# # Generating data:
# Ntests = 5
# Nsamples = 1000
# nPoints = 180
#
# # phi follows uniform distribution
# phiSamples = np.random.uniform(-np.pi/2, np.pi/2, size=Nsamples)  # values between [-pi/2, pi/2)
# # theta is same for all fibres
# thtSamples = np.ones(Nsamples)
# thtChoices = np.random.uniform(-np.pi/3, np.pi/3, size=Ntests)
#
# # Projection settings
# refdirRad = (0, 0)
# upsDomainRad = (0, np.pi)
# psiDomainRad = (0, np.pi)
# projdir = (0, 0)  # using value passed to this function
#
# rmserrs2 = []
# rmserrs4 = []
#
# for testno in range(Ntests):
#         fpath = os.path.join(outDir, 'test-' + str(testno+1))
#         theta0 = thtChoices[testno]
#         print("\n\nTest - ", testno)
#         print("theta0 = ", theta0)  # value of theta
#
#         result = test_basis_function(phiSamples, thtSamples * theta0, nbins=36,
#                                      projdir=projdir, upsDomainRad=upsDomainRad, psiDomainRad=psiDomainRad,
#                                      nUps=nPoints, nPsi=nPoints, refdirRad=refdirRad
#                                      )
#         jointProbFig, thtFig, phiFig, alphFig, biasPhi, biasTht, rmserr2, rmserr4 = result
#         rmserrs2.append(rmserr2)
#         rmserrs4.append(rmserr4)
#
#         # print all stats to the figure as text for verification.
#         info = "$\phi ~ U(-\pi/2, \pi/2)$"\
#                 + "\n$\\theta = " + str(theta0)
#
#         jointProbFig.text(0.8, 0.5, info, fontsize=9)
#         jointProbFig.savefig(fpath+"_jointHist.tiff")
#
#         phiFig.savefig(fpath + "_phiHist.tiff")
#         thtFig.savefig(fpath + "_thtHist.tiff")
#         alphFig.savefig(fpath + "_alphHist.tiff")
#
# columns = ['theta0', 'rms err 2nd', 'rms err 4th']
# summaryDF = pd.DataFrame(data=np.array([thtChoices.data, rmserrs2, rmserrs4]).T, columns=columns)
# summaryDF.to_csv(os.path.join(outDir, "summary.csv"), index=False)

# f.close()
# sys.stdout = orig_stdout


# ---------------------------------------------------------------------------
# # TEST TYPE - 1
#
# teststring = "TEST-Type-1: All fibres in same direction,\n" +\
# "\t p(theta, phi) = delta(theta - theta_0) delta(phi - phi_0) \n" +\
# "Then, by projecting to XY-plane (phi-plane), \n" +\
# "        p(phi) = delta(phi - phi_0) \n " +\
# "Several tests to be performed by varying: theta_0, phi_0"
#
# outDir = os.path.join("tests_basis_function", "Type-1")
# if not os.path.exists(outDir):
#         os.mkdir(outDir)
#
# f = open(os.path.join(outDir, 'test_summary.txt'), 'w+')
# sys.stdout = f
# print(teststring)
#
# # Generating data:
# Ntests = 5
# Nsamples = 1
# nPoints = 180
#
# # phi follows uniform distribution
# # phiSamples = np.random.uniform(-np.pi/2, np.pi/2, size=Nsamples)  # values between [-pi/2, pi/2)
# # theta is same for all fibres
# thtSamples = np.ones(Nsamples)
# phiChoices = np.random.uniform(-np.pi/3, np.pi/3, size=Ntests)
# thtChoices = np.random.uniform(-np.pi/3, np.pi/3, size=Ntests)
#
# # Projection settings
# refdirRad = (0, 0)
# upsDomainRad = (0, np.pi)
# psiDomainRad = (0, np.pi)
# projdir = (0, 0)  # using value passed to this function
#
# rmserrs2 = []
# rmserrs4 = []
# for testno in range(Ntests):
#         fpath = os.path.join(outDir, 'test-' + str(testno+1))
#         phi0 = phiChoices[testno]
#         theta0 = thtChoices[testno]
#         print("\n\nTest - ", testno)
#         print("theta0 = ", theta0)  # value of theta
#         print("phi0 = ", phi0)  # value of phi
#
#         result = test_basis_function(thtSamples * phi0, thtSamples * theta0, nbins=36,
#                                      projdir=projdir, upsDomainRad=upsDomainRad, psiDomainRad=psiDomainRad,
#                                      nUps=nPoints, nPsi=nPoints, refdirRad=refdirRad
#                                      )
#         jointProbFig, thtFig, phiFig, alphFig, biasPhi, biasTht, rmserr2, rmserr4 = result
#         rmserrs2.append(rmserr2)
#         rmserrs4.append(rmserr4)
#
#         # print all stats to the figure as text for verification.
#         info = "$\phi \~ U(-\pi/2, \pi/2)$"\
#                 + "\n$\\theta$ = " + str(np.round(np.pi/2+theta0, 3))
#
#         jointProbFig.text(0.8, 0.5, info, fontsize=9)
#         jointProbFig.savefig(fpath+"_jointHist.tiff")
#
#         ax = alphFig.gca()
#         if phi0 < 0:
#                 ax.legend(loc='upper right')
#         else:
#                 ax.legend(loc='upper left')
#
#         phiFig.savefig(fpath + "_phiHist.tiff")
#         thtFig.savefig(fpath + "_thtHist.tiff")
#         alphFig.savefig(fpath + "_alphHist.tiff")
#
# columns = ['theta0', 'phi0', 'rms err 2nd', 'rms err 4th']
# summaryDF = pd.DataFrame(data=np.array([thtChoices.data, phiChoices.data, rmserrs2, rmserrs4]).T, columns=columns)
# summaryDF.to_csv(os.path.join(outDir, "summary.csv"), index=False)

# f.close()
# sys.stdout = orig_stdout


# ---------------------------------------------------------------------------
# TEST TYPE - 3

teststring = "TEST-Type-3: vonMises distribution for phi and theta,\n" +\
"\t p(theta, phi) = p(theta) p(phi) \n" +\
"Several tests to be performed by varying: mu_theta, mu_phi, kappa_theta, kappa_phi"

outDir = os.path.join("tests_basis_function", "Type-3")
if not os.path.exists(outDir):
        os.mkdir(outDir)

f = open(os.path.join(outDir, 'test_summary.txt'), 'w+')
sys.stdout = f
print(teststring)

# Generating data:
Ntests = 6
Nsamples = 500
nPoints = 180
nbins = 36

mu_thtChoices = [-np.pi/6, -np.pi/3, np.pi/3, np.pi/4, -np.pi/3, -np.pi/6]
mu_phiChoices = [np.pi/6, -np.pi/3, -np.pi/3, -np.pi/6, np.pi/3, np.pi/4]
k_thtChoices  = [2, 2, 2, 2, 2, 2]  # kappa >= 1
k_phiChoices  = [2, 2, 2, 2, 2, 2]  # kappa >= 1

# Projection settings
refdirRad = (0, 0)
upsDomainRad = (0, np.pi)
psiDomainRad = (0, np.pi)

rmserrs2phi = []
rmserrs4phi = []
rmserrs2alph = []
rmserrs4alph = []
for testno in range(Ntests):
        fpath = os.path.join(outDir, 'test-' + str(testno + 1))
        muPhiRad = mu_phiChoices[testno]
        muThtRad = mu_thtChoices[testno]
        kappaPhi = k_phiChoices[testno]
        kappaTht = k_thtChoices[testno]
        print("\n\nTest - ", testno)
        print("mu_phi: ", muPhiRad)  # central value of phi
        print("kappa_phi: ", kappaPhi)  # spread of phi values
        print("mu_theta: ", muThtRad)  # central value of phi
        print("kappa_theta: ", kappaTht)  # spread of theta values

        # GENERATING DATA using vonMises distribution  # Checked
        phiSamples = 0.5 * np.random.vonmises(2 * muPhiRad, kappaPhi, size=Nsamples)  # values between [-pi/2, pi/2)
        thtSamples = 0.5 * np.random.vonmises(2 * muThtRad, kappaTht, size=Nsamples)  # values between [-pi/2, pi/2)
        print("Phi: min: {0}\t max: {1}\t mean: {2}".format(phiSamples.min(), phiSamples.max(), phiSamples.mean()))
        print("Theta: min: {0}\t max: {1}\t mean: {2}".format(thtSamples.min(), thtSamples.max(), thtSamples.mean()))

        # projdir = (0, 0)
        # result = test_basis_function(thtSamples, phiSamples, nbins=36,
        #                              projdir=projdir, upsDomainRad=upsDomainRad, psiDomainRad=psiDomainRad,
        #                              nUps=nPoints, nPsi=nPoints, refdirRad=refdirRad
        #                              )
        # jointProbFig, thtFig, phiFig, _, biasPhi, biasTht, rmserr2phi, rmserr4phi = result
        # rmserrs2phi.append(rmserr2phi)
        # rmserrs4phi.append(rmserr4phi)

        projdir = (0, np.pi/2)
        result = test_basis_function(phiSamples, thtSamples, nbins=nbins,
                                     projdir=projdir, upsDomainRad=upsDomainRad, psiDomainRad=psiDomainRad,
                                     nUps=nPoints, nPsi=nPoints, refdirRad=refdirRad
                                     )
        jointProbFig, thtFig, phiFig, alphFig, biasPhi, biasTht, rmserr2alph, rmserr4alph = result
        rmserrs2alph.append(rmserr2alph)
        rmserrs4alph.append(rmserr4alph)

        # print all stats to the figure as text for verification.
        info = "$\mu_{\phi}$="+ str(muPhiRad) + " rad"\
                + "\n$\kappa_{\phi}$=" + str(kappaPhi) \
                + "\n$\mu_{\\theta}$=" +  str(muThtRad) + " rad" \
                + "\n$\kappa_{\\theta}$=" + str(kappaTht) \
                + "\n$\phi_0$=" + str(round(biasPhi, 2)) + " rad"  \
                + "\n$\\theta_0$="+str(round(biasTht, 2)) + " rad"

        jointProbFig.text(0.8, 0.5, info, fontsize=9)
        jointProbFig.savefig(fpath+"_jointHist.tiff")

        # Check if projection on XZ is correct
        alphSamples = np.arctan(np.tan(thtSamples), np.cos(phiSamples))
        ax = alphFig.gca()
        bins = np.linspace(-np.pi/2, np.pi/2, nbins + 1, endpoint=True)
        ax.hist(alphSamples, bins=bins, density=True, alpha=0.5)

        phiFig.savefig(fpath + "_phiHist.tiff")
        thtFig.savefig(fpath + "_thtHist.tiff")
        alphFig.savefig(fpath + "_alphHist.tiff")

# columns = ['muPhiRad', 'kappaPhi', 'muThtRad', 'kappaTht', 'rms err phi 2nd', 'rms err phi 4th', 'rms err alp 2nd', 'rms err alp 4th']
# summaryDF = pd.DataFrame(data=np.array([mu_phiChoices.data, k_phiChoices, mu_thtChoices.data, k_thtChoices.data,
#                                         rmserrs2phi, rmserrs4phi, rmserrs2alph, rmserrs4alph]).T, columns=columns)
columns = ['muPhiRad', 'kappaPhi', 'muThtRad', 'kappaTht', 'rms err alp 2nd', 'rms err alp 4th']
summaryDF = pd.DataFrame(data=np.array([mu_phiChoices, k_phiChoices, mu_thtChoices, k_thtChoices,
                                        rmserrs2alph, rmserrs4alph]).T, columns=columns)
summaryDF.to_csv(os.path.join(outDir, "summary.csv"), index=False)

f.close()
sys.stdout = orig_stdout


# ---------------------------------------------------------------------------
# # TEST TYPE - 4
#
# teststring = "TEST-Type-3: vonMises distribution for phi and theta,\n" +\
# "\t p(theta, phi) = p(theta) p(phi) \n" +\
# "Several tests to be performed by varying: mu_theta, mu_phi, kappa_theta, kappa_phi"
#
# outDir = os.path.join("tests_basis_function", "Type-3")
# if not os.path.exists(outDir):
#         os.mkdir(outDir)
#
# f = open(os.path.join(outDir, 'test_summary.txt'), 'w+')
# sys.stdout = f
# print(teststring)
#
# # Generating data:
# Ntests = 3
# Nsamples = 500
# nPoints = 180
# nbins = 36
#
# mu_thtChoices = np.random.uniform(-np.pi*5/12, np.pi*5/12, size=Ntests)
# mu_phiChoices = np.random.uniform(-np.pi*5/12, np.pi*5/12, size=Ntests)
# k_thtChoices  = np.random.uniform(0, 1, size=Ntests) * 2  # kappa <= 1
# k_phiChoices  = np.random.uniform(0, 1, size=Ntests) * 2  # kappa <= 1
#
# # Projection settings
# refdirRad = (0, 0)
# upsDomainRad = (0, np.pi)
# psiDomainRad = (0, np.pi)
#
# rmserrs2phi = []
# rmserrs4phi = []
# rmserrs2alph = []
# rmserrs4alph = []
# for testno in range(Ntests):
#         fpath = os.path.join(outDir, 'test-' + str(testno + 1))
#         muPhiRad = mu_phiChoices[testno]
#         muThtRad = mu_thtChoices[testno]
#         kappaPhi = k_phiChoices[testno]
#         kappaTht = k_thtChoices[testno]
#         print("\n\nTest - ", testno)
#         print("mu_phi: ", muPhiRad)  # central value of phi
#         print("kappa_phi: ", kappaPhi)  # spread of phi values
#         print("mu_theta: ", muThtRad)  # central value of phi
#         print("kappa_theta: ", kappaTht)  # spread of theta values
#
#         # GENERATING DATA using vonMises distribution
#         phiSamples = 0.5 * np.random.vonmises(2*muPhiRad, kappaPhi, size=Nsamples)  # values between [-pi/2, pi/2)
#         thtSamples = 0.5 * np.random.vonmises(2*muThtRad, kappaTht, size=Nsamples)  # values between [-pi/2, pi/2)
#
#         # projdir = (0, 0)
#         # result = test_basis_function(thtSamples, phiSamples, nbins=36,
#         #                              projdir=projdir, upsDomainRad=upsDomainRad, psiDomainRad=psiDomainRad,
#         #                              nUps=nPoints, nPsi=nPoints, refdirRad=refdirRad
#         #                              )
#         # jointProbFig, thtFig, phiFig, _, biasPhi, biasTht, rmserr2phi, rmserr4phi = result
#         # rmserrs2phi.append(rmserr2phi)
#         # rmserrs4phi.append(rmserr4phi)
#
#         projdir = (np.pi/2, 0)
#         result = test_basis_function(thtSamples, phiSamples, nbins=nbins,
#                                      projdir=projdir, upsDomainRad=upsDomainRad, psiDomainRad=psiDomainRad,
#                                      nUps=nPoints, nPsi=nPoints, refdirRad=refdirRad
#                                      )
#         jointProbFig, thtFig, phiFig, alphFig, biasPhi, biasTht, rmserr2alph, rmserr4alph = result
#         rmserrs2alph.append(rmserr2alph)
#         rmserrs4alph.append(rmserr4alph)
#
#         # print all stats to the figure as text for verification.
#         info = "$\mu_{\phi}$="+ str(muPhiRad) + " rad"\
#                 + "\n$\kappa_{\phi}$=" + str(kappaPhi) \
#                 + "\n$\mu_{\\theta}$=" +  str(muThtRad) + " rad" \
#                 + "\n$\kappa_{\\theta}$=" + str(kappaTht) \
#                 + "\n$\phi_0$=" + str(round(biasPhi, 2)) + " rad"  \
#                 + "\n$\\theta_0$="+str(round(biasTht, 2)) + " rad"
#
#         jointProbFig.text(0.8, 0.5, info, fontsize=9)
#         jointProbFig.savefig(fpath+"_jointHist.tiff")
#
#         # Check if projection on XZ is correct
#         alphSamples = np.arctan2(np.tan(thtSamples), 1 / np.cos(phiSamples))
#         ax = alphFig.gca()
#         bins = np.linspace(-np.pi/2, np.pi/2, nbins + 1, endpoint=True)
#         ax.hist(alphSamples, bins=bins, density=True, alpha=0.5)
#
#         phiFig.savefig(fpath + "_phiHist.tiff")
#         thtFig.savefig(fpath + "_thtHist.tiff")
#         alphFig.savefig(fpath + "_alphHist.tiff")
#
# # columns = ['muPhiRad', 'kappaPhi', 'muThtRad', 'kappaTht', 'rms err phi 2nd', 'rms err phi 4th', 'rms err alp 2nd', 'rms err alp 4th']
# # summaryDF = pd.DataFrame(data=np.array([mu_phiChoices.data, k_phiChoices, mu_thtChoices.data, k_thtChoices.data,
# #                                         rmserrs2phi, rmserrs4phi, rmserrs2alph, rmserrs4alph]).T, columns=columns)
# columns = ['muPhiRad', 'kappaPhi', 'muThtRad', 'kappaTht', 'rms err alp 2nd', 'rms err alp 4th']
# summaryDF = pd.DataFrame(data=np.array([mu_phiChoices.data, k_phiChoices, mu_thtChoices.data, k_thtChoices.data,
#                                         rmserrs2alph, rmserrs4alph]).T, columns=columns)
# summaryDF.to_csv(os.path.join(outDir, "summary.csv"), index=False)
