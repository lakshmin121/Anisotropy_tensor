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
#
# f.close()
# sys.stdout = orig_stdout


# ---------------------------------------------------------------------------
# TEST TYPE - 1

teststring = "TEST-Type-2: All fibres in same direction,\n" +\
"\t p(theta, phi) = delta(theta - theta_0) delta(phi - phi_0) \n" +\
"Then, by projecting to XY-plane (phi-plane), \n" +\
"        p(phi) = delta(phi - phi_0) \n " +\
"Several tests to be performed by varying: theta_0, phi_0"

outDir = os.path.join("tests_basis_function", "Type-1")
if not os.path.exists(outDir):
        os.mkdir(outDir)

f = open(os.path.join(outDir, 'test_summary.txt'), 'w+')
sys.stdout = f
print(teststring)

# Generating data:
Ntests = 5
Nsamples = 500
nPoints = 180

# phi follows uniform distribution
# phiSamples = np.random.uniform(-np.pi/2, np.pi/2, size=Nsamples)  # values between [-pi/2, pi/2)
# theta is same for all fibres
thtSamples = np.ones(Nsamples)
phiChoices = np.random.uniform(-np.pi/3, np.pi/3, size=Ntests)
thtChoices = np.random.uniform(-np.pi/3, np.pi/3, size=Ntests)

# Projection settings
refdirRad = (0, 0)
upsDomainRad = (0, np.pi)
psiDomainRad = (0, np.pi)
projdir = (0, 0)  # using value passed to this function

rmserrs2 = []
rmserrs4 = []
for testno in range(Ntests):
        fpath = os.path.join(outDir, 'test-' + str(testno+1))
        phi0 = phiChoices[testno]
        theta0 = thtChoices[testno]
        print("\n\nTest - ", testno)
        print("theta0 = ", theta0)  # value of theta
        print("phi0 = ", phi0)  # value of phi

        result = test_basis_function(thtSamples * phi0, thtSamples * theta0, nbins=36,
                                     projdir=projdir, upsDomainRad=upsDomainRad, psiDomainRad=psiDomainRad,
                                     nUps=nPoints, nPsi=nPoints, refdirRad=refdirRad
                                     )
        jointProbFig, thtFig, phiFig, alphFig, biasPhi, biasTht, rmserr2, rmserr4 = result
        rmserrs2.append(rmserr2)
        rmserrs4.append(rmserr4)

        # print all stats to the figure as text for verification.
        info = "$\phi \~ U(-\pi/2, \pi/2)$"\
                + "\n$\\theta$ = " + str(np.round(np.pi/2+theta0, 3))

        jointProbFig.text(0.8, 0.5, info, fontsize=9)
        jointProbFig.savefig(fpath+"_jointHist.tiff")

        ax = alphFig.gca()
        if phi0 < 0:
                ax.legend(loc='upper right')
        else:
                ax.legend(loc='upper left')

        phiFig.savefig(fpath + "_phiHist.tiff")
        thtFig.savefig(fpath + "_thtHist.tiff")
        alphFig.savefig(fpath + "_alphHist.tiff")

columns = ['theta0', 'phi0', 'rms err 2nd', 'rms err 4th']
summaryDF = pd.DataFrame(data=np.array([thtChoices.data, phiChoices.data, rmserrs2, rmserrs4]).T, columns=columns)
summaryDF.to_csv(os.path.join(outDir, "summary.csv"), index=False)

f.close()
sys.stdout = orig_stdout


# ---------------------------------------------------------------------------
# fpath = os.path.join(outDir, "test1")
# muPhiRad = np.round(np.deg2rad(155), 2)
# muThtRad = np.round(np.deg2rad(135), 2)
# kappaPhi = 1
# kappaTht = 2
# Nsamples = 1000
#
# nPoints = 180
# print("mu_phi: ", muPhiRad)  # central value of phi
# print("kappa_phi: ", kappaPhi)  # spread of phi values
# print("mu_theta: ", muThtRad)  # central value of phi
# print("kappa_theta: ", kappaTht)  # spread of theta values
#
# # GENERATING DATA using vonMises distribution
# phiSamples = 0.5 * np.random.vonmises(2*(muPhiRad - np.pi/2), kappaPhi, size=Nsamples)  # values between [-pi/2, pi/2)
# thtSamples = 0.5 * np.random.vonmises(2*(muThtRad - np.pi/2), kappaTht, size=Nsamples)  # values between [-pi/2, pi/2)
#
# jointProbFig, thtFig, phiFig, alphFig, biasPhi, biasTht = test_basis_function(phiSamples, thtSamples, nbins=36,
#                                                                               projdir=(0, 0))
#
# # print all stats to the figure as text for verification.
# info = "$\mu_{\phi}$="+ str(muPhiRad) + " rad"\
#         + "\n$\kappa_{\phi}$=" + str(kappaPhi) \
#         + "\n$\mu_{\\theta}$=" +  str(muThtRad) + " rad" \
#         + "\n$\kappa_{\\theta}$=" + str(kappaTht) \
#         + "\n$\phi_0$=" + str(round(biasPhi, 2)) + " rad"  \
#         + "\n$\\theta_0$="+str(round(biasTht, 2)) + " rad"
#
# jointProbFig.text(0.8, 0.5, info, fontsize=9)
# jointProbFig.savefig(fpath+"_jointHist.tiff")
#
# phiFig.savefig(fpath + "_phiHist.tiff")
# thtFig.savefig(fpath + "_thtHist.tiff")
# alphFig.savefig(fpath + "_alphHist.tiff")

# columns = ['muPhiRad', 'kappaPhi', 'muThtRad', 'kappaTht', 'rms err 2nd', 'rms err 4th']
# summaryDF = pd.DataFrame(data=np.array([muPhiRad, kappaPhi, muThtRad, kappaTht, rms_errs2, rms_errs4]).T, columns=columns)
# summaryDF.to_csv(os.path.join(outDir, "summary.csv"), index=False)
