"""
Implementation of the Fourier Transform (FT) analysis of 2D FRC images.
Objective: Compare the ODF obtained from FT orientation tensor with original ODF used for the 2D simulation of the
artificial FRC image.
Challenges:
1. A complete efficient implementation of ODF from FT.
2. Develop a method to compare ODF from 2 similar distributions with slight error.
------------------------------------------------------------------------------------------------
@ Lakshminarayanan Mohana Kumar
Updated: 30th July 2021
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
dataDir = "../data/test_images_2D/vf20_ar50_tk50"
outDir = "../data/test_images_2D/vf20_ar50_tk50_results"
err_summary_file = "error_summary.csv"
if not os.path.exists(outDir):
    os.mkdir(outDir)

# Analysis parameters
mu = [0, 90]
kappa = [0.1, 0.25]  #, 0.5, 1, 5]
windowName = 'hann'  # window applied to image before FT
xWidth = 50  # px, image local neighbourhood width in x-direction
yWidth = xWidth  # A square neighbourhood is well-conditioned for FT analysis
xStep = 20  # px, image local neighbourhood sliding step distance in x-direction
yStep = xStep


def get_yticks(ymax, s):
    """
    Local function to generate yticks for the ODF plots.
    """
    ymaxid = np.ceil(ymax / s)
    return np.arange(0, (ymaxid + 0.2) * s, s)


def tensor_error(tensor1, tensor2):
    eigvals1 = np.linalg.eigvals(tensor1)
    eigvals2 = np.linalg.eigvals(tensor2)
    rms_err = np.linalg.norm(eigvals1.sort() - eigvals2.sort())
    return rms_err, eigvals1, eigvals2


#  ANALYSIS
relerr_FT_series = []
relerr_FT_fact_series = []
relerr_locFT_series = []
relerr_locFT_fact_series = []
for m in mu:
    for k in kappa:
        imgName = 'vm_m{0}k{1}'.format(m, k)
        img_fname = imgName + '.tiff'  # image to be analysed
        prob_fname = imgName + '_prob.csv'  # theoretical ODF (discrete) = PMF used for generation of fibre orientations
        txt_fname = imgName + '_out.txt'  # summary of results as an output text file.
        plot_fname = imgName + '_fit.tiff'  # plot of all estimated tensor-based ODFs
        print("\n\n", imgName)

        # Read image
        img = skio.imread(os.path.join(dataDir, img_fname), as_gray=True)

        # Fibre Orientation (Histogram) used during generation of image.
        phiDF = pd.read_csv(os.path.join(dataDir, prob_fname))  # read histogram data from CSV file.
        phiBins = phiDF['phiBins'].to_numpy()
        phiMin, phiMax = (round(np.min(phiBins), 0), round(np.max(phiBins), 0))
        phiBinc = 0.5 * (phiBins[1:] + phiBins[:-1])
        phiRange = phiBins[-1] - phiBins[0]

        phiHist = phiDF['phiHist'].to_numpy()
        phiHist = phiHist[:-1]
        dphi = np.mean(phiBins[1:] - phiBins[:-1])
        print("Check: Total probability from theoretical PMF = ", np.sum(phiHist) * dphi)


        # Theoretical tensor and ODF from histogram
        Q2_theo, A2_theo = orient_tensor_2D(phiHist, phiBinc)
        phiODF2_theo = tensor2odf_2D(phiBinc, A2_theo) * 2 * np.pi / 180
        print("Check: Total probability from theoretical ODF = ", np.sum(phiODF2_theo) * dphi)


        # Tensor and ODF from image
        Q2, A2 = fourier_orient_tensor(img, windowName=windowName)
        phiODF2 = tensor2odf_2D(phiBinc, A2*2) * 2 * np.pi / 180
        print("Check: Total probability from FT of image = ", np.sum(phiODF2) * dphi)


        # Tensor and ODF from image (windowed approach)
        Q2w, A2w = localised_fourier_orient_tensor(img, x_width=xWidth, y_width=yWidth, x_step=xStep, y_step=yStep,
                                                   windowName=windowName, order=2)
        phiODF2w = tensor2odf_2D(phiBinc, A2w*2) * 2 * np.pi / 180  # A2w * factor = 2 is empirically observed.
        # It is obsereved that multiplying with factor=2 gives excellent fit the theoretical ODF.
        # This is likely the result of symmetry of the fibres in image and their FT.
        print("Check: Total probability from localised FT of image = ", np.sum(phiODF2w) * dphi)
        odf_min = np.min(phiODF2w)
        # if odf_min < 0:
        #     phiODF2w = phiODF2w - odf_min
        # print("Check: Total probability from localised FT of image = ", np.sum(phiODF2w) * dphi)

        # Error estimates
        eigvals_theo = np.linalg.eigvals(A2_theo)
        eigvals2_ = np.linalg.eigvals(A2)
        eigvals2 = np.linalg.eigvals(2*A2)
        eigvals2w_ = np.linalg.eigvals(A2w)
        eigvals2w = np.linalg.eigvals(2*A2w)
        print("Eigenvalues:")
        print("\t Theoretical: ", eigvals_theo)
        print("\t FT: ", eigvals2_)
        print("\t FT factored: ", eigvals2)
        print("\t localised FT: ", eigvals2w_)
        print("\t localised FT factored: ", eigvals2w)
        print("Error estimates: ")
        relerr_FT = np.linalg.norm(eigvals2_ - eigvals_theo) / np.abs(eigvals_theo[0])
        relerr_FT_fact = np.linalg.norm(eigvals2 - eigvals_theo) / np.abs(eigvals_theo[0])
        relerr_locFT =  np.linalg.norm(eigvals2w_ - eigvals_theo) / np.abs(eigvals_theo[0])
        relerr_locFT_fact = np.linalg.norm(eigvals2w - eigvals_theo) / np.abs(eigvals_theo[0])
        print("\t FT: ", relerr_FT)
        print("\t FT factored: ", relerr_FT_fact)
        print("\t localised FT: ", relerr_locFT)
        print("\t localised FT factored: ", relerr_locFT_fact)
        relerr_FT_series.append(relerr_FT)
        relerr_FT_fact_series.append(relerr_FT_fact)
        relerr_locFT_series.append(relerr_locFT)
        relerr_locFT_fact_series.append(relerr_locFT_fact)

        # # DOCUMENTATION
        # fig = plt.figure(figsize=(3.7, 2))
        # ax = fig.gca()
        # plt.bar(phiBins, np.append(phiHist, np.nan), width=0.9 * dphi, align='edge', lw=0)
        # plt.plot(phiBinc, phiODF2_theo, color=np.asarray([176, 21, 21]) / 255, lw=0.75, linestyle='-', label='Theoretical')
        # plt.plot(phiBinc, phiODF2, color=np.asarray([230, 100, 20]) / 255, lw=0.75, linestyle='--', label='FT')
        # plt.plot(phiBinc, phiODF2w, color='black', lw=0.75, linestyle='dotted', label='Localised FT 2nd')
        # legend = plt.legend(loc='upper center', fontsize=10, ncol=3, numpoints=5,
        #                     frameon=True, mode=None, fancybox=False,
        #                     columnspacing=0.5, borderaxespad=0.15, edgecolor='darkgray')
        # legend.get_frame().set_linewidth(0.5)
        #
        # xticks = phiBins[::3]
        # ax.set_xlim([phiBins[0], phiBins[-1]])
        # ymax = np.max(phiHist)
        # if ymax > 0.01:
        #     yticks = get_yticks(ymax, s=0.01)
        # else:
        #     yticks = get_yticks(ymax, s=0.005)
        # ax.set_ylim([yticks[0], yticks[-1]])
        # ax.set_yticks(yticks)
        # ax.set_xticks(xticks)
        # plt.xticks(rotation=90)
        # plt.yticks(rotation=90)
        # ax.set_xticklabels(xticks)
        # ax.set_xlabel("$\phi$ [degrees]")
        # ax.set_ylabel("p($\phi$)")
        #
        # fig.savefig(os.path.join(outDir, plot_fname), dpi=300)
        #
        # # General information to text file
        # with open(os.path.join(outDir, txt_fname), 'w+') as f:
        #     f.write("# Input \n")
        #     f.write("# ----- \n")
        #     f.write("Image: {}\n".format(img_fname))
        #     f.write("PMF (histogram) data: {}\n".format(prob_fname))
        #
        #     f.write("\n\n# Theoretical Distribution (used to generate image)\n")
        #     f.write("# ------------------------ \n")
        #     f.write("Check: Total probability from PMF = {}\n".format(np.sum(phiHist) * dphi))
        #     f.write("\nEstimating the tensor representation of theoretical PMF\n")
        #     f.write("Orientation tensor (theoretical): \n")
        #     f.write("{}\n".format(Q2_theo))
        #     f.write("Anisotropy tensor (theoretical): \n")
        #     f.write("{}\n".format(A2_theo))
        #     f.write("\nEstimating ODF (continuous) from the tensor representation of theoretical PMF\n")
        #     f.write("Check: Total probability from PMF = {}\n".format(np.sum(phiODF2_theo) * dphi))
        #
        #     f.write("\n\n# Distribution from Fourier Analysis of Image\n")
        #     f.write("# -------------------------------------------- \n")
        #     f.write("Image shape: {}\n".format(img.shape))
        #     f.write("Image window used: {}\n".format(windowName))
        #     f.write("\nEstimating the tensor representation in Fourier space\n")
        #     f.write("Orientation tensor (FT, 2nd order): \n")
        #     f.write("{}\n".format(Q2))
        #     f.write("Anisotropy tensor (FT, 2nd order): \n")
        #     f.write("{}\n".format(A2))
        #     f.write("\nEstimating ODF (continuous) from the tensor representation of theoretical PMF\n")
        #     f.write("Check: Total probability from FT ODF = {}\n".format(np.sum(phiODF2) * dphi))
        #
        #     f.write("\n\n# Distribution from Localised Fourier Analysis of Image\n")
        #     f.write("# ------------------------------------------------------ \n")
        #     f.write("Local neighbourhood selection: \n")
        #     f.write("\t x-width: {} px\n".format(xWidth))
        #     f.write("\t y-width: {} px\n".format(yWidth))
        #     f.write("\t x-step: {} px\n".format(xStep))
        #     f.write("\t x-step: {} px\n".format(yStep))
        #     f.write("\nEstimating the tensor representation using Localised Fourier Transform\n")
        #     f.write("Orientation tensor (localised FT, 2nd order): \n")
        #     f.write("{}\n".format(Q2w))
        #     f.write("Anisotropy tensor (localised FT, 2nd order): \n")
        #     f.write("{}\t Note: anisotropy is multiplied by factor=2.\n".format(A2w*2))
        #
        #     f.write("\n\n# Eigenvalues: \n")
        #     f.write("\t Theoretical: {}\n".format(eigvals_theo))
        #     f.write("\t FT: {}\n".format(eigvals2_))
        #     f.write("\t FT factored: {}\n".format(eigvals2))
        #     f.write("\t localised FT: {}\n".format(eigvals2w_))
        #     f.write("\t localised FT factored: {}\n".format(eigvals2w))
        #
        #     f.write("\n\n# Relative Error: \n")
        #     f.write("\t FT: {}\n".format(relerr_FT))
        #     f.write("\t FT factored: {}\n".format(relerr_FT_fact))
        #     f.write("\t localised FT: {}\n".format(relerr_locFT))
        #     f.write("\t localised FT factored: {}\n".format(relerr_locFT_fact))


# ERROR SUMMARY PLOT
keys = ['FT', 'factored FT', 'localised-FT', 'factored localised-FT']
dframe = dict(zip(keys, [relerr_FT_series, relerr_FT_fact_series, relerr_locFT_series, relerr_locFT_fact_series]))
# sns.color_palette("rocket")
# assert len(relerr_FT_series) == len(mu) * len(kappa)
# errfig = plt.figure(figsize=set_fig_size(3.54, 1))
# ax = errfig.gca()
# x = np.array(range(len(relerr_FT_series)))
# yax_loc = len(relerr_FT_series) // 2 + 0.5
# ax.set_xlim([x[0]-0.1, x[-1]+0.1])
# ax.set_xticks(x)
# ax.bar(x-0.4, relerr_FT_series, width=0.2, label='FT')
# ax.bar(x-0.2, relerr_FT_fact_series, width=0.2, label='FT factored')
# ax.bar(x+0.2, relerr_locFT_series, width=0.2, label='localised FT')
# ax.bar(x+0.4, relerr_locFT_fact_series, width=0.2, label='localised FT factored')
# ax.set_ylim([0, 2])
# ax.set_yticks([0, 1, 2])
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_position(('data', yax_loc))
# ax.spines['bottom'].set_position(('data', 1))
# legend = plt.legend(loc='upper left', fontsize=10, ncol=2, numpoints=5, frameon=True, mode=None, fancybox=False,
#                     columnspacing=0.5, borderaxespad=0.15, edgecolor='darkgray')
# errfig.savefig(os.path.join(outDir, 'error_summary.tiff'), dpi=300)
