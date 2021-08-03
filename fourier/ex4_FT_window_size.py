"""
To study the effect of window size on the localised FT of the image, and corresponding anisotropy tensor.
Objective:
    1. Parametric study of FT window size.
    2. Importantly, evaluating the effect of fibre size relative to image width in the multiplication factor to
    anisotropy tensor.
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
outDir = "../data/test_images_2D/vf20_ar50_tk50_windowsize"
if not os.path.exists(outDir):
    os.mkdir(outDir)

# Analysis parameters
# TODO: read from file.
factor = 2
imscale = 20  # px/mm
fibdia = 0.2  # mm
fibAR = 50  # mm / mm
df = int(np.ceil(fibdia * imscale))  # diameter of fibre, px
lf = int(np.ceil(df * fibAR))  # length of fibre, px

xWidths = (np.array([2.5, 1.25, 1.0, 0.75, 0.5]) * lf).astype(int)

mu = [0, 90]
kappa = [0.1, 0.25, 0.5, 1, 5]
windowName = 'hann'  # window applied to image before FT

def get_yticks(ymax, s):
    """
    Local function to generate yticks for the ODF plots.
    """
    ymaxid = np.ceil(ymax / s)
    return np.arange(0, (ymaxid + 0.2) * s, s)

# Image file selection
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
        phiMin, phiMax = (round(np.min(phiBins), 0), round(np.max(phiBins), 0)) # lower and upper bounds of phi -> domain
        phiBinc = 0.5 * (phiBins[1:] + phiBins[:-1])  # centres of phi bins
        phiRange = phiBins[-1] - phiBins[0]  # range of phi values, (0, 180) => 180-0 = 180

        phiHist = phiDF['phiHist'].to_numpy()
        phiHist = phiHist[:-1]  # last element is NaN
        dphi = np.mean(phiBins[1:] - phiBins[:-1])  # phi bin width, dphi for integration of discrete PMF.
        print("Check: Total probability from theoretical PMF = ", np.sum(phiHist) * dphi)


        # Theoretical tensor and ODF from histogram
        Q2_theo, A2_theo = orient_tensor_2D(phiHist, phiBinc)
        phiODF2_theo = tensor2odf_2D(phiBinc, A2_theo) * 2 * np.pi / 180
        eigvals_theo = np.sort(np.linalg.eigvals(A2_theo))
        print("Check: Total probability from theoretical ODF = ", np.sum(phiODF2_theo) * dphi)


        # Tensor and ODF from image
        Q2, A2 = fourier_orient_tensor(img, windowName=windowName)
        phiODF2 = tensor2odf_2D(phiBinc, A2 * factor) * 2 * np.pi / 180
        eigvals2 = np.sort(np.linalg.eigvals(A2 * factor))
        err2 = np.linalg.norm(eigvals2 - eigvals_theo)
        relerr2 = err2 / np.abs(eigvals_theo[0])
        print("Check: Total probability from FT of image = ", np.sum(phiODF2) * dphi)

        # ANALYSIS & DOCUMENTATION
        fig = plt.figure(figsize=(3.7, 2))
        ax = fig.gca()
        plt.bar(phiBins, np.append(phiHist, np.nan), width=0.9 * dphi, align='edge', lw=0)
        plt.plot(phiBinc, phiODF2_theo, color=np.asarray([176, 21, 21]) / 255, lw=0.75, linestyle='-', label='Theoretical')
        plt.plot(phiBinc, phiODF2, color=np.asarray([230, 100, 20]) / 255, lw=0.75, linestyle='--', label='FT')

        # General information to text file
        with open(os.path.join(outDir, txt_fname), 'w+') as f:
            f.write("# Input \n")
            f.write("# ----- \n")
            f.write("Image: {}\n".format(img_fname))
            f.write("PMF (histogram) data: {}\n".format(prob_fname))

            f.write("\n\n# Theoretical Distribution (used to generate image)\n")
            f.write("# ------------------------ \n")
            f.write("Check: Total probability from PMF = {}\n".format(np.sum(phiHist) * dphi))
            f.write("\nEstimating the tensor representation of theoretical PMF\n")
            f.write("Orientation tensor (theoretical): \n")
            f.write("{}\n".format(Q2_theo))
            f.write("Anisotropy tensor (theoretical): \n")
            f.write("{}\n".format(A2_theo))
            f.write("\nEstimating ODF (continuous) from the tensor representation of theoretical PMF\n")
            f.write("Check: Total probability from ODF = {}\n".format(np.sum(phiODF2_theo) * dphi))

            f.write("\n\n# Distribution from Fourier Analysis of Image\n")
            f.write("# -------------------------------------------- \n")
            f.write("Image shape: {}\n".format(img.shape))
            f.write("Image window used: {}\n".format(windowName))
            f.write("\nEstimating the tensor representation in Fourier space\n")
            f.write("Orientation tensor (FT, 2nd order): \n")
            f.write("{}\n".format(Q2))
            f.write("Anisotropy tensor (FT, 2nd order): \n")
            f.write("{}\n".format(A2))
            f.write("\nEstimating ODF (continuous) from the tensor representation of theoretical PMF\n")
            f.write("Check: Total probability from FT ODF = {}\n".format(np.sum(phiODF2) * dphi))
            f.write("Eigenvalues: {}\n".format(eigvals2))
            f.write("Error (normed): {}\n".format(err2))
            f.write("Rel. error: {}\n".format(relerr2))

            f.write("\n\n# Distribution from Localised Fourier Analysis of Image\n")
            f.write("# ------------------------------------------------------ \n")

            eigList = []
            errList = []
            relerrList = []
            for wno, xWidth in enumerate(xWidths):
                yWidth = xWidth  # A square neighbourhood is well-conditioned for FT analysis
                xStep = xWidth  # px, image local neighbourhood sliding step distance in x-direction
                yStep = xStep
                # yWidth = xWidth => square neighbourhood.
                # xStep = xWidth => no overlap between adjacent neighbourhoods.
                # yStep = xStep => no overlap in y-direction as well.


                # Tensor and ODF from image (windowed approach)
                Q2w, A2w = localised_fourier_orient_tensor(img, x_width=xWidth, y_width=yWidth, x_step=xStep, y_step=yStep,
                                                           windowName=windowName, order=2)
                phiODF2w = tensor2odf_2D(phiBinc, A2w * factor) * 2 * np.pi / 180  # A2w * factor = 2 is empirically observed.
                eigvals2w = np.sort(np.linalg.eigvals(A2w * factor))
                err2w =  np.linalg.norm(eigvals2w - eigvals_theo)
                relerr2w = err2w / np.abs(eigvals_theo[0])
                # It is obsereved that multiplying with factor=2 gives excellent fit the theoretical ODF.
                # This is likely the result of symmetry of the fibres in image and their FT.
                print("Check: Total probability from localised FT of image = ", np.sum(phiODF2w) * dphi)

                eigList.append(eigvals2w)
                errList.append(err2w)
                relerrList.append(relerr2w)
                plt.plot(phiBinc, phiODF2w, lw=0.5, label='local {} px'.format(xWidth))

                # Documentation of each iteration.
                f.write("Case: {}\n".format(wno))
                f.write("\t x-width: {} px\n".format(xWidth))
                f.write("\t y-width: {} px\n".format(yWidth))
                f.write("\t x-step: {} px\n".format(xStep))
                f.write("\t x-step: {} px\n".format(yStep))
                f.write("\tOrientation tensor (localised FT, 2nd order): \n")
                f.write("\t{}\n".format(Q2w))
                f.write("\tAnisotropy tensor (localised FT, 2nd order): \n")
                f.write("\t{}.\n".format(A2w))
                f.write("\tEigenvalues: {}\n".format(eigvals2w))
                f.write("\tError (normed): {}\n".format(err2w))
                f.write("\tRel. error: {}\n".format(relerr2w))

            # Error summary
            f.write("\n\nError Summary:\n")
            f.write("----------------\n")
            f.write("\txWidth [px] \t Eigval \t Error \t Rel. error\n")
            f.write("\t{0} \t\t {1} \t {2} \t {3}\n".format(img.shape[0], eigvals2, err2, relerr2))
            for n in range(len(xWidths)):
                f.write("\t{0} \t\t {1} \t {2} \t {3}\n".format(xWidths[n], eigList[n], errList[n], relerrList[n]))


        legend = plt.legend(loc='upper center', fontsize=9, ncol=3, numpoints=5,
                            frameon=True, mode=None, fancybox=False,
                            columnspacing=0.5, labelspacing=0.25, borderaxespad=0.15, edgecolor='darkgray')
        legend.get_frame().set_linewidth(0.5)

        xticks = phiBins[::3]
        ax.set_xlim([phiBins[0], phiBins[-1]])
        ymax = np.max(phiHist)
        if ymax > 0.01:
            yticks = get_yticks(ymax, s=0.01)
        else:
            yticks = get_yticks(ymax, s=0.005)
        ax.set_ylim([yticks[0], yticks[-1]])
        ax.set_yticks(yticks)
        ax.set_xticks(xticks)
        plt.xticks(rotation=90)
        plt.yticks(rotation=90)
        ax.set_xticklabels(xticks)
        ax.set_xlabel("$\phi$ [degrees]")
        ax.set_ylabel("p($\phi$)")

        fig.savefig(os.path.join(outDir, plot_fname), dpi=300)
