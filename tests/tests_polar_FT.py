
import os
import sys
sys.path.append("..")
import numpy as np
import pandas as pd
from skimage import img_as_float
import skimage.io as skio
from skimage.transform import rotate, rescale, warp_polar
from skimage.filters import window
from scipy.fftpack import fft2, fftshift
from scipy.interpolate import interp2d
from matplotlib import pyplot as plt
from matplotlib_settings import *
from fiborient import tensor2odf_2D, orient_tensor_2D


dataDir = "../data/test_images_2D/vf20_ar50_tk50"
outDir = "../data/test_images_2D/vf20_ar50_tk50_polar"
muDeg = [0, 90]
kappa = [0.1, 0.25, 0.5, 1, 5]
# ncases = len(mu) * len(kappa)
windowFT = 'hann'
zpad_factor = 10.5

# Selection of image file
# m = mu[0]
# k = kappa[0]

for mu in muDeg:
    for k in kappa:
# mu = 90
# k = 1

        print(mu, k)
        imgName = 'vm_m{0}k{1}'.format(mu, k)
        img_fname = imgName + '.tiff'  # image to be analysed
        prob_fname = imgName + '_prob.csv'  # theoretical ODF (discrete) = PMF used for generation of fibre orientations
        out_fname = imgName + '_out.txt'

        fibdia = 4  # px
        fiblen = 200  # px

        if not os.path.exists(outDir):
            os.mkdir(outDir)

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

        phiBinsExtnd = np.arange(0, 361, dphi)
        phiHistExtnd = np.concatenate((phiHist / 2, phiHist / 2))
        msg = "len(phiBinsExtnd) = {0} while len(phiHistExtnd) = {1}".format(len(phiBinsExtnd), len(phiHistExtnd))
        assert len(phiBinsExtnd) == len(phiHistExtnd) + 1, print(msg)


        # Theoretical tensor and ODF from histogram
        Q2_theo, A2_theo = orient_tensor_2D(phiHist, phiBinc)
        phiODF2_theo = tensor2odf_2D(phiBinc, A2_theo) * 2 * np.pi / 180
        print("Check: Total probability from theoretical ODF = ", np.sum(phiODF2_theo) * dphi)


        # Fourier Analysis
        wimg = img * window(windowFT, img.shape)  # windowed image.
        shparr = np.array(img.shape)
        zpad_shp = np.round(zpad_factor * shparr, 0).astype(np.int)
        wimgFTcomplx = fftshift(fft2(wimg, tuple(zpad_shp)))  # FFT of zero-padded and windowed image.
        wimgFT = np.absolute(wimgFTcomplx)  # FFT magnitude -> spectrum
        wimgFT = wimgFT - np.mean(wimgFT)
        wimgFTlog = 20 * np.log(1+wimgFT)  # FFT as power (log)


        # Polar Analysis
        # rmin = wimgFT.shape[0] // fiblen
        # rmax = wimgFT.shape[0] // fibdia
        rmax = 2 * fiblen * zpad_factor
        rmin = 2 * fibdia * zpad_factor
        phiRad_p = np.deg2rad(np.arange(0, 360))
        print("FT max radius: ", rmax)
        print("FT min radius: ", rmin)

        xc, yc = np.asarray(wimgFT.shape) // 2
        xl, yl = xc + rmax * np.cos(phiRad_p), yc + rmax * np.sin(phiRad_p)
        xh, yh = xc + rmin * np.cos(phiRad_p), yc + rmin * np.sin(phiRad_p)

        imgFTpolar = warp_polar(wimgFT, radius=rmax)
        imgFTpolar = imgFTpolar.T
        print("imgFTpolar.shape: ", imgFTpolar.shape)

        r0 = int(rmin)
        phiHistp = np.sum(imgFTpolar[r0:], axis=0)  # Radial sum
        m = len(phiHistp) // 2
        phiHistp = phiHistp[:m] + phiHistp[m:]
        phiHistsum = np.sum(phiHistp)
        phiHistp = phiHistp / phiHistsum
        phiHistp = np.roll(phiHistp, -90)
        phiBinsp = np.arange(0, 181, 1)
        phiBinspc = 0.5 * (phiBinsp[1:] + phiBinsp[:-1])

        # Correction
        excess = np.sum(imgFTpolar[:r0+1])
        phiHistp = phiHistp * (1 + excess / phiHistsum)  # approximate
        phiHistp = phiHistp / np.sum(phiHistp)

        # Tensor from FT
        Q2, A2 = orient_tensor_2D(phiHistp, phiBinspc)
        phiODF2 = tensor2odf_2D(phiBinspc, A2) * 2 * np.pi / 180
        dphip = np.mean(phiBinsp[1:] - phiBinsp[:-1])
        print("Check: Total probability from theoretical ODF = ", np.sum(phiODF2) * dphip)


        # DOCUMENTATION
        fig, axes = plt.subplots(1, 3, figsize=(6, 2), dpi=300)
        axs = axes.ravel()
        axs[0].imshow(img, cmap='gray')
        axs[0].set_title("SFRC image")
        axs[1].imshow(wimg)
        axs[1].set_title("windowed image")
        axs[2].imshow(wimgFTlog)
        axs[2].plot(xl, yl, color="0.25", lw=0.5)
        axs[2].plot(xh, yh, color="0.25", lw=0.5)
        axs[2].set_title("FT spectrum (log)")
        fig.savefig(os.path.join(outDir, imgName+'_FT.tiff'))


        figp = plt.figure(figsize=(2, 2), dpi=300)
        ax = figp.gca()
        ax.imshow(np.log(1+imgFTpolar))
        ax.set_xlim([0, 360])
        ax.set_ylim([0, imgFTpolar.shape[0]])
        xticks = np.arange(0, 361, 120)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)
        ax.set_aspect(360 / imgFTpolar.shape[0])
        ax.plot(xticks, [rmin]*len(xticks), color="0.25", lw=0.5, linestyle='dashed')
        figp.savefig(os.path.join(outDir, imgName+'_polar.tiff'))


        figb = plt.figure(figsize=(4, 2), dpi=300)
        ax = figb.gca()
        # ax.bar(phiBinsExtnd[:-1], phiHistExtnd, width=dphi, align='edge',linewidth=0, alpha=0.5)
        ax.bar(phiBins[:-1], phiHist, width=dphi, align='edge',linewidth=0, alpha=0.5)
        ax.bar(phiBinsp[:-1], phiHistp, width=1.0, align='edge', linewidth=0, alpha=0.5)
        plt.plot(phiBinc, phiODF2_theo, color=np.asarray([176, 21, 21]) / 255, lw=0.75, linestyle='-', label='Theoretical')
        plt.plot(phiBinspc, phiODF2, color=np.asarray([230, 100, 20]) / 255, lw=0.75, linestyle='--', label='FT')
        ax.set_xticks(phiBinsp[::30])
        ax.set_xticklabels(phiBinsp[::30])
        figb.savefig(os.path.join(outDir, imgName+'_hist.tiff'))

        # # plt.axis('off')
        # plt.show()


        with open(os.path.join(outDir, out_fname), 'w+') as f:
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
            f.write("Image window used: {}\n".format(windowFT))
            f.write("\nEstimating the tensor representation in Fourier space\n")
            f.write("Orientation tensor (FT, 2nd order): \n")
            f.write("{}\n".format(Q2))
            f.write("Anisotropy tensor (FT, 2nd order): \n")
            f.write("{}\n".format(A2))
            f.write("\nEstimating ODF (continuous) from the tensor representation of theoretical PMF\n")
            f.write("Check: Total probability from FT ODF = {}\n".format(np.sum(phiODF2) * dphip))