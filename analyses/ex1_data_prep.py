"""
Code to learn the 3D orientation distribution fucntion (ODF) model based on
spherical harmonics to the orientation distribution observed in multiple projections.

All projections (x, y, and z) of a 3D image are read, the fibre orientation PMFs on
these images are stored along with the direction of projection.

PART-1: Data preparation
Refer: ex5_FT_3Dimages.py
_______________________________________________________________________________________
@ Lakshminarayanan Mohana Kumar
09 Oct 2021
"""

import os
import sys
sys.path.append('..')
from glob import glob
from itertools import groupby
import numpy as np
from skimage.io import imread
from fibfourier import fourier_orient_hist


dataDir = "../data/art_images"
outDir = dataDir
ncfPaths = glob(os.path.join(dataDir, "*.nc"))
ZPAD_FACTOR = 5


def read_projimg(projimgfpath):
    projdirRad = (0, 0)
    ORIENTDICT = {'xproj': (np.pi/2, 0),
                  'yproj': (np.pi / 2, np.pi / 2),
                  'zproj': (0, 0),
                  }
    print(projimgfpath)
    fname = os.path.split(projimgfpath)[-1]
    imgname = os.path.splitext(fname)[0]
    orientkey = imgname.split('_')[-1]

    try:
        projdirRad = ORIENTDICT[orientkey]
    except KeyError:
        sections = groupby(orientkey, key=lambda char: char.isdigit())
        substrs  = [''.join(sec) for cond, sec in sections]
        if substrs[0] == 'tht' and substrs[2] == 'phi':
            projdirRad = int(substrs[1]), int(substrs[3])
            projdirRad = np.deg2rad(np.asarray(projdirRad))
        else:
            return None, None

    projimg = imread(projimgfpath, as_gray=True)

    return projdirRad, projimg


# Get data
if __name__ == '__main__':  # Required for multiprocessing used in fourier_orient_hist
    for ncfPath in ncfPaths:
        ncfName = os.path.split(ncfPath)[-1]
        imgName = os.path.splitext(ncfName)[0]
        print(imgName)

        projimgFpaths = glob(os.path.join(dataDir, imgName+'*.tiff'))

        #   Get projected images, projection directions
        projDirsRad = []
        orientHists = []

        for projimgFpath in projimgFpaths:
            projDirRad, projImg = read_projimg(projimgFpath)
            if projDirRad is not None:
                projDirsRad.append(projDirRad)

                #   Extract PMF from projected images
                orientHist, orientBins = fourier_orient_hist(projImg, windowName='hann', zpad_factor=ZPAD_FACTOR)
                orientHists.append(orientHist)

        print("No. of projections: ", len(projDirsRad))
        print("Projection directions: ", projDirsRad)

        #   Save data in simplified format for learning
        np.savez(os.path.join(outDir, imgName+'_projPMFData'), projDirsRad=projDirsRad, orientHists=orientHists)

    # Preprocess data if needed
    # Set up model
    # Learning algorithm -
    #   Convex Cost Function
    #   Gradient
    #   Convex optimization
    #   Learning monitoring - learning rates, errors, variance, and bias.
    # Results
    # Plots
    # Accuracy and error estimates
