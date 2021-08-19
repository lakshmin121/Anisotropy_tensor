"""
Generate projections of the 3D images.
"""

import os
import sys
sys.path.append("..")
from glob import glob
from PyImModules.NCData import NCData
from skimage import img_as_float, img_as_ubyte
from skimage import io as skio
import numpy as np
from PyImModules.FileFolderUtils import append2fname, replace_extn
from matplotlib import pyplot as plt
from matplotlib_settings import *


def projection(img, axis):
    """
    Returns normalized projection of an image along the specified axis.
    :param img: input image as ndarray.
    :param axis: integer specifying axis along which projection is calculated.
    :return: normalized projection as ndarray.
    """
    proj = np.sum(img, axis=axis)
    proj = np.exp(-proj)
    proj = 1 - proj / np.max(proj)
    return proj

dataDir = "../data/art_images"
outDir = dataDir

# List of all 3D images
filenames = glob(os.path.join(dataDir, '*.nc'))
nfiles = len(filenames)
print("{} nc files detected.".format(nfiles))

for imgfpath in filenames:
    # imgfpath = filenames[0]
    imgfName = os.path.split(imgfpath)[-1]
    imgName = os.path.splitext(imgfName)[0]
    phifName = imgName + '_phiHist.csv'
    thtfName = imgName + '_thetaHist.csv'
    infoName = imgName + '.txt'
    print("Image filename: " + imgfName)
    print("Image: " + imgName)

    # Read image
    tomo = NCData.read_ncfile(imgfpath)
    img3D = img_as_float(tomo.ncdata)
    print(img3D.shape)

    # Projections
    z_proj = projection(img3D, axis=0)
    y_proj = projection(img3D, axis=1)
    x_proj = projection(img3D, axis=2)

    # Save projections
    skio.imsave(append2fname(replace_extn(imgfpath, '.tiff'), '_zproj'), img_as_ubyte(z_proj), plugin='tifffile')
    skio.imsave(append2fname(replace_extn(imgfpath, '.tiff'), '_yproj'), img_as_ubyte(y_proj), plugin='tifffile')
    skio.imsave(append2fname(replace_extn(imgfpath, '.tiff'), '_xproj'), img_as_ubyte(x_proj), plugin='tifffile')
