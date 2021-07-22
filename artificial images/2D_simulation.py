"""
Simulation of artificial projected images of FRC.
__________________________________________________
@ Lakshminarayanan Mohana Kumar
16th July 2021
"""

import os
from os import path
import numpy as np
import skimage.io as skio
from skimage import img_as_ubyte
from fibre2D import Fibre
from skimage.draw import line, bezier_curve
from skimage.morphology import dilation
from matplotlib import pyplot as plt
from matplotlib_settings import *
import orientation_probabilities as op
from fiborient import orient_tensor_2D
import pandas as pd

m= 0
for k in [0.1, 0.25, 0.5, 1, 5]:
    # INPUT
    outDir = "../data/test_images_2D/vf20p"
    imgName = 'vm_m{0}k{1}'.format(m, k)
    img_fname = imgName + '.tiff'
    txt_fname = imgName + '.txt'
    probplot_fname = imgName + '_prob.tiff'
    prob_fname = imgName + '_prob.csv'

    # Specimen and fibre properties:
    imscale = 20  # px/mm
    sp_shape = (50, 50)  # mm
    sp_tk = 50  # mm
    fibdia = 0.2  # mm
    fibAR = 50  # mm / mm
    fibdosage = 0.002

    odf = 'vonmises'  # Orietation Distribution Function
    parameters = {'muDeg': m,
                  'kappa': k,
                  'spreadDeg': 180}

    # simulated properties:
    df = int(np.ceil(fibdia * imscale))  # px
    lf = int(np.ceil(df * fibAR))  # px
    img_shp = tuple(np.array(np.asarray(sp_shape) * imscale, dtype=int))  # px
    fibvol = 0.25 * np.pi * fibAR * fibdia**3
    Nf = int(fibdosage * np.prod(np.asarray(sp_shape)) * sp_tk / fibvol)  # number of fibres
    print("Number of fibres: ", Nf)


    # GENERATION OF FIBRES


    def generate_fibre_orientations(odf='vonmises', size=1, **kwargs):
        if odf=='vonmises':
            try:
                kappa = kwargs['kappa']
            except KeyError as e:
                raise e("Require value of kappa.")
            phi_vals = op.vonmises(size=size, **kwargs)
        elif odf=='uniform':
            try:
                phidomainDeg = kwargs['phidomainDeg']
                phi_vals = op.uniform(phidomainDeg=phidomainDeg, size=size)
            except KeyError:
                print("using default domain (0, 180) degrees for uniform distribution.")
                phi_vals = op.uniform(size=size)
        else:
            raise ValueError("Invalid odf: {}".format(odf))
        return phi_vals


    # Fibre Orientation
    phivals = generate_fibre_orientations(odf=odf, size=Nf, **parameters)
    phiMin, phiMax = (round(np.min(phivals), 1), round(np.max(phivals), 1))

    # Fibre Locations
    xvals, yvals = tuple(np.random.uniform(low=0, high=img_shp[dim], size=Nf) for dim in range(len(img_shp)))
    spcentre = np.asarray(img_shp)//2

    # Fibres:
    fibs = [Fibre(df, lf, (x, y), phiDegxy=phi, spcentre=spcentre) for x, y, phi in zip(xvals, yvals, phivals)]


    # IMAGE
    img = np.zeros(img_shp)


    def insert_fibres(image, fibres, kind='line', cval=1, **lineargs):
        for fibre in fibres:
            r0, c0 = tuple(fibre.start)
            r1, c1 = tuple(fibre.centroid)
            r2, c2 = tuple(fibre.stop)
            if kind == 'line':
                linecoords = line(r0, c0, r2, c2)
            elif kind == 'bezier':
                linecoords = bezier_curve(r0, c0, r1, c1, r2, c2, **lineargs)
            else:
                raise ValueError("kind must be 'line' or 'bezier'")
            image[linecoords] = cval


    # Insert fibres
    insert_fibres(img, fibs, kind='bezier', weight=2)

    # Impart fibre thickness
    rf = df // 2
    img = dilation(img, selem=np.ones((rf, rf)))

    # ODF plot
    fig = plt.figure()
    ax = fig.gca()
    myround = lambda x: int(round(x, 0))
    phiBins = np.arange(myround(phiMin), myround(phiMax)+1, 10)
    hist, bins, _ = ax.hist(phivals, bins=phiBins, density=True)
    ax.set_xlim([phiBins[0], phiBins[-1]])
    ax.set_xticks(phiBins[::3])
    ax.set_xticklabels(phiBins[::3])
    ax.set_xlabel("$\phi$ [degrees]")

    # Orientation Tensor
    Q, A = orient_tensor_2D(hist, bins)  # orientation and anisotropy tensors



    # DOCUMENTATION
    assert path.exists(outDir)
    skio.imsave(path.join(outDir, img_fname), img_as_ubyte(img), plugin='tifffile')

    # Discrete probability (histogram) plot to figure
    fig.savefig(path.join(outDir, probplot_fname), dpi=300)

    # Histogram data points to CSV
    histDF = {'phiBins': bins, 'phiHist': np.append(hist, np.nan)}  # nan as last element to match lengths of bins and hist
    histDF = pd.DataFrame(histDF)
    histDF.to_csv(path.join(outDir, prob_fname))

    # General information to text file
    with open(path.join(outDir, txt_fname), 'w+') as f:
        f.write("#FRC properties\n")
        f.write("Specimen dimensions: {} mm\n".format(sp_shape))
        f.write("Specimen thickness: {} mm\n".format(sp_tk))
        f.write("Fibre:\n")
        f.write("\tDiameter: {} mm\n".format(fibdia))
        f.write("\tAspect ratio: {}\n".format(fibAR))
        f.write("\tVolume fraction: {}\n".format(fibdosage))

        f.write("\n\n#Image Properties\n")
        f.write("Image scale: {}\n".format(imscale))
        f.write("Image shape: {}\n".format(img_shp))
        f.write("Fibre:\n")
        f.write("\tDiameter: {} px\n".format(df))
        f.write("\tLength: {}px\n".format(lf))
        f.write("\tNumbers: {}\n".format(Nf))

        f.write("\n\n#Fibre Orientation\n")
        f.write("ODF: {}\n".format(odf))
        f.write("ODF parameters: {}\n".format(parameters))
        f.write("Orientation tensor: {}".format(Q.ravel()))
