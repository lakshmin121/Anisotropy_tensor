"""
Simulation of artificial projected images (2D) of FRC.
__________________________________________________
@ Lakshminarayanan Mohana Kumar
updated: 29th July 2021
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
from matplotlib_settings import *   # custom configurations for plotting.
import orientation_probabilities as op
from fiborient import orient_tensor_2D
import pandas as pd


for m in [0, 90]:
    for k in [0.1, 0.25, 0.5, 1, 5]:
        # 1. INPUT
        outDir = "../data/test_images_2D/vf20_ar50_tk50"
        imgName = 'vm_m{0}k{1}'.format(m, k)
        img_fname = imgName + '.tiff'  # filename of FRC image -> tiff file
        txt_fname = imgName + '.txt'  # filename of FRC image information -> text file
        probplot_fname = imgName + '_prob.tiff'  # file name of FRC orientaion histogram (plot) -> tiff file
        prob_fname = imgName + '_prob.csv'  # file name of FRC orientation histogram data as CSV table

        # Specimen and fibre properties:
        imscale = 20  # px/mm
        sp_shape = (50, 50)  # mm
        sp_tk = 50  # mm
        fibdia = 0.2  # mm
        fibAR = 50  # mm / mm
        fibdosage = 0.002

        # Orietation Distribution Function
        odf = 'vonmises'
        parameters = {'muDeg': m,
                      'kappa': k,
                      'spreadDeg': 180}
        phiDomainDeg = (m-90, m+90)

        # Simulated properties:
        df = int(np.ceil(fibdia * imscale))  # diameter of fibre, px
        lf = int(np.ceil(df * fibAR))  # length of fibre, px
        img_shp = tuple(np.array(np.asarray(sp_shape) * imscale, dtype=int))  # shape of projected image, px
        fibvol = 0.25 * np.pi * fibAR * fibdia**3  # volume of single fibre in physical units, mm^3
        Nf = int(fibdosage * np.prod(np.asarray(sp_shape)) * sp_tk / fibvol)  # number of fibres
        print("Number of fibres: ", Nf)


        # 2. GENERATION OF FIBRES


        def generate_fibre_orientations(odf='vonmises', size=1, **kwargs):
            """
            Local function to choose between vonmises and uniform distributions of fibre orientation.
            :param odf: 'vonmises' or 'uniform'.
            :param size: number of samples to be generated = number of fibres.
            :param kwargs: other parameters relevant to the chosen odf.
            :return: 1-D numpy array of samples drawn from the given odf with parameters set using kwargs.
            """
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
        phiMin, phiMax = phiDomainDeg  # phi domain

        # Fibre Locations
        # Generating (x, y) locations (centroids) of fibres as a uniform random distribution.
        xvals, yvals = tuple(np.random.uniform(low=0, high=img_shp[dim], size=Nf) for dim in range(len(img_shp)))
        spcentre = np.asarray(img_shp)//2  # centre of the specimen in image units

        # Fibres:
        # Generating individual fibres with attributes such as location, length, orientation etc.
        # A custom class named Fibre is used to store attributes for each fibre. Each fibre is an object of the class.
        fibs = [Fibre(df, lf, (x, y), phiDegxy=phi, spcentre=spcentre) for x, y, phi in zip(xvals, yvals, phivals)]


        # 3. IMAGE
        # Generation of the image
        img = np.zeros(img_shp)  # initializing blank image (appears fully black).


        def insert_fibres(image, fibres, kind='line', cval=1, **lineargs):
            """
            Local function to insert individual fibres into the image. For a set of fibres, each fibre is
            inserted iteratively. Each fibre must be an object of class Fibre. The fibre is drawn as a line
            or a bezier curve (future use for curved fibres) of unit pixel thickness.
            :param image: Image in which fibres must be drawn.
            :param fibres: Iterable collection of fibres (objects of class Fibre).
            :param kind: 'line' or 'bezier'
            :param cval: Image intensity value for fibre lines.
            :type cval: int; value in the range 0-255.
            :param lineargs: additional kwargs to be used with 'bezier' curve.
            :return: image populated with fibres as lines of unit pixel thickness.
            """
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
        # unit pixel lines in image are dilated to make them one fibre-diameter thick lines.
        rf = df // 2
        img = dilation(img, selem=np.ones((rf, rf)))  # dilation is a morphological operator.

        # ODF plot (histogram)
        fig = plt.figure()
        ax = fig.gca()
        if phiMin < 0:
            # phiDomainDeg = np.asarray(phiDomainDeg) - phiMin  # setting domain to (0, 180).
            phivals = np.where(phivals < 0, phivals + 180, phivals)
            phimin, phimax = np.min(phivals), np.max(phivals)
            assert phimin >= 0 and phimax <= 180, print("Domain of phi: {}".format((phimin, phimax)))
            # phiMin, phiMax = phiDomainDeg
        phiBins = np.arange(0, 181, 10)
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
        # nan as last element to match lengths of bins and hist
        histDF = {'phiBins': phiBins, 'phiHist': np.append(hist, np.nan)}
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
