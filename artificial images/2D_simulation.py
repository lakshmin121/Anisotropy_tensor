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
from skimage import img_as_float, img_as_ubyte
from fibre2D import Fibre
from skimage.draw import line, bezier_curve
from skimage.morphology import dilation
from matplotlib import pyplot as plt
import orientation_probabilities as op

# Say the probability distribution of phi is given.
# Generate a set of lines representing fibres.
# We can consider equal amounts of fibres in equal layers.
# We can consider an relative attenuation value for concrete.
# We can adjust fibre length based on distribution of theta.

outDir = "../data/art_images"
fname = 'test_2fib.tiff'

# Generating a single fibre of given properties.
imscale = 20 # px/mm
sp_shape = (50, 50)  # mm
sp_tk = 50  # mm
fibdia = 0.2  # mm
fibAR = 100  # mm / mm
fibdosage = 0.002

# single fibre properties:
df = int(np.ceil(fibdia * imscale))  # px
lf = int(np.ceil(df * fibAR))  # px
img_shp = tuple(np.array(np.asarray(sp_shape) * imscale, dtype=int))  # px
fibvol = 0.25 * np.pi * fibAR * fibdia**3
Nf = int(fibdosage * np.prod(np.asarray(sp_shape)) * sp_tk / fibvol)  # number of fibres

print("Number of fibres: ", Nf)

phivals = op.vonmises(kappa=5, size=Nf)
# phivals = op.uniform(size=Nf)
xvals = np.random.uniform(low=0, high=img_shp[0], size=Nf)
yvals = np.random.uniform(low=0, high=img_shp[1], size=Nf)
spcentre = np.asarray(img_shp)//2

img = np.zeros(img_shp)
fibs = [Fibre(df, lf, (x, y), phiDegxy=phi, spcentre=spcentre) for x, y, phi in zip(xvals, yvals, phivals)]


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
rf = df // 2 + 1
img = dilation(img, selem=np.ones((rf, rf)))

f = plt.figure()
ax = f.gca()
ax.imshow(img, cmap='gray')
# ax.plot(spcentre[1], spcentre[0], 'go')
# ax.plot(N[1], N[0], 'r*')
plt.show()
#
# assert path.exists(outDir)
# skio.imsave(path.join(outDir, fname), img_as_ubyte(img), plugin='tifffile')
