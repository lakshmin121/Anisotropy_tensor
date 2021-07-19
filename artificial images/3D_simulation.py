"""
Simulation of artificial projected images of FRC.
__________________________________________________
@ Lakshminarayanan Mohana Kumar
16th July 2021
"""
import os
import sys
cwd = os.getcwd()
print(cwd)
sys.path.append(os.getcwd())

from os import path
import time
import numpy as np
from joblib import Parallel, delayed
from fibre3D import Fibre
from skimage.draw import line_nd
from skimage.morphology import dilation
from matplotlib import pyplot as plt
import orientation_probabilities as op
from skimage.morphology import ball
from PyImModules.NCData import NCData
from skimage.feature import blob_dog

# job details
ostart = time.time()
# outDir = path.join(cwd, "data", "art_images")
outDir = "../data/art_images"
fname = 'test1_3D_img1.nc'

# INPUT
# Generating a single fibre of given properties.
imscale = 10  # px/mm
sp_shape = (50, 50, 50)  # (z, y, x) mm
fibdia = 0.2  # mm
fiblen = 20  # mm
fibdosage = 0.0005

# single fibre properties:
df = int(np.ceil(fibdia * imscale))  # px
lf = int(np.ceil(fiblen * imscale))  # px
fibAR = lf / df
border = int(0.125 * imscale)
img_shp = tuple(np.array(np.asarray(sp_shape) * imscale, dtype=np.int) + 2*border)  # px
fibvol = 0.25 * np.pi * fiblen * fibdia**2
Nf = int(fibdosage * np.prod(np.asarray(sp_shape)) / fibvol)  # number of fibres

print("Number of fibres: ", Nf)

#  PROBABILITY DEFINITIONS
# phivals = op.vonmises(kappa=5, size=Nf)
phivals = op.uniform(size=Nf)
thetavals = op.sin((0, 180), symmetric=True, size=Nf)
xvals, yvals, zvals = tuple(np.random.uniform(low=0, high=img_shp[dim], size=Nf) for dim in range(len(img_shp)))
spcentre = np.asarray(img_shp)//2

# --------------------------------------------------------------------------------------------------------------
# FIBRES
img = np.zeros(img_shp, dtype=np.int8)
fibs = [Fibre(df, lf, (z, y, x), phiDegxy=phi, thetaDegxy=theta, spcentre=spcentre)
        for z, y, x, phi, theta in zip(zvals, yvals, xvals, phivals, thetavals)]
tstop = time.time()
print("Time taken to generate fibres: {} s".format(round(tstop - ostart, 0)))


def insert_fibres(image, fibres, cval=1, **lineargs):
    try:
        assert len(cval) == len(fibres), "len(cval) does not match len(fibres)."
    except TypeError:
        cval = [cval] * len(fibres)

    for fibre, c in zip(fibres, cval):
        linecoords = line_nd(fibre.start, fibre.stop, **lineargs)
        image[linecoords] = image[linecoords] + c

# Insert fibres
# processed_list = Parallel(n_jobs=-1, prefer='threads')(delayed(insert_fibre)(img, fib) for fib in fibs)
fib_attn = np.random.binomial(n=100, p=0.7, size=Nf)
print(np.min(fib_attn), np.max(fib_attn))
insert_fibres(img, fibs, cval=fib_attn)


# check intersections:
no_of_intersections = np.sum(img > 1)
print("Number of intersections = ", no_of_intersections)
t2stop = time.time()
print("Time taken to insert fibres: {} s".format(round(t2stop - tstop, 0)))

# Impart fibre thickness
rf = df // 2
img = dilation(img, selem=ball(rf))

# check contacts:
no_of_contacts = np.sum(img > 1)
print("Number of contacts = ", no_of_contacts)
t3stop = time.time()
print("Time taken to dilate fibres: {} s".format(round(t3stop - t2stop, 0)))

# --------------------------------------------------------------------------------------------------------------
# PORES
pore_dia = np.arange(0.5, 3.1, 0.25)
db_vals = pore_dia * imscale
rel_freq = np.array([0.0765, 0.1600, 0.1718, 0.1311, 0.1116, 0.0939, 0.0602, 0.0744, 0.0638, 0.0531, 0.0035])
total_pore_volfrac = 0.08  # 10%
rel_vol = rel_freq * total_pore_volfrac * np.prod(np.asarray(sp_shape))
print(rel_vol)
num_pores = np.array([int(vol / (np.pi*db**3 / 6)) for vol, db in zip(rel_vol, pore_dia)])
Np = np.sum(num_pores)
print("number of pores: ", Np)
locs = tuple(np.random.randint(low=border, high=img_shp[dim]-border, size=Np) for dim in range(len(img_shp)))


def insert_air_bubble(image, diameter, location):
    z, y, x = location
    r = int(diameter) // 2
    bub = ball(r)
    bub_shp = bub.shape
    slc_z = slice(z - r, z - r + bub_shp[0], None)
    slc_y = slice(y - r, y - r + bub_shp[1], None)
    slc_x = slice(x - r, x - r + bub_shp[2], None)
    image[slc_z, slc_y, slc_x] = bub


def insert_air_bubbles(image, diameters, numbers, locations):
    z, y, x = locations
    counter = 0
    for idia, dia in enumerate(diameters):
        r = int(dia) // 2
        bub = ball(r)
        bub_shp = bub.shape
        for num in range(numbers[idia]):
            try:
                slc_z = slice(z[counter] - r, z[counter] - r + bub_shp[0], None)
                slc_y = slice(y[counter] - r, y[counter] - r + bub_shp[1], None)
                slc_x = slice(x[counter] - r, x[counter] - r + bub_shp[2], None)
                image[slc_z, slc_y, slc_x] = bub
                counter += 1
            except Exception as e:
                print("r: ", r)
                print("slc_y: ", slc_y)
                print("slc_x: ", slc_x)
                print(bub.shape)
                raise e


pore_img = np.zeros(img_shp, dtype=np.int8)
insert_air_bubbles(pore_img, pore_dia, num_pores, locs)

# -------------------------------------------------------------------------------------------------------------
# Concrete
conc_img = np.ones(img_shp, dtype=np.int8)

# --------------------------------------------------------------------------------------------------------------
# FINAL
img_inv = img == 0
img = img + img_inv.astype(np.int8) * pore_img * 12
img_inv = img == 0
img = img + img_inv.astype(np.int8) * conc_img * 50
# img = img.astype(np.int8)

if not path.exists(outDir):
    os.mkdir(outDir)
ncdata = NCData(img[border:img_shp[0]-border, border:img_shp[1]-border, border:img_shp[2]-border] * 3)
print(ncdata)
ncdata.write_ncfile(path.join(outDir, fname))
print("Time taken to write .nc file: {} s".format(round(time.time() - t3stop, 0)))

print("Overall time taken: {} s".format(round(time.time() - ostart, 0)))
