"""
Simulation of artificial projected images of FRC.
Simulation includse straight fibres of given geometric properties and 3D orientation state. Concrete and pores are also
added in the simulation. Output in the form of netCDF file (format can be changed. Default: netCDF3_CLASSIC).
__________________________________________________
@ Lakshminarayanan Mohana Kumar
Updated: 1st Aug 2021
"""

import os
from os import path
import time
import numpy as np
from joblib import Parallel, delayed
from fibre3D import Fibre
from skimage.draw import line_nd
from skimage.morphology import dilation, closing
from skimage.filters import gaussian, median
from matplotlib import pyplot as plt
import orientation_probabilities as op
from skimage.morphology import ball
from PyImModules.NCData import NCData
from skimage import img_as_ubyte, img_as_float


ostart = time.time()

# 1. INPUT
outDir = "../data/art_images"
fname = 'test1_3D_img1.nc'

# Specimen and fibre properties:
imscale = 10  # px/mm
sp_shape = (50, 50, 50)  # (z, y, x) mm
fibdia = 0.3  # mm
fibAR = 50  # aspect ratio, mm / mm
fibdosage = 0.0005

# Orietation Distribution Function
# phi
phiODF = 'uniform'
phi_parameters = {}
# phiODF = 'vonmises'
# m = 0
# k = 1
# phi_parameters = {'muDeg': m,
#               'kappa': k,
#               'spreadDeg': 180}
# phiDomainDeg = (m-90, m+90)

# theta
thetaODF = 'sin'
thetaDomainDeg = (0, 180)
theta_parameters = {'domainDeg': thetaDomainDeg}

# Grayscale values
steel = 90
concrete = 50
pores = 10


# Simulated properties:
df = int(np.ceil(fibdia * imscale))  # diameter of fibre, px
lf = int(np.ceil(df * fibAR))  # length of fibre, px
border = int(0.125 * lf)  # image border = 12.5% of fibre length on each edge. This border will be cropped at the end.
img_shp = tuple(np.array(np.asarray(sp_shape) * imscale, dtype=int) + 2*border)  # shape of 3D specimen image, px
fibvol = 0.25 * np.pi * fibAR * fibdia**3  # volume of single fibre in physical units, mm^3
Nf = int(fibdosage * np.prod(np.asarray(sp_shape)) / fibvol)  # number of fibres
print("Number of fibres: ", Nf)

# --------------------------------------------------------------------------------------------------------------
# 2.  GENERATION OF FIBRES


def generate_fibre_orientations(odf='vonmises', size=1, **odfprops):
    """
    Local function to choose between vonmises and uniform distributions of fibre orientation.
    :param odf: 'vonmises' or 'uniform'.
    :param size: number of samples to be generated = number of fibres.
    :param odfprops: other parameters relevant to the chosen odf.
    :return: 1-D numpy array of samples drawn from the given odf with parameters set using kwargs.
    """
    if odf == 'vonmises':
        try:
            kappa = odfprops['kappa']
        except KeyError as e:
            raise e("Require value of kappa.")
        phi_vals = op.vonmises(size=size, **odfprops)
    elif odf == 'uniform':
        try:
            phidomainDeg = odfprops['domainDeg']
            phi_vals = op.uniform(domainDeg=phidomainDeg, size=size)
        except KeyError:
            print("using default domain (0, 180) degrees for uniform distribution.")
            phi_vals = op.uniform(size=size)
    elif odf== 'sin':
        phi_vals = op.sin(size=size, **odfprops)
    else:
        raise ValueError("Invalid odf: {}".format(odf))
    return phi_vals


# Fibre Orientation
# Generating (x, y) locations (centroids) of fibres as a uniform random distribution.
phivals = generate_fibre_orientations(phiODF, size=Nf, **phi_parameters)
thetavals = generate_fibre_orientations(thetaODF, size=Nf, **theta_parameters)

# Fibre Locations
xvals, yvals, zvals = tuple(np.random.uniform(low=0, high=img_shp[dim], size=Nf) for dim in range(len(img_shp)))
spcentre = np.asarray(img_shp)//2  # centre of the specimen in image units


# Fibres:
# Generating individual fibres with attributes such as location, length, orientation etc.
# A custom class named Fibre is used to store attributes for each fibre. Each fibre is an object of the class.
fibs = [Fibre(df, lf, (z, y, x), phiDegxy=phi, thetaDegxy=theta, spcentre=spcentre)
        for z, y, x, phi, theta in zip(zvals, yvals, xvals, phivals, thetavals)]
tstop = time.time()
print("Time taken to generate fibres: {} s".format(round(tstop - ostart, 0)))


# 3. IMAGE
# Generation of the image
img = np.zeros(img_shp, dtype=np.uint8)

def insert_fibres(image, fibres, cval=1, **lineargs):
    """
    Local function to insert individual fibres into the image. For a set of fibres, each fibre is
    inserted iteratively. Each fibre must be an object of class Fibre.
    :param image: Image in which fibres must be drawn.
    :param fibres: Iterable collection of fibres (objects of class Fibre).
    :param cval: Image intensity value (constant) for fibre lines.
    :type cval: int; value in the range 0-255.
    :param lineargs: additional kwargs
    :return: image populated with fibres as lines of unit pixel thickness.
    """
    for fibre in fibres:
        linecoords = line_nd(fibre.start, fibre.stop, **lineargs)
        image[linecoords] = image[linecoords] + cval


# Insert fibres
insert_fibres(img, fibs)

# Check intersections:
no_of_intersections = np.sum(img > 1)
print("Number of intersections = ", no_of_intersections)
t2stop = time.time()
print("Time taken to insert fibres: {} s".format(round(t2stop - tstop, 0)))

# Impart fibre thickness
# unit pixel lines in image are dilated to make them one fibre-diameter thick lines.
rf = df // 2
img = dilation(img, selem=ball(rf))

# check contacts:
no_of_contacts = np.sum(img > 1)
img = img > 0  # converting to binary
print("Number of contacts = ", no_of_contacts)
t3stop = time.time()
print("Time taken to dilate fibres: {} s".format(round(t3stop - t2stop, 0)))

# # --------------------------------------------------------------------------------------------------------------
# PORES
# pore size distribution adopted from:
# Ríos, J.D., Leiva, C., Ariza, M.P., Seitl, S. and Cifuentes, H., 2019. Analysis of the tensile fracture properties of
# ultra-high-strength fiber-reinforced concrete with different types of steel fibers by X-ray tomography.
# Materials & Design, 165, p.107582.
pore_dia = np.arange(0.5, 3.1, 0.25)  # pore diameters at 0.25 mm interval, mm
db_vals = pore_dia * imscale  # pore diameter in image units, px
rel_freq = np.array([0.0765, 0.1600, 0.1718, 0.1311, 0.1116, 0.0939, 0.0602, 0.0744, 0.0638, 0.0531, 0.0035])
total_pore_volfrac = 0.08  # 10%
rel_vol = rel_freq * total_pore_volfrac * np.prod(np.asarray(sp_shape))  # relative volume of pores of a specific dia.
print(rel_vol)
num_pores = np.array([int(vol / (np.pi*db**3 / 6)) for vol, db in zip(rel_vol, pore_dia)]) # converting rel. vol to num.
Np = np.sum(num_pores)  # Total number of pores.
print("number of pores: ", Np)
locs = tuple(np.random.randint(low=border, high=img_shp[dim]-border, size=Np) for dim in range(len(img_shp)))


def insert_air_bubbles(image, diameters, numbers, locations):
    """
    Local function to insert spherical pores of given diameters in pixels.
    :param image: Image to insert pores
    :param diameters: Iterable list of distinct pore diameters.
    :param numbers: number of pores of each distinct diameter.
    :param locations: locations of pores (z, y, x) image coordinates.
    :return:
    """
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


# # --------------------------------------------------------------------------------------------------------------
# FINAL
# Combining fibres, concrete, and pores.
img = img_as_ubyte(gaussian(img, sigma=rf))  # blurring the fibres to add effect of noise.
img = (img / np.max(img) * (steel - concrete) + concrete).astype(np.uint8)  # adding fibre and concrete grayscale values.
img = img - pore_img * img  # masking the locations of pores
img = img + pore_img * pores  # adding grayscale values of pores.
# print(np.min(img), np.max(img))

# Writing to .nc file.
if not path.exists(outDir):
    os.mkdir(outDir)
# Border of the image is cropped before writing.
ncdata = NCData(img[border:img_shp[0]-border, border:img_shp[1]-border, border:img_shp[2]-border], vartype='i1')
print(ncdata)  # print properties of the .nc file being written.
ncdata.write_ncfile(path.join(outDir, fname))  # write file.
print("Time taken to write .nc file: {} s".format(round(time.time() - t3stop, 0)))

print("Overall time taken: {} s".format(round(time.time() - ostart, 0)))
