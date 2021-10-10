"""
Code to test the concept of generating 2D projectional-plane basis functions from 3D basis functions through integration.
Also check the multiplication with anisotropy tensor.
_______________________________________________________________________________________
@ Lakshminarayanan Mohana Kumar
10 Oct 2021
"""

import os
import sys
import numpy as np
sys.path.append('..')
# from glob import glob
from itertools import product
from odffit import basisfunc_3D


dataDir = "../data/art_images"
outDir = "../data/art3D_FT"
imgName = "vf10_ar100_phiuni_thsin"
orientPMFDataFile = imgName + "_projPMFData.npz"

SECONDORDER_SIZE = 3

# Read Data
npzfile = np.load(os.path.join(dataDir, orientPMFDataFile))
projDirsRad, orientHists = npzfile['projDirsRad'], npzfile['orientHists']
print(len(projDirsRad))

nData, nPoints = orientHists.shape
thtVals = np.linspace(0, np.pi, nPoints)
phiVals = np.linspace(0, np.pi, nPoints)

# orienttns = tuple(product(phiVals, thtVals))
F2ndOrder_3D = np.zeros((SECONDORDER_SIZE, SECONDORDER_SIZE, nPoints, nPoints))

uvecs = np.zeros((SECONDORDER_SIZE, nPoints, nPoints))
uvecs[0, :, :] = np.outer(np.cos(phiVals), np.sin(thtVals))
uvecs[1, :, :] = np.outer(np.sin(phiVals), np.sin(thtVals))
uvecs[2, :, :] = np.outer(np.ones(nPoints), np.cos(thtVals))

for i in range(SECONDORDER_SIZE):
    for j in range(SECONDORDER_SIZE):
        F2ndOrder_3D[i, j, :, :] = uvecs[i, :, :] * uvecs[j, :, :]

F2ndOrder_3D_fromdef = basisfunc_3D()

F2ndOrder_2D = np.trapz(F2ndOrder_3D, thtVals)
F2ndOrder_2D_fromdef = np.trapz(F2ndOrder_3D_fromdef, thtVals)
randidx = np.random.randint(0, nPoints)
print(F2ndOrder_2D.shape)
print(F2ndOrder_2D[:, :, randidx])
print(F2ndOrder_2D_fromdef[:, :, randidx])
print("\nAfter transposing: \n")
Ftrans = F2ndOrder_2D.T
print(Ftrans[randidx, :, :])

A = np.arange(9).reshape((3, 3))
res1 = np.einsum('ij, ijk -> k', A, F2ndOrder_2D)
res2 = A @ Ftrans
print(res1[randidx])
print(res2[randidx])

#  Preprocess data if needed
# Learning algorithm -
#   Set up model
#       Basis Functions in respective projection directions.
#   Use Normal Equation?
#   Convex Cost Function
#   Gradient
#   Convex optimization
#   Learning monitoring - learning rates, errors, variance, and bias.
# Results
# Plots
# Accuracy and error estimates
