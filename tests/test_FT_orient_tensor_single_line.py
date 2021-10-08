"""
Testing the estimation of orientation tensor in FT space of images with single line (fibre).
Objectives:
    1. Generate image with single fibre.
    2. Rotate image to obtain single fibre in different orientations
    3. Calculate actual orientation tensor
    4. Calculation orientation tensor in FT space
    5. Compare tensors form 3 and 4.
____________________________________________________________________________________________
@ Lakshminarayanan Mohana Kumar
25th July 2021
"""

import os
from os import path
import numpy as np
import skimage.io as skio
from skimage.transform import rotate
from fibfourier import fourier_orient_tensor
from fiborient import theo_orient_tensor_2D

outDir = "tests_FT_of_lines"
txt_fname = "single_lines_halflen.txt"

if not path.exists(outDir):
    os.mkdir(outDir)

img = np.zeros((50, 50))
img[24:27, 24:] = 1  # image with a straight line along X-axis passing through origin with thickness 3 px
rotations = np.arange(0, 91, 5)

with open(path.join(outDir, txt_fname), 'w+') as f:
    for rot in rotations:
        imgRottd = rotate(img, rot)
        # Theoretical orientation tensor
        Q_theo = theo_orient_tensor_2D(rot)
        A_theo = Q_theo - 0.5 * np.eye(2)
        # FT orientation tensor
        Q_FT, A = fourier_orient_tensor(imgRottd, windowName='hann', order=2)
        print("Tr(Q_FT): ", np.trace(Q_FT))
        print("Det(A) = ", np.linalg.det(A))
        # Q_FT = Q_FT / np.trace(Q_FT)
        # Error
        errQ = np.linalg.norm(Q_FT - Q_theo)
        errA = np.linalg.norm(A - A_theo)

        f.write("Rotation: {} deg\n".format(rot))
        f.write("Theoretical orientation tensor:\n")
        f.write("{}".format(Q_theo))
        f.write("\nFT orientation tensor:\n")
        f.write("{}".format(Q_FT))
        f.write("\nL2-norm error: {}\n".format(errQ))
        f.write("Theoretical anisotropy tensor:\n")
        f.write("{}".format(A_theo))
        f.write("\nFT anisotropy tensor:\n")
        f.write("{}".format(A))
        f.write("\nL2-norm error: {}\n\n".format(errA))

