"""
Implementation of functions for FT analysis of 2D FRC images.
------------------------------------------------------------------------------------------------
@ Lakshminarayanan Mohana Kumar
22nd July 2021
"""

import numpy as np
# from matplotlib_settings import *
# from matplotlib import pyplot as plt
# import skimage.io as skio
from skimage import img_as_float
from skimage.color import rgb2gray
from skimage.transform import rotate  # EuclideanTransform, warp,
# import skimage.feature as skfeature
# from fiborient import orient_tensor_2D, theo_orient_tensor_2D
from skimage.filters import window
from scipy.fftpack import fft2, fftshift
from itertools import product, combinations


__version__ = '0.1'


def delta(i, j):
    if i==j:
        return 1
    else:
        return 0


def fourier_orient_tensor_2order(image, windowName='hann'):
    if image.ndim > 2:
        image = rgb2gray(image)
    image = img_as_float(image)

    # Fourier Transform
    wimg = image * window(windowName, image.shape)  # windowing the image
    wimgFT = np.abs(fftshift(fft2(wimg)))  # FT

    # Calculation of orientation tensor in Fourier space:
    m, n = wimgFT.shape
    u = np.arange(0, m) - m // 2
    v = np.arange(0, n) - n // 2
    uu, vv = np.meshgrid(u, v)  # all points in Fourier space
    r = np.sqrt(uu ** 2 + vv ** 2)  # radial distance to each point.
    r = np.where(r == 0, 1, r)

    ku = np.divide(uu, r)  # spatial frequency (unit vector component) in u-direction (x)
    kv = np.divide(vv, r)  # spatial frequency (unit vector component) in v-direction (y)
    E = np.sum(wimgFT)  # Total energy in Fourier space (sum of values at all points)

    # elements of the orientation tensor
    Quu = np.sum(ku ** 2 * wimgFT) / E
    Quv = np.sum(ku * kv * wimgFT) / E
    Qvv = np.sum(kv ** 2 * wimgFT) / E

    Q = np.array([[Quu, Quv], [Quv, Qvv]])  # orientation tensor in Fspace

    # estimating the actual orientation tensor from the tensor obtained in Fspace
    # rot = -np.pi / 2  # The axes in Fourier space are rotated by 90 degress w.r.t the axes in original image.
    R = np.array([[0, -1], [1, 0]])

    # transformation of orientation tensor for 90 degrees rotation.
    Q = R @ Q @ R.T  # Q is from Fourier space. Q2 is in original image.
    A = Q - 0.5 * np.eye(2)
    return Q, A


def fourier_orient_tensor_4order(image, windowName='hann'):
    if image.ndim > 2:
        image = rgb2gray(image)
    image = img_as_float(image)

    # Fourier Transform
    wimg = image * window(windowName, image.shape)  # windowing the image
    wimgFT = np.abs(fftshift(fft2(wimg)))  # FT
    wimgFT = rotate(wimgFT, 90, order=3)  # rotating image to match orientation of FT space with real space


    # Calculation of orientation tensor in Fourier space:
    m, n = wimgFT.shape
    u = np.arange(0, m) - m // 2
    v = np.arange(0, n) - n // 2
    uu, vv = np.meshgrid(u, v)  # all points in Fourier space
    r = np.sqrt(uu ** 2 + vv ** 2)  # radial distance to each point.
    r = np.where(r == 0, 1, r)

    k = np.zeros((2, m, n))
    k[0, :, :] = np.divide(uu, r)  # spatial frequency (unit vector component) in u-direction (x)
    k[1, :, :] = np.divide(vv, r)  # spatial frequency (unit vector component) in v-direction (y)
    E = np.sum(wimgFT)  # Total energy in Fourier space (sum of values at all points)

    coords = [(0, 1)]
    order = 4
    base = tuple(coords * order)
    indices = product(*base)

    Q = []
    # elements of the orientation tensor
    for indx in indices:
        elem = wimgFT.copy()
        for i in indx:
            elem = elem * k[i, :, :]
        Q.append(np.sum(elem))

    Q = np.array(Q).reshape((order, order)) / E  # orientation tensor in FT space

    # Anisotropy Tensor
    Q2, A2 = fourier_orient_tensor_2order(image, windowName=windowName)
    A = Q.ravel()

    for itrno, indx in enumerate(indices):
        l = set(indx)
        term1 = 0
        term2 = 0
        for comb in combinations(l, 2):
            rem = tuple(l.difference(comb))
            term1 += delta(comb) * Q2[rem]
            term2 += delta(comb) * delta(*rem)

        A[itrno] = A[itrno] - (term1 / 6) + (term2 / 24)
    A = A.reshape(Q.shape)

    return Q, A
