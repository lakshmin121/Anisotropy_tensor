"""
Implementation of functions for FT analysis of 2D FRC images.
------------------------------------------------------------------------------------------------
@ Lakshminarayanan Mohana Kumar
22nd July 2021
"""

import time
import numpy as np
from skimage import img_as_float
from skimage.color import rgb2gray
from skimage.transform import warp_polar, rotate
from multiprocessing import cpu_count, Pool
from joblib import Parallel, delayed
from skimage.filters import window
from scipy.ndimage.filters import gaussian_filter1d
from scipy.fftpack import fft2, fftshift
from itertools import product, combinations
from functools import partial
from fiborient import orient_tensor_2D, tensor2odf_2D


__version__ = '0.3'

# TODO: To change: fourier_orient_tensor
# TODO: Obsolete: sliding_window, localised_fourier_orient_tensor.
# TODO: To delete: fourier_orient_tensor_2order

# List of functions used in other scripts: imgfft, fourier_orient_tensor, localised_fourier_orient_tensor

def delta(i, j):
    """
    Kronecker delta function.
    :return: delta(i, j) = { 1, when i==j,
                             0, when i~=j.
    where i and j are integer indices.
    """
    if i == j:
        return 1
    else:
        return 0


def imgfft(img, windowName=None, zpad_factor=1):
    """
    Spectrum of an image.
    :param img: Input image
    :type img: ndarray
    :param windowName: Name of window to be used. Preferred 'hann' or 'blackmanharris'
    :param zpad_factor: Input image shape multiplied by zpad_factor gives size of zero-padded image for FT.
    :type zpad_factor: float
    :return: spectrum (magnitude) of input image. Output is an ndarray with shape = input image shape * zpad_factor.
    Notes:
    1. Windowing minimizes the effects on FT due to the use of a finite signal (image). FT assumes the data to
    represent a full cycle such that the image repeats itself beyond the boundaries. Windowing smoothly
    reduce the image intensity towards the boundaries of the image. Heuristically, the windows apt for the specific
    purpose of this function are 'hann' and 'blackmanharris'. Refer: https://en.wikipedia.org/wiki/Window_function
    2. Zero-padding is the process of padding the image with zeros beyond its boundaries. This allows the calculation
    of more Fourier coefficients, thus, a smoother frequency spectrum. For the purpose of this function, consider this
    as a better resolution of the image in FT space.
    Refer: https://dspillustrations.com/pages/posts/misc/spectral-leakage-zero-padding-and-frequency-resolution.html
    """
    if img.ndim > 2:  # if color image, convert to gray.
        img = rgb2gray(img)
    img = img_as_float(img)

    # Fourier Transform
    if windowName is not None:
        # windowed and zero-padded FT
        shp = img.shape  # image shape as tuple
        shparr = np.array(shp)  # shape as array
        zpad_shp = np.round(shparr * zpad_factor, 0).astype(np.int)  # zero-padded shape of imager
        wimg = img * window(windowName, img.shape)  # windowing the image
        wimgFT = np.absolute(fftshift(fft2(wimg, tuple(zpad_shp))))  # FT of windowed and zero padded image.
    else:
        # windowless FT
        wimgFT = np.absolute(fftshift(fft2(img)))  # FT
    return wimgFT


def sliding_patches(img, x_width, y_width, x_step=1, y_step=1): # TODO: change to patch
    """
    Returns an iterator containing the location and image captured by a sliding window - patch.
    :param img: original image across which the window has to slide.
    :param x_width: width of sliding window in x-direction.
    :param y_width: width of sliding window in y-direction.
    :param x_step: distance by which window slides in x-direction.
    :param y_step: distance by which window slides in y-direction.
    :return: x, y, sub-image captured by the sliding window,
    where x, y are location (centre) of the sliding window.
    Note:
    Here window refers to a local patch of image selected. It is different from window used in signal
    processing, particularly Fourier Transform.
    """
    m, n = img.shape    # size of original image
    x0 = x_width // 2   # border thickness = 1/2 of window width
    y0 = y_width // 2

    img_bordered = np.zeros((m + x_width, n + y_width))  # image with border
    img_bordered[x0:x0+m, y0:y0+n] = img  # central region of bordered image = original image.
    # currently border region is black (zero values).

    # Border boundary condition = reflection
    img_bordered[0:x0, :] = img_bordered[x0:2*x0, :][::-1, :]               # Left border
    img_bordered[m+x0:m+2*x0, :]    = img_bordered[m:m+x0, :][::-1, :]      # right border
    img_bordered[:, 0:y0]           = img_bordered[:, y0:2*y0][:, ::-1]     # bottom border
    img_bordered[:, n+y0:n + 2*y0]  = img_bordered[:, n:n + y0][:, ::-1]    # top border

    # ########################################### WRONG ###########################################
    # for x in range(x0, m+x0, x_step):  # x-coordinate of sliding window centre
    #     for y in range(y0, n+y0, y_step):  # y-coordinate of sliding window centre
    #         yield x-x0, y-y0, img_bordered[x:x + x_width, y:y + y_width]
    # ########################################### #### ###########################################
    for x in range(0, m, x_step):  # x-coordinate of sliding window centre
        for y in range(0, n, y_step):  # y-coordinate of sliding window centre
            yield x, y, img_bordered[x:x + x_width, y:y + y_width]


# TODO: update sliding window to patch
def localised_fourier_orient_tensor(img, x_width, y_width, x_step=1, y_step=1, windowName=None, order=2):
    """
    Applies FT to a sub-image selected by a sliding window - patch and calculates orientation tensor
    from the spectrum, repeated by sliding across the entire image.
    :param func: function to be applied
    :param img: input image
    :param x_width: width of sliding window in x-direction
    :param y_width: width of sliding window in y-direction
    :param x_step: distance by which window slides in x-direction.
    :param y_step: distance by which window slides in y-direction.
    :param apply_median_filter: whether to smooth the output using a median filter.
    :return: ndarray (image) resulting from operation of func on img.
    """
    def func2(XYwindow):
        x, y, window = XYwindow
        return fourier_orient_tensor(window, windowName=windowName, order=order)

    parallelStart = time.time()
    num_cores = cpu_count() - 1
    processed_list = Parallel(n_jobs=num_cores, prefer='threads')(delayed(func2)(xywindow)
                                                                  for xywindow in sliding_patches(img,
                                                                                                  x_width,
                                                                                                  y_width,
                                                                                                  x_step,
                                                                                                  y_step)
                                                                  )
    print("Parallel operation: elapsed time: ", time.time() - parallelStart)

    opStart = time.time()
    # print(len(processed_list))
    Qitems = [item[0] for item in processed_list]
    # Aitems = [item[1] for item in processed_list]
    Q = np.dstack(Qitems)
    # A = np.dstack(Aitems)
    Q = np.nan_to_num(Q, nan=0)
    Q = np.mean(Q, axis=-1)
    Q = Q / np.trace(Q)
    if order == 2:
        A = Q - 0.5 * np.eye(order)
    elif order == 4:
        Q2, A2 = localised_fourier_orient_tensor(img,  x_width=x_width, y_width=y_width,
                                                 x_step=x_step, y_step=y_step,
                                                 windowName=windowName, order=2)
        A = np.copy(Q).ravel()

        # setup
        coords = (0, 1)  # possible coordinates in 2D space
        base = tuple([coords] * order)  # tensor space dimension = coords * order
        indices = list(product(*base))  # all possible tensor indices Qijkl

        for itrno, indx in enumerate(indices):
            s = set(range(4))
            term1 = 0
            term2 = 0
            for comb in combinations(s, 2):
                i, j = tuple(indx[m] for m in comb)
                k, l = tuple(indx[m] for m in s.difference(set(comb)))
                # print("i, j, k, l: ", i, j, k, l)
                term1 += delta(i, j) * Q2[k, l]
                term2 += delta(i, j) * delta(k, l)
            A[itrno] = A[itrno] - (term1 / 6) + (term2 / 48)
        A = A.reshape(Q.shape)
    else:
        raise NotImplementedError
    print("Local operation: elapsed time: ", time.time() - opStart)
    return Q, A


# TODO: To be removed
# def fourier_orient_tensor_2order(image, windowName=None, zpad_factor=1):
#     """
#     Estimation of orientation and anisotropy tensors from the FT of image.
#     :param image:
#     :param windowName:
#     :param zpad_factor:
#     :return:
#     """
#     if image.ndim > 2:
#         image = rgb2gray(image)
#     image = img_as_float(image)
#
#     # Fourier Transform
#     wimgFT = imgfft(image, windowName=windowName, zpad_factor=zpad_factor)
#     # wimgFT -= np.mean(wimgFT)  # removing mean
#
#     # Calculation of orientation tensor in Fourier space:
#     m, n = wimgFT.shape
#     u = np.arange(0, m) - m // 2
#     v = np.arange(0, n) - n // 2
#     uu, vv = np.meshgrid(u, v)  # all points in Fourier space
#     r = np.sqrt(uu ** 2 + vv ** 2)  # radial distance to each point.
#     r = np.where(r == 0, 1, r)
#
#     ku = np.divide(uu, r)  # spatial frequency (unit vector component) in u-direction (x)
#     kv = np.divide(vv, r)  # spatial frequency (unit vector component) in v-direction (y)
#     E = np.sum(wimgFT)  # Total energy in Fourier space (sum of values at all points)
#
#     # elements of the orientation tensor
#     Quu = np.sum(ku ** 2 * wimgFT) / E
#     Quv = np.sum(ku * kv * wimgFT) / E
#     Qvv = np.sum(kv ** 2 * wimgFT) / E
#
#     Q = np.array([[Quu, Quv], [Quv, Qvv]])  # orientation tensor in Fspace
#
#     # estimating the actual orientation tensor from the tensor obtained in Fspace
#     # rot = -np.pi / 2  # The axes in Fourier space are rotated by 90 degress w.r.t the axes in original image.
#     R = np.array([[0, -1], [1, 0]])
#
#     # transformation of orientation tensor for 90 degrees rotation.
#     Q = R @ Q @ R.T  # Q is from Fourier space. Q2_theo is in original image.
#     A = Q - 0.5 * np.eye(2)
#     return Q, A


def confined_energy(imageFT, xx, yy, xc, yc, r):
    mask = (xx - xc) ** 2 + (yy - yc) ** 2 <= r ** 2
    return np.sum(imageFT * mask).item()


def FTbound(imageFT, step=1):
    imageFT = np.nan_to_num(imageFT, 0)
    xc, yc = np.asarray(imageFT.shape) // 2
    s = min(*imageFT.shape)
    xv = np.arange(0, s)
    yv = xv
    xx, yy = np.meshgrid(xv, yv)
    rvals = np.arange(0, s // 2, step)

    apool = Pool(processes=cpu_count() - 2)
    area_vals = apool.map(partial(confined_energy, imageFT, xx, yy, xc, yc), rvals)
    area_vals = np.array(area_vals)
    # print(area_vals.shape)
    # print(rvals.shape)
    return rvals, np.array(area_vals, dtype=np.float)


def energy_boundary(imageFT, step=1):
    rvals, energy = FTbound(imageFT, step=step)
    grad = np.gradient(energy)
    grad = gaussian_filter1d(grad, step / 4)  # smoothens the gradVals values to a continuous func.
    gradmax_idx = np.argmax(grad)  # location of maximum gradVals.
    ridx = gradmax_idx + np.argmin(np.abs(grad[gradmax_idx:]))

    return ridx, rvals, energy, grad


# TODO: Test 90 deg shift and 180 deg reflection of the histogram.
def polarFT_hist(imageFT, rmin=None, rmax=None, correction=True):
    if rmin is None:
        rmin = 0
    if rmax is None:
        rmax = int(np.min(imageFT.shape) // 2)
    imageFTpolar = warp_polar(imageFT, radius=rmax)
    imageFTpolar = imageFTpolar.T
    hist = np.sum(imageFTpolar[rmin:], axis=0)  # Radial sum
    m = len(hist) // 2
    hist = hist[:m] + hist[m:]
    histsum = np.sum(hist)
    hist = hist / histsum

    # Correction
    if correction:
        excess = np.sum(imageFT[:rmin + 1])
        hist = hist * (1 + excess / histsum)  # approximate
        hist = hist / np.sum(hist)
    hist = np.roll(hist, -90)
    return hist, imageFTpolar


# TODO: include the effect of rmin, rmax? change estimation of wimgFT
# TODO: Add docstring.
def fourier_orient_hist(image, windowName=None, zpad_factor=1, fibdia=1):
    """
    Estimates fibre orientation histogram from FT of projected image.
    :param image:
    :param windowName:
    :param zpad_factor:
    :param fibdia:
    :return:
    """
    if image.ndim > 2:
        image = rgb2gray(image)
    image = img_as_float(image)

    # Fourier Transform
    wimgFT = imgfft(image, windowName=windowName, zpad_factor=zpad_factor)
    wimgFT -= np.mean(wimgFT)  # removing the average (similar to thresholding).
    wimgFTlog = np.nan_to_num(np.log(1 + wimgFT), 0)  # spectrum in log scale + convert any nan generated to zero.

    # Bounds in FT space as a minimization problem
    rindx, rVals, energyVals, gradVals = energy_boundary(wimgFTlog, step=zpad_factor * 5)
    rBound = rVals[rindx]

    # Polar warp
    rmin = min([int(fibdia * zpad_factor), int(round(np.pi * rBound / (50 * zpad_factor)))])
    phiHist, imgzFTpolar = polarFT_hist(wimgFT, rmin=rmin, rmax=rBound)
    phiBins = np.arange(0, 181, 1)

    return phiHist, phiBins


def fourier_orient_tensor(image, windowName=None, order=2, zpad_factor=1, fibdia=1):
    phiHistp, phiBinsp = fourier_orient_hist(image, windowName=windowName,
                                             zpad_factor=zpad_factor, fibdia=fibdia)
    phiBinspc = 0.5 * (phiBinsp[1:] + phiBinsp[:-1])

    # Tensor from FT
    Qphi, Aphi = orient_tensor_2D(phiHistp, phiBinspc, order=order)
    # phiODF2 = tensor2odf_2D(phiBinspc, Aphi) * 2 * np.pi / 180
    # # dphip = np.mean(phiBinsp[1:] - phiBinsp[:-1])
    # print("Check: Total probability from FT ODF (phi) = ", np.trapz(phiODF2, phiBinspc))

    return Qphi, Aphi
