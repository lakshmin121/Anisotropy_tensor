"""
Popular orientation probability distributions for simulation of fibres in concrete.
"""

import numpy as np
from numpy import random
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

def uniform(domainDeg=(0, 180), size=None):
    phiLow, phiHigh = domainDeg  # angular domain in deg
    if phiLow > phiHigh:
        phiHigh, phiLow = domainDeg  # angular domain in deg
    return random.uniform(low=phiLow, high=phiHigh, size=size)


def vonmises(muDeg=0.0, kappa=1.0, spreadDeg=180, size=None):
    randvals = random.vonmises(mu=0, kappa=kappa, size=size)  # values between [-pi, pi] centred at 0.
    factor = spreadDeg / 360
    phivals = factor * np.rad2deg(randvals) + muDeg
    return phivals


def sin(domainDeg=(-90, 90), symmetric=True, size=None):
    phiLow, phiHigh = domainDeg  # angular domain in deg
    if phiLow > phiHigh:
        phiHigh, phiLow = domainDeg  # angular domain in deg
    if symmetric:
        if size is None:
            sin_(domainDeg, size)
        else:
            m = 0.5 * (phiLow + phiHigh)
            sz1 = size//2
            sz2 = size - sz1
            domain1 = (phiLow, m)
            domain2 = (phiHigh, m)
            phivals1 = sin_(domain1, sz1)
            phivals2 = sin_(domain2, sz2)
            phivals = np.concatenate((phivals1, phivals2))
            return phivals
    else:
        phivals = sin_(domainDeg, size)
        return phivals


def sin_(domainDeg, size=None):
    phiLow, phiHigh = domainDeg  # angular domain in deg
    s = 2 * (phiHigh - phiLow) / np.pi  # scale
    u = random.uniform(0, 1, size=size)
    phivals = s * np.arccos(1 - u) + phiLow
    return phivals


def map_rv2cos(x, pmf_x, y):
    """
    Estimates PMF p(y) for the tranformation y=cos(x) when the PMF p(x) is given.
    :param x: input RV
    :param pmf_x: PMF p(x) defined at the input x values; len(pmf_x) must be equal to len(x)
    :param y: output RV values where the transformed PMF p(y = cos x) must be estimated.
    :return: PMF of transformed RV.
    """
    assert len(x) == len(pmf_x), print("check input: len(x) not equal to len(pmf_x).")
    F = interp1d(x, pmf_x, kind='quadratic', fill_value='extrapolate')
    pmf_y = F(np.arccos(-y)) * (1 / np.sqrt(1 - y**2))
    assert len(pmf_y) == len(y)
    return pmf_y


def map_rv2tan(x, pmf_x, y):
    """
    Estimates PMF p(y) for the tranformation y=tan(x) when the PMF p(x) is given.
    :param x: input RV
    :param pmf_x: PMF p(x) defined at the input x values; len(pmf_x) must be equal to len(x)
    :param y: output RV values where the transformed PMF p(y = cos x) must be estimated.
    :return: PMF of transformed RV.
    """
    assert len(x) == len(pmf_x), print("check input: len(x) not equal to len(pmf_x).")
    F = interp1d(x, pmf_x, kind='quadratic', fill_value='extrapolate')
    pmf_y = F(np.arctan(y)) * (1 / (1 + y**2))
    assert len(pmf_y) == len(y)
    return pmf_y


def map_rv2arctan(x, pmf_x, y):
    """
    Estimates PMF p(y) for the tranformation y=arctan(x) when the PMF p(x) is given.
    """
    assert len(x) == len(pmf_x), print("check input: len(x) not equal to len(pmf_x).")
    F = interp1d(x, pmf_x, kind='quadratic', fill_value='extrapolate')
    pmf_y = F(np.tan(y)) * (1 / np.cos(y))**2
    assert len(pmf_y) == len(y)
    return pmf_y


def product_distr(x, y, z, pmfx, pmfy, atol=1e-4):
    assert len(x) == len(pmfx)
    assert len(y) == len(pmfy)

    fx = interp1d(x, pmfx, kind='quadratic', bounds_error=False, fill_value=(0, 0))
    pmfz = []

    # Masking to avoid zero division error.
    mask = np.isclose(np.abs(y), 0, atol=atol)
    y_mskd = y[~mask]
    pmfy_mskd = pmfy[~mask]

    for k in z:
        p_mskd = pmfy_mskd * fx(k / y_mskd) / np.abs(y_mskd)
        p_mskd[np.abs(p_mskd) < atol] = 0  # removing numerically very small values
        p = np.trapz(p_mskd, y_mskd)  # integration (trapezoidal rule)
        pmfz.append(p)
    return pmfz


def ratio_distr(x, y, z, pmfx, pmfy, atol=1e-4):
    assert len(x) == len(pmfx)
    assert len(y) == len(pmfy)

    fx = interp1d(x, pmfx, kind='quadratic', bounds_error=False, fill_value=(0, 0))
    pmfz = []

    # # Masking to avoid zero division error.
    mask1 = np.isclose(np.abs(y), 0, atol=atol)
    mask2 = np.isclose(np.abs(y), np.pi/2, atol=atol)
    mask = mask1 & mask2
    y_mskd = y[~mask]
    pmfy_mskd = pmfy[~mask]

    for k in z:
        p_mskd = np.abs(y_mskd) * pmfy_mskd * fx(k * y_mskd)
        # p_mskd[np.abs(p_mskd) < atol] = 0  # removing numerically very small values
        p = np.trapz(p_mskd, y_mskd)  # integration (trapezoidal rule)
        pmfz.append(p)
    return pmfz