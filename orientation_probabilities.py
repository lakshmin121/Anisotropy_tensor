"""
Popular orientation probability distributions for simulation of fibres in concrete.
"""

import numpy as np
from numpy import random


def uniform(phidomainDeg=(0, 180), size=None):
    phiLow, phiHigh = phidomainDeg  # angular domain in deg
    if phiLow > phiHigh:
        phiHigh, phiLow = phidomainDeg  # angular domain in deg
    return random.uniform(low=phiLow, high=phiHigh, size=size)


def vonmises(muDeg=0.0, kappa=1.0, spreadDeg=180, size=None):
    randvals = random.vonmises(mu=0, kappa=kappa, size=size)  # values between [-pi, pi] centred at 0.
    factor = spreadDeg / 360
    phivals = factor * np.rad2deg(randvals) + muDeg
    return phivals


def sin(phidomainDeg=(-90, 90), symmetric=True, size=None):
    phiLow, phiHigh = phidomainDeg  # angular domain in deg
    if phiLow > phiHigh:
        phiHigh, phiLow = phidomainDeg  # angular domain in deg
    if symmetric:
        if size is None:
            sin_(phidomainDeg, size)
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
        phivals = sin_(phidomainDeg, size)
        return phivals


def sin_(phidomainDeg, size=None):
    phiLow, phiHigh = phidomainDeg  # angular domain in deg
    s = 2 * (phiHigh - phiLow) / np.pi  # scale
    u = random.uniform(0, 1, size=size)
    phivals = s * np.arccos(1 - u) + phiLow
    return phivals
