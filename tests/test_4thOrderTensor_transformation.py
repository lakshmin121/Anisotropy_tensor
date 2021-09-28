"""
Code to test the mathematical relationship derived to capture transformation (rotation of coordinate system) of
4th order tensor.
Objectives:
----------
    1. Reliable calculation of 4th order tensor.
    2. Input / choose the rotation of coordinate system.
    3. Calculate tensor in rotated system using rotated basis vectors.
    4. Calculate tensor in rotated system by transforming the tensor defined in reference system using the transformation
    relationship derived.
    5. A positive match between the same tensor estimated through 2 independent methods (3) and (4) affirms the validity
    of the mathematical relationship derived for transformation of 4th order tensor.
    6. Test for several cases.
------------------------------------------------------------------------------------------------------------------------
@ Lakshminarayanan Mohana Kumar
created: 26th Sep 2021.
"""

import os
import numpy as np
from matplotlib_settings import *
from orientation_probabilities import sin

outDir = "tests_tensorTransformation"
SECONDORDER_SIZE = 3
ANGLE_TICK_LABELS = ["0", "$\\frac{\pi}{6}$", "$\\frac{\pi}{3}$", "$\\frac{\pi}{2}$", "$\\frac{2\pi}{3}$",
                     "$\\frac{5\pi}{6}$", "$\pi$"]

def joint_probability(thetaVals, phiVals, nbins = 36):
    # Joint Probability: p(theta, phi).
    fig = plt.figure(figsize=(3.5, 3), dpi=300)  # see a plot of the joint distribution
    ax = fig.gca()
    jointProb, xEdges, yEdges, histImg = ax.hist2d(phiVals, thetaVals, bins=nbins, density=True)
    ax.set_aspect('equal')
    ax.set_ylabel("$\\theta$")
    ax.set_xlabel("$\phi$")
    step = int(nbins / 6)
    xticks = xEdges[::step]
    yticks = yEdges[::step]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    global ANGLE_TICK_LABELS
    ax.set_yticklabels(ANGLE_TICK_LABELS)
    ax.set_xticklabels(ANGLE_TICK_LABELS)
    plt.colorbar(histImg)

    return jointProb, xEdges, yEdges, fig


def tensor_2nd_order(probThetaPhi, thetaVals, phiVals):
    global SECONDORDER_SIZE
    # unit vectors:
    uvec = np.zeros((3, len(thetaVals), len(phiVals)))
    uvec[0, :, :] = np.outer(np.sin(thetaVals), np.cos(phiVals))
    uvec[1, :, :] = np.outer(np.sin(thetaVals), np.sin(phiVals))
    uvec[2, :, :] = np.outer(np.cos(thetaVals), np.ones(len(phiVals)))

    assert uvec[0].shape == probThetaPhi.shape, \
        print("Shapes of joint probability {0} and unit-vector {1} does not match.".format(uvec[0].shape, p_tht_phi.shape))
    orientTensor = np.zeros((SECONDORDER_SIZE, SECONDORDER_SIZE))
    for i in range(SECONDORDER_SIZE):
        for j in range(SECONDORDER_SIZE):
            orientTensor[i, j] = np.trapz(np.trapz(p_tht_phi * uvec[i] * uvec[j], thetaVals, axis=0), phiVals, axis=-1)
    # print("Orientation Tensor - 2nd order: {}".format(orientTensor))

    anisoTensor = orientTensor - 1/3 * np.eye(SECONDORDER_SIZE)

    return orientTensor, anisoTensor


def cycle_indices(size):  # Tested OK
    all_indices = np.arange(size).astype(np.int)  # indices are integers
    for s in range(size):
        yield tuple(np.roll(all_indices, shift=s))


def rotate_2nd_order(rotAnglesDeg=(0, 0, 0), rotAnglesRad=None):  # Tested OK
    if rotAnglesRad is None:
        rotAnglesRad = np.deg2rad(rotAnglesDeg)

    rotMat = np.eye(3)
    for r, i, j in cycle_indices(3):
        # r is the axis about which rotation is performed.
        rotRad = rotAnglesRad[r]
        if rotRad > 0:
            rCos, rSin = np.cos(rotRad), np.sin(rotRad)
            rotMat_r = np.eye(3)
            rotMat_r[i, i], rotMat_r[i, j], rotMat_r[j, i], rotMat_r[j, j] = rCos, - rSin, rSin, rCos
            rotMat = rotMat @ rotMat_r

    return rotMat


'''
A simple working code for 2nd order tensor in the lead up to testing for 4th order.
A spherical coordinate system with azimuthal angle phi and elevation theta are used to record orientations.
'''

# Probability function - the random input that varies for different test cases.
Nfib = 5000  # number of fibres

# # Domain: (0, pi), (0, pi)
# phiDomain = (0, np.pi)  # Domain of phi
# thtDomain = (0, np.pi)  # Domain of theta
# # phiObs = np.random.rand(Nfib) * np.pi
# # thtObs = np.random.rand(Nfib) * np.pi
# phiObs = np.random.uniform(size=Nfib) * np.pi
# iud = np.random.uniform(size=Nfib)
# thtObs = np.arccos(1 - 2*iud)

# Domain: (0, 2pi), (0, pi/2)
phiDomain = (0, 2*np.pi)  # Domain of phi
thtDomain = (0, np.pi/2)  # Domain of theta
# phiObs = np.random.rand(Nfib) * np.pi
# thtObs = np.random.rand(Nfib) * np.pi
phiObs = np.random.uniform(size=Nfib) * 2*np.pi
iud = np.random.uniform(size=Nfib)
thtObs = np.arccos(1 - iud)

fig_tht_phi, axes = plt.subplots(1, 2, figsize=(6, 2.5), dpi=300)
axs = axes.ravel()
nbins = 36
h, b, _ = axs[0].hist(phiObs, bins=nbins, density=True)
axs[0].set_xlabel("$\phi$")
h, b, _ = axs[1].hist(thtObs, bins=nbins, density=True)
axs[1].set_xlabel("$\\theta$")
for ax in axs:
    ax.set_xticks(b[::int(nbins/6)])
    ax.set_xticklabels(ANGLE_TICK_LABELS)

p_tht_phi, phiEdges, thtEdges, jointProbFig = joint_probability(thtObs, phiObs, nbins=72)
jointProbFig.savefig(os.path.join(outDir, "jointProbabilityHist.tiff"))

thtCentres = 0.5 * (thtEdges[1:] + thtEdges[:-1])
phiCentres = 0.5 * (phiEdges[1:] + phiEdges[:-1])
totalIntegral = np.trapz(np.trapz(p_tht_phi, thtCentres, axis=0), phiCentres, axis=-1)
assert np.isclose(totalIntegral, 1, atol=1e-1), print("Total integral not close to unity: {}".format(totalIntegral))
p_tht_phi = p_tht_phi / totalIntegral  # normalizing
print("Total probability: ", np.trapz(np.trapz(p_tht_phi, thtCentres, axis=0), phiCentres, axis=-1))


# Tensor
orientTensor2nd, anisoTensor2nd = tensor_2nd_order(p_tht_phi, thtCentres, phiCentres)
print("Orientation tensor: \n", orientTensor2nd)
print()
print("Anisotropy tensor: \n", anisoTensor2nd)

# Rotation


# print("Transformed tensor: ", R@A@R.T)

# plt.show()
