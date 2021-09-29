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
ANGLE_TICK_LABELS_PI = ["0", "$\\frac{\pi}{6}$", "$\\frac{\pi}{3}$", "$\\frac{\pi}{2}$",
                        "$\\frac{2\pi}{3}$", "$\\frac{5\pi}{6}$", "$\pi$"]
ANGLE_TICK_LABELS_2PI = ["0", "$\\frac{\pi}{6}$", "$\\frac{\pi}{3}$", "$\\frac{\pi}{2}$",
                         "$\\frac{2\pi}{3}$", "$\\frac{5\pi}{6}$", "$\pi$",
                         "$\\frac{7\pi}{6}$", "$\\frac{4\pi}{3}$", "$\\frac{3\pi}{2}$",
                         "$\\frac{5\pi}{3}$", "$\\frac{11\pi}{6}$", "$2\pi$"]
ANGLE_TICK_LABELS_PIBY2 = ["0", "$\\frac{\pi}{6}$", "$\\frac{\pi}{3}$", "$\\frac{\pi}{2}$"]
ANGLE_TICK_LABEL_DICT = {len(ANGLE_TICK_LABELS_PIBY2): ANGLE_TICK_LABELS_PIBY2,
                         len(ANGLE_TICK_LABELS_PI): ANGLE_TICK_LABELS_PI,
                         len(ANGLE_TICK_LABELS_2PI): ANGLE_TICK_LABELS_2PI
                        }


def joint_probability(thetaVals, phiVals, step=np.deg2rad(5)):
    global ANGLE_TICK_LABEL_DICT
    # Joint Probability: p(theta, phi).
    fig = plt.figure(figsize=(3.5, 3), dpi=300)  # see a plot of the joint distribution
    ax = fig.gca()
    xbins = np.ceil(np.max(phiVals) / step).astype(np.int)
    ybins = np.ceil(np.max(thetaVals) / step).astype(np.int)
    jointProb, xEdges, yEdges, histImg = ax.hist2d(phiVals, thetaVals,
                                                   bins=(xbins, ybins),
                                                   density=True)
    ax.set_aspect('equal')
    ax.set_ylabel("$\\theta$")
    ax.set_xlabel("$\phi$")
    xticks = np.arange(0, np.max(xEdges)+np.pi/12, np.pi/6)
    yticks = np.arange(0, np.max(yEdges)+np.pi/12, np.pi/6)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ANGLE_TICK_LABEL_DICT[len(yticks)])
    ax.set_xticklabels(ANGLE_TICK_LABEL_DICT[len(xticks)])
    plt.colorbar(histImg, orientation='horizontal')

    return jointProb.T, xEdges, yEdges, fig


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


def rotation_matrix(rotAnglesDeg=(0, 0, 0), rotAnglesRad=None):  # Tested OK
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


def rotate_2nd_order(tensor, rotAnglesDeg=(0, 0, 0), rotAnglesRad=None):
    R = rotation_matrix(rotAnglesDeg=rotAnglesDeg, rotAnglesRad=rotAnglesRad)
    assert R.shape == tensor.shape, "Shapes of 2nd order tensor and rotation matrix do not match."
    return R @ tensor @ R.T
