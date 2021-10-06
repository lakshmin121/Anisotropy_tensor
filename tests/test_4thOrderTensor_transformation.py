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

import numpy as np
from matplotlib_settings import *
from itertools import product, combinations

outDir = "tests_tensorTransformation"
SECONDORDER_SIZE = 3
FOURTHORDER_SIZE = 9
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
        ValueError("Shapes of joint probability {0} and unit-vector {1} does not match.".format(uvec[0].shape,
                                                                                                probThetaPhi.shape
                                                                                                ))
    orientTensor = np.zeros((SECONDORDER_SIZE, SECONDORDER_SIZE))
    for i in range(SECONDORDER_SIZE):
        for j in range(SECONDORDER_SIZE):
            orientTensor[i, j] = np.trapz(np.trapz(probThetaPhi * uvec[i] * uvec[j], thetaVals, axis=0), phiVals, axis=-1)
    # print("Orientation Tensor - 2nd order: {}".format(orientTensor))

    anisoTensor = orientTensor - 1/3 * np.eye(SECONDORDER_SIZE)

    return orientTensor, anisoTensor


def cycle_indices(size):  # Tested OK
    all_indices = np.arange(size).astype(np.int)  # indices are integers
    for s in range(size):
        yield tuple(np.roll(all_indices, shift=s))


def delta(i, j):
    if i == j:
        return 1
    else:
        return 0


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
    assert R.shape == tensor.shape, ValueError("Shapes of 2nd order tensor and rotation matrix do not match.")
    return R @ tensor @ R.T


def rotate_4th_order(tensor, rotAnglesDeg=(0, 0, 0), rotAnglesRad=None):
    R = rotation_matrix(rotAnglesDeg=rotAnglesDeg, rotAnglesRad=rotAnglesRad)
    assert tensor.shape == (FOURTHORDER_SIZE, FOURTHORDER_SIZE), ValueError("Unacceptable shape of 4th order tensor.")
    modshape = (SECONDORDER_SIZE, SECONDORDER_SIZE,SECONDORDER_SIZE, SECONDORDER_SIZE)
    tensor_rottd = np.einsum('im,jn,kp,lq,mnpq->ijkl', R, R, R, R, tensor.reshape(modshape))

    return tensor_rottd.reshape((FOURTHORDER_SIZE, FOURTHORDER_SIZE))




def binedges_to_centres(edges):
    if not isinstance(edges, np.ndarray):
        edges = np.array(edges)
    return 0.5 * (edges[:-1] + edges[1:])


# def bincentres_to_edges(centres, round=8):  # Not tested for non-uniform binwidth. But expected to work.
#     if not isinstance(centres, np.ndarray):
#         centres = np.array(centres)
#     d = np.diff(centres)
#     e1 = set(np.round(centres[:-1] - d/2, round))
#     e2 = set(np.round(centres[1:] + d/2, round))
#     edges = e1.union(e2)
#     return np.array(list(edges))


def rotate_joint_probability(refElvEdges, refAzmEdges, refProb,
                             rotElvEdges, rotAzmEdges,
                             rotAnglesDeg=(0, 0, 0), rotAnglesRad=None):
    # psiDomain = (0, np.pi / 2)  # Domain of theta
    # upsDomain = (0, 2 * np.pi)  # Domain of phi
    # step = np.deg2rad(5)
    psiEdges = rotElvEdges
    upsEdges = rotAzmEdges
    psiCentres = 0.5 * (psiEdges[1:] + psiEdges[:-1])
    upsCentres = 0.5 * (upsEdges[1:] + upsEdges[:-1])

    R = rotation_matrix(rotAnglesDeg=rotAnglesDeg, rotAnglesRad=rotAnglesRad)

    # unit vectors in rotated system:
    uvec_rottd = np.zeros((3, len(psiCentres), len(upsCentres)))
    uvec_rottd[0, :, :] = np.outer(np.sin(psiCentres), np.cos(upsCentres))
    uvec_rottd[1, :, :] = np.outer(np.sin(psiCentres), np.sin(upsCentres))
    uvec_rottd[2, :, :] = np.outer(np.cos(psiCentres), np.ones(len(upsCentres)))

    uvec = np.tensordot(R.T, uvec_rottd, axes=([1, 0]))
    thtCentresTransformed = np.arccos(uvec[2])
    phiCentresTransformed = np.arctan2(uvec[1], uvec[0])
    phiCentresTransformed = np.where(phiCentresTransformed > 0,
                                     phiCentresTransformed,
                                     2 * np.pi + phiCentresTransformed
                                     )
    thtEdges = refElvEdges
    phiEdges = refAzmEdges
    print("Theta edges: \n", thtEdges)
    # print("Phi edges: \n", phiEdges)
    thtIndices = np.digitize(thtCentresTransformed, thtEdges) - 1
    phiIndices = np.digitize(phiCentresTransformed, phiEdges) - 1
    # print(thtIndices.min(), thtIndices.max())
    # print(phiIndices.min(), phiIndices.max())
    p_psi_ups = refProb[thtIndices, phiIndices]
    p_psi_ups = p_psi_ups.reshape((len(psiCentres), len(upsCentres)))
    totalIntegral = np.trapz(np.trapz(p_psi_ups, psiCentres, axis=0), upsCentres, axis=-1)
    assert np.isclose(totalIntegral, 1, atol=1e-1), \
        ValueError("Total integral not close to unity: {}".format(totalIntegral))
    # print("Total integral: ", totalIntegral)

    fig = plt.figure(figsize=(3.5, 3), dpi=300)
    ax = fig.gca()
    ax.set_aspect('equal')
    ax.set_ylabel("$\psi$")
    ax.set_xlabel("$\\upsilon$")
    xscale = len(upsCentres) / phiEdges[-1]
    yscale = len(psiCentres) / thtEdges[-1]
    xticks = np.arange(0, np.floor(xscale * (np.max(phiEdges) + np.pi / 12)), np.floor(xscale * np.pi / 6))
    yticks = np.arange(0, np.floor(yscale * (np.max(thtEdges) + np.pi / 12)), np.floor(yscale * np.pi / 6))

    imgTransformed = ax.imshow(p_psi_ups, origin='lower',
                               extent=[xticks.min(), xticks.max(), yticks.min(), yticks.max()])
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ANGLE_TICK_LABEL_DICT[len(yticks)])
    ax.set_xticklabels(ANGLE_TICK_LABEL_DICT[len(xticks)])
    plt.colorbar(imgTransformed, orientation='horizontal')

    return p_psi_ups, psiCentres, upsCentres, fig



def orient_tensor_3D(probThetaPhi, thetaValsRad, phiValsRad, order=2):
    # --------------------------------------------------------------------
    # Input Checks
    m, n = probThetaPhi.shape
    if len(thetaValsRad) == m + 1:
        thetaValsRad = 0.5 * (thetaValsRad[:-1] + thetaValsRad[1:])
    elif len(thetaValsRad) == m:
        pass
    else:
        raise ValueError("check dimensions of probThetaPhi and thetaValsRad: {0}, {1}".format(m, len(thetaValsRad)))

    if len(phiValsRad) == n + 1:
        phiValsRad = 0.5 * (phiValsRad[:-1] + phiValsRad[1:])
    elif len(phiValsRad) == n:
        pass
    else:
        raise ValueError("check dimensions of probThetaPhi and phiValsRad: {0}, {1}".format(n, len(phiValsRad)))

    # check for total probability = 1
    total_prob = np.trapz(np.trapz(probThetaPhi, thetaValsRad, axis=0), phiValsRad, axis=-1)
    if not np.isclose(total_prob, 1., atol=1e-1):
        print("Total probability not 1: {:1.2f}".format(total_prob))
    # else:
    #     print("Total probability: {:1.2f}".format(total_prob))
    # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    # Set-up
    coords = (0, 1, 2)  # possible coordinates in 2D space
    base = tuple([coords] * order)  # tensor space dimension = coords * order
    indices = list(product(*base))  # all possible tensor indices Qijkl
    # dphi = np.mean(np.diff(phiValsRad))
    # dtht = np.mean(np.diff(thetaValsRad))
    order_size = int(np.sqrt(len(coords) ** order))

    # unit vectors: direction cosines:
    uvec = np.zeros((3, len(thetaValsRad), len(phiValsRad)))
    uvec[0, :, :] = np.outer(np.sin(thetaValsRad), np.cos(phiValsRad))
    uvec[1, :, :] = np.outer(np.sin(thetaValsRad), np.sin(phiValsRad))
    uvec[2, :, :] = np.outer(np.cos(thetaValsRad), np.ones(len(phiValsRad)))

    assert uvec[0].shape == probThetaPhi.shape, \
        ValueError("Shapes of joint probability {0} and unit-vector {1} does not match.".format(uvec[0].shape,
                                                                                           probThetaPhi.shape))
    # Orientation Tensor
    Q = []
    for indx in indices:
        # print("index: ", indx)
        elem = probThetaPhi
        for i in indx:
            elem = elem * uvec[i, :]
        Q.append(np.trapz(np.trapz(elem, thetaValsRad, axis=0), phiValsRad, axis=-1))
    Q = np.array(Q).reshape((order_size, order_size))  # * dphi * dtht

    if order==2:
        A = Q - np.eye(order_size) / 3

    elif order==4:
        Q2, A2 = orient_tensor_3D(probThetaPhi, thetaValsRad, phiValsRad, order=2)
        # print("Tr(Q2_theo): ", np.trace(Q2_theo))
        A = np.copy(Q).ravel()

        for itrno, indx in enumerate(indices):
            s = set(range(order))
            term1 = 0
            term2 = 0
            for comb in combinations(s, 2):
                i, j = tuple(indx[m] for m in comb)
                k, l = tuple(indx[m] for m in s.difference(set(comb)))
                # print("i, j, k, l: ", i, j, k, l)
                term1 += delta(i, j) * Q2[k, l]
                term2 += delta(i, j) * delta(k, l)  # this calculates term2 twice.
            A[itrno] = A[itrno] - (term1 / 7) + (term2 / 70)
        A = A.reshape(Q.shape)

    # Output Checks:
    tr = Q.trace()
    if not np.isclose(tr, 1.):
        print("Trace of orientation tensor not 1: {:1.2f}".format(tr))
    tr = A.trace()
    if not np.isclose(tr, 0.):
        print("Trace of anisotropy tensor not 0: {:1.2f}".format(tr))

    return Q, A