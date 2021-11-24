"""
Functions for analysis of fibre orientation state in projected images of FRC.
"""
import numpy as np
from matplotlib_settings import *
from itertools import product, combinations


SECONDORDER_SIZE_3D = 3
FOURTHORDER_SIZE_3D = 9


# TODO: change all angular arguments to radian units.

def delta(i, j):
    if i==j:
        return 1
    else:
        return 0

def theo_orient_tensor_2D(phiDeg):
    ux = np.cos(np.deg2rad(phiDeg))
    uy = np.sin(np.deg2rad(phiDeg))
    orient_tensor = np.array([[ux*ux, ux*uy],
                              [uy*ux, uy*uy]]
                             )

    return orient_tensor


def theo_orient_tensor_3D(thetaDeg, phiDeg):
    up = np.sin(np.deg2rad(thetaDeg))
    ux = up*np.cos(np.deg2rad(phiDeg))
    uy = up*np.sin(np.deg2rad(phiDeg))
    uz = np.cos(np.deg2rad(thetaDeg))
    orient_tensor = np.array([[ux*ux, ux*uy, ux*uz],
                              [uy*ux, uy*uy, uy*uz],
                              [uz*ux, uz*uy, uz*uz]]
                             )

    return orient_tensor


def orient_tensor_2D(prob_phi, phi_valsDeg, order=2):
    if len(phi_valsDeg) == len(prob_phi) + 1:
        phi_valsDeg = 0.5 * (phi_valsDeg[:-1] + phi_valsDeg[1:])
    elif len(phi_valsDeg) == len(prob_phi):
        pass
    else:
        raise ValueError("check dimensions of prob_phi and phi_vals: {0}, {1}".format(len(prob_phi), len(phi_valsDeg)))

    # check for total probability = 1
    d_phi = np.mean(phi_valsDeg[1:] - phi_valsDeg[:-1])  # mean bin width of phi values (delta_phi)
    total_prob = np.sum(prob_phi) * d_phi
    if not np.isclose(total_prob, 1.):
        print("Total probability not 1: {:1.2f}".format(total_prob))
    # else:
    #     print("Total probability: {:1.2f}".format(total_prob))

    # setup
    coords = (0, 1)  # possible coordinates in 2D space
    base = tuple([coords] * order)  # tensor space dimension = coords * order
    indices = list(product(*base))  # all possible tensor indices Qijkl

    # direction cosines
    u = np.zeros((2, len(phi_valsDeg)))
    u[0, :] = np.cos(np.deg2rad(phi_valsDeg))
    u[1, :] = np.sin(np.deg2rad(phi_valsDeg))

    # Orientation Tensor
    Q = []
    for indx in indices:
        # print("index: ", indx)
        elem = prob_phi
        for i in indx:
            elem = elem * u[i, :]
        Q.append(np.sum(elem))
    Q = np.array(Q).reshape((order, order)) * d_phi

    if order==2:
        A = Q - 0.5*np.eye(order)

    elif order==4:
        Q2, A2 = orient_tensor_2D(prob_phi, phi_valsDeg, order=2)
        # print("Tr(Q2_theo): ", np.trace(Q2_theo))
        A = np.copy(Q).ravel()

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

    # check:
    tr = Q.trace()
    if not np.isclose(tr, 1.):
        print("Trace of orientation tensor not 1: {:1.2f}".format(tr))

    # check:
    tr = A.trace()
    if not np.isclose(tr, 0.):
        print("Trace of anisotropy tensor not 0: {:1.2f}".format(tr))

    return Q, A


def orient_tensor_from_FTimage(wimgFT, coordsys='image'):
    # Calculation of orientation tensor in Fourier space:
    m, n = wimgFT.shape
    u = np.arange(0, m) - m // 2  # shifting origin to center of image
    v = np.arange(0, n) - n // 2
    uu, vv = np.meshgrid(u, v)  # all points in Fourier space
    r = np.sqrt(uu ** 2 + vv ** 2)  # radial distance to each point.
    r = np.where(r == 0, 1, r)

    ku = np.divide(uu, r)  # spatial frequency (unit vector component) in u-direction (x)
    kv = np.divide(vv, r)  # spatial frequency (unit vector component) in v-direction (y)
    E = np.sum(wimgFT)  # Total energy in Fourier space (sum of values at all points)

    Quu = np.sum(ku ** 2 * wimgFT) / E
    Quv = np.sum(ku * kv * wimgFT) / E
    Qvv = np.sum(kv ** 2 * wimgFT) / E

    Q = np.array([[Quu, Quv], [Quv, Qvv]])  # orientation tensor in Fspace
    A = Q - 0.5*np.eye(2)

    if coordsys=='image':
        # estimating the actual orientation tensor from the tensor obtained in Fspace
        # The axes in Fourier space are rotated by 90 degress w.r.t the axes in original image.
        R = np.array([[0., -1.], [1., 0.]])

        # transformation of orientation tensor for 90 degrees rotation.
        Q2 = R @ Q @ R.T  # Q is from Fourier space. Q2_theo is in original image.
        A2 = Q2-0.5*np.eye(2)

        return Q2, A2
    else:
        return Q, A


def sanitize_2Dtensor(tensor):
    if not type(tensor) == np.ndarray:
        tensor = np.ndarray(tensor)
    return tensor


def tensor2odf_2D(phivalsDeg, tensors):
    if type(tensors) in (list, tuple):
        A = sanitize_2Dtensor(tensors[0])
    else:
        A = sanitize_2Dtensor(tensors)
    assert A.shape == (2, 2), 'Only two-dimensional tensors accepted'
    phivals = np.deg2rad(phivalsDeg)
    c = np.cos(phivals)
    s = np.sin(phivals)
    a = A.ravel()
    afunc = a[0] * (c*c - 0.5) + a[1] * (c*s) + a[2] * (s*c) + a[3] * (s*s - 0.5)
    odf = 1 / (2*np.pi) + (2 / np.pi) * afunc

    if type(tensors) in (list, tuple):
        if len(tensors) == 2:
            A = sanitize_2Dtensor(tensors[1])
        else:
            print("tensors size not equal to 2. Returning second-order approximated ODF.")
            return odf
        assert A.shape == (4, 4), 'Only four-dimensional tensors accepted'
        coords = [(0, 1)]
        order = 4
        base = tuple(coords * order)
        indices = list(product(*base))

        u = np.array([c, s])
        func = []
        a = A.ravel()
        # elements of basis function
        for indx in indices:
            elem = 1
            for i in indx:
                elem = elem * u[i, :]
            func.append(elem)

        afunc = 0
        for itrno, indx in enumerate(indices):
            s = set(range(4))
            term1 = 0
            term2 = 0
            for comb in combinations(s, 2):
                i, j = tuple(indx[m] for m in comb)
                k, l = tuple(indx[m] for m in s.difference(set(comb)))
                # print("i, j, k, l: ", i, j, k, l)
                term1 += delta(i, j) * u[k, :] * u[l, :]
                term2 += delta(i, j) * delta(k, l)
            func[itrno] = func[itrno] - (term1 / 6) + (term2 / 48)
            afunc = a[itrno] * func[itrno]

        odf = odf + (8 / np.pi) * afunc

    return odf


def joint_probability_3D(thetaVals, phiVals, step=np.deg2rad(5)):
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


def orient_tensor_3D(probThetaPhi, thetaValsRad, phiValsRad, phiDomainRad=(0, np.pi), thetaDomainRad=(0, np.pi), order=2):
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
    coords = (0, 1, 2)  # possible coordinates in 3D space
    base = tuple([coords] * order)  # tensor space dimension = coords * order
    indices = list(product(*base))  # all possible tensor indices Qijkl
    order_size = int(np.sqrt(len(coords) ** order))

    # unit vectors: direction cosines:
    uvec = np.zeros((3, len(phiValsRad), len(thetaValsRad)))
    if phiDomainRad == (0, np.pi) and thetaDomainRad == (0, np.pi):
        uvec[0, :, :] = np.outer(np.cos(phiValsRad), np.sin(thetaValsRad))
        uvec[1, :, :] = np.outer(np.sin(phiValsRad), np.sin(thetaValsRad))
        uvec[2, :, :] = np.outer(np.ones(len(phiValsRad)), np.cos(thetaValsRad))
    elif phiDomainRad == (-np.pi/2, np.pi/2) and thetaDomainRad == (-np.pi/2, np.pi/2):
        uvec[0, :, :] = np.outer(np.cos(phiValsRad), np.cos(thetaValsRad))
        uvec[1, :, :] = np.outer(np.sin(phiValsRad), np.cos(thetaValsRad))
        uvec[2, :, :] = np.outer(np.ones(len(phiValsRad)), np.sin(thetaValsRad))


    assert uvec[0].shape == probThetaPhi.shape, \
        ValueError("Shapes of joint probability {0} and unit-vector {1} does not match.".format(uvec[0].shape,
                                                                                           probThetaPhi.shape))
    # Orientation Tensor
    Q = []
    for indx in indices:
        # print("index: ", indx)
        elem = probThetaPhi.T  # if not transposed, row -> theta
        for i in indx:
            elem = elem * uvec[i, :, :]
        Q.append(np.trapz(np.trapz(elem, thetaValsRad, axis=-1), phiValsRad))
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


def cycle_indices(size):  # Tested OK
    all_indices = np.arange(size).astype(np.int)  # indices are integers
    for s in range(size):
        yield tuple(np.roll(all_indices, shift=s))


def delta(i, j):
    if i == j:
        return 1
    else:
        return 0


def rotation_matrix_3D(rotAnglesDeg=(0, 0, 0), rotAnglesRad=None):  # Tested OK
    if rotAnglesRad is None:
        rotAnglesRad = np.deg2rad(rotAnglesDeg)
    assert len(rotAnglesRad) == 3, ValueError("rotAnglesDeg or rotAnglesRad is not iterable of length 3.")

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


def rotate_2nd_order_3D(tensor, rotAnglesDeg=(0, 0, 0), rotAnglesRad=None):
    R = rotation_matrix_3D(rotAnglesDeg=rotAnglesDeg, rotAnglesRad=rotAnglesRad)
    assert R.shape == tensor.shape, ValueError("Shapes of 2nd order tensor and rotation matrix do not match.")
    return R @ tensor @ R.T


def rotate_4th_order_3D(tensor, rotAnglesDeg=(0, 0, 0), rotAnglesRad=None):
    R = rotation_matrix_3D(rotAnglesDeg=rotAnglesDeg, rotAnglesRad=rotAnglesRad)
    assert tensor.shape == (FOURTHORDER_SIZE_3D, FOURTHORDER_SIZE_3D), ValueError("Unacceptable shape of 4th order tensor.")
    modshape = (SECONDORDER_SIZE_3D, SECONDORDER_SIZE_3D, SECONDORDER_SIZE_3D, SECONDORDER_SIZE_3D)
    tensor_rottd = np.einsum('im,jn,kp,lq,mnpq->ijkl', R, R, R, R, tensor.reshape(modshape))

    return tensor_rottd.reshape((FOURTHORDER_SIZE_3D, FOURTHORDER_SIZE_3D))


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

    R = rotation_matrix_3D(rotAnglesDeg=rotAnglesDeg, rotAnglesRad=rotAnglesRad)

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
