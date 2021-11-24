"""
Functions for the fitting of ODF to projection data.
"""

import numpy as np
from itertools import product, combinations

SECONDORDER_SIZE_3D = 3
FOURTHORDER_SIZE_3D = 9


def delta(i, j):
    if i == j:
        return 1
    else:
        return 0


def basisfunc_3D(phiDomainRad=(0, np.pi),
                 thetaDomainRad=(0, np.pi),
                 nPhi=180, nTheta=180, order=2):

    phiVals = np.linspace(phiDomainRad[0], phiDomainRad[-1], nPhi)
    thtVals = np.linspace(thetaDomainRad[0], thetaDomainRad[-1], nTheta)

    coords = (0, 1, 2)  # possible coordinates in 3D space
    base = tuple([coords] * order)  # tensor space dimension = coords * order
    indices = list(product(*base))  # all possible tensor indices Qijkl
    order_size = int(np.sqrt(len(coords) ** order))

    uvec = np.zeros((len(coords), nPhi, nTheta))
    if phiDomainRad == (0, np.pi) and thetaDomainRad == (0, np.pi):
        uvec[0, :, :] = np.outer(np.cos(phiVals), np.sin(thtVals))
        uvec[1, :, :] = np.outer(np.sin(phiVals), np.sin(thtVals))
        uvec[2, :, :] = np.outer(np.ones(nPhi), np.cos(thtVals))
    elif phiDomainRad == (-np.pi/2, np.pi/2) and thetaDomainRad == (-np.pi/2, np.pi/2):
        uvec[0, :, :] = np.outer(np.cos(phiVals), np.cos(thtVals))
        uvec[1, :, :] = np.outer(np.sin(phiVals), np.cos(thtVals))
        uvec[2, :, :] = np.outer(np.ones(nPhi), np.sin(thtVals))

    Q = []
    for indx in indices:
        elem = 1
        for i in indx:
            elem = elem * uvec[i, :, :]
        Q.append(elem)
    Q = np.array(Q).reshape((order_size, order_size, nPhi, nTheta))

    if order==2:
        I = np.zeros(Q.shape)  # This is for future applications > 2nd order. Hence, np.eye is not used.
        i = np.arange(SECONDORDER_SIZE_3D)
        I[i, i, :, :] = 1
        F = Q - I / 3

    elif order==4:
        F2, Q2 = basisfunc_3D(phiDomainRad=phiDomainRad, thetaDomainRad=thetaDomainRad,
                              nPhi=nPhi, nTheta=nTheta, order=2)
        F = np.copy(Q)  # Fijkl = Qijkl
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
            I = itrno // FOURTHORDER_SIZE_3D  # row number
            J = itrno % FOURTHORDER_SIZE_3D  # col number
            # print(I, J)
            F[I, J, :, :] = F[I, J, :, :] - (term1 / 7) + (term2 / 70)

    return F, Q


def isiterable(var):
    try:
        it = iter(var)
    except TypeError:
        return False
    return True


def projdir_rotation_3D(thetaValsRad, phiValsRad, refdirRad=(0, 0)):
    """
    Returns the rotation matrix to transform the given projection direction
    to a reference direction.
    Reference: https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    :param thetaValsRad: Spherical coordinate - elevation.
    :param phiValsRad: Spherical coordinate - azimuth
    :return: rotation matrix to transform the given projection direction (theta, phi) to the reference direction,
    global Z- axis <0, 0, 1> by default
    """
    if not isiterable(thetaValsRad):
        thetaValsRad, phiValsRad = np.array([thetaValsRad]), np.array([phiValsRad])
    assert len(thetaValsRad) == len(phiValsRad), ValueError("len(thetaValsRad) does not match len(phiValsRad).")
    thetaref, phiref = refdirRad
    cosphiref, sinphiref = np.cos(phiref), np.sin(phiref)
    costhtref, sinthtref = np.cos(thetaref), np.sin(thetaref)

    cosphi, sinphi = np.cos(phiValsRad), np.sin(phiValsRad)
    costht, sintht = np.cos(thetaValsRad), np.sin(thetaValsRad)
    uvecs = np.zeros((len(thetaValsRad), 3))

    uref = np.array([sinthtref * cosphiref, sinthtref * sinphiref, costhtref])
    uvecs[:, 0] = sintht * cosphi
    uvecs[:, 1] = sintht * sinphi
    uvecs[:, 2] = costht

    rotaxs = np.cross(uvecs, uref)
    cosrotangles = np.dot(uvecs, uref)
    print("cosrotangles: ", cosrotangles)

    rotmats = np.zeros((len(thetaValsRad), 3, 3))
    for i, cosang in enumerate(cosrotangles):
        v = rotaxs[i, :]
        c = cosang
        vcross = np.cross(v, np.eye(v.shape[0]) * -1)
        rotmats[i, :, :] = np.eye(3) \
                           + vcross \
                           + np.dot(vcross, vcross) / (1 + c)

    return np.squeeze(rotmats)


# TODO: test this using alpha_xz
def basisfunc_proj(projRotMat, upsDomainRad=(0, np.pi), psiDomainRad=(0, np.pi),
                 nUps=180, nPsi=180, order=2):
    F3D, Q3D = basisfunc_3D(phiDomainRad=upsDomainRad, thetaDomainRad=psiDomainRad,
                            nPhi=nUps, nTheta=nPsi, order=order)
    # Check F3D:
    # Check-1: Trace = 0
    traceF = np.trace(F3D, axis1=0, axis2=1)
    traceQ = np.trace(Q3D, axis1=0, axis2=1)
    print("Checking trace of F3D: \n", np.all(np.isclose(traceF, np.zeros(traceF.shape))))  # checking traces of all 2D subarrays are zeroes
    print("Checking trace of Q3D: \n", np.all(np.isclose(traceQ, np.ones(traceQ.shape))))  # checking traces of all 2D subarrays are ones

    psiVals = np.linspace(psiDomainRad[0], psiDomainRad[1], nPsi, endpoint=True)
    # print(F3D.shape)
    # Fproj = np.trapz(F3D, psiVals)
    if psiDomainRad == (0, np.pi):
        for psi in range(F3D.shape[-1]):
            F3D[..., psi] = F3D[..., psi] * np.sin(psiVals[psi])
    elif psiDomainRad == (-np.pi/2, np.pi/2):
        for psi in range(F3D.shape[-1]):
            F3D[..., psi] = F3D[..., psi] * np.cos(psiVals[psi])
    Fproj = np.trapz(F3D, psiVals, axis=-1)

    print("Fproj[:, :, 90]: \n", Fproj[:, :, 90])
    print("Fproj[:, :, 20]: \n", Fproj[:, :, 20])

    # Check Fproj in local coordinates
    # Check-1: psi component is zero
    print("Check psi component of Fproj local: ", np.sum(Fproj[2, 2, ...]))
    # Check-2: trace is zero
    traceFproj = np.trace(Fproj, axis1=0, axis2=1)
    print("Check trace of Fproj local: ", np.all(np.isclose(traceFproj, np.zeros(traceFproj.shape))))

    # if projdir == refdirRad:
    #     return Fproj
    # else:
        # R is rotation matrix
    # R = projdir_rotation_3D(projdir[0], projdir[1], refdirRad=refdirRad)
    # print("Rotation matrix: \n", R)
    R = projRotMat
    if order == 2:
        # Fproj_rottd = np.einsum('im, jn, ijk -> mnk', R, R, Fproj)
        Fproj_rottd = np.einsum('mi, nj, ijk -> mnk', R, R, Fproj)
    elif order==4:
        m, n, npoints = Fproj.shape
        assert (m, n) == (FOURTHORDER_SIZE_3D, FOURTHORDER_SIZE_3D), ValueError("shape of 4th order tensor must be 9x9")
        modshape = (SECONDORDER_SIZE_3D, SECONDORDER_SIZE_3D, SECONDORDER_SIZE_3D, SECONDORDER_SIZE_3D)
        # Fproj_rottd = np.einsum('im, jn, kp, lq,mnpqs->ijkls', R, R, R, R, Fproj.reshape((*modshape, npoints)))
        Fproj_rottd = np.einsum('mi, nj, pk, ql,mnpqs->ijkls', R, R, R, R, Fproj.reshape((*modshape, npoints)))
        Fproj_rottd = Fproj_rottd.reshape(Fproj.shape)
    else:
        raise NotImplementedError
    return Fproj_rottd
