"""
Functions for analysis of fibre orientation state in projected images of FRC.
"""
import numpy as np
from itertools import product, combinations


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


def orient_tensor_2D(prob_phi, phi_valsDeg):
    if len(phi_valsDeg) == len(prob_phi) + 1:
        phi_valsDeg = 0.5 * (phi_valsDeg[:-1] + phi_valsDeg[1:])
    elif len(phi_valsDeg) == len(prob_phi):
        pass
    else:
        raise ValueError("check dimensions of prob_phi and phi_vals")

    # check for total probability = 1
    d_phi = np.mean(phi_valsDeg[1:] - phi_valsDeg[:-1])  # mean bin width of phi values (delta_phi)
    prob_phi = prob_phi * 180 / np.pi
    total_prob = np.sum(prob_phi) * d_phi * np.pi / 180
    if not np.isclose(total_prob, 1.):
        print("Total probability not 1: {:1.2f}".format(total_prob))

    # direction cosines on 2D plane
    ux = np.cos(np.deg2rad(phi_valsDeg))
    uy = np.sin(np.deg2rad(phi_valsDeg))
    # orientation tensor
    orient_tensor = np.array([[np.multiply(ux, ux), np.multiply(ux, uy)],
                              [np.multiply(uy, ux), np.multiply(uy, uy)]]
                         )
    orient_tensor = orient_tensor @ prob_phi
    orient_tensor = orient_tensor * d_phi

    # check:
    assert orient_tensor.shape == (2, 2)
    tr = orient_tensor.trace()
    if not np.isclose(tr, 1.):
        print("Trace of orientation tensor not 1: {:1.2f}".format(tr))

    aniso_tensor = orient_tensor - 0.5 * np.eye(2)
    # check:
    tr = aniso_tensor.trace()
    if not np.isclose(tr, 0.):
        print("Trace of anisotropy tensor not 0: {:1.2f}".format(tr))

    return orient_tensor, aniso_tensor


def orientation_tensor_4order(prob_phi, phi_valsDeg):
    if len(phi_valsDeg) == len(prob_phi) + 1:
        phi_valsDeg = 0.5 * (phi_valsDeg[:-1] + phi_valsDeg[1:])
    elif len(phi_valsDeg) == len(prob_phi):
        pass
    else:
        raise ValueError("check dimensions of prob_phi and phi_vals")

    # check for total probability = 1
    d_phi = np.mean(phi_valsDeg[1:] - phi_valsDeg[:-1])  # mean bin width of phi values (delta_phi)
    total_prob = np.sum(prob_phi) * d_phi

    if not np.isclose(total_prob, 1.):
        print("Total probability not 1: {:1.2f}".format(total_prob))
    else:
        print("Total probability: {:1.2f}".format(total_prob))

    # direction cosines on 2D planea
    u = np.zeros((2, len(phi_valsDeg)))
    u[0, :] = np.cos(np.deg2rad(phi_valsDeg))
    u[1, :] = np.sin(np.deg2rad(phi_valsDeg))

    coords = [(0, 1)]
    order = 4
    base = tuple(coords * order)
    indices = list(product(*base))

    Q = []
    # elements of the orientation tensor
    for indx in indices:
        elem = prob_phi
        for i in indx:
            elem = elem * u[i, :]
        Q.append(np.sum(elem))

    Q = np.array(Q).reshape((order, order))  # orientation tensor
    print("Tr(Q): ", np.trace(Q))

    # Anisotropy Tensor
    Q2, A2 = orient_tensor_2D(prob_phi, phi_valsDeg)
    print("Tr(Q2): ", np.trace(Q2))
    A = Q.ravel()

    for itrno, indx in enumerate(indices):
        s = set(range(order))
        term1 = 0
        term2 = 0
        counter = 0
        for comb in combinations(s, 2):
            i, j = tuple(indx[m] for m in comb)
            k, l = tuple(indx[m] for m in s.difference(set(comb)))
            term1 += delta(i, j) * Q2[k, l]
            term2 += delta(i, j) * delta(k, l)
            counter += 1
        # print("counter: ", counter)
        print("term1: ", term1)
        print("term2: ", term2)
        # A[itrno] = A[itrno] - (term1 / 6)  #+ (term2 / 48)
    A = A.reshape(Q.shape)
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
        Q2 = R @ Q @ R.T  # Q is from Fourier space. Q2 is in original image.
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
    # odf = 1 / (2*np.pi) + (2 / np.pi) * afunc
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
        indices = product(*base)

        dircosine = np.array([c, s])
        func = []
        a = A.ravel()
        # elements of basis function
        for indx in indices:
            elem = 1
            for i in indx:
                elem = elem * dircosine[i, :]
            func.append(elem)

        afunc = 0
        for itrno, indx in enumerate(indices):
            l = set(indx)
            term1 = 0
            term2 = 0
            for comb in combinations(l, 2):
                rem = tuple(l.difference(comb))
                term1 += delta(*comb) * dircosine[rem[0]] * dircosine[rem[1]]
                term2 += delta(*comb) * delta(*rem)

            func[itrno] = func[itrno] - (term1 / 6) + (term2 / 24)
            afunc = a[itrno] * func[itrno]

        odf = odf + (8 / np.pi) * afunc

    return odf
