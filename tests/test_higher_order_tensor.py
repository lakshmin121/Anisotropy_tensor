"""
Construction of higher order tensor.
Objectives:
    1. generic code for construction of tensor.
    2. test
----------------------------------------------------
@ Lakshminarayanan Mohana Kumar
25th July 2021
"""

import numpy as np
from itertools import product, combinations


def delta(i, j):
    if i==j:
        return 1
    else:
        return 0


def tensor2D(prob_phi, phiDeg, order=2):
    assert len(prob_phi) == len(phiDeg)

    coords = (0, 1)
    base = tuple([coords]*order)
    indices = list(product(*base))

    # direction cosines
    u = np.zeros((2, len(phiDeg)))
    u[0, :] = np.cos(np.deg2rad(phiDeg))
    u[1, :] = np.sin(np.deg2rad(phiDeg))

    Q = []
    for indx in indices:
        # print("index: ", indx)
        elem = prob_phi
        for i in indx:
            elem = elem * u[i, :]
        Q.append(np.sum(elem))
    Q = np.array(Q).reshape((order, order))

    if order==2:
        A = Q - 0.5*np.eye(order)

    elif order==4:
        Q2, A2 = tensor2D(prob_phi, phiDeg, order=2)
        print("Tr(Q2_theo): ", np.trace(Q2))
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

    return Q, A


Q2, A2 = tensor2D([1], [30])
print("Q2_theo: \n", Q2)
print("Tr(Q2_theo) = ", np.trace(Q2))
print("A2: \n", A2)
print("Tr(A2) = ", np.trace(A2))
print()

Q4, A4 =  tensor2D([1], [30], order=4)
print("Q4: \n", Q4)
print("Tr(Q4) = ", np.trace(Q4))
print("A4: \n", A4)
print("Tr(A4) = ", np.trace(A4))
