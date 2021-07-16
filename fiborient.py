"""
Functions for analysis of fibre orientation state in projected images of FRC.
"""
import numpy as np


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
    total_prob = np.sum(prob_phi) * d_phi
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