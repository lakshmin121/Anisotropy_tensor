"""
Code to test the conversion of spherical to Euler angles of the projection direction (normal to
projection plane).

Tested against different proj directions and reference directions.
____________________________________
@ Lakshminarayanan Mohana Kumar
9th Sep 2021
"""
import numpy as np
from fiborient import projdir_rotation_3D

Nsamples = 100
thetaVals = np.random.uniform(size=Nsamples) * np.pi
phiVals = np.random.uniform(size=Nsamples) * np.pi

print(thetaVals, phiVals)
rotmats = projdir_rotation_3D(thetaValsRad=thetaVals, phiValsRad=phiVals, refdirRad=(np.pi/2, np.pi/4))

uvecs = np.zeros((len(thetaVals), 3))
cosphi, sinphi = np.cos(phiVals), np.sin(phiVals)
costht, sintht = np.cos(thetaVals), np.sin(thetaVals)
uvecs[:, 0] = sintht * cosphi
uvecs[:, 1] = sintht * sinphi
uvecs[:, 2] = costht

utransfrmd = np.zeros((len(thetaVals), 3))
for i in range(len(thetaVals)):
    rotmat = rotmats[i, :, :]
    utransfrmd[i, :] = uvecs[i, :] @ rotmat.T

print("\n\n unit vectors: \n", uvecs)
print("\n\n rotation mat: \n", rotmats)
print("\n\n unit transformed: \n", utransfrmd)
