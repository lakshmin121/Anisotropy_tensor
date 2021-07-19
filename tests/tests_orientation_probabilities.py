"""
Testing of orientation probabilities.
______________________________________________
@ Lakshminarayanan Mohana Kumar
17th July 2021
"""

import os
import sys
import numpy as np
from matplotlib import pyplot as plt
sys.path.append("../artificial images/")
from matplotlib_settings import *
import orientation_probabilities as op

outDir = 'tests_orient_prob_res'
if not os.path.exists(outDir):
    os.mkdir(outDir)

# Possible phi, theta domains: domain2 used mostly in the study
phi_domain1 = (-90, 90)
theta_domain1 = (-90, 90)
phi_domain2 = (0, 180)
theta_domain2 = (0, 180)
phi_domain3 = (0, 360)
theta_domain3 = (0, 90)

phi_add_domain1 = (-45, 45)

distinct_domains = [phi_domain1, phi_domain2, phi_domain3, theta_domain3, phi_add_domain1]
symmetry = [True, True, True, False, True]

# Test uniform
# Technique: plot discrete probability distribution (histogram).
# Probability (frequency) of all bins must be roughly equal.

# set up figure
# n = 500  # generate n values
# dphi = 10  # binwidth
# for itmno, domain in enumerate(distinct_domains):
#     philow, phihigh = domain
#     phibins = np.arange(philow, phihigh+1, dphi)
#     phivals = op.uniform(domain, size=n)
#     prob_theo = 1 / (len(phibins) * dphi)
#
#     fig = plt.figure(figsize=(3, 2))
#     ax = fig.gca()
#     h, b, _ = ax.hist(phivals, bins=phibins, density=True)
#     ax.set_xlabel("$\phi$, [degrees]")
#     ax.set_ylabel("p($\phi$)")
#     xticks = phibins[::3]
#     ax.set_xticks(xticks)
#     ax.set_xticklabels(xticks)
#     ax.set_xlim([philow, phihigh])
#     ax.ticklabel_format(axis='y', style='sci', scilimits=(-1, 2))
#     plt.yticks(rotation=0)
#     plt.xticks(rotation=90)
#     ax.plot(phibins, [prob_theo]*len(phibins), 'r--')
#     outfname = "uniform_{0}_to_{1}".format(philow, phihigh)
#     fig.savefig(os.path.join(outDir, outfname+'.tiff'))



# # Test Von Mises
# n = 500  # generate n values
# dphi = 10  # binwidth
# for itmno, domain in enumerate(distinct_domains):
#     philow, phihigh = domain
#     spread = phihigh - philow
#     mu = 0.5 * (philow + phihigh)
#     phibins = np.arange(philow, phihigh+1, dphi)
#     phivals = op.vonmises(spreadDeg=spread, size=n) + mu
#     # prob_theo = 1 / (len(phibins) * dphi)
#
#     fig = plt.figure(figsize=(3, 2))
#     ax = fig.gca()
#     h, b, _ = ax.hist(phivals, bins=phibins, density=True)
#     ax.set_xlabel("$\phi$, [degrees]")
#     ax.set_ylabel("p($\phi$)")
#     xticks = phibins[::3]
#     ax.set_xticks(xticks)
#     ax.set_xticklabels(xticks)
#     ax.set_xlim([philow, phihigh])
#     ax.ticklabel_format(axis='y', style='sci', scilimits=(-1, 2))
#     plt.yticks(rotation=0)
#     plt.xticks(rotation=90)
#     # ax.plot(phibins, [prob_theo]*len(phibins), 'r--')
#     outfname = "vonmises_{0}_to_{1}".format(philow, phihigh)
#     fig.savefig(os.path.join(outDir, outfname+'.tiff'))


# # Test sin
# n = 500  # generate n values
# dphi = 10  # binwidth
# for itmno, (domain, symm) in enumerate(zip(distinct_domains, symmetry)):
#     print(itmno)
#     philow, phihigh = domain
#     phibins = np.arange(philow, phihigh+1, dphi)
#     phivals = op.sin(domain, symmetric=symm, size=n)
#     # prob_theo = 1 / (len(phibins) * dphi)
#
#     fig = plt.figure(figsize=(3, 2))
#     ax = fig.gca()
#     h, b, _ = ax.hist(phivals, bins=phibins, density=True)
#     ax.set_xlabel("$\phi$, [degrees]")
#     ax.set_ylabel("p($\phi$)")
#     xticks = phibins[::3]
#     ax.set_xticks(xticks)
#     ax.set_xticklabels(xticks)
#     ax.set_xlim([philow, phihigh])
#     ax.ticklabel_format(axis='y', style='sci', scilimits=(-1, 2))
#     plt.yticks(rotation=0)
#     plt.xticks(rotation=90)
#     # ax.plot(phibins, [prob_theo]*len(phibins), 'r--')
#     outfname = "sin_{0}_to_{1}".format(philow, phihigh)
#     fig.savefig(os.path.join(outDir, outfname+'.tiff'))
