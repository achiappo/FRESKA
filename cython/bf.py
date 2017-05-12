import numpy as np
from sys import argv
from scipy.optimize import brute, fmin

from profiles import build_profile, build_kernel
from dispersion import SphericalJeansDispersion
from likelihood import GaussianLikelihood

##############################################################################

rh = 0.04
D = 39.81
theta = 2*rh/D

directory = '/home/andrea/Desktop/work/DWARF/dsphsim'
R, v = np.loadtxt(directory+'/Ret2_data/dsph_001.txt',usecols=(5, 7),unpack=True)
vnan = np.isnan(v)
v = v[~vnan]
R = R[~vnan]
dv = np.zeros_like(v)

##############################################################################

dm = build_profile('NFW')
st = build_profile('plummer',rh=rh) 
kr = build_kernel('iso')
dwarf_props = {'D':D, 'theta':theta, 'rt':np.inf}

Sigma = SphericalJeansDispersion(dm, st, kr, dwarf_props)

LL = GaussianLikelihood([R, v, dv, 0.], Sigma)
LL.set_free('dm_a')
#LL.set_free('dm_b')
#LL.set_free('dm_c')
LL.set_free('dm_r0')

##############################################################################

def logLike(p, *param):
    J, = param
    a, r = p
    L = LL(a, J, 10**r)
    if np.isnan(L) or np.isinf(L):
    	return 1e8
    else:
    	return L

J = 16.72
rranges = ( (0.5, 3), (-3, 2) )

res = brute(logLike, ranges=rranges, args=(J,), Ns=100, full_output=True, disp=True)

np.save( 'LikeBF', res)

