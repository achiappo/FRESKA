import numpy as np
from yaml import dump
from math import log10
from pymc import Uniform, MCMC, deterministic

from scipy.interpolate import interp1d as interp
from scipy.optimize import brentq, minimize_scalar

from profiles import build_profile, build_kernel
from dispersion import SphericalJeansDispersion
from likelihood import GaussianLikelihood
from fitter import *

##############################################################################

rh = 0.04
D = 39.81
theta = 2*rh/D

R, v = np.loadtxt('Ret2_data/dsph_001.txt',usecols=(5, 7),unpack=True)
v = v[~np.isnan(v)]
R = R[~np.isnan(v)]
dv = np.zeros_like(v)

##############################################################################

dm = build_profile('NFW')
st = build_profile('plummer',**{'rh':rh}) # Plummer Stellar profile
kr = build_kernel('iso') # isotropic kernel
dwarf_props = {'D':D, 'theta':theta, 'rt':np.inf, 'with_errs':False}

Sigma = SphericalJeansDispersion(dm, st, kr, dwarf_props)

LL = GaussianLikelihood([R, v, dv, 0.], Sigma)
LL.set_free('dm_a')
LL.set_free('dm_b')
LL.set_free('dm_c')
LL.set_free('dm_r0')

##############################################################################

def model(J):
    a = Uniform('a', lower=0, upper=5)
    b = Uniform('b', lower=0, upper=5)
    c = Uniform('c', lower=0, upper=5)
    r = Uniform('r', lower=0, upper=5)
    
    @deterministic
    def loglike(J=J, a=a, b=b, c=c, r=r):
        return LL(a, b, J, r, c)
    
    return locals()

##############################################################################

J_array = np.linspace(15,18)

L_arr = np.empty_like(J_array)
a_arr = np.empty_like(J_array)
b_arr = np.empty_like(J_array)
c_arr = np.empty_like(J_array)
r_arr = np.empty_like(J_array)

for j,J in enumerate(J_array):
    M = MCMC(model(J))
    M.sample(1000)
    
    L_sample = M.trace('loglike')[:]
    true_indx = ~np.isnan(L_sample)
    L_sample = L_sample[true_indx]
    min_indx = np.where( L_sample==L_sample.min() )[0][0]
    L_arr[j] = L_sample[min_indx]
    
    for par in ['a', 'b', 'c', 'r']:
        exec( "{p}_sample = M.trace('{p}')[:]".format(p=par) )
        exec( "{p}_sample = {p}_sample[true_indx]".format(p=par) )
        exec( "{p}_arr[j] = {p}_sample[min_indx]".format(p=par) )

interp_L = interp(J_array, L_arr-L_arr.min())
interp_r = interp(J_array, r_arr)
interp_a = interp(J_array, a_arr)
interp_b = interp(J_array, b_arr)
interp_c = interp(J_array, c_arr)

##############################################################################

eval_Like_J = np.linspace(J_array.min(), J_array.max(), 1e3)
min_Like_J = interp_L(eval_Like_J).min()
Jmin = eval_Like_J[ np.where( interp_L(eval_Like_J) == min_Like_J )[0][0] ]
J_plt = np.linspace(J_array.min(),J_array.max())

Jr = float( interp_r(Jmin) )
Ja = float( interp_a(Jmin) )
Jb = float( interp_b(Jmin) )
Jc = float( interp_c(Jmin) )

dm.r0 = Jr
dm.a = Ja
dm.b = Jb
dm.c = Jc

minrho = lambda rho : abs(Jmin - log10( dm.Jfactor(**dwarf_props) ) - 2*rho)
Jrho = float( 10**minimize_scalar(minrho).x )

##############################################################################

np.save('results/LikeJ_Mcmc',np.vstack( (J_plt, interp_L(J_plt) ) ) )
dump( { 'Nstars':R.size, 'Jmin':Jmin, 'rho':Jrho, 'r':Jr,
		'a':Ja, 'b':Jb, 'c':Jc }, open('results/results_Mcmc.yaml','w') )
