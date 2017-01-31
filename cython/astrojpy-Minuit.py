import numpy as np
from yaml import dump
from math import log10
import numpy as np

from scipy.interpolate import interp1d as interp
from scipy.optimize import minimize_scalar

from profiles import build_profile, build_kernel
from dispersion import SphericalJeansDispersion
from likelihood import GaussianLikelihood
from fitter import MinuitFitter


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

J_array = np.linspace(15,18,10)

J_new = np.empty([0])
L_arr = np.empty([0])
a_arr = np.empty([0])
b_arr = np.empty([0])
c_arr = np.empty([0])
r_arr = np.empty([0])

for J in J_array:
    M = MinuitFitter(LL)
    # J
    M.set_value('J',J)
    M.set_fixed('J')
    # r0
    M.set_value('dm_r0',rh*4.)
    M.set_error('dm_r0',0.01)
    # a
    M.set_value('dm_a',1.)
    M.set_error('dm_a',0.01)
    # b
    M.set_value('dm_b',3.)
    M.set_error('dm_b',0.01)
    # c
    M.set_value('dm_c',1.)
    M.set_error('dm_c',0.01)
    
    M.set_minuit(**{'tol':1,'strategy':2})
    valid = False
    maxval = 1
    while not valid and maxval>0 :
        BF = M.migrad_min()
        valid = BF[0]['is_valid']
        maxval = BF[0]['fval'] - M.minuit.tol*BF[0]['up']*1e-4
        M.minuit.tol *= 10
    else:
        J_new = np.append(J_new,J)
        L_arr = np.append(L_arr,BF[0]['fval'])
        a_arr = np.append(a_arr,BF[1][0]['value'])
        b_arr = np.append(b_arr,BF[1][1]['value'])
        c_arr = np.append(c_arr,BF[1][3]['value'])
        r_arr = np.append(r_arr,BF[1][4]['value'])

##############################################################################

interp_L = interp(J_new, L_arr-L_arr.min())
interp_r = interp(J_new, r_arr)
interp_a = interp(J_new, a_arr)
interp_b = interp(J_new, b_arr)
interp_c = interp(J_new, c_arr)

eval_Like_J = np.linspace(J_new.min(), J_new.max(), 1e3)
min_Like_J = interp_L(eval_Like_J).min()
Jmin = eval_Like_J[ np.where( interp_L(eval_Like_J) == min_Like_J )[0][0] ]

Jr = float(interp_r(Jmin))
Ja = float(interp_a(Jmin))
Jb = float(interp_b(Jmin))
Jc = float(interp_c(Jmin))

minrho = lambda rho : abs(Jmin - np.log10( dm.Jfactor(**dwarf_props) ) - 2*rho)
Jrho = float(10**minimize_scalar(minrho).x)

##############################################################################

np.save('results/LikeJ',np.vstack( (J_new, interp_L(J_new) ) ) )
dump( {'Nstars':R.size, 'Jmin':Jmin, 'rho':Jrho, 'r':Jr, 
		'a':Ja, 'b':Jb, 'c':Jc }, open('results/results.yaml','w') )
