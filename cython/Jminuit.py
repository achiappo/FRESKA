import numpy as np
from sys import argv

from profiles import build_profile, build_kernel
from dispersion import SphericalJeansDispersion
from likelihood import GaussianLikelihood
from fitter import MinuitFitter

rh = 0.04
D = 39.81
theta = 2*rh/D

dm = build_profile('NFW', r0=0.35)
st = build_profile('plummer', rh=rh) 
kr = build_kernel('iso')  
dwarf_props = {'D':D, 'theta':theta, 'rt':np.inf}
Sigma = SphericalJeansDispersion(dm, st, kr, dwarf_props)

n = argv[1]
R, v = np.loadtxt('Ret2_data/dsph_%03d.txt'%int(n), usecols=(5, 7), unpack=True)
vnan = np.isnan(v)
v = v[~vnan]
R = R[~vnan]
dv = np.zeros_like(v)

LL = GaussianLikelihood([R, v, dv, 0.], Sigma)
LL.set_free('dm_r0')
LL.set_free('dm_a')
LL.set_free('dm_b')
LL.set_free('dm_c')

J_array = np.linspace(15, 19, 100)
L_arr = np.empty([0])
parameters = [par.split('_')[-1] for par in LL.free_pars.keys()]
for par in parameters:
    exec( '%s_arr = np.empty([0])'%par)

for j,J in enumerate(J_array):
    M = MinuitFitter(LL)
    # J
    M.set_value('J',J)
    M.set_fixed('J')
    # r0
    M.set_value('dm_r0', 0.35)
    M.set_error('dm_r0', 0.01)
    M.set_bound('dm_r0', (0.001,100))
    # a
    M.set_value('dm_a', 1.)
    M.set_error('dm_a', 0.01)
    M.set_bound('dm_a', (0.5,3))
    # b
    M.set_value('dm_b', 3.)
    M.set_error('dm_b', 0.01)
    M.set_bound('dm_b', (2,6))
    # c
    M.set_value('dm_c', 1.)
    M.set_error('dm_c', 0.01)
    M.set_bound('dm_c', (0,1.2))
    # beta
    #M.set_value('ker_beta',0.)
    #M.set_error('ker_beta',0.01)
    
    M.settings['print_level'] = 1
    M.set_minuit(tol=1e4,strategy=2)
    
    valid = False
    while not valid:
        BF = M.migrad_min()
        valid = BF[0]['is_valid']
        if M.minuit.tol>1e6 and not valid:
            break
        else:
            M.minuit.tol *= 10
    else:
        if not np.isnan(BF[0]['edm']):
            L_arr = np.append(L_arr, BF[0]['fval'])
            for n,par in enumerate(parameters):
                exec( "{0}_arr = np.append({0}_arr, BF[1][n]['value'])".format(par))

np.save('results/Minuit_%s'%n, np.vstack(L_arr, r0_arr, a_arr, b_arr, c_arr))

