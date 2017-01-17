import numpy as np
from math import log10
import pylab as plt
from scipy.interpolate import interp1d as interp
from scipy.optimize import brentq, minimize_scalar
from profiles import build_profile, build_kernel
from dispersion import SphericalJeansDispersion
from likelihood import GaussianLikelihood
from fitter import MinuitFitter

directory = '/home/andrea/Desktop/work/DWARF/dsphsim/'
rh = 0.04
D = 39.81
theta = 2*rh/D
dm = build_profile('NFW')
st = build_profile('plummer',**{'rh':rh}) # non-Plummer Stellar profile
kr = build_kernel('iso') # isotropic kernel
dwarf_props = {'D':D, 'theta':theta, 'rt':np.inf, 'with_errs':False}
Sigma = SphericalJeansDispersion(dm, st, kr, dwarf_props)

BF = np.zeros([100,2])
for i in range(100):
    R, v = np.loadtxt(directory+'Ret2_data/dsph_%03d.txt'%i,usecols=(5, 9),unpack=True)
    v = v[~np.isnan(v)]
    R = R[~np.isnan(v)]
    dv = np.ones_like(v)*2.
    LL = GaussianLikelihood([R, v, dv, 0.], Sigma)
    LL.set_free('dm_r0')
    global global_loglike
    global_loglike = LL
    M = MinuitFitter(LL)
    
    # J
    M.set_value('J',17)
    M.set_error('J',0.01)
    #M.set_fixed('J')
    #M.set_bound('J',(15,22))
    # r0
    M.set_value('dm_r0',rh*2.)
    M.set_error('dm_r0',0.01)
    #M.set_bound('dm_r0',(R.min(),R.max()*100))
    '''
    LL.set_free('dm_a')
    LL.set_free('dm_b')
    LL.set_free('dm_c')
        # a
    M.set_value('dm_a',1.)
    M.set_error('dm_a',0.01)
    #M.set_bound('dm_a',(1e-10,10))
    # b
    M.set_value('dm_b',3.)
    M.set_error('dm_b',0.01)
    #M.set_bound('dm_b',(1e-10,10))
    # c
    M.set_value('dm_c',1.)
    M.set_error('dm_c',0.01)
    #M.set_bound('dm_c',(1e-10,10))
    '''
    M.set_minuit(**{'tol':1e4,'strategy':2})
    Min = M.migrad_min()
    BF[i] = Min[1][0]['value'],Min[1][1]['value']

np.save('BF_Ret2_Vmean',BF)

J = np.array([BF[i][0] for i in range(100)])
rs = np.array([BF[i][1] for i in range(100)])

print 'J = %.2f +- %.2f'%(J.mean(),J.std()/np.sqrt(J.size))
print 'rs = %.2f +- %.2f'%(rs.mean(),rs.std()/np.sqrt(rs.size))
