import numpy as np
from profiles import *
from dispersion import SphericalJeansDispersion
from likelihood import GaussianLikelihood
from fitter import MinuitFitter
from utils import *

homedir = '/home/andrea/Desktop/work/DWARF/Jvalue/project1/test/Isotrop_Core_nonPlum'
MockSize = 100
dataSize = 100
dset = 1
# enter model choice - cf. casedir (options 1,2,3,4)
mod = 1
theta = 0.5
D = 100

data = load_gaia(homedir, MockSize, dataSize, dset, mod, D, True)

kinematic = data[0:3]
rh = data[-3]
r0_true = data[-2]
rho0_true = data[-1]

dm = build_profile('Zhao', a=1., b=3., c=0.) # Cored DM profile
st = build_profile('plummer', **{'c':1, 'rh':rh}) # non-Plummer Stellar profile
kr = build_kernel('iso') # isotropic kernel

dwarf_props = {'D':D, 'theta':theta, 'rt':np.inf, 'errs':False}
Sigma = SphericalJeansDispersion(dm, st, kr, dwarf_props)

LL = GaussianLikelihood(kinematic, Sigma)
LL.set_free('dm_r0')
LL.set_free('dm_a')
LL.set_free('dm_b')
LL.set_free('dm_c')

global global_loglike
global_loglike = LL
M = MinuitFitter(LL)

# J
M.set_value('J',19)
M.set_error('J',0.01)
M.set_bound('J',(15,22))
# r0
M.set_value('dm_r0',1.)
M.set_error('dm_r0',0.01)
M.set_bound('dm_r0',(kinematic[0].min(),kinematic[0].max()))
# a
M.set_value('dm_a',1.)
M.set_error('dm_a',0.01)
M.set_bound('dm_a',(1e-10,10))
# b
M.set_value('dm_b',1.)
M.set_error('dm_b',0.01)
M.set_bound('dm_b',(1e-10,10))
# c
M.set_value('dm_c',1.)
M.set_error('dm_c',0.01)
M.set_bound('dm_c',(1e-10,10))

M.settings['print_level'] = 1

M.set_minuit(**{'tol':1e2,'strategy':2})
M.migrad_min()
M.minos_profile('J',**{'bound':(18,22),'subract_min':True})

