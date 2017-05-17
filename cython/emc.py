import numpy as np
import pylab as plt
from math import log10
from emcee import EnsembleSampler
from emcee.utils import MPIPool

from scipy.interpolate import interp1d as interp
from scipy.optimize import brentq, minimize_scalar
from profiles import build_profile, build_kernel
from dispersion import SphericalJeansDispersion
from likelihood import GaussianLikelihood

rh = 0.04
D = 39.81
theta = 2*rh/D

dm = build_profile('NFW')
st = build_profile('plummer',rh=rh) # Plummer Stellar profile
kr = build_kernel('iso') # isotropic kernel
dwarf_props = {'D':D, 'theta':theta, 'rt':np.inf, 'with_errs':False}
Sigma = SphericalJeansDispersion(dm, st, kr, dwarf_props)

directory = '/home/andrea/Desktop/work/DWARF/'
R, v = np.loadtxt(directory+'dsphsim/Ret2_data/dsph_001.txt',usecols=(5, 7),unpack=True)
#R, v = np.load('results/equiRdSphs.npy')
vnan = np.isnan(v)
v = v[:100]#[~vnan]
R = R[:100]#[~vnan]
dv = np.zeros_like(v)

LL = GaussianLikelihood([R, v, dv, 0.], Sigma)
LL.set_free('dm_a')
LL.set_free('dm_b')
LL.set_free('dm_c')
LL.set_free('dm_r0')

def lnprior(theta):
    r, a, b, c = theta
    if -3. < r < 2. and 0.5 < a < 3. and 2. < b < 6. and .0 < c < 1.2:
        return 0.0
    return -np.inf

def loglike(theta, J):
    r, a, b, c = theta
    return LL(a, b, J, 10**r, c)

def lnprob(theta, J):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglike(theta, J)

ndim = len(LL.free_pars)-1
nwalkers = ndim*4

p0 = [-1., 1., 3., 1.]
pos = [ p0 + 0.1*np.random.randn(ndim) for i in range(nwalkers)]

J = 16.72

# Initialize the MPI-based pool used for parallelization.
pool = MPIPool()

if not pool.is_master():
    # Wait for instructions from the master process.
    pool.wait()
    sys.exit(0)

sampler = EnsembleSampler(nwalkers, ndim, lnprob, args=(J,), pool=pool)

# Run 100 steps as a burn-in.
pos, prob, state = sampler.run_mcmc(pos, 100)

# Reset the chain to remove the burn-in samples.
sampler.reset()

# Starting from the final position in the burn-in chain, 
# sample for 1000 steps.
sampler.run_mcmc(pos, 1000, rstate0=state)

np.save('samples', sampler.chain)
np.save('lnprobs', sampler.lnprobability)
