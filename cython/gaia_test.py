import numpy as np
from sys import float_info

from emcee import EnsembleSampler
#from emcee.utils import MPIPool

from profiles import build_profile, build_kernel
from dispersion import SphericalJeansDispersion
from likelihood import GaussianLikelihood

rh = 0.25
D = 40.
theta = 2*rh/D

dm = build_profile('NFW')           # NFW DM profile
st = build_profile('plummer',rh=rh) # Plummer Stellar profile
kr = build_kernel('iso')            # isotropic kernel

dwarf_props = {'D':D, 'theta':theta, 'rt':np.inf, 'with_errs':False}
Sigma = SphericalJeansDispersion(dm, st, kr, dwarf_props)

##############################################################################
# Gaia data extraction

x,y,z,vx,vy,vz = np.loadtxt('gs010_bs050_rcrs100_rarcinf_core_0400mpc3_df_10000_0.dat',
                            unpack=True)
R = np.sqrt(x**2+y**2)          # assumed direction of observation along z-axis
d = np.sqrt(x**2+y**2+(D-z)**2) # for simplicity (as suggested on the Gaia wiki)

Evx,Evy,Evz = np.loadtxt('gs010_bs050_rcrs100_rarcinf_core_0400mpc3_df_10000_0_err.dat', 
                         unpack=True, 
                         usecols=(3,4,5))
Ex,Ey,Ez = Evx-vx, Evy-vy,Evz-vz
v = (x*Evx+y*Evy+(D-z)*Evz)/d
dv = (x*Ex+y*Ey+(D-z)*Ez)/d

i = 0
N = 10
#idx = np.random.randint(low=R.size, size=N)
#R, v, dv = R[idx], v[idx], dv[idx]
R, v, dv = R[ N*i : N*(i+1) ], v[ N*i : N*(i+1) ], dv[ N*i : N*(i+1) ]

##############################################################################
# likelihood object instantiation

LL = GaussianLikelihood([R, v, dv, v.mean()], Sigma)
LL.set_free('dm_a')
LL.set_free('dm_b')
LL.set_free('dm_c')
LL.set_free('dm_r0')

##############################################################################
# likelihood model

def lnprior(theta):
    J, r, a, b, c = theta
    if 12. < J < 28 and -3. < r < 2. and 0.5 < a < 3. and 2. < b < 6. and .0 < c < 1.2:
        return 0.0
    return -np.inf

def loglike(theta):
    J, r, a, b, c = theta
    ll = -LL(a, b, J, 10**r, c)
    if not np.isfinite(ll):
        return -float_info.max
    return ll

def lnprob(theta):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglike(theta)

##############################################################################
# sampling

ndim = len(LL.free_pars)
nwalkers = ndim*4
nsteps = 10000

p0 = [ 20., -0.5, 2., 6., 0.5 ]
pos0 = [ p0 + np.concatenate( ([2*randn()], 0.1*randn(ndim-1)) )\
         for i in range(nwalkers) ]

# multi-threads computation
sampler = EnsembleSampler(nwalkers, ndim, lnprob, threads=4)

# Run 500 steps as a burn-in.
pos, prob, state = sampler.run_mcmc(pos0, 500)

# Reset the chain to remove the burn-in samples.
sampler.reset()

# Starting from the final position in the burn-in chain, 
# sample for 10000 steps.
sampler.run_mcmc(pos, nsteps, rstate0=state)

##############################################################################
# save results

samples = sampler.flatchain
lnprobs = sampler.flatlnprobability

# print results in the log file (just in case...)
print 'results'
for sample, loglike in zip(samples, lnprobs):
    print sample, loglike

np.save('results/gaia_cov%s_n%s'%(N, n), 
        np.column_stack( (samples, lnprobs) ) )

