
__author__ = "Andrea Chiappo"
__email__ = "chiappo.andrea@gmail.com"

#
# Template code to perform a Jeans analysis
# on real stellar kinematic data from a
# dwarf spheroidal satellite galaxy
#
# outputs:  
# - .npy file containing the profile likelihood of the J factor
# - .yaml file containing the the results of statistical inference
#

import numpy as np
from utils import load_data, envelope

###############################################################################
# select the candidate dwarf galaxy (Draco)
# import the corresponding stellar kinematic data

dwarf = 'dra' 
R,v,dv,D,rh,rt = load_data(dwarf)

###############################################################################
# select the model used in the Jeans equation 

from profiles import build_profile, build_kernel
from dispersion import SphericalJeansDispersion
from likelihood import GaussianLikelihood

ker = 'iso'
theta = 0.5

dm = build_profile('nfw')
st = build_profile('plummer',rh=rh)
kr = build_kernel(ker)

dwarf_props = {'D':D, 'theta':theta, 'rt':np.inf, 'with_errs':False}

Sigma = SphericalJeansDispersion(dm, st, kr, dwarf_props)

###############################################################################
# likelihood object instantiation

LL = GaussianLikelihood([R, v, dv, v.mean()], Sigma)
LL.set_free('dm_r0')
LL.set_free('dm_a')
LL.set_free('dm_b')
LL.set_free('dm_c')

# enter allowed ranges on the parameter values
# for the exploration of the parameter space

priors = {'J': (10, 30),
          'dm_r0' : (-3, 2),
          'dm_a' : (0, 3),
          'dm_b' : (0.5, 6),
          'dm_c' : (0, 1.5)
         }

if not ker == 'iso':
    ker_param = 'ker_'+kr.params[0]
    LL.set_free(ker_param)
    if ker == 'om':
        priors[ker_param] = (-3, 2)
    else:
        priors[ker_param] = (-9, 0.9)

# print on screen likelihood arguments
LLargs = LL.free_pars.keys()
print('LLargs: ',LLargs)

###############################################################################
# definition of the three elements entering the emcee sampler:
# - log prior
# - log likelihood
# - log posterior
###############################################################################
from sys import float_info

def lnprior(theta):
    for val,par in zip(theta, LLargs):
        pi, pf = priors[par]
        if not pi < val < pf:
            return -np.inf
    return 0.0

def lnlike(theta):
    new_theta = []
    # copy theta into new list
    # where r0 is in natural units
    for val,par in zip(theta, LLargs):
        new_theta.append(10**val if par=='dm_r0' or par=='ker_ra' else val)
    try:
        ll = -LL(*new_theta)
    except:
        ll = -float_info.max
    if not np.isfinite(ll):
        return -float_info.max
    return ll

def lnprob(theta):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta)

###############################################################################
# loglikelihood parameter space sampling
###############################################################################

# definition of number of steps in the sampling chain
# the value reported were found to be a compromise 
# between sampling time and thoroughness

ndim = len(LLargs)
nwalkers = 100 
if R.size<100:
    nsteps = 8000
elif 100<R.size<500:
    nsteps = 4000
elif 500<R.size<1000:
    nsteps = 2000
elif 1000<R.size<2000:
    nsteps = 1000
else:
    nsteps = 500

# initial positions of random walkers as
# randomly sampled points from parameter ranges

pos0 = np.empty([nwalkers,ndim])
for w in range(nwalkers):
    for p,par in enumerate(LLargs):
        pL,pR = priors[par]
        p0 = np.random.uniform(low=pL,high=pR)
        pos0[w,p] = p0

from emcee import EnsembleSampler

# multi-threads computation
sampler = EnsembleSampler(nwalkers, ndim, lnprob, threads=1)

# Run initial burn-in steps
pos, prob, state = sampler.run_mcmc(pos0, 200)

# Reset the chain to remove the burn-in samples
sampler.reset()

# Starting from the final position in the burn-in chain, sample for nsteps.
sampler.run_mcmc(pos, nsteps, rstate0=state)

###############################################################################
# statistical inference from sampling the loglikelihood parameter space
###############################################################################

# instruction to save the result of sampling:
# - the coordinates of the sampled points in the chain
# - the corresponding logprobability
np.save('results/{0}/{1}/samples_{0}_{1}'.format(ker,dwarf), sampler.chain)
np.save('results/{0}/{1}/lnprobs_{0}_{1}'.format(ker,dwarf), sampler.lnprobability)

# instruction to flatten the chain to successively 
# envelope the results along the desired direction

flatsamples = sampler.flatchain
flatlnprobs = sampler.flatlnprobability

# envelope lowermost -lnlikelihood values over sampled J range
# to obtain the profile of another variable, change 'param' argument

Penv, Senv, Lenv = envelope(flatsamples, flatlnprobs, param=2)

# print results of envelope on screen 

print 'results'
for P,S,L in zip(Penv, Senv, Lenv):
    print P, S, L


