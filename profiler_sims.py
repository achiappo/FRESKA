
__author__ = "Andrea Chiappo"
__email__ = "chiappo.andrea@gmail.com"

#
# Template code to perform a Jeans analysis
# on simulated stellar kinematic data from
# the Gaia Challenge simulation suite
# available at http://astrowiki.ph.surrey.ac.uk/dokuwiki/doku.php?id=workshop
#
# outputs:  
# - .npy file containing the profile likelihood of the J factor
# - .yaml file containing the the results of statistical inference
#

import numpy as np
from utils import load_data, envelope

###############################################################################
# select the Gaia Challenge:
# - model
# - mock size (number of stars desired to enter the analysis)
# - data set size (size of the sample provided by the Gaia Challenge team) 
# - realisation N. (number identifying the different sample realisations) 

mod, mocksize, dset = 0, 100, 0

# to enter these quantities from command line, uncomment the following
#from sys import argv
#mod, mocksize, dset = argv[1:]

# select distance (in kpc) to center of mock galaxy
D = 100.

# Gaia data extraction
data = load_gaia(homedir, MockSize, dataSize, dset, mod, D)
R, v, dv, rh, cst, ker, r0_true, rho0_true = data

###############################################################################
# select the model used in the Jeans equation 

from profiles import build_profile, build_kernel
from dispersion import SphericalJeansDispersion
from likelihood import GaussianLikelihood

theta = 0.5

dm = build_profile('nfw')
st = build_profile('plummer',rh=rh)
kr = build_kernel(ker)

dwarf_props = {'D':D, 'theta':theta, 'rt':np.inf, 'with_errs':False}

Sigma = SphericalJeansDispersion(dm, st, kr, dwarf_props)

###############################################################################
# likelihood object instantiation

LL = GaussianLikelihood([R, v, dv, v.mean()], Sigma)
LL.set_free('dm_a')
LL.set_free('dm_b')
LL.set_free('dm_c')
LL.set_free('dm_r0')

# enter allowed ranges on the parameter values
# for the exploration of the parameter space

priors = {'J': (10, 30),
          'dm_r0' : (-3, 2),
          'dm_a' : (0, 3),
          'dm_b' : (0.5, 6),
          'dm_c' : (0, 1.5)
         }

if mod>3:
	LL.set_free('ker_ra')
	priors['ker_ra'] = {'range' : (-3, 2)}

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
nwalkers = ndim*100
nsteps = 3000

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
sampler = EnsembleSampler(nwalkers, ndim, lnprob, threads=4)

# Run initial burn-in steps
pos, prob, state = sampler.run_mcmc(pos0, 500)

# Reset the chain to remove the burn-in samples.
sampler.reset()

# Starting from the final position in the burn-in chain, sample for nsteps.
sampler.run_mcmc(pos, nsteps, rstate0=state)

###############################################################################
# statistical inference from sampling the loglikelihood parameter space
###############################################################################

# instruction to flatten the chain to successively 
# envelope the results along the desired direction

samples = sampler.flatchain
lnprobs = sampler.flatlnprobability

# determine positional index of J in samples

for p,par in enumerate( LLargs ):
	if 'J' in par:
		Jind = p

# envelope lowermost -lnlikelihood values over sampled J range
# to obtain the profile of another variable, change 'param' argument

Jenv, Senv, Lenv = envelope(flatsamples, flatlnprobs, param=Jind)

# print results of envelope on screen 

print 'results'
for J,S,L in zip(Jenv, Senv, Lenv):
  print J, S, L

# save results
np.save( 'results/gaia_%s_fit_s%s_n%s_LikeJ'%(casedir[mod], mocksize, dset), 
         np.column_stack( (Jenv, Senv, Lenv) ) )

#------------------------------------------------------------------------------
# save global minimum from samples
Lmin = min(Lnew)
Jmin = Jnew[ np.where( Lnew==Lmin )[0][0] ]

print 'Jmin: ', Jmin

from yaml import dump
with open('results/gaia_%s_fit_s%s_n%s_Jmin.yaml'%(casedir[mod], mocksize, dset), 'w') as yml:
  dump( {'Jmin':Jmin}, yml )

