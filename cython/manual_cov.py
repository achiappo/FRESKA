import numpy as np
from numpy.random import randn
from emcee import EnsembleSampler

from scipy.interpolate import interp1d as interp
from scipy.optimize import brentq, minimize_scalar

from profiles import build_profile, build_kernel
from dispersion import SphericalJeansDispersion
from likelihood import GaussianLikelihood

rh = 0.04
D = 39.81
theta = 2*rh/D

dm = build_profile('NFW')           # NFW profile for DM
st = build_profile('plummer',rh=rh) # Plummer Stellar profile
kr = build_kernel('iso')            # isotropic kernel
dwarf_props = {'D':D, 'theta':theta, 'rt':np.inf, 'with_errs':False}
Sigma = SphericalJeansDispersion(dm, st, kr, dwarf_props)

n = argv[1]
R, v = np.loadtxt('Ret2_data/dsph_%03d.txt'%int(n), usecols=(5, 7), unpack=True)
#R, v = np.load('equiRdSphs.npy')
vnan = np.isnan(v)
R = R[~vnan]
v = v[~vnan]
# truncate the data vectors
N = argv[2]
v = v[:int(N)]
R = R[:int(N)]
dv = np.zeros_like(v)

LL = GaussianLikelihood([R, v, dv, 0.], Sigma)
LL.set_free('dm_a')
LL.set_free('dm_b')
LL.set_free('dm_c')
LL.set_free('dm_r0')

###############################################################################
# likelihood model

def lnprior(theta, freeJ=False):
    if freeJ:
        J, r, a, b, c = theta
        if 12 < J < 24 and -3 < r < 2 and 0.5 < a < 3 and 2 < b < 6 and 0 < c < 1.2:
            return 0.0
        return -np.inf
    r, a, b, c = theta
    if -3 < r < 2 and 0.5 < a < 3 and 2 < b < 6 and 0 < c < 1.2:
        return 0.0
    return -np.inf

def lnlike(theta, args=[], freeJ=False):
    if freeJ:
        J, r, a, b, c = theta
        return -LL(a, b, J, 10**r, c)
    J = args
    r, a, b, c = theta
    return -LL(a, b, J, 10**r, c)

def lnprob(theta, args=[], freeJ=False):
    if freeJ:
        lp = lnprior(theta, freeJ)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike(theta, args, freeJ)
    lp = lnprior(theta, freeJ)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, args, freeJ)


###############################################################################

nthreads = 1
burnin = 1
nsteps = 10

#-----------------------------------------------------------------------------
## initial scan to locate the minimum J

initial_dim = len(LL.free_pars)
nwalkers = initial_dim*4
initial_p0 = [17, -0.5, 1.5, 4., 0.5]
initial_pos0 = [ initial_p0 + \
                 np.concatenate( ([2*randn()], 0.1*randn(initial_dim-1)) ) \
                 for i in range(nwalkers) ]

# multi-threads computation
sampler = EnsembleSampler(nwalkers, 
                          initial_dim, 
                          lnprob, 
                          kwargs={'freeJ':True}, 
                          threads=nthreads )
# Run initial_burnin steps as a burn-in.
pos, prob, state = sampler.run_mcmc(initial_pos0, burnin)
# Reset the chain to remove the burn-in samples.
sampler.reset()
# Starting from the final position in the burn-in chain, 
# sample for initial_nsteps.
sampler.run_mcmc(pos, nsteps, rstate0=state)
# extract the flat samples and flat likelihood
samples = sampler.flatchain
lnprobs = sampler.flatlnprobability
# find global minimum likelihood value
Like_gmin = min( -lnprobs )
indLike_gmin = np.where( -lnprobs==Like_gmin )[0][0]
# save global minimum likelihood value and corresponding parameters
J_gmin = samples[indLike_gmin, 0]
params_gmin = samples[indLike_gmin, 1:]

#-----------------------------------------------------------------------------
## successive scans to build the wings of the profile

new_dim = len(LL.free_pars)-1
nwalkers = new_dim*4
# initialise sampler with global maximum likelihood parameters
new_p0 = params_gmin
new_pos0 = [ new_p0 + 0.1*randn(new_dim) for i in range(nwalkers)]

###################################
### left wing of profile likelihood
stopL = False
nL = 0
JL = J_gmin - 0.2
LikeL, paramsL, JL_array = [], [], []

# build left wing of profile likelihood 
while not stopL:
    nL += 1
    new_pos0 = [ new_p0 + 0.1*randn(new_dim) for i in range(nwalkers)]
    # instantiate emcee sampler object
    sampler = EnsembleSampler(nwalkers, 
                              new_dim, 
                              lnprob, 
                              args=[JL], 
                              threads=nthreads)
    # Run new_burnin steps as a burn-in.
    pos, prob, state = sampler.run_mcmc(new_pos0, burnin)
    # Reset the chain to remove the burn-in samples.
    sampler.reset()
    # Starting from the final position in the burn-in chain, 
    # sample for new_nsteps.
    sampler.run_mcmc(pos, nsteps, rstate0=state)
    # extract samples and likelihood
    samples = sampler.flatchain
    lnprobs = sampler.flatlnprobability
    # find maximum likelihood value
    Like_min = min( -lnprobs )
    indLike_min = np.where( -lnprobs==Like_min )[0][0]
    # append quantities to their list
    JL_array.append( JL )
    LikeL.append( Like_min ) # save -logLikelihood value
    paramsL.append( samples[indLike_min] )
    # check if we reached the 3sigma level
    if LikeL[-1] < Like_gmin+4.5:
        JL -= 0.2
        # set new initial guesses to maximum likelihood parameters just found
        new_p0 = samples[indLike_min]
    else:
        stopL = True

# reverse and store results in final results arrays
J_array = sorted(JL_array, reverse=True)
Likes = sorted(LikeL, reverse=True)
params = sorted(paramsL, reverse=True)

####################################
### right wing of profile likelihood
stopR = False
nR = 0
JR = J_gmin + 0.2
LikeR, paramsR, JR_array = [], [], []

new_p0 = params_gmin

# build left wing of profile likelihood 
while not stopR:
    nR += 1
    new_pos0 = [ new_p0 + 0.1*randn(new_dim) for i in range(nwalkers)]
    # instantiate emcee sampler object
    sampler = EnsembleSampler(nwalkers, 
                              new_dim, 
                              lnprob, 
                              args=[JR], 
                              threads=nthreads)
    # Run new_burnin steps as a burn-in.
    pos, prob, state = sampler.run_mcmc(new_pos0, burnin)
    # Reset the chain to remove the burn-in samples.
    sampler.reset()
    # Starting from the final position in the burn-in chain, sample for new_nsteps.
    sampler.run_mcmc(pos, nsteps, rstate0=state)
    # extract samples and likelihood
    samples = sampler.flatchain
    lnprobs = sampler.flatlnprobability
    # find maximum likelihood value
    Like_min = min( -lnprobs )
    indLike_min = np.where( -lnprobs==Like_min )[0][0]
    # append quantities to their list
    JR_array.append( JR )
    LikeR.append( Like_min ) # save -logLikelihood value
    paramsR.append( samples[indLike_min] )
    # check if we reached the 3sigma level
    if LikeR[-1] < Like_gmin+4.5:
        JR += 0.2
        # set new initial guesses to maximum likelihood parameters just found
        new_p0 = samples[indLike_min]
    else:
        stopR = True

# complete final results array
J_array.append( J_gmin )    ; J_array.append( JR_array )
Likes.append( Like_gmin )   ; Likes.append( LikeR )
params.append( params_gmin ); params.extend( paramsR )


np.save('manual_coverage%s_%s'%(N, n), np.vstack( (J_array, Likes, params) ) )

print nL, nR
