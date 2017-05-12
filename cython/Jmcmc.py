import inspect
import numpy as np
from yaml import dump
from sys import argv
from pymc import * 

from profiles import build_profile, build_kernel
from dispersion import SphericalJeansDispersion
from likelihood import GaussianLikelihood

##############################################################################

rh = 0.04
D = 39.81
theta = 2*rh/D

n = argv[1]
R, v = np.loadtxt('Ret2_data/dsph_%03d.txt'%int(n),usecols=(5, 7),unpack=True)
vnan = np.isnan(v)
v = v[~vnan]
R = R[~vnan]
dv = np.zeros_like(v)

##############################################################################

dm = build_profile('NFW')
st = build_profile('plummer',rh=rh) 
kr = build_kernel('iso')
dwarf_props = {'D':D, 'theta':theta, 'rt':np.inf}

Sigma = SphericalJeansDispersion(dm, st, kr, dwarf_props)

LL = GaussianLikelihood([R, v, dv, 0.], Sigma)
LL.set_free('dm_a')
LL.set_free('dm_b')
LL.set_free('dm_c')
LL.set_free('dm_r0')

############################################################################## 

def model(J, R=R, v=v, dv=dv):
    #J = Uniform('J', lower=15, upper=19)
    r = Uniform('r', lower=-3., upper=2.)
    a = Uniform('a', lower=.5, upper=3.)
    b = Uniform('b', lower=2., upper=6.)
    c = Uniform('c', lower=0., upper=1.2)
    
    @deterministic
    def sigma(a=a, b=b, J=J, r=10**r, c=c):
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        values.pop('frame')
        for name,value in zip(LL.free_pars.keys(), values.values()):
            Sigma.setparams(name, value)
        return dv**2 + Sigma.compute(R)
    
    @deterministic
    def loglike(J=J, a=a, b=b, c=c, r=r):
        return LL(a, b, J, 10**r, c)
    
    v_obs = Normal('v', mu=v.mean(), tau=1/sigma, value=v, observed=True)
    
    return locals()

############################################################################## 
J = float( argv[2] )

M = MCMC( model(J), db='pickle', dbname='results/mcmc_1e4.pickle' )
M.sample( iter=11e3, burn=1e3 )
'''
#-----------------------------------------------------------------
#J = Uniform('J', lower=15, upper=19)
r = Uniform('r', lower=-3., upper=2.)
a = Uniform('a', lower=.5, upper=3.)
b = Uniform('b', lower=2., upper=6.)
c = Uniform('c', lower=0., upper=1.2)

@deterministic
def loglike(J=J, a=a, b=b, c=c, r=r):
    return LL(a, b, J, 10**r, c)

M = MCMC( [a, b, c, r, loglike], db='pickle', dbname='results/mcmc_AdaMet_1e5.pickle' )
M.use_step_method(AdaptiveMetropolis, [r, a, b, c] ) 
M.sample( iter=11e4, burn=1e4 )
#-----------------------------------------------------------------
'''
L_sample = M.trace('loglike')[:]
Lnan = np.logical_or( np.isnan(L_sample), np.isinf(L_sample) )
#L = L_sample[~Lnan]
L = L_sample[:]
'''
Lmin = np.where( L_sample==min(L_sample) )[0][0]
L = L_sample[Lmin]

for par in ['r', 'a', 'b', 'c']:
    exec( "{p} = M.trace('{p}')[~Lnan]".format(p=par) )
    exec( "{p} = {p}[Lmin]".format(p=par) )

np.save( 'results/params_%s_%05d'%(n, int(J*1000)), np.vstack( (J, L, a, b, c, r) ) )
'''
for par in ['a', 'b', 'c', 'r']:
    #exec( "{p} = M.trace('{p}')[~Lnan]".format(p=par) )
    exec( "{p} = M.trace('{p}')[:]".format(p=par) )

np.save( 'results/MCtest_1e4', np.vstack( (L, a, b, c, r) ) )

Mscores = geweke( M, intervals=100 )
dump( Mscores, open('results/Mscores_1e4.yaml', 'w') )
dump( M.stats(), open('results/Mstats_1e4.yaml', 'w') )

