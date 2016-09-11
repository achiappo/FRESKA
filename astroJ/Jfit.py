import yaml
import numpy as np
from math import *
from sys import argv
from iminuit import Minuit
from scipy.integrate import quad
from multiprocessing import Pool
from scipy import optimize as sciopt
from scipy.interpolate import interp1d as interp
from functions import integral2, Jfactor, proxy_integral2, compute_I_array, get_data

######################################################################################################################################
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('config', type=file)
parser.add_argument('-b', '--base_config', type=file,default='../astroJ/default.yaml')
args = parser.parse_args()

config = yaml.load(args.base_config)
config.update(yaml.load(args.config))

err = 0.01
Minuit_setting = {}
fitting_params = {}
for prof in config:
    if config[prof]['use']:
        for var in config[prof]: # add instrution to neglect element from dictionary (to remove need for try-except)
            try:
                if config[prof][var]['free']:
                    Minuit_setting.update({var:config[prof][var]['val'],
                    	'error_%s'%var:err})
                    fitting_params.update({var:config[prof][var]['val']})
                else:
                    Minuit_setting.update({var:config[prof][var]['val'],
                    	'fixed_%s'%var:True})
            except: pass

R,v,dv = np.load('input.npy')
u=v.mean()
nstars=np.size(R)

gamma_array = R/rh

class logLike:
    def __init__(self, data, numprocs=1):
        self.data = data
        self.numprocs = numprocs
        self.cache = {}
        # the following performs cythonisation of the functions.pyx module
        system('python setup.py build_ext --inplace')

        
    def retrieve(self, fitting_params):
        if all(fitting_params) in self.cache.keys():
            I=self.cache[*fitting_params]
            iscached=True
        else:
            I=0#dummy value
            iscached=False
        return iscached, I
    
    def store(self, I, fitting_params):
        self.cache[fitting_params]=I
        return

    def pool_compute(self, alpha, delta, beta):
        pool = Pool(processes=self.numprocs)
        results = pool.map(proxy_integral2, 
        					itertools.izip(self.data, 
        				   	itertools.repeat(alpha), 
        				   	itertools.repeat(delta), 
        				   	itertools.repeat(beta)))
        pool.close()
        pool.join()
        return results
    
    def compute(self, J, r0, fitting_params):
        #treat r0 first, as it can't be vectorized
        if 'ra' in fitting_params.keys():
        	delta = ra/
        r0=10**r0
        ra=10**ra
        alpha = rh / r0
        delta = ra / rh
        gamma = self.data / rh
        params = [ alpha , delta, beta]

        if r0<=0 or ra<=0:
            if np.isscalar(J):
                return np.nan
            else:
                return np.nan*np.ones_like(J)
        if np.all(J<0):
            return np.nan*np.ones_like(J)
        else:
            res=np.zeros_like(J)
            mask=J>0
            res[np.logical_not(mask)]=np.nan
            iscached, Icached = self.retrieve(r0, ra, beta)
            if iscached:
                I=Icached
            else:
                if self.numprocs==1 or self.data.size<100:
                    #I_array = compute_I_array(self.data, rh/r0)
                    I_array = np.array([integral2(gamma, params, ) for gamma in self.data])
                else:
                    I_array = self.pool_compute(ra/rh, rh/r0)
                I = r0**3*I_array*np.power(10,J/2.)/np.sqrt(Jfactor(D,np.inf,r0,1.,theta))
                self.store(ra,r0,I)
            S = (dv**2.)[:,np.newaxis] + I[:,np.newaxis] 
        res = (np.log(S) + ((v-u)**2.)[:,np.newaxis]/S)
        res = res.sum(axis=0)
        return res/2.

######################################################################################################################################

lh = logLike(gamma_array)
settings = {'errordef':0.5,'print_level':0,'pedantic':False,'J':19,'ra':np.log10(rh/2.),'r0':np.log10(rh*2.),'error_J':0.1,'error_ra':0.01,'error_r0':0.01}
Jfit = Minuit(lh.compute,**settings)
Jfit.tol = 0.01
Jfit.set_strategy(2)


# fitting scheme
J_array = np.linspace(params['J_i'],params['J_f'],params['NJ'])
J_new = np.empty([0])
min_LikeJ = np.empty([0])
min_ra_arr = np.empty([0])
min_r0_arr = np.empty([0])

######################################################################################################################################
# minimum and C.I. determination

interp_r0 = float(interp(J_new,min_r0_arr))
interp_LikeJ = float(interp(J_new,min_LikeJ))
interp_Like_ra = float(interp(J_new,min_ra_arr))

J_min = sciopt.minimize_scalar(interp_Like,method='Bounded',bounds=(J_new[0],J_new[-1])).x
min_ra = interp_Like_ra(J_min)

J_r0 = interp_r0(J_min)
J_rho0 = 10**sciopt.minimize_scalar(lambda log10rho0:abs(J_min-np.log10(Jfactor(D,np.inf,J_r0,1.,0.5))-2*log10rho0)).x

def sigma_CI(J,n):
	return np.abs(interp_Like(J)-interp_Like(J_min)-n)

# 1-sigma uncertainties
one_sigma_l = sciopt.minimize_scalar(lambda J : sigma_CI(J,0.5), method='Bounded', bounds=(J_min-1,J_min)).x-J_min
one_sigma_r = sciopt.minimize_scalar(lambda J : sigma_CI(J,0.5), method='Bounded', bounds=(J_min,J_min+1)).x-J_min

# 2-sigma uncertainties
two_sigma_l = sciopt.minimize_scalar(lambda J : sigma_CI(J,2.0), method='Bounded', bounds=(J_min-1,J_min)).x-J_min
two_sigma_r = sciopt.minimize_scalar(lambda J : sigma_CI(J,2.0), method='Bounded', bounds=(J_min,J_min+1)).x-J_min

# 3-sigma uncertainties
three_sigma_l = sciopt.minimize_scalar(lambda J : sigma_CI(J,4.5), method='Bounded', bounds=(J_min-1,J_min)).x-J_min
three_sigma_r = sciopt.minimize_scalar(lambda J : sigma_CI(J,4.5), method='Bounded', bounds=(J_min,J_min+1)).x-J_min

yaml.dump({'N':argv[1],'Jmin':J_min,'r0':J_r0,'rho0':J_rho0,'J1sL':one_sigma_l,'J1sR':one_sigma_r,
	'J2sL':two_sigma_l,'J2sR':two_sigma_r,'J3sL':three_sigma_l,'J3sR':three_sigma_r},open('results.yaml','wb'))

LikeJ_array = np.linspace(J_min+three_sigma_l-0.5,J_min+three_sigma_r+0.5,100)
np.save('LikeJ',np.vstack((LikeJ_array,interp_Like(LikeJ_array)-spline_LikeJ(J_min))))
