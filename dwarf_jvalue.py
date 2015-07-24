#!/usr/bin/python

import sys
import math
import time
import numpy as np
from iminuit import Minuit
from ATminuit import get_data,get_sigmalos,M
from scipy.integrate import quadrature

# paramters
nmax     = 1000
nparams  = 7
pi       = math.pi
rstar    = 0.3e0                        # 0.3 kpc
kpctom   = 3.085e19                     # kpc in m
Msun     = 1.9891e30                    # Solar mass unit
Mhalo    = 1.e9 * Msun                  # Halo mass
sigma_MW = 200                          # velocity dispersion of Milky Way in km s^-1
G        = 6.67e-11*Msun                # m^3 Msun^-1 s^-2          

galaxy  = sys.argv[1]                   # get the galaxy name from the command line

#######################################################################################################
#                                               MAIN CODE
#######################################################################################################

# log(Likelihood) function
def LogLike(data,a,b,c): #Mvar,rs,beta,a,b,c):
    x,v,dv,rh,rt,nstars,D,pa = data
    Mvar,rs,beta = pa[:3]   # beta = pa[2]
    u = pa[-1]
    arg1 = 0.
    arg2 = 0.
    #rcut  = pow(G*Mhalo*pow(D,2)/2./pow(sigma_MW,2),1/3.)      # truncation scale on DM density profile
    rho0    = Mvar/M(rstar,rs,a,b,c)
    for i in range(nstars):
        s     = get_sigmalos(abs(x[i]),rho0,rt,rh,rs,beta,a,b,c)
        arg1 += 0.5e0*pow(v[i]-u,2)/(pow(dv[i],2)+pow(s,2))
        arg2 += 0.5e0*math.log(2*pi*pow(dv[i],2)+pow(s,2))
    dlike  = arg1+arg2
    return dlike

# class necessary to fit the LogLike function evaluated at its data points
class LogLH:    
    def __init__(self,data):
        self.data = data
    
    def compute(self,a,b,c): #Mvar,rs,beta,a,b,c):
        return LogLike(self.data,a,b,c) #Mvar,rs,beta,a,b,c)          # FITTED PARAMETERS: a,b,c

# integrand of the J factor, i.e. DM density profile squared
def profile(r,rho0,rs,a,b,c):
    return pow(rho0/pow(r/rs,a)/pow(1+pow(r/rs,b),(c-a)/b),2.)

#a_init,b_init,c_init = get_data(galaxy)[-1][-4:-1] # extract values from param file
data = get_data(galaxy)

# building a function object to be passed to Minuit and evaluation of -MLE parameters
lh = LogLH(data)
#kwdargs = dict(Mvar=1.,rs=1.,beta=1.,a=1.,b=1.,c=1.,error_Mvar=0.01,error_rs=0.01,error_beta=0.01,\
#    error_a=0.01,error_b=0.01,error_c=0.01,limit_Mvar=(6,8),limit_rs=(1.e-3,1.e2),limit_beta=(-2.,2.),\
#    limit_a=(0.1,2.),limit_b=(0.1,2.),limit_c=(2.,6.))
kwdargs = dict(a=1.,b=1.,c=1.,error_a=0.01,error_b=0.01,error_c=0.01,\
    limit_a=(0.1,2.),limit_b=(0.1,2.),limit_c=(2.,6.))
m = Minuit(lh.compute,**kwdargs)
start = time.time()
bestfit = m.migrad()
end = time.time()
print 'time elapsed = ',int(end-start),' s'

'''
Mvar = bestfit[1][0]["value"]
rs   = bestfit[1][1]["value"]
beta = bestfit[1][2]["value"]
a    = bestfit[1][3]["value"]
b    = bestfit[1][4]["value"]
c    = bestfit[1][5]["value"]

rt = data[4]
rho0    = 10.e0**Mvar/M(rstar,rs,a,b,c)
Jvalue  = quadrature(profile,0.,rt,args=(rho0,rs,a,b,c),maxiter=150)[0]
print math.log10(Jvalue)
'''
