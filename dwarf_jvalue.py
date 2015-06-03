import sys
import math
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
def LogLike(data,a,b,c):
    x,v,dv,rh,rt,nstars,D,pa = data
    Mvar,rs,beta = pa[:3]
    u = pa[-1]
    arg1 = 0.
    arg2 = 0.
    #rcut  = pow(G*Mhalo*pow(D,2)/2./pow(sigma_MW,2),1/3.)
    p0    = 10.e0**Mvar/M(rstar,rs,a,b,c)
    for i in range(nstars):
        s     = get_sigmalos(abs(x[i]),p0,rt,beta,rh,rs,a,b,c)
        arg1 += 0.5e0*pow(v[i]-u,2)/(pow(dv[i],2)+pow(s,2))
        arg2 += 0.5e0*math.log(2*pi*pow(dv[i],2)+pow(s,2))
    dlike  = -arg1-arg2
    return dlike

# class necessary to fit the LogLike function evaluated at its data points
class LogLH:    
    def __init__(self,data):
        self.data = data
    
    def compute(self,a,b,c):
        return LogLike(self.data,a,b,c)      # FITTED PARAMETERS: a,b,c

# integrand of the J factor, i.e. DM density profile squared
def profile(r,rho0,rs,a,b,c):
    return pow(rho0/pow(r/rs,a)/pow(1+pow(r/rs,b),(c-a)/b),2.)

#a_init,b_init,c_init = get_data(galaxy)[-1][-4:-1] # extract values from param file
data = get_data(galaxy)

# building a function object to be passed to Minuit and evaluation of -MLE parameters
lh = LogLH(data)
l = (0.1,10.)
m = Minuit(lh.compute,a=1.,b=1.,c=1.,error_a=0.01,error_b=0.01,error_c=0.01,limit_a=l,limit_b=l,limit_c=l)
bestfit = m.migrad()

a = bestfit[1][0]["value"]
b = bestfit[1][1]["value"]
c = bestfit[1][2]["value"]

'''
rt = data[4]
Mvar,rs     = data[-1][:2]
a,b,c  = 1.,1.5,5.
rho0    = 10.e0**Mvar/M(rstar,rs,a,b,c)
Jvalue = quadrature(profile,0.,rt,args=(rho0,rs,a,b,c),maxiter=150)[0]

print math.log10(Jvalue)
'''
