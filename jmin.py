#!/usr/bin/python
import sys
import math
from iminuit import Minuit
from iminuit.util import describe
from ATminuit import get_data,get_sigmalos
from scipy.integrate import quadrature
import numpy as np

# paramters
pi       = math.pi
Msun     = 1.9891e30                    # Solar mass unit
Mhalo    = 1.e9 * Msun                  # Halo mass
sigma_MW = 200                          # velocity dispersion of Milky Way in km s^-1
G        = 6.67e-11*Msun                # m^3 Msun^-1 s^-2          

galaxy  = sys.argv[1]                   # get the galaxy name from the command line

#######################################################################################################
#                                               MAIN CODE
#######################################################################################################
  
# class necessary to fit the LogLike function evaluated at its data points
class LogLike:
    def __init__(self,data):
        self.data = data
    
    def compute(self,rho0,rs):
        x,v,dv,rh,rt,nstars,D,pa = self.data
        beta,u    = pa[2],pa[-1]
        arg1,arg2 = 0.,0.
        a,b,c = 1.,1.,3.
        #rcut  = pow(G*Mhalo*pow(D,2)/2./pow(sigma_MW,2),1/3.)      # truncation scale on DM density profile
        for i in range(nstars):
            s     = get_sigmalos(abs(x[i]),rho0,rt,rh,beta,rs,a,b,c)
            arg1 += 0.5e0*pow(v[i]-u,2)/(pow(dv[i],2)+pow(s,2))
            arg2 += 0.5e0*math.log(2*pi*(pow(dv[i],2)+pow(s,2)))
        dlike  = arg1+arg2
        return dlike

data = get_data(galaxy)
'''
# building a function object to be passed to Minuit and evaluation of -MLE parameters
lh = LogLike(data)
kwdargs = dict(rho0=1.e7,rs=1.,error_rho0=0.01,error_rs=0.01,limit_rho0=(1.e5,1.e9),limit_rs=(1.e-3,1.e2))
m = Minuit(lh.compute,**kwdargs)
bestfit = m.migrad()
'''

#rho0 = bestfit[1][0]["value"]
#rs   = bestfit[1][1]["value"]
#beta = bestfit[1][2]["value"]
#a    = bestfit[1][3]["value"]
#b    = bestfit[1][4]["value"]
#c    = bestfit[1][5]["value"]

# integrand of the J factor along l.o.s.
def profile(s,phi,D,rs,a,b,c):
	r = np.sqrt(pow(s,2)+pow(D,2)-2*s*D*np.cos(phi))
	return pow(r/rs,-2.*a)*pow(1+pow(r/rs,b),2*(a-c)/b)

# integrand of the J factor over the solid angle
def int_profile(phi,D,rs,a,b,c):
	return quadrature(profile,0.,D+rt,args=(phi,D,rs,a,b,c))[0]*np.sin(phi)

rt,D = data[4],data[-2]
rho0    = 1.879E+08
theta = pi/360.
Dphi = 2*pi*(1-math.cos(0.5))#(1-math.cos(theta))	# I KNOW IT'S NOT RIGHT, BUT THE CORRECT VERSION
rs,a,b,c = 1.071,1.,1.,4.				# LEADS TO A PYTHON ValueError
Jvalue  = 2*pi*pow(rho0,2.)*quadrature(int_profile,0.,Dphi,args=(D,rs,a,b,c))[0]
print math.log10(Jvalue)
