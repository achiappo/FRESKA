#!/usr/bin/python
# author Andrea Chiappo		<andrea.chiappo@fysik.su.se>
import sys
import math
import yaml
from iminuit import Minuit
from iminuit.util import describe
from AT_profile_minuit import get_data,get_sigmalos
import numpy as np

# paramters
pi       = math.pi
Msun     = 1.9891e30                    # Solar mass unit
Mhalo    = 1.e9 * Msun                  # Halo mass
sigma_MW = 200                          # velocity dispersion of Milky Way in km s^-1
G        = 6.67e-11*Msun                # m^3 Msun^-1 s^-2          

dwarf  = sys.argv[1]                    	# get the galaxy name from the command line

x,v,dv,rh,rt,nstars,D,pa = get_data(dwarf) 	#	extraction of data relative 
beta,u = pa[2],pa[-1] 						#	to the examined dwarf
a,b,c = 1.,1.,3.    # NFW

#######################################################################################################
# class necessary to fit the LogLike function evaluated at its data points

class LogLike:
    def __init__(self,data):
        self.data = data
    
    def compute(self,rho0):
    	s = self.data
        arg1,arg2 = 0.,0.
        for i in range(nstars):
            arg1 += pow(v[i]-u,2)/(pow(dv[i],2)+pow(s[i],2)*10**rho0)
            arg2 += math.log(2*pi*(pow(dv[i],2)+pow(s[i],2)*10**rho0))
        dlike  = arg1+arg2
        return dlike/2.

#######################################################################################################
# profile likelihood technique
BFrho0 = []
for rs in np.linspace(0.01,2.,num=50):									# scan over the parameters rs
	s = np.empty([0])
	for i in range(len(x)):
		s = np.append(s,get_sigmalos(abs(x[i]),rt,rh,beta,rs,a,b,c))	# evaluation of the sigma_p
	
	# building a function object to be passed to Minuit and evaluation of -MLE parameters
	lh = LogLike(s)
	m = Minuit(lh.compute,errordef=0.5,pedantic=False,rho0=7.,error_rho0=1.e-2,limit_rho0=(5.,9.))
	bestfit = m.migrad()
	BFrho0.append(bestfit[1][0]["value"])
	
for rs,rho0 in zip(np.linspace(0.01,2.,num=50),BFrho0):
	print rs,rho0 


