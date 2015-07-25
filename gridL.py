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
dwarf  = sys.argv[1]                    	# get the galaxy name from the command line

x,v,dv,rh,rt,nstars,D,pa = get_data(dwarf) 	#	extraction of data relative 
beta,u = pa[2],pa[-1] 						#	to the examined dwarf
a,b,c = 1.,1.,3.    # NFW

#######################################################################################################
# class necessary to fit the LogLike function evaluated at its data points

class LogLike(object):
    def __init__(self,data):
        self.data = data
    
    def compute(self,rhos):
    	s = self.data
        Sigma = dv*dv+s*s*rhos
        dlike  = (np.log(2*pi*Sigma)+(v-u)**2/Sigma).sum()
        return dlike/2.

#######################################################################################################
# profile likelihood technique
rhos_values = np.logspace(6.,9.,100)						# build r_s grid points
rs_values   = np.logspace(np.log10(0.1),np.log10(5.),100)	# build rho0 grid points 
pts = np.zeros([len(rhos_values),len(rs_values)])			# build 2D empty grid

for j,rs in enumerate(rs_values):							# scan over the parameters rs
	s = np.empty([0])
	for i in range(nstars):
		s = np.append(s,get_sigmalos(abs(x[i]),rh,beta,rs,a,b,c))	# evaluation of the sigma_p
	
	# building a function object to be passed to Minuit and evaluation of -MLE parameters
	L = LogLike(s)
	# generate grid
	for i,rhos in enumerate(rhos_values):
		pts[i,j] = L.compute(rhos)
		print '%4.0f %8.3f %4.0f %12.3f %10.4f'%(j+1,rs,i+1,rhos,pts[i,j])

np.save('output/Lgrid_'+dwarf,pts)	# save the grid values into python-exacutable binary for plotting purposes
