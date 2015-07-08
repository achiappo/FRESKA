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
    
    def compute(self,f):			# f = rho0*rs**3
    	s = self.data
        arg1,arg2 = 0.,0.
        for i in range(nstars):
            arg1 += pow(v[i]-u,2)/(pow(dv[i],2)+pow(s[i],2)*10**f)
            arg2 += math.log(2*pi*(pow(dv[i],2)+pow(s[i],2)*10**f))
        dlike  = arg1+arg2
        return dlike/2.

#######################################################################################################
# profile likelihood technique
BFf = []
val = []
num = 10**2
rs_values = np.linspace(1.e-3,10.,num=num)			# build rho0 grid points 
f_values  = np.linspace(2.,12.,num=num)				# build r_s grid points
pts = np.zeros([len(rs_values),len(f_values)])		# build 2D empty grid

for j,rs in enumerate(rs_values):					# scan over the parameters rs
	s = np.empty([0])
	for i in range(len(x)):
		s = np.append(s,get_sigmalos(abs(x[i]),rt,rh,beta,rs,a,b,c))	# evaluation of the sigma_p
	
	# building a function object to be passed to Minuit and evaluation of -MLE parameters
	lh = LogLike(s)
	m  = Minuit(lh.compute,errordef=0.5,pedantic=False,f=5.,error_f=1.e-3,limit_f=(4.+2*rs**1/3.,7.+2*rs**1/3.))
	m.tol = 1.e-7
	bestfit = m.migrad()
	BFf.append(bestfit[1][0]["value"])
	val.append(bestfit[0]["fval"])

	# generate grid
	for i,f in enumerate(f_values):
		pts[i,j] = lh.compute(f)

np.save('output/'+dwarf,pts)	# save the grid values into python-exacutable binary for plotting purposes
save = open('output/vals_%s.dat'%dwarf,'w')
save.write(('%6s %10s %10s %s')%('rs','rho0*rs^3','-logLike','\n'))
for rs,f,val in zip(rs_values,BFf,val):
	save.write('%6.3f %10.2f %10.4f %s'%(rs,f,val,'\n'))
save.close()

