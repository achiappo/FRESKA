#!/usr/bin/python
# author Andrea Chiappo		<andrea.chiappo@fysik.su.se>
import sys
import math
import yaml
from iminuit import Minuit
from iminuit.util import describe
from ATminuit import get_data,get_sigmalos
from scipy.integrate import quadrature,quad
import numpy as np

# paramters
pi       = math.pi
Msun     = 1.9891e30                    # Solar mass unit
Mhalo    = 1.e9 * Msun                  # Halo mass
sigma_MW = 200                          # velocity dispersion of Milky Way in km s^-1
G        = 6.67e-11*Msun                # m^3 Msun^-1 s^-2          

dwarf  = sys.argv[1]                    # get the galaxy name from the command line

#######################################################################################################
#			MAIN CODE: MINUIT MINIMISATION OF -log(LIKE) TO OBTAIN BEST-FIT PARAMETERS
#######################################################################################################
  
# class necessary to fit the LogLike function evaluated at its data points
class LogLike:
    def __init__(self,data):
        self.data = data
        print data

    def compute(self,rho0,rs):
        x,v,dv,rh,rt,nstars,D,pa = self.data
        beta,u    = pa[2],pa[-1]
        arg1,arg2 = 0.,0.
        a,b,c = 1.,1.,3.    # NFW
        #rcut  = pow(G*Mhalo*pow(D,2)/2./pow(sigma_MW,2),1/3.)      # truncation scale on DM density profile
        for i in range(nstars):
            s     = get_sigmalos(abs(x[i]),10**rho0,rt,rh,rs,beta,a,b,c)
            arg1 += 0.5*pow(v[i]-u,2)/(pow(dv[i],2)+pow(s,2))
            arg2 += 0.5*math.log(2*pi*(pow(dv[i],2)+pow(s,2)))
        dlike  = arg1+arg2
        return dlike

data = get_data(dwarf)
# building a function object to be passed to Minuit and evaluation of -MLE parameters
lh = LogLike(data)

kwdargs = dict(rho0=7.,rs=0.5,error_rho0=1.e-2,error_rs=1.e-2,limit_rho0=(5.,9.),limit_rs=(0.,2.))
m = Minuit(lh.compute,errordef=0.5,pedantic=False,**kwdargs)
#m.tol = 1.e-6
bestfit = m.migrad()
rho0 = bestfit[1][0]["value"]
rs   = bestfit[1][1]["value"]
yaml.dump(bestfit,open("output/%s.yaml"%dwarf,"wb"))

#######################################################################################################
#			CONSTRUCTION OF PARAMETERS GRID TO VERIFY THE NON-LOCALITY OF BEST-FIT ARRAY
#######################################################################################################

npts = 20												# parameter controlling the density of the grid
rho0_array = np.linspace(rho0-1.,rho0+1.,num=npts)  	# build rho0 grid points 
rs_array   = np.linspace(.1,rs+1.,num=npts)				# build r_s grid points
pts = np.zeros([len(rs_array),len(rho0_array)])			# build 2D empty grid
for i,rho0 in enumerate(rho0_array):
    for j,rs in enumerate(rs_array):	                # fill the grid with -log(Like)
        pts[i,j] = lh.compute(rho0,rs)					# evaluated at each point

np.save('output/'+dwarf,pts)	# save the grid values into python-exacutable binary for plotting purposes
