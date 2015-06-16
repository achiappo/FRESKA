#!/usr/bin/python
# author Andrea Chiappo		<andrea.chiappo@fysik.su.se>
import sys
import math
from iminuit import Minuit
from iminuit.util import describe
from ATminuit import get_data,get_sigmalos
from scipy.integrate import quadrature,quad
import numpy as np
import matplotlib.pyplot as plt

# paramters
pi       = math.pi
Msun     = 1.9891e30                    # Solar mass unit
Mhalo    = 1.e9 * Msun                  # Halo mass
sigma_MW = 200                          # velocity dispersion of Milky Way in km s^-1
G        = 6.67e-11*Msun                # m^3 Msun^-1 s^-2          

dwarf  = sys.argv[1]                    # get the galaxy name from the command line

#######################################################################################################
#		MAIN CODE: MINUIT MINIMISATION OF -log(LIKE) TO OBTAIN BEST-FIT PARAMETERS
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

data = get_data(dwarf)
# building a function object to be passed to Minuit and evaluation of -MLE parameters
lh = LogLike(data)
'''
kwdargs = dict(rho0=1.e7,rs=1.,error_rho0=1.e5,error_rs=0.01,limit_rho0=(1.e5,1.e9),limit_rs=(1.e-2,1.e2))
m = Minuit(lh.compute,**kwdargs)
bestfit = m.migrad()			# UNCOMMENT THIS BLOCK TO OBTAIN THE BEST-FIT PARAMETERS
'''

#######################################################################################################
#		CONSTRUCTION OF PARAMETERS GRID TO VERIFY THE NON-LOCALITY OF BEST-FIT ARRAY
#######################################################################################################

npts = 20						# parameter controlling the density of the grid
rho0_array = np.logspace(5.,9.,num=npts,dtype=float)	# build rho0 grid points 
rs_array   = np.logspace(-2.,2.,num=npts,dtype=float)	# build r_s grid points
pts = np.zeros([len(rs_array),len(rho0_array)])		# build 2D empty grid
for i,rs in enumerate(rs_array):
	for j,rho0 in enumerate(rho0_array):		# fill the grid with -log(Like)
		pts[i,j] = lh.compute(rho0,rs)		# value at each point

np.save(dwarf,pts)		# save the grid values into python-exacutable binary for plotting purposes

#######################################################################################################
#		EVALUATION OF THE J-FACTOR FROM BEST-FIT PARAMETERS
#######################################################################################################
'''
# integrand of the J factor along l.o.s.
def profile(s,phi,D,rs,a,b,c):
	r = np.sqrt(np.power(s,2)+pow(D,2)-2*s*D*np.cos(phi))
	return pow(r/rs,-2.*a)*pow(1+pow(r/rs,b),2*(a-c)/b)

# integrand of the J factor over the solid angle
def int_profile(phi,D,rs,a,b,c):
	return quad(profile,0.,D+rt,args=(phi,D,rs,a,b,c),limit=100,points=(D+rs,D))[0]*np.sin(phi)

rt,D = data[4],data[-2]
rho0 = 1.908E+08
Dphi = 2*pi*(1-math.cos(pi/360.))		# 0.008726646259971648 rad = 0.5 deg
Dphi = math.atan(rt/D)					# 0.02499479361892016 rad ~= 1.5 deg 
rs,a,b,c = 0.8035,1.,1.,3.
print int_profile(0.,D,rs,a,b,c)
#Jvalue   = 2*pi*pow(rho0,2.)*quad(int_profile,0.,Dphi,args=(D,rs,a,b,c),epsabs=1.e-4)[0]
#print math.log10(Jvalue)
'''

