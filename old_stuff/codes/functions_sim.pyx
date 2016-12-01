from __future__ import division
from scipy.integrate import quad,nquad
from math import sqrt,cos, log, pi
import numpy as np

###################################################################################################
#					FUNCTIONS DEFINITIONS
###################################################################################################
# stellar density profile 
def nu(double r, double rh):
	#return (r/rh)**(-.1)*(1+(r/rh)**2)**(-2.45) #(used for gamma* = 0.1 , Plum)
	return (r/rh)**-1*(1+(r/rh)**2)**-2 #(used for gamma* = 1 , non-Plum)

##########################################################################
# Mass of cusped NFW DMH
def get_M_NFW_cusp(double x):
	return np.log(1.+x)-x/(1.+x)

# Mass of cored NFW DMH
def get_M_NFW_core(double x):
	return np.log(1.+x)-(2.*x+3.*x**2)*0.5/(1.+x)**2

##########################################################################
# numerical integrals in sigma_los

def integrand1(double s, double rh, double r0):
	#result = nu(s,rh)*get_M_NFW_cusp(s/r0)/s**2	# for Cusped NFW
	result = nu(s,rh)*get_M_NFW_core(s/r0)/s**2 	# for Cored NFW
	return result

def integral1(double smin, double rh, double r0):
	res,err = quad(integrand1,smin,+np.inf,args=(rh, r0),epsabs=1.e-3,epsrel=1.e-3,limit=100)
	return res

def integrand2(double z, double rh, double r0, double R):
	result = z/np.sqrt(z*z-1.)
	res = integral1(z*R,rh,r0)
	return result * res

def integral2(double r0, double rh, double R):
	res,err = quad(integrand2,1.,+np.inf,args=(rh,r0,R),epsabs=1.e-3,epsrel=1.e-3,limit=100)
	return res

##########################################################################
# jfactor evaluation functions

def func(double u, double y, double D, double rt, double ymin):
	return (1.+u)**(-4)/u/sqrt(u*u-D**2*(1-y*y))

def lim_u(double y, double D, double rt, double ymin):
	return [D*sqrt(1-y*y), rt]

def lim_y(double D, double rt, double ymin):
	return [ymin,1]

def Jfactor(double D, double rt, double r0, double rho0, double tmax):
	"""
	returns the Jfactor computed in the solid angle of
	semi apex angle tmax, in degree, for a NFW halo profile of 
	shape parameters (r0,rho0) at distance D. 
	rt is the maximal radius of integration 
	D, r0 and rt are in kpc, and rho0 is in Msun.kpc^-3
	"""
	ymin=cos(np.radians(tmax))
	Dprime=D/r0
	rtprime=rt/r0
	Msun2kpc5_GeVcm5 = 4463954.894661358
	cst = 4*pi*rho0**2*r0*Msun2kpc5_GeVcm5
	res = nquad(func, ranges=[lim_u, lim_y], args=(Dprime,rtprime,ymin))
	return cst*res[0]

