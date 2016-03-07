from __future__ import division
from scipy.integrate import quad,nquad
from math import sqrt,cos, log, pi
import numpy as np

###################################################################################################
#					FUNCTIONS DEFINITIONS
###################################################################################################
# stellar density profile 
def nu(double s):
	return 1./s**0.1/(1+s**2)**2.45 #(used for gamma* = 0.1 , Plum)
	#return 1./s/(1+s**2)**2 #(used for gamma* = 1 , non-Plum)

##########################################################################
# Mass of cusped NFW DMH
def get_M_NFW_cusp(double x):
	return np.log(1.+x)-x/(1.+x)

# Mass of cored NFW DMH
def get_M_NFW_core(double x):
	return np.log(1.+x)-(2.*x+3.*x**2)*0.5/(1.+x)**2

##########################################################################
# numerical integrals in sigma_los

def integrand1(double y, double alpha):
	#return nu(y)*get_M_NFW_cusp(y*alpha)/y**2		# for Cusped NFW
	return nu(y)*get_M_NFW_core(y*alpha)/y**2 		# for Cored NFW

def integral1(double smin, double alpha):
	return quad(integrand1,smin,+np.inf,args=(alpha),epsabs=1.e-2,epsrel=1.e-2)[0]

def integrand2(double z, double gamma, double alpha):
	return integral1(z*gamma, alpha)*z/np.sqrt(z*z-1.)

def integral2(double gamma, double alpha):
	return quad(integrand2,1.,+np.inf,args=(gamma, alpha),epsabs=1.e-2,epsrel=1.e-2)[0]

##########################################################################
# jfactor evaluation functions

def func(double u, double y, double D, double rt, double ymin):
	#return 1./(1.+u)**4/u/sqrt(u*u-D**2*(1.-y*y))	# for Cusped NFW
	return u/(1.+u)**6/sqrt(u*u-D**2*(1.-y*y))	 	# for Cored NFW

def lim_u(double y, double D, double rt, double ymin):
	return [D*sqrt(1.-y*y), rt]

def lim_y(double D, double rt, double ymin):
	return [ymin,1.]

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

