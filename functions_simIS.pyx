from __future__ import division
from scipy.integrate import quad,nquad
from math import sqrt,cos, log, pi
import numpy as np
cimport numpy as np

###############################################################################################################################
#											FUNCTIONS DEFINITIONS
###############################################################################################################################
# stellar density profile 
def nu(double s, int mod):
	return 1./s/(1+s**2)**2 if mod==0 or mod==2 else 1./s**0.1/(1+s**2)**2.45

######################################################################################################################
# Mass of cusped NFW DMH
def get_M_NFW(double x, int mod):
	return np.log(1.+x)-(2.*x+3.*x**2)*0.5/(1.+x)**2 if mod==0 or mod==1 else np.log(1.+x)-x/(1.+x)

######################################################################################################################
# numerical integrals in sigma_los

def integrand1(double y, double alpha, int mod):
	return nu(y, mod)*get_M_NFW(y*alpha, mod)/y**2

def integral1(double smin, double alpha, int mod):
	return quad(integrand1,smin,+np.inf,args=(alpha, mod),epsabs=1.e-2,epsrel=1.e-2)[0]

def integrand2(double z, double gamma, double alpha, int mod):
	return integral1(z*gamma, alpha, mod)*z/np.sqrt(z*z-1.)

def integral2(double gamma, double alpha, int mod):
	return quad(integrand2,1.,+np.inf,args=(gamma, alpha, mod),epsabs=1.e-2,epsrel=1.e-2)[0]

def proxy_integral2(tuple args):
	return integral2(args[0], args[1], args[2])

DTYPE=np.double
ctypedef np.double_t DTYPE_t
def compute_I_array(np.ndarray[DTYPE_t, ndim=1] gamma_array, double rh_over_r0, int mod):
	cdef np.ndarray[DTYPE_t, ndim=1] res = np.zeros_like(gamma_array)
	cdef double gamma
	cdef int i
	cdef int n = len(gamma_array)
	for i in range(n):
		res[i] = integral2(gamma_array[i],rh_over_r0, mod)
	return res

######################################################################################################################
# jfactor evaluation functions

def func(double u, double y, double D, double rt, double ymin, int mod):
	return u/(1.+u)**6/sqrt(u*u-D**2*(1.-y*y)) if mod==0 or mod==1 else 1./(1.+u)**4/u/sqrt(u*u-D**2*(1.-y*y))

def lim_u(double y, double D, double rt, double ymin, int mod):
	return [D*sqrt(1.-y*y), rt]

def lim_y(double D, double rt, double ymin, int mod):
	return [ymin,1.]

def Jfactor(double D, double rt, double r0, double rho0, double tmax, int mod):
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
	res = nquad(func, ranges=[lim_u, lim_y], args=(Dprime, rtprime, ymin, mod), opts=[{'limit':1000},{'limit':1000}])
	return cst*res[0]
