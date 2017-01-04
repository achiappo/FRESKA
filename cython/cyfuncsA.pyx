from exceptions import Exception, ValueError, OverflowError, ZeroDivisionError
from scipy.special import betainc, hyp2f1, gamma
from scipy.integrate import quad, nquad
from math import pi, atan, asin, sqrt
import numpy as np
from cpython cimport array

# functions cythonized
def _inv_csch(double x):
	# inverse hyperbolic cosecant (used for c* = 1 , non-Plum)
	return np.log( np.sqrt( 1.+1./x/x ) + 1./x )

def plummer1_func(double x) :
	return ((2+x*x)*_inv_csch(x) - np.sqrt(1+x*x))/(1+x*x)**1.5

def zhao_func(double x, double a, double b, double c):
	try:
		return 1. / x**c / (1.+x**a)**((b-c) / a)
	except (OverflowError, ZeroDivisionError):
		return np.nan

def mass_func(double x, double a, double b, double c):
	try:
		H = hyp2f1((3.-c)/a, (b-c)/a, (a-c+3.)/a, -x**a)
		return x**(3.-c) * H / (3.-c)
	except (OverflowError, ZeroDivisionError):
		return np.nan

##############################################################################

ctypedef double (*f_type)(double)

def _radius(double z, double y, double Dprime):
	try:
		return sqrt( z*z + Dprime*Dprime*(1-y*y))
	except (OverflowError, ZeroDivisionError):
		return np.nan

def _limu(double y, double rtprime, double Dprime):
	bounds = [0, sqrt(rtprime**2 - Dprime**2*(1-y*y))]
	cdef array.array bounds_arr = array.array('d', bounds)
	return bounds_arr

def _limy(double ymin):
	bounds = [ymin,1.]
	cdef array.array bounds_arr = array.array('d', bounds)
	return bounds_arr

def _integrand(f_type density, double z, double y, double D):
	try:
		return density( _radius(z, y, D) )**2
	except (OverflowError, ZeroDivisionError):
		return np.nan

def integral(f_type density, double ymin, double rtprime, double Dprime):
	integrand = lambda z, y : _integrand(density, z, y, Dprime)
	lim_u = lambda y : _limu(y, rtprime, Dprime)
	lim_y = lambda : _limy(ymin)
	res = nquad(integrand, ranges=[lim_u, lim_y], \
				opts=[{'limit':1000, 'epsabs':1.e-3, 'epsrel':1.e-3},\
						{'limit':1000, 'epsabs':1.e-3, 'epsrel':1.e-3}])
	return res

##############################################################################
# Mamon-Lokas kernel functions

def func_isotropic_kernel(double r, double R):
	u = r / R
	return sqrt(1.-1./u/u)

def func_radial_kernel(double r, double R):
	u = r / R
	return pi*u/4. - 0.5*sqrt(1. - 1./u/u) - u*asin(1./u)/2.

def func_constant_kernel(double r, double R, double beta):
	u = r / R
	ker1 = sqrt(1.-1./u/u) / (1.-2.*beta)
	ker2 = sqrt(pi)/2. * gamma(beta-0.5)/gamma(beta) * (1.5-beta)
	ker3 = u**(2*beta-1) * (1.-betainc(1./u/u, beta+0.5, 0.5))
	return ker1 + ker2 * ker3

def func_OM_kernel(double r, double R, double ra):
	u = r / R
	w = ra / R
	ker1 = (w*w + 0.5) * (u*u + w*w) / u / (w*w + 1.)**1.5
	ker2 = atan( sqrt( (u*u - 1.) / (w*w + 1.) ) )
	ker3 = 0.5 / (w*w + 1.) * sqrt(1. - 1./u/u)
	return ker1 * ker2 - ker3
