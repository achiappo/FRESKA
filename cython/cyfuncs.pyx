from exceptions import Exception, ValueError, OverflowError, ZeroDivisionError
from scipy.special import betainc, hyp2f1, gamma
from math import pi, atan, asin, sqrt
import numpy as np

# cythonizable functions
##############################################################################
# stellar component

def _inv_csch(double x):
	# inverse hyperbolic cosecant (used for c* = 1 , non-Plum)
	return np.log( np.sqrt( 1.+x**(-2.) ) + 1./x )

def plummer0_func(double x) :
	return 4. / 3. / (1.+x*x)**2

def plummer1_func(double x) :
	return ((2+x*x)*_inv_csch(x) - np.sqrt(1+x*x))/(1+x*x)**1.5

##############################################################################
# DM component

def zhao_func(double x, double a, double b, double c):
	try:
		return 1. / x**c / (1.+x**a)**((b-c) / a)
	except (OverflowError, ZeroDivisionError):
		return np.nan

def mass_func(double x, double a, double b, double c):
	try:
		F = hyp2f1( (3-c)/a, (b-c)/a, (a-c+3)/a, -x**a )
		return x**(3-c) * F / (3-c)
	except (OverflowError, ZeroDivisionError):
		return np.nan

def radius(double z, double y, double Dprime):
	try:
		return sqrt( z*z + Dprime*Dprime*(1-y*y) )
	except (OverflowError, ZeroDivisionError):
		return np.nan

##############################################################################
# Mamon-Lokas kernel functions

def func_isotropic_kernel(double r, double R):
	u = r / R
	return sqrt( 1. - u**(-2) )

def func_radial_kernel(double r, double R):
	u = r / R
	return pi*u/4. - 0.5*sqrt(1. - u**(-2)) - u*asin(1./u)/2.

def func_constant_kernel(double r, double R, double beta):
	u = r / R
	ker1 = sqrt( 1.-u**(-2.) ) / (1.-2.*beta)
	ker2 = sqrt(pi)/2. * gamma(beta-0.5)/gamma(beta) * (1.5-beta)
	ker3 = u**( 2*beta-1. ) * ( 1.-betainc( u**(-2), beta+0.5, 0.5) )
	return ker1 + ker2 * ker3

def func_OM_kernel(double r, double R, double ra):
	u = r / R
	w = ra / R
	ker1 = (w*w + 0.5) * (u*u + w*w) / u / (w*w + 1.)**1.5
	ker2 = atan( sqrt( (u*u - 1.) / (w*w + 1.) ) )
	ker3 = 0.5 / (w*w + 1.) * sqrt( 1. - u**(-2.) )
	return ker1 * ker2 - ker3
