
__author__ = "Johann Cohen Tanugi, Andrea Chiappo"
__email__ = "chiappo.andrea@gmail.com"

##############################################################################
# module containing cythonizable functions
# to speed up evaluation of core mathematical calculations via C language
# to compile with the following command:
# python setup.py build_ext --inplace

from exceptions import Exception, ValueError, OverflowError, ZeroDivisionError
from math import pi, atan, asin, sqrt
import numpy as np
from scipy import integrate as sciint
from scipy import special as scispec

# stellar component

def _inv_csch(double x):
	# inverse hyperbolic cosecant (used for c* = 1 , non-Plum)
	try:
		return np.log( np.sqrt( 1.+x**(-2.) ) + 1./x )
	except (OverflowError, ZeroDivisionError, ValueError):
		return np.nan

def plummer0_func(double x) :
	try:
		return 4. / 3. / (1.+x*x)**2
	except (OverflowError, ZeroDivisionError, ValueError):
		return np.nan

def plummer1_func(double x) :
	try:
		return ((2+x*x)*_inv_csch(x) - np.sqrt(1+x*x)) / (1+x*x)**1.5
	except (OverflowError, ZeroDivisionError, ValueError):
		return np.nan

##############################################################################
# DM component

def zhao_func(double x, double a, double b, double c):
	try:
		return 1. / x**c / (1.+x**a)**((b-c) / a)
	except (OverflowError, ZeroDivisionError, ValueError):
		return np.nan

def mass_func(double x, double a, double b, double c):
	try:
		F = scispec.hyp2f1( (3-c)/a, (b-c)/a, (a-c+3)/a, -x**a )
		return x**(3-c) * F / (3-c)
	except (OverflowError, ZeroDivisionError, ValueError):
		return np.nan

def radius(double z, double y, double Dprime):
	try:
		return sqrt( z*z + Dprime*Dprime*(1-y*y) )
	except (OverflowError, ZeroDivisionError, ValueError):
		return np.nan

##############################################################################
# Mamon-Lokas kernel functions

def func_isotropic_kernel(double r, double R):
	u = r / R
	try:
		return sqrt( 1. - u**(-2) )
	except (OverflowError, ZeroDivisionError, ValueError):
		return np.nan

def func_radial_kernel(double r, double R):
	u = r / R
	try:
		return pi*u/4. - 0.5*sqrt(1. - u**(-2)) - u*asin(1./u)/2.
	except (OverflowError, ZeroDivisionError, ValueError):
		return np.nan

def func_OM_kernel(double r, double R, double ra):
	u = r / R
	w = ra / R
	try:
		ker1 = (w*w + 0.5) * (u*u + w*w) / u / (w*w + 1.)**1.5
		ker2 = atan( sqrt( (u*u - 1.) / (w*w + 1.) ) )
		ker3 = 0.5 * sqrt(1. - u**(-2.)) / (w*w + 1.)
		return ker1 * ker2 - ker3
	except (OverflowError, ZeroDivisionError, ValueError):
		return np.nan

def func_constant_kernel(double r, double R, double beta):
	u = r / R
	try:
		ker1 = sqrt( 1.-u**(-2.) ) / (1.-2.*beta)
		ker2 = sqrt(pi)/2. * scispec.gamma(beta-0.5)/scispec.gamma(beta) * (1.5-beta)
		ker3 = u**( 2*beta-1. ) * modified_betainc( 1-u**(-2), beta)
		return ker1 + ker2 * ker3
	except (OverflowError, ZeroDivisionError, ValueError):
		return np.nan

def modified_betainc(double y, double bb):
	"""
	this custom function makes use of the equality
	I(y=1-x,b,a) = 1-I(x,a,b)
	where I is the regularized incomplete beta function, and a=bb+0.5, b=0.5. 
	It prolongs the 
	scipy version analytically by using the recursion relation
	$$
	I(y,b,a) = I(y,b,a+1) - \frac{\Gamma(a+b)}{\Gamma(b)\Gamma(a+1)}\left[u^b(1-u)^a\right]_0^{y}
	$$
	Compared to the proposal by Mamon&Lokas 2004, which is not in practice 
	applicable, inverting the beta integral allows to make sure that u^b 
	is not infinite at 0.
	"""
	b = bb
	if b<0 and (int(b+0.5)==b+0.5 or int(b)==b):
		return np.inf
	if b+0.5>0:
		B = scispec.betainc(0.5, b+0.5, y)
	else:
		G = scispec.gamma(b+1) / scispec.gamma(0.5) / scispec.gamma(b+1.5)
		B = -G * (1-y)**(b+0.5) * np.sqrt(y) + modified_betainc(y, b+1)
	return B
