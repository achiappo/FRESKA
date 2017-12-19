from exceptions import Exception, ValueError, OverflowError, ZeroDivisionError
from math import pi, atan, asin, sqrt
import numpy as np
from scipy import integrate as sciint
from scipy import special as scispec

# cythonizable functions
##############################################################################
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
        return ((2+x*x)*_inv_csch(x) - np.sqrt(1+x*x))/(1+x*x)**1.5
    except (OverflowError, ZeroDivisionError, ValueError):
        return np.nan

##############################################################################
# DM component

def zhao_func(double x, double a, double b, double c):
    try:
        return 1. / x**c / (1.+x**a)**((b-c) / a)
    except (OverflowError, ZeroDivisionError, ValueError):
        return np.nan

def zhao_mass_integral(double x, double a, double b, double c):
    res = sciint.quad(zhao_func, 0,x, args=(a, b-2,c-2))
    return res

def mass_func(double x, double a, double b, double c):
    F = scispec.hyp2f1( (3-c)/a, (b-c)/a, (a-c+3)/a, -x**a )
    if np.isinf(F):
        F,errF = zhao_mass_integral(x,a,b,c)
        if F<0 or errF>F:
            print "Warning: quadpack integration failed for robust_zhao_mass"
            F=np.nan
    else:
        F *= x**(3-c) / (3-c)
    return F

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
		I(y,a,b) = I(y,a,b+1) - \frac{\Gamma(a+b)}{b\Gamma(b)\Gamma(a)}\left[u^a(1-u)^b\right]_0^{y}
		$$
    Compared to the proposal by Mamon&Lokas 2004, which is not in practice 
    applicable, inverting the beta integral allows to make sure that u^b 
    is not infinite at 0.
    """
    b=bb
    gamma=scispec.gamma
    if b<0 and (int(b+0.5)==b+0.5 or int(b)==b):
        return np.inf
    if b+0.5>0:
        B = scispec.betainc(0.5, b+0.5, y)
    else:
        G = gamma(b+1) / gamma(0.5) / gamma(b+0.5)
        B = modified_betainc(y, b+1) - G * (1-y)**(b+0.5) * np.sqrt(y) / (b+0.5)
    return B
    
    
def func_OM_kernel(double r, double R, double ra):
    u = r / R
    w = ra / R
    try:
        ker1 = (w*w + 0.5) * (u*u + w*w) / u / (w*w + 1.)**1.5
        ker2 = atan( sqrt( (u*u - 1.) / (w*w + 1.) ) )
        ker3 = 0.5 / (w*w + 1.) * sqrt( 1. - u**(-2.) )
        return ker1 * ker2 - ker3
    except (OverflowError, ZeroDivisionError, ValueError):
        return np.nan

