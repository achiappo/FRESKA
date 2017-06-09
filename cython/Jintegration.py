import numpy as np
from math import cos, sqrt
from scipy import integrate
from cyfuncs import zhao_func
from itertools import product
from exceptions import Exception, ValueError, OverflowError, ZeroDivisionError


rh = 0.04
D = 40.

theta = 2*rh/D
rhosat = 1e19
ymin = cos( np.radians(theta) )

#-----------------------------------------------------------------------------

def density(x, a, b, c, r0):
    if c>1e-5:
        if x > r0*(1e-10)**(1/c):
            return 2*zhao_func(x, a, b, c)
        else:
            return rhosat
    else:
        return 2*zhao_func(x, a, b, 0.)

def radius(z, y, D):
    try:
        return sqrt( z*z + D*D * (1 - y*y ) )
    except (OverflowError, ZeroDivisionError, ValueError):
        return np.nan

def integrand(x, y, ymin, rt, D, a, b, c, r0):
    try:
        return density( radius(x, y, D), a, b, c, r0 )**2
    except (OverflowError, ZeroDivisionError):
        return np.nan

def lim_l(y, ymin, rt, D, a, b, c, r0):
    l_min = 0#- sqrt( rt*rt - D*D * (1. - y*y) )
    l_max = + sqrt( rt*rt - D*D * (1. - y*y) )
    return [l_min, l_max]

def lim_y(ymin, rt, D, a, b, c, r0):
    return [ymin, 1.]

#-----------------------------------------------------------------------------

R = [-3, 2]
A = [0.5, 3]
B = [1, 10]
C = [0, 1.5]


print '%5s %5s %5s %5s %10s %10s %8s %8s'%\
		('a', 'b', 'c', 'r0', 'I', 'Ierr', 'I>Ierr', 'DIerr')

for p in product( range(2), repeat=4 ):
	r0, a, b, c = R[p[0]], A[p[1]], B[p[2]], C[p[3]]
	I = integrate.nquad(integrand, ranges=[lim_l, lim_y], 
	                    args=(ymin, np.inf, D/10**r0, a, b, c, 10**r0), 
	                    opts=[{'limit':1000, 'epsabs':1.e-6, 'epsrel':1.e-8},
	                    	  {'limit':1000, 'epsabs':1.e-6, 'epsrel':1.e-8}]) 

	print '%5.2f %5.2f %5.2f %5.2f %10.2e %10.2e %8s %10.3e'%\
	(r0, a, b, c, I[0], I[1], I[0]>I[1], round(I[1]/I[0],3))
