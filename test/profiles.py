from exceptions import Exception, ValueError
from scipy.special import betainc, kn, hyp2f1, gammainc
from scipy import integrate as sciint
from math import cos
import numpy as np

##############################################################################
# inverse hyperbolic cosecant (used for gamma* = 1 , non-Plum)
def inv_csch(x):
    return np.log(np.sqrt(1+x**-2.)+x**-1.)

def plummer1_func(x) :
    x2 = x*x
    return ((2+x2)*inv_csch(x) - np.sqrt(1+x2))/(1+x2)**1.5

def zhao_func(x, a, b, c):
    return 1. / x**c / (1.+x**a)**((b-c) / a)

##############################################################################

class Profile(object):
    """
    Base class for DM or stellar profiles. Only a density function is
    expected for both. The base class just loads the arguments passed at
    instantiation.
    """
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

    def density(self, *args, **kwargs):
        return Exception('Not implemented')

class StellarProfile(Profile):
    """
    Define a stellar profile, for which the 2 universal parameters are
    - rh : scale radius
    - rhoh : scale density, at scale radius
    This is still a base class, but it implements the Abel transform for 
    the surface brightness computation, as a default for inherited classes.
    """
    def __init__(self, **kwargs):
        """
        ensure existence of rh and rhoh attributes
        """
        super(StellarProfile, self).__init__(**kwargs)        
        self.rh =  kwargs['rh'] if 'rh' in kwargs else 1
        self.rhoh = kwargs['rhoh'] if 'rhoh' in kwargs else 1
        self.params = ['rh', 'rhoh']

    def surface_brightness(self, *args, **kwargs):
        """
        Compute the surface brightness from the density, 
        using the Abel transform
        """
        R = kwargs['R']
        rh = kwargs['rh']
        if np.isscalar(R):
            integrand = lambda r,rh,R: self.density(r/rh)*r/np.sqrt(r**2-R**2)
            return 2*quad(integrand, R, +np.inf, args=(rh,R))[0] 
        else :
            res = np.zeros_like(R)
            for i, RR in enumerate(R):
                integrand = lambda r,rh,RR: self.density(r/rh)*r/np.sqrt(r**2-RR**2)
                res[i] = 2*quad(integrand, RR, +np.inf, args=(rh,RR))[0] 
            return res
        
class genPlummerProfile(StellarProfile):
    """
    Generalized Plummer stellar profile, where the exponent of the central 
    slope can be different from 0, the standard Plummer profile's value.
    """
    def __init__(self, **kwargs):
        """
        Use the Zhao general formula for the Plummer density, with exponents
        (a,b,c) = (2, 5, c), c been defaulted to 0 (standard Plummer) if not
        provided.
        """
        super(genPlummerProfile, self).__init__(**kwargs)
        if 'a' in kwargs or 'b' in kwargs:
            print "exponent parameters a and b are fixed to 2 and 5, \
            respectively, in generalized Plummer profiles. Use Zhao \
            profiles instead."
        self.a = 2
        self.b = 5
        if 'c' not in kwargs:
            self.c = 0 #standard Plummer
        self.params += ['c']

    def density(self, x):
        """
        return the stellar density.
        input : x=r/rh (can be array-like)
        output : rhoh * rh * x**(-c) * (1.+x**2)**(-(5-c)/2)
        """
        return self.rhoh  * self.rh * zhao_func(x, self.a, self.b, self.c)
    
    def surface_brightness(self, x):
        """
        Return the analytical solution for the brightness profile of a Plummer density
        input : x=R/rh
        output : rhoh * rh * (1+x*x)**(-2) if c==0 ,
        rhoh * rh * ((2+x**2)*inv_csch(x) - np.sqrt(1+x**2))/(1+x**2)**1.5 if c==1
        otherwise, default to base class Abel integration.
        """
        result = self.rhoh * self.rh
        c = self.c
        if c == 0: #standard Plummer
            return result * (1+x*x)**(-2)
        elif c == 1:
            return result * plummer1_func(x)
        else :
            return super(genPlummerProfile, self).surface_brightness(rh=self.rh, R=x*self.rh)

class DMProfile(Profile):
    def __init__(self, **kwargs):
        super(DMProfile, self).__init__(**kwargs)
        if 'r0' not in kwargs:
            self.r0 = 1
        if 'rho0' not in kwargs:
            self.rho0 = 1
        self.params=['r0', 'rho0']

    def reducedJ(self, D, theta, rt, with_errs=False):
        """
        compute the reduced J factor \int_ymin^1 dy \int_0^zmax dx f^2(r(z,y)),
        where 
        - ymin = \cos(\theta_max)
        - z = r/r0 - D'y
        - the density reads rho(r)=rho0*f(r/r0)
        - zmax = sqrt(r_t'**2-D'^2*(1-y^2)) with r_t'=rt/r0 and D'=D/r0.
        The usual J factor is recovered by multiplying by 4pi*rho0^2*r0. The 
        reduced J factor is dimensionless
        """
        r0 = self.r0
        Dprime = D/r0
        rtprime = rt/r0
        ymin = cos(np.radians(theta))
        def radius(z,y):
            return sqrt( z*z + Dprime**2*(1-y*y)) 
        def integrand(z,y):
            return self.density(radius(z,y))**2
        def lim_u(y):
            return [0, sqrt(rtprime**2 - Dprime**2*(1-y*y))]
        def lim_y():
            return [ymin,1.]

        res = sciint.nquad(integrand, ranges=[lim_u, lim_y], \
                    opts=[{'limit':1000, 'epsabs':1.e-10, 'epsrel':1.e-10},\
                          {'limit':1000, 'epsabs':1.e-10, 'epsrel':1.e-10}])
        if with_errs:
            return res[0], res[1]
        else:
            return res[0]

    def Jfactor(self, D, theta, rt, with_errs=False):
        Msun2kpc5_GeVcm5 = 4463954.894661358
        cst = 4*pi*rho0**2*r0*Msun2kpc5_GeVcm5
        return cst * self.reducedJ(D, theta, rt, with_errs=False)

class  ZhaoProfile(DMProfile):
    def __init__(self, **kwargs):
        super(ZhaoProfile, self).__init__(**kwargs)
        #default to NFW
        if 'a' not in kwargs:
            self.a = 1
        if 'b' not in kwargs:
            self.b = 3
        if 'c' not in kwargs:
            self.c = 1
        self.params+=['a','b','c']
        
    def density(self,x):
        a, b, c = self.a, self.b, self.c
        return zhao_func(x, a, b, c)

    def mass(self, x):
        a, b, c = self.a, self.b, self.c
        return x**(3.-c) * hyp2f1((3.-c)/a, (b-c)/a, (a-c+3.)/a, -x**a) / (3.-c)

###########################################################################
#Anisotropy kernels
class IsotropicKernel(object):
    """
    Mamon-Lokas integral for isotropic kernels
    """
    def __init__(self, **kwargs):
        self.__dict__ = kwargs
        #isotropic kernel does not take any new parameters
        self.params ={}
    def __call__(self, u):
        """
        return the isotropic kernel for u=r/R, r is a running radius 
        (the variable in the integrand), and R is the star radius (data)
        """
        mod=self.model
        if mod == 'iso':
            return sqrt(1.-u**(-2))

###########################################################################
#Helper functions
def build_profile(profile_type, **kwargs):
    if profile_type.upper() == 'PLUMMER':
        return PlummerProfile(**kwargs)
    elif profile_type.upper() == 'NFW':
        return ZhaoProfile(a=1, b=3, c=1, **kwargs)
    elif profile_type.upper() == 'ZHAO':
        if not set(['a', 'b', 'c']).issubset(kwargs):
            raise Exception('ZHAO profiles require inputs for exponents a, b, and c')
        return ZhaoProfile(**kwargs)
    else:
        raise ValueError("Unrecognized type %s"%profile_type)

def build_kernel(kernel_type, **kwargs):
    if kernel_type.upper()=='ISO':
        return IsotropicKernel(**kwargs)
    else:
        raise ValueError("Unrecognized type %s"%profile_type)
