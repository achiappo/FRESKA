from exceptions import Exception, ValueError, OverflowError, ZeroDivisionError
from scipy.special import betainc, hyp2f1, gamma
from scipy.integrate import quad, nquad
from math import pi, cos, atan, asin, sqrt
import numpy as np
import cyfuncs

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
    This is still a base class, but it implements the Abel transform for the 
    surface brightness computation, as a default for inherited classes.
    """
    def __init__(self, **kwargs):
        """
        ensure existence of rh and rhoh attributes
        """
        super(StellarProfile, self).__init__(**kwargs)        
        self.rh =  kwargs['rh'] if 'rh' in kwargs else 1
        self.rhoh = kwargs['rhoh'] if 'rhoh' in kwargs else 1
        self.params = ['rh', 'rhoh']

    def surface_brightness(self, **kwargs):
        """
        Compute the surface brightness from the density, 
        using the Abel transform
        """
        R=kwargs['R']
        rh=kwargs['rh']
        if np.isscalar(R):
            integrand = lambda r,rh,R: self.density(r/rh)*r/np.sqrt(r**2-R**2)
            return 2 * quad(integrand, R, +np.inf, args=(rh,R))[0]
        else:
            res = np.zeros_like(R)
            for i, RR in enumerate(R):
                integrand = lambda r,rh,RR: self.density(r/rh)*r/np.sqrt(r**2-RR**2)
                res[i] = 2 * quad(integrand, RR, +np.inf, args=(rh,RR))[0]
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
            print "exponent parameters a and b are fixed to 2 and 5, "+\
            "respectively, in generalized Plummer profiles. "+\
            "Use ZhaoProfile() instead."
        self.a = 2
        self.b = 5
        # default to Plummer
        if 'c' not in kwargs:
            self.c = 0
        self.params += ['c']

    def density(self, x):
        """
        return the stellar density.
        input : x=r/rh (can be array-like)
        output : rhoh * rh * x**(-c) * (1.+x**2)**(-(5-c)/2)
        """
        return self.rhoh * cyfuncs.zhao_func(x, self.a, self.b, self.c)

    def surface_brightness(self, R):
        """
        Return the analytical solution for the brightness profile of a 
        Plummer density
        input : R
        output : 
        rhoh * rh * (1+x*x)**(-2) if c==0 ,
        rhoh * rh * ((2+x**2)*inv_csch(x) - np.sqrt(1+x**2))/(1+x**2)**1.5 if c==1
        otherwise, default to base class Abel integration.
        """
        x = R/self.rh
        result = self.rhoh * self.rh
        c = self.c
        if c == 0: #standard Plummer
            return result * cyfuncs.plummer0_func(x)
        elif c == 1:
            return result * cyfuncs.plummer1_func(x)
        else:
            return super(genPlummerProfile, self).surface_brightness(rh=self.rh, R=R)

class DMProfile(Profile):
    """ 
    Base class for the DM profile
    contains the generic formula for the J-factor calculation
    """
    def __init__(self, **kwargs):
        super(DMProfile, self).__init__(**kwargs)
        if 'r0' not in kwargs:
            self.r0 = 1
        if 'rho0' not in kwargs:
            self.rho0 = 1
        self.params = ['r0', 'rho0']
        self.__cached_Jreduced = {}
        
    def cached_Jreduced(self, D, theta, rt, with_errs=False):
        cache_params = tuple(getattr(self, par) for par in self.params_Jreduced)
        if (not hasattr(self,'D')) or\
          (self.D!=D or self.theta!=theta or self.rt!=rt):
            self.D = D
            self.theta = theta
            self.rt = rt
            self.__cached_Jreduced.clear()
            J = self.Jreduced(D, theta, rt, with_errs)
            self.__cached_Jreduced[cache_params] = J
        else :
            if cache_params in self.__cached_Jreduced.keys():
                J = self.__cached_Jreduced[cache_params]
                return J
            else:
                J = self.Jreduced(D, theta, rt, with_errs)
                self.__cached_Jreduced[cache_params] = J
        return J
        
    def Jreduced(self, D, theta, rt, with_errs=False):
        """
        compute the reduced J factor \int_ymin^1 dy \int_0^zmax dx f^2(r(z,y))
        where
        - ymin=\cos(\theta_max)
        - z=r/r0-D'y
        - the density reads rho(r)=rho0*f(r/r0)
        - zmax = sqrt(r_t'**2-D'^2*(1-y^2)) with r_t'=rt/r0 and D'=D/r0.
        The usual J factor is recovered by multiplying by 4pi*rho0^2*r0. 
        The reduced J factor is dimensionless
        """
        r0 = self.r0
        
        Dprime = D/r0
        rtprime = rt/r0
        ymin = cos(np.radians(theta))

        def integrand(z,y):
            try:
                return self.density(cyfuncs.radius(z, y, Dprime))**2
            except (OverflowError, ZeroDivisionError):
                return sys.float_info.max

        def lim_u(y):
            return [ 0., sqrt( rtprime*rtprime - Dprime*Dprime*(1-y*y) ) ]

        def lim_y():
            return [ ymin, 1. ]

        res = nquad(integrand, ranges=[lim_u, lim_y], \
                    opts=[{'limit':1000, 'epsabs':1.e-6, 'epsrel':1.e-8},\
                          {'limit':1000, 'epsabs':1.e-6, 'epsrel':1.e-8}])
        if with_errs:
            return res[0], res[1]
        else:
            return res[0]

    def Jcst(self):
        Msun2kpc5_GeVcm5 = 4463954.894661358
        cst = 4 * pi * self.r0 * self.rho0**2 * Msun2kpc5_GeVcm5
        return cst
    
    def Jfactor(self, D, theta, rt, with_errs=False):
        cst = self.Jcst()
        Jred = self.Jreduced(D, theta, rt, with_errs)
        if with_errs:
            return cst * Jred[0], cst * Jred[1]
        else:
            return cst * Jred

class ZhaoProfile(DMProfile):
    """
    class for defining a DM density profile
    belonging to the generic family of Zhao profiles
    """
    def __init__(self, **kwargs):
        super(ZhaoProfile, self).__init__(**kwargs)
        #default to NFW
        if 'a' not in kwargs:
            self.a = 1.
        if 'b' not in kwargs:
            self.b = 3.
        if 'c' not in kwargs:
            self.c = 1.
        self.params += ['a','b','c']
        self.params_Jreduced = [par for par in self.params if par != 'rho0']

    def density(self,x):
        a, b, c = self.a, self.b, self.c
        rhosat = 1e19
        if c>1e-5:
            if x > self.r0*(1e-10)**(1/c):
                return cyfuncs.zhao_func(x, a, b, c)
            else:
                return rhosat
        else:
            return cyfuncs.zhao_func(x, a, b, 0.)

    def mass(self, x):
        a, b, c = self.a, self.b, self.c
        return cyfuncs.mass_func(x, a, b, c)

##############################################################################
#Anisotropy kernels

class AnisotropyKernel(object):
    """
    Mamon-Lokas (2005) Kernel functions for calculating the intrinsic
    velocity dispersion for the following anistropy models:
    - isotropic 
    - radial anisotropy 
    - constant Beta anistropy 
    - Osipkov-Merritt anistropy profile
    """
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

class IsotropicKernel(AnisotropyKernel):
    """
    Kernel function for the isotropic 
    velocity distribution case
    """
    def __init__(self, **kwargs):
        super(IsotropicKernel,self).__init__(**kwargs)
        self.params = []

    def __call__(self, r, R):
        return cyfuncs.func_isotropic_kernel(r, R)

class RadialKernel(AnisotropyKernel):
    """
    Kernel function for the radial 
    velocity distribution case
    """
    def __init__(self, **kwargs):
        super(RadialKernel,self).__init__(**kwargs)
        self.params = []

    def __call__(self, r, R):
        return cyfuncs.func_radial_kernel(r, R)

class ConstBetaKernel(AnisotropyKernel):
    """
    Kernel function for the constant 
    velocity anistropy case
    """
    def __init__(self, **kwargs):
        super(ConstBetaKernel,self).__init__(**kwargs)
        self.beta = kwargs['beta'] if 'beta' in kwargs else 0.
        self.params = ['beta']

    def __call__(self, r, R):
        beta = self.beta
        return cyfuncs.func_constant_kernel(r, R, beta)

class OMKernel(AnisotropyKernel):
    """
    Kernel function for the Osipkov-Merritt varying
    anisotropy case
    """
    def __init__(self, **kwargs):
        super(OMKernel,self).__init__(**kwargs)
        self.a = kwargs['a'] if 'a' in kwargs else 0.
        self.params = ['a']

    def __call__(self, r, R):
        ra = self.a
        return cyfuncs.func_OM_kernel(r, R, ra)

##############################################################################
#Helper functions

def build_profile(profile_type, **kwargs):
    if profile_type.upper() == 'PLUMMER':
        return genPlummerProfile(**kwargs)
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
    elif kernel_type.upper()=='RAD':
        return RadialKernel(**kwargs)
    elif kernel_type.upper()=='CONSTBETA':
        return ConstBetaKernel(**kwargs)
    elif kernel_type.upper()=='OM':
        return OMKernel(**kwargs)
    else:
        raise ValueError("Unrecognized anisotropy type %s"%kernel_type)
