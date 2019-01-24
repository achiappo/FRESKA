from exceptions import Exception, ValueError, OverflowError, ZeroDivisionError
from scipy.special import betainc, hyp2f1, gamma, kn, gammaincc
from scipy.integrate import quad, nquad
from math import pi, cos, atan, asin, sqrt
import numpy as np
import cyfuncs

###############################################################################

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

###############################################################################
#                           STELLAR PROFILES
# options:
# - generalised Plummer (DOI: 10.1093/mnras/71.5.460)
# - exponential         (DOI: 10.1111/j.1745-3933.2008.00596.x)
# - King                (DOI: 10.1086/108756)
# - Sersic              (1968adga.book.....S , 1997A&A...321..111P)

class StellarProfile(Profile):
    """
    Define a stellar profile, for which the 2 universal parameters are
    - rh : scale radius
    - nuh : scale density, at scale radius
    This is still a base class, but it implements the Abel transform for the 
    surface brightness computation, as a default for inherited classes.
    """
    def __init__(self, **kwargs):
        """
        ensure existence of rh and nuh attributes
        """
        super(StellarProfile, self).__init__(**kwargs)        
        self.rh =  kwargs['rh'] if 'rh' in kwargs else 1
        self.nuh = kwargs['nuh'] if 'nuh' in kwargs else 1
        self.params = ['rh', 'nuh']

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
        Return the stellar density.
        input : x=r/rh (can be array-like)
        output : nuh * rh * x**(-c) * (1.+x**2)**(-(5-c)/2)
        """
        return self.nuh * cyfuncs.zhao_func(x, self.a, self.b, self.c)

    def surface_brightness(self, R):
        """
        Return the analytical solution for the brightness profile of a 
        Plummer density
        input : R
        output : 
        nuh * rh * (1+x*x)**(-2) if c==0 ,
        nuh * rh * ((2+x**2)*inv_csch(x) - np.sqrt(1+x**2))/(1+x**2)**1.5 if c==1
        otherwise, default to base class Abel integration.
        """
        x = R/self.rh
        result = self.nuh * self.rh
        c = self.c
        if c == 0: #standard Plummer
            return result * cyfuncs.plummer0_func(x)
        elif c == 1:
            return result * cyfuncs.plummer1_func(x)
        else:
            return super(genPlummerProfile, self).surface_brightness(rh=self.rh, R=R)

class ExponentialProfile(object):
    """
    Exponential stellar density profile, with parameter rc 
    indicating the size of a constant density core
    """
    def __init__(self, **kwargs):
        super(ExponentialProfile, self).__init__(**kwargs)
        self.rc = kwargs['rc'] if 'rc' in kwargs else 1.
        self.params += ['rc']

    def density(self, x):
        """
        Return the stellar density of an exponential profile
        input : x=r/rc (can be array-like)
        output : nuh Bessel0(x) / pi / rc
        """
        return self.nuh * kn(0,x) / pi / self.rc

    def surface_brightness(self, R):
        """
        Return the surface brightness of an exponential profile
        input : R
        output : nuh * exp(-R/rc)
        """
        x = R/self.rc
        return self.nuh * np.exp(-x)

class KingProfile(StellarProfile):
    """
    King stellar profile, with parameters
    - rc : core radius
    - rlim : maximum radius
    """
    def __init__(self, **kwargs):
        super(KingProfile, self).__init__(**kwargs)
        self.rc = kwargs['rc'] if 'rc' in kwargs else 1.
        # set a large dummy value for the maximum radius
        self.rlim = kwargs['rlim'] if 'rlim' in kwargs else 1000. 
        self.params += ['rc','rlim']

    def density(self, x):
        """
        Return the stellar density of a King profile.
        input : x=r/rc (can be array-like)
        output : nuh * (1 + x^2 + sqrt(1+x^2)sqrt(x^2-rlim^2/rc^2))) / pi / rc
        """
        lc = self.rlim / self.rc
        res = (1. + x*x + np.sqrt(1+x*x) * np.sqrt(x*x-lc*lc))
        return self.nuh / res / pi / self.rc

    def surface_brightness(self, R):
        """
        Return the surface brightness of a King profile
        input : R
        output : nuh * (1/sqrt(1+R^2/rc^2) - 1/sqrt(1+rlim^2/rc^2))
        """
        x = R/self.rc
        lc = self.rlim / self.rc
        return self.nuh * (1./np.sqrt(1.+x*x) - 1./np.sqrt(1.+lc*lc))

class SersicProfile(StellarProfile):
    """
    Sersic stellar profile, with parameters
    - rc : core radius
    - n : index controlling the sharpness of logarithmic decrease
    - bn = 2n − 1/3 + 0.009876/n 
    """
    def __init__(self, **kwargs):
        super(SersicProfile, self).__init__(**kwargs)
        self.rc = kwargs['rc'] if 'rc' in kwargs else 1.
        self.n = kwargs['n'] if 'n' in kwargs else 1. 
        self.params += ['rc','n']
    
    def density(self, x):
        """
        Return the stellar density.
        input : x=r/rc (can be array-like)
        output : nuh * bn * Int(x,inf) / n / pi
        where 
        Int(x,inf) = int^inf_r exp(-bn(y^(1/n)-1)) y^(1/n-2) / sqrt(1-x^2/y^2) dy
        """
        bn = 2./self.n - 1/3. + 0.009876/self.n 
        
        def integrand(y,x,bn,n): 
            return np.exp(-bn*(y**(1./n)-1)) * y**(1./n-2.) / np.sqrt(1-x*x*y*y)
        
        if np.isscalar(x):
            res = quad(integrand, x, +np.inf, args=(x,bn,n))[0]
            return self.nuh * bn * res / pi / n
        else:
            res = np.zeros_like(x)
            for i, xx in enumerate(xx):
                res[i] = quad(integrand, xx, +np.inf, args=(xx,bn,n))[0]
            return self.nuh * bn * res / pi / n

    def surface_brightness(self, R):
        """
        Return the surface brightness of a Sersic profile
        input : R
        output : nuh * exp( -bn ((R/rc)^(1/n)-1) )
        where bn = 2n − 1/3 + 0.009876/n 
        """
        x = R/self.rc
        bn = 2./self.n - 1/3. + 0.009876/self.n 
        return self.nuh * np.exp(-bn * (x**(1./n) - 1.))

##############################################################################
#                           DARK MATTER PROFILES
# options:
# - ZHAO (generalised NFW)  (DOI: 10.1086/168845, 10.1093/mnras/278.2.488 )
# - Einasto                 (DOI: 10.1093/mnrasl/slw216)
        
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

    def Jcst(self):
        """
        Dimensional factor of the J-factor
        Given r0 in kpc and rho0 in Msun/kpc^3
        returns a quantity with units of GeV^2/cm^5
        """
        Msun2kpc5_GeVcm5 = 4463954.894661358
        cst = 4 * pi * self.r0 * self.rho0**2 * Msun2kpc5_GeVcm5
        return cst
    
    def Jfactor(self, D, theta, rt, with_errs=False):
        """
        Return the J-factor, with or without integration errors
        """
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
        """
        (dimensionless) Zhao profile of the DM density distribution
        """
        a, b, c = self.a, self.b, self.c
        rhosat = 1e19
        if c>1e-5:
            if x > self.r0*(1e-10)**(1./c):
                return cyfuncs.zhao_func(x, a, b, c)
            else:
                return rhosat
        else:
            return cyfuncs.zhao_func(x, a, b, 0.)

    def mass(self, x):
        """
        (dimensionless) Mass function of the Zhao profile
        """
        a, b, c = self.a, self.b, self.c
        return cyfuncs.mass_func(x, a, b, c)

    def assert_range(self,a,b,c):
        if a<0 or b<c or b<=0.5 or c>=1.5:
            raise Exception("a,b,c values not allowed: %s, %s, %s"%\
                            (str(a), str(b), str(c)))
    
    def Jreduced(self, D, theta, rt, with_errs=False):
        """
        # In the case of a general Zhao profile, the J integration is 
        # more stable after a change of variable, and the separation of 
        # the resulting double integral in two steps
        """

        a, b, c = self.a, self.b, self.c
        r0 = self.r0
        Dprime = D/r0
        #rtprime = rt/r0
        ymin = cos(np.radians(theta))

        self.assert_range(a,b,c)
        opts = {'limit':1000, 'epsabs':1.e-8, 'epsrel':1.e-8}

        #first integral
        def integrand_one(u,t,a,b,c):
            val1 = (1+(u*t)**a)**(-2*(b-c)/a)
            val = val1 * u**(1-2*c) / np.sqrt(u**2-1)
            return val

        def integral_one(t, a, b, c):
            res = quad(integrand_one, 1, +np.inf, args=(t,a,b,c), **opts)
            return res

        #second integral
        def integrand_two(t, a, b, c):
            res = integral_one(t,a,b,c)
            return res[0] * t**(2-2*c) / np.sqrt(1.-(t/Dprime)**2)

        res = quad(integrand_two, 0, Dprime*np.sqrt(1.-ymin**2),\
                   args=(a,b,c), **opts)

        if with_errs:
            return res[0]/Dprime**2, res[1]/Dprime**2
        else:
            return res[0]/Dprime**2

class EinastoProfile(DMProfile):
    """
    class to define an Einasto DM profile
    """
    def __init__(self, **kwargs):
        super(EinastoProfile, self).__init__(**kwargs)
        self.alpha = kwargs['alpha'] if 'alpha' in kwargs else 1.
        self.params += ['alpha']
        self.params_Jreduced = [par for par in self.params if par != 'rho0']

    def density(self,x):
        """
        (dimensionless) Einasto profile of the DM density distribution
        """
        return np.exp(-2. * (x**self.alpha - 1.) / self.alpha)

    def mass(self, x):
        """
        (dimensionless) Mass function of the Einasto profile
        """
        alpha = self.alpha
        factor = (np.exp(2) * alpha**(3.-alpha) / 8.)**(1./alpha)
        return (1. - gammaincc(3./alpha, 2.*x**alpha/alpha)) * factor

    def assert_range(self,alpha):
        if a<0:
            raise Exception("alpha must be positive (%g)"%alpha)

    def Jreduced(self, D, theta, rt, with_errs=False):
        r0 = self.r0
        alpha = self.alpha
        Dprime = D/r0
        ymin = cos(np.radians(theta))

        self.assert_range(a,b,c)
        opts = {'limit':1000, 'epsabs':1.e-8, 'epsrel':1.e-8}

        #first integral
        def integrand_one(u,t,alpha):
            val = np.exp(-4. * u**alpha * t**alpha / alpha)
            return u * val / np.sqrt(u*u-1.)

        def integral_one(t, alpha):
            res = quad(integrand_one, 1, +np.inf, args=(t,alpha), **opts)
            if res[0]/res[1] < 10 and res[1]<1:
                #find the scale of the error and force epsrel and epsabs to 
                #aim for one order of magnitude smaller error
                eexp = ("%e"%res[1]).split("e-")[1]
                neweps = eval("1.e-%d"%(int(eexp)+1))
                res = quad(integrand_one, 1, +np.inf, args=(t,alpha),\
                           epsabs=neweps, epsrel=neweps, limit=1000)
            return res

        #second integral
        def integrand_two(t, alpha):
            res = integral_one(t,alpha)
            return res[0] * t**2 / np.sqrt(1.-(t/Dprime)**2)

        res = quad(integrand_two, 0, Dprime*np.sqrt(1.-ymin**2),\
                   args=(alpha,), **opts)
        if res[0]/res[1] < 10 and res[1]<1:
            #find the scale of the error and force epsrel and epsabs to 
            #aim for one order of magnitude smaller error
            eexp = ("%e"%res[1]).split("e-")[1]
            neweps = eval("1.e-%d"%(int(eexp)+1))
            res = quad(integrand_two, 0, Dprime*np.sqrt(1.-ymin**2),\
                       args=(alpha,), epsabs=neweps, epsrel=neweps, limit=1000)
        
        if with_errs:
            return res[0]/Dprime**2, res[1]/Dprime**2
        else:
            return res[0]/Dprime**2
        
##############################################################################
#                       ANISOTROPY KERNEL FUNCTIONS

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
        self.ra = kwargs['ra'] if 'ra' in kwargs else 0.
        self.params = ['ra']

    def __call__(self, r, R):
        ra = self.ra
        return cyfuncs.func_OM_kernel(r, R, ra)

##############################################################################
#                           HELPER FUNCTIONS

def build_profile(profile_type, **kwargs):
    if profile_type.upper() == 'PLUMMER':
        return genPlummerProfile(**kwargs)
    elif profile_type.upper() == 'EXPONENTIAL':
        return ExponentialProfile(**kwargs)
    elif profile_type.upper() == 'KING':
        return KingProfile(**kwargs)
    elif profile_type.upper() == 'SERSIC':
        return SersicProfile(**kwargs)
    elif profile_type.upper() == 'NFW':
        return ZhaoProfile(a=1, b=3, c=1, **kwargs)
    elif profile_type.upper() == 'ZHAO':
        if not set(['a', 'b', 'c']).issubset(kwargs):
            raise Exception('ZHAO profiles require inputs for exponents a, b, and c')
        return ZhaoProfile(**kwargs)
    elif profile_type.upper() == 'EINASTO':
        return EinastoProfile(**kwargs)
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

