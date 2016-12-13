from exceptions import Exception, ValueError
from scipy.special import betainc, kn, hyp2f1, gamma
from scipy.integrate import quad, nquad
from math import cos, atan, asin, sqrt
import numpy as np

##############################################################################
# inverse hyperbolic cosecant (used for c* = 1 , non-Plum)
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
    This is still a base class, but it implements the Abel transform for the 
    surface brightness computation, as a default for inherited classes.
    """
    def __init__(self, **kwargs):
        """
        ensure existence of rh and rhoh attributes
        """
        super(StellarProfile, self).__init__(**kwargs)        
        self.rh =  kwargs['rh'] if 'rh' in kwargs else 1
        self.rhoh= kwargs['rhoh'] if 'rhoh' in kwargs else 1
        self.params=['rh', 'rhoh']

    def surface_brightness(self, *args, **kwargs):
        """
        Compute the surface brightness from the density, 
        using the Abel transform
        """
        R=kwargs['R']
        rh=kwargs['rh']
        if np.isscalar(R):
            integrand = lambda r,rh,R: self.density(r/rh)*r/np.sqrt(r**2-R**2)
            return 2 * quad(integrand, R, +np.inf, args=(rh,R))[0]
        else :
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
            print "exponent parameters a and b are fixed to 2 and 5," +\
            "respectively, in generalized Plummer profiles. "+\
            "Use ZhaoProfile() instead."
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
        self.params = ['r0', 'rho0']

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
        def radius(z,y):
            return sqrt( z*z + Dprime**2*(1-y*y)) 
        def integrand(z,y):
            return self.density(radius(z,y))**2
        def lim_u(y):
            return [0, sqrt(rtprime**2 - Dprime**2*(1-y*y))]
        def lim_y():
            return [ymin,1.]

        res = nquad(integrand, ranges=[lim_u, lim_y], \
        			opts=[{'limit':1000, 'epsabs':1.e-10, 'epsrel':1.e-10},\
        				{'limit':1000, 'epsabs':1.e-10, 'epsrel':1.e-10}])
        if with_errs:
            return res[0], res[1]
        else:
            return res[0]

    def Jfactor(self, D, theta, rt, with_errs=False):
        Msun2kpc5_GeVcm5 = 4463954.894661358
        cst = 4*pi*rho0**2*r0*Msun2kpc5_GeVcm5
        return cst * self.Jreduced(D, theta, rt, with_errs=False)

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
        return 1. / x**c / (1.+x**a)**((b-c) / a)

    def mass(self, x):
        a, b, c = self.a, self.b, self.c
        return x**(3.-a) * hyp2f1( (3-a)/b, (c-a)/b, (b-a+3.)/b, -x**b )

##############################################################################
#Anisotropy kernels
'''
class AnisotropyKernel(object):
    """
    Mamon-Lokas integral for isotropic kernels
    """
    def __init__(self, model, **kwargs):
        self.__dict__ = kwargs
        self.model = model
        self.params = []
        if self.model == 'CONSTBETA':
        	#default to constant zero anisotropy
        	self.beta = kwargs['beta'] if 'beta' in kwargs else 0.
        	self.params = ['beta']
        elif self.model == 'OM':
        	#default to radial anisotropy
        	self.ra = kwargs['ra'] if 'ra' in kwargs else 0.
        	self.params = ['ra']

    def __call__(self, r, R):
        """
        return the isotropic kernel for u=r/R, r is a running radius 
        (the variable in the integrand), and R is the star radius (data)
        """
        u = r / R
        mod = self.model
        # isotropic model kernel function
        if mod == 'ISO':
            return sqrt(1.-u**(-2))
        # radial anisotropy kernel function
        elif mod == 'RAD':
        	return pi*u/4. - 0.5*np.sqrt(1. - 1./u/u) - u*asin(1./u)/2.
        # cosntant beta anisotropy kernel function
        elif mod == 'CONSTBETA':
        	beta = self.beta
        	ker1 = sqrt(1.-1./u/u) / (1.-2.*beta)
        	ker2 = sqrt(pi)/2. * gamma(beta-0.5)/gamma(beta) * (1.5-beta)
        	ker3 = u**(2*beta-1) * (1.-betainc(1./u/u, beta+0.5, 0.5))
        	return ker1 + ker2 * ker3
        # Osipkov-Merrit model kernel function
        elif mod == 'OM':
        	w = self.ra / R
        	ker1 = (w*w + 0.5)*(u*u + w*w)/u/(w*w + 1.)**1.5
        	ker2 = atan(np.sqrt((u*u - 1.)/(w*w + 1.)))
        	ker3 = 0.5/(w*w + 1.)*sqrt(1. - 1./u*u)
        	return ker1 * ker2 - ker3
        else:
        	raise ValueError("Unrecognized anisotropy type %s"%mod)
'''
class AnisotropyKernel(object):
    """
    Mamon-Lokas integral for isotropic kernels
    """
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

class IsotropicKernel(AnisotropyKernel):
	"""docstring for IsotropicKernel"""
	def __init__(self, **kwargs):
		super(IsotropicKernel,self).__init__(**kwargs)
		self.params = []

	def __call__(self, r, R):
		u = r / R
		return sqrt(1.-u**(-2))

class RadialKernel(AnisotropyKernel):
	"""docstring for RadialKernel"""
	def __init__(self, **kwargs):
		super(RadialKernel,self).__init__(**kwargs)
		self.params = []

	def __call__(self, r, R):
		u = r / R
		return pi*u/4. - 0.5*sqrt(1. - 1./u/u) - u*asin(1./u)/2.

class ConstBetaKernel(AnisotropyKernel):
	"""docstring for ConstBetaKernel"""
	def __init__(self, **kwargs):
		super(ConstBetaKernel,self).__init__(**kwargs)
		self.beta = kwargs['beta'] if 'beta' in kwargs else 0.
		self.params = ['beta']

	def __call__(self, r, R):
		u = r / R
		beta = self.beta
		ker1 = sqrt(1.-1./u/u) / (1.-2.*beta)
		ker2 = sqrt(pi)/2. * gamma(beta-0.5)/gamma(beta) * (1.5-beta)
		ker3 = u**(2*beta-1) * (1.-betainc(1./u/u, beta+0.5, 0.5))
		return ker1 + ker2 * ker3

class OMKernel(AnisotropyKernel):
	"""docstring for OMKernel"""
	def __init__(self, **kwargs):
		super(OMKernel,self).__init__(**kwargs)
		self.ra = kwargs['ra'] if 'ra' in kwargs else 0.
		self.params = ['ra']

	def __call__(self, r, R):
		ra = self.ra
		u = r / R
		w = ra / R
		ker1 = (w*w + 0.5)*(u*u + w*w)/u/(w*w + 1.)**1.5
		ker2 = atan(np.sqrt((u*u - 1.)/(w*w + 1.)))
		ker3 = 0.5/(w*w + 1.)*sqrt(1. - 1./u*u)
		return ker1 * ker2 - ker3

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
'''
def build_kernel(kernel_type, **kwargs):
    if kernel_type.upper()=='ISO':
        return AnisotropyKernel(kernel_type.upper(),**kwargs)
    elif kernel_type.upper()=='RAD':
        return AnisotropyKernel(kernel_type.upper(),**kwargs)
    elif kernel_type.upper()=='CONSTBETA':
        return AnisotropyKernel(kernel_type.upper(),**kwargs)
    elif kernel_type.upper()=='OM':
        return AnisotropyKernel(kernel_type.upper(),**kwargs)
    else:
        raise ValueError("Unrecognized anisotropy type %s"%profile_type)
'''
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
        raise ValueError("Unrecognized anisotropy type %s"%profile_type)
