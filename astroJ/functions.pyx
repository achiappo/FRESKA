from __future__ import division
from math import *

import yaml
import numpy as np
from scipy.integrate import quad,nquad
from scipy.special import betainc, kn, hyp2f1, gammainc
cimport numpy as np

params = yaml.load(open('default.yaml'))

##############################################################################
##############################################################################
#							FUNCTIONS DEFINITIONS
##############################################################################
##############################################################################
# extract data from data files

def get_data(str gal):
	"""This function extracts the parameters and the stellar kinematic
		data of the system in exam . The user can modify this function
		in order for the """

	data = open('data/params/params_%s.dat'%gal, 'r').readlines()
	parameters = []
	for line in data:
		parameters.append(line[:-1])

	D  = float(parameters[1])
	rh = float(parameters[2])
	rt = float(parameters[3])

	x,v,dv = np.loadtxt('data/velocities/velocities_%s.dat'%gal, 
						dtype=float, usecols=(0, 1, 2), unpack=True)
	return x, v, dv, D, rh, rt

##############################################################################
##############################################################################
# Dark Matter density profile definition

if params['DM_prof'] == 'Zhao' :

	def rho(double x, tuple DM_params):
		"""Zhao profile for the dark matter distribution (x := r/r_s)"""

		aDM, bDM, cDM = DM_params
		return 1. / x**cDM / (1. + x**aDM)**((bDM - cDM)/aDM)

	def mass(double x, tuple DM_params):
		"""Dark matter halo mass from Zhao profile"""

		aDM, bDM, cDM = DM_params
		hypa = (3.- aDM) / bDM
		hypb = (cDM - aDM) / bDM
		hypc = (bDM - aDM + 3.)/bDM
		return x**(3.-aDM) * hyp2f1(hypa, hypb, hypc, -x**bDM)

	#----------------------------------------------------------------------
elif params['DM_prof']=='Zhao':

	def rho(double x, tuple DM_params):
		"""Burkert profile for the dark matter distribution (x := r/r_s)"""

		return 1. / (1.+x) / (1.+x**2)
	
	def mass( double x, tuple DM_params ):
		"""Dark matter halo mass from Burkert profile"""

		return (log(1 + x**2) + 2*log(1. + x) - 2*atan(x))/4.

	#----------------------------------------------------------------------
elif params['DM_prof']=='Einasto':

	def rho(double x, tuple DM_params):
		"""Einasto profile for the dark matter distribution (x := r/r_2)"""

		alpha = DM_params
		return exp(-2.*(x**alpha - 1.)/alpha)
	
	def mass(double x, tuple DM_params):
		"""Dark matter halo mass from Einasto profile"""

		alpha = DM_params
		gam_inc = gammainc(3./alpha,2*x**alpha/alpha)
		return -8**(-1./alpha)*exp(2./alpha) * x**3 * gam_inc/alpha

##############################################################################
##############################################################################
# Stellar density profile and surface brightness definitions

if params['stellar']=='Hernquist':

	def nu(double y, tuple ST_params):
		"""Hernquist profile for stellar density distribution"""

		aST, bST, cST, rST = ST_params
		return 1. / y**cST / (1.+y**aST)**((bST-cST) / aST)
	
	# obtain surface brightness I(R) using Abel transform
	def integrand_I(double r, double R, tuple ST_params):
		return nu(r, ST_params) * r / np.sqrt(r**2-R**2)

	def I(double R, tuple ST_params):
		"""Surface brightness profile obtained 
			via the Abel transformation of
			the Hernquist profile"""

		return quad(integrand_I, R, +np.inf, args=(R, ST_params))[0]

	#----------------------------------------------------------------------
elif params['stellar']=='surface_bright':
	
	if params['I_prof']=='Plummer':

		def I(double gamma, tuple ST_params):
			"""Plummer profile for dSphs surface brightness"""
			rST = ST_params
			return 4.*rST/3./pi/(1. + gamma**2)**2

		def nu(double r, tuple ST_params):
			"""Stellar density obtained from the inverse 
				Abel transform of Plummer profile"""
		
			return 1. / (1.+r**2)**2.5

	#----------------------------------------------------------------------
	elif params['I_prof']=='exponential':

		def I(double gamma, tuple ST_params):
			"""exponential profile for dSphs surface brightness"""
			return exp(-gamma)

		def nu(double y, tuple ST_params):
			"""stellar density obtained from the inverse 
				Abel transform of exponential profile"""

			r_c = ST_params
			return kn(0, y) / r_c

	#----------------------------------------------------------------------
	elif params['I_prof']=='King':

		def I(double gamma, tuple ST_params):
			"""King profile for dSphs surface brightness"""
			
			r_c, r_lim = ST_params
			term1 = 1./np.sqrt(1. + gamma**2) 
			term2 = 1./np.sqrt(1. + r_lim**2/r_c**2)
			return  term1 - term2

		def nu(double y, tuple ST_params):
			"""Stellar density obtained from the inverse 
				Abel transform of King profile"""

			r_c, r_lim = ST_params
			return 1./(1. + y**2)/r_c

	#----------------------------------------------------------------------
	elif params['I_prof']=='Sersic':

		def I(double gamma, tuple ST_params):
			"""Sersic profile for dSphs surface brightness"""

			n, r_c = ST_params
			bn = 2*n - 1/3. + 0.009876/n
			return exp(-bn*(gamma**(1./n) - 1))

		def integrand_nu(double z, double n, double y):
			return exp(-bn*z)/z**n/sqrt(1. - y**2/z**(2*n))

		def nu(double y, tuple ST_params):
			"""stellar density obtained from the inverse 
				Abel transform of Sersic profile"""

			n, r_c = ST_params
			bn = 2*n - 1./3. + 0.009876/n
			intgrl = quad(integrand_nu, y**(1./n), +np.inf, args=(n, y))
			return bn*exp(bn)*intgrl/r_c

##############################################################################
##############################################################################
# Stellar velocity anisotropy profile definition

if params['anisotropy']=='IS':

	def kernel(double u, double w, double beta):
		"""Kernel function for isotropic velocity distribution"""

		return np.sqrt(1. - 1./u)

#----------------------------------------------------------------------------
if params['anisotropy']=='RD':

	def kernel(double u, double w, double beta):
		"""Kernel function for radially isotropic velocity distribution"""

		term1 = pi*u/4.
		term2 = 0.5*np.sqrt(1. - 1./u**2)
		term3 = u*asin(1./u)/2.
		return term1 - term2 - term3

#----------------------------------------------------------------------------
if params['anisotropy']=='CA':

	def kernel(double u, double w, double beta):
		"""Kernel function for constant anisotropy velocity distribution"""

		term1 = (1.5 - beta)*np.sqrt(pi)*gamma(beta - 0.5)/gamma(beta)
		term2 = beta*betainc(1./u**2, beta+0.5, 0.5)
		term3 = betainc(1./u**2, beta-0.5, 0.5)
		return 0.5*u**(2*beta - 1.)*(term1 + term2 - term3)

#----------------------------------------------------------------------------
if params['anisotropy']=='OM':

	def kernel(double u, double w, double beta):
		"""Kernel function for Osipkov-Merritt velocity distribution"""

		term1 = (w**2 + 0.5)*(u**2 + w**2)/u/(w**2 + 1.)**1.5
		term2 = atan(np.sqrt((u**2 - 1.)/(w**2 + 1.)))
		term3 = 0.5/(w**2 + 1.)*sqrt(1. - 1./u**2)
		return term1 * term2 - term3

##############################################################################
##############################################################################
# Numerical integral in sigma_los

def integrand_s(double y, double gamma, 
				tuple params, tuple DM_params, tuple ST_params):
	"""Integrand of sigma_los function"""

	alpha, delta, beta = params
	Kuw = kernel(y/gamma, delta/gamma, beta)
	return Kuw*nu(y, ST_params)*mass(	y*alpha, DM_params)/y

def integral_s(double gamma, tuple params, 
			   tuple DM_params, tuple ST_params):
	"""Integral of the sigma_los function"""
	
	arguments = (gamma, params, DM_params, ST_params)
	return quad(integrand_S, gamma, +np.inf, args=arguments)[0]

def proxy_sigma(tuple args):
	"""Proxy function for the integral of the sigma_los
	   to perform a multi-processors computation"""

	cst = 8.*np.pi*4.3e-6
	Int_S = integral_S(args[0], args[1], args[2], args[3])
	return cst*Int_S/I(args[0], args[3])

#----------------------------------------------------------------------------

DTYPE=np.double
ctypedef np.double_t DTYPE_t
def compute_I_array(np.ndarray[DTYPE_t, ndim=1] gamma_array, 
	        		tuple params, tuple DM_params, tuple ST_params):
	
	cdef np.ndarray[DTYPE_t, ndim=1] res = np.zeros_like(gamma_array)
	cdef double gamma
	cdef int i
	cdef double cst = 8.*np.pi*4.3e-6
	cdef int n = len(gamma_array)
	for i in range(n):
		Int_S = integral2(gamma_array[i], params, DM_params, ST_params)
		res[i] = Int_S / I(gamma_array[i], ST_params)
	return res

##############################################################################
##############################################################################
# Jfactor evaluation functions

def func(double z, double y, double D, double rt, double ymin):
	"""Integrand of the J-factor"""

	return rho(sqrt(z**2 - D**2 * (1. - ymin**2 ) ) )

def lim_u( double y, double D, double rt, double ymin ):
	return [0, sqrt(rt**2 - D**2 * (1. - ymin**2))]

def lim_y(double D, double rt, double ymin):
	return [ymin, 1.]

def Jfactor(double D, double rt, double r0, double tmax):
	"""
	Returns the Jfactor computed in the solid angle of
	semi apex angle tmax, in degree, for a NFW halo profile of 
	shape parameters (r0,rho0) at distance D. 
	rt is the maximal radius of integration 
	D, r0 and rt are in kpc, and rho0 is in Msun.kpc^-3
	"""

	ymin = cos(np.radians(tmax))
	Dprime = D/r0
	rtprime = rt/r0
	Msun2kpc5_GeVcm5 = 4463954.894661358
	cst = 4.*np.pi*r0*Msun2kpc5_GeVcm5
	options = [{'limit':1000, 'epsabs':1e-5}, {'limit':1000, 'epsabs':1e-5}]
	rgs = [lim_u, lim_y]
	arguments = (Dprime, rtprime, ymin)
	res = nquad(func, ranges=rgs, args=arguments, opts=options)
	return cst*res[0]
