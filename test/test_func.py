from __future__ import division
from math import *

import yaml
import numpy as np
from operator import itemgetter
from scipy.integrate import quad,nquad
from scipy.special import betainc, kn, hyp2f1, gammainc

##############################################################################
# Likelihood definition

class Like(object):
	""" Likelihood class"""
	def __init__(self, data, profiles):
		self.R = data[0]
		self.v = data[1]
		self.dv = data[2]
		self.profiles = profiles
	
	def unbinned(self, params):
		gamma = self.R / params['rh']
		alpha = params['rh'] / params['r0']
		S = Sigma(self.profiles, params)
		I = ConfigStellar(self.profiles, params)
		S_arr = np.array([S(g, alpha)/I.surface_brightness(R) 
			for g,R in zip(gamma,self.R)])
		term1 = (self.v-self.v.mean())**2 / (self.dv**2+S_arr)
		term2 = nl.log(self.dv**2+S_arr)
		return 0.5*(term1+term2).sum()

	def __call__(self, params):
		if 'unbinned' in self.profiles:
			return self.unbinned(params)

##############################################################################
# Dark Matter component section

class ZhaoProfiles(object):
	""" Class defining the mass and density of the Zhao profile"""
	
	def __init__(self, params):
		self.params = params

	def density(self,x):
		a, b, c = sorted(self.params.values())
		return 1. / x**c / (1.+x**a)**((b-c) / a)

	def mass(self, x):
		a, b, c = sorted(self.params.values())
		return x**(3.-a) * hyp2f1( (3-a)/b, (c-a)/b, (b-a+3.)/b, -x**b )

class ConfigDM(object):
	""" Class returning the DM mass and density functions,
		depending on the profile chosen"""

	def __init__(self, profiles, params):
		if 'Zhao' in profiles:
			self.profile = ZhaoProfiles(params)

	def DM_density(self, x):
		return self.profile.density(x) 

	def DM_mass(self,x):
		return self.profile.mass(x)

##############################################################################
# Stellar component section

class Plummer(object):
	""" Class defining the  Plummer surface brightness profile 
		and its stellar density distribution"""
    def __init__(self, params):
        self.params = params

    def surface_brightness(self, R):
        rh = self.params['rh']
        return 4.*rh/3./pi/(1. + R**2/rh**2)**2
    
    def stellar_density(self, r):
        return 1. / (1.+r**2)**2.5

class ConfigStellar(object):
    def __init__(self, profiles, params):
        if 'Plummer' in profiles.keys():
            self.profile = Plummer(params)
    
    def surface_brightness(self, R):
        return self.profile.surface_brightness(R)
    
    def stellar_density(self, r):
        return self.profile.stellar_density(r)

##############################################################################
# Stellar velocity anisotropy section

class ConfigKernel(object):
	def __init__(self, profiles, params):
		self.profiles = profiles
		self.params = params

	def IS_kernel(self, y, gamma):
		"""Kernel function for isotropic velocity distribution"""
		u = y / gamma
		return np.sqrt(1. - 1./u)

	def RD_kernel(self, y, gamma):
		"""Kernel function for radially anisotropic velocity distribution"""
		u = y / gamma
		return pi*u/4. - 0.5*np.sqrt(1. - 1./u**2) - u*asin(1./u)/2.

	def CA_kernel(self, y, gamma, beta):
		"""Kernel function for constant anisotropy velocity distribution"""
		u = y / gamma
		term1 = sqrt(1.-1./u**2) / (1.-2.*beta)
		term2 = sqrt(pi)/2. * gamma(beta-0.5)/gamma(beta) * (1.5-beta)
		term3 = u**(2*beta-1) * (1.-betainc(1./u**2, beta+0.5, 0.5))
		return term1 + term2 * term3


	def OM_kernel(self, y, gamma, delta):
		"""Kernel function for Osipkov-Merritt velocity distribution"""
		u = y / gamma
		term1 = (w**2 + 0.5)*(u**2 + w**2)/u/(w**2 + 1.)**1.5
		term2 = atan(np.sqrt((u**2 - 1.)/(w**2 + 1.)))
		term3 = 0.5/(w**2 + 1.)*sqrt(1. - 1./u**2)
		return term1 * term2 - term3

	def __call__(self, y, gamma):
		if 'IS' in self.profiles:
			return self.IS_kernel(y, gamma)

		elif 'RD' in self.profiles:
			return self.RD_kernel(y, gamma)

		elif 'CA' in self.profiles:
			beta = self.params['beta']
			return self.CA_kernel(y, gamma, beta)

		elif 'OM' in self.profiles:
			ra = self.params['ra']
			rh = self.params['rh']
			return self.OM_kernel(y, gamma, ra/rh)

##############################################################################
# Sigma_los integral

class Sigma(object):
	""" Class containing the integral involved in sigma_los"""
	
	def __init__(self, profiles, params):
		self.mass = ConfigDM(profiles, params)
		self.star = ConfigStellar(profiles, params)
		self.kern = ConfigKernel(profiles, params)


	def integrand_s(self, y, gamma, alpha):
		Kern = self.kern(y, gamma)
		star = self.star.stellar_density(y)
		mass = self.mass.DM_mass(y*alpha)
		return Kern*star*mass/y

	def integral_s(self, gamma, alpha):
		arguments = (gamma, alpha)
		return quad(self.integrand_S, gamma, +np.inf, args=arguments)[0]

	def __call__(self, args):
		""" proxy for the integral in Sigma"""

		cst = 8.*np.pi*4.3e-6
		Int_S = self.integral_S(args[0], args[1])
		return cst*Int_S
