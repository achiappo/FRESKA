from exceptions import Exception, ValueError, OverflowError, ZeroDivisionError
from scipy.integrate import quad, nquad
from math import pi, cos, sqrt
import numpy as np
import cyfuncsA

class DMProfile(object):
	def __init__(self, **kwargs):
		if 'r0' not in kwargs:
			self.r0 = 1
		if 'rho0' not in kwargs:
			self.rho0 = 1
		self.params = ['r0', 'rho0']

	def Jreduced(self, D, theta, rt, with_errs=False):
		r0 = self.r0
		Dprime = D/r0
		rtprime = rt/r0
		ymin = cos(np.radians(theta))
		res = cyfuncsA.integral(self.density, ymin, rtprime, Dprime)
		if with_errs:
			return res[0], res[1]
		else:
			return res[0]

	def Jfactor(self, D, theta, rt, with_errs=False):
		Msun2kpc5_GeVcm5 = 4463954.894661358
		cst = 4 * pi * self.r0 * Msun2kpc5_GeVcm5
		return cst * self.rho0**2 * self.Jreduced(D, theta, rt, with_errs=False)

class ZhaoProfile(DMProfile):
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

	def density(self,x):
		a, b, c = self.a, self.b, self.c
		return cyfuncsA.zhao_func(x, a, b, c)

	def mass(self, x):
		a, b, c = self.a, self.b, self.c
		return cyfuncsA.mass_func(x, a, b, c)
