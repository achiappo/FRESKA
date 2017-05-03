import numpy as np
from exceptions import ValueError

class LogLikelihood(object):
	def __init__(self, data, sigma, numprocs=1):
		self.data = data
		self.sigma = sigma
		self.sigma.data = data # instruction to pass the data back to the SigmaLos object
		self.free_pars = {'J':18}
		self.numprocs = numprocs
		self.cache = {}

	def set_free(self, parname, **kwargs):
		if parname not in self.sigma.params:
			raise ValueError('%s not a parameter of logLike function.\n'%parname\
				+'\t The parameters are %s'%self.sigma.params.keys())
		if parname not in self.free_pars:
			if kwargs == {}:
				kwargs['val'] = self.sigma.params[parname]
			self.free_pars[parname] = kwargs

	def _retrieve(self, *freepars):
		if freepars in self.cache.keys():
			S = self.cache[freepars]
			iscached = True
		else:
			S = 0 #dummy value
			iscached = False
		return iscached, S

	def _store(self, S, *freepars):
		self.cache[freepars] = S

	def __call__(self, *par_array):
		if np.any(np.isnan(par_array)):
			return np.nan
		else:
			iscached, Scached = self._retrieve(*par_array)
			if iscached:
				S = Scached
			else:
				for i,key in enumerate(self.free_pars.keys()):
					self.sigma.setparams(key, par_array[i])
				S = self.compute()
				#self._store(S, *par_array)
			return S

class GaussianLikelihood(LogLikelihood):
    def __init__(self, *args):
        super(GaussianLikelihood, self).__init__(*args)

    def compute(self):
    	R = self.data[0]#['R']
        v = self.data[1]#['v']
        dv = self.data[2]#['dv']
        vsys = self.data[3]
        #need to properly deal with parallelizing over R and possibly J
        #note : the fitter below uses fit to fit for J
        #in case of an array of Js, one should write another scan function
        #so it might be that here only the R parallelization is in order
        S = dv**2 + self.sigma.compute(R) #this is an array like R array
        res = np.log(S) + ((v-vsys)**2)/S
        return res.sum() / 2.

    def contour(self, J, r):
    	R = self.data[0]
    	v = self.data[1]
    	dv = self.data[2]
    	vsys = self.data[3]
    	iscached, Scached = self._retrieve(r)
    	if iscached:
    		s = Scached
    	else:
    		self.sigma.setparams('J', 0)
    		self.sigma.setparams('dm_r0', r)
    		s = self.sigma.compute(R)
    		self._store(s, r)
    	S = dv**2 +  s * np.power(10, J/2.)
    	res = np.log(S) + ((v-vsys)**2)/S
    	return res.sum() / 2.
