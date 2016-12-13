import numpy as np
from exceptions import ValueError

class LogLikelihood(object):
	def __init__(self, data, sigma):
		self.data = data
		self.sigma = sigma
		self.sigma.data = data # instruction to pass the data back to the SigmaLos object
		self.free_pars = {'J':18}

	def set_free(self, parname, **kwargs):
		if parname not in self.sigma.params:
			raise ValueError('%s not a parameter of logLike function.\n'%parname\
				+'\t The parameters are %s'%self.sigma.params.keys())
		if parname not in self.free_pars:
			if kwargs == {}:
				kwargs['val'] = self.sigma.params[parname]
			self.free_pars[parname] = kwargs

	def __call__(self, *par_array):
		# would be worthwhile putting the following in an indented block
		# and add the caching+retrieving instructions
		for i,key in enumerate(self.free_pars.keys()):
			#defer to sigma object the actual setting, so that 
			#the loglike object does not need to know which param 
			#comes from which part of the sigma computation
			#note : J is a parameter here
			self.sigma.setparams(key, par_array[i])
		return self.compute()

class GaussianLikelihood(LogLikelihood):
    def __init__(self, *args):
        super(GaussianLikelihood, self).__init__(*args)

    def compute(self):
    	R = self.data[0]#['R']
        v = self.data[1]#['v']
        dv = self.data[2]#['dv']
        #need to properly deal with parallelizing over R and possibly J
        #note : the fitter below uses fit to fit for J
        #in case of an array of Js, one should write another scan function
        #so it might be that here only the R parallelization is in order
        S = dv**2 + self.sigma.compute(R) #this is an array like R array
        res = np.log(S) + ((v-v.mean())**2)/S
        return res.sum() / 2.


##############################################################################
class GaussianLikelihood2(object):
    def __init__(self, sigma, v, dv):
        self.vbar2 = (v - v.mean())**2
        self.dv2 = dv**2
        self.sigma = sigma
        self.__dict__.update(sigma.mass.__dict__)
        self.__dict__.update(sigma.star.__dict__)

    def __call__(**kwargs):
        v = self.v
        dv2 = self.dv2
        sigma=self.sigma(kwargs)
        term1 = self.vbar2 / (dv2 + sigma)
        term2 = np.log(dv2 + sigma)
        return 0.5*(term1+term2).sum()
