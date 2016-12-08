import numpy as np

class LogLikelihood(object):
    def __init__(self, data, sigma):
        self.data = data
        self.sigma = sigma
        self.sigma.data = data
        self.free_pars = {}
    def get_free_pars(self):
        return self.free_pars
    def set_free(self, parname, **kwargs):
        if parname not in self.free_pars:
            if kwargs == {}:
                kwargs['val']=self.sigma.params[parnames]
            self.free_pars[parname] = kwargs
    def __call__(self, par_array):
        for i,key in enumerate(self.free_pars().keys()):
            #defer to sigma object the actual setting, so that 
            #the loglike object does not need to know which param 
            #comes from which part of the sigma computation
            #note : J is a parameter here
            self.sigma.setparam(key, par_array[i])
        self.compute()
        
class GaussianLikelihood(LogLikelihood):
    def __init__(self, **kwargs):
    def compute(self):
        R = self.data['R']
        dv = self.data['dv']
        v = self.data['v']
        #need to properly deal with parallelizing over R and possibly J
        #note : the fitter below uses fit to fit for J
        #in case of an array of Js, one should write another scan function
        #so it might be that here only the R parallelization is in order
        S = (dv**2.) + self.sigma.compute(R) #this is an array like R array
        res = (np.log(S) + ((v-u)**2.)/S)
        res = res.sum()
        return res


###################################################
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
