import numpy as np
from sys import float_info
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
            return float_info.max
        else:
            iscached, Scached = self._retrieve(*par_array)
            if iscached:
                S = Scached
            else:
                for i,key in enumerate(self.free_pars.keys()):
                    self.free_pars[key] = par_array[i]
                    self.sigma.setparams(key, par_array[i])
                S = self.compute()
                #self._store(S, *par_array)
            return S

class GaussianLikelihood(LogLikelihood):
    def __init__(self, *args):
        super(GaussianLikelihood, self).__init__(*args)
        self.R = self.data[0] #['R']
        self.v = self.data[1] #['v']
        self.dv2 = self.data[2]**2 #['dv']
        self.vsys = self.data[3]
        self.Dv2 = (self.v-self.vsys)**2
        
    def compute(self):
        #need to properly deal with parallelizing over R and possibly J
        #note : the fitter below uses fit to fit for J
        #in case of an array of Js, one should write another scan function
        #so it might be that here only the R parallelization is in order
        S = self.dv2 + self.sigma.compute(self.R) #this is an array like R array
        res = np.log(S) + self.Dv2/S
        return res.sum() / 2.
