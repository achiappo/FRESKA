import numpy as np
from sys import float_info
from exceptions import ValueError, Exception
from dispersion import SphericalJeansDispersion

class LogLikelihood(object):
    def __init__(self, data, sigma, numprocs=1):
        if not isinstance(sigma, SphericalJeansDispersion):
            raise Exception("sigma must be an istance of SphericalJeansDispersion")
        self.data = data
        self.sigma = sigma
        self.sigma.data = data
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

    def set_fixed(self, parname, **kwargs):
        if parname not in self.sigma.params:
            raise ValueError('%s not a parameter of logLike function.\n'%parname\
                +'\t The parameters are %s'%self.sigma.params.keys())
        if parname not in self.free_pars:
            raise ValueError('%s is already fixed at value %g'%\
                             (parname, self.sigma.params[parname]))
        else:
            del self.free_pars[parname]

    def set_value(self, parname, parvalue):
        if parname not in self.sigma.params:
            raise ValueError('%s not a parameter of logLike function.\n'%parname\
                +'\t The parameters are %s'%self.sigma.params.keys())
        if parname in self.free_pars:
            self.free_pars[parname] = parvalue
        self.sigma.setparams(parname, parvalue)

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
        self.R = self.data[0]
        self.v = self.data[1]
        self.dv2 = self.data[2]**2
        self.vsys = self.data[3]
        self.Dv2 = (self.v-self.vsys)**2
        
    def compute(self):
        S = self.dv2 + self.sigma.compute(self.R) #this is an array like R array
        res = np.log(S) + self.Dv2/S
        return res.sum() / 2.

