import numpy as np
from iminuit import Minuit
from pymc import MCMC, Uniform
from exceptions import ValueError, RuntimeError

class Fitter(object):
    def __init__(self, loglike):
        self.loglike = loglike
        global global_loglike
        global_loglike = loglike
        self.settings = {}
        self._synch()
        self._set_function()
        
    def _synch(self):
        for key,val in self.loglike.free_pars.items():
            if hasattr(val, '__iter__'):
                self.settings[key] = val.values()[0]
            else:
                self.settings[key] = val

    def _set_function(self):
        freepars = self.loglike.free_pars.keys()
        strargs = ", ".join(freepars)
        fit_func = eval("lambda %s : global_loglike(%s)"%(strargs,strargs))
        self.fit_func = fit_func

    def set_free(self, parnames):
        parnames=np.array(parnames, ndmin=1, copy=False)
        for par in parnames:
            if par not in self.loglike.sigma.params:
                raise ValueError('%s not a parameter of logLike function.\n'%par\
                    +'\t The parameters are %s'%self.loglike.sigma.params.keys())
            else:
                self.loglike.set_free(par)
        self._synch()
        self._set_function()

class MinuitFitter(Fitter):
    """ subclass for Minuit minimisation"""
    def __init__(self, *args):
        super(MinuitFitter, self).__init__(*args)
        self.settings.update({'errordef':0.5,'print_level':0,'pedantic':False})

    def set_fixed(self, parnames):
        parnames=np.array(parnames, ndmin=1, copy=False)
        for par in parnames:
            if par not in self.loglike.free_pars:
                raise ValueError('%s not a free parameter of logLike function.\n'%par\
                    +'\t The free parameters are %s'%self.loglike.free_pars.keys())
            self.settings['fix_%s'%par] = True

    def set_value(self, par, value):
        if par not in self.loglike.free_pars:
            raise ValueError('%s not a free parameter of logLike function.\n'%par\
                +'\t The free parameters are %s'%self.loglike.free_pars.keys())
        self.settings['%s'%par] = value

    def set_error(self, par, value):
        if par not in self.loglike.free_pars:
            raise ValueError('%s not a free parameter of logLike function.\n'%par\
                +'\t The free parameters are %s'%self.loglike.free_pars.keys())
        self.settings['error_%s'%par] = value

    def set_bound(self, par, value):
        if par not in self.loglike.free_pars:
            raise ValueError('%s not a free parameter of logLike function.\n'%par\
                +'\t The free parameters are %s'%self.loglike.free_pars.keys())
        self.settings['limit_%s'%par] = value

    def set_minuit(self, **kwargs):
        minuit = Minuit(self.fit_func, **self.settings)
        if 'tol' in kwargs: 
            minuit.tol = kwargs['tol']
        if 'strategy' in kwargs:
            minuit.set_strategy(kwargs['strategy'])
        self.minuit = minuit

    def migrad_min(self, **kwargs):
        self.migrad_args = kwargs
        return self.minuit.migrad(**kwargs)

    def minos_profile(self, var, **kwargs):
        try:
            self.minuit.hesse()
            var_array, Like, res = self.minuit.mnprofile(var, **kwargs)
        except RuntimeError:
            print('Function not at minimum. Running migrad first.')
            self.minuit.migrad(**self.migrad_args)
            self.minuit.hesse()
            var_array, Like, res = self.minuit.mnprofile(var, **kwargs)
        return var_array, Like

class McmcFitter(Fitter):
    """ subclass for MCMC minimisation"""
    def __init__(self, *args):
        super(McmcFitter, self).__init__(*args)

#   def _set_MCMC(self):
#       self.M = MCMC()
        
