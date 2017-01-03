from iminuit import Minuit
from exceptions import ValueError, RuntimeError
import numpy as np

class MinuitFitter(object):
    def __init__(self, loglike):
        self.loglike = loglike
        global global_loglike
        global_loglike = loglike
        self.settings = {'errordef':0.5,'print_level':0,'pedantic':False}
        self._synch()
        
    def _synch(self):
        for key,val in self.loglike.free_pars.items():
            if hasattr(val, '__iter__'):
                self.settings[key] = val.values()[0]
            else:
                self.settings[key] = val

    def set_free(self, parnames):
        parnames=np.array(parnames, ndmin=1, copy=False)
        for par in parnames:
            if par not in self.loglike.sigma.params:
                raise ValueError('%s not a parameter of logLike function.\n'%par\
				+'\t The parameters are %s'%self.loglike.sigma.params.keys())
            else:
            	self.loglike.set_free(par)
        self._synch()

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
        freepars = self.loglike.free_pars.keys()
        strargs = ", ".join(freepars)
        fit_func = eval("lambda %s : global_loglike(%s)"%(strargs,strargs))
        minuit = Minuit(fit_func, **self.settings)
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
