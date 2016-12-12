from iminuit import Minuit
from exceptions import ValueError
import numpy as np

class MinuitFitter(object):
    def __init__(self, loglike):
        self.loglike = loglike
        global global_loglike
        global_loglike = loglike
        self.settings = {'errordef':0.5,'print_level':0,'pedantic':False}
        self.synch()
        
    def synch(self):
        for par in self.loglike.sigma.params:
            if par in self.loglike.free_pars:
                self.settings['fix_%s'%par] = False
        for key,val in self.loglike.get_free_pars().items():
            if hasattr(val, '__iter__'):
                #make room for future add of boundaries and/or errors
                self.settings[key]= val.values()[0]
            else:
                self.settings[key]= val

    def set_free(self, parnames):
        parnames=np.array(parnames, ndmin=1, copy=False)
        for par in parnames:
            if par not in self.loglike.sigma.params:
                raise ValueError('%s not a parameter of logLike function.\n'%par\
				+'The parameters are %s'%self.loglike.sigma.params.keys())
            self.settings['fix_%s'%par]=False

    def set_fixed(self, parnames):
        parnames=np.array(parnames, ndmin=1, copy=False)
        for par in parnames:
            if par not in self.loglike.sigma.params:
                raise ValueError('%s not a parameter of logLike function.\n'%par\
				+'The parameters are %s'%self.loglike.sigma.params.keys())
            self.settings['fix_%s'%par]=True

    def set_error(self, par, value):
        if par not in self.loglike.sigma.params:
            raise ValueError('%s not a parameter of logLike function.\n'%par\
				+'The parameters are %s'%self.loglike.sigma.params.keys())
        if par not in self.loglike.free_pars:
            raise ValueError('%s not a free parameter of logLike function.\n'%par\
				+'The free parameters are %s'%self.loglike.free_pars.keys())
        self.settings['error_%s'%par] = value
        
    def set_bound(self, par, value):
        if par not in self.loglike.sigma.params:
            raise ValueError('%s not a parameter of logLike function.\n'%par\
				+'The parameters are %s'%self.loglike.sigma.params.keys())
        if par not in self.loglike.free_pars:
        	raise ValueError('%s not a free parameter of logLike function.\n'%par\
				+'The free parameters are %s'%self.loglike.free_pars.keys())
        self.settings['limit_%s'%par] = value

    def fit(self, **kwargs):
        freepars = self.loglike.get_free_pars().keys()
        strargs = ", ".join(freepars)
        fit_func = eval("lambda %s : global_loglike(%s)"%(strargs,strargs))
        minuit = Minuit(fit_func, **self.settings)
        if 'tol' in kwargs: minuit.tol = kwargs['tol']
        fitresult = minuit.migrad()
        return fitresult


##############################################################################
class MinuitFitter2(object):
    def __init__(self, fcn):
        self.fcn = fcn
        self.settings = {'errordef':0.5,'print_level':0,'pedantic':False,}

    def fit(self):
        #this does not work, as Minuit will check the signature of the compute function to know the parameters. This would force the 
        #compute signature to include all possible parameters for all possible DM stellar and anisotropic parameters, which is untenable.
        Jfit = Minuit(lh.compute,**self.settings)
        Jfit.tol = 0.01
        BF = Jfit.migrad()
        return BF

    def set_free(self, parnames):
        parnames=np.array(parnames, ndmin=1, copy=False)
        for par in parnames:
            if par not in self.fcn.__dict__:
                raise ValueError('%s not a parameter of logLike function'%par)
            self.settings['fix_%s'%par]=False

    def set_fixed(self, parnames):
        parnames=np.array(parnames, ndmin=1, copy=False)
        for par in parnames:
            if par not in self.fcn.__dict__:
                raise ValueError('%s not a parameter of logLike function'%par)
            self.settings['fix_%s'%par]=True

    def set_value(self, par, value):
        if par not in self.fcn.__dict__:
            raise ValueError('%s not a parameter of logLike function'%par)
        self.settings['%s'%par]=value


    def set_error(self, par, value):
        if par not in self.fcn.__dict__:
            raise ValueError('%s not a parameter of logLike function'%par)
        self.settings['error_%s'%par]=value
        
    def set_bound(self, par, value):
        if par not in self.fcn.__dict__:
            raise ValueError('%s not a parameter of logLike function'%par)
        self.settings['limit_%s'%par]=value

