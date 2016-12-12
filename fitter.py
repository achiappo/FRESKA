from iminuit import Minuit
from exceptions import ValueError
import numpy as np

class MinuitFitter(object):
    def __init__(self, loglike):
        self.loglike = loglike
        self.settings = {'errordef':0.5,'print_level':0,'pedantic':False}
        
    def synch(self):
        for key,val in self.loglike.get_free_pars().items():
            if hasattr(val, '__iter__'):
                #make room for future add of boundaries and/or errors
                self.settings[key]= val.values()[0]
            else:
                self.settings[key]= val
        return self.settings
    
    def fit(self, **kwargs):
        settings = self.synch()
        freepars = self.loglike.get_free_pars().keys()
        minuit = Minuit(lambda freepars: self.loglike(freepars),**settings)
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

