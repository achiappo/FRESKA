import numpy as np
from sys import float_info
from scipy.integrate import quad
from profiles import DMProfile, StellarProfile, AnisotropyKernel

class SphericalJeansDispersion(object):
    def __init__(self, dm, stellar, anisotropy, dwarf_props, **kwargs):
        if isinstance(dm, DMProfile):
            self.dm = dm
        elif isinstance(dm, str):
            self.dm = build_profile(dm,**kwargs)
        if isinstance(stellar, StellarProfile):
            self.stellar = stellar
        elif isinstance(stellar, str):
            self.stellar = build_profile(stellar,**kwargs)
        if isinstance(anisotropy, AnisotropyKernel):
            self.kernel = anisotropy
        elif isinstance(anisotropy, str):
            self.kernel = build_kernel(anisotropy,**kwargs)
        G = 4.302e-6
        self.dwarf_props = dwarf_props
        self.cst = 8.*np.pi*G
        self.params = {'J':18} #dummy value
        setattr(self, 'J', 18)
        self._synch()
        
    def _synch(self):
        for par in self.dm.params:
            self.params['dm_'+par] = self.dm.__dict__[par]
        for par in self.stellar.params:
            self.params['st_'+par] = self.stellar.__dict__[par]
        for par in self.kernel.params:
            self.params['ker_'+par] = self.kernel.__dict__[par]
    
    def setparams(self, name, value):
        if name.split('dm_')[-1] in self.dm.__dict__:
            setattr(self.dm, name.split('dm_')[-1], value)
        if name.split('st_')[-1] in self.stellar.__dict__:
            setattr(self.stellar, name.split('st_')[-1], value)
        if name.split('ker_')[-1] in self.kernel.__dict__:
            setattr(self.kernel, name.split('ker_')[-1], value)
        if name == 'J':
            setattr(self, name, value)
            self.params[name] = value
        self._synch()
        
    def integrand(self, s, R, **kwargs):
        #not correct : R could be provided which is not in inital self.R
        #this is weak anyway, why are r0 and rh singled out parameters?
        r0 = kwargs['r0'] if 'r0' in kwargs else self.dm.r0
        rh = kwargs['rh'] if 'rh' in kwargs else self.stellar.rh
        val = self.stellar.density(s/rh) * self.dm.mass(s/r0) * self.kernel(s, R) / s
        return val

    def compute(self, R):
        #called by a LogLike object
        if any([getattr(self.dm,par)<0 for par in self.dm.params]):
            return float_info.max
        else:
            Jreduced = self.dm.cached_Jreduced( **self.dwarf_props )
            if 'with_errs' in self.dwarf_props and\
              self.dwarf_props['with_errs']==True:
                Jred = Jreduced[0]
                Jerr = Jreduced[1]
                #print self.dm.r0, Jred, Jerr
            else:
                Jred = Jreduced
                
            if np.isscalar(R):
                integral = self._sigma_integral(R)
                sigma2 = integral / self.stellar.surface_brightness(R) / np.sqrt(Jred)
            else:
                sigma2 = np.zeros_like(R)
                for i,rr in enumerate(R):
                    eps=1.e-8
                    integral = self._sigma_integral(rr, epsrel=eps,epsabs=eps)
                    I_of_R = self.stellar.surface_brightness(rr)
                    sigma2[i] = integral / I_of_R / np.sqrt(Jred)
                    
            cst = self.cst / np.sqrt(self.dm.Jcst())
            return sigma2 * self.dm.r0**3 * cst * np.power(10, self.J/2.)

    def _sigma_integral(self, rr, epsrel=1.e-8, epsabs=1.e-8):
        a,b,c,r0 = self.dm.a, self.dm.b, self.dm.c, self.dm.r0
        res = quad(self.integrand, t, np.inf, args=(t,beta,a,b,c),
               epsabs=epsabs, epsrel=epsrel)
        if res[0]/res[1] < 10 and res[1]<1:
            #find the scale of the error and force epsrel and epsabs to 
            #aim for one order of magnitude smaller error
            eexp = ("%e"%res[1]).split("e-")[1]
            neweps = eval("1.e-%d"%(int(eexp)+1))
            res = quad(self.integrand, t, np.inf, args=(t,beta,a,b,c),
                       epsabs=neweps, epsrel=neweps, limit=1000)
        return res[0], res[1]

    def _sigma_integral2(self, rr):
        """
        This integration scheme is a work in progress
        """
        res1 = quad(self.integrand, rr, 2*rr, args=(rr,), epsrel=1.e-10, epsabs=1.e-10)
        res2 = quad(self.integrand, 2*rr, np.inf, args=(rr,), epsrel=1.e-10, epsabs=1.e-10)
        if res1[0]<0 or res1[1]>res1[0]:
            print res1, self.dm.a, self.dm.b, self.dm.c, self.dm.r0, rr
        if res2[0]<0 or res2[1]>res2[0]:
            print res2, self.dm.a, self.dm.b, self.dm.c, self.dm.r0, rr
        integral = res1[0]+res2[0]
        return integral
