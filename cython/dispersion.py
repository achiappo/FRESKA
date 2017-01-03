import numpy as np
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
        G=4.3e-6
        self.dwarf_props = dwarf_props
        self.cst = 8.*np.pi*G
        self.params = {'J':18} #dummy value
        self._synch()
        
    def _synch(self):
        for par in self.dm.params:
            self.params['dm_'+par] = self.dm.__dict__[par]
        for par in self.stellar.params:
            self.params['st_'+par] = self.stellar.__dict__[par]
        for par in self.kernel.params:
            self.params['ker_'+par] = self.kernel.__dict__[par]
    
    def setparams(self, name, value):
        if name.strip('dm_') in self.dm.__dict__:
            setattr(self.dm, name.strip('dm_'), value)
        if name.strip('st_') in self.stellar.__dict__:
            setattr(self.stellar, name.strip('st_'), value)
        if name.strip('ker_') in self.kernel.__dict__:
            setattr(self.kernel, name.strip('ker_'), value)
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
        D = self.dwarf_props['D']
        theta = self.dwarf_props['theta']
        rt = self.dwarf_props['rt']
        errs = self.dwarf_props['errs']
        Jred = np.sqrt(self.dm.Jfactor(D, theta, rt, errs))
        if np.isscalar(R):
            integral, error = quad(self.integrand, R, np.inf, args=(R,))
            sigma2 = integral / self.stellar.surface_brightness(R) / Jred
        else:
            sigma2 = np.zeros_like(R)
            for i,rr in enumerate(R):
                integral, error = quad(self.integrand, rr, np.inf, args=(rr,))
                I_of_R = self.stellar.surface_brightness(rr)
                sigma2[i] = integral / I_of_R / Jred
        return sigma2 * self.dm.r0**3 * self.cst * np.power(10,self.J/2.)
