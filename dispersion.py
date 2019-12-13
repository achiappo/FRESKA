
__author__ = "Johann Cohen Tanugi, Andrea Chiappo"
__email__ = "chiappo.andrea@gmail.com"

#
# This module contains the class to compute the 
# stellar line-of-sight velocity dispersion sigma2_los(R)
# (evaluated at the projected radial distance R from the center of the system)
#

import numpy as np
from sys import float_info
from scipy.integrate import quad
from profiles import DMProfile, StellarProfile, AnisotropyKernel

class SphericalJeansDispersion(object):
    """
    Class to evaluate the stellar line-of-sight velocity dispersion 
    predicted by Jeans equation
    """
    def __init__(self, dm, stellar, anisotropy, dwarf_props, **kwargs):
        """
        To build an istance, you must provide
        
        - dm : istance of StellarProfile class
               describing the Dark Matter component of the system
               
        - stellar : istance of StellarProfile class
                    describing the stellar component of the system
                    
        - anisotropy : istance of AnisotropyKernel class
                       describing the stellar velocity anisotropy
        
        - dwarf_props : dictionary containing information to compute the J-factor
                        the keys must be: 
                        > 'D' : distance to the center of the system 
                        > 'rt' : tidal radius of the system 
                        > 'theta' : aperture of the integration angle to calculate J
                        > 'with_errs' : (bool) option to return errors on J
        """
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
        self.cst = 2*G
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
        """
        function to update the values of the parameters
        entering the various profiles (stellar, dark matter, anisotropy) 
        
        input:
            - name : (full) parameter name
            - value : new parameter value
        output:
            (no returned value)
        """
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
        """
        integrand of the stellar velocity dispersion 
        as given by Jeans equation
        
        input:
            - s : integration variable
            - R : stellar projected radial distance
        output:
            integrand value
        """
        r0 = kwargs['r0'] if 'r0' in kwargs else self.dm.r0
        rh = kwargs['rh'] if 'rh' in kwargs else self.stellar.rh
        val = self.stellar.density(s/rh) * self.dm.mass(s/r0) * self.kernel(s, R) / s
        return val
    
    def sigma_integral(self, rr, epsrel=1.e-8, epsabs=1.e-8):
        """
        function to evaluate the integral of the stellar velocity dispersion
        as given by Jeans equation
        
        input: 
            - r : radial distance coordinate from center of the system 
        output:
            - result of integration (error omitted)
        """
        res = quad(self.integrand, rr, np.inf, args=(rr,),\
                   epsabs=epsabs, epsrel=epsrel)
        if res[0]/res[1] < 10 and res[1]<1:
            #find the scale of the error and force epsrel and epsabs to 
            #aim for one order of magnitude smaller error
            eexp = ("%e"%res[1]).split("e-")[1]
            neweps = eval("1.e-%d"%(int(eexp)+1))
            res = quad(self.integrand, rr, np.inf, args=(rr,),\
                       epsabs=neweps, epsrel=neweps, limit=1000)
        return res[0]
    
    def compute(self, R):
        """
        function to evaluate the integral entering 
        the velocity dispersion given by Jeans equation and 
        the J-factor of the dark matter distribution
        
        input: 
            - R : project radial distance from center of the system
        output: 
            - sigma_los(R) : stellar line-of-sight velocity dispersion
        """
        if any([getattr(self.dm,par)<0 for par in self.dm.params]):
            return float_info.max
        else:
            Jreduced = self.dm.cached_Jreduced( **self.dwarf_props )
            if 'with_errs' in self.dwarf_props and\
                self.dwarf_props['with_errs']==True:
                Jred = Jreduced[0]
                Jerr = Jreduced[1]
            else:
                Jred = Jreduced
            			
            if np.isscalar(R):
                integral = self.sigma_integral(R)
                sigma2 = integral / self.stellar.surface_brightness(R) / np.sqrt(Jred)
            else:
                sigma2 = np.zeros_like(R)
                for i,rr in enumerate(R):
                    eps = 1.e-8
                    integral = self.sigma_integral(rr, epsrel=eps,epsabs=eps)
                    I_of_R = self.stellar.surface_brightness(rr)
                    sigma2[i] = integral / I_of_R / np.sqrt(Jred)
            cst = self.cst / np.sqrt(self.dm.Jcst())
            return cst * np.power(10, self.J/2.) * sigma2

