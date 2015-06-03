import math
import time
import numpy as np
from scipy.integrate import quad,quadrature
from scipy.interpolate import UnivariateSpline

# paramters
nmax 	 = 1000
nparams  = 7
pi 		 = math.pi
rstar	 = 0.3e0						# 0.3 kpc
kpctom 	 = 3.085e19				# kpc in m
Msun 	 = 1.9891e30 				# Solar mass unit
Mhalo 	 = 1.e9 * Msun		# Halo mass
sigma_MW = 200 						# velocity dispersion of Milky Way in km s^-1
G 		 = 6.67e-11*Msun		# m^3 Msun^-1 s^-2			


def get_data(gal):
    y    = np.empty([nmax])
    r    = np.empty([nmax])
    azm  = np.empty([nmax])
    pa   = np.empty([nparams])

 #  Read the parameter from the input file
    data = open('data/params/params_'+gal+'.dat','r').readlines()
    parameters = []
    for line in data:
        parameters.append(line.split(','))

    dwarfname       = str(parameters[0][0])             # name of the dwarf
    D               = float(parameters[1][0])           # distance to galaxy in kpc
    rh              = float(parameters[2][0])           # half-light radius
    rt              = float(parameters[3][0])           # tidal radius 
    like_val        = float(parameters[4][0])           # initial (arbitrary) value of the likelihood
    pmin = np.empty([0])
    pmax = np.empty([0])
    for i in range(5,len(data)):                        
        pmin = np.append(pmin,float(parameters[i][0]))  # extract min,max values of M300pc[Msun],log10(rs[kpc])
        pmax = np.append(pmax,float(parameters[i][1]))  # beta(velocity anisotropy) a,b,c NFW shape parameters
    
    velocities = open('data/velocities/velocities_'+gal+'.dat','r').readlines()
    x  = np.empty([0])
    v  = np.empty([0])
    dv = np.empty([0])
    for line in velocities:
        x  = np.append(x,float(line.split()[0]))        # star position
        v  = np.append(v,float(line.split()[1]))        # star velocity
        dv = np.append(dv,float(line.split()[2]))       # star velocity dispersion
    nstars = len(velocities)
    ave,adev,sdev,var,skew,curt = moment(v,nstars)  
    vsys_min = ave-6.e0
    vsys_max = ave+6.e0

    rmin = math.sqrt(x[0]**2)                                           #
    for i in range(nstars):                                             #   PROBABLY UNNECESSARY
        if math.sqrt(x[i]**2) < rmin : rmin = math.sqrt(x[i]**2)        #   PART OF CODE
    rmax = rt                                                           #

 #      Overwrite systematic velocity?
    pmin = np.append(pmin,vsys_min)
    pmax = np.append(pmax,vsys_max)
    pa = 0.5e0* (pmax+pmin)
    pa[1] = 10.**pa[1]

    return x,v,dv,rh,rt,nstars,D,pa


##########################################################################################################
#   get the mass, after the initial spline 

def M(x_in,rs,a,b,c):
    nmass = 25                              # differently from the Fortran code, init_mass()
    xa,Sy = init_mass(rs,a,b,c)             # now passes the function object UnivariateSpline
    x = np.log(x_in)
    if x.any() > xa[-1]:
        print 'M(r) called with r > r_tidal. Spline does not'
        print 'extend beyond r_tidal. Stopping.'
        print math.exp(xa[0]),math.exp(xa[nmass-1]),x_in
    if x.any() < xa[0]:
        print 'M(r) called with r < r_min where r_min is the'
        print 'innermost data point radius. Spline does not'
        print 'extend to r < r_min. Stopping.'
        print math.exp(xa[0]),math.exp(xa[nmass-1]),x_in
    M = np.exp(Sy(x))
    return M

##########################################################################################################
#   spline the mass distribution

def init_mass(rs,a,b,c):
    nmass  = 25
    rmin   = 1.e-4
    rmax   = 1.e1
    rmass  = np.empty([nmass])
    massa  = np.empty([nmass])
    mass2a = np.empty([nmass])
    dr = math.log(rmax/rmin)/(nmass-1.e0)
    for i in range(nmass):
        r = rmin*math.exp(dr*i)
        mass = get_M(r,rs,a,b,c)
        rmass[i] = math.log(r)
        massa[i] = math.log(mass)

    return rmass,UnivariateSpline(rmass,massa,s=0)

# with this Spline setting, the difference between M(r) obtained via integration or interpolation is O(10^-6)

##########################################################################################################

#   Get the line-of-sight velocity dispersion
#   Note the variable substution: t^2 = r-R, where r and R are 
#   defined in appendix eq 2 of Strigari et al, Nature 2008. 
#   This makes sure that the integral doesn't numerically diverge. 

def get_sigmalos(R,p0,rt,beta,rh,rs,a,b,c):
    a = 0.e0                                                                # lower bound of outer integral
    b = np.sqrt(rt-R)                                                       # upper bound of outer integral
    ss = quad(funcr,a,b,args=(R,rt,beta,rh,rs,a,b,c))[0]*p0/kpctom/1.e6     # outer integral [in km^2 s^-2]
    s  = math.sqrt(G*ss/istar(R,rh))                                        # projected 2-D velocity dispersion
    return s
    
##########################################################################################################
#   double integration

# integrand of Eq. 2 in notes (with variable substitution t^2 = r - R)
def funcs(s,beta,rh,rs,a,b,c):
    return pow(s,2.*beta-2.)*rhostar(s,rh)*M(s,rs,a,b,c)
# integrand of Eq. 3 in notes
def funcr(t,R,rt,beta,rh,rs,a,b,c):
    x = R + pow(t,2.)
    return (1-beta*pow(R,2)/pow(x,2))*pow(x,1.-2.*beta)*\
    quadrature(funcs,x,rt,args=(beta,rh,rs,a,b,c),maxiter=120)[0]/np.sqrt(x+R)

##########################################################################################################
#   3d stellar density

def rhostar(s,rh):
    return pow(1.e0+s**2/rh**2,-5./2.)
     
##########################################################################################################
#   projected 2d stellar density 

def istar(R,rh):
    return pow(1.e0+R**2/rh**2,-2.)*rh/3.e0

##########################################################################################################
#   evaluation of M(r)

def get_M(x,rs,a,b,c):
    return quad(dmass,0.e0,x,args=(rs,a,b,c))[0]

##########################################################################################################
#   integrand of M(r): 4pi*r^2*rho_DM(r)

def dmass(x,rs,a,b,c):
    return 4.e0*pi*pow(rs,a)*pow(x,2.e0-a)/pow(1.e0+pow(x/rs,b),(c-a)/b) # miss exponential cut 

##########################################################################################################
#   returned: ave,adev,sdev,var,skew,curt (moments of data)

def moment(data,n):
  if(n <= 1):
    print 'n must be at least 2 in moment'
  s = 0.
  for j in range(n):
    s += data[j]

  ave  = s/n
  adev = 0.
  var  = 0.
  skew = 0.
  curt = 0.
  eps  = 0.
  ep   = 0.
  for j in range(n):
    s     = data[j]-ave
    ep   += s
    adev += abs(s)
    p     = s*s
    var  += p
    p    *= s
    skew += p
    p    *= s
    curt += p

  adev /= n
  var  = (var-ep**2/n)/(n-1)
  sdev = math.sqrt(var)
  if(var != 0.):
    skew /= (n*sdev**3)
    curt /= (n*var**2)-3.
  else:
    print 'no skew or kurtosis when zero variance in moment'

  return ave,adev,sdev,var,skew,curt
