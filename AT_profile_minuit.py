import math
import time
import numpy as np
from scipy.integrate import quad,quadrature
from scipy.interpolate import UnivariateSpline
from scipy.special import hyp2f1

# paramters
nmax 	 = 1000
nparams  = 7
pi 		 = math.pi
rstar	 = 0.3e0						# 0.3 kpc
kpctom 	 = 3.085e19						# kpc in m
Msun 	 = 1.9891e30 					# Solar mass unit
Mhalo 	 = 1.e9 * Msun					# Halo mass
sigma_MW = 200 							# velocity dispersion of Milky Way in km s^-1
G 		 = 6.67e-11*Msun				# m^3 Msun^-1 s^-2			


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

    dwarfname = str(parameters[0][0])             # name of the dwarf
    D         = float(parameters[1][0])           # distance to galaxy in kpc
    rh        = float(parameters[2][0])           # half-light radius
    rt        = float(parameters[3][0])           # tidal radius 
    like_val  = float(parameters[4][0])           # initial (arbitrary) value of the likelihood
    pmin = np.empty([0])
    pmax = np.empty([0])
    for i in range(5,len(data)):                        
        pmin = np.append(pmin,float(parameters[i][0]))  # extract min,max values of M300pc[Msun],log10(rs[kpc])
        pmax = np.append(pmax,float(parameters[i][1]))  # beta(velocity anisotropy) a,b,c NFW shape parameters
    
    x,v,dv = np.loadtxt('data/velocities/velocities_'+gal+'.dat',dtype=float,usecols=(0,1,2),unpack=True)
    nstars = len(x)
    ave = moment(v,nstars)[0]
    vsys_min = ave-6.e0
    vsys_max = ave+6.e0
 # Overwrite systematic velocity?
    pmin  = np.append(pmin,vsys_min)
    pmax  = np.append(pmax,vsys_max)
    pa    = 0.5*(pmax+pmin)
    pa[0] = 10.**pa[0]
    pa[1] = 10.**pa[1]

    return x,v,dv,rh,rt,nstars,D,pa

##########################################################################################################
#   Get the line-of-sight velocity dispersion (variable substution: t^2 = r-R, Strigari et al, Nature 2008)

def get_sigmalos(R,rt,rh,beta,rs,a,b,c):
    ss = quad(funcr,0.,np.inf,args=(R,rt,rh,beta,rs,a,b,c),epsabs=1.e-5)[0] # outer integral [in km^2 s^-2]
    s  = math.sqrt(G*ss/kpctom/1.e6/istar(R,rh))	                        # projected 2-D velocity dispersion
    return s
    
##########################################################################################################
#   double integration

# integrand of Eq. 2 in notes (with variable substitution t^2 = r - R)
def funcs(s,rh,beta,rs,a,b,c):
    return pow(s,2.*beta-2.)*rhostar(s,rh)*M(s,rs,a,b,c)
# integrand of Eq. 3 in notes
def funcr(t,R,rt,rh,beta,rs,a,b,c):
    x = R + pow(t,2.)
    return (1-beta*pow(R,2)/pow(x,2))*pow(x,1.-2.*beta)*\
    quad(funcs,x,np.inf,args=(rh,beta,rs,a,b,c),epsabs=1.e-5)[0]/np.sqrt(x+R)

##########################################################################################################
#   get the mass, after the initial spline 
'''	
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
'''
def M(x,rs,a,b,c):								# analytic formulation of M(r) in NFW case
	return 4*pi*(math.log(1+x/rs)-1/(rs/x+1))

##########################################################################################################
#   spline the mass distribution

def init_mass(rs,a,b,c):
    r = np.logspace(-4.,1.,num=25,base=np.e)
    x=r/rs
    #mass = 4*pi*rs**3*x**(3.-a)*hyp2f1((3.-a)/b, (c-a)/b, (-a+b+3)/b,-x**b)/(3.-a)
    mass = 4*pi*x**(3.-a)*hyp2f1((3.-a)/b,(c-a)/b,(-a+b+3)/b,-x**b)/(3.-a) # uncomment to fit rho0*rs**3
    rmass = np.log(r)
    massa = np.log(mass)

    return rmass,UnivariateSpline(rmass,massa,s=0)

# with this Spline setting, the difference between M(r) obtained via integration or interpolation is O(10^-6)

##########################################################################################################
#   3d stellar density

def rhostar(s,rh):
    return pow(1.+s**2/rh**2,-5./2.)
     
##########################################################################################################
#   projected 2d stellar density 

def istar(R,rh):
    return pow(1.+R**2/rh**2,-2.)*rh/3.

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
