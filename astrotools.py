import numpy as np
from scipy.integrate import quad,romberg,dblquad
from scipy.interpolate import UnivariateSpline
import math

# paramters
nmax 	 = 1000
nparams  = 7
pi 		 = 3.14159e0
rstar	 = 0.3e0
kpctokm	 = 3.0856e16
mtokpc   = 3.085e19
Msun 	 = 1.9891e30					# Solar mass unit
Mhalo 	 = 1.e9 * Msun					# Halo mass
sigma_MW = 200 							# velocity dispersion of Milky Way in km s^-1
G 		 = 6.67e-11*1.e30*1.e-9 		# km^3 Msun^-1 s^-2 

#	gets the data from the input file
def get_data(gal):
	y    = np.empty([nmax])
	r    = np.empty([nmax])
	azm  = np.empty([nmax])
	pa   = np.empty([nparams])

 #	Read the parameter from the input file
 	data = open('data/params/params_'+gal+'.dat','r').readlines()
 	parameters = []
	for line in data:
		parameters.append(line.split(','))

	dwarfname 	 	= str(parameters[0][0]) 			# name of the dwarf
	D 				= float(parameters[1][0]) 			# distance to galaxy in kpc
	rc  			= float(parameters[2][0]) 			# core radius
	rt  			= float(parameters[3][0]) 			# tidal radius 
	like_val		= float(parameters[4][0])		 	# initial (arbitrary) value of the likelihood
	pmin = np.empty([0])
	pmax = np.empty([0])
	for i in range(5,len(data)):
		pmin = np.append(pmin,float(parameters[i][0]))
		pmax = np.append(pmax,float(parameters[i][1]))
	
	velocities = open('data/velocities/velocities_'+gal+'.dat','r').readlines()
	x  = np.empty([0])
	v  = np.empty([0])
	dv = np.empty([0])
	for line in velocities:
		x  = np.append(x,float(line.split()[0]))		# star position
		v  = np.append(v,float(line.split()[1]))		# star velocity
		dv = np.append(dv,float(line.split()[2]))		# star velocity dispersion
	nstars = len(velocities)
	ave,adev,sdev,var,skew,curt = moment(v,nstars)	
	vsys_min = ave-6.e0
	vsys_max = ave+6.e0
	rmin = math.sqrt(x[0]**2)
	for i in range(nstars):
		if math.sqrt(x[i]**2) < rmin : rmin = math.sqrt(x[i]**2)
	
	rmax = rt

 #  	Overwrite systematic velocity?
	pmin = np.append(pmin,vsys_min)
	pmax = np.append(pmax,vsys_max)
	pa = 0.5e0* (pmax+pmin)

	return x,v,dv,rc,rt,nstars,D,like_val,pa,pmin,pmax

#########################################################################################

def dlike(gal):
	x,v,dv,rc,rt,nstars,D,like_val,pa = get_data(gal)[:9]
	Mvar  = pa[0]			# Mvar = mass within 300 pc. 
	rs 	  = 10.e0**pa[1]
	beta  = pa[2]
	a1    = pa[3]
	b1    = pa[4]	
	c1    = pa[5]
	u 	  = pa[6]
	arg1  = 0.e0
	p2 	  = 0.e0
	rcut = pow(G*Mhalo*pow(D,2)/2./pow(sigma_MW,2),1/3.)
	p0    = 10.e0**Mvar/M(rstar,pa,rcut)
	for i in range(nstars):
		radius = math.sqrt(pow(x[i],2))
        s = get_sigmalos(radius,gal,rc,p0)
        arg1 += 0.5e0*pow(v[i]-u,2)/(pow(dv[i],2)+pow(s,2))
        p2   += 0.5e0*math.log(pow(dv[i],2)+pow(s,2))		# added 2*pi
	q = -arg1-p2 + like_val
	dlike = math.exp(q)
	return dlike,p0

##########################################################################################################
#	spline the mass distribution

def init_mass(pa,rcut):
	nmass  = 25
	rmin   = 1.e-4
	rmax   = 1.e1
	rmass  = np.empty([nmass])
	massa  = np.empty([nmass])
	mass2a = np.empty([nmass])
	dr = math.log(rmax/rmin)/(nmass-1.e0)
	for i in range(nmass):
		#r = dr*(i-1.e0)+rmin
		r = rmin*math.exp(dr*i)
		mass = get_M(r,pa,rcut)
		rmass[i] = math.log(r)
        massa[i] = math.log(mass)

	Mspline = UnivariateSpline(rmass,massa,s=1)
	mass2a = Mspline(rmass)
	return rmass,massa,mass2a

##########################################################################################################
#	get the mass, after the initial spline 

def M(x_in,pa,rcut):
	nmass = 25
	xa,ya,y2a = init_mass(pa,rcut)

	x = math.log(x_in)
	if (x > xa[nmass-1]):
		print 'M(r) called with r > r_tidal. Spline does not'
		print 'extend beyond r_tidal. Stopping.'
		print math.exp(xa[0]),math.exp(xa[nmass-1]),x_in
	if (x < xa[0]):
		print 'M(r) called with r < r_min where r_min is the'
		print 'innermost data point radius. Spline does not'
		print 'extend to r < r_min. Stopping.'
		print math.exp(xa[0]),math.exp(xa[nmass-1]),x_in
	klo = 1
	khi = nmass
	while khi-klo > 1:
		k = (khi+klo)/2
		if xa[k] > x :
			khi=k
		else:
			klo=k

	h = xa[khi]-xa[klo]
	if h == 0. :
		print 'bad xa input in splint'
	a = (xa[khi]-x)/h
	b = (x-xa[klo])/h
	y = a*ya[klo]+b*ya[khi]+(a**3-a)*y2a[klo]+(b**3-b)*y2a[khi]*(h**2)/6.
	M = math.exp(y)
	return M

##########################################################################################################

#	Get the line-of-sight velocity dispersion
#	Note the variable substution: t^2 = r-R, where r and R are 
# 	defined in appendix eq 2 of Strigari et al, Nature 2008. 
# 	This makes sure that the integral doesn't numerically diverge. 

def get_sigmalos(xr,gal,rcut,p0):
	rc,rt = get_data(gal)[3:5]
	R = xr
	a = 0.e0 											# lower bound on outer integral
	b = math.sqrt(rt-R)									# upper bound on outer integral 
	ss = dblquad(func,a,b,lambda x:R*pow(x,2),lambda x:rt,args=(R,gal,rcut,p0))	# double integration
	s  = math.sqrt(ss[0]/istar(R,rc))						# projected 2-D velocity dispersion
	return s
    
##########################################################################################################

def func(y,x,R,gal,rcut,p0):
#	valid only for constant beta
#	r_3D = R_projected + x**2
	rc = get_data(gal)[3]
	pa  = get_data(gal)[8]
	beta = pa[2]
	func = 4.e0*(R+x**2)*((R+x**2)**(-2.e0*beta))*(y**(2.e0*beta))*rhostar(y,rc)*G*M(y,pa,rcut)*\
	(1-beta*R**2/(R+x**2)**2)/y/y/math.sqrt(2.e0*R+x**2)
	func *= p0 / mtokpc / 1.e6
	return func

##########################################################################################################
#	3d stellar density

def rhostar(x,rc):
	rhostar = (1.e0+x**2/rc**2)**(-5.e0/2.e0)
	return rhostar
     
##########################################################################################################
#	projected 2d stellar density 

def istar(R,rc):
	istar = 4.e0/3.e0*rc*(1.e0+R**2/rc**2)**(-2.e0)
	return istar

##########################################################################################################
#	M(r)

def get_M(x,pa,rcut):
	return quad(dmass,0.e0,x,args=(pa,rcut))[0]

##########################################################################################################
#	integrand of M(r): 4pi*r^2*rho_DM(r)

def dmass(x,pa,rcut):
	r0 	  = pow(10.e0,pa[1])		# r0 of NFW coincides with rt
	a1    = pa[3]					# a in NFW
	b1    = pa[4]					# b in NFW
	c1    = pa[5]					# c in NFW
	return 4.e0*math.pi*math.exp(-x/rcut)*pow(r0,a1)*pow(x,2.e0-a1)/pow(1.e0+pow(x/r0,b1),(c1-a1)/b1)

##########################################################################################################
#	returned: ave,adev,sdev,var,skew,curt (moments of data1)

def moment(data1,n):
  if(n <= 1):
    print 'n must be at least 2 in moment'
  s = 0.
  for j in range(n):
    s += data1[j]

  ave  = s/n
  adev = 0.
  var  = 0.
  skew = 0.
  curt = 0.
  eps  = 0.
  ep   = 0.
  for j in range(n):
    s     = data1[j]-ave
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
