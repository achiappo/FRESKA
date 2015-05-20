import numpy as np
from scipy.integrate import quad,romberg,quadrature,dblquad,nquad
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
	rs  			= float(parameters[2][0]) 			# half-light radius
	rt  			= float(parameters[3][0]) 			# tidal radius 
	like_val		= float(parameters[4][0])		 	# initial (arbitrary) value of the likelihood
	pmin = np.empty([0])
	pmax = np.empty([0])
	for i in range(5,len(data)):						
		pmin = np.append(pmin,float(parameters[i][0]))	# extract min,max values of M300pc[Msun],log10(rs[kpc])
		pmax = np.append(pmax,float(parameters[i][1]))	# beta(velocity anisotropy) a,b,c NFW shape parameters
	
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

	return x,v,dv,rs,rt,nstars,D,like_val,pa,pmin,pmax

#########################################################################################

def dlike(gal):
	x,v,dv,rs,rt,nstars,D,like_val,pa = get_data(gal)[:9]
	Mvar,rs,beta,a,b,c,u = pa
	rs = 10.e0**rs
	arg1  = 0.e0
	p2 	  = 0.e0
	rcut = pow(G*Mhalo*pow(D,2)/2./pow(sigma_MW,2),1/3.)
	p0    = 10.e0**Mvar/M(rstar,pa,rcut)
	for i in range(nstars):
		radius = math.sqrt(pow(x[i],2))
        s = get_sigmalos(radius,gal,rcut,p0)
        arg1 += 0.5e0*pow(v[i]-u,2)/(pow(dv[i],2)+pow(s,2))
        p2   += 0.5e0*math.log(2*math.pi*(pow(dv[i],2)+pow(s,2)))
	q = -arg1-p2 + like_val
	dlike = math.exp(q)
	return dlike,p0

##########################################################################################################
#	get the mass, after the initial spline 

def M(x_in,pa,rcut):
	nmass = 25
	xa,ya,y2a = init_mass(pa,rcut)
	#print 'x_in = ',x_in
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
	klo = 0
	khi = nmass-1
	while khi-klo > 1:
		k = (khi+klo)/2
		if xa[k] > x :
			khi=k
		else:
			klo=k

	h = xa[khi]-xa[klo]
	if h == 0. : print 'bad xa input in splint'
	a = (xa[khi]-x)/h
	b = (x-xa[klo])/h
	y = a*ya[klo]+b*ya[khi]+(a**3-a)*y2a[klo]+(b**3-b)*y2a[khi]*(h**2)/6.
	M = math.exp(y)
	return M

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

#	Get the line-of-sight velocity dispersion
#	Note the variable substution: t^2 = r-R, where r and R are 
# 	defined in appendix eq 2 of Strigari et al, Nature 2008. 
# 	This makes sure that the integral doesn't numerically diverge. 

def get_sigmalos(R,gal,rcut,p0):
	rs,rt = get_data(gal)[3:5]
	pa  = get_data(gal)[8]
	beta = pa[2]
	a = 0.e0											# lower bound on outer integral
	b = math.sqrt(rt-R)									# upper bound on outer integral
	#lmt = {'points':np.linspace(a,b,10),'limit':100.}
	#ss = nquad(func,[[a,b],[lambda x:R+pow(x,2),lambda x:rt]],args=(R,gal,rcut,p0),opts=[lmt])
	#ss = dblquad(func,a,np.inf,lambda x:R+pow(x,2),lambda x:np.inf,args=(R,gal,rcut,p0))
	ss = dblquad(func,a,b,lambda x:R+pow(x,2),lambda x:rt,args=(R,gal,rcut,p0))
	#ss = 4.e0*G*quadrature(funcr,a,b,args=(rs,rt,beta,R,pa,rcut))[0]
	s  = math.sqrt(ss[0]/istar(R,rs))						# projected 2-D velocity dispersion
	return s 							# NO NEED TO ROOT IF I THEN SQUARE IT IN DLIKE !!!!!
    
##########################################################################################################
# integrand of Eq. 2 in Walker et al. 2009
def funcs(s,rs,beta,pa,rcut):
	return pow(s,2.*beta-2.)*rhostar(s,rs)*M(s,pa,rcut)
# integral of above
def int_funcs(t,rs,rt,beta,R,pa,rcut):
	return quadrature(funcs,lambda t:R+t**2,rt,args=(rs,beta,R,pa,rcut))[0]
# integrand of Eq. 3 Walker et al. 2009
def funcr(r,rs,rt,beta,R,pa,rcut):
	return (1-beta*R**2/(R+r**2)**2)*pow(R+r**2,1.-2.*beta)*\
	int_funcs(r,rs,rt,beta,R,pa,rcut)/math.sqrt(2*R+r**2)

# (alternative) two-dimensional function
def func(y,x,R,gal,rcut,p0):
#	valid only for constant beta
	rs = get_data(gal)[3]
	pa  = get_data(gal)[8]
	beta = pa[2]
	#print 'y = ',y
	func = 4.e0*pow(R+x**2,1.e0-2.e0*beta)*pow(y,2.e0*beta-2.e0)*rhostar(y,rs)*G*M(y,pa,rcut)*\
	(1-beta*R**2/(R+x**2)**2)/math.sqrt(2.e0*R+x**2)
	func *= p0 / mtokpc / 1.e6
	return func

##########################################################################################################
#	3d stellar density

def rhostar(x,rs):
	rhostar = (1.e0+x**2/rs**2)**(-5.e0/2.e0)
	return rhostar
     
##########################################################################################################
#	projected 2d stellar density 

def istar(R,rs):
	istar = 4.e0/3.e0*rs*(1.e0+R**2/rs**2)**(-2.e0)
	return istar

##########################################################################################################
#	evaluation of M(r)

def get_M(x,pa,rcut):
	return romberg(dmass,0.e0,x,args=(pa,rcut))

##########################################################################################################
#	integrand of M(r): 4pi*r^2*rho_DM(r)

def dmass(x,pa,rcut):
	r0 = pow(10.e0,pa[1])	# r0 of NFW coincides with rs, the half-light radius
	a,b,c = pa[-3:]			# a,b,c shape parameters of NFW profile
	return 4.e0*math.pi*pow(r0,a)*pow(x,2.e0-a)/pow(1.e0+pow(x/r0,b),(c-a)/b)

##########################################################################################################
#	returned: ave,adev,sdev,var,skew,curt (moments of data)

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
