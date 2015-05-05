from sys import arg
import numpy as np
import NR
import math

# paramters

nmax 	= 1000
nparams = 7
pi 		= 3.14159e0
rstar	= 0.3e0
kpctokm	= 3.0856e16
G 		= 6.67e-11*1.e30*1.e-9 # km^3 Msun^-1 s^-2 

#########################################################################################

#	Get the values for the next step in mcmc. For now next step is just randomly 
# 	determined within the fixed parameter ranges described above. 
      
def getparams(pa,pmin,pmax,xx):
	for i in range(nparams):
		pa[i] = xx[i]*(pmax[i]-pmin[i])+pmin[i]
    
    return pa

#########################################################################################

def dlike(pa):
	iwksp 	= np.empty([nmax])
	v_arr 	= np.empty([nmax])
	x 		= np.empty([nmax])
	y 		= np.empty([nmax])
	v 		= np.empty([nmax])
	dv		= np.empty([nmax])
	azm 	= np.empty([nmax])
	r_arr 	= np.empty([nmax])
	wksp 	= np.empty([nmax])
	y2 		= np.empty([nmax])
	y2a 	= np.empty([nmax])   
	radius_store = np.empty([nmax])
	sigma_store  = np.empty([nmax])

	yp1   = 0.
	ypn   = 0.
	Mvar  = pa[1]			# Mvar = mass within 300 pc. 
	r 	  = 10.e0**pa[2]
	beta  = pa[3]
	a1 	  = pa[4]
	b1 	  = pa[5]
	c1 	  = pa[6]
	u 	  = pa[7]
	p0 	  = 10.e0**Mvar/M(rstar)
	p2 	  = 0.e0
	arg1  = 0.e0 

	y2 = init_mass()

	# 	start of the loop for calculation of the likelihood function 
	# 	to speed up, use the spline and the if statement. 
	#
	#   call spline(radius_store,sigma_store,nrs,yp1,ypn,y2)
      
	for i in range(nstars):
		radius = math.sqrt(x[i]**2)
	#	if radius < 0.02e0 :
	#		splint(radius_store,sigma_store,y2a,nrs,radius,s)
	#	else:
        s = get_sigmalos(radius) 				# correct the indentation in case of IF usage
        arg1 += 0.5e0*(v[i]-u)**2/(dv[i]**2+s**2)
        p2   += 0.5e0*math.log(s**2+dv[i]**2)

    q = -arg1-p2 + like_val 				# add an arbitrary number to the likelihood to get a reasonable numerical value
    dlike = math.exp(q)						# Mrtstore 

    return dlike
#	Only use for a Vmax-rmax prior. Currently unused. 
#
#	vmax_calc = (math.log10(rmax_halo)+0.196e0)/1.35e0 + 1.e0 
#	sigmavmax = 0.20e0
#   lognormal = math.exp(-0.5e0*((vmax_calc-math.log10(vmax_halo))/sigmavmax)**2)/sigmavmax/vmax_halo**5   
#	dlike = dlike*lognormal
#	print*,dlike,lognormal 

##########################################################################################################

#	spline the mass distribution

def init_mass():
	nmass  = 25
	rmin   = 1.3-4
	rmax   = 1.31
	rmass  = np.empty([nmass])
	massa  = np.empty([nmass])
	mass2a = np.empty([nmass])
	dr = math.log(rmax/rmin)/(nmass-1.e0)
	for i in range(nmass):
		r = dr*(i-1.e0)+rmin
		r = rmin*math.exp(dr*(i-1))
		mass = get_M(r)
		rmass[i] = math.log(r)
        massa[i] = math.log(mass)
	
	return NR.spline(rmass,massa,nmass,1.e30,1.e30,mass2a)

##########################################################################################################

#	Get the line-of-sight velocity dispersion
#	Note the variable substution: t^2 = r-R, where r and R are 
# 	defined in appendix eq 2 of Strigari et al, Nature 2008. 
# 	This makes sure that the integral doesn't numerically diverge. 

def get_sigmalos(xr):
	R = xr
	a = 0.e0 						# lower bound on outer integral (Note that the variable is t as defined above)
	b = math.sqrt(rt-R)				# upper bound on outer integral 

	ss = QUAD3D(a,b)				# call the integration routine betweeb a and b
	s  = math.sqrt(ss/istar(R))		# projected 2-D velocity dispersion 

    return s
    
##########################################################################################################

def QUAD3D(x1,x2):
	return QGAUSSX(h,x1,x2) 

##########################################################################################################

def h(x):
	return QGAUSSY(f,y1(x),y2(x))

##########################################################################################################

def f(y):
	return func(x,y)

##########################################################################################################

def QGAUSSX(function,a,b):
	w = np.array([.2955242247, .2692667193, .2190863625, .1494513491,.0666713443])
	x = np.array([.1488743389, .4333953941, .6794095682, .8650633666,.9739065285])
	xm = 0.5*(b+a)
	xr = 0.5*(b-a)
	ss = 0.
	for j in range(5):
		dx  = xr*x[j]
		ss += w[j]*(function(xm+dx)+function(xm-dx))
	ss *= xr
	return ss

##########################################################################################################

def QGAUSSY(function,a,b):
	w = np.array([.2955242247, .2692667193, .2190863625, .1494513491,.0666713443])
	x = np.array([.1488743389, .4333953941, .6794095682, .8650633666,.9739065285])
	xm = 0.5*(b+a)
	xr = 0.5*(b-a)
	ss = 0
	for j in range(5):
		dx  = xr*x[j]
		ss += w[j]*(function(xm+dx)+function(xm-dx))
	ss *= xr
	return ss

##########################################################################################################

#	The actual 2d integral to evaluate. 
#	Note the variable transformation described in get_sigmalos

def func(x,y):
	mtokpc = 3.085e+19
#	valid only for constant beta
#	r_3D = R_projected + x**2
	func = 2.e0*2.e0*(R+x*x)*((R+x*x)**(-2.e0*beta))*(y**(2.e0*beta))*rhostar(y)*Gn*M(y)*\
	(1-beta*R*R/(R+x*x)**2)/y/y4/math.sqrt(2.e0*R+x*x)
	r3d  = R+x*x
#	func = 2.d0*2.d0*r3d*exp(-2.d0*beta_int(r3d))*exp(2.d0*beta_int(y))*rhostar(y)*G*M(y)*\
#	(1-beta_func(r3d)*R*R/r3d**2)/y/y/dsqrt(R+r3d)
	func *= p0
	func /= mtokpc
	func /= 1.e+6 	#	Convert m^2 to km^2 
	return func

##########################################################################################################

#	Lower bound for inner component of the 2D integral
#	Note the variable transformation described in get_sigmalos

def y1(x):
    return R + x*x 

##########################################################################################################

#	Upper bound for the inner compoment of the 2D integral. 
#	Note the variable transformation described in get_sigmalos

def y2(x):
	y2 = rt 
	return

##########################################################################################################

#	3d stellar density

def rhostar(x):
#      if( surface_density_flag .eq. 1)then 
#         z = (1+x*x/rc/rc)/(1+rt*rt/rc/rc)
#         z = math.sqrt(z) 
#         if(x < rt):
#         	rhostar=(math.cos(z)/z-math.sqrt(1-z*z))/z/z/pi/rc/(1.e0+rt**2/rc**2)**1.5e0 
#         else:
#            rhostar = 0.e0
#      if( surface_density_flag .eq. 2 )then  
	rhostar = (1.e0+x**2/rc**2)**(-5.e0/2.e0)
	return rhostar

##########################################################################################################

#	get the mass, after the initial spline 

def M(x_in):
	nmass = 25
	xa  = np.empty([nmass])
	ya  = np.empty([nmass])
	y2a = np.empty([nmass])
	
	x = math.log(x_in)
	if (x > xa[nmass]):
		print 'M(r) called with r > r_tidal. Spline does not'
		print 'extend beyond r_tidal. Stopping.'
		print math.exp(xa[1]),math.exp(xa[nmass]),x_in
		break
	if (x < xa[1]):
		print 'M(r) called with r < r_min where r_min is the'
		print 'innermost data point radius. Spline does not'
		print'extend to r < r_min. Stopping.'
		print math.exp(xa[1]),math.exp(xa[nmass]),x_in
		break
	klo = 1
	khi = nmass
	while khi-klo > 1:
		k = (khi+klo)/2
		if xa[k] > x :
			khi=k
		else:
			klo=k

	h = xa[khi]-xa[klo]
	if h != 0. :
		print 'bad xa input in splint'
		break
	a = (xa[khi]-x)/h
	b = (x-xa[klo])/h
	y = a*ya[klo]+b*ya[khi]+(a**3-a)*y2a[klo]+(b**3-b)*y2a[khi])*(h**2)/6.
	M = math.exp(y)
	return M
     
##########################################################################################################

#	projected 2d stellar density 

def istar(R):
#	if(surface_density_flag == 1):
#		istar =(1.d0/math.sqrt(1.e0+R**2/rc**2)-1.e0/math.sqrt(1.e0+rt**2/rc**2))**2
#	if(surface_density_flag == 2):
	istar = 4.e0/3.e0*rc*(1.e0+R**2/rc**2)**(-2.e0)
#	endif 
	return istar

##########################################################################################################

def get_M(x):
	tol=1.e-4
	return NR.rombint(dmass,0.e0,x,tol)

##########################################################################################################

def dmass(x):
	pi = 3.14159e0
	return 4.e0*pi*pstar[x]

##########################################################################################################

#	dark matter density profile 
	
def pstar(x):
	return math.exp(-x/rcut*0.e0)*r0**a1*x**(2.e0-a1)/(1.e0+(x/r0)**b1)**( (c1-a1)/b1 )
	
