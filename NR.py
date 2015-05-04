import math
import numpy as np

def dfridr(func,x,h,err):
  CON  = 1.4
  CON2 = CON*CON
  BIG  = 1.e30
  NTAB = 10
  SAFE = 2.
  if(h == 0.):
    print 'h must be nonzero in dfridr'
    break
  hh = h
  a  = np.empty([NTAB,NTAB])
  a[1,1] = (func(x+hh)-func(x-hh))/(2.0*hh)
  err=BIG
  for i in range(1,NTAB):
    x = hh/CON
    a[1,i] = (func(x+hh)-func(x-hh))/(2.0*hh)
    fac = CON2
    for j in range(1,i):
      a[j,i] = (a[j-1,i]*fac-a[j-1,i-1])/(fac-1.)
      fac = CON2*fac
      errt = max(abs(a[j,i]-a[j-1,i]),abs[a(j,i]-a[j-1,i-1]))
      if (errt <= err):
        err = errt
        dfridr = a[j,i]
    if (abs(a[i,i]-a[i-1,i-1]) >= SAFE*err):
      return
  return

def rombint(f,a,b,tol):
# Rombint returns the integral from a to b of using Romberg integration.
# The method converges provided that f(x) is continuous in (a,b).
# f must be double precision and must be declared external in the calling
# routine.  tol indicates the desired relative accuracy in the integral.
  MAXITER = 40
  MAXJ  = 5
  h     = 0.5e0*(b-a)
  gmax  = h*(f(a)+f(b))
  g     = np.empty([MAXJ+1])
  g[1]  = gmax
  nint  = 1
  error = 1.0e20
  i = 1
  while (i > MAXITER or (i > 5 and abs(error) < tol)) == False:
    i = i+1                     # Calculate next trapezoidal rule approximation to integral.
    g0 = 0.0e0
    for k=1 in range(nint):
      g0 = g0+f(a+(k+k-1)*h) 
    g0 = 0.5e0*g[1]+h*g0
    h = 0.5e0*h
    nint *= 2
    jmax = min(i,MAXJ)
    fourj = 1.0e0
    for j=1 in range(jmax):     # Use Richardson extrapolation.
      fourj = 4.0e0*fourj
      g1 = g0+(g0-g[j])/(fourj-1.0e0)
      g[j] = g0
      g0 = g1
    if (abs(g0) > tol):
      error = 1.0e0-gmax/g0
    else:
      error = gmax
    gmax = g0
    g[jmax+1] = g0
  else:
    rombint=g0
  if (i > MAXITER and abs(error) > tol):
    print 'Rombint failed to converge; integral, error = ',rombint,' , ',error
  return

#########################################################################################################
#
#     Fifth order Runge-Kutta Method with Adaptive Stepsize.
#     Integrate 'func' with parameters array fp(np) which contains any extra parameters other than the
#     integration variable from a to b, with initial step size dxinit and fractional accuracy eps.
#
#     In other words,
#          _b
#         /
#        |
#        |  FUNC(x,fp)dx
#        |
#        /
#       _ a
#
#     fp of length np, (i.e. real*8 fp(np)) contains all variables other
#     than the integration variable, say, x.
#
#########################################################################################################

def INTEGRATE(FUNC,fp,np,a,b,dxinit,eps):
  fp = np.empty([np])
  maxsteps =1.e8
  Nstep = 0
  x  = a
  dx = dxinit
  y  = 0.e0

  while (x-b)*(b-a) < 0.e0 and Nstep < maxsteps:
    Nstep += 1
    dydx = FUNC(x,fp,np)
    yscale = max(abs(y) + abs(dx*dydx), 1.e-12)
    if ((x+dx-b)*(x+dx-a) > 0.e0): dx = b - x 				# If stepsize overshoots, decrease it.
    dxnext = RUNGE5VAR(y,dydx,x,dx,eps,yscale,dxnext,FUNC,fp,np)[2]
    dx = dxnext
  
  if (Nstep >= maxsteps): print 'WARNING: failed to converge in INTEGRATE.'
  y = RUNGE5VAR(y,dydx,x,dx,eps,yscale,dxnext,FUNC,fp,np)[0]
  INTEGRATE = y

  return
  
###################################################################################################################
#
#   Fifth-order Runge-Kutta step with monitoring of local truncation error to ensure accuracy and adjust stepsize.
#   Input are the dependent-variable y and its derivative dydx at the starting value of the independent variable x.
#   Also input are the stepsize to be attempted htry, the required accuracy eps, and the value yscale, against 
#   which the error is scaled.  On output, y and x are replaced by their new values. hdid is the stepsize that
#   was actually accomplished, and hnext is the estimated next stepsize. DERIVS is the user-supplied routine that
#   computes right-hand-side derivatives.  The argument fparm is for an optional second argument to DERIVS 
#   (NOT integrated over).
#
###################################################################################################################

 def RUNGE5VAR(y,dydx,x,htry,eps,yscale,hnext,DERIVS,fp,np):
  safety  =  0.9e0
  pgrow   = -0.2e0
  pshrink = -0.25e0
  errcon  =  1.89e-4
  errmax  = 10.e0
  ytemp = 0.
  yerr  = 0.
  h = htry													# Set stepsize to initial accuracy.
  
  while errmax > 1.e0 :
    yerr = RUNGE(y,dydx,x,h,ytemp,yerr,DERIVS,fp,np)[1]
    errmax = abs(yerr/yscale)/eps               			# Scale relative to required accuracy.
    if (errmax > 1.e0):                         			# Truncation error too large; reduce h
      htemp = safety*h*(errmax**pshrink)
      hold = h
      h = math.copysign(max(abs(htemp),0.1e0*abs(h)),h)  	# No more than factor of 10
      xnew = x + h
      if (xnew == x):
        print 'WARNING: ','Stepsize underflow in RUNGE5VAR().'
        h = hold
        errmax = 0.e0

#     Step succeeded.  Compute estimated size of next step.
  if (errmax > errcon):
    hnext = safety*h*(errmax**pgrow)
  else:
    hnext = 5.e0 * h                            			# No more than factor of 5 increase.
    
  x = x + h
  y = ytemp

  return y,x,hnext
 
###################################################################################################################
# 
#     Given values for a variable y and its derivative dydx known at x, use the fifth-order Cash-Karp Runge-Kutta 
#     method to advance the solution over an interval h and return the incremented variables as yout. Also
#     return an estimate of the local truncation error in yout using the embedded fourth order method.  The user 
#     supplies the routine DERIVS(x,y,dydx), which returns derivatives dydx at x.
#
###################################################################################################################

def RUNGE(y,dydx,x,h,yout,yerr,DERIVS,fp,np):
  a2  = 0.2e0
  a3  = 0.3e0
  a4  = 0.6e0
  a5  = 1.e0
  a6  = 0.875e0
  c1  = 37.e0/378.e0
  c3  = 250.e0/621.e0
  c4  = 125.e0/594.e0
  c6  = 512.e0/1771.e0
  dc1 = c1 - 2825.e0/27648.e0
  dc3 = c3 - 18575.e0/48384.e0
  dc4 = c4 - 13525.e0/55296.e0
  dc5 = -277.e0/14336.e0
  dc6 = c6 - 0.25e0

  ak3 = DERIVS(x+a3*h,fp,np)
  ak4 = DERIVS(x+a4*h,fp,np)
  ak5 = DERIVS(x+a5*h,fp,np)
  ak6 = DERIVS(x+a6*h,fp,np)

# Estimate the fifth order value.
  yout = y + h*(c1*dydx + c3*ak3 + c4*ak4  + c6*ak6)

# Estimate error as difference between fourth and fifth order
  yerr = h*(dc1*dydx + dc3*ak3 + dc4*ak4 + dc5*ak5 + dc6*ak6)

  return yout,yerr

###################################################################################################################
#   Spline fit subroutines
##########################

#	returned: y2

def spline(x,y,n,yp1,ypn,y2):
  NMAX=100010
  u  = np.empty([NMAX])

  if (yp1 > 0.99e30):
    y2[0] = 0.e0
    u[0]  = 0.e0
  else:
    y2[0] = -0.5e0
    u[0]  = (3.e0/(x[2]-x[1]))*((y[2]-y[1])/(x[2]-x[1])-yp1)
  
  for i in range(1,n):
    sig = (x[i]-x[i-1])/(x[i+1]-x[i-1])
    p   = sig*y2[i-1]+2.e0
    y2[i] = (sig-1.e0)/p
    u[i]  = (6.e0*((y[i+1]-y[i])/(x[i+1]-x[i])-(y[i]-y[i-1])/(x[i]-x[i-1]))/(x[i+1]-x[i-1])-sig*u[i-1])/p
  
  if (ypn > .99e30):
    qn=0.e0
    un=0.e0
  else:
    qn=0.5e0
    un=(3.e0/(x[n]-x[n-1]))*(ypn-(y[n]-y[n-1])/(x[n]-x[n-1]))
  
  y2[n]=(un-qn*u[n-1])/(qn*y2[n-1]+1.e0)
  
  for k in range(n-1,0,-1):
    y2[k]=y2[k]*y2[k+1]+u[k]
  
  return y2

#---------------------------------------------------------------------------------

#	returned: y

def splint(xa,ya,y2a,n,x,y):
	klo = 1
	khi = n

  while khi-klo > 1:
    k=(khi+klo)/2
    if(xa[k] > x):
      khi=k
    else:
      klo=k
    
  h=xa[khi]-xa[klo]

  if(h == 0.): 
    print 'bad xa input in splint'
    break
  
  a = (xa[khi]-x)/h
  b = (x-xa[klo])/h
  y = a*ya[klo]+b*ya[khi]+((a**3-a)*y2a[klo]+(b**3-b)*y2a[khi])*(h**2)/6.
  
  return y

#---------------------------------------------------------------------------------

#	returned: ran2

def ran2(idum):
	IM1   = 2147483563
 	IM2   = 2147483399
 	AM    = 1./IM1
 	IMM1  = IM1-1
 	IA1   = 40014
 	IA2   = 40692
 	IQ1   = 53668
 	IQ2   = 52774
 	IR1   = 12211
 	IR2   = 3791
 	NTAB  = 32
 	NDIV  = 1+IMM1/NTAB
 	EPS   = 1.2e-7
 	RNMX  = 1.-EPS
 	idum2 = 123456789
 	iv    = np.zeros([NTAB])
 	iy    = 0

	if (idum <= 0):
		idum  = max(-idum,1)
		idum2 = idum
		for j in range(NTAB+8,0,-1):
			k = idum/IQ1
			idum = IA1*(idum-k*IQ1)-k*IR1
			if (idum < 0): idum += IM1
			if (j <= NTAB): iv[j] = idum
	
		iy = iv[0]
  
  k = idum/IQ1
  idum = IA1*(idum-k*IQ1)-k*IR1
  if (idum < 0): idum += IM1
  k = idum2/IQ2
  idum2 = IA2*(idum2-k*IQ2)-k*IR2
  if (idum2 < 0): idum2 += IM2
  j = 1+iy/NDIV
  iy = iv[j]-idum2
  iv[j] = idum
  if(iy < 1): iy += IMM1
  ran2 = min(AM*iy,RNMX)

  return

#---------------------------------------------------------------------------------

#	returned: gasdev

def gasdev(idum):
	iset = 0
	rsq  = 0
	if (iset == 0):
		while rsq >= 1. or rsq == 0.:
			v1  = 2.*ran1(idum)-1.
			v2  = 2.*ran1(idum)-1.
			rsq = v1**2+v2**2
    
  	fac = math.sqrt(-2.*math.log(rsq)/rsq)
   	gset = v1*fac
   	gasdev = v2*fac
   	iset = 1
  else:
   	gasdev = gset
   	iset = 0
  	
  return

#---------------------------------------------------------------------------------

#	returned: ran1

def ran1(idum):
	IA 	 = 16807
 	IM   = 2147483647
 	AM   = 1./IM
 	IQ   = 127773
 	IR 	 = 2836
 	NTAB = 32
 	NDIV = 1+(IM-1)/NTAB
 	EPS  = 1.2e-7
 	RNMX = 1.-EPS
 	iv   = np.zeros([NTAB])
 	iy   = 0

 	if idum <= 0 or iy == 0:
    idum = max(-idum,1)
   	for j in range(NTAB+8,0,-1):
     		k = idum/IQ
     		idum = IA*(idum-k*IQ)-IR*k
     		if (idum < 0): idum += IM
     		if (j <= NTAB): iv[j] = idum
   	iy = iv[0]
  	
 	k = idum/IQ
 	idum = IA*(idum-k*IQ)-IR*k
 	if (idum < 0): idum += IM
 	j = 1+iy/NDIV
 	iy = iv[j]
 	iv[j] = idum
 	ran1 = min(AM*iy,RNMX)
  	
 	return

#---------------------------------------------------------------------------------

#	returned: a

def choldc(a,n,np,p):
  
 	for i in range(0,n):
 		for j in range(i,n):
    		sum = a[i,j]
      		for k in range(i-1,0,-1):
        		sum -= a[i,k]*a[j,k]
      		if(i == j):
        		if(sum <= 0.):
          		print 'choldc failed'
          		break
        		p[i] = math.sqrt(sum)
      		else:
	        	a[j,i]=sum/p[i]
    	
   return a,p                                    # CHECK RETURN VALUE

#---------------------------------------------------------------------------------

#	returned: a,d,v

def jacobi(a,n,np,d,v,nrot):
	NMAX=500
 	b = np.empty([NMAX])
 	z = np.empty([NMAX])
  	
 	for ip in range(n):
   	for iq in range(n):
     		v[ip,iq] = 0.
   	v[ip,ip] = 1.
 	for ip in range(n):
   	b[ip] = a[ip,ip]
   	d[ip] = b[ip]
   	z[ip] = 0.
  	
 	nrot=0
 	for i in range(50):
   	sm=0.
   	for ip in range(n-1):
     		for iq in range(ip+1,n):
     			sm += abs(a[ip,i])
 		
		if(sm == 0.): return
 		if(i < 4):
      tresh = 0.2*sm/n**2
 		else:
   		tresh = 0.
  		
 		for ip in range(n-1):
   		for iq in range(ip+1,n):
     			g = 100.*abs(a[ip,iq])
     			if (i > 4) and (abs(d[ip])+g == abs(d(ip))) and (abs(d[iq])+g == abs(d[iq])):
       			a[p,iq]=0.
     			elif abs(a[ip,iq]) > tresh:
       			h=d[iq]-d[ip]
       			if(abs[h]+g == abs[h]):
         				t = a[ip,iq]/h
       			else:
         				theta = 0.5*h/a[ip,iq]
         				t = 1./(abs(theta)+math.sqrt(1.+theta**2))
         				if theta <0.: t = -t
        
       			c = 1./math.sqrt(1+t**2)
       			s = t*c
       			tau = s/(1.+c)
       			h = t*a[ip,iq]
       			z[ip] -= h
       			z[iq] += h
       			d[ip] -= h
       			d[iq] += h
       			a[ip,iq] = 0.
       			for j in range(ip-1):
       				g = a[j,ip]
       				h = a[j,iq]
       				a[j,ip] = g-s*(h+g*tau)
       				a[j,iq] = h+s*(g-h*tau)
       			
       			for j in range(ip,iq-1):
       				g = a[ip,j]
       				h = a[j,iq]
       				a[ip,j] = g-s*(h+g*tau)
       				a[j,iq] = h+s*(g-h*tau)
       
        			for j in range(iq,n):
        				g = a[ip,j]
        				h = a[iq,j]
         				a[ip,j] = g-s*(h+g*tau)
         				a[iq,j] = h+s*(g-h*tau)
        
        			for j in range(n):
        				g = v[j,ip]
         				h = v[j,iq]
        				v[j,ip] = g-s*(h+g*tau)
        				v[j,iq] = h+s*(g-h*tau)
        			
       			nrot += 1
    for ip in range(n):
    	b[ip] += z[ip]
   	 	d[ip] = b[ip]
   	 	z[ip] = 0.
    
  	print 'too many iterations in jacobi'
  	break

 	return a,d,v 									               # CHECK RETURN VALUE

#---------------------------------------------------------------------------------

#	returned: ave,adev,sdev,var,skew,curt (moments of data1)

def moment(data1,n,ave,adev,sdev,var,skew,curt):
	if(n <= 1):
   	print 'n must be at least 2 in moment'
   	break
 	s = 0.
 	for j in range(n):
   	s += data1[j]
	  	
 	ave  = s/n
 	adev = 0.
 	var  = 0.
 	skew = 0.
 	curt = 0.
 	eps  = 0.
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
   	break
 	
 	return ave,adev,sdev,var,skew,curt

#---------------------------------------------------------------------------------

#	returned: gammp

def gammp(a,x):
  	
 	if x < 0. or a <= 0.:
   	print 'bad arguments in gammp'
   	break
 	
 	if x < a+1.:
   	gammp = gser(gamser,a,x,gln)
 	else:
    gammp = 1.- gcf(gammcf,a,x,gln)
  	
 	return gammp

#---------------------------------------------------------------------------------

#	returned: gammcf

def gcf(gammcf,a,x,gln):
 	ITMAX = 100
 	EPS   = 3.e-7
 	FPMIN = 1.e-30
 	gln   = gammln(a)
 	b     = x+1.-a
 	c 	  = 1./FPMIN
 	d     = 1./b
 	h 	  = d
 	for i in range(ITMAX):
   	an = -i*(i-a)
   	b += 2.
   	d  = an*d+b
   	if abs(d) < FPMIN : d = FPMIN
   	c  = b+an/c
   	if abs(c) < FPMIN : c = FPMIN
   	d  = 1./d
   	de = d*c
   	h *= de
   	if abs(de-1.) < EPS :
   		print 'a too large, ITMAX too small in gcf'
   		break
  
  gammcf = math.exp(-x+a*math.log(x)-gln)*h
  
  return gammcf

#---------------------------------------------------------------------------------

#	returned: gamser

def gser(gamser,a,x,gln):
 	ITMAX = 100
 	EPS   = 3.e-7
 	gln   = gammln(a)
 	if x <= 0. :
   	if(x < 0.): 
     		print 'x < 0 in gser'
     		break
    	
   	gamser=0.
   	return gamser
  	
 	ap  = a
 	sum = 1./a
 	de  = sum
 	for n in range(ITMAX):
   	ap += 1.
   	de *= x/ap
   	sum += de
   	if abs(de) < abs(sum)*EPS :
 			print 'a too large, ITMAX too small in gser'
 			break
  
 	gamser = sum*exp(-x+a*math.log(x)-gln)
  
 	return gamser

#---------------------------------------------------------------------------------      

#	returned: gammln

def gammln(xx):
	cof = np.array([76.18009172947146e0,-86.50532032941677e0,24.01409824083091e0,\
		-1.231739572450155e0,.1208650973866179e-2,-.5395239384953e-5])
	stp = np.array([76.18009172947146e0,-86.50532032941677e0,24.01409824083091e0,\
		-1.231739572450155e0,.1208650973866179e-2,-.5395239384953e-5,2.5066282746310005e0])
 	x   = xx
 	y   = x
 	tmp = x+5.5e0
 	tmp = (x+0.5e0)*math.log(tmp)-tmp
 	ser=1.000000000190015e0
 	for j in range(6):
   	y += 1.e0
   	ser += cof[j]/y
  	
 	gammln = tmp+np.log(stp*ser/x)
  
 	return gammln

#---------------------------------------------------------------------------------      

#	returned: rtbis

def rtbis(func,x1,x2,xacc):
	JMAX = 500
 	fmid = func(x2)
 	f    = func(x1)
 	if f*fmid >= 0.e0 :
 		print 'root must be bracketed in rtbis'
   	break
  	
 	if f < 0.e0 :
   	rtbis = x1
   	dx 	  = x2-x1
 	else:
   	rtbis = x2
   	dx    = x1-x2
  	
 	for j in range(JMAX):
   	dx  *= 0.5e0
   	xmid = rtbis+dx
   	fmid = func(xmid)
   	if fmid <= 0.e0 : rtbis = xmid
   	if abs(dx) < xacc or fmid == 0.e0 : return rtbis
  	
 	print 'too many bisections in rtbis'
 	break

#---------------------------------------------------------------------------------      

#	returned: 

def zbrac(func,x1,x2,succes):
 	FACTOR = 1.6e0
 	NTRY   = 1000
 	if x1 == x2:
   	print 'you have to guess an initial range in zbrac'
   	break
  	
 	f1 = func(x1)
 	f2 = func(x2)
 	succes = True
 	for j in range(NTRY):
   	if f1*f2 < 0.e0 : return f1,f2
   	if abs(f1) < abs(f2) :
   		x1 += FACTOR*(x1-x2)
 	  	f1  = func(x1)
   	else:
   		x2 += FACTOR*(x2-x1)
   		f2  = func(x2)
  	
 	succes = False
 	return f1,f2      									# CHECK RETURN VALUE

#---------------------------------------------------------------------------------      

#	returned: 

def zbrak(fx,x1,x2,n,xb1,xb2,nb):
 	nbb = 0
 	x   = x1
 	dx  = (x2-x1)/n
 	fp  = fx(x)
 	for i in range(n):
   	x += dx
   	fc = fx(x)
   	if fc*fp < 0.e0 :
   		nbb 	+= 1
   		xb1[nbb] = x-dx
   		xb2[nbb] = x
   		if nbb == nb : break
    	
   	fp=fc
 	
 	nb=nbb
 	return xb1,xb2,nb 			     					# CHECK RETURN VALUE

#---------------------------------------------------------------------------------      

#	returned: zbrent

def zbrent(func,x1,x2,tol):
	ITMAX = 100
 	EPS   = 3.e-8
 	a  = x1
 	b  = x2
 	fa = func(a)
 	fb = func(b)
 	if (fa >0. and fb > 0.) or (fa < 0. and fb < 0.) :
   	print 'root must be bracketed for zbrent'
  	
 	c  = b
 	fc = fb
 	for ite in range(ITMAX):
   	if (fb > 0 and fc > 0.) or (fb < 0. and fc < 0.) :
   		c  = a
   		fc = fa
   		d  = b-a
   		e  = d
    	
   	if abs(fc) < abs(fb) :
   		a  = b
   		b  = c
   		c  = a
   		fa = fb
   		fb = fc
   		fc = fa
    	
   	tol1 = 2.*EPS*abs(b)+0.5*tol
   	xm   = .5*(c-b)
   	if abs(xm) <= tol1  or fb == 0. :
   		zbrent = b
   		return zbrent
   	if abs(e) >= tol1  and abs(fa) > abs(fb) :
   		s = fb/fa
   		if a == c :
     		p = 2.*xm*s
     		q = 1.-s
   		else:
        q = fa/fc
     		r = fb/fc
     		p = s*(2.*xm*q*(q-r)-(b-a)*(r-1.))
     		q = (q-1.)*(r-1.)*(s-1.)
      		
     	if p > 0. : q += -1.
      p = abs(p)
      if(2.*p < min(3.*xm*q-abs(tol1*q),abs(e*q))):
      	e = d
      	d = p/q
      else:
      	d = xm
      	e = d
    	
    else:
    	d = xm
    	e = d
    	
  	a  = b
    fa = fb
    if abs(d) > tol1 :
    	b += d
    else:
    	b += math.copysign(tol1,xm)
    	
    fb = func(b)
  
  print 'zbrent exceeding maximum iterations'
  zbrent = b
  
  return zbrent
      
#---------------------------------------------------------------------------------      

#	returned: arr

def sort(n,arr):
  NSTACK = 50
  istack = np.empty([NSTACK])
 	jstack = 0
 	M  = 7
 	l  = 1
 	ir = n
 	while ir-l < M:
   	for j in range(l,ir):
     	a = arr[j]
     	for i in range(j,l,-1):
        if arr[i] < a :
   	      arr[i+1] = a
       	  break
       	arr[i+1] = arr[i]
     	
      i = l-1                       # POTENTIAL SOURCE OF BUG
     	arr(i+1)=a
   	
    if jstack == 0 : return arr
   	ir = istack[jstack]
   	l  = istack[jstack-1]
   	jstack -= 2
 	else:
   	k = (l+ir)/2.
    temp     = arr[k]
    arr[k]   = arr[l+1]
   	arr[l+1] = temp
   	if arr[l] > arr[ir] :
     	temp    = arr[l]
     	arr[l]  = arr[ir]
     	arr[ir] = temp
   	
    if arr[l+1] > arr[ir] :
     	temp     = arr[l+1]
     	arr[l+1] = arr[ir]
     	arr[ir]  = temp
   	
    if arr[l] > arr[l+1] :
     	temp     = arr[l]
     	arr[l]   = arr[l+1]
     	arr[l+1] = temp
   	
    i = l+1
   	j = ir
   	a = arr[l+1]
   	while arr[i] < a and j > i:       #
     	i = i+1                         #
     	while (arr[j] > a):             #
        j = j-1                       #     POTENTIAL SOURCE OF BUGS
   	    temp   = arr[i]               #     (ll. 904-912 fortran)
        arr[i] = arr[j]               #
       	arr[j] = temp                 #
   	arr[l+1] = arr[j]                 #
   	arr[j] = a
   	jstack += 2
   	if jstack > NSTACK :
   		print 'NSTACK too small in sort'
   		break
   	
    if ir-i+1 >= j-l :
   		istack[jstack]   = ir
   		istack[jstack-1] = i
   		ir = j-1
   	else:
   		istack[jstack]   = j-1
   		istack[jstack-1] = l
   		l = i

#---------------------------------------------------------------------------------

#	returned: ra,rb,rc

def sort3(n,ra,rb,rc,wksp,iwksp):
	ra    = indexx(n,ra,iwksp)[0]
	iwksp = indexx(n,ra,iwksp)[1]
	
	for j in range(n)
        wksp[j] = ra[j]
	
	for j in range(n)
        ra[j]   = wksp[iwksp[j]]
	
	for j in range(n)
        wksp[j] = rb[j]
	
	for j in range(n)
        rb(j)   = wksp[iwksp[j]]
	
	for j in range(n)
        wksp[j] = rc[j]
	
	for j in range(n)
        rc[j]   = wksp[iwksp[j]]

	return ra,rb,rc 				  					# CHECK RETURN VALUE

#---------------------------------------------------------------------------------

#	returned: arr,indx

def indexx(n,arr,indx):
  M=7
  NSTACK=50
  istack = np.empty([NSTACK])

  for j in range(1,n):
    indx[j]=j
    jstack=0
    l=1
    ir=n
    while (ir-l < M):
      if (ir-l < M):
        for j in range(l+1,ir):
          indxt=indx[j]
          a=arr[indxt]
          for i in range(j-1,1,-1):                       #
            while (arr[indx[i]] <= a) is False:           # POTENTIAL SOURCE OF BUGS
              indx[i+1]=indx[i]                           # (ll. 978-983 fortran)
          i=0                                             #
          indx(i+1)=indxt                                 #
        if(jstack == 0): return arr,indexx  # CHECK RETURN VALUE
        ir=istack[jstack]
        l=istack[jstack-1]
        jstack-=2
      else:
        k=(l+ir)/2.
        itemp=indx[k]
        indx[k]=indx[l+1]
        indx[l+1]=itemp
        if(arr[indx[l+1]] > arr[indx[ir]]):
          itemp=indx(l+1)
          indx[l+1]=indx[ir]
          indx[ir]=itemp
        if(arr[indx[l]] > arr[indx[ir]]):
          itemp=indx[l]
          indx[l]=indx[ir]
          indx[ir]=itemp
        if(arr[indx[l+1]] > arr[indx[l]]):
          itemp=indx[l+1]
          indx[l+1]=indx[l]
          indx[l]=itemp
        i=l+1
        j=ir
        indxt=indx[l]
        a=arr[indxt]
        while (arr[indx[i]] < a):                   #
          i=i+1                                     #
        while (arr[indx[j]] > a):                   #
          j=j-1                                     #    POTENTIAL SOURCE OF BUGS
        while (j < i):                              #      (ll. 1014-1018 fortran)
          itemp=indx[i]                             #
          indx[i]=indx[j]                           #
          indx[j]=itemp                             #
          goto 3                                    #
        indx[l]=indx[j]                             #
        indx(j)=indxt
        jstack+=2
        if(jstack > NSTACK):
          print 'NSTACK too small in indexx'
          break
        if(ir-i+1 >= j-l):
          istack[jstack]=ir
          istack[jstack-1]=i
          ir=j-1
        else:
          istack[jstack]=j-1
          istack[jstack-1]=l
          l=i

#---------------------------------------------------------------------------------

#	returned: s

def qtrap(func,a,b,s):
  EPS  = 1.e-6
  JMAX = 20
  olds = 0.
  for j in range(JMAX):
   	s = trapzd(func,a,b,s,j)
   	if j > 5 :
   		if abs(s-olds) < EPS*abs(olds) or (s == 0. and olds == 0.) : return s
    	
   	olds=s
  	
  print 'too many steps in qtrap'
  	
#---------------------------------------------------------------------------------

#	returned: s

def trapzd(func,a,b,s,n):
 	if n == 1:
   	s = 0.5*(b-a)*(func(a)+func(b))
 	else:
   	it  = 2**(n-2)
    tnm = it
   	de  = (b-a)/tnm
    x   = a+0.5*de
   	sum  = 0.
   	for j in range(it):
     		sum += func(x)
         	x += de
    	
   	s = 0.5*(s+(b-a)*sum/tnm)
  
 	return s

#---------------------------------------------------------------------------------

#	returned: erf (error function for x)

def erf(x):
  if x < 0. :
   	erf = -gammp(.5e0,x**2)
  else:
   	erf =  gammp(.500,x**2)
  
  return erf

#---------------------------------------------------------------------------------

#     Linear equation solution by Gauss-Jordan elimination, equation 
#     (2.1.1) above. a(1:n,1:n) is an input matrix stored in an array 
#     of physical dimensions np by np. b(1:n,1:m) is an input matrix 
#     containing the m right-hand side vectors, stored in an array of 
#     physical dimensions np by mp. On output, a(1:n,1:n) is replaced 
#     by its matrix inverse, and b(1:n,1:m) is replaced by the 
#     corresponding set of solution vectors.
#     Parameter: NMAX is the largest anticipated value of n.

#	returned: a, b

def gaussj(a,n,np,b,m,mp):
 	NMAX=50
 	indxc = np.empty([NMAX])
	indxr = np.empty([NMAX])
 	ipiv  = np.empty([NMAX])
 	for j in range(n):
   	ipiv[j] = 0
  	
 	for i in range(n):
   	big = 0.
   	for j in range(n):
     		if ipiv[j] != 1 :
     			for k in range(n):
       			if ipiv[k] == 0 :
       				if abs(a[j,k]) >= big :
           				big  = abs(a[j,k])
           				irow = j
           				icol = k
  		
 		ipiv[icol] += 1

#     We now have the pivot element, so we interchange rows, if needed, 
#     to put the pivot element on the diagonal. The columns are not 
#     physically interchanged, only relabeled:
#     indxc(i), the column of the ith pivot element, is the ith column 
#     that is reduced, while indxr(i) is the row in which that pivot 
#     element was originally located. If indxr(i) =indxc(i) there is an 
#     implied column interchange. With this form of bookkeeping, the
#     solution b's will end up in the correct order, and the inverse 
#     matrix will be scrambled by columns.

 		if irow != icol :
   		for l in range(n):
     			dum = a[irow,l]
     			a[irow,l] = a[icol,l]
     			a[icol,l] = dum
    
   		for lin range(m):
     			dum = b[irow,l]
     			b[irow,l] = b[icol,l]
     			b[icol,l] = dum
  
 		indxr[i]=irow
 		indxc[i]=icol
 		if a[icol,icol] == 0. :
   		print 'singular matrix in gaussj'
   		break
  
 		pivinv = 1./a[icol,icol]
 		a[icol,icol]=1.
 		for l in range(n):
   		a[icol,l] *= pivinv
  		
 		for l in range(m):
   		b[icol,l] *= pivinv
  
 		for ll in range(n):           # Next, we reduce the rows...
   		if ll != icol :               #...except for the pivot one, of course.
     			dum = a[ll,icol]
     			a[ll,icol] = 0.
     			for l in range(n):
       			a[ll,l] -= a[icol,l]*dum 
     			
     			for l in range(m):
       			b(ll,l) -= b(icol,l)*dum  
  
 	for l in range(n,0,-1):
   	if indxr[l] != indxc[l] :
     		for k in range(1,n):
       		dum = a[k,indxr[l]]
       		a[k,indxr[l]] = a[k,indxc[l]]
       		a[k,indxc[l]] = dum
  
 	return a,b                          # And we are done.

#---------------------------------------------------------------------------------

#	returned: xi

def rebin(rc,nd,r,xin,xi):
 	k  = 0
 	xo = 0.
 	dr = 0.
 	for i in range(nd-1):
   	while (rc > dr):
     		k += 1
      	dr += r[k]
    	
   	if k > 1 : xo = xi[k-1]
   	xn  = xi[k]
   	dr -= rc
   	xin[i] = xn-(xn-xo)*dr/r[k]
  	
 	for i in range(nd-1):
   	xi[i] = xin[i]
  	
 	xi[nd]=1.
  
 	return xi

#---------------------------------------------------------------------------------

#	returned: factorial of n

def fact(n):
	if n <= 0 :
   	fact = 1
   	print  'n <=0 and fact = 1'
   	return fact
 	else:
   	fact = 1         
  	
 	for i in range(1,n+1):
   	fact *= i
    
   return fact

#---------------------------------------------------------------------------------

#	returned: prob

def ksone(data,n,func,d,prob):
 	data = sort(n,data)
 	en = n
 	d  = 0.
 	fo = 0.
 	for j in range(n):
   	fn = j/en
   	ff = func(data[j])
   	dt = max(abs(fo-ff),abs(fn-ff))
   	if dt > d : d = dt
   	fo = fn
  	
 	en = math.sqrt(en)
 	prob = probks((en+0.12+0.11/en)*d)  
   
 	return  prob

#---------------------------------------------------------------------------------

#	returned: prob

def kstwo(data1,n1,data2,n2,d,prob):
 	data1 = sort(n1,data1)
 	data2 = sort(n2,data2)
 	en1 = n1
 	en2 = n2
 	j1  = 0
 	j2  = 0
 	fn1 = 0.
 	fn2 = 0.
 	d = 0.
 	while j1 <= n1 and j2 <= n2 :
   	d1 = data1[j1]
   	d2 = data2[j2]
   	if d1 <= d2 :
   	  	fn1 = j1/en1
     		j1 += 1  
   	
   	if d2 <= d1 :
     		fn2 = j2/en2
     		j2 += 1  
    	
   	dt = abs(fn2-fn1)  
   	if dt > d : d = dt    
 	
 	en = math.sqrt(en1*en2/(en1+en2))
 	prob = probks((en+0.12+0.11/en)*d)
  	
 	return  prob

#---------------------------------------------------------------------------------

#	returned: probks

def probks(alam):
	EPS1 = 0.001
 	EPS2 = 1.e-8
 	a2   = -2.*alam**2
 	fac  = 2.
 	probks = 0.
 	termbf = 0.
 	for j in range(1,100):
   	term = fac*math.exp(a2*j**2)
   	probks += term
   	if abs(term) <= EPS1*termbf or abs(term) <= EPS2*probks : return probks
   	fac *= -1
   	termbf = abs(term)  
  	
 	probks=1.
  
 	return probks
      
