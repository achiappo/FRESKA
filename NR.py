import numpy as np

def dfridr(func,x,h,err):
  CON  = 1.4
  CON2 = CON*CON
  BIG  = 1.e30
  NTAB = 10
  SAFE = 2.
  if(h == 0.):
    print 'h must be nonzero in dfridr'
    pass
  hh = h
  a  = np.empty([NTAB,NTAB])
  a[1,1] = (func(x+hh)-func(x-hh))/(2.0*hh)
  err=BIG
  for i=2 in range(NTAB):
    x = hh/CON
    a[1,i] = (func(x+hh)-func(x-hh))/(2.0*hh)
    fac = CON2
    for j=2 in range(i):
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
  maxsteps =1.e8
  Nstep = 0
  x  = a
  dx = dxinit
  y  = 0.e0
  while (x-b)*(b-a) < 0.e0 and Nstep < maxsteps:
    Nstep = Nstep + 1
    dydx = FUNC(x,fp,np)
    yscale = max(abs(y) + abs(dx*dydx), 1.e-12)
    if ((x+dx-b)*(x+dx-a) > 0.e0):  # If stepsize overshoots, decrease it.
      dx = b - x
    RUNGE5VAR(y,dydx,x,dx,eps,yscale,dxnext,FUNC,fp,np)
    dx = dxnext
  if (Nstep >= maxsteps):
    print 'WARNING: failed to converge in INTEGRATE.'
  INTEGRATE = y
  return
  
###################################################################################################################
#
#   Fifth-order Runge-Kutta step with monitoring of local truncation error to ensure accuracy and adjust stepsize.
#   Input are the dependentvariable y and its derivative dydx at the starting value of the independent variable x.
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
  ytemp = 0.
  yerr  = 0.
  h = htry                                      # Set stepsize to initial accuracy.
  errmax = 10.e0
  while errmax > 1.e0 :
    RUNGE(y,dydx,x,h,ytemp,yerr,DERIVS,fp,np)
    errmax = abs(yerr/yscale)/eps               # Scale relative to required accuracy.
    if (errmax > 1.e0):                         # Truncation error too large; reduce h
      htemp = safety*h*(errmax**pshrink)
      hold = h
      h = sign(max(abs(htemp),0.1e0*abs(h)),h)  # No more than factor of 10
      xnew = x + h
      if (xnew == x):
        print 'WARNING: ','Stepsize underflow in RUNGE5VAR().'
        h = hold
        errmax = 0.e0

#     Step succeeded.  Compute estimated size of next step.
  if (errmax > errcon):
    hnext = safety*h*(errmax**pgrow)
  else:
    hnext = 5.e0 * h                            # No more than factor of 5 increase.
    
  x = x + h
  y = ytemp
  return
  



