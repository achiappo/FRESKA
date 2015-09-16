import numpy as np
from os import mkdir
from AT import get_data
from matplotlib import pylab as plt
from scipy.integrate import quad,quadrature,nquad
from math import sqrt,cos, log, pi
from scipy import optimize as sciopt
from scipy.interpolate import UnivariateSpline as spline

'''
import logging
LOG = logging.getLogger("")

interactive = True # THIS IS A GLOBAL VARIABLE, HANDLE WITH CARE!
try:             os.environ['DISPLAY']
except KeyError, KE:
    LOG.error(KE)
    matplotlib.use('Agg')
    interactive = False
'''

names = {'dra':"Draco",'seg1':"Segue 1",'umaI':"Ursa Major 1",'booI':"Bootes 1",
         'wil1':"Willman 1",'scl':"Sculptor",'for':"Fornax",'sgr':"Sagittarius",
         'com':"Coma Berenices"}

###################################################################################################
#										FUNCTIONS DEFINITIONS
###################################################################################################
# stellar density profile

def nu(r,rh):
    return (1. + (r/rh)**2.)**(-5./2.)

# dwarf surface brightness profile

def I(R,rh):
    return 4./3. * rh/(1.+(R/rh)**2.)**2.

##########################################################################
# Mass of NFW DMH

def get_M_NFW(x):
    #this is the analytic formula
    #the constant is 4pi*rho0*rs^3
    return np.log(1.+x)-x/(1.+x)

##########################################################################
# numerical integrals in sigma_los

def integrand1(y,alpha,beta):
    result = nu(y,1.)*get_M_NFW(y*alpha)/y**(2.-2.*beta)
    return result

def integral1(ymin,alpha,beta):
    res,err = quad(integrand1,ymin,+np.inf,args=(alpha,beta),
    	epsabs=1.e-10,epsrel=1.e-10,limit=1000)
    return res

def integrand2(z,alpha,beta,gamma):
    result = (1.-beta/z**2.) * z**(1.-2.*beta)/np.sqrt(z*z-1.)
    res = integral1(gamma*z,alpha,beta)
    return result * res

def integral2(alpha,beta,gamma):
    res, err = quad(integrand2,1.,+np.inf,args=(alpha,beta,gamma),
    	epsabs=1.e-10,epsrel=1.e-10,limit=1000)
    return res

##########################################################################
# jfactor evaluation functions

def func(u,y, D, rt, ymin):
    return (1.+u)**(-4.)/u/sqrt(u*u-D**2.*(1.-y*y))

def lim_u(y, D, rt, ymin):
    return [D*sqrt(1.-y*y), rt]

def lim_y(D, rt, ymin):
    return [ymin,1.]

def Jfactor(D,rt,r0,rho0,tmax):
    """
    returns the Jfactor computed in the solid angle of
    semi apex angle tmax, in degree, for a NFW halo profile of 
    shape parameters (r0,rho0) at distance D. 
    rt is the maximal radius of integration 
    D, r0 and rt are in kpc, and rho0 is in Msun.kpc^-3
    """
    ymin=cos(np.radians(tmax))
    Dprime=D/r0
    rtprime=rt/r0
    Msun2kpc5_GeVcm5 = 4463954.894661358
    cst = 4.*pi*rho0**2.*r0*Msun2kpc5_GeVcm5
    res = nquad(func, ranges=[lim_u, lim_y], args=(Dprime,rtprime,ymin),
    	opts=[{'epsabs':1.e-10,'epsrel':1.e-10,'limit':1000},
    	{'epsabs':1.e-10,'epsrel':1.e-10,'limit':1000}])
    return cst*res[0]

###################################################################################################
#										BODY OF THE CODE
###################################################################################################

# extract the dwarf parameter from file
dwarf = 'dra'
#mkdir('/home/andrea/Desktop/work/DWARF/Jvalue/output/%s'%dwarf)
R,v,dv,rh,rt,nstars,D = get_data(dwarf)
u = v.mean()

gamma_array = R/rh
r0_array 	= np.logspace(np.log10(0.1),np.log10(5.),100)
alpha_array = rh/r0_array
beta_array 	= np.linspace(-2.,2.,100)

sigma_array = np.array([[[integral2(alpha,beta,gamma)*gamma**(1.-2.*beta)/
	I(gamma*rh,rh) for alpha in alpha_array] for gamma in gamma_array] for beta in beta_array],dtype=float)

np.save('output/%s/sigma_array_%s'%(dwarf,dwarf),sigma_array)
#sigma_array = np.load('output/%s/sigma_beta_%s.npy'%(dwarf,dwarf))

# Likelihood definition (only for fixed beta!)
def LLBeta(log10rho0,i,b):
	cst = 8.*np.pi*4.3e-6
	M0 = 10.**log10rho0*r0_array[i]**3
	I = cst*M0*sigma_array[b,:,i]
	S = dv**2+I
	res = (np.log(S) + (v-u)**2/S).sum()
	return res/2.

log10Jrho1 = np.log10([Jfactor(D,np.inf,r0,1.,0.5) for r0 in r0_array])
def deltaJ(log10rho0,J,j):
    return abs(J-log10Jrho1[j]-2.*log10rho0)

J_array = np.linspace(15.,21.,100)
LikeJ = np.empty([0])
J_new = np.empty([0])
LikeBeta = np.zeros_like(beta_array)
LikeJ = np.zeros_like(r0_array)
for j,J in enumerate(J_array):
    for i,r0 in enumerate(r0_array):
        log10rho0 = sciopt.minimize_scalar(deltaJ,args=(J,i),tol=1.e-10).x
        for b,beta in enumerate(beta_array):
            LikeBeta[b] = LLBeta(log10rho0,i,b)
        spline_LikeBeta = spline(beta_array,LikeBeta,s=0)
        min_beta = sciopt.minimize_scalar(spline_LikeBeta,method='Bounded',
        	bounds=(beta_array[0]+0.01,beta_array[-1]-0.01),tol=1.e-10).x
        neg_spline_beta = lambda J : -spline_LikeBeta(J)
        if min_beta<beta_array[0]+0.1:
            LikeJ[j] = spline_LikeBeta(sciopt.minimize_scalar(neg_spline_beta,method='Bounded',
            	bounds=(beta_array[0]+0.01,beta_array[-1]-0.01),tol=1.e-10).x)
        else:
            LikeJ[j] = spline_LikeBeta(min_beta)
    r0_new = np.delete(r0_array,np.where(np.isnan(LikeJ))[0])
    LikeJnew = np.delete(LikeJ,np.where(np.isnan(LikeJ))[0])
    #for i in range(len(r0_new)):
    #    print '%10.4f %10.4f %10.4f'%(J,r0_new[i],LikeJnew[i])
    #print '\n'
    plt.plot(r0_new,LikeJnew,'-')
    if len(r0_new) is not 0:
        spline_LikeJ = spline(r0_new,LikeJnew,s=0)
        min_LikeJ = sciopt.minimize_scalar(spline_LikeJ,method='Bounded',
        	bounds=(r0_new[0]+0.01,r0_new[-1]-0.01),tol=1.e-10).x
        neg_splineJ = lambda J : -spline_LikeJ(J)
        if min_LikeJ>r0_new[-1]-0.1:
            min_LikeJ_neg = sciopt.minimize_scalar(neg_splineJ,method='Bounded',
            	bounds=(r0_new[0]+.01,r0_new[-1]-.01),tol=1.e-10).x
            if min_LikeJ_neg>r0_new[0]+0.1:
                J_new = np.append(J_new,J)
                LikeJ = np.append(LikeJ,spline_LikeJ(min_LikeJ_neg))
                #print J,spline_LikeJ(min_LikeJ_neg)
        else:
            J_new = np.append(J_new,J)
            LikeJ = np.append(LikeJ,spline_LikeJ(min_LikeJ))
            #print J,spline_LikeJ(min_LikeJ)


def one_sigmaJ(J):
    return np.abs(spline_Like(J)-spline_Like(min_J.x)-0.5)

plt.plot(J_new,LikeJ,'-')
plt.show()
