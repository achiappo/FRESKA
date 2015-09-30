import yaml
import numpy as np
from sys import argv
from os import mkdir
from AT import get_data
from matplotlib import pyplot as plt
from scipy.integrate import quad,quadrature,nquad
from math import sqrt,cos, log, pi
from scipy import optimize as sciopt
from scipy.interpolate import UnivariateSpline as spline

names = {'booI':"Bootes I",'booII':"Bootes II",'car':"Carina",'com':"Coma Berenices",
'cvnI':"Canes Venatici I",'cvnII':"Canes Venatici II",'dra':"Draco",'for':"Fornax",
'her':"Hercules",'leoI':"Leo I",'leoIV':"Leo IV",'leoT':"Leo T",'scl':"Sculptor",
'seg1':"Segue 1",'sex':"Sextans",'sgr':"Sagittarius",'umaI':"Ursa Major I",
'umaII':"Ursa Major II",'umi':"Ursa Minor",'wil1':"Willman 1",}

###################################################################################################
#									          	FUNCTIONS DEFINITIONS
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
#										                    BODY OF THE CODE
###################################################################################################

# extract the dwarf parameter from file
dwarf = argv[1]
#mkdir('/home/andrea/Desktop/work/DWARF/Jvalue/output/new/%s'%dwarf)
R,v,dv,rh,rt,nstars,D = get_data(dwarf)
u = v.mean()

gamma_array = R/rh
r0_array 	= np.logspace(np.log10(0.1),np.log10(5.),50)
rho0_array = np.linspace(6.,9.,50)
alpha_array = rh/r0_array
beta_array 	= np.linspace(-1.,1.,50)

# vectorisation array for r0-rho0-beta profiling (very long calculation)
sigma_array = np.array([[[integral2(alpha,beta,gamma)*gamma**(1.-2.*beta)/
	I(gamma*rh,rh) for alpha in alpha_array] 
	for gamma in gamma_array] for beta in beta_array],dtype=float)
np.save('output/new/%s/sigma_beta_array_%s'%(dwarf,dwarf),sigma_array)

#sigma_array = np.load('output/new/%s/sigma_beta_array_%s.npy'%(dwarf,dwarf))

# Likelihood definition (only for fixed beta!)
def LikeBeta(log10rho0,i,b):
	cst = 8.*np.pi*4.3e-6
	M0 = 10.**log10rho0*r0_array[i]**3
	I = cst*M0*sigma_array[b,:,i]
	S = dv**2+I
	res = (np.log(S) + (v-u)**2/S).sum()
	return res/2.

log10Jrho1 = np.log10([Jfactor(D,np.inf,r0,1.,0.5) for r0 in r0_array])
def deltaJ(log10rho0,J,j):
    return abs(J-log10Jrho1[j]-2.*log10rho0)

LikeJ = np.empty([0])
J_new = np.empty([0])
J_array = np.linspace(15.,21.,100)
LikeBetak = np.zeros_like(beta_array)
for c,J in enumerate(J_array):
    LikeJbeta = np.empty([0])
    r0_new = np.empty([0])
    for j,r0 in enumerate(r0_array):
        log10rho0 = sciopt.minimize_scalar(deltaJ,args=(J,j),tol=1e-8).x
        for b,beta in enumerate(beta_array):
            LikeBetak[b] = LikeBeta(log10rho0,j,b)
        spline_LikeBeta = spline(beta_array,LikeBetak,s=0)
        min_beta = sciopt.minimize_scalar(spline_LikeBeta,method='Bounded',
        	bounds=(-.99,.99),tol=1.e-10)
        neg_spline_beta = lambda J : -spline_LikeBeta(J)
        if beta_array[0]+0.1<min_beta.x<beta_array[-1]-0.1:
            minb = min_beta.x
            r0_new = np.append(r0_new,r0)
            LikeJbeta = np.append(LikeJbeta,spline_LikeBeta(minb))
        else:
            maxb = sciopt.minimize_scalar(neg_spline_beta,method='Bounded',
            	bounds=(-.99,.99),tol=1.e-10).x
            if beta_array[0]+0.1<maxb<beta_array[-1]-0.1:
                r0_new = np.append(r0_new,r0)
                LikeJbeta = np.append(LikeJbeta,spline_LikeBeta(maxb))
    if len(r0_new)>3:
    	spline_Like_r0 = spline(r0_new,LikeJbeta,s=0)
    	min_LikeJ_r0 = sciopt.minimize_scalar(spline_Like_r0,method='Bounded',
    		bounds=(r0_new[0],r0_new[-1]),tol=1.e-10).x
    	neg_spline_r0 = lambda J : -spline_Like_r0(J)
    	if r0_new[0]+0.2<min_LikeJ_r0<r0_new[-1]-0.2:
    		J_new = np.append(J_new,J)
    		LikeJ = np.append(LikeJ,spline_Like_r0(min_LikeJ_r0))
    	else:
    		min_LikeJ_neg = sciopt.minimize_scalar(neg_spline_r0,method='Bounded',
    			bounds=(r0_new[0],r0_new[-1]),tol=1.e-10).x
    		if r0_new[0]+0.2<min_LikeJ_neg<r0_new[-1]-0.2:
    			J_new = np.append(J_new,J)
    			LikeJ = np.append(LikeJ,spline_Like_r0(min_LikeJ_neg))

spline_Like = spline(J_new,LikeJ)
min_J = sciopt.minimize_scalar(spline_Like,method='Bounded',
	bounds=(J_new[0],J_new[-1]),tol=1.e-10).x

def one_sigmaJ(J):
    return abs(spline_Like(J)-spline_Like(min_J)-0.5)
err_l = sciopt.minimize_scalar(one_sigmaJ,method='Bounded',
	bounds=(J_new[0],min_J),tol=1.e-10).x-min_J
err_r = sciopt.minimize_scalar(one_sigmaJ,method='Bounded',
	bounds=(min_J,J_new[-1]),tol=1.e-10).x-min_J

results={'dwarf':dwarf,'Nstars':nstars,'Jmini':min_J,
'Jplus':err_r,'Jminus':err_l,'LikeJmin':float(spline_Like(min_J))}
yaml.dump(results,open('output/new/%s/results_%s.yaml'%(dwarf,dwarf),'wb'))
np.save('output/new/%s/Like_%s'%(dwarf,dwarf),np.vstack((J_new,LikeJ)))

J_plt = np.linspace(min_J-1,min_J+1,100)
plt.plot(J_plt,spline_Like(J_plt)-spline_Like(min_J))
plt.plot(min_J,0.,'^',label='J = %.2f +%.2f %.2f'%(min_J,err_r,err_l))
plt.hlines(0.,min(J_plt),max(J_plt),linestyles='dashed')
plt.hlines(.5,min(J_plt),max(J_plt),colors=('r'),linestyles='dashed',
	label=r'$1-\sigma$'+'\t'+r'$N_{stars}$ = %i'%nstars)
plt.legend(numpoints=1,loc='upper center',fontsize=12)
plt.ylabel(r'$-\Delta log$Like(J)',fontsize=14)
plt.xlabel(r'$log_{10}$ J [GeV$^2$ cm$^{-5}$]',fontsize=14)
plt.ylim(-1,10)
plt.xlim(min_J-1,min_J+1)
plt.suptitle('%s'%names[dwarf]+'\t'+r'(fit for $\beta$)',fontsize=16)
plt.savefig('output/new/%s/LikeJbeta_%s.png'%(dwarf,dwarf),dpi=300,format='png')
plt.show()
