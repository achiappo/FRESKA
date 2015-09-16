from scipy import special
import numpy as np
from os import mkdir
from AT import get_data
from matplotlib import pylab as plt
from scipy.integrate import quad,quadrature,nquad
from math import sqrt,cos, log, pi
from scipy import optimize as sciopt
from scipy.interpolate import UnivariateSpline as spline

names = {'dra':"Draco",'seg1':"Segue 1",'umaI':"Ursa Major 1",'booI':"Bootes 1",
         'wil1':"Willman 1",'scl':"Sculptor",'for':"Fornax",'sgr':"Sagittarius",
         'com':"Coma Berenices"}

###################################################################################################
#										FUNCTIONS DEFINITIONS
###################################################################################################
# stellar density profile

def nu(r,rh):
    return (1 + (r/rh)**2)**(-5./2.)

# dwarf surface brightness profile

def I(R,rh):
    return 4./3. * rh/(1+(R/rh)**2)**2

##########################################################################
# Mass of NFW DMH

def get_M_NFW(x):
    #this is the analytic formula
    #the constant is 4pi*rho0*rs^3
    return np.log(1.+x)-x/(1.+x)

##########################################################################
# numerical integrals in sigma_los

def integrand1(y,alpha,beta):
    result = nu(y,1)*get_M_NFW(y*alpha)/y**(2.-2.*beta)
    return result

def integral1(ymin,alpha,beta):
    res,err = quad(integrand1,ymin,+np.inf,args=(alpha,beta))
    return res

def integrand2(z,alpha,beta,gamma):
    result = (1.-beta/z**2) * z**(1.-2.*beta)/np.sqrt(z*z-1.)
    res = integral1(gamma*z,alpha,beta)
    return result * res

def integral2(alpha,beta,gamma):
    res, err = quad(integrand2,1.,+np.inf,args=(alpha,beta,gamma) )
    return res, err

##########################################################################
# jfactor evaluation functions

def func(u,y, D, rt, ymin):
    return (1.+u)**(-4)/u/sqrt(u*u-D**2*(1-y*y))

def lim_u(y, D, rt, ymin):
    return [D*sqrt(1-y*y), rt]

def lim_y(D, rt, ymin):
    return [ymin,1]

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
    cst = 4*pi*rho0**2*r0*Msun2kpc5_GeVcm5
    res = nquad(func, ranges=[lim_u, lim_y], args=(Dprime,rtprime,ymin),
    	opts=[{'epsabs':1.e-10,'epsrel':1.e-10,'limit':1000},
    	{'epsabs':1.e-10,'epsrel':1.e-10,'limit':1000}])
    return cst*res[0]

###################################################################################################
#										BODY OF THE CODE
###################################################################################################

# extract the dwarf parameter from file
dwarf = 'umaI'
#mkdir('/home/andrea/Desktop/work/DWARF/Jvalue/output/%s'%dwarf)
R,v,dv,rh,rt,nstars,D = get_data(dwarf)
u=v.mean()

beta=-0.005

gamma_array = R/rh
A_array = gamma_array**(1.-2.*beta)/I(R,rh)
r0_array = np.logspace(np.log10(0.1),np.log10(5.),100)
alpha_array = rh/r0_array
I_array=np.zeros(shape=(len(A_array),len(r0_array)))

for i,gamma in enumerate(gamma_array):
    for j,alpha in enumerate(alpha_array):
        res,err = integral2(alpha,beta,gamma)
        I_array[i,j] = res *A_array[i]

cst = 8.*np.pi*4.3e-6 #this is 1.08e-4, which means that the natural numerical unit for M0 is ~10^4 M_solar

# Likelihood definition (only for fixed beta!)
def logLike(M0,j):
    I = cst*M0*I_array[:,j]
    S = dv**2+I
    res = (np.log(S) + (v-u)**2/S).sum()
    return res/2.

rho0_array = np.logspace(6.,9.,100)
Jgrid = np.empty([len(rho0_array),len(r0_array)])
Lgrid = np.empty([len(rho0_array),len(r0_array)])
for i,rho0 in enumerate(rho0_array):
    for j,r0 in enumerate(r0_array):						# EVALUATION OF THE GRID OF THE LIKELIHOOD  
        Jgrid[i,j] = Jfactor(D,np.inf,r0,rho0,0.5)			# AND THE J-FACTOR
        Lgrid[i,j] = logLike(rho0*r0**3,j)
#np.save('output/%s/Jgrid_%s'%(dwarf,dwarf),Jgrid)
#np.save('output/%s/Lgrid_%s'%(dwarf,dwarf),Lgrid)

log10Jrho1 = np.log10([Jfactor(D,np.inf,r0,1.,0.5) for r0 in r0_array])
def deltaJ(log10rho0,J,j):
    return abs(J-log10Jrho1[j]-2.*log10rho0)

rho0star = np.empty([4])
r0star = np.empty([4])
for i,J in enumerate((15,16,18,20,21)):
    min_rho0J = sciopt.minimize_scalar(deltaJ,args=(J,59+i*2),tol=1.e-10)
    rho0star[i] = min_rho0J.x
    r0star[i] = r0_array[59+i*2]

lplot = plt.pcolormesh(r0_array,rho0_array,Lgrid,shading='gouraud')
ctJ = plt.contour(r0_array,rho0_array,np.log10(Jgrid),colors='k')
ctL = plt.contour(r0_array,rho0_array,Lgrid,linestyles='dashed',colors='m')
plt.scatter(r0star,np.power(10,rho0star),marker='*',c='w',s=100,edgecolor='w')
plt.clabel(ctJ,inline=1,fmt='%1.0f',colors='k')
plt.clabel(ctL,inline=1,fmt='%1.0f',colors='m')
cx = plt.colorbar(lplot,pad=0)
cx.set_label(r'-$log$Like$(\rho_0,r_0|\vec v,\vec \sigma_v)$',fontsize=14)
plt.semilogy()
plt.ylabel(r'$\rho_0 [M_\odot$ kpc$^{-3}$]',fontsize=14)
plt.xlabel(r'$r_0$ [kpc]',fontsize=14)
plt.suptitle('%s'%names[dwarf],fontsize=16)
plt.show()
