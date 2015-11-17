import yaml
import numpy as np
from sys import argv
from scipy import special
from scipy.integrate import quad
from scipy import optimize as sciopt
from scipy.interpolate import UnivariateSpline as spline
from matplotlib import pyplot as plt
from functions_sim import integral2, Jfactor, nu
from multiprocessing import Pool

##########################################################################################################################################################
# inverse hyperbolic cosecant (used for gamma* = 1 , non-Plum)
def inv_csch(x):
    return np.log(np.sqrt(1+x**-2.)+x**-1.)
# integrand of I(R) (used for gamma* = 0.1 , Plum)
def integrand_I(r,rh,R):
    return nu(r,rh)*r/np.sqrt(r**2-R**2)

# dwarf surface brightness profile
def I(R,rh):
    return rh**2*((2*rh**2+R**2)*inv_csch(R/rh)-rh*np.sqrt(rh**2+R**2))/(rh**2+R**2)**(3/2.) # for gamma*=1 , non-Plum
    #return 2*quad(integrand_I,R,+np.inf,args=(rh,R))[0] # for gamma*=0.1 , Plum

##########################################################################################################################################################
# data extraction and preparation
x,y,z,vx,vy,vz = np.loadtxt('gs100_bs050_rcrs025_rarcinf_cusp_0064mpc3_df_1000_0.dat',unpack=True)
R = np.sqrt(x**2+y**2) # assumed direction of observation along z-axis for simplicity (as suggested on the Gaia wiki)
D=50.
d = np.sqrt(x**2+y**2+(D-z)**2)
v = (x*vx+y*vy+(D-z)*vz)/d
# Errors (from mock data) preparation
Evx,Evy,Evz = np.loadtxt('gs100_bs050_rcrs025_rarcinf_cusp_0064mpc3_df_1000_0_err.dat',unpack=True,usecols=(3,4,5))
Ex,Ey,Ez = np.absolute(Evx-vx),np.absolute(Evy-vy),np.absolute(Evz-vz)
dv = (x*Ex+y*Ey+(D-z)*Ez)/d

##########################################################################################################################################################
# computation of intrinsic velocity dispersion (from Jeans equation)

u=0.
rh=0.25
nstars=np.size(R)

A_array = np.array([Ri/I(Ri,rh) for Ri in R])
r0_array = np.logspace(-2.,2.,100)
I_array=np.zeros(shape=(len(A_array),len(r0_array)))

def array_builder(r0_array, R_array):
    for i,R in enumerate(R_array):
        for j,r0 in enumerate(r0_array):
            yield (i, j), (r0, rh, R)

def proxy(args):
    return args[0], integral2(*args[1])*A_array[args[0][0]]

pool = Pool(processes=4)
results = pool.map(proxy, array_builder(r0_array, R))
pool.close()
pool.join()
for idx,value in results:
    I_array[idx] = value

cst = 8.*np.pi*4.3e-6
# Likelihood definition (only for fixed beta!)
def logLike(M0,j):
    I = cst*M0*I_array[:,j]
    S = dv**2.+I
    res = (np.log(S) + (v-u)**2./S).sum()
    return res/2.

# |J-J(rho0,r0)| for J inversion (in log scale)
log10Jrho1 = np.log10([Jfactor(D,np.inf,r0,1.,0.5) for r0 in r0_array])
def deltaJ(log10rho0,log10J,j):
    return abs(log10J-log10Jrho1[j]-2*log10rho0)

##########################################################################################################################################################
# fitting scheme
J_array = np.linspace(15.,25.,100)
LikeJ = np.zeros_like(r0_array)
J_new = np.empty([0])
min_LikeJ = np.empty([0])
for i,J in enumerate(J_array):
    for j,r0 in enumerate(r0_array):
        log10rho0 = sciopt.minimize_scalar(deltaJ,args=(J,j)).x
        LikeJ[j] = logLike(10**log10rho0*r0**3,j)
    spline_LikeJ = spline(r0_array,LikeJ,s=0)
    min_r0R = sciopt.minimize_scalar(spline_LikeJ,method='Bounded',bounds=(r0_array[0],r0_array[np.size(r0_array)/2]))
    min_r0L = sciopt.minimize_scalar(spline_LikeJ,method='Bounded',bounds=(r0_array[np.size(r0_array)/2],r0_array[-1]))
    if min_r0R.fun>min_r0L.fun:
        if r0_array[1]<min_r0L.x<r0_array[-2]:
            J_new = np.append(J_new,J)
            min_LikeJ = np.append(min_LikeJ,min_r0L.fun)
    else:
        if r0_array[1]<min_r0R.x<r0_array[-2]:
            J_new = np.append(J_new,J)
            min_LikeJ = np.append(min_LikeJ,min_r0R.fun)

##########################################################################################################################################################
# minimum and C.I. determination
spline_Like = spline(J_new,min_LikeJ,s=0)
min_J = sciopt.minimize_scalar(spline_Like,method='Bounded',bounds=(J_new[0],J_new[-1])).x

def one_sigmaJ(J):
    return np.abs(spline_Like(J)-spline_Like(min_J)-0.5)

one_sigma_l = sciopt.minimize_scalar(one_sigmaJ,method='Bounded',bounds=(J_new[0],min_J)).x-min_J
one_sigma_r = sciopt.minimize_scalar(one_sigmaJ,method='Bounded',bounds=(min_J,J_new[-1])).x-min_J

def two_sigmaJ(J):
    return np.abs(spline_Like(J)-spline_Like(min_J)-2.)

two_sigma_l = sciopt.minimize_scalar(two_sigmaJ,method='Bounded',bounds=(J_new[0],min_J)).x-min_J
two_sigma_r = sciopt.minimize_scalar(two_sigmaJ,method='Bounded',bounds=(min_J,J_new[-1])).x-min_J

def three_sigmaJ(J):
    return np.abs(spline_Like(J)-spline_Like(min_J)-4.)

three_sigma_l = sciopt.minimize_scalar(three_sigmaJ,method='Bounded',bounds=(J_new[0],min_J)).x-min_J
three_sigma_r = sciopt.minimize_scalar(three_sigmaJ,method='Bounded',bounds=(min_J,J_new[-1])).x-min_J

trueJ = round(np.log10(Jfactor(D,np.inf,1.,6.4e7,0.5)),2)

J_plt = np.linspace(min_J-1,min_J+1,100)
plt.plot(J_plt,spline_Like(J_plt)-spline_Like(min_J))
plt.plot(min_J,0,'b^',markersize=12,label='J$_{MLE}$ = %.2f'%min_J)
plt.plot(trueJ,0,'r*',markersize=12,label='J$_{TRUE}$ = %.2f'%trueJ)
plt.hlines(0.,min(J_plt),max(J_plt),linestyles='dashed')
plt.hlines(.5,min(J_plt),max(J_plt),colors=('r'),linestyles='dashed',
           label=r'$1-\sigma$'+'\t'+'[%.2f,%+.2f]'%(one_sigma_l,one_sigma_r))
plt.hlines(2,min(J_plt),max(J_plt),colors=('g'),linestyles='dashed',
           label=r'$2-\sigma$'+'\t'+'[%.2f,%+.2f]'%(two_sigma_l,two_sigma_r))
plt.hlines(4,min(J_plt),max(J_plt),colors=('c'),linestyles='dashed',
           label=r'$3-\sigma$'+'\t'+'[%.2f,%+.2f]'%(three_sigma_l,three_sigma_r))
plt.legend(numpoints=1,loc='upper right',fontsize=14)
plt.ylabel(r'$\mathcal{L}$(J)',fontsize=14)
plt.xlabel(r'$log_{10}$  J [GeV$^2$ cm$^{-5}$]',fontsize=14)
plt.ylim(-0.5,8)
plt.xlim(min_J-1,min_J+1)
casedir = 'Isotrop+Cusp+nonPlum'
plt.suptitle(r'Isotropic-Cusped-non Plummer (N$_\star$ = %i)'%nstars,fontsize=16)
plt.savefig('Sim%i_%s.png'%(nstars,casedir),dpi=300,format='png')
