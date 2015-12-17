import yaml
import numpy as np
from sys import argv
from scipy import special
from scipy.integrate import quad
from scipy import optimize as sciopt
from scipy.interpolate import UnivariateSpline as spline
from scipy.interpolate import interp1d as interp
from functions_sim import integral2, Jfactor, nu
from multiprocessing import Pool

##########################################################################################################################################################
# inverse hyperbolic cosecant
def inv_csch(x):
    return np.log(np.sqrt(1+x**-2.)+x**-1.)
# dwarf surface brightness profile
def I(R,rh):
    return rh**2*((2*rh**2+R**2)*inv_csch(R/rh)-rh*np.sqrt(rh**2+R**2))/(rh**2+R**2)**(3/2.)

##########################################################################################################################################################
# computation of intrinsic velocity dispersion (from Jeans equation)

# Mock data preparation
homedir = '/home/andrea/Desktop/work/DWARF/Jvalue/output/test/'
casedir = 'Isotrop_Core_nonPlum'
data = '/gs100_bs050_rcrs100_rarcinf_core_0400mpc3_df_100_0.dat'
err  = '/gs100_bs050_rcrs100_rarcinf_core_0400mpc3_df_100_0_err.dat'
x,y,z,vx,vy,vz = np.loadtxt(homedir+casedir+data,unpack=True)
R = np.sqrt(x**2+y**2) # assumed direction of observation along z-axis for simplicity (as suggested on the Gaia wiki)
D=50.
d = np.sqrt(x**2+y**2+(D-z)**2)
v = (x*vx+y*vy+(D-z)*vz)/d
# Errors (from mock data) preparation
Evx,Evy,Evz = np.loadtxt(homedir+casedir+err,unpack=True,usecols=(3,4,5))
Ex,Ey,Ez = np.absolute(Evx-vx),np.absolute(Evy-vy),np.absolute(Evz-vz)
dv = (x*Ex+y*Ey+(D-z)*Ez)/d

u=0.
rh=1.
nstars=np.size(R)

A_array = np.array([Ri/I(Ri,rh) for Ri in R])
r0_array = np.logspace(-4,4,200)
I_array=np.zeros(shape=(len(A_array),len(r0_array)))

def array_builder(r_array, rh, R_array):
    for i,R in enumerate(R_array):
        for j,r in enumerate(r_array):
            yield (i, j), (r, rh, R)

def proxy(args):
    return args[0], integral2(*args[1])*A_array[args[0][0]]

pool = Pool(processes=4)
results = pool.map(proxy, array_builder(r0_array, rh, R))
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
J_array = np.linspace(15,25,100)
LikeJ = np.zeros_like(r0_array)
J_new = np.empty([0])
min_LikeJ = np.empty([0])
num = 8
div = np.size(r0_array)/num
acc = 3.
for J in J_array:
    min_r0 = np.zeros(shape=(div,2))
    for j,r0 in enumerate(r0_array):
        log10rho0 = sciopt.minimize_scalar(deltaJ,args=(J,j)).x
        LikeJ[j] = logLike(10**log10rho0*r0**3,j)
    spline_LikeJ = interp(r0_array,LikeJ)
    
    a, b = 0, div-1
    for i in range(num):
        loc_min_r0 = sciopt.minimize_scalar(spline_LikeJ,method='Bounded',bounds=(r0_array[a],r0_array[b]))
        if (r0_array[a]+r0_array[a+1])/acc<loc_min_r0.x<(r0_array[b-1]+r0_array[b])/acc:
            min_r0[i,:] = (loc_min_r0.x,loc_min_r0.fun)
        a = b
        b += div
    
    min_r0 = np.delete(min_r0,np.where(min_r0==0.),axis=0)
    new_min_r0 = np.array([min_r0[i] for i in sorted(range(len(min_r0)),key = lambda k : min_r0[k,1])])
    if new_min_r0.size!=0:
        min_LikeJ = np.append(min_LikeJ,new_min_r0[0,1])
        J_new = np.append(J_new,J)

##########################################################################################################################################################
# minimum and C.I. determination

spline_LikeJ = spline(J_new,min_LikeJ)
min_J = sciopt.minimize_scalar(spline_LikeJ,method='Bounded',bounds=(J_new[0],J_new[-1])).x

def one_sigmaJ(J):
    return np.abs(spline_LikeJ(J)-spline_LikeJ(min_J)-0.5)

one_sigma_l = sciopt.minimize_scalar(one_sigmaJ,method='Bounded',bounds=(min_J-2,min_J)).x-min_J
one_sigma_r = sciopt.minimize_scalar(one_sigmaJ,method='Bounded',bounds=(min_J,min_J+2)).x-min_J

def two_sigmaJ(J):
    return np.abs(spline_LikeJ(J)-spline_LikeJ(min_J)-2.)

two_sigma_l = sciopt.minimize_scalar(two_sigmaJ,method='Bounded',bounds=(min_J-2,min_J)).x-min_J
two_sigma_r = sciopt.minimize_scalar(two_sigmaJ,method='Bounded',bounds=(min_J,min_J+2)).x-min_J

def three_sigmaJ(J):
    return np.abs(spline_LikeJ(J)-spline_LikeJ(min_J)-4.)

three_sigma_l = sciopt.minimize_scalar(three_sigmaJ,method='Bounded',bounds=(min_J-2,min_J)).x-min_J
three_sigma_r = sciopt.minimize_scalar(three_sigmaJ,method='Bounded',bounds=(min_J,min_J+2)).x-min_J

J_plt = np.linspace(min_J+three_sigma_l-0.5,min_J+three_sigma_r+0.5,100)
np.save('LikeJ',np.vstack((J_plt,spline_LikeJ(J_plt)-spline_LikeJ(min_J))))
yaml.dump({'N':int(nstars),'Jmin':min_J,'J1sL':one_sigma_l,'J1sR':one_sigma_r,'J2sL':two_sigma_l,'J2sR':two_sigma_r,
	'J3sL':three_sigma_l,'J3sR':three_sigma_r},open('results.yaml','wb'))
