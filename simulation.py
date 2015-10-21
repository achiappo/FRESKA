import yaml
import numpy as np
from sys import argv
from scipy import special
from scipy.integrate import quad
from scipy import optimize as sciopt
from scipy.interpolate import UnivariateSpline as spline
from functions_sim import integral2, Jfactor
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

R,v = np.load('data_sim%i.npy'%argv[1])
u=0
dv=np.zeros_like(R)
D=50
rh=0.25
nstars=np.size(R)

A_array = [Ri/I(Ri,rh) for Ri in R]
r0_array = np.logspace(-2.,1.,100)
I_array=np.zeros(shape=(len(A_array),len(r0_array)))

def array_builder(r_array, R_array):
    for i,R in enumerate(R_array):
        for j,r in enumerate(r_array):
            yield (i, j), (r, rh, R)

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
    return abs(log10J-log10Jrho1[j]-2.*log10rho0)

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
    min_r0 = sciopt.minimize_scalar(spline_LikeJ)
    if r0_array[0]<min_r0.x<r0_array[-1]:
        J_new = np.append(J_new,J)
        min_LikeJ = np.append(min_LikeJ,min_r0.fun)

##########################################################################################################################################################
# minimum and C.I. determination

spline_Like = spline(J_new,min_LikeJ)
min_J = sciopt.minimize_scalar(spline_Like,method='Bounded',bounds=(J_new[0],J_new[-1])).x

def one_sigmaJ(J):
    return np.abs(spline_Like(J)-spline_Like(min_J)-0.5)

err_l = sciopt.minimize_scalar(one_sigmaJ,method='Bounded',bounds=(J_new[0],min_J)).x-min_J
err_r = sciopt.minimize_scalar(one_sigmaJ,method='Bounded',bounds=(min_J,J_new[-1])).x-min_J

yaml.dump({'N':argv[1],'Jmin':min_J,'Jr':err_r,'Jl':err_l},open('results%i'%argv[1],'wb'))
