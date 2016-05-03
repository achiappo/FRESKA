import yaml
import numpy as np
from math import *
from sys import argv
from scipy import special
from scipy.integrate import quad
from scipy import optimize as sciopt
from scipy.interpolate import interp1d as interp
from functions import integral2, Jfactor, get_data
from multiprocessing import Pool

################################################################################################################
# dwarf surface brightness profile
def I(R,rh):
	return 4./3. * rh/(1+(R/rh)**2)**2

###########################################################

dwarf = argv[1]
R,v,dv,D,rh,rt = get_data(dwarf)
u=v.mean()
theta=0.5

r0_i,r0_f,Nr0 = 4,4,200
r0_array = np.logspace(-r0_i,r0_f,Nr0)

case = 'IS_%s_%i%i%i%i_%i'%(dwarf,r0_i,r0_f,theta*10)

gamma_array = R/rh
alpha_array = rh/r0_array
A_array = np.array([gamma_array[i]/I(Ri,rh) for i,Ri in enumerate(R)])
I_array=np.zeros(shape=(len(A_array),len(r0_array)))

def array_builder(gamma_array, alpha_array):
    for i,gamma in enumerate(gamma_array):
        for j,alpha in enumerate(alpha_array):
            yield (i, j), (gamma, alpha)

def proxy(args):
    return args[0], A_array[args[0][0]]*integral2(*args[1])

pool = Pool(processes=4)
results = pool.map(proxy, array_builder(gamma_array, alpha_array))
pool.close()
pool.join()
for idx,value in results:
    I_array[idx] = value

Jf = np.sqrt([Jfactor(D,np.inf,r0,1.,0.5) for r0 in r0_array])

cst = 8.*np.pi*4.3e-6
# Likelihood definition (for free beta)
def logLike(J,i):
    I = cst*sqrt(J)*r0_array[i]**3*I_array[:,i]/Jf[i]
    S = dv**2.+I
    res = (np.log(S) + (v-u)**2./S).sum()
    return res/2.

##########################################################################################################################################################
#											MAXIMUM LIKELIHOOOD SCHEME
##########################################################################################################################################################
# fitting scheme
J_array = np.linspace(14,21,200)
J_new 	= np.empty([0])
min_LikeJ  = np.empty([0])
min_r0_arr = np.empty([0])

for J in J_array:                                                  # scan over an array of J values
    LikeJr0 = np.zeros_like(r0_array)
    for i in range(r0_array.size): LikeJr0[i] = logLike(10**J,i) # for each J scan over an array of r0 values
    interp_r0 = interp(r0_array,LikeJr0)                      # build the profile likelihood along ra
        
    eval_Like_r0 = np.logspace(log10(r0_array.min()),log10(r0_array.max()),1e3)
    min_Like_r0  = interp_r0(eval_Like_r0).min()
    min_r0 	     = eval_Like_r0[np.where(interp_r0(eval_Like_r0)==min_Like_r0)[0][0]]
        
    if r0_array[1]<min_r0<r0_array[-2]:
        min_r0_arr = np.append(min_r0_arr,min_r0)
        min_LikeJ  = np.append(min_LikeJ,min_Like_r0)
        J_new 	   = np.append(J_new,J)

##########################################################################################################################################################
# minimum and C.I. determination
interp_Like_J  = interp(J_new,min_LikeJ)
interp_Like_r0 = interp(J_new,min_r0_arr)

eval_Like_J = np.linspace(J_new.min(),J_new.max(),1e3)
min_Like_J  = interp_Like_J(eval_Like_J).min()
J_min       = eval_Like_J[np.where(interp_Like_J(eval_Like_J)==min_Like_J)[0][0]]

J_r0 	= float(interp_Like_r0(J_min))
J_rho0  = 10**sciopt.minimize_scalar(lambda log10rho0:abs(J_min-np.log10(Jfactor(D,np.inf,J_r0,1.,0.5))-2*log10rho0)).x

J1sL = round(sciopt.minimize_scalar(lambda J : np.abs(interp_Like_J(J)-interp_Like_J(J_min)-0.5),method='Bounded',bounds=(J_new[0],J_min)).x-J_min,2)
J1sR = round(sciopt.minimize_scalar(lambda J : np.abs(interp_Like_J(J)-interp_Like_J(J_min)-0.5),method='Bounded',bounds=(J_min,J_new[-1])).x-J_min,2)

J2sL = round(sciopt.minimize_scalar(lambda J : np.abs(interp_Like_J(J)-interp_Like_J(J_min)-2.),method='Bounded',bounds=(J_new[0],J_min)).x-J_min,2)
J2sR = round(sciopt.minimize_scalar(lambda J : np.abs(interp_Like_J(J)-interp_Like_J(J_min)-2.),method='Bounded',bounds=(J_min,J_new[-1])).x-J_min,2)

J3sL = round(sciopt.minimize_scalar(lambda J : np.abs(interp_Like_J(J)-interp_Like_J(J_min)-4.),method='Bounded',bounds=(J_new[0],J_min)).x-J_min,2)
J3sR = round(sciopt.minimize_scalar(lambda J : np.abs(interp_Like_J(J)-interp_Like_J(J_min)-4.),method='Bounded',bounds=(J_min,J_new[-1])).x-J_min,2)
J_true = log10(Jfactor(D,np.inf,1.,4e8,0.5))

if J_min+J3sL-0.1>J_new[0]: J_i = J_min+J3sL-0.1
else: J_i = J_new[0]
if J_min+J3sR+0.1<J_new[-1]: J_f = J_min+J3sR+0.1
else: J_f = J_new[-1]
J_plt = np.linspace(J_i,J_f,100)

np.save('results/LikeJ_%s'%case,np.vstack((J_plt,interp_Like_J(J_plt)-interp_Like_J(J_min))))
yaml.dump({'Nstars':R.size,'Jmin':J_min,'r0':J_r0,'rho0':J_rho0,'J1sL':J1sL,'J1sR':J1sR,
	'J2sL':J2sL,'J2sR':J2sR,'J3sL':J3sL,'J3sR':J3sR},open('results/results_%s.yaml'%case,'w'))
