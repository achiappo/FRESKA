
#	PYTHON code to fit the J-factor of dSphs using Maximum Likelihood. This version assumes a CONSTANT ANISOTROPY velocity distribution of the
# 	dSphs stellar component. Three possible stellar light profiles can be chosen from the Hernquist profile (select stelprof) and two
# 	versions of the NFW DM profile: Cusped and Cored (select DMprof)

import yaml
from math import *
import numpy as np
from sys import argv
from iminuit import Minuit
from multiprocessing import Pool
from scipy.integrate import quad
from scipy import optimize as sciopt
from scipy.interpolate import interp1d as interp
from functionsCA import integral2, Jfactor, nu, get_data

stelprof = ['Plum','truePlum','nonPlum']
DMprof = ['NFW_Cusp','NFW_Core']
stelmod = 1
DMmod = 0

dwarf = argv[1]
R,v,dv,D,rh,rt = get_data(dwarf)
u=v.mean()
theta=0.5

case = 'CA_%s_0%i_%s_%s'%(dwarf,theta*10,DMprof[DMmod],stelprof[stelmod])

################################################################################################################
# inverse hyperbolic cosecant (used for gamma* = 1 , non-Plum)
def inv_csch(x):
    return np.log(np.sqrt(1+x**-2.)+x**-1.)
# integrand of I(R) (used for gamma* = 0.1 , Plum)
def integrand_I(r,rh,R):
    return nu(r/rh)*r/np.sqrt(r**2-R**2)

# dwarf surface brightness profile
def I(R,rh):
    if stelmod==0: return 2*quad(integrand_I,R,+np.inf,args=(rh,R))[0] 
    elif stelmod==1: return 4./3.*rh/(1+(R/rh)**2)**2
    else: return rh**2*((2*rh**2+R**2)*inv_csch(R/rh)-rh*np.sqrt(rh**2+R**2))/(rh**2+R**2)**(3/2.)        

################################################################################################################
# definition of the (Gaussian) Likelihood used by the fitter

cst = 8.*np.pi*4.3e-6
gamma_array = R/rh

class logLike:
    def __init__(self,data):
        self.data = data

    def compute(self,J,beta,r0):
        gamma = self.data
        A_array = np.array([gamma[i]**(1.-2*beta)/I(Ri,rh) for i,Ri in enumerate(R)])
        I_array = np.array([integral2(gamma[i],beta,rh/r0) for i in range(R.size)])
        Int = cst*sqrt(10**J)*r0**3*A_array*I_array/np.sqrt(Jfactor(D,np.inf,r0,1.,theta))
        S = dv**2.+Int
        res = (np.log(S)+(v-u)**2./S).sum()
        return res/2.

##########################################################################################################################################################
# fitting scheme
J_array = np.linspace(17,21,50)
J_new 	= np.empty([0])
min_LikeJ  = np.empty([0])
min_b_arr  = np.empty([0])
min_r0_arr = np.empty([0])
for J in J_array:
    lh = logLike(gamma_array)
    settings = {'errordef':0.5,'print_level':0,'pedantic':False,'beta':0,'r0':rt/2.,'J':J,'fix_J':True,
    'error_beta':0.001,'error_r0':0.001,'limit_beta':(-10.,1.),'limit_r0':(R.min(),rt*100)}
    Jfit = Minuit(lh.compute,**settings)
    Jfit.tol = 0.01
    BF = Jfit.migrad()
    if BF[0]['is_valid']:
    	J_new = np.append(J_new,J)
    	min_LikeJ  = np.append(min_LikeJ,BF[0]['fval'])
    	min_b_arr  = np.append(min_b_arr,BF[1][1]['value'])
    	min_r0_arr = np.append(min_r0_arr,BF[1][2]['value'])

##########################################################################################################################################################
# minimum and C.I. determination

interp_Like_J  = interp(J_new,min_LikeJ)
interp_Like_b  = interp(J_new,min_b_arr)
interp_Like_r0 = interp(J_new,min_r0_arr)

eval_Like_J = np.linspace(J_new.min(),J_new.max(),1e3)
min_Like_J  = interp_Like_J(eval_Like_J).min()
J_min       = eval_Like_J[np.where(interp_Like_J(eval_Like_J)==min_Like_J)[0][0]]

J_b    = float(interp_Like_b(J_min))
J_r0   = float(interp_Like_r0(J_min))
J_rho0 = 10**sciopt.minimize_scalar(lambda log10rho0 : abs(J_min-np.log10(Jfactor(D,np.inf,J_r0,1.,theta))-2*log10rho0)).x

J1sL = round(sciopt.minimize_scalar(lambda J : np.abs(interp_Like_J(J)-interp_Like_J(J_min)-0.5),method='Bounded',bounds=(J_new[0],J_min)).x-J_min,2)
J1sR = round(sciopt.minimize_scalar(lambda J : np.abs(interp_Like_J(J)-interp_Like_J(J_min)-0.5),method='Bounded',bounds=(J_min,J_new[-1])).x-J_min,2)

J2sL = round(sciopt.minimize_scalar(lambda J : np.abs(interp_Like_J(J)-interp_Like_J(J_min)-2.),method='Bounded',bounds=(J_new[0],J_min)).x-J_min,2)
J2sR = round(sciopt.minimize_scalar(lambda J : np.abs(interp_Like_J(J)-interp_Like_J(J_min)-2.),method='Bounded',bounds=(J_min,J_new[-1])).x-J_min,2)

J3sL = round(sciopt.minimize_scalar(lambda J : np.abs(interp_Like_J(J)-interp_Like_J(J_min)-4.),method='Bounded',bounds=(J_new[0],J_min)).x-J_min,2)
J3sR = round(sciopt.minimize_scalar(lambda J : np.abs(interp_Like_J(J)-interp_Like_J(J_min)-4.),method='Bounded',bounds=(J_min,J_new[-1])).x-J_min,2)

if J_min+J3sL-0.1>J_new[0]: J_i = J_min+J3sL-0.1
else: J_i = J_new[0]
if J_min+J3sR+0.1<J_new[-1]: J_f = J_min+J3sR+0.1
else: J_f = J_new[-1]
J_plt = np.linspace(J_i,J_f,100)

J1sL = J1sL if J_min+J1sL-0.01>=J_i else ''
J1sR = J1sR if J_min+J1sR+0.01<=J_f else ''
J2sL = J2sL if J_min+J2sL-0.01>=J_i and not J2sL==J1sL else ''
J2sR = J2sR if J_min+J2sR+0.01<=J_f and not J2sR==J1sR else ''
J3sL = J3sL if J_min+J3sL-0.01>=J_i and not J3sL==J2sL else ''
J3sR = J3sR if J_min+J3sR+0.01<=J_f and not J3sR==J2sR else ''

##########################################################################################################################################################
# output storing: Likelihood in a npy file and results in a yaml
np.save('results/LikeJ_%s'%case,np.vstack((J_plt,interp_Like_J(J_plt)-interp_Like_J(J_min))))
yaml.dump({'Nstars':R.size,'Jmin':J_min,'r0':J_r0,'rho0':J_rho0,'b':J_b,'J1sL':J1sL,'J1sR':J1sR,
	'J2sL':J2sL,'J2sR':J2sR,'J3sL':J3sL,'J3sR':J3sR},open('results/results_%s.yaml'%case,'w'))
