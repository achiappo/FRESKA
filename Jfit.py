import yaml
import numpy as np
from sys import argv
from scipy import special
from scipy.integrate import quad
from scipy import optimize as sciopt
from scipy.interpolate import UnivariateSpline as spline
from scipy.interpolate import interp1d as interp
from functions import integral2, Jfactor, nu
from multiprocessing import Pool

params = yaml.load(open('input.yaml','rb'))

######################################################################################################################################
# integrand of I(R)
def integrand_I(r,R,rh,a,b,c):
    return nu(r/rh,a,b,c)*r/np.sqrt(r**2-R**2)

# dwarf surface brightness profile
def I(R,rh,a,b,c):
	return 2*quad(integrand_I,R,+np.inf,args=(R,rh,a,b,c))[0]

######################################################################################################################################
# computation of intrinsic velocity dispersion grid (from Jeans equation)

R,v,dv = np.load('input.npy')
u=v.mean()
nstars=np.size(R)

gamma_array = R/rh
r0_array = np.logspace(params['r0_i'],params['r0_f'],params['Nr0'])
ra_array = np.logspace(params['ra_i'],params['ra_f'],params['Nra'])

alpha_array = rh/r0_array
delta_array = ra_array/rh
A_array = np.array([gamma_array[i]/I(Ri,rh,params['aST'],params['bST'],params['cST']) for i,Ri in enumerate(R)])
I_array=np.zeros(shape=(len(A_array),len(delta_array),len(alpha_array)))

def array_builder(gamma_array, delta_array, alpha_array):
    for k,gamma in enumerate(gamma_array):
        for i,delta in enumerate(delta_array):
            for j,alpha in enumerate(alpha_array):
                yield (k, i, j), (gamma,delta,alpha)

def proxy(args):
	integral = integral2(*args[1],params['aST'],params['bST'],params['cST'],params['aDM'],params['bDM',params['cDM'])
    return args[0], A_array[args[0][0]]*integral

pool = Pool(processes=4)
results = pool.map(proxy, array_builder(gamma_array,delta_array,alpha_array))
pool.close()
pool.join()
for idx,value in results:
    I_array[idx] = value

Jf = np.sqrt([Jfactor(params['D'],np.inf,r0,1.,params['theta'],params['aDM'],params['bDM',params['cDM']]) for r0 in r0_array])

cst = 8.*np.pi*4.3e-6
# Likelihood definition
def logLike(J,i):
    I = cst*np.sqrt(J)*r0_array[i]**3*I_array[:,i]/Jf[i]
    S = dv**2.+I
    res = (np.log(S) + (v-u)**2./S).sum()
    return res/2.

######################################################################################################################################
# fitting scheme
J_array = np.linspace(params['J_i'],params['J_f'],params['NJ'])
J_new = np.empty([0])
min_LikeJ = np.empty([0])
min_ra_arr = np.empty([0])
min_r0_arr = np.empty([0])
num1 = params['Nmin1']
div = ra_array.size/num1
if ra_array.size%num1!=0 : num1+=1
for J in J_array:                                                    # scan over an array of J values
    num2 = params['Nmin2']
    LikeJbeta = np.empty([0])
    r0_new = np.empty([0])
    ra_new = np.empty([0])
    min_r0 = np.zeros(shape=(num2+1,2))
    for j,r0 in enumerate(r0_array):                                 # for each J scan over an array of r0 values
        min_ra = np.zeros(shape=(num1,2))
        LikeJ = np.zeros_like(ra_array)
        for i in range(ra_array.size): LikeJ[i]=logLike(10**J,i,j) # likelihood evaluation for each (r0,rho0)
        interp_LikeBeta = interp(ra_array,LikeJ)                     # build the profile likelihood along ra
        
        a, b = 0, div-1
        for i in range(num1):                                       # adaptive minima finding routine along ra
        	if b>=ra_array.size : b=-1
            loc_min_ra = sciopt.minimize_scalar(interp_LikeBeta,method='Bounded',bounds=(ra_array[a],ra_array[b]))
            if ra_array[a+1]<loc_min_ra.x<ra_array[b-1]: min_ra[i,:] = (loc_min_ra.x,loc_min_ra.fun)
            a = b
            b += div

        min_ra = np.delete(min_ra,np.where(min_ra[:,1]==0.),axis=0) # find the lowest minimum (in the range examined)
        new_min_ra = np.array([min_ra[i] for i in sorted(range(len(min_ra)),key = lambda k : min_ra[k,1])])
        if new_min_ra.size!=0:
        	LikeJbeta = np.append(LikeJbeta,new_min_ra[0,1])
            ra_new = np.append(ra_new,new_min_ra[0,0])
            r0_new = np.append(r0_new,r0)
    
    if np.size(r0_new)>num2*4:    # this condition guarantees that in the minimum-finding part c /= c+1 /= d-1 /= c
        interp_ra = interp(r0_new,ra_new)
        interp_LikeJ = interp(r0_new,LikeJbeta)                     # build the profile likelihood along r0
        
        div2 = r0_new.size/num2
        c, d = 0, div2-1
        if r0_new.size%num2!=0: num2+=1
        for i in range(num2):                                       # adaptive minima finding routine along r0
            if d>=r0_new.size: d=-1
            loc_min_r0 = sciopt.minimize_scalar(interp_LikeJ,method='Bounded',bounds=(r0_new[c],r0_new[d]))
            if r0_new[c+1]<loc_min_r0.x<r0_new[d-1]: min_r0[i,:] = (loc_min_r0.x,loc_min_r0.fun)
            c = d
            d += div2

        min_r0 = np.delete(min_r0,np.where(min_r0[:,1]==0.),axis=0) # find the lowest minimum (in the range examined)
        new_min_r0 = np.array([min_r0[i] for i in sorted(range(len(min_r0)),key = lambda k : min_r0[k,1])])
        if new_min_r0.size!=0:
            min_ra_arr = np.append(min_ra_arr,interp_ra(new_min_r0[0,0]))
            min_r0_arr = np.append(min_r0_arr,new_min_r0[0,0])
            min_LikeJ = np.append(min_LikeJ,new_min_r0[0,1])
            J_new = np.append(J_new,J)

######################################################################################################################################
# minimum and C.I. determination

interp_r0 = interp(J_new,min_r0_arr)
spline_LikeJ = spline(J_new,min_LikeJ)
interp_Like_ra = interp(J_new,min_ra_arr)

min_J = sciopt.minimize_scalar(interp_Like,method='Bounded',bounds=(J_new[0],J_new[-1])).x
min_ra = interp_Like_ra(min_J)

J_r0 = interp_r0(min_J)
J_rho0 = 10**sciopt.minimize_scalar(lambda log10rho0:abs(min_J-np.log10(Jfactor(D,np.inf,J_r0,1.,0.5))-2*log10rho0)).x

def one_sigmaJ(J):
    return np.abs(interp_Like(J)-interp_Like(min_J)-0.5)

one_sigma_l = sciopt.minimize_scalar(one_sigmaJ,method='Bounded',bounds=(min_J-1,min_J)).x-min_J
one_sigma_r = sciopt.minimize_scalar(one_sigmaJ,method='Bounded',bounds=(min_J,min_J+1)).x-min_J

def two_sigmaJ(J):
    return np.abs(interp_Like(J)-interp_Like(min_J)-2.)

two_sigma_l = sciopt.minimize_scalar(two_sigmaJ,method='Bounded',bounds=(min_J-1,min_J)).x-min_J
two_sigma_r = sciopt.minimize_scalar(two_sigmaJ,method='Bounded',bounds=(min_J,min_J+1)).x-min_J

def three_sigmaJ(J):
    return np.abs(interp_Like(J)-interp_Like(min_J)-4.)

three_sigma_l = sciopt.minimize_scalar(three_sigmaJ,method='Bounded',bounds=(min_J-1,min_J)).x-min_J
three_sigma_r = sciopt.minimize_scalar(three_sigmaJ,method='Bounded',bounds=(min_J,min_J+1)).x-min_J

yaml.dump({'N':argv[1],'Jmin':min_J,'r0':J_r0,'rho0':J_rho0,'J1sL':one_sigma_l,'J1sR':one_sigma_r,
	'J2sL':two_sigma_l,'J2sR':two_sigma_r,'J3sL':three_sigma_l,'J3sR':three_sigma_r},open('results.yaml','wb'))

LikeJ_array = np.linspace(min_J+three_sigma_l-0.5,min_J+three_sigma_r+0.5,100)
np.save('LikeJ',np.vstack((LikeJ_array,interp_Like(LikeJ_array)-spline_LikeJ(min_J))))
