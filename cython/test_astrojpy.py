import numpy as np

from scipy.interpolate import interp1d as interp
from scipy.optimize import brentq, minimize_scalar

from profiles import build_profile, build_kernel
from dispersion import SphericalJeansDispersion
from likelihood import GaussianLikelihood
from fitter import MinuitFitter

directory = '/home/andrea/Desktop/work/DWARF/dsphsim/Ret2_data/'
rh = 0.04
rs = 0.757/2.163
D = 39.81
theta = 2*rh/D
Jtrue = 16.7168

dm = build_profile('NFW',r0=rs)
st = build_profile('plummer',rh=rh)
kr = build_kernel('iso')

dwarf_props = {'D':D, 'theta':theta, 'rt':np.inf, 'with_errs':False}
Sigma = SphericalJeansDispersion(dm, st, kr, dwarf_props)

Jmin = np.zeros([100])
J1s = np.zeros([100,2])
J2s = np.zeros([100,2])
J3s = np.zeros([100,2])

in1s,in2s,in3s = 0,0,0

for i in range(100):
	R, v = np.loadtxt(directory+'dsph_%03d.txt'%(i+1),usecols=(5, 7),unpack=True)
	vnan = ~np.isnan(v) 
	v = v[vnan]
	R = R[vnan]
	dv = np.zeros_like(v)
	LL = GaussianLikelihood([R, v, dv, 0.], Sigma)

	J_array = np.linspace(16,18,30)
	J_new = np.empty([0])
	L_arr = np.empty([0])

	for J in J_array:
		M = MinuitFitter(LL)
		M.set_value('J',J)
		M.set_fixed('J')
		M.set_minuit()
		#-----------------------------------------------
		BF = M.migrad_min()
		if BF[0]['is_valid']:
			J_new = np.append(J_new,J)
			L_arr = np.append(L_arr,BF[0]['fval'])

	#####################################################################

	interp_L = interp( J_new, L_arr-L_arr.min() )

	eval_Like_J = np.linspace(J_new.min(), J_new.max(), 1e3)
	min_Like_J = interp_L(eval_Like_J).min()
	jmin = eval_Like_J[ np.where( interp_L(eval_Like_J) == min_Like_J )[0][0] ]
	Jmin[i] = jmin

	for n,c in enumerate([0.5,2.0,4.5]):
		exec( 'J%is[i,0] = %g'%(n+1, brentq(lambda j : interp_L(j)-c, a=J_new.min(), b=jmin) ) )
		exec( 'J%is[i,1] = %g'%(n+1, brentq(lambda j : interp_L(j)-c, a=jmin, b=J_new.max()) ) )

	if J1s[i,0] < Jtrue < J1s[i,1] : in1s += 1
	if J2s[i,0] < Jtrue < J2s[i,1] : in2s += 1
	if J3s[i,0] < Jtrue < J3s[i,1] : in3s += 1

print 'J = %g +- %g'%( Jmin.mean(), Jmin.std()/np.sqrt(Jmin.size) )

print '1-sigma: ', in1s
print '2-sigma: ', in2s
print '3-sigma: ', in3s
