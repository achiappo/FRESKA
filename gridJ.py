import sys
import math
import yaml
from jcalc import *
from iminuit import Minuit
from iminuit.util import describe
from ATminuit import get_data,get_sigmalos
import numpy as np

dwarf  = sys.argv[1]		# get the galaxy name from the command line
D = get_data(dwarf)[-2] 	# get its distance (in kpc) from parameter file

density=Units.msun_kpc3
distance=Units.kpc

def Jvalue(dist,rhos,rs):				# evaluate the J factor integrating it on a solid angle given by psi
	rhos  *= density
	rs    *= distance
	dist  *= distance
	params = dict(type='nfw',rs=rs,rhos=rhos)
	dp = DensityProfile.create(params)
	psi = 0.5 # deg
	jprofile = JProfile.create(dp,dist,None)
	J_int = jprofile.integrate(psi)/Units.gev2_cm5
	return J_int
'''
data = yaml.load(open('output/umaI.yaml','r'))
rhos = data[1][0]['value']
rs = data[1][1]['value']
print math.log10(Jvalue(D,10**rhos,rs))
'''	
rhos_values = np.logspace(9.,5.,100)				#
rs_values 	= np.linspace(1.e-2,2.,100)				# initialise the grid of size 100
pts = np.empty([len(rhos_values),len(rs_values)])	#

for j,rhos in enumerate(rhos_values):				#
	for i,rs in enumerate(rs_values):				# fill the grid of J(rhos,rs)
		pts[j,i] = Jvalue(D,rhos,rs)				#
		print j,rhos,i,rs,pts[j,i]

np.save('output/jgrid_'+dwarf,pts)
