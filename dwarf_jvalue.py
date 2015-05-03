# python version of fortran code to calculate the j-value of dSphs

import NR 
from sys import arg 
import astrotools as AT
import NR

galaxy 	  = sys.arg[1]

# various parameters definitions
nparams = 7 
nmax 	= 1000
pi 		= 3.14159e0
rstar	= 0.3e0
kpctokm	= 3.0856e16
G 		= 6.67e-11*1.e30*1.e-9  # km^3 Msun^-1 s^-2
rt_max 	= 1.e1 					# the maximum value for tidal radius. Set for Segue 1 at 0.6 kpc motivated by arguments of Geha et al. 2009
ncalls 	= 10000					# number of mcmc accepts
idum 	= -46812					# seed for random number generators

get_basedir(directory)		# get the base directory from the environment
get_galaxy(galaxy) 			# get the galaxy name
get_data(galaxy) 			# get the data for the galaxy 

# Keep track of accepts and rejects
count_accept = 0 
count_reject = 0 

# Get initial values for the likelihood function at pa
l 		= []
lmax 	= []
laccept = []
l[1] 	= dlike(pa)
lmax 	= l[1] 	
laccept = l[1]

filename = 'out_'+galaxy+'.dat'
output = open(filename,'w')					# output file for the list of parameters 
accept = 0
reject = 0 

#	Core of mcmc code. Skip to end of main program if not interested 
print 'Fraction of accepted points,  # of accepted pts, value of likleihood'
while (accept <= ncalls):
	if(dlike[pa] >= laccept):
		laccept = dlike[pa]
		accept += 1
		output.write(pa(j), j = 1,nparams,p0) 					# POTENTIAL SOURCE OF BUG
		print float(accept)/float(ncalls),accept,laccept 
	else:
		ratio = dlike(pa)/laccept
		xxx = ran2(idum)
		if(xxx <= ratio):
			laccept = dlike(pa)
			accept += 1
			output.write(pa(j), j = 1,nparams,p0) 			# POTENTIAL SOURCE OF BUG
			print float(accept)/float(ncalls),accept,laccept 
		else:
			reject +=1 
	for j in range(1,nparams):
		xx[j] = ran2(idum)
	getparams(pa,pmin,pmax,xx)
output.close()
print accept,' accepts and ', reject,' rejects',accept+reject  

