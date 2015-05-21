import sys
from astrotools import get_data,dlike
from numpy.random import random_sample

# various parameters definitions
nparams = 7
nmax 	= 1000
ncalls 	= 10000				# number of mcmc accepts

galaxy  = sys.argv[1]								# get the galaxy name from the command line
like_val,pa,pmin,pmax = get_data(galaxy)[7:]		# get the data for the galaxy 

# Get initial values for the likelihood function at pa
laccept = dlike(galaxy)[0]
xx 		= np.empty([nparams])

output = open('out_'+galaxy+'.dat','w')				# output file for the list of parameters 

# Keep track of accepts and rejects
accept = 0
reject = 0 

#	Core of mcmc code
print 'Fraction of accepted points,  # of accepted pts, value of likleihood'
while (accept <= ncalls):
	if(dlike(galaxy)[0] >= laccept):
		laccept,p0 = dlike(galaxy)
		accept += 1
		output.write(('%14.5e %14.5e %14.5e %14.5e %14.5e %14.5e %14.5e %14.5e')%\
			(pa[0],pa[1],pa[2],pa[3],pa[4],pa[5],pa[6],p0))
		print ('%20f %20f %20f')%(float(accept)/float(ncalls),accept,laccept)
	else:
		ratio = dlike(galaxy)[0]/laccept
		xxx = random_sample()
		if(xxx <= ratio):
			laccept,p0 = dlike(galaxy)
			accept += 1
			output.write(('%14.5e %14.5e %14.5e %14.5e %14.5e %14.5e %14.5e %14.5e')%\
				(pa[0],pa[1],pa[2],pa[3],pa[4],pa[5],pa[6],p0))
			print ('%20f %20f %20f')%(float(accept)/float(ncalls),accept,laccept)
		else:
			reject +=1 
	for j in range(nparams):
		xx[j] = random_sample()
	pa = xx*(pmax-pmin)+pmin

output.close()
print accept,' accepts and ', reject,' rejects',' | total: ',accept+reject  

