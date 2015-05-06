import NR 
from sys import arg 
import astrotools as AT
import NR

# various parameters definitions
nparams = 7 
nmax 	= 1000
pi 		= 3.14159e0
rstar	= 0.3e0
kpctokm	= 3.0856e16
G 		= 6.67e-11*1.e30*1.e-9  	# km^3 Msun^-1 s^-2
rt_max 	= 1.e1 						# the maximum value for tidal radius
ncalls 	= 10000						# number of mcmc accepts
idum 	= -46812					# seed for random number generators

galaxy  = sys.arg[1]							# get the galaxy name from the command line
pa,pmin,pmax,like_val = AT.get_data(galaxy)		# get the data for the galaxy 

# Keep track of accepts and rejects
count_accept = 0 
count_reject = 0 

# Get initial values for the likelihood function at pa
l 		= []
lmax 	= []
laccept = []
l[1] 	= dlike(pa,like_val)
lmax 	= l[1] 	
laccept = l[1]
xx 		= np.empty([nparams])

filename = 'out_'+galaxy+'.dat'
output = open(filename,'w')					# output file for the list of parameters 
accept = 0
reject = 0 

#	Core of mcmc code
print 'Fraction of accepted points,  # of accepted pts, value of likleihood'
while (accept <= ncalls):
	if(dlike[pa] >= laccept):
		laccept = AT.dlike(pa)
		accept += 1
		output.write([pa(j) for j in range(nparams)],p0)
		print float(accept)/float(ncalls),accept,laccept 
	else:
		ratio = AT.dlike(pa)/laccept
		xxx = NR.ran2(idum)
		if(xxx <= ratio):
			laccept = AT.dlike(pa)
			accept += 1
			output.write([pa[j] for j in range(nparams)],p0)
			print float(accept)/float(ncalls),accept,laccept 
		else:
			reject +=1 
	for j in range(nparams):
		xx[j] = NR.ran2(idum)
	pa = AT.getparams(pmin,pmax,xx)

output.close()
print accept,' accepts and ', reject,' rejects',accept+reject  

