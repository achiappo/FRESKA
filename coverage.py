import yaml
import numpy as np
from os import system,remove

##########################################################################################################################################################
# Mock data preparation
x,y,z,vx,vy,vz = np.loadtxt('gs100_bs050_rcrs025_rarcinf_cusp_0064mpc3_df_10000_0.dat',unpack=True)
R = np.sqrt(x**2+y**2) # assumed direction of observation along z-axis for simplicity (as suggested on the Gaia wiki)
D=50.
d = np.sqrt(x**2+y**2+(D-z)**2)
v = (x*vx+y*vy+(D-z)*vz)/d
trueJ = 19.35
within = 0
outside = 0

for i in range(100):
	np.save('data_sim%i'%i,np.vstack((R[100*i:100*(i+1)],v[100*i:100*(i+1)])))
	system('python simulation.py %i'%i)
	remove('data_sim%i.npy'%i)
	results = yaml.load(open('results%i.yaml'%argv[1],'rb'))
	#remove('results%i.yaml'%argv[1])
	if results['Jmin']-results['Jl']<trueJ<results['Jmin']+results['Jr']:
		within +=1
		print ('%10s %10.2f %+10.2f %10.2f')%('J = ',results['Jmin'],results['Jr'],results['Jl'])
	else:
		outside += 1

print 'N of times the true value is contained = ',within/100.,'\n','N of times the true value is outside = ',outside/100.

