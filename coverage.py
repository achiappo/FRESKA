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
# Errors (from mock data) preparation
Evx,Evy,Evz = np.loadtxt('gs100_bs050_rcrs025_rarcinf_cusp_0064mpc3_df_10000_0_err.dat',unpack=True,usecols=(3,4,5))
Ex,Ey,Ez = np.absolute(Evx-vx),np.absolute(Evy-vy),np.absolute(Evz-vz)
dv = (x*Ex+y*Ey+(D-z)*Ez)/d

trueJ = round(np.log10(Jfactor(D,np.inf,1.,6.4e7,0.5)),2)
meanJ,in1s,in2s,in3s,Err1sL,Err1sR,Err2sL,Err2sR,Err3sL,Err3sR = 0,0,0,0,0,0,0,0,0,0

for i in range(100):
	np.save('data_sim%i'%i,np.vstack((R[100*i:100*(i+1)],v[100*i:100*(i+1)],dv[100*i:100*(i+1)])))
	system('python Jfit.py %i'%i)
	remove('data_sim%i.npy'%i)
	results = yaml.load(open('results/results%i.yaml'%i,'rb'))
	#remove('results%s.yaml'%argv[1])
	meanJ  += results['Jmin']/100 		# mean of best fit J
	Err1sL += results['J1sL']/100. 		# mean of 1-sigma left  errorbars
	Err1sR += results['J1sR']/100. 		# mean of 1-sigma right errorbars
	Err2sL += results['J2sL']/100. 		# mean of 2-sigma left  errorbars
	Err2sR += results['J2sR']/100. 		# mean of 2-sigma right errorbars
	Err3sL += results['J3sL']/100. 		# mean of 3-sigma left  errorbars
	Err3sR += results['J3sR']/100. 		# mean of 3-sigma right errorbars
	if results['Jmin']+results['J1sL']<trueJ<results['Jmin']+results['J1sR']: in1s +=1 	# check 1-sigma containment
	if results['Jmin']+results['J2sL']<trueJ<results['Jmin']+results['J2sR']: in2s +=1 	# check 2-sigma containment
	if results['Jmin']+results['J3sL']<trueJ<results['Jmin']+results['J3sR']: in3s +=1 	# check 3-sigma containment

output = open('coverage_Isotrop+Cusp+nonPlum.txt','w')
output.write('%20s %4.2f  \n'%('N of times the true value is inside 1-sigma = ',in1s/100.))
output.write('%20s %4.2f  \n'%('N of times the true value is inside 2-sigma = ',in2s/100.))
output.write('%20s %4.2f  \n'%('N of times the true value is inside 3-sigma = ',in3s/100.))
output.write('%8s  %4.2f  \n'%('Mean J = ',meanJ))
output.write('%10s %4.2f  \n'%('Mean 1-sigma left   errorbar =  ',Err1sL))
output.write('%10s %+4.2f \n'%('Mean 1-sigma right  errorbar =  ',Err1sR))
output.write('%10s %4.2f  \n'%('Mean 2-sigma left   errorbar =  ',Err2sL))
output.write('%10s %+4.2f \n'%('Mean 2-sigma right  errorbar =  ',Err2sR))
output.write('%10s %4.2f  \n'%('Mean 3-sigma left   errorbar =  ',Err3sL))
output.write('%10s %+4.2f \n'%('Mean 3-sigma right  errorbar =  ',Err3sR))
output.close()
