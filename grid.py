#!/usr/bin/python
# author Andrea Chiappo		<andrea.chiappo@fysik.su.se>
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

# this script is meant to extract the -log(Like) Best-Fit parameter values calculated
# by iMinuit and stored in the logfile, load the grid values of the -log(Like) - also
# evaluated in the Minuit scprit (i.e. jmin.py) - and display the correspoding colour map

# i) extract BF parameters
dwarf  = sys.argv[1]							# get the galaxy name from the command line
params = 'rho0','rs'							# best-fit parameters to be extracted from the logfile		
BFparams = []
logfile = open('out_%s.txt'%dwarf,'r').readlines()
for param in params:
	for line in logfile:
		if param in line:
			data = line.strip().split()
			if len(data) > 4:
				BFparams.append(float(data[5]))

# ii) load -log(Like) grid values
pts = np.load("%s.npy"%dwarf)

# iii) display colour map
m = plt.imshow(pts,cmap='rainbow',extent=[1e-2,1e2,1e5,1e9])
plt.loglog()
plt.xlabel(r'$r_s$',fontsize=18)
plt.ylabel(r'$\rho_0$',fontsize=18)
plt.grid()
cx = plt.colorbar(m,pad=0)
cx.set_label(r'$-log(Like)$',fontsize=18)
plt.scatter(BFparams[1],BFparams[0],s=200,marker='*',c='k')		# display Best-Fit value
plt.show()
