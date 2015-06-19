#!/usr/bin/python
# author Andrea Chiappo		<andrea.chiappo@fysik.su.se>
import sys
import math
import yaml
import numpy as np
import matplotlib.pyplot as plt

# this script is meant to extract the -log(Like) Best-Fit parameter values calculated
# by iMinuit and stored in the logfile, load the grid values of the -log(Like) - also
# evaluated in the Minuit scprit (i.e. jmin.py) - and display the correspoding colour map

# i) extract BF parameters
dwarf  = sys.argv[1]							# get the galaxy name from the command line

bestfit = yaml.load(open("output/%s.yaml"%dwarf,"rb"))
rho0 = bestfit[1][0]["value"]
rs   = bestfit[1][1]["value"]

# ii) load -log(Like) grid values
pts = np.load("output/%s.npy"%dwarf)

# iii) display colour map
m = plt.imshow(np.flipud(pts),cmap='rainbow',extent=[0.1,rs+1.,rho0-1.,rho0+1.])
ct = plt.contour(np.linspace(0.1,rs+1.,20),np.linspace(rho0-1.,rho0+1.,20),pts)
plt.clabel(ct,inline=1,fmt='%1.0f',colors='k')
plt.semilogy()
plt.xlabel(r'$r_s [kpc]$',fontsize=18)
plt.ylabel(r'$\rho_0 [M_{sun}  kpc^{-3}]$',fontsize=18)
plt.yticks([int(rho0),int(rho0+1.)],['%1.0e'%10**int(rho0),'%1.0e'%10**int(rho0+1.)])
plt.grid()
cx = plt.colorbar(m,pad=0)
cx.set_label(r'$-log(Like)$',fontsize=18)
plt.scatter(rs,rho0,s=200,marker='*',c='k',label=r'$r_s = $ %1.1f, $log_{10}(\rho_0) = $ %1.1f'%(rs,rho0))
plt.legend(scatterpoints=1,loc='lower right',fontsize=12)
plt.savefig('output/%s_im.png'%dwarf,dpi=100,format='png')
plt.show()
