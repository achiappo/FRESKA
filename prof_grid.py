#!/usr/bin/python
# author Andrea Chiappo		<andrea.chiappo@fysik.su.se>
import sys
import math
import yaml
import numpy as np
import matplotlib.pyplot as plt

# this script is meant to extract the -log(Like) Best-Fit parameter values calculated
# by iMinuit, load the grid values of the -log(Like) - also evaluated in the Minuit scprit 
# (i.e. jprof_like.py) - and display the correspoding colour map

# i) extract BF parameters
dwarf  = sys.argv[1]							# get the galaxy name from the command line

# ii) load -log(Like) grid values and get minimum
pts = np.load("output/%s.npy"%dwarf)
rs,f,val = np.loadtxt('output_%s.dat'%dwarf,unpack=True,skiprows=1)
min = np.where(val == min(val))[0][0]
BFrs = rs[min]
BFf = f[min]

# iii) display colour map
fig = plt.figure()
fig.suptitle('%s'%dwarf,fontsize=18)
m = plt.imshow(np.flipud(pts),cmap='rainbow',extent=[1.e-3,10.,2.,12.])
ct = plt.contour(np.linspace(1.e-3,10.,num=1000),np.linspace(2.,12.,num=1000),pts)
plt.clabel(ct,inline=1,fmt='%1.0f',colors='k')
plt.xlabel(r'$r_s [kpc]$',fontsize=18)
plt.ylabel(r'$log_{10}(\rho_0 r_s^3 [M_{sun}])$',fontsize=18)
#plt.yticks(range(10),['%1.0e'%10**i for i in range(10)])
plt.grid()
cx = plt.colorbar(m,pad=0)
cx.set_label(r'$-log$Like$(\rho_{0,min},r_s|\sigma_p)$',fontsize=18)
BFrho0 = BFf-3*math.log10(BFrs)
plt.scatter(BFrs,BFf,s=200,marker='*',c='y',label=r'$r_s = $ %1.2f, $log_{10}(\rho_0) = $ %1.2f'%(BFrs,BFrho0))
plt.legend(scatterpoints=1,loc='lower right',fontsize=16)
plt.savefig('output/%s.png'%dwarf,dpi=100,format='png')
plt.show()
