import matplotlib.pyplot as plt
import numpy as np
import sys

dwarf = sys.argv[1]

rs,rho0,fval = np.loadtxt('output_%s.dat'%dwarf,unpack=True,skiprows=1)

fig = plt.figure()
fig.suptitle('%s'%dwarf,fontsize=18)
plt.plot(rs,fval-min(fval),'-')
plt.plot(rs[np.where(fval == min(fval))],0.,'D',label=r'$r_s = $'+str(rs[np.where(fval == min(fval))[0]]))
plt.xlabel(r'$r_s [kpc]$',fontsize=18)
plt.ylabel(r'$\Delta log$Like$(\rho_{0,min},r_s|\sigma_p)$',fontsize=18)
plt.ylim(-0.05,2.)
plt.hlines(0.,0.,10.,linestyles='dashed')
plt.hlines(.5,0.,10.,colors=('r'),linestyles='dashed',label=r'$1-\sigma$')
#plt.hlines(.,0.,10.,colors=('r'),linestyles='dashed',label=r'$1-\sigma$')
plt.legend(numpoints=1,loc='upper left',fontsize=16)
plt.savefig('output/profL_%s.png'%dwarf,dpi=100,format='png')
plt.show()
