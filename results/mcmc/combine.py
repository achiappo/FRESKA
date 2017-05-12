import numpy as np
from os import listdir

files = listdir('.')
paramfiles = sorted([f for f in files if 'params_1' in f], key=lambda x:x.split('_')[1].split('.')[0])

Npoints = int( raw_input('enter number of points in J: ') )
Nsample = int( raw_input('enter number of sample points: ') )
Nparams = int( raw_input('enter number of parameters: ') )

params = np.empty([Npoints, Nparams])
for i,p in enumerate(paramfiles):
	params[i] = np.load(p)

np.save('params_Mc_{0}p{1}s'.format(Npoints, Nsample), params)

