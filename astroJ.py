import yaml
import numpy as np
from os import system,remove
from AT import get_data

# this performs cythonisation of the functions.pyx module
system('python setup.py build_ext --inplace')

###################################################################################################
#										PARAMETERS INPUT
###################################################################################################

# SHAPE PARAMETERS OF STELLAR DENSITY PROFILE
aST, bST, cST = 2, 5, 0  # Plummer profile

# SHAPE PARAMETERS OF DM DENSITY PROFILE
aDM, bDM, cDM = 1, 1, 3  # Cusped NFW

# LIKELIHOOD GRID INITIALISATION
# For each of the chosen arrays (ra,r0,b,J) input: INITIAL value, FINAL value, Number 
# of evaluated points in log scale (eg. 4 means 1e4)

# r0 ARRAY
r0_i, r0_f, Nr0 = -3, 3, 100
# ra ARRAY
ra_i, ra_f, Nra = -3, 3, 100
# beta ARRAY
beta_i, beta_f, Nbeta  	= -3, 3, 100

# J GRID: Input INITIAL value, FINAL value and Number of points for the J values examined
J_i, J_f, NJ	= 17, 21, 100

# Number of subintervals into which divide the ra array when profiling
Nmin1 = 10

# Number of subintervals into which divide the r0 array when profiling
Nmin2 = 10

# Select anisotropy profile to adopt: Isotropic (IS), Charbonnier (CA), Osipkov-Merrit (OM)
ASTY  = 'IS'

# Max integration angle when evaluating the J factor
theta = 0.5

# Extract the data and parameters from the data and parameter files
R,v,dv,rh,rt,nstars,D = get_data(dwarf)

# save data vectors into numpy object
data = np.save('input',np.vstack(R,v,dv))

# Save all the input into a yaml file to be read by the execution file
yaml.dump({'aST':aST,'bST':bST,'cST':cST,'aDM':aDM,'bDM':bDM,'cDM':cDM,
	'r0_i':r0_i,'r0_f':r0_f,'Nr0':Nr0,'ra_i':ra_i,'ra_f':ra_f,'Nra':Nra,
	'beta_i':beta_i,'r0_f':r0_f,'Nr0':Nr0,'J_i':J_i,'J_f':J_f,'NJ':NJ,'Nmin1':Nmin1,
	'Nmin2':Nmin2,'ASTY':ASTY,'rh':rh,'rt':rt,'D':D,'theta':theta},open('input.yaml',''))

###################################################################################################
#										CODE EXECUTION
###################################################################################################

system('python Jfit.py')

