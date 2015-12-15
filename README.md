# ASTROJPY
This Python script allows to build the (-log) profile Likelihood for the J factor of Dwarf Spheroidal Satellite Galaxies (dSPhs) 
of the Milky Way. It uses as input the kinematic data from the dSphs member stars. It also performs a basic statistical analysis,
which comprises of: determination of Maximum Likelihood value and its Confidence Intervals.
This README contains the instructions on how to use it correctly.

FUNCTIONS.PYX MODULE
This file contains the definitions of various functions used by the main script (ASTROJPY). For a faster execution, these are
written in a format compatible with Python Cythonize package. In order to use it, it must first be compiled with a C++ compiler.
A Python script which does this is also included, "setup.py", which should be executed with the following commaned from the 
command line
$ python setup.py build_ext --inplace

DATA INPUT
Data should be input into the code as a three-columns datafile consisting of
1) projected distance of a star from the dSphs center
2) measured line-of-sight velocity of each star
3) measurement error on the line-of-sight velocity

DATA OUTPUT
The code produces two files:
a) "Like.npy" consists of the 2D, vertically stacked array of the profile likelihood components of the dSphs J factor. 
  Its extension means that it is a python numpy-saved objected, thus loadable with the command
  np.load("path/to/file/Like.npy")
b) "results.yaml" contains a (binary) python dictionary with the results of the statistical analysis on the profile
  Likelihood. These consist of: 
