import yaml
import numpy as np
from os import system,remove

# the following performs cythonisation of the functions.pyx module
system('python setup.py build_ext --inplace')


system('python Jfit.py')
