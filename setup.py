from distutils.core import setup
from Cython.Build import cythonize
import os 
setup(
    ext_modules = cythonize(["astrojpy/cyfuncs.pyx"])
)
os.rename('cyfuncs.so', 'astrojpy/cyfuncs.so')
