from distutils.core import setup
from Cython.Build import cythonize

setup(
    #ext_modules = cythonize(["profiles.pyx"])
    ext_modules = cythonize(["cyfuncs.pyx"])
)
