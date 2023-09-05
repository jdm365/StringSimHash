from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("minhash.pyx"),
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp']
)

