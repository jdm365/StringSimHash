from setuptools import setup
from Cython.Build import cythonize
import numpy

## make python lib
setup(
    name='jaccard_kernel',
    ext_modules=cythonize("jaccard_kernel.pyx"),
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp']
)

