from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy
from torch.utils.cpp_extension import include_paths

extensions = [
    Extension("env", ["env.pyx"],
        include_dirs=[numpy.get_include()] + include_paths(),
        extra_compile_args=['-O2', '-msse4.2'])
]
setup(
    name="env",
    ext_modules=cythonize(extensions),
)