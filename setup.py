from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np


ext_core = Extension(
        "metaseg_metrics",
        sources=["semseg_density/metaseg_metrics.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"])

setup(
    name='semseg_density',
    version="0.0",
    ext_modules=cythonize(ext_core),
    install_requires=['hnswlib', 'sacred', 'gdown', 'numpy'],
    packages=['semseg_density', 'semseg_density.data', 'semseg_density.model', 'deeplab'])
