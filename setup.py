from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np

ext_core = Extension("metaseg_metrics",
                     sources=["semsegcluster/metaseg_metrics.pyx"],
                     include_dirs=[np.get_include()],
                     extra_compile_args=["-O3"])

setup(name='semsegcluster',
      version="0.0",
      ext_modules=cythonize(ext_core),
      install_requires=[
          'hnswlib', 'sacred', 'incense', 'gdown', 'numpy', 'scikit-learn',
          'scikit-optimize', 'markov_clustering', 'hdbscan', 'open3d',
          'torchmetrics', 'kornia', 'pymongo==3.12'
      ],
      packages=[
          'semsegcluster', 'semsegcluster.data', 'semsegcluster.model',
          'deeplab'
      ])
