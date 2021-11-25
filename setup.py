from setuptools import setup

setup(
    name='semseg_density',
    version="0.0",
    install_requires=['hnswlib', 'sacred', 'gdown', 'numpy'],
    packages=['semseg_density', 'semseg_density.data', 'semseg_density.model'])
