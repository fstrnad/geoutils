"""setup.py for climnet."""

from setuptools import find_packages
from setuptools import setup

setup(
    name='geoutils',
    version='1.0',
    description=('Library for processing and plotting of climate data.'),
    author='Felix Strnad',
    author_email='felix.strnad@uni-tuebingen.de',
    packages=find_packages()
)
