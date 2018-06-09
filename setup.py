#! /usr/bin/env python

from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='EVChargingCoordination',
    version='1.0',
    description='Simulation/optimisation model for EV charging coordination',
    long_description=readme,
    author='Fabian Neumann',
    author_email='fabian.neumann@outlook.de',
    url='https://github.com/fneum/ev_chargingcoordination2017',
    license=license,
    packages=find_packages()
)
