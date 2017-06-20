from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='EVcharger',
    version='1.0',
    description='Simulation/optimisation model for EV charging coordination',
    long_description=readme,
    author='Fabian Neumann',
    author_email='s1674190@sms.ed.ac.uk',
    url='https://github.com/orley-enterprises/ev_chargingcoordination2017',
    license=license,
    packages=find_packages()
)