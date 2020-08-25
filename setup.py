import os
from setuptools import setup, find_packages

with open(os.path.join('requirements.txt')) as f:
    required = f.read().splitlines()

setup(name='dragg-gym',
      license='MIT',
      version='0.0.1',
      install_requires=required,
      py_modules=['dragg_gym']
     )
