# -*- coding: utf-8 -*-

from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='bifacial',
      version='0.0.3',
      description='''This package can be used to simulate the irradiance of direct and diffuse light onto a bifacial solar cell.''',
      #long_description=readme(),
      url='https://github.com/P-Tillmann/bifacial',
      author='HZB Nano-Sippe Group',
      author_email='Peter.tillmann@helmholtz-berlin.de',
      license='MIT',
      packages=['bifacial_geo'],
      #setup_requires=['pytest-runner'],
      #test_require=['pytest'],
      install_requires=[
          'numpy',
      ],
      zip_safe=False)
