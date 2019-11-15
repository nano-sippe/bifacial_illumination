# -*- coding: utf-8 -*-

from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='bifacial_illumination',
      version='0.0.1',
      description='''This package can be used to simulate the irradiance of direct and diffuse light onto a bifacial solar cell.''',
      #long_description=readme(),
      url='https://github.com/nano-sippe/bifacial_illumination',
      author='Peter Tillmann, Klaus Jäger',
      author_email='Peter.tillmann@helmholtz-berlin.de',
      license='MIT',
      packages=['bifacial_illumination'],
      #setup_requires=['pytest-runner'],
      #test_require=['pytest'],
      install_requires=[
          'numpy'
      ],
      zip_safe=False)
