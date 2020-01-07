.. Bifacial Illumination documentation master file, created by
   sphinx-quickstart on Mon Jan  6 17:54:26 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Bifacial Illumination's documentation!
=================================================

Bifacial Illumination is a simple tool to evaluate illumination on front and back side in solar arrays. It assumes endless dimensions of the PV field. It was designed for fast evaluation of timeseries and tmy data of solar radiation.

Installation
____________

For installation you need to clone the project from github, then you can install it with pip.

.. code-block:: console

    git clone https://github.com/nano-sippe/bifacial_illumination
    cd bifacial_illumination
    pip install bifacial_illumination

Getting Started
_______________

You can define the parameters of the PV field and the position of the sun to evaluate front and back illumination. Results are an 1d array (for different positions on the module) if a single sun position is modeled.

.. code-block:: python

    input_dict = dict(module_length=1.96, module_tilt=52,
                      mount_height=0.5, module_spacing=7.3,
                      dni=100, dhi=120, zenith_sun=60,
                      azimuth_sun=143.3, albedo=0.3)

    model = ModuleIllumination(**input_dict)
    fig = plt.subplot()
    fig.plot(model.results['irradiance_module_front'], label='front')
    fig.plot(model.results['irradiance_module_back'], label='back')
    fig.set_ylim(0, 250)
    fig.set_xlabel('module position')
    fig.set_ylabel('Irradiance (W/m2)')
    plt.legend(loc="right")

.. figure:: ./_static/irrad_example.png
    :align: center
    :figwidth: 400px



.. toctree::
    :maxdepth: 2
    :caption: Contents:

.. automodule:: geo

.. autoclass:: ModuleIllumination
    :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
