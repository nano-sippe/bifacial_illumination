Calculate yearly yield
======================

After downloading a radiation data set we create an illumination object, containg the radiation data, solar positions and some metadata. Then we can easyly run a simualtion.


.. ipython:: python
    :okwarning:
    :suppress:

    import bifacial_illumination as bi
    import os.path as path
    module_path = bi.__path__[0]
    example_path = path.join(module_path, '..', 'example_data', 'berlin_2014.nc')
    berlin_illumination = bi.Illumination(example_path,
                                          file_format='copernicus',
                                          tmy=False)
    berlin_illumination.df = berlin_illumination.df.query('zenith<90')


.. ipython::

    @verbatim
    In [5]: berlin_illumination = bi.Illumination('berlin_2014.nc',
       ...:                                   file_format='copernicus',
       ...:                                   tmy=False)

    In [6]: simulator = bi.YieldSimulator(berlin_illumination,
       ...:                               tmy_data=False)

    In [7]: yearly_yield = simulator.calculate_yield(spacing=6, tilt=35)
       ...: print('Yearly yield in kWh/year: {}'.format(yearly_yield))

By default the simulation assumes a constant efficiency of 20 % on the front side and 18 % at the back. For more details see the API documenttion.

