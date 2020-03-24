Optimize PV Array
=================

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
    :okwarning:

    @verbatim
    In [5]: berlin_illumination = bi.Illumination('berlin_2014.nc',
       ...:                                   file_format='copernicus',
       ...:                                   tmy=False)

    In [3]: optimizer = bi.CostOptimizer(berlin_illumination, invest_kwp = 1500,
       ...:                              price_per_m2_land = 5, tmy_data=False)

    In [6]: import skopt

    In [6]: print(skopt.__version__)

    In [6]: print(dir(skopt))

    In [6]: optimizer.optimize(ncalls=40);

    @savefig optimize.png width=4in
    In [5]: optimizer.plot_objective()

asdf