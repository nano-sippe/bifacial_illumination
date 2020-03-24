Copernicus
==========

The `Copernicus Project <https://www.copernicus.eu/>`_ is a collection of satelite derived climate and weather measurements funded by the European Union. Among many other things they provide radiation data for Europe, Africa and West-Asia from 2004 onwards in different time intervalls. The user guide of the solar radiation service can be found `here <https://atmosphere.copernicus.eu/sites/default/files/2019-01/CAMS72_2015SC3_D72.1.3.1_2018_UserGuide_v1_201812.pdf>`_

Copernicus Data is provided by serveral services, `Wekeo <https://www.wekeo.eu>`_ is one of them with free API support. Bifacial illumination includes a small tool to directly download radiation time series from wekeo, all you need to do is `register a free account <https://www.wekeo.eu/user/register>`_ and get an API key and secret.

When inilizing the API wrapper you can either specify key and secret explicitly or provide it (implicit) via enviromental variables:

.. code-block:: python

    #this will only work when WEKEO_KEY and WEKEO_SECRET envriomental variables
    #are set
    wrapper = CopernicusWrapper(latitude=30, longitude=10, start='2014-01-01',
                                end='2014-01-10')

    #otherwise you will have to provide both explicitly

    wrapper = CopernicusWrapper(latitude=30, longitude=10, start='2014-01-01',
                                end='2014-01-10', key='your_key', secret='your_secret')

    wrapper.download_data(filepath='berlin.nc')

By default the data is downlaoded in netcdf fromat and a 15 minute intervall.
