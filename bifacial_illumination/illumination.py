# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pvlib


class Illumination():
    def __init__(self, filepath, file_format='nsrdb', use_perez=False, tmy=True):
        '''
        Read irradiance data from different formats and adds solar position.
        Instances of this class can be directly used for yield simulation and
        optimization.
        Parameters
        ----------
        filepath : filepath
            Filepath to radiation data

        file_format : str
            Currently supported are csv data from the nsrdb and netcdf data from
            copernicus.

        use_perez : boolean
            Determines wether the perez model for circumsolar and horizontal
            brightening should be used.

        tmy : boolean
            Specifies if the provided data represents typical meteorological year
            (True) or a continuous time series.
        '''
        if file_format == 'nsrdb':
            df = pd.read_csv(filepath, skiprows=2)
            df['dt'] = pd.to_datetime(df.Year.astype(str) +
                                       df.Month.apply('{:0>2}'.format) +
                                       df.Day.apply('{:0>2}'.format) +
                                       df.Hour.apply('{:0>2}'.format) +
                                       df.Minute.apply('{:0>2}'.format),
                                       format='%Y%m%d%H%M', utc=True)
            df = df[['dt','GHI','DNI','DHI']]
            meta = pd.read_csv(filepath, nrows=1).iloc[0]
            self.lat, self.long = meta[['Latitude','Longitude']]
            df = df.set_index('dt')

        elif file_format == 'copernicus':
            import netCDF4
            ds = netCDF4.Dataset(filepath)
            df = pd.DataFrame({
                    'GHI':ds['GHI'][:,0,0,0],
                    'DHI':ds['DHI'][:,0,0,0],
                    'DNI':ds['BNI'][:,0,0,0]})
            df['dt'] = pd.to_datetime(ds['time'][:], utc=True, unit='s')
            sampling = (df['dt'] - df['dt'].shift(periods=1)).value_counts().index[0]
            df['dt'] = df['dt'] - sampling/2
            df = df.set_index('dt')
            df = df*(pd.Timedelta('1H') / sampling)
            self.lat, self.long = ds['latitude'][0], ds['longitude'][0]

        elif file_format == 'native':
            with pd.HDFStore(filepath) as store:
                self.df = store['table']
                self.lat = store.root._v_attrs.latitude
                self.long = store.root._v_attrs.longitude
            return None

        solar_position = pvlib.solarposition.get_solarposition(df.index,
                                                               self.lat,
                                                               self.long)
        self.df = pd.concat([df, solar_position[['zenith','azimuth']]], axis=1)

        if use_perez:
            raise NotImplementedError('Not yet implemented')

    def save_as_native(self, filepath):
        with pd.HDFStore(filepath) as store:
            store['table'] = self.df
            store.root._v_attrs.latitude = self.lat
            store.root._v_attrs.longitude = self.long


if __name__ == '__main__':
    filepath = '701624_32.77_-96.82_tmy-2018.csv'
    test = Illumination(filepath)
    test.save_as_native('native_test.h5')
    test2 = Illumination('native_test.h5', file_format='native')
