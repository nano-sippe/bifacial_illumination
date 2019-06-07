# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# =============================================================================
# def download_data():
# config_path = os.path.abspath('config.yaml')
#     attr_dict = {'api_key':os.environ['NSRDB_API_KEY']}
#
#     # coordinates of Dallas
#     latlong = (47.7, -122.26)
#
#     #attr_dict['utc']='false'
#     #attr_dict['mailing_list']='false'
#
#     wrapper = SpectralTMYWrapper(latlong,
#                                  request_attr_dict=attr_dict,
#                                  config_path=config_path)
#
#     wrapper.request_data()
#     wrapper.df.dt = wrapper.df.dt.dt.tz_localize('utc')
#     wrapper.add_zenith_azimuth()
#
#     wrapper.df.to_hdf('tmy_spec_seattle.h5', 'table')
# =============================================================================

def load_dataframe(location):
    if location == 'dallas':
        df = pd.read_hdf('tmy_spec_dallas.h5', 'table')

        #correct for local time
        df[['zenith','dt','azimuth']] = df[['zenith','dt','azimuth']].shift(-6)
        df = df.dropna(axis=0)

    if location == 'seattle':
        df = pd.read_hdf('tmy_spec_seattle.h5', 'table')

    df = df.set_index('dt')

    ghi_column_filter = df.columns.str.contains('GHI')
    dni_column_filter = df.columns.str.contains('DNI')

    df['ghi_total'] = df.loc[:,ghi_column_filter].sum(axis=1)*10
    df['dni_total'] = df.loc[:,dni_column_filter].sum(axis=1)*10

    #converting to zenith angle
    df.zenith = 90 - df.zenith

    df['diffuse_total'] = df['ghi_total']-df['dni_total']*np.cos(df.zenith/180*np.pi)

    #removing datapoints where the sun is below the horizon
    df = df.loc[df.zenith < 90]
    return df