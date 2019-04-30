# -*- coding: utf-8 -*-

from nsrdb_wrapper.wrapper import SpectralTMYWrapper
#import yaml
import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geo

config_path = os.path.abspath('config.yaml')

attr_dict = {'api_key':os.environ['NSRDB_API_KEY']}

# coordinates of Dallas
# =============================================================================
# latlong = (47.7, -122.26)
#
# #attr_dict['utc']='false'
# #attr_dict['mailing_list']='false'
#
# wrapper = SpectralTMYWrapper(latlong,
#                              request_attr_dict=attr_dict,
#                              config_path=config_path)
#
# wrapper.request_data()
# wrapper.df.dt = wrapper.df.dt.dt.tz_localize('utc')
# wrapper.add_zenith_azimuth()
#
# wrapper.df.to_hdf('tmy_spec_seattle.h5', 'table')
# =============================================================================

df = pd.read_hdf('tmy_spec_seattle.h5', 'table')

df = df.set_index('dt')

ghi_column_filter = df.columns.str.contains('GHI')
dni_column_filter = df.columns.str.contains('DNI')

df['ghi_total'] = df.loc[:,ghi_column_filter].sum(axis=1)*10
df['dni_total'] = df.loc[:,dni_column_filter].sum(axis=1)*10

df.zenith = 90-df.zenith

df['diffuse_total'] = df['ghi_total']-df['dni_total']*np.cos(df.zenith/180*np.pi)

# =============================================================================
# test_df = df.loc['20120801']
#
# test_df.ghi_total.plot()
# test_df.diffuse_total.plot()
#
# ax = test_df[['ghi_total', 'diffuse_total']].plot(figsize=(12,8))
# ax.set(ylabel='irradiance [W/m²]', xlabel='UTC time')
# plt.show()
#
# =============================================================================

std_paras = {
    'L': 1.650,# module length, standard is 1650 mm or 1960 mm
    'theta_m_deg': 52., # Angle of the module with respect to the ground (first guess optimal theta_m = latitude on Earth)
    'D': 3.000, # distance between modules
    'H': 0.500, # height of module base above ground
    'DNI': 1, # direct normal irradiance
    'DHI': 1, # diffuse horizontal irradiance
    'theta_S_deg': 35, # zenith of the Sun
    'phi_S_deg': 135, # azimuth of the Sun
    'albedo': 0.3, # albedo of the ground
    'x_array': np.linspace(-3,12,2001), # x-values for which the module and light ray functions are evaluated
    'ground_steps': 101, #number of steps into which irradiance on the ground is evaluated in the interval [0,D]
    'angle_step_deg': 0.1, # angular step at which diffuse illumination onto the ground is evaluated
    'module_steps': 6 # SET THIS NUMBER HIGHER FOR REAL EVALUATION! (SUGGESTION 20) number of lengths steps at which irradiance is evaluated on the module
}

simulation = geo.ModuleIllumination(std_paras)

# =============================================================================
# result = {
#         'front_ground_diffuse':[],
#         'back_ground_diffuse':[],
#         'front_ground_direct':[],
#         'back_ground_direct':[],
#         'front_sky_diffuse':[],
#         'back_sky_diffuse':[],
#         'front_direct':[],
#         'back_direct':[]
#         }
# =============================================================================

df = df.loc[df.zenith < 90]

simulation.theta_S_rad = geo.deg2rad(df.zenith)
simulation.phi_S_rad = geo.deg2rad(df.azimuth)

simulation.calc_irradiance_module_sky_direct()
simulation.calc_radiance_ground_direct()
simulation.calc_irradiance_module_ground_direct_from_matrix()

dni = df.loc[:,'dni_total']
dhi = df.loc[:,'diffuse_total']

result = {}
result['front_ground_diffuse'] = simulation.dict['irradiance_module_front_ground_diffuse_mean']*dhi
result['front_ground_direct'] = simulation.dict['irradiance_module_front_ground_direct_mean']*dni
result['back_ground_diffuse'] = simulation.dict['irradiance_module_back_ground_diffuse_mean']*dhi
result['back_ground_direct'] = simulation.dict['irradiance_module_back_ground_direct_mean']*dni
result['front_sky_diffuse'] = simulation.dict['irradiance_module_front_sky_diffuse_mean']*dhi
result['back_sky_diffuse'] = simulation.dict['irradiance_module_back_sky_diffuse_mean']*dhi
result['front_direct'] = simulation.dict['irradiance_module_front_sky_direct']*dni
result['back_direct'] = simulation.dict['irradiance_module_back_sky_direct']*dni

result = pd.DataFrame(result, index = df.index)

front_filter = result.columns.str.contains('front')
back_filter = result.columns.str.contains('back')

print('Back contribution: {} %'.format(
        result.loc[:, back_filter].sum().sum() /
        result.loc[:, front_filter].sum().sum()))

ax = (result.sum()/1000).plot.bar(figsize=(8,6))
ax.set_ylabel('Irrandiance (kWh/m²)')
plt.show()

(result.loc[:, back_filter].sum()/1000).plot.bar(figsize=(8,6))
ax.set_ylabel('Irrandiance (kWh/m²)')
plt.show()

result['total'] = result.sum(axis=1)

asdf

result.groupby(result.index.dayofyear).total.sum().plot()

df.groupby(df.index.dayofyear).ghi_total.sum().plot()

test3 = df.loc[df.index.dayofyear==180]

sns.heatmap(test3.groupby([test3.zenith.round(), test3.azimuth.round()]).size().unstack('azimuth'))


test = result.groupby([df.zenith.round(), df.azimuth.round()]).back_ground_direct.mean()

import seaborn as sns
from matplotlib.colors import LogNorm
log_norm = LogNorm(vmin=0.1, vmax=test.max())

sns.heatmap(test.unstack('azimuth').clip(0.01, test.max()))#, norm=log_norm)

test2 = result.groupby([df.zenith.round(), df.azimuth.round()]).size()

sns.heatmap(test2.unstack('azimuth'))

back_groud_direct_eff = pd.Series(simulation.dict['irradiance_module_front_ground_direct_mean'],
                                  index=df.index)

back_groud_direct_eff.groupby([df.zenith.round(), df.azimuth.round()]).mean().unstack('azimuth').to_excel('test2.xlsx')

plt.figure(figsize=(8,6))
sns.heatmap(back_groud_direct_eff.groupby([df.zenith.round(), df.azimuth.round()]).mean().unstack('azimuth'))


asdf

simulation.dict['irradiance_module_front_ground_direct'].shape
simulation.dict['irradiance_module_back_ground_direct']

test_1 = pd.Series(self.dict['irradiance_module_front_ground_direct_mean'], index=df.index)
test_2 = pd.Series(self.dict['irradiance_module_back_ground_direct_mean'], index=df.index)
test_4


test_1 = test_1 * df.dni_total
test_2 = test_2 * df.dni_total

test_1.loc['20120801'].plot()
test_2.loc['20120801'].plot()

for i in range(24):
    zenith = test_df.loc[:,'zenith'].iloc[i]
    azimuth = test_df.loc[:,'azimuth'].iloc[i]
    dni = test_df.loc[:,'dni_total'].iloc[i]
    dhi = test_df.loc[:,'diffuse_total'].iloc[i]
    simulation.update_zenith_azimuth(zenith, azimuth)

    result['front_ground_diffuse'].append(
            simulation.dict['irradiance_module_front_ground_diffuse'].mean()*dhi)
    result['front_ground_direct'].append(
            simulation.dict['irradiance_module_front_ground_direct'].mean()*dni)
    result['back_ground_diffuse'].append(
            simulation.dict['irradiance_module_back_ground_diffuse'].mean()*dhi)
    result['back_ground_direct'].append(
            simulation.dict['irradiance_module_back_ground_direct'].mean()*dni)
    result['front_sky_diffuse'].append(
            simulation.dict['irradiance_module_front_sky_diffuse'].mean()*dhi)
    result['back_sky_diffuse'].append(
            simulation.dict['irradiance_module_back_sky_diffuse'].mean()*dhi)
    result['front_direct'].append(
            simulation.dict['irradiance_module_front_sky_direct']*dni)
    result['back_direct'].append(
            simulation.dict['irradiance_module_back_sky_direct']*dni)

result = pd.DataFrame(result)
result.index = test_df.index

ax = result.plot(figsize=(8,8))
ax.set(ylabel='irradiance [W/m²]', xlabel='UTC time')
plt.show()
result.drop('front_direct', axis=1).plot(figsize=(8,8))
ax.set(ylabel='irradiance [W/m²]', xlabel='UTC time')
plt.show()

front_filter = result.columns.str.contains('front')
back_filter = result.columns.str.contains('back')

front_sum = result.loc[:, front_filter].sum().sum()
back_sum = result.loc[:, back_filter].sum().sum()


df.loc['20120801'].ghi_total.plot()
df.loc['20120801'].dni_total.plot()
df.loc['20120801'].zenith.plot()

(df['dni_total']*np.cos(df.zenith*180/np.pi)).loc['20120801'].plot()



df['month'] = df.dt.dt.month
df['day'] = df.dt.dt.day
test_plot = df.loc[df.dt.dt.hour==12].sort_values(['month','day'])
ghi_column_filter = test_plot.columns.str.contains('GHI')
test_plot.columns = test_plot
test_plot = test_plot.loc[:,ghi_column_filter]