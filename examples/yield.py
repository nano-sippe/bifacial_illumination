# -*- coding: utf-8 -*-

from nsrdb_wrapper.wrapper import SpectralTMYWrapper
#import yaml
import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bifacial_geo as geo

config_path = os.path.abspath('config.yaml')

download_data = False

if download_data:
    attr_dict = {'api_key':os.environ['NSRDB_API_KEY']}

    # coordinates of Dallas
    latlong = (47.7, -122.26)

    #attr_dict['utc']='false'
    #attr_dict['mailing_list']='false'

    wrapper = SpectralTMYWrapper(latlong,
                                 request_attr_dict=attr_dict,
                                 config_path=config_path)

    wrapper.request_data()
    wrapper.df.dt = wrapper.df.dt.dt.tz_localize('utc')
    wrapper.add_zenith_azimuth()

    wrapper.df.to_hdf('tmy_spec_seattle.h5', 'table')

location = 'dallas'

if location == 'dallas':
    df = pd.read_hdf('tmy_spec_dallas.h5', 'table')
    df[['zenith','dt','azimuth']] = df[['zenith','dt','azimuth']].shift(-6)
    df = df.dropna(axis=0)

if location == 'seattle':
    df = pd.read_hdf('tmy_spec_seattle.h5', 'table')

df = df.set_index('dt')

ghi_column_filter = df.columns.str.contains('GHI')
dni_column_filter = df.columns.str.contains('DNI')

df['ghi_total'] = df.loc[:,ghi_column_filter].sum(axis=1)*10
df['dni_total'] = df.loc[:,dni_column_filter].sum(axis=1)*10

df.zenith = 90 - df.zenith

df['diffuse_total'] = df['ghi_total']-df['dni_total']*np.cos(df.zenith/180*np.pi)
df = df.loc[df.zenith < 90]

print('Diffuse/Direct ratio: {}'.format(df['diffuse_total'].sum()/(df['ghi_total']-df['diffuse_total']).sum()))

inputDict = {
        'L': 1.65,# module length, standard is 1650 mm or 1960 mm
        'theta_m_deg': 1, # Angle of the module with respect to the ground (first guess optimal theta_m = latitude on Earth)
        'D': 3.000, # distance between modules
        'H': 1, # height of module base above ground
        'DNI': 1, # direct normal irradiance
        'DHI': 1, # diffuse horizontal irradiance
        'theta_S_deg': 30, # zenith of the Sun
        'phi_S_deg': 150, # azimuth of the Sun
        'albedo': 0.3, # albedo of the ground
        'ground_steps': 101, #number of steps into which irradiance on the ground is evaluated in the interval [0,D]
        'module_steps': 12, # SET THIS NUMBER HIGHER FOR REAL EVALUATION! (SUGGESTION 20) number of lengths steps at which irradiance is evaluated on the module
        'angle_steps': 180 # Number at which angle discretization of ground light on module should be set
    }

inputDict['theta_S_deg'] = df.zenith
inputDict['phi_S_deg'] = df.azimuth

dni = df.loc[:,'dni_total']
dhi = df.loc[:,'diffuse_total']

grid = 'fine'
optimize = 'min'

if grid == 'fine':
    if location == 'dallas':
        distance_scan = np.linspace(3,6,7)
        tilt_scan = np.linspace(22,38,9)
    if location == 'seattle':
        #distance_scan = np.linspace(3,6,7)
        distance_scan = np.linspace(6,12,7)
        tilt_scan = np.linspace(26,48,12)

if grid=='coarse':
    distance_scan = np.linspace(3,6,4)
    tilt_scan = np.linspace(1,60,7)

scan_results = []
scan_result_all = []

for distance in distance_scan:
    for tilt in tilt_scan:
        inputDict['D'] = distance
        inputDict['theta_m_deg'] = tilt
        simulation = geo.ModuleIllumination(inputDict)
        print("distance: {} m, tilt: {} deg".format(distance, tilt))

        result = {}
        if optimize=='mean':
            result['front_ground_diffuse'] = simulation.results['irradiance_module_front_ground_diffuse_mean']*dhi
            result['front_ground_direct'] = simulation.results['irradiance_module_front_ground_direct_mean']*dni
            result['back_ground_diffuse'] = simulation.results['irradiance_module_back_ground_diffuse_mean']*dhi
            result['back_ground_direct'] = simulation.results['irradiance_module_back_ground_direct_mean']*dni
            #plt.plot(simulation.results['irradiance_module_back_ground_direct_mean'])
            #plt.show()
            result['front_sky_diffuse'] = simulation.results['irradiance_module_front_sky_diffuse_mean']*dhi
            result['back_sky_diffuse'] = simulation.results['irradiance_module_back_sky_diffuse_mean']*dhi
            result['front_direct'] = simulation.results['irradiance_module_front_sky_direct_mean']*dni
            result['back_direct'] = simulation.results['irradiance_module_back_sky_direct_mean']*dni

        if optimize=='min':
            result['front_ground_diffuse'] = simulation.results['irradiance_module_front_ground_diffuse'].min(axis=-1)*dhi
            result['front_ground_direct'] = simulation.results['irradiance_module_front_ground_direct'].min(axis=-1)*dni
            result['back_ground_diffuse'] = simulation.results['irradiance_module_back_ground_diffuse'].min(axis=-1)*dhi
            result['back_ground_direct'] = simulation.results['irradiance_module_back_ground_direct'].min(axis=-1)*dni
            #plt.plot(simulation.results['irradiance_module_back_ground_direct_mean'])
            #plt.show()
            result['front_sky_diffuse'] = simulation.results['irradiance_module_front_sky_diffuse'].min(axis=-1)*dhi
            result['back_sky_diffuse'] = simulation.results['irradiance_module_back_sky_diffuse'].min(axis=-1)*dhi
            result['front_direct'] = simulation.results['irradiance_module_front_sky_direct'].min(axis=-1)*dni
            result['back_direct'] = simulation.results['irradiance_module_back_sky_direct'].min(axis=-1)*dni

        result = pd.DataFrame(result, index = df.index)
        scan_result_all.append(result.sum())

        front_filter = result.columns.str.contains('front')
        back_filter = result.columns.str.contains('back')
        tmp = pd.Series({'distance':distance,
                   'tilt':tilt,
                   'back':result.loc[:,back_filter].sum().sum(),
                   'front':result.loc[:,front_filter].sum().sum(),
                   'front_direct':result['front_direct'].sum(),
                   'test':simulation.results['irradiance_module_front_sky_direct_mean'].sum()
                   })
        tmp['sum'] = tmp['front'] + tmp['back']
        scan_results.append(tmp)

df_yield = pd.concat(scan_result_all, axis=1).T
df_yield_all = pd.concat(scan_result_all, axis=1).T
df_yield = pd.concat(scan_results, axis=1).T

df_yield[['sum','front','back']] = df_yield[['sum','front','back']]/1000
df_yield['tilt'] = df_yield['tilt'].round().astype(int)

df_yield['front_back_ratio'] = df_yield['back'] / df_yield['front']

ax = df_yield.iloc[df_yield.groupby('distance')['sum'].idxmax()].set_index('distance')['tilt'].plot()#.plot(label='optimal tilt')
df_yield.iloc[df_yield.groupby('distance')['sum'].idxmax()].set_index('distance')['front_back_ratio'].plot(ax=ax, secondary_y=True)#, label='back contribution')
df_yield.iloc[df_yield.groupby('distance')['front'].idxmax()].set_index('distance')['tilt'].plot(ax=ax)
ax.legend([ax.get_lines()[0], ax.right_ax.get_lines()[0], ax.get_lines()[1]],
          ['optimal tilt total', 'back contribution', 'optimal tilt front'])
plt.show()

df_yield.groupby('distance')[['sum','front']].max().plot(title='yearly yield')
plt.show()

max_idx = df_yield['sum'].idxmax()
max_stuff = df_yield.iloc[max_idx]
df_yield.query('distance==3')[['sum','front','back']]

df['ghi_total'].sum()/1000
df['dni_total'].sum()/1000

sns.heatmap(df_yield.set_index(['distance','tilt'])['sum'].unstack('distance'))
plt.show()
sns.heatmap(df_yield.set_index(['distance','tilt'])['front'].unstack('distance'))
plt.show()
sns.heatmap(df_yield.set_index(['distance','tilt'])['back'].unstack('distance'))
plt.show()

# =============================================================================
# sns.heatmap(df_yield.set_index(['distance','tilt'])['test'].unstack('distance'))
# plt.show()
# =============================================================================

asdf

test = df_yield_all.front_ground_direct.to_frame()
test[['distance', 'tilt']] = df_yield[['distance', 'tilt']]
sns.heatmap(test.set_index(['distance','tilt']).unstack('distance'))

test = df_yield_all.back_ground_direct.to_frame()
test[['distance', 'tilt']] = df_yield[['distance', 'tilt']]
sns.heatmap(test.set_index(['distance','tilt']).unstack('distance'))

test = df_yield_all.back_direct.to_frame()
test[['distance', 'tilt']] = df_yield[['distance', 'tilt']]
sns.heatmap(test.set_index(['distance','tilt']).unstack('distance'))

test = df_yield_all.front_direct.to_frame()
test[['distance', 'tilt']] = df_yield[['distance', 'tilt']]
sns.heatmap(test.set_index(['distance','tilt']).unstack('distance'))

test = df_yield_all.front_direct.to_frame()
test[['distance', 'tilt']] = df_yield[['distance', 'tilt']]
sns.heatmap(test.set_index(['distance','tilt']).unstack('distance'))

asdf

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