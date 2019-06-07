# -*- coding: utf-8 -*-

#import yaml
import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bifacial_geo as geo
import helper

from joblib import Parallel, delayed, Memory

def _run(df, distance, tilt, bifacial=True):
    simulator = geo.Simulator(df, bifacial=bifacial)
    return simulator.simulate(distance, tilt)

class ParallelWrapper():
    def __init__(self, location, cache=True):
        self.location = location
        self.df = helper.load_dataframe(self.location)
        self.cache = cache

    def parallel_run(self, distance_scan, tilt_scan, bifacial=True):
        if self.cache:
            cache_location = './cachedir'
            memory = Memory(cache_location, verbose=0)
            cached_run = memory.cache(_run)

        return Parallel(n_jobs=4)(delayed(cached_run)\
                       (self.df, distance, tilt,  bifacial=True)
                       for distance in distance_scan
                       for tilt in tilt_scan)

def analyse_results_mean_agg(results):
    detailed_yield = results.groupby(level='contribution', axis=1).mean()\
                       .groupby(['distance','tilt']).sum()/1000

    direction = ['front' if 'front' in name else 'back'
                 for name in detailed_yield.columns]
    yearly_yield = detailed_yield.groupby(direction, axis=1).sum()
    yearly_yield['total'] = yearly_yield['front'] + yearly_yield['back']
    return yearly_yield

def analyse_results_min_agg(results):
    detailed_yield = results.groupby(level='contribution', axis=1).mean()\
                       .groupby(['distance','tilt']).sum()/1000

    direction = ['front' if 'front' in name else 'back'
                 for name in detailed_yield.columns]

    front_share = detailed_yield.groupby(direction, axis=1).sum()\
                                .eval('front/(front+back)')

    yearly_yield = results.groupby(level='module_position', axis=1).sum()\
                         .min(axis=1)\
                         .groupby(['distance', 'tilt']).sum()\
                         .rename('total').to_frame()
    yearly_yield['front'] = yearly_yield['total']*front_share
    yearly_yield['back'] = yearly_yield['total']-yearly_yield['front']
    return yearly_yield

def plot_heatmaps(stats):
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, dpi=100, figsize=(14,4))
    sns.heatmap(stats['total'].unstack('distance'), ax=ax1).set_title('total')
    sns.heatmap(stats['front'].unstack('distance'), ax=ax2).set_title('front')
    sns.heatmap(stats['back'].unstack('distance'), ax=ax3).set_title('back')
    plt.show()

def analyse_yield(location='dallas', grid='fine', optimize='min'):
    if grid == 'fine':
        if location == 'dallas':
            distance_scan = np.linspace(3,6,7)
            tilt_scan = np.linspace(22,38,9)
        if location == 'seattle':
            distance_scan = np.linspace(6,12,7)
            tilt_scan = np.linspace(26,48,12)

    if grid=='coarse':
        distance_scan = np.linspace(3,6,4)
        tilt_scan = np.linspace(1,60,7)

    wrapper = ParallelWrapper(location)

    scan_results = wrapper.parallel_run(distance_scan, tilt_scan)

    mindex = pd.MultiIndex.from_product([distance_scan, tilt_scan],
                                    names=['distance', 'tilt'])

    scan_results = pd.concat(scan_results, keys=mindex, names=['distance', 'tilt'])

    stats = analyse_results_min_agg(scan_results)
    stats = stats/1000 #convert to kWh
    plot_heatmaps(stats)

analyse_yield(location='dallas', grid='fine', optimize='min')




# =============================================================================
# fig, (ax1, ax2, ax3) = plt.subplots(1,3, dpi=100, figsize=(14,4))
# sns.heatmap(stats['total'].unstack('distance'), ax=ax1).set_title('total')
# sns.heatmap(stats['front'].unstack('distance'), ax=ax2).set_title('front')
# sns.heatmap(stats['back'].unstack('distance'), ax=ax3).set_title('back')
#
# asdf
#
# for distance in distance_scan:
#     for tilt in tilt_scan:
#
#         scan_results.append(tmp)
#
# df_yield = pd.concat(scan_result_all, axis=1).T
# df_yield_all = pd.concat(scan_result_all, axis=1).T
# df_yield = pd.concat(scan_results, axis=1).T
#
# df_yield[['sum','front','back']] = df_yield[['sum','front','back']]/1000
# df_yield['tilt'] = df_yield['tilt'].round().astype(int)
#
# df_yield['front_back_ratio'] = df_yield['back'] / df_yield['front']
#
# ax = df_yield.iloc[df_yield.groupby('distance')['sum'].idxmax()].set_index('distance')['tilt'].plot()#.plot(label='optimal tilt')
# df_yield.iloc[df_yield.groupby('distance')['sum'].idxmax()].set_index('distance')['front_back_ratio'].plot(ax=ax, secondary_y=True)#, label='back contribution')
# df_yield.iloc[df_yield.groupby('distance')['front'].idxmax()].set_index('distance')['tilt'].plot(ax=ax)
# ax.legend([ax.get_lines()[0], ax.right_ax.get_lines()[0], ax.get_lines()[1]],
#           ['optimal tilt total', 'back contribution', 'optimal tilt front'])
# plt.show()
#
# df_yield.groupby('distance')[['sum','front']].max().plot(title='yearly yield')
# plt.show()
#
# max_idx = df_yield['sum'].idxmax()
# max_stuff = df_yield.iloc[max_idx]
# df_yield.query('distance==3')[['sum','front','back']]
#
# df['ghi_total'].sum()/1000
# df['dni_total'].sum()/1000
#
# sns.heatmap(df_yield.set_index(['distance','tilt'])['sum'].unstack('distance'))
# plt.show()
# sns.heatmap(df_yield.set_index(['distance','tilt'])['front'].unstack('distance'))
# plt.show()
# sns.heatmap(df_yield.set_index(['distance','tilt'])['back'].unstack('distance'))
# plt.show()
#
# # =============================================================================
# # sns.heatmap(df_yield.set_index(['distance','tilt'])['test'].unstack('distance'))
# # plt.show()
# # =============================================================================
#
# asdf
#
# test = df_yield_all.front_ground_direct.to_frame()
# test[['distance', 'tilt']] = df_yield[['distance', 'tilt']]
# sns.heatmap(test.set_index(['distance','tilt']).unstack('distance'))
#
# test = df_yield_all.back_ground_direct.to_frame()
# test[['distance', 'tilt']] = df_yield[['distance', 'tilt']]
# sns.heatmap(test.set_index(['distance','tilt']).unstack('distance'))
#
# test = df_yield_all.back_direct.to_frame()
# test[['distance', 'tilt']] = df_yield[['distance', 'tilt']]
# sns.heatmap(test.set_index(['distance','tilt']).unstack('distance'))
#
# test = df_yield_all.front_direct.to_frame()
# test[['distance', 'tilt']] = df_yield[['distance', 'tilt']]
# sns.heatmap(test.set_index(['distance','tilt']).unstack('distance'))
#
# test = df_yield_all.front_direct.to_frame()
# test[['distance', 'tilt']] = df_yield[['distance', 'tilt']]
# sns.heatmap(test.set_index(['distance','tilt']).unstack('distance'))
#
# asdf
#
# print('Back contribution: {} %'.format(
#         result.loc[:, back_filter].sum().sum() /
#         result.loc[:, front_filter].sum().sum()))
#
# ax = (result.sum()/1000).plot.bar(figsize=(8,6))
# ax.set_ylabel('Irrandiance (kWh/m²)')
# plt.show()
#
# (result.loc[:, back_filter].sum()/1000).plot.bar(figsize=(8,6))
# ax.set_ylabel('Irrandiance (kWh/m²)')
# plt.show()
#
# result['total'] = result.sum(axis=1)
#
# asdf
#
# result.groupby(result.index.dayofyear).total.sum().plot()
#
# df.groupby(df.index.dayofyear).ghi_total.sum().plot()
#
# test3 = df.loc[df.index.dayofyear==180]
#
# sns.heatmap(test3.groupby([test3.zenith.round(), test3.azimuth.round()]).size().unstack('azimuth'))
#
#
# test = result.groupby([df.zenith.round(), df.azimuth.round()]).back_ground_direct.mean()
#
# import seaborn as sns
# from matplotlib.colors import LogNorm
# log_norm = LogNorm(vmin=0.1, vmax=test.max())
#
# sns.heatmap(test.unstack('azimuth').clip(0.01, test.max()))#, norm=log_norm)
#
# test2 = result.groupby([df.zenith.round(), df.azimuth.round()]).size()
#
# sns.heatmap(test2.unstack('azimuth'))
#
# back_groud_direct_eff = pd.Series(simulation.dict['irradiance_module_front_ground_direct_mean'],
#                                   index=df.index)
#
# back_groud_direct_eff.groupby([df.zenith.round(), df.azimuth.round()]).mean().unstack('azimuth').to_excel('test2.xlsx')
#
# plt.figure(figsize=(8,6))
# sns.heatmap(back_groud_direct_eff.groupby([df.zenith.round(), df.azimuth.round()]).mean().unstack('azimuth'))
#
#
# asdf
#
# simulation.dict['irradiance_module_front_ground_direct'].shape
# simulation.dict['irradiance_module_back_ground_direct']
#
# test_1 = pd.Series(self.dict['irradiance_module_front_ground_direct_mean'], index=df.index)
# test_2 = pd.Series(self.dict['irradiance_module_back_ground_direct_mean'], index=df.index)
# test_4
#
#
# test_1 = test_1 * df.dni_total
# test_2 = test_2 * df.dni_total
#
# test_1.loc['20120801'].plot()
# test_2.loc['20120801'].plot()
#
# for i in range(24):
#     zenith = test_df.loc[:,'zenith'].iloc[i]
#     azimuth = test_df.loc[:,'azimuth'].iloc[i]
#     dni = test_df.loc[:,'dni_total'].iloc[i]
#     dhi = test_df.loc[:,'diffuse_total'].iloc[i]
#     simulation.update_zenith_azimuth(zenith, azimuth)
#
#     result['front_ground_diffuse'].append(
#             simulation.dict['irradiance_module_front_ground_diffuse'].mean()*dhi)
#     result['front_ground_direct'].append(
#             simulation.dict['irradiance_module_front_ground_direct'].mean()*dni)
#     result['back_ground_diffuse'].append(
#             simulation.dict['irradiance_module_back_ground_diffuse'].mean()*dhi)
#     result['back_ground_direct'].append(
#             simulation.dict['irradiance_module_back_ground_direct'].mean()*dni)
#     result['front_sky_diffuse'].append(
#             simulation.dict['irradiance_module_front_sky_diffuse'].mean()*dhi)
#     result['back_sky_diffuse'].append(
#             simulation.dict['irradiance_module_back_sky_diffuse'].mean()*dhi)
#     result['front_direct'].append(
#             simulation.dict['irradiance_module_front_sky_direct']*dni)
#     result['back_direct'].append(
#             simulation.dict['irradiance_module_back_sky_direct']*dni)
#
# result = pd.DataFrame(result)
# result.index = test_df.index
#
# ax = result.plot(figsize=(8,8))
# ax.set(ylabel='irradiance [W/m²]', xlabel='UTC time')
# plt.show()
# result.drop('front_direct', axis=1).plot(figsize=(8,8))
# ax.set(ylabel='irradiance [W/m²]', xlabel='UTC time')
# plt.show()
#
# front_filter = result.columns.str.contains('front')
# back_filter = result.columns.str.contains('back')
#
# front_sum = result.loc[:, front_filter].sum().sum()
# back_sum = result.loc[:, back_filter].sum().sum()
#
#
# df.loc['20120801'].ghi_total.plot()
# df.loc['20120801'].dni_total.plot()
# df.loc['20120801'].zenith.plot()
#
# (df['dni_total']*np.cos(df.zenith*180/np.pi)).loc['20120801'].plot()
#
#
#
# df['month'] = df.dt.dt.month
# df['day'] = df.dt.dt.day
# test_plot = df.loc[df.dt.dt.hour==12].sort_values(['month','day'])
# ghi_column_filter = test_plot.columns.str.contains('GHI')
# test_plot.columns = test_plot
# test_plot = test_plot.loc[:,ghi_column_filter]
# =============================================================================
