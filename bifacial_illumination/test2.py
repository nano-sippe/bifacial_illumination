# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import plots
import matplotlib.pyplot as plt

asdf



plt.rcParams['font.size'] = 8.0
plt.rcParams['text.latex.preamble'] = r'\usepackage{arev}'
plt.rc('text', usetex=True)




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

import bifacial_illumination as bi

berlin_illumination = bi.Illumination('berlin_2014.nc',
                                      file_format='copernicus',
                                      tmy=False)

simulator = bi.YieldSimulator(berlin_illumination,
                              tmy_data=False)
#spacing is specified in m, tilt in deg
yearly_yield = simulator.calculate_yield(spacing=6, tilt=35)
print(yearly_yield)


optimizer = bi.CostOptimizer(berlin_illumination, invest_kwp = 1500,
                             price_per_m2_land = 5, tmy_data=False)

optimizer.optimize(ncalls=40)
optimizer.plot_objective()



asdf
if not os.path.exists('berlin_native.h5'):
    illumination = bi.Illumination('irradiation-518f75e4-fa61-11e9-bbc8-b8ca3a6792fc.nc', file_format='copernicus')
    illumination.save_as_native('berlin_native.h5')
else:
    illumination = bi.Illumination('berlin_native.h5', file_format='native')

simulator = bi.YieldSimulator(illumination, tmy_data=False)
self = simulator
#print(simulator.calculate_yield(6, 20))
optimizer = bi.CostOptimizer(illumination, invest_kwp = 1500,
                             price_per_m2_land = 5, tmy_data=False)

optimizer = bi.CostOptimizer(berlin_illumination, invest_kwp = 1500,
                             price_per_m2_land = 5, tmy_data=False)

optimizer.optimize(ncalls=40)

df_results = optimizer.simulate(7.2, 33.9)
df_results['distance'] = 7.2
df_results['tilt'] = 33.9

df_results = df_results.set_index(['distance', 'tilt'], append=True)

res = optimizer.res
level_min = np.floor(res.fun*10)/10
level_max = level_min+2.5

fig, ax = plt.subplots(dpi=100)

cs = plots.plot_objective(optimizer.res, level_min=level_min, level_max=level_max, ax=ax)

ticks = list(np.linspace(level_min, level_max, 6).astype(np.float32))

cbar_ax = fig.add_axes([0.2, 0.1, 0.6, 0.04])
cbar = fig.colorbar(cs, label = r'\textbf{LCOE} (0.01 \$/kWh)', cax=cbar_ax,
                    ticks = ticks, orientation='horizontal')

cbar.ax.set_xticklabels(ticks[:-1]  + ['> {:.1f}'.format(ticks[-1])])
plt.tight_layout(rect=[0, 0.14, 0.98, 1])


optimizer_mono = bi.CostOptimizer(illumination, invest_kwp = 1500,
                             price_per_m2_land = 5, tmy_data=False, bifacial=False)

optimizer_mono.optimize(tilt_max=60)

res_mono = optimizer_mono.res
level_min = np.floor(res_mono.fun*10)/10
level_max = level_min+2.5

fig, ax = plt.subplots(dpi=100)

cs = plots.plot_objective(optimizer_mono.res, level_min=level_min, level_max=level_max, ax=ax)

ticks = list(np.linspace(level_min, level_max, 6).astype(np.float32))

cbar_ax = fig.add_axes([0.2, 0.1, 0.6, 0.04])
cbar = fig.colorbar(cs, label = r'\textbf{LCOE} (0.01 \$/kWh)', cax=cbar_ax,
                    ticks = ticks, orientation='horizontal')

cbar.ax.set_xticklabels(ticks[:-1]  + ['> {:.1f}'.format(ticks[-1])])
plt.tight_layout(rect=[0, 0.14, 0.98, 1])