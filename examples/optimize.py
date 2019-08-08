# -*- coding: utf-8 -*-



import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bifacial_geo as geo
import helper
from skopt import gp_minimize
from joblib import Parallel, delayed, Memory
import os
from yielding import analyse_results_min_agg
import plots

plt.rcParams['font.size'] = 8.0
plt.rcParams['text.latex.preamble'] = r'\usepackage{arev}'
plt.rc('text', usetex=True)

def _run(df, land_cost, invest_kwp, bifacial=True, albedo=0.3):
    simulator = geo.CostOptimizer(
                    df,
                    price_per_m2_land=land_cost,
                    bifacial=bifacial,
                    invest_kwp = invest_kwp,
                    module_length=1.96,
                    albedo = albedo)
    simulator.optimize()
    return simulator.res

class ParallelWrapper():
    def __init__(self, location, cache=True):
        self.df = helper.load_dataframe(location)
        self.cache = cache

    def parallel_run(self, land_costs, invest_kwp, **kwargs):
        if self.cache:
            cache_location = './cachedir'
            memory = Memory(cache_location, verbose=0)
            cached_run = memory.cache(_run)

        return Parallel(n_jobs=4)(delayed(cached_run)\
                       (self.df, land_cost, invest_kwp, **kwargs)
                       for land_cost in land_costs)

class DebugWrapper():
    def __init__(self, location, cache=True):
        self.location = location
        self.df = helper.load_dataframe(self.location)
        self.cache = cache

    def parallel_run(self, land_costs, invest_kwp, **kwargs):
        return [_run(self.df, land_cost, invest_kwp, **kwargs)
                      for land_cost in land_costs]

def process_results(results, ts_irrad, bifacial, albedo):
    df_dict = {
    'cost': [results[i].fun for i in range(len(results))],
    'dist': [results[i].x[0] for i in range(len(results))],
    'tilt': [results[i].x[1] for i in range(len(results))]
        }
    df = pd.DataFrame(df_dict, index=land_costs)
    df_tmp = []
    for dist, tilt in zip(df['dist'], df['tilt']):
        simulator = geo.Simulator(ts_irrad, albedo=albedo, bifacial=bifacial)
        result = simulator.simulate(tilt=tilt, distance=dist)
        result['tilt'] = tilt
        result['distance'] = dist
        result = result.set_index(['tilt','distance'],append=True)/1000
        df_tmp.append(analyse_results_min_agg(result))

    df_tmp = pd.concat(df_tmp)
    df_tmp = df_tmp.reset_index(drop=True)
    df = pd.concat([df.reset_index(), df_tmp], axis=1)
    df = df.rename(columns={'index':'cost_scenario'})
    land_cost = (100*land_costs*df.dist)
    pp_cost = (0.2*2000*1.96)*100
    total_cost = land_cost+pp_cost

    df['land_cost'] = df['cost']*(land_cost/total_cost)
    df['pp_cost'] = df['cost']*(pp_cost/total_cost)
    return df

land_costs = np.array([1, 2.5, 5 ,10 ,20])
invest_kwp = 2000

location = 'seattle'

debug=False
if debug:
    wrapper = DebugWrapper(location)
else:
    wrapper = ParallelWrapper(location)

stats_list = []

for albedo in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    results_bifacial = wrapper.parallel_run(land_costs, invest_kwp, bifacial=True, albedo=albedo)
    results_front = wrapper.parallel_run(land_costs, invest_kwp, bifacial=False, albedo=albedo)

    #plots.plot_objective_oct([results_bifacial[:-1], results_front[:-1]], location,
    #                   land_costs)

    stats_bifacial = process_results(results_bifacial, ts_irrad = wrapper.df,
                                     bifacial=True, albedo=albedo)
    stats_bifacial['type'] = 'bi'
    stats_front = process_results(results_front,  ts_irrad = wrapper.df,
                                  bifacial=False, albedo=albedo)
    stats_front['type'] = 'mono'

    stats_tmp = pd.concat([stats_bifacial, stats_front])

    stats_tmp['albedo'] = albedo
    stats_tmp['location'] = location
    stats_list.append(stats_tmp)

stats = pd.concat(stats_list)
stats.to_csv('./stats/opt_stats_{}.csv'.format(location))
