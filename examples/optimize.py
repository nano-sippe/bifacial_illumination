# -*- coding: utf-8 -*-



import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bifacial_geo as geo
import helper
from skopt import gp_minimize
from skopt import plots
from joblib import Parallel, delayed, Memory

plt.rcParams['font.size'] = 8.0
plt.rcParams['text.latex.preamble'] = r'\usepackage{arev}'
plt.rc('text', usetex=True)

def _run(df, land_cost, invest_kwp, bifacial=True):
    simulator = geo.CostOptimizer(
                    df,
                    price_per_m2_land=land_cost,
                    bifacial=bifacial,
                    invest_kwp = invest_kwp,
                    module_length=1.96)
    simulator.optimize()
    return simulator.res

class ParallelWrapper():
    def __init__(self, location, cache=True):
        self.location = location
        self.df = helper.load_dataframe(self.location)
        self.cache = cache

    def parallel_run(self, land_costs, invest_kwp, location, bifacial=True):
        if self.cache:
            cache_location = './cachedir'
            memory = Memory(cache_location, verbose=0)
            cached_run = memory.cache(_run)

        return Parallel(n_jobs=4)(delayed(cached_run)\
                       (self.df, land_cost, invest_kwp,  bifacial=bifacial)
                       for land_cost in land_costs)

class DebugWrapper():
    def __init__(self, location, cache=True):
        self.location = location
        self.df = helper.load_dataframe(self.location)
        self.cache = cache

    def parallel_run(self, land_costs, invest_kwp, location, bifacial=True):
        return [_run(self.df, land_cost, invest_kwp,  bifacial=bifacial)
                for land_cost in land_costs]

def process_results(results):
    df_dict = {
    'cost': [results[i].fun for i in range(len(results))],
    'dist': [results[i].x[0] for i in range(len(results))],
    'tilt': [results[i].x[1] for i in range(len(results))]
        }

    df = pd.DataFrame(df_dict, index=land_costs)
    df['yield'] = (invest_kwp*0.2*100 + 100*land_costs*df.dist/1.65)/(df.cost*25*0.2)
    df['land_cost'] = (100*land_costs*df.dist/1.65) / (df['yield']*25*0.2)
    df['invest'] = df['cost'] - df['land_cost']
    return df

def plot_objective(result, ax=None, level_min=5, level_max=9, axis_title_x=True, axis_title_y=True):
    space = result.space
    samples = np.asarray(result.x_iters)
    n_samples=250
    levels = np.linspace(level_min, level_max, 30)
    n_points=40
    rvs_transformed = space.transform(space.rvs(n_samples=n_samples))

    if ax is None:
        fig, ax = plt.subplots(dpi=150)
    #fig, ax = plt.subplots(figsize=(3.375, 2.25))
    xi, yi, zi = plots.partial_dependence(space, result.models[-1],
                                    1, 0,
                                    rvs_transformed, n_points)
    ax.set_xlim([2, 15])
    ax.set_ylim([1, 50])
    cs = ax.contourf(xi, yi, zi.clip(0,levels.max()), levels, cmap='viridis_r')#,vmin=5.5, vmax=8.5)
    #cs = ax.contourf(xi, yi, zi, levels, cmap='viridis_r')
    ax.scatter(samples[:, 0], samples[:, 1],
                     c='k', s=10, lw=0.)
    ax.scatter(result.x[0], result.x[1],
                     c=['r'], s=20, lw=0.)
    #ax.tick_params(bottom=axis_title_x, left=axis_title_y)
    ax.tick_params(left=axis_title_y)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", direction="in")
    ax.xaxis.set_label_position('top')
    if axis_title_x:
        ax.set_xlabel('module spacing (m)')
    else:
        ax.set_xticklabels([])
    if axis_title_y:
        ax.set_ylabel(r'module tilt (deg)')
    return cs

def plot_objective_quarted(results, location, fb):
    fig, axes = plt.subplots(2, 2, figsize=(6.75, 4.5), sharey=True, sharex=True)
    cs_list = []
    legend_list = [(False, True),
                   (False, False),
                   (True, True),
                   (True, False)]
    if location=='dallas':
        level_min = 4
        level_max = 7
    else:
        level_min = 5.5
        level_max = 8.5

    ticks = list(np.linspace(level_min, level_max, 7))
    for i, result in enumerate(results):
        #if i == 2:
        #    continue
        cs_list.append(plot_objective(result, axes[int(i/2)][i%2], level_min, level_max, *legend_list[i]))
        axes[int(i/2)][i%2].set_title(r'\textbf{{({0})}}'.format(chr(97+i)))

    #fig.subplots_adjust(right=0.80)
    cbar_ax = fig.add_axes([0.85, 0.2, 0.04, 0.6])
    cbar = fig.colorbar(cs_list[0], label = r'0.01 \$/kWh', cax=cbar_ax, ticks = ticks)
    #bar = fig.colorbar(cs_list[0], cax=cbar_ax, ticks = np.linspace(4,7,7))
    cbar.ax.set_yticklabels(ticks[:-1]  + ['> {}'.format(ticks[-1])])

    figname = 'opt_{}_{}.pdf'.format(location, fb)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(figname, format='pdf', dpi=300)

def plot_objective_quarted2(results, location, fb, land_costs):
    fig, axes = plt.subplots(1, 4, figsize=(6.75, 3), sharey=True, sharex=True)
    cs_list = []

    if location=='dallas':
        level_min = 4
        level_max = 7
    else:
        level_min = 5.5
        level_max = 8.5

    ticks = list(np.linspace(level_min, level_max, 7))
    for i, result in enumerate(results):
        axis_title_y = (i==0)
        cs_list.append(plot_objective(result, axes[i], level_min, level_max,
                       axis_title_y=axis_title_y))
        title = r'\textbf{{({0})}} c$_g$={1} \$/m$^2$'.format(chr(97+i),
                         land_costs[i])
        axes[i].set_title(title)

    #fig.subplots_adjust(bottom=0.30)
    cbar_ax = fig.add_axes([0.2, 0.15, 0.6, 0.04])

    #gs = fig.add_gridspec(2, 4)
    #cbar_ax = fig.add_subplot(245, gridspec_kw={"height_ratios":[1,1,0.1]})
    cbar = fig.colorbar(cs_list[0], label = r'0.01 \$/kWh', cax=cbar_ax,
                        ticks = ticks, orientation='horizontal')
    #bar = fig.colorbar(cs_list[0], cax=cbar_ax, ticks = np.linspace(4,7,7))
    cbar.ax.set_yticklabels(ticks[:-1]  + ['> {}'.format(ticks[-1])])

    figname = 'opt_{}_{}.pdf'.format(location, fb)
    plt.tight_layout(rect=[0, 0.2, 1, 1])
    plt.savefig(figname, format='pdf', dpi=300)

def plot_objective_helper(axes, res_arr, level_min, level_max,
                          axis_title_x=False):
    for i, result in enumerate(res_arr):
        axis_title_y = (i==0)
        last_cs = plot_objective(result, axes[i], level_min, level_max,
                       axis_title_y=axis_title_y, axis_title_x=axis_title_x)
    return last_cs

def plot_objective_oct(results, location, land_costs):
    fig, (axes_bi, axes_mono)  = plt.subplots(2, 4, figsize=(6.75, 4), sharey=True, sharex=False)

    if location=='dallas':
        level_min = 4
        level_max = 7
    else:
        level_min = 5.5
        level_max = 10.5

    ticks = list(np.linspace(level_min, level_max, 6))

    _ = plot_objective_helper(axes_bi, results[0], level_min, level_max, True)
    cs_last = plot_objective_helper(axes_mono, results[1], level_min, level_max, False)

    for i, ax in enumerate(axes_bi):
        title = r'c$_g$=\textbf{{{1} \$/m$^2$}}'.format(chr(97+i),
                         land_costs[i])
        ax.set_title(title)

    cbar_ax = fig.add_axes([0.2, 0.1, 0.6, 0.04])

    cbar = fig.colorbar(cs_last, label = r'\textbf{LCOE} (0.01 \$/kWh)', cax=cbar_ax,
                        ticks = ticks, orientation='horizontal')
    #bar = fig.colorbar(cs_list[0], cax=cbar_ax, ticks = np.linspace(4,7,7))
    cbar.ax.set_xticklabels(ticks[:-1]  + ['> {}'.format(ticks[-1])])
    mono = axes_bi[-1].annotate(r'\textbf{bifacial}', xy=(0.965, 0.73),
                   xycoords='figure fraction',
                   fontsize=10)
    mono.set_rotation(-90)

    bi = axes_mono[-1].annotate(r'\textbf{monofacial}', xy=(0.965, 0.41),
                    xycoords='figure fraction',
                    fontsize=10)
    bi.set_rotation(-90)

    figname = 'opt_{}.pdf'.format(location)
    plt.tight_layout(rect=[0, 0.14, 1.05, 1])

    axes_bi[0].annotate(r'\textbf{(a)}', xy=(0.01, 0.88),
                   xycoords='figure fraction',
                   fontsize=10)
    axes_mono[0].annotate(r'\textbf{(b)}', xy=(0.01, 0.50),
                   xycoords='figure fraction',
                   fontsize=10)

    plt.savefig(figname, format='pdf', dpi=300)

land_costs = np.array([1, 2.5, 5 ,10 ,20])
invest_kwp = 2000

location = 'dallas'

debug=False

if debug:
    wrapper = DebugWrapper(location)
else:
    wrapper = ParallelWrapper(location)

results_bifacial = wrapper.parallel_run(land_costs, invest_kwp, location, bifacial=True)
results_front = wrapper.parallel_run(land_costs, invest_kwp, location, bifacial=False)

stats_bifacial = process_results(results_bifacial)
stats_front = process_results(results_front)

#plot_objective(results_bifacial[2])

plot_objective_oct([results_bifacial[:-1], results_front[:-1]], location,
                   land_costs)

#plot_objective_quarted2(results_bifacial[:-1], location, 'bi', land_costs)

#plot_objective_quarted(results_bifacial[:-1], location, 'bi')
#plot_objective_quarted(results_front[:-1], location, 'front')

#stats_bifacial.to_csv('opt_stats_{}_bi.csv'.format(location))
#stats_front.to_csv('opt_stats_{}_front.csv'.format(location))

