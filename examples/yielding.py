# -*- coding: utf-8 -*-

#import yaml
import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bifacial_geo as geo
import helper
from matplotlib import ticker
import matplotlib

from joblib import Parallel, delayed, Memory

plt.rcParams['font.size'] = 8.0
plt.rcParams['text.latex.preamble'] = r'\usepackage{arev}'
plt.rc('text', usetex=True)

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

def contour_helper(stats, axes, fig, pad=0.05, cmap='viridis', top=True):
    contribs = ['total','front','back']
    cbar_kws={"orientation": "horizontal", 'pad':pad}
    for i, contrib in enumerate(contribs):
        levels = np.linspace(np.floor(stats[contrib].min()/10)*10,
                             np.ceil(stats[contrib].max()/10)*10, 20)
        cs = axes[i].contourf(stats.index.unique('distance'),
            stats.index.unique('tilt'), stats.unstack('distance')[contrib],
            levels=levels, cmap=cmap)
        if contrib != 'back':
            line = stats[contrib].groupby(level='distance',axis=0).idxmax().apply(pd.Series).iloc[:,1]
            axes[i].plot(line, c='black')
        if i != 0:
            axes[i].yaxis.label.set_visible(False)
            axes[i].tick_params(left=False)
        if top==True:
            #axes[i].tick_params(bottom=False)
            axes[i].set_title(['bifacial','front','back'][i])
        else:
            cbar_kws['label']= 'annual radiation yield (kWh)'
            axes[i].set_xticklabels([])
            axes[i].tick_params(axis="x",direction="in")
            axes[i].xaxis.tick_top()

        cb = fig.colorbar(cs, ax=axes[i], **cbar_kws)
        #tick_locator = ticker.MaxNLocator(nbins=5)
        cb.locator = ticker.MaxNLocator(nbins=5)
        cb.update_ticks()
        cb.ax.set_xticklabels(cb.ax.get_xticklabels(), rotation=-45)

def plot_contourfs(stats_d, stats_s, filename=None):
    fig, (axes_d, axes_s) = \
        plt.subplots(2,3, dpi=100, figsize=(6.75, 4.5), sharey=True, sharex=False,
                     gridspec_kw={'height_ratios': [1, 1]})
    contour_helper(stats_d, axes_d, fig, pad =0.06, cmap='Reds_r', top=True)
    contour_helper(stats_s, axes_s, fig, pad=0.06, cmap='Blues_r', top=False)

    axes_d[0].set_ylabel('module tilt (deg)')
    axes_s[0].set_ylabel('module tilt (deg)')
    for i in range(3):
        axes_d[i].set_xlabel('module spacing (m)')
        axes_d[i].xaxis.set_label_position('top')
        axes_d[i].xaxis.tick_top()
        axes_d[i].tick_params(axis="x",direction="in")

    plt.tight_layout(rect=[0, 0, 0.98, 1])

    dallas = axes_d[-1].annotate(r'\textbf{Dallas}', xy=(0.965, 0.75),
                   xycoords='figure fraction',
                   fontsize=10)
    dallas.set_rotation(-90)

    seattle = axes_d[-1].annotate(r'\textbf{Seattle}', xy=(0.965, 0.35),
                    xycoords='figure fraction',
                    fontsize=10)
    seattle.set_rotation(-90)
    axes_d[0].annotate(r'\textbf{(a)}', xy=(0.01, 0.89),
                   xycoords='figure fraction',
                   fontsize=10)
    axes_s[0].annotate(r'\textbf{(b)}', xy=(0.01, 0.47),
                   xycoords='figure fraction',
                   fontsize=10)

    plt.savefig(filename, format='pdf', dpi=300)

def plot_heatmaps(stats, filename=None):
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, dpi=100, figsize=(6.75, 3.5), sharey=True)
    _, cb1 = sns.heatmap(stats['total'].unstack('distance'), ax=ax1,
                cbar_kws={'label': 'annual radiation yield (kWh)',
                          "orientation": "horizontal",
                          'pad':0.2})
    ax1.set_title('bifacial')
    _, cb2 = sns.heatmap(stats['front'].unstack('distance'), ax=ax2,
                cbar_kws={'label': 'annual radiation yield (kWh)',
                          "orientation": "horizontal",
                          'pad':0.2})
    ax2.set_title('front')
    ax2.tick_params(left=False)
    _, cb3 = sns.heatmap(stats['back'].unstack('distance'), ax=ax3,
                cbar_kws={'label': 'annual radiation yield (kWh)',
                          "orientation": "horizontal",
                          'pad':0.2})
    ax3.set_title('back')
    ax3.tick_params(left=False)

    cb1.ax.set_xticklabels(cb1.ax.get_xticklabels(), rotation=-45)
    cb2.ax.set_xticklabels(cb2.ax.get_xticklabels(), rotation=-45)
    cb3.ax.set_xticklabels(cb3.ax.get_xticklabels(), rotation=-45)

    ax1.set_ylabel('tilt (deg)')
    ax1.set_xlabel('module spacing (m)')
    ax2.yaxis.label.set_visible(False)
    ax2.set_xlabel('module spacing (m)')
    ax3.yaxis.label.set_visible(False)
    ax3.set_xlabel('module spacing (m)')
    plt.tight_layout()

    if filename:
        plt.savefig(filename, format='pdf', dpi=300)

    plt.show()

def analyse_yield(location='dallas', grid='fine', optimize='min'):
    if grid == 'fine':
        distance_scan = np.linspace(6,12,7).astype(int)
        tilt_scan = np.linspace(22,48,14).astype(int)
# =============================================================================
#         if location == 'dallas':
#             distance_scan = np.linspace(3,6,7)
#             tilt_scan = np.linspace(22,38,9)
#         if location == 'seattle':
#             distance_scan = np.linspace(6,12,7)
#             tilt_scan = np.linspace(26,48,12)
# =============================================================================

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
    #plot_heatmaps(stats, filename='rad_yield_{}.pdf'.format(location))
    return stats, scan_results

def create_donut(stats, location, distance=10, tilt=34):
    detailed_yield = stats.groupby(level='contribution', axis=1).mean()\
                       .groupby(['distance','tilt']).sum()/1000

    yearly_yield = stats.groupby(level='module_position', axis=1).sum()\
                     .min(axis=1)\
                     .groupby(['distance', 'tilt']).sum()\
                     .rename('total')

    detailed_yield = detailed_yield.div(detailed_yield.sum(axis=1), axis=0).multiply(yearly_yield, axis=0)

    data_back = detailed_yield.loc[(distance, tilt), detailed_yield.columns.str.contains('back')]
    data_front = detailed_yield.loc[(distance, tilt), detailed_yield.columns.str.contains('front')]
    data_front['back'] = data_back.sum()

    data_front['ground'] = data_front['front_ground_diffuse'] + data_front['front_ground_direct']
    data_front= data_front.drop(['front_ground_diffuse', 'front_ground_direct'])

    data_front = data_front.reindex(['front_sky_diffuse','back',
                                     'front_sky_direct', 'ground'])

    share_front = (data_front/(data_front.sum())*100).round(1)
    #labels_front = [r'front sky\\diffuse', 'back', r'front sky\\direct', 'front ground']
    labels_front = [r'$I_{\textrm{diff,f}}^{\textrm{sky}}$', 'back',
                    r'$I_{\textrm{dir,b}}^{\textrm{sky}}$', r'$I_{\textrm{f}}^{\textrm{gr}}$']

    share_back = (data_back/(data_back.sum())*100).round(1)
    #labels_back = ['ground diffuse', 'ground direct', 'sky diffuse', 'sky direct']
    labels_back = [r'$I_{\textrm{diff,b}}^{\textrm{gr}}$', r'$I_{\textrm{dir,b}}^{\textrm{gr}}$',
                   r'$I_{\textrm{diff,b}}^{\textrm{sky}}$', r'$I_{\textrm{dir,b}}^{\textrm{sky}}$']

    front_help = data_front.copy()
    front_help['front'] = front_help.drop('back').sum()
    front_help = front_help.loc[['front','back']]

    fig, (ax1, ax2) = plt.subplots(1,2, dpi=100, figsize=(6.75, 2.5))

# =============================================================================
#     wedges, texts = ax1.pie(front_help, radius=0.7, wedgeprops=dict(width=0.3),
#                                startangle=28, labels=['front',''], colors=['r','b'],
#                                labeldistance=0.6)
# =============================================================================

    if location == 'dallas':
        startangle1 = -104
        line1 = matplotlib.lines.Line2D((0.355,0.68),(0.605,0.847), transform=fig.transFigure,
                                        figure=fig, c='#999999', linewidth=1)
        line2 = matplotlib.lines.Line2D((0.350,0.68),(0.354,0.155), transform=fig.transFigure,
                                        figure=fig, c='#999999', linewidth=1)

    if location == 'seattle':
        startangle1 = -126
        line1 = matplotlib.lines.Line2D((0.36,0.68),(0.605,0.825), transform=fig.transFigure,
                                        figure=fig, c='#999999', linewidth=1)
        line2 = matplotlib.lines.Line2D((0.355,0.68),(0.35,0.177), transform=fig.transFigure,
                                        figure=fig, c='#999999', linewidth=1)

    labels_back = [labels_back[i] + ' ({}\%)'.format(share_back.iloc[i])
                    for i in range(len(labels_back))]
    labels_front = [labels_front[i] + ' ({}\%)'.format(share_front.iloc[i])
                    for i in range(len(labels_front))]

    wedges, texts = ax1.pie(data_front, radius=1, wedgeprops=dict(width=0.3),
                               startangle=startangle1, labels=labels_front, colors=['#1f77b4','#999999','#ff7f0e','#7E7468'])

    wedges, texts = ax2.pie(data_back, radius=1, wedgeprops=dict(width=0.3),
                               startangle=50, labels=labels_back, colors=['#396686','#c28249','#1f77b4','#ff7f0e'])
    plt.tight_layout(rect=[0.02, 0, 0.98, 1])



    if location == 'dallas':
        ax1.annotate(r'\textbf{total}', xy=(0.205, 0.48),
                       xycoords='figure fraction',
                       fontsize=10)

    if location == 'seattle':
         ax1.annotate(r'\textbf{total}', xy=(0.2345, 0.48),
                       xycoords='figure fraction',
                       fontsize=10)

    ax2.annotate(r'\textbf{back side}', xy=(0.6467, 0.48),
                   xycoords='figure fraction',
                   fontsize=10)

    #line2 = matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
    #                           transform=fig.transFigure)
# =============================================================================
#     fig.lines.append(line1)
#     fig.lines.append(line2)
# =============================================================================
    plt.savefig('{}_yield_contrib.pdf'.format(location), format='pdf', dpi=300)
    plt.show()

if __name__ == '__main__':
    stats_d, data_d = analyse_yield(location='dallas', grid='fine', optimize='min')
    stats_s, data_s = analyse_yield(location='seattle', grid='fine', optimize='min')
    #plot_contourfs(stats_d, stats_s, filename='radiation_yield.pdf')
    create_donut(data_d, location='dallas', distance=10, tilt=34,)
    create_donut(data_s, location='seattle', distance=10, tilt=42)