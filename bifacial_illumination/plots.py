# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skopt import plots as sk_plots
from matplotlib import ticker
import os

def plot_objective(result, ax=None, level_min=5, level_max=9, axis_title_x=True, axis_title_y=True):
    space = result.space
    samples = np.asarray(result.x_iters)
    n_samples=250
    levels = np.linspace(level_min, level_max, 26)
    n_points=40
    rvs_transformed = space.transform(space.rvs(n_samples=n_samples))

    if ax is None:
        fig, ax = plt.subplots(dpi=100)
    #fig, ax = plt.subplots(figsize=(3.375, 2.25))
    xi, yi, zi = sk_plots.partial_dependence(space, result.models[-1],
                                    1, 0,
                                    rvs_transformed, n_points)
    ax.set_xlim([2, 14])
    ax.set_ylim([1, 50])
    cs = ax.contourf(xi, yi, zi.clip(0,levels.max()), levels, cmap='viridis_r')#,vmin=5.5, vmax=8.5)
    #cs = ax.contourf(xi, yi, zi, levels, cmap='viridis_r')
    ax.scatter(samples[:, 0], samples[:, 1],
                     c='k', s=10, lw=0.)
    ax.scatter(result.x[0], result.x[1],
                     c=['r'], s=20, lw=0.)
    #ax.tick_params(bottom=axis_title_x, left=axis_title_y)
    ax.tick_params(left=axis_title_y)
    #ax.xaxis.tick_top()
    ax.tick_params(axis="x", direction="in")
    ax.xaxis.set_label_position('bottom')
    if axis_title_x:
        ax.set_xlabel('module spacing (m)')
    else:
        ax.set_xticklabels([])
    if axis_title_y:
        ax.set_ylabel(r'module tilt (deg)')
    return cs

def plot_objective_helper(axes, res_arr, level_min, level_max,
                          axis_title_x=False, shadow=None):
    for i, result in enumerate(res_arr):
        axis_title_y = (i==0)
        last_cs = plot_objective(result, axes[i], level_min, level_max,
                       axis_title_y=axis_title_y, axis_title_x=axis_title_x)
        if shadow is not None:
            axes[i].plot(shadow, np.arange(1,50))

    return last_cs

def calc_shadow(theta=1, phi=1, location=None):
    #theta_S = [72, 62, 81, 72]
    #phi_S = [137, 180, 140, 180]
    if location=='dallas':
        theta, phi = 72, 137
    if location=='seattle':
        theta, phi = 81, 137

    theta_S_rad = np.deg2rad(theta)
    phi_S_rad = np.deg2rad(phi)

    n_S = np.array([np.sin(theta_S_rad)*np.cos(-phi_S_rad),
                             np.sin(theta_S_rad)*np.sin(-phi_S_rad),
                             np.cos(theta_S_rad)])

    h = np.sin(np.deg2rad(np.arange(1,50)))*1.96
    s1 = np.cos(np.deg2rad(np.arange(1,50)))*1.96

    s2 = h/n_S[2]/sum(n_S[[0,2]]**2)**0.5

    return s1+s2

def plot_objective_special(result):
    fig, ax = plt.subplots(figsize=(5, 4))
    level_min = 3.3
    level_max = 5.8
    shadow = calc_shadow(location='seattle')
    ticks = list(np.linspace(level_min, level_max, 6).astype(np.float32))

    cs = plot_objective(result, ax=ax, level_min=level_min, level_max=level_max)
    ax.plot(shadow, np.arange(1,50))
    cbar_ax = fig.add_axes([0.85, 0.12, 0.05, 0.84])

    cbar = fig.colorbar(cs, label = r'\textbf{LCOE} (0.01 \$/kWh)', cax=cbar_ax,
                        ticks = ticks, orientation='vertical')
    #bar = fig.colorbar(cs_list[0], cax=cbar_ax, ticks = np.linspace(4,7,7))
    cbar.ax.set_xticklabels(ticks[:-1]  + ['> {:.1f}'.format(ticks[-1])])
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plt.savefig(os.path.join('./figures/special_opt'), format='pdf', dpi=300)

def plot_objective_oct(results, location, land_costs):
    fig, (axes_bi, axes_mono)  = plt.subplots(2, 4, figsize=(6.75, 4), sharey=True, sharex=False)

    level_min = location.lcoe_min
    level_max = location.lcoe_max

    shadow = location.calc_shadow()
    ticks = list(np.linspace(level_min, level_max, 6).astype(np.float32))

    _ = plot_objective_helper(axes_bi, results[0], level_min, level_max, False, shadow=shadow)
    cs_last = plot_objective_helper(axes_mono, results[1], level_min, level_max, True, shadow=shadow)

    for i, ax in enumerate(axes_bi):
        title = r'c$_L$=\textbf{{{1} \$/m$^2$}}'.format(chr(97+i),
                         land_costs[i])
        ax.set_title(title)

    cbar_ax = fig.add_axes([0.2, 0.1, 0.6, 0.04])

    cbar = fig.colorbar(cs_last, label = r'\textbf{LCOE} (0.01 \$/kWh)', cax=cbar_ax,
                        ticks = ticks, orientation='horizontal')
    #bar = fig.colorbar(cs_list[0], cax=cbar_ax, ticks = np.linspace(4,7,7))
    cbar.ax.set_xticklabels(ticks[:-1]  + ['> {:.1f}'.format(ticks[-1])])
    plt.tight_layout(rect=[0, 0.14, 0.98, 1])
    mono = axes_bi[-1].annotate(r'\textbf{bifacial}', xy=(0.965, 0.82),
                   xycoords='figure fraction',
                   fontsize=10)
    mono.set_rotation(-90)

    bi = axes_mono[-1].annotate(r'\textbf{monofacial}', xy=(0.965, 0.49),
                    xycoords='figure fraction',
                    fontsize=10)
    bi.set_rotation(-90)

    figname = 'opt_{}.pdf'.format(location.name)

    axes_bi[0].annotate(r'\textbf{(a)}', xy=(0.01, 0.94),
                   xycoords='figure fraction',
                   fontsize=10)
    axes_mono[0].annotate(r'\textbf{(b)}', xy=(0.01, 0.59),
                   xycoords='figure fraction',
                   fontsize=10)

    plt.savefig(os.path.join('./figures', figname), format='pdf', dpi=300)

def contour_helper(stats, axes, fig, pad=0.05, cmap='viridis', top=True, location=None):
    contribs = ['total','front','back']
    contrib_label = {'total':'',#'annual radiant exposure (kWh)',
                     'front':'annual radiant exposure (kWh)',
                     'back':'',}#'bifacial gain (\%)'}
    cbar_kws={"orientation": "horizontal", 'pad':pad}
    #stats['back'] = stats['back'] / stats['front'] * 100
    for i, contrib in enumerate(contribs):
        if contrib == 'back':
            levels = np.linspace(np.floor(stats[contrib].min()),
                                 np.ceil(stats[contrib].max()), 20)
        else:
            levels = np.linspace(np.floor(stats[contrib].min()/10)*10,
                                 np.ceil(stats[contrib].max()/10)*10, 20)
        cs = axes[i].contourf(stats.index.unique('distance'),
            stats.index.unique('tilt'), stats.unstack('distance')[contrib],
            levels=levels, cmap=cmap)
        axes[i].set_xlim([4, 14])
# =============================================================================
#         if contrib != 'back':
#             line = stats[contrib].groupby(level='distance',axis=0).idxmax().apply(pd.Series).iloc[:,1]
#             axes[i].plot(line, c='black')
# =============================================================================
# =============================================================================
#         if contrib == 'total':
#             shadow = location.calc_shadow()
#             shadow = np.stack([shadow,np.arange(1,50)])
#             shadow = shadow[:,(shadow[0]>4)&(shadow[1]<=48)&(shadow[1]>=12)]
#             #axes[i].plot(shadow[0], shadow[1], c='black')
# =============================================================================

        if i != 0:
            axes[i].yaxis.label.set_visible(False)
            axes[i].tick_params(left=False)
        if top==True:
            #axes[i].tick_params(bottom=False)
            axes[i].set_title(['bifacial','front','back'][i])
        else:
            cbar_kws['label']= contrib_label[contrib]
            axes[i].set_xticklabels([])
            axes[i].tick_params(axis="x",direction="in")
            axes[i].xaxis.tick_top()

        cb = fig.colorbar(cs, ax=axes[i], **cbar_kws)
        #tick_locator = ticker.MaxNLocator(nbins=5)
        cb.locator = ticker.MaxNLocator(nbins=5)
        cb.update_ticks()
        cb.ax.set_xticklabels(cb.ax.get_xticklabels(), rotation=-45)

def plot_contourfs(stats_d, stats_s, filename=None, locations=(None, None)):
    fig, (axes_d, axes_s) = \
        plt.subplots(2,3, dpi=100, figsize=(6.75, 4.5), sharey=True, sharex=False,
                     gridspec_kw={'height_ratios': [1, 1]})
    contour_helper(stats_d, axes_d, fig, pad =0.06, cmap='Reds_r', top=True, location=locations[0])
    if locations[1].short_name == 'seattle':
        contour_helper(stats_s, axes_s, fig, pad=0.06, cmap='Blues_r', top=False, location=locations[1])
    else:
        contour_helper(stats_s, axes_s, fig, pad=0.06, cmap='Reds_r', top=False, location=locations[1])

    axes_d[0].set_ylabel('module tilt (deg)')
    axes_s[0].set_ylabel('module tilt (deg)')
    for i in range(3):
        axes_d[i].set_xlabel('module spacing (m)')
        axes_d[i].xaxis.set_label_position('top')
        axes_d[i].xaxis.tick_top()
        axes_d[i].tick_params(axis="x",direction="in")

    plt.tight_layout(rect=[0, 0, 0.98, 1])

    dallas = axes_d[-1].annotate(r'\textbf{{{}}}'.format(locations[0].name), xy=(0.965, 0.75),
                   xycoords='figure fraction',
                   fontsize=10)
    dallas.set_rotation(-90)

    seattle = axes_d[-1].annotate(r'\textbf{{{}}}'.format(locations[1].name), xy=(0.965, 0.35),
                    xycoords='figure fraction',
                    fontsize=10)
    seattle.set_rotation(-90)
    axes_d[0].annotate(r'\textbf{(a)}', xy=(0.01, 0.89),
                   xycoords='figure fraction',
                   fontsize=10)
    axes_s[0].annotate(r'\textbf{(b)}', xy=(0.01, 0.47),
                   xycoords='figure fraction',
                   fontsize=10)
    fig.align_labels()
    plt.savefig(filename, format='pdf', dpi=300)