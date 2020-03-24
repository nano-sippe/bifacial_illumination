# -*- coding: utf-8 -*-

from bifacial_illumination import ModuleIllumination
import bifacial_illumination as bi
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import time
import pandas as pd
import datetime
import pysolar.solar as sol
import pickle
import seaborn as sns
import io_

from scipy import interpolate

a = pd.DataFrame({'a':np.random.random(20)*4,'b':np.random.random(20)*4, 'c':np.random.random(20)})
b = pd.DataFrame({'a':np.random.random(20)*4,'b':np.random.random(20)*4, 'd':np.random.random(20)})

def merge_asof_2d(a, b, on_1, on_2, direction='nearest'):
    b['merge_idx_1'] = b.groupby(on_1).ngroup()
    b['merge_idx_2'] = b.groupby(on_2).ngroup()

    a = a.sort_values(on_1)
    b = b.sort_values(on_1)
    a = pd.merge_asof(a, b[[on_1, 'merge_idx_1']], on=on_1)
    a = a.sort_values(on_2)
    b = b.sort_values(on_2)
    a = pd.merge_asof(a, b[[on_2, 'merge_idx_2']], on=on_2)

    b = b.drop([on_1, on_2], axis=1)

    a = a.merge(b, on=['merge_idx_1', 'merge_idx_2'])

    a = a.drop(['merge_idx_1', 'merge_idx_2'], axis=1)
    return a


def plot_polar_heatmap(df, filepath=None):

    rad = df.index#np.sin(np.deg2rad(df.index))
    azi = np.deg2rad(df.columns+180)

    r, th = np.meshgrid(rad, azi)

    fig = plt.figure()
    plt.subplot(projection="polar")

    im = plt.pcolormesh(th, r, df.T)
    #plt.pcolormesh(th, z, r)

    plt.plot(azi, r, color='k', ls='none')
    ax1 = plt.gca()
    #ax1.set_yticks(np.linspace(20,80,4))
    #import matplotlib as mpl
    #mpl.rcParams['ytick.color'] = 'white'
    plt.yticks(ticks=np.linspace(40,60,2),labels = map(lambda x: str(int(x)) + '°', np.linspace(30,60,2)))

    cax = fig.add_axes([0.3, 0, 0.4, 0.05])

    #cax = divider.append_axes('bottom', size='80%', pad=0.6)

    plt.colorbar(im, orientation='horizontal', cax=cax)
    cax.set_xlabel(r'Time integrated radiance $\left(\frac{\mathrm{kWh}}{\mathrm{year} \cdot \mathrm{sr}\cdot \mathrm{m}^2}\right)$')

    ax1.grid()#ax1.grid(axis='y')

    #plt.grid()

    if filepath is not None:
        plt.savefig(filepath, format='svg')

    plt.show()
    #mpl.rcParams['ytick.color'] = 'black'

def calculate_ground_back_with_aoi_loss(geo_instance, angular_loss, join_phi=False):
    df = calculate_angular_dist_ground(geo_instance, matrix_name, illumination_name)
    df = df.sort_values('theta')

    beta_dist = np.sin(np.linspace(0,np.pi,180))**2 /\
                (np.sin(np.linspace(0,np.pi,180))**2).sum()
    alpha_dist = np.ones(geo_instance.angle_steps)
    mindex = pd.MultiIndex.from_product([np.linspace(-np.pi/2, np.pi/2, geo_instance.angle_steps),
                                         np.linspace(0,np.pi,180)],
        names=['alpha', 'beta'])

    dist_2d = np.multiply.outer(alpha_dist, beta_dist)

    loss_matrix = pd.DataFrame({'int': dist_2d.flatten()}, index=mindex)
    loss_matrix = loss_matrix.reset_index()


    loss_matrix['theta'] = np.arccos(np.sin(loss_matrix['beta'])*np.cos(loss_matrix['alpha']))
    loss_matrix['phi'] = np.arctan2(1/np.tan(loss_matrix['beta']), np.sin(loss_matrix['alpha']))
    if join_phi==False:
        #loss_matrix = loss_matrix.sort_values('theta')
        angular_loss = pd.DataFrame({'theta':np.arange(91)})
        angular_loss['theta'] = angular_loss['theta']*np.pi/180
        angular_loss['aoi_loss'] = np.linspace(0, 2, 91)
        angular_loss['aoi_loss'] = 1/np.exp(angular_loss['aoi_loss'])
        #angular_loss['aoi_loss'].plot()
        #loss_matrix = pd.merge_asof(loss_matrix, angular_loss, on=join)
        angular_loss = interpolate.interp1d(angular_loss['theta'], angular_loss['aoi_loss'])

        loss_matrix['int'] = loss_matrix['int']*angular_loss(loss_matrix['theta'])
        loss_matrix = loss_matrix.groupby(['alpha'])['int'].sum()


def calculate_angular_dist_ground(geo_instance, matrix_name, illumination_name):
    eff_matrix = geo_instance.results[matrix_name]

    ground_illumination = geo_instance.results[illumination_name]

    if len(ground_illumination.shape) > 1:
        ground_illumination = ground_illumination.sum(axis=0)

    beta_distribution = np.sin(np.linspace(0,np.pi,180))**2 /\
                (np.sin(np.linspace(0,np.pi,180))**2).sum()

    angular_dist_1d = (eff_matrix*ground_illumination).sum(axis=-1)

    angular_dist_2d = np.multiply.outer(angular_dist_1d, beta_distribution)

    mindex = pd.MultiIndex.from_product([range(geo_instance.module_steps),
                                         range(geo_instance.angle_steps),
                                         np.linspace(0,np.pi,180)],
        names=['module_position', 'alpha', 'beta'])

    df = pd.DataFrame({'int':angular_dist_2d.flatten()}, index=mindex)
    df = df.reset_index()
    df['alpha'] = -np.pi/2 + df['alpha']/180*np.pi
    df['theta'] = np.arccos(np.sin(df['beta'])*np.cos(df['alpha']))
    df['phi'] = np.arctan2(1/np.tan(df['beta']), np.sin(df['alpha']))
    return df

def plot_angular_dist_ground(geo_instance, matrix_name, illumination_name):
    df = calculate_angular_dist_ground(geo_instance, matrix_name, illumination_name)
    binned = bin_theta_phi(df, agg_func=np.mean)
    plot_polar_heatmap(binned)

def sum_binned_ground_data(geo_instance, theta_bins='iso'):
    ground_direct_back = calculate_angular_dist_ground(geo_instance,
                                matrix_name='module_back_ground_matrix',
                                illumination_name='radiance_ground_direct_emitted')
    ground_direct_back = bin_theta_phi(ground_direct_back, agg_func=np.sum, theta_bins=theta_bins)

    ground_diffuse_back = calculate_angular_dist_ground(geo_instance,
                                matrix_name='module_back_ground_matrix',
                                illumination_name='radiance_ground_diffuse_emitted')
    ground_diffuse_back = bin_theta_phi(ground_diffuse_back, agg_func=np.sum, theta_bins=theta_bins)

    ground_diffuse_front = calculate_angular_dist_ground(geo_instance,
                                matrix_name='module_front_ground_matrix',
                                illumination_name='radiance_ground_diffuse_emitted')
    ground_diffuse_front = bin_theta_phi(ground_diffuse_front, agg_func=np.sum, theta_bins=theta_bins)

    ground_direct_front = calculate_angular_dist_ground(geo_instance,
                                matrix_name='module_front_ground_matrix',
                                illumination_name='radiance_ground_direct_emitted')
    ground_direct_front = bin_theta_phi(ground_direct_front, agg_func=np.sum, theta_bins=theta_bins)

    ground_back = (ground_direct_back + ground_diffuse_back)/4
    ground_front = (ground_diffuse_front + ground_direct_front)/4

    return {'front': ground_front,
            'back': ground_back}

def bin_theta_phi(df, module_position=6, agg_func=np.mean, theta_bins='iso'):
    df = df.loc[df['module_position']==module_position]

    if theta_bins == 'iso':
        theta_bins = np.arccos(np.linspace(0, 1, 21))[::-1]

    binned_data = df.groupby([pd.cut(df.theta, theta_bins, labels=False),
                    pd.cut(df.phi, np.linspace(-np.pi,np.pi,20), labels=False)])\
          ['int'].apply(agg_func).unstack('phi')

    mindex = pd.MultiIndex.from_product([range(len(theta_bins)-1), range(20)], names=['theta','phi'])
    template = pd.DataFrame(index=mindex)
    template['int'] = 0
    template = template['int'].unstack('phi')

    binned_data = (template+binned_data).fillna(0)

    binned_data.index = theta_bins[:-1]/np.pi*180
    binned_data.columns = binned_data.columns*20-10

    return binned_data

def process_sky_direct(df_res, geo_instance, theta_bins='iso'):
    df = df_res[['front_sky_direct','back_sky_direct']].copy()#.groupby(level="contribution", axis=1).mean()
    df['theta'] = geo_instance.tmp['theta'].copy()
    df['phi'] = geo_instance.tmp['phi'].copy()
    df = df.set_index(['theta', 'phi'], append=True)
    df = df.stack('module_position')

    df = df.reset_index()

    df_front = df[['theta', 'phi', 'front_sky_direct', 'module_position']].copy()
    df_front = df_front.rename(columns={'front_sky_direct':'int'})

    df_back = df[['theta', 'phi', 'back_sky_direct', 'module_position']].copy()
    df_back['theta'] = np.pi - (df_back['theta'])
    df_back = df_back.rename(columns={'back_sky_direct':'int'})

    return {'front': bin_theta_phi(df_front, agg_func=np.sum, theta_bins=theta_bins)/4,
            'back': bin_theta_phi(df_back, agg_func=np.sum, theta_bins=theta_bins)/4}


def process_sky_diffuse(geo_instance, theta_bins='iso'):
    def calc_alpha_dist(alpha_array):
        spacing_alpha = np.linspace(-np.pi/2, np.pi/2, 360)
        dist_alpha = np.cos(spacing_alpha)
        dist_alpha = dist_alpha/(dist_alpha).sum()

        selector = np.greater.outer(spacing_alpha, alpha_array).T
        dist_alpha = np.tile(dist_alpha, (geo_instance.module_steps, 1))
        dist_alpha[selector] = 0
        return dist_alpha

    def calc_beta_dist(alpha_dist):
        beta_dist = np.sin(np.linspace(0,np.pi,360))**2 /\
                (np.sin(np.linspace(0,np.pi,360))**2).sum()
        return np.multiply.outer(alpha_dist, beta_dist)


    alpha_front = geo_instance.tmp['alpha_2_front']
    alpha_front_dist = calc_alpha_dist(alpha_front)
    alpha_front_dist = alpha_front_dist[:, ::-1]

    front_dist = calc_beta_dist(alpha_front_dist)

    alpha_back = -np.pi/2 + geo_instance.tmp['epsilon_1_back']
    alpha_back_dist = calc_alpha_dist(alpha_back)

    # function is not aware from where to check for shadow
    alpha_back_dist = alpha_back_dist[:,::-1]
    back_dist = calc_beta_dist(alpha_back_dist)

    mindex = pd.MultiIndex.from_product([range(geo_instance.module_steps),
                                         np.linspace(0,np.pi,360),
                                         np.linspace(0,np.pi,360)],
        names=['module_position', 'alpha', 'beta'])

    df_front = pd.DataFrame({'int':front_dist.flatten()}, index=mindex).reset_index()
    df_back = pd.DataFrame({'int':back_dist.flatten()}, index=mindex).reset_index()
    df = pd.concat([df_front, df_back], keys=['front','back'], names=['side'])
    df['alpha'] = -np.pi/2 + df['alpha']
    df['theta'] = np.arccos(np.sin(df['beta'])*np.cos(df['alpha']))
    df['phi'] = np.arctan2(1/np.tan(df['beta']), np.sin(df['alpha']))

    binned_front = bin_theta_phi(df.loc['front'], agg_func=np.sum, theta_bins=theta_bins)*geo_instance.DHI/4
    binned_back = bin_theta_phi(df.loc['back'], agg_func=np.sum, theta_bins=theta_bins)*geo_instance.DHI/4

    return {'front': binned_front,
            'back': binned_back}

def calc_front_back(geo_sinatcen, df_res, theta_bins='iso'):
    sky_diffuse = process_sky_diffuse(simulator.simulation, theta_bins=theta_bins)
    sky_direct = process_sky_direct(df_res, simulator.simulation, theta_bins=theta_bins)
    ground = sum_binned_ground_data(simulator.simulation, theta_bins=theta_bins)

    front = sky_diffuse['front'] + sky_direct['front'] + ground['front']
    back = sky_diffuse['back'] + sky_direct['back'] + ground['back']

    return front, back


if __name__ == '__main__':

    berlin_illumination = bi.Illumination('berlin_2014.nc',
                                      file_format='copernicus',
                                      tmy=False)

    simulator = bi.YieldSimulator(berlin_illumination,
                                  tmy_data=False, module_height=0.5)
    #spacing is specified in m, tilt in deg
    yearly_yield = simulator.calculate_yield(spacing=6, tilt=35)
    print('Yearly yield in kWh/year: {}'.format(yearly_yield))

    df_res = simulator.simulate(spacing=6, tilt=35)
    print((df_res.loc[:,(slice(None),6)].sum()/4/1000))

    front_iso, back_iso = calc_front_back(simulator.simulation, df_res, theta_bins='iso')

    plot_polar_heatmap(front_iso/1000*2*np.pi, filepath='/home/peter/tmp/front.svg')
    plot_polar_heatmap(back_iso/1000*2*np.pi, filepath='/home/peter/tmp/back.svg')

    ground_direct_back = calculate_angular_dist_ground(simulator.simulation,
                                matrix_name='module_back_ground_matrix',
                                illumination_name='radiance_ground_direct_emitted')
    ground_direct_back = bin_theta_phi(ground_direct_back, agg_func=np.sum)
    plot_polar_heatmap(ground_direct_back)

    sky_diffuse = process_sky_diffuse(simulator.simulation)
    plot_polar_heatmap(sky_diffuse['front'])

    ground = sum_binned_ground_data(simulator.simulation, theta_bins='iso')
    plot_polar_heatmap(ground['front'])

    front_iso, back_iso = calc_front_back(simulator.simulation, df_res, theta_bins='iso')

    front, back = calc_front_back(simulator.simulation, df_res, theta_bins=np.linspace(0, np.pi/2, 19))

    front.sum(axis=1).plot.bar()
    back.sum(axis=1).plot.bar()

    data = pd.read_csv('refl_angle.csv', header=None)
    data.columns =['angle', 'textured', 'flat']
    data = data.set_index('angle')
    data = data.append(pd.DataFrame({'flat':0,'textured':0}, index=[88]))

    width=.35

    loss = (data['textured'] / data['textured'].max())
    interp_loss = interpolate.interp1d(loss.index, loss)

    back_plot = back.sum(axis=1)/1000
    back_plot.index = back_plot.index.astype(int)

    loss = pd.Series(interp_loss(back.index+2.5), index=back_plot.index)

    back_plot = back_plot.rename('absorbed').to_frame()
    back_plot['reflected'] = back_plot['absorbed']*(1-loss)
    back_plot['absorbed'] = back_plot['absorbed']-back_plot['reflected']
    back_plot = back_plot.rename(columns={'absorbed':'utilizable',
                                            'reflected':'loss from angle of incidence'})
    ax1 = back_plot.plot.bar(width=width, stacked=True)
    plt.ylim([0,18])

    ax2 = (100-loss*100).reset_index(drop=True).plot(secondary_y=True, c='#ff7f0e')
    plt.xlim([-width-0.1, len(back_plot)-width])

    ax1.set_xlabel('Angle of incidence (deg)')
    ax1.set_ylabel(r'Radiant exposure $\left(\frac{\mathrm{kWh}}{\mathrm{year} \cdot \mathrm{m}^2}\right)$')
    ax2.set_ylabel('Angle of incidence loss (%)', rotation=270)
    ax2.yaxis.label.set_color('#ff7f0e')
    ax2.tick_params(axis='y', colors='#ff7f0e')
    ax2.set_ylim(0,100)
    ax1.get_legend()._loc = (0.03, 0.82)
    plt.savefig('/home/peter/tmp/fig3_back.svg', format='svg')
    plt.show()


    front_plot = front.sum(axis=1)/1000
    front_plot.index = front_plot.index.astype(int)

    loss = pd.Series(interp_loss(front.index+2.5), index=front_plot.index)

    front_plot = front_plot.rename('absorbed').to_frame()
    front_plot['reflected'] = front_plot['absorbed']*(1-loss)
    front_plot['absorbed'] = front_plot['absorbed']-front_plot['reflected']
    front_plot = front_plot.rename(columns={'absorbed':'utilizable',
                                            'reflected':'loss from angle of incidence'})
    ax1 = front_plot.plot.bar(width=width, stacked=True)
    plt.ylim([0,150])

    ax2 = (100-loss*100).reset_index(drop=True).plot(secondary_y=True, c='#ff7f0e')
    plt.xlim([-width-0.1, len(front_plot)-width])

    ax1.set_xlabel('Agnle of incidence (deg)')
    ax1.set_ylabel(r'Radiant exposure $\left(\frac{\mathrm{kWh}}{\mathrm{year} \cdot \mathrm{m}^2}\right)$')
    ax2.set_ylabel('Angle of incidence loss (%)', rotation=270)
    ax2.yaxis.label.set_color('#ff7f0e')
    ax2.set_ylim(0,100)
    ax2.tick_params(axis='y', colors='#ff7f0e')
    ax1.get_legend()._loc = (0.03, 0.82)
    plt.savefig('/home/peter/tmp/fig3_front.svg', format='svg')
    plt.show()

    print(1-front_plot['absorbed'].sum()/front_plot.sum().sum())
    print(1-back_plot['absorbed'].sum()/back_plot.sum().sum())

    plt.figure(dpi=400, figsize=(8,6))
    ax = data.rename(columns={'flat':'flat interface', 'textured':'textured interface'}).plot()
    ax.set_xlabel('incindent angle (deg)')
    ax.set_ylabel('photocurrent density (mA/cm²)')
    plt.savefig('/home/peter/tmp/img1.svg', format='svg')








