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



def plot_polar_heatmap(df):

    rad = np.sin(np.deg2rad(df.index))
    azi = np.deg2rad(df.columns+180)

    r, th = np.meshgrid(rad, azi)

    fig = plt.figure()
    plt.subplot(projection="polar")

    plt.pcolormesh(th, r, df.T)
    #plt.pcolormesh(th, z, r)

    plt.plot(azi, r, color='k', ls='none')
    plt.grid()

    plt.show()

def plot_angular_dist_ground(geo_instance, matrix_name, illumination_name):

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

    binned = bin_theta_phi(df)
    plot_polar_heatmap(binned)


def bin_theta_phi(df, module_position=6):
    df = df.loc[df['module_position']==module_position]
    bins = np.arccos(np.linspace(0, 1, 11))[::-1]

    binned_data = df.groupby([pd.cut(df.theta, bins, labels=False),
                    pd.cut(df.phi, np.linspace(-np.pi,np.pi,20), labels=False)])\
          ['int'].mean().unstack('phi')

    binned_data.index = binned_data.index*10
    binned_data.columns = binned_data.columns*20-10

    return binned_data

def process_angular_diffuse_direct(geo_instance):
    def calc_alpha_dist(alpha_array):
        spacing_alpha = np.linspace(-np.pi/2, np.pi/2, 360)
        dist_alpha = np.cos(spacing_alpha)

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
    front_dist = calc_beta_dist(alpha_front_dist)

    alpha_back = -np.pi/2 + geo_instance.tmp['epsilon_1_back']
    alpha_back_dist = calc_alpha_dist(alpha_back)
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

    binned = bin_theta_phi(df.loc['front'])

    plot_polar_heatmap(binned)





if __name__ == '__main__':
    # Define the dictionary for the class
    InputDict = {
        'L': 1.650,# module length, standard is 1650 mm or 1960 mm
        'theta_m_deg': 52., # Angle of the module with respect to the ground (first guess optimal theta_m = latitude on Earth)
        'D': 3.000, # distance between modules
        'H': 0.500, # height of module base above ground
        'DNI': 1, # direct normal irradiance
        'DHI': 1, # diffuse horizontal irradiance
        'theta_S_deg': np.array([30]), # zenith of the Sun
        'phi_S_deg': np.array([150]), # azimuth of the Sun
        'albedo': 0.3, # albedo of the ground
        'ground_steps': 101, #number of steps into which irradiance on the ground is evaluated in the interval [0,D]
        'module_steps': 12, # SET THIS NUMBER HIGHER FOR REAL EVALUATION! (SUGGESTION 20) number of lengths steps at which irradiance is evaluated on the module
        'angle_steps': 180 # Number at which angle discretization of ground light on module should be set
    }

# =============================================================================
#     GI = geo2.ModuleIllumination(module_length=1.92, module_tilt=52, mount_height=0.5,
#                  module_distance=7.1, dni=1, dhi=1, zenith_sun=30, azimuth_sun=150,
#                  albedo=0.3, ground_steps=101, module_steps=12, angle_steps=180)
# =============================================================================



    input_dict = dict(module_length=1.96, module_tilt=35,
                      mount_height=0.5, module_spacing=3.5,
                      dni=100, dhi=120, zenith_sun=27,
                      azimuth_sun=180.3, albedo=0.3, ground_steps=101, angle_steps=180)

    model = ModuleIllumination(**input_dict)

    #plt.plot(model.results['radiance_ground_direct_emitted'])
# =============================================================================
#     alpha_dist = (model.results['module_back_ground_matrix']@model.results['radiance_ground_direct_emitted'])
#     import matplotlib
#     matplotlib.rcParams.update({'font.size': 12})
#
#     alpha_dist = pd.Series(alpha_dist[6])
#     alpha_dist.index = alpha_dist.index.rename('alpha')
#
#     alpha_dist.plot()
# =============================================================================
    plt.xlabel('alpha', fontsize=18)

    #process_angular_distribution(model, 'module_back_ground_matrix', 'radiance_ground_direct_emitted')


    berlin_illumination = bi.Illumination('berlin_2014.nc',
                                      file_format='copernicus',
                                      tmy=False)

    simulator = bi.YieldSimulator(berlin_illumination,
                                  tmy_data=False, module_height=1)
    #spacing is specified in m, tilt in deg
    yearly_yield = simulator.calculate_yield(spacing=6, tilt=35)
    #print('Yearly yield in kWh/year: {}'.format(yearly_yield))

    df_res = simulator.simulate(spacing=6, tilt=35)

    plot_angular_dist_ground(simulator.simulation,
                                 matrix_name='module_back_ground_matrix',
                                 illumination_name='radiance_ground_direct_emitted')

    plot_angular_dist_ground(simulator.simulation,
                                 matrix_name='module_back_ground_matrix',
                                 illumination_name='radiance_ground_diffuse_emitted')

    process_angular_diffuse_direct(simulator.simulation)

    asdf

    df_test = df_res[['front_sky_direct','back_sky_direct']].groupby(level="contribution", axis=1).mean()
    df_test['theta'] = simulator.simulation.tmp['theta']
    df_test['phi'] = simulator.simulation.tmp['phi']
    df_test['theta'] = (df_test['theta'].apply(np.rad2deg)/5).apply(np.floor)*5
    df_test['phi'] = (df_test['phi'].apply(np.rad2deg)/10).apply(np.floor)*10

    test = df_test.groupby(['theta', 'phi'])['front_sky_direct'].mean()
    test = test.loc[:90]
    test.groupby('theta').sum().plot.bar()

    #sns.heatmap(test.groupby(['theta', 'phi']).sum().unstack('phi'))
    plot_polar_heatmap(test.unstack('phi').fillna(0))

    test = df_test.groupby(['theta', 'phi'])['back_sky_direct'].sum()
    test = test.reset_index()
    test['theta'] = 90-(test['theta']-90)
    test = test.loc[test['theta']<=90]
    test = test.set_index(['theta','phi'])
    test.groupby('theta').sum().plot.bar()

    test = test.groupby(['theta', 'phi'])['back_sky_direct'].sum()

    #sns.heatmap(test.groupby(['theta', 'phi']).sum().unstack('phi'))
    plot_polar_heatmap(test.unstack('phi').fillna(0))

# =============================================================================
#     distrib = df_test.loc['20140501':'20140801'].groupby(['theta', 'phi']).sum()
#     distrib = df_test.query('11<=hour<13').groupby(['theta', 'phi']).sum()
#     sns.heatmap(distrib['front_sky_direct'].unstack('phi'))
# =============================================================================


    #plt.imshow(GI.results['radiance_ground_direct_emitted'])

    asdf
    dt_1238 = datetime.datetime(2019, 9, 23, 12, 38, 0, tzinfo = datetime.timezone(datetime.timedelta(hours=2)))
    inputs = input_dict
    inputs.update({
        'dni': 883, # direct normal irradiance
        'dhi': 134, # diffuse horizontal irradiance
        'zenith_sun': 90-sol.get_altitude(52.5, 13.25, dt_1238), # zenith of the Sun (Berlin, 2019-09-20, 16:00 CEST)
        'azimuth_sun': sol.get_azimuth(52.5, 13.25, dt_1238), # azimuth of the Sun (Berlin, 2019-09-20, 16:00 CEST)
    })
    GI_1238 = ModuleIllumination(**inputs)

    asdf

    hour_range = np.linspace(4,21,18,dtype = int)
    minute_range = np.linspace(0,50,6,dtype = int)
    time_array = np.zeros(len(hour_range)*len(minute_range))
    ground_direct = np.zeros([3,len(hour_range)*len(minute_range),InputDict['ground_steps']])
    month_array = [6,8,11]

    zenith_arr = []
    azimuth_arr = []

    for k,month in enumerate(month_array):
        for i,hh in enumerate(hour_range):
            #print(hh,'o clock')
            for j,mm in enumerate(minute_range):
                time_array[6*i+j] = hh + mm/60
                date_temp = datetime.datetime(2019, month, 20, hh, mm, 0, 0, tzinfo = datetime.timezone(datetime.timedelta(hours=2)) )
                zenith_arr.append(90 - sol.get_altitude(52.5, 13.25, date_temp))
                azimuth_arr.append(sol.get_azimuth(52.5, 13.25, date_temp))

    zenith_arr = np.array(zenith_arr)
    azimuth_arr = np.array(azimuth_arr)
    input_dict.update({'zenith_sun':zenith_arr, 'azimuth_sun':azimuth_arr})

    GI = ModuleIllumination(**input_dict)
    ground_direct = (GI.results['irradiance_module_back_ground_direct'] * np.pi / GI.albedo).reshape(3,108,-1)
    for i in range(3):
        fig, ax = plt.subplots(figsize=(4,4))
        im = ax.imshow(ground_direct[i],origin='lower',vmax=0.88)

    validate_ts = True
    if validate_ts is True:
        df = pd.read_hdf('mojave.h5', 'table')
        df = df.rename(columns={'GHI':'ghi_total',
                            'DNI':'dni_total',
                            'DHI':'diffuse_total'})
        df.zenith = 90 - df.zenith

        df = df.set_index('dt')

        #removing datapoints where the sun is below the horizon
        df = df.loc[df.zenith < 90]
        #remove datapoints where illumination is 0
        df = df.loc[df['ghi_total']>0]

        input_dict['zenith_sun'] = df.zenith
        input_dict['azimuth_sun'] = df.azimuth

        GI_ts = ModuleIllumination(**input_dict)
        #with open("reference_ts.pickle", "wb") as output_file:
        #    pickle.dump(GI_ts.results, output_file)

        with open("reference_ts.pickle", "rb") as output_file:
            referenze = pickle.load(output_file)

        for k, v in GI_ts.results.items():
            print(k)
            print(np.allclose(v, referenze[k], rtol=1e-04, atol=1e-06))

    validate_yield = True
    if validate_yield is True:
        from tmy_yield import Simulator
        sim = Simulator(df)
        start = time.time()
        res = sim.simulate(6, 18)
        print(time.time()-start)
        #res.to_hdf('ref3.h5', 'table')

        ref = pd.read_hdf('ref3.h5', 'table')

        ref['front_sky_direct'].mean(axis=1).groupby(df.index.hour).sum().plot()
        res['front_sky_direct'].mean(axis=1).groupby(df.index.hour).sum().plot()

        ref['back_ground_direct'].mean(axis=1).groupby(df.index.hour).sum().plot()
        res['back_ground_direct'].mean(axis=1).groupby(df.index.hour).sum().plot()

        ref['back_ground_diffuse'].mean(axis=1).groupby(df.index.hour).sum().plot()
        res['back_ground_diffuse'].mean(axis=1).groupby(df.index.hour).sum().plot()

        ref['front_ground_direct'].mean(axis=1).groupby(df.index.hour).sum().plot()
        res['front_ground_direct'].mean(axis=1).groupby(df.index.hour).sum().plot()

        ref['front_ground_diffuse'].mean(axis=1).groupby(df.index.hour).sum().plot()
        res['front_ground_diffuse'].mean(axis=1).groupby(df.index.hour).sum().plot()

# =============================================================================
#         ax.set_xticks(x_ticks)
#         ax.set_xticklabels(np.round(GI.x_g_array[x_ticks],1))
#         ax.set_yticks(y_ticks)
#         ax.set_yticklabels(time_array[y_ticks])
#         ax.set_xlabel('distance (m)')
#         ax.set_ylabel('time (h)')
#         #if i==2:
#         #    cbar = fig.colorbar(im, shrink =0.8, ax=ax)
#         #    cbar.set_label('geom.\ dist.\ func.\ $\gamma_\mathrm{dir}$')
#         figname = 'figures/ground_illum_{}.pdf'.format(i)
#         plt.tight_layout()
#         #plt.savefig(figname, format='pdf', dpi=300)
#         plt.show()
# =============================================================================


    asdf

    df = pd.read_hdf('tmy_spec_seattle.h5', 'table')

    df = df.set_index('dt')

    ghi_column_filter = df.columns.str.contains('GHI')
    dni_column_filter = df.columns.str.contains('DNI')

    df['ghi_total'] = df.loc[:,ghi_column_filter].sum(axis=1)*10
    df['dni_total'] = df.loc[:,dni_column_filter].sum(axis=1)*10

    df.zenith = 90 - df.zenith

    df['diffuse_total'] = df['ghi_total']-df['dni_total']*np.cos(df.zenith/180*np.pi)
    df = df.loc[df.zenith < 90]


    InputDict['theta_S_deg'] = df.zenith
    InputDict['phi_S_deg'] = df.azimuth

    GI = ModuleIllumination(InputDict) # GroundIllumination
    asdf

    figure = plt.figure(figsize=(6,4))
    plt.plot(GI.l_array,GI.results['irradiance_module_front_ground_diffuse'])
    plt.plot(GI.l_array,GI.results['irradiance_module_back_ground_diffuse'])
    plt.plot(GI.l_array,GI.irradiance_module_front_ground_direct)
    plt.plot(GI.l_array,GI.irradiance_module_back_ground_direct)
    figure.legend(['front_ground_diff','back_ground_diff','front_ground_dir','back_ground_dir'])
    plt.show()


    figure = plt.figure(figsize=(6,4))
    plt.plot(GI.l_array,GI.results['irradiance_module_front_sky_direct'],label='front')
    plt.plot(GI.l_array,GI.results['irradiance_module_back_sky_direct'],label='back')
    plt.legend()
    plt.title('Direct Illumination from the Sky')
    plt.xlabel('position on module (m)')
    plt.ylabel('Irradiance on module (m$^{-2}$) (DHI = 1)')
    plt.show()

    #dict['theta_S_deg'] = 30 # zenith of the Sun
    #dict['phi_S_deg'] = 135 # azimuth of the Sun
    #GI = ModuleIllumination(dict) # GroundIllumination
    figure = plt.figure(figsize=(6,4))
    plt.plot(GI.x_g_array,GI.results['radiance_ground_diffuse_emitted'],label='diffuse ground')
    plt.plot(GI.x_g_array,GI.results['radiance_ground_direct_emitted'],label='direct ground')
    plt.legend()
    #plt.title('Diffuse illumination from the sky')
    plt.xlabel('position on ground (m)')
    plt.ylabel('radiance from ground (m$^{-2}$) (DHI = 1)')
    plt.show()

    figure = plt.figure(figsize=(6,4))
    plt.plot(GI.l_array,GI.results['irradiance_module_front_ground_diffuse'],label='front, diffuse')
    plt.plot(GI.l_array,GI.results['irradiance_module_front_ground_direct'],label='front, direct')
    plt.plot(GI.l_array,GI.results['irradiance_module_back_ground_diffuse'],label='back, diffuse')
    plt.plot(GI.l_array,GI.results['irradiance_module_back_ground_direct'],label='back, direct')
    plt.legend()
    plt.title('Illumination from the ground')
    plt.xlabel('position on module (m)')
    plt.ylabel('Irradiance on module (m$^{-2}$) (DHI = 1)')
    figure = plt.figure(figsize=(6,4))
    plt.plot(GI.l_array,GI.results['irradiance_module_front_sky_diffuse'],label='sky front')
    plt.plot(GI.l_array,GI.results['irradiance_module_back_sky_diffuse'],label='sky back')
    plt.legend()
    plt.title('Diffuse illumination from the sky')
    plt.xlabel('position on module (m)')
    plt.ylabel('Irradiance on module (m$^{-2}$) (DHI = 1)')
    plt.show()

# =============================================================================
#     from numpy import cross, eye, dot
#     from scipy.linalg import expm, norm
#
#     def M(axis, theta):
#         return expm(cross(eye(3), axis/norm(axis)*theta))
#
#     v1 = np.array([1,0,0])
#     v2 = np.array([0.8, 0.4, 0])
#     v2 = v2/np.linalg.norm(v2)
#     v_axis_1 = np.array([0,-1,0])
#     v_axis_2 = np.array([v2[1], -v2[0], 0])
#
#     axis_1 = np.cross(v1, v_axis_1)/np.linalg.norm(np.cross(v1, v_axis_1))
#     axis_2 = np.cross(v2, v_axis_2)/np.linalg.norm(np.cross(v2, v_axis_2))
#     theta_array = np.linspace(0, np.pi/2, 91)
#     distance = np.zeros_like(theta_array)
#
#     for i, theta in enumerate(theta_array):
#         v1_rot = np.dot(M(v_axis_1, theta), v1)
#         v2_rot = np.dot(M(v_axis_2, theta), v2)
#         distance[i] = np.arccos(np.dot(v1_rot, v2_rot))
#     plt.plot(theta_array, distance)
#
#     v, axis, theta = [3,5,0], [4,4,1], 1.2
#     M0 = M(axis, theta)
#     print(dot(M0,v))
# =============================================================================


