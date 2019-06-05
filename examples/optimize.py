# -*- coding: utf-8 -*-



import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bifacial_geo as geo
from skopt import gp_minimize
from skopt import plots
optimize = 'min'
#area_cost_ratio = 0.001

class Simulator():
    def __init__(self, location='dallas', target='min', bifacial=True, module_length = 1.65,
                 invest_kwp = 1500, price_per_m2_land = 5):
        self.bifacial = bifacial
        self.module_cost_kwp = invest_kwp
        self.price_per_m2_land = price_per_m2_land
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
        self.dni = df.loc[:,'dni_total']
        self.dhi = df.loc[:,'diffuse_total']
        self.inputDict = {
            'L': module_length,# module length, standard is 1650 mm or 1960 mm
            'theta_m_deg': 1, # Angle of the module with respect to the ground (first guess optimal theta_m = latitude on Earth)
            'D': 1, # distance between modules
            'H': 0.5, # height of module base above ground
            'DNI': 1, # direct normal irradiance
            'DHI': 1, # diffuse horizontal irradiance
            'theta_S_deg': 30, # zenith of the Sun
            'phi_S_deg': 150, # azimuth of the Sun
            'albedo': 0.3, # albedo of the ground
            'ground_steps': 101, #number of steps into which irradiance on the ground is evaluated in the interval [0,D]
            'module_steps': 12, # SET THIS NUMBER HIGHER FOR REAL EVALUATION! (SUGGESTION 20) number of lengths steps at which irradiance is evaluated on the module
            'angle_steps': 180 # Number at which angle discretization of ground light on module should be set
        }
        self.inputDict['theta_S_deg'] = df.zenith
        self.inputDict['phi_S_deg'] = df.azimuth


    def simulate(self, distance, tilt):
        self.inputDict['theta_m_deg'] = tilt
        self.inputDict['D'] = distance
        simulation = geo.ModuleIllumination(self.inputDict)
        result = {}
        if optimize=='mean':
            result['front_ground_diffuse'] = simulation.results['irradiance_module_front_ground_diffuse_mean']*self.dhi
            result['front_ground_direct'] = simulation.results['irradiance_module_front_ground_direct_mean']*self.dni
            result['front_sky_diffuse'] = simulation.results['irradiance_module_front_sky_diffuse_mean']*self.dhi
            result['front_sky_direct'] = simulation.results['irradiance_module_front_sky_direct_mean']*self.dni
            if self.bifacial:
                result['back_sky_diffuse'] = simulation.results['irradiance_module_back_sky_diffuse_mean']*self.dhi
                result['back_ground_diffuse'] = simulation.results['irradiance_module_back_ground_diffuse_mean']*self.dhi
                result['back_ground_direct'] = simulation.results['irradiance_module_back_ground_direct_mean']*self.dni
                result['back_sky_direct'] = simulation.results['irradiance_module_back_sky_direct_mean']*self.dni

        if optimize=='min':
            result['front_ground_diffuse'] = np.outer(self.dhi, simulation.results['irradiance_module_front_ground_diffuse'])
            result['front_ground_direct'] = simulation.results['irradiance_module_front_ground_direct']*self.dni[:, None]
            result['front_sky_diffuse'] = np.outer(self.dhi, simulation.results['irradiance_module_front_sky_diffuse'])
            result['front_direct'] = simulation.results['irradiance_module_front_sky_direct']*self.dni[:, None]
            if self.bifacial:
                result['back_sky_diffuse'] = np.outer(self.dhi, simulation.results['irradiance_module_back_sky_diffuse'])
                result['back_direct'] = simulation.results['irradiance_module_back_sky_direct']*self.dni[:, None]
                result['back_ground_diffuse'] = np.outer(self.dhi, simulation.results['irradiance_module_back_ground_diffuse'])
                result['back_ground_direct'] = simulation.results['irradiance_module_back_ground_direct']*self.dni[:, None]

            result = np.stack([arr for _, arr in result.items()])
            result = result.min(axis=-1).sum()

        #import pdb
        #pdb.set_trace()

        #result = pd.DataFrame(result)
        yearly_yield = result/1000
        return yearly_yield

    def calculate_cost(self, yearly_yield):
        price_per_m2_module = self.module_cost_kwp * 0.2 * 100 # in cents
        price_per_m2_land = self.price_per_m2_land * 100
        land_cost_per_m2_module = price_per_m2_land*self.inputDict['D']/self.inputDict['L']
        cost = (price_per_m2_module + land_cost_per_m2_module)/(yearly_yield * 0.2 * 25)
        return cost

    def optimization_wrapper(self, para_list):
        distance, tilt = para_list
        print(distance, tilt)
        yearly_yield = self.simulate(distance, tilt)
        return self.calculate_cost(yearly_yield)

    def optimize(self, dist_low=1.65, dist_high=15, tilt_low=1, tilt_high=50, ncalls=50):
        self.res = gp_minimize(self.optimization_wrapper,
                               [(dist_low, dist_high), (tilt_low, tilt_high)],
                               n_random_starts=20, n_jobs=1, n_calls=ncalls)

results_bifacial = []
land_costs = np.array([1, 2.5, 5, 10, 20])
invest_kwp = 1500
for land_cost in land_costs:
    simulator = Simulator(price_per_m2_land=land_cost,
                          bifacial=True,
                          invest_kwp = invest_kwp,
                          location='seattle')
    simulator.optimize()
    results_bifacial.append(simulator.res)

results_front = []
land_costs = np.array([1, 2.5, 5, 10, 20])
for land_cost in land_costs:
    simulator = Simulator(price_per_m2_land=land_cost,
                          bifacial=False,
                          invest_kwp = invest_kwp,
                          location='seattle')
    simulator.optimize()
    results_front.append(simulator.res)

df_dict = {
    'cost': [results_bifacial[i].fun for i in range(len(results_bifacial))],
    'dist': [results_bifacial[i].x[0] for i in range(len(results_bifacial))],
    'tilt': [results_bifacial[i].x[1] for i in range(len(results_bifacial))]
        }

df_bi = pd.DataFrame(df_dict, index=land_costs)
df_bi['yield'] = (invest_kwp*0.2*100 + 100*land_costs*df_bi.dist/1.65)/(df_bi.cost*25*0.2)
df_bi['land_cost'] = (100*land_costs*df_bi.dist/1.65) / (df_bi['yield']*25*0.2)
df_bi['invest'] = df_bi['cost'] - df_bi['land_cost']



df_dict = {
    'cost': [results_front[i].fun for i in range(len(results_bifacial))],
    'dist': [results_front[i].x[0] for i in range(len(results_bifacial))],
    'tilt': [results_front[i].x[1] for i in range(len(results_bifacial))]
        }

df_front = pd.DataFrame(df_dict, index=land_costs)
df_front['yield'] = (invest_kwp*0.2*100 + 100*land_costs*df_front.dist/1.65)/(df_front.cost*25*0.2)
df_front['land_cost'] = (100*land_costs*df_front.dist/1.65) / (df_front['yield']*25*0.2)
df_front['invest'] = df_front['cost'] - df_front['land_cost']



asdf
plots.plot_objective(simulator.res[1])

plots.plot_objective(results_bifacial[2], size=4)

plt.figure(dpi=100)
ax = df_bi.cost.plot()
ax2 = df_bi.dist.plot(secondary_y=True)

ax.set_ylim((3,6))
ax2.set_ylim((2,12))

ax.set_ylabel('electr. cost (cent/kWh)')
ax.set_xlabel('land cost (€/m²)')
ax2.set_ylabel('optimal distance (m)')
plt.show()

plt.figure(dpi=100)
ax = df_front.cost.plot()
ax2 = df_front.dist.plot(secondary_y=True)

ax.set_ylim((3,4))
ax2.set_ylim((2,10))

ax.set_ylabel('electr. cost (cent/kWh)')
ax.set_xlabel('land cost (€/m²)')
ax2.set_ylabel('optimal distance (m)')


plt.figure(dpi=100)
ax = df_bi.tilt.plot()
ax.set_ylim((10, 35))
ax.set_ylabel('optimal tilt')
ax.set_xlabel('land cost (€/m²)')
#plt.show()

#plt.figure(dpi=100)
ax = df_front.tilt.plot()
ax.set_ylim((10, 35))
ax.set_ylabel('optimal tilt')
ax.set_xlabel('land cost (€/m²)')
plt.show()

