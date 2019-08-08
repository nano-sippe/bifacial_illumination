# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bifacial_geo as geo
from skopt import gp_minimize

class Simulator():
    def __init__(self, df, module_agg_func='min', bifacial=True, albedo=0.3,
                 module_length = 1.96, front_eff=0.2, back_eff=0.18,
                 inputDict={}):
        self.bifacial = bifacial
        self.front_eff = front_eff
        self.back_eff = back_eff
        self.module_agg_func = module_agg_func

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
            'albedo': albedo, # albedo of the ground
            'ground_steps': 101, #number of steps into which irradiance on the ground is evaluated in the interval [0,D]
            'module_steps': 12, # SET THIS NUMBER HIGHER FOR REAL EVALUATION! (SUGGESTION 20) number of lengths steps at which irradiance is evaluated on the module
            'angle_steps': 180 # Number at which angle discretization of ground light on module should be set
        }
        self.inputDict.update(inputDict)
        self.inputDict['theta_S_deg'] = df.zenith
        self.inputDict['phi_S_deg'] = df.azimuth
        print('Albedo: {}'.format(albedo))

    def simulate(self, distance, tilt):
        self.inputDict['theta_m_deg'] = tilt
        self.inputDict['D'] = distance
        simulation = geo.ModuleIllumination(self.inputDict)

        diffuse = np.concatenate([
                simulation.results['irradiance_module_front_ground_diffuse'],
                simulation.results['irradiance_module_front_sky_diffuse'],
                simulation.results['irradiance_module_back_sky_diffuse'],
                simulation.results['irradiance_module_back_ground_diffuse']
                ])
        direct = np.concatenate([
                simulation.results['irradiance_module_front_ground_direct'],
                simulation.results['irradiance_module_front_sky_direct'],
                simulation.results['irradiance_module_back_sky_direct'],
                simulation.results['irradiance_module_back_ground_direct']
                ], axis=1)
        diffuse_ts = np.outer(self.dhi, diffuse)
        direct_ts = self.dni[:, None]*direct

        column_names = ['front_ground', 'front_sky', 'back_sky', 'back_ground']
        prefixes = ['_diffuse', '_direct']
        column_names = [name+prefix
                        for prefix in prefixes
                        for name in column_names]

        level_names = ['contribution', 'module_position']
        multi_index = pd.MultiIndex.from_product([
                                          column_names,
                                          range(self.inputDict['module_steps'])
                                          ],
                                    names=level_names)

        results = pd.DataFrame(np.concatenate([diffuse_ts, direct_ts], axis=1),
                                  columns = multi_index)
        return results

    def calculate_yield(self, distance, tilt):
        results = self.simulate(distance, tilt)
        back_columns = results.columns.get_level_values('contribution').str.contains('back')
        front_columns = ~back_columns

        if self.bifacial:
            results.loc[:, back_columns] *= self.back_eff
            results.loc[:, front_columns] *= self.front_eff
        else:
            results = results.loc[:, front_columns]
            results *= self.front_eff

        yearly_yield = results.groupby(level='module_position', axis=1).sum()\
                       .apply(self.module_agg_func, axis=1)\
                       .sum()

        return yearly_yield/1000 #convert to kWh


class CostOptimizer(Simulator):
    def __init__(self, df, module_agg_func='min', bifacial=True,
                 module_length = 1.65, invest_kwp = 1500,
                 price_per_m2_land = 5, inputDict={}, **kwargs):
        self.module_cost_kwp = invest_kwp
        self.price_per_m2_land = price_per_m2_land
        self.module_length = module_length

        super().__init__(df, module_agg_func=module_agg_func, bifacial=bifacial, module_length=module_length,
             inputDict=inputDict, **kwargs)

    def calculate_cost(self, yearly_yield):
        price_per_m2_module = self.module_cost_kwp * self.front_eff * 100 # in cents
        price_per_m2_land = self.price_per_m2_land * 100
        land_cost_per_m2_module = price_per_m2_land*self.inputDict['D']/self.inputDict['L']
        cost = (price_per_m2_module + land_cost_per_m2_module)/(yearly_yield * 25)
        print('Cost: {}'.format(cost))
        return cost

    def optimization_wrapper(self, para_list):
        distance, tilt = para_list
        print(distance, tilt)
        yearly_yield = self.calculate_yield(distance, tilt)
        return self.calculate_cost(yearly_yield)

    def optimize(self, dist_low=1.65, dist_high=15, tilt_low=1, tilt_high=50,
                 ncalls=50):

        #minimal spacing has to at least module length
        dist_low = max(dist_low, self.module_length)

        self.res = gp_minimize(self.optimization_wrapper,
                               [(dist_low, dist_high), (tilt_low, tilt_high)],
                               n_random_starts=20, n_jobs=1, n_calls=ncalls,
                               random_state=1,)
