# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import geo
from skopt import gp_minimize

class YieldSimulator():
    def __init__(self, illumination, module_agg_func='min', bifacial=True, albedo=0.3,
                 module_length = 1.96, front_eff=0.2, back_eff=0.18,
                 module_height=0.5, kw_parameter={}):
        self.bifacial = bifacial
        self.front_eff = front_eff
        self.back_eff = back_eff
        self.module_agg_func = module_agg_func

        #whether the perez model should be used to determine the components of diffuse irradiance
        #self.perez_diffuse = perez_diffuse

        self.dni = illumination.df.loc[:,'DNI']
        self.dhi = illumination.df.loc[:,'DHI']

        self.input_parameter = dict(module_length=module_length,
                                    mount_height=module_height,
                                    albedo=albedo)
        self.input_parameter.update(kw_parameter)
        self.input_parameter['zenith_sun'] = illumination.df.zenith
        self.input_parameter['azimuth_sun'] = illumination.df.azimuth

    def simulate(self, spacing, tilt):
        self.input_parameter['module_tilt'] = tilt
        self.input_parameter['module_spacing'] = spacing
        simulation = geo.ModuleIllumination(**self.input_parameter)

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
                                          range(simulation.module_steps)
                                          ],
                                    names=level_names)

        results = pd.DataFrame(np.concatenate([diffuse_ts, direct_ts], axis=1),
                                  columns = multi_index)
        return results

    def calculate_yield(self, spacing, tilt):
        results = self.simulate(spacing, tilt)
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

# =============================================================================
# class CostOptimizer(YieldSimulator):
#     def __init__(self, df, module_agg_func='min', bifacial=True,
#                  module_length = 1.65, invest_kwp = 1500,
#                  price_per_m2_land = 5, inputDict={}, **kwargs):
#         self.module_cost_kwp = invest_kwp
#         self.price_per_m2_land = price_per_m2_land
#         self.module_length = module_length
#
#         super().__init__(df, module_agg_func=module_agg_func, bifacial=bifacial, module_length=module_length,
#              inputDict=inputDict, **kwargs)
#
#     def calculate_cost(self, yearly_yield):
#         price_per_m2_module = self.module_cost_kwp * self.front_eff * 100 # in cents
#         price_per_m2_land = self.price_per_m2_land * 100
#         land_cost_per_m2_module = price_per_m2_land*self.inputDict['D']/self.inputDict['L']
#         cost = (price_per_m2_module + land_cost_per_m2_module)/(yearly_yield * 25)
#         #print('Cost: {}'.format(cost))
#         return cost
#
#     def calc_lcoe(self, para_list):
#         distance, tilt = para_list
#         #print(distance, tilt)
#         yearly_yield = self.calculate_yield(distance, tilt)
#         return self.calculate_cost(yearly_yield)
#
#     def optimize(self, dist_low=1.65, dist_high=14, tilt_low=1., tilt_high=50.,
#                  ncalls=60):
#
#         #minimal spacing has to at least module length
#         dist_low = max(dist_low, self.module_length)
#
#         self.res = gp_minimize(self.calc_lcoe,
#                                [(dist_low, dist_high), (tilt_low, tilt_high)],
#                                n_random_starts=20, n_jobs=1, n_calls=ncalls,
#                                random_state=1)
# =============================================================================
