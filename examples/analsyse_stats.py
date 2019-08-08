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
from glob2 import glob

# =============================================================================
# file_list = ['opt_stats_dallas_bi.csv',
#              'opt_stats_dallas_front.csv',
#              'opt_stats_seattle_bi.csv',
#              'opt_stats_seattle_front.csv']
# =============================================================================

# =============================================================================
# file_list = glob('./stats/opt_stats_*.csv')
#
# df_list = []
# for file in file_list:
#     df_list.append(
#             pd.read_csv(file, index_col=0)
#             .reset_index()
#             .rename(columns={'index':'cost_szenario'})
#             )
# =============================================================================

df = pd.read_csv('stats/opt_stats_dallas.csv', index_col=0).reset_index(drop=True)

#df = pd.concat(df_list, keys=[file.split('.csv')[0] for file in file_list])
print(df)
df = df.query('location=="dallas"')

contour_data = df.query('type=="bi"').set_index(['cost_scenario', 'albedo']).unstack('cost_scenario')
contour_cost = contour_data['cost']
plt.contourf(contour_cost.columns, contour_cost.index, contour_cost)

contour_bigain= df.query('type=="bi"')\
                  .set_index(['cost_scenario', 'albedo'])\
                  .eval('back/front')\
                  .unstack('cost_scenario')

plt.contourf(contour_bigain.columns, contour_bigain.index, contour_bigain)

df['rel_land_cost'] = df.eval('land_cost/cost')*100
df['bifacial_gain'] = df.eval('back/front')*100

df.query('(albedo==0.3)&(type=="bi")')[['cost_scenario', 'cost', 'rel_land_cost', 'dist',
        'tilt','bifacial_gain']].round(1).to_latex(index=False)

df.query('(albedo==0.3)&(type=="mono")')[['cost_scenario', 'cost', 'rel_land_cost', 'dist',
        'tilt','bifacial_gain']].round(1).to_latex(index=False, sparsify=True).strip()

asdf
df.head()
df['albedo'] = df

asdf

df = df.reset_index().set_index(['level_0','cost_szenario']).drop('level_1', axis=1)

cost_data = df['cost'].unstack('level_0')
tilt_data = df['tilt'].unstack('level_0')

plt.rcParams['font.size'] = 8.0
plt.rcParams['text.latex.preamble'] = r'\usepackage{arev}'
plt.rc('text', usetex=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 2.5))

# =============================================================================
# l1 = ax1.plot(cost_data.loc[:,'opt_stats_dallas_bi'], label='Dallas bifacial')
# l2 = ax1.plot(cost_data.loc[:,'opt_stats_dallas_front'], label='Dallas front')
# l3 = ax1.plot(cost_data.loc[:,'opt_stats_seattle_bi'], label='Seattle bifacial')
# l4 = ax1.plot(cost_data.loc[:,'opt_stats_seattle_front'], label='Seattle front')
# =============================================================================
# =============================================================================
# ax1.plot(cost_data.loc[:,'opt_stats_dallas_bi'])
# ax1.plot(cost_data.loc[:,'opt_stats_dallas_front'])
# ax1.plot(cost_data.loc[:,'opt_stats_seattle_bi'])
# ax1.plot(cost_data.loc[:,'opt_stats_seattle_front'])
# =============================================================================
l1, = ax1.plot(cost_data.loc[:,'opt_stats_dallas_bi'], c='crimson',)
l2, = ax1.plot(cost_data.loc[:,'opt_stats_dallas_front'], c='crimson', linestyle='--')
l3, = ax1.plot(cost_data.loc[:,'opt_stats_seattle_bi'], c='b',)
l4, = ax1.plot(cost_data.loc[:,'opt_stats_seattle_front'], C='b', linestyle='--')
ax1.set_ylim((3.5, 8))
ax1.set_ylabel('LCOE (cents/kWh)')
ax1.set_xlabel('land cost (\$/m$^2$)')
ax1.set_title(r'\textbf{(a)}')

ax2.plot(tilt_data.loc[:,'opt_stats_dallas_bi'], c='crimson',)
ax2.plot(tilt_data.loc[:,'opt_stats_dallas_front'], c='crimson', linestyle='--')
ax2.plot(tilt_data.loc[:,'opt_stats_seattle_bi'], c='b',)
ax2.plot(tilt_data.loc[:,'opt_stats_seattle_front'], C='b', linestyle='--')
ax2.set_ylim((10, 50))
ax2.set_ylabel('optimal tilt (deg)')
ax2.set_xlabel('land cost (\$/m$^2$)')
ax2.set_title(r'\textbf{(b)}')
fig.subplots_adjust(right=0.80, left=0.05, bottom=0.18)
fig.legend((l1, l2, l3, l4), ('Dallas bifacial', 'Dallas monofacial', 'Seattle bifacial', 'Seattle monofacial'),
           (0.81, 0.61),frameon=False)
#plt.tight_layout()
plt.savefig('opt_lcoe_tilt.pdf', format='pdf', dpi=300)
