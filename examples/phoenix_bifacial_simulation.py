# -*- coding: utf-8 -*-

import pandas as pd
import pvlib
import bifacial_illumination as bi

df = pd.read_csv('Phoenix_tmy_2020.csv', skiprows=2)
metadata = pd.read_csv('Phoenix_tmy_2020.csv', nrows=1).iloc[0]

df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])

solar_position = pvlib.solarposition.get_solarposition(df['datetime'],
                                                       metadata['Latitude'],
                                                       metadata['Longitude'])

df[['zenith', 'azimuth']] = solar_position.reset_index()[['zenith', 'azimuth']]

simulator = bi.YieldSimulator(df)

df_res = simulator.simulate(tilt=15, spacing=5)
df_res = df_res.groupby(level="contribution", axis=1).mean().sum(axis=1)

df_res.index = df['datetime']
df_res.groupby(df_res.index.dayofyear).sum().plot()


