# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pvlib
from typing import Union
from pathlib import Path
import json

# Flags for optional libraries
HAS_NETCDF4 = False
HAS_PYARROW = False

# Try importing optional libraries
try:
    import netCDF4
    HAS_NETCDF4 = True
except ImportError:
    pass

try:
    import pyarrow
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    pass

class Illumination:
    def __init__(
        self,
        filepath: Union[str, Path],
        file_format: str = "nsrdb",
        use_perez: bool = False,
    ):
        """
        Read irradiance data from different formats and adds solar position.
        Instances of this class can be directly used for yield simulation and
        optimization.
        Parameters
        ----------
        filepath : filepath
            Filepath to radiation data

        file_format : str
            Currently supported are csv data from the nsrdb and netcdf data from
            copernicus.

        use_perez : boolean
            Determines wether the perez model for circumsolar and horizontal
            brightening should be used.

        """
        if file_format == "nsrdb":
            df = pd.read_csv(filepath, skiprows=2)
            df["dt"] = pd.to_datetime(
                df.Year.astype(str)
                + df.Month.apply("{:0>2}".format)
                + df.Day.apply("{:0>2}".format)
                + df.Hour.apply("{:0>2}".format)
                + df.Minute.apply("{:0>2}".format),
                format="%Y%m%d%H%M",
                utc=True,
            )
            df = df[["dt", "GHI", "DNI", "DHI"]]
            meta = pd.read_csv(filepath, nrows=1).iloc[0]
            self.lat, self.long = meta[["Latitude", "Longitude"]]
            df = df.set_index("dt")

        elif file_format == "copernicus":
            if not HAS_NETCDF4:
                raise ImportError(
                    "The 'netCDF4' library is required for handling 'copernicus' file format. "
                )

            ds = netCDF4.Dataset(filepath)
            df = pd.DataFrame(
                {
                    "GHI": ds["GHI"][:, 0, 0, 0],
                    "DHI": ds["DHI"][:, 0, 0, 0],
                    "DNI": ds["BNI"][:, 0, 0, 0],
                }
            )
            df["dt"] = pd.to_datetime(ds["time"][:], utc=True, unit="s")
            sampling = (
                (df["dt"] - df["dt"].shift(periods=1)).value_counts().index[0]
            )
            df["dt"] = df["dt"] - sampling / 2
            df = df.set_index("dt")
            df = df * (pd.Timedelta("1H") / sampling)
            self.lat, self.long = ds["latitude"][0], ds["longitude"][0]

        elif file_format == "native":
            with pd.HDFStore(filepath) as store:
                self.df = store["table"]
                self.lat = store.root._v_attrs.latitude
                self.long = store.root._v_attrs.longitude
            return None

        elif file_format == "parquet":
            if not HAS_PYARROW:
                raise ImportError(
                    "The 'pyarrow' library is required for loading a Parquet file. "
                )

            table = pq.read_pandas(filepath)
            self.df = table.to_pandas()
            location_metadata = json.loads(
                table.schema.metadata[b"location_metadata"]
            )
            self.lat = location_metadata["latitude"]
            self.long = location_metadata["longitude"]
            return None

        solar_position = pvlib.solarposition.get_solarposition(
            df.index, self.lat, self.long
        )
        self.df = pd.concat(
            [df, solar_position[["zenith", "azimuth"]]], axis=1
        )

        if use_perez:
            raise NotImplementedError("Not yet implemented")

    def save_as_parquet(self, filepath: Union[str, Path]) -> None:
        """
        Writes the DataFrame object as a Parquet file with updated schema metadata.

        This method converts the internal DataFrame representation to a PyArrow Table and updates the schema metadata with latitude and longitude values. It then writes the table to a Parquet file at the specified file path.

        Args:
        filepath (str): The path to the output Parquet file, including the file name.
        """

        if not HAS_PYARROW:
            raise ImportError(
                "The 'pyarrow' library is required for saving as a Parquet file. "
            )

        table = pyarrow.Table.from_pandas(self.df)
        schema_metadata = table.schema.metadata or {}
        schema_metadata[b"location_metadata"] = json.dumps(
            {"latitude": self.lat, "longitude": self.long}
        )
        table = table.replace_schema_metadata(schema_metadata)
        pq.write_table(table, filepath)

    def save_as_native(self, filepath: Union[str, Path]) -> None:
        """
        Saves the DataFrame object as a native HDF5 file with custom attributes for latitude and longitude.

        This method writes the internal DataFrame representation to an HDF5 file at the specified file path, and
        adds custom attributes for latitude and longitude to the root group of the HDF5 file.

        Args:
            filepath (Union[str, Path]): The path to the output HDF5 file, including the file name.
        """
        with pd.HDFStore(filepath) as store:
            store["table"] = self.df
            store.root._v_attrs.latitude = self.lat
            store.root._v_attrs.longitude = self.long


if __name__ == "__main__":
    filepath = "cams_radiation_Berlin.h5"
    berlin_illumination = Illumination(filepath, file_format='native')
    berlin_illumination.save_as_parquet("cams_radiation_Berlin.parquet")
