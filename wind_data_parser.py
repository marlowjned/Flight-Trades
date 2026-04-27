#!/usr/bin/env python3
"""
ERA5 Wind Data Processor (Combined Output)

Variant of wind_data_parser.py that outputs a single CSV with columns:
    pressure (hPa), wind_east (m/s), wind_north (m/s)

Author: Bryan Huang
Date: 2025-08-06
"""

import os
import warnings
from datetime import datetime
from typing import List, Tuple, Optional

import cdsapi
import numpy as np
import xarray as xr

warnings.filterwarnings('ignore')


class ERA5WindProcessor:
    """
    Processes ERA5 reanalysis wind data and saves a single combined CSV.
    """

    # ERA5 available pressure levels (hPa)
    ERA5_PRESSURE_LEVELS = [
        1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700,
        650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 175, 150,
        125, 100, 70, 50, 30, 20, 10, 7, 5, 3, 2, 1
    ]

    def __init__(self,
                 launch_location: Tuple[float, float],
                 years: Optional[List[str]] = None,
                 months: Optional[List[str]] = None,
                 days: Optional[List[str]] = None,
                 times: Optional[List[str]] = None,
                 output_dir: str = "./wind_data"):
        # P1: DEC 7, 2025 ~13:00
        # LADHAD2: MAR 1, 2025 ~10:00
        self.lat, self.lon = launch_location
        self.years = years or ["2025"]
        self.months = months or ["12"]
        self.days = days or ["07"]
        self.times = times or ["13:00"]
        self.output_dir = output_dir

        try:
            self.c = cdsapi.Client()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CDS API client: {e}. "
                               f"Make sure you have configured your CDS API key in ~/.cdsapirc")

        os.makedirs(output_dir, exist_ok=True)

    def download_era5_data(self) -> str:
        delta = 0.01
        area = [
            self.lat + delta,  # North
            self.lon - delta,  # West
            self.lat - delta,  # South
            self.lon + delta   # East
        ]

        filename = os.path.join(self.output_dir, "era5_wind_data.nc")
        pressure_levels_str = [str(p) for p in self.ERA5_PRESSURE_LEVELS]

        request = {
            "product_type": ["reanalysis"],
            "variable": ["u_component_of_wind", "v_component_of_wind"],
            "year": self.years,
            "month": self.months,
            "day": self.days,
            "time": self.times,
            "pressure_level": pressure_levels_str,
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": area
        }

        try:
            print(f"Downloading ERA5 data for location ({self.lat}, {self.lon})...")
            self.c.retrieve("reanalysis-era5-pressure-levels", request).download(filename)
            print(f"Download completed: {filename}")
            return filename
        except Exception as e:
            raise RuntimeError(f"Failed to download ERA5 data: {e}")

    def process_wind_data(self, netcdf_file: str) -> Tuple[np.ndarray, np.ndarray]:
        try:
            ds = xr.open_dataset(netcdf_file)
            ds_point = ds.sel(latitude=self.lat, longitude=self.lon, method='nearest')

            u_wind = ds_point['u'].values  # East component
            v_wind = ds_point['v'].values  # North component

            u_wind = np.nan_to_num(u_wind, nan=0.0, posinf=100.0, neginf=-100.0)
            v_wind = np.nan_to_num(v_wind, nan=0.0, posinf=100.0, neginf=-100.0)

            if u_wind.size == 0 or v_wind.size == 0:
                raise ValueError("Empty wind data arrays")

            u_wind = u_wind.transpose()
            v_wind = v_wind.transpose()

            east_means = np.mean(u_wind, axis=1)
            north_means = np.mean(v_wind, axis=1)

            return east_means, north_means

        except Exception as e:
            raise ValueError(f"Failed to process wind data: {e}")

    def save_combined_csv(self,
                          east_means: np.ndarray,
                          north_means: np.ndarray) -> str:
        """
        Save a single CSV with columns: pressure, wind_east, wind_north.

        Args:
            east_means: Array of eastward wind component means (m/s)
            north_means: Array of northward wind component means (m/s)

        Returns:
            str: Path to the saved CSV file
        """
        pressures = np.array(self.ERA5_PRESSURE_LEVELS, dtype=float)

        # Stack columns: pressure | wind_east | wind_north
        combined = np.column_stack([pressures, east_means, north_means])

        output_file = os.path.join(self.output_dir, "wind_combined.csv")
        header = "pressure,wind_east,wind_north"
        np.savetxt(output_file, combined, delimiter=',', fmt='%.6f', header=header, comments='')

        return output_file

    def run_processing(self) -> str:
        """
        Run the complete wind data processing pipeline.

        Returns:
            str: Path to the combined CSV output file
        """
        try:
            netcdf_file = self.download_era5_data()
            east_means, north_means = self.process_wind_data(netcdf_file)
            output_file = self.save_combined_csv(east_means, north_means)
            return output_file
        except Exception as e:
            raise RuntimeError(f"Processing pipeline failed: {e}")


def main():
    # Configuration parameters
    LAUNCH_LOCATION = (35.35, -117.81)  # FAR launch site

    YEARS = ["2025"]   # in format "2025"
    MONTHS = ["12"]    # in format "03"
    DAYS = ["07"]      # in format "01"
    TIMES = ["13:00"]  # in format "08:00"

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    OUTPUT_DIR_HOME = "./parser_output/recent_output"

    # Clear directory if it already exists
    if os.path.exists(OUTPUT_DIR_HOME):
        import shutil
        shutil.rmtree(OUTPUT_DIR_HOME)
        print(f"Cleared existing directory: {OUTPUT_DIR_HOME}")

    os.makedirs(OUTPUT_DIR_HOME, exist_ok=True)

    print("Starting ERA5 wind data processing (combined CSV output)...")
    print(f"Launch location: {LAUNCH_LOCATION}")
    print(f"Output directory: {OUTPUT_DIR_HOME}")
    print("-" * 50)

    for day in DAYS:
        days = [day]

        for time in TIMES:
            times = [time]

            month_name = datetime.strptime(MONTHS[0], "%m").strftime("%B").lower()
            year = YEARS[0]
            output_dir = (f"{OUTPUT_DIR_HOME}/{month_name}_{days[0]}_{year}_"
                          f"{time.replace(':', '')}_at_{LAUNCH_LOCATION[0]}_{LAUNCH_LOCATION[1]}")

            print(f"Processing: {month_name.capitalize()} {days[0]}, {year} at {time}")

            processor = ERA5WindProcessor(
                launch_location=LAUNCH_LOCATION,
                years=YEARS,
                months=MONTHS,
                days=days,
                times=times,
                output_dir=output_dir
            )

            try:
                output_file = processor.run_processing()
                print("SUCCESS! File generated:")
                print(f"  Combined wind CSV (pressure, wind_east, wind_north): {output_file}")
                print()

            except Exception as e:
                print(f"ERROR processing {month_name.capitalize()} {days[0]} at {time}: {e}")
                print("Make sure you have:")
                print("  1. CDS API key configured (~/.cdsapirc)")
                print("  2. Required Python packages installed:")
                print("     pip install cdsapi numpy xarray netcdf4")
                print("  3. Internet connection for ERA5 download")
                print()

    print("Processing complete!")


if __name__ == "__main__":
    main()
