#!/usr/bin/env python
"""
fetch_month.py — Download and preprocess one month of ERA5 wind data.

Usage:
    python fetch_month.py --year 2023 --month 6 --lat 35.0 --lon -117.8 --dz 50
"""

import argparse
import numpy as np
from seb_wind_model.utils import uniform_alt_grid, wind_speed
from seb_wind_model.cds_fetch import load_or_fetch


def main():
    parser = argparse.ArgumentParser(
        description="Download and preprocess one month of ERA5 wind profiles."
    )
    parser.add_argument("--year",   type=int, required=True)
    parser.add_argument("--month",  type=int, required=True)
    parser.add_argument("--lat",    type=float, required=True,
                        help="Site latitude (decimal degrees)")
    parser.add_argument("--lon",    type=float, required=True,
                        help="Site longitude (decimal degrees)")
    parser.add_argument("--dz",     type=float, default=50.0,
                        help="Altitude grid step in metres (default: 50)")
    parser.add_argument("--zmax",   type=float, default=20000.0,
                        help="Maximum altitude in metres (default: 20000)")
    parser.add_argument("--surface-elev", type=float, default=0.0,
                        dest="surface_elev",
                        help="Site elevation MSL in metres (default: 0)")
    parser.add_argument("--cache-dir", type=str, default="./era5_cache",
                        dest="cache_dir")
    args = parser.parse_args()

    alt_grid = uniform_alt_grid(z_min=0.0, z_max=args.zmax, dz=args.dz)

    print(f"Fetching ERA5 data for {args.year}-{args.month:02d} at "
          f"({args.lat:.2f}, {args.lon:.2f})...")

    data = load_or_fetch(
        year=args.year,
        month=args.month,
        lat=args.lat,
        lon=args.lon,
        alt_grid_m=alt_grid,
        cache_dir=args.cache_dir,
        surface_elev_m=args.surface_elev,
    )

    u, v = data["u"], data["v"]
    n_profiles = u.shape[0]
    print(f"\n  Profiles loaded : {n_profiles}")
    print(f"  Altitude range  : {alt_grid[0]:.0f} – {alt_grid[-1]:.0f} m AGL "
          f"({len(alt_grid)} levels)")

    for z_check in [500.0, 2000.0, 5000.0]:
        idx = int(np.argmin(np.abs(alt_grid - z_check)))
        spd = wind_speed(u[:, idx], v[:, idx])
        print(f"  Mean wind speed at {z_check:.0f} m : "
              f"{spd.mean():.1f} m/s  (std {spd.std():.1f} m/s)")

    out_path = (
        f"{args.cache_dir}/era5_{args.year}_{args.month:02d}_preprocessed.npz"
    )
    print(f"\nPreprocessed data saved to {out_path}")


if __name__ == "__main__":
    main()
