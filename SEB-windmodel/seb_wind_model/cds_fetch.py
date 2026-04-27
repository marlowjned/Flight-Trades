# cds_fetch.py
# ERA5 download and preprocessing from Copernicus CDS

import calendar
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import PchipInterpolator


PRESSURE_LEVELS = [
    1000, 975, 950, 925, 900, 875, 850, 825, 800,
    775, 750, 700, 650, 600, 550, 500, 450, 400,
    350, 300, 250, 200, 150, 100,
]

_G = 9.80665  # standard gravity (m/s²)

# ISA constants for pressure → altitude conversion
_T0    = 288.15   # K, sea-level temperature
_L     = 0.0065   # K/m, tropospheric lapse rate
_P0    = 101325.0 # Pa, sea-level pressure
_R     = 287.058  # J/(kg·K), specific gas constant for dry air
_T11   = 216.65   # K, isothermal stratosphere temperature
_P11   = 22632.1  # Pa, pressure at 11 km


def _pressure_to_altitude_isa(pressure_hpa: np.ndarray) -> np.ndarray:
    """
    Convert pressure levels (hPa) to approximate geometric altitude (m MSL)
    using the International Standard Atmosphere. Used as fallback when
    geopotential is not present in the ERA5 file.
    """
    p = pressure_hpa * 100.0  # hPa → Pa
    z = np.where(
        p >= _P11,
        # Troposphere
        (_T0 / _L) * (1.0 - (p / _P0) ** (_R * _L / _G)),
        # Stratosphere
        11000.0 + (_R * _T11 / _G) * np.log(_P11 / p),
    )
    return z


def _check_cds_credentials():
    rc = Path.home() / ".cdsapirc"
    if not rc.exists():
        raise EnvironmentError(
            "CDS API credentials not found. Create ~/.cdsapirc with your key:\n"
            "  url: https://cds.climate.copernicus.eu/api/v2\n"
            "  key: <UID>:<API-KEY>\n"
            "Register at https://cds.climate.copernicus.eu to obtain a key."
        )


def _preprocess_ds(
    ds: xr.Dataset,
    alt_grid_m: np.ndarray,
    lat: float,
    lon: float,
    surface_elev_m: float = 0.0,
) -> dict:
    """
    Core preprocessing: xarray Dataset → interpolated (u, v) arrays on a
    uniform altitude grid. Handles both spatially-averaged and single-point
    ERA5 exports.

    Parameters
    ----------
    ds : xr.Dataset
        Opened ERA5 pressure-level dataset.
    alt_grid_m : np.ndarray
        Target altitude grid (m AGL).
    lat, lon : float
        Site coordinates — stored in the output dict for reference only.
    surface_elev_m : float
        Site MSL elevation subtracted from geopotential altitudes so z=0 is AGL.

    Returns
    -------
    dict with keys: u, v, alt_grid, times, lat, lon.
    """
    # Resolve variable names — ERA5 pressure-level fields are 'u', 'v', 'z'
    u_var = "u" if "u" in ds else "u_component_of_wind"
    v_var = "v" if "v" in ds else "v_component_of_wind"
    z_var = "z" if "z" in ds else "geopotential"

    # Spatial dimensions present in this file
    spatial_dims = [d for d in ("latitude", "longitude") if d in ds.dims]

    if spatial_dims:
        u_col = ds[u_var].mean(dim=spatial_dims)
        v_col = ds[v_var].mean(dim=spatial_dims)
    else:
        u_col = ds[u_var].squeeze()
        v_col = ds[v_var].squeeze()

    u_arr = u_col.values.astype(np.float64)
    v_arr = v_col.values.astype(np.float64)

    # Build altitude array: prefer geopotential, fall back to ISA
    if z_var in ds:
        if spatial_dims:
            z_col = ds[z_var].mean(dim=spatial_dims)
        else:
            z_col = ds[z_var].squeeze()
        z_arr = (z_col.values / _G).astype(np.float64)
    else:
        # No geopotential — derive altitude from pressure levels via ISA
        level_dim = next(
            (d for d in u_col.dims if d in ("pressure_level", "level", "isobaricInhPa")),
            u_col.dims[-1],
        )
        pressure_hpa = ds.coords[level_dim].values.astype(np.float64)
        z_isa = _pressure_to_altitude_isa(pressure_hpa)  # (N_levels,)
        # Broadcast to (N_times, N_levels)
        z_arr = np.broadcast_to(z_isa, u_arr.shape).copy()

    # Ensure shape is (N_times, N_levels) — add time axis if missing
    if u_arr.ndim == 1:
        u_arr = u_arr[np.newaxis, :]
        v_arr = v_arr[np.newaxis, :]
        z_arr = z_arr[np.newaxis, :]

    # Forward-fill NaNs along the level dimension
    for arr in (u_arr, v_arr, z_arr):
        for i in range(1, arr.shape[1]):
            mask = np.isnan(arr[:, i])
            arr[mask, i] = arr[mask, i - 1]

    n_times, _ = u_arr.shape
    n_out = len(alt_grid_m)
    u_out = np.empty((n_times, n_out), dtype=np.float64)
    v_out = np.empty((n_times, n_out), dtype=np.float64)

    for t in range(n_times):
        z_agl = z_arr[t] - surface_elev_m
        sort_idx = np.argsort(z_agl)
        z_s = z_agl[sort_idx]
        u_s = u_arr[t, sort_idx]
        v_s = v_arr[t, sort_idx]

        u_out[t] = PchipInterpolator(z_s, u_s, extrapolate=True)(alt_grid_m)
        v_out[t] = PchipInterpolator(z_s, v_s, extrapolate=True)(alt_grid_m)

    # Time coordinate
    time_coord = ds.coords.get("valid_time", ds.coords.get("time", None))
    times = pd.DatetimeIndex(time_coord.values) if time_coord is not None else pd.DatetimeIndex([])

    return {"u": u_out, "v": v_out, "alt_grid": alt_grid_m, "times": times,
            "lat": lat, "lon": lon}


def preprocess_nc(
    nc_path: str,
    alt_grid_m: np.ndarray,
    lat: float,
    lon: float,
    surface_elev_m: float = 0.0,
    output_npz: str = None,
) -> dict:
    """
    Preprocess any ERA5 pressure-level netCDF file into the wind model format.

    Use this when you already have a downloaded .nc file (e.g. from the CDS
    web interface or another tool) and want to convert it without re-downloading.

    Parameters
    ----------
    nc_path : str
        Path to an ERA5 pressure-level netCDF file.
    alt_grid_m : np.ndarray
        Uniform altitude grid (m AGL) to interpolate onto.
    lat, lon : float
        Site coordinates stored in the output for reference.
    surface_elev_m : float
        Site MSL elevation (m). Subtracted from geopotential altitudes.
    output_npz : str, optional
        If provided, save the preprocessed result to this .npz path.

    Returns
    -------
    dict with keys: u, v, alt_grid, times, lat, lon.
    """
    ds = xr.open_dataset(nc_path)
    result = _preprocess_ds(ds, alt_grid_m, lat, lon, surface_elev_m)
    ds.close()

    if output_npz is not None:
        _save_npz(result, output_npz)

    return result


def _save_npz(result: dict, path: str):
    np.savez(
        path,
        u=result["u"],
        v=result["v"],
        alt_grid=result["alt_grid"],
        times=np.array(result["times"], dtype="datetime64[ns]"),
        lat=np.float64(result["lat"]),
        lon=np.float64(result["lon"]),
    )


def load_npz(npz_path: str) -> dict:
    """Load a preprocessed wind profile .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    times = pd.DatetimeIndex(data["times"].astype("datetime64[ns]"))
    return {
        "u": data["u"],
        "v": data["v"],
        "alt_grid": data["alt_grid"],
        "times": times,
        "lat": float(data["lat"]),
        "lon": float(data["lon"]),
    }


def fetch_era5_month(
    year: int,
    month: int,
    lat: float,
    lon: float,
    alt_grid_m: np.ndarray,
    cache_dir: str = "./era5_cache",
    surface_elev_m: float = 0.0,
) -> dict:
    """
    Download one calendar month of ERA5 wind profiles from the CDS and
    preprocess onto a uniform altitude grid.

    Requires a valid ~/.cdsapirc credentials file.
    The raw .nc file is cached at {cache_dir}/era5_{year}_{month:02d}.nc and
    reused on subsequent calls.
    """
    _check_cds_credentials()
    import cdsapi

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    nc_file = cache_path / f"era5_{year}_{month:02d}.nc"

    if not nc_file.exists():
        n_days = calendar.monthrange(year, month)[1]
        area = [lat + 0.5, lon - 0.5, lat - 0.5, lon + 0.5]
        cdsapi.Client().retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type": "reanalysis",
                "variable": ["u_component_of_wind", "v_component_of_wind", "geopotential"],
                "pressure_level": [str(p) for p in PRESSURE_LEVELS],
                "year": str(year),
                "month": f"{month:02d}",
                "day": [f"{d:02d}" for d in range(1, n_days + 1)],
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "area": area,
                "format": "netcdf",
            },
            str(nc_file),
        )

    return preprocess_nc(str(nc_file), alt_grid_m, lat, lon, surface_elev_m)


def load_or_fetch(
    year: int,
    month: int,
    lat: float,
    lon: float,
    alt_grid_m: np.ndarray,
    cache_dir: str = "./era5_cache",
    surface_elev_m: float = 0.0,
) -> dict:
    """
    Load preprocessed data from .npz cache if available; otherwise fetch and cache.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    npz_file = cache_path / f"era5_{year}_{month:02d}_preprocessed.npz"

    if npz_file.exists():
        return load_npz(str(npz_file))

    result = fetch_era5_month(year, month, lat, lon, alt_grid_m, cache_dir, surface_elev_m)
    _save_npz(result, str(npz_file))
    return result
