#!/usr/bin/env python
"""
validate_model.py — Diagnostic plots for the EOF wind model and Von Kármán filter.

Run after fetch_month.py. Produces four plots in ./diagnostics/:
  - eof_variance.png      Scree plot
  - ensemble_spread.png   Ensemble u-profiles vs ERA5 raw profiles
  - vk_psd.png            Von Kármán PSD vs theoretical
  - altitude_params.png   sigma_u(z) and L(z) vs altitude
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

from seb_wind_model.utils import uniform_alt_grid
from seb_wind_model.cds_fetch import load_or_fetch
from seb_wind_model.eof import EOFModel
from seb_wind_model.von_karman import VonKarmanFilter, sigma_u, scale_length, L_FREE_ATM
from seb_wind_model.wind_model import WindModel


def plot_eof_variance(eof_model: EOFModel, out_dir: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    modes = np.arange(1, eof_model.n_modes + 1)
    cumvar = eof_model.variance_explained() * 100

    ax.plot(modes, cumvar, "o-", color="steelblue")
    ax.axhline(95, color="red", linestyle="--", label="95% threshold")
    ax.set_xlabel("Mode number")
    ax.set_ylabel("Cumulative variance explained (%)")
    ax.set_title("EOF Scree Plot")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "eof_variance.png"), dpi=150)
    plt.close(fig)
    print("  Saved eof_variance.png")


def plot_ensemble_spread(eof_model: EOFModel, data: dict, out_dir: str):
    alt = eof_model.alt_grid / 1000  # km
    model = WindModel(eof_model, dt=0.05)
    members = model.ensemble(n=50)

    fig, ax = plt.subplots(figsize=(6, 8))

    # Raw ERA5 profiles (red, behind)
    for i in range(min(50, data["u"].shape[0])):
        ax.plot(data["u"][i], alt, color="red", alpha=0.15, linewidth=0.5)

    # Ensemble members (grey)
    u_ensemble = np.array([m.base_wind(z)[0]
                           for m in members
                           for z in eof_model.alt_grid]).reshape(50, -1)

    for i in range(50):
        ax.plot(u_ensemble[i], alt, color="grey", alpha=0.25, linewidth=0.5)

    mean_u = u_ensemble.mean(axis=0)
    std_u  = u_ensemble.std(axis=0)
    ax.plot(mean_u, alt, color="black", linewidth=2, label="Ensemble mean")
    ax.fill_betweenx(alt, mean_u - std_u, mean_u + std_u,
                     alpha=0.3, color="steelblue", label="±1σ")

    ax.set_xlabel("u-wind (m/s)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title("Ensemble Spread vs ERA5 Profiles")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ensemble_spread.png"), dpi=150)
    plt.close(fig)
    print("  Saved ensemble_spread.png")


def plot_vk_psd(out_dir: str):
    dt = 0.01
    V = 100.0
    z = 500.0
    n_steps = int(60.0 / dt)

    filt = VonKarmanFilter(dt=dt, airspeed=V, z_m=z)
    rng = np.random.default_rng(0)
    u_ts = np.array([filt.step(rng)[0] for _ in range(n_steps)])

    freqs, psd = welch(u_ts, fs=1.0 / dt, nperseg=1024)
    # Avoid DC
    mask = freqs > 0

    # Theoretical Von Kármán PSD
    sigma = sigma_u(z)
    L = scale_length(z)
    f = freqs[mask]
    phi_theory = (sigma ** 2 * (2 * L / V) /
                  (1 + (1.339 * L * 2 * np.pi * f / V) ** 2) ** (5 / 6))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(f, psd[mask], color="steelblue", alpha=0.7, label="Simulated PSD")
    ax.loglog(f, phi_theory, color="red", linestyle="--", linewidth=2,
              label="Theoretical Von Kármán")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (m²/s² / Hz)")
    ax.set_title(f"Von Kármán PSD  (z={z} m, V={V} m/s)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "vk_psd.png"), dpi=150)
    plt.close(fig)
    print("  Saved vk_psd.png")


def plot_altitude_params(out_dir: str):
    z_vals = np.linspace(0, 15000, 500)
    sigma_vals = np.array([sigma_u(z) for z in z_vals])
    L_vals     = np.array([scale_length(z) for z in z_vals])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    ax1.plot(sigma_vals, z_vals / 1000, color="steelblue")
    ax1.axhline(0.305, color="red", linestyle="--", linewidth=0.8,
                label="305 m transition")
    ax1.set_xlabel("σ_u (m/s)")
    ax1.set_ylabel("Altitude (km)")
    ax1.set_title("Turbulence Intensity")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(L_vals, z_vals / 1000, color="darkorange")
    ax2.axhline(0.305, color="red", linestyle="--", linewidth=0.8,
                label="305 m transition")
    ax2.set_xlabel("L_u (m)")
    ax2.set_title("Scale Length")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "altitude_params.png"), dpi=150)
    plt.close(fig)
    print("  Saved altitude_params.png")


def main():
    parser = argparse.ArgumentParser(description="Validate the ERA5 EOF wind model.")
    parser.add_argument("--year",   type=int, required=True)
    parser.add_argument("--month",  type=int, required=True)
    parser.add_argument("--lat",    type=float, required=True)
    parser.add_argument("--lon",    type=float, required=True)
    parser.add_argument("--dz",     type=float, default=50.0)
    parser.add_argument("--surface-elev", type=float, default=0.0,
                        dest="surface_elev")
    parser.add_argument("--cache-dir", type=str, default="./era5_cache",
                        dest="cache_dir")
    parser.add_argument("--out-dir", type=str, default="./diagnostics",
                        dest="out_dir")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    alt_grid = uniform_alt_grid(z_min=0.0, z_max=20000.0, dz=args.dz)
    print(f"Loading ERA5 data for {args.year}-{args.month:02d}...")
    data = load_or_fetch(args.year, args.month, args.lat, args.lon,
                         alt_grid, args.cache_dir, args.surface_elev)

    print("Fitting EOF model...")
    eof_model = EOFModel(data["u"], data["v"], data["alt_grid"])
    print(f"  Retained {eof_model.n_modes} modes "
          f"({eof_model.variance_explained()[-1]*100:.1f}% variance)")

    print("Generating diagnostic plots...")
    plot_eof_variance(eof_model, args.out_dir)
    plot_ensemble_spread(eof_model, data, args.out_dir)
    plot_vk_psd(args.out_dir)
    plot_altitude_params(args.out_dir)

    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
