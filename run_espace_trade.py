"""
run_espace_trade.py
===================
ESPACE Thrust Trade Study Runner.

For each thrust level (8–14 kN in 1 kN steps):
  1. Sizes propellant / tank (bisection, 1-D no-drag model)
  2. Generates an interpolated RasAero aero-table CSV if not already present
  3. Runs N Monte Carlo wind trials via SimulationHandler (temp YAML config)
  4. Aggregates apogee + landing-dispersion statistics across trials
  5. Saves per-config results CSVs + a single aggregate CSV

Usage (from project root):
    # Full run — 30 MC trials per thrust level, SEB/ERA5 wind
    python run_espace_trade.py

    # Smoke test — 1 trial, simple turbulent wind (no ERA5 file needed)
    python run_espace_trade.py --trials 1 --simple-wind

    # Custom trial count with ERA5 wind
    python run_espace_trade.py --trials 10

    # Export full flight snapshots for the 8 kN case (perm 0, trial 0)
    python run_espace_trade.py --trials 1 --simple-wind --snapshots results/espace_trade/F08kN_snap.csv
"""

import argparse
import math
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, '6DOF'))
sys.path.insert(0, os.path.join(ROOT, 'user_inputs', 'rocket_data'))

from flight_sim.core.sim_handler import SimulationHandler   # noqa: E402
from generate_aero_csvs import (                            # noqa: E402
    load_stability_interpolator, make_aero_csv,
)
from espace_sizing import (                                  # noqa: E402
    THRUST_VALUES_N, TARGET_ALT_M, MDOT_NOM, F_NOM,
    find_propellant_mass, compute_tank_lengths,
)

# ── Constants ─────────────────────────────────────────────────────────────────
ROCKET_DATA_DIR = os.path.join(ROOT, 'user_inputs', 'rocket_data')
RESULTS_DIR     = os.path.join(ROOT, 'results', 'espace_trade')

# ERA5 wind data (same dataset used for p1 trade study)
ERA5_PATH    = os.path.join(ROOT, 'data', 'wind',
               'december_07_2025_1300_at_35.35_-117.81', 'era5_wind_data.nc')
ERA5_LAT     = 35.35
ERA5_LON     = -117.81
SURF_ELEV_M  = 700.0

# Reference area — 8" OD body tube
REF_AREA_M2  = math.pi / 4 * (8 * 0.0254) ** 2   # ≈ 0.032429 m²


# ── Config helpers ────────────────────────────────────────────────────────────
def _wind_block(simple: bool) -> dict:
    if simple:
        return {
            'type': 'simple',
            'magnitude': 5.0,
            'direction': 0.0,
            'alt_min': 0.0,
            'alt_max': 15000.0,
            'alt_steps': 100,
            'turbulence_intensity': 1.0,
        }
    return {
        'type': 'SEB-windmodel',
        'data_path': ERA5_PATH,
        'lat': ERA5_LAT,
        'lon': ERA5_LON,
        'surface_elev_m': SURF_ELEV_M,
        'alt_max_m': 15000.0,
        'alt_dz_m': 50.0,
        'scale': 1.0,
        'n_modes': None,
    }


def _build_config(ork_fname, aero_fname, n_trials, simple_wind) -> dict:
    return {
        'simulation': {
            'type': 'Trade',
            'dt': 0.05,
            'max_runtime': 1500,        # 25 min sim time; spaceshot + descent
            'iterations_per_trial': n_trials,
            'recovery_sim': True,
            'launch_rail_length': 5.0,
            'launchrail_orientation': [0, 0, 1],
        },
        'rocket': {
            'ORK_path':         'user_inputs/rocket_data/',
            'ORK_filename':     ork_fname,
            'RasAero_path':     'user_inputs/rocket_data/',
            'RasAero_filename': aero_fname,
            'reference_area':   REF_AREA_M2,
        },
        'recovery': {
            'CdA': [1.0, 10.0],                  # drogue, main  (UPDATE w/ actual CdA)
            'deployment_altitude': [None, 500],   # drogue at apogee, main at 500 m
        },
        'wind': _wind_block(simple_wind),
        'record': ['apogee', 'landing_x', 'landing_y', 'max_mach'],
    }


# ── Per-config runner ─────────────────────────────────────────────────────────
def _run_config(tag, ork_fname, aero_fname, n_trials, simple_wind,
                snap_path=None):
    cfg = _build_config(ork_fname, aero_fname, n_trials, simple_wind)

    # Write a temporary YAML so SimulationHandler can load it
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml',
                                     delete=False, dir=RESULTS_DIR) as tf:
        yaml.dump(cfg, tf, default_flow_style=False, allow_unicode=True)
        tmp_yaml = tf.name

    try:
        handler = SimulationHandler(tmp_yaml)
        results = handler.run()
        per_csv = os.path.join(RESULTS_DIR, f'{tag}_results.csv')
        handler.export_csv(per_csv)
        if snap_path:
            handler.export_snapshots_csv(snap_path)
            print(f'  Snapshots → {snap_path}')
        return results, per_csv
    finally:
        os.unlink(tmp_yaml)


# ── Statistics helpers ────────────────────────────────────────────────────────
def _print_stats(results):
    apogees = [r['apogee']    for r in results if r.get('apogee')    is not None]
    lx      = [r['landing_x'] for r in results if r.get('landing_x') is not None]
    ly      = [r['landing_y'] for r in results if r.get('landing_y') is not None]

    if apogees:
        apo_ft = np.array(apogees) / 0.3048
        print(f'  Apogee  : med={np.median(apo_ft):>10,.0f} ft  '
              f'min={np.min(apo_ft):>10,.0f}  max={np.max(apo_ft):>10,.0f}')
    if lx and ly:
        disp_km = np.sqrt(np.array(lx) ** 2 + np.array(ly) ** 2) / 1000
        print(f'  Landing : med={np.median(disp_km):>6.2f} km  '
              f'p95={np.percentile(disp_km, 95):>6.2f} km  '
              f'max={np.max(disp_km):>6.2f} km')


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='ESPACE thrust trade study.')
    parser.add_argument('--trials', type=int, default=30,
                        help='Monte Carlo trials per thrust level (default: 30)')
    parser.add_argument('--simple-wind', action='store_true',
                        help='Simple turbulent wind instead of SEB/ERA5')
    parser.add_argument('--snapshots', default=None,
                        help='Export full flight snapshots CSV for the first '
                             'thrust config (perm 0, trial 0)')
    parser.add_argument('--thrust', type=int, default=None, nargs='+',
                        help='Run only specific thrust values, e.g. --thrust 8000 10000')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    thrusts = args.thrust if args.thrust else THRUST_VALUES_N

    # Load stability tables once for all configs
    print('Loading stability tables …')
    interp_fn, col_names, nrows, ncols = load_stability_interpolator()
    print(f'Ready.\n')

    all_rows = []

    for i, F_N in enumerate(thrusts):
        tag = f'espace_F{int(F_N / 1000):02d}kN'
        print(f"{'─'*65}")
        print(f'Thrust = {F_N} N  [{tag}]  ({args.trials} trial(s))')

        # Sizing
        m_prop_g    = find_propellant_mass(F_N, TARGET_ALT_M)
        L_NOS_ft, _ = compute_tank_lengths(m_prop_g)
        m_dot       = MDOT_NOM * (F_N / F_NOM)
        t_burn      = (m_prop_g / 1000.0) / m_dot
        print(f'  m_prop={m_prop_g/1000:.2f} kg  '
              f'L_NOS={L_NOS_ft:.2f} ft  '
              f't_burn={t_burn:.1f} s')

        # Aero CSV
        aero_path = os.path.join(ROCKET_DATA_DIR, f'{tag}_aero.csv')
        if not os.path.exists(aero_path):
            make_aero_csv(interp_fn, col_names, nrows, ncols, L_NOS_ft, aero_path)
            print(f'  Generated aero CSV → {aero_path}')
        else:
            print(f'  Aero CSV exists    → {aero_path}')

        # Snapshot path only for first thrust in list (perm 0 trial 0)
        snap = args.snapshots if (i == 0 and args.snapshots) else None

        # Run
        results, per_csv = _run_config(
            tag,
            ork_fname=f'{tag}.csv',
            aero_fname=f'{tag}_aero.csv',
            n_trials=args.trials,
            simple_wind=args.simple_wind,
            snap_path=snap,
        )
        print(f'  Results → {per_csv}')
        _print_stats(results)

        for r in results:
            r['thrust_N']  = F_N
            r['L_NOS_ft']  = round(L_NOS_ft, 3)
            r['m_prop_kg'] = round(m_prop_g / 1000, 2)
            r['t_burn_s']  = round(t_burn, 2)
        all_rows.extend(results)

    # Aggregate CSV
    if all_rows:
        df = pd.DataFrame(all_rows)
        agg_csv = os.path.join(RESULTS_DIR, 'espace_trade_aggregate.csv')
        df.to_csv(agg_csv, index=False)

        # Summary table
        print(f"\n{'='*70}")
        print('SUMMARY TABLE')
        print(f"{'='*70}")
        hdr = (f"{'F(N)':>6} | {'L_NOS':>6} | {'m_prop':>7} | {'t_burn':>7} | "
               f"{'Apogee_med':>12} | {'Disp_med':>9} | {'Disp_p95':>9}")
        units = (f"{'':>6} | {'(ft)':>6} | {'(kg)':>7} | {'(s)':>7} | "
                 f"{'(ft)':>12} | {'(km)':>9} | {'(km)':>9}")
        print(hdr)
        print(units)
        print('─' * 70)
        for F_N in thrusts:
            sub = df[df['thrust_N'] == F_N]
            if sub.empty:
                continue
            apo  = sub['apogee'].dropna().values / 0.3048
            disp = (np.sqrt(sub['landing_x'].dropna() ** 2 +
                            sub['landing_y'].dropna() ** 2) / 1000)
            print(
                f"{F_N:>6.0f} | {sub['L_NOS_ft'].iloc[0]:>6.2f} | "
                f"{sub['m_prop_kg'].iloc[0]:>7.2f} | "
                f"{sub['t_burn_s'].iloc[0]:>7.1f} | "
                f"{np.median(apo):>12,.0f} | "
                f"{np.median(disp.values):>9.2f} | "
                f"{np.percentile(disp.values, 95):>9.2f}"
            )
        print(f'\nAggregate → {agg_csv}')


if __name__ == '__main__':
    main()
