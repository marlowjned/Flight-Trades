"""
run_new_trade.py
================
Trade study runner for the redesigned stable ESPACE airframes.

Usage:
    python run_new_trade.py                        # v2, all configs, 1 trial, simple wind
    python run_new_trade.py --ver v1               # use v1 aero tables / rocket CSV
    python run_new_trade.py --trials 30            # full MC, simple wind
    python run_new_trade.py --thrust 10000         # override thrust
    python run_new_trade.py --era5                 # use SEB/ERA5 wind (needs .nc file)
"""

import argparse
import math
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, '6DOF'))

from flight_sim.core.sim_handler import SimulationHandler  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
NEW_AERO_DIR    = os.path.join(ROOT, 'data', 'new-aero-tables')
ROCKET_DATA_DIR = os.path.join(ROOT, 'user_inputs', 'rocket_data')
RESULTS_DIR     = os.path.join(ROOT, 'results', 'new_trade')

# ── Version-specific config ────────────────────────────────────────────────────
# Each entry: dict with keys
#   rocket_csv  – path to rocket config CSV
#   aero_dir    – directory containing aero table CSVs
#   tanks       – list of tank lengths (ft), in CSV row order
#   aero_name   – callable(ft) → filename inside aero_dir
#   cols        – dict mapping logical keys to actual column names
#                 (strip=True strips leading/trailing whitespace first)
#   strip       – whether to strip column names before lookup

_PREFIX = 'Copy of Stability Analysis + Upthrust Info  - '

VER_CONFIGS = {
    'v0': {
        'rocket_csv': os.path.join(NEW_AERO_DIR,
                      _PREFIX + 'Sheet9.csv'),
        'aero_dir':   NEW_AERO_DIR,
        'tanks':      [8, 12, 16],
        'aero_name':  lambda ft: _PREFIX + f'[AERO-TABLE] {ft}ft.csv',
        'cols': {
            'wet_mass': 'Wet Mass (g)',
            'wet_cg':   'Wet CG (cm)',
            'wet_I':    'Wet I_xx/I_yy',
            'dry_mass': 'Dry Mass',
            'dry_cg':   'Dry CG (cm)',
            'dry_I':    'Dry I_xx/I_yy',
        },
        'strip': False,
    },
    'v1': {
        'rocket_csv': os.path.join(NEW_AERO_DIR, 'v1',
                      _PREFIX + '[v1] rocket.csv'),
        'aero_dir':   os.path.join(NEW_AERO_DIR, 'v1'),
        'tanks':      [8, 12, 16],
        'aero_name':  lambda ft: _PREFIX + f'[v1-AT] {ft}ft.csv',
        'cols': {
            'wet_mass': 'Wet Mass (g)',
            'wet_cg':   'CG (cm)',
            'wet_I':    'I_long (g cm2)',
            'dry_mass': 'Dry Mass (g)',
            'dry_cg':   'CG (cm).1',
            'dry_I':    'I_long (g cm2).1',
        },
        'strip': True,
    },
    'v2': {
        'rocket_csv': os.path.join(NEW_AERO_DIR, 'v2',
                      _PREFIX + '[v2] rocket.csv'),
        'aero_dir':   os.path.join(NEW_AERO_DIR, 'v2'),
        'tanks':      [12, 16, 20],
        # rocket CSV rows are 8/12/16/20ft in order; map ft→row index explicitly
        'tank_row':   {8: 0, 12: 1, 16: 2, 20: 3},
        'aero_name':  lambda ft: _PREFIX + f'[v2-AT] {ft}ft.csv',
        # Artificial CP offsets (calibers, aft-positive) applied before sim.
        # D_ref = 8 in, so 1 cal = 8 in added to the 'CP' column.
        'cp_offsets_cal': {20: 1.0},
        'cols': {
            'wet_mass': 'Wet Mass',
            'wet_cg':   'Wet CG',
            'wet_I':    'Wet I_long',
            'dry_mass': 'Dry Mass',
            'dry_cg':   'Dry CG (cm)',
            'dry_I':    'Dry I_long',
        },
        'strip': True,
    },
}

ERA5_PATH   = os.path.join(ROOT, 'data', 'wind',
              'december_07_2025_1300_at_35.35_-117.81', 'era5_wind_data.nc')
ERA5_LAT    = 35.35
ERA5_LON    = -117.81
SURF_ELEV_M = 700.0

# Reference area — 8" OD body tube
D_REF_M    = 8 * 0.0254          # 0.2032 m
REF_AREA   = math.pi / 4 * D_REF_M ** 2   # ≈ 0.03243 m²
R_OUTER_M  = D_REF_M / 2         # for I_roll estimate

# Engine reference (constant Isp scaling)
G0       = 9.80665   # m/s²
F_NOM    = 8_000.0   # N
MDOT_NOM = 4.1       # kg/s  (Isp ≈ 198.7 s)
ISP      = F_NOM / (MDOT_NOM * G0)

# MOI unit conversion: Sheet9 stores I in g·cm²; ork_loader expects kg·m²
GCM2_TO_KGM2 = 1e-7

# ── Tumble-abort settings ──────────────────────────────────────────────────────
# Abort a trial early if the rocket is tumbling at high speed.
# Tumbling = alpha > threshold AND speed > threshold.
# This saves the long ballistic coast of a tumbled trajectory.
TUMBLE_ALPHA_DEG  = 45.0   # degrees — clearly off-axis
TUMBLE_SPEED_M_S  = 300.0  # m/s    — ~Mach 0.9 at sea level; above this abort matters

# Heartbeat interval: print intra-trial progress every N wall-clock seconds
HEARTBEAT_INTERVAL_S = 4.0

def _make_abort_fn():
    """
    Returns a stateful abort/heartbeat callable for one trial.
    Prints altitude/Mach every HEARTBEAT_INTERVAL_S wall-clock seconds,
    and signals abort if the rocket is tumbling during ascent.
    """
    _last_hb = [time.perf_counter()]

    def fn(snap):
        now = time.perf_counter()
        if now - _last_hb[0] >= HEARTBEAT_INTERVAL_S:
            print(f"      ... t={snap.time:6.1f}s  "
                  f"alt={snap.altitude/0.3048:>9,.0f} ft  "
                  f"Mach={snap.mach:.2f}  α={snap.alpha_deg:.1f}°",
                  flush=True)
            _last_hb[0] = now
        return (snap.vel_z > 0
                and snap.speed > TUMBLE_SPEED_M_S
                and snap.alpha_deg > TUMBLE_ALPHA_DEG)

    return fn


# ── ORK CSV generation ────────────────────────────────────────────────────────

def _write_ork_csv(path, F_N, wet_mass_g, wet_cg_cm, wet_I_long_gcm2,
                   dry_mass_g, dry_cg_cm, dry_I_long_gcm2, n_points=100):
    """
    Linear interpolation from wet → dry over the burn, then constant at dry.
    I_roll estimated from tube outer radius (much smaller than I_long; sim
    uses it only for roll dynamics which are negligible for a symmetric rocket).
    """
    mdot    = MDOT_NOM * (F_N / F_NOM)
    prop_g  = wet_mass_g - dry_mass_g
    t_burn  = (prop_g / 1000.0) / mdot

    wet_I = wet_I_long_gcm2 * GCM2_TO_KGM2
    dry_I = dry_I_long_gcm2 * GCM2_TO_KGM2

    header = ("# Time (s),Mass (g),Motor mass (g),"
              "Longitudinal moment of inertia (kg\u00b7m\u00b2),"
              "Rotational moment of inertia (kg\u00b7m\u00b2),"
              "CP location (cm),CG location (cm),Thrust (N)")

    rows = []
    times = np.linspace(0, t_burn, n_points)
    for t in times:
        frac = 1.0 - t / t_burn   # 1.0 at liftoff → 0.0 at burnout
        mass_g  = dry_mass_g  + frac * prop_g
        cg_cm   = wet_cg_cm   + (1 - frac) * (dry_cg_cm  - wet_cg_cm)
        I_long  = wet_I       + (1 - frac) * (dry_I      - wet_I)
        I_roll  = (mass_g / 1000.0) * R_OUTER_M ** 2 / 2.0
        rows.append((t, mass_g, frac * prop_g, I_long, I_roll, float('nan'), cg_cm, F_N))

    # Burnout + far-future hold
    I_roll_dry = (dry_mass_g / 1000.0) * R_OUTER_M ** 2 / 2.0
    rows.append((t_burn + 0.001, dry_mass_g, 0.0,
                 dry_I, I_roll_dry, float('nan'), dry_cg_cm, 0.0))
    rows.append((t_burn + 600.0, dry_mass_g, 0.0,
                 dry_I, I_roll_dry, float('nan'), dry_cg_cm, 0.0))

    with open(path, 'w') as fh:
        fh.write(header + '\n')
        for t, mass, mot, ilong, iroll, cp, cg, thr in rows:
            fh.write(f"{t:.4f},{mass:.2f},{mot:.2f},"
                     f"{ilong:.4f},{iroll:.5f},"
                     f"NaN,{cg:.3f},{thr:.1f}\n")

    return t_burn


# ── Sim config helpers ────────────────────────────────────────────────────────

def _wind_block(era5: bool) -> dict:
    if not era5:
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


def _build_config(ork_fname, aero_fname, n_trials, era5, dt=0.05, dt_coast=None) -> dict:
    sim_block = {
        'type': 'Trade',
        'dt': dt,
        'max_runtime': 1500,
        'iterations_per_trial': n_trials,
        'recovery_sim': True,
        'launch_rail_length': 5.0,
        'launchrail_orientation': [0, 0, 1],
    }
    if dt_coast is not None:
        sim_block['dt_coast'] = dt_coast
    return {
        'simulation': sim_block,
        'rocket': {
            'ORK_path':         'user_inputs/rocket_data/',
            'ORK_filename':     ork_fname,
            'RasAero_path':     'user_inputs/rocket_data/',
            'RasAero_filename': aero_fname,
            'reference_area':   REF_AREA,
        },
        'recovery': {
            'CdA': [1.0, 10.0],
            'deployment_altitude': [None, 500],
        },
        'wind': _wind_block(era5),
        'record': ['apogee', 'landing_x', 'landing_y', 'max_mach'],
    }


def _run_config(tag, ork_fname, aero_fname, n_trials, era5, dt=0.05, dt_coast=None, trial_callback=None):
    cfg = _build_config(ork_fname, aero_fname, n_trials, era5, dt=dt, dt_coast=dt_coast)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml',
                                     delete=False, dir=RESULTS_DIR) as tf:
        yaml.dump(cfg, tf, default_flow_style=False, allow_unicode=True)
        tmp_yaml = tf.name

    # Each trial gets a fresh abort/heartbeat fn so the timer resets
    # We intercept the per-trial boundary via a wrapper around trial_callback
    _trial_abort_fns = []

    def _per_trial_abort_wrapper(snap):
        # Lazily populated — the first snap of a new trial triggers fn creation
        if not _trial_abort_fns or _trial_abort_fns[-1] is None:
            _trial_abort_fns.append(_make_abort_fn())
        return _trial_abort_fns[-1](snap)

    def _reset_and_callback(record):
        # Signal that this trial is done; next snap starts a new fn
        _trial_abort_fns.append(None)
        if trial_callback is not None:
            trial_callback(record)

    try:
        handler = SimulationHandler(tmp_yaml)
        results = handler.run(abort_fn=_per_trial_abort_wrapper,
                              trial_callback=_reset_and_callback)
        per_csv = os.path.join(RESULTS_DIR, f'{tag}_results.csv')
        handler.export_csv(per_csv)
        return results, per_csv
    finally:
        os.unlink(tmp_yaml)


def _print_stats(tag, results):
    good    = [r for r in results if not r.get('aborted', False)]
    apogees = [r['apogee']    for r in good if r.get('apogee')    is not None]
    lx      = [r['landing_x'] for r in good if r.get('landing_x') is not None]
    ly      = [r['landing_y'] for r in good if r.get('landing_y') is not None]
    if apogees:
        ft = np.array(apogees) / 0.3048
        print(f'  [{tag}] Apogee  med={np.median(ft):>10,.0f} ft  '
              f'min={np.min(ft):>10,.0f}  max={np.max(ft):>10,.0f}')
    if lx and ly:
        disp = np.sqrt(np.array(lx)**2 + np.array(ly)**2) / 1000
        print(f'  [{tag}] Landing med={np.median(disp):>6.2f} km  '
              f'p95={np.percentile(disp, 95):>6.2f} km  '
              f'max={np.max(disp):>6.2f} km')


# ── Main ──────────────────────────────────────────────────────────────────────

TARGET_ALT_FT = 380_000.0
TARGET_ALT_M  = TARGET_ALT_FT * 0.3048


def main():
    parser = argparse.ArgumentParser(description='New stable ESPACE design trade study.')
    parser.add_argument('--ver',        type=str,   default='v2',
                        choices=list(VER_CONFIGS.keys()),
                        help='Aero/rocket version to use (default: v2)')
    parser.add_argument('--trials',     type=int,   default=1,
                        help='MC trials per config per thrust level (default: 1)')
    parser.add_argument('--thrust',     type=float, default=F_NOM,
                        help=f'Starting thrust in N (default: {F_NOM:.0f})')
    parser.add_argument('--sweep-to',   type=float, default=None,
                        help='Sweep thrust up to this value in N, stopping early '
                             'once all configs reach target apogee')
    parser.add_argument('--step',       type=float, default=2000.0,
                        help='Thrust step size in N for sweep (default: 2000)')
    parser.add_argument('--era5',       action='store_true',
                        help='Use SEB/ERA5 wind instead of simple turbulent')
    parser.add_argument('--dt',         type=float, default=0.05,
                        help='Sim timestep during powered flight in seconds (default: 0.05)')
    parser.add_argument('--dt-coast',   type=float, default=0.5,
                        help='Sim timestep after burnout in seconds (default: 0.5 — 10x speedup on descent)')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    vc = VER_CONFIGS[args.ver]
    cols = vc['cols']

    # Load rocket CSV
    rocket = pd.read_csv(vc['rocket_csv'])
    if vc['strip']:
        rocket.columns = rocket.columns.str.strip()
    print(f"[{args.ver}] Loaded {len(rocket)} config row(s) from {os.path.basename(vc['rocket_csv'])}\n")

    # Build thrust list
    if args.sweep_to is not None:
        thrust_list = list(np.arange(args.thrust, args.sweep_to + 1, args.step))
    else:
        thrust_list = [args.thrust]

    # Pre-copy aero tables once (they don't change with thrust).
    # Apply any artificial CP offsets (calibers → inches, D_ref = 8 in).
    cp_offsets = vc.get('cp_offsets_cal', {})
    D_REF_IN   = 8.0  # reference diameter in inches
    for tank_ft in vc['tanks']:
        src_name = vc['aero_name'](tank_ft)
        src  = os.path.join(vc['aero_dir'], src_name)
        dest = os.path.join(ROCKET_DATA_DIR, f'new_{tank_ft:02d}ft_aero.csv')
        offset_cal = cp_offsets.get(tank_ft, 0.0)
        if offset_cal == 0.0:
            shutil.copy2(src, dest)
        else:
            df_aero = pd.read_csv(src)
            df_aero['CP'] += offset_cal * D_REF_IN
            df_aero.to_csv(dest, index=False)
            print(f"  [aero] {tank_ft}ft CP shifted {offset_cal:+.2f} cal "
                  f"({offset_cal * D_REF_IN:+.2f} in) aft", flush=True)

    # Collect configs from rocket CSV
    # If version has an explicit tank_row map, use it; otherwise fall back to
    # sequential indexing (row 0 = tanks[0], row 1 = tanks[1], …)
    tank_row_map = vc.get('tank_row', {ft: i for i, ft in enumerate(vc['tanks'])})
    configs = []
    for tank_ft in vc['tanks']:
        row_idx = tank_row_map[tank_ft]
        if row_idx >= len(rocket):
            print(f"  WARNING: no row {row_idx} in rocket CSV for {tank_ft}ft — skipping")
            continue
        row = rocket.iloc[row_idx]
        configs.append({
            'tank_ft':    tank_ft,
            'wet_mass_g': float(row[cols['wet_mass']]),
            'wet_cg_cm':  float(row[cols['wet_cg']]),
            'wet_I_gcm2': float(row[cols['wet_I']]),
            'dry_mass_g': float(row[cols['dry_mass']]),
            'dry_cg_cm':  float(row[cols['dry_cg']]),
            'dry_I_gcm2': float(row[cols['dry_I']]),
        })

    all_rows      = []
    configs_done  = set()   # tank_ft values that have reached target

    for F_N in thrust_list:
        print(f"\n{'='*70}")
        print(f"THRUST = {F_N:.0f} N")
        print(f"{'='*70}")

        thrust_apogees = {}   # tank_ft → median apogee (ft) at this thrust

        for cfg in configs:
            tank_ft    = cfg['tank_ft']
            tag        = f'new_{tank_ft:02d}ft'
            prop_g     = cfg['wet_mass_g'] - cfg['dry_mass_g']
            mdot       = MDOT_NOM * (F_N / F_NOM)
            t_burn     = (prop_g / 1000.0) / mdot

            print(f"  {tank_ft:2d}ft  t_burn={t_burn:5.1f}s  F={F_N:.0f}N", flush=True)

            # Regenerate ORK CSV for this thrust level
            ork_fname = f'{tag}.csv'
            ork_path  = os.path.join(ROCKET_DATA_DIR, ork_fname)
            _write_ork_csv(
                ork_path, F_N,
                cfg['wet_mass_g'], cfg['wet_cg_cm'], cfg['wet_I_gcm2'],
                cfg['dry_mass_g'], cfg['dry_cg_cm'], cfg['dry_I_gcm2'],
            )

            trial_counter = [0]
            def _trial_cb(record, _tc=trial_counter, _nt=args.trials):
                _tc[0] += 1
                apo_ft   = (record.get('apogee') or 0.0) / 0.3048
                aborted  = record.get('aborted', False)
                disp_km  = math.sqrt(
                    (record.get('landing_x') or 0.0)**2 +
                    (record.get('landing_y') or 0.0)**2
                ) / 1000.0
                status = ' [TUMBLE-ABORT]' if aborted else ''
                print(f"    trial {_tc[0]:>2}/{_nt}  "
                      f"apogee={apo_ft:>10,.0f} ft  "
                      f"disp={disp_km:6.2f} km{status}", flush=True)

            results, _ = _run_config(
                tag, ork_fname, f'{tag}_aero.csv', args.trials, args.era5,
                dt=args.dt, dt_coast=args.dt_coast, trial_callback=_trial_cb)

            good = [r for r in results if not r.get('aborted', False)
                    and r.get('apogee') is not None]
            apogees_ft = [r['apogee'] / 0.3048 for r in good]
            med_ft = np.median(apogees_ft) if apogees_ft else 0.0
            n_tumble = sum(1 for r in results if r.get('aborted', False))
            thrust_apogees[tank_ft] = med_ft

            pct = med_ft / TARGET_ALT_FT * 100
            hit = ' *** TARGET' if med_ft >= TARGET_ALT_FT else ''
            tumble_note = f'  ({n_tumble}/{len(results)} tumbled)' if n_tumble else ''
            print(f"  → {tank_ft:2d}ft  med apogee={med_ft:>10,.0f} ft  "
                  f"({pct:5.1f}%){tumble_note}{hit}", flush=True)

            for r in results:
                r['tank_ft']   = tank_ft
                r['thrust_N']  = F_N
                r['t_burn_s']  = round(t_burn, 2)
                r['wet_kg']    = round(cfg['wet_mass_g'] / 1000, 2)
                r['dry_kg']    = round(cfg['dry_mass_g'] / 1000, 2)
                r['wet_cg_cm'] = round(cfg['wet_cg_cm'], 2)
                r['dry_cg_cm'] = round(cfg['dry_cg_cm'], 2)
            all_rows.extend(results)

            if med_ft >= TARGET_ALT_FT:
                configs_done.add(tank_ft)

        # Stop sweep once all available configs have hit target
        if args.sweep_to is not None and len(configs_done) == len(configs):
            print(f"\nAll configs reached {TARGET_ALT_FT/1000:.0f}k ft target — stopping sweep.")
            break

    # Aggregate CSV
    if all_rows:
        df      = pd.DataFrame(all_rows)
        agg_csv = os.path.join(RESULTS_DIR, 'new_trade_aggregate.csv')
        df.to_csv(agg_csv, index=False)

        print(f"\n{'='*70}")
        print(f"FULL SUMMARY  (target = {TARGET_ALT_FT/1000:.0f}k ft)")
        print(f"{'='*70}")
        print(f"{'F(N)':>7} | {'Tank':>5} | {'t_burn':>7} | "
              f"{'Apogee_med':>12} | {'% target':>9} | {'Disp_med':>9}")
        print(f"{'':>7} | {'(ft)':>5} | {'(s)':>7} | "
              f"{'(ft)':>12} | {'':>9} | {'(km)':>9}")
        print('─' * 70)
        for F_N in sorted(df['thrust_N'].unique()):
            for tank_ft in sorted(df['tank_ft'].unique()):
                sub = df[(df['thrust_N'] == F_N) & (df['tank_ft'] == tank_ft)]
                if sub.empty:
                    continue
                good = sub[sub.get('aborted', pd.Series(False, index=sub.index)) != True]
                if good.empty:
                    good = sub  # fallback if all aborted
                apo  = good['apogee'].dropna().values / 0.3048
                disp = np.sqrt(good['landing_x'].dropna()**2 +
                               good['landing_y'].dropna()**2) / 1000
                n_t  = int(sub.get('aborted', pd.Series(False, index=sub.index)).sum())
                med_ft   = np.median(apo) if len(apo) else 0.0
                med_disp = np.median(disp.values) if len(disp) else 0.0
                pct      = med_ft / TARGET_ALT_FT * 100
                tumble_s = f'  ({n_t}T)' if n_t else ''
                print(f"{F_N:>7.0f} | {tank_ft:>5} | "
                      f"{sub['t_burn_s'].iloc[0]:>7.1f} | "
                      f"{med_ft:>12,.0f} | {pct:>8.1f}% | "
                      f"{med_disp:>9.2f}{tumble_s}")
        print(f'\nAggregate → {agg_csv}')


if __name__ == '__main__':
    main()
