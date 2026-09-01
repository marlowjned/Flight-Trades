"""
stability_quality.py
====================
Quantifies passive aerodynamic stability quality for a given ESPACE rocket
configuration throughout its powered flight phase.

Physics background
------------------
For small pitch perturbations α about a 1-D (vertical) trajectory the
linearised pitch equation is:

    I_pitch · α̈  =  −CNα · q · S_ref · (x_CP − x_CG) · α   +  damping

This is a second-order linear system with:

    Natural frequency   ω_n = sqrt(CNα · q · S_ref · |x_CP − x_CG| / I_pitch)
                                [rad/s]

    When SM > 0 (stable):   perturbations oscillate at ω_n, decaying at rate
                             τ = 1 / (ζ · ω_n)

    When SM < 0 (unstable): perturbations grow.  Time-to-double:
                             T₂ = ln(2) / ω_n   [s]

    Damping ratio ζ requires the pitch-damping coefficient Cmq, which is not
    in our RASAero tables.  For fin-stabilised sounding rockets ζ ≈ 0.05–0.15
    (very lightly damped), meaning static SM is the dominant design driver.
    ζ is therefore not computed here; add it when Cmq data are available via:
        ζ = −Cmq · q · S_ref · D² / (4 · I_pitch · ω_n · V)

Static stability margin
-----------------------
    SM = (x_CP − x_CG) / D_ref    [calibers]
    SM > 0  →  stable   (CP aft of CG)
    SM < 0  →  unstable (CP forward of CG; perturbations diverge)

Stability quality score Q
--------------------------
    Q = min(SM_powered) × fraction_of_powered_flight_with_SM_>_0

Interpretation:
    Q ≥ 1    → solid positive margin everywhere (min SM ≥ 1 cal)
    0 < Q < 1 → either thin margin or briefly unstable
    Q ≤ 0    → unstable at some point; |Q| quantifies severity

Inputs
------
The script needs, per thrust level:
  • Propellant mass / tank geometry   → from espace_sizing.py functions
  • CG(t), I_pitch(t), m(t)          → from simulate_trajectory()
  • CP(Mach), CNα(Mach) at α = 0     → interpolated from data/stability-analysis tables
  • Standard atmosphere               → ambiance library (clamped to 81 020 m)

Usage
-----
    python stability_quality.py                     # all 7 thrust levels
    python stability_quality.py --thrust 8000 12000 # specific thrusts only
    python stability_quality.py --thrust 8000 --plot --csv out.csv
    python stability_quality.py --dt 0.1            # finer time resolution
"""

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'user_inputs', 'rocket_data'))

from ambiance import Atmosphere  # noqa: E402

from espace_sizing import (     # noqa: E402
    THRUST_VALUES_N, TARGET_ALT_M,
    MDOT_NOM, F_NOM, G0, M_STRUCT_G, NOSECONE_LEN_CM,
    find_propellant_mass, compute_tank_lengths, compute_tank_struct_mass,
    total_cg_cm, compute_moi,
)

# ── Fixed rocket geometry ─────────────────────────────────────────────────────
D_REF   = 8 * 0.0254                       # reference diameter  [m] = 0.2032 m
S_REF   = math.pi / 4 * D_REF ** 2        # reference area      [m²] ≈ 0.03243 m²
ATM_CAP = 81_020.0                         # ambiance altitude limit [m]

TABLE_DIR = os.path.join(ROOT, 'data', 'stability-analysis',
                         'Interpolator Data',
                         'Tuning - Percent Stability', 'Tables')
LENGTHS   = list(range(8, 21))             # 8 … 20 ft table filenames

# CNα column name as it appears in the CSV files
CNA_COL = 'CNalpha (0 to 4 deg) (per rad)'


# ══════════════════════════════════════════════════════════════════════════════
# AERO TABLE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_raw_interp():
    """
    Load and stack the 13 Spaceshot_Xft CSV tables once.
    Returns (stack_fn, col_names, nrows, ncols) where
    stack_fn(L_ft) → DataFrame interpolated to that tank length.
    """
    tables    = [pd.read_csv(os.path.join(TABLE_DIR, f'Spaceshot_{L}ft.CSV'))
                 for L in LENGTHS]
    col_names = tables[0].columns.tolist()
    S         = np.stack([df.to_numpy() for df in tables], axis=2)   # (R,C,13)
    nrows, ncols, npages = S.shape
    S_flat    = S.transpose(2, 0, 1).reshape(npages, -1)             # (13, R*C)
    raw_fn    = interp1d(LENGTHS, S_flat, axis=0, kind='linear')

    def stack_fn(L_ft):
        arr = raw_fn(L_ft).reshape(nrows, ncols)
        return pd.DataFrame(arr, columns=col_names)

    return stack_fn


def build_aero_fns(stack_fn, L_NOS_ft):
    """
    Given the stacking function and a tank length, return:
        cp_fn(mach)  → CP in metres from nose  [m]
        cna_fn(mach) → CNα at α ≈ 0            [1/rad]
    Both are 1-D scipy interpolators (LASTVAL extrapolation).
    """
    df     = stack_fn(L_NOS_ft)
    alpha0 = df[df['Alpha'] < 0.5].sort_values('Mach')

    machs   = alpha0['Mach'].values
    cp_m    = alpha0['CP'].values * 0.0254             # inches → metres
    cna     = alpha0[CNA_COL].values                   # 1/rad

    kw = dict(bounds_error=False)
    cp_fn  = interp1d(machs, cp_m,  fill_value=(cp_m[0],  cp_m[-1]),  **kw)
    cna_fn = interp1d(machs, cna,   fill_value=(cna[0],   cna[-1]),   **kw)
    return cp_fn, cna_fn


# ══════════════════════════════════════════════════════════════════════════════
# 1-D TRAJECTORY (returns full time series)
# ══════════════════════════════════════════════════════════════════════════════

def simulate_trajectory(F_N, m_prop_g, L_NOS_ft, L_IPA_ft,
                        m_nos_tank, m_ipa_tank, dt=0.2):
    """
    1-D Euler integration (no drag, constant g) returning arrays:
        t, h, v, cg_cm, i_long, thrust
    Also returns scalar t_burn.
    """
    m_dot       = MDOT_NOM * (F_N / F_NOM)
    t_burn      = (m_prop_g / 1000.0) / m_dot
    bulkhead_cm = NOSECONE_LEN_CM + L_NOS_ft * 30.48
    m_dry_g     = M_STRUCT_G + m_nos_tank + m_ipa_tank

    t = h = v = 0.0
    rows = []

    while True:
        frac         = max(1.0 - t / t_burn, 0.0) if t <= t_burn else 0.0
        m_prop_rem_g = m_prop_g * frac
        m_total      = (m_dry_g + m_prop_rem_g) / 1000.0

        cg_cm, _  = total_cg_cm(L_NOS_ft, L_IPA_ft, m_nos_tank, m_ipa_tank,
                                  frac, bulkhead_cm)
        i_long, _ = compute_moi(L_NOS_ft, L_IPA_ft, m_nos_tank, m_ipa_tank,
                                 frac, bulkhead_cm, cg_cm)

        thrust = F_N if t <= t_burn else 0.0
        rows.append((t, h, v, cg_cm, i_long, thrust))

        a_net = thrust / m_total - G0
        v += a_net * dt
        h += v * dt
        t += dt

        if h < 0 and v < 0 and t > t_burn:
            break
        if t > 2000:
            break

    cols = ['t', 'h', 'v', 'cg_cm', 'i_long', 'thrust']
    df = pd.DataFrame(rows, columns=cols)
    df['t_burn'] = t_burn
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STABILITY METRICS (per time step)
# ══════════════════════════════════════════════════════════════════════════════

def compute_stability_series(traj_df, cp_fn, cna_fn):
    """
    For each row in the trajectory DataFrame, compute:
        sm        : static stability margin  [calibers]
        omega_n   : undamped natural frequency  [rad/s]
        t_double  : time-to-double (unstable only, else inf)  [s]
        q_pa      : dynamic pressure  [Pa]
        mach      : Mach number

    Returns a new DataFrame with these columns appended.
    """
    sm_list, wn_list, td_list, q_list, mach_list = [], [], [], [], []

    for _, row in traj_df.iterrows():
        h_clamped = float(np.clip(row['h'], 0.0, ATM_CAP))
        v         = float(row['v'])

        atm     = Atmosphere(h_clamped)
        rho     = float(atm.density[0])
        a_snd   = float(atm.speed_of_sound[0])

        mach  = v / a_snd if v > 0 else 0.0
        q     = 0.5 * rho * v ** 2

        m_lkp = max(mach, 0.01)
        cp_m  = float(cp_fn(m_lkp))
        cna   = float(cna_fn(m_lkp))

        cg_m   = row['cg_cm'] / 100.0
        i_long = row['i_long']

        sm = (cp_m - cg_m) / D_REF

        if q > 0 and i_long > 0 and cna > 0:
            omega_n = math.sqrt(cna * q * S_REF * abs(cp_m - cg_m) / i_long)
        else:
            omega_n = 0.0

        t_double = math.log(2) / omega_n if (sm < 0 and omega_n > 0) else math.inf

        sm_list.append(sm)
        wn_list.append(omega_n)
        td_list.append(t_double)
        q_list.append(q)
        mach_list.append(mach)

    out = traj_df.copy()
    out['sm']       = sm_list
    out['omega_n']  = wn_list
    out['t_double'] = td_list
    out['q_pa']     = q_list
    out['mach']     = mach_list
    return out


# ══════════════════════════════════════════════════════════════════════════════
# QUALITY SCORE
# ══════════════════════════════════════════════════════════════════════════════

def quality_score(stab_df):
    """
    Compute scalar stability quality metrics from the full time-series DataFrame.
    All metrics refer to the *powered* phase only.

    Returns a dict with:
        Q                  composite quality score (higher = better; < 0 = unstable)
        min_sm_cal         worst-case SM during powered flight  [calibers]
        max_sm_cal         best-case SM during powered flight   [calibers]
        mean_sm_cal        time-averaged SM                     [calibers]
        sm_liftoff_cal     SM at first powered step             [calibers]
        sm_burnout_cal     SM at last powered step              [calibers]
        frac_stable        fraction of powered flight with SM > 0  [0–1]
        sm_slope_cal_per_s linear trend of SM vs time (negative = getting worse)
        worst_t_double_s   minimum T₂ observed (inf if always stable)  [s]
        t_unstable_start_s time when instability begins (None if never)  [s]
        dur_unstable_s     total duration of SM < 0              [s]
    """
    powered = stab_df[stab_df['thrust'] > 0].copy()
    if powered.empty:
        return {}

    sm = powered['sm'].values
    t  = powered['t'].values
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0

    frac_stable = float((sm > 0).mean())
    min_sm      = float(sm.min())
    max_sm      = float(sm.max())
    mean_sm     = float(sm.mean())

    Q = min_sm * frac_stable

    # Linear trend (calibers / second)
    sm_slope = float(np.polyfit(t - t[0], sm, 1)[0]) if len(t) > 2 else 0.0

    # Worst time-to-double
    td = powered['t_double'].replace(math.inf, np.nan).dropna()
    worst_t2 = float(td.min()) if len(td) > 0 else math.inf

    # Unstable window
    unstable = sm < 0
    t_start = float(t[unstable][0])  if unstable.any() else None
    dur      = float(unstable.sum() * dt)

    return {
        'Q':                   Q,
        'min_sm_cal':          min_sm,
        'max_sm_cal':          max_sm,
        'mean_sm_cal':         mean_sm,
        'sm_liftoff_cal':      float(sm[0]),
        'sm_burnout_cal':      float(sm[-1]),
        'frac_stable':         frac_stable,
        'sm_slope_cal_per_s':  sm_slope,
        'worst_t_double_s':    worst_t2,
        't_unstable_start_s':  t_start,
        'dur_unstable_s':      dur,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOP-LEVEL ANALYSIS FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def analyse(F_N, stack_fn, dt=0.2, verbose=True):
    """
    Full stability quality analysis for a single thrust level.

    Parameters
    ----------
    F_N      : float  thrust [N]
    stack_fn : callable returned by _load_raw_interp()
    dt       : float  trajectory integration step [s]
    verbose  : bool   print per-config report

    Returns
    -------
    stab_df  : DataFrame  full time-series with stability columns
    qm       : dict       quality metrics (see quality_score())
    """
    m_prop_g           = find_propellant_mass(F_N, TARGET_ALT_M)
    L_NOS_ft, L_IPA_ft = compute_tank_lengths(m_prop_g)
    m_nos_tank, m_ipa_tank = compute_tank_struct_mass(L_NOS_ft, L_IPA_ft)

    cp_fn, cna_fn = build_aero_fns(stack_fn, L_NOS_ft)
    traj_df       = simulate_trajectory(F_N, m_prop_g, L_NOS_ft, L_IPA_ft,
                                        m_nos_tank, m_ipa_tank, dt=dt)
    stab_df       = compute_stability_series(traj_df, cp_fn, cna_fn)
    qm            = quality_score(stab_df)

    if verbose:
        t_burn = traj_df['t_burn'].iloc[0]
        tag    = 'STABLE' if qm.get('min_sm_cal', -1) > 0 else 'UNSTABLE'
        print(f"\n{'─'*68}")
        print(f"  F = {F_N:,} N │ L_NOS = {L_NOS_ft:.2f} ft │ "
              f"m_prop = {m_prop_g/1000:.1f} kg │ t_burn = {t_burn:.1f} s │ [{tag}]")
        print(f"{'─'*68}")
        print(f"  SM at liftoff    : {qm['sm_liftoff_cal']:+.3f} cal")
        print(f"  SM at burnout    : {qm['sm_burnout_cal']:+.3f} cal")
        print(f"  SM minimum       : {qm['min_sm_cal']:+.3f} cal  "
              f"(mean {qm['mean_sm_cal']:+.3f},  max {qm['max_sm_cal']:+.3f})")
        print(f"  SM slope         : {qm['sm_slope_cal_per_s']:+.4f} cal/s")
        print(f"  Fraction stable  : {qm['frac_stable']*100:.1f}%")
        if qm['t_unstable_start_s'] is not None:
            print(f"  Unstable from    : t = {qm['t_unstable_start_s']:.1f} s  "
                  f"(duration {qm['dur_unstable_s']:.1f} s of {t_burn:.1f} s burn)")
            print(f"  Worst T₂         : {qm['worst_t_double_s']:.2f} s  "
                  f"  ← perturbation doubles in {qm['worst_t_double_s']:.2f} s at peak instability")
        else:
            print("  Unstable phase   : none")
        print(f"  Quality score Q  : {qm['Q']:+.3f}")

    return stab_df, qm


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Rocket stability quality analyser for ESPACE configurations.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--thrust', type=int, nargs='+', default=None,
                        help='Thrust values to analyse [N]. Default: all 7.')
    parser.add_argument('--dt', type=float, default=0.2,
                        help='Trajectory integration timestep [s]. Default: 0.2')
    parser.add_argument('--plot', action='store_true',
                        help='Show SM(t) and ω_n(t) plots.')
    parser.add_argument('--csv', default=None,
                        help='Write summary table to CSV path.')
    args = parser.parse_args()

    thrusts = args.thrust or THRUST_VALUES_N

    print('ESPACE Stability Quality Analysis')
    print('=' * 68)
    print(f'D_ref = {D_REF*100:.2f} cm  │  S_ref = {S_REF*1e4:.2f} cm²  │  dt = {args.dt} s')
    print(f'Target altitude: {TARGET_ALT_M/1000:.1f} km  ({TARGET_ALT_M/0.3048/1000:.0f}k ft)')
    print(f'Note: damping ratio not computed (Cmq not in aero tables).')
    print(f'      For fin-stabilised rockets ζ ≈ 0.05–0.15; SM dominates design.')
    print()
    print('Loading stability tables …')
    stack_fn = _load_raw_interp()
    print('Ready.\n')

    all_qm   = []
    all_dfs  = {}

    for F_N in thrusts:
        df, qm = analyse(F_N, stack_fn, dt=args.dt)
        qm['thrust_N'] = F_N
        all_qm.append(qm)
        all_dfs[F_N] = df

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print('SUMMARY')
    print(f"{'='*80}")

    hdr   = (f"{'F(N)':>7} │ {'Q':>6} │ {'SM_lo':>6} │ {'SM_bo':>6} │ "
             f"{'SM_min':>6} │ {'slope':>9} │ {'%stable':>8} │ "
             f"{'T2_min':>7} │ {'instab_start':>13}")
    units = (f"{'':>7} │ {'':>6} │ {'cal':>6} │ {'cal':>6} │ "
             f"{'cal':>6} │ {'cal/s':>9} │ {'':>8} │ "
             f"{'s':>7} │ {'s':>13}")
    print(hdr)
    print(units)
    print('─' * 80)

    for qm in all_qm:
        t2  = f"{qm['worst_t_double_s']:.2f}" if math.isfinite(qm['worst_t_double_s']) else '     ∞'
        t_s = f"{qm['t_unstable_start_s']:.1f}" if qm['t_unstable_start_s'] is not None else '        —'
        flag = '  ✓' if qm['min_sm_cal'] > 0 else '  ✗'
        print(
            f"{qm['thrust_N']:>7} │ {qm['Q']:>+6.3f} │ "
            f"{qm['sm_liftoff_cal']:>+6.3f} │ {qm['sm_burnout_cal']:>+6.3f} │ "
            f"{qm['min_sm_cal']:>+6.3f} │ {qm['sm_slope_cal_per_s']:>+9.4f} │ "
            f"{qm['frac_stable']*100:>7.1f}% │ "
            f"{t2:>7} │ {t_s:>13}{flag}"
        )

    if args.csv:
        pd.DataFrame(all_qm).to_csv(args.csv, index=False)
        print(f'\nSummary → {args.csv}')

    # ── Optional plots ─────────────────────────────────────────────────────
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
            colors = plt.cm.tab10(np.linspace(0, 1, len(thrusts)))

            for (F_N, df), col in zip(all_dfs.items(), colors):
                powered = df[df['thrust'] > 0]
                qm_row  = next(q for q in all_qm if q['thrust_N'] == F_N)
                label   = f"{F_N//1000}kN  (Q={qm_row['Q']:+.2f})"
                ax1.plot(powered['t'], powered['sm'], color=col, label=label)

            ax1.axhline(0.0, color='red',    linestyle='--', lw=1.0, label='SM = 0 (neutral)')
            ax1.axhline(1.0, color='orange', linestyle=':',  lw=0.8, label='SM = 1 cal (rule-of-thumb min)')
            ax1.fill_between([ax1.get_xlim()[0], ax1.get_xlim()[1]], 0, -99,
                             color='red', alpha=0.05)
            ax1.set_ylabel('Stability Margin (calibers)')
            ax1.set_title('Static Stability Margin During Powered Flight')
            ax1.legend(fontsize=8, ncol=2)
            ax1.grid(True, alpha=0.3)
            ax1.yaxis.set_minor_locator(mticker.AutoMinorLocator())

            for (F_N, df), col in zip(all_dfs.items(), colors):
                powered = df[df['thrust'] > 0]
                wn = powered['omega_n'].copy()
                wn[wn == 0] = np.nan
                ax2.plot(powered['t'], wn, color=col,
                         label=f"{F_N//1000}kN")

            ax2.set_ylabel('Natural Frequency ω_n  (rad/s)')
            ax2.set_xlabel('Time (s)')
            ax2.set_title('Undamped Pitch Natural Frequency During Powered Flight')
            ax2.legend(fontsize=8, ncol=2)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except ImportError:
            print('matplotlib not available — skipping plot.')


if __name__ == '__main__':
    main()
