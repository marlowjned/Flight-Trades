"""
generate_aero_csvs.py
=====================
Generates RasAero-compatible aero coefficient CSV files for each ESPACE
thrust configuration by interpolating across the pre-computed RASAero tables
in 'data/stability-analysis/Interpolator Data/Tuning - Percent Stability/Tables/'.

The "rocket_length" parameter is the propulsion body tube length (tank section)
in feet, matching the Spaceshot_Xft.CSV naming convention (8–20 ft range).

Each output CSV contains columns expected by rasaero_loader.py:
    Mach, Alpha, CD Power-Off, CD Power-On, CL, CP
CP is left in inches — rasaero_loader.py multiplies by 0.0254 on read.

Run from project root:
    python user_inputs/rocket_data/generate_aero_csvs.py
"""

import sys
import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ── Paths ─────────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT      = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
TABLE_DIR = os.path.join(ROOT, 'data', 'stability-analysis',
                         'Interpolator Data',
                         'Tuning - Percent Stability', 'Tables')
OUT_DIR   = _THIS_DIR   # aero CSVs land next to the ORK CSVs

sys.path.insert(0, _THIS_DIR)          # for espace_sizing import
from espace_sizing import (
    THRUST_VALUES_N, TARGET_ALT_M,
    find_propellant_mass, compute_tank_lengths,
)

# ── Constants ─────────────────────────────────────────────────────────────────
LENGTHS   = list(range(8, 21))   # 8 … 20 ft (must match CSV filenames)
AERO_COLS = ['Mach', 'Alpha', 'CD Power-Off', 'CD Power-On', 'CL', 'CP']


# ── Table loading / interpolation ─────────────────────────────────────────────
def load_stability_interpolator():
    """
    Load all 13 Spaceshot_Xft.CSV tables and return a callable interpolator.

    Returns
    -------
    interp_fn : callable
        interp_fn(L_ft) → 1-D array of length (nrows * ncols)
    col_names : list[str]
    nrows, ncols : int
    """
    tables    = [pd.read_csv(os.path.join(TABLE_DIR, f'Spaceshot_{L}ft.CSV'))
                 for L in LENGTHS]
    col_names = tables[0].columns.tolist()
    S         = np.stack([df.to_numpy() for df in tables], axis=2)  # (R, C, 13)
    nrows, ncols, npages = S.shape
    S_flat    = S.transpose(2, 0, 1).reshape(npages, -1)            # (13, R*C)
    interp_fn = interp1d(LENGTHS, S_flat, axis=0, kind='linear')
    return interp_fn, col_names, nrows, ncols


def make_aero_csv(interp_fn, col_names, nrows, ncols, L_NOS_ft, output_path):
    """
    Interpolate to L_NOS_ft and write a RasAero-format CSV.
    Selects only the columns needed by rasaero_loader.py.
    """
    arr    = interp_fn(L_NOS_ft).reshape(nrows, ncols)
    df_all = pd.DataFrame(arr, columns=col_names)
    df_all[AERO_COLS].to_csv(output_path, index=False)
    return len(df_all)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Loading stability tables …')
    interp_fn, col_names, nrows, ncols = load_stability_interpolator()
    print(f'Loaded {len(LENGTHS)} tables, {nrows} rows × {ncols} columns each.\n')

    print(f'{"F (N)":>7}  {"L_NOS (ft)":>11}  {"Output file":<42}  {"Rows":>5}')
    print('─' * 72)

    for F_N in THRUST_VALUES_N:
        m_prop_g     = find_propellant_mass(F_N, TARGET_ALT_M)
        L_NOS_ft, _  = compute_tank_lengths(m_prop_g)
        tag          = f'espace_F{int(F_N / 1000):02d}kN'
        out_path     = os.path.join(OUT_DIR, f'{tag}_aero.csv')
        n            = make_aero_csv(interp_fn, col_names, nrows, ncols,
                                     L_NOS_ft, out_path)
        print(f'{F_N:>7.0f}  {L_NOS_ft:>11.3f}  {os.path.basename(out_path):<42}  {n:>5}')

    print(f'\nGenerated {len(THRUST_VALUES_N)} aero CSVs in:\n  {OUT_DIR}/')
    print(f'Columns: {AERO_COLS}')
    print('CP is in inches — rasaero_loader.py converts × 0.0254 to metres on read.')
