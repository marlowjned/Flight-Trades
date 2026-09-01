"""
ESPACE Spaceshot Sizing Script
================================
Computes propellant requirements for each thrust level in the trade study,
generates ORK-style time-series CSVs for the 6DOF simulator.

Geometry / propellant model
---------------------------
- Outer body tube (NOS tank): OD=8.0 in, ID=7.834 in
- Inner tube (IPA tank, coaxial, nestled at TOP of outer tube):
    OD=5.0 in, ID=4.834 in
- As propellant drains, vapour accumulates at the TOP of each tank
  (liquid settles to the bottom under thrust)

Volume formulae (matching sizing spreadsheet):
    V_NOS = A_outer * L_NOS - A_IPA * L_IPA   (NOS fills outer, IPA displaces some)
    V_IPA = A_IPA * L_IPA

Where A uses the tube OD for the inner calculation (spreadsheet convention).

Propellant densities (user-specified, g/ft³):
    NOS (N₂O):  14 800 g/ft³
    IPA:        22 200 g/ft³

Tank structure mass = wall cross-section area × length × material density
    Material: Aluminium 6061, ρ = 2700 kg/m³ = 76 453 g/ft³  ← CONFIRM w/ user
    Wall thickness both tubes = 0.083 in (OD-ID)/2

Engine scaling: F ∝ m_dot (constant Isp)
    Ref: F_nom=8000 N, m_dot_nom=4.1 kg/s → Isp=198.7 s

Structural dry masses (from data/ESPACEmassdata.xlsx, Sheet 1):
    These are fixed regardless of thrust/propellant.
"""

import math
import numpy as np
import os

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
G0      = 9.80665   # m/s²  standard gravity
IN_M    = 0.0254    # m per inch
FT_M    = 0.3048    # m per foot
FT3_M3  = FT_M**3  # m³ per ft³

# ─────────────────────────────────────────────────────────────
# ENGINE REFERENCE
# ─────────────────────────────────────────────────────────────
F_NOM    = 8_000.0   # N  nominal thrust
MDOT_NOM = 4.1       # kg/s nominal mass flow rate
ISP      = F_NOM / (MDOT_NOM * G0)      # ≈ 198.7 s  (constant with scaling)
OF_RATIO = 4.0       # oxidiser-to-fuel mass ratio (N2O : IPA)

# ─────────────────────────────────────────────────────────────
# PROPELLANT DENSITIES  (g/ft³)
# ─────────────────────────────────────────────────────────────
RHO_NOS_GFT3 = 14_800.0   # g/ft³  liquid N2O at fill conditions
RHO_IPA_GFT3 = 22_200.0   # g/ft³  IPA

# ─────────────────────────────────────────────────────────────
# TUBE GEOMETRY  (inches, then converted)
# ─────────────────────────────────────────────────────────────
OD_OUTER_IN   = 8.000   # in  NOS/body outer tube OD
ID_OUTER_IN   = 7.834   # in  NOS/body outer tube ID
OD_INNER_IN   = 5.000   # in  IPA inner tube OD
ID_INNER_IN   = 4.834   # in  IPA inner tube ID

# Cross-section areas using spreadsheet convention (OD as effective diameter):
#   A_outer = π/4 * (OD_OUTER/12)²  ft²
#   A_IPA   = π/4 * (OD_INNER/12)²  ft²
A_OUTER_FT2  = math.pi / 4 * (OD_OUTER_IN / 12)**2   # 0.34907 ft²
A_IPA_FT2    = math.pi / 4 * (OD_INNER_IN / 12)**2   # 0.13635 ft²

# Wall cross-section areas for structural mass calculation:
A_WALL_OUTER_FT2 = math.pi / 4 * ((OD_OUTER_IN/12)**2 - (ID_OUTER_IN/12)**2)  # 0.01433 ft²
A_WALL_INNER_FT2 = math.pi / 4 * ((OD_INNER_IN/12)**2 - (ID_INNER_IN/12)**2)  # 0.00890 ft²

# Material density for tank structural mass
RHO_ALU_GFT3 = 2700 * 1000 / (1 / FT3_M3)   # g/ft³
# simpler:
RHO_ALU_GFT3 = 2700e3 * FT3_M3               # 76 453 g/ft³

# Useful radii in metres (for MOI calcs)
R_OUTER_M      = (ID_OUTER_IN / 2) * IN_M    # ≈ 0.0995 m  (NOS region outer radius)
R_IPA_OD_M     = (OD_INNER_IN / 2) * IN_M   # ≈ 0.0635 m  (IPA tube outer radius)
R_IPA_ID_M     = (ID_INNER_IN / 2) * IN_M   # ≈ 0.0614 m  (IPA propellant radius)

# ─────────────────────────────────────────────────────────────
# ROCKET GEOMETRY / REFERENCE POSITIONS
# ─────────────────────────────────────────────────────────────
NOSECONE_LEN_IN  = 56.0        # in  nosecone length (tip to tank top)
NOSECONE_LEN_CM  = NOSECONE_LEN_IN * 2.54  # cm from tip to tank top
NOSECONE_LEN_M   = NOSECONE_LEN_IN * IN_M  # m

# ─────────────────────────────────────────────────────────────
# STRUCTURAL DRY MASSES (from data/ESPACEmassdata.xlsx, Sheet 1)
# CG_from_tip: cm from nosecone tip (positive toward engine)
# For Prospect 2: directly from "CG (z cm from nosecone TOP)"
# For Eureka 3: CG_from_tip = bulkhead_cm + z_from_bulkhead_cm
#   where bulkhead_cm = (56 in + L_tank_ft * 12 in/ft) * 2.54
# ─────────────────────────────────────────────────────────────

# Prospect 2 (top section) – CG fixed regardless of tank length
PROSPECT2_COMPONENTS = [
    # (name, mass_g, cg_from_tip_cm)
    ("Nosecone",          3982.172, 97.205),
    ("Drogue Parachute",  867.444,  43.563),
    ("Main Parachute",    1862.037, 99.195),
    ("Avbay w/ Coupler",  1472.254, 132.505),
]

# Eureka 3 (bottom section) – CG offsets from the Bulkhead reference
# Per user: bulkhead position = nosecone_len + tank_len (in inches from tip)
# "z_from_bulkhead_cm" = distance BELOW the bulkhead junction
EUREKA3_COMPONENTS = [
    # (name, mass_g, z_from_bulkhead_cm)
    ("Bulkhead",               569.874,   1.8791),
    ("IPA Poppet Valve",       374.404,   9.1101),
    ("Nitrous Poppet Valve",   1495.596,  2.9981),
    ("Other Valves",           487.114,   8.0861),
    ("Injector",               5389.35,  16.8301),
    ("Chamber Exterior",       7906.469, 28.8911),
    ("Engine w/ Fin Mounts",  26011.07,  31.3131),
]

# Total structural dry mass (no tanks, no propellant)
M_P2_G     = sum(c[1] for c in PROSPECT2_COMPONENTS)    # ≈ 8 184 g
M_E3_G     = sum(c[1] for c in EUREKA3_COMPONENTS)      # ≈ 42 234 g
M_STRUCT_G = M_P2_G + M_E3_G                            # ≈ 50 418 g

# ─────────────────────────────────────────────────────────────
# TANK SIZING FUNCTIONS
# ─────────────────────────────────────────────────────────────

def compute_tank_lengths(m_prop_g):
    """
    Given total propellant mass (g), return (L_NOS_ft, L_IPA_ft) using O/F = 4.
    """
    m_nos_g = (OF_RATIO / (1 + OF_RATIO)) * m_prop_g
    m_ipa_g = m_prop_g - m_nos_g

    V_IPA_ft3 = m_ipa_g / RHO_IPA_GFT3
    L_IPA_ft  = V_IPA_ft3 / A_IPA_FT2                        # IPA tank length

    V_NOS_ft3 = m_nos_g / RHO_NOS_GFT3
    # V_NOS = A_OUTER * L_NOS - A_IPA * L_IPA  →  L_NOS:
    L_NOS_ft  = (V_NOS_ft3 + A_IPA_FT2 * L_IPA_ft) / A_OUTER_FT2

    return L_NOS_ft, L_IPA_ft


def compute_tank_struct_mass(L_NOS_ft, L_IPA_ft):
    """
    Tank structural mass (g) assuming aluminium walls.
    """
    m_nos_tank_g = A_WALL_OUTER_FT2 * L_NOS_ft * RHO_ALU_GFT3
    m_ipa_tank_g = A_WALL_INNER_FT2 * L_IPA_ft * RHO_ALU_GFT3
    return m_nos_tank_g, m_ipa_tank_g


def compute_prop_masses(m_prop_g):
    m_nos_g = (OF_RATIO / (1 + OF_RATIO)) * m_prop_g
    m_ipa_g = m_prop_g - m_nos_g
    return m_nos_g, m_ipa_g


# ─────────────────────────────────────────────────────────────
# CG CALCULATION (time-varying, as propellant depletes)
# ─────────────────────────────────────────────────────────────

def tank_cg_cm(L_NOS_ft, L_IPA_ft, frac_remaining):
    """
    CG (cm from nosecone tip) of the PROPELLANT remaining at a given burn fraction.
    frac_remaining: 1.0 = full tanks, 0.0 = empty (calls with 0 should return bottom CG).
    Propellant drains from the TOP of each tank (vapour head grows at top).
    """
    f   = frac_remaining
    z0  = NOSECONE_LEN_CM          # tank top, cm from tip
    L_N = L_NOS_ft * 30.48        # cm
    L_I = L_IPA_ft * 30.48        # cm

    # ── IPA (simple cylinder, top of tank section) ──
    # Liquid occupies bottom f*L_I of IPA tank
    # IPA tank: from z0 to z0+L_I  (top=z0, bottom=z0+L_I)
    L_ipa_liq_cm = f * L_I
    z_cg_ipa = (z0 + L_I) - L_ipa_liq_cm / 2.0   # midpoint of remaining liquid

    m_ipa_rel = A_IPA_FT2 * (f * L_IPA_ft) * RHO_IPA_GFT3   # prop mass

    # ── NOS (two regions: annular 0→L_I, full L_I→L_N) ──
    V_nos_full_region = A_OUTER_FT2 * (L_NOS_ft - L_IPA_ft)   # ft³  (below IPA)
    V_nos_ann_region  = (A_OUTER_FT2 - A_IPA_FT2) * L_IPA_ft  # ft³  (beside IPA)
    V_nos_total       = V_nos_full_region + V_nos_ann_region

    V_nos_rem = V_nos_total * f

    z_nos_full_bot = z0 + L_N   # bottom of NOS tank (cm from tip)
    z_nos_full_top = z0 + L_I   # junction between annular and full regions

    if V_nos_rem <= V_nos_full_region:
        # Liquid only in full region (bottom of tank)
        h_liq_cm = (V_nos_rem / A_OUTER_FT2) * 30.48
        z_cg_nos = z_nos_full_bot - h_liq_cm / 2.0
        m_nos_rel = V_nos_rem * RHO_NOS_GFT3
    else:
        # Full region entirely liquid + some annular region
        V_ann_rem = V_nos_rem - V_nos_full_region
        h_ann_liq_cm = (V_ann_rem / (A_OUTER_FT2 - A_IPA_FT2)) * 30.48

        # Full region centroid
        z_cg_full = z_nos_full_top + (z_nos_full_bot - z_nos_full_top) / 2.0
        m_full = V_nos_full_region * RHO_NOS_GFT3

        # Annular remaining liquid sits at BOTTOM of annular zone (z0+L_I is top)
        # So liquid spans from (z0+L_I) down to (z0+L_I - h_ann_liq_cm)
        z_ann_liq_bot = z_nos_full_top   # = z0+L_I
        z_cg_ann = z_ann_liq_bot - h_ann_liq_cm / 2.0
        m_ann = V_ann_rem * RHO_NOS_GFT3

        m_nos_rel = m_full + m_ann
        z_cg_nos  = (m_full * z_cg_full + m_ann * z_cg_ann) / m_nos_rel

    return z_cg_nos, m_nos_rel, z_cg_ipa, m_ipa_rel


def total_cg_cm(L_NOS_ft, L_IPA_ft, m_nos_tank_g, m_ipa_tank_g, frac_remaining,
                bulkhead_cm):
    """
    Full rocket CG (cm from nosecone tip) at a given propellant fraction.
    """
    # ── Structural / fixed components ──
    mass_terms = []   # list of (mass_g, cg_cm)

    # Prospect 2
    for _, m, cg in PROSPECT2_COMPONENTS:
        mass_terms.append((m, cg))

    # Eureka 3 (CG offset from bulkhead)
    for _, m, dz in EUREKA3_COMPONENTS:
        mass_terms.append((m, bulkhead_cm + dz))

    # Tank structure: CG at mid-length of each tube
    z_tank_top = NOSECONE_LEN_CM
    cg_nos_tank = z_tank_top + (L_NOS_ft * 30.48) / 2.0
    cg_ipa_tank = z_tank_top + (L_IPA_ft * 30.48) / 2.0
    mass_terms.append((m_nos_tank_g, cg_nos_tank))
    mass_terms.append((m_ipa_tank_g, cg_ipa_tank))

    # ── Propellant ──
    z_cg_nos, m_nos_rem, z_cg_ipa, m_ipa_rem = tank_cg_cm(
        L_NOS_ft, L_IPA_ft, frac_remaining)
    mass_terms.append((m_nos_rem, z_cg_nos))
    mass_terms.append((m_ipa_rem, z_cg_ipa))

    total_mass = sum(m for m, _ in mass_terms)
    cg = sum(m * c for m, c in mass_terms) / total_mass
    return cg, total_mass


# ─────────────────────────────────────────────────────────────
# MOMENT OF INERTIA
# ─────────────────────────────────────────────────────────────

def moi_cylinder(m_kg, r_m, l_m, d_m):
    """
    Pitch MOI (kg·m²) and roll MOI of a solid cylinder about rocket CG.
    d_m = axial distance from cylinder centroid to rocket CG.
    """
    i_pitch_self = m_kg * (r_m**2 / 4 + l_m**2 / 12)
    i_pitch      = i_pitch_self + m_kg * d_m**2        # parallel axis
    i_roll       = m_kg * r_m**2 / 2
    return i_pitch, i_roll


def compute_moi(L_NOS_ft, L_IPA_ft, m_nos_tank_g, m_ipa_tank_g,
                frac_remaining, bulkhead_cm, cg_cm):
    """
    Returns (I_long_kg_m2, I_roll_kg_m2) for the full rocket at given propellant fraction.
    Structural components treated as point masses on the centreline.
    """
    cg_m = cg_cm / 100.0

    I_long = 0.0
    I_roll = 0.0

    # ── Structural point masses (on axis → roll ≈ 0 from these) ──
    for _, m_g, cg_c in PROSPECT2_COMPONENTS:
        d = (cg_c / 100.0) - cg_m
        I_long += (m_g / 1000) * d**2

    for _, m_g, dz in EUREKA3_COMPONENTS:
        cg_c = (bulkhead_cm + dz) / 100.0
        d    = cg_c - cg_m
        I_long += (m_g / 1000) * d**2

    # ── Tank structure (thin-walled cylinders) ──
    z_top_m = NOSECONE_LEN_M
    L_NOS_m = L_NOS_ft * FT_M
    L_IPA_m = L_IPA_ft * FT_M
    cg_nos_tank_m = z_top_m + L_NOS_m / 2.0
    cg_ipa_tank_m = z_top_m + L_IPA_m / 2.0

    # NOS tank: thin annular tube → approximate as solid cylinder at R_OUTER_M
    ip, ir = moi_cylinder(m_nos_tank_g / 1000, R_OUTER_M, L_NOS_m,
                          cg_nos_tank_m - cg_m)
    I_long += ip; I_roll += ir

    ip, ir = moi_cylinder(m_ipa_tank_g / 1000, R_IPA_OD_M, L_IPA_m,
                          cg_ipa_tank_m - cg_m)
    I_long += ip; I_roll += ir

    # ── Propellant ──
    z_cg_nos, m_nos_rem_g, z_cg_ipa, m_ipa_rem_g = tank_cg_cm(
        L_NOS_ft, L_IPA_ft, frac_remaining)

    # NOS remaining: approximate as one equivalent cylinder centred at z_cg_nos
    # Use effective radius of full-tube region (below IPA)
    L_nos_rem_m = (frac_remaining * (A_OUTER_FT2 * (L_NOS_ft - L_IPA_ft) +
                   (A_OUTER_FT2 - A_IPA_FT2) * L_IPA_ft) / A_OUTER_FT2) * FT_M
    ip, ir = moi_cylinder(m_nos_rem_g / 1000, R_OUTER_M, max(L_nos_rem_m, 0.001),
                          z_cg_nos / 100.0 - cg_m)
    I_long += ip; I_roll += ir

    # IPA remaining
    L_ipa_rem_m = frac_remaining * L_IPA_ft * FT_M
    ip, ir = moi_cylinder(m_ipa_rem_g / 1000, R_IPA_ID_M, max(L_ipa_rem_m, 0.001),
                          z_cg_ipa / 100.0 - cg_m)
    I_long += ip; I_roll += ir

    return I_long, I_roll


# ─────────────────────────────────────────────────────────────
# 1-D TRAJECTORY SIZING  (no drag, constant g)
# ─────────────────────────────────────────────────────────────

def simulate_1d(F_N, m_prop_g, bulkhead_cm, L_NOS_ft, L_IPA_ft,
                m_nos_tank_g, m_ipa_tank_g, dt=0.1):
    """
    Euler-integration 1-D vertical trajectory (no drag, constant g).
    Returns max altitude in metres.
    """
    m_dot = MDOT_NOM * (F_N / F_NOM)   # kg/s, scaled with thrust
    m0_g  = M_STRUCT_G + m_nos_tank_g + m_ipa_tank_g + m_prop_g
    m0    = m0_g / 1000.0              # kg
    mf    = (M_STRUCT_G + m_nos_tank_g + m_ipa_tank_g) / 1000.0   # kg

    t = 0.0; h = 0.0; v = 0.0; h_max = 0.0
    t_burn = m_prop_g / 1000.0 / m_dot

    while True:
        m = max(m0 - m_dot * t, mf)
        thrust = F_N if t < t_burn else 0.0
        a = thrust / m - G0
        v += a * dt
        h += v * dt
        t += dt
        if h > h_max:
            h_max = h
        if h < 0 and v < 0:
            break
        if t > 1500:   # safety cap
            break

    return h_max


def find_propellant_mass(F_N, target_alt_m, tol_m=500):
    """
    Bisection search for the propellant mass (g) that achieves target_alt_m
    in the 1-D no-drag simulation.  Returns m_prop_g.
    """
    lo = 20_000.0    # g  lower bound
    hi = 300_000.0   # g  upper bound

    for _ in range(50):
        mid = (lo + hi) / 2.0
        L_NOS, L_IPA = compute_tank_lengths(mid)
        m_nos_tank, m_ipa_tank = compute_tank_struct_mass(L_NOS, L_IPA)
        bulkhead = NOSECONE_LEN_CM + L_NOS * 30.48   # cm from tip

        alt = simulate_1d(F_N, mid, bulkhead, L_NOS, L_IPA, m_nos_tank, m_ipa_tank)
        if abs(alt - target_alt_m) < tol_m:
            return mid
        if alt < target_alt_m:
            lo = mid
        else:
            hi = mid

    return mid


# ─────────────────────────────────────────────────────────────
# ORK CSV GENERATION
# ─────────────────────────────────────────────────────────────

def generate_ork_csv(F_N, m_prop_g, output_path, n_burn_points=100):
    """
    Write an ORK-compatible time-series CSV for the given thrust / propellant.
    Format (matches p1ORK-6dof.csv):
      # Time (s), Mass (g), Motor mass (g), Long MOI (kg·m²),
        Rot MOI (kg·m²), CP (cm), CG (cm), Thrust (N)
    """
    m_dot   = MDOT_NOM * (F_N / F_NOM)           # kg/s
    t_burn  = (m_prop_g / 1000.0) / m_dot         # seconds
    L_NOS, L_IPA   = compute_tank_lengths(m_prop_g)
    m_nos_tank, m_ipa_tank = compute_tank_struct_mass(L_NOS, L_IPA)
    bulkhead_cm = NOSECONE_LEN_CM + L_NOS * 30.48  # cm from nosecone tip

    m_dry_g = M_STRUCT_G + m_nos_tank + m_ipa_tank   # g

    rows = []

    # ── Burn phase (n_burn_points steps) ──
    burn_times = np.linspace(0, t_burn, n_burn_points)
    for t in burn_times:
        frac = 1.0 - t / t_burn if t_burn > 0 else 0.0
        frac = max(frac, 0.0)

        m_prop_rem_g = m_prop_g * frac
        total_mass_g = m_dry_g + m_prop_rem_g
        motor_mass_g = m_prop_rem_g

        cg, _ = total_cg_cm(L_NOS, L_IPA, m_nos_tank, m_ipa_tank,
                              frac, bulkhead_cm)
        i_long, i_roll = compute_moi(L_NOS, L_IPA, m_nos_tank, m_ipa_tank,
                                     frac, bulkhead_cm, cg)
        rows.append((t, total_mass_g, motor_mass_g, i_long, i_roll, float('nan'), cg, F_N))

    # ── Burnout point (thrust = 0) ──
    cg_dry, _ = total_cg_cm(L_NOS, L_IPA, m_nos_tank, m_ipa_tank,
                              0.0, bulkhead_cm)
    i_long_dry, i_roll_dry = compute_moi(L_NOS, L_IPA, m_nos_tank, m_ipa_tank,
                                         0.0, bulkhead_cm, cg_dry)
    rows.append((t_burn + 0.001, m_dry_g, 0.0, i_long_dry, i_roll_dry,
                 float('nan'), cg_dry, 0.0))

    # ── Post-burnout (one far-future point to hold last values) ──
    rows.append((t_burn + 600.0, m_dry_g, 0.0, i_long_dry, i_roll_dry,
                 float('nan'), cg_dry, 0.0))

    header = ("# Time (s),Mass (g),Motor mass (g),"
              "Longitudinal moment of inertia (kg·m²),"
              "Rotational moment of inertia (kg·m²),"
              "CP location (cm),CG location (cm),Thrust (N)")

    with open(output_path, "w") as fh:
        fh.write(header + "\n")
        for row in rows:
            t_, mass, mot, ilong, iroll, cp, cg_, thr = row
            cp_str = "NaN"
            fh.write(f"{t_:.4f},{mass:.2f},{mot:.2f},"
                     f"{ilong:.4f},{iroll:.4f},"
                     f"{cp_str},{cg_:.3f},{thr:.1f}\n")

    return L_NOS, L_IPA, m_nos_tank, m_ipa_tank, bulkhead_cm, t_burn


# ─────────────────────────────────────────────────────────────
# MAIN: run sizing for each thrust level
# ─────────────────────────────────────────────────────────────

TARGET_ALT_FT = 380_000.0
TARGET_ALT_M  = TARGET_ALT_FT * 0.3048   # 115 824 m

# Thrust sweep: 8 kN → 14 kN in 1 kN steps
THRUST_VALUES_N = [8_000, 9_000, 10_000, 11_000, 12_000, 13_000, 14_000]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print(f"\n{'='*75}")
    print(f"ESPACE Spaceshot Sizing — target {TARGET_ALT_FT/1000:.0f}k ft "
          f"({TARGET_ALT_M/1000:.1f} km) — 1-D no-drag model")
    print(f"Isp = {ISP:.1f} s  |  O/F = {OF_RATIO}  |  Struct dry = {M_STRUCT_G/1000:.2f} kg")
    print(f"{'='*75}")
    print(f"{'F (N)':>7} | {'m_dot':>6} | {'m_prop':>8} | {'m_wet':>8} | "
          f"{'L_NOS':>7} | {'L_IPA':>7} | {'m_tank':>8} | {'t_burn':>7}")
    print(f"{'':>7} | {'(kg/s)':>6} | {'(kg)':>8} | {'(kg)':>8} | "
          f"{'(ft)':>7} | {'(ft)':>7} | {'(kg)':>8} | {'(s)':>7}")
    print("-"*75)

    sizing_results = {}

    for F in THRUST_VALUES_N:
        m_dot = MDOT_NOM * (F / F_NOM)
        m_prop_g = find_propellant_mass(F, TARGET_ALT_M)

        L_NOS, L_IPA = compute_tank_lengths(m_prop_g)
        m_nos_tank, m_ipa_tank = compute_tank_struct_mass(L_NOS, L_IPA)
        m_tank_g = m_nos_tank + m_ipa_tank
        m_wet_g  = M_STRUCT_G + m_tank_g + m_prop_g
        t_burn   = (m_prop_g / 1000.0) / m_dot

        print(f"{F:>7.0f} | {m_dot:>6.3f} | {m_prop_g/1000:>8.2f} | "
              f"{m_wet_g/1000:>8.2f} | {L_NOS:>7.2f} | {L_IPA:>7.2f} | "
              f"{m_tank_g/1000:>8.2f} | {t_burn:>7.1f}")

        # Generate ORK CSV
        tag  = f"espace_F{int(F/1000):02d}kN"
        path = os.path.join(OUTPUT_DIR, f"{tag}.csv")
        generate_ork_csv(F, m_prop_g, path)

        sizing_results[F] = {
            "m_dot": m_dot,
            "m_prop_g": m_prop_g,
            "L_NOS_ft": L_NOS,
            "L_IPA_ft": L_IPA,
            "m_tank_g": m_tank_g,
            "m_wet_g":  m_wet_g,
            "t_burn_s": t_burn,
            "ork_file": path,
        }

    print(f"{'='*75}")
    print(f"\nGenerated {len(THRUST_VALUES_N)} ORK CSV files in {OUTPUT_DIR}/")
    print("NOTE: Tank structure density assumed aluminium 6061 (2700 kg/m³). "
          "Confirm or update RHO_ALU_GFT3 if different material.")
    print("NOTE: 1-D no-drag sizing — actual 6DOF apogee will be lower due to drag.")
    print("      Recommend running smoke-test and adjusting target altitude if needed.")
