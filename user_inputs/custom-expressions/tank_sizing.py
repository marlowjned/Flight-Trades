# tank_sizing.py
# Rocket assembly model for a nested-cylinder NOS/IPA tank configuration.
# IPA occupies the inner cylinder; NOS occupies the surrounding annulus.
# Both share the same axial span. The longer of the two required lengths
# sets L_tank; the other fluid has ullage at one end.
#
# Public API: size_at_prop_mass(components, tank_section, T, m_prop)

import numpy as np

G0 = 9.80665  # m/s²

FLUID_DENSITY = {
    'NOS': 793.0,  # kg/m³ — liquid NOS at operating conditions
    'IPA': 786.0,  # kg/m³ — IPA at operating conditions
}

TANK_WALL_DENSITY = 2700.0  # kg/m³ — aluminium


# --- Inertia helpers (about component's own CG) ---

def _i_long_cylinder(mass, length):
    """Longitudinal MOI of a uniform-density cylinder about its CG (kg·m²)."""
    return mass * length ** 2 / 12.0


def _i_rot_solid_cylinder(mass, radius):
    return mass * radius ** 2 / 2.0


def _i_rot_annulus(mass, r_inner, r_outer):
    return mass * (r_inner ** 2 + r_outer ** 2) / 2.0


# --- Assembly ---

def _assemble(components, tank_section, m_prop):
    """
    Resolve absolute CG positions for the full component stack at a given m_prop.

    components : list of dicts — ref ('forward'|'aft'), name, mass, length,
                 cg (from nose for forward; from aft face for aft), i_long, i_rot
    tank_section: dict — fwd_gap, aft_gap, od_outer, od_inner, wall_t, of_ratio, isp

    Returns (entries, total_length) where each entry is
    (mass, abs_cg, i_long_own, i_rot_own, is_propellant).
    """
    fwd = [c for c in components if c['ref'] == 'forward']
    aft = [c for c in components if c['ref'] == 'aft']

    L_fwd    = sum(c['length'] for c in fwd)
    L_aft    = sum(c['length'] for c in aft)
    fwd_gap  = tank_section['fwd_gap']
    aft_gap  = tank_section['aft_gap']
    od_outer = tank_section['od_outer']
    od_inner = tank_section['od_inner']
    wall_t   = tank_section['wall_t']
    of_ratio = tank_section['of_ratio']   # NOS : IPA by mass (e.g. 4.0)

    m_nos = m_prop * of_ratio / (1.0 + of_ratio)
    m_ipa = m_prop / (1.0 + of_ratio)

    # Wetted radii
    r_nos_outer = od_outer / 2.0 - wall_t        # inner face of outer wall
    r_nos_inner = od_inner / 2.0                 # outer face of inner tank (NOS touches this)
    r_ipa       = od_inner / 2.0 - wall_t        # inner face of inner tank

    A_nos = np.pi * (r_nos_outer ** 2 - r_nos_inner ** 2)
    A_ipa = np.pi * r_ipa ** 2

    L_nos   = m_nos / (FLUID_DENSITY['NOS'] * A_nos)
    L_ipa   = m_ipa / (FLUID_DENSITY['IPA'] * A_ipa)
    L_tank  = max(L_nos, L_ipa)

    # Structural mass — thin-wall cylinders, no end caps
    m_struct_outer = np.pi * od_outer * wall_t * L_tank * TANK_WALL_DENSITY  # TODO: REPLACE THIS — add end caps
    m_struct_inner = np.pi * od_inner * wall_t * L_tank * TANK_WALL_DENSITY  # TODO: REPLACE THIS — add end caps
    m_struct       = m_struct_outer + m_struct_inner

    abs_cg_tank  = L_fwd + fwd_gap + L_tank / 2.0
    total_length = L_fwd + fwd_gap + L_tank + aft_gap + L_aft

    entries = []

    for c in fwd:
        entries.append((c['mass'], c['cg'], c['i_long'], c['i_rot'], False))

    # Tank structure
    entries.append((m_struct, abs_cg_tank, 0.0, 0.0, False))  # TODO: REPLACE THIS — add structural inertia

    # NOS — annular fluid
    entries.append((m_nos, abs_cg_tank,
                    _i_long_cylinder(m_nos, L_tank),
                    _i_rot_annulus(m_nos, r_nos_inner, r_nos_outer),
                    True))

    # IPA — inner cylinder fluid
    entries.append((m_ipa, abs_cg_tank,
                    _i_long_cylinder(m_ipa, L_tank),
                    _i_rot_solid_cylinder(m_ipa, r_ipa),
                    True))

    for c in aft:
        abs_cg = total_length - c['cg']
        entries.append((c['mass'], abs_cg, c['i_long'], c['i_rot'], False))

    return entries, total_length


def _composite(entries, include_propellant):
    """Composite mass, CG (mass-weighted), and inertia (parallel axis) from entry list."""
    subset = [(m, cg, il, ir)
              for (m, cg, il, ir, is_prop) in entries
              if not is_prop or include_propellant]

    total_mass = sum(m for m, *_ in subset)
    if total_mass == 0.0:
        return dict(mass=0.0, cg=0.0, i_long=0.0, i_rot=0.0)

    cg     = sum(m * c for m, c, _, _ in subset) / total_mass
    i_long = sum(il + m * (c - cg) ** 2 for m, c, il, _ in subset)
    i_rot  = sum(ir for _, _, _, ir in subset)

    return dict(mass=total_mass, cg=cg, i_long=i_long, i_rot=i_rot)


# --- Public API ---

def size_at_prop_mass(components, tank_section, T, m_prop):
    """
    Assemble the rocket at a specific propellant mass and thrust level.
    Returns a dict of wet and dry properties plus burn_time.
    """
    entries, total_length = _assemble(components, tank_section, m_prop)
    dry = _composite(entries, include_propellant=False)
    wet = _composite(entries, include_propellant=True)

    burn_time = m_prop * tank_section['isp'] * G0 / T

    return {
        'mass':         wet['mass'],
        'cg':           wet['cg'],
        'i_long_wet':   wet['i_long'],
        'i_rot_wet':    wet['i_rot'],
        'm_dry':        dry['mass'],
        'cg_dry':       dry['cg'],
        'i_long_dry':   dry['i_long'],
        'i_rot_dry':    dry['i_rot'],
        'm_prop':       m_prop,
        'burn_time':    burn_time,
        'total_length': total_length,
    }
