# spaceshot_sizing.py
# Vehicle component definitions for the spaceshot thrust trade.
# Imported by SimulationHandler — exposes COMPONENTS and TANK_SECTION directly.
#
# Forward-referenced components: cg measured from nose tip (m).
# Aft-referenced components:     cg measured from the aft face of the rocket (m).
# length is the axial extent of the component (m).
#
# All values marked TODO: REPLACE THIS are placeholder estimates.

COMPONENTS = [
    dict(ref='forward', name='nose_cone',    length=0.60, mass=3.0,  cg=0.35, i_long=0.090, i_rot=0.015),  # TODO: REPLACE THIS
    dict(ref='forward', name='avionics_bay', length=0.50, mass=5.0,  cg=0.85, i_long=0.104, i_rot=0.025),  # TODO: REPLACE THIS
    dict(ref='aft',     name='engine',       length=0.50, mass=15.0, cg=0.25, i_long=0.313, i_rot=0.075),  # TODO: REPLACE THIS
    dict(ref='aft',     name='fins',         length=0.30, mass=4.0,  cg=0.15, i_long=0.030, i_rot=0.020),  # TODO: REPLACE THIS
]

TANK_SECTION = dict(
    fwd_gap  = 0.05,    # TODO: REPLACE THIS — gap from forward section aft face to tank forward face (m)
    aft_gap  = 0.05,    # TODO: REPLACE THIS — gap from tank aft face to aft section forward face (m)
    od_outer = 0.20,    # TODO: REPLACE THIS — outer tank OD, set by airframe inner diameter (m)
    od_inner = 0.10,    # TODO: REPLACE THIS — inner (IPA) tank OD (m)
    wall_t   = 0.003,   # TODO: REPLACE THIS — wall thickness applied to both inner and outer walls (m)
    isp      = 220.0,   # TODO: REPLACE THIS — reference Isp (s)
    of_ratio = 4.0,     # NOS : IPA mass ratio
)
