# sim_handler.py
# Handles config file, creates actual trade framework, and packages data

import itertools
import copy
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from flight_sim.core.config_loader import load_config
from flight_sim.core.sim_loop import FlightSim
from flight_sim.core.sim_snapshot import export_snapshots_csv
from flight_sim.flight_components.rocket import Rocket
from flight_sim.flight_components.gravity_model import GravityModel
from flight_sim.flight_components.recovery import Recovery
from flight_sim.wind.simple_wind_model import SimpleWindModel
from flight_sim.data_helpers.vector3d import Vector3D


RECORD_EXTRACTORS = {
    "apogee":       lambda snaps: max(s.altitude for s in snaps),
    "flight_time":  lambda snaps: snaps[-1].time,
    "max_velocity": lambda snaps: max(s._state.velocity.magnitude for s in snaps),
    "max_mach":     lambda snaps: max(s.mach for s in snaps),
    "landing_x":    lambda snaps: snaps[-1].pos_x,
    "landing_y":    lambda snaps: snaps[-1].pos_y,
}


class SimulationHandler:

    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self._permutations = self._generate_permutations()
        self._results   = []
        self._snapshots = {}  # (perm_idx, trial_idx) -> list[SimSnapshot]

    def _generate_permutations(self) -> list[dict]:
        blocks = self.config.get("trade_study", {}).get("blocks", [])

        if not blocks:
            return [{}]

        per_block = []
        for block in blocks:
            params = block.get("parameters", {})
            if not params:
                continue
            lengths = [len(v) for v in params.values()]
            if len(set(lengths)) > 1:
                raise ValueError(
                    f"Trade block '{block.get('name')}' has params with unequal lengths: "
                    f"{dict(zip(params.keys(), lengths))}"
                )
            keys = list(params.keys())
            block_perms = [dict(zip(keys, combo)) for combo in zip(*params.values())]
            per_block.append(block_perms)

        combined = []
        for combo in itertools.product(*per_block):
            merged = {}
            for d in combo:
                merged.update(d)
            combined.append(merged)

        return combined

    def _merge_config(self, base_config: dict, permutation: dict) -> dict:
        merged = copy.deepcopy(base_config)
        merged["_trade_overrides"] = permutation
        return merged

    def _build_settings(self, config: dict) -> FlightSim.SimulationSettings:
        sim_cfg   = config.get("simulation", {})

        dt               = sim_cfg.get("dt", 0.05)
        max_runtime      = sim_cfg.get("max_runtime", 200)
        recovery_sim     = sim_cfg.get("recovery_sim", False)
        launch_rail_len  = sim_cfg.get("launch_rail_length", 5.0)

        rail_dir = np.array(sim_cfg.get("launchrail_orientation", [0, 0, 1]), dtype=float)
        launch_rail_orientation = Rotation.align_vectors([rail_dir], [[0, 0, 1]])[0]

        return FlightSim.SimulationSettings(
            dt=dt,
            max_runtime=max_runtime,
            recovery_sim=recovery_sim,
            launch_rail_len=launch_rail_len,
            launch_rail_orientation=launch_rail_orientation,
        )

    def _build_init_state(self, config: dict, orientation: Rotation) -> FlightSim.FlightState:
        dcm = orientation.as_matrix()
        return FlightSim.FlightState(
            time=0.0,
            position=Vector3D(np.zeros(3), dcm),
            velocity=Vector3D(np.zeros(3), dcm),
            orientation=orientation,
            omega=Vector3D(np.zeros(3), dcm, True),
        )

    def _build_wind_model(self, config: dict, seed: int):
        wind_cfg  = config.get("wind", {})
        wind_type = wind_cfg.get("type", "simple")

        if wind_type == "SEB-windmodel":
            return self._build_seb_wind_model(wind_cfg, config, seed)

        magnitude            = wind_cfg.get("magnitude", 0.0)
        direction            = wind_cfg.get("direction", 0.0)
        alt_min              = wind_cfg.get("alt_min", 0.0)
        alt_max              = wind_cfg.get("alt_max", 10000.0)
        alt_steps            = wind_cfg.get("alt_steps", 100)
        altitudes            = np.linspace(alt_min, alt_max, alt_steps)
        turbulence_intensity = wind_cfg.get("turbulence_intensity", 1.0)
        return SimpleWindModel(magnitude, direction, altitudes,
                               turbulence_seed=seed,
                               turbulence_intensity=turbulence_intensity)

    def _build_seb_wind_model(self, wind_cfg: dict, config: dict, seed: int):
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'SEB-windmodel'))
        from seb_wind_model.cds_fetch import load_npz, preprocess_nc
        from seb_wind_model.eof import EOFModel
        from flight_sim.wind.seb_wind_model import SEBWindModel

        data_path = wind_cfg.get("data_path")
        if data_path is None:
            raise ValueError(
                "SEB-windmodel requires 'data_path' pointing to a preprocessed "
                ".npz file or a raw ERA5 .nc file."
            )

        if data_path.endswith(".nc"):
            from seb_wind_model.utils import uniform_alt_grid
            alt_grid = uniform_alt_grid(z_max=wind_cfg.get("alt_max_m", 20000.0),
                                        dz=wind_cfg.get("alt_dz_m", 50.0))
            data = preprocess_nc(
                data_path, alt_grid,
                lat=wind_cfg.get("lat", 0.0),
                lon=wind_cfg.get("lon", 0.0),
                surface_elev_m=wind_cfg.get("surface_elev_m", 0.0),
            )
        else:
            data = load_npz(data_path)

        n_modes = wind_cfg.get("n_modes", None)
        scale   = wind_cfg.get("scale", 1.0)
        dt      = config.get("simulation", {}).get("dt", 0.05)

        eof_model = EOFModel(data["u"], data["v"], data["alt_grid"], n_modes=n_modes)
        return SEBWindModel(eof_model, dt=dt, seed=seed, scale=scale)

    def _build_recovery(self, config: dict):
        recovery_cfg = config.get("recovery", None)
        if recovery_cfg is None:
            return None

        cdas = recovery_cfg.get("CdA", [])
        alts = recovery_cfg.get("deployment_altitude", [])

        if len(cdas) != len(alts):
            raise ValueError(
                f"Recovery config mismatch: {len(cdas)} CdA value(s) vs {len(alts)} deployment_altitude(s)"
            )

        devices = [Recovery.Device(cda=cda, deploy_alt=alt) for cda, alt in zip(cdas, alts)]
        return Recovery(devices)

    def _build_rocket(self, config: dict) -> Rocket:
        rocket_cfg = config.get("rocket", {})

        ork_path = rocket_cfg.get("ORK_path", "") + rocket_cfg.get("ORK_filename", "")
        ras_path = rocket_cfg.get("RasAero_path", "") + rocket_cfg.get("RasAero_filename", "")
        ref_area = rocket_cfg.get("reference_area")

        rocket = Rocket.from_ork(ork_path, ras_path, ref_area, recovery=self._build_recovery(config))

        # Merge static overrides (from rocket.overrides:) with trade study overrides.
        # Trade study values take precedence.
        static_overrides = rocket_cfg.get("overrides", {}) or {}
        trade_overrides  = config.get("_trade_overrides", {})
        all_overrides    = {**static_overrides, **trade_overrides}
        if all_overrides:
            rocket.apply_overrides(all_overrides)

        return rocket

    def _build_sim(self, config: dict, seed: int) -> FlightSim:
        settings = self._build_settings(config)
        return FlightSim(
            rocket=self._build_rocket(config),
            wind_model=self._build_wind_model(config, seed),
            gravity=GravityModel(),
            init_state=self._build_init_state(config, settings.launch_rail_orientation),
            settings=settings,
        )

    def _extract_record(self, snaps: list, permutation: dict, perm_idx: int, trial_idx: int) -> dict:
        record_keys = self.config.get("record", [])

        row = {"perm_idx": perm_idx, "trial_idx": trial_idx}
        row.update(permutation)

        for key in record_keys:
            extractor = RECORD_EXTRACTORS.get(key)
            if extractor is None:
                warnings.warn(f"Unknown record key '{key}' — recording None")
                row[key] = None
            else:
                try:
                    row[key] = extractor(snaps)
                except Exception as e:
                    warnings.warn(f"Failed to extract '{key}': {e}")
                    row[key] = None

        return row

    def run(self) -> list[dict]:
        iterations_per_trial = self.config.get("simulation", {}).get("iterations_per_trial", 1)

        for perm_idx, permutation in enumerate(self._permutations):
            merged = self._merge_config(self.config, permutation)
            for trial_idx in range(iterations_per_trial):
                seed = perm_idx * iterations_per_trial + trial_idx
                sim  = self._build_sim(merged, seed)
                snaps = sim.run()
                self._snapshots[(perm_idx, trial_idx)] = snaps
                self._results.append(self._extract_record(snaps, permutation, perm_idx, trial_idx))

        return self._results

    def export_csv(self, output_path: str):
        pd.DataFrame(self._results).to_csv(output_path, index=False)

    def export_snapshots_csv(self, output_path: str, perm_idx: int = 0, trial_idx: int = 0):
        snaps = self._snapshots.get((perm_idx, trial_idx))
        if snaps is None:
            raise KeyError(f"No snapshots found for perm={perm_idx}, trial={trial_idx}")
        export_snapshots_csv(snaps, output_path)
