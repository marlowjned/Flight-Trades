# SimulationHandler.py
# Handles config file, creates actual trade framework, and packages data

import itertools
import copy
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import ConfigLoader
import Rocket
import Environment
import GravityModel
import SimulationLoop
import SimSnapshot
import Vector3D


RECORD_EXTRACTORS = {
    "apogee":       lambda snaps: max(s.altitude for s in snaps),
    "flight_time":  lambda snaps: snaps[-1].time,
    "max_velocity": lambda snaps: max(s._state.velocity.magnitude for s in snaps),
}


class SimulationHandler:

    def __init__(self, configPath: str):
        self.config = ConfigLoader.loadConfig(configPath)
        self._permutations = self._generate_permutations()
        self._results = []

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

    def _build_settings(self, config: dict) -> SimulationLoop.FlightSim.SimulationSettings:
        sim_cfg = config.get("simulation", {})
        overrides = config.get("_trade_overrides", {})

        dt = sim_cfg.get("dt", 0.05)
        max_runtime = sim_cfg.get("max_runtime", 200)
        recovery_sim = sim_cfg.get("recovery_sim", False)
        launch_rail_len = sim_cfg.get("launch_rail_length", 5.0)

        rail_dir = np.array(sim_cfg.get("launchrail_orientation", [0, 0, 1]), dtype=float)
        launch_rail_orientation = Rotation.align_vectors([rail_dir], [[0, 0, 1]])[0]

        return SimulationLoop.FlightSim.SimulationSettings(
            dt=dt,
            maxRuntime=max_runtime,
            recoverySim=recovery_sim,
            launchRailLen=launch_rail_len,
            launchRailOrientation=launch_rail_orientation,
        )

    def _build_init_state(self, config: dict, orientation: Rotation) -> SimulationLoop.FlightSim.FlightState:
        dcm = orientation.as_matrix()
        return SimulationLoop.FlightSim.FlightState(
            time=0.0,
            position=Vector3D.Vector3D(np.zeros(3), dcm),
            velocity=Vector3D.Vector3D(np.zeros(3), dcm),
            orientation=orientation,
            omega=Vector3D.Vector3D(np.zeros(3), dcm, True),
        )

    def _build_environment(self, config: dict, seed: int) -> Environment.Environment:
        env = Environment.Environment()
        env._config_overrides = config.get("_trade_overrides", {})
        return env

    def _build_recovery(self, config: dict):
        return None

    def _build_rocket(self, config: dict) -> Rocket.Rocket:
        rocket_cfg = config.get("rocket", {})
        overrides = config.get("_trade_overrides", {})

        ork_path = rocket_cfg.get("ORK_path", "") + rocket_cfg.get("ORK_filename", "")
        ras_path = rocket_cfg.get("RasAero_path", "") + rocket_cfg.get("RasAero_filename", "")

        rocket = Rocket.Rocket.from_ork(ork_path, ras_path, recovery=self._build_recovery(config))

        config_overrides = {
            k[len("OVERRIDE_"):]: v
            for k, v in rocket_cfg.items()
            if k.startswith("OVERRIDE_")
        }
        config_overrides.update(overrides)
        rocket._config_overrides = config_overrides

        return rocket

    def _build_sim(self, config: dict, seed: int) -> SimulationLoop.FlightSim:
        settings = self._build_settings(config)
        return SimulationLoop.FlightSim(
            rocket=self._build_rocket(config),
            environment=self._build_environment(config, seed),
            gravity=GravityModel.GravityModel(),
            init_state=self._build_init_state(config, settings.launchRailOrientation),
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
                sim = self._build_sim(merged, seed)
                snaps = sim.run()
                self._results.append(self._extract_record(snaps, permutation, perm_idx, trial_idx))

        return self._results

    def export_csv(self, output_path: str):
        pd.DataFrame(self._results).to_csv(output_path, index=False)
