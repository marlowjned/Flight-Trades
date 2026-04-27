# main.py
# Entry point for running flight simulations from a config file.
#
# Usage:
#   python main.py <config_path> [--output <output_path>]
#
# Examples:
#   python main.py user_inputs/configs/test_config.yaml
#   python main.py user_inputs/configs/test_config.yaml --output results/run1.csv

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "6DOF"))

from flight_sim.core.sim_handler import SimulationHandler


def default_output_path(config_path: str) -> str:
    base = os.path.splitext(os.path.basename(config_path))[0]
    return f"{base}_results.csv"


def main():
    parser = argparse.ArgumentParser(description="Run a flight simulation trade study.")
    parser.add_argument("config", help="Path to the YAML config file")
    parser.add_argument("--output", "-o", default=None, help="Output CSV path (default: <config>_results.csv)")
    parser.add_argument("--snapshots", "-s", default=None, help="Export full flight snapshots CSV for perm 0 trial 0")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: config file not found: {args.config}")
        sys.exit(1)

    output_path = args.output or default_output_path(args.config)

    print(f"Config:  {args.config}")
    print(f"Output:  {output_path}")

    handler = SimulationHandler(args.config)

    n_perms  = len(handler._permutations)
    n_trials = handler.config.get("simulation", {}).get("iterations_per_trial", 1)
    print(f"Running: {n_perms} permutation(s) x {n_trials} trial(s) = {n_perms * n_trials} simulation(s)\n")

    results = handler.run()

    handler.export_csv(output_path)

    record_keys = handler.config.get("record", [])
    if results and record_keys:
        print("Summary:")
        for key in record_keys:
            vals = [r[key] for r in results if r.get(key) is not None]
            if vals:
                print(f"  {key}: min={min(vals):.2f}  max={max(vals):.2f}  mean={sum(vals)/len(vals):.2f}")

    print(f"\nResults written to {output_path}")

    if args.snapshots:
        handler.export_snapshots_csv(args.snapshots)
        print(f"Snapshots written to {args.snapshots}")


if __name__ == "__main__":
    main()
