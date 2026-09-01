# Flight-Trades

A 6DOF rocket flight-simulation and trade-study framework, paired with a
reduced-order stochastic wind model (SEB-windmodel) for Monte Carlo
dispersion analysis. Used here for the ESPACE sounding-rocket sizing and
stability trade studies.

## Repo layout

```
Flight-Trades/
├── main.py                    # generic entry point: run any YAML config through the 6DOF sim
├── run_espace_trade.py        # ESPACE thrust trade study (sizing + aero + Monte Carlo wind)
├── run_new_trade.py           # trade study for the redesigned stable ESPACE airframes (v0/v1/v2)
├── stability_quality.py       # standalone static-stability-margin analyser (no full 6DOF run needed)
├── wind_data_parser.py        # fetches + reduces ERA5 wind data (via cdsapi) into CSVs
│
├── 6DOF/                      # flight_sim package: 6DOF equations of motion, sim loop, config loader
├── SEB-windmodel/             # seb_wind_model package: EOF/Von Kármán stochastic wind model
│
├── user_inputs/
│   ├── configs/                # YAML sim configs (see "Running a simulation" below)
│   └── rocket_data/            # ORK mass/CG/MOI + RasAero aero-coefficient CSVs, keyed by rocket name
│                                #   p1*        — Prospect 1
│                                #   espace_F*  — ESPACE thrust trade (8-14 kN)
│                                #   new_*      — redesigned stable airframe, v2 tables (default)
│                                #   v1_* / v2_*— redesigned airframe, explicit version tables
│                                #   espace_sizing.py       — propellant/tank sizing model for ESPACE
│                                #   generate_aero_csvs.py  — builds espace_F*_aero.csv from stability tables
│
├── data/                       # static reference/input datasets (not simulation output)
│   ├── wind/
│   │   └── december_07_2025_1300_at_35.35_-117.81/   # ERA5 wind data + reduced CSVs for the Mojave/FAR site
│   ├── new-aero-tables/        # source aero/mass spreadsheets for the new airframe, v0/v1/v2
│   ├── stability-analysis/     # RASAero stability tables + interpolator scripts (used by generate_aero_csvs.py
│   │                           #   and stability_quality.py)
│   └── ESPACEmassdata.xlsx     # structural dry-mass breakdown referenced by espace_sizing.py
│
├── results/                    # simulation output (git-trackable CSVs land here, organized per study)
│   ├── espace_trade/
│   ├── new_trade/
│   └── legacy/                 # older ad-hoc run outputs kept for reference
│
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
pip install -e SEB-windmodel/
pip install -e 6DOF/
```

`cdsapi` (used only by `wind_data_parser.py` to download ERA5 data) requires a
CDS API key configured in `~/.cdsapirc` — not needed if you're just running
simulations against the existing wind data in `data/wind/`.

## Running a simulation

`main.py` is the generic entry point — point it at any YAML config under
`user_inputs/configs/`:

```bash
python main.py user_inputs/configs/test_config.yaml
python main.py user_inputs/configs/seb_wind_test_config.yaml --output results/run1.csv
python main.py user_inputs/configs/test_config.yaml --snapshots results/run1_snapshots.csv
```

A config specifies (see `user_inputs/configs/sample_config.yaml` for an
annotated template):
- `simulation` — timestep, single-trial vs. trade-study sweep, iterations per trial
- `rocket` — path + filename of the ORK-style mass/CG/MOI CSV and the RasAero
  aero-coefficient CSV (both from `user_inputs/rocket_data/`)
- `trade_study` — swept parameters (mass, thrust, etc.) for trade-study runs
- `recovery` — parachute CdA / deployment altitudes
- `wind` — `simple` (analytic turbulence) or `SEB-windmodel` (ERA5 + Von
  Kármán stochastic model, pointed at a `.nc` file under `data/wind/`)
- `record` — which output fields to summarize/export

## Trade study scripts

These are self-contained studies built on top of the same `SimulationHandler`
used by `main.py`. Run them from the project root:

```bash
# ESPACE thrust trade (8-14 kN): sizes propellant/tanks, generates aero
# tables, runs Monte Carlo wind trials, aggregates apogee/dispersion stats
python run_espace_trade.py --trials 1 --simple-wind   # smoke test
python run_espace_trade.py                             # full run (30 trials, ERA5 wind)

# Redesigned stable-airframe trade study (v0/v1/v2 tank lengths)
python run_new_trade.py --ver v2 --trials 30
python run_new_trade.py --ver v1 --era5

# Static stability margin / pitch-dynamics analysis (fast, no full 6DOF run)
python stability_quality.py --thrust 8000 12000 --plot --csv results/stability.csv
```

Each writes its results under `results/<study_name>/`, including a
per-config CSV and a single aggregate CSV across all configs/thrust levels.

## Regenerating input data

- `user_inputs/rocket_data/espace_sizing.py` — recomputes propellant mass,
  tank lengths, CG/MOI curves, and ORK CSVs for each ESPACE thrust level
  from first principles (uses structural masses from `data/ESPACEmassdata.xlsx`).
- `user_inputs/rocket_data/generate_aero_csvs.py` — interpolates the RASAero
  stability tables in `data/stability-analysis/` to each tank length and
  writes `espace_F*_aero.csv`.
- `wind_data_parser.py` — downloads and reduces ERA5 wind data via `cdsapi`
  into the CSV format used under `data/wind/`.

These are only needed if you're changing the rocket geometry, thrust levels,
or launch site/date — the generated CSVs are already checked into
`user_inputs/rocket_data/` and `data/wind/`.
