# Space debris simulation

An Earth-orbit simulation scaffold using SatKit. It supports a single-orbit
diagnostic run and a configurable multi-satellite run with per-satellite CSV
output and optional plots.

## Quick start

Use Python 3.10 or newer. From the repository root, create an environment and
install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the single-satellite diagnostic:

```bash
PYTHONPATH=.:src python src/main.py
```

Run the multi-satellite template:

```bash
PYTHONPATH=.:src python scripts/simulation_template.py
```

The template currently propagates three configured satellites for 100 days.
For a shorter first run, edit `RUN_DURATION_S` near the top of
`scripts/simulation_template.py`. Edit `SATELLITES` there to change orbital and
spacecraft inputs, and set `force_complexity` to choose the force configuration:

- `0`: Earth gravity only; disables the configured external forces.
- `1`: enables the configured third-body, drag, SRP, solid-tide, and
  relativistic effects.
- `2`: uses the force selections written in the custom branch of the template.

The template writes one CSV per satellite to
`results/timesteps/simulation_template/`. Enabled plots are saved to
`results/images/simulation_template/`. Edit `SCRIPT_NAME` if you copy the
template for another run; it determines both output folders.

SatKit may need to retrieve ephemeris, Earth-orientation, or space-weather data
on the first run. If its default data directory is not writable, set
`SATKIT_DATA` to a writable directory before running the simulation.

## Simulation model

The model is Earth-only. Earth constants are available from the immutable
`EARTH` object in `src/scenario.py`, including `EARTH.mu`, `EARTH.radius`,
`EARTH.mass`, and `EARTH.J2`. Orbit and Cartesian state units are meters,
meters/second, and seconds; the input orbit angles are in degrees. States use
the GCRF frame. `RunConfig.dt` controls the requested output cadence, while
SatKit chooses internal integration steps. The default single-run duration is
one orbit.

Earth gravity uses SatKit's EGM2008 spherical-harmonic model through degree and
order 8. Force switches are defined by `ForceConfig` in `src/scenario.py` and
mapped to SatKit settings in `src/propagator.py`. Drag and solar radiation
pressure use each satellite's mass, area, and coefficients. The template's
force-complexity setting is applied to every satellite run.

## Project layout

- `src/scenario.py`: orbit, spacecraft, force, and run configuration plus
  validation.
- `src/state.py`: conversion from Earth-centered orbital elements to GCRF
  position and velocity.
- `src/propagator.py`: SatKit propagation and sampled state/element results.
- `src/main.py`: single-satellite diagnostic entry point.
- `scripts/simulation_template.py`: multi-satellite run configuration,
  propagation, CSV writing, and plot switches.
- `results/plots.py`: plots generated from the per-satellite CSV files.

Generated CSV and PNG files are ignored by Git.
