"""
Template for configuring and starting a multi-satellite run.

To run: 
PYTHONPATH=.:src python scripts/SCRIPT_NAME.py
                                ^^^^^^^^^^^ You'll have to add this in yourself.
This scripts execute line:
PYTHONPATH=.:src python scripts/simulation_template.py
"""

import math
import csv
from pathlib import Path

from state import elements_to_state
from propagator import propagate
from scenario import external_forces
from scenario import (
    EARTH,
    OrbitConfig, 
    RunConfig, 
    SatelliteConfig, 
    validate_run_config,
)
from results.plots import (
    plot_apsis_altitudes,
    plot_orbital_element_changes,
    plot_trajectories_3d,
)
from orbit_control import OrbitControlConfig, propagate_with_orbit_control


# Knobs and Switches
# Name switch:
SCRIPT_NAME = "simulation_template"


# Time switches:
RUN_DURATION_S = 86_400 * 1
# 86_400 = 1 day [s]
OUTPUT_INTERVAL_S = 900
# 900 = 15 minutes [s]

# Orbit control switch: when enabled, propagation is split into windows so
# orbital events can be checked and a policy can maneuver at those events.
ORBIT_CONTROL_ENABLED = True
CONTROL_CHECK_INTERVAL_S = OUTPUT_INTERVAL_S
ORBIT_EVENTS = ("ascending_node", "descending_node", "apogee", "perigee")
CONTROL_POLICY = None # TBD
VERBOSE = False
# Policy actions are discrete delta-v choices in the local RTN frame [m/s].
# Keep a zero-action entry so a policy can choose to coast.
MANEUVER_ACTIONS_RTN_M_S = (
    (0.0, 0.0, 0.0),
    (0.0, 0.01, 0.0),
    (0.0, -0.01, 0.0),
)


# Force switches:
force_complexity = 1
    # 0 - Low complexity, no external forces
    # 1 - High complexity, all external forces
    # 2 - Custom complexity, manually set

if force_complexity == 0:
    external_forces.use_sun_gravity = False
    external_forces.use_moon_gravity = False
    external_forces.use_drag = False
    external_forces.use_solar_radiation_pressure = False
    external_forces.use_spaceweather = False
    external_forces.use_solid_tides = False
    external_forces.use_relativistic_correction = False
    
elif force_complexity == 1:
    external_forces.use_sun_gravity = True
    external_forces.use_moon_gravity = True
    external_forces.use_drag = True
    external_forces.use_solar_radiation_pressure = True
    external_forces.use_spaceweather = True
    external_forces.use_solid_tides = True
    external_forces.use_relativistic_correction = True

elif force_complexity == 2:
    external_forces.use_sun_gravity = True
    external_forces.use_moon_gravity = False
    external_forces.use_drag = True
    external_forces.use_solar_radiation_pressure = False
    external_forces.use_spaceweather = True
    external_forces.use_solid_tides = True
    external_forces.use_relativistic_correction = True

else: 
    raise ValueError("Incorrect force complexity.")



# Plot switches: only plots set to True are generated.
PLOT_TRAJECTORIES_3D = True
PLOT_ORBITAL_ELEMENT_CHANGES = True
PLOT_APSIS_ALTITUDES = True

# Checklist for adding a plot:
# 1. Add a plot function in results/plots.py that saves under
#    results/images/<script name>/.
# 2. Import that function above and add a matching PLOT_* switch here.
# 3. Call the function in run_simulation under `if PLOT_*:`.



# Satellite(s) Configurations
SATELLITES = [                  # length: 1..N satellites
    {
        "sat_id": "sat_0001", 
        "a_m": 20_000_000.0,    # [m]
        "e": 0.50,             # [-]
        "i_deg": 45.0,          # [deg]
        "Omega_deg": 60.0,      # [deg]
        "w_deg": 90.0,          # [deg]
        "v_deg": 0.0,           # [deg]
        "mass_kg": 12.0,        # [kg]
        "area_m2": 1.0,         # [m^2]
        "drag_coef": 2.2,       # [-]
        "refl_coef": 1.3,       # [-]
    },
    {
        "sat_id": "sat_0002",
        "a_m": 21_000_000,
        "e": 0.002,
        "i_deg": 1.0,
        "Omega_deg": 60.0,
        "w_deg": 90.0,
        "v_deg": 120.0,      
        "mass_kg": 12.0,
        "area_m2": 1.0,
        "drag_coef": 2.2,
        "refl_coef": 1.3,
    },
    {
        "sat_id": "sat_0003",
        "a_m": 15_000_000,
        "e": 0.003,
        "i_deg": 45.0,
        "Omega_deg": 60.0,
        "w_deg": 90.0,
        "v_deg": 240.0,    
        "mass_kg": 12.0,
        "area_m2": 1.0,
        "drag_coef": 2.2,
        "refl_coef": 1.3,
    },
]



def build_run_configs() -> list[RunConfig]:
    """
    In:
        SATELLITES (list, length 1..N) 
            sat_id      - satellite identifier (string)
            a_m         - [m]
            e           - [-]
            i_deg       - [deg]
            Omega_deg   - [deg]
            w_deg       - [deg]
            v_deg       - [deg]
        mass_kg                 - [kg]
        area_m2 (flt)              - [m^2]
        drag_coef (flt)         - [-]
        refl_coef (flt)         - [-]
        RUN_DURATION_S (flt)    - Total simulation duration [s].
        OUTPUT_INTERVAL_S (flt) - Requested output interval [s].
    Out:
        (list[RunConfig]) - One independently configured run per satellite.

    Each dictionary is translated into the typed config objects used by the
    simulation. OrbitConfig receives only orbital elements; satellite identity
    and mass belong to RunConfig and SatelliteConfig respectively.
    """

    configs = []

    for sat in SATELLITES:
        # Build an orbit first, then pass that instance into RunConfig along
        # with the shared timing settings for this satellite's run.
        orbit = OrbitConfig(
            a=sat["a_m"],
            e=sat["e"],
            i=sat["i_deg"],
            Omega=sat["Omega_deg"],
            w=sat["w_deg"],
            v=sat["v_deg"],
        )
        # Keep sat_id on the run and mass in SatelliteConfig; neither is an
        # orbital element, so neither should be added to OrbitConfig.
        config = RunConfig(
            sat_id=sat["sat_id"],
            orbit=orbit,
            satellite=SatelliteConfig(
                mass_kg=sat["mass_kg"],
                area_m2=sat["area_m2"],
                drag_coef=sat["drag_coef"],
                refl_coef=sat["refl_coef"],
            ),
            forces=external_forces,
            dt=OUTPUT_INTERVAL_S,
            dt_End=RUN_DURATION_S,
        )

        configs.append(config)

    return configs


def write_satellite_csvs(
    configs: list[RunConfig],
    all_states: dict[str, list],
    all_elements: dict[str, list],
) -> None:
    """
    In:
        configs: RunConfig objects, one per satellite.
        all_states: State sample lists keyed by satellite ID.
        all_elements: Kepler element sample lists keyed by satellite ID.
            
    Out:
        None - Write one CSV per satellite under
              results/timesteps/<script name>/.

    """

    headers = [
        "sat_id", "time_s",
        "position_x_m", "position_y_m", "position_z_m",
        "velocity_x_m_s", "velocity_y_m_s", "velocity_z_m_s",
        "semi_major_axis_m", "eccentricity", "inclination_deg",
        "raan_deg", "argument_of_periapsis_deg", "true_anomaly_deg",
    ]

    output_dir = Path("results") / "timesteps" / SCRIPT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    for config in configs:
        # Results are keyed by the ID on this full RunConfig object.
        states = all_states[config.sat_id]
        elements = all_elements[config.sat_id]
        if len(states) != len(elements):
            raise ValueError(
                f"State/element sample count mismatch for {config.sat_id}: "
                f"{len(states)} states, {len(elements)} element sets"
            )

        # Each file contains one satellite's samples, one sample per row.
        output_path = output_dir / f"{config.sat_id}.csv"
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            # propagate returns matching lists: one State and one Kepler
            # element object for every sampled time.
            for state, kep in zip(states, elements):
                writer.writerow([
                    config.sat_id,
                    state.t,
                    *state.r,
                    *state.v,
                    kep.a,
                    kep.eccen,
                    math.degrees(kep.inclination),
                    math.degrees(kep.raan),
                    math.degrees(kep.argp),
                    math.degrees(kep.nu),
                ])



def run_simulation() -> None:
    """
    In:
        Script-level satellite, duration, and output interval settings above.
    Out:
        Propagated samples and one CSV per satellite under
        results/timesteps/<script name>/.
        Populated plots based on switches
    """

    # Call list of configs for each sat
    configs = build_run_configs()
    print(f"{len(configs)} Configs built!")

    all_states = {}
    all_elements = {}

    for config in configs:
        validate_run_config(config)
        print(f"Config validated for {config.sat_id}")

        # Build the initial cartesian state for propagation
        init_state = elements_to_state(
            config.orbit, 
            mu=EARTH.mu
            )

        # Run Propagation
        if ORBIT_CONTROL_ENABLED:
            states, elements, event_log = propagate_with_orbit_control(
                init_state,
                config.dt_End,
                config,
                OrbitControlConfig(
                    enabled=True,
                    check_interval_s=CONTROL_CHECK_INTERVAL_S,
                    events=ORBIT_EVENTS,
                    actions_rtn_m_s=MANEUVER_ACTIONS_RTN_M_S,
                    policy=CONTROL_POLICY,
                    verbose=VERBOSE,
                ),
            )
            print(f"Orbit-control events for {config.sat_id}: {len(event_log)}")
        else:
            states, elements = propagate(
                init_state,
                config.dt_End,
                config=config
            )

        all_states[config.sat_id] = states
        all_elements[config.sat_id] = elements

    # TODO: Add calculations for sat-sat distances

    print("States and Elements propagated for all satellites.\n"
          "Writing CSVs now.")
    # Write one CSV per satellite, with that satellite's samples as rows.
    write_satellite_csvs(
        configs,
        all_states, 
        all_elements,
        )

    print("CSVs written. Generating plots.")
    # Each plot is called here; its switch controls whether it is generated.
    if PLOT_TRAJECTORIES_3D:
        plot_trajectories_3d(SCRIPT_NAME)
    if PLOT_ORBITAL_ELEMENT_CHANGES:
        plot_orbital_element_changes(SCRIPT_NAME)
    if PLOT_APSIS_ALTITUDES:
        plot_apsis_altitudes(SCRIPT_NAME)

    print("Enabled plots generated.")

    pass


if __name__ == "__main__":
    run_simulation()
