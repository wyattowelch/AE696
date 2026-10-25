"""Plot satellite trajectories and orbital-element changes from run CSVs."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EARTH_RADIUS_M = 6_378_136.6


def _load_satellite_csvs(script_name: str) -> dict[str, dict[str, np.ndarray]]:
    """Load one CSV per satellite from results/timesteps/<script_name>."""
    csv_dir = Path("results") / "timesteps" / script_name
    if not csv_dir.is_dir():
        raise FileNotFoundError(f"Satellite CSV directory not found: {csv_dir}")

    satellites = {}
    for csv_path in sorted(csv_dir.glob("*.csv")):
        with csv_path.open("r", newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            continue

        sat_id = rows[0]["sat_id"]
        columns = (
            "time_s", "position_x_m", "position_y_m", "position_z_m",
            "semi_major_axis_m", "eccentricity", "inclination_deg",
            "raan_deg", "argument_of_periapsis_deg", "true_anomaly_deg",
        )
        satellites[sat_id] = {
            name: np.asarray([float(row[name]) for row in rows], dtype=float)
            for name in columns
        }
    if not satellites:
        raise ValueError(f"No satellite samples found in {csv_dir}")
    return satellites


def _plot_output_dir(script_name: str) -> Path:
    """Create and return the output directory for a simulation's plots."""
    output_dir = Path("results") / "images" / script_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_trajectories_3d(script_name: str) -> None:
    """Plot 3D trajectories for all satellites in this simulation."""
    satellites = _load_satellite_csvs(script_name)
    output_dir = _plot_output_dir(script_name)

    # Plot 1 — 3D paths show that each satellite has a distinct propagated orbit.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for sat_id, data in satellites.items():
        ax.plot(
            data["position_x_m"] / 1e6,
            data["position_y_m"] / 1e6,
            data["position_z_m"] / 1e6,
            label=sat_id,
        )
    # Draw Earth as a translucent reference sphere in the same kilometer scale.
    u = np.linspace(0, 2 * np.pi, 36)
    v = np.linspace(0, np.pi, 18)
    earth_r = EARTH_RADIUS_M / 1e6
    ax.plot_surface(
        earth_r * np.outer(np.cos(u), np.sin(v)),
        earth_r * np.outer(np.sin(u), np.sin(v)),
        earth_r * np.outer(np.ones_like(u), np.cos(v)),
        color="steelblue", alpha=0.25, linewidth=0,
    )
    ax.set_title("Satellite 3D trajectories")
    ax.set_xlabel("GCRF x (10³ km)")
    ax.set_ylabel("GCRF y (10³ km)")
    ax.set_zlabel("GCRF z (10³ km)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "trajectories_3d.png", dpi=160)
    plt.close(fig)


def plot_orbital_element_changes(script_name: str) -> None:
    """Plot changes in semi-major axis, eccentricity, inclination, and RAAN."""
    satellites = _load_satellite_csvs(script_name)
    output_dir = _plot_output_dir(script_name)

    # Element changes from each initial sample expose perturbation drift.
    element_specs = [
        ("semi_major_axis_m", "Semi-major axis change (m)"),
        ("eccentricity", "Eccentricity change"),
        ("inclination_deg", "Inclination change (deg)"),
        ("raan_deg", "RAAN change (deg)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, (field, ylabel) in zip(axes.flat, element_specs):
        for sat_id, data in satellites.items():
            elapsed_days = data["time_s"] / 86400.0
            change = data[field] - data[field][0]
            ax.plot(elapsed_days, change, label=sat_id)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    for ax in axes[-1, :]:
        ax.set_xlabel("Elapsed time (days)")
    axes[0, 0].legend()
    fig.suptitle("Orbital element changes from initial values")
    fig.tight_layout()
    fig.savefig(output_dir / "orbital_element_changes.png", dpi=160)
    plt.close(fig)


def plot_apsis_altitudes(script_name: str) -> None:
    """Plot each satellite's perigee and apogee altitude over time."""
    satellites = _load_satellite_csvs(script_name)
    output_dir = _plot_output_dir(script_name)

    # Apsis altitudes show changes in each orbit's closest and farthest points.
    fig, ax = plt.subplots(figsize=(10, 6))
    for sat_id, data in satellites.items():
        perigee_altitude_km = (
            data["semi_major_axis_m"] * (1.0 - data["eccentricity"])
            - EARTH_RADIUS_M
        ) / 1000.0
        apogee_altitude_km = (
            data["semi_major_axis_m"] * (1.0 + data["eccentricity"])
            - EARTH_RADIUS_M
        ) / 1000.0
        elapsed_days = data["time_s"] / 86400.0
        ax.plot(elapsed_days, perigee_altitude_km, label=f"{sat_id} perigee")
        ax.plot(
            elapsed_days, apogee_altitude_km, linestyle="--",
            label=f"{sat_id} apogee",
        )
    ax.set_title("Perigee and apogee altitude over time")
    ax.set_xlabel("Elapsed time (days)")
    ax.set_ylabel("Altitude above Earth (km)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "apsis_altitudes.png", dpi=160)
    plt.close(fig)
