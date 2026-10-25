"""
SatKit-backed propagation plus a place for a future local RK4 implementation.
"""

from __future__ import annotations

from datetime import timezone
from math import ceil
import numpy as np
import satkit as sk
from scenario import EARTH, RunConfig
from state import State


# This adapter uses SatKit's adaptive RKV98 integrator. 
# Config.dt is the requested output sampling interval.
# SatKit chooses its own internal step sizes.
def _satkit_settings(config: RunConfig):
    """
    Build explicit, reproducible SatKit settings from this run's config.
    """
    return sk.propsettings(
        # Include Earth harmonics through degree/order 8; the separate force
        # switches below control third-body gravity and space-weather use.
        gravity_degree=8,       # J8 harmonics (way more than enough)
        gravity_order=8,
        use_sun_gravity=config.forces.use_sun_gravity,
        use_moon_gravity=config.forces.use_moon_gravity,
        use_spaceweather=config.forces.use_spaceweather,
        tide_model=(
            sk.tidemodel.solid_step1
            if config.forces.use_solid_tides
            else sk.tidemodel.none
        ),
        use_relativistic_correction=config.forces.use_relativistic_correction,
        integrator=sk.integrator.rkv98,
        abs_error=config.abs_error_m,
        rel_error=config.rel_error,
        enable_interp=True,
    )


# Satellite drag/solar pressure forces solved using configurated values.
# SatKit handles moon and sun ephemeris / direction vectors.
def _satkit_satellite_properties(config: RunConfig):
    sat = config.satellite
    cdaoverm = 0.0
    craoverm = 0.0
    if sat.mass_kg is not None and sat.mass_kg > 0.0:
        if config.forces.use_drag:
            cdaoverm = sat.drag_coef * sat.area_m2 / sat.mass_kg
        if config.forces.use_solar_radiation_pressure:
            craoverm = sat.refl_coef * sat.area_m2 / sat.mass_kg
    return sk.satproperties(cdaoverm=cdaoverm, craoverm=craoverm)


# Propagate to an elapsed simulation end-time and return State samples
# at config.dt intervals, including the exact final endpoint. 
def propagate(
    initial_state: State, end_time_s: float, config: RunConfig
) -> tuple[list[State], list[sk.kepler]]:
    """
    Return sampled states through ``end_time_s`` after run start.

    Integrates with adaptive RKV98 . ``dt`` controls samples returned to the 
    caller, not the adaptive integrator step.

    State vectors are GCRF meters and meters/second throughout this module.
    """
    if not np.isfinite(end_time_s) or end_time_s < initial_state.t:
        raise ValueError("end_time_s must be finite and >= initial_state.t")
    if not np.isfinite(config.dt) or config.dt <= 0.0:
        raise ValueError("config.dt must be finite and greater than zero")

    start_t = float(initial_state.t)
    elapsed = float(end_time_s) - start_t
    samples = [initial_state]
    elements = [
        sk.kepler.from_pv(
            initial_state.r, initial_state.v, mu=EARTH.mu
        )
    ]
    if elapsed == 0.0:
        return samples, elements

    # Normalize offsets before passing date components as UTC.
    epoch = config.start_epoch_utc.astimezone(timezone.utc)
    begin = sk.time(
        epoch.year,
        epoch.month,
        epoch.day,
        epoch.hour,
        epoch.minute,
        epoch.second + epoch.microsecond / 1_000_000.0,
        scale=sk.timescale.UTC,
    ) + sk.duration(seconds=start_t)

    # Here is the actual propagator from SatKit
    result = sk.propagate(
        np.concatenate((initial_state.r, initial_state.v)),
        begin,
        duration_secs=elapsed,
        propsettings=_satkit_settings(config),
        satproperties=_satkit_satellite_properties(config),
    )

    # Interpolate the adaptive solution on the requested output grid. Include
    # one final shorter interval when duration is not divisible by config.dt.
    count = ceil(elapsed / config.dt)
    for index in range(1, count + 1):
        sample_t = min(start_t + index * config.dt, float(end_time_s))
        sample_epoch = begin + sk.duration(seconds=sample_t - start_t)
        state_vector = np.asarray(result.interp(sample_epoch), dtype=float).reshape(6)
        r_vec = state_vector[:3]
        v_vec = state_vector[3:]
        samples.append(
            State(r=r_vec, v=v_vec, t=sample_t)
        )
        elements.append(
            sk.kepler.from_pv(r_vec, v_vec, mu=EARTH.mu)
        )
    return samples, elements

