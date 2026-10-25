"""
Scenario inputs for frequent referrals and constant calls.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import ClassVar
import math
import warnings

@dataclass(frozen=True)
class EarthConstants:
    """Fixed physical constants used by this Earth-only simulator."""

    G: float = 6.67384e-11
    mass: float = 5.972e24
    radius: float = 6.3781366e6
    J2: float = 0.0010826359
    mu: float = 3.986004418e14
    day: float = 86164.10
    a_GEO: float = 42164000.2


EARTH = EarthConstants()


@dataclass
class OrbitConfig:
    """
    User-editable initial orbit, expressed with classical elements.
    This stores only orbital elements; defaults are supplied by RunConfig.

    In:
        a (flt)         - semi-major axis [m]
        e (flt)         - eccentricity
        i (flt)         - inclination [degrees]
        Omega (flt)     - right ascension of ascending node [degrees]
        w (flt)         - argument of periapsis [degrees]
        v (flt)         - true anomaly [degrees]
    """

    a: float        # default: 20,000,000.0 m
    e: float        # default: 0.4
    i: float        # default: 45.0 deg
    Omega: float    # default: 60.0 deg
    w: float        # default: 90.0 deg
    v: float        # default: 0.0  deg



@dataclass
class SatelliteConfig:
    """
    Store the physical spacecraft inputs.
    Used to form Cd*A/m and Cr*A/m for drag and solar radiation pressure models
        SatKit has built in functionality for this
    """
    mass_kg: float = 12.0       # Random Value
    area_m2: float = 1.0        # Random Value
    drag_coef: float = 2.2      # Industry standard value
    refl_coef: float = 1.3      # Industry standard value


@dataclass
class ForceConfig:
    """
    Additional body forces. Current defaults enable the available perturbations.
    SatKit will provide Sun/Moon positions from ephemerides when enabled    
    """
    use_sun_gravity: bool = True
    use_moon_gravity: bool = True
    use_drag: bool = True
    use_solar_radiation_pressure: bool = True
    use_spaceweather: bool = True
    use_solid_tides: bool = True
    use_relativistic_correction: bool = True

external_forces = ForceConfig(
        use_sun_gravity=True, 
        use_moon_gravity=True, 
        use_drag=True,
        use_solar_radiation_pressure=True,
        use_spaceweather=True,
        use_solid_tides=True,
        use_relativistic_correction=True,
    )


@dataclass
class RunConfig:
    """
    sat_id                      - Identifier kept with this satellite's run
    orbit                       - Orbital elements for this run
    dt                          - Requested output sample interval [s]
    dt_End                      - Elapsed run duration [s]
    """

    defaultOrbit: ClassVar[OrbitConfig] = OrbitConfig(
        a=20_000_000.0,
        e=0.4,
        i=45.0,
        Omega=60.0,
        w=90.0,
        v=0.0,
    )

    # Start with a one-orbit Earth run. The 15 s interval controls output
    # samples; SatKit chooses its own adaptive internal integration steps.
    # The template supplies these fields for per-satellite runs; defaults keep
    # RunConfig useful for the single-run entry point in src/main.py.
    sat_id: str = "sat_0001"
    orbit: OrbitConfig = field(default_factory=lambda: OrbitConfig(
        a=RunConfig.defaultOrbit.a,
        e=RunConfig.defaultOrbit.e,
        i=RunConfig.defaultOrbit.i,
        Omega=RunConfig.defaultOrbit.Omega,
        w=RunConfig.defaultOrbit.w,
        v=RunConfig.defaultOrbit.v,
    ))
    dt: float = 15.0
    dt_End: float | None = None


    start_epoch_utc: datetime = field(
        default_factory=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc) 
    ) # January 01, 2025 0000z Start Time
    satellite: SatelliteConfig = field(default_factory=SatelliteConfig)
    forces: ForceConfig = field(default_factory=ForceConfig)
    abs_error_m: float = 1.0e-3
    rel_error: float = 1.0e-10

    def __post_init__(self):
        if self.dt_End is None:
            self.dt_End = 2.0 * math.pi * math.sqrt(
                self.orbit.a**3 / EARTH.mu
            ) # 1 orbit default



def validate_run_config(config: RunConfig):
    """
    Raise a useful error when an input is missing or physically invalid.
    """

    orbit = config.orbit

    # Check eccentricity
    e = orbit.e
    if not math.isfinite(e):
        raise ValueError(f"Eccentricity must be a finite value. e={e}")
    if e == 0:
        orbitType = "Circular"
    elif e > 0 and e < 1:
        orbitType = "Elliptical"
    elif e == 1:
        orbitType = "Parabolic"
        raise ValueError(f"Eccentricity is equal to 1, and therefore {orbitType}.")
    elif e > 1:
        orbitType = "Hyperbolic"
        raise ValueError(f"Eccentricity is greater than 1, "
                          f"and therefore {orbitType}. e={e}")
    else:
        raise ValueError(f"Eccentricity is not a valid value of "
                          "0 or greater. e={e}")


    # Check semi-major axis
    if not math.isfinite(orbit.a) or orbit.a <= 0:
        raise ValueError(f"Semi-major axis must be a finite value of "
                          "greater than 0 m. a={orbit.a}")


    # Check inclination
    if not math.isfinite(orbit.i) or not 0 <= orbit.i <= 180:
        raise ValueError(f"Inclination must be between "
                          "0 and 180 degrees. i={orbit.i}")


    # check orbital angles
    for angle_name in ("Omega", "w", "v"):
        angle = getattr(orbit, angle_name)
        if not math.isfinite(angle):
            raise ValueError(f"{angle_name} must be a finite angle "
                             f"in degrees. {angle_name}={angle}")


    # check tiem step and end time
    if not math.isfinite(config.dt) or config.dt <= 0:
        raise ValueError(f"Time step must be a finite value greater "
                          "than 0 seconds. dt={config.dt}")

    if not math.isfinite(config.dt_End) or config.dt_End <= 0:
        raise ValueError(f"End time must be a finite value greater "
                          "than 0 seconds. end_time={config.dt_End}")
    
    if config.dt > config.dt_End:
        warnings.warn("Output interval exceeds the run duration; "
                      "the trajectory will contain the initial and final states.")


    # Check UTC origin and spacecraft model parameters.
    if (config.start_epoch_utc.tzinfo is None or
            config.start_epoch_utc.utcoffset() is None):
        raise ValueError("start_epoch_utc must include a timezone (use UTC).")

    sat = config.satellite
    if sat.mass_kg is not None and (not math.isfinite(sat.mass_kg) or sat.mass_kg <= 0):
        raise ValueError("Satellite mass must be finite and greater than 0 kg.")
    if not math.isfinite(sat.area_m2) or sat.area_m2 < 0:
        raise ValueError("Satellite area must be finite and >= 0 m^2.")
    if not math.isfinite(sat.drag_coef) or sat.drag_coef < 0:
        raise ValueError("Drag coefficient must be finite and >= 0.")
    if not math.isfinite(sat.refl_coef) or sat.refl_coef < 0:
        raise ValueError("Reflectivity coefficient must be finite and >= 0.")
    if config.forces.use_drag or config.forces.use_solar_radiation_pressure:
        if sat.mass_kg is None or sat.area_m2 <= 0:
            raise ValueError("Drag/SRP requires satellite mass and positive area.")

    if not math.isfinite(config.abs_error_m) or config.abs_error_m <= 0:
        raise ValueError("abs_error_m must be finite and greater than zero.")
    if not math.isfinite(config.rel_error) or config.rel_error <= 0:
        raise ValueError("rel_error must be finite and greater than zero.")
