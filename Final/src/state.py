"""Cartesian state and conversion from an initial orbit definition."""

from dataclasses import dataclass, field
from scenario import EARTH
import numpy as np



@dataclass
class State:
    """
    Position (m) and velocity (m/s) at one simulation time (s).
    SatKit's numerical propagator requires the GCRF frame.
    
    In:
        r (3, flt) - [m]
        v (3, flt) - [m/s]
        t (flt)    - Elapsed time from simulation start [s]
    Usage:
        state = State(r=np.array([1.0, 2.0, 3.0]),
                      v=np.array([0.1, 0.2, 0.3]))
        print(state.r)  # Output: [1. 2. 3.]
        print(state.v)  # Output: [0.1 0.2 0.3]
    """

    r: np.ndarray  # shape (3,), m
    v: np.ndarray  # shape (3,), m/s
    t: float = 0.0 # elapsed seconds from the simulation start

    def __post_init__(self):
        self.r = np.asarray(self.r, dtype=float).reshape(3).copy()
        self.v = np.asarray(self.v, dtype=float).reshape(3).copy()



def elements_to_state(
    orbit,
    time_s: float = 0.0,
    mu: float | None = None,
) -> State:
    """
    Convert classical elements to inertial Cartesian position and velocity.
    We start with calculating p and v in the perifocal frame, then converting
    to an Earth-centered inertial equatorial frame (treated as GCRF here).

    In:
        orbit - input as an "OrbitConfig" class call.
            a (flt)  - semi-major axis [m]
            e (flt)  - eccentricity
            v (flt)  - true anomaly [deg]
        time_s (flt) - current sim time
    Out:
        state = State(r=position_vector, v=velocity_vector, t=time_s)
            state.r = [rx ry rz]
            state.v = [vx vy vz]
    """

    if mu is None:
        mu = EARTH.mu
    a = orbit.a
    e = orbit.e
    v = orbit.v
    v_rad = np.deg2rad(v)

    p = a * (1 - e ** 2)

    # Position magnitude
    r_mag = p / (1 + e * np.cos(v_rad))

    # Periapsis vectors
    mu_p_ratio = np.sqrt(mu / p)
    r_pqw = np.array([r_mag * np.cos(v_rad),
                      r_mag * np.sin(v_rad),
                      0])

    v_pqw = np.array([-mu_p_ratio * np.sin(v_rad),
                      mu_p_ratio * (e + np.cos(v_rad)),
                      0])    

    # Perifocal to GCRF-like inertial equatorial coordinates.
    r_gcrf, v_gcrf = perifocal_to_gcrf(orbit, r_pqw, v_pqw)

    # Assign to State
    initial_state = State(r=r_gcrf, v=v_gcrf, t=time_s)

    return initial_state



def perifocal_to_gcrf(orbit, r_pqw, v_pqw):
    """
    Transforms position and velocity vectors from Perifocal (pqw) to 
    geocentric inertial equatorial axes used as GCRF for this prototype.

    In:
        orbit
            a (flt)         - semi-major axis [m]        {NOT USED}
            e (flt)         - eccentricity                {NOT USED}
            i (flt)         - inclination [degrees]
            Omega (flt)     - right ascension of ascending node [degrees]
            w (flt)         - argument of periapsis [degrees]
            v (flt)         - true anomaly [degrees]      {NOT USED}
        r_pqw (1x3 float)   - periapsis position [m]
        v_pqw (1x3 float)   - periapsis velocity [m/s]

    Out:
        r_gcrf (3 float)    - inertial position [m]
        v_gcrf (3 float)    - inertial velocity [m/s]
    """


    # Initialize
    Omega = orbit.Omega
    w = orbit.w
    i = orbit.i

    cos_O = np.cos(np.deg2rad(Omega))
    sin_O = np.sin(np.deg2rad(Omega))

    cos_w = np.cos(np.deg2rad(w))
    sin_w = np.sin(np.deg2rad(w))

    cos_i = np.cos(np.deg2rad(i))
    sin_i = np.sin(np.deg2rad(i))


    # Matrix Transformation Components
    # Row 1
    q11 = cos_O * cos_w - sin_O * sin_w * cos_i
    q12 = -cos_O * sin_w - sin_O * cos_w * cos_i
    q13 = sin_O * sin_i

    # Row 2
    q21 = sin_O * cos_w + cos_O * sin_w * cos_i
    q22 = -sin_O * sin_w + cos_O * cos_w * cos_i
    q23 = -cos_O * sin_i

    # Row 3
    q31 = sin_w * sin_i
    q32 = cos_w * sin_i
    q33 = cos_i

    
    # Reassemble matrix
    rot_matrix = np.array([
        [q11, q12, q13],
        [q21, q22, q23],
        [q31, q32, q33]
    ])

    
    # Matrix multiply vectors
    r_gcrf = np.dot(rot_matrix, r_pqw)
    v_gcrf = np.dot(rot_matrix, v_pqw)


    return r_gcrf, v_gcrf
