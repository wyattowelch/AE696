import numpy as np
import math
import satkit as sk
from scenario import EARTH


def _perifocal_to_gcrf(r_j_pf, v_j_pf, omega, inclination, raan):
    """
    Convert position and velocity from perifocal coordinates to GCRF.

    Angles must be in radians. Position and velocity vectors must each
    contain three components. SatKit's Keplerian angles are in radians.
    """

    def R1(angle):
        c = np.cos(angle)
        s = np.sin(angle)

        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, c,  -s],
            [0.0, s,   c],
        ])

    def R3(angle):
        c = np.cos(angle)
        s = np.sin(angle)

        return np.array([
            [ c, -s, 0.0],
            [ s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ])

    # Perifocal -> GCRF rotation
    Q_pf_to_gcrf = (
        R3(raan)
        @ R1(inclination)
        @ R3(omega)
    )

    r_j_gcrf = Q_pf_to_gcrf @ np.asarray(r_j_pf, dtype=float).reshape(3)
    v_j_gcrf = Q_pf_to_gcrf @ np.asarray(v_j_pf, dtype=float).reshape(3)

    return r_j_gcrf, v_j_gcrf

def find_ideal_events(init, control_config):
    """
    Map each configured orbital event to its ideal GCRF (position, velocity).

    The returned values are ``(r, v)`` NumPy vectors in metres and metres
    per second, respectively.
    """
    elements = sk.kepler.from_pv(init.r, init.v, mu=EARTH.mu)
    omega = elements.argp
    e = elements.eccen
    a = elements.a
    inclination = elements.inclination
    raan = elements.raan
    event_anomalies = {
        "ascending_node": -omega,
        "descending_node": math.pi - omega,
        "apogee": math.pi,
        "perigee": 0.0,
    }

    p = a * (1.0 - e**2)
    ideals = {}
    for event_name in control_config.events:
        try:
            nu = event_anomalies[event_name]
        except KeyError:
            raise ValueError(f"Unsupported orbital event: {event_name}") from None

        r_mag = p / (1.0 + e * np.cos(nu))
        r_pf = np.array([r_mag * np.cos(nu), r_mag * np.sin(nu), 0.0])
        v_pf = math.sqrt(EARTH.mu / p) * np.array(
            [-np.sin(nu), e + np.cos(nu), 0.0]
        )

        ideals[event_name] = _perifocal_to_gcrf(
            r_pf, v_pf, omega, inclination, raan
        )

    return ideals
