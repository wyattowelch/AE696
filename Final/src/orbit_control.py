"""Optional orbital-event checks and policy-driven impulsive maneuver scaffold."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import satkit as sk
import logging

from propagator import propagate
from scenario import EARTH, RunConfig
from state import State


logger = logging.getLogger(__name__)


@dataclass
class OrbitControlConfig:
    """
    Settings for event checks and the policy action interface.

    Policy left to none, see orbit_control_scaffold.md for implementation
    once policy has been created
    """

    enabled: bool = False
    check_interval_s: float = 900.0
    events: tuple[str, ...] = ("ascending_node", "descending_node", "apogee", "perigee")
    actions_rtn_m_s: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),)
    policy: object | None = None
    verbose: bool = False


def _event_value(state: State, event_name: str) -> float:
    """
    Return the scalar whose zero crossing defines the requested event.

    Returns k at the 'left' and 'right' states if checking asc/desc nodes
    Returns r dot v for 'left' and 'right' if checking apogee/perigee
    """
    if event_name in ("ascending_node", "descending_node"):
        return float(state.r[2])
    if event_name in ("apogee", "perigee"):
        return float(np.dot(state.r, state.v))
    raise ValueError(f"Unsupported orbital event: {event_name}")


def _crossed_event(left: State, right: State, event_name: str) -> bool:
    """
    Check whether a pair of propagated states brackets an orbital event.
    """
    a = _event_value(left, event_name)
    b = _event_value(right, event_name)
    if a == 0.0 or a * b > 0.0:
        return False
    if event_name == "ascending_node":
        return a < 0.0 and b >= 0.0
    if event_name == "descending_node":
        return a > 0.0 and b <= 0.0
    if event_name == "apogee":
        return a > 0.0 and b <= 0.0
    return a < 0.0 and b >= 0.0


def _interpolate_event(left: State, right: State, event_name: str) -> State:
    """
    Linearly interpolate position and velocity to the event surface.
    """
    # TODO: Replace the scalar linear root estimate with SatKit's dense
    # interpolant once the propagator exposes it to this event layer.
    a = _event_value(left, event_name)
    b = _event_value(right, event_name)
    fraction = float(np.clip(-a / (b - a), 0.0, 1.0))
    return State(
        r=left.r + fraction * (right.r - left.r),
        v=left.v + fraction * (right.v - left.v),
        t=left.t + fraction * (right.t - left.t),
    )


def _refine_event(
    left: State,
    right: State,
    event_name: str,
    run_config: RunConfig,
    time_tolerance_s: float = 0.1,
) -> State:
    """Refine a sampled event bracket using repeated SatKit propagation."""
    while right.t - left.t > time_tolerance_s:
        midpoint_t = (left.t + right.t) / 2.0
        midpoint_states, _ = propagate(left, midpoint_t, run_config)
        midpoint = midpoint_states[-1]
        if _crossed_event(left, midpoint, event_name):
            right = midpoint
        else:
            left = midpoint
    return _interpolate_event(left, right, event_name)


def _policy_action(config: OrbitControlConfig, event_name: str, state: State) -> int:
    """
    Ask the configured policy for an action index at an orbital event.
    """
    # TODO: Build and normalize the observation expected by the trained DQN.
    # Suggested observation: event type, elapsed time, RTN state, and elements.
    if config.policy is None:
        return 0
    event_index = ("ascending_node", "descending_node", "apogee", "perigee").index(event_name)
    observation = np.concatenate(([state.t], state.r, state.v, [event_index]))
    action, _state = config.policy.predict(observation, deterministic=True)
    return int(np.asarray(action).reshape(-1)[0])


def _apply_rtn_delta_v(state: State, delta_v_rtn_m_s: tuple[float, float, float]) -> State:
    """
    Apply an impulsive RTN-frame delta-v and return the maneuvered state.
    """
    radial = state.r / np.linalg.norm(state.r)
    normal = np.cross(state.r, state.v)
    normal = normal / np.linalg.norm(normal)
    transverse = np.cross(normal, radial)
    basis = np.column_stack((radial, transverse, normal))
    delta_v_gcrf = basis @ np.asarray(delta_v_rtn_m_s, dtype=float)
    return State(r=state.r, v=state.v + delta_v_gcrf, t=state.t)


def propagate_with_orbit_control(
    initial_state: State,
    end_time_s: float,
    run_config: RunConfig,
    control_config: OrbitControlConfig,
) -> tuple[list[State], list[sk.kepler], list[dict]]:
    """
    Propagate in windows, check event crossings, and apply policy maneuvers.
    """
    if not control_config.enabled:
        states, elements = propagate(initial_state, end_time_s, run_config)
        return states, elements, []
    if not np.isfinite(control_config.check_interval_s) or control_config.check_interval_s <= 0:
        raise ValueError("check_interval_s must be finite and greater than zero")

    states = [initial_state]
    event_log = []
    last_event_time = {}
    current = initial_state
    while current.t < end_time_s:
        window_end = min(current.t + control_config.check_interval_s, end_time_s)
        try:
            window_states, _ = propagate(current, window_end, run_config)
        except RuntimeError as exc:
            logger.exception(
                "SatKit propagation failed for %s over elapsed-time window "
                "[%.9f, %.9f] s",
                run_config.sat_id,
                current.t,
                window_end,
            )
            raise
        event_hit = None
        # Find the earliest event in this window, then restart propagation at
        # its interpolated state so no already-computed future state survives.
        for left, right in zip(window_states, window_states[1:]):
            candidates = []
            for name in control_config.events:
                if not _crossed_event(left, right, name):
                    continue
                event_state = _refine_event(left, right, name, run_config)
                if event_state.t - last_event_time.get(name, -np.inf) > 30.0:
                    candidates.append((event_state, name))
            if candidates:
                event_state, event_name = min(candidates, key=lambda item: item[0].t)
                event_hit = (left, event_state, event_name)
                break

        if event_hit is None: # If there is NO EVENT
            states.extend(window_states[1:]) # Build out states until window end
            current = window_states[-1] # start new current at end of old window
            continue

        # If there IS EVENT
        left, event_state, event_name = event_hit
        states.extend(state for state in window_states[1:] if state.t < event_state.t)
        if not states or states[-1].t != event_state.t:
            states.append(event_state) # append final if none/if times dont line up
        action_index = _policy_action(control_config, event_name, event_state)
        if not 0 <= action_index < len(control_config.actions_rtn_m_s):
            raise ValueError(f"Policy action {action_index} is outside the configured action list")
        delta_v = control_config.actions_rtn_m_s[action_index]
        current = _apply_rtn_delta_v(event_state, delta_v)
        if control_config.verbose:
            logger.warning(
                "Orbit event: sat=%s event=%s t=%.6f s action=%d delta_v_rtn_m_s=%s",
                run_config.sat_id,
                event_name,
                event_state.t,
                action_index,
                tuple(delta_v),
            )
        # Replace the event sample with its post-maneuver velocity so saved
        # history and all continuing propagation use the commanded velocity.
        states[-1] = current
        event_log.append({
            "event": event_name,
            "time_s": event_state.t,
            "action_index": action_index,
            "delta_v_rtn_m_s": tuple(delta_v),
            "pre_maneuver_state": event_state,
            "post_maneuver_state": current,
        })
        last_event_time[event_name] = event_state.t

    elements = [sk.kepler.from_pv(state.r, state.v, mu=EARTH.mu) for state in states]
    return states, elements, event_log
