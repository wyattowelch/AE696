"""Write per-satellite orbit-control records under ``results/logs``."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping

import numpy as np


def _format_json(value, depth: int = 0) -> str:
    """Pretty-print JSON while keeping simple vectors and lists on one line."""
    indent = "  " * depth
    child_indent = "  " * (depth + 1)

    if isinstance(value, Mapping):
        if not value:
            return "{}"
        items = [
            f"{child_indent}{json.dumps(str(key))}: {_format_json(item, depth + 1)}"
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(items) + f"\n{indent}}}"

    if isinstance(value, list):
        if not value:
            return "[]"
        if all(not isinstance(item, (Mapping, list)) for item in value):
            return json.dumps(value, separators=(", ", ": "))
        items = [f"{child_indent}{_format_json(item, depth + 1)}" for item in value]
        return "[\n" + ",\n".join(items) + f"\n{indent}]"

    return json.dumps(value)


def _state_record(state) -> dict:
    return {
        "time_s": float(state.t),
        "r_gcrf_m": np.asarray(state.r, dtype=float).tolist(),
        "v_gcrf_m_s": np.asarray(state.v, dtype=float).tolist(),
    }


def write_satellite_event_log(
    script_name: str,
    sat_id: str,
    event_log: list[dict],
    *,
    control_enabled: bool,
    states: list,
    events: tuple[str, ...] = (),
) -> Path:
    """Write a JSON record for one satellite and return its path."""
    output_dir = Path("results") / "logs" / script_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{sat_id}.json"

    event_records = []
    for event in event_log:
        event_records.append({
            "event": event["event"],
            "time_s": float(event["time_s"]),
            "action_index": int(event["action_index"]),
            "delta_v_rtn_m_s": list(event["delta_v_rtn_m_s"]),
            "ideal_r_gcrf_m": np.asarray(event["ideal_r_gcrf_m"], dtype=float).tolist(),
            "ideal_v_gcrf_m_s": np.asarray(event["ideal_v_gcrf_m_s"], dtype=float).tolist(),
            "position_error_gcrf_m": np.asarray(event["position_error_gcrf_m"], dtype=float).tolist(),
            "velocity_error_gcrf_m_s": np.asarray(event["velocity_error_gcrf_m_s"], dtype=float).tolist(),
            "pre_maneuver_state": _state_record(event["pre_maneuver_state"]),
            "post_maneuver_state": _state_record(event["post_maneuver_state"]),
        })

    record = {
        "script": script_name,
        "satellite": sat_id,
        "orbit_control_enabled": bool(control_enabled),
        "configured_events": list(events),
        "sample_count": len(states),
        "start_time_s": float(states[0].t) if states else None,
        "end_time_s": float(states[-1].t) if states else None,
        "detected_event_count": len(event_records),
        "events": event_records,
    }
    output_path.write_text(_format_json(record) + "\n", encoding="utf-8")
    return output_path
