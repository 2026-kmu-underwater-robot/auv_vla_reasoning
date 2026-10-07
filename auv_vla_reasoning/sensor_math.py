"""Sensor conversions matching the physical KMU26 collector state layout.

Derived from auv_vla_data_collector's ``contract.py``. These helpers do not
import ROS, the recorder, or the model runtime.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

STATE_SLICES = {
    "prev_command": (0, 4),
    "dvl_velocity": (4, 7),
    "angular_velocity": (7, 10),
    "linear_acceleration": (10, 13),
    "attitude": (13, 17),
    "depth": (17, 18),
    "altitude": (18, 19),
    "validity": (19, 23),
}


def _vector(values: Sequence[float], size: int, name: str) -> np.ndarray:
    value = np.asarray(values, dtype=np.float32)
    if value.shape != (size,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain {size} finite values")
    return value


def normalize_quaternion_wxyz(values: Sequence[float]) -> np.ndarray:
    """Return a unit quaternion in the collector's w, x, y, z order."""
    quaternion = _vector(values, 4, "attitude")
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1.0e-6:
        raise ValueError("Quaternion must have a finite, non-zero norm")
    return quaternion / norm


def body_velocity(
    values: Sequence[float], frame_id: str, input_frame: str, convention: str
) -> np.ndarray:
    """Rotate DVL axes to FLU [m/s], preserving the acoustic-origin reference."""
    if frame_id != input_frame:
        raise ValueError(f"Expected DVL frame {input_frame!r}, got {frame_id!r}")
    value = _vector(values, 3, "DVL velocity")
    if convention == "FRD":
        return value * np.asarray([1, -1, -1], dtype=np.float32)
    if convention == "FLU":
        return value
    raise ValueError("dvl_convention must be FRD or FLU")


def sample_is_fresh(source_time: float, received_time: float, now: float, max_age: float) -> bool:
    """Require capture and receipt ages in [0, max_age] on the ROS clock [s]."""
    return bool(
        np.isfinite(max_age)
        and max_age > 0
        and all(
            np.isfinite(stamp) and 0 <= now - stamp <= max_age
            for stamp in (source_time, received_time)
        )
    )


def build_state(
    previous_command: Sequence[float],
    dvl_velocity: Sequence[float],
    angular_velocity: Sequence[float],
    linear_acceleration: Sequence[float],
    attitude_wxyz: Sequence[float],
    depth_m: float,
    altitude_m: float,
    validity: Sequence[float],
) -> np.ndarray:
    """Build the 23-value collector state, with depth/altitude in metres.

    Velocities are body FLU [m/s], angular velocity [rad/s], acceleration
    [m/s²], and the quaternion is normalized wxyz. Previous actions are the
    normalized [surge, sway, heave, yaw] commands actually sent by the adapter.
    """
    return np.concatenate(
        [
            _vector(previous_command, 4, "previous command"),
            _vector(dvl_velocity, 3, "DVL velocity"),
            _vector(angular_velocity, 3, "angular velocity"),
            _vector(linear_acceleration, 3, "linear acceleration"),
            normalize_quaternion_wxyz(attitude_wxyz),
            _vector([depth_m, altitude_m], 2, "depth/altitude"),
            _vector(validity, 4, "validity"),
        ]
    )
