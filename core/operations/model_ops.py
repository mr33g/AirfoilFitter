from __future__ import annotations

import numpy as np
from scipy import interpolate

from core import config


def finalize_curves(proc) -> None:
    """Final cleanup and curve rebuilding."""
    if proc.upper_control_points is not None and proc.lower_control_points is not None:
        shared_p0 = (proc.upper_control_points[0] + proc.lower_control_points[0]) / 2
        proc.upper_control_points[0] = shared_p0
        proc.lower_control_points[0] = shared_p0

        proc.upper_control_points[0] = [0.0, 0.0]
        proc.lower_control_points[0] = [0.0, 0.0]
        proc.upper_control_points[1, 0] = 0.0
        proc.lower_control_points[1, 0] = 0.0

        if proc.is_sharp_te:
            te_point = np.array([1.0, 0.0])
            proc.upper_control_points[-1] = te_point
            proc.lower_control_points[-1] = te_point

    if proc.upper_control_points is not None and proc.upper_knot_vector is not None:
        proc.upper_curve = interpolate.BSpline(proc.upper_knot_vector, proc.upper_control_points, proc.degree_upper)

    if proc.lower_control_points is not None and proc.lower_knot_vector is not None:
        proc.lower_curve = interpolate.BSpline(proc.lower_knot_vector, proc.lower_control_points, proc.degree_lower)


def validate_trailing_edge_tangents(proc, upper_te_dir: np.ndarray | None, lower_te_dir: np.ndarray | None) -> None:
    _ = proc, upper_te_dir, lower_te_dir
    pass


def validate_continuity(proc) -> None:
    if not config.DEBUG_WORKER_LOGGING:
        return
    if not proc.fitted or proc.upper_curve is None or proc.lower_curve is None:
        return
    print("[DEBUG] Control points:")
    for i in range(len(proc.upper_control_points)):
        print(f"[DEBUG]   Upper P{i}: ({proc.upper_control_points[i,0]:.6f}, {proc.upper_control_points[i,1]:.6f})")
    for i in range(len(proc.lower_control_points)):
        print(f"[DEBUG]   Lower P{i}: ({proc.lower_control_points[i,0]:.6f}, {proc.lower_control_points[i,1]:.6f})")

