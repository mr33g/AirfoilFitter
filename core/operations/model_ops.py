from __future__ import annotations

from scipy import interpolate

from core import config
from core.debug_log import write_debug_line


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

        upper_te_y = float(proc.upper_control_points[-1, 1])
        lower_te_y = float(proc.lower_control_points[-1, 1])
        te_half_gap = 0.0 if proc.is_sharp_te else 0.5 * (upper_te_y - lower_te_y)

        proc.upper_control_points[-1, 0] = 1.0
        proc.lower_control_points[-1, 0] = 1.0
        proc.upper_control_points[-1, 1] = te_half_gap
        proc.lower_control_points[-1, 1] = -te_half_gap

    if proc.upper_control_points is not None and proc.upper_knot_vector is not None:
        proc.upper_curve = interpolate.BSpline(proc.upper_knot_vector, proc.upper_control_points, proc.degree_upper)

    if proc.lower_control_points is not None and proc.lower_knot_vector is not None:
        proc.lower_curve = interpolate.BSpline(proc.lower_knot_vector, proc.lower_control_points, proc.degree_lower)


def validate_continuity(proc) -> None:
    if not config.DEBUG_WORKER_LOGGING:
        return
    if not proc.fitted or proc.upper_curve is None or proc.lower_curve is None:
        return
    write_debug_line("[DEBUG] Control points:")
    for i in range(len(proc.upper_control_points)):
        write_debug_line(f"[DEBUG]   Upper P{i}: ({proc.upper_control_points[i,0]:.6f}, {proc.upper_control_points[i,1]:.6f})")
    for i in range(len(proc.lower_control_points)):
        write_debug_line(f"[DEBUG]   Lower P{i}: ({proc.lower_control_points[i,0]:.6f}, {proc.lower_control_points[i,1]:.6f})")

