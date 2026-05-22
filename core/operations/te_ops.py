from __future__ import annotations

import numpy as np
from scipy import interpolate

from core import config
from utils import bspline_helper


def apply_te_thickening(proc, te_thickness: float) -> bool:
    """
    Apply trailing edge thickening as a post-processing step.
    """
    if not proc.fitted or proc.upper_control_points is None or proc.lower_control_points is None:
        return False

    if te_thickness < 0.0:
        return False

    try:
        proc._backup_upper_control_points = proc.upper_control_points.copy()
        proc._backup_lower_control_points = proc.lower_control_points.copy()
        proc._backup_upper_knot_vector = None if proc.upper_knot_vector is None else proc.upper_knot_vector.copy()
        proc._backup_lower_knot_vector = None if proc.lower_knot_vector is None else proc.lower_knot_vector.copy()

        if proc.upper_curve is None or proc.lower_curve is None:
            return False

        num_samples = max(200, int(config.PLOT_POINTS_PER_SURFACE))

        _, upper_pts = bspline_helper.sample_curve(proc.upper_curve, num_samples)
        _, lower_pts = bspline_helper.sample_curve(proc.lower_curve, num_samples)

        upper_x = np.clip(upper_pts[:, 0], 0.0, 1.0)
        lower_x = np.clip(lower_pts[:, 0], 0.0, 1.0)
        f_upper = bspline_helper.smoothstep_quintic(upper_x)
        f_lower = bspline_helper.smoothstep_quintic(lower_x)

        half_thickness = 0.5 * te_thickness

        thick_upper = upper_pts.copy()
        thick_lower = lower_pts.copy()
        thick_upper[:, 1] = thick_upper[:, 1] + half_thickness * f_upper
        thick_lower[:, 1] = thick_lower[:, 1] - half_thickness * f_lower

        proc._fit_g1_independent(
            thick_upper,
            thick_lower,
            (proc.num_cp_upper, proc.num_cp_lower),
            upper_te_dir=None,
            lower_te_dir=None,
            enable_soft_te_handle_quality=False,
        )

        proc.is_sharp_te = False
        proc._finalize_curves()
        proc.fitted = True
        return True

    except Exception:
        return False


def remove_te_thickening(proc) -> bool:
    """
    Remove trailing edge thickening by restoring from backup.
    """
    if not proc.fitted or proc.upper_control_points is None or proc.lower_control_points is None:
        return False

    try:
        if proc._backup_upper_control_points is not None and proc._backup_lower_control_points is not None:
            proc.upper_control_points = proc._backup_upper_control_points
            proc.lower_control_points = proc._backup_lower_control_points
            if proc._backup_upper_knot_vector is not None:
                proc.upper_knot_vector = proc._backup_upper_knot_vector
            if proc._backup_lower_knot_vector is not None:
                proc.lower_knot_vector = proc._backup_lower_knot_vector

            if proc.upper_knot_vector is not None:
                proc.upper_curve = interpolate.BSpline(
                    proc.upper_knot_vector, proc.upper_control_points, proc.degree
                )
            if proc.lower_knot_vector is not None:
                proc.lower_curve = interpolate.BSpline(
                    proc.lower_knot_vector, proc.lower_control_points, proc.degree
                )

            proc._backup_upper_control_points = None
            proc._backup_lower_control_points = None
            proc._backup_upper_knot_vector = None
            proc._backup_lower_knot_vector = None

            proc.is_sharp_te = bool(np.allclose(proc.upper_control_points[-1], proc.lower_control_points[-1], atol=1e-12))
            return True

        return False

    except Exception:
        return False
