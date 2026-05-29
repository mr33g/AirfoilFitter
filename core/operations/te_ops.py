from __future__ import annotations

import numpy as np
from scipy import interpolate, optimize

from core import config
from core.optimization.continuity_metrics import (
    curvature_value_and_cp_grad,
    start_derivative_weights,
)
from utils import bspline_helper


def _clear_te_backup(proc) -> None:
    proc._backup_upper_control_points = None
    proc._backup_lower_control_points = None
    proc._backup_upper_knot_vector = None
    proc._backup_lower_knot_vector = None


def _curve_curvature(curve: interpolate.BSpline, u_values: np.ndarray) -> np.ndarray:
    d1 = curve.derivative(1)(u_values)
    d2 = curve.derivative(2)(u_values)
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    speed_sq = d1[:, 0] * d1[:, 0] + d1[:, 1] * d1[:, 1]
    denom = np.power(speed_sq, 1.5)
    return np.divide(cross, denom, out=np.zeros_like(cross), where=denom > 1.0e-18)


def _curve_sample_params(curve: interpolate.BSpline, sample_count: int) -> np.ndarray:
    t_start = float(curve.t[curve.k])
    t_end = float(curve.t[-(curve.k + 1)])
    u_values = np.linspace(t_start, t_end, max(8, int(sample_count)))
    if len(u_values) > 0:
        eps = max(1.0e-12, 1.0e-12 * abs(t_end - t_start))
        u_values[-1] = min(u_values[-1], t_end - eps)
    return u_values


def _surface_constraints(
    original_cp: np.ndarray,
    knot_vector: np.ndarray,
    degree: int,
    target_te_y: float,
    offset: int,
    total_vars: int,
) -> list[dict]:
    num_cp = int(original_cp.shape[0])
    constraints: list[dict] = []
    linear_rows: list[np.ndarray] = []

    def row_constraint(row_local: np.ndarray, rhs_local: float) -> dict:
        row_full = np.zeros(total_vars, dtype=float)
        row_full[offset:offset + num_cp] = row_local
        linear_rows.append(row_full.copy())
        return {
            "type": "eq",
            "fun": lambda x, r=row_full, b=float(rhs_local): float(np.dot(r, x) - b),
            "jac": lambda x, r=row_full: r,
        }

    row = np.zeros(num_cp, dtype=float)
    row[0] = 1.0
    constraints.append(row_constraint(row, 0.0))

    row = np.zeros(num_cp, dtype=float)
    row[-1] = 1.0
    constraints.append(row_constraint(row, float(target_te_y - original_cp[-1, 1])))

    weights1 = start_derivative_weights(num_cp, knot_vector, degree, 1)
    if weights1 is not None:
        constraints.append(row_constraint(weights1[0], 0.0))

    weights2 = start_derivative_weights(num_cp, knot_vector, degree, 2)
    kappa_original, _ = curvature_value_and_cp_grad(original_cp, knot_vector, degree, weights2)
    _, grad_original = curvature_value_and_cp_grad(original_cp, knot_vector, degree, weights2)
    curvature_row = np.zeros(total_vars, dtype=float)
    curvature_row[offset:offset + num_cp] = grad_original[:, 1]
    if linear_rows:
        linear_matrix = np.vstack(linear_rows)
        coeffs, *_ = np.linalg.lstsq(linear_matrix.T, curvature_row, rcond=None)
        curvature_row_residual = curvature_row - coeffs @ linear_matrix
    else:
        curvature_row_residual = curvature_row

    if np.linalg.norm(curvature_row_residual) <= 1.0e-10 * max(1.0, np.linalg.norm(curvature_row)):
        return constraints

    def curvature_fun(x, cp0=original_cp.copy(), kv=knot_vector.copy(), deg=int(degree),
                      weights=weights2, k0=float(kappa_original), off=int(offset), n=int(num_cp)):
        cp = cp0.copy()
        cp[:, 1] += x[off:off + n]
        kappa, _ = curvature_value_and_cp_grad(cp, kv, deg, weights)
        return float(kappa - k0)

    def curvature_jac(x, cp0=original_cp.copy(), kv=knot_vector.copy(), deg=int(degree),
                      weights=weights2, off=int(offset), n=int(num_cp), total=int(total_vars)):
        cp = cp0.copy()
        cp[:, 1] += x[off:off + n]
        _, grad_cp = curvature_value_and_cp_grad(cp, kv, deg, weights)
        jac = np.zeros(total, dtype=float)
        jac[off:off + n] = grad_cp[:, 1]
        return jac

    constraints.append({"type": "eq", "fun": curvature_fun, "jac": curvature_jac})

    return constraints


def apply_te_thickening(proc, te_thickness: float) -> bool:
    """
    Apply trailing edge thickening as a constrained displacement of the
    existing B-spline control points.
    """
    if not proc.fitted or proc.upper_control_points is None or proc.lower_control_points is None:
        return False

    if te_thickness < 0.0 or proc.upper_knot_vector is None or proc.lower_knot_vector is None:
        return False

    try:
        proc._backup_upper_control_points = proc.upper_control_points.copy()
        proc._backup_lower_control_points = proc.lower_control_points.copy()
        proc._backup_upper_knot_vector = None if proc.upper_knot_vector is None else proc.upper_knot_vector.copy()
        proc._backup_lower_knot_vector = None if proc.lower_knot_vector is None else proc.lower_knot_vector.copy()

        if proc.upper_curve is None or proc.lower_curve is None:
            _clear_te_backup(proc)
            return False

        upper_cp0 = np.asarray(proc.upper_control_points, dtype=float).copy()
        lower_cp0 = np.asarray(proc.lower_control_points, dtype=float).copy()
        upper_kv = np.asarray(proc.upper_knot_vector, dtype=float).copy()
        lower_kv = np.asarray(proc.lower_knot_vector, dtype=float).copy()
        degree_upper = int(proc.degree_upper)
        degree_lower = int(proc.degree_lower)

        sample_count = max(80, int(getattr(config, "TE_THICKENING_CURVATURE_SAMPLES", 500)))
        upper_u = _curve_sample_params(proc.upper_curve, sample_count)
        lower_u = _curve_sample_params(proc.lower_curve, sample_count)
        upper_kappa0 = _curve_curvature(proc.upper_curve, upper_u)
        lower_kappa0 = _curve_curvature(proc.lower_curve, lower_u)

        half_thickness = 0.5 * float(te_thickness)
        upper_target_te_y = half_thickness
        lower_target_te_y = -half_thickness

        upper_delta_target = upper_target_te_y - float(upper_cp0[-1, 1])
        lower_delta_target = lower_target_te_y - float(lower_cp0[-1, 1])
        upper_init = upper_delta_target * bspline_helper.smoothstep_quintic(np.clip(upper_cp0[:, 0], 0.0, 1.0))
        lower_init = lower_delta_target * bspline_helper.smoothstep_quintic(np.clip(lower_cp0[:, 0], 0.0, 1.0))
        upper_init[0] = 0.0
        lower_init[0] = 0.0
        upper_init[-1] = upper_delta_target
        lower_init[-1] = lower_delta_target
        initial_vars = np.concatenate([upper_init, lower_init])

        num_upper = len(upper_cp0)
        num_lower = len(lower_cp0)
        total_vars = num_upper + num_lower

        displacement_weight = float(getattr(config, "TE_THICKENING_OBJECTIVE_DISPLACEMENT_WEIGHT", 1.0e-5))
        smoothness_weight = float(getattr(config, "TE_THICKENING_OBJECTIVE_SMOOTHNESS_WEIGHT", 1.0e-4))

        def make_curves(vars_flat: np.ndarray) -> tuple[interpolate.BSpline, interpolate.BSpline, np.ndarray, np.ndarray]:
            upper_delta = np.asarray(vars_flat[:num_upper], dtype=float)
            lower_delta = np.asarray(vars_flat[num_upper:], dtype=float)
            upper_cp = upper_cp0.copy()
            lower_cp = lower_cp0.copy()
            upper_cp[:, 1] += upper_delta
            lower_cp[:, 1] += lower_delta
            return (
                interpolate.BSpline(upper_kv, upper_cp, degree_upper),
                interpolate.BSpline(lower_kv, lower_cp, degree_lower),
                upper_delta,
                lower_delta,
            )

        def objective(vars_flat: np.ndarray) -> float:
            upper_curve, lower_curve, upper_delta, lower_delta = make_curves(vars_flat)
            upper_dk = _curve_curvature(upper_curve, upper_u) - upper_kappa0
            lower_dk = _curve_curvature(lower_curve, lower_u) - lower_kappa0
            curvature_term = float(np.mean(upper_dk * upper_dk) + np.mean(lower_dk * lower_dk))
            displacement_term = float(np.mean(upper_delta * upper_delta) + np.mean(lower_delta * lower_delta))
            smoothness_term = 0.0
            if num_upper >= 3:
                upper_d2 = np.diff(upper_delta, n=2)
                smoothness_term += float(np.mean(upper_d2 * upper_d2))
            if num_lower >= 3:
                lower_d2 = np.diff(lower_delta, n=2)
                smoothness_term += float(np.mean(lower_d2 * lower_d2))
            return (
                curvature_term
                + displacement_weight * displacement_term
                + smoothness_weight * smoothness_term
            )

        constraints = []
        constraints.extend(
            _surface_constraints(
                upper_cp0,
                upper_kv,
                degree_upper,
                upper_target_te_y,
                0,
                total_vars,
            )
        )
        constraints.extend(
            _surface_constraints(
                lower_cp0,
                lower_kv,
                degree_lower,
                lower_target_te_y,
                num_upper,
                total_vars,
            )
        )

        result = optimize.minimize(
            objective,
            initial_vars,
            method="SLSQP",
            constraints=constraints,
            options={"ftol": 1.0e-11, "maxiter": max(250, total_vars * 40), "disp": False},
        )

        if not result.success:
            _clear_te_backup(proc)
            return False

        upper_delta = np.asarray(result.x[:num_upper], dtype=float)
        lower_delta = np.asarray(result.x[num_upper:], dtype=float)
        proc.upper_control_points = upper_cp0.copy()
        proc.lower_control_points = lower_cp0.copy()
        proc.upper_control_points[:, 1] += upper_delta
        proc.lower_control_points[:, 1] += lower_delta

        proc.is_sharp_te = False
        proc._finalize_curves()
        proc.fitted = True
        return True

    except Exception:
        _clear_te_backup(proc)
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

            _clear_te_backup(proc)

            proc.is_sharp_te = bool(np.allclose(proc.upper_control_points[-1], proc.lower_control_points[-1], atol=1e-12))
            return True

        return False

    except Exception:
        return False
