from __future__ import annotations

import numpy as np

from core import config
from core.optimization import pure_surface_fit_error
from utils import bspline_helper


def apply_knot_insertions(proc, new_knots: list[float], surface: str | None = None) -> bool:
    if not new_knots:
        return True

    sorted_new_knots = sorted(new_knots)
    new_upper_cps = proc.upper_control_points
    new_upper_knots = proc.upper_knot_vector
    new_lower_cps = proc.lower_control_points
    new_lower_knots = proc.lower_knot_vector
    new_num_cp_upper = int(proc.num_cp_upper)
    new_num_cp_lower = int(proc.num_cp_lower)
    insertion_records: list[dict[str, float | bool | str]] = []

    if surface is None or surface == "upper":
        if proc.upper_control_points is None or proc.upper_knot_vector is None:
            proc.last_error_message = "Knot insertion failed: upper curve state is missing."
            return False
        new_upper_cps = proc.upper_control_points.copy()
        new_upper_knots = proc.upper_knot_vector.copy()
        for knot in sorted_new_knots:
            insert_result = insert_knot_with_spacing_fallback(
                proc, new_upper_cps, new_upper_knots, proc.degree_upper, float(knot), "upper"
            )
            if insert_result is None:
                return False
            new_upper_cps, new_upper_knots, actual_knot, used_fallback = insert_result
            insertion_records.append(
                {
                    "surface": "upper",
                    "requested_knot": float(knot),
                    "actual_knot": float(actual_knot),
                    "used_fallback": bool(used_fallback),
                }
            )
        new_num_cp_upper = len(new_upper_cps)

    if surface is None or surface == "lower":
        if proc.lower_control_points is None or proc.lower_knot_vector is None:
            proc.last_error_message = "Knot insertion failed: lower curve state is missing."
            return False
        new_lower_cps = proc.lower_control_points.copy()
        new_lower_knots = proc.lower_knot_vector.copy()
        for knot in sorted_new_knots:
            insert_result = insert_knot_with_spacing_fallback(
                proc, new_lower_cps, new_lower_knots, proc.degree_lower, float(knot), "lower"
            )
            if insert_result is None:
                return False
            new_lower_cps, new_lower_knots, actual_knot, used_fallback = insert_result
            insertion_records.append(
                {
                    "surface": "lower",
                    "requested_knot": float(knot),
                    "actual_knot": float(actual_knot),
                    "used_fallback": bool(used_fallback),
                }
            )
        new_num_cp_lower = len(new_lower_cps)

    proc.upper_control_points = new_upper_cps
    proc.upper_knot_vector = new_upper_knots
    proc.lower_control_points = new_lower_cps
    proc.lower_knot_vector = new_lower_knots
    proc.num_cp_upper = new_num_cp_upper
    proc.num_cp_lower = new_num_cp_lower
    proc.last_insertion_info = {"records": insertion_records}
    return True


def insert_knot_with_spacing_fallback(
    proc,
    control_points: np.ndarray,
    knot_vector: np.ndarray,
    degree: int,
    requested_knot: float,
    surface_name: str,
) -> tuple[np.ndarray, np.ndarray, float, bool] | None:
    trial_cps, trial_knots = bspline_helper.insert_knot(control_points, knot_vector, degree, requested_knot)
    if not cp_spacing_violates_minimum(proc, trial_cps):
        return trial_cps, trial_knots, float(requested_knot), False

    fallback_knot = largest_span_midpoint_knot(proc, knot_vector, degree, exclude=requested_knot)
    if fallback_knot is None:
        proc.last_error_message = (
            f"Knot insertion rejected: {surface_name} control points became too tightly clustered."
        )
        return None

    fallback_cps, fallback_knots = bspline_helper.insert_knot(control_points, knot_vector, degree, fallback_knot)
    if cp_spacing_violates_minimum(proc, fallback_cps):
        proc.last_error_message = (
            f"Knot insertion rejected: {surface_name} control points became too tightly clustered."
        )
        return None
    return fallback_cps, fallback_knots, float(fallback_knot), True


def largest_span_midpoint_knot(
    proc,
    knot_vector: np.ndarray,
    degree: int,
    *,
    exclude: float | None = None,
) -> float | None:
    _ = proc
    kv = np.asarray(knot_vector, dtype=float)
    if kv.size < 2:
        return None

    start = max(int(degree), 0)
    stop = max(start + 1, int(len(kv) - degree - 1))
    spans: list[tuple[float, float]] = []
    for i in range(start, stop):
        left = float(kv[i])
        right = float(kv[i + 1])
        width = right - left
        if width <= 1e-12:
            continue
        mid = 0.5 * (left + right)
        spans.append((width, mid))

    if not spans:
        return None

    spans.sort(key=lambda s: s[0], reverse=True)
    for _, mid in spans:
        if exclude is not None and abs(mid - float(exclude)) <= 1e-12:
            continue
        return float(mid)
    return None


def cp_spacing_violates_minimum(proc, control_points: np.ndarray) -> bool:
    min_dist = float(getattr(proc, "min_cp_neighbor_distance", 0.0))
    if min_dist <= 0.0:
        return False
    if control_points is None or len(control_points) < 2:
        return False
    seg = np.diff(np.asarray(control_points, dtype=float), axis=0)
    if len(seg) == 0:
        return False
    d = np.linalg.norm(seg, axis=1)
    return bool(np.min(d) < min_dist)


def refit_after_knot_insertion(proc) -> bool:
    if proc.upper_original_data is None or proc.lower_original_data is None:
        return True

    metric = str(getattr(config, "FIT_ERROR_OBJECTIVE", "msr")).strip()
    if metric not in {"msr", "vertical"}:
        metric = "msr"
    baseline_upper_cp = None if proc.upper_control_points is None else np.asarray(proc.upper_control_points, dtype=float).copy()
    baseline_lower_cp = None if proc.lower_control_points is None else np.asarray(proc.lower_control_points, dtype=float).copy()
    baseline_upper_knots = None if proc.upper_knot_vector is None else np.asarray(proc.upper_knot_vector, dtype=float).copy()
    baseline_lower_knots = None if proc.lower_knot_vector is None else np.asarray(proc.lower_knot_vector, dtype=float).copy()

    def pure_fit_total() -> float:
        if (
            proc.upper_control_points is None
            or proc.lower_control_points is None
            or proc.upper_knot_vector is None
            or proc.lower_knot_vector is None
        ):
            return float("inf")
        upper_error = pure_surface_fit_error(
            metric,
            proc.upper_original_data,
            proc.upper_control_points,
            proc.upper_knot_vector,
            proc.degree_upper,
            exponent_guess=float(getattr(proc, "param_exponent_upper", 0.5)),
        )
        lower_error = pure_surface_fit_error(
            metric,
            proc.lower_original_data,
            proc.lower_control_points,
            proc.lower_knot_vector,
            proc.degree_lower,
            exponent_guess=float(getattr(proc, "param_exponent_lower", 0.5)),
        )
        return float(upper_error + lower_error)

    baseline_error = pure_fit_total()

    cp_counts = (proc.num_cp_upper, proc.num_cp_lower)
    if proc.enforce_g2:
        success = proc._fit_with_g2_optimization(
            proc.upper_original_data,
            proc.lower_original_data,
            cp_counts,
            upper_te_dir=None,
            lower_te_dir=None,
            enforce_te_tangency=False,
            use_existing_knot_vectors=True,
            warm_start_from_current=True,
        )
        if not success:
            proc._fit_g1_independent(
                proc.upper_original_data,
                proc.lower_original_data,
                cp_counts,
                upper_te_dir=None,
                lower_te_dir=None,
                enforce_te_tangency=False,
                use_existing_knot_vectors=True,
            )
    else:
        proc._fit_g1_independent(
            proc.upper_original_data,
            proc.lower_original_data,
            cp_counts,
                upper_te_dir=None,
                lower_te_dir=None,
                enforce_te_tangency=False,
                use_existing_knot_vectors=True,
            )
    final_error = pure_fit_total()
    tolerance = max(1e-12, abs(baseline_error) * 1e-9)
    if np.isfinite(baseline_error) and np.isfinite(final_error) and final_error > baseline_error + tolerance:
        proc.upper_control_points = baseline_upper_cp
        proc.lower_control_points = baseline_lower_cp
        proc.upper_knot_vector = baseline_upper_knots
        proc.lower_knot_vector = baseline_lower_knots
        proc.last_error_message = (
            f"Refit after knot insertion worsened {metric} fit error "
            f"({baseline_error:.6e} -> {final_error:.6e}); reverted to shape-preserving inserted curve."
        )
        if isinstance(proc.last_optimizer_info, dict):
            proc.last_optimizer_info["insertion_refit_reverted"] = True
            proc.last_optimizer_info["insertion_refit_baseline_error"] = float(baseline_error)
            proc.last_optimizer_info["insertion_refit_final_error"] = float(final_error)
            proc.last_optimizer_info["insertion_refit_metric"] = metric
    return True


def refine_curve_with_knots(proc, new_knots: list[float], surface: str | None = None) -> bool:
    if not proc.fitted or proc.upper_curve is None or proc.lower_curve is None:
        return False
    if not new_knots:
        return True
    try:
        if not apply_knot_insertions(proc, new_knots, surface=surface):
            return False
        if not refit_after_knot_insertion(proc):
            return False
        proc._finalize_curves()
        proc._validate_continuity()
        return True
    except Exception as exc:
        proc.last_error_message = f"refine_curve_with_knots failed: {exc}"
        return False


def refine_curves_with_surface_knots(
    proc,
    *,
    upper_knots: list[float] | None = None,
    lower_knots: list[float] | None = None,
) -> bool:
    if not proc.fitted or proc.upper_curve is None or proc.lower_curve is None:
        return False

    upper_knots = upper_knots or []
    lower_knots = lower_knots or []
    if not upper_knots and not lower_knots:
        return True

    try:
        if upper_knots and not apply_knot_insertions(proc, upper_knots, surface="upper"):
            return False
        if lower_knots and not apply_knot_insertions(proc, lower_knots, surface="lower"):
            return False
        if not refit_after_knot_insertion(proc):
            return False
        proc._finalize_curves()
        proc._validate_continuity()
        return True
    except Exception as exc:
        proc.last_error_message = f"refine_curves_with_surface_knots failed: {exc}"
        return False

