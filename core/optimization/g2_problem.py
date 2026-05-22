from __future__ import annotations

import numpy as np
from core import config
from core.optimization.continuity_metrics import (
    curvature_derivative_value_and_cp_grad,
    curvature_value_and_cp_grad,
    finite_diff_jacobian,
    start_derivative_weights,
)
from core.optimization.control_point_mapping import (
    OptimizationLayout,
    build_bounds,
    fourth_difference_smoothing_weights,
)
from core.optimization.fit_metrics import vertical_distance_and_grad
from core.optimization.te_handle_quality import te_handle_quality_penalty_and_grad


def build_g2_problem(
    *,
    upper_data: np.ndarray,
    lower_data: np.ndarray,
    basis_upper: np.ndarray,
    basis_lower: np.ndarray,
    upper_knot_vector: np.ndarray,
    lower_knot_vector: np.ndarray,
    degree_upper: int,
    degree_lower: int,
    te_point_upper: np.ndarray,
    te_point_lower: np.ndarray,
    upper_te_dir: np.ndarray | None,
    lower_te_dir: np.ndarray | None,
    enable_soft_te_handle_quality: bool,
    smoothing_weight: float,
    initial_vars: np.ndarray,
    layout: OptimizationLayout,
    vars_to_control_points_fn,
    enforce_g3: bool,
    fit_error_metric: str = "msr",
) -> dict:
    num_cp_upper = int(layout.num_cp_upper)
    num_cp_lower = int(layout.num_cp_lower)
    num_vars = int(layout.num_vars)

    smooth_w_upper = fourth_difference_smoothing_weights(num_cp_upper)
    smooth_w_lower = fourth_difference_smoothing_weights(num_cp_lower)

    metric = str(fit_error_metric).strip()
    if metric not in {"msr", "vertical"}:
        metric = "msr"

    vertical_eval_state = {
        "last_x": None,
        "last_error_upper": 0.0,
        "last_error_lower": 0.0,
        "last_grad_upper": None,
        "last_grad_lower": None,
        "last_u_upper": None,
        "last_u_lower": None,
    }
    eval_cache: dict[str, np.ndarray | None] = {
        "x": None,
        "cp_upper": None,
        "cp_lower": None,
    }
    fit_eval_state: dict[str, np.ndarray | float | None] = {
        "x": None,
        "error_upper": None,
        "error_lower": None,
        "grad_upper": None,
        "grad_lower": None,
    }

    def cached_control_points(vars: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(vars, dtype=float)
        cached_x = eval_cache["x"]
        if cached_x is None or not np.array_equal(cached_x, x):
            cp_u, cp_l = vars_to_control_points_fn(x)
            eval_cache["x"] = x.copy()
            eval_cache["cp_upper"] = cp_u
            eval_cache["cp_lower"] = cp_l
        cp_upper = eval_cache["cp_upper"]
        cp_lower = eval_cache["cp_lower"]
        if cp_upper is None or cp_lower is None:
            cp_upper, cp_lower = vars_to_control_points_fn(x)
            eval_cache["x"] = x.copy()
            eval_cache["cp_upper"] = cp_upper
            eval_cache["cp_lower"] = cp_lower
        return cp_upper, cp_lower

    def ensure_vertical_eval(cp_upper: np.ndarray, cp_lower: np.ndarray, x: np.ndarray) -> None:
        last_x = vertical_eval_state["last_x"]
        x_arr = np.asarray(x, dtype=float)
        if last_x is not None and np.array_equal(np.asarray(last_x, dtype=float), x_arr):
            return

        err_u, grad_u, solved_u_upper, _ = vertical_distance_and_grad(
            upper_data,
            cp_upper,
            upper_knot_vector,
            degree_upper,
            initial_u=np.asarray(vertical_eval_state["last_u_upper"], dtype=float)
            if vertical_eval_state["last_u_upper"] is not None
            else None,
        )
        err_l, grad_l, solved_u_lower, _ = vertical_distance_and_grad(
            lower_data,
            cp_lower,
            lower_knot_vector,
            degree_lower,
            initial_u=np.asarray(vertical_eval_state["last_u_lower"], dtype=float)
            if vertical_eval_state["last_u_lower"] is not None
            else None,
        )
        vertical_eval_state["last_x"] = x_arr.copy()
        vertical_eval_state["last_error_upper"] = float(err_u)
        vertical_eval_state["last_error_lower"] = float(err_l)
        vertical_eval_state["last_grad_upper"] = grad_u
        vertical_eval_state["last_grad_lower"] = grad_l
        vertical_eval_state["last_u_upper"] = solved_u_upper
        vertical_eval_state["last_u_lower"] = solved_u_lower

    def ensure_fit_eval(vars: np.ndarray) -> None:
        x = np.asarray(vars, dtype=float)
        cached_x = fit_eval_state["x"]
        if cached_x is not None and np.array_equal(np.asarray(cached_x, dtype=float), x):
            return

        cp_upper, cp_lower = cached_control_points(x)
        if metric == "msr":
            fitted_upper = basis_upper @ cp_upper
            fitted_lower = basis_lower @ cp_lower
            residual_upper = fitted_upper - upper_data
            residual_lower = fitted_lower - lower_data
            error_upper = float(np.sum(residual_upper * residual_upper))
            error_lower = float(np.sum(residual_lower * residual_lower))
            grad_cp_upper = 2.0 * (basis_upper.T @ residual_upper)
            grad_cp_lower = 2.0 * (basis_lower.T @ residual_lower)
        else:
            ensure_vertical_eval(cp_upper, cp_lower, x)
            error_upper = float(vertical_eval_state["last_error_upper"])
            error_lower = float(vertical_eval_state["last_error_lower"])
            grad_cp_upper = np.asarray(vertical_eval_state["last_grad_upper"], dtype=float).copy()
            grad_cp_lower = np.asarray(vertical_eval_state["last_grad_lower"], dtype=float).copy()

        fit_eval_state["x"] = x.copy()
        fit_eval_state["error_upper"] = error_upper
        fit_eval_state["error_lower"] = error_lower
        fit_eval_state["grad_upper"] = grad_cp_upper
        fit_eval_state["grad_lower"] = grad_cp_lower

    def raw_smoothing_penalty_and_grad(cp: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
        grad = np.zeros_like(cp, dtype=float)
        if weights.size == 0 or len(cp) < 5:
            return 0.0, grad

        penalty = 0.0
        fourth_diff = np.diff(cp, n=4, axis=0)
        for i, w in enumerate(weights):
            weight = float(w)
            if weight <= 0.0:
                continue
            diff = fourth_diff[i]
            penalty += weight * float(np.dot(diff, diff))
            grad_fourth = 2.0 * weight * diff
            grad[i] += grad_fourth
            grad[i + 1] -= 4.0 * grad_fourth
            grad[i + 2] += 6.0 * grad_fourth
            grad[i + 3] -= 4.0 * grad_fourth
            grad[i + 4] += grad_fourth

        return penalty, grad

    def raw_smoothing_breakdown(cp: np.ndarray, weights: np.ndarray) -> dict[str, float]:
        if weights.size == 0 or len(cp) < 5:
            return {"total": 0.0, "fourth_diff": 0.0}

        fourth_penalty = 0.0
        fourth_diff = np.diff(cp, n=4, axis=0)
        for i, w in enumerate(weights):
            weight = float(w)
            if weight <= 0.0:
                continue
            diff = fourth_diff[i]
            fourth_penalty += weight * float(np.dot(diff, diff))

        total = fourth_penalty
        return {
            "total": float(total),
            "fourth_diff": float(fourth_penalty),
        }

    initial_vars_arr = np.asarray(initial_vars, dtype=float)
    initial_cp_upper, initial_cp_lower = cached_control_points(initial_vars_arr)
    ensure_fit_eval(initial_vars_arr)
    baseline_fit_error_total = float(fit_eval_state["error_upper"]) + float(fit_eval_state["error_lower"])
    raw_smooth_upper_initial = raw_smoothing_breakdown(initial_cp_upper, smooth_w_upper)
    raw_smooth_lower_initial = raw_smoothing_breakdown(initial_cp_lower, smooth_w_lower)
    raw_smoothing_baseline_total = float(raw_smooth_upper_initial["total"] + raw_smooth_lower_initial["total"])
    smoothing_reference_fit = max(baseline_fit_error_total, 1e-6)
    smoothing_objective_scale = 0.0
    if raw_smoothing_baseline_total > 1e-16 and float(smoothing_weight) > 0.0:
        smoothing_objective_scale = smoothing_reference_fit / raw_smoothing_baseline_total

    te_handle_weight = float(getattr(config, "DEFAULT_TE_HANDLE_QUALITY_WEIGHT", 0.0))
    te_handle_enabled = (
        bool(enable_soft_te_handle_quality)
        and bool(getattr(config, "ENABLE_SOFT_TE_HANDLE_QUALITY", False))
        and te_handle_weight > 0.0
    )
    te_handle_min_length = float(getattr(config, "TE_HANDLE_MIN_LENGTH", 0.040))
    te_handle_short_length_weight = float(getattr(config, "TE_HANDLE_SHORT_LENGTH_WEIGHT", 0.25))

    def raw_te_handle_penalty_and_grad(
        cp: np.ndarray,
        target_dir: np.ndarray | None,
    ) -> tuple[float, np.ndarray, dict[str, float]]:
        if not te_handle_enabled or target_dir is None:
            return 0.0, np.zeros_like(cp, dtype=float), {
                "angle": 0.0,
                "short_length": 0.0,
                "total": 0.0,
                "length": 0.0,
            }
        return te_handle_quality_penalty_and_grad(
            cp,
            target_dir,
            min_length=te_handle_min_length,
            short_length_weight=te_handle_short_length_weight,
        )

    raw_te_upper_initial, _, raw_te_upper_initial_parts = raw_te_handle_penalty_and_grad(
        initial_cp_upper,
        upper_te_dir,
    )
    raw_te_lower_initial, _, raw_te_lower_initial_parts = raw_te_handle_penalty_and_grad(
        initial_cp_lower,
        lower_te_dir,
    )
    raw_te_handle_baseline_total = float(raw_te_upper_initial + raw_te_lower_initial)
    te_handle_objective_scale = 0.0
    if raw_te_handle_baseline_total > 1e-16 and te_handle_enabled:
        te_handle_objective_scale = smoothing_reference_fit / raw_te_handle_baseline_total

    def te_handle_penalty_and_grad(
        cp: np.ndarray,
        target_dir: np.ndarray | None,
    ) -> tuple[float, np.ndarray]:
        raw_penalty, raw_grad, _ = raw_te_handle_penalty_and_grad(cp, target_dir)
        scale = te_handle_weight * te_handle_objective_scale
        if scale == 0.0:
            return 0.0, np.zeros_like(raw_grad, dtype=float)
        return scale * raw_penalty, scale * raw_grad

    def te_handle_breakdown(cp: np.ndarray, target_dir: np.ndarray | None) -> dict[str, float]:
        raw_penalty, _, parts = raw_te_handle_penalty_and_grad(cp, target_dir)
        scale = te_handle_weight * te_handle_objective_scale
        return {
            "total": float(scale * raw_penalty),
            "angle": float(scale * parts["angle"]),
            "short_length": float(scale * parts["short_length"]),
            "length": float(parts["length"]),
        }

    def smoothing_penalty_and_grad(cp: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
        raw_penalty, raw_grad = raw_smoothing_penalty_and_grad(cp, weights)
        scale = float(smoothing_weight) * float(smoothing_objective_scale)
        if scale == 0.0:
            return 0.0, np.zeros_like(raw_grad, dtype=float)
        return scale * raw_penalty, scale * raw_grad

    def smoothing_breakdown(cp: np.ndarray, weights: np.ndarray) -> dict[str, float]:
        raw = raw_smoothing_breakdown(cp, weights)
        scale = float(smoothing_weight) * float(smoothing_objective_scale)
        return {
            "total": float(scale * raw["total"]),
            "fourth_diff": float(scale * raw["fourth_diff"]),
        }

    def append_monotonic_x_constraints(is_upper: bool, num_cp: int) -> None:
        for i in range(1, num_cp - 1):
            idx_current = layout.var_index(is_upper, i, 0)
            idx_next = layout.var_index(is_upper, i + 1, 0)
            if idx_current is None and idx_next is None:
                continue

            def monotonic_fun(vars, is_upper_local=is_upper, i_local=i):
                cp_upper, cp_lower = cached_control_points(vars)
                cp = cp_upper if is_upper_local else cp_lower
                return float(cp[i_local + 1, 0] - cp[i_local, 0])

            def monotonic_jac(vars, idx_current_local=idx_current, idx_next_local=idx_next):
                _ = vars
                jac = np.zeros(num_vars, dtype=float)
                if idx_next_local is not None:
                    jac[idx_next_local] += 1.0
                if idx_current_local is not None:
                    jac[idx_current_local] -= 1.0
                return jac

            constraints.append({"type": "ineq", "fun": monotonic_fun, "jac": monotonic_jac})

    def objective(vars):
        cp_upper, cp_lower = cached_control_points(vars)
        ensure_fit_eval(np.asarray(vars, dtype=float))
        error_upper = float(fit_eval_state["error_upper"])
        error_lower = float(fit_eval_state["error_lower"])
        penalty_upper, _ = smoothing_penalty_and_grad(cp_upper, smooth_w_upper)
        penalty_lower, _ = smoothing_penalty_and_grad(cp_lower, smooth_w_lower)
        te_penalty_upper, _ = te_handle_penalty_and_grad(cp_upper, upper_te_dir)
        te_penalty_lower, _ = te_handle_penalty_and_grad(cp_lower, lower_te_dir)
        return error_upper + error_lower + penalty_upper + penalty_lower + te_penalty_upper + te_penalty_lower

    def objective_jac(vars):
        cp_upper, cp_lower = cached_control_points(vars)
        ensure_fit_eval(np.asarray(vars, dtype=float))
        grad_cp_upper = np.asarray(fit_eval_state["grad_upper"], dtype=float).copy()
        grad_cp_lower = np.asarray(fit_eval_state["grad_lower"], dtype=float).copy()

        _, smooth_grad_upper = smoothing_penalty_and_grad(cp_upper, smooth_w_upper)
        _, smooth_grad_lower = smoothing_penalty_and_grad(cp_lower, smooth_w_lower)
        grad_cp_upper += smooth_grad_upper
        grad_cp_lower += smooth_grad_lower

        _, te_grad_upper = te_handle_penalty_and_grad(cp_upper, upper_te_dir)
        _, te_grad_lower = te_handle_penalty_and_grad(cp_lower, lower_te_dir)
        grad_cp_upper += te_grad_upper
        grad_cp_lower += te_grad_lower
        return layout.gradients_to_vars(grad_cp_upper, grad_cp_lower)

    def evaluate_diagnostics(vars: np.ndarray) -> dict[str, float]:
        x = np.asarray(vars, dtype=float)
        cp_upper, cp_lower = cached_control_points(x)
        ensure_fit_eval(x)
        smooth_upper = smoothing_breakdown(cp_upper, smooth_w_upper)
        smooth_lower = smoothing_breakdown(cp_lower, smooth_w_lower)
        te_handle_upper = te_handle_breakdown(cp_upper, upper_te_dir)
        te_handle_lower = te_handle_breakdown(cp_lower, lower_te_dir)
        fit_upper = float(fit_eval_state["error_upper"])
        fit_lower = float(fit_eval_state["error_lower"])
        return {
            "fit_error_upper": fit_upper,
            "fit_error_lower": fit_lower,
            "fit_error_total": fit_upper + fit_lower,
            "smoothing_penalty_upper": float(smooth_upper["total"]),
            "smoothing_penalty_lower": float(smooth_lower["total"]),
            "smoothing_penalty_total": float(smooth_upper["total"] + smooth_lower["total"]),
            "smoothing_penalty_upper_fourth_diff": float(smooth_upper["fourth_diff"]),
            "smoothing_penalty_lower_fourth_diff": float(smooth_lower["fourth_diff"]),
            "raw_smoothing_baseline_total": float(raw_smoothing_baseline_total),
            "smoothing_reference_fit": float(smoothing_reference_fit),
            "smoothing_objective_scale": float(smoothing_objective_scale),
            "te_handle_penalty_upper": float(te_handle_upper["total"]),
            "te_handle_penalty_lower": float(te_handle_lower["total"]),
            "te_handle_penalty_total": float(te_handle_upper["total"] + te_handle_lower["total"]),
            "te_handle_penalty_upper_angle": float(te_handle_upper["angle"]),
            "te_handle_penalty_lower_angle": float(te_handle_lower["angle"]),
            "te_handle_penalty_upper_short_length": float(te_handle_upper["short_length"]),
            "te_handle_penalty_lower_short_length": float(te_handle_lower["short_length"]),
            "te_handle_upper_length": float(te_handle_upper["length"]),
            "te_handle_lower_length": float(te_handle_lower["length"]),
            "raw_te_handle_baseline_total": float(raw_te_handle_baseline_total),
            "te_handle_objective_scale": float(te_handle_objective_scale),
            "te_handle_weight": float(te_handle_weight),
            "te_handle_min_length": float(te_handle_min_length),
        }

    weights_upper_2 = start_derivative_weights(num_cp_upper, upper_knot_vector, degree_upper, max_order=2)
    weights_lower_2 = start_derivative_weights(num_cp_lower, lower_knot_vector, degree_lower, max_order=2)
    weights_upper_3 = start_derivative_weights(num_cp_upper, upper_knot_vector, degree_upper, max_order=3)
    weights_lower_3 = start_derivative_weights(num_cp_lower, lower_knot_vector, degree_lower, max_order=3)

    def curvature_constraint(vars):
        cp_upper, cp_lower = cached_control_points(vars)
        kappa_upper, _ = curvature_value_and_cp_grad(cp_upper, upper_knot_vector, degree_upper, weights_upper_2)
        kappa_lower, _ = curvature_value_and_cp_grad(cp_lower, lower_knot_vector, degree_lower, weights_lower_2)
        return kappa_upper - kappa_lower

    def curvature_constraint_jac(vars):
        cp_upper, cp_lower = cached_control_points(vars)
        _, grad_upper = curvature_value_and_cp_grad(cp_upper, upper_knot_vector, degree_upper, weights_upper_2)
        _, grad_lower = curvature_value_and_cp_grad(cp_lower, lower_knot_vector, degree_lower, weights_lower_2)
        if weights_upper_2 is None or weights_lower_2 is None:
            return finite_diff_jacobian(curvature_constraint, vars)
        return layout.gradients_to_vars(grad_upper, -grad_lower)

    def curvature_derivative_constraint(vars):
        cp_upper, cp_lower = cached_control_points(vars)
        dk_upper, _ = curvature_derivative_value_and_cp_grad(cp_upper, upper_knot_vector, degree_upper, weights_upper_3)
        dk_lower, _ = curvature_derivative_value_and_cp_grad(cp_lower, lower_knot_vector, degree_lower, weights_lower_3)
        return dk_upper - dk_lower

    def curvature_derivative_constraint_jac(vars):
        cp_upper, cp_lower = cached_control_points(vars)
        _, grad_upper = curvature_derivative_value_and_cp_grad(cp_upper, upper_knot_vector, degree_upper, weights_upper_3)
        _, grad_lower = curvature_derivative_value_and_cp_grad(cp_lower, lower_knot_vector, degree_lower, weights_lower_3)
        if weights_upper_3 is None or weights_lower_3 is None:
            return finite_diff_jacobian(curvature_derivative_constraint, vars)
        return layout.gradients_to_vars(grad_upper, -grad_lower)

    constraints = [{"type": "eq", "fun": curvature_constraint, "jac": curvature_constraint_jac}]
    if enforce_g3:
        constraints.append(
            {"type": "eq", "fun": curvature_derivative_constraint, "jac": curvature_derivative_constraint_jac}
        )

    def te_constraint_upper(vars):
        cp_upper, _ = cached_control_points(vars)
        return cp_upper[-1] - te_point_upper

    def te_constraint_lower(vars):
        _, cp_lower = cached_control_points(vars)
        return cp_lower[-1] - te_point_lower

    def te_constraint_upper_jac(vars):
        _ = vars
        jac = np.zeros((2, num_vars), dtype=float)
        ix = layout.var_index(True, num_cp_upper - 1, 0)
        iy = layout.var_index(True, num_cp_upper - 1, 1)
        if ix is not None:
            jac[0, ix] = 1.0
        if iy is not None:
            jac[1, iy] = 1.0
        return jac

    def te_constraint_lower_jac(vars):
        _ = vars
        jac = np.zeros((2, num_vars), dtype=float)
        ix = layout.var_index(False, num_cp_lower - 1, 0)
        iy = layout.var_index(False, num_cp_lower - 1, 1)
        if ix is not None:
            jac[0, ix] = 1.0
        if iy is not None:
            jac[1, iy] = 1.0
        return jac

    constraints.extend(
        [
            {"type": "eq", "fun": te_constraint_upper, "jac": te_constraint_upper_jac},
            {"type": "eq", "fun": te_constraint_lower, "jac": te_constraint_lower_jac},
        ]
    )

    if metric == "vertical":
        append_monotonic_x_constraints(True, num_cp_upper)
        append_monotonic_x_constraints(False, num_cp_lower)

    bounds = build_bounds(num_cp_upper - 3, num_cp_lower - 3)
    return {
        "initial_vars": np.asarray(initial_vars, dtype=float),
        "objective": objective,
        "objective_jac": objective_jac,
        "constraints": constraints,
        "bounds": bounds,
        "fit_error_metric": metric,
        "fit_error_samples": -1,
        "fit_error_samples_coarse": -1,
        "fit_error_samples_medium": -1,
        "fit_error_refresh_every": -1,
        "fit_error_force_full_precision": False,
        "evaluate_diagnostics": evaluate_diagnostics,
    }
