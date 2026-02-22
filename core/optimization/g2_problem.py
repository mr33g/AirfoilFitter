from __future__ import annotations

import numpy as np

from core.optimization.continuity_metrics import (
    curvature_derivative_value_and_cp_grad,
    curvature_value_and_cp_grad,
    finite_diff_jacobian,
    start_derivative_weights,
)
from core.optimization.control_point_mapping import OptimizationLayout, build_bounds, smoothing_weights
from utils import bspline_helper


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
    enforce_te_tangency: bool,
    smoothing_weight: float,
    initial_vars: np.ndarray,
    layout: OptimizationLayout,
    vars_to_control_points_fn,
    enforce_g3: bool,
) -> dict:
    num_cp_upper = int(layout.num_cp_upper)
    num_cp_lower = int(layout.num_cp_lower)
    num_vars = int(layout.num_vars)

    smooth_w_upper = smoothing_weights(num_cp_upper)
    smooth_w_lower = smoothing_weights(num_cp_lower)

    eval_cache: dict[str, np.ndarray | None] = {
        "x": None,
        "cp_upper": None,
        "cp_lower": None,
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

    weights_upper_2 = start_derivative_weights(num_cp_upper, upper_knot_vector, degree_upper, max_order=2)
    weights_lower_2 = start_derivative_weights(num_cp_lower, lower_knot_vector, degree_lower, max_order=2)
    weights_upper_3 = start_derivative_weights(num_cp_upper, upper_knot_vector, degree_upper, max_order=3)
    weights_lower_3 = start_derivative_weights(num_cp_lower, lower_knot_vector, degree_lower, max_order=3)

    def objective(vars):
        cp_upper, cp_lower = cached_control_points(vars)

        fitted_upper = basis_upper @ cp_upper
        fitted_lower = basis_lower @ cp_lower

        error_upper = float(np.sum((upper_data - fitted_upper) ** 2))
        error_lower = float(np.sum((lower_data - fitted_lower) ** 2))

        smoothing_penalty = 0.0
        if smooth_w_upper.size:
            diff_upper = np.diff(cp_upper, n=2, axis=0)
            smoothing_penalty += float(np.sum((diff_upper ** 2) * smooth_w_upper[:, np.newaxis]))
        if smooth_w_lower.size:
            diff_lower = np.diff(cp_lower, n=2, axis=0)
            smoothing_penalty += float(np.sum((diff_lower ** 2) * smooth_w_lower[:, np.newaxis]))

        return error_upper + error_lower + smoothing_weight * smoothing_penalty

    def objective_jac(vars):
        cp_upper, cp_lower = cached_control_points(vars)

        fitted_upper = basis_upper @ cp_upper
        fitted_lower = basis_lower @ cp_lower
        grad_cp_upper = 2.0 * (basis_upper.T @ (fitted_upper - upper_data))
        grad_cp_lower = 2.0 * (basis_lower.T @ (fitted_lower - lower_data))

        if smooth_w_upper.size:
            diff_upper = np.diff(cp_upper, n=2, axis=0)
            for i, w in enumerate(smooth_w_upper):
                scale = 2.0 * smoothing_weight * float(w)
                grad_cp_upper[i] += scale * diff_upper[i]
                grad_cp_upper[i + 1] += -2.0 * scale * diff_upper[i]
                grad_cp_upper[i + 2] += scale * diff_upper[i]
        if smooth_w_lower.size:
            diff_lower = np.diff(cp_lower, n=2, axis=0)
            for i, w in enumerate(smooth_w_lower):
                scale = 2.0 * smoothing_weight * float(w)
                grad_cp_lower[i] += scale * diff_lower[i]
                grad_cp_lower[i + 1] += -2.0 * scale * diff_lower[i]
                grad_cp_lower[i + 2] += scale * diff_lower[i]

        return layout.gradients_to_vars(grad_cp_upper, grad_cp_lower)

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

    constraints = [
        {"type": "eq", "fun": curvature_constraint, "jac": curvature_constraint_jac},
    ]

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

    if upper_te_dir is not None and lower_te_dir is not None and enforce_te_tangency:
        def te_tangent_jacobian(cp: np.ndarray, knot_vector: np.ndarray, degree: int, is_upper: bool) -> np.ndarray:
            jac = np.zeros((2, num_vars), dtype=float)
            n = len(cp) - 1
            if n < 1:
                return jac
            p = int(degree)
            denom = float(knot_vector[n + 1] - knot_vector[n - p + 1])
            if abs(denom) < 1e-12:
                scale = 1.0
                vec = cp[-1] - cp[-2]
            else:
                scale = float(p / denom)
                vec = scale * (cp[-1] - cp[-2])
            norm = float(np.linalg.norm(vec))
            if norm <= 1e-12:
                return jac
            proj = np.eye(2, dtype=float) / norm - np.outer(vec, vec) / (norm ** 3)
            dt_dpn = scale * proj
            dt_dpnm1 = -scale * proj
            for coord in (0, 1):
                idx_last = layout.var_index(is_upper, n, coord)
                idx_prev = layout.var_index(is_upper, n - 1, coord)
                if idx_last is not None:
                    jac[:, idx_last] += dt_dpn[:, coord]
                if idx_prev is not None:
                    jac[:, idx_prev] += dt_dpnm1[:, coord]
            return jac

        def te_tangent_constraint_upper(vars):
            cp_upper, _ = cached_control_points(vars)
            computed_tangent = bspline_helper.compute_tangent_at_trailing_edge(cp_upper, upper_knot_vector, degree_upper)
            return computed_tangent - upper_te_dir

        def te_tangent_constraint_lower(vars):
            _, cp_lower = cached_control_points(vars)
            computed_tangent = bspline_helper.compute_tangent_at_trailing_edge(cp_lower, lower_knot_vector, degree_lower)
            return computed_tangent - lower_te_dir

        def te_tangent_constraint_upper_jac(vars):
            cp_upper, _ = cached_control_points(vars)
            return te_tangent_jacobian(cp_upper, upper_knot_vector, degree_upper, True)

        def te_tangent_constraint_lower_jac(vars):
            _, cp_lower = cached_control_points(vars)
            return te_tangent_jacobian(cp_lower, lower_knot_vector, degree_lower, False)

        constraints.extend(
            [
                {"type": "eq", "fun": te_tangent_constraint_upper, "jac": te_tangent_constraint_upper_jac},
                {"type": "eq", "fun": te_tangent_constraint_lower, "jac": te_tangent_constraint_lower_jac},
            ]
        )

    n_free_upper = num_cp_upper - 3
    n_free_lower = num_cp_lower - 3
    bounds = build_bounds(n_free_upper, n_free_lower)

    return {
        "initial_vars": np.asarray(initial_vars, dtype=float),
        "objective": objective,
        "objective_jac": objective_jac,
        "constraints": constraints,
        "bounds": bounds,
    }

