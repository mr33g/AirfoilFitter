from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from core import config
from core.optimization.continuity_metrics import (
    curvature_derivative_value_and_cp_grad,
    curvature_value_and_cp_grad,
    finite_diff_jacobian,
    start_derivative_weights,
)
from core.optimization.control_point_mapping import OptimizationLayout, build_bounds, smoothing_weights
from utils import bspline_helper


def _nearest_sampled_distance_and_grad(
    data_points: np.ndarray,
    sampled_curve: np.ndarray,
    sampled_basis: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Approximate point-to-curve error using nearest sampled points."""
    if data_points.size == 0 or sampled_curve.size == 0:
        return 0.0, np.zeros((sampled_basis.shape[1], 2), dtype=float)

    diff = data_points[:, np.newaxis, :] - sampled_curve[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    nearest_idx = np.argmin(dist_sq, axis=1)
    nearest_curve_points = sampled_curve[nearest_idx]
    residual = nearest_curve_points - data_points

    error = float(np.sum(np.einsum("ij,ij->i", residual, residual)))
    nearest_basis = sampled_basis[nearest_idx]
    grad_cp = 2.0 * (nearest_basis.T @ residual)
    return error, grad_cp


def _distance_and_grad_from_indices(
    data_points: np.ndarray,
    sampled_curve: np.ndarray,
    sampled_basis: np.ndarray,
    nearest_idx: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Compute distance objective using preselected nearest sampled indices."""
    if data_points.size == 0 or sampled_curve.size == 0:
        return 0.0, np.zeros((sampled_basis.shape[1], 2), dtype=float)

    idx = np.asarray(nearest_idx, dtype=int)
    idx = np.clip(idx, 0, max(0, sampled_curve.shape[0] - 1))
    nearest_curve_points = sampled_curve[idx]
    residual = nearest_curve_points - data_points
    error = float(np.sum(np.einsum("ij,ij->i", residual, residual)))
    nearest_basis = sampled_basis[idx]
    grad_cp = 2.0 * (nearest_basis.T @ residual)
    return error, grad_cp


def _nearest_indices_kdtree(
    data_points: np.ndarray,
    sampled_curve: np.ndarray,
) -> np.ndarray:
    """Nearest sampled-point indices via KD-tree (used on refresh steps)."""
    if data_points.size == 0 or sampled_curve.size == 0:
        return np.zeros((len(data_points),), dtype=int)
    tree = cKDTree(sampled_curve)
    _, idx = tree.query(data_points, k=1)
    return np.asarray(idx, dtype=int)


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
    fit_error_metric: str = "euclidean",
    num_fit_samples: int | None = None,
    euclidean_force_full_precision: bool = False,
) -> dict:
    num_cp_upper = int(layout.num_cp_upper)
    num_cp_lower = int(layout.num_cp_lower)
    num_vars = int(layout.num_vars)

    smooth_w_upper = smoothing_weights(num_cp_upper)
    smooth_w_lower = smoothing_weights(num_cp_lower)
    metric = str(fit_error_metric).strip().lower()
    if metric not in {"euclidean", "msr"}:
        metric = "euclidean"

    effective_fit_samples = int(
        max(
            128,
            num_fit_samples
            if num_fit_samples is not None
            else getattr(config, "NUM_POINTS_CURVE_OPTIMIZATION_EUCLIDEAN", 1500),
        )
    )
    sampled_basis_upper = None
    sampled_basis_lower = None
    euclidean_levels: list[dict[str, np.ndarray]] = []
    euclidean_level_upper_idx: list[np.ndarray | None] = []
    euclidean_level_lower_idx: list[np.ndarray | None] = []
    euclidean_eval_state = {
        "new_x_count": 0,
        "last_x": None,
        "last_level": -1,
        "last_error_upper": 0.0,
        "last_error_lower": 0.0,
        "last_grad_upper": None,
        "last_grad_lower": None,
    }
    if metric == "euclidean":
        if euclidean_force_full_precision:
            sample_levels = [effective_fit_samples]
        else:
            coarse_default = max(256, effective_fit_samples // 8)
            medium_default = max(512, effective_fit_samples // 3)
            coarse_samples = int(
                max(128, min(effective_fit_samples, getattr(config, "EUCLIDEAN_COARSE_SAMPLES", coarse_default)))
            )
            medium_samples = int(
                max(coarse_samples, min(effective_fit_samples, getattr(config, "EUCLIDEAN_MEDIUM_SAMPLES", medium_default)))
            )
            sample_levels = [coarse_samples, medium_samples, effective_fit_samples]
        sample_levels_unique: list[int] = []
        for level in sample_levels:
            if level not in sample_levels_unique:
                sample_levels_unique.append(level)

        for level_samples in sample_levels_unique:
            sample_u_upper = np.linspace(
                float(upper_knot_vector[degree_upper]),
                float(upper_knot_vector[-(degree_upper + 1)]),
                int(level_samples),
            )
            sample_u_lower = np.linspace(
                float(lower_knot_vector[degree_lower]),
                float(lower_knot_vector[-(degree_lower + 1)]),
                int(level_samples),
            )
            euclidean_levels.append(
                {
                    "basis_upper": bspline_helper.build_basis_matrix(sample_u_upper, upper_knot_vector, degree_upper),
                    "basis_lower": bspline_helper.build_basis_matrix(sample_u_lower, lower_knot_vector, degree_lower),
                }
            )
            euclidean_level_upper_idx.append(None)
            euclidean_level_lower_idx.append(None)

        sampled_basis_upper = euclidean_levels[-1]["basis_upper"]
        sampled_basis_lower = euclidean_levels[-1]["basis_lower"]

        if euclidean_force_full_precision:
            stage1_evals = 0
            stage2_evals = 0
            refresh_every = 1
        else:
            stage1_evals = int(max(0, getattr(config, "EUCLIDEAN_COARSE_EVALS", 20)))
            stage2_evals = int(max(stage1_evals, getattr(config, "EUCLIDEAN_MEDIUM_EVALS", 70)))
            refresh_every = int(max(1, getattr(config, "EUCLIDEAN_NEAREST_REFRESH_EVERY", 4)))

        def select_level(eval_count: int) -> int:
            if len(euclidean_levels) == 1:
                return 0
            if len(euclidean_levels) == 2:
                return 0 if eval_count < stage1_evals else 1
            if eval_count < stage1_evals:
                return 0
            if eval_count < stage2_evals:
                return 1
            return len(euclidean_levels) - 1

        def ensure_euclidean_eval(cp_upper: np.ndarray, cp_lower: np.ndarray, x: np.ndarray) -> None:
            last_x = euclidean_eval_state["last_x"]
            if last_x is not None and np.array_equal(last_x, x):
                return

            euclidean_eval_state["new_x_count"] = int(euclidean_eval_state["new_x_count"]) + 1
            eval_count = int(euclidean_eval_state["new_x_count"])
            level_idx = select_level(eval_count)
            level_changed = level_idx != int(euclidean_eval_state["last_level"])
            level_data = euclidean_levels[level_idx]
            level_basis_upper = level_data["basis_upper"]
            level_basis_lower = level_data["basis_lower"]
            sampled_upper = level_basis_upper @ cp_upper
            sampled_lower = level_basis_lower @ cp_lower

            upper_idx = euclidean_level_upper_idx[level_idx]
            lower_idx = euclidean_level_lower_idx[level_idx]
            need_refresh = (
                level_changed
                or upper_idx is None
                or lower_idx is None
                or (eval_count % refresh_every == 0)
            )
            if need_refresh:
                upper_idx = _nearest_indices_kdtree(upper_data, sampled_upper)
                lower_idx = _nearest_indices_kdtree(lower_data, sampled_lower)
                euclidean_level_upper_idx[level_idx] = upper_idx
                euclidean_level_lower_idx[level_idx] = lower_idx

            err_u, grad_u = _distance_and_grad_from_indices(upper_data, sampled_upper, level_basis_upper, upper_idx)
            err_l, grad_l = _distance_and_grad_from_indices(lower_data, sampled_lower, level_basis_lower, lower_idx)
            euclidean_eval_state["last_x"] = np.asarray(x, dtype=float).copy()
            euclidean_eval_state["last_level"] = int(level_idx)
            euclidean_eval_state["last_error_upper"] = float(err_u)
            euclidean_eval_state["last_error_lower"] = float(err_l)
            euclidean_eval_state["last_grad_upper"] = grad_u
            euclidean_eval_state["last_grad_lower"] = grad_l
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

        if metric == "msr":
            fitted_upper = basis_upper @ cp_upper
            fitted_lower = basis_lower @ cp_lower
            residual_upper = fitted_upper - upper_data
            residual_lower = fitted_lower - lower_data
            error_upper = float(np.sum(residual_upper * residual_upper))
            error_lower = float(np.sum(residual_lower * residual_lower))
        else:
            if sampled_basis_upper is None or sampled_basis_lower is None:
                raise ValueError("Sampled basis matrices are unavailable for Euclidean objective.")
            ensure_euclidean_eval(cp_upper, cp_lower, np.asarray(vars, dtype=float))
            error_upper = float(euclidean_eval_state["last_error_upper"])
            error_lower = float(euclidean_eval_state["last_error_lower"])

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

        if metric == "msr":
            fitted_upper = basis_upper @ cp_upper
            fitted_lower = basis_lower @ cp_lower
            residual_upper = fitted_upper - upper_data
            residual_lower = fitted_lower - lower_data
            grad_cp_upper = 2.0 * (basis_upper.T @ residual_upper)
            grad_cp_lower = 2.0 * (basis_lower.T @ residual_lower)
        else:
            if sampled_basis_upper is None or sampled_basis_lower is None:
                raise ValueError("Sampled basis matrices are unavailable for Euclidean objective.")
            ensure_euclidean_eval(cp_upper, cp_lower, np.asarray(vars, dtype=float))
            grad_cp_upper = np.asarray(euclidean_eval_state["last_grad_upper"], dtype=float)
            grad_cp_lower = np.asarray(euclidean_eval_state["last_grad_lower"], dtype=float)

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
        "fit_error_metric": metric,
        "fit_error_samples": int(effective_fit_samples) if metric == "euclidean" else -1,
        "fit_error_samples_coarse": int(euclidean_levels[0]["basis_upper"].shape[0]) if metric == "euclidean" else -1,
        "fit_error_samples_medium": int(euclidean_levels[min(1, len(euclidean_levels) - 1)]["basis_upper"].shape[0]) if metric == "euclidean" else -1,
        "fit_error_refresh_every": int(max(1, getattr(config, "EUCLIDEAN_NEAREST_REFRESH_EVERY", 4)))
        if metric == "euclidean"
        else -1,
        "fit_error_force_full_precision": bool(euclidean_force_full_precision) if metric == "euclidean" else False,
    }

