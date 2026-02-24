from __future__ import annotations

import numpy as np
from scipy import optimize

from core import config
from core.optimization.control_point_mapping import smoothing_weights
from utils import bspline_helper


def _resolve_pure_fit_error_metric() -> str:
    metric = str(getattr(config, "FIT_ERROR_OBJECTIVE", "euclidean")).strip().lower()
    if metric == "msr":
        return "msr"
    return "euclidean"


def _nearest_sampled_distance_and_grad(
    data_points: np.ndarray,
    sampled_curve: np.ndarray,
    sampled_basis: np.ndarray,
) -> tuple[float, np.ndarray]:
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


def _pack_control_points(cp: np.ndarray) -> np.ndarray:
    num_cp = int(cp.shape[0])
    out = np.zeros(2 * num_cp, dtype=float)
    out[:num_cp] = cp[:, 0]
    out[num_cp:] = cp[:, 1]
    return out


def _unpack_control_points(vars_flat: np.ndarray, num_cp: int) -> np.ndarray:
    x = np.asarray(vars_flat[:num_cp], dtype=float)
    y = np.asarray(vars_flat[num_cp:], dtype=float)
    return np.column_stack((x, y))


def _build_linear_g1_guess(
    proc,
    basis_matrix: np.ndarray,
    surface_data: np.ndarray,
    num_control_points: int,
    te_tangent_vector: np.ndarray | None,
    te_point: np.ndarray | None,
) -> np.ndarray:
    """Legacy constrained least-squares initializer."""
    A_data = np.zeros((2 * len(surface_data), 2 * num_control_points))
    b_data = np.zeros(2 * len(surface_data))

    A_data[: len(surface_data), :num_control_points] = basis_matrix
    b_data[: len(surface_data)] = surface_data[:, 0]

    A_data[len(surface_data) :, num_control_points:] = basis_matrix
    b_data[len(surface_data) :] = surface_data[:, 1]

    constraints = []
    constraint_rhs = []

    row = np.zeros(2 * num_control_points)
    row[0] = 1.0
    constraints.append(row)
    constraint_rhs.append(0.0)

    row = np.zeros(2 * num_control_points)
    row[num_control_points] = 1.0
    constraints.append(row)
    constraint_rhs.append(0.0)

    row = np.zeros(2 * num_control_points)
    row[1] = 1.0
    constraints.append(row)
    constraint_rhs.append(0.0)

    if te_tangent_vector is not None:
        row = np.zeros(2 * num_control_points)
        row[num_control_points - 1] = -te_tangent_vector[1]
        row[2 * num_control_points - 1] = te_tangent_vector[0]
        row[num_control_points - 2] = te_tangent_vector[1]
        row[2 * num_control_points - 2] = -te_tangent_vector[0]
        constraints.append(row)
        constraint_rhs.append(0.0)

    if te_point is not None:
        row_x = np.zeros(2 * num_control_points)
        row_x[num_control_points - 1] = 1.0
        constraints.append(row_x)
        constraint_rhs.append(te_point[0])

        row_y = np.zeros(2 * num_control_points)
        row_y[2 * num_control_points - 1] = 1.0
        constraints.append(row_y)
        constraint_rhs.append(te_point[1])

    constraint_weight = 1000.0
    A_constraints = np.array(constraints) * constraint_weight
    b_constraints = np.array(constraint_rhs) * constraint_weight

    A_smoothing = np.zeros(((num_control_points - 2) * 2, 2 * num_control_points))
    b_smoothing = np.zeros((num_control_points - 2) * 2)

    for i in range(num_control_points - 2):
        gradient = 0.5 + 1.5 * (i / (num_control_points - 3)) if num_control_points > 3 else 1.0
        current_weight = proc.smoothing_weight * gradient

        A_smoothing[i, i] = current_weight
        A_smoothing[i, i + 1] = -2 * current_weight
        A_smoothing[i, i + 2] = current_weight

        A_smoothing[i + (num_control_points - 2), num_control_points + i] = current_weight
        A_smoothing[i + (num_control_points - 2), num_control_points + i + 1] = -2 * current_weight
        A_smoothing[i + (num_control_points - 2), num_control_points + i + 2] = current_weight

    A_all = np.vstack([A_data, A_constraints, A_smoothing])
    b_all = np.hstack([b_data, b_constraints, b_smoothing])
    return np.linalg.lstsq(A_all, b_all, rcond=None)[0]


def _build_linear_constraints(
    num_control_points: int,
    te_tangent_vector: np.ndarray | None,
    te_point: np.ndarray | None,
) -> list[dict]:
    constraints: list[dict] = []

    def add_linear_eq(row: np.ndarray, rhs: float) -> None:
        row_local = np.asarray(row, dtype=float).copy()
        rhs_local = float(rhs)
        constraints.append(
            {
                "type": "eq",
                "fun": lambda x, r=row_local, b=rhs_local: float(np.dot(r, x) - b),
                "jac": lambda x, r=row_local: r,
            }
        )

    row = np.zeros(2 * num_control_points, dtype=float)
    row[0] = 1.0
    add_linear_eq(row, 0.0)

    row = np.zeros(2 * num_control_points, dtype=float)
    row[num_control_points] = 1.0
    add_linear_eq(row, 0.0)

    row = np.zeros(2 * num_control_points, dtype=float)
    row[1] = 1.0
    add_linear_eq(row, 0.0)

    if te_tangent_vector is not None:
        row = np.zeros(2 * num_control_points, dtype=float)
        row[num_control_points - 1] = -te_tangent_vector[1]
        row[2 * num_control_points - 1] = te_tangent_vector[0]
        row[num_control_points - 2] = te_tangent_vector[1]
        row[2 * num_control_points - 2] = -te_tangent_vector[0]
        add_linear_eq(row, 0.0)

    if te_point is not None:
        row = np.zeros(2 * num_control_points, dtype=float)
        row[num_control_points - 1] = 1.0
        add_linear_eq(row, te_point[0])

        row = np.zeros(2 * num_control_points, dtype=float)
        row[2 * num_control_points - 1] = 1.0
        add_linear_eq(row, te_point[1])

    return constraints


def fit_g1_independent(
    proc,
    upper_data: np.ndarray,
    lower_data: np.ndarray,
    num_control_points: int | tuple[int, int],
    upper_te_dir: np.ndarray | None,
    lower_te_dir: np.ndarray | None,
    enforce_te_tangency: bool = True,
    use_existing_knot_vectors: bool = False,
) -> None:
    """Fit surfaces independently with G1 constraint only."""
    _ = num_control_points
    te_point_upper = upper_data[-1]
    te_point_lower = lower_data[-1]

    u_params_upper = bspline_helper.create_parameter_from_x_coords(upper_data, proc.param_exponent_upper)
    u_params_lower = bspline_helper.create_parameter_from_x_coords(lower_data, proc.param_exponent_lower)

    if not use_existing_knot_vectors:
        proc.upper_knot_vector = bspline_helper.create_knot_vector(proc.num_cp_upper, proc.degree_upper)
        proc.lower_knot_vector = bspline_helper.create_knot_vector(proc.num_cp_lower, proc.degree_lower)

    if proc.upper_knot_vector is None or proc.lower_knot_vector is None:
        raise ValueError("Knot vectors are unexpectedly None when building basis matrices in G1-independent fit.")

    basis_upper = bspline_helper.build_basis_matrix(u_params_upper, proc.upper_knot_vector, proc.degree_upper)
    basis_lower = bspline_helper.build_basis_matrix(u_params_lower, proc.lower_knot_vector, proc.degree_lower)

    num_control_points_upper = len(proc.upper_knot_vector) - proc.degree_upper - 1
    num_control_points_lower = len(proc.lower_knot_vector) - proc.degree_lower - 1

    proc.upper_control_points = fit_single_surface_g1(
        proc,
        basis_upper,
        upper_data,
        num_control_points_upper,
        is_upper=True,
        te_tangent_vector=upper_te_dir if enforce_te_tangency else None,
        te_point=te_point_upper,
    )
    proc.lower_control_points = fit_single_surface_g1(
        proc,
        basis_lower,
        lower_data,
        num_control_points_lower,
        is_upper=False,
        te_tangent_vector=lower_te_dir if enforce_te_tangency else None,
        te_point=te_point_lower,
    )
    num_samples = int(max(128, getattr(config, "NUM_POINTS_CURVE_OPTIMIZATION_EUCLIDEAN", 1500)))
    pure_metric = _resolve_pure_fit_error_metric()
    proc.last_optimizer_info = {
        "success": True,
        "accepted": True,
        "accepted_via_relaxed_criteria": False,
        "status": 0,
        "message": (
            "G1 independent fit solved with MSR objective."
            if pure_metric == "msr"
            else "G1 independent fit solved with Euclidean objective."
        ),
        "iterations": -1,
        "objective": float("nan"),
        "max_constraint_violation": 0.0,
        "solver_ftol": float("nan"),
        "solver_maxiter": -1,
        "insertion_mode": bool(use_existing_knot_vectors),
        "fit_error_metric": pure_metric,
        "fit_error_samples": int(num_samples),
        "mode": "g1_independent",
    }


def fit_single_surface_g1(
    proc,
    basis_matrix: np.ndarray,
    surface_data: np.ndarray,
    num_control_points: int,
    is_upper: bool,
    te_tangent_vector: np.ndarray | None = None,
    te_point: np.ndarray | None = None,
) -> np.ndarray:
    """Fit single surface with G1/TE constraints and Euclidean distance objective."""
    knot_vector = proc.upper_knot_vector if is_upper else proc.lower_knot_vector
    degree = proc.degree_upper if is_upper else proc.degree_lower
    if knot_vector is None:
        raise ValueError("Knot vector is unexpectedly None in G1 fitting.")

    linear_guess = _build_linear_g1_guess(
        proc,
        basis_matrix,
        surface_data,
        num_control_points,
        te_tangent_vector,
        te_point,
    )
    x0 = np.asarray(linear_guess, dtype=float)

    num_samples = int(max(128, getattr(config, "NUM_POINTS_CURVE_OPTIMIZATION_EUCLIDEAN", 1500)))
    sample_u = np.linspace(
        float(knot_vector[degree]),
        float(knot_vector[-(degree + 1)]),
        num_samples,
    )
    sampled_basis = bspline_helper.build_basis_matrix(sample_u, knot_vector, degree)
    smooth_w = smoothing_weights(num_control_points) * (float(proc.smoothing_weight) ** 2)

    def objective_euclidean(vars_flat: np.ndarray) -> float:
        cp = _unpack_control_points(vars_flat, num_control_points)
        sampled_curve = sampled_basis @ cp
        error, _ = _nearest_sampled_distance_and_grad(surface_data, sampled_curve, sampled_basis)

        if smooth_w.size:
            diff = np.diff(cp, n=2, axis=0)
            error += float(np.sum((diff ** 2) * smooth_w[:, np.newaxis]))
        return error

    def objective_euclidean_jac(vars_flat: np.ndarray) -> np.ndarray:
        cp = _unpack_control_points(vars_flat, num_control_points)
        sampled_curve = sampled_basis @ cp
        _, grad_cp = _nearest_sampled_distance_and_grad(surface_data, sampled_curve, sampled_basis)

        if smooth_w.size:
            diff = np.diff(cp, n=2, axis=0)
            for i, w in enumerate(smooth_w):
                scale = 2.0 * float(w)
                grad_cp[i] += scale * diff[i]
                grad_cp[i + 1] += -2.0 * scale * diff[i]
                grad_cp[i + 2] += scale * diff[i]

        return _pack_control_points(grad_cp)

    def objective_msr(vars_flat: np.ndarray) -> float:
        cp = _unpack_control_points(vars_flat, num_control_points)
        fitted = basis_matrix @ cp
        residual = fitted - surface_data
        msr = float(np.sum(residual * residual))
        if smooth_w.size:
            diff = np.diff(cp, n=2, axis=0)
            msr += float(np.sum((diff ** 2) * smooth_w[:, np.newaxis]))
        return msr

    def objective_msr_jac(vars_flat: np.ndarray) -> np.ndarray:
        cp = _unpack_control_points(vars_flat, num_control_points)
        fitted = basis_matrix @ cp
        residual = fitted - surface_data
        grad_cp = 2.0 * (basis_matrix.T @ residual)
        if smooth_w.size:
            diff = np.diff(cp, n=2, axis=0)
            for i, w in enumerate(smooth_w):
                scale = 2.0 * float(w)
                grad_cp[i] += scale * diff[i]
                grad_cp[i + 1] += -2.0 * scale * diff[i]
                grad_cp[i + 2] += scale * diff[i]
        return _pack_control_points(grad_cp)

    constraints = _build_linear_constraints(num_control_points, te_tangent_vector, te_point)
    bounds: list[tuple[float | None, float | None]] = [(None, None)] * (2 * num_control_points)
    y1_idx = num_control_points + 1
    if is_upper:
        bounds[y1_idx] = (0.0, None)
    else:
        bounds[y1_idx] = (None, 0.0)

    max_iter = max(300, 20 * num_control_points)
    pure_metric = _resolve_pure_fit_error_metric()
    if pure_metric == "msr":
        objective_fn = objective_msr
        objective_jac_fn = objective_msr_jac
    else:
        objective_fn = objective_euclidean
        objective_jac_fn = objective_euclidean_jac
    result = optimize.minimize(
        objective_fn,
        x0,
        method="SLSQP",
        jac=objective_jac_fn,
        constraints=constraints,
        bounds=bounds,
        options={"ftol": 1e-8, "maxiter": max_iter, "disp": False},
    )

    final_vars = result.x if bool(result.success or result.status == 0) else x0
    control_points = _unpack_control_points(final_vars, num_control_points)
    control_points[0] = [0.0, 0.0]
    control_points[1, 0] = 0.0

    if is_upper and control_points[1, 1] < 0:
        control_points[1, 1] = abs(control_points[1, 1])
    elif not is_upper and control_points[1, 1] > 0:
        control_points[1, 1] = -abs(control_points[1, 1])

    return control_points

