from __future__ import annotations

import numpy as np
from scipy import optimize

from core import config
from core.optimization.fit_metrics import vertical_distance_and_grad
from core.optimization.te_handle_quality import te_handle_quality_penalty_and_grad
from utils import bspline_helper


def _resolve_pure_fit_error_metric() -> str:
    metric = str(getattr(config, "FIT_ERROR_OBJECTIVE", "msr")).strip().lower()
    if metric in {"msr", "vertical"}:
        return metric
    return "msr"


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


def _fourth_difference_weights(num_control_points: int) -> np.ndarray:
    if num_control_points <= 4:
        return np.zeros(0, dtype=float)
    idx = np.arange(num_control_points - 4, dtype=float)
    grad = (
        0.5 + 1.5 * (idx / (num_control_points - 5))
        if num_control_points > 5
        else np.ones(num_control_points - 4, dtype=float)
    )
    return grad * grad


def _fourth_difference_penalty_and_grad(
    cp: np.ndarray,
    weights: np.ndarray,
    smoothing_weight: float,
) -> tuple[float, np.ndarray]:
    grad = np.zeros_like(cp, dtype=float)
    if weights.size == 0 or len(cp) < 5 or smoothing_weight == 0.0:
        return 0.0, grad

    penalty = 0.0
    diff = np.diff(cp, n=4, axis=0)
    for i, w in enumerate(weights):
        scale = float(smoothing_weight) * float(w)
        if scale <= 0.0:
            continue
        d4 = diff[i]
        penalty += scale * float(np.dot(d4, d4))
        grad_d4 = 2.0 * scale * d4
        grad[i] += grad_d4
        grad[i + 1] -= 4.0 * grad_d4
        grad[i + 2] += 6.0 * grad_d4
        grad[i + 3] -= 4.0 * grad_d4
        grad[i + 4] += grad_d4
    return penalty, grad


def _scaled_te_handle_penalty_and_grad(
    cp: np.ndarray,
    target_direction: np.ndarray | None,
    scale: float,
) -> tuple[float, np.ndarray, dict[str, float]]:
    raw_penalty, raw_grad, parts = te_handle_quality_penalty_and_grad(
        cp,
        target_direction,
        min_length=float(getattr(config, "TE_HANDLE_MIN_LENGTH", 0.040)),
        short_length_weight=float(getattr(config, "TE_HANDLE_SHORT_LENGTH_WEIGHT", 0.25)),
    )
    if scale == 0.0:
        return 0.0, np.zeros_like(raw_grad, dtype=float), parts
    return float(scale * raw_penalty), scale * raw_grad, {
        key: float(scale * value) if key != "length" else float(value)
        for key, value in parts.items()
    }


def _build_linear_g1_guess(
    proc,
    basis_matrix: np.ndarray,
    surface_data: np.ndarray,
    num_control_points: int,
    te_point: np.ndarray | None,
) -> np.ndarray:
    """Constrained least-squares initializer for endpoint/G1 conditions."""
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

    num_d4 = max(0, num_control_points - 4)
    A_smoothing = np.zeros((num_d4 * 2, 2 * num_control_points))
    b_smoothing = np.zeros(num_d4 * 2)

    for i in range(num_d4):
        gradient = 0.5 + 1.5 * (i / (num_d4 - 1)) if num_d4 > 1 else 1.0
        current_weight = proc.smoothing_weight * gradient

        A_smoothing[i, i] = current_weight
        A_smoothing[i, i + 1] = -4 * current_weight
        A_smoothing[i, i + 2] = 6 * current_weight
        A_smoothing[i, i + 3] = -4 * current_weight
        A_smoothing[i, i + 4] = current_weight

        y_row = i + num_d4
        A_smoothing[y_row, num_control_points + i] = current_weight
        A_smoothing[y_row, num_control_points + i + 1] = -4 * current_weight
        A_smoothing[y_row, num_control_points + i + 2] = 6 * current_weight
        A_smoothing[y_row, num_control_points + i + 3] = -4 * current_weight
        A_smoothing[y_row, num_control_points + i + 4] = current_weight

    A_all = np.vstack([A_data, A_constraints, A_smoothing])
    b_all = np.hstack([b_data, b_constraints, b_smoothing])
    return np.linalg.lstsq(A_all, b_all, rcond=None)[0]


def _build_linear_constraints(
    num_control_points: int,
    te_point: np.ndarray | None,
    enforce_monotonic_x: bool = False,
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

    def add_linear_ineq(row: np.ndarray, rhs: float) -> None:
        row_local = np.asarray(row, dtype=float).copy()
        rhs_local = float(rhs)
        constraints.append(
            {
                "type": "ineq",
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

    if te_point is not None:
        row = np.zeros(2 * num_control_points, dtype=float)
        row[num_control_points - 1] = 1.0
        add_linear_eq(row, te_point[0])

        row = np.zeros(2 * num_control_points, dtype=float)
        row[2 * num_control_points - 1] = 1.0
        add_linear_eq(row, te_point[1])

    if enforce_monotonic_x:
        for i in range(1, num_control_points - 1):
            row = np.zeros(2 * num_control_points, dtype=float)
            row[i + 1] = 1.0
            row[i] = -1.0
            add_linear_ineq(row, 0.0)

    return constraints


def fit_g1_independent(
    proc,
    upper_data: np.ndarray,
    lower_data: np.ndarray,
    num_control_points: int | tuple[int, int],
    upper_te_dir: np.ndarray | None,
    lower_te_dir: np.ndarray | None,
    enable_soft_te_handle_quality: bool = True,
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
        soft_te_tangent_vector=upper_te_dir if enable_soft_te_handle_quality else None,
        te_point=te_point_upper,
    )
    proc.lower_control_points = fit_single_surface_g1(
        proc,
        basis_lower,
        lower_data,
        num_control_points_lower,
        is_upper=False,
        soft_te_tangent_vector=lower_te_dir if enable_soft_te_handle_quality else None,
        te_point=te_point_lower,
    )
    pure_metric = _resolve_pure_fit_error_metric()
    proc.last_optimizer_info = {
        "success": True,
        "accepted": True,
        "status": 0,
        "message": (
            "G1 independent fit solved with MSR objective."
            if pure_metric == "msr"
            else "G1 independent fit solved with vertical objective."
        ),
        "iterations": -1,
        "objective": float("nan"),
        "max_constraint_violation": 0.0,
        "solver_ftol": float("nan"),
        "solver_maxiter": -1,
        "insertion_mode": bool(use_existing_knot_vectors),
        "fit_error_metric": pure_metric,
        "fit_error_samples": -1,
        "te_handle_weight": float(getattr(config, "DEFAULT_TE_HANDLE_QUALITY_WEIGHT", np.nan)),
        "te_handle_min_length": float(getattr(config, "TE_HANDLE_MIN_LENGTH", np.nan)),
        "mode": "g1_independent",
    }


def fit_single_surface_g1(
    proc,
    basis_matrix: np.ndarray,
    surface_data: np.ndarray,
    num_control_points: int,
    is_upper: bool,
    soft_te_tangent_vector: np.ndarray | None = None,
    te_point: np.ndarray | None = None,
) -> np.ndarray:
    """Fit single surface with G1/endpoint constraints and the selected fit objective."""
    knot_vector = proc.upper_knot_vector if is_upper else proc.lower_knot_vector
    degree = proc.degree_upper if is_upper else proc.degree_lower
    if knot_vector is None:
        raise ValueError("Knot vector is unexpectedly None in G1 fitting.")

    linear_guess = _build_linear_g1_guess(
        proc,
        basis_matrix,
        surface_data,
        num_control_points,
        te_point,
    )
    x0 = np.asarray(linear_guess, dtype=float)

    smooth_w = _fourth_difference_weights(num_control_points)
    smoothing_weight = float(proc.smoothing_weight)
    te_handle_weight = float(getattr(config, "DEFAULT_TE_HANDLE_QUALITY_WEIGHT", 0.0))
    te_handle_enabled = bool(getattr(config, "ENABLE_SOFT_TE_HANDLE_QUALITY", False)) and te_handle_weight > 0.0
    vertical_eval_state: dict[str, np.ndarray | float | None] = {
        "x": None,
        "error": None,
        "grad": None,
        "u": None,
    }

    def ensure_vertical_eval(vars_flat: np.ndarray, cp: np.ndarray) -> None:
        cached_x = vertical_eval_state["x"]
        x = np.asarray(vars_flat, dtype=float)
        if cached_x is not None and np.array_equal(np.asarray(cached_x, dtype=float), x):
            return
        error, grad_cp, solved_u, _ = vertical_distance_and_grad(
            surface_data,
            cp,
            knot_vector,
            degree,
            initial_u=np.asarray(vertical_eval_state["u"], dtype=float) if vertical_eval_state["u"] is not None else None,
            exponent_guess=proc.param_exponent_upper if is_upper else proc.param_exponent_lower,
        )
        vertical_eval_state["x"] = x.copy()
        vertical_eval_state["error"] = float(error)
        vertical_eval_state["grad"] = grad_cp
        vertical_eval_state["u"] = solved_u

    def fit_error_for_scaling(cp: np.ndarray, vars_flat: np.ndarray) -> float:
        if _resolve_pure_fit_error_metric() == "vertical":
            ensure_vertical_eval(vars_flat, cp)
            return float(vertical_eval_state["error"])
        fitted = basis_matrix @ cp
        residual = fitted - surface_data
        return float(np.sum(residual * residual))

    cp0 = _unpack_control_points(x0, num_control_points)
    raw_te0, _, _ = te_handle_quality_penalty_and_grad(
        cp0,
        soft_te_tangent_vector if te_handle_enabled else None,
        min_length=float(getattr(config, "TE_HANDLE_MIN_LENGTH", 0.040)),
        short_length_weight=float(getattr(config, "TE_HANDLE_SHORT_LENGTH_WEIGHT", 0.25)),
    )
    fit_ref = max(fit_error_for_scaling(cp0, x0), 1e-6)
    te_handle_scale = te_handle_weight * fit_ref / raw_te0 if raw_te0 > 1e-16 and te_handle_enabled else 0.0

    def objective_vertical(vars_flat: np.ndarray) -> float:
        cp = _unpack_control_points(vars_flat, num_control_points)
        ensure_vertical_eval(vars_flat, cp)
        error = float(vertical_eval_state["error"])
        penalty, _ = _fourth_difference_penalty_and_grad(cp, smooth_w, smoothing_weight)
        error += penalty
        te_penalty, _, _ = _scaled_te_handle_penalty_and_grad(cp, soft_te_tangent_vector, te_handle_scale)
        error += te_penalty
        return error

    def objective_vertical_jac(vars_flat: np.ndarray) -> np.ndarray:
        cp = _unpack_control_points(vars_flat, num_control_points)
        ensure_vertical_eval(vars_flat, cp)
        grad_cp = np.asarray(vertical_eval_state["grad"], dtype=float).copy()

        _, smooth_grad = _fourth_difference_penalty_and_grad(cp, smooth_w, smoothing_weight)
        grad_cp += smooth_grad
        _, te_grad, _ = _scaled_te_handle_penalty_and_grad(cp, soft_te_tangent_vector, te_handle_scale)
        grad_cp += te_grad

        return _pack_control_points(grad_cp)

    def objective_msr(vars_flat: np.ndarray) -> float:
        cp = _unpack_control_points(vars_flat, num_control_points)
        fitted = basis_matrix @ cp
        residual = fitted - surface_data
        error = float(np.sum(residual * residual))
        penalty, _ = _fourth_difference_penalty_and_grad(cp, smooth_w, smoothing_weight)
        error += penalty
        te_penalty, _, _ = _scaled_te_handle_penalty_and_grad(cp, soft_te_tangent_vector, te_handle_scale)
        error += te_penalty
        return error

    def objective_msr_jac(vars_flat: np.ndarray) -> np.ndarray:
        cp = _unpack_control_points(vars_flat, num_control_points)
        fitted = basis_matrix @ cp
        residual = fitted - surface_data
        grad_cp = 2.0 * (basis_matrix.T @ residual)
        _, smooth_grad = _fourth_difference_penalty_and_grad(cp, smooth_w, smoothing_weight)
        grad_cp += smooth_grad
        _, te_grad, _ = _scaled_te_handle_penalty_and_grad(cp, soft_te_tangent_vector, te_handle_scale)
        grad_cp += te_grad
        return _pack_control_points(grad_cp)

    pure_metric = _resolve_pure_fit_error_metric()
    constraints = _build_linear_constraints(
        num_control_points,
        te_point,
        enforce_monotonic_x=(pure_metric == "vertical"),
    )
    bounds: list[tuple[float | None, float | None]] = [(None, None)] * (2 * num_control_points)
    y1_idx = num_control_points + 1
    if is_upper:
        bounds[y1_idx] = (0.0, None)
    else:
        bounds[y1_idx] = (None, 0.0)

    max_iter = max(300, 20 * num_control_points)
    if pure_metric == "vertical":
        objective_fn = objective_vertical
        objective_jac_fn = objective_vertical_jac
    else:
        objective_fn = objective_msr
        objective_jac_fn = objective_msr_jac
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

