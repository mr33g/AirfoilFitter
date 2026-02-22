from __future__ import annotations

import numpy as np

from utils import bspline_helper


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


def fit_single_surface_g1(
    proc,
    basis_matrix: np.ndarray,
    surface_data: np.ndarray,
    num_control_points: int,
    is_upper: bool,
    te_tangent_vector: np.ndarray | None = None,
    te_point: np.ndarray | None = None,
) -> np.ndarray:
    """Fit single surface with G1 LE constraint and optional TE constraints."""
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

    solution = np.linalg.lstsq(A_all, b_all, rcond=None)[0]

    control_points = np.zeros((num_control_points, 2))
    control_points[:, 0] = solution[:num_control_points]
    control_points[:, 1] = solution[num_control_points:]

    control_points[0] = [0.0, 0.0]
    control_points[1, 0] = 0.0

    if is_upper and control_points[1, 1] < 0:
        control_points[1, 1] = abs(control_points[1, 1])
    elif not is_upper and control_points[1, 1] > 0:
        control_points[1, 1] = -abs(control_points[1, 1])

    return control_points

