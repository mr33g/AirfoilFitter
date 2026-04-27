from __future__ import annotations

import numpy as np
from scipy import optimize

from core import config
from core.optimization import (
    OptimizationLayout,
    build_g2_problem,
    control_points_to_initial_vars,
    vars_to_control_points,
)
from utils import bspline_helper


def _resolve_pure_fit_error_metric() -> str:
    metric = str(getattr(config, "FIT_ERROR_OBJECTIVE", "msr")).strip()
    if metric in {"msr", "vertical"}:
        return metric
    return "msr"


def fit_bspline(
    proc,
    upper_data: np.ndarray,
    lower_data: np.ndarray,
    num_control_points: int | tuple[int, int],
    is_thickened: bool = False,
    upper_te_tangent_vector: np.ndarray | None = None,
    lower_te_tangent_vector: np.ndarray | None = None,
    enforce_g2: bool = False,
    enforce_g3: bool = False,
    enforce_te_tangency: bool = True,
    preserve_existing_knots: bool = False,
) -> bool:
    """Fit B-splines with G1 and optional G2/G3 constraints at leading edge."""
    try:
        proc.last_error_message = None
        proc.last_optimizer_info = None
        if isinstance(num_control_points, tuple):
            num_cp_upper, num_cp_lower = num_control_points
        else:
            num_cp_upper = num_cp_lower = num_control_points

        proc.num_cp_upper = num_cp_upper
        proc.num_cp_lower = num_cp_lower
        proc.enforce_g2 = enforce_g2
        proc.enforce_g3 = enforce_g3 if enforce_g2 else False

        proc.degree_upper = proc.degree
        proc.degree_lower = proc.degree
        proc.last_insertion_info = None

        proc.is_sharp_te = not is_thickened

        le_point = (upper_data[0] + lower_data[0]) / 2
        upper_data_corrected = upper_data.copy()
        lower_data_corrected = lower_data.copy()
        upper_data_corrected[0] = le_point
        lower_data_corrected[0] = le_point

        proc.upper_original_data = upper_data_corrected.copy()
        proc.lower_original_data = lower_data_corrected.copy()
        proc.error_reference_available = True

        if proc.is_sharp_te:
            te_point = np.array([1.0, 0.0])
            upper_data_corrected[-1] = te_point
            lower_data_corrected[-1] = te_point

        upper_te_dir = bspline_helper.normalize_vector(upper_te_tangent_vector)
        lower_te_dir = bspline_helper.normalize_vector(lower_te_tangent_vector)

        can_reuse_existing_knots = (
            bool(preserve_existing_knots)
            and proc.upper_knot_vector is not None
            and proc.lower_knot_vector is not None
            and len(proc.upper_knot_vector) == int(proc.num_cp_upper) + int(proc.degree_upper) + 1
            and len(proc.lower_knot_vector) == int(proc.num_cp_lower) + int(proc.degree_lower) + 1
        )

        if proc.enforce_g2:
            success = fit_with_g2_optimization(
                proc,
                upper_data_corrected,
                lower_data_corrected,
                (proc.num_cp_upper, proc.num_cp_lower),
                upper_te_dir,
                lower_te_dir,
                enforce_te_tangency,
                use_existing_knot_vectors=can_reuse_existing_knots,
                warm_start_from_current=can_reuse_existing_knots,
            )
            if not success:
                proc.enforce_g2 = False
                proc.enforce_g3 = False

        if not proc.enforce_g2:
            proc._fit_g1_independent(
                upper_data_corrected,
                lower_data_corrected,
                (proc.num_cp_upper, proc.num_cp_lower),
                upper_te_dir,
                lower_te_dir,
                enforce_te_tangency,
                use_existing_knot_vectors=can_reuse_existing_knots,
            )

        proc._finalize_curves()
        proc.fitted_degree = (proc.degree_upper, proc.degree_lower)
        proc.fitted = True

        proc.num_cp_upper = len(proc.upper_control_points)
        proc.num_cp_lower = len(proc.lower_control_points)

        proc._validate_continuity()
        if upper_te_dir is not None and lower_te_dir is not None and enforce_te_tangency:
            proc._validate_trailing_edge_tangents(upper_te_dir, lower_te_dir)

        return True

    except Exception as exc:
        proc.last_error_message = f"fit_bspline failed: {exc}"
        proc.fitted = False
        return False


def fit_with_g2_optimization(
    proc,
    upper_data: np.ndarray,
    lower_data: np.ndarray,
    num_control_points: int | tuple[int, int],
    upper_te_dir: np.ndarray | None,
    lower_te_dir: np.ndarray | None,
    enforce_te_tangency: bool = True,
    use_existing_knot_vectors: bool = False,
    warm_start_from_current: bool = False,
) -> bool:
    """Fit both surfaces with G2 continuity using constrained optimization."""
    _ = num_control_points
    te_point_upper = upper_data[-1]
    te_point_lower = lower_data[-1]

    u_params_upper = bspline_helper.create_parameter_from_x_coords(upper_data, proc.param_exponent_upper)
    u_params_lower = bspline_helper.create_parameter_from_x_coords(lower_data, proc.param_exponent_lower)

    if not use_existing_knot_vectors:
        proc.upper_knot_vector = bspline_helper.create_knot_vector(proc.num_cp_upper, proc.degree_upper)
        proc.lower_knot_vector = bspline_helper.create_knot_vector(proc.num_cp_lower, proc.degree_lower)

    if proc.upper_knot_vector is None or proc.lower_knot_vector is None:
        raise ValueError("Knot vectors are unexpectedly None when building basis matrices in G2 optimization.")

    num_cp_upper = len(proc.upper_knot_vector) - proc.degree_upper - 1
    num_cp_lower = len(proc.lower_knot_vector) - proc.degree_lower - 1

    use_warm_start = bool(warm_start_from_current and use_existing_knot_vectors)
    warm_start_available = (
        use_warm_start
        and proc.upper_control_points is not None
        and proc.lower_control_points is not None
        and len(proc.upper_control_points) == num_cp_upper
        and len(proc.lower_control_points) == num_cp_lower
    )
    if not warm_start_available:
        proc._fit_g1_independent(
            upper_data,
            lower_data,
            (num_cp_upper, num_cp_lower),
            upper_te_dir,
            lower_te_dir,
            enforce_te_tangency,
            use_existing_knot_vectors,
        )
    else:
        proc.upper_control_points = np.asarray(proc.upper_control_points, dtype=float).copy()
        proc.lower_control_points = np.asarray(proc.lower_control_points, dtype=float).copy()

    initial_vars = control_points_to_initial_vars(
        proc.upper_control_points,
        proc.lower_control_points,
        num_cp_upper,
        num_cp_lower,
    )
    layout = OptimizationLayout(num_cp_upper, num_cp_lower)

    num_vars = layout.num_vars

    insertion_mode = bool(use_existing_knot_vectors and warm_start_from_current)
    max_deg = max(proc.degree_upper, proc.degree_lower)
    max_iter = max(200, num_vars * 20)
    ftol = 1e-7

    if insertion_mode:
        insertion_target = max(
            proc.insertion_solver_min_maxiter,
            int(np.ceil(num_vars * proc.insertion_solver_maxiter_factor)),
        )
        max_iter = max(max_iter, insertion_target)
        ftol = proc.insertion_solver_ftol

    if max_deg > 10:
        max_iter += (max_deg - 10) * 100

    def build_problem(
        metric: str,
        start_vars: np.ndarray,
        basis_upper_local: np.ndarray,
        basis_lower_local: np.ndarray,
    ) -> dict:
        return build_g2_problem(
            upper_data=upper_data,
            lower_data=lower_data,
            basis_upper=basis_upper_local,
            basis_lower=basis_lower_local,
            upper_knot_vector=proc.upper_knot_vector,
            lower_knot_vector=proc.lower_knot_vector,
            degree_upper=proc.degree_upper,
            degree_lower=proc.degree_lower,
            te_point_upper=te_point_upper,
            te_point_lower=te_point_lower,
            upper_te_dir=upper_te_dir,
            lower_te_dir=lower_te_dir,
            enforce_te_tangency=enforce_te_tangency,
            smoothing_weight=proc.smoothing_weight,
            initial_vars=np.asarray(start_vars, dtype=float),
            layout=layout,
            vars_to_control_points_fn=lambda x: vars_to_control_points(x, num_cp_upper, num_cp_lower),
            enforce_g3=proc.enforce_g3,
            fit_error_metric=metric,
        )

    pure_metric = _resolve_pure_fit_error_metric()
    basis_upper = bspline_helper.build_basis_matrix(u_params_upper, proc.upper_knot_vector, proc.degree_upper)
    basis_lower = bspline_helper.build_basis_matrix(u_params_lower, proc.lower_knot_vector, proc.degree_lower)
    problem = build_problem(pure_metric, initial_vars, basis_upper, basis_lower)
    constraints = problem["constraints"]
    bounds = problem["bounds"]
    result = optimize.minimize(
        problem["objective"],
        problem["initial_vars"],
        method="SLSQP",
        jac=problem["objective_jac"],
        constraints=constraints,
        bounds=bounds,
        options={"ftol": ftol, "maxiter": max_iter, "disp": False},
    )
    total_iterations = int(getattr(result, "nit", -1))

    max_constraint_violation = 0.0
    if getattr(result, "x", None) is not None:
        x_final = np.asarray(result.x, dtype=float)
        for constraint in constraints:
            cval = np.asarray(constraint["fun"](x_final), dtype=float).ravel()
            if cval.size:
                max_constraint_violation = max(
                    max_constraint_violation,
                    float(np.max(np.abs(cval))),
                )
    relaxed_success = (
        int(getattr(result, "status", -1)) == 9
        and np.isfinite(max_constraint_violation)
        and max_constraint_violation <= 2e-5
    )
    accepted = bool(result.success or result.status == 0 or relaxed_success)
    proc.last_optimizer_info = {
        "success": bool(result.success),
        "accepted": accepted,
        "accepted_via_relaxed_criteria": bool(relaxed_success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(total_iterations),
        "objective": float(getattr(result, "fun", np.nan)),
        "max_constraint_violation": float(max_constraint_violation),
        "solver_ftol": float(ftol),
        "solver_maxiter": int(max_iter),
        "insertion_mode": bool(insertion_mode),
        "fit_error_metric": str(problem.get("fit_error_metric", "unknown")),
        "fit_error_samples": int(problem.get("fit_error_samples", -1)),
        "fit_error_samples_coarse": int(problem.get("fit_error_samples_coarse", -1)),
        "fit_error_samples_medium": int(problem.get("fit_error_samples_medium", -1)),
        "fit_error_refresh_every": int(problem.get("fit_error_refresh_every", -1)),
        "fit_error_force_full_precision": bool(problem.get("fit_error_force_full_precision", False)),
        "mode": "g3" if bool(proc.enforce_g3) else "g2",
    }

    if accepted:
        proc.upper_control_points, proc.lower_control_points = vars_to_control_points(result.x, num_cp_upper, num_cp_lower)
        return True

    proc.last_error_message = f"G2 optimization failed (status={result.status}): {result.message}"
    return False
