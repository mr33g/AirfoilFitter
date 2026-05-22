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
    metric = str(getattr(config, "FIT_ERROR_OBJECTIVE", "vertical")).strip().lower()
    if metric in {"msr", "vertical"}:
        return metric
    return "vertical"


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
        proc.upper_te_dir = None if upper_te_dir is None else np.asarray(upper_te_dir, dtype=float).copy()
        proc.lower_te_dir = None if lower_te_dir is None else np.asarray(lower_te_dir, dtype=float).copy()

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
                enable_soft_te_handle_quality=True,
                use_existing_knot_vectors=can_reuse_existing_knots,
            )

        proc._finalize_curves()
        proc.fitted_degree = (proc.degree_upper, proc.degree_lower)
        proc.fitted = True

        proc.num_cp_upper = len(proc.upper_control_points)
        proc.num_cp_lower = len(proc.lower_control_points)

        proc._validate_continuity()
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
    use_existing_knot_vectors: bool = False,
    warm_start_from_current: bool = False,
    use_insertion_solver_settings: bool = False,
) -> bool:
    """Fit both surfaces with G2 continuity using constrained optimization."""
    _ = num_control_points
    enable_soft_te_handle_quality = True
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
            enable_soft_te_handle_quality=enable_soft_te_handle_quality,
            use_existing_knot_vectors=use_existing_knot_vectors,
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
    insertion_mode = bool(use_insertion_solver_settings)
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
    if enable_soft_te_handle_quality:
        max_iter += 80
    if bool(proc.enforce_g3):
        max_iter += 300

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
            enable_soft_te_handle_quality=enable_soft_te_handle_quality,
            smoothing_weight=float(proc.smoothing_weight),
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
    initial_diag = dict(problem["evaluate_diagnostics"](initial_vars))

    result = optimize.minimize(
        problem["objective"],
        problem["initial_vars"],
        method="SLSQP",
        jac=problem["objective_jac"],
        constraints=problem["constraints"],
        bounds=problem["bounds"],
        options={"ftol": ftol, "maxiter": max_iter, "disp": False},
    )
    total_iterations = int(getattr(result, "nit", -1))

    max_constraint_violation = 0.0
    if getattr(result, "x", None) is not None:
        x_final = np.asarray(result.x, dtype=float)
        for constraint in problem["constraints"]:
            cval = np.asarray(constraint["fun"](x_final), dtype=float).ravel()
            if cval.size:
                ctype = str(constraint.get("type", "eq")).strip().lower()
                if ctype == "ineq":
                    violation = float(np.max(np.maximum(0.0, -cval)))
                else:
                    violation = float(np.max(np.abs(cval)))
                max_constraint_violation = max(max_constraint_violation, violation)
    else:
        x_final = np.asarray(initial_vars, dtype=float)

    final_diag = dict(problem["evaluate_diagnostics"](x_final))
    accepted = bool(result.success)

    proc.last_optimizer_info = {
        "success": bool(result.success),
        "accepted": accepted,
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(total_iterations),
        "objective": float(getattr(result, "fun", np.nan)),
        "max_constraint_violation": float(max_constraint_violation),
        "solver_ftol": float(ftol),
        "solver_maxiter": int(max_iter),
        "insertion_mode": bool(insertion_mode),
        "solver_profile": "insertion" if insertion_mode else "default",
        "smoothing_weight": float(proc.smoothing_weight),
        "fit_error_metric": str(problem.get("fit_error_metric", "unknown")),
        "initial_fit_error_total": float(initial_diag["fit_error_total"]),
        "final_fit_error_total": float(final_diag["fit_error_total"]),
        "initial_smoothing_penalty_total": float(initial_diag["smoothing_penalty_total"]),
        "final_smoothing_penalty_total": float(final_diag["smoothing_penalty_total"]),
        "final_smoothing_penalty_upper_fourth_diff": float(final_diag["smoothing_penalty_upper_fourth_diff"]),
        "final_smoothing_penalty_lower_fourth_diff": float(final_diag["smoothing_penalty_lower_fourth_diff"]),
        "raw_smoothing_baseline_total": float(final_diag.get("raw_smoothing_baseline_total", np.nan)),
        "smoothing_reference_fit": float(final_diag.get("smoothing_reference_fit", np.nan)),
        "smoothing_objective_scale": float(final_diag.get("smoothing_objective_scale", np.nan)),
        "te_handle_weight": float(final_diag.get("te_handle_weight", np.nan)),
        "te_handle_min_length": float(final_diag.get("te_handle_min_length", np.nan)),
        "initial_te_handle_penalty_total": float(initial_diag.get("te_handle_penalty_total", np.nan)),
        "final_te_handle_penalty_total": float(final_diag.get("te_handle_penalty_total", np.nan)),
        "final_te_handle_penalty_upper_angle": float(final_diag.get("te_handle_penalty_upper_angle", np.nan)),
        "final_te_handle_penalty_lower_angle": float(final_diag.get("te_handle_penalty_lower_angle", np.nan)),
        "final_te_handle_penalty_upper_short_length": float(final_diag.get("te_handle_penalty_upper_short_length", np.nan)),
        "final_te_handle_penalty_lower_short_length": float(final_diag.get("te_handle_penalty_lower_short_length", np.nan)),
        "final_te_handle_upper_length": float(final_diag.get("te_handle_upper_length", np.nan)),
        "final_te_handle_lower_length": float(final_diag.get("te_handle_lower_length", np.nan)),
        "raw_te_handle_baseline_total": float(final_diag.get("raw_te_handle_baseline_total", np.nan)),
        "te_handle_objective_scale": float(final_diag.get("te_handle_objective_scale", np.nan)),
        "mode": "g3" if bool(proc.enforce_g3) else "g2",
    }

    if accepted:
        proc.upper_control_points, proc.lower_control_points = vars_to_control_points(result.x, num_cp_upper, num_cp_lower)
        return True

    proc.last_error_message = f"G2 optimization failed (status={result.status}): {result.message}"
    return False
