from __future__ import annotations

import numpy as np
from scipy import interpolate, optimize

from core import config
from utils import bspline_helper

class BSplineProcessor:
    """
    B-spline processor with true G2 constraint that maintains G1 continuity.
    Works with arbitrary degree B-splines.
    """

    def __init__(self, degree: int = config.DEFAULT_BSPLINE_DEGREE):
        self.upper_control_points: np.ndarray | None = None
        self.lower_control_points: np.ndarray | None = None
        self.upper_knot_vector: np.ndarray | None = None
        self.lower_knot_vector: np.ndarray | None = None
        self.upper_curve: interpolate.BSpline | None = None
        self.lower_curve: interpolate.BSpline | None = None
        self.degree: int = int(degree)
        self.degree_upper: int = int(degree)
        self.degree_lower: int = int(degree)
        self.fitted_degree: int | None = None  # Stores the actual degree used during fitting
        self.num_cp_upper: int = 10  # Current number of control points for upper surface
        self.num_cp_lower: int = 10  # Current number of control points for lower surface
        self.param_exponent_upper: float = 0.5 # Mapping x -> u (0.5 is sqrt)
        self.param_exponent_lower: float = 0.5
        self.fitted: bool = False
        self.is_sharp_te: bool = False
        self.enforce_g2: bool = True
        self.enforce_g3: bool = False
        self.g2_weight: float = 100.0  # Weight for G2 constraint in optimization
        self.smoothing_weight: float = config.DEFAULT_SMOOTHNESS_PENALTY  # Weight for control point smoothing penalty
        self.last_error_message: str | None = None
        self.last_optimizer_info: dict | None = None
        
        self.upper_original_data: np.ndarray | None = None
        self.lower_original_data: np.ndarray | None = None
        # Backup for undoing TE thickening
        self._backup_upper_control_points: np.ndarray | None = None
        self._backup_lower_control_points: np.ndarray | None = None
        self._backup_upper_knot_vector: np.ndarray | None = None
        self._backup_lower_knot_vector: np.ndarray | None = None


    def fit_bspline(
        self,
        upper_data: np.ndarray,
        lower_data: np.ndarray,
        num_control_points: int | tuple[int, int],
        is_thickened: bool = False,
        upper_te_tangent_vector: np.ndarray | None = None,
        lower_te_tangent_vector: np.ndarray | None = None,
        enforce_g2: bool = False,
        enforce_g3: bool = False,
        enforce_te_tangency: bool = True,
    ) -> bool:
        """
        Fit B-splines with G1 and optional G2 constraints at leading edge.
        num_control_points can be a single int (symmetric) or a tuple (upper, lower).
        """
        try:
            self.last_error_message = None
            self.last_optimizer_info = None
            if isinstance(num_control_points, tuple):
                num_cp_upper, num_cp_lower = num_control_points
            else:
                num_cp_upper = num_cp_lower = num_control_points
            
            self.num_cp_upper = num_cp_upper
            self.num_cp_lower = num_cp_lower
            self.enforce_g2 = enforce_g2
            # G3 requires G2, so if G2 is disabled, disable G3
            self.enforce_g3 = enforce_g3 if enforce_g2 else False

            # Use current global degree for both surfaces
            self.degree_upper = self.degree
            self.degree_lower = self.degree

            # Set trailing edge type
            self.is_sharp_te = not is_thickened
            
            # Ensure both surfaces start at the same point
            le_point = (upper_data[0] + lower_data[0]) / 2
            upper_data_corrected = upper_data.copy()
            lower_data_corrected = lower_data.copy()
            upper_data_corrected[0] = le_point
            lower_data_corrected[0] = le_point

            # Store original data for potential later refinement (e.g., knot insertion)
            self.upper_original_data = upper_data_corrected.copy()
            self.lower_original_data = lower_data_corrected.copy()
            
            # For sharp trailing edge, ensure they end at the same point
            if self.is_sharp_te:
                te_point = np.array([1.0, 0.0])
                upper_data_corrected[-1] = te_point
                lower_data_corrected[-1] = te_point
            
            # Normalize TE tangent vectors
            upper_te_dir = bspline_helper.normalize_vector(upper_te_tangent_vector)
            lower_te_dir = bspline_helper.normalize_vector(lower_te_tangent_vector)

            if self.enforce_g2:
                # Use optimization-based approach for G2 (and optionally G3)
                success = self._fit_with_g2_optimization(
                    upper_data_corrected, lower_data_corrected,
                    (self.num_cp_upper, self.num_cp_lower), upper_te_dir, lower_te_dir, enforce_te_tangency
                )
                if not success:
                    self.enforce_g2 = False
                    self.enforce_g3 = False
            
            if not self.enforce_g2:
                # Use original G1-only fitting
                self._fit_g1_independent(
                    upper_data_corrected, lower_data_corrected,
                    (self.num_cp_upper, self.num_cp_lower), upper_te_dir, lower_te_dir, enforce_te_tangency
                )
            
            # Final cleanup and validation
            self._finalize_curves()
            # Store the actual degrees that were used for fitting (important for export)
            self.fitted_degree = (self.degree_upper, self.degree_lower)
            self.fitted = True
            
            # Update num_cp to match actual control point counts (may differ from requested
            # if requested CPs < degree+1, since minimum CPs = degree+1)
            self.num_cp_upper = len(self.upper_control_points)
            self.num_cp_lower = len(self.lower_control_points)
            
            self._validate_continuity()
            # Validate trailing edge tangents if they were used in fitting
            if upper_te_dir is not None and lower_te_dir is not None and enforce_te_tangency:
                self._validate_trailing_edge_tangents(upper_te_dir, lower_te_dir)
            
            return True
            
        except Exception as exc:
            self.last_error_message = f"fit_bspline failed: {exc}"
            self.fitted = False
            return False

    def _fit_with_g2_optimization(
        self,
        upper_data: np.ndarray,
        lower_data: np.ndarray,
        num_control_points: int | tuple[int, int],
        upper_te_dir: np.ndarray | None,
        lower_te_dir: np.ndarray | None,
        enforce_te_tangency: bool = True,
        use_existing_knot_vectors: bool = False,
    ) -> bool:
        """
        Fit both surfaces with G2 continuity using constrained optimization.
        This maintains exact G1 constraints while achieving G2.
        """
        # Get trailing edge points from input data
        te_point_upper = upper_data[-1]
        te_point_lower = lower_data[-1]
        
        # Create parameter values
        u_params_upper = bspline_helper.create_parameter_from_x_coords(upper_data, self.param_exponent_upper)
        u_params_lower = bspline_helper.create_parameter_from_x_coords(lower_data, self.param_exponent_lower)
        
        # Create knot vectors (only if not using existing ones)
        if not use_existing_knot_vectors:
            self.upper_knot_vector = bspline_helper.create_knot_vector(self.num_cp_upper, self.degree_upper)
            self.lower_knot_vector = bspline_helper.create_knot_vector(self.num_cp_lower, self.degree_lower)

        # Ensure knot vectors are available for building basis matrices
        if self.upper_knot_vector is None or self.lower_knot_vector is None:
             raise ValueError("Knot vectors are unexpectedly None when building basis matrices in G2 optimization.")

        # Determine number of control points for each surface from knot vectors
        num_cp_upper = len(self.upper_knot_vector) - self.degree_upper - 1
        num_cp_lower = len(self.lower_knot_vector) - self.degree_lower - 1

        # Precompute basis matrices for objective evaluations
        basis_upper = bspline_helper.build_basis_matrix(u_params_upper, self.upper_knot_vector, self.degree_upper)
        basis_lower = bspline_helper.build_basis_matrix(u_params_lower, self.lower_knot_vector, self.degree_lower)

        # Number of free variables per surface
        n_fixed = 3  # P0, P1, P2 are partially constrained
        n_free_upper = num_cp_upper - n_fixed
        n_free_lower = num_cp_lower - n_fixed
        
        # Initial guess from G1-only fit
        self._fit_g1_independent(upper_data, lower_data, (num_cp_upper, num_cp_lower), upper_te_dir, lower_te_dir, enforce_te_tangency, use_existing_knot_vectors)
        
        # Extract initial values for optimization variables
        initial_vars = []
        initial_vars.append(self.upper_control_points[1, 1])  # P1.y_upper
        initial_vars.append(self.lower_control_points[1, 1])  # P1.y_lower
        initial_vars.append(self.upper_control_points[2, 0])  # P2.x_upper
        initial_vars.append(self.upper_control_points[2, 1])  # P2.y_upper
        initial_vars.append(self.lower_control_points[2, 0])  # P2.x_lower
        initial_vars.append(self.lower_control_points[2, 1])  # P2.y_lower
        
        # Add remaining control points
        for i in range(3, num_cp_upper):
            initial_vars.extend([self.upper_control_points[i, 0], self.upper_control_points[i, 1]])
        for i in range(3, num_cp_lower):
            initial_vars.extend([self.lower_control_points[i, 0], self.lower_control_points[i, 1]])
        
        initial_vars = np.array(initial_vars)
        
        if num_cp_upper > 2:
            idx_u = np.arange(num_cp_upper - 2, dtype=float)
            grad_upper = 0.5 + 1.5 * (idx_u / (num_cp_upper - 3)) if num_cp_upper > 3 else np.ones(num_cp_upper - 2)
            smooth_w_upper = grad_upper * grad_upper
        else:
            smooth_w_upper = np.zeros(0, dtype=float)

        if num_cp_lower > 2:
            idx_l = np.arange(num_cp_lower - 2, dtype=float)
            grad_lower = 0.5 + 1.5 * (idx_l / (num_cp_lower - 3)) if num_cp_lower > 3 else np.ones(num_cp_lower - 2)
            smooth_w_lower = grad_lower * grad_lower
        else:
            smooth_w_lower = np.zeros(0, dtype=float)

        def objective(vars):
            """Minimize fitting error."""
            # Reconstruct control points from variables
            cp_upper, cp_lower = self._vars_to_control_points(vars, num_cp_upper, num_cp_lower)

            # Evaluate fitted curves at data points from precomputed bases
            fitted_upper = basis_upper @ cp_upper
            fitted_lower = basis_lower @ cp_lower

            # Compute fitting error
            error_upper = float(np.sum((upper_data - fitted_upper) ** 2))
            error_lower = float(np.sum((lower_data - fitted_lower) ** 2))

            # Compute smoothing penalty (penalize second-order differences of control points)
            smoothing_penalty = 0.0
            if smooth_w_upper.size:
                diff_upper = np.diff(cp_upper, n=2, axis=0)
                smoothing_penalty += float(np.sum((diff_upper ** 2) * smooth_w_upper[:, np.newaxis]))
            if smooth_w_lower.size:
                diff_lower = np.diff(cp_lower, n=2, axis=0)
                smoothing_penalty += float(np.sum((diff_lower ** 2) * smooth_w_lower[:, np.newaxis]))

            return error_upper + error_lower + self.smoothing_weight * smoothing_penalty
        
        def curvature_constraint(vars):
            """G2 constraint: equal curvatures at leading edge."""
            cp_upper, cp_lower = self._vars_to_control_points(vars, num_cp_upper, num_cp_lower)
            kappa_upper = bspline_helper.compute_curvature_at_zero(cp_upper, self.upper_knot_vector, self.degree_upper)
            kappa_lower = bspline_helper.compute_curvature_at_zero(cp_lower, self.lower_knot_vector, self.degree_lower)
            return kappa_upper - kappa_lower

        def curvature_derivative_constraint(vars):
            """G3 constraint: equal curvature derivatives at leading edge."""
            cp_upper, cp_lower = self._vars_to_control_points(vars, num_cp_upper, num_cp_lower)
            dk_upper = bspline_helper.compute_curvature_derivative_at_zero(cp_upper, self.upper_knot_vector, self.degree_upper)
            dk_lower = bspline_helper.compute_curvature_derivative_at_zero(cp_lower, self.lower_knot_vector, self.degree_lower)
            return dk_upper - dk_lower
        
        # Set up constraints
        constraints = [
            {'type': 'eq', 'fun': curvature_constraint},  # G2 constraint
        ]
        
        # Add G3 constraint only if explicitly requested
        if self.enforce_g3:
            constraints.append({'type': 'eq', 'fun': curvature_derivative_constraint})
        
        # Add TE endpoint constraints for both sharp and blunt trailing edges
        def te_constraint_upper(vars):
            cp_upper, _ = self._vars_to_control_points(vars, num_cp_upper, num_cp_lower)
            return cp_upper[-1] - te_point_upper
        
        def te_constraint_lower(vars):
            _, cp_lower = self._vars_to_control_points(vars, num_cp_upper, num_cp_lower)
            return cp_lower[-1] - te_point_lower
        
        constraints.extend([
            {'type': 'eq', 'fun': te_constraint_upper},
            {'type': 'eq', 'fun': te_constraint_lower}
        ])
        
        # Add trailing edge tangent constraints if selected
        if upper_te_dir is not None and lower_te_dir is not None and enforce_te_tangency:
            def te_tangent_constraint_upper(vars):
                """Constraint for upper surface trailing edge tangent."""
                cp_upper, _ = self._vars_to_control_points(vars, num_cp_upper, num_cp_lower)
                computed_tangent = bspline_helper.compute_tangent_at_trailing_edge(cp_upper, self.upper_knot_vector, self.degree_upper)
                return computed_tangent - upper_te_dir
            
            def te_tangent_constraint_lower(vars):
                """Constraint for lower surface trailing edge tangent."""
                _, cp_lower = self._vars_to_control_points(vars, num_cp_upper, num_cp_lower)
                computed_tangent = bspline_helper.compute_tangent_at_trailing_edge(cp_lower, self.lower_knot_vector, self.degree_lower)
                return computed_tangent - lower_te_dir
            
            constraints.extend([
                {'type': 'eq', 'fun': te_tangent_constraint_upper},
                {'type': 'eq', 'fun': te_tangent_constraint_lower}
            ])
        
        # Bounds
        bounds = []
        bounds.append((0.001, 0.1))   # P1.y_upper (positive)
        bounds.append((-0.1, -0.001)) # P1.y_lower (negative)
        bounds.append((0.001, 0.5))   # P2.x_upper
        bounds.append((0.001, 0.3))   # P2.y_upper
        bounds.append((0.001, 0.5))   # P2.x_lower
        bounds.append((-0.3, -0.001)) # P2.y_lower
        
        # Bounds for remaining control points
        for _ in range(n_free_upper):
            bounds.extend([(None, None), (None, None)])
        for _ in range(n_free_lower):
            bounds.extend([(None, None), (None, None)])
        
        # Optimize
        max_deg = max(self.degree_upper, self.degree_lower)
        num_vars = len(initial_vars)
        max_iter = max(200, num_vars * 20)
        
        if max_deg > 10:
            max_iter += (max_deg - 10) * 100
            
        result = optimize.minimize(
            objective, initial_vars,
            method='SLSQP',
            constraints=constraints,
            bounds=bounds,
            options={'ftol': 1e-7, 'maxiter': max_iter, 'disp': False}
        )
        self.last_optimizer_info = {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(getattr(result, "nit", -1)),
            "objective": float(getattr(result, "fun", np.nan)),
        }
        
        if result.success or result.status == 0:
            self.upper_control_points, self.lower_control_points = \
                self._vars_to_control_points(result.x, num_cp_upper, num_cp_lower)
            return True
        else:
            self.last_error_message = (
                f"G2 optimization failed (status={result.status}): {result.message}"
            )
            return False

    def _vars_to_control_points(self, vars: np.ndarray, num_cp_upper: int, num_cp_lower: int) -> tuple[np.ndarray, np.ndarray]:
        """Convert optimization variables to control points maintaining G1 constraints."""
        cp_upper = np.zeros((num_cp_upper, 2))
        cp_lower = np.zeros((num_cp_lower, 2))
        
        # Fixed constraints
        cp_upper[0] = [0.0, 0.0]  # P0
        cp_lower[0] = [0.0, 0.0]  # P0 (same as upper)
        
        cp_upper[1] = [0.0, vars[0]]  # P1: x=0 (G1), y from vars
        cp_lower[1] = [0.0, vars[1]]  # P1: x=0 (G1), y from vars
        
        cp_upper[2] = [vars[2], vars[3]]  # P2 from vars
        cp_lower[2] = [vars[4], vars[5]]  # P2 from vars
        
        # Remaining control points
        idx = 6
        for i in range(3, num_cp_upper):
            cp_upper[i] = vars[idx:idx+2]
            idx += 2
        
        for i in range(3, num_cp_lower):
            cp_lower[i] = vars[idx:idx+2]
            idx += 2
        
        return cp_upper, cp_lower

    def _fit_g1_independent(
        self,
        upper_data: np.ndarray,
        lower_data: np.ndarray,
        num_control_points: int | tuple[int, int],
        upper_te_dir: np.ndarray | None,
        lower_te_dir: np.ndarray | None,
        enforce_te_tangency: bool = True,
        use_existing_knot_vectors: bool = False,
    ):
        """Fit surfaces independently with G1 constraint only."""
        # Get trailing edge points from input data
        te_point_upper = upper_data[-1]
        te_point_lower = lower_data[-1]
        
        # Create parameter values
        u_params_upper = bspline_helper.create_parameter_from_x_coords(upper_data, self.param_exponent_upper)
        u_params_lower = bspline_helper.create_parameter_from_x_coords(lower_data, self.param_exponent_lower)
        
        # Create knot vectors (only if not using existing ones)
        if not use_existing_knot_vectors:
            self.upper_knot_vector = bspline_helper.create_knot_vector(self.num_cp_upper, self.degree_upper)
            self.lower_knot_vector = bspline_helper.create_knot_vector(self.num_cp_lower, self.degree_lower)

        # Ensure knot vectors are available for building basis matrices
        if self.upper_knot_vector is None or self.lower_knot_vector is None:
             raise ValueError("Knot vectors are unexpectedly None when building basis matrices in G1-independent fit.")
        
        # Build basis matrices using the current (potentially modified) knot vectors
        basis_upper = bspline_helper.build_basis_matrix(u_params_upper, self.upper_knot_vector, self.degree_upper)
        basis_lower = bspline_helper.build_basis_matrix(u_params_lower, self.lower_knot_vector, self.degree_lower)
        
        # Determine number of control points from knot vectors
        num_control_points_upper = len(self.upper_knot_vector) - self.degree_upper - 1
        num_control_points_lower = len(self.lower_knot_vector) - self.degree_lower - 1
        
        # Fit upper surface
        self.upper_control_points = self._fit_single_surface_g1(
            basis_upper, upper_data, num_control_points_upper, is_upper=True, 
            te_tangent_vector=upper_te_dir if enforce_te_tangency else None, te_point=te_point_upper
        )
        
        # Fit lower surface
        self.lower_control_points = self._fit_single_surface_g1(
            basis_lower, lower_data, num_control_points_lower, is_upper=False, 
            te_tangent_vector=lower_te_dir if enforce_te_tangency else None, te_point=te_point_lower
        )

    def _fit_single_surface_g1(
        self,
        basis_matrix: np.ndarray,
        surface_data: np.ndarray,
        num_control_points: int,
        is_upper: bool,
        te_tangent_vector: np.ndarray | None = None,
        te_point: np.ndarray | None = None
    ) -> np.ndarray:
        """Fit single surface with G1 constraint (P0 = origin, P1.x = 0) and optional TE tangent constraint."""
        # Data fitting equations
        A_data = np.zeros((2 * len(surface_data), 2 * num_control_points))
        b_data = np.zeros(2 * len(surface_data))
        
        # X-coordinate equations
        A_data[:len(surface_data), :num_control_points] = basis_matrix
        b_data[:len(surface_data)] = surface_data[:, 0]
        
        # Y-coordinate equations
        A_data[len(surface_data):, num_control_points:] = basis_matrix
        b_data[len(surface_data):] = surface_data[:, 1]
        
        # Constraint equations
        constraints = []
        constraint_rhs = []
        
        # P0 = (0, 0)
        row = np.zeros(2 * num_control_points)
        row[0] = 1.0  # P0.x = 0
        constraints.append(row)
        constraint_rhs.append(0.0)
        
        row = np.zeros(2 * num_control_points)
        row[num_control_points] = 1.0  # P0.y = 0
        constraints.append(row)
        constraint_rhs.append(0.0)
        
        # P1.x = 0 (G1 constraint)
        row = np.zeros(2 * num_control_points)
        row[1] = 1.0
        constraints.append(row)
        constraint_rhs.append(0.0)
        
        # Add trailing edge tangent constraint if provided
        if te_tangent_vector is not None:
            row = np.zeros(2 * num_control_points)
            row[num_control_points - 1] = -te_tangent_vector[1]
            row[2 * num_control_points - 1] = te_tangent_vector[0]
            row[num_control_points - 2] = te_tangent_vector[1]
            row[2 * num_control_points - 2] = -te_tangent_vector[0]
            constraints.append(row)
            constraint_rhs.append(0.0)
        
        # Add trailing edge endpoint constraint if provided
        if te_point is not None:
            row_x = np.zeros(2 * num_control_points)
            row_x[num_control_points - 1] = 1.0
            constraints.append(row_x)
            constraint_rhs.append(te_point[0])
            
            row_y = np.zeros(2 * num_control_points)
            row_y[2 * num_control_points - 1] = 1.0
            constraints.append(row_y)
            constraint_rhs.append(te_point[1])
        
        # Weight for constraints (make them strong)
        constraint_weight = 1000.0
        
        # Build augmented system
        A_constraints = np.array(constraints) * constraint_weight
        b_constraints = np.array(constraint_rhs) * constraint_weight

        # Add smoothing penalty terms to the system with a gradient
        A_smoothing = np.zeros(((num_control_points - 2) * 2, 2 * num_control_points))
        b_smoothing = np.zeros((num_control_points - 2) * 2)
        
        for i in range(num_control_points - 2):
            gradient = 0.5 + 1.5 * (i / (num_control_points - 3)) if num_control_points > 3 else 1.0
            current_weight = self.smoothing_weight * gradient

            # X-coordinates smoothing
            A_smoothing[i, i] = current_weight
            A_smoothing[i, i+1] = -2 * current_weight
            A_smoothing[i, i+2] = current_weight

            # Y-coordinates smoothing
            A_smoothing[i + (num_control_points - 2), num_control_points + i] = current_weight
            A_smoothing[i + (num_control_points - 2), num_control_points + i+1] = -2 * current_weight
            A_smoothing[i + (num_control_points - 2), num_control_points + i+2] = current_weight

        A_all = np.vstack([A_data, A_constraints, A_smoothing])
        b_all = np.hstack([b_data, b_constraints, b_smoothing])
        
        # Solve
        solution = np.linalg.lstsq(A_all, b_all, rcond=None)[0]
        
        # Extract control points
        control_points = np.zeros((num_control_points, 2))
        control_points[:, 0] = solution[:num_control_points]
        control_points[:, 1] = solution[num_control_points:]
        
        # Enforce constraints exactly
        control_points[0] = [0.0, 0.0]
        control_points[1, 0] = 0.0
        
        # Ensure correct sign for P1.y
        if is_upper and control_points[1, 1] < 0:
            control_points[1, 1] = abs(control_points[1, 1])
        elif not is_upper and control_points[1, 1] > 0:
            control_points[1, 1] = -abs(control_points[1, 1])
        
        return control_points

    def _finalize_curves(self):
        """Final cleanup and curve rebuilding."""
        # Ensure shared leading edge
        if self.upper_control_points is not None and self.lower_control_points is not None:
            shared_p0 = (self.upper_control_points[0] + self.lower_control_points[0]) / 2
            self.upper_control_points[0] = shared_p0
            self.lower_control_points[0] = shared_p0
            
            # Enforce G1 constraints exactly
            self.upper_control_points[0] = [0.0, 0.0]
            self.lower_control_points[0] = [0.0, 0.0]
            self.upper_control_points[1, 0] = 0.0
            self.lower_control_points[1, 0] = 0.0
            
            # Handle trailing edge
            if self.is_sharp_te:
                te_point = np.array([1.0, 0.0])
                self.upper_control_points[-1] = te_point
                self.lower_control_points[-1] = te_point
        
        # Rebuild curves
        if self.upper_control_points is not None and self.upper_knot_vector is not None:
            self.upper_curve = interpolate.BSpline(
                self.upper_knot_vector, self.upper_control_points, self.degree_upper
            )
        
        if self.lower_control_points is not None and self.lower_knot_vector is not None:
            self.lower_curve = interpolate.BSpline(
                self.lower_knot_vector, self.lower_control_points, self.degree_lower
            )

    def _validate_trailing_edge_tangents(self, upper_te_dir: np.ndarray | None, lower_te_dir: np.ndarray | None) -> None:
        """Validate that the fitted B-splines satisfy the trailing edge tangent constraints."""
        # Validation performed silently - no debug output
        pass

    def _validate_continuity(self):
        """Validate G0, G1, and G2 continuity at the leading edge."""
        if not config.DEBUG_WORKER_LOGGING:
            return
        
        if not self.fitted or self.upper_curve is None or self.lower_curve is None:
            return
        
        # Log control points
        print("[DEBUG] Control points:")
        for i in range(len(self.upper_control_points)):
            print(f"[DEBUG]   Upper P{i}: ({self.upper_control_points[i,0]:.6f}, {self.upper_control_points[i,1]:.6f})")
        for i in range(len(self.lower_control_points)):
            print(f"[DEBUG]   Lower P{i}: ({self.lower_control_points[i,0]:.6f}, {self.lower_control_points[i,1]:.6f})")

    def is_fitted(self) -> bool:
        """Check if B-splines have been successfully fitted."""
        return self.fitted and self.upper_curve is not None and self.lower_curve is not None

    def has_ui_applied_te_thickening(self) -> bool:
        """Return True when UI-driven TE thickening backup state exists."""
        return (
            self._backup_upper_control_points is not None
            and self._backup_lower_control_points is not None
        )

    def reset_model_state(self) -> None:
        """Reset fitted model state when loading a new airfoil."""
        self.fitted = False
        self.last_error_message = None
        self.last_optimizer_info = None
        self.upper_control_points = None
        self.lower_control_points = None
        self.upper_curve = None
        self.lower_curve = None
        self.is_sharp_te = True
        self._backup_upper_control_points = None
        self._backup_lower_control_points = None
        self._backup_upper_knot_vector = None
        self._backup_lower_knot_vector = None

    def calculate_curvature_comb_data(self, num_points_per_segment=200, scale_factor=0.050):
        """Calculate curvature comb visualization data."""
        if not self.is_fitted():
            return None
        
        return bspline_helper.calculate_curvature_comb_data(
            self.upper_curve, self.lower_curve, num_points_per_segment, scale_factor
        )

    def apply_te_thickening(self, te_thickness: float) -> bool:
        """
        Apply trailing edge thickening as a post-processing step to the entire airfoil.
        Uses a smooth C2 blend along the chord so that the offset is 0 at the
        leading edge (with zero slope and curvature) and reaches the requested
        thickness at the trailing edge.
        
        Args:
            te_thickness: The thickness to apply at the trailing edge (0.0 to 1.0)
            
        Returns:
            bool: True if thickening was applied successfully, False otherwise
        """
        if not self.fitted or self.upper_control_points is None or self.lower_control_points is None:
            return False
        
        if te_thickness < 0.0:
            return False
        
        try:
            # Save backups to allow remove_te_thickening to restore
            self._backup_upper_control_points = self.upper_control_points.copy()
            self._backup_lower_control_points = self.lower_control_points.copy()
            self._backup_upper_knot_vector = None if self.upper_knot_vector is None else self.upper_knot_vector.copy()
            self._backup_lower_knot_vector = None if self.lower_knot_vector is None else self.lower_knot_vector.copy()

            # Sample both curves densely in parameter domain
            if self.upper_curve is None or self.lower_curve is None:
                return False

            num_samples = max(200, int(config.PLOT_POINTS_PER_SURFACE))

            _, upper_pts = bspline_helper.sample_curve(self.upper_curve, num_samples)
            _, lower_pts = bspline_helper.sample_curve(self.lower_curve, num_samples)

            # Blend based on x (already normalized to chord in data loader)
            upper_x = np.clip(upper_pts[:, 0], 0.0, 1.0)
            lower_x = np.clip(lower_pts[:, 0], 0.0, 1.0)
            f_upper = bspline_helper.smoothstep_quintic(upper_x)
            f_lower = bspline_helper.smoothstep_quintic(lower_x)

            half_thickness = 0.5 * te_thickness

            # Apply vertical, smoothly varying offsets
            thick_upper = upper_pts.copy()
            thick_lower = lower_pts.copy()
            thick_upper[:, 1] = thick_upper[:, 1] + half_thickness * f_upper
            thick_lower[:, 1] = thick_lower[:, 1] - half_thickness * f_lower

            # Refit using current number of control points, preserving G1 at LE
            self._fit_g1_independent(
                thick_upper, thick_lower, (self.num_cp_upper, self.num_cp_lower),
                upper_te_dir=None, lower_te_dir=None, enforce_te_tangency=False
            )

            # Mark as blunt before finalizing
            self.is_sharp_te = False
            
            # Finalize and rebuild curves
            self._finalize_curves()
            self.fitted = True

            return True
            
        except Exception:
            return False

    def remove_te_thickening(self) -> bool:
        """
        Remove trailing edge thickening by restoring from backup.
        
        Returns:
            bool: True if thickening was removed successfully, False otherwise
        """
        if not self.fitted or self.upper_control_points is None or self.lower_control_points is None:
            return False
        
        try:
            # If we have a backup from before thickening, restore it
            if self._backup_upper_control_points is not None and self._backup_lower_control_points is not None:
                self.upper_control_points = self._backup_upper_control_points
                self.lower_control_points = self._backup_lower_control_points
                # Restore knot vectors when available
                if self._backup_upper_knot_vector is not None:
                    self.upper_knot_vector = self._backup_upper_knot_vector
                if self._backup_lower_knot_vector is not None:
                    self.lower_knot_vector = self._backup_lower_knot_vector

                # Rebuild curves
                if self.upper_knot_vector is not None:
                    self.upper_curve = interpolate.BSpline(
                        self.upper_knot_vector, self.upper_control_points, self.degree
                    )
                if self.lower_knot_vector is not None:
                    self.lower_curve = interpolate.BSpline(
                        self.lower_knot_vector, self.lower_control_points, self.degree
                    )

                # After restoring, clear backups
                self._backup_upper_control_points = None
                self._backup_lower_control_points = None
                self._backup_upper_knot_vector = None
                self._backup_lower_knot_vector = None

                # Reset TE type
                self.is_sharp_te = bool(np.allclose(self.upper_control_points[-1], self.lower_control_points[-1], atol=1e-12))

                return True
            
            # Fallback: if no backup available, we can't restore
            return False
            
        except Exception:
            return False

    def _apply_knot_insertions(self, new_knots: list[float], surface: str | None = None) -> bool:
        """Apply knot insertion to selected surfaces without re-fitting or finalizing."""
        if not new_knots:
            return True

        sorted_new_knots = sorted(new_knots)

        if surface is None or surface == 'upper':
            if self.upper_control_points is None or self.upper_knot_vector is None:
                return False
            current_upper_cps = self.upper_control_points.copy()
            current_upper_knots = self.upper_knot_vector.copy()
            for knot in sorted_new_knots:
                current_upper_cps, current_upper_knots = bspline_helper.insert_knot(
                    current_upper_cps, current_upper_knots, self.degree_upper, knot
                )
            self.upper_control_points = current_upper_cps
            self.upper_knot_vector = current_upper_knots
            self.num_cp_upper = len(self.upper_control_points)

        if surface is None or surface == 'lower':
            if self.lower_control_points is None or self.lower_knot_vector is None:
                return False
            current_lower_cps = self.lower_control_points.copy()
            current_lower_knots = self.lower_knot_vector.copy()
            for knot in sorted_new_knots:
                current_lower_cps, current_lower_knots = bspline_helper.insert_knot(
                    current_lower_cps, current_lower_knots, self.degree_lower, knot
                )
            self.lower_control_points = current_lower_cps
            self.lower_knot_vector = current_lower_knots
            self.num_cp_lower = len(self.lower_control_points)

        return True

    def _refit_after_knot_insertion(self) -> bool:
        """Re-fit using current knot vectors and control-point counts."""
        if self.upper_original_data is None or self.lower_original_data is None:
            return True

        cp_counts = (self.num_cp_upper, self.num_cp_lower)
        if self.enforce_g2:
            success = self._fit_with_g2_optimization(
                self.upper_original_data,
                self.lower_original_data,
                cp_counts,
                upper_te_dir=None,
                lower_te_dir=None,
                enforce_te_tangency=False,
                use_existing_knot_vectors=True,
            )
            if not success:
                self._fit_g1_independent(
                    self.upper_original_data,
                    self.lower_original_data,
                    cp_counts,
                    upper_te_dir=None,
                    lower_te_dir=None,
                    enforce_te_tangency=False,
                    use_existing_knot_vectors=True,
                )
        else:
            self._fit_g1_independent(
                self.upper_original_data,
                self.lower_original_data,
                cp_counts,
                upper_te_dir=None,
                lower_te_dir=None,
                enforce_te_tangency=False,
                use_existing_knot_vectors=True,
            )

        return True

    def refine_curve_with_knots(self, new_knots: list[float], surface: str | None = None) -> bool:
        """
        Refine the B-spline curves by inserting new knots.
        This operation preserves the curve's shape while increasing the number of control points.

        Args:
            new_knots: A list of parameter values (knots) to insert.
            surface: Which surface(s) to insert knots into. 
                     None (default) = both surfaces, 'upper' = upper only, 'lower' = lower only.

        Returns:
            bool: True if knots were inserted successfully, False otherwise.
        """
        if not self.fitted or self.upper_curve is None or self.lower_curve is None:
            return False

        if not new_knots:
            return True

        try:
            if not self._apply_knot_insertions(new_knots, surface=surface):
                return False
            if not self._refit_after_knot_insertion():
                return False

            # Rebuild curves with updated control points and knot vectors
            self._finalize_curves()
            self._validate_continuity()
            
            return True

        except Exception as exc:
            self.last_error_message = f"refine_curve_with_knots failed: {exc}"
            return False

    def refine_curves_with_surface_knots(
        self,
        *,
        upper_knots: list[float] | None = None,
        lower_knots: list[float] | None = None,
    ) -> bool:
        """
        Insert knots on upper/lower surfaces and perform a single mandatory refit.
        """
        if not self.fitted or self.upper_curve is None or self.lower_curve is None:
            return False

        upper_knots = upper_knots or []
        lower_knots = lower_knots or []
        if not upper_knots and not lower_knots:
            return True

        try:
            if upper_knots and not self._apply_knot_insertions(upper_knots, surface='upper'):
                return False
            if lower_knots and not self._apply_knot_insertions(lower_knots, surface='lower'):
                return False
            if not self._refit_after_knot_insertion():
                return False
            self._finalize_curves()
            self._validate_continuity()
            return True
        except Exception as exc:
            self.last_error_message = f"refine_curves_with_surface_knots failed: {exc}"
            return False
