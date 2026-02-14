"""
B-spline helper functions for airfoil processing.

This module contains utility functions for B-spline operations including:
- Parameter creation and knot vector generation
- Basis function evaluation
- Curvature and tangent computations
- Vector normalization
- Smoothstep functions for blending
"""

import numpy as np
from scipy import interpolate


def normalize_vector(vec: np.ndarray | None) -> np.ndarray | None:
    """
    Normalize a vector.
    
    Args:
        vec: Input vector to normalize
        
    Returns:
        Normalized vector or None if normalization fails
    """
    if vec is None:
        return None
    try:
        v = np.asarray(vec, dtype=float)
        n = float(np.hypot(v[0], v[1]))
        if n <= 1e-12:
            return None
        return v / n
    except Exception:
        return None


def create_parameter_from_x_coords(surface_data: np.ndarray, exponent: float = 0.5) -> np.ndarray:
    """
    Create parameter values using a power-law mapping x = u^(1/exponent).
    Default exponent 0.5 corresponds to the standard x = u² mapping.
    
    Args:
        surface_data: Surface data points
        exponent: The power used for parameterization (default 0.5)
        
    Returns:
        Parameter values for B-spline fitting
    """
    x_coords = surface_data[:, 0]
    x_coords = np.clip(x_coords, 0.0, 1.0 - 1e-12)
    
    # Use np.power for flexible parameterization density
    # u = x^exponent
    u_params = np.power(x_coords, exponent)
    return u_params


def create_knot_vector(num_control_points: int, degree: int) -> np.ndarray:
    """
    Create clamped knot vector.
    
    Args:
        num_control_points: Number of control points
        degree: B-spline degree
        
    Returns:
        Clamped knot vector
    """
    n = num_control_points - 1
    p = degree
    num_interior = n - p
    
    if num_interior <= 0:
        knot_vector = np.concatenate([
            np.zeros(p + 1),
            np.ones(p + 1)
        ])
    else:
        uniform_knots = np.linspace(0.0, 1.0, num_interior + 2)[1:-1]
        knot_vector = np.concatenate([
            np.zeros(p + 1),
            uniform_knots,
            np.ones(p + 1)
        ])
    return knot_vector


def insert_knot(
    control_points: np.ndarray,
    knot_vector: np.ndarray,
    degree: int,
    new_knot: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Insert a single knot into a B-spline curve using the Boehm algorithm.
    This increases the number of control points by one and lengthens the knot vector by one.
    The shape of the curve remains unchanged.

    Args:
        control_points: Original B-spline control points (N_c, D)
        knot_vector: Original B-spline knot vector (N_k)
        degree: B-spline degree (p)
        new_knot: The new knot value to insert (u_new)

    Returns:
        A tuple containing:
        - new_control_points: Updated control points (N_c + 1, D)
        - new_knot_vector: Updated knot vector (N_k + 1)
    """
    # Number of control points and knots
    n_c = len(control_points)
    n_k = len(knot_vector)
    p = degree

    # 1. Find the knot span k such that knot_vector[k] <= new_knot < knot_vector[k+1]
    # This also handles the case where new_knot is at the end of the curve (new_knot = knot_vector[n_k - p - 1])
    # by making sure the interval is valid [knot_vector[k], knot_vector[k+1])
    k = p # Initialize k to handle clamped start
    for i in range(p + 1, n_k - p -1): # Iterate through internal knots
        if new_knot >= knot_vector[i]:
            k = i
        else:
            break
    
    # Handle the special case where new_knot is at the end (for clamped curves)
    if new_knot >= knot_vector[n_k - p - 1] - 1e-9: # Allowing a small tolerance for floating point
        k = n_k - p - 2 # The last valid span for insertion


    # 2. Create the new knot vector
    new_knot_vector = np.insert(knot_vector, k + 1, new_knot)

    # 3. Compute new control points
    new_control_points = np.zeros((n_c + 1, control_points.shape[1]))

    # Copy initial control points
    for i in range(k - p + 1):
        new_control_points[i] = control_points[i]

    # Compute new intermediate control points
    for i in range(k - p + 1, k + 1):
        alpha = (new_knot - knot_vector[i]) / (knot_vector[i + p] - knot_vector[i])
        new_control_points[i] = alpha * control_points[i] + (1 - alpha) * control_points[i - 1]
    
    # Copy remaining control points
    for i in range(k + 1, n_c + 1):
        new_control_points[i] = control_points[i - 1]

    return new_control_points, new_knot_vector


def build_basis_matrix(t_values: np.ndarray, knot_vector: np.ndarray, degree: int) -> np.ndarray:
    """
    Build B-spline basis matrix.
    
    Args:
        t_values: Parameter values
        knot_vector: B-spline knot vector
        degree: B-spline degree
        
    Returns:
        Basis matrix for B-spline fitting
    """
    num_points = len(t_values)
    num_basis = len(knot_vector) - degree - 1
    if num_points == 0 or num_basis <= 0:
        return np.zeros((num_points, max(num_basis, 0)))

    t_min = float(knot_vector[degree])
    t_max = float(knot_vector[-(degree + 1)])
    if t_max <= t_min:
        t_eval = np.full_like(t_values, t_min, dtype=float)
    else:
        eps = 1e-12 * max(1.0, abs(t_max - t_min))
        t_eval = np.clip(np.asarray(t_values, dtype=float), t_min, t_max - eps)

    try:
        design = interpolate.BSpline.design_matrix(t_eval, knot_vector, degree, extrapolate=False)
        return design.toarray()
    except Exception:
        basis_matrix = np.zeros((num_points, num_basis))
        for i in range(num_basis):
            for j, t in enumerate(t_eval):
                basis_matrix[j, i] = evaluate_basis_function(i, degree, float(t), knot_vector)
        return basis_matrix


def evaluate_basis_function(i: int, degree: int, t: float, knots: np.ndarray) -> float:
    """
    Evaluate B-spline basis function.
    
    Args:
        i: Basis function index
        degree: B-spline degree
        t: Parameter value
        knots: Knot vector
        
    Returns:
        Basis function value
    """
    t = max(knots[0], min(knots[-1] - 1e-12, t))
    
    if degree == 0:
        if i < len(knots) - 1:
            if knots[i] <= t < knots[i + 1]:
                return 1.0
            elif i == len(knots) - 2 and abs(t - knots[-1]) < 1e-12:
                return 1.0
        return 0.0
    
    result = 0.0
    
    if i + degree < len(knots) and abs(knots[i + degree] - knots[i]) > 1e-15:
        alpha1 = (t - knots[i]) / (knots[i + degree] - knots[i])
        if 0 <= i < len(knots) - degree:
            left_basis = evaluate_basis_function(i, degree - 1, t, knots)
            result += alpha1 * left_basis
    
    if i + degree + 1 < len(knots) and abs(knots[i + degree + 1] - knots[i + 1]) > 1e-15:
        alpha2 = (knots[i + degree + 1] - t) / (knots[i + degree + 1] - knots[i + 1])
        if 0 <= i + 1 < len(knots) - degree:
            right_basis = evaluate_basis_function(i + 1, degree - 1, t, knots)
            result += alpha2 * right_basis
    
    return result


def compute_curvature_at_zero(control_points: np.ndarray, knot_vector: np.ndarray, degree: int) -> float:
    """
    Compute curvature at u=0 for given control points.
    
    Args:
        control_points: B-spline control points
        knot_vector: B-spline knot vector
        degree: B-spline degree
        
    Returns:
        Curvature value at u=0
    """
    # Fast path: compute LE derivatives directly from derivative control nets.
    # Falls back to scipy derivative evaluation only if the knot structure is degenerate.
    d1, d2, _ = _compute_start_derivatives(control_points, knot_vector, degree, max_order=2)
    if d1 is None or d2 is None:
        curve = interpolate.BSpline(knot_vector, control_points, degree)
        d1 = curve.derivative(1)(0.0)
        d2 = curve.derivative(2)(0.0)
    
    # Compute curvature: κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    norm_d1 = np.linalg.norm(d1)
    
    if norm_d1 > 1e-12:
        return abs(cross) / (norm_d1 ** 3)
    else:
        return 0.0


def compute_curvature_derivative_at_zero(control_points: np.ndarray, knot_vector: np.ndarray, degree: int) -> float:
    """
    Compute the derivative of curvature with respect to the parameter u at u=0.
    
    The formula for curvature derivative is:
    κ' = [ (x'y''' - y'x''') (x'² + y'²) - 3 (x'y'' - y'x'') (x'x'' + y'y'') ] / (x'² + y'²)^(5/2)
    """
    # Fast path: direct derivative-control-net evaluation at LE.
    d1, d2, d3 = _compute_start_derivatives(control_points, knot_vector, degree, max_order=3)
    if d1 is None or d2 is None or d3 is None:
        curve = interpolate.BSpline(knot_vector, control_points, degree)
        d1 = curve.derivative(1)(0.0)
        d2 = curve.derivative(2)(0.0)
        d3 = curve.derivative(3)(0.0)
    
    x1, y1 = d1[0], d1[1]
    x2, y2 = d2[0], d2[1]
    x3, y3 = d3[0], d3[1]
    
    v2 = x1**2 + y1**2
    if v2 < 1e-12:
        return 0.0
        
    numerator = (x1*y3 - y1*x3) * v2 - 3 * (x1*y2 - y1*x2) * (x1*x2 + y1*y2)
    denominator = v2**(2.5)
    
    return numerator / denominator


def _compute_start_derivatives(
    control_points: np.ndarray,
    knot_vector: np.ndarray,
    degree: int,
    max_order: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Compute C'(u0), C''(u0), C'''(u0) via derivative control nets at the start knot.
    Returns (d1, d2, d3); entries above max_order are None.
    """
    if max_order < 1:
        return None, None, None

    cp = np.asarray(control_points, dtype=float)
    kv = np.asarray(knot_vector, dtype=float)
    d = int(degree)
    out: list[np.ndarray | None] = [None, None, None]

    try:
        for order in range(1, min(max_order, 3) + 1):
            if d <= 0 or len(cp) < 2 or len(kv) < d + 2:
                break

            new_cp = np.zeros((len(cp) - 1, cp.shape[1]), dtype=float)
            for i in range(len(new_cp)):
                denom = kv[i + d + 1] - kv[i + 1]
                if abs(denom) <= 1e-15:
                    # Degenerate local knot span; bail to fallback path.
                    return None, None, None
                new_cp[i] = d * (cp[i + 1] - cp[i]) / denom

            cp = new_cp
            kv = kv[1:-1]
            d -= 1
            out[order - 1] = cp[0].copy()
    except Exception:
        return None, None, None

    return out[0], out[1], out[2]


def compute_tangent_at_trailing_edge(control_points: np.ndarray, knot_vector: np.ndarray, degree: int) -> np.ndarray:
    """
    Compute the tangent vector at the trailing edge (u=1) of a B-spline curve.
    
    For a B-spline of degree p, the tangent at u=1 is:
    tangent = p * (P_n - P_{n-1}) / (t_{n+1} - t_{n-p+1})
    
    Args:
        control_points: Control points of the B-spline
        knot_vector: Knot vector of the B-spline
        degree: B-spline degree
        
    Returns:
        Tangent vector at the trailing edge
    """
    n = len(control_points) - 1  # Number of control points minus 1
    p = degree
    
    # Get the last two control points
    P_n = control_points[-1]
    P_n_minus_1 = control_points[-2]
    
    # Get the relevant knot values
    t_n_plus_1 = knot_vector[n + 1]
    t_n_minus_p_plus_1 = knot_vector[n - p + 1]
    
    # Compute the denominator
    denominator = t_n_plus_1 - t_n_minus_p_plus_1
    
    if abs(denominator) < 1e-12:
        # Fallback: use simple difference
        tangent = P_n - P_n_minus_1
    else:
        # Compute tangent using B-spline formula
        tangent = p * (P_n - P_n_minus_1) / denominator
    
    # Normalize the tangent vector
    norm = np.linalg.norm(tangent)
    if norm > 1e-12:
        tangent = tangent / norm
    
    return tangent


def smoothstep_quintic(u: np.ndarray) -> np.ndarray:
    """
    C2 smoothstep (quintic) function.
    
    Ensures f(0)=0, f'(0)=f''(0)=0 and f(1)=1, f'(1)=f''(1)=0
    
    Args:
        u: Input parameter values
        
    Returns:
        Smoothstep values
    """
    u = np.clip(u, 0.0, 1.0)
    return u**3 * (10 - 15*u + 6*u*u)


def sample_curve(curve: interpolate.BSpline, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample a B-spline curve densely in parameter domain.
    
    Args:
        curve: B-spline curve to sample
        num_samples: Number of samples to take
        
    Returns:
        Tuple of (parameter values, curve points)
    """
    t_start = curve.t[curve.k]
    t_end = curve.t[-(curve.k + 1)]
    t_vals = np.linspace(t_start, t_end, num_samples)
    # Avoid exact end to keep derivatives well behaved
    if len(t_vals) > 0:
        t_vals[-1] = min(t_vals[-1], t_end - 1e-12)
    pts = curve(t_vals)
    return t_vals, pts


def calculate_curvature_comb_data(upper_curve: interpolate.BSpline, lower_curve: interpolate.BSpline, 
                                 num_points_per_segment: int = 200, scale_factor: float = 0.050):
    """
    Calculate curvature comb visualization data.
    
    Args:
        upper_curve: Upper surface B-spline curve
        lower_curve: Lower surface B-spline curve
        num_points_per_segment: Number of points per curve segment
        scale_factor: Scale factor for comb visualization
        
    Returns:
        List of curvature comb data for both curves
    """
    all_curves_combs = []
    curves = [upper_curve, lower_curve]
    
    for curve in curves:
        curve_comb_hairs = []
        t_start = curve.t[curve.k]
        t_end = curve.t[-(curve.k + 1)]
        t_vals = np.linspace(t_start, t_end, num_points_per_segment)
        curve_points = curve(t_vals)
        derivatives_1 = curve.derivative(1)(t_vals)
        derivatives_2 = curve.derivative(2)(t_vals)
        
        x_prime = derivatives_1[:, 0]
        y_prime = derivatives_1[:, 1]
        x_double_prime = derivatives_2[:, 0]
        y_double_prime = derivatives_2[:, 1]
        
        numerator = x_prime * y_double_prime - y_prime * x_double_prime
        denominator = (x_prime**2 + y_prime**2)**(3/2)
        curvatures = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)
        
        tangent_norms = np.sqrt(x_prime**2 + y_prime**2)
        unit_tangents = np.zeros_like(derivatives_1)
        valid_tangents = tangent_norms > 1e-12
        unit_tangents[valid_tangents] = derivatives_1[valid_tangents] / tangent_norms[valid_tangents, np.newaxis]
        
        normals = np.zeros_like(unit_tangents)
        normals[:, 0] = -unit_tangents[:, 1]
        normals[:, 1] = unit_tangents[:, 0]
        
        comb_lengths = -curvatures * scale_factor
        end_points = curve_points + normals * comb_lengths[:, np.newaxis]
        
        for j in range(num_points_per_segment):
            hair_segment = np.array([curve_points[j], end_points[j]])
            curve_comb_hairs.append(hair_segment)
        
        all_curves_combs.append(curve_comb_hairs)
    
    return all_curves_combs if all_curves_combs else None
