import numpy as np
from scipy.interpolate import CubicSpline

def _remove_consecutive_duplicates(points: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    if len(points) == 0:
        return points
    diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate(([True], diffs > tol))
    return points[keep]

def _compute_arc_length(points: np.ndarray) -> np.ndarray:
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    s = np.zeros(len(points), dtype=float)
    s[1:] = np.cumsum(seg_lengths)
    return s

def _build_contour_from_surfaces(upper_surface: np.ndarray, lower_surface: np.ndarray) -> np.ndarray:
    upper_te_to_le = np.flipud(upper_surface)
    if len(lower_surface) > 0 and np.allclose(upper_te_to_le[-1], lower_surface[0]):
        lower_part = lower_surface[1:]
    else:
        lower_part = lower_surface
    return np.vstack([upper_te_to_le, lower_part])

def _find_real_le_on_spline(contour: np.ndarray, logger_func=print) -> tuple[np.ndarray, int]:
    contour = _remove_consecutive_duplicates(contour)
    if len(contour) < 4:
        raise ValueError("Not enough points to compute cubic spline for leading edge detection.")

    s = _compute_arc_length(contour)
    if not np.all(np.diff(s) > 0):
        raise ValueError("Arc-length parameter is not strictly increasing; check for duplicate points.")

    x = contour[:, 0]
    y = contour[:, 1]
    x_spline = CubicSpline(s, x)
    y_spline = CubicSpline(s, y)

    x_te = 0.5 * (x[0] + x[-1])
    y_te = 0.5 * (y[0] + y[-1])

    sle = None
    for i in range(2, len(contour) - 2):
        dxte = x[i] - x_te
        dyte = y[i] - y_te
        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]
        dotp = dxte * dx + dyte * dy
        if dotp < 0.0:
            sle = s[i]
            break
    if sle is None:
        sle = s[int(np.argmin(x))]

    ds_eps = (s[-1] - s[0]) * 1.0e-10
    for _ in range(50):
        x_le = x_spline(sle)
        y_le = y_spline(sle)
        dxds = x_spline(sle, 1)
        dyds = y_spline(sle, 1)
        dxdd = x_spline(sle, 2)
        dydd = y_spline(sle, 2)

        x_chord = x_le - x_te
        y_chord = y_le - y_te
        res = x_chord * dxds + y_chord * dyds
        ress = dxds * dxds + dyds * dyds + x_chord * dxdd + y_chord * dydd
        if np.isclose(ress, 0.0):
            break
        dsle = -res / ress
        max_step = 0.02 * max(abs(x_chord) + abs(y_chord), 1.0e-12)
        dsle = np.clip(dsle, -max_step, max_step)
        sle = np.clip(sle + dsle, s[0], s[-1])
        if abs(dsle) < ds_eps:
            break
    else:
        logger_func("Warning: LE spline search did not converge; using last estimate.")

    x_le = float(x_spline(sle))
    y_le = float(y_spline(sle))
    le_point = np.array([x_le, y_le], dtype=float)

    insert_idx = int(np.searchsorted(s, sle))
    if insert_idx < len(s) and abs(s[insert_idx] - sle) < ds_eps:
        contour[insert_idx] = le_point
        le_index = insert_idx
        return contour, le_index

    # Avoid adding a point if the real LE is already represented in the data.
    bbox = np.ptp(contour, axis=0)
    scale = float(max(bbox[0], bbox[1], 1.0))
    min_dist_tol = max(1.0e-7 * scale, 1.0e-12)
    dists = np.linalg.norm(contour - le_point, axis=1)
    closest_idx = int(np.argmin(dists))
    if dists[closest_idx] <= min_dist_tol:
        logger_func(
            "Real LE matches existing point within tolerance; no point inserted. "
            f"Closest idx {closest_idx}, dist {dists[closest_idx]:.3e}."
        )
        return contour, closest_idx

    contour = np.insert(contour, insert_idx, le_point, axis=0)
    logger_func(
        "Inserted real LE point from spline definition. "
        f"Index {insert_idx}, point ({x_le:.8f}, {y_le:.8f})."
    )
    le_index = insert_idx
    return contour, le_index

def _split_contour_at_le(contour: np.ndarray, le_index: int) -> tuple[np.ndarray, np.ndarray]:
    if le_index <= 0 or le_index >= len(contour) - 1:
        raise ValueError("Leading edge index is at contour boundary; input may be malformed.")
    upper_surface = np.flipud(contour[:le_index + 1])
    lower_surface = contour[le_index:]
    return upper_surface, lower_surface

def prepare_surfaces_with_real_le(upper_surface: np.ndarray, lower_surface: np.ndarray, logger_func=print) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds a closed contour from upper/lower surfaces, finds the real leading edge
    using the spline-normal definition, inserts that point, and splits back into
    LE->TE ordered surfaces.
    """
    contour = _build_contour_from_surfaces(upper_surface, lower_surface)
    contour, le_index = _find_real_le_on_spline(contour, logger_func)
    return _split_contour_at_le(contour, le_index)

def prepare_surfaces_from_selig_contour(all_coords: np.ndarray, logger_func=print) -> tuple[np.ndarray, np.ndarray]:
    """
    Takes a Selig-style contour (TE upper -> around LE -> TE lower),
    finds/inserts the real leading edge, and splits into LE->TE surfaces.
    """
    contour, le_index = _find_real_le_on_spline(all_coords, logger_func)
    return _split_contour_at_le(contour, le_index)

def normalize_airfoil_data(upper_surface, lower_surface, logger_func=print, real_le_prepared: bool = False):
    """
    Normalizes airfoil coordinates to have a chord length of 1, with the
    leading edge at (0,0) and the trailing edge chord-line at (1,0) if appropriate.

    This process preserves the shape of the airfoil by performing translation,
    rotation, and scaling. For thick trailing edges, y-normalization is only performed
    if the TE y-values are not symmetric about y=0.

    Args:
        upper_surface (np.ndarray): Original upper surface coordinates (N, 2), ordered LE to TE.
        lower_surface (np.ndarray): Original lower surface coordinates (N, 2), ordered LE to TE.
        logger_func (callable, optional): A function for logging messages.

    Returns:
        tuple: (normalized_upper_surface, normalized_lower_surface)
    """
    # Ensure LE is the real spline-defined LE and surfaces are ordered LE->TE.
    if not real_le_prepared:
        upper_surface, lower_surface = prepare_surfaces_with_real_le(upper_surface, lower_surface, logger_func)

    le_point = upper_surface[0].copy()
    te_upper = upper_surface[-1]
    te_lower = lower_surface[-1]
    te_point = (te_upper + te_lower) / 2.0
    
    logger_func(f"Normalizing airfoil. Original LE: {le_point}, Original TE (midpoint): {te_point}")

    # Translate so LE is at the origin (0,0)
    upper_translated = upper_surface - le_point
    lower_translated = lower_surface - le_point
    te_translated = te_point - le_point

    # Rotate so the TE point lies on the positive x-axis.
    if np.allclose(te_translated, [0.0, 0.0]):
        logger_func("Warning: Leading and trailing edges are coincident. Cannot determine rotation.")
        rotation_angle = 0.0
    else:
        rotation_angle = -np.arctan2(te_translated[1], te_translated[0])

    if abs(rotation_angle) > 0:
        rotation_matrix = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle),  np.cos(rotation_angle)]
        ])
        upper_rotated = upper_translated @ rotation_matrix.T
        lower_rotated = lower_translated @ rotation_matrix.T
        te_rotated = te_translated @ rotation_matrix.T
    else:
        upper_rotated = upper_translated
        lower_rotated = lower_translated
        te_rotated = te_translated
    
    chord_length = te_rotated[0]

    if np.isclose(chord_length, 0):
        raise ValueError("Cannot normalize airfoil with zero chord length.")
    
    logger_func(f"Detected chord length: {chord_length:.6f}, rotating by {np.rad2deg(rotation_angle):.4f} degrees.")

    upper_normalized = upper_rotated / chord_length
    lower_normalized = lower_rotated / chord_length

    # --- Trailing edge y-normalization logic ---
    y_te_upper = upper_normalized[-1, 1]
    y_te_lower = lower_normalized[-1, 1]
    tol = 1e-8
    if np.isclose(y_te_upper, y_te_lower, atol=tol):
        # Not thickened, proceed as before: shift so TE is at y=0
        y_shift = (y_te_upper + y_te_lower) / 2.0
        logger_func(f"TE y-values equal (not thickened). Shifting y by {-y_shift:.8f}.")
        upper_normalized[:, 1] -= y_shift
        lower_normalized[:, 1] -= y_shift
    elif np.isclose(y_te_upper, -y_te_lower, atol=tol):
        # Thickened and symmetric, do not shift y
        logger_func("TE y-values are symmetric about y=0 (thickened trailing edge). No y-shift performed.")
    else:
        # Not symmetric, shift so TE midpoint is at y=0
        y_shift = (y_te_upper + y_te_lower) / 2.0
        logger_func(f"TE y-values not symmetric. Shifting y by {-y_shift:.8f}.")
        upper_normalized[:, 1] -= y_shift
        lower_normalized[:, 1] -= y_shift

    return upper_normalized, lower_normalized

def load_airfoil_data(filename, logger_func=print):
    """
    Loads airfoil coordinates from a file, supporting Selig and Lednicer .dat formats.

    Selig format:
        Line 1: Airfoil name
        Subsequent lines: x y coordinates, starting from upper TE, around LE, to lower TE.

    Lednicer format:
        Line 1: Airfoil name
        Line 2: NUM_UPPER NUM_LOWER (number of points on upper and lower surfaces) [optional]
        Subsequent lines: Upper surface points (x y), LE to TE.
        Then: Lower surface points (x y), LE to TE.

    Args:
        filename (str): Path to the airfoil data file.
        logger_func (callable, optional): A function (like print or a signal's emit method) to send log messages to.
                                          Defaults to `print`.

    Returns:
        tuple: (upper_surface_coords, lower_surface_coords, airfoil_name, thickened)
               Each is a NumPy array (N, 2) with points ordered LE to TE.
               'thickened' is True if the trailing edge is thickened and symmetric about y=0.
    Raises:
        ValueError: If the file is empty or malformed.
    """
    with open(filename, 'r') as f:
        # Ignore blank lines and comment lines starting with '#'.
        lines = [
            line.strip()
            for line in f.readlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    if not lines:
        raise ValueError(f"Airfoil data file '{filename}' is empty or contains no valid data.")

    airfoil_name = lines[0]
    coords_start_line = 1

    is_lednicer = False
    num_upper_points = 0
    num_lower_points = 0

    # 1. Try Lednicer with count line (both numbers close to int and > 1)
    if len(lines) > 1:
        parts = lines[1].split()
        if len(parts) == 2:
            try:
                n1 = float(parts[0])
                n2 = float(parts[1])
                is_n1_int = abs(n1 - int(round(n1))) < 1e-6 and n1 > 1
                is_n2_int = abs(n2 - int(round(n2))) < 1e-6 and n2 > 1
                if is_n1_int and is_n2_int:
                    num_upper_points = int(round(n1))
                    num_lower_points = int(round(n2))
                    total_points = num_upper_points + num_lower_points
                    if abs(len(lines) - (2 + total_points)) <= 1:
                        is_lednicer = True
                        coords_start_line = 2
            except ValueError:
                pass # Not numbers, so check slope below

    # 2. Slope-based detection if not already Lednicer
    if not is_lednicer:
        # Try to parse the next 5 lines as coordinates (skip count line if present)
        slope_start = coords_start_line
        sample_coords = []
        for i in range(slope_start, min(slope_start + 5, len(lines))):
            try:
                x, y = map(float, lines[i].split())
                sample_coords.append(x)
            except Exception:
                break
        if len(sample_coords) >= 3:
            # Only check if x goes up or down from first to last sample
            if sample_coords[-1] > sample_coords[0]: #and abs(sample_coords[0]) < 1e-3 and abs(sample_coords[-1] - 1.0) < 1e-2:
                is_lednicer = True
                logger_func(f"Detected Lednicer format for '{airfoil_name}' (by x slope).")
            elif sample_coords[-1] < sample_coords[0]: # and abs(sample_coords[0] - 1.0) < 1e-2 and abs(sample_coords[-1]) < 1e-3:
                is_lednicer = False
                logger_func(f"Detected Selig-like format for '{airfoil_name}' (by x slope).")
            else:
                logger_func(f"Ambiguous x slope for '{airfoil_name}'. Defaulting to Selig-like format.")
                is_lednicer = False
        else:
            logger_func(f"Insufficient coordinate data to determine format for '{airfoil_name}'. Defaulting to Selig-like format.")
            is_lednicer = False

    if is_lednicer:
        if coords_start_line == 2:
            logger_func(f"Detected Lednicer format for '{airfoil_name}' (by count line). Expected upper points: {num_upper_points}, lower points: {num_lower_points}.")
            all_coords_raw = np.array([list(map(float, line.split())) for line in lines[coords_start_line:]])
            if len(all_coords_raw) != (num_upper_points + num_lower_points):
                logger_func(f"Warning: Declared points ({num_upper_points + num_lower_points}) do not match actual points read ({len(all_coords_raw)}). Attempting to proceed.")
            upper_surface = all_coords_raw[:num_upper_points]
            lower_surface = all_coords_raw[num_upper_points:]
        else:
            # Lednicer without count line: split at midpoint
            logger_func(f"Detected Lednicer format for '{airfoil_name}' (by x slope, no count line).")
            all_coords_raw = np.array([list(map(float, line.split())) for line in lines[coords_start_line:]])
            midpoint = len(all_coords_raw) // 2
            upper_surface = all_coords_raw[:midpoint]
            lower_surface = all_coords_raw[midpoint:]
        # Basic validation: check if LE point is consistent
        if not (np.allclose(upper_surface[0], lower_surface[0]) and np.allclose(upper_surface[0], [0.0, 0.0])):
            logger_func("Warning: Leading edge points are not consistent or not at (0,0) in Lednicer format. Data may need normalization.")
    else:
        logger_func(f"Detected Selig-like format for '{airfoil_name}'.")
        coords_str = lines[coords_start_line:]
        all_coords = np.array([list(map(float, line.split())) for line in coords_str if line])
        upper_surface, lower_surface = prepare_surfaces_from_selig_contour(all_coords, logger_func)
    # Always normalize the data to ensure consistency.
    upper_surface, lower_surface = normalize_airfoil_data(
        upper_surface,
        lower_surface,
        logger_func,
        real_le_prepared=is_lednicer is False,
    )

    # Detect thickened trailing edge (symmetric about y=0)
    y_te_upper = upper_surface[-1, 1]
    y_te_lower = lower_surface[-1, 1]
    tol = 1e-8
    thickened = False
    if not np.isclose(y_te_upper, y_te_lower, atol=tol) and np.isclose(y_te_upper, -y_te_lower, atol=tol):
        thickened = True
        logger_func("Trailing edge is thickened and symmetric about y=0.")

    return upper_surface, lower_surface, airfoil_name, thickened

def find_shoulder_x_coords(upper_data, lower_data):
    """
    Identifies the x-coordinates for the upper and lower shared vertices (shoulder points)
    based on maximum/minimum y-values in the original airfoil data.

    Args:
        upper_data (np.ndarray): Coordinates of the upper surface (N, 2).
        lower_data (np.ndarray): Coordinates of the lower surface (N, 2).

    Returns:
        tuple: (upper_shoulder_x, lower_shoulder_x)
    """
    # Find the x-coordinate corresponding to the maximum y-value on the upper surface
    upper_max_y_idx = np.argmax(upper_data[:, 1])
    upper_shoulder_x = upper_data[upper_max_y_idx, 0]

    # Find the x-coordinate corresponding to the minimum y-value on the lower surface
    lower_min_y_idx = np.argmin(lower_data[:, 1])
    lower_shoulder_x = lower_data[lower_min_y_idx, 0]

    return upper_shoulder_x, lower_shoulder_x

def export_airfoil_to_selig_format(upper_surface, lower_surface, airfoil_name, filename):
    """
    Export airfoil data in Selig format to a .dat file.
    
    Selig format:
        Line 1: Airfoil name
        Subsequent lines: x y coordinates, starting from upper TE, around LE, to lower TE.
    
    Args:
        upper_surface (np.ndarray): Upper surface coordinates (N, 2), ordered LE to TE.
        lower_surface (np.ndarray): Lower surface coordinates (N, 2), ordered LE to TE.
        airfoil_name (str): Name of the airfoil.
        filename (str): Output file path.
    """
    with open(filename, 'w') as f:
        # Write airfoil name
        f.write(f"{airfoil_name}\n")
        
        # Write coordinates in Selig format: upper TE -> LE -> lower TE
        # Upper surface from TE to LE (reverse order)
        for i in range(len(upper_surface) - 1, -1, -1):
            x, y = upper_surface[i]
            f.write(f"{x:.6f} {y:.6f}\n")
        
        # Lower surface from LE to TE (skip first point as it's the same as upper LE)
        for i in range(1, len(lower_surface)):
            x, y = lower_surface[i]
            f.write(f"{x:.6f} {y:.6f}\n")
