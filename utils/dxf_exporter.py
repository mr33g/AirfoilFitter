import traceback

import ezdxf
import numpy as np
from scipy import interpolate

DXF_EXPORT_MODE_NURBS = "nurbs"
DXF_EXPORT_MODE_BEZIER = "bezier"


def _get_degree_pair(bspline_processor):
    """Return (upper_degree, lower_degree) from processor state."""
    fitted_degree = getattr(bspline_processor, "fitted_degree", None)
    if isinstance(fitted_degree, tuple):
        return int(fitted_degree[0]), int(fitted_degree[1])
    degree = fitted_degree if fitted_degree is not None else getattr(bspline_processor, "degree", 3)
    return int(degree), int(degree)


def _decompose_bspline_to_bezier_segments(control_points, knot_vector, degree):
    """
    Decompose one clamped B-spline into Bezier control polygons.

    Returns:
        list[np.ndarray]: Each element has shape ((degree + 1), dim).
    """
    if degree < 1:
        raise ValueError("Bezier decomposition requires degree >= 1.")

    ctrl = np.asarray(control_points, dtype=float)
    knots = np.asarray(knot_vector, dtype=float)
    if ctrl.ndim != 2:
        raise ValueError("Control points must be a 2D array.")
    if knots.ndim != 1:
        raise ValueError("Knot vector must be a 1D array.")

    spline = interpolate.BSpline(knots, ctrl, degree)

    unique_knots, counts = np.unique(knots, return_counts=True)
    if unique_knots.size < 2:
        return []

    start = unique_knots[0]
    end = unique_knots[-1]
    for knot, count in zip(unique_knots, counts):
        if np.isclose(knot, start) or np.isclose(knot, end):
            continue
        missing_mult = degree - int(count)
        if missing_mult > 0:
            spline = spline.insert_knot(float(knot), m=missing_mult)

    full_knots = np.asarray(spline.t, dtype=float)
    full_ctrl = np.asarray(spline.c, dtype=float)
    k = int(spline.k)

    segments = []
    n = len(full_ctrl) - 1
    for i in range(k, n + 1):
        if i + 1 >= len(full_knots):
            break
        if full_knots[i + 1] <= full_knots[i]:
            continue
        segment_ctrl = full_ctrl[i - k : i + 1]
        if len(segment_ctrl) == k + 1:
            segments.append(segment_ctrl.copy())
    return segments


def _add_nurbs_surface(msp, points, degree, knots, layer, color, logger_func, name):
    spline = msp.add_open_spline(control_points=points, degree=degree, knots=knots)
    spline.dxf.layer = layer
    spline.dxf.color = color
    logger_func(f"  {name}: degree {degree} B-spline with {len(points)} control points")


def _add_bezier_surface(msp, control_points, knot_vector, degree, layer, color, logger_func, name):
    segments = _decompose_bspline_to_bezier_segments(control_points, knot_vector, degree)
    if not segments:
        raise ValueError(f"No Bezier segments generated for {name}.")

    bezier_knots = [0.0] * (degree + 1) + [1.0] * (degree + 1)
    for segment in segments:
        spline = msp.add_open_spline(
            control_points=[tuple(pt.tolist()) for pt in segment],
            degree=degree,
            knots=bezier_knots,
        )
        spline.dxf.layer = layer
        spline.dxf.color = color
    logger_func(f"  {name}: exported {len(segments)} Bezier segment(s), degree {degree}")


def export_bspline_to_dxf(bspline_processor, chord_length_mm, logger_func, export_mode=DXF_EXPORT_MODE_NURBS):
    """
    Export B-spline curves to DXF as NURBS curves.
    
    Args:
        bspline_processor: BSplineProcessor instance with fitted B-spline data
        chord_length_mm (float): The desired chord length in millimeters for scaling
        logger_func (callable): A function to send log messages to
        
    Returns:
        ezdxf.document.Drawing: The created DXF document object, or None if an error occurred
    """
    try:
        if not bspline_processor.fitted:
            logger_func("Error: No B-spline fit available for DXF export.")
            return None
            
        if chord_length_mm <= 0:
            logger_func("Error: Chord length must be positive for DXF export.")
            return None

        mode = str(export_mode).strip().lower()
        if mode not in {DXF_EXPORT_MODE_NURBS, DXF_EXPORT_MODE_BEZIER}:
            logger_func(f"Warning: Unknown DXF export mode '{export_mode}', falling back to NURBS.")
            mode = DXF_EXPORT_MODE_NURBS

        logger_func(
            f"Preparing B-spline DXF export with chord length: {chord_length_mm:.2f} mm "
            f"(mode: {mode})..."
        )

        # Get B-spline control points and knot vectors
        upper_ctrl_pts = bspline_processor.upper_control_points
        lower_ctrl_pts = bspline_processor.lower_control_points
        upper_knots = bspline_processor.upper_knot_vector
        lower_knots = bspline_processor.lower_knot_vector
        
        if upper_ctrl_pts is None or lower_ctrl_pts is None:
            logger_func("Error: B-spline control points not available for DXF export.")
            return None
        
        if upper_knots is None or lower_knots is None:
            logger_func("Error: B-spline knot vectors not available for DXF export.")
            return None
        
        # Scale control points by chord length
        upper_ctrl_pts_scaled = upper_ctrl_pts * chord_length_mm
        lower_ctrl_pts_scaled = lower_ctrl_pts * chord_length_mm
        
        # Create DXF document
        doc = ezdxf.new('R2000')
        doc.header["$INSUNITS"] = 4  # millimeters
        msp = doc.modelspace()
        
        # Convert control points to format expected by ezdxf
        upper_points = [tuple(pt.tolist()) for pt in upper_ctrl_pts_scaled]
        lower_points = [tuple(pt.tolist()) for pt in lower_ctrl_pts_scaled]
        
        # Convert knot vectors to lists (ezdxf expects a list)
        upper_knots_list = upper_knots.tolist()
        lower_knots_list = lower_knots.tolist()
        
        degree_upper, degree_lower = _get_degree_pair(bspline_processor)
        logger_func(f"Creating curves: degrees U={degree_upper}, L={degree_lower}")
        logger_func(f"  Upper knot vector: {len(upper_knots_list)} knots")
        logger_func(f"  Lower knot vector: {len(lower_knots_list)} knots")

        if mode == DXF_EXPORT_MODE_BEZIER:
            _add_bezier_surface(
                msp,
                upper_ctrl_pts_scaled,
                upper_knots,
                degree_upper,
                "AIRFOIL_UPPER",
                1,
                logger_func,
                "Upper surface",
            )
            _add_bezier_surface(
                msp,
                lower_ctrl_pts_scaled,
                lower_knots,
                degree_lower,
                "AIRFOIL_LOWER",
                5,
                logger_func,
                "Lower surface",
            )
        else:
            _add_nurbs_surface(
                msp,
                upper_points,
                degree_upper,
                upper_knots_list,
                "AIRFOIL_UPPER",
                1,
                logger_func,
                "Upper surface",
            )
            _add_nurbs_surface(
                msp,
                lower_points,
                degree_lower,
                lower_knots_list,
                "AIRFOIL_LOWER",
                5,
                logger_func,
                "Lower surface",
            )
        
        # Add trailing edge connector if needed (for blunt trailing edge)
        if not bspline_processor.is_sharp_te:
            if not np.allclose(upper_ctrl_pts_scaled[-1], lower_ctrl_pts_scaled[-1]):
                msp.add_line(
                    tuple(upper_ctrl_pts_scaled[-1].tolist()), 
                    tuple(lower_ctrl_pts_scaled[-1].tolist()),
                    dxfattribs={'layer': 'TRAILING_EDGE_CONNECTOR', 'color': 2}  # Yellow
                )
                logger_func("  Added trailing edge connector for blunt trailing edge")
        
        logger_func("B-spline DXF export completed successfully.")
        return doc
        
    except Exception as e:
        logger_func(f"Error during B-spline DXF export: {e}")
        logger_func(traceback.format_exc())
        return None
