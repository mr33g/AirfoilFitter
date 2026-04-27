"""Export B-spline control points and knots to .bsp format."""

from __future__ import annotations

import json
from typing import Any


def _surface_payload(control_points: Any, knot_vector: Any, degree: int) -> dict[str, object]:
    return {
        "px": [float(point[0]) for point in control_points],
        "py": [float(point[1]) for point in control_points],
        "knots": [float(knot) for knot in knot_vector],
        "degree": int(degree),
    }


def export_bspline_to_bsp(
    bspline_processor: Any,
    airfoil_name: str,
    file_path: str,
    logger_func=print,
) -> bool:
    """Write a .bsp file using the JSON surface format."""
    if bspline_processor is None:
        logger_func("Error: No B-spline processor available for BSP export.")
        return False

    upper_cp = getattr(bspline_processor, "upper_control_points", None)
    lower_cp = getattr(bspline_processor, "lower_control_points", None)
    upper_knots = getattr(bspline_processor, "upper_knot_vector", None)
    lower_knots = getattr(bspline_processor, "lower_knot_vector", None)
    degree_upper = int(getattr(bspline_processor, "degree_upper", getattr(bspline_processor, "degree", 0)))
    degree_lower = int(getattr(bspline_processor, "degree_lower", getattr(bspline_processor, "degree", 0)))

    if upper_cp is None or lower_cp is None or upper_knots is None or lower_knots is None:
        logger_func("Error: B-spline control points or knot vectors not available for BSP export.")
        return False

    if degree_upper < 1 or degree_lower < 1:
        logger_func("Error: Invalid B-spline degree; cannot export BSP file.")
        return False

    payload = {
        "name": airfoil_name,
        "upper": _surface_payload(upper_cp, upper_knots, degree_upper),
        "lower": _surface_payload(lower_cp, lower_knots, degree_lower),
    }

    try:
        with open(file_path, "w+", encoding="utf-8") as file:
            json.dump(payload, file, indent=4)
            file.write("\n")

        return True
    except OSError as exc:
        logger_func(f"Error writing BSP file: {exc}")
        return False
