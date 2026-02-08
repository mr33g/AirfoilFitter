"""Export B-spline control points and knots to .bsp format (AirfoilEditor-compatible)."""

from __future__ import annotations

from typing import Any


def export_bspline_to_bsp(
    bspline_processor: Any,
    airfoil_name: str,
    file_path: str,
    logger_func=print,
) -> bool:
    """Write a .bsp file using the AirfoilEditor sectioned format."""
    if bspline_processor is None:
        logger_func("Error: No B-spline processor available for BSP export.")
        return False

    upper_cp = getattr(bspline_processor, "upper_control_points", None)
    lower_cp = getattr(bspline_processor, "lower_control_points", None)
    upper_knots = getattr(bspline_processor, "upper_knot_vector", None)
    lower_knots = getattr(bspline_processor, "lower_knot_vector", None)

    if upper_cp is None or lower_cp is None or upper_knots is None or lower_knots is None:
        logger_func("Error: B-spline control points or knot vectors not available for BSP export.")
        return False

    try:
        with open(file_path, "w+", encoding="utf-8") as file:
            file.write(f"{airfoil_name}\n")

            file.write("Top Start\n")
            for p in upper_cp:
                file.write("%13.10f %13.10f\n" % (float(p[0]), float(p[1])))
            file.write("Top End\n")

            file.write("Top Knots Start\n")
            for k in upper_knots:
                file.write("%13.10f\n" % float(k))
            file.write("Top Knots End\n")

            file.write("Bottom Start\n")
            for p in lower_cp:
                file.write("%13.10f %13.10f\n" % (float(p[0]), float(p[1])))
            file.write("Bottom End\n")

            file.write("Bottom Knots Start\n")
            for k in lower_knots:
                file.write("%13.10f\n" % float(k))
            file.write("Bottom Knots End\n")

        return True
    except OSError as exc:
        logger_func(f"Error writing BSP file: {exc}")
        return False
