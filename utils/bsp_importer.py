"""Import B-spline control points and knots from .bsp format."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class BSPModelData:
    airfoil_name: str
    upper_control_points: np.ndarray
    lower_control_points: np.ndarray
    upper_knots: np.ndarray
    lower_knots: np.ndarray
    upper_degree: int
    lower_degree: int
def _parse_surface_json(payload: object, *, surface_name: str) -> tuple[np.ndarray, np.ndarray, int]:
    if not isinstance(payload, dict):
        raise ValueError(f"Surface '{surface_name}' must be a JSON object.")

    px = payload.get("px")
    py = payload.get("py")
    knots = payload.get("knots")
    degree = payload.get("degree")

    if not isinstance(px, list) or not isinstance(py, list) or not isinstance(knots, list):
        raise ValueError(f"Surface '{surface_name}' must contain list fields 'px', 'py', and 'knots'.")
    if len(px) != len(py):
        raise ValueError(f"Surface '{surface_name}' has mismatched 'px'/'py' lengths.")
    if len(px) == 0:
        raise ValueError(f"Surface '{surface_name}' must contain at least one control point.")

    try:
        control_points = np.column_stack(
            [
                np.asarray(px, dtype=float),
                np.asarray(py, dtype=float),
            ]
        )
        knot_vector = np.asarray(knots, dtype=float)
        degree_value = int(degree)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Surface '{surface_name}' contains invalid numeric values.") from exc

    expected_knot_count = len(control_points) + degree_value + 1
    if degree_value < 1:
        raise ValueError(f"Surface '{surface_name}' has invalid degree {degree_value}.")
    if len(knot_vector) != expected_knot_count:
        raise ValueError(
            f"Surface '{surface_name}' has {len(knot_vector)} knots, expected {expected_knot_count} "
            f"for {len(control_points)} control points and degree {degree_value}."
        )

    return control_points, knot_vector, degree_value


def _load_bspline_from_json(path: Path, payload: object) -> BSPModelData:
    if not isinstance(payload, dict):
        raise ValueError("BSP JSON root must be an object.")

    airfoil_name = str(payload.get("name") or path.stem)
    upper_cp, upper_knots, upper_degree = _parse_surface_json(payload.get("upper"), surface_name="upper")
    lower_cp, lower_knots, lower_degree = _parse_surface_json(payload.get("lower"), surface_name="lower")

    return BSPModelData(
        airfoil_name=airfoil_name,
        upper_control_points=upper_cp,
        lower_control_points=lower_cp,
        upper_knots=upper_knots,
        lower_knots=lower_knots,
        upper_degree=upper_degree,
        lower_degree=lower_degree,
    )


def load_bspline_from_bsp(file_path: str | Path) -> BSPModelData:
    """Parse a JSON-based .bsp file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"BSP file not found: {path}")

    raw_text = path.read_text(encoding="utf-8")
    payload = json.loads(raw_text)
    return _load_bspline_from_json(path, payload)
