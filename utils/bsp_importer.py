"""Import B-spline control points and knots from .bsp format."""

from __future__ import annotations

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


def _parse_float_list(lines: list[str], *, expected_cols: int, section_name: str) -> np.ndarray:
    values: list[list[float]] = []
    for idx, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != expected_cols:
            raise ValueError(
                f"Section '{section_name}' line {idx} has {len(parts)} columns, expected {expected_cols}."
            )
        try:
            row = [float(part) for part in parts]
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value in section '{section_name}', line {idx}: {line}") from exc
        values.append(row)

    if not values:
        raise ValueError(f"Section '{section_name}' is empty.")
    return np.asarray(values, dtype=float)


def _slice_section(lines: list[str], start_label: str, end_label: str) -> list[str]:
    try:
        start_idx = lines.index(start_label)
        end_idx = lines.index(end_label, start_idx + 1)
    except ValueError as exc:
        raise ValueError(f"Missing section markers '{start_label}'/'{end_label}'.") from exc
    if end_idx <= start_idx + 1:
        return []
    return lines[start_idx + 1 : end_idx]


def load_bspline_from_bsp(file_path: str | Path) -> BSPModelData:
    """Parse an AirfoilEditor-compatible .bsp file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"BSP file not found: {path}")

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    lines = [line.strip() for line in raw_lines if line.strip()]
    if len(lines) < 9:
        raise ValueError("BSP file is too short or malformed.")

    airfoil_name = lines[0]

    upper_cp_lines = _slice_section(lines, "Top Start", "Top End")
    upper_knot_lines = _slice_section(lines, "Top Knots Start", "Top Knots End")
    lower_cp_lines = _slice_section(lines, "Bottom Start", "Bottom End")
    lower_knot_lines = _slice_section(lines, "Bottom Knots Start", "Bottom Knots End")

    upper_cp = _parse_float_list(upper_cp_lines, expected_cols=2, section_name="Top")
    lower_cp = _parse_float_list(lower_cp_lines, expected_cols=2, section_name="Bottom")
    upper_knots = _parse_float_list(upper_knot_lines, expected_cols=1, section_name="Top Knots").reshape(-1)
    lower_knots = _parse_float_list(lower_knot_lines, expected_cols=1, section_name="Bottom Knots").reshape(-1)

    return BSPModelData(
        airfoil_name=airfoil_name,
        upper_control_points=upper_cp,
        lower_control_points=lower_cp,
        upper_knots=upper_knots,
        lower_knots=lower_knots,
    )
