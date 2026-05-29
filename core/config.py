"""Central project configuration constants.

This module gathers default numeric parameters and tunable hyper-parameters
used across the airfoil processing library so they live in one place.
Import these values instead of hard-coding magic numbers inside
algorithms or UI widgets.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

# Debugging & Logging
DEBUG_WORKER_LOGGING: bool = True

# B-spline settings
DEFAULT_BSPLINE_DEGREE: int = 4  # Degree of B-spline curves (3-7 recommended for airfoils)
DEFAULT_BSPLINE_CP: int = 9   # Initial number of control points per surface (must be >= degree + 1)
DEFAULT_SMOOTHNESS_PENALTY: float = 0.0  # Weight for control point smoothing penalty (higher = smoother, lower = more accurate)

# ---- Manufacturing / Export defaults -------------------------------------
DEFAULT_CHORD_LENGTH_MM: float = 200.0
DEFAULT_TE_THICKNESS_MM: float = 0.0
ENABLE_BSP_EXPORT: bool = True
ENABLE_DAT_EXPORT: bool = True
ENABLE_DXF_BEZIER_EXPORT: bool = False
MIN_CP_NEIGHBOR_DISTANCE: float = 1.0e-3

# ---- Sampling & Debugging -----------------------------------------------
NUM_POINTS_CURVE_ERROR: int = 35000
FIT_ERROR_OBJECTIVE: str = "msr"
KNOT_INSERTION_STRATEGY: str = "midspan"  # "adaptive" or "midspan"

# Input repaneling samples normalized fit data from the temporary cubic spline
# that is already used for real-leading-edge detection. The defaults mirror
# AirfoilEditor's matching target repaneling: 100 total panels, LE bunch 0.85,
# TE bunch 0.3.
ENABLE_INPUT_REPANELING: bool = True
INPUT_REPANEL_POINTS_PER_SURFACE: int = 51  # <= 0 keeps the original side counts
INPUT_REPANEL_LE_BUNCH: float = 0.85
INPUT_REPANEL_TE_BUNCH: float = 0.3

# Plot sampling settings
# Curvature-adaptive sampling improves visual smoothness near the leading edge
# while keeping performance reasonable.
PLOT_POINTS_PER_SURFACE: int = 500
PLOT_CURVATURE_WEIGHT: float = 0.85  # 0 = uniform, 1 = fully curvature-driven
SHOW_CP_FOURTH_DIFFERENCES: bool = True
CP_FOURTH_DIFF_MAX_PLOT_LENGTH: float = 0.040  # chord fraction for largest displayed D4 vector

# Curvature comb UI ranges
# Old max density (100) becomes the new minimum. Allow much denser combs.
COMB_DENSITY_MIN: int = 100
COMB_DENSITY_MAX: int = 1000
COMB_DENSITY_DEFAULT: int = 200
COMB_SCALE_DEFAULT: float = 0.020  # Reduced from 0.050 for better initial viewport fit


# Number of points used for trailing edge vector calculations
# Higher numbers provide more robust tangent estimates but may be less sensitive to local geometry
DEFAULT_TE_VECTOR_POINTS: int = 2

# Soft TE handle quality penalty. It gently prefers the last free control point
# to form a reasonable handle direction and length with the trailing edge.
ENABLE_SOFT_TE_HANDLE_QUALITY: bool = True
DEFAULT_TE_HANDLE_QUALITY_WEIGHT: float = 0.05
TE_HANDLE_MIN_LENGTH: float = 0.040
TE_HANDLE_SHORT_LENGTH_WEIGHT: float = 0.25

# UI-applied TE thickening directly displaces existing B-spline control points.
# The TE gap itself is exact; LE tangency and curvature are always preserved
# when those derivative constraints are available for the fitted spline.
TE_THICKENING_CURVATURE_SAMPLES: int = 500
TE_THICKENING_OBJECTIVE_DISPLACEMENT_WEIGHT: float = 1.0e-5
TE_THICKENING_OBJECTIVE_SMOOTHNESS_WEIGHT: float = 1.0e-4


USER_CONFIG_FILENAME = "airfoilfitter.config.json"
LOADED_USER_CONFIG_PATH: str | None = None


def _candidate_user_config_paths() -> list[Path]:
    candidates: list[Path] = []

    env_path = os.environ.get("AIRFOILFITTER_CONFIG", "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())

    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / USER_CONFIG_FILENAME)
    else:
        project_root = Path(__file__).resolve().parents[1]
        candidates.append(project_root / USER_CONFIG_FILENAME)

    return candidates


def _coerce_override_value(default_value: object, override_value: object) -> object | None:
    if isinstance(default_value, str):
        return override_value if isinstance(override_value, str) else None

    if isinstance(default_value, bool):
        return override_value if isinstance(override_value, bool) else None

    if isinstance(default_value, int):
        if isinstance(override_value, bool):
            return None
        return override_value if isinstance(override_value, int) else None

    if isinstance(default_value, float):
        if isinstance(override_value, bool):
            return None
        if isinstance(override_value, (int, float)):
            return float(override_value)
        return None

    return None


def _apply_user_overrides() -> None:
    global LOADED_USER_CONFIG_PATH

    for cfg_path in _candidate_user_config_paths():
        if not cfg_path.is_file():
            continue

        try:
            with cfg_path.open("r", encoding="utf-8-sig") as handle:
                payload = json.load(handle)
        except Exception as exc:
            print(f"[config] Failed to read '{cfg_path}': {exc}")
            return

        if not isinstance(payload, dict):
            print(f"[config] Ignoring '{cfg_path}': top-level JSON object expected.")
            return

        for name, value in payload.items():
            if not isinstance(name, str) or not name.isupper():
                continue
            if name not in globals():
                print(f"[config] Ignoring unknown key '{name}' in '{cfg_path}'.")
                continue

            current_value = globals()[name]
            new_value = _coerce_override_value(current_value, value)
            if new_value is None:
                print(
                    f"[config] Ignoring invalid value for '{name}' in '{cfg_path}'. "
                    f"Expected type like {type(current_value).__name__}."
                )
                continue

            globals()[name] = new_value

        LOADED_USER_CONFIG_PATH = str(cfg_path)
        return


_apply_user_overrides()
