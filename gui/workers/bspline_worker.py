"""Worker thread for B-spline fitting operations."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal
import numpy as np
from core import config
from core.bspline_processor import BSplineProcessor
from core.optimization import vertical_distance_and_grad
from utils import bspline_helper


class BSplineWorker(QThread):
    """Worker thread for performing B-spline fitting operations without blocking the UI."""
    
    # Signals emitted when operations complete
    finished = Signal(bool, str)  # success: bool, message: str
    error = Signal(str)  # error_message: str
    progress_message = Signal(str)  # For intermediate status updates
    
    def __init__(self, bspline_processor: BSplineProcessor, parent=None):
        super().__init__(parent)
        self.bspline_processor = bspline_processor
        self.operation_type = None  # 'fit' or 'insert_knot'
        
        # Parameters for fitting operation
        self.upper_data = None
        self.lower_data = None
        self.num_control_points = None
        self.is_thickened = False
        self.upper_te_tangent_vector = None
        self.lower_te_tangent_vector = None
        self.enforce_g2 = False
        self.enforce_g3 = False
        self.preserve_existing_knots = False
        
        # Parameters for insert knot operation
        self.target_surface = None
        self.target_data = None
    
    def setup_fit_operation(
        self,
        upper_data: np.ndarray,
        lower_data: np.ndarray,
        num_control_points: int | tuple[int, int],
        is_thickened: bool,
        upper_te_tangent_vector: np.ndarray | None,
        lower_te_tangent_vector: np.ndarray | None,
        enforce_g2: bool,
        enforce_g3: bool,
        preserve_existing_knots: bool = False,
    ):
        """Set up parameters for a B-spline fitting operation."""
        self.operation_type = 'fit'
        self.upper_data = upper_data.copy()
        self.lower_data = lower_data.copy()
        self.num_control_points = num_control_points
        self.is_thickened = is_thickened
        self.upper_te_tangent_vector = upper_te_tangent_vector.copy() if upper_te_tangent_vector is not None else None
        self.lower_te_tangent_vector = lower_te_tangent_vector.copy() if lower_te_tangent_vector is not None else None
        self.enforce_g2 = enforce_g2
        self.enforce_g3 = enforce_g3
        self.preserve_existing_knots = preserve_existing_knots
    
    def setup_insert_knot_operation(
        self,
        target_surface: str,
        target_data: np.ndarray,
    ):
        """Set up parameters for a knot insertion operation."""
        self.operation_type = 'insert_knot'
        self.target_surface = target_surface
        self.target_data = target_data.copy()
    
    def run(self) -> None:
        """Execute the operation in the worker thread."""
        try:
            if self.operation_type == 'fit':
                self._run_fit()
            elif self.operation_type == 'insert_knot':
                self._run_insert_knot()
            else:
                self.error.emit("No operation type specified")
        except Exception as e:
            self.error.emit(f"Error during operation: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _run_fit(self) -> None:
        """Execute B-spline fitting in the worker thread."""
        self.progress_message.emit("Processing...")

        success = self.bspline_processor.fit_bspline(
            self.upper_data,
            self.lower_data,
            self.num_control_points,
            self.is_thickened,
            self.upper_te_tangent_vector,
            self.lower_te_tangent_vector,
            self.enforce_g2,
            self.enforce_g3,
            self.preserve_existing_knots,
        )
        
        if success:
            # Create a summary message
            g2_status = "enabled" if self.enforce_g2 else "disabled"
            g3_status = "enabled" if self.enforce_g3 else "disabled"
            message = f"B-spline fitting completed (G2: {g2_status}, G3: {g3_status}, TE handle: enabled)"
            self.finished.emit(True, message)
        else:
            error_message = getattr(self.bspline_processor, "last_error_message", None) or "B-spline fitting failed."
            self.finished.emit(False, error_message)
    
    def _run_insert_knot(self) -> None:
        """Execute knot insertion in the worker thread."""
        if self.target_surface not in {"upper", "lower"}:
            self.finished.emit(False, "Invalid target surface.")
            return
        if self.target_data is None or self.target_data.size == 0:
            self.finished.emit(False, "No target data available for knot insertion.")
            return

        target_curve = (
            self.bspline_processor.upper_curve
            if self.target_surface == "upper"
            else self.bspline_processor.lower_curve
        )
        existing_knots = (
            self.bspline_processor.upper_knot_vector
            if self.target_surface == "upper"
            else self.bspline_processor.lower_knot_vector
        )
        if target_curve is None or existing_knots is None:
            self.finished.emit(False, "Target curve is unavailable for knot insertion.")
            return

        objective_metric = str(getattr(config, "FIT_ERROR_OBJECTIVE", "vertical")).strip().lower()
        if objective_metric not in {"msr", "vertical"}:
            objective_metric = "vertical"
        insertion_strategy = str(getattr(config, "KNOT_INSERTION_STRATEGY", "adaptive")).strip().lower()
        exponent_guess = float(
            getattr(
                self.bspline_processor,
                "param_exponent_upper" if self.target_surface == "upper" else "param_exponent_lower",
                0.5,
            )
        )
        if insertion_strategy == "midspan":
            point_u, point_errors = self._collect_pointwise_objective_errors(
                target_curve,
                self.target_data,
                objective_metric,
                exponent_guess,
            )
            if point_errors.size == 0:
                self.finished.emit(False, "Failed to determine insertion target from error data.")
                return
            target_idx = int(np.argmax(point_errors))
            new_knot = self._select_midspan_knot(np.asarray(existing_knots, dtype=float), float(point_u[target_idx]))
        else:
            point_u, point_errors = self._collect_pointwise_objective_errors(
                target_curve,
                self.target_data,
                objective_metric,
                exponent_guess,
            )
            new_knot = self._select_insertion_knot(
                np.asarray(existing_knots, dtype=float),
                point_u,
                point_errors,
            )
        if new_knot is None:
            self.finished.emit(False, "Failed to determine a valid insertion knot.")
            return

        self.progress_message.emit(f"Inserting knot at u={new_knot:.4f} on {self.target_surface} surface...")
        success = self.bspline_processor.refine_curve_with_knots(
            [new_knot],
            surface=self.target_surface
        )
        
        if success:
            info = getattr(self.bspline_processor, "last_insertion_info", None)
            actual_knot = float(new_knot)
            used_fallback = False
            if isinstance(info, dict):
                for record in reversed(info.get("records", [])):
                    if record.get("surface") == self.target_surface:
                        actual_knot = float(record.get("actual_knot", actual_knot))
                        used_fallback = bool(record.get("used_fallback", False))
                        break
            message = f"Knot inserted successfully at u={actual_knot:.4f} on {self.target_surface} surface"
            if used_fallback:
                message += " (spacing fallback applied)"
            self.finished.emit(True, message)
        else:
            self.finished.emit(False, "Failed to insert knot.")

    def _collect_pointwise_objective_errors(
        self,
        target_curve,
        target_data: np.ndarray,
        objective_metric: str,
        exponent_guess: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if objective_metric == "vertical":
            _, _, solved_u, y_residual = vertical_distance_and_grad(
                target_data,
                np.asarray(target_curve.c, dtype=float),
                np.asarray(target_curve.t, dtype=float),
                int(target_curve.k),
                exponent_guess=exponent_guess,
            )
            return np.asarray(solved_u, dtype=float), np.abs(np.asarray(y_residual, dtype=float))

        if objective_metric == "msr":
            u_params = bspline_helper.create_parameter_from_x_coords(target_data, exponent_guess)
            basis = bspline_helper.build_basis_matrix(u_params, np.asarray(target_curve.t, dtype=float), int(target_curve.k))
            fitted = basis @ np.asarray(target_curve.c, dtype=float)
            residual = fitted - target_data
            return np.asarray(u_params, dtype=float), np.linalg.norm(residual, axis=1)
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)

    def _select_insertion_knot(
        self,
        existing_knots: np.ndarray,
        point_u: np.ndarray,
        point_errors: np.ndarray,
    ) -> float | None:
        kv = np.asarray(existing_knots, dtype=float)
        if kv.size < 2:
            return None

        nonzero_spans: list[tuple[float, float]] = []
        for i in range(len(kv) - 1):
            left = float(kv[i])
            right = float(kv[i + 1])
            if right - left > 1e-10:
                nonzero_spans.append((left, right))
        if not nonzero_spans:
            return None

        span_widths = np.asarray([right - left for left, right in nonzero_spans], dtype=float)
        median_width = float(np.median(span_widths)) if span_widths.size else 0.0
        min_preferred_width = max(1e-5, 0.40 * median_width) if median_width > 0.0 else 1e-5

        candidates: list[dict[str, float | int]] = []
        u_values = np.asarray(point_u, dtype=float)
        errors = np.asarray(point_errors, dtype=float)
        for left, right in nonzero_spans:
            width = float(right - left)
            if width <= 1e-10:
                continue
            in_span = (u_values >= left) & (u_values < right)
            if right >= nonzero_spans[-1][1] - 1e-12:
                in_span = (u_values >= left) & (u_values <= right)
            if not np.any(in_span):
                continue
            span_errors = errors[in_span]
            span_u = u_values[in_span]
            local_idx = int(np.argmax(span_errors))
            error_sq_sum = float(np.sum(span_errors * span_errors))
            weight_sum = float(np.sum(span_errors * span_errors))
            weighted_u = (
                float(np.sum(span_u * (span_errors * span_errors)) / weight_sum)
                if weight_sum > 0.0
                else float(np.mean(span_u))
            )
            max_error = float(span_errors[local_idx])
            density_penalty = min(1.0, width / max(min_preferred_width, 1e-12))
            score = error_sq_sum * density_penalty
            candidates.append(
                {
                    "left": left,
                    "right": right,
                    "width": width,
                    "score": score,
                    "raw_score": error_sq_sum,
                    "max_error": max_error,
                    "u_target": float(span_u[local_idx]),
                    "u_weighted": weighted_u,
                }
            )

        if not candidates:
            return None

        preferred = [c for c in candidates if float(c["width"]) >= min_preferred_width]
        pool = preferred if preferred else candidates
        best = max(pool, key=lambda c: (float(c["score"]), float(c["max_error"]), float(c["width"])))

        best_left = float(best["left"])
        best_right = float(best["right"])
        span_width = float(best["width"])
        u_target = float(best["u_target"])
        u_weighted = float(best.get("u_weighted", u_target))
        midpoint = 0.5 * (best_left + best_right)
        margin = min(0.20 * span_width, max(1e-4, 0.10 * span_width))
        requested = float(np.clip(0.65 * u_weighted + 0.35 * midpoint, best_left + margin, best_right - margin))

        nearest_existing_dist = float(np.min(np.abs(kv - requested))) if kv.size else float("inf")
        if nearest_existing_dist <= max(1e-6, 0.12 * span_width):
            requested = 0.5 * (best_left + best_right)

        return float(np.clip(requested, best_left + margin, best_right - margin))

    def _select_midspan_knot(self, existing_knots: np.ndarray, u_target: float) -> float | None:
        kv = np.asarray(existing_knots, dtype=float)
        if kv.size < 2:
            return None
        idx = int(np.searchsorted(kv, u_target))
        idx = max(1, min(idx, len(kv) - 1))
        t_left = float(kv[idx - 1])
        t_right = float(kv[idx])
        if (t_right - t_left) < 1e-4:
            span_left = float(t_left - kv[idx - 2]) if idx > 1 else 0.0
            span_right = float(kv[idx + 1] - t_right) if idx < len(kv) - 1 else 0.0
            if idx > 1 and span_left > span_right:
                return float((kv[idx - 2] + t_left) / 2.0)
            if idx < len(kv) - 1:
                return float((t_right + kv[idx + 1]) / 2.0)
        return float((t_left + t_right) / 2.0)

