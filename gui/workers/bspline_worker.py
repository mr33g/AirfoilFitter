"""Worker thread for B-spline fitting operations."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal
import numpy as np
from scipy.spatial import cKDTree
from core import config
from core.bspline_processor import BSplineProcessor


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
        self.enforce_te_tangency = True
        
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
        enforce_te_tangency: bool,
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
        self.enforce_te_tangency = enforce_te_tangency
    
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
            self.enforce_te_tangency,
        )
        
        if success:
            # Create a summary message
            g2_status = "enabled" if self.enforce_g2 else "disabled"
            g3_status = "enabled" if self.enforce_g3 else "disabled"
            te_tangency_status = "enabled" if self.enforce_te_tangency else "disabled"
            message = f"B-spline fitting completed (G2: {g2_status}, G3: {g3_status}, TE tangency: {te_tangency_status})"
            self.finished.emit(True, message)
        else:
            self.finished.emit(False, "B-spline fitting failed.")
    
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

        num_points_curve = int(max(128, getattr(config, "NUM_POINTS_CURVE_ERROR", 35000)))
        t_samples = np.linspace(0.0, 1.0, num_points_curve)
        if len(t_samples) > 0:
            t_samples[-1] = min(t_samples[-1], 1.0 - 1e-12)
        sampled_curve_points = target_curve(t_samples)
        sort_idx = np.argsort(sampled_curve_points[:, 0])
        sampled_curve_points = sampled_curve_points[sort_idx]
        t_sorted = t_samples[sort_idx]
        tree = cKDTree(sampled_curve_points)
        min_dists, nn_curve_idx = tree.query(self.target_data, k=1)
        max_error_idx = int(np.argmax(min_dists))
        nearest_curve_idx = int(nn_curve_idx[max_error_idx])
        nearest_curve_idx = max(0, min(nearest_curve_idx, len(t_sorted) - 1))
        u_at_max = float(t_sorted[nearest_curve_idx])

        idx = int(np.searchsorted(existing_knots, u_at_max))
        idx = max(1, min(idx, len(existing_knots) - 1))
        t_left = float(existing_knots[idx - 1])
        t_right = float(existing_knots[idx])
        if (t_right - t_left) < 1e-4:
            span_left = float(t_left - existing_knots[idx - 2]) if idx > 1 else 0.0
            span_right = float(existing_knots[idx + 1] - t_right) if idx < len(existing_knots) - 1 else 0.0
            if idx > 1 and span_left > span_right:
                new_knot = float((existing_knots[idx - 2] + t_left) / 2.0)
            elif idx < len(existing_knots) - 1:
                new_knot = float((t_right + existing_knots[idx + 1]) / 2.0)
            else:
                new_knot = float((t_left + t_right) / 2.0)
        else:
            new_knot = float((t_left + t_right) / 2.0)

        self.progress_message.emit(f"Inserting knot at u={new_knot:.4f} on {self.target_surface} surface...")
        success = self.bspline_processor.refine_curve_with_knots(
            [new_knot],
            surface=self.target_surface
        )
        
        if success:
            message = f"Knot inserted successfully at u={new_knot:.4f} on {self.target_surface} surface"
            self.finished.emit(True, message)
        else:
            self.finished.emit(False, "Failed to insert knot.")

