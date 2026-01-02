"""Worker thread for B-spline fitting operations."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal
import numpy as np
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
        self.single_span = False
        
        # Parameters for insert knot operation
        self.knot_u_value = None
        self.target_surface = None
    
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
        single_span: bool,
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
        self.single_span = single_span
    
    def setup_insert_knot_operation(
        self,
        knot_u_value: float,
        target_surface: str,
    ):
        """Set up parameters for a knot insertion operation."""
        self.operation_type = 'insert_knot'
        self.knot_u_value = knot_u_value
        self.target_surface = target_surface
    
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
            self.single_span,
        )
        
        if success:
            # Create a summary message
            g2_status = "enabled" if self.enforce_g2 else "disabled"
            g3_status = "enabled" if self.enforce_g3 else "disabled"
            te_tangency_status = "enabled" if self.enforce_te_tangency else "disabled"
            single_span_status = "enabled" if self.single_span else "disabled"
            message = f"B-spline fitting completed (G2: {g2_status}, G3: {g3_status}, TE tangency: {te_tangency_status}, Single Span: {single_span_status})"
            self.finished.emit(True, message)
        else:
            self.finished.emit(False, "B-spline fitting failed.")
    
    def _run_insert_knot(self) -> None:
        """Execute knot insertion in the worker thread."""
        self.progress_message.emit(f"Inserting knot at u={self.knot_u_value:.4f} on {self.target_surface} surface...")
        
        success = self.bspline_processor.refine_curve_with_knots(
            [self.knot_u_value],
            surface=self.target_surface
        )
        
        if success:
            message = f"Knot inserted successfully at u={self.knot_u_value:.4f} on {self.target_surface} surface"
            self.finished.emit(True, message)
        else:
            self.finished.emit(False, "Failed to insert knot.")

