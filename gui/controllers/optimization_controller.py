"""Optimization controller for the Airfoil Fitter GUI.

Handles TE vector recalculation and related functionality.
"""

from __future__ import annotations

from typing import Any

from core.airfoil_processor import AirfoilProcessor


class OptimizationController:
    """Handles TE vector recalculation and related functionality."""
    
    def __init__(self, processor: AirfoilProcessor, window: Any, ui_state_controller: Any = None):
        self.processor = processor
        self.window = window
        self.ui_state_controller = ui_state_controller
    
    def recalculate_te_vectors(self) -> None:
        """Delegate to the canonical TE-vector flow in BSplineController."""
        bspline_controller = getattr(self.window, "bspline_controller", None)
        if bspline_controller is not None:
            bspline_controller.handle_te_vector_points_changed()
            return
        self.processor.log_message.emit("B-spline controller not initialized; cannot recalculate TE vectors.")
