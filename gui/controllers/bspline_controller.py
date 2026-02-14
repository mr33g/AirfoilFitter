from __future__ import annotations

from typing import Any
import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial import cKDTree
from core import config
from core.bspline_processor import BSplineProcessor
from gui.workers.bspline_worker import BSplineWorker


class BSplineController:
    """Controller for B-spline operations, following existing architecture."""
    
    def __init__(self, processor, window: Any):
        self.processor = processor
        self.window = window
        # Reuse the instance created in MainWindow to keep a single source
        self.bspline_processor = getattr(window, "bspline_processor", None) or BSplineProcessor()
        # Store the B-spline processor in the window for access by other controllers
        self.window.bspline_processor = self.bspline_processor
        
        # Worker thread for long-running operations
        self._current_worker: BSplineWorker | None = None

    def refit_if_fitted(self) -> None:
        """Re-fit B-spline if one is already fitted. Used for parameter changes.
        
        Preserves the current control point configuration.
        """
        if not self.bspline_processor.is_fitted():
            return  # No existing fit, do nothing

        if getattr(self.processor, "upper_data", None) is None:
            return  # No data loaded

        self.window.status_log.append("Parameter changed, re-fitting B-spline...")
        # Set flag to preserve current CP counts instead of resetting to defaults
        self._refitting = True
        self.fit_bspline()

    def handle_te_vector_points_changed(self) -> None:
        """Handle TE vector points dropdown changes.
        
        Always recalculates TE vectors from the input data.
        If a fit exists and tangency is disabled: preserves the fit and just updates vectors.
        If a fit exists and tangency is enabled: re-fits with new TE vectors.
        """
        if getattr(self.processor, "upper_data", None) is None:
            return  # No data loaded
        
        opt = self.window.optimizer_panel
        try:
            te_vector_points = int(opt.te_vector_points_combo.currentText())
        except ValueError:
            return
        
        is_fitted = self.bspline_processor.is_fitted()
        tangency_enabled = opt.enforce_te_tangency_checkbox.isChecked()
        
        # Recalculate TE vectors (this updates processor's TE vector data)
        self.processor.recalculate_te_vectors(te_vector_points)
        self.window.status_log.append(f"TE vectors recalculated with {te_vector_points} points.")
        
        if is_fitted:
            if tangency_enabled:
                # Re-fit with new TE vectors, preserving current CP configuration
                self.window.status_log.append("TE tangency enabled, re-fitting B-spline...")
                self._refitting = True
                self.fit_bspline()
            else:
                # Just update the plot with existing B-spline (preserves fit)
                self._update_plot_with_bsplines()
        else:
            # No fit exists, just update the plot with TE vectors
            self.processor.update_plot()



    def fit_bspline(self) -> None:
        """Fit B-spline curves to loaded airfoil data."""
        if getattr(self.processor, "upper_data", None) is None or getattr(self.processor, "lower_data", None) is None:
            self.window.status_log.append("No airfoil data loaded. Please load an airfoil first.")
            return

        # Check if a worker is already running
        if self._current_worker is not None and self._current_worker.isRunning():
            if hasattr(self, "_refitting"):
                self._refitting = False
            self.window.status_log.append("A B-spline operation is already in progress. Please wait.")
            return

        try:
            # Get control point count and degree from GUI
            gui_cp = int(self.window.optimizer_panel.initial_cp_spin.value())
            gui_degree = int(self.window.optimizer_panel.bspline_degree_spin.value())

            # Get G2 flag from GUI checkbox
            enforce_g2 = self.window.optimizer_panel.g2_checkbox.isChecked()
            
            # Get G3 flag from GUI checkbox
            enforce_g3 = self.window.optimizer_panel.g3_checkbox.isChecked()
            
            # Get TE tangency flag from GUI checkbox
            enforce_te_tangency = self.window.optimizer_panel.enforce_te_tangency_checkbox.isChecked()
            
            # Get smoothness penalty from GUI
            smoothing_weight = float(self.window.optimizer_panel.smoothness_penalty_spin.value())
            self.bspline_processor.smoothing_weight = smoothing_weight
            
            # Determine control point counts
            # If we're coming from an automatic refinement (knot insertion), we might have asymmetric counts
            if hasattr(self, "_refitting") and self._refitting:
                num_cp_upper = self.bspline_processor.num_cp_upper
                num_cp_lower = self.bspline_processor.num_cp_lower
                self._refitting = False # Reset flag
            else:
                # Normal fit from button click: reset to symmetric GUI value and reset exponents
                num_cp_upper = num_cp_lower = gui_cp
                self.bspline_processor.param_exponent_upper = 0.5
                self.bspline_processor.param_exponent_lower = 0.5

            # Set degree from GUI
            self.bspline_processor.degree = gui_degree
            
            # Store parameters for use in completion handler
            self._pending_fit_params = {
                'num_cp_upper': num_cp_upper,
                'num_cp_lower': num_cp_lower,
                'enforce_g2': enforce_g2,
                'enforce_g3': enforce_g3,
                'enforce_te_tangency': enforce_te_tangency,
            }
            
            # Create and configure worker
            self._current_worker = BSplineWorker(self.bspline_processor, self.window)
            self._current_worker.setup_fit_operation(
                self.processor.upper_data,
                self.processor.lower_data,
                (num_cp_upper, num_cp_lower),
                self.processor.is_trailing_edge_thickened(),
                self.processor.upper_te_tangent_vector,
                self.processor.lower_te_tangent_vector,
                enforce_g2,
                enforce_g3,
                enforce_te_tangency,
            )
            
            # Connect worker signals
            self._current_worker.finished.connect(self._on_fit_finished)
            self._current_worker.error.connect(self._on_worker_error)
            self._current_worker.progress_message.connect(self._on_worker_progress)
            
            # Start spinner and disable buttons
            self.window.status_log.start_spinner("Fitting B-spline")
            self._set_buttons_enabled(False)
            
            # Start worker thread
            self._current_worker.start()

        except Exception as e:  # pragma: no cover
            self.window.status_log.stop_spinner()
            self.window.status_log.append(f"Error during B-spline fitting setup: {e}")
            self._set_buttons_enabled(True)
    
    def _on_fit_finished(self, success: bool, message: str) -> None:
        """Handle completion of B-spline fitting operation."""
        self.window.status_log.stop_spinner()
        self._set_buttons_enabled(True)
        
        if success:
            params = getattr(self, '_pending_fit_params', {})
            enforce_g2 = params.get('enforce_g2', False)
            enforce_g3 = params.get('enforce_g3', False)
            enforce_te_tangency = params.get('enforce_te_tangency', True)
            num_cp_upper = params.get('num_cp_upper', 10)
            num_cp_lower = params.get('num_cp_lower', 10)

            # Log continuity settings used
            g2_status = "enabled" if enforce_g2 else "disabled"
            g3_status = "enabled" if enforce_g3 else "disabled"
            te_tangency_status = "enabled" if enforce_te_tangency else "disabled"
            self.window.status_log.append(f"B-spline fitting with G2: {g2_status}, G3: {g3_status}, TE tangency: {te_tangency_status}")
            
            # Calculate and display errors for each surface
            upper_sum_sq, upper_max_err, upper_max_err_idx, _ = self.calculate_bspline_fitting_error(
                self.bspline_processor.upper_curve,
                self.processor.upper_data,
                return_max_error=True,
            )
            lower_sum_sq, lower_max_err, lower_max_err_idx, _ = self.calculate_bspline_fitting_error(
                self.bspline_processor.lower_curve,
                self.processor.lower_data,
                return_max_error=True,
            )
            
            # Store max error information for plotting
            self.bspline_processor.last_upper_max_error = upper_max_err
            self.bspline_processor.last_upper_max_error_idx = upper_max_err_idx
            self.bspline_processor.last_lower_max_error = lower_max_err
            self.bspline_processor.last_lower_max_error_idx = lower_max_err_idx
            
            # Use the degree that was actually used for fitting
            max_cp = max(num_cp_upper, num_cp_lower)
            max_deg = max(self.bspline_processor.degree_upper, self.bspline_processor.degree_lower)
            num_spans = max_cp - max_deg
            span_info = f"{num_spans} span" if num_spans == 1 else f"{num_spans} spans"
            self.window.status_log.append(
                f"B-spline fit OK (degrees {self.bspline_processor.degree_upper}/{self.bspline_processor.degree_lower}, {span_info}). "
                f"Upper max error: {upper_max_err:.6e}, Lower max error: {lower_max_err:.6e}"
            )
            
            # Update control point labels in the UI (use actual values from processor)
            self.window.optimizer_panel.upper_cp_label.setText(f"Upper CPs: {self.bspline_processor.num_cp_upper}")
            self.window.optimizer_panel.lower_cp_label.setText(f"Lower CPs: {self.bspline_processor.num_cp_lower}")

            # Update button text based on whether CP counts differ from defaults
            self._update_fit_button_text()

            # Trigger plot update with B-spline curves
            self._update_plot_with_bsplines()
        else:
            self.window.status_log.append(message)
        
        # Clean up worker
        if self._current_worker:
            self._current_worker.deleteLater()
            self._current_worker = None
        
        # Clean up pending params
        if hasattr(self, '_pending_fit_params'):
            del self._pending_fit_params
    
    def _on_worker_error(self, error_message: str) -> None:
        """Handle errors from worker thread."""
        self.window.status_log.stop_spinner()
        self.window.status_log.append(error_message)
        self._set_buttons_enabled(True)
        
        # Clean up worker
        if self._current_worker:
            self._current_worker.deleteLater()
            self._current_worker = None
    
    def _on_worker_progress(self, message: str) -> None:
        """Handle progress messages from worker thread."""
        # Update spinner message
        self.window.status_log.stop_spinner()
        self.window.status_log.start_spinner(message)
    
    def _set_buttons_enabled(self, enabled: bool) -> None:
        """Enable or disable B-spline operation buttons."""
        opt = self.window.optimizer_panel
        is_file_loaded = getattr(self.processor, "upper_data", None) is not None
        is_model_built = self.bspline_processor.is_fitted()

        opt.fit_bspline_button.setEnabled(enabled and is_file_loaded)

        # Knot control buttons
        opt.upper_insert_btn.setEnabled(enabled and is_model_built)
        opt.lower_insert_btn.setEnabled(enabled and is_model_built)

        # Fit-driving optimizer controls are read-only while a fit is running
        opt.initial_cp_spin.setEnabled(enabled)
        opt.bspline_degree_spin.setEnabled(enabled)
        opt.smoothness_penalty_spin.setEnabled(enabled)
        opt.g2_checkbox.setEnabled(enabled)
        opt.g3_checkbox.setEnabled(enabled and opt.g2_checkbox.isChecked())
        opt.enforce_te_tangency_checkbox.setEnabled(enabled)
        opt.te_vector_points_combo.setEnabled(enabled)

        # Comb controls are read-only while a fit is running
        self.window.comb_panel.comb_scale_slider.setEnabled(enabled and is_model_built)
        self.window.comb_panel.comb_density_slider.setEnabled(enabled and is_model_built)

    def _update_fit_button_text(self) -> None:
        """Update the fit button text based on whether CP counts differ from defaults."""
        default_cp = config.DEFAULT_BSPLINE_CP

        # Check if current CP counts differ from defaults
        current_upper_cp = self.bspline_processor.num_cp_upper
        current_lower_cp = self.bspline_processor.num_cp_lower

        if current_upper_cp != default_cp or current_lower_cp != default_cp:
            self.window.optimizer_panel.fit_bspline_button.setText("Reset fit")
        else:
            self.window.optimizer_panel.fit_bspline_button.setText("Fit B-spline")

    def calculate_bspline_fitting_error(
                self,
                bspline_curve: BSpline,
                original_data: np.ndarray,
                *,
                return_max_error: bool = False,
                return_all: bool = False,
            ):
            """
            Calculate fitting error for a B-spline curve against original data.
            """
            # Approximate orthogonal by dense sampling
            num_points_curve = config.NUM_POINTS_CURVE_ERROR
            t_samples = np.linspace(0.0, 1.0, num_points_curve)
            if len(t_samples) > 0:
                t_samples[-1] = min(t_samples[-1], 1.0 - 1e-12)
            sampled_curve_points = bspline_curve(t_samples)
            sampled_curve_points = sampled_curve_points[np.argsort(sampled_curve_points[:, 0])]
            tree = cKDTree(sampled_curve_points)
            min_dists, _ = tree.query(original_data, k=1)
            sum_sq = float(np.sum(min_dists ** 2))
            if return_all:
                rms = float(np.sqrt(np.mean(min_dists ** 2)))
                return min_dists, rms, (sum_sq, int(np.argmax(min_dists)))
            if return_max_error:
                max_error = float(np.max(min_dists))
                max_error_idx = int(np.argmax(min_dists))
                
                # Get the parameter value (u) corresponding to the max error index in original_data
                from utils import bspline_helper # Import here to avoid circular dependency
                u_params_original = bspline_helper.create_parameter_from_x_coords(original_data)
                u_at_max_error = u_params_original[max_error_idx]

                return sum_sq, max_error, max_error_idx, u_at_max_error
            return sum_sq
    

    def apply_te_thickening(self, te_thickness_percent: float) -> bool:
        """
        Apply trailing edge thickening to fitted B-splines.
        
        Args:
            te_thickness_percent: The thickness percentage to apply (0.0 to 100.0)
            
        Returns:
            bool: True if thickening was applied successfully, False otherwise
        """
        if not self.bspline_processor.is_fitted():
            self.window.status_log.append("No B-spline model fitted. Please fit B-splines first.")
            return False
        
        try:
            # Convert percentage to decimal
            te_thickness = te_thickness_percent / 100.0
            
            success = self.bspline_processor.apply_te_thickening(te_thickness)
            
            if success:
                self.window.status_log.append(f"Applied {te_thickness_percent:.2f}% trailing edge thickness to B-splines.")
                # Update the plot with thickened B-splines
                self._update_plot_with_bsplines()
                return True
            else:
                self.window.status_log.append("Failed to apply trailing edge thickening to B-splines.")
                return False
                
        except Exception as e:
            self.window.status_log.append(f"Error applying trailing edge thickening to B-splines: {e}")
            return False

    def remove_te_thickening(self) -> bool:
        """
        Remove trailing edge thickening from fitted B-splines.
        
        Returns:
            bool: True if thickening was removed successfully, False otherwise
        """
        if not self.bspline_processor.is_fitted():
            self.window.status_log.append("No B-spline model fitted. Please fit B-splines first.")
            return False
        
        try:
            success = self.bspline_processor.remove_te_thickening()
            
            if success:
                self.window.status_log.append("Removed trailing edge thickening from B-splines.")
                # Update the plot with sharp B-splines
                self._update_plot_with_bsplines()
                return True
            else:
                self.window.status_log.append("Failed to remove trailing edge thickening from B-splines.")
                return False
                
        except Exception as e:
            self.window.status_log.append(f"Error removing trailing edge thickening from B-splines: {e}")
            return False

    def insert_knot(self, surface: str) -> None:
        """Insert a knot at the location of maximum deviation on the specified surface."""
        if not self.bspline_processor.is_fitted():
            self.window.status_log.append("No B-spline model fitted. Please fit B-splines first.")
            return

        if self.processor.upper_data is None or self.processor.lower_data is None:
            self.window.status_log.append("No airfoil data loaded. Cannot determine max deviation.")
            return

        target_surface = surface
        target_data = self.processor.upper_data if target_surface == 'upper' else self.processor.lower_data
        target_curve = self.bspline_processor.upper_curve if target_surface == 'upper' else self.bspline_processor.lower_curve

        # Insert a knot
        try:
            _, max_err, _, u_at_max = self.calculate_bspline_fitting_error(
                target_curve,
                target_data,
                return_max_error=True,
            )

            existing_knots = self.bspline_processor.upper_knot_vector if target_surface == 'upper' else self.bspline_processor.lower_knot_vector
            new_knot = u_at_max
            if existing_knots is not None:
                idx = np.searchsorted(existing_knots, u_at_max)
                t_left = existing_knots[idx-1]
                t_right = existing_knots[idx]
                if t_right - t_left < 1e-4:
                    span_left = t_left - existing_knots[idx-2] if idx > 1 else 0
                    span_right = existing_knots[idx+1] - t_right if idx < len(existing_knots)-1 else 0
                    new_knot = (existing_knots[idx-2] + t_left) / 2 if span_left > span_right else (t_right + existing_knots[idx+1]) / 2
                else:
                    new_knot = (t_left + t_right) / 2

            self.window.status_log.append(f"Inserting knot at u={new_knot:.4f} on {target_surface} surface.")
            
            success = self.bspline_processor.refine_curve_with_knots([new_knot], surface=target_surface)

            if success:
                self._on_fit_finished(True, "")
            else:
                self.window.status_log.append(f"Failed to insert knot on {target_surface} surface.")

        except Exception as e:
            self.window.status_log.append(f"Error during knot insertion: {e}")

    def is_te_thickened(self) -> bool:
        """
        Check if the B-spline model has trailing edge thickening applied.
        
        Returns:
            bool: True if trailing edge is thickened, False otherwise
        """
        if not self.bspline_processor.is_fitted():
            return False
        return not self.bspline_processor.is_sharp_te

    
    def _update_plot_with_bsplines(self) -> None:
        """Update plot to display B-spline curves and control points."""
        # Get comb parameters from the UI
        comb_scale = self.window.comb_panel.comb_scale_slider.value() / 1000.0
        comb_density = self.window.comb_panel.comb_density_slider.value()
        
        # Calculate B-spline comb data
        comb_bspline = self.bspline_processor.calculate_curvature_comb_data(
            num_points_per_segment=comb_density,
            scale_factor=comb_scale,
        )
        
        self.processor.emit_plot_update(
            bspline_processor=self.bspline_processor,
            comb_bspline=comb_bspline,
        )


