from __future__ import annotations

from typing import Any
import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial import cKDTree
from core import config
from core.bspline_processor import BSplineProcessor
from core.debug_log import write_debug_line
from core.optimization import vertical_error_metrics
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

    def _append_debug_message(self, message: str) -> None:
        if config.DEBUG_WORKER_LOGGING:
            write_debug_line(message)

    def _append_debug_control_points(self) -> None:
        if not config.DEBUG_WORKER_LOGGING:
            return
        if self.bspline_processor.upper_control_points is None or self.bspline_processor.lower_control_points is None:
            return
        self._append_debug_message("[DEBUG] Final control points:")
        for i, point in enumerate(np.asarray(self.bspline_processor.upper_control_points)):
            self._append_debug_message(f"[DEBUG]   Upper P{i}: ({point[0]:.6f}, {point[1]:.6f})")
        for i, point in enumerate(np.asarray(self.bspline_processor.lower_control_points)):
            self._append_debug_message(f"[DEBUG]   Lower P{i}: ({point[0]:.6f}, {point[1]:.6f})")

    def _append_debug_cp_fourth_differences(self) -> None:
        """Analysis only: log fourth finite differences of the control polygons."""
        if not config.DEBUG_WORKER_LOGGING:
            return

        for surface_name, control_points in (
            ("Upper", self.bspline_processor.upper_control_points),
            ("Lower", self.bspline_processor.lower_control_points),
        ):
            if control_points is None or len(control_points) < 5:
                continue
            d4 = np.diff(np.asarray(control_points, dtype=float)[:, :2], n=4, axis=0)
            magnitudes = np.linalg.norm(d4, axis=1)
            max_idx = int(np.argmax(magnitudes)) if magnitudes.size else -1
            max_mag = float(magnitudes[max_idx]) if max_idx >= 0 else 0.0
            self._append_debug_message(
                "[DEBUG] CP fourth differences "
                f"{surface_name}: count={len(d4)}, max_norm={max_mag:.3e}, "
                f"max_window=P{max_idx}..P{max_idx + 4}"
            )

    def refit_if_fitted(self) -> None:
        """Re-fit B-spline if one is already fitted. Used for parameter changes.
        
        Preserves the current control point configuration.
        """
        has_existing_model = (
            self.bspline_processor.upper_control_points is not None
            and self.bspline_processor.lower_control_points is not None
            and self.bspline_processor.upper_knot_vector is not None
            and self.bspline_processor.lower_knot_vector is not None
        )
        if not self.bspline_processor.is_fitted() and not has_existing_model:
            return  # No existing fit, do nothing

        if getattr(self.processor, "upper_data", None) is None:
            return  # No data loaded

        self.window.status_log.append("Parameter changed, re-fitting B-spline...")
        # Set flag to preserve current CP counts instead of resetting to defaults
        self._refitting = True
        self.fit_bspline()

    def refit_smoothing_full(self) -> None:
        """Re-fit B-spline from a fresh solve when the smoothness slider is released."""
        has_existing_model = (
            self.bspline_processor.upper_control_points is not None
            and self.bspline_processor.lower_control_points is not None
        )
        if not self.bspline_processor.is_fitted() and not has_existing_model:
            return

        if getattr(self.processor, "upper_data", None) is None:
            return

        self.window.status_log.append("Smoothness changed, fitting B-spline...")
        if hasattr(self, "_refitting"):
            self._refitting = False
        self._fresh_refit_preserve_counts = True
        self.fit_bspline()

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
            
            # Get smoothness penalty from GUI
            smoothing_weight = float(self.window.optimizer_panel.smoothness_penalty_value())
            self.bspline_processor.smoothing_weight = smoothing_weight

            # Determine control point counts
            # If we're coming from an automatic refinement (knot insertion), we might have asymmetric counts
            if bool(getattr(self, "_fresh_refit_preserve_counts", False)):
                num_cp_upper = self.bspline_processor.num_cp_upper or gui_cp
                num_cp_lower = self.bspline_processor.num_cp_lower or gui_cp
                self.bspline_processor.param_exponent_upper = 0.5
                self.bspline_processor.param_exponent_lower = 0.5
                preserve_existing_knots = False
                self._fresh_refit_preserve_counts = False
            elif hasattr(self, "_refitting") and self._refitting:
                num_cp_upper = self.bspline_processor.num_cp_upper
                num_cp_lower = self.bspline_processor.num_cp_lower
                preserve_existing_knots = (
                    self.bspline_processor.upper_knot_vector is not None
                    and self.bspline_processor.lower_knot_vector is not None
                    and int(self.bspline_processor.degree_upper) == gui_degree
                    and int(self.bspline_processor.degree_lower) == gui_degree
                )
                self._refitting = False # Reset flag
            else:
                # Normal fit from button click: reset to symmetric GUI value and reset exponents
                num_cp_upper = num_cp_lower = gui_cp
                self.bspline_processor.param_exponent_upper = 0.5
                self.bspline_processor.param_exponent_lower = 0.5
                preserve_existing_knots = False

            # Set degree from GUI
            self.bspline_processor.degree = gui_degree
            
            # Store parameters for use in completion handler
            self._pending_fit_params = {
                'num_cp_upper': num_cp_upper,
                'num_cp_lower': num_cp_lower,
                'enforce_g2': enforce_g2,
                'enforce_g3': enforce_g3,
                'preserve_existing_knots': preserve_existing_knots,
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
                preserve_existing_knots,
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
            requested_g2 = params.get('enforce_g2', False)
            requested_g3 = params.get('enforce_g3', False)
            actual_g2 = bool(self.bspline_processor.enforce_g2)
            actual_g3 = bool(self.bspline_processor.enforce_g3)
            num_cp_upper = params.get('num_cp_upper', 10)
            num_cp_lower = params.get('num_cp_lower', 10)

            # Log requested vs actual continuity settings.
            g2_status = "enabled" if actual_g2 else "disabled"
            g3_status = "enabled" if actual_g3 else "disabled"
            self.window.status_log.append(
                f"B-spline fitting with G2: {g2_status}, G3: {g3_status}, TE handle: enabled"
            )
            if requested_g2 and not actual_g2:
                self.window.status_log.append(
                    "Requested G2 fit was not accepted; fit fell back to G1-independent mode."
                )
            if requested_g3 and not actual_g3:
                self.window.status_log.append(
                    "Requested G3 continuity was not active in the final fit."
                )
            opt_info = getattr(self.bspline_processor, "last_optimizer_info", None)
            if isinstance(opt_info, dict):
                mode = opt_info.get("mode")
                metric = opt_info.get("fit_error_metric")
                self._append_debug_message(
                    "[DEBUG] Smoothness: "
                    f"weight={float(opt_info.get('smoothing_weight', self.bspline_processor.smoothing_weight)):.3f}, "
                    "mode=control_point_fourth_diff"
                )
                self._append_debug_message(
                    "[DEBUG] Solver: "
                    f"profile={str(opt_info.get('solver_profile', 'unknown'))}, "
                    f"success={bool(opt_info.get('success', False))}, "
                    f"accepted={bool(opt_info.get('accepted', False))}, "
                    f"status={int(opt_info.get('status', -1))}, "
                    f"iterations={int(opt_info.get('iterations', -1))}, "
                    f"objective={float(opt_info.get('objective', float('nan'))):.3e}, "
                    f"constraint violation={float(opt_info.get('max_constraint_violation', float('nan'))):.3e}"
                )
                self._append_debug_message(
                    "[DEBUG] Objective terms: "
                    f"initial fit/smooth={float(opt_info.get('initial_fit_error_total', float('nan'))):.3e}/"
                    f"{float(opt_info.get('initial_smoothing_penalty_total', float('nan'))):.3e}, "
                    f"final fit/smooth={float(opt_info.get('final_fit_error_total', float('nan'))):.3e}/"
                    f"{float(opt_info.get('final_smoothing_penalty_total', float('nan'))):.3e}"
                )
                self._append_debug_message(
                    "[DEBUG] Smoothing scale: "
                    f"baseline raw={float(opt_info.get('raw_smoothing_baseline_total', float('nan'))):.3e}, "
                    f"fit ref={float(opt_info.get('smoothing_reference_fit', float('nan'))):.3e}, "
                    f"scale={float(opt_info.get('smoothing_objective_scale', float('nan'))):.3e}"
                )
                self._append_debug_message(
                    "[DEBUG] Smoothing terms: "
                    f"upper fourth={float(opt_info.get('final_smoothing_penalty_upper_fourth_diff', float('nan'))):.3e}, "
                    f"lower fourth={float(opt_info.get('final_smoothing_penalty_lower_fourth_diff', float('nan'))):.3e}"
                )
                te_handle_final = float(opt_info.get('final_te_handle_penalty_total', float('nan')))
                if np.isfinite(te_handle_final):
                    self._append_debug_message(
                        "[DEBUG] TE handle quality: "
                        f"weight={float(opt_info.get('te_handle_weight', float('nan'))):.3f}, "
                        f"min_len={float(opt_info.get('te_handle_min_length', float('nan'))):.3f}, "
                        f"initial/final={float(opt_info.get('initial_te_handle_penalty_total', float('nan'))):.3e}/"
                        f"{te_handle_final:.3e}, "
                        f"scale={float(opt_info.get('te_handle_objective_scale', float('nan'))):.3e}"
                    )
                    self._append_debug_message(
                        "[DEBUG] TE handle terms: "
                        f"upper angle/short/len="
                        f"{float(opt_info.get('final_te_handle_penalty_upper_angle', float('nan'))):.3e}/"
                        f"{float(opt_info.get('final_te_handle_penalty_upper_short_length', float('nan'))):.3e}/"
                        f"{float(opt_info.get('final_te_handle_upper_length', float('nan'))):.3f}, "
                        f"lower angle/short/len="
                        f"{float(opt_info.get('final_te_handle_penalty_lower_angle', float('nan'))):.3e}/"
                        f"{float(opt_info.get('final_te_handle_penalty_lower_short_length', float('nan'))):.3e}/"
                        f"{float(opt_info.get('final_te_handle_lower_length', float('nan'))):.3f}"
                    )
                if mode is not None:
                    self.window.status_log.append(f"Fit mode used: {mode}.")
                if metric is not None:
                    self.window.status_log.append(f"Fit objective metric: {metric}.")
            
            # Use the degree that was actually used for fitting
            max_cp = max(num_cp_upper, num_cp_lower)
            max_deg = max(self.bspline_processor.degree_upper, self.bspline_processor.degree_lower)
            num_spans = max_cp - max_deg
            span_info = f"{num_spans} span" if num_spans == 1 else f"{num_spans} spans"
            upper_vertical, lower_vertical = self._update_final_error_metrics()
            self.window.status_log.append(
                f"B-spline fit OK (degrees {self.bspline_processor.degree_upper}/{self.bspline_processor.degree_lower}, {span_info}). "
                f"Vertical max upper/lower (% chord) = "
                f"{upper_vertical['max_error'] * 100.0:.4f}% / {lower_vertical['max_error'] * 100.0:.4f}%"
            )
            self.window.status_log.append(
                "Vertical RMS upper/lower (% chord) = "
                f"{upper_vertical['rms'] * 100.0:.4f}% / {lower_vertical['rms'] * 100.0:.4f}%"
            )
            self._append_debug_cp_fourth_differences()
            self._append_debug_control_points()
            
            # Update control point labels in the UI (use actual values from processor)
            self.window.optimizer_panel.upper_cp_label.setText(f"Upper CPs: {self.bspline_processor.num_cp_upper}")
            self.window.optimizer_panel.lower_cp_label.setText(f"Lower CPs: {self.bspline_processor.num_cp_lower}")

            # Update button text based on whether CP counts differ from defaults
            self._update_fit_button_text()

            # Trigger plot update with B-spline curves
            self._update_plot_with_bsplines()
        else:
            self.window.status_log.append(message)
            opt_info = getattr(self.bspline_processor, "last_optimizer_info", None)
            if isinstance(opt_info, dict):
                status = opt_info.get("status")
                detail = opt_info.get("message")
                violation = opt_info.get("max_constraint_violation")
                self._append_debug_message(
                    "[DEBUG] G2 failure: "
                    f"status={status}, violation={float(violation) if violation is not None else float('nan'):.3e}, "
                    f"message={detail}"
                )
        
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
        fp = self.window.file_panel
        opt = self.window.optimizer_panel
        is_file_loaded = getattr(self.processor, "upper_data", None) is not None
        is_model_built = self.bspline_processor.is_fitted()

        # Disable file load/export controls while worker operations are running.
        fp.load_button.setEnabled(enabled)
        fp.export_dxf_button.setEnabled(enabled and is_model_built)
        fp.export_bsp_button.setEnabled(enabled and is_model_built and config.ENABLE_BSP_EXPORT)
        fp.export_dat_button.setEnabled(enabled and is_model_built and config.ENABLE_DAT_EXPORT)

        opt.fit_bspline_button.setEnabled(enabled and is_file_loaded)

        # Knot control buttons
        opt.upper_insert_btn.setEnabled(enabled and is_model_built)
        opt.lower_insert_btn.setEnabled(enabled and is_model_built)

        # Fit-driving optimizer controls are read-only while a fit is running
        opt.initial_cp_spin.setEnabled(enabled)
        opt.bspline_degree_spin.setEnabled(enabled)
        opt.smoothness_penalty_slider.setEnabled(enabled)
        opt.g2_checkbox.setEnabled(enabled)
        opt.g3_checkbox.setEnabled(enabled and opt.g2_checkbox.isChecked())
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

    def _error_reference_data(self) -> tuple[np.ndarray, np.ndarray]:
        upper_ref = getattr(self.processor, "upper_display_reference_data", None)
        lower_ref = getattr(self.processor, "lower_display_reference_data", None)
        if upper_ref is None or lower_ref is None or len(upper_ref) == 0 or len(lower_ref) == 0:
            return self.processor.upper_data, self.processor.lower_data
        return upper_ref, lower_ref

    def _update_final_error_metrics(
        self,
    ) -> tuple[dict, dict]:
        upper_error_data, lower_error_data = self._error_reference_data()
        upper_sum_sq, upper_max_err, upper_max_err_idx, _ = self.calculate_bspline_fitting_error(
            self.bspline_processor.upper_curve,
            upper_error_data,
            return_max_error=True,
        )
        lower_sum_sq, lower_max_err, lower_max_err_idx, _ = self.calculate_bspline_fitting_error(
            self.bspline_processor.lower_curve,
            lower_error_data,
            return_max_error=True,
        )
        _ = upper_sum_sq, lower_sum_sq

        self.bspline_processor.last_upper_max_error = upper_max_err
        self.bspline_processor.last_upper_max_error_idx = upper_max_err_idx
        self.bspline_processor.last_lower_max_error = lower_max_err
        self.bspline_processor.last_lower_max_error_idx = lower_max_err_idx

        upper_vertical = self.calculate_bspline_vertical_error(
            self.bspline_processor.upper_curve,
            upper_error_data,
            exponent_guess=float(getattr(self.bspline_processor, "param_exponent_upper", 0.5)),
        )
        lower_vertical = self.calculate_bspline_vertical_error(
            self.bspline_processor.lower_curve,
            lower_error_data,
            exponent_guess=float(getattr(self.bspline_processor, "param_exponent_lower", 0.5)),
        )
        return upper_vertical, lower_vertical

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
            sort_idx = np.argsort(sampled_curve_points[:, 0])
            sampled_curve_points = sampled_curve_points[sort_idx]
            t_sorted = t_samples[sort_idx]
            tree = cKDTree(sampled_curve_points)
            min_dists, nn_curve_idx = tree.query(original_data, k=1)
            sum_sq = float(np.sum(min_dists ** 2))
            if return_all:
                rms = float(np.sqrt(np.mean(min_dists ** 2)))
                return min_dists, rms, (sum_sq, int(np.argmax(min_dists)))
            if return_max_error:
                max_error = float(np.max(min_dists))
                max_error_idx = int(np.argmax(min_dists))
                nearest_curve_idx = int(nn_curve_idx[max_error_idx])
                nearest_curve_idx = max(0, min(nearest_curve_idx, len(t_sorted) - 1))
                u_at_max_error = float(t_sorted[nearest_curve_idx])

                return sum_sq, max_error, max_error_idx, u_at_max_error
            return sum_sq

    def calculate_bspline_vertical_error(
                self,
                bspline_curve: BSpline,
                original_data: np.ndarray,
                *,
                exponent_guess: float = 0.5,
            ) -> dict[str, float | int | np.ndarray]:
            """
            Calculate vertical fitting error by matching each data x-position to x(u) on the spline.
            """
            return vertical_error_metrics(
                bspline_curve,
                original_data,
                exponent_guess=float(exponent_guess),
            )
    

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

        if self._current_worker is not None and self._current_worker.isRunning():
            self.window.status_log.append("A B-spline operation is already in progress. Please wait.")
            return

        target_surface = surface
        target_data = self.processor.upper_data if target_surface == 'upper' else self.processor.lower_data

        # Insert a knot
        try:
            self.window.status_log.append(f"Inserting knot on {target_surface} surface...")
            self._current_worker = BSplineWorker(self.bspline_processor, self.window)
            self._current_worker.setup_insert_knot_operation(target_surface, target_data)
            self._current_worker.finished.connect(self._on_insert_knot_finished)
            self._current_worker.error.connect(self._on_worker_error)
            self._current_worker.progress_message.connect(self._on_worker_progress)

            self.window.status_log.start_spinner("Inserting knot")
            self._set_buttons_enabled(False)
            self._current_worker.start()

        except Exception as e:
            self.window.status_log.append(f"Error during knot insertion: {e}")

    def _on_insert_knot_finished(self, success: bool, message: str) -> None:
        """Handle completion of knot insertion operation."""
        self.window.status_log.stop_spinner()
        self._set_buttons_enabled(True)
        self.window.status_log.append(message)

        if success and self.bspline_processor.upper_curve is not None and self.bspline_processor.lower_curve is not None:
            upper_vertical, lower_vertical = self._update_final_error_metrics()
            self.window.status_log.append(
                "Post-insert vertical max upper/lower (% chord) = "
                f"{upper_vertical['max_error'] * 100.0:.4f}% / {lower_vertical['max_error'] * 100.0:.4f}%"
            )
            self.window.status_log.append(
                "Post-insert vertical RMS upper/lower (% chord) = "
                f"{upper_vertical['rms'] * 100.0:.4f}% / {lower_vertical['rms'] * 100.0:.4f}%"
            )
            self.window.optimizer_panel.upper_cp_label.setText(f"Upper CPs: {self.bspline_processor.num_cp_upper}")
            self.window.optimizer_panel.lower_cp_label.setText(f"Lower CPs: {self.bspline_processor.num_cp_lower}")
            self._update_fit_button_text()
            self._update_plot_with_bsplines()

        if self._current_worker:
            self._current_worker.deleteLater()
            self._current_worker = None

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


