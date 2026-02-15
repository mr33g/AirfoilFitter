"""File operations controller for the Airfoil Fitter GUI.

Handles loading airfoil data files and exporting B-spline models.
"""

from __future__ import annotations

import os
from typing import Any
import numpy as np

from PySide6.QtWidgets import QFileDialog
from scipy import interpolate

from core.airfoil_processor import AirfoilProcessor
from core import config
from utils.dxf_exporter import (
    DXF_EXPORT_MODE_BEZIER,
    DXF_EXPORT_MODE_NURBS,
    export_bspline_to_dxf,
)
from utils.bsp_exporter import export_bspline_to_bsp
from utils.bsp_importer import load_bspline_from_bsp
from utils.data_loader import export_airfoil_to_selig_format, load_airfoil_data


class FileController:
    """Handles file loading and export operations."""
    
    def __init__(self, processor: AirfoilProcessor, window: Any, ui_state_controller: Any = None):
        self.processor = processor
        self.window = window
        self.ui_state_controller = ui_state_controller

    def _get_bspline_processor(self):
        """Return the canonical B-spline processor instance."""
        bspline_controller = getattr(self.window, "bspline_controller", None)
        if bspline_controller is not None:
            bspline_proc = getattr(bspline_controller, "bspline_processor", None)
            if bspline_proc is not None:
                return bspline_proc
        return getattr(self.window, "bspline_processor", None)
    
    def load_airfoil_file(self) -> None:
        """Handle loading an airfoil data file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self.window,
            "Load Airfoil Data / BSP",
            "",
            "Airfoil/BSP Files (*.dat *.bsp);;Airfoil Data Files (*.dat);;BSP Files (*.bsp);;All Files (*)",
        )

        if not file_path:
            return

        # Clear only once a new file has actually been selected
        self.window.plot_widget.clear()

        self.window.file_panel.file_path_label.setText(os.path.basename(file_path))

        try:
            suffix = os.path.splitext(file_path)[1].lower()
            if suffix == ".bsp":
                if self._load_bsp_file(file_path):
                    self.processor.log_message.emit(
                        f"Successfully loaded BSP '{os.path.basename(file_path)}'."
                    )
                    if self.ui_state_controller:
                        self.ui_state_controller.update_button_states()
                else:
                    self.processor.log_message.emit(
                        f"Failed to load BSP '{os.path.basename(file_path)}'. Check file format and content."
                    )
            elif self.processor.load_airfoil_data_and_initialize_model(file_path):
                self.processor.log_message.emit(
                    f"Successfully loaded '{os.path.basename(file_path)}'."
                )
                
                # Reset UI state for new airfoil
                if self.ui_state_controller:
                    self.ui_state_controller.reset_ui_for_new_airfoil()
            else:
                self.processor.log_message.emit(
                    f"Failed to load '{os.path.basename(file_path)}'. Check file format and content."
                )
        except Exception as exc:  # pragma: no cover – unexpected error path
            self.processor.log_message.emit(
                f"An unexpected error occurred during file loading: {exc}"
            )

    def _load_bsp_file(self, file_path: str) -> bool:
        """Load a .bsp model and hydrate processor + GUI state for inspection."""
        bsp_data = load_bspline_from_bsp(file_path)
        bspline_proc = self._get_bspline_processor()
        if bspline_proc is None:
            self.processor.log_message.emit("Error: No B-spline processor available.")
            return False

        upper_cp = np.asarray(bsp_data.upper_control_points, dtype=float)
        lower_cp = np.asarray(bsp_data.lower_control_points, dtype=float)
        upper_knots = np.asarray(bsp_data.upper_knots, dtype=float)
        lower_knots = np.asarray(bsp_data.lower_knots, dtype=float)

        deg_upper = int(len(upper_knots) - len(upper_cp) - 1)
        deg_lower = int(len(lower_knots) - len(lower_cp) - 1)
        if deg_upper < 1 or deg_lower < 1:
            self.processor.log_message.emit(
                "Error: Invalid BSP degree inferred from knot/control-point counts."
            )
            return False

        try:
            upper_curve = interpolate.BSpline(upper_knots, upper_cp, deg_upper)
            lower_curve = interpolate.BSpline(lower_knots, lower_cp, deg_lower)
        except Exception as exc:
            self.processor.log_message.emit(f"Error: Could not construct BSP curves: {exc}")
            return False

        bspline_proc.reset_model_state()
        bspline_proc.upper_control_points = upper_cp
        bspline_proc.lower_control_points = lower_cp
        bspline_proc.upper_knot_vector = upper_knots
        bspline_proc.lower_knot_vector = lower_knots
        bspline_proc.upper_curve = upper_curve
        bspline_proc.lower_curve = lower_curve
        bspline_proc.degree_upper = deg_upper
        bspline_proc.degree_lower = deg_lower
        bspline_proc.degree = max(deg_upper, deg_lower)
        bspline_proc.fitted_degree = (deg_upper, deg_lower)
        bspline_proc.num_cp_upper = int(len(upper_cp))
        bspline_proc.num_cp_lower = int(len(lower_cp))
        bspline_proc.is_sharp_te = bool(np.allclose(upper_cp[-1], lower_cp[-1], atol=1e-12))
        bspline_proc.fitted = True

        # Prefer sibling DAT (same stem) as reference/source data for error calculations.
        # If unavailable, fall back to sampled points from the imported BSP curves.
        ref_dat_path = self._find_matching_dat_for_bsp(file_path)
        has_reference_data = False
        if ref_dat_path is not None:
            try:
                upper_data, lower_data, dat_name, blunt_te = load_airfoil_data(
                    ref_dat_path,
                    logger_func=lambda _msg: None,
                )
                has_reference_data = True
                self.processor.log_message.emit(
                    f"Loaded reference DAT for BSP comparison: '{os.path.basename(ref_dat_path)}'."
                )
                if dat_name:
                    self.processor.airfoil_name = dat_name
                self.processor._is_blunt_TE = blunt_te
            except Exception as exc:
                self.processor.log_message.emit(
                    f"Warning: Could not load matching DAT '{os.path.basename(ref_dat_path)}': {exc}. "
                    "Falling back to BSP-sampled reference data."
                )
                upper_data, lower_data = self._sample_bsp_curves(upper_curve, lower_curve)
                self.processor.airfoil_name = bsp_data.airfoil_name or os.path.splitext(os.path.basename(file_path))[0]
                self.processor._is_blunt_TE = not bspline_proc.is_sharp_te
        else:
            upper_data, lower_data = self._sample_bsp_curves(upper_curve, lower_curve)
            self.processor.airfoil_name = bsp_data.airfoil_name or os.path.splitext(os.path.basename(file_path))[0]
            self.processor._is_blunt_TE = not bspline_proc.is_sharp_te

        self.processor.upper_data = upper_data
        self.processor.lower_data = lower_data
        self.processor._last_plot_data = None
        self.processor.upper_te_tangent_vector, self.processor.lower_te_tangent_vector = (
            self.processor._calculate_te_tangent(
                self.processor.upper_data,
                self.processor.lower_data,
                config.DEFAULT_TE_VECTOR_POINTS,
            )
        )

        bspline_proc.upper_original_data = upper_data.copy()
        bspline_proc.lower_original_data = lower_data.copy()
        bspline_proc.error_reference_available = has_reference_data
        if not has_reference_data:
            bspline_proc.last_upper_max_error = None
            bspline_proc.last_upper_max_error_idx = None
            bspline_proc.last_lower_max_error = None
            bspline_proc.last_lower_max_error_idx = None

        if self.ui_state_controller:
            self.ui_state_controller._calculate_initial_thickness()
            self.window.optimizer_panel.upper_cp_label.setText(f"Upper CPs: {bspline_proc.num_cp_upper}")
            self.window.optimizer_panel.lower_cp_label.setText(f"Lower CPs: {bspline_proc.num_cp_lower}")
            fit_btn = getattr(self.window, "bspline_controller", None)
            if fit_btn is not None:
                fit_btn._update_fit_button_text()

        bspline_controller = getattr(self.window, "bspline_controller", None)
        if (
            bspline_controller is not None
            and has_reference_data
            and bspline_proc.upper_curve is not None
            and bspline_proc.lower_curve is not None
            and self.processor.upper_data is not None
            and self.processor.lower_data is not None
        ):
            try:
                _, upper_max_err, upper_max_err_idx, _ = bspline_controller.calculate_bspline_fitting_error(
                    bspline_proc.upper_curve,
                    self.processor.upper_data,
                    return_max_error=True,
                )
                _, lower_max_err, lower_max_err_idx, _ = bspline_controller.calculate_bspline_fitting_error(
                    bspline_proc.lower_curve,
                    self.processor.lower_data,
                    return_max_error=True,
                )
                bspline_proc.last_upper_max_error = upper_max_err
                bspline_proc.last_upper_max_error_idx = upper_max_err_idx
                bspline_proc.last_lower_max_error = lower_max_err
                bspline_proc.last_lower_max_error_idx = lower_max_err_idx
                self.processor.log_message.emit(
                    f"BSP error metrics updated. Upper max error: {upper_max_err:.6e}, "
                    f"Lower max error: {lower_max_err:.6e}"
                )
            except Exception as exc:
                self.processor.log_message.emit(f"Warning: Could not compute BSP error metrics: {exc}")

        if bspline_controller is not None:
            bspline_controller._update_plot_with_bsplines()
        else:
            self.processor.emit_plot_update(bspline_processor=bspline_proc, comb_bspline=None)

        return True

    def _sample_bsp_curves(self, upper_curve, lower_curve) -> tuple[np.ndarray, np.ndarray]:
        sample_count = max(200, int(config.PLOT_POINTS_PER_SURFACE))
        t_values = np.linspace(0.0, 1.0, sample_count)
        if len(t_values) > 0:
            t_values[-1] = min(t_values[-1], 1.0 - 1e-12)
        return upper_curve(t_values), lower_curve(t_values)

    def _find_matching_dat_for_bsp(self, bsp_path: str) -> str | None:
        bsp = os.path.abspath(bsp_path)
        base_dir = os.path.dirname(bsp)
        stem = os.path.splitext(os.path.basename(bsp))[0].lower()
        try:
            for name in os.listdir(base_dir):
                path = os.path.join(base_dir, name)
                if not os.path.isfile(path):
                    continue
                file_stem, ext = os.path.splitext(name)
                if ext.lower() == ".dat" and file_stem.lower() == stem:
                    return path
        except OSError:
            return None
        return None
    

    def export_dxf(self) -> None:
        """Export the current B-spline model as a DXF file."""
        # Check if B-spline is available and fitted
        bspline_proc = self._get_bspline_processor()
        bspline_fitted = False
        if bspline_proc is not None:
            try:
                bspline_fitted = bool(getattr(bspline_proc, "fitted", False))
            except Exception:
                bspline_fitted = False

        if not bspline_fitted or getattr(bspline_proc, "upper_control_points", None) is None or \
           getattr(bspline_proc, "lower_control_points", None) is None:
            self.processor.log_message.emit(
                "Error: B-spline model not available for export. Please fit B-spline first."
            )
            return

        # Use B-spline export
        self.export_bspline_dxf()
    
    def export_bspline_dxf(self) -> None:
        """Export the current B-spline model as a DXF file."""
        # Check if B-spline processor is available and fitted
        bspline_proc = self._get_bspline_processor()
        if bspline_proc is None or not getattr(bspline_proc, "fitted", False):
            self.processor.log_message.emit(
                "Error: B-spline model not available for export. Please fit B-spline first."
            )
            return

        try:
            chord_length_mm = float(
                self.window.airfoil_settings_panel.chord_length_input.text()
            )
        except ValueError:
            self.processor.log_message.emit(
                "Error: Invalid chord length. Please enter a number."
            )
            return

        export_mode = DXF_EXPORT_MODE_NURBS
        file_panel = getattr(self.window, "file_panel", None)
        if file_panel is not None:
            as_bezier_checkbox = getattr(file_panel, "export_dxf_as_bezier_checkbox", None)
            if as_bezier_checkbox is not None and as_bezier_checkbox.isChecked():
                export_mode = DXF_EXPORT_MODE_BEZIER

        dxf_doc = export_bspline_to_dxf(
            bspline_proc,
            chord_length_mm,
            self.processor.log_message.emit,
            export_mode=export_mode,
        )

        if not dxf_doc:
            self.processor.log_message.emit(
                "B-spline DXF export failed during document creation."
            )
            return

        default_filename = self._get_default_dxf_filename()
        file_path, _ = QFileDialog.getSaveFileName(
            self.window,
            "Save B-spline DXF File",
            default_filename,
            "DXF Files (*.dxf)",
        )
        if not file_path:
            self.processor.log_message.emit("B-spline DXF export cancelled by user.")
            return

        try:
            dxf_doc.saveas(file_path)
            self.processor.log_message.emit(
                f"B-spline DXF export successful to '{os.path.basename(file_path)}'."
            )
            self.processor.log_message.emit(
                "Note: For correct scale in CAD software, ensure import settings are configured for millimeters."
            )
        except IOError as exc:
            self.processor.log_message.emit(f"Could not save DXF file: {exc}")
    
    def _get_default_dxf_filename(self) -> str:
        """Return a safe default filename based on the loaded profile."""
        import re

        profile_name = getattr(self.processor, "airfoil_name", None)
        if profile_name:
            sanitized = re.sub(r"[^A-Za-z0-9\-_]+", "_", profile_name)
            if sanitized:
                return f"{sanitized}.dxf"
        return "airfoil.dxf" 

    def export_dat_file(self) -> None:
        """Export the current B-spline model as a high-resolution .dat file."""
        if not config.ENABLE_DAT_EXPORT:
            self.processor.log_message.emit("DAT export is disabled in config.")
            return

        # Get the number of points per surface from the UI
        try:
            points_per_surface = self.window.file_panel.points_per_surface_input.value()
        except ValueError:
            self.processor.log_message.emit(
                "Error: Invalid number of points. Please enter a valid number."
            )
            return

        # Get the current airfoil name
        airfoil_name = getattr(self.processor, "airfoil_name", "airfoil")
        if not airfoil_name:
            airfoil_name = "airfoil"

        # Check if B-spline is available and fitted
        bspline_proc = self._get_bspline_processor()
        try:
            bspline_fitted = bool(getattr(bspline_proc, "fitted", False))
        except Exception:
            bspline_fitted = False

        if not bspline_fitted or getattr(bspline_proc, "upper_curve", None) is None or \
           getattr(bspline_proc, "lower_curve", None) is None:
            self.processor.log_message.emit(
                "Error: B-spline model not available for export. Please fit B-spline first."
            )
            return

        try:
            t_values = np.linspace(0.0, 1.0, points_per_surface)
            if len(t_values) > 0:
                t_values[-1] = min(t_values[-1], 1.0 - 1e-12)
            upper_points = bspline_proc.upper_curve(t_values)
            lower_points = bspline_proc.lower_curve(t_values)

            # Default filename
            default_filename = self._get_default_dat_filename(f"{airfoil_name}_bspline")

            file_path, _ = QFileDialog.getSaveFileName(
                self.window,
                "Save B-spline High-Resolution .dat File",
                default_filename,
                "DAT Files (*.dat);;All Files (*)",
            )
            if not file_path:
                self.processor.log_message.emit(".dat export cancelled by user.")
                return

            export_airfoil_to_selig_format(upper_points, lower_points, airfoil_name, file_path)
            self.processor.log_message.emit(
                f"B-spline .dat export successful to '{os.path.basename(file_path)}'."
            )
            self.processor.log_message.emit(
                f"Exported {len(upper_points)} points per surface in Selig format."
            )
        except Exception as exc:
            self.processor.log_message.emit(
                f"Error during B-spline .dat export: {exc}"
            )

    def _get_default_dat_filename(self, airfoil_name: str) -> str:
        """Return a safe default filename for .dat export based on the loaded profile."""
        import re

        if airfoil_name:
            sanitized = re.sub(r"[^A-Za-z0-9\-_]+", "_", airfoil_name)
            if sanitized:
                return f"{sanitized}_highres.dat"
        return "airfoil_highres.dat"

    def export_bsp_file(self) -> None:
        """Export the current B-spline model as a .bsp file."""
        if not config.ENABLE_BSP_EXPORT:
            self.processor.log_message.emit("BSP export is disabled in config.")
            return

        bspline_proc = self._get_bspline_processor()
        try:
            bspline_fitted = bool(getattr(bspline_proc, "fitted", False))
        except Exception:
            bspline_fitted = False

        if not bspline_fitted:
            self.processor.log_message.emit(
                "Error: B-spline model not available for export. Please fit B-spline first."
            )
            return

        airfoil_name = getattr(self.processor, "airfoil_name", "airfoil") or "airfoil"
        default_filename = self._get_default_bsp_filename(f"{airfoil_name}_bspline")

        file_path, _ = QFileDialog.getSaveFileName(
            self.window,
            "Save B-spline .bsp File",
            default_filename,
            "BSP Files (*.bsp);;All Files (*)",
        )
        if not file_path:
            self.processor.log_message.emit(".bsp export cancelled by user.")
            return

        ok = export_bspline_to_bsp(
            bspline_proc,
            airfoil_name,
            file_path,
            self.processor.log_message.emit,
        )
        if ok:
            self.processor.log_message.emit(
                f"B-spline .bsp export successful to '{os.path.basename(file_path)}'."
            )
        else:
            self.processor.log_message.emit("Error during B-spline .bsp export.")

    def _get_default_bsp_filename(self, airfoil_name: str) -> str:
        """Return a safe default filename for .bsp export based on the loaded profile."""
        import re

        if airfoil_name:
            sanitized = re.sub(r"[^A-Za-z0-9\-_]+", "_", airfoil_name)
            if sanitized:
                return f"{sanitized}.bsp"
        return "airfoil.bsp"
