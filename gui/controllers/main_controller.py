"""Main controller for the Airfoil Fitter GUI.

Orchestrates the other controllers and handles signal routing between GUI and processor.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject

from core.airfoil_processor import AirfoilProcessor
from gui.main_window import MainWindow

from .file_controller import FileController
from .optimization_controller import OptimizationController
from .ui_state_controller import UIStateController
from .bspline_controller import BSplineController


class MainController(QObject):
    """Main controller that orchestrates all other controllers and handles signal routing."""

    def __init__(self, window: MainWindow):
        super().__init__(window)

        self.window = window
        self.processor = AirfoilProcessor()
        
        # Initialize sub-controllers
        self.ui_state_controller = UIStateController(self.processor, self.window)
        self.file_controller = FileController(self.processor, self.window, self.ui_state_controller)
        self.optimization_controller = OptimizationController(self.processor, self.window, self.ui_state_controller)
        self.bspline_controller = BSplineController(self.processor, self.window)
        
        # Store B-spline controller in window for access by other controllers
        self.window.bspline_controller = self.bspline_controller

        # ------------------------------------------------------------------
        # Wire up processor signals
        # ------------------------------------------------------------------
        self.processor.log_message.connect(self.window.status_log.append)
        self.processor.plot_update_requested.connect(self._update_plot_from_processor)

        # ------------------------------------------------------------------
        # Connect widget signals -> controller slots
        # ------------------------------------------------------------------
        self._connect_signals()

        # ------------------------------------------------------------------
        # Initial UI state
        # ------------------------------------------------------------------
        self.ui_state_controller.update_comb_labels()
        self.ui_state_controller.update_button_states()

        self.processor.log_message.emit("Application started. Load an airfoil .dat file to begin.")

    def _connect_signals(self) -> None:
        """Connect all GUI signals to their respective controller methods."""
        # File operations
        fp = self.window.file_panel
        fp.load_button.clicked.connect(self.file_controller.load_airfoil_file)
        fp.export_dxf_button.clicked.connect(self.file_controller.export_dxf)

        fp.export_dat_button.clicked.connect(self.file_controller.export_dat_file)

        # Optimization operations
        opt = self.window.optimizer_panel
        opt.fit_bspline_button.clicked.connect(self.bspline_controller.fit_bspline)
        opt.upper_insert_btn.clicked.connect(lambda: self.bspline_controller.insert_knot('upper'))
        opt.lower_insert_btn.clicked.connect(lambda: self.bspline_controller.insert_knot('lower'))
        
        # Parameter changes that trigger re-fit (only if already fitted)
        opt.bspline_degree_spin.valueChanged.connect(self.bspline_controller.refit_if_fitted)
        opt.smoothness_penalty_spin.valueChanged.connect(self.bspline_controller.refit_if_fitted)
        opt.g2_checkbox.toggled.connect(self.bspline_controller.refit_if_fitted)
        opt.g3_checkbox.toggled.connect(self.bspline_controller.refit_if_fitted)
        opt.enforce_te_tangency_checkbox.toggled.connect(self.bspline_controller.refit_if_fitted)
        
        # TE vector points dropdown - always updates TE vectors, refits only if tangency enabled and fitted
        opt.te_vector_points_combo.currentIndexChanged.connect(self.bspline_controller.handle_te_vector_points_changed)


        # Airfoil settings
        airfoil = self.window.airfoil_settings_panel
        airfoil.toggle_thickening_button.clicked.connect(self.ui_state_controller.handle_toggle_thickening)
        airfoil.te_thickness_input.textChanged.connect(self.ui_state_controller.handle_thickness_input_changed)

        # Comb parameters
        comb = self.window.comb_panel
        comb.comb_scale_slider.valueChanged.connect(self.ui_state_controller.handle_comb_params_changed)
        comb.comb_density_slider.valueChanged.connect(self.ui_state_controller.handle_comb_params_changed)

    def _update_plot_from_processor(self, plot_data: dict[str, Any]) -> None:
        """Receive plot data from the processor and forward to the widget."""
        # Cache so we can recompute comb later
        self._last_plot_data = plot_data.copy()

        try:
            chord_length_mm = float(self.window.airfoil_settings_panel.chord_length_input.text())
        except Exception:
            chord_length_mm = None

        # Clear the plot
        self.window.plot_widget.clear()
        
        # Remove chord_length_mm from plot_data to avoid duplicate parameter
        plot_data_without_chord = {k: v for k, v in plot_data.items() if k != 'chord_length_mm'}
        
        self.window.plot_widget.plot_airfoil(
            **plot_data_without_chord,
            chord_length_mm=chord_length_mm,
        )

        # Update button states
        self.ui_state_controller.update_button_states() 