import numpy as np
from PySide6.QtCore import QObject, Signal
import logging

from core import config
from utils.data_loader import load_airfoil_data



class SignalLogHandler(logging.Handler):
    """A logging handler that emits a Qt signal."""
    def __init__(self, signal_emitter):
        super().__init__()
        self.signal_emitter = signal_emitter

    def emit(self, record):
        msg = self.format(record)
        self.signal_emitter.emit(msg)

class AirfoilProcessor(QObject):
    """
    Acts as a bridge between the GUI and the CoreProcessor.
    It holds an instance of the CoreProcessor and uses Qt Signals
    to communicate with the GUI.
    """
    log_message = Signal(str)
    plot_update_requested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Core airfoil data
        self.upper_data = None
        self.lower_data = None
        self.upper_display_reference_data = None
        self.lower_display_reference_data = None
        self.upper_te_tangent_vector = None
        self.lower_te_tangent_vector = None
        self._last_plot_data = None # Cache for the last plot data dictionary
        self._is_blunt_TE = False # True if original airfoil has thickened TE


    def load_airfoil_data_and_initialize_model(self, file_path):
        """
        Loads airfoil data and initializes the AirfoilModel.
        Resets internal flags and state.
        """
        self._last_plot_data = None
        self.upper_data = None
        self.lower_data = None
        self.upper_display_reference_data = None
        self.lower_display_reference_data = None
        self.upper_te_tangent_vector = None
        self.lower_te_tangent_vector = None
        self._is_blunt_TE = False

        try:
            upper, lower, airfoil_name, blunt_te = load_airfoil_data(file_path, logger_func=self.log_message.emit)
            self.upper_data = upper
            self.lower_data = lower
            self._load_display_reference_data(file_path, upper, lower)
            self.airfoil_name = airfoil_name
            self._is_blunt_TE = blunt_te
            # Recalculate TE tangent vectors using configured default
            te_vector_points = config.DEFAULT_TE_VECTOR_POINTS
            self.upper_te_tangent_vector, self.lower_te_tangent_vector = self._calculate_te_tangent(
                self.upper_data, self.lower_data, te_vector_points)
            self.log_message.emit("Airfoil data loaded.")
            self._request_plot_update()
            return True
        except Exception as e:
            self.log_message.emit(f"Failed to load or initialize airfoil data: {e}")
            return False

    def _load_display_reference_data(self, file_path, fallback_upper, fallback_lower) -> None:
        """Keep the non-repaneled normalized input as the visual reference."""
        try:
            upper_ref, lower_ref, _name, _blunt_te = load_airfoil_data(
                file_path,
                logger_func=lambda _msg: None,
                repanel_input=False,
            )
            self.upper_display_reference_data = upper_ref
            self.lower_display_reference_data = lower_ref
        except Exception as exc:
            self.log_message.emit(
                f"Warning: Could not load non-repaneled visual reference: {exc}"
            )
            self.upper_display_reference_data = fallback_upper.copy()
            self.lower_display_reference_data = fallback_lower.copy()

    def is_trailing_edge_thickened(self):
        """Returns True if the loaded airfoil has a thickened trailing edge."""
        return self._is_blunt_TE

    def _request_plot_update(self):
        """Emits a signal to request a plot update with current model data from the core."""
        if self.upper_data is None or self.lower_data is None:
            self.log_message.emit("No airfoil data available to plot.")
            return

        self.emit_plot_update()

    def update_plot(self) -> None:
        """Request a plot update with the current airfoil data (without B-spline)."""
        self._request_plot_update()

    def build_plot_payload(
        self,
        *,
        bspline_processor=None,
        comb_bspline=None,
    ) -> dict:
        """Build a complete plot payload from current core state and optional B-spline state."""
        display_upper = self.upper_display_reference_data
        display_lower = self.lower_display_reference_data
        if display_upper is None or display_lower is None:
            display_upper = self.upper_data
            display_lower = self.lower_data

        plot_data = {
            'upper_data': display_upper,
            'lower_data': display_lower,
            'upper_te_tangent_vector': self.upper_te_tangent_vector,
            'lower_te_tangent_vector': self.lower_te_tangent_vector,
            'geometry_metrics': None,
        }

        if bspline_processor is None:
            return plot_data

        plot_data.update(
            {
                'bspline_upper_curve': bspline_processor.upper_curve,
                'bspline_lower_curve': bspline_processor.lower_curve,
                'bspline_upper_control_points': bspline_processor.upper_control_points,
                'bspline_lower_control_points': bspline_processor.lower_control_points,
                'comb_bspline': comb_bspline,
                'bspline_is_blunt': not bspline_processor.is_sharp_te,
                'bspline_num_cp_upper': bspline_processor.num_cp_upper,
                'bspline_num_cp_lower': bspline_processor.num_cp_lower,
                'bspline_fourth_difference_data': (
                    bspline_processor.calculate_control_point_fourth_difference_data()
                    if bool(getattr(config, "SHOW_CP_FOURTH_DIFFERENCES", False))
                    else None
                ),
            }
        )

        if bool(getattr(bspline_processor, "error_reference_available", False)):
            plot_data['bspline_upper_max_error'] = bspline_processor.last_upper_max_error
            plot_data['bspline_upper_max_error_idx'] = bspline_processor.last_upper_max_error_idx
            plot_data['bspline_lower_max_error'] = bspline_processor.last_lower_max_error
            plot_data['bspline_lower_max_error_idx'] = bspline_processor.last_lower_max_error_idx

        return plot_data

    def emit_plot_update(
        self,
        *,
        bspline_processor=None,
        comb_bspline=None,
    ) -> None:
        """Emit a plot update request from a centrally generated payload."""
        plot_data = self.build_plot_payload(
            bspline_processor=bspline_processor,
            comb_bspline=comb_bspline,
        )
        self._last_plot_data = plot_data.copy()
        self.plot_update_requested.emit(plot_data)

    def _calculate_te_tangent(self, upper_data, lower_data, te_vector_points):
        """
        Calculate trailing edge tangent vectors for upper and lower surfaces using the last N points.
        Returns (upper_te_tangent_vector, lower_te_tangent_vector)
        """
        def tangent(data, n):
            # Use the last n points to estimate the tangent at the trailing edge
            if n < 2 or len(data) < n:
                n = min(3, len(data))
            pts = data[-n:]
            dx = pts[-1, 0] - pts[0, 0]
            dy = pts[-1, 1] - pts[0, 1]
            norm = np.hypot(dx, dy)
            if norm == 0:
                return np.array([1.0, 0.0])
            return np.array([dx, dy]) / norm
        upper_te_tangent = tangent(upper_data, te_vector_points)
        lower_te_tangent = tangent(lower_data, te_vector_points)
        return upper_te_tangent, lower_te_tangent
