"""Interactive airfoil visualisation widget.

This is the *actual* implementation of :class:`AirfoilPlotWidget`, moved
from ``gui/airfoil_plot_widget.py`` into the new widgets package as part
of the GUI refactor (2025-07-17).
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt

from core.config import (
    PLOT_POINTS_PER_SURFACE,
)


KEY_ORIGINAL_DATA = "Original Data"
KEY_BSPLINE_CURVES = "B-spline Curves"
KEY_BSPLINE_CONTROL_POINTS = "B-spline Control Points"
KEY_BSPLINE_KNOT_MARKERS = "B-spline Knot Markers"
KEY_BSPLINE_COMB = "B-spline Curvature Comb"
KEY_BSPLINE_COMB_TIPS = "B-spline Comb Tips Polyline"
KEY_TE_TANGENT_UPPER = "TE Tangent (Upper)"
KEY_TE_TANGENT_LOWER = "TE Tangent (Lower)"
KEY_BSPLINE_ERROR_TEXT = "B-spline Error Text"
KEY_BSPLINE_MAX_ERROR_MARKERS = "B-spline Max. Error Markers"
KEY_GEOMETRY_METRICS_TEXT = "Geometry Metrics Text"


class AirfoilPlotWidget(pg.PlotWidget):
    """Custom `pyqtgraph.PlotWidget` tailored for airfoil visualisation."""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setParent(parent)
        self.setSizePolicy(
            pg.QtWidgets.QSizePolicy.Policy.Expanding,
            pg.QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.updateGeometry()

        pg.setConfigOptions(antialias=True)
        self.setAspectLocked(True)
        self.showGrid(x=True, y=True)
        self.addLegend(offset=(30, 10))
        self.setLabel("bottom", "x/c (Chord)")
        self.setLabel("left", "y/c (Chord)")

        self.plot_items: dict[str, object] = {}
        self._item_visibility_by_name: dict[str, bool] = {}
        self._first_plot_done = False
        self.getViewBox().sigRangeChanged.connect(self._update_error_text_positions)

    def plot_airfoil(
        self,
        upper_data,
        lower_data,
        upper_te_tangent_vector=None,
        lower_te_tangent_vector=None,
        chord_length_mm=None,
        geometry_metrics=None,
        bspline_upper_curve=None,
        bspline_lower_curve=None,
        bspline_upper_control_points=None,
        bspline_lower_control_points=None,
        bspline_upper_max_error=None,
        bspline_lower_max_error=None,
        bspline_upper_max_error_idx=None,
        bspline_lower_max_error_idx=None,
        comb_bspline=None,
        bspline_is_blunt=False,
        bspline_num_cp_upper=None,
        bspline_num_cp_lower=None,
    ):
        """Render current airfoil and optional B-spline layers."""
        palette = self._build_palette()
        self._capture_visibility_state()
        self._reset_plot_canvas()

        self._plot_original_data(upper_data, lower_data, palette)
        self._plot_bspline_layers(
            bspline_upper_curve,
            bspline_lower_curve,
            bspline_upper_control_points,
            bspline_lower_control_points,
            bspline_is_blunt,
            palette,
        )
        self._plot_curvature_comb(comb_bspline, palette)
        self._plot_te_tangent_vectors(
            upper_data,
            lower_data,
            upper_te_tangent_vector,
            lower_te_tangent_vector,
            palette,
        )
        self._plot_error_annotations(
            upper_data,
            lower_data,
            chord_length_mm,
            bspline_upper_max_error,
            bspline_lower_max_error,
            bspline_upper_max_error_idx,
            bspline_lower_max_error_idx,
            bspline_num_cp_upper,
            bspline_num_cp_lower,
        )
        self._plot_geometry_metrics(geometry_metrics)

        self._update_error_text_positions()
        self._restore_visibility_state()
        self._set_initial_view_range_if_needed(upper_data, lower_data)

    def _build_palette(self) -> dict[str, object]:
        return {
            "original_data": (100, 149, 237, 220),
            "te_tangent_upper": pg.mkPen((220, 20, 60), width=2, style=Qt.PenStyle.SolidLine),
            "te_tangent_lower": pg.mkPen((138, 43, 226), width=2, style=Qt.PenStyle.SolidLine),
            "comb": pg.mkPen((150, 150, 150), width=1.5),
            "comb_outline": pg.mkPen((255, 215, 0), width=2, style=Qt.PenStyle.DotLine),
            "curve_sharp_a": pg.mkPen((255, 0, 127), width=2.5),
            "curve_sharp_b": pg.mkPen((0, 191, 255), width=2.5),
            "control_sharp_line": pg.mkPen((255, 20, 147), width=1.5, style=Qt.PenStyle.DashLine),
            "control_sharp_brush": pg.mkBrush((255, 20, 147, 220)),
            "control_sharp_pen": pg.mkPen((255, 20, 147), width=1.5),
            "curve_blunt_a": pg.mkPen((199, 21, 133), width=3.0),
            "curve_blunt_b": pg.mkPen((0, 139, 139), width=3.0),
            "control_blunt_line": pg.mkPen((148, 0, 211), width=1.5, style=Qt.PenStyle.DotLine),
            "control_blunt_brush": pg.mkBrush((148, 0, 211, 220)),
            "control_blunt_pen": pg.mkPen((148, 0, 211), width=1.5),
            "knot_pen": pg.mkPen((255, 215, 0), width=1.5),
            "knot_brush": pg.mkBrush((255, 215, 0, 220)),
        }

    def _reset_plot_canvas(self) -> None:
        self.clear()
        self.addLegend(offset=(30, 10))
        self.plot_items = {}

    def _register_item(self, key: str, item, *, is_group: bool = False) -> None:
        if is_group:
            if key not in self.plot_items:
                self.plot_items[key] = []
            self.plot_items[key].append(item)
            return
        self.plot_items[key] = item

    def _plot_original_data(self, upper_data, lower_data, palette: dict[str, object]) -> None:
        all_original_data = np.concatenate([upper_data, lower_data])
        item = self.plot(
            all_original_data[:, 0],
            all_original_data[:, 1],
            pen=None,
            symbol="o",
            symbolSize=5,
            symbolBrush=pg.mkBrush(palette["original_data"]),
            name=KEY_ORIGINAL_DATA,
        )
        self._register_item(KEY_ORIGINAL_DATA, item)

    def _plot_bspline_layers(
        self,
        bspline_upper_curve,
        bspline_lower_curve,
        bspline_upper_control_points,
        bspline_lower_control_points,
        bspline_is_blunt: bool,
        palette: dict[str, object],
    ) -> None:
        if bspline_upper_curve is None and bspline_lower_curve is None:
            return

        self.plot_items[KEY_BSPLINE_CURVES] = []
        self.plot_items[KEY_BSPLINE_CONTROL_POINTS] = []
        self.plot_items[KEY_BSPLINE_KNOT_MARKERS] = []

        if bspline_is_blunt:
            curve_pens = [palette["curve_blunt_a"], palette["curve_blunt_b"]]
            control_pen = palette["control_blunt_line"]
            control_brush = palette["control_blunt_brush"]
            control_symbol_pen = palette["control_blunt_pen"]
            curve_name_prefix = "Thickened B-spline"
        else:
            curve_pens = [palette["curve_sharp_a"], palette["curve_sharp_b"]]
            control_pen = palette["control_sharp_line"]
            control_brush = palette["control_sharp_brush"]
            control_symbol_pen = palette["control_sharp_pen"]
            curve_name_prefix = "B-spline"

        for curve, surface_name in ((bspline_upper_curve, "Upper"), (bspline_lower_curve, "Lower")):
            if curve is None:
                continue
            self._plot_bspline_curve_spans(curve, surface_name, curve_pens, curve_name_prefix)
            self._plot_bspline_knot_markers(curve, surface_name, palette)

        if bspline_upper_control_points is not None:
            item = self.plot(
                bspline_upper_control_points[:, 0],
                bspline_upper_control_points[:, 1],
                pen=control_pen,
                symbol="s",
                symbolSize=6,
                symbolBrush=control_brush,
                symbolPen=control_symbol_pen,
                name=f"{curve_name_prefix} Control Points Upper",
            )
            self._register_item(KEY_BSPLINE_CONTROL_POINTS, item, is_group=True)

        if bspline_lower_control_points is not None:
            item = self.plot(
                bspline_lower_control_points[:, 0],
                bspline_lower_control_points[:, 1],
                pen=control_pen,
                symbol="s",
                symbolSize=6,
                symbolBrush=control_brush,
                symbolPen=control_symbol_pen,
                name=f"{curve_name_prefix} Control Points Lower",
            )
            self._register_item(KEY_BSPLINE_CONTROL_POINTS, item, is_group=True)

    def _plot_bspline_curve_spans(self, curve, surface_name: str, curve_pens: list[object], curve_name_prefix: str) -> None:
        knots = curve.t
        k = curve.k
        active_knots = knots[k : len(knots) - k]
        unique_knots = np.unique(active_knots)

        for idx in range(len(unique_knots) - 1):
            t_start = unique_knots[idx]
            t_end = unique_knots[idx + 1]
            num_points_span = max(5, PLOT_POINTS_PER_SURFACE // (len(unique_knots) - 1))
            t_vals = np.linspace(t_start, t_end, num_points_span)
            if t_end == unique_knots[-1]:
                t_vals[-1] = min(t_vals[-1], unique_knots[-1] - 1e-12)

            span_points = curve(t_vals)
            name = f"{curve_name_prefix} {surface_name}" if idx == 0 else None
            item = self.plot(
                span_points[:, 0],
                span_points[:, 1],
                pen=curve_pens[idx % 2],
                antialias=True,
                name=name,
            )
            self._register_item(KEY_BSPLINE_CURVES, item, is_group=True)

    def _plot_bspline_knot_markers(self, curve, surface_name: str, palette: dict[str, object]) -> None:
        knots = curve.t
        k = curve.k
        active_knots = knots[k : len(knots) - k]
        unique_knots = np.unique(active_knots)
        knot_points = curve(unique_knots)
        item = self.plot(
            knot_points[:, 0],
            knot_points[:, 1],
            pen=None,
            symbol="o",
            symbolSize=4,
            symbolBrush=palette["knot_brush"],
            symbolPen=palette["knot_pen"],
            name=f"Knots ({surface_name})" if surface_name == "Upper" else None,
        )
        self._register_item(KEY_BSPLINE_KNOT_MARKERS, item, is_group=True)

    def _plot_curvature_comb(self, comb_bspline, palette: dict[str, object]) -> None:
        if comb_bspline is None or not any(comb_bspline):
            return

        all_hairs: list[np.ndarray] = []
        all_tip_segments: list[np.ndarray] = []
        for comb_segments in comb_bspline:
            if not comb_segments:
                continue
            all_hairs.extend(comb_segments)

            comb_tips = np.array([hair[1] for hair in comb_segments])
            for idx in range(len(comb_tips) - 1):
                p1 = comb_tips[idx]
                p2 = comb_tips[idx + 1]
                if p1[1] != 0 or p2[1] != 0:
                    all_tip_segments.append(p1)
                    all_tip_segments.append(p2)

        if not all_hairs:
            return

        comb_array = np.concatenate(all_hairs)
        main_item = self.plot(
            comb_array[:, 0],
            comb_array[:, 1],
            pen=palette["comb"],
            name=KEY_BSPLINE_COMB,
            connect="pairs",
        )
        self._register_item(KEY_BSPLINE_COMB, main_item, is_group=True)

        if not all_tip_segments:
            return

        tips_array = np.array(all_tip_segments)
        tips_item = self.plot(
            tips_array[:, 0],
            tips_array[:, 1],
            pen=palette["comb_outline"],
            connect="pairs",
        )
        self._register_item(KEY_BSPLINE_COMB_TIPS, tips_item, is_group=True)

        main_item.visibleChanged.connect(
            lambda: tips_item.setVisible(main_item.isVisible())
        )
        tips_item.setVisible(main_item.isVisible())

    def _plot_te_tangent_vectors(
        self,
        upper_data,
        lower_data,
        upper_te_tangent_vector,
        lower_te_tangent_vector,
        palette: dict[str, object],
    ) -> None:
        if upper_te_tangent_vector is None or lower_te_tangent_vector is None:
            return
        if len(upper_data) == 0 or len(lower_data) == 0:
            return

        tangent_length = 0.05
        upper_te_point = upper_data[-1]
        lower_te_point = lower_data[-1]

        tangent_start_upper = upper_te_point - upper_te_tangent_vector * tangent_length
        tangent_end_upper = upper_te_point + upper_te_tangent_vector * tangent_length
        upper_item = self.plot(
            [tangent_start_upper[0], tangent_end_upper[0]],
            [tangent_start_upper[1], tangent_end_upper[1]],
            pen=palette["te_tangent_upper"],
            name=KEY_TE_TANGENT_UPPER,
        )
        self._register_item(KEY_TE_TANGENT_UPPER, upper_item)

        tangent_start_lower = lower_te_point - lower_te_tangent_vector * tangent_length
        tangent_end_lower = lower_te_point + lower_te_tangent_vector * tangent_length
        lower_item = self.plot(
            [tangent_start_lower[0], tangent_end_lower[0]],
            [tangent_start_lower[1], tangent_end_lower[1]],
            pen=palette["te_tangent_lower"],
            name=KEY_TE_TANGENT_LOWER,
        )
        self._register_item(KEY_TE_TANGENT_LOWER, lower_item)

    def _plot_error_annotations(
        self,
        upper_data,
        lower_data,
        chord_length_mm,
        max_bspline_upper,
        max_bspline_lower,
        max_bspline_upper_idx,
        max_bspline_lower_idx,
        bspline_num_cp_upper,
        bspline_num_cp_lower,
    ) -> None:
        if chord_length_mm is not None and (max_bspline_upper is not None or max_bspline_lower is not None):
            error_html = '<div style="text-align: right; color: #FF6B6B; font-size: 10pt;">'
            error_html += self._build_cp_info_html(bspline_num_cp_upper, bspline_num_cp_lower)

            if max_bspline_upper is not None and max_bspline_lower is not None:
                max_upper_mm = max_bspline_upper * chord_length_mm
                max_lower_mm = max_bspline_lower * chord_length_mm
                error_html += (
                    "B-spline Max Error (Upper/Lower): "
                    f"{max_bspline_upper:.2e} ({max_upper_mm:.3f} mm) / "
                    f"{max_bspline_lower:.2e} ({max_lower_mm:.3f} mm)"
                )
            elif max_bspline_upper is not None:
                max_upper_mm = max_bspline_upper * chord_length_mm
                error_html += f"B-spline Max Error (Upper): {max_bspline_upper:.2e} ({max_upper_mm:.3f} mm)"
            elif max_bspline_lower is not None:
                max_lower_mm = max_bspline_lower * chord_length_mm
                error_html += f"B-spline Max Error (Lower): {max_bspline_lower:.2e} ({max_lower_mm:.3f} mm)"

            error_html += "</div>"
            text_item = pg.TextItem(html=error_html, anchor=(1, 1))
            self.addItem(text_item)
            self._register_item(KEY_BSPLINE_ERROR_TEXT, text_item)

        marker_x: list[float] = []
        marker_y: list[float] = []
        if max_bspline_upper_idx is not None and 0 <= max_bspline_upper_idx < len(upper_data):
            pt = upper_data[max_bspline_upper_idx]
            marker_x.append(pt[0])
            marker_y.append(pt[1])
        if max_bspline_lower_idx is not None and 0 <= max_bspline_lower_idx < len(lower_data):
            pt = lower_data[max_bspline_lower_idx]
            marker_x.append(pt[0])
            marker_y.append(pt[1])

        if marker_x:
            marker_item = self.plot(
                marker_x,
                marker_y,
                pen=None,
                symbol="s",
                symbolSize=16,
                symbolBrush=None,
                symbolPen=pg.mkPen((255, 165, 0), width=3),
                name=KEY_BSPLINE_MAX_ERROR_MARKERS,
            )
            self._register_item(KEY_BSPLINE_MAX_ERROR_MARKERS, marker_item)

    @staticmethod
    def _build_cp_info_html(bspline_num_cp_upper, bspline_num_cp_lower) -> str:
        if bspline_num_cp_upper is not None and bspline_num_cp_lower is not None:
            if bspline_num_cp_upper == bspline_num_cp_lower:
                return f"Control Points: {bspline_num_cp_upper}<br/>"
            return f"Control Points: Upper={bspline_num_cp_upper}, Lower={bspline_num_cp_lower}<br/>"
        if bspline_num_cp_upper is not None:
            return f"Control Points (Upper): {bspline_num_cp_upper}<br/>"
        if bspline_num_cp_lower is not None:
            return f"Control Points (Lower): {bspline_num_cp_lower}<br/>"
        return ""

    def _plot_geometry_metrics(self, geometry_metrics) -> None:
        if not geometry_metrics:
            return

        t_pct = geometry_metrics.get("thickness_percent")
        c_pct = geometry_metrics.get("camber_percent")
        wedge = geometry_metrics.get("te_wedge_angle_deg")
        le_r = geometry_metrics.get("le_radius_percent")
        x_t = geometry_metrics.get("x_t_percent", 0.0)
        x_c = geometry_metrics.get("x_c_percent", 0.0)
        geo_html = (
            '<div style="text-align: right; color: #F0E68C; font-size: 10pt;">'
            f"Thickness: {t_pct:.2f}% (x: {x_t:.1f}%)<br/>"
            f"Camber: {c_pct:.2f}% (x: {x_c:.1f}%)<br/>"
            f"TE wedge: {wedge:.2f}&deg;<br/>"
            f"LE Radius: {le_r:.3f}%"
            "</div>"
        )
        geo_item = pg.TextItem(html=geo_html, anchor=(1, 1))
        self.addItem(geo_item)
        self._register_item(KEY_GEOMETRY_METRICS_TEXT, geo_item)

    def _set_initial_view_range_if_needed(self, upper_data, lower_data) -> None:
        if self._first_plot_done:
            return

        all_x = np.concatenate([upper_data[:, 0], lower_data[:, 0]])
        all_y = np.concatenate([upper_data[:, 1], lower_data[:, 1]])

        x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
        y_min, y_max = float(np.min(all_y)), float(np.max(all_y))

        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.2

        self.setXRange(x_min - x_padding, x_max + x_padding)
        self.setYRange(y_min - y_padding, y_max + y_padding)
        self._first_plot_done = True

    def _iter_plot_items(self):
        """Yield all plot-backed items currently tracked in `self.plot_items`."""
        for entry in self.plot_items.values():
            if isinstance(entry, list):
                for item in entry:
                    yield item
            elif entry is not None:
                yield entry

    @staticmethod
    def _item_name(item) -> str | None:
        """Return the plot item legend name when available."""
        if hasattr(item, "opts") and isinstance(item.opts, dict):
            name = item.opts.get("name")
            if name:
                return str(name)
        return None

    def _capture_visibility_state(self) -> None:
        """Capture visibility per named plot item before clearing/replotting."""
        for item in self._iter_plot_items():
            name = self._item_name(item)
            if name and hasattr(item, "isVisible"):
                self._item_visibility_by_name[name] = bool(item.isVisible())

    def _restore_visibility_state(self) -> None:
        """Reapply captured visibility to newly created named items."""
        if not self._item_visibility_by_name:
            return
        for item in self._iter_plot_items():
            name = self._item_name(item)
            if name and name in self._item_visibility_by_name and hasattr(item, "setVisible"):
                item.setVisible(self._item_visibility_by_name[name])

    def _update_error_text_positions(self):
        """Keep error text anchored to the top-right corner on zoom/pan."""
        vb = self.getViewBox()
        if not vb:
            return

        x_range, y_range = vb.viewRange()
        x_padding = (x_range[1] - x_range[0]) * 0.04
        y_padding = (y_range[1] - y_range[0]) * 0.08

        top_right_x = x_range[1] - x_padding
        current_y = y_range[1] - y_padding

        text_bspline = self.plot_items.get(KEY_BSPLINE_ERROR_TEXT)
        text_geo = self.plot_items.get(KEY_GEOMETRY_METRICS_TEXT)

        y_offset = (y_range[1] - y_range[0]) * 0.06
        if text_bspline:
            text_bspline.setPos(top_right_x, current_y)
            current_y -= y_offset
        if text_geo:
            bottom_right_y = y_range[0] + (y_range[1] - y_range[0]) * 0.02
            text_geo.setPos(top_right_x, bottom_right_y)
