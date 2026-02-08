"""Widget holding settings related to the B-spline optimiser."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QCheckBox,
    QPushButton,
    QWidget,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
)

from core import config


class OptimizerSettingsWidget(QGroupBox):
    """Panel exposing parameters for the B-spline airfoil optimiser."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Optimizer Settings", parent)

        # --- Inputs ------------------------------------------------------
        # TE Vector Points dropdown
        self.te_vector_points_combo = QComboBox()
        self.te_vector_points_combo.addItems([str(i) for i in range(2, 6)])  # 2-5
        self.te_vector_points_combo.setCurrentText(str(config.DEFAULT_TE_VECTOR_POINTS))
        self.te_vector_points_combo.setFixedWidth(80)


        # Enforce G2 at leading edge
        self.g2_checkbox = QCheckBox("G2")
        self.g2_checkbox.setToolTip(
            "G2 continuity ensures smooth curvature transition at the leading edge.\n"
            "When enabled: Both surfaces share the same leading edge radius.\n"
            "When disabled: Only G1 (tangent) continuity is enforced."
        )
        
        # Enforce G3 at leading edge
        self.g3_checkbox = QCheckBox("G3")
        self.g3_checkbox.setToolTip(
            "G3 continuity ensures smooth curvature derivative transition at the leading edge.\n"
            "When enabled: Both surfaces share the same rate of change of curvature.\n"
            "Note: G3 requires G2 to be enabled. Enabling G3 will automatically enable G2."
        )


        # Enforce TE vector tangency
        self.enforce_te_tangency_checkbox = QCheckBox("TE tangency")
        self.enforce_te_tangency_checkbox.setChecked(False)  # Default to disabled
        self.enforce_te_tangency_checkbox.setToolTip(
            "When enabled: B-splines are constrained to be tangent to the computed trailing edge vectors.\n"
            "When disabled: Only endpoint constraints are applied, allowing better fit for some airfoils."
        )

        # Initial control point count
        self.initial_cp_label = QLabel("Initial CP count:")
        self.initial_cp_spin = QSpinBox()
        self.initial_cp_spin.setMaximum(200)
        self.initial_cp_spin.setValue(config.DEFAULT_BSPLINE_CP)
        self.initial_cp_spin.setToolTip(
            "Initial number of control points per surface. Must be at least degree + 1."
        )

        # B-spline settings (new layout for knots)
        self.upper_cp_label = QLabel("Upper CPs: -")
        self.upper_insert_btn = QPushButton("+")
        self.upper_insert_btn.setFixedWidth(100)
        self.upper_insert_btn.setToolTip("Insert a knot on the upper surface.")

        self.lower_cp_label = QLabel("Lower CPs: -")
        self.lower_insert_btn = QPushButton("+")
        self.lower_insert_btn.setFixedWidth(100)
        self.lower_insert_btn.setToolTip("Insert a knot on the lower surface.")


        # B-spline degree setting
        self.bspline_degree_label = QLabel("Degree:")
        self.bspline_degree_spin = QSpinBox()
        self.bspline_degree_spin.setMinimum(4)
        self.bspline_degree_spin.setMaximum(12)
        self.bspline_degree_spin.setValue(config.DEFAULT_BSPLINE_DEGREE)
        self.bspline_degree_spin.setToolTip(
            "The degree of the B-spline curves. Higher degree allows for smoother curves (G3+) but may be less stable."
        )

        # Smoothness penalty setting
        self.smoothness_penalty_label = QLabel("Smoothness:")
        self.smoothness_penalty_spin = QDoubleSpinBox()
        self.smoothness_penalty_spin.setMinimum(0.0)
        self.smoothness_penalty_spin.setMaximum(1.0)
        self.smoothness_penalty_spin.setSingleStep(0.00001)
        self.smoothness_penalty_spin.setDecimals(5)
        self.smoothness_penalty_spin.setValue(config.DEFAULT_SMOOTHNESS_PENALTY)
        self.smoothness_penalty_spin.setToolTip(
            "Controls the tradeoff between smoothness and accuracy.\n"
            "Lower values (0.0-0.1): Prioritize accuracy, may have more wiggles.\n"
            "Higher values (0.5-1.0): Prioritize smoothness, may reduce accuracy."
        )


        # Action buttons - make them more prominent
        self.fit_bspline_button = QPushButton("Fit B-spline")
        #self.fit_bspline_button.setMinimumHeight(25)  # Make button taller
        self.fit_bspline_button.setStyleSheet("font-weight: bold;")

        # --- Layout ------------------------------------------------------
        layout = QVBoxLayout()

        # Initial control point count
        inital_row = QHBoxLayout()
        inital_row.addWidget(self.initial_cp_label)
        inital_row.addWidget(self.initial_cp_spin)
        layout.addLayout(inital_row)

        # Upper B-spline controls
        upper_row = QHBoxLayout()
        upper_row.addWidget(self.upper_cp_label)
        upper_row.addStretch(1)
        upper_row.addWidget(self.upper_insert_btn)
        layout.addLayout(upper_row)

        # Lower B-spline controls
        lower_row = QHBoxLayout()
        lower_row.addWidget(self.lower_cp_label)
        lower_row.addStretch(1)
        lower_row.addWidget(self.lower_insert_btn)
        layout.addLayout(lower_row)

        # Degree and Smoothness in same row
        params_row = QHBoxLayout()
        params_row.addWidget(self.bspline_degree_label)
        params_row.addWidget(self.bspline_degree_spin)
        params_row.addWidget(self.smoothness_penalty_label)
        params_row.addWidget(self.smoothness_penalty_spin)
        params_row.addStretch(1)
        layout.addLayout(params_row)

        # G2, G3 in same row
        continuity_row = QHBoxLayout()
        continuity_row.addWidget(self.g2_checkbox)
        continuity_row.addWidget(self.g3_checkbox)
        continuity_row.addStretch(1)
        layout.addLayout(continuity_row)

        # TE Vector Points row with tangency checkbox
        te_row = QHBoxLayout()
        te_row.addWidget(QLabel("TE Vector Points:"))
        te_row.addWidget(self.te_vector_points_combo)
        te_row.addWidget(self.enforce_te_tangency_checkbox)
        te_row.addStretch(1)
        layout.addLayout(te_row)


        # Action buttons row
        action_row = QHBoxLayout()
        action_row.addWidget(self.fit_bspline_button, 1)  # Give buttons equal space
        layout.addLayout(action_row)

        self.setLayout(layout)

        # Connect G2/G3 checkboxes for dependency logic
        self.g2_checkbox.toggled.connect(self._update_g3_checkbox_state)
        self.g3_checkbox.toggled.connect(self._update_g2_from_g3)
        self._update_g3_checkbox_state()    # Set initial G3 state
        self.bspline_degree_spin.valueChanged.connect(self._sync_initial_cp_min)
        self._sync_initial_cp_min()

    def _update_g3_checkbox_state(self):
        """Enable/disable G3 checkbox based on G2 state."""
        is_g2_enabled = self.g2_checkbox.isChecked()
        self.g3_checkbox.setEnabled(is_g2_enabled)
        if not is_g2_enabled and self.g3_checkbox.isChecked():
            # If G2 is unchecked while G3 is checked, uncheck G3
            self.g3_checkbox.setChecked(False)
    
    def _update_g2_from_g3(self):
        """When G3 is checked, automatically check G2 if not already checked."""
        if self.g3_checkbox.isChecked() and not self.g2_checkbox.isChecked():
            # Temporarily disconnect to avoid recursion
            self.g2_checkbox.blockSignals(True)
            self.g2_checkbox.setChecked(True)
            self.g2_checkbox.blockSignals(False)
            self._update_g3_checkbox_state()  # Update G3 state (should enable it) 

    def _sync_initial_cp_min(self) -> None:
        """Keep initial control point count >= degree + 1."""
        min_cp = int(self.bspline_degree_spin.value()) + 1
        if self.initial_cp_spin.minimum() != min_cp:
            self.initial_cp_spin.setMinimum(min_cp)
        if self.initial_cp_spin.value() < min_cp:
            self.initial_cp_spin.setValue(min_cp)
