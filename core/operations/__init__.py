from .fit_ops import fit_bspline, fit_with_g2_optimization
from .g1_ops import fit_g1_independent, fit_single_surface_g1
from .insertion_ops import (
    apply_knot_insertions,
    cp_spacing_violates_minimum,
    insert_knot_with_spacing_fallback,
    largest_span_midpoint_knot,
    refine_curve_with_knots,
    refine_curves_with_surface_knots,
    refit_after_knot_insertion,
)
from .model_ops import finalize_curves, validate_continuity, validate_trailing_edge_tangents
from .te_ops import apply_te_thickening, remove_te_thickening
