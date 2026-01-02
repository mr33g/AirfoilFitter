# Airfoil Fitter

A desktop application for fitting B-spline curves to airfoil coordinate data and exporting the result for use in CAD software.

## Installation

### Requirements

- Python 3.10 or later
- Windows, macOS, or Linux

### Setup

1. Clone or download this repository.

2. Create and activate a virtual environment (recommended):
   ```
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   source .venv/bin/activate   # macOS/Linux
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Dependencies

| Package   | Purpose                              |
|-----------|--------------------------------------|
| numpy     | Numerical operations                 |
| scipy     | B-spline fitting and optimization    |
| PySide6   | Qt GUI framework                     |
| pyqtgraph | Interactive plotting                 |
| ezdxf     | DXF file export                      |
| joblib    | Parallel processing utilities        |

## Usage

### Starting the Application

```
python run_gui.py
```

### Workflow

1. **Load Airfoil Data**  
   Click *Load Airfoil File* and select a `.dat` file. Both Selig and Lednicer formats are supported. The application normalizes coordinates to unit chord with the leading edge at the origin.

2. **Fit B-spline**  
   Click *Fit B-spline* to perform the initial fit. The default configuration uses 5 control points per surface with degree 4.

3. **Adjust Parameters**  
   - **Degree**: B-spline polynomial degree (4–12). Higher degrees allow smoother curves but may be less stable.
   - **Smoothness**: Regularization weight. Higher values produce smoother control point distributions at the cost of fitting accuracy.
   - **G2 / G3**: Enable curvature (G2) or curvature-derivative (G3) continuity at the leading edge.
   - **TE tangency**: Constrain trailing edge tangent direction to match the input data.
   - **Single Span**: Force degree = (control points − 1), producing a single polynomial segment per surface. The result can be imported in Fusion 360.

4. **Refine the Fit**  
   Use the **+** and **−** buttons next to each surface label to insert or remove control points. In multi-span mode, knots are inserted at the location of maximum deviation. In single-span mode, this elevates the polynomial degree.

5. **Trailing Edge Thickening**  
   Enter a thickness value in millimeters and click *Apply* to add a blunt trailing edge. The offset is applied using a C² quintic blend that preserves leading edge geometry.

6. **Export**  
   - *Export DXF*: Saves upper and lower B-spline curves as NURBS entities scaled to the specified chord length.
   - *Export DAT*: Saves a resampled coordinate file in Selig format.

### Configuration

Runtime defaults are defined in `core/config.py`:

| Parameter                  | Default | Description                                      |
|----------------------------|---------|--------------------------------------------------|
| `DEBUG_WORKER_LOGGING`     | False   | Enable verbose control point logging             |
| `DEFAULT_BSPLINE_DEGREE`   | 4       | Initial B-spline degree                          |
| `DEFAULT_BSPLINE_CP`       | 5       | Initial control points per surface               |
| `DEFAULT_SMOOTHNESS_PENALTY` | 0.01  | Regularization weight                            |
| `DEFAULT_CHORD_LENGTH_MM`  | 200.0   | Default chord length for export                  |

## Background

### Problem Statement

Airfoil coordinates from sources like the UIUC database are provided as discrete point sets. Importing these directly into CAD software typically results in polylines or low-quality splines that are difficult to edit and may introduce surface artifacts. Extreme cases might even cause problems in CAM due to excessive tool acceleration. 

This application fits smooth B-spline curves to the coordinate data, enforcing geometric constraints that ensure the resulting curves are suitable for CAD modeling.

### B-spline Fitting Method

The fitting algorithm solves a constrained least-squares problem:

1. **Parameterization**: Input coordinates are mapped to parameter values using `u = x^0.5`, which concentrates parameter density near the leading edge where curvature is highest.

2. **Knot Vector**: A clamped uniform knot vector is generated based on the number of control points and the selected degree.

3. **Constraints**:
   - **G1 at leading edge**: Both surfaces share a common leading edge point at the origin, with the first control point constrained to lie on the y-axis. This ensures tangent continuity.
   - **G2 at leading edge** (optional): Curvatures are matched by solving an optimization problem with an equality constraint on the curvature values at u=0.
   - **G3 at leading edge** (optional): Curvature derivatives are additionally matched.
   - **Trailing edge position**: The last control point is constrained to the trailing edge coordinates.
   - **Trailing edge tangent** (optional): The direction of the curve at u=1 is constrained to match a tangent vector computed from the input data.

4. **Optimization**: When G2/G3 constraints are enabled, the problem is solved using SLSQP (Sequential Least Squares Programming). The objective function combines fitting error with a smoothness penalty on second-order differences of control points.

5. **Smoothness Penalty**: A gradient-weighted penalty is applied, with lower weight near the leading edge (allowing the curve to follow high curvature) and higher weight toward the trailing edge (reducing oscillation in low-curvature regions).

### Single-Span Mode

When enabled, the degree is set to `(control points − 1)`, resulting in a Bézier curve with a single polynomial segment. This mode is useful for CAD software that has limited support for multi-span NURBS curves (e.g., Fusion 360's native spline tools).

### Trailing Edge Thickening

Blunt trailing edges are applied as a post-processing step:

1. Both fitted curves are densely sampled.
2. A vertical offset is computed using a quintic smoothstep function: `f(x) = x³(10 − 15x + 6x²)`.
3. The offset is zero at x=0 (leading edge) and reaches the target half-thickness at x=1 (trailing edge).
4. The curves are re-fitted to the offset points, preserving G1 continuity at the leading edge.

### Project Structure

```
AirfoilFitter/
├── run_gui.py              # Application entry point
├── core/
│   ├── config.py           # Configuration constants
│   ├── airfoil_processor.py # Coordinate loading and normalization
│   └── bspline_processor.py # B-spline fitting algorithms
├── gui/
│   ├── main_window.py      # Main window layout
│   ├── controllers/        # Application logic
│   └── widgets/            # UI components
└── utils/
    ├── bspline_helper.py   # B-spline utility functions
    ├── data_loader.py      # File format parsing
    └── dxf_exporter.py     # DXF output
```

## License

MIT License
