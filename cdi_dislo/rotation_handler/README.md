# Rotation Utilities (`rotation_utils`)

**Comprehensive functions for rotation, orthogonalization, and geometric transformations in 3D datasets**

---

## Overview

This script provides an extensive suite of utilities dedicated to handling rotations, coordinate transformations, and orthogonalization tasks, especially suited for applications in crystallography and diffraction imaging analysis.

---

## Table of Contents

- [Orthogonalization & Normalization](#orthogonalization--normalization)
- [Geometric Transformations](#geometric-transformations)
- [Rotation Matrix Computations](#rotation-matrix-computations)
- [Grid & Data Transformations](#grid--data-transformations)
- [Dislocation Analysis](#dislocation-analysis)
- [Utility & Debugging](#utility--debugging)
- [Suggestions & Improvements](#suggestions--improvements)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Author](#author)

---

## Orthogonalization & Normalization

- `orthogonalize_basis`: Orthogonalizes 3x3 matrices using SVD or QR decomposition.
- `normalize_vector`: Ensures vectors are normalized to unit length.
- `normalize_vectors_3d`: Normalizes multiple 3D vectors simultaneously.

## Geometric Transformations

- Coordinate system conversions: Cartesian, Polar, Cylindrical, Spherical (`cart2pol`, `xyz_to_thetaphir`, etc.).
- 3D coordinate reference transformations (`referent_rotzy_trans_xyz`, `referent_roty`, `referent_rotz_`).
- Transform coordinates into crystallographic basis (`transform_coordinates_to_crystallographic`).

## Rotation Matrix Computations

- Compute rotation matrices aligning vectors (`compute_rotation_matrix`).
- Generate rotation matrices using ParaView's ZYX convention (`compute_rotation_matrix_paraview_style`).
- Individual axis rotations: X, Y, Z (`rotation_matrix_x`, `rotation_matrix_y`, `rotation_matrix_z`).
- Normalize and validate rotation matrices (`normalize_rotation_matrix`).

## Grid & Data Transformations

- Rotate 3D NumPy arrays with padding support (`apply_rotation_to_data`).
- Preserve axes while applying rotation transformations (`transform_data_paraview_style_with_new_axes`).
- Convert data grids to crystallographic basis (`transform_grid_to_crystallographic`).

## Dislocation Analysis

- Extract circular selections around dislocations (`get_select_circle_data_2d`, `get_select_circle_data`).
- Analyze and transform crystallographic vectors (`analyze_and_transform_vectors`).

## Utility & Debugging

- Verify rotations and alignments (`test_rotation`).
- Calculate Euler rotation angles (`calculate_rotation_angles`).
- Determine angles between vectors (`angle_between_vectors`).

---

## Suggestions & Improvements

- **Code Optimization:** Consolidate similar functions and refactor repetitive code blocks.
- **Performance Enhancements:** Leverage `numba` or `scipy.ndimage` for faster array transformations.
- **Modularity & Readability:** Organize related utilities into separate modules (`rotation_utils.py`, `coordinate_transform.py`).
- **Validation & Error Handling:** Add comprehensive input checks and clear error messaging.
- **Testing & Debugging:** Implement unit tests and structured logging to replace print statements.
- **Vectorized Operations:** Optimize loops using NumPy's vectorized operations.
- **Documentation:** Provide detailed docstrings, usage examples, and visualization aids.

---

## Dependencies

- NumPy
- SciPy
- Matplotlib
- Optional: Numba (for performance improvements)

---

## Usage

This module can be imported into Python scripts or interactive sessions for handling rotation tasks:

```python
from rotation_utils import compute_rotation_matrix

vector_a = [1, 0, 0]
vector_b = [0, 1, 0]
rotation_matrix = compute_rotation_matrix(vector_a, vector_b)
```

---

## Author

**Abdelrahman Zakaria**  
Date: 19/02/2025

