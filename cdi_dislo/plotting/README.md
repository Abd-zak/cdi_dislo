# Visualization Utilities for 3D Data, Diffraction, and Mechanical Properties

**Comprehensive visualization tools for analyzing 3D scientific data, diffraction patterns, and mechanical properties of materials.**

---

## Overview

This script provides a versatile suite of visualization utilities tailored for analyzing and interpreting complex datasets commonly used in scientific research, especially for Bragg Coherent Diffraction Imaging (BCDI), mechanical properties analysis, and strain visualization.

---

## Table of Contents

- [Key Features](#key-features)
- [Visualization Functions](#visualization-functions)
- [Usage Examples](#usage-examples)
- [Implemented Improvements](#implemented-improvements)
- [Suggestions for Further Enhancements](#suggestions-for-further-enhancements)
- [Dependencies](#dependencies)
- [Author](#author)

---

## Key Features

- **Interactive 3D visualizations** for efficient exploration of volumetric data.
- **Diffraction pattern analysis** with projections and intensity mappings.
- Visualization of **mechanical properties** such as stiffness, yield displacement, and force.
- Advanced tools for **strain and deformation mapping** in nanoparticle datasets.
- **Interactive widgets and plotting tools** for dynamic data exploration.

---

## Visualization Functions

### 3D Data Visualization

- `visualize_3d_data()`: Interactive volume rendering using PyVista.
- `plot_interactive_slices()`: Dynamic slicing and 2D visualizations of 3D arrays.
- `plot_3d_array()`, `plot_3d_array_ipv()`: 3D scatter and volume plots.

### Diffraction and Phase Analysis

- `plot_summary_difraction()`: Diffraction intensity and COM projections.
- `plot_qxqyqzI()`, `plotqxqyqzi_imshow()`: Detailed Q-space diffraction intensity visualizations.
- `plot_def_()`: Deformation maps based on phase analysis.

### Mechanical Properties and Strain Analysis

- `plot_mechanical_properties()`: Stiffness, yield force, and displacement analysis.
- `plot_data_disp()`, `plot_data_disp_projection()`: Strain mapping in nanoparticles.

### Statistical and Evolutionary Analysis

- `plot_stast_evolution_id27()`: Statistical metrics evolution over experimental series.
- `plot_xyz_com()`, `plot_combined_xyz()`: Center-of-mass displacement tracking.
- `plot_phases()`: Phase evolution tracking under varying conditions.

### Utility Functions

- `format_vector()`: Numeric vector formatting for readability.
- `get_color_list()`: Generation of distinct color palettes for plotting.
- `summary_slice_plot_abd()`: Annotated summary plots for slices.

---

## Usage Examples

```python
# Example of visualizing 3D data interactively
visualize_3d_data(data_array=my_3d_data, cmap="viridis", plot_title="Sample Visualization")

# Example of plotting mechanical properties
plot_mechanical_properties(slopes_elastic=stiffness_data, f_max_elastic_x=displacement_data, f_max_elastic_y=force_data, test_part=sample_labels, plot_fit=True)
```

---

## Implemented Improvements

- Removed redundant imports and streamlined dependencies.
- Consolidated plotting functionalities to reduce repetition and enhance modularity.
- Optimized error handling and added robust input validation.
- Enhanced performance with improved plotting routines.
- Improved readability and consistency in code formatting and structure.

---

## Suggestions for Further Enhancements

- Optimize computational performance using parallel processing or tools like `numba` or `cython`.
- Integrate interactive web-based visualization libraries (Dash, Bokeh).
- Extend plot customization and interactivity with GUI-based controls.
- Develop batch processing features for automated visualization of large datasets.
- Enhance documentation with detailed examples and Jupyter notebooks.

---

## Dependencies

- NumPy
- Matplotlib
- PyVista
- Plotly
- Scipy
- Widgets from IPython

---

## Author

**Abdelrahman Zakaria**  
Date: 19/02/2025

