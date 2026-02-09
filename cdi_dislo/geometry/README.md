# Phase Retrieval, Preprocessing, and Strain Analysis Utilities for Bragg Coherent Diffraction Imaging (BCDI)

**Comprehensive utilities for phase retrieval, preprocessing, phase ramp removal, strain computation, and visualization in Bragg Coherent Diffraction Imaging (BCDI).**

---

## Summary Block

This script provides a robust set of utilities essential for analyzing volumetric datasets, computing strain fields, and performing phase retrieval and preprocessing in BCDI. Key functionalities include:

✅ **Phase Retrieval Setup (`setup_phase_retrieval`)**  
- Configures RAAR, HIO, ER, and ML algorithms with user-defined parameters.
- Handles energy-to-wavelength conversion and algorithm parameter processing.

✅ **Preprocessing Setup (`setup_preprocessing`)**  
- Manages voxel binning, defect detection, clustering, and preparation tasks for optimal data reconstruction.

✅ **Phase Ramp Removal (`remove_phase_ramp`, `remove_phase_ramp_clement`)**  
- Uses linear regression techniques to correct unwanted phase ramps from 3D datasets.

✅ **Strain and Displacement Gradient Computation (`get_displacement_gradient`, `get_het_normal_strain`)**  
- Employs hybrid gradient calculations for precise displacement gradients and strain estimations.

✅ **Lattice Parameter and Strain Estimation (`get_lattice_parametre`, `get_strain_from_lattice_parametre`)**  
- Extracts lattice parameters based on Bragg conditions to quantify heterogeneous strain in materials.

✅ **Visualization and Export to VTK (`getting_strain_mapvti`)**  
- Identifies dislocation nodes and defect lines by minimizing the gradient of the phase.
- Generates detailed strain maps and exports data to VTI for visualization.
- Includes debugging plots for thorough inspection of strain and phase gradients.

---

## Implemented Improvements

🔹 Explicitly refined import statements, removing `import *` for clarity.  
🔹 Validated critical imports such as `plt`, `array()`, `gu`, and `en2lam()` for robustness.  
🔹 Enhanced exception handling for missing datasets in HDF5 files.  
🔹 Optimized calls to gradient calculations to reduce redundant computations.  
🔹 Improved handling of nested Miller indices using correct NumPy functions.  
🔹 Validated NaN handling utilities (`nan_to_zero()` and `zero_to_nan()`) for consistent data integrity.

---

## Suggestions for Further Enhancements

- Integrate parallel processing or GPU acceleration for computationally intensive gradient calculations.
- Expand automated error handling and logging for long-running batch processes.
- Develop interactive visualization and GUI options for dynamic user interactions.
- Enhance script modularity by separating functionality into clearly defined modules or classes.
- Expand documentation with comprehensive usage examples and troubleshooting guides.

---

## Dependencies

- NumPy
- Matplotlib
- SciPy
- h5py
- scikit-learn
- PyVista (optional for VTK exports)

---

## Author

**Abdelrahman Zakaria**  
Date: 19/02/2025

