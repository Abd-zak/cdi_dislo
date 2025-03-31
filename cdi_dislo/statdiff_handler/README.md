# CDI Intensity Analysis

**Advanced functions for analyzing intensity distributions in 3D diffraction datasets, tailored for Bragg Coherent Diffraction Imaging (BCDI)**

---

## Table of Contents

- [Overview](#overview)
- [Key Functionalities](#key-functionalities)
- [Suggestions for Improvements](#suggestions-for-improvements)
- [Dependencies](#dependencies)
- [Usage](#usage)

---

## Overview

This script provides comprehensive methods for analyzing intensity distributions within 3D data sets from diffraction experiments, particularly suited for Bragg Coherent Diffraction Imaging (BCDI). It includes robust tools for statistical analysis, peak fitting, visualization, and advanced data handling.

## Key Functionalities

- **Intensity Distribution Analysis**:
  - Plotting 1D intensity distributions along various axes
  - Computation of key statistical measures (FWHM, barycenter, skewness, kurtosis)

- **Peak Profile Fitting**:
  - Gaussian, Lorentzian, Voigt, and Pseudo-Voigt profiles
  - Calculation of fitting quality metrics (R-squared values)

- **FWHM Calculation**:
  - Various methods for determining Full Width at Half Maximum (FWHM) geometrically and through fitting

- **Visualization Tools**:
  - Detailed intensity plots with fitted profiles, mean value indicators, and annotations

- **Advanced Data Handling**:
  - Noise management techniques
  - Scherrer equation calculations
  - Bragg angle determinations

## Suggestions for Improvements

- **Code Optimization**:
  - Modularize repeated operations, such as summations along different axes

- **Exception Handling**:
  - Improve logging and detailed error handling for fitting procedures

- **Performance Enhancements**:
  - Leverage libraries like `numba` or `cython` for intensive computational tasks (peak fitting, FWHM calculations)

- **Visualization Enhancements**:
  - Add gridlines, annotations, dynamic scaling, and structured subplots using Matplotlib

- **Configurable Parameters**:
  - Enable users to specify noise levels, fitting algorithms, and detection thresholds through configurable function arguments

- **Generalization**:
  - Extend script capabilities to support datasets of arbitrary dimensionality for broader experimental adaptability

- **Save and Export Options**:
  - Implement data export functionalities (JSON, CSV)
  - Support high-resolution figure saving for publication quality visuals

- **Documentation and Readability**:
  - Enhance docstrings with detailed parameter and return value descriptions
  - Replace `print()` statements with structured logging for improved debugging and traceability

## Dependencies

- NumPy
- SciPy
- Matplotlib
- Optionally: Numba/Cython for performance

## Usage

Designed for interactive use in Jupyter notebooks or automated batch analysis:

```python
from cdi_intensity_analysis import analyze_intensity

# Example usage:
analyze_intensity(your_3d_dataset)
```

---