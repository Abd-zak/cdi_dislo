# Mechanical Analysis Submodule

This submodule provides various utilities for analyzing mechanical properties of materials, focusing specifically on force-displacement analysis obtained from nanoindentation experiments. It encompasses data processing, regression analysis, denoising, and visualization capabilities to support comprehensive material science research.

---

## Overview

This module addresses several functionalities:

- Mathematical computations related to mechanical properties.
- Processing and analyzing nanoindentation experimental data.
- Regression analysis to extract mechanical properties.
- Data loading, saving, and handling utilities.
- Advanced visualization of mechanical behavior and plastic deformation.

---

## Key Functionalities

### Mathematical & Utility Functions
- `calculate_burgers_modulus()`: Magnitude of Burgers vector for FCC nanoparticles.
- `calculate_tau()`: Calculates resolved shear stress.
- `get_dislocation_speed()`: Determines dislocation speed from applied forces.

### Data Analysis and Processing
- `analyze_results()`: Comprehensive analysis of nanoindentation force-displacement.
- `analyze_force_displacement()`: Extracts elastic and plastic deformation phases with regression analysis.
- `dict_compare()`: Recursively compares dictionaries including NumPy arrays.

### Data Loading and Saving
- `load_and_save_as_dict_femtotools()`: Converts `.npz` files into Python dictionaries.
- `save_data_hdf5()`: Stores processed data efficiently in HDF5 format.

### Regression & Machine Learning
- `fit_linear_regression()`: Basic linear regression implementation.
- `fit_and_plot_data()`: Fits constrained regression models and visualizes them.
- `fit_with_min_residual()`: Optimizes regression fit based on minimal residual error.
- `get_x_fromy_linear()`: Computes corresponding x-values for linear regression predictions.

### Nanoindentation Data Processing
- `process_femtotools_data()`: Calibrates and cleans raw experimental data.
- `denoise_data_femtotools()`: Implements multiple denoising techniques.
- `get_xmin_nonzero_fromnoiseddata_femtotools()`: Identifies initial contact points.

### Plastic Deformation Analysis
- `get_alldrops_fits_coeficients_and_functions()`: Identifies plastic deformation events and fits regression models.
- `get_x_y_of_drops_plastic()`: Extracts coordinates of plastic deformation events.
- `constrained_polyfit()`: Performs polynomial fits with constraints.
- `get_reordering_indices()`: Generates reordering indices based on specified criteria.

### Visualization
- Extensive plotting capabilities using Matplotlib, including:
  - Force-displacement curves
  - Elastic/plastic deformation profiles
  - Dislocation velocity plots
  - Stress distribution analyses
  - Histograms of mechanical property distributions

---

## Dependencies

Main dependencies:
- NumPy
- SciPy
- Matplotlib
- h5py

Install via pip:
```bash
pip install numpy scipy matplotlib h5py
```

---

## Usage Example

A simple usage example:
```python
from your_package_name.mechanical_analysis import process_femtotools_data, analyze_force_displacement

# Load and preprocess data
data = process_femtotools_data('experiment_data.npz')

# Analyze force-displacement relationship
results = analyze_force_displacement(data)
```

---

## Testing

Run tests using pytest:
```bash
pytest tests/test_mechanical_analysis.py
```

---

## Future Improvements

- Explicit dependency management and imports
- Advanced outlier handling and argument validation
- Performance optimization through vectorization
- Enhanced error handling mechanisms

---

## License

Distributed under the MIT License. See [LICENSE](../../LICENSE) for more information.

