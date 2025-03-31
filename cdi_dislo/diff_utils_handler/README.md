# CDI Dislocation Diffutils

**Utility functions for Reciprocal and Real Space Optimization, Data Processing, and Visualization**

---

## Table of Contents

- [Voxel Optimization](#voxel-optimization)
- [Data Resampling and Adjustments](#data-resampling-and-adjustments)
- [Data Reading and Processing](#data-reading-and-processing)
- [Data Visualization](#data-visualization)
- [X-ray Diffraction Utilities](#x-ray-diffraction-utilities)
- [Latest Updates](#latest-updates)

---

## Voxel Optimization

Functions for optimizing voxel sizes in reciprocal and real spaces:

- `optimize_voxel_reciproque_space`: Finds optimal voxel size along a specified reciprocal-space direction.
- `optimize_voxel_real_space`: Adjusts voxel size along a single real-space direction.
- `optimize_voxel_real_space_all_directions`: Optimizes voxel sizes simultaneously in all three real-space directions.

## Data Resampling and Adjustments

Utilities for resampling and adjusting voxel sizes in 3D data:

- `adjust_voxel_size`: Rescales voxel size in a 3D array to match a specified target voxel size.
- `array_to_dict`: Converts a NumPy array into a dictionary for structured data handling.

## Data Reading and Processing

Functions to read, process, and transform experimental data:

- `read_orth_rho_phi`: Reads and processes phase and intensity data from input files.
- `orth_ID27_gridder_def_new`: Computes reciprocal-space coordinates specifically for ID27 beamline data.
- `orth_ID27_gridder_def`: Alternative orthonormalization method for ID27 beamline datasets.
- `get_abc_direct_space`: Computes direct-space coordinates from reciprocal-space data.
- `get_abc_direct_space_sixs2019`: Specialized computation of direct-space coordinates tailored to SIXS 2019 data.
- `orth_sixs2019_gridder_def`: Converts SIXS 2019 experimental data into reciprocal space.

## Data Visualization

Visualization tools for analyzing 3D intensity and phase data:

- `plot_qxqyqzI`: Plots 3D intensity maps in reciprocal space.
- `plotqxqyqzi_imshow`: Visualizes slices of 3D intensity data using `imshow`.
- `ortho_data_phaserho_sixs2019`: Processes and plots orthogonalized phase/intensity data from SIXS 2019.

## X-ray Diffraction Utilities

Specialized tools for X-ray diffraction analysis:

- `detectframe_to_labframe`: Converts detector-frame data into the laboratory frame.
- `orth_SIXS2019`: Performs orthonormalization for SIXS 2019 datasets.
- `orth_SIXS2019_gridder_def`: Gridder function specifically designed for processing SIXS 2019 data.

## Latest Updates

- Implemented optimized voxel size search algorithms for both reciprocal and real spaces.
- Enhanced gridder-based data transformation methods.
- Improved data visualization capabilities through updated plotting functions.

