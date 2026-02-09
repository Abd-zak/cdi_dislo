# CDI Dislocation Analysis (`dislocation`)

**Comprehensive utilities for dislocation analysis in Bragg Coherent Diffraction Imaging (BCDI)**

---

## Table of Contents

- [Overview](#overview)
- [Data Processing & Filtering](#data-processing--filtering)
- [Dislocation Analysis](#dislocation-analysis)
- [Visualization & Plotting](#visualization--plotting)
- [Machine Learning & Clustering](#machine-learning--clustering)
- [File Handling & Data Export](#file-handling--data-export)
- [General Utilities](#general-utilities)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Author](#author)

---

## Overview

The `dislocation` script provides extensive tools for analyzing dislocations in Bragg Coherent Diffraction Imaging (BCDI) datasets. It includes preprocessing, visualization, quantitative analysis, and machine learning approaches to examine dislocation structures comprehensively.

## Data Processing & Filtering

- Preprocessing of BCDI data (noise reduction, interpolation, filtering)
- Calculation of dislocation density and Burgers vector components
- Phase unwrapping techniques

## Dislocation Analysis

- Detection of dislocations via phase gradients and topological charges
- Computation and visualization of dislocation lines and networks
- Automated and interactive tools for detailed dislocation characterization

## Visualization & Plotting

- 2D and 3D rendering of dislocations, phase maps, and diffraction data
- Interactive visualization tools for volume slicing and tracing dislocations
- Supports Matplotlib, PyVista, and Plotly visualization backends

## Machine Learning & Clustering

- Feature extraction and classification using clustering (KMeans, DBSCAN)
- Dimensionality reduction and anomaly detection using PCA

## File Handling & Data Export

- Efficient management of large datasets using HDF5 format
- Export functionality for processed data and visualization outputs

## General Utilities

- Mathematical and statistical tools for signal processing and peak analysis
- Custom coordinate transformations and rotation matrix operations

## Dependencies

This script utilizes scientific computing libraries:

- NumPy
- SciPy
- Matplotlib
- Scikit-learn
- PyVista
- Xrayutilities
- Custom CDI analysis modules

## Usage

The script can be utilized interactively within Jupyter notebooks or as a standalone module for batch processing BCDI datasets.

```bash
python dislocation.py
```

## Author

**Abdelrahman Zakaria**  
Date: 19/02/2025

