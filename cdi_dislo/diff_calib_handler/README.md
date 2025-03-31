# CDI Dislocation DiffCalib

**Utility Functions for Diffraction Analysis, Temperature Calibration, and Strain Measurement**

---

## Table of Contents

- [Overview](#overview)
- [Diffraction Analysis Utilities](#diffraction-analysis-utilities)
- [Temperature and Lattice Parameter Predictions](#temperature-and-lattice-parameter-predictions)
- [Strain and Thermal Expansion](#strain-and-thermal-expansion)
- [Data Processing Functions](#data-processing-functions)

---

## Overview

The `cdi_dislo_diffcalib` module contains a suite of functions for analyzing diffraction data, calibrating temperature measurements, predicting lattice parameters, and calculating strain based on thermal expansion. These tools are particularly useful for X-ray diffraction experiments involving sapphire and other crystalline materials.

## Diffraction Analysis Utilities

- `plot_temperature_and_lattice`: Plots comparisons between experimental and theoretical temperatures, temperature differences, lattice parameters, and lattice parameter versus temperature.
- `calculate_epsilon_and_prediction`: Calculates strain (epsilon) and predicts temperatures using polynomial fitting.
- `calculate_epsilon_and_prediction_2nd_degree`: Specifically performs 2nd-degree polynomial fits for temperature and strain calculations.

## Temperature and Lattice Parameter Predictions

- `get_prediction_from_theo_sapphire_lattice`: Predicts sapphire lattice parameters based on temperature data, ensuring accuracy through iterative adjustment.
- `optimize_energy_value_and_calibrate_temperature`: Optimizes X-ray energy values to minimize temperature discrepancies and calibrate temperature measurements.

## Strain and Thermal Expansion

- `get_theo_strain_thermal_expansion`: Calculates theoretical strain due to thermal expansion for specified materials, such as platinum or sapphire (parallel/perpendicular to the c-axis).

## Data Processing Functions

Functions specifically tailored for processing and visualizing diffraction data from various experiments:

- `process_scan_data_COM_CRISTAL_2022`: Processes Center of Mass (COM) analysis data from the CRISTAL beamline.
- `process_scan_data_COM_id1_avril_2023`: Handles COM data from the ID1 beamline (April 2023 experiments).
- `process_bcdi_data_B12S1P1_id1_Jan_2024`: Processes Bragg Coherent Diffraction Imaging (BCDI) data collected in January 2024.
- `process_scan_data_COM_id1_june_2024`: Manages COM analysis data for June 2024 ID1 beamline experiments.
- `process_scan_id1_june_2024`: Individual scan processing for June 2024 ID1 data.

These functions facilitate data visualization, masking, cropping, and analysis for optimized diffraction data interpretation and calibration.