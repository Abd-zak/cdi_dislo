# cdi_dislo

**cdi_dislo** is a Python package for analyzing Bragg Coherent Diffraction Imaging (BCDI) data, with a special focus on dislocation detection and characterization. The package provides utilities for data processing, diffraction calibration, peak fitting, thermal strain calculations, and visualization of 3D intensity distributions.

## 🚀 Features
- 📡 **Diffraction Analysis**: Extracts lattice parameters and calibrates diffraction setups.
- 📊 **Data Processing & Visualization**: 3D projections, intensity distributions, and peak fitting.
- 🔬 **Dislocation Detection**: Identifies dislocations in BCDI data.
- 📏 **FWHM & Strain Analysis**: Computes Full-Width at Half Maximum (FWHM) and integral FWHM.
- 🧩 **Machine Learning & Fitting**: Supports polynomial and exponential fitting models.
- 🔍 **General Utilities**: Masking, cropping, and statistical analysis tools.

---

## 📦 Installation
```bash
# Clone the repository
git clone git@github.com:Abd-zak/cdi_dislo.git
cd cdi_dislo

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
