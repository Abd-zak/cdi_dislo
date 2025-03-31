# cdi_dislo

**cdi_dislo** is a comprehensive Python package designed for analyzing Bragg Coherent Diffraction Imaging (BCDI) data, with a particular focus on dislocation detection and characterization. The package offers utilities for data processing, diffraction calibration, peak fitting, thermal strain calculations, nanoindentation mechanical property analysis, and visualization of 3D intensity distributions.

---

## 🚀 Features

- 📡 **Diffraction Analysis**: Extract lattice parameters and calibrate diffraction setups.
- 📊 **Data Processing & Visualization**: Generate 3D projections, visualize intensity distributions, and perform peak fitting.
- 🔬 **Dislocation Detection**: Identify and characterize dislocations in BCDI data.
- 📏 **FWHM & Strain Analysis**: Compute Full-Width at Half Maximum (FWHM) and integral FWHM.
- 🧩 **Machine Learning & Fitting**: Support polynomial and exponential fitting models.
- 🔍 **General Utilities**: Includes masking, cropping, and statistical analysis tools.
- 🔨 **Nanoindentation Analysis**: Tools for analyzing mechanical properties from nanoindentation experiments.

---

## 📁 Project Structure

```bash
cdi_dislo/
├── LICENSE
├── README.md
├── requirements.txt
└── cdi_dislo/
    ├── common_imports.py
    ├── diff_calib_handler
    │   └── cdi_dislo_diffcalib.py
    ├── diff_utils_handler
    │   └── cdi_dislo_diffutils.py
    ├── dislo_handler
    │   └── cdi_dislo_dislocation.py
    ├── ewen_utilities  https://github.com/ewbellec/alienclustering.git
    │   ├── Reconstruction.py
    │   ├── PostProcessing.py
    │   ├── Orthogonalization_real_space.py
    │   ├── Orthogonalization_reciprocal_space.py
    │   ├── Object_utilities.py
    │   └── (and other utilities...)
    ├── femtotools_handler
    │   └── cdi_dislo_femto.py
    ├── general_utilities
    │   └── cdi_dislo_utils.py
    ├── genetic_handler
    │   └── cdi_dislo_genetic.py
    ├── orthogonalisation_handler
    │   └── cdi_dislo_ortho_handler.py
    ├── plotutilities
    │   └── cdi_dislo_plotutilities.py
    ├── reco_handler
    │   └── cdi_dislo_reconstruction.py
    ├── rotation_handler
    │   └── cdi_dislo_rotation.py
    └── statdiff_handler
        └── cdi_dislo_statdiff_handler.py
```

---

## 📦 Installation

Clone the repository:

```bash
git clone git@github.com:Abd-zak/cdi_dislo.git
cd cdi_dislo

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📚 Modules Overview

Each submodule has a dedicated `README.md` for detailed information:

- **[`diff_calib_handler`](cdi_dislo/diff_calib_handler/README.md)**: Diffraction data calibration.
- **[`diff_utils_handler`](cdi_dislo/diff_utils_handler/README.md)**: Diffraction utilities.
- **[`dislo_handler`](cdi_dislo/dislo_handler/README.md)**: Dislocation and deformation analysis.
- **[`ewen_utilities`](cdi_dislo/ewen_utilities/README.md)**: Reconstruction and post-processing.
- **[`femtotools_handler`](cdi_dislo/femtotools_handler/README.md)**: Nanoindentation utilities.
- **[`general_utilities`](cdi_dislo/general_utilities/README.md)**: General-purpose utilities.
- **[`genetic_handler`](cdi_dislo/genetic_handler/README.md)**: Genetic optimization algorithms.
- **[`orthogonalisation_handler`](cdi_dislo/orthogonalisation_handler/README.md)**: Data orthogonalization.
- **[`plotutilities`](cdi_dislo/plotutilities/README.md)**: Visualization tools.
- **[`reco_handler`](cdi_dislo/reco_handler/README.md)**: Reconstruction selection.
- **[`rotation_handler`](cdi_dislo/rotation_handler/README.md)**: Data rotation and alignment.
- **[`statdiff_handler`](cdi_dislo/statdiff_handler/README.md)**: Statistical diffraction analysis.

---

## 📝 Example Usage

```python
from cdi_dislo.ewen_utilities.Reconstruction import CDI_one_reconstruction

# Perform a CDI reconstruction
result = CDI_one_reconstruction(data, params)
```

---

## 🧪 Testing

Run tests:

```bash
pytest tests/
```

---

## 🤝 Contributing

Contributions are welcome! Check [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## 📜 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

