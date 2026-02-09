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
    ├── calibration
    │   ├── cdi_dislo_diffcalib.py
    │   ├── diffcalib.py
    │   └── README.md
    ├── diffraction
    │   ├── diffutils.py
    │   └── README.md
    ├── dislocations
    │   ├── dislocation.py
    │   └── README.md
    ├── ewen_utilities  https://github.com/ewbellec/alienclustering.git
    │   ├── Reconstruction.py
    │   ├── PostProcessing.py
    │   ├── Orthogonalization_real_space.py
    │   ├── Orthogonalization_reciprocal_space.py
    │   ├── Object_utilities.py
    │   └── (and other utilities...)
    ├── utils
    │   ├── utils.py
    │   └── README.md
    ├── genetic
    │   ├── genetic.py
    │   └── README.md
    ├── geometry
    │   ├── ortho_handler.py
    │   └── README.md
    ├── plotting
    │   ├── plotutilities.py
    │   ├── linecut.py
    │   └── README.md
    ├── reconstruction
    │   ├── reconstruction.py
    │   └── README.md
    ├── rotation
    │   ├── rotation.py
    │   └── README.md
    └── statistics
        ├── statdiff_handler.py
        └── README.md
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

- **[`calibration`](cdi_dislo/calibration/README.md)**: Diffraction data calibration.
- **[`diffraction`](cdi_dislo/diffraction/README.md)**: Diffraction utilities.
- **[`dislocations`](cdi_dislo/dislocations/README.md)**: Dislocation and deformation analysis.
- **[`ewen_utilities`](cdi_dislo/ewen_utilities/README.md)**: Reconstruction and post-processing.
- **[`utils`](cdi_dislo/utils/README.md)**: General-purpose utilities.
- **[`genetic`](cdi_dislo/genetic/README.md)**: Genetic optimization algorithms.
- **[`geometry`](cdi_dislo/geometry/README.md)**: Data orthogonalization.
- **[`plotting`](cdi_dislo/plotting/README.md)**: Visualization tools.
- **[`reconstruction`](cdi_dislo/reconstruction/README.md)**: Reconstruction selection.
- **[`rotation`](cdi_dislo/rotation/README.md)**: Data rotation and alignment.
- **[`statistics`](cdi_dislo/statistics/README.md)**: Statistical diffraction analysis.

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

