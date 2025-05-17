from setuptools import setup, find_packages

setup(
    name="cdi_dislo",
    version="0.1.0",
    description="cdi_dislo is a comprehensive Python package designed for analyzing Bragg Coherent Diffraction Imaging (BCDI) data, with a particular focus on dislocation detection and characterization. The package offers utilities for data processing, diffraction calibration, peak fitting, thermal strain calculations, nanoindentation mechanical property analysis, and visualization of 3D intensity distributions.",
    author="Abdelrahman Zakaria",
    packages=find_packages(),
    install_requires=[
        # Core Scientific Libraries
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "h5py",
        "seaborn",
        "scikit-learn",
        # Signal Processing & Image Analysis
        "pywavelets",
        "scikit-image",
        "vtk",
        "pyvista",
        "imageio",
        "ruptures",
        # Interactive & Visualization
        "ipywidgets",
        "ipyvolume",
        "plotly",
        "mpl_interactions",
        "networkx",
        "tabulate",
        "mpld3",
        # Mathematical & Statistical Utilities
        "xrayutilities",
        "astropy",
        "cryptography",
        "gekko",
        "yapf",
        # File Handling & Utilities
        "hdf5plugin",
        "silx",
        # Miscellaneous
        "pytest",
    ],
)
