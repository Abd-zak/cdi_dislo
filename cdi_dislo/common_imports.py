# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", module="h5py")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os , sys , subprocess , glob, traceback, copy, json, time, re, shutil, logging, math, cmath, h5py, scipy ,importlib , mpld3, astropy, vtk, silx.io, hdf5plugin, imageio, pywt ,gekko ,yapf.yapflib.yapf_api,matplotlib
from math import gcd, acos, degrees, sqrt
from numpy import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets                     import interact, widgets
from fractions                      import Fraction
from typing                         import Tuple,List,Union,Optional
from decimal                        import Decimal, getcontext
from contextlib import redirect_stdout
import pyvista as pv
import random as py_random 
from scipy.stats import pearsonr

import xrayutilities as xu
from xrayutilities                  import lam2en

import ipyvolume as ipv
import ruptures as rpt
import networkx as nx
import plotly.graph_objects as go

# Sklearn
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.decomposition import PCA, FastICA

# SciPy
from scipy                          import stats, ndimage, signal, fftpack, optimize, spatial
from scipy.optimize                 import curve_fit, differential_evolution, bisect, minimize
from scipy.ndimage                  import zoom,label, gaussian_filter1d, median_filter, shift, fourier_shift, center_of_mass as C_O_M
from scipy.spatial.transform        import Rotation as R
from scipy.spatial.distance         import cdist
from scipy.spatial                  import KDTree, cKDTree
from scipy.fftpack                  import fftn, ifftn, fftshift
from scipy                          import fftpack              as fft
from scipy.signal                   import find_peaks, savgol_filter
from scipy.interpolate              import RegularGridInterpolator, interp1d, splprep, splev, CubicSpline, PchipInterpolator
from scipy.stats                    import zscore
from scipy                          import ndimage              as nd
from scipy.stats                    import skew, kurtosis
from scipy.stats                    import * 

# skimage
from skimage                        import measure              as skm
from skimage.restoration            import unwrap_phase
from skimage.feature                import canny
from skimage.transform              import hough_circle, hough_circle_peaks
from skimage.draw                   import circle_perimeter
from skimage.morphology             import binary_closing, ball

from xrayutilities                  import en2lam

# Matplotlib settings
from matplotlib import animation, rcParams, colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm, Normalize
from matplotlib.table import Table
from matplotlib import pyplot as plt
from matplotlib import rcParams             as rc
from matplotlib.backends import backend_pdf as be_pdf
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_interactions import ipyplot as iplt, hyperslicer
import matplotlib.gridspec as gridspec
from tabulate import tabulate

# Date and file handling
from datetime import datetime
from os.path import isfile, join, isdir
from os import listdir
from numpy.linalg import norm
from functools import reduce
from IPython.display import display, Math

# Cryptography
from cryptography.utils import CryptographyDeprecationWarning
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

# Plotting configurations
rcParams.update({
    "font.family": "Liberation Serif",  # Use for titles, labels, and body text
    "font.size": 12,         # General font size
    "axes.labelsize": 14,    # X and Y labels font size
    "xtick.labelsize": 12,   # X-axis tick labels
    "ytick.labelsize": 12,   # Y-axis tick labels
    "legend.fontsize": 12,   # Legend font size
    "axes.titlesize": 16,    # Title font size
    "savefig.bbox": "tight",
})

# ANSI colors for terminal output
RED_COLOR = "\033[91m"
RESET_COLOR = "\033[0m"
BLUE_COLOR = "\033[94m"







# bcdi


import bcdi.utils.utilities as util




from bcdi.graph                     import graph_utils          as gu
from bcdi.postprocessing            import postprocessing_utils as pu
from bcdi.graph                     import linecut              as linecut
from bcdi                           import *










import cdiutils

def import_all_cdiutils():
    import cdiutils
    imported_modules = {}
    
    # Import all submodules
    for submodule_name in cdiutils.__submodules__:
        module = importlib.import_module(f"cdiutils.{submodule_name}")
        imported_modules[submodule_name] = module
    
    # Import all classes from class_submodules
    for class_name, module_name in cdiutils.__class_submodules__.items():
        try:
            module = importlib.import_module(f"cdiutils.{module_name}")
            imported_modules[class_name] = getattr(module, class_name)
        except Exception as e:
            print(f"Can't load {class_name} from {module_name}: {str(e)}")
    
    # Import utility functions
    utility_functions = [
        'energy_to_wavelength', 'wavelength_to_energy', 'make_support',
        'get_centred_slices', 'CroppingHandler', 'hot_pixel_filter'
    ]
    for func_name in utility_functions:
        imported_modules[func_name] = getattr(cdiutils, func_name)
    
    # Additional imports
    try:
        from cdiutils.plot.formatting import set_plot_configs, white_interior_ticks_labels, get_figure_size
        imported_modules.update({
            'set_plot_configs': set_plot_configs,
            'white_interior_ticks_labels': white_interior_ticks_labels,
            'get_figure_size': get_figure_size
        })
    except ImportError:
        print("Failed to import formatting functions.")
    
    try:
        from cdiutils.plot.slice import plot_contour
        imported_modules['plot_contour'] = plot_contour
    except ImportError:
        print("Failed to import plot_contour function.")
    
    try:
        from cdiutils import BcdiPipeline
        imported_modules['BcdiPipeline'] = BcdiPipeline
    except ImportError:
        try:
            from cdiutils.process import BcdiPipeline
            imported_modules['BcdiPipeline'] = BcdiPipeline
        except ImportError:
            print("Failed to import BcdiPipeline.")
    
    try:
        from cdiutils.pipeline import get_params_from_variables
        imported_modules['get_params_from_variables'] = get_params_from_variables
    except ImportError:
        try:
            from cdiutils.process import get_parameters_from_notebook_variables as get_params_from_variables
            imported_modules['get_params_from_variables'] = get_params_from_variables
        except ImportError:
            print("Failed to import get_params_from_variables.")
    
    print("All cdiutils submodules and classes have been imported.")
    return imported_modules

# Automatically load CDI utilities on import
globals().update(import_all_cdiutils())

# Initialize CDI utilities
def initialize():
    global chain_centring
    import cdiutils
    chain_centring = cdiutils.CroppingHandler.chain_centring

initialize()
try:
    from cdiutils.utils import *
    from cdiutils.plot.volume import plot_3d_object , plot_3d_vector_field , plot_3d_voxels
    from cdiutils.plot.slice import plot_multiple_volume_slices , plot_contour ,  plot_slices
except ImportError:
    print("Failed to import cdiutlis subpackages please check the version or the existance of the package.")    

from scipy.optimize import OptimizeWarning
warnings.simplefilter("ignore", category=OptimizeWarning)



