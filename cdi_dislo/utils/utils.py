"""
###########################################################################################################
                                   SCRIPT IMPROVEMENT NOTES                                              #
###########################################################################################################

## 1. Code Optimization & Readability
- Reduce `import *` usage; explicitly import required functions/classes.
- Consolidate repeated imports at the top.
- Replace `print()` with `logging.info()`, `logging.warning()`, etc., for better debugging.
- Maintain a consistent function naming convention (`snake_case` preferred in Python).

## 2. Function-Level Improvements
A. `extract_coefficient_and_exponent(number)`:
   - Handle `number = 0` to avoid `log10(0)` error.
   - Fix: Add a zero condition.

B. `generate_burgers_directions(m, G, ...)`:
   - Validate that `G != [0,0,0]` to prevent division errors.

C. `find_max_and_com_3d(data, window_size=10)`:
   - Ensure `data` is not all NaNs before calling `np.argmax(data)`.

D. `normalize_vector(v)`:
   - Prevent division by zero by adding a small epsilon.

## 3. Optimization of Large Functions
- `crop_data_and_update_coordinates()`: Break into `compute_new_coordinates()`, `apply_padding()`, `crop_data()`, `validate_results()`.
- `data_processing()`: Use NumPy vectorized operations instead of loops.

## 4. Handling Large Datasets Efficiently
- Use `numpy.memmap` for large arrays instead of in-memory `np.array()`.
- Implement multiprocessing with `multiprocessing.Pool()` for speed improvements.
- Avoid modifying global arrays in-place; return new arrays instead.

## 5. Structural Improvements
- Move utility functions (`normalize_vector()`, `check_array_empty()`, etc.) to `utilities.py` for reusability.
- Group related functions into classes (e.g., `CroppingHandler` for cropping functions).

## 6. Final Checklist
✅ Avoid Redundant Imports
✅ Refactor Large Functions
✅ Improve Exception Handling
✅ Use Logging Instead of Print
✅ Consider Memory Optimization Techniques

###########################################################################################################
                                  END OF IMPROVEMENT NOTES                                              #
###########################################################################################################
###########################################################################################################
                                   SCRIPT OVERVIEW                                                      #
###########################################################################################################

This script contains various utilities for processing 3D diffraction data, analyzing dislocations,
and handling BCDI (Bragg Coherent Diffraction Imaging) reconstructions. It includes functions for:

1. **Imports & Dependencies**
   - Imports essential libraries for scientific computing, image processing, and machine learning.

2. **Mathematical & Utility Functions**
   - `extract_coefficient_and_exponent()`: Extracts the coefficient and exponent from a number.
   - `normalize_vector()`: Normalizes a given vector to unit length.
   - `project_vector()`: Projects a vector onto a plane perpendicular to another vector.
   - `rotation_matrix_from_angles()`: Computes a 3D rotation matrix from Euler angles.

3. **3D Data Processing & Cropping**
   - `crop_data_and_update_coordinates()`: Crops 3D data and updates spatial coordinates accordingly.
   - `crop_3d_obj_pos_and_update_coordinates()`: Crops a complex 3D object and recalculates coordinates.
   - `fill_up_support()`: Fills holes inside a binary support mask.

4. **Dislocation & Burgers Vector Analysis**
   - `generate_burgers_directions()`: Computes possible Burgers vector directions based on a reciprocal lattice vector.
   - `get_dislocation_position()`: Identifies dislocation positions from a given mask.

5. **Statistical Analysis & Feature Extraction**
   - `std_data()`: Computes the standard deviation of nonzero elements in a dataset.
   - `mean_z_run()`, `mean_y_run()`, `mean_x_run()`: Computes mean values along different axes.
   - `std_rho()`: Computes the standard deviation of density across multiple scans.

6. **Data Masking & Clustering**
   - `mask_clusters()`: Identifies and masks clusters in 3D data.
   - `get_largest_component()`: Extracts the largest connected component from a binary mask.

7. **BCDI Data Processing & Reconstruction Handling**
   - `data_processing()`: Processes raw reconstruction data from multiple scans and extracts key parameters.
   - `load_reco_from_cxinpz()`: Loads reconstruction data from `.cxi`, `.h5`, or `.npz` files.

8. **Regression & Machine Learning Utilities**
   - `fit_model_regression()`: Fits various regression models (Linear, Decision Tree, SVR, MLP, etc.).

9. **Coordinate System Transformations**
   - `transform_data_paraview_style()`: Rotates and transforms 3D data with a padding buffer to prevent cropping artifacts.

10. **Miscellaneous Functions**
   - `check_array_empty()`: Checks whether a NumPy array is empty.
   - `pad_to_shape()`: Pads an array to a given shape.
   - `optimize_cropping()`: Determines the best cropping size for a dataset.

11. **Execution & Main Script Logic**
   - The script does not currently contain a `main()` function but provides modular utilities
     that can be used independently or within larger pipelines.

###########################################################################################################
                                  END OF SCRIPT OVERVIEW                                                #
###########################################################################################################
"""

import numpy as np
from decimal import Decimal, getcontext
import re
from scipy.ndimage import center_of_mass as C_O_M
from cdiutils.utils import CroppingHandler

import logging
import time


#####################################################################################################################
#####################################################################################################################
# 🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀
# 🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀
# 🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀
def save_vti_from_dictdata(
    dict_data, filename_save, voxel_sizes, amplitude_threshold=0.01
):
    from cdiutils.io.vtk import save_as_vti

    save_as_vti(
        output_path=filename_save,
        voxel_size=tuple(voxel_sizes),
        **{key: item for key, item in dict_data.items()},
    )
    return


def center_angles(angles):
    """
    Centers a list of angles between -max_angle and max_angle.

    Parameters:
        angles (list or np.ndarray): List of angles in degrees or radians.
        max_angle (float): Maximum angle for centering.

    Returns:
        np.ndarray: Angles centered between -max_angle and max_angle.
    """
    import numpy as np

    min_angle = np.nanmin(angles)
    # Convert angles to a numpy array for vectorized operations
    angles = np.array(angles)
    shift_tozero = angles - min_angle
    # Normalize angles to [-max_angle, max_angle]
    max_angle_new = (np.nanmax(shift_tozero)) / 2
    centered_angles = shift_tozero - max_angle_new

    return centered_angles


# Normalize vector
def normalize_vector(v):
    """Normalizes a vector to unit length."""
    import numpy as np

    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-12 else v


# Project vector onto plane perpendicular to another vector
def project_vector(v, t):
    """Projects vector v onto the plane perpendicular to vector t."""
    import numpy as np

    v = np.array(v, dtype=np.float64)  # Ensure `v` is a NumPy array
    t = np.array(t, dtype=np.float64)  # Ensure `t` is a NumPy array
    return v - (np.dot(v, t) / np.linalg.norm(t) ** 2) * t


def fill_up_support(support, plot=False):
    """
    Modify the support by filling any hole inside.
    """
    import numpy as np
    from cdi_dislo.ewen_utilities.plot_utilities import (
        plot_2D_slices_middle_one_array3D,
    )

    def process_axis(support, axis):
        support_cum = np.cumsum(support, axis=axis)
        support_cum_inv = np.flip(
            np.cumsum(np.flip(support, axis=axis), axis=axis), axis=axis
        )
        support_combine = support_cum * support_cum_inv
        return support_combine

    def fill_up_support_parallel(support):
        support_convex = np.zeros(support.shape)

        # Create remote tasks for each axis
        futures = [process_axis(support, axis) for axis in range(support.ndim)]

        # Get results
        results = futures

        # Combine results
        for result in results:
            support_convex[result != 0] = 1

        return support_convex

    support_convex = fill_up_support_parallel(support)

    if plot:
        plot_2D_slices_middle_one_array3D(
            support, cmap="gray_r", fig_title="original support"
        )
        plot_2D_slices_middle_one_array3D(
            support_convex, cmap="gray_r", fig_title="filled support"
        )

    return support_convex


def nan_to_zero(phase):
    import numpy as np

    return np.nan_to_num(phase, nan=0)


def zero_to_nan(data: np.ndarray, boolean_values: bool = False) -> np.ndarray:
    """Convert zero values to np.nan."""
    import numpy as np

    return np.where(data == 0, np.nan, 1 if boolean_values else data)


# 🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀
# 🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀
# 🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀
def extract_coefficient_and_exponent(number):
    """Extracts the coefficient and exponent from a given number."""
    import math

    if number == 0:
        return 0, 0

    # Extract the exponent
    exponent = int(math.log10(np.abs(number)))
    # Calculate the coefficient
    coefficient = number / (10**exponent)
    return coefficient, exponent


def transform_miller_indices(miller_list):
    """Transforms a list of Miller indices from string format to a 2D NumPy array of floats."""
    import numpy as np

    transformed_list = []
    for indices in miller_list:
        transformed = []
        i = 0
        while i < len(indices):
            if indices[i] == "-":  # Handle negative numbers
                transformed.append(-int(indices[i + 1]))
                i += 2  # Skip the next character since it's part of the negative number
            else:
                transformed.append(int(indices[i]))
                i += 1  # Move to the next character
        transformed_list.append(transformed)
    return np.array(transformed_list).astype(float)


def generate_burgers_directions(
    m, G, hkl_max=5, sort_by_hkl=True, angle_vector=None
):
    """
    Generate all possible primitive Burgers vector directions [h, k, l] and their negatives [-h, -k, -l]
    for a given reciprocal lattice vector G, a specific integer m, removing redundant cases where b is
    perpendicular to G.

    Parameters:
        m (int): Integer m to compute possible Burgers vectors. If m = 0, function returns an empty list.
        G (list or numpy array): Reciprocal lattice vector G = [Gx, Gy, Gz].
        hkl_max (int): Maximum np.absolute value for h, k, l components.
        sort_by_hkl (bool): If True, sort by h, k, l. If False, sort by angle.
        angle_vector (list or numpy array): Vector to compute angle with. If None, use G.

    Returns:
        list: List of tuples (direction, angle) where direction is [h, k, l] and angle is in degrees.
    """
    from math import gcd, degrees
    from functools import reduce
    import numpy as np

    if m == 0:
        return []  # Skip m = 0 as it corresponds to invisible dislocations.

    def is_primitive_vector(vector):
        non_zero = [np.abs(x) for x in vector if x != 0]
        if not non_zero:
            return False
        common_divisor = reduce(gcd, non_zero)
        return common_divisor == 1

    def calculate_angle(v1, v2):
        """Compute the angle between two vectors in degrees."""
        dot_product = np.sum(v1[i] * v2[i] for i in range(len(v1)))  # type: ignore
        magnitude_v1 = np.sqrt(np.sum(x**2 for x in v1))  # type: ignore
        magnitude_v2 = np.sqrt(np.sum(x**2 for x in v2))  # type: ignore
        cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
        cos_theta = np.max(
            -1.0, np.min(1.0, cos_theta)
        )  # Ensure cos_theta is within [-1, 1]
        return degrees(np.arccos(cos_theta))

    valid_directions = []
    seen_vectors = set()  # To track unique vectors before sorting
    angle_vector = angle_vector if angle_vector is not None else G

    for h in range(-hkl_max, hkl_max + 1):
        for k in range(-hkl_max, hkl_max + 1):
            for l_ in range(-hkl_max, hkl_max + 1):
                if h == 0 and k == 0 and l_ == 0:
                    continue  # Skip the zero vector

                dot_product = G[0] * h + G[1] * k + G[2] * l_

                if dot_product == 0:
                    continue  # Skip invisible vectors

                if dot_product == 2 * m:
                    if is_primitive_vector([h, k, l_]):
                        vector = tuple([h, k, l_])
                        neg_vector = tuple([-h, -k, -l_])

                        # Ensure we store both b and -b uniquely
                        for v in [vector, neg_vector]:
                            if v not in seen_vectors:
                                seen_vectors.add(v)
                                angle = calculate_angle(angle_vector, v)
                                valid_directions.append((list(v), angle))

    # Sorting improvements
    if sort_by_hkl:
        valid_directions.sort(key=lambda x: (x[0][0], x[0][1], x[0][2]))
    else:
        valid_directions.sort(key=lambda x: x[1])

    return valid_directions


def find_max_and_com_3d(data, window_size=10):
    """
    Finds the position of the maximum value in a 3D array and computes the rounded center of mass (COM)
    within a window around the maximum.

    Parameters:
    - data: 3D numpy array.
    - window_size: Defines the region around the max for COM computation.

    Returns:
    - max_pos: (z, y, x) coordinates of the maximum value.
    - com_pos: (z, y, x) rounded center of mass.
    """
    import numpy as np

    # Ensure data is a NumPy array
    data = np.array(data)

    # Find the coordinates of the maximum value
    max_pos = np.unravel_index(np.argmax(data), data.shape)

    # Define the window around the maximum
    z_min = max(0, max_pos[0] - window_size // 2)
    z_max = min(data.shape[0], max_pos[0] + window_size // 2 + 1)
    y_min = max(0, max_pos[1] - window_size // 2)
    y_max = min(data.shape[1], max_pos[1] + window_size // 2 + 1)
    x_min = max(0, max_pos[2] - window_size // 2)
    x_max = min(data.shape[2], max_pos[2] + window_size // 2 + 1)

    # Extract the sub-region around the max
    sub_region = data[z_min:z_max, y_min:y_max, x_min:x_max]

    # Compute the center of mass in the sub-region
    com_local = C_O_M(sub_region)

    # Adjust COM coordinates to global positions and round them
    com_pos = (
        round(com_local[0] + z_min),  # type: ignore
        round(com_local[1] + y_min),  # type: ignore
        round(com_local[2] + x_min),  # type: ignore
    )

    return max_pos, com_pos


def format_as_4digit_string(number):
    return f"{number:04d}"


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def IfStringRepresentsFloat(x):
    for i in range(len(x)):
        # print(x[0:len(x)-i])
        if isfloat(x[0 : len(x) - i]):
            if float(x[0 : len(x) - i]) > 10000:
                return 0.0
            else:
                return float(x[0 : len(x) - i])
            break
        #        else:
        #            if isfloat(x[0:-1]):                return float(x[0:-1])
        #            else:
        #                if isfloat(x[0:-2]):        return float(x[0:-2])
        else:
            if i < len(x):
                continue
            else:
                return False


def check_array_empty(array__):
    """
    Check if a NumPy array is empty.

    Parameters:
    - array (numpy.ndarray): The array to check.

    Returns:
    - bool: True if the array is empty, False otherwise.
    """
    if np.array(array__).size == 0:
        return True
    else:
        return False


def array_to_dict(array):
    """Converts a NumPy array to a dictionary.

    Args:
      array: A NumPy array.

    Returns:
      A dictionary.
    """
    if array.dtype.names is None:
        array = array.flatten()
        dictionary = {}
        for index, value in enumerate(array):
            dictionary[index] = value
    else:
        dictionary = {}
        for key, value in zip(array.dtype.names, array.tolist()):
            dictionary[key] = value
    return dictionary


def check_type(var):
    if isinstance(var, (list, tuple, np.ndarray)):
        if len(var) > 0:
            # Determine the type of elements
            all_int = all(
                [isinstance(item, (int, np.integer)) for item in var]
            )
            all_float = all(
                [isinstance(item, (float, np.floating)) for item in var]
            )
            all_str = all([isinstance(item, (str, np.str_)) for item in var])
            all_bool = all(
                [isinstance(item, (bool, np.bool_)) for item in var]
            )

            if isinstance(var, np.ndarray):
                container_type = "array"
            else:
                container_type = type(var).__name__  # "list" or "tuple"

            # Identify the type of elements
            if all_int:
                element_type = "int"
            elif all_float:
                element_type = "float"
            elif all_str:
                element_type = "string"
            elif all_bool:
                element_type = "bool"
            else:
                element_type = "mixed"

            return f"{container_type} of {element_type}"
        else:
            return f"empty {type(var).__name__}"
    elif isinstance(
        var,
        (int, float, str, bool, np.integer, np.floating, np.str_, np.bool_),
    ):
        return f"{type(var).__name__}"
    else:
        return "Unknown type"


def normalize_methods(methods):
    from collections.abc import Iterable
    import numpy as np

    out = []
    for item in methods:
        # Exclude strings explicitly (they are iterable)
        if isinstance(item, (str, bytes)):
            out.append(item)
            continue

        # Detect numeric containers (tuple/list/np.ndarray)
        if isinstance(item, Iterable):
            try:
                vals = []
                for v in item:
                    # convert NumPy scalars → Python scalars
                    if isinstance(v, np.generic):
                        v = v.item()
                    # keep int or float only
                    if isinstance(v, (int, float)):
                        vals.append(v)
                    else:
                        raise TypeError
                out.append(tuple(vals))
                continue
            except TypeError:
                pass  # not a pure numeric container

        # Fallback: leave untouched
        out.append(item)

    return tuple(out)


def multiply_list_elements(arr):
    result = 1
    for num in arr:
        result *= num
    return result


# Define the DualOutput class for dual printing
class DualOutput:
    def __init__(self, *streams, log_only=False):
        self.streams = streams
        self.log_only = log_only  # If True, output only to the log file

    def write(self, message):
        if (
            self.log_only
        ):  # Output only to the last stream (assumed to be the log file)
            self.streams[-1].write(message)
            self.streams[-1].flush()
        else:  # Output to all streams
            for stream in self.streams:
                stream.write(message)
                stream.flush()  # Ensure immediate writing to output

    def flush(self):
        for stream in self.streams:
            stream.flush()


class Float(float):
    """A custom float class that performs arithmetic operations with 4 significant digits."""

    def __new__(cls, value):
        return float.__new__(cls, value)

    def __init__(self, value):
        self.value = Decimal(str(value))

    def __add__(self, other):
        return Float(self._operate(other, Decimal.__add__))

    def __sub__(self, other):
        return Float(self._operate(other, Decimal.__sub__))

    def __mul__(self, other):
        return Float(self._operate(other, Decimal.__mul__))

    def __truediv__(self, other):
        return Float(self._operate(other, Decimal.__truediv__))

    def _operate(self, other, operation):
        getcontext().prec = 4
        other_value = (
            other.value if isinstance(other, Float) else Decimal(str(other))
        )
        result = operation(self.value, other_value)
        return float(result)

    def __repr__(self):
        return f"{float(self.value):.4f}"


def get_numbers_from_string(string):
    """Extracts the numbers from a string in a professional way.

    Args:
      string: A string containing the numbers.

    Returns:
      A list of numbers, or an empty list if the string does not contain any numbers.
    """
    # Use regular expression to find all numbers in the string
    numbers = re.findall(r"\d+", string)
    # Concatenate the numbers into a single word
    result = "".join(numbers)

    return result


def std_data(data):
    data = data.flatten()
    data = data[data != 0]
    mean_d = np.sum(data) / len(data)
    std = np.sqrt(np.square(data - mean_d).sum() / len(data))
    return std


#####################################################################################################################
#####################################################################################################################
############################################      Rotation utility ##################################################
#####################################################################################################################
#####################################################################################################################
def build_rotation_matrix_from_axes(
    angle_x=0, angle_y=0, angle_z=0, degrees=True, order="xyz"
):
    from scipy.spatial.transform import Rotation as R

    """
    Builds a composite rotation matrix from sequential rotations around X, Y, Z axes.
    
    Parameters:
        angle_x, angle_y, angle_z (float): Rotation angles (degrees or radians)
        degrees (bool): Whether the input angles are in degrees (default: True)
        order (str): Order of application, e.g., 'xyz', 'zyx', etc.
    
    Returns:
        (3, 3) ndarray: Final rotation matrix.
    """
    if degrees:
        angle_x = np.radians(angle_x)
        angle_y = np.radians(angle_y)
        angle_z = np.radians(angle_z)

    # Axis rotation matrices
    Rx = R.from_rotvec(angle_x * np.array([1, 0, 0])).as_matrix()
    Ry = R.from_rotvec(angle_y * np.array([0, 1, 0])).as_matrix()
    Rz = R.from_rotvec(angle_z * np.array([0, 0, 1])).as_matrix()

    # Apply in specified order (default 'xyz')
    rotation_matrices = {"x": Rx, "y": Ry, "z": Rz}
    R_total = np.eye(3)
    for axis in order:
        R_total = rotation_matrices[axis] @ R_total

    return R_total


def alignment_euler_angles(source_vec, target_vec, order="xyz", degrees=True):
    """
    Computes the Euler angles to rotate source_vec to align with target_vec.

    Parameters:
        source_vec (array-like): Starting 3D vector.
        target_vec (array-like): Target 3D vector.
        order (str): Rotation order for Euler angles (e.g. 'xyz', 'zyx').
        degrees (bool): Return angles in degrees if True, radians if False.

    Returns:
        angles (tuple): Rotation angles in the given order.
        rotation_matrix (ndarray): The rotation matrix that aligns the vectors.
    """
    from scipy.spatial.transform import Rotation as R

    source_vec = np.array(source_vec, dtype=float)
    target_vec = np.array(target_vec, dtype=float)
    source_vec /= np.linalg.norm(source_vec)
    target_vec /= np.linalg.norm(target_vec)

    # Compute optimal rotation
    rot, _ = R.align_vectors([target_vec], [source_vec])
    rotation_matrix = rot.as_matrix()

    # Convert to Euler angles
    angles = rot.as_euler(order, degrees=degrees)
    return tuple(angles), rotation_matrix


# Function to apply centered affine transformation
def centered_affine_transform(data, transformation_matrix, order=1):
    """Applies a centered orthogonal transformation to a 3D array."""
    from scipy.ndimage import map_coordinates

    # Validation checks (keep from previous version)
    assert transformation_matrix.shape == (3, 3), "Matrix must be 3x3"
    assert np.allclose(
        transformation_matrix @ transformation_matrix.T, np.eye(3), atol=1e-6
    ), "Matrix must be orthogonal"

    shape = np.array(data.shape)
    center = (shape - 1) / 2

    # Fixed coordinate grid generation
    indices = np.indices(shape)  # type: ignore # Shape (3, dim_x, dim_y, dim_z)
    x, y, z = indices[0], indices[1], indices[2]
    coords = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=-1)

    # Rest remains unchanged
    coords_centered = coords - center
    transformed_coords = (transformation_matrix @ coords_centered.T).T + center
    transformed_data = map_coordinates(
        data, transformed_coords.T, order=order, mode="constant", cval=0
    )

    return transformed_data.reshape(shape)


# Function to create a rotation matrix from angles
def rotation_matrix_from_angles(angle_x, angle_y, angle_z):
    """Compute the 3D rotation matrix from given Euler angles (in degrees)."""
    angle_x, angle_y, angle_z = np.radians([angle_x, angle_y, angle_z])

    # Rotation matrices
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)],
        ]
    )

    Ry = np.array(
        [
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)],
        ]
    )

    Rz = np.array(
        [
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1],
        ]
    )

    # Combined rotation order: Z -> Y -> X
    R = Rx @ Ry @ Rz
    return R


#####################################################################################################################
#####################################################################################################################
############################################crop & general utility ##################################################
#####################################################################################################################
#####################################################################################################################
def remove_bordersurface_mask(mask, order=2):
    """
    Docstring for remove_bordersurface_mask
    Removes border surface from a 3D binary mask based on the specified order.

    :param mask:    Description
    :param order: Description
    """
    from cdi_dislo.geometry.ortho_handler import (
        get_displacement_gradient,
    )

    graadient_modes_mask = (
        np.max(
            nan_to_zero(
                np.abs(
                    np.array(
                        get_displacement_gradient((mask), voxel_size=(1, 1, 1))
                    )
                )
            ),
            axis=0,
        )
        != 0
    ).astype(float)
    if order == 2:
        graadient_graadient_modes_mask = (
            np.max(
                nan_to_zero(
                    np.abs(
                        np.array(
                            get_displacement_gradient(
                                (graadient_modes_mask), voxel_size=(1, 1, 1)
                            )
                        )
                    )
                ),
                axis=0,
            )
            != 0
        )
        mask = (
            ((graadient_modes_mask) * (graadient_graadient_modes_mask) - mask)
            < 0
        ).astype(float)
    else:
        # print("order is one")
        mask = (((graadient_modes_mask) - mask) < 0).astype(float)

    return mask


def get_largest_component(binary_mask):
    """
    Finds and returns the largest connected component in a 3D binary mask.

    Parameters:
    - binary_mask: 3D numpy array (binary mask with values 0 and 1)

    Returns:
    - largest_component_mask: 3D numpy array with the same shape as `binary_mask`,
      where only the largest connected component is set to 1, and all other areas are 0.
    - max_label: the label of the largest component
    - max_size: the size of the largest component
    """
    import numpy as np
    from scipy.ndimage import label

    # Label connected components in the binary mask
    labeled_mask, num_features = label(binary_mask)  # type: ignore

    # Count occurrences of each label, ignoring background label 0
    label_sizes = np.bincount(labeled_mask.flatten())
    label_sizes[0] = 0  # Exclude background

    # Find the label with the maximum size
    max_label = np.argmax(label_sizes)
    max_size = label_sizes[max_label]

    # Create a new mask with only the largest component
    largest_component_mask = (labeled_mask == max_label).astype(int)

    return largest_component_mask, max_label, max_size


def crop_data_and_update_coordinates(
    x, y, z, data, finalshape, pos="com", k_add=20, verbose=True
):
    """
    Crop 3D data around a specified position and update the spatial coordinates accordingly."""
    import numpy as np

    # Get the original data shape
    orig_shape = np.array(data.shape)
    finalshape = np.array(finalshape)
    # Ensure pos and output_shape are numpy arrays for easier manipulation
    type_of_methods = check_type(pos)
    if type_of_methods == "str":
        if pos == "max":
            pos_pad = ("max",)
        elif pos == "com":
            pos_pad = ("com",)
    elif (type_of_methods == "array of float") or (
        type_of_methods == "array of int"
    ):
        pos_pad = (tuple(pos),)
    elif (type_of_methods == "list of float") or (
        type_of_methods == "list of int"
    ):
        pos_pad = (tuple(pos),)
    elif "mixed" in type_of_methods:
        if ("list" in type_of_methods) or ("array" in type_of_methods):
            pos_pad = (tuple(pos),)
    if ("max" in pos) or ("com" in pos):
        pos = np.array([int(np.round(i)) for i in C_O_M(data)])
    else:
        pos = pos
    k_add = 10
    # Calculate padding
    pad_before = finalshape // 2 - pos  # type: ignore
    pad_after = (
        np.maximum((pos + finalshape // 2 + finalshape % 2) - orig_shape, 0)
        + k_add
    )  # type: ignore
    pad_after = [i if i > 0 else 0 for i in pad_after]
    pad_before = [i if i > 0 else 0 for i in pad_before]
    # Pad the array
    padded_data = np.pad(
        data,
        list(zip(pad_before, pad_after)),
        mode="constant",
        constant_values=0,
    )
    # Determine the position for cropping

    (
        crop_data_cdi,  # the output cropped data
        det_ref,  # the detector reference voxel in the full detector frame
        cropped_det_ref,  # the detector reference voxel in the cropped detector frame
        roi,  # the region of interest (ROI) used to crop the data
    ) = CroppingHandler.chain_centring(  # type: ignore
        padded_data,
        methods=pos_pad,  # the list of methods used sequentially # type: ignore
        output_shape=finalshape,  # the output shape you want to work with # type: ignore
        verbose=verbose,  # whether to print logs during the reference voxel search
    )
    # print(array(det_ref)-array(pad_before),array(det_ref),array(pad_before))
    pos_pad_after_crop = np.array(det_ref) - np.array(pad_before)

    mean_step_x = np.mean(np.diff(x))
    mean_step_y = np.mean(np.diff(y))
    mean_step_z = np.mean(np.diff(z))

    # Define the new coordinate ranges based on final shape and mean steps
    start_x = x[pos_pad_after_crop[0]] - ((roi[1] - roi[0]) // 2) * mean_step_x
    end_x = start_x + finalshape[0] * mean_step_x
    new_x_crop = np.linspace(start_x, end_x, finalshape[0])

    start_y = y[pos_pad_after_crop[1]] - ((roi[3] - roi[2]) // 2) * mean_step_y
    end_y = start_y + finalshape[1] * mean_step_y
    new_y_crop = np.linspace(start_y, end_y, finalshape[1])

    start_z = z[pos_pad_after_crop[2]] - ((roi[5] - roi[4]) // 2) * mean_step_z
    end_z = start_z + finalshape[2] * mean_step_z
    new_z_crop = np.linspace(start_z, end_z, finalshape[2])

    # Assertions to check if the cropping and coordinate adjustments are correct
    try:
        assert crop_data_cdi.shape == tuple(finalshape), (
            "Cropped data shape doesn't match finalshape"
        )
        assert (
            len(new_x_crop) == finalshape[0]
            and len(new_y_crop) == finalshape[1]
            and len(new_z_crop) == finalshape[2]
        ), "Cropped coordinate array lengths don't match finalshape"

        new_com = cropped_det_ref
        assert all(
            np.abs(com - finalshape[i] // 2) < 2
            for i, com in enumerate(new_com)
        ), "Center of mass is not near the center of the new coordinate system"
    except AssertionError as e:
        logging.error(f"Assertion failed: {str(e)}")
        raise

    if verbose:
        print(
            f"Original shape: {data.shape}, Cropped shape: {crop_data_cdi.shape}"
        )
        print(
            f"New coordinate ranges: x[{new_x_crop[0]:.2f}, {new_x_crop[-1]:.2f}], y[{new_y_crop[0]:.2f}, {new_y_crop[-1]:.2f}], z[{new_z_crop[0]:.2f}, {new_z_crop[-1]:.2f}]"
        )

    return crop_data_cdi, new_x_crop, new_y_crop, new_z_crop


def crop_3d_obj_pos_and_update_coordinates(
    obj,
    x,
    y,
    z,
    output_shape=(100, 100, 100),
    pos="com",
    k_add=20,
    verbose=False,
):
    """
    Crop complex 3D data around a specified position, with padding if necessary, and update spatial coordinates.
    """
    import numpy as np

    density = np.abs(obj)
    phase = np.angle(obj)

    orig_shape = np.array(obj.shape)
    output_shape = np.array(output_shape)

    if pos == "com":
        pos = np.array([int(np.round(i)) for i in C_O_M(density)])
    elif pos == "max":
        pos = np.array(
            [int(np.round(i)) for i in np.where(density == density.max())]
        )
    else:
        pos = np.array(pos)
    # Calculate padding
    k_add = 20
    pad_before = np.maximum(output_shape // 2 - pos, 0) + k_add
    pad_after = (
        np.maximum(
            (pos + output_shape // 2 + output_shape % 2) - orig_shape, 0
        )
        + k_add
    )

    # Pad the data
    padded_density = np.pad(
        density,
        list(zip(pad_before, pad_after)),
        mode="constant",
        constant_values=0,
    )
    padded_phase = np.pad(
        phase,
        list(zip(pad_before, pad_after)),
        mode="constant",
        constant_values=0,
    )
    print(pad_before, pad_after)

    # Calculate cropping
    if pos == "com":
        new_pos = np.array([int(np.round(i)) for i in C_O_M(padded_density)])
    elif pos == "max":
        new_pos = np.array(
            [
                int(np.round(i))
                for i in np.where(padded_density == padded_density.max())
            ]
        )
    else:
        new_pos = pos + (pad_before)
    print(new_pos, pos + (pad_before))
    start = new_pos - (np.round(output_shape / 2)).astype(int)
    end = start + output_shape

    # Crop the padded data
    cropped_density = padded_density[
        start[0] : end[0], start[1] : end[1], start[2] : end[2]
    ]
    cropped_phase = padded_phase[
        start[0] : end[0], start[1] : end[1], start[2] : end[2]
    ]

    # Estimate mean steps
    mean_step_x = np.mean(np.diff(x))
    mean_step_y = np.mean(np.diff(y))
    mean_step_z = np.mean(np.diff(z))

    start_x, end_x = (
        x[pos[0]] - (pad_before[0] + pos[0]) * mean_step_x,
        x[pos[0]] + (pad_after[0] + orig_shape[0] - pos[0]) * mean_step_x,
    )
    start_y, end_y = (
        y[pos[1]] - (pad_before[1] + pos[1]) * mean_step_y,
        y[pos[1]] + (pad_after[1] + orig_shape[1] - pos[1]) * mean_step_y,
    )
    start_z, end_z = (
        z[pos[2]] - (pad_before[2] + pos[2]) * mean_step_z,
        z[pos[2]] + (pad_after[2] + orig_shape[2] - pos[2]) * mean_step_z,
    )

    # Update and extend coordinates
    new_x = np.arange(start_x, end_x, mean_step_x)
    new_y = np.arange(start_y, end_y, mean_step_y)
    new_z = np.arange(start_z, end_z, mean_step_z)

    new_x_crop = new_x[start[0] : end[0]]
    new_y_crop = new_y[start[1] : end[1]]
    new_z_crop = new_z[start[2] : end[2]]

    # Reconstruct the cropped complex object
    cropped_obj = cropped_density * np.exp(1j * cropped_phase)
    try:
        assert cropped_obj.shape == tuple(output_shape), (
            "Cropped object shape doesn't match output_shape"
        )
        assert (
            len(new_x_crop) == output_shape[2]
            and len(new_y_crop) == output_shape[1]
            and len(new_z_crop) == output_shape[0]
        ), "Cropped coordinate array lengths don't match output_shape"

        if pos == "com":
            new_com = C_O_M(np.abs(cropped_obj))
            assert all(
                np.abs(com - output_shape[i] // 2) < 2
                for i, com in enumerate(new_com)
            ), (
                "Center of mass is not near the center of the new coordinate system"
            )
        elif pos == "max":
            new_max = np.unravel_index(
                np.argmax(np.abs(cropped_obj)), cropped_obj.shape
            )
            assert all(
                np.abs(m - output_shape[i] // 2) < 2
                for i, m in enumerate(new_max)
            ), (
                "Maximum value is not near the center of the new coordinate system"
            )
    except AssertionError as e:
        logging.error(f"Assertion failed: {str(e)}")
        raise

    if verbose:
        logging.info(
            f"Original shape: {obj.shape}, Cropped shape: {cropped_obj.shape}"
        )
        logging.info(
            f"New coordinate ranges: x[{new_x_crop[0]:.2f}, {new_x_crop[-1]:.2f}], y[{new_y_crop[0]:.2f}, {new_y_crop[-1]:.2f}], z[{new_z_crop[0]:.2f}, {new_z_crop[-1]:.2f}]"
        )

    return np.array(cropped_obj), new_x_crop, new_y_crop, new_z_crop


def crop_3d_obj_pos(
    obj, methods=["max", "com"], verbose=True, output_shape=(100, 100, 100)
):
    """
    Crop complex 3D data around a specified position, with padding if necessary.

    Parameters:
    obj (numpy.ndarray): Complex 3D input array
    methods (list): Methods to determine the cropping center (e.g., ["max", "com"] or a specific position)
    verbose (bool): Whether to print detailed information
    output_shape (tuple): Desired shape of the cropped output

    Returns:
    numpy.ndarray: Cropped complex 3D array
    """
    density = np.abs(obj)
    phase = 0.0 - np.angle(np.exp(1j * np.angle(obj)))
    type_of_methods = check_type(methods)

    if ("float" in type_of_methods) or ("int" in type_of_methods):
        # Handle cropping based on a specified position
        if verbose:
            print(f"Will crop around the specified position: {methods}")
        # Use the updated crop_3darray_pos for density and phase separately
        density = crop_3darray_pos(
            density,
            output_shape=output_shape,
            methods=tuple(methods),
            verbose=verbose,
        )
        phase = crop_3darray_pos(
            phase, output_shape=output_shape, methods=methods, verbose=verbose
        )

        # Reconstruct the cropped complex object
        obj = density * np.exp(1j * phase)  # type: ignore

    else:
        # Use the updated crop_3darray_pos for density and phase separately
        det_ref, density = crop_3darray_pos(
            density,
            output_shape=output_shape,
            methods=methods,
            verbose=verbose,
            det_ref_return=True,
        )
        phase = crop_3darray_pos(
            phase, output_shape=output_shape, methods=det_ref, verbose=verbose
        )

        # Reconstruct the cropped complex object
        obj = density * np.exp(1j * phase)  # type: ignore

    return obj


def crop_3darray_pos_v00(
    data,
    output_shape=[100, 100, 100],
    methods=["max", "com"],
    verbose=True,
    det_ref_return=False,
):
    """
    Crop a 3D array around a specified position, with padding if necessary.

    Parameters:
    data (numpy.ndarray): 3D input array
    pos (tuple): Center position for cropping (z, y, x)
    output_shape (tuple): Desired shape of the output (z, y, x)

    Returns:
    numpy.ndarray: Cropped array of shape output_shape
    """
    import numpy as np

    # Ensure pos and output_shape are numpy arrays for easier manipulation
    if ("max" in methods) or ("com" in methods):
        pos = np.array([int(np.round(i)) for i in C_O_M(data)])
    else:
        pos = methods

    output_shape = np.array(output_shape)
    orig_shape = np.array(data.shape)
    k_add = 10
    # Calculate padding
    pad_before = output_shape // 2 - pos
    pad_after = (
        np.maximum(
            (pos + output_shape // 2 + output_shape % 2) - orig_shape, 0
        )
        + k_add
    )
    pad_after = [i if i > 0 else 0 for i in pad_after]
    pad_before = [i if i > 0 else 0 for i in pad_before]
    # Pad the array
    padded_data = np.pad(
        data,
        list(zip(pad_before, pad_after)),
        mode="constant",
        constant_values=0,
    )
    if not isinstance(methods, (np.ndarray, list, tuple)):
        if methods == "max":
            methods = ["max"]
        elif methods == "com":
            methods = ["com"]
    if ("max" in methods) or ("com" in methods):
        print(
            "the methods choosed to centering is max or com or combination of both"
        )
        (
            crop_data_cdi,  # the output cropped data
            det_ref,  # the detector reference voxel in the full detector frame
            _,  # the detector reference voxel in the cropped detector frame
            _,  # the region of interest (ROI) used to crop the data
        ) = CroppingHandler.chain_centring(  # type: ignore
            padded_data,
            methods=methods,  # the list of methods used sequentially # type: ignore
            output_shape=output_shape,  # the output shape you want to work with # type: ignore
            verbose=verbose,  # whether to print logs during the reference voxel search
        )
    else:
        print("the origin of the crop is given by user ", methods)
        # Calculate crop indices
        start = methods - output_shape // 2  # type: ignore
        end = start + output_shape
        print(start, end)
        # Crop the padded array
        crop_data_cdi = padded_data[
            start[0] : end[0], start[1] : end[1], start[2] : end[2]
        ]

    if det_ref_return:
        return det_ref, crop_data_cdi  # type: ignore
    else:
        return crop_data_cdi


def crop_3darray_pos(
    data,
    output_shape=[100, 100, 100],
    methods=["max", "com"],
    verbose=True,
    det_ref_return=False,
):
    """
    Crop a 3D array around a specified position, with padding if necessary.

    Parameters:
    data (numpy.ndarray): 3D input array
    pos (tuple): Center position for cropping (z, y, x)
    output_shape (tuple): Desired shape of the output (z, y, x)

    Returns:
    numpy.ndarray: Cropped array of shape output_shape
    """
    # Ensure pos and output_shape are numpy arrays for easier manipulation
    type_of_methods = check_type(methods)
    if verbose:
        print(f"Type of methods: {type_of_methods}")
    if type_of_methods != "str":
        methods = tuple(methods)
        type_of_methods = check_type(methods)

    if type_of_methods == "str":
        pos = np.array([int(np.round(i)) for i in C_O_M(data)])
        methods = (methods,)
        if methods[0] == "max":
            if verbose:
                print("the method choosed to centering is max")
        elif methods[0] == "com":
            if verbose:
                print("the method choosed to centering is com")
    elif type_of_methods == "tuple of string":
        pos = np.array([int(np.round(i)) for i in C_O_M(data)])
        if verbose:
            print("the methods choosed to centering is a tuple of strings")
    elif (type_of_methods == "tuple of int") or (
        type_of_methods == "tuple of float"
    ):
        if np.array(methods).size == 3:
            methods = (tuple(methods),)
        if len(methods) == 1:
            if verbose:
                print(
                    "the method choosed to centering is a position given by user ",
                    methods,
                )
            pos = np.array([int(v) for v in methods[0]])
            methods = tuple([[int(v) for v in methods[0]]])
        else:
            if verbose:
                print(
                    "the method choosed to centering is a tuple of positions given by user ",
                    methods,
                )
            pos = np.array([int(v) for v in methods[0]])
            methods = tuple([[int(v) for v in methods[0]]])
    else:  # tuple of mixed
        if verbose:
            print(
                "the method choosed to centering is a mixte given by user ",
                methods,
            )
        pos = np.array([int(np.round(i)) for i in C_O_M(data)])
        methods = normalize_methods(methods)

    output_shape = np.array(output_shape)
    orig_shape = np.array(data.shape)
    k_add = 10
    # Calculate padding
    pad_before = output_shape // 2 - pos
    pad_after = (
        np.maximum(
            (pos + output_shape // 2 + output_shape % 2) - orig_shape, 0
        )
        + k_add
    )
    # print(pad_before)
    # print(pad_after)
    pad_after = [i if i > 0 else 0 for i in pad_after]
    pad_before = [i if i > 0 else 0 for i in pad_before]
    # Pad the array
    padded_data = np.pad(
        data,
        list(zip(pad_before, pad_after)),
        mode="constant",
        constant_values=0,
    )
    (
        crop_data_cdi,  # the output cropped data
        det_ref,  # the detector reference voxel in the full detector frame
        _,  # the detector reference voxel in the cropped detector frame
        _,  # the region of interest (ROI) used to crop the data
    ) = CroppingHandler.chain_centring(  # type: ignore
        padded_data,
        methods=methods,  # the list of methods used sequentially # type: ignore
        output_shape=output_shape,  # the output shape you want to work with # type: ignore
        verbose=verbose,  # whether to print logs during the reference voxel search
    )

    if det_ref_return:
        return det_ref, crop_data_cdi
    else:
        return crop_data_cdi


def transform_data_paraview_style(
    data, angle_x, angle_y, angle_z, padding_factor=1.5
):
    """
    Transform a 3D numpy array using rotation angles in a way similar to ParaView,
    with padding to avoid data loss at edges.

    :param data: 3D numpy array of any shape
    :param angle_x: Rotation angle around x-axis in degrees
    :param angle_y: Rotation angle around y-axis in degrees
    :param angle_z: Rotation angle around z-axis in degrees
    :param padding_factor: Factor by which to increase the grid size (default: 1.5)
    :return: Transformed 3D numpy array of the same shape as input
    """
    from scipy.interpolate import RegularGridInterpolator
    import numpy as np

    original_shape = data.shape

    # Pad the data
    pad_width = tuple(
        (int(s * (padding_factor - 1) / 2),) * 2 for s in original_shape
    )
    padded_data = np.pad(data, pad_width, mode="constant", constant_values=0)
    padded_shape = padded_data.shape

    # Convert angles to radians
    angle_x, angle_y, angle_z = np.radians([angle_x, angle_y, angle_z])

    # Create rotation matrices
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)],
        ]
    )

    Ry = np.array(
        [
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)],
        ]
    )

    Rz = np.array(
        [
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1],
        ]
    )

    # Combine rotations (order: Z, Y, X)
    R = Rx @ Ry @ Rz

    # Create coordinate grid for padded shape
    x, y, z = np.meshgrid(
        np.arange(padded_shape[0]),
        np.arange(padded_shape[1]),
        np.arange(padded_shape[2]),
        indexing="ij",
    )
    coords = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    # Center coordinates
    center = np.array([(s - 1) / 2 for s in padded_shape])
    coords_centered = coords - center

    # Apply rotation
    rotated_coords = (R @ coords_centered.T).T + center

    # Interpolate
    interpolator = RegularGridInterpolator(
        (
            np.arange(padded_shape[0]),
            np.arange(padded_shape[1]),
            np.arange(padded_shape[2]),
        ),
        padded_data,
        bounds_error=False,
        fill_value=0,
    )
    transformed_data = interpolator(rotated_coords).reshape(padded_shape)

    # Crop back to original size
    start = tuple(
        int((ps - os) / 2) for ps, os in zip(padded_shape, original_shape)
    )
    end = tuple(s + os for s, os in zip(start, original_shape))
    transformed_data = transformed_data[
        start[0] : end[0], start[1] : end[1], start[2] : end[2]
    ]

    return transformed_data


def crop_3darray_pos_old(data, finalshape, pos="com"):
    """
    Crop a 3D array around a specified position, with padding if necessary.

    Parameters:
    data (numpy.ndarray): 3D input array
    pos (tuple): Center position for cropping (z, y, x)
    finalshape (tuple): Desired shape of the output (z, y, x)

    Returns:
    numpy.ndarray: Cropped array of shape finalshape
    """
    from cdi_dislo.ewen_utilities.plot_utilities import plot_3D_projections

    # Ensure pos and finalshape are numpy arrays for easier manipulation
    if pos == "com":
        pos = np.array([int(np.round(i)) for i in C_O_M(data)])
    elif pos == "max":
        pos = np.array(
            [int(np.round(i)) for i in np.where(data == np.nanmax(data))[0]]
        )
    else:
        pos = np.array(pos)
    finalshape = np.array(finalshape)
    orig_shape = np.array(data.shape)
    k_add = 10
    # Calculate padding
    pad_before = finalshape // 2 - pos
    pad_after = (
        np.maximum((pos + finalshape // 2 + finalshape % 2) - orig_shape, 0)
        + k_add
    )
    pad_after = [i if i > 0 else 0 for i in pad_after]
    pad_before = [i if i > 0 else 0 for i in pad_before]
    print(pad_before)
    # Pad the array
    padded_data = np.pad(
        data,
        list(zip(pad_before, pad_after)),
        mode="constant",
        constant_values=0,
    )
    plot_3D_projections(
        zero_to_nan(padded_data),
        log_scale=True,
        cmap="jet",
    )

    # Adjust position for padded array

    if pos == "com":
        new_pos = np.array([int(np.round(i)) for i in C_O_M(padded_data)])
    elif pos == "max":
        new_pos = np.array(
            [
                int(np.round(i))
                for i in np.where(padded_data == np.nanmax(padded_data))[0]
            ]
        )
    else:
        new_pos = pos + pad_before

    # Calculate crop indices
    start = new_pos - finalshape // 2
    end = start + finalshape
    print(start, end)
    # Crop the padded array
    cropped = padded_data[
        start[0] : end[0], start[1] : end[1], start[2] : end[2]
    ]

    return cropped


def mask_clusters(
    data,
    threshold_factor=0.2,
    dilation_iterations=20,
    return_second_cluster=False,
):
    """
    Mask the cluster with the highest maximum value (or second highest if specified) and its surrounding region in 3D data.

    Parameters:
    data (numpy.ndarray): 3D input data
    threshold_factor (float): Factor to determine the threshold (default: 0.2)
    dilation_iterations (int): Number of iterations for mask dilation (default: 20)
    return_second_cluster (bool): Whether to return the second highest cluster instead of the first (default: False)

    Returns:
    tuple: (masked_data, data_after_masking, dilated_mask)
        masked_data: MaskedArray where the selected cluster and surroundings are unmasked
        data_after_masking: Original data with the selected cluster and surroundings set to zero
        dilated_mask: Boolean mask of the identified region
    """
    import numpy as np
    from scipy import ndimage

    # Step 1: Find first global maximum
    max_value1 = np.max(data)
    max_coords1 = np.unravel_index(np.argmax(data), data.shape)

    # Step 2: Identify first cluster containing maximum
    threshold = max_value1 * threshold_factor
    binary = data > threshold
    labeled, num_features = ndimage.label(binary)  # type: ignore
    max_label1 = labeled[max_coords1]

    # Step 3: Create mask for first cluster and surrounding region
    cluster_mask1 = labeled == max_label1
    dilated_mask1 = ndimage.binary_dilation(
        cluster_mask1, iterations=dilation_iterations
    )

    # Step 4: Apply mask to original data
    masked_data1 = np.ma.array(data, mask=~dilated_mask1)  # type: ignore
    data_after_first_mask = data * (1 - dilated_mask1)

    if not return_second_cluster:
        return masked_data1, data_after_first_mask, dilated_mask1

    # If second cluster is requested, continue processing
    # Step 5: Find second global maximum
    max_value2 = np.max(data_after_first_mask)
    max_coords2 = np.unravel_index(
        np.argmax(data_after_first_mask), data.shape
    )

    # Step 6: Identify second cluster containing maximum
    threshold2 = max_value2 * threshold_factor
    binary2 = data_after_first_mask > threshold2
    labeled2, num_features2 = ndimage.label(binary2)  # type: ignore
    max_label2 = labeled2[max_coords2]

    # Step 7: Create mask for second cluster and surrounding region
    cluster_mask2 = labeled2 == max_label2
    dilated_mask2 = ndimage.binary_dilation(
        cluster_mask2, iterations=dilation_iterations
    )

    # Step 8: Apply masks to original data
    masked_data2 = np.ma.array(data, mask=~dilated_mask2)  # type: ignore
    data_after_masking = data * (1 - dilated_mask2)

    return masked_data2, data_after_masking, dilated_mask2


# data processing
def data_processing(files, dirlist):
    """
    Docstring for data_processing

    :param files: Description
    :param dirlist: Description
    """

    def mean_z_run(data):
        sum_ = data[0]
        for i in range(1, len(data)):
            sum_ = sum_ + data[i]

        return sum_ / len(data)

    import time

    (
        data_allscans_LLK,
        data_allscans_runs,
        data_allscans_rho,
        data_allscans_phi,
        data_allscans_runs_meanz_rho,
        data_allscans_runs_meanz_phi,
        data_allscans_mask,
        data_allscans_COM,
    ) = {}, {}, {}, {}, {}, {}, {}, {}
    rho_min = 1
    for dir_ in range(len(dirlist)):
        start = time.time()
        (
            data_run_nb,
            LLK_runs,
            rho_data,
            phi_data,
            i_COM,
            rho_data_cut,
            phi_data_cut,
            data_run_meanz_rho,
            data_run_meanz_phi,
        ) = [], [], [], [], [], [], [], [], []
        # size = len(files[dir_])
        scan_nb = files[dir_][0][
            files[dir_][0].find("results/S") + 9 : files[dir_][0].find(
                "results/S"
            )
            + 12
        ]
        print(
            "********** scan "
            + str(scan_nb)
            + " with "
            + str(len(files[dir_]))
            + "runs"
            + "**********"
        )
        data_run_nb = [
            int(
                IfStringRepresentsFloat(
                    i[i.find("Run") + 3 : i.find("Run") + 8]
                )  # type: ignore
            )
            for i in files[dir_]
        ]  # type: ignore
        LLK_runs = [
            IfStringRepresentsFloat(
                i[i.find("LLKf") + 16 : i.find("LLKf") + 27]
            )
            for i in files[dir_]
        ]

        rho_data = [np.abs(np.array(np.load(i)["obj"])) for i in files[dir_]]
        phi_data = [np.angle(np.array(np.load(i)["obj"])) for i in files[dir_]]
        # phi_data = [np.mod(np.angle(np.array(np.load(i)['obj']),deg=True), 2*180) for i in files[dir_]]
        rho_data_mask = np.asarray(rho_data) * 0
        for i in range(len(rho_data)):
            rho_data_mask[i][np.where(rho_data[i] > rho_min)] = 1
        i_COM = [
            np.asarray(C_O_M(rho_data_mask[i]))
            for i in range(len(files[dir_]))
        ]
        i_COM = [list(map(int, i_COM[i])) for i in range(len(i_COM))]
        rho_data_mask = [
            rho_data_mask[i][
                i_COM[i][0] - 39 : i_COM[i][0] + 40,
                i_COM[i][1] - 50 : i_COM[i][1] + 50,
                i_COM[i][2] - 50 : i_COM[i][2] + 50,
            ]
            for i in range(len(i_COM))
        ]
        rho_data_cut = [
            rho_data[i][
                i_COM[i][0] - 39 : i_COM[i][0] + 40,
                i_COM[i][1] - 50 : i_COM[i][1] + 50,
                i_COM[i][2] - 50 : i_COM[i][2] + 50,
            ]
            * rho_data_mask[i]
            for i in range(len(i_COM))
        ]
        phi_data_cut = [
            phi_data[i][
                i_COM[i][0] - 39 : i_COM[i][0] + 40,
                i_COM[i][1] - 50 : i_COM[i][1] + 50,
                i_COM[i][2] - 50 : i_COM[i][2] + 50,
            ]
            * rho_data_mask[i]
            for i in range(len(i_COM))
        ]
        if len(rho_data_cut) == 0:
            continue
        i_COM = i_COM - 50 * np.asarray(i_COM) ** 0
        data_run_meanz_rho = [
            mean_z_run(rho_data_cut[i]) for i in range(len(i_COM))
        ]
        data_run_meanz_phi = [
            mean_z_run(phi_data_cut[i]) for i in range(len(i_COM))
        ]
        data_allscans_LLK = {
            **data_allscans_LLK,
            str(scan_nb): np.asarray(LLK_runs),
        }
        data_allscans_runs = {
            **data_allscans_runs,
            str(scan_nb): np.asarray(data_run_nb),
        }
        data_allscans_rho = {
            **data_allscans_rho,
            str(scan_nb): np.asarray(rho_data_cut),
        }
        data_allscans_phi = {
            **data_allscans_phi,
            str(scan_nb): np.asarray(phi_data_cut),
        }
        data_allscans_runs_meanz_rho = {
            **data_allscans_runs_meanz_rho,
            str(scan_nb): data_run_meanz_rho,
        }
        data_allscans_runs_meanz_phi = {
            **data_allscans_runs_meanz_phi,
            str(scan_nb): data_run_meanz_phi,
        }
        data_allscans_mask = {
            **data_allscans_mask,
            str(scan_nb): np.asarray(rho_data_mask),
        }
        data_allscans_COM = {
            **data_allscans_COM,
            str(scan_nb): np.asarray(i_COM),
        }
        end = time.time()
        print(str(int(((end - start) / 60) * 10) / 10) + "min")
        # if dir_>=1: break
    return (
        data_run_meanz_rho,
        data_run_meanz_phi,
        data_allscans_LLK,
        data_allscans_runs,
        data_allscans_rho,
        data_allscans_phi,
        data_allscans_runs_meanz_rho,
        data_allscans_runs_meanz_phi,
        data_allscans_mask,
        data_allscans_COM,
    )  # type: ignore


# modes processing
def modes_processing(files, mypath="/home/abdelrahman/data_sixs_2019/"):
    """
    Docstring for modes_processing

    :param files: Description
    :param mypath: Description
    """
    import h5py
    import numpy as np

    final_selected_runs_allscsan = {
        "100": [np.array([2, 4])],
        "103": [np.array([0, 6])],
        "113": [np.array([0, 1])],
        "400": [np.array([1, 4])],
        "439": [np.array([2, 3])],
        "657": [np.array([1, 16])],
        "668": [np.array([3, 19])],
        "671": [np.array([18, 22])],
    }
    ii_scan = 0
    (
        data_allscans_runs_mode,
        data_allscans_rho_modes,
        data_allscans_phi_modes,
        data_allscans_mask_modes,
        data_allscans_COM_modes,
    ) = {}, {}, {}, {}, {}
    rho_min = 3
    for i_scan in final_selected_runs_allscsan.keys():
        start = time.time()
        data_run_nb, rho_data, phi_data, i_COM, rho_data_cut, phi_data_cut = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        destination_ = [
            mypath
            + "S"
            + i_scan
            + "/selected_runs/"
            + str(i_run_list)
            + "/modes.h5"
            for i_run_list in range(len(final_selected_runs_allscsan[i_scan]))
        ]
        print(
            "**********"
            + str(i_scan)
            + " with "
            + str(len(destination_))
            + "modes"
            + "**********"
        )
        data_run_nb = np.array(
            [
                i_run_list
                for i_run_list in range(
                    len(final_selected_runs_allscsan[i_scan])
                )
            ]
        )
        rho_data = np.array(
            [
                np.abs(
                    np.array(
                        h5py.File(f, "r+")["entry_1"]["data_1"]["data"][0]
                    )
                )
                for f in destination_
            ]
        )  # type: ignore
        phi_data = np.array(
            [
                np.angle(
                    np.array(
                        h5py.File(f, "r+")["entry_1"]["data_1"]["data"][0]
                    )
                )
                for f in destination_
            ]
        )  # type: ignore
        rho_data_mask = np.asarray(rho_data) * 0
        for i in range(len(rho_data)):
            rho_data_mask[i][np.where(rho_data[i] > rho_min)] = 1
        i_COM = np.array(
            [
                np.asarray(C_O_M(rho_data_mask[i]))
                for i in range(len(destination_))
            ]
        )
        i_COM = np.array([list(map(int, i_COM[i])) for i in range(len(i_COM))])
        rho_data_mask = np.array(
            [
                rho_data_mask[i][
                    i_COM[i][0] - 39 : i_COM[i][0] + 40,
                    i_COM[i][1] - 50 : i_COM[i][1] + 50,
                    i_COM[i][2] - 50 : i_COM[i][2] + 50,
                ]
                for i in range(len(i_COM))
            ]
        )
        rho_data_cut = np.array(
            [
                rho_data[i][
                    i_COM[i][0] - 39 : i_COM[i][0] + 40,
                    i_COM[i][1] - 50 : i_COM[i][1] + 50,
                    i_COM[i][2] - 50 : i_COM[i][2] + 50,
                ]
                * rho_data_mask[i]
                for i in range(len(i_COM))
            ]
        )
        phi_data_cut = np.array(
            [
                phi_data[i][
                    i_COM[i][0] - 39 : i_COM[i][0] + 40,
                    i_COM[i][1] - 50 : i_COM[i][1] + 50,
                    i_COM[i][2] - 50 : i_COM[i][2] + 50,
                ]
                * rho_data_mask[i]
                for i in range(len(i_COM))
            ]
        )
        if len(rho_data_cut) == 0:
            continue
        i_COM = i_COM - 50 * np.asarray(i_COM) ** 0

        data_allscans_runs_modes = {
            **data_allscans_runs_mode,
            str(i_scan): np.asarray(data_run_nb),
        }
        data_allscans_rho_modes = {
            **data_allscans_rho_modes,
            str(i_scan): np.asarray(rho_data_cut),
        }
        data_allscans_phi_modes = {
            **data_allscans_phi_modes,
            str(i_scan): np.asarray(phi_data_cut),
        }
        data_allscans_mask_modes = {
            **data_allscans_mask_modes,
            str(i_scan): np.asarray(rho_data_mask),
        }
        data_allscans_COM_modes = {
            **data_allscans_COM_modes,
            str(i_scan): np.asarray(i_COM),
        }
        end = time.time()
        print(str(int(((end - start) / 60) * 1000) / 1000) + "min")
        ii_scan = ii_scan + 1
    return (
        final_selected_runs_allscsan,
        data_allscans_runs_modes,
        data_allscans_rho_modes,
        data_allscans_phi_modes,
        data_allscans_mask_modes,
        data_allscans_COM_modes,
    )  # type: ignore


def runsmode_read(files_sel):
    """read the selected modes from npy files"""
    import numpy as np
    import time

    scan_nb = np.array(
        [
            str(
                int(
                    IfStringRepresentsFloat(
                        files_sel[dir_][0][
                            files_sel[dir_][0].find("/S") + 2 : files_sel[
                                dir_
                            ][0].find("/S")
                            + 6
                        ]
                    )
                )
            )
            for dir_ in range(len(files_sel.keys()))
        ]
    )  # type: ignore
    ii_scan = 0
    data_allscans_rho_modes = {}
    data_allscans_phi_modes = {}
    data_allscans_data_complex_modes = {}
    data_allscans_mask_modes = {}
    for i_scan in scan_nb:
        start = time.time()
        data_complex = []
        rho_data = []
        phi_data = []
        destination_ = files_sel[ii_scan]
        print(
            "**********"
            + str(i_scan)
            + " with "
            + str(len(destination_))
            + "modes"
            + "**********"
        )
        rho_data = np.array(
            [np.abs(np.array(np.load(i)["obj"])) for i in destination_]
        )
        phi_data = np.array(
            [np.angle(np.array(np.load(i)["obj"])) for i in destination_]
        )
        data_complex = np.array(
            [np.array(np.load(i)["obj"]) for i in destination_]
        )
        mask = np.asarray(rho_data) * 0
        mask[np.where(rho_data > 5)] = 1
        rho_data = rho_data * mask
        phi_data = phi_data * mask
        data_complex = data_complex * mask
        data_allscans_rho_modes = {
            **data_allscans_rho_modes,
            str(i_scan): np.asarray(rho_data),
        }
        data_allscans_phi_modes = {
            **data_allscans_phi_modes,
            str(i_scan): np.asarray(phi_data),
        }
        data_allscans_data_complex_modes = {
            **data_allscans_data_complex_modes,
            str(i_scan): np.asarray(data_complex),
        }
        data_allscans_mask_modes = {
            **data_allscans_mask_modes,
            str(i_scan): np.asarray(mask),
        }
        end = time.time()
        print(str(int((end - start) * 1000) / 1000) + "sec")
        ii_scan = ii_scan + 1
    return (
        data_allscans_rho_modes,
        data_allscans_phi_modes,
        data_allscans_data_complex_modes,
        data_allscans_mask_modes,
    )


def modes_read(mypath="/home/abdelrahman/data_sixs_2019/"):
    """read the selected modes from h5 files"""
    import h5py
    import numpy as np
    import time

    final_selected_runs_allscsan = {
        "100": [np.array([2, 4])],
        "103": [np.array([0, 6])],
        "113": [np.array([0, 1])],
        "400": [np.array([1, 4])],
        "439": [np.array([2, 3])],
        "657": [np.array([1, 16])],
        "668": [np.array([3, 19])],
        "671": [np.array([18, 22])],
    }
    ii_scan = 0
    data_allscans_rho_modes = {}
    data_allscans_phi_modes = {}
    data_allscans_data_complex_modes = {}
    data_allscans_mask_modes = {}
    for i_scan in final_selected_runs_allscsan.keys():
        start = time.time()
        data_complex = []
        rho_data = []
        phi_data = []

        destination_ = [
            mypath
            + "S"
            + i_scan
            + "/selected_runs/"
            + str(i_run_list)
            + "/modes.h5"
            for i_run_list in range(len(final_selected_runs_allscsan[i_scan]))
        ]
        print(
            "**********"
            + str(i_scan)
            + " with "
            + str(len(destination_))
            + "modes"
            + "**********"
        )
        rho_data = np.array(
            [
                np.abs(
                    np.array(
                        h5py.File(f, "r+")["entry_1"]["data_1"]["data"][0]
                    )
                )
                for f in destination_
            ]
        )  # type: ignore
        phi_data = np.array(
            [
                np.angle(
                    np.array(
                        h5py.File(f, "r+")["entry_1"]["data_1"]["data"][0]
                    )
                )
                for f in destination_
            ]
        )  # type: ignore
        data_complex = np.array(
            [
                np.array(h5py.File(f, "r+")["entry_1"]["data_1"]["data"][0])
                for f in destination_
            ]
        )  # type: ignore
        mask = np.asarray(rho_data) * 0
        mask[np.where(rho_data > 5)] = 1
        rho_data = rho_data * mask
        phi_data = phi_data * mask
        data_complex = data_complex * mask
        data_allscans_rho_modes = {
            **data_allscans_rho_modes,
            str(i_scan): np.asarray(rho_data),
        }
        data_allscans_phi_modes = {
            **data_allscans_phi_modes,
            str(i_scan): np.asarray(phi_data),
        }
        data_allscans_data_complex_modes = {
            **data_allscans_data_complex_modes,
            str(i_scan): np.asarray(data_complex),
        }
        data_allscans_mask_modes = {
            **data_allscans_mask_modes,
            str(i_scan): np.asarray(mask),
        }
        end = time.time()
        print(str(int((end - start) * 1000) / 1000) + "sec")
        ii_scan = ii_scan + 1
    return (
        final_selected_runs_allscsan,
        data_allscans_rho_modes,
        data_allscans_phi_modes,
        data_allscans_data_complex_modes,
        data_allscans_mask_modes,
    )


def get_dislocation_position(data_mask):
    shape_ = data_mask.shape
    c = np.zeros(shape_)
    for x_i in range(shape_[0]):
        for y_i in range(shape_[1]):
            for z_i in range(2, shape_[2]):
                if data_mask[x_i, y_i, z_i - 1] == 0:
                    if data_mask[x_i, y_i, z_i] == 0:
                        c[x_i, y_i, z_i] = 0
                    else:
                        c[x_i, y_i, z_i] = 10
                else:
                    c[x_i, y_i, z_i] = 0
    for x_i in range(shape_[0]):
        for y_i in range(shape_[1]):
            for z_i in range(shape_[2]):
                if c[x_i, y_i, z_i] == 10:
                    c[x_i, y_i, z_i] = 0
                    break
                else:
                    continue
    dislo_positions = np.array(np.where(c != 0))
    # dislo_positions= np.array([ dislo_positions[i][(dislo_positions[0]==44) | (dislo_positions[0]==35)] for i in range(len(dislo_positions))])
    most_freq_z = np.bincount(dislo_positions[0]).argmax()
    dislo_positions = np.array(
        [
            dislo_positions[i][(dislo_positions[0] == most_freq_z)]
            for i in range(len(dislo_positions))
        ]
    )
    for i_index in range(len(dislo_positions[0])):
        if dislo_positions[0][i_index] != most_freq_z:
            np.delete()  # type: ignore
    return dislo_positions, c


def sum_z_run(data):
    sum_ = data[0]
    for i in range(1, len(data)):
        sum_ = sum_ + data[i]

    return sum_


def sum_y_run(data):
    sum_ = data[:, 0, :]
    for i in range(1, data.shape[1]):
        sum_ = sum_ + data[:, i, :]
    return sum_


def maxcorr_index(corr_data, shift_from_max, allowed_min):
    for i_scan in corr_data.keys():
        min_corr = np.max(corr_data[i_scan]) - shift_from_max
        if min_corr < allowed_min:
            print(
                "scan "
                + str(i_scan)
                + "reconstruction problem. Runs are not well reconstructed."
            )
            continue
        corr_data[i_scan][np.where(corr_data[i_scan] == 1)] = 0
        index_best_run = np.array(
            np.sort(
                list(
                    set(
                        np.array(
                            np.where(corr_data[i_scan] > min_corr)
                        ).flatten()
                    )
                )
            )
        )

        n_frames = len(index_best_run)
        print(
            "**********"
            + i_scan
            + " with "
            + str(n_frames)
            + "selected runs"
            + "**********"
            + "max corr: "
            + str(int(100 * (min_corr + shift_from_max)))
        )
        print(index_best_run)


def std_dltaphi(data_phi, data_mask):
    std_phi_allscan = {}
    for i_scan in data_phi.keys():
        start = time.time()
        print(
            "**********"
            + i_scan
            + " with "
            + str(len(data_phi[i_scan]))
            + "runs"
            + "**********"
        )
        nb_frames = len(data_phi[i_scan])
        std_phi_run = np.zeros((nb_frames, nb_frames))
        for j_run in range(nb_frames):
            for i_run in range(j_run + 1):
                if i_run < j_run:
                    mask = data_mask[i_scan][j_run] * data_mask[i_scan][i_run]
                    len_eff = mask.sum()
                    delta_phi = (
                        np.angle(
                            np.exp(
                                1j
                                * (
                                    data_phi[i_scan][j_run]
                                    - data_phi[i_scan][i_run]
                                )
                            )
                        )
                        * mask
                    )
                    mean_dphi = delta_phi.sum() / len_eff
                    std_phi_run[j_run, i_run] = np.sqrt(
                        np.square((delta_phi - mean_dphi) * mask).sum()
                        / len_eff
                    )
            std_phi_run[:, j_run] = std_phi_run[j_run, :]
            std_phi_run[j_run, j_run] = 7
        std_phi_allscan = {**std_phi_allscan, str(i_scan): std_phi_run}
        end = time.time()
        print(str(int(((end - start) / 60) * 10) / 10) + "min")
    return std_phi_allscan


def dltaphi(data_phi, data_mask):
    # nb_scan = len(data_phi)
    dphi_allscan = {}
    for i_scan in data_phi.keys():
        start = time.time()
        shape_data_ = data_phi[i_scan][0].shape
        print(
            "**********"
            + i_scan
            + " with "
            + str(len(data_phi[i_scan]))
            + "runs"
            + "**********"
        )
        nb_frames = len(data_phi[i_scan])
        delta_phi = np.zeros(
            (
                nb_frames,
                nb_frames,
                shape_data_[0],
                shape_data_[1],
                shape_data_[2],
            )
        )
        for j_run in range(nb_frames):
            for i_run in range(j_run + 1):
                mask = data_mask[i_scan][j_run]
                delta_phi[j_run, i_run] = (
                    np.angle(
                        np.exp(
                            1j
                            * (
                                data_phi[i_scan][j_run]
                                - data_phi[i_scan][i_run]
                            )
                        )
                    )
                    * mask
                )
                # mean_dphi = delta_phi0.sum()/len(mask)
                # delta_phi1 = np.angle(np.exp(1j * (data_phi[i_scan][j_run] -np.flip(data_phi[i_scan][i_run]))))*mask
                # mean_dphi1 = delta_phi1.sum()/len(mask)
                # if mean_dphi1<=mean_dphi0:
                #    mean_dphi=mean_dphi1
                #    delta_phi=delta_phi1
                # else:
                #    mean_dphi=mean_dphi0
                #    delta_phi=delta_phi0
        dphi_allscan = {**dphi_allscan, str(i_scan): delta_phi}
        end = time.time()
        print(str(int(((end - start) / 60) * 10) / 10) + "min")
    return dphi_allscan


def std_rho(data_):
    """calculate the std of rho data for all scans and all runs"""

    def mean_z_run(data):
        sum_ = data[0]
        for i in range(1, len(data)):
            sum_ = sum_ + data[i]

        return sum_ / len(data)

    def mean_y_run(data):
        sum_ = data[:, 0, :]
        nb_ = data.shape[1]
        for i in range(1, nb_):
            sum_ = sum_ + data[:, i, :]
        return sum_ / nb_

    # nb_scan = len(data_)
    std_data_allscsans = {}
    mean_overy_data_allscsans = {}
    mean_overz_data_allscsans = {}
    for i_scan in data_.keys():
        start = time.time()
        print(
            "**********"
            + i_scan
            + " with "
            + str(len(data_[i_scan]))
            + "runs"
            + "**********"
        )
        nb_frames = len(data_[i_scan])
        mean_overy_data_allruns = np.zeros(
            (nb_frames, data_[i_scan][0].shape[0], data_[i_scan][0].shape[2])
        )
        mean_overz_data_allruns = np.zeros(
            (nb_frames, data_[i_scan][0].shape[1], data_[i_scan][0].shape[2])
        )
        std_data_allruns = np.zeros((nb_frames))
        for j_run in range(nb_frames):
            ####### statistical information calculation:STD#############
            combined_data = data_[i_scan][j_run].flatten()
            combined_data = combined_data[np.where(combined_data != 0)]
            # mean_data = np.mean(combined_data)
            std_data_allruns[j_run] = std_data(combined_data)
            mean_overy_data_allruns[j_run] = mean_y_run(data_[i_scan][j_run])
            mean_overz_data_allruns[j_run] = mean_z_run(data_[i_scan][j_run])
            # std_data_allruns            = np.append(std_data_allruns, std_data_run)
            # mean_overy_data_allruns     = np.append(std_data_allruns, mean_y_run(data_[i_scan][j_run]))
            # if std_rho_data<=0.0145:
            #    selected_runs=np.append(selected_runs,data_allscans_runs[i_scan][j_run])
        std_data_allscsans = {
            **std_data_allscsans,
            str(i_scan): std_data_allruns,
        }
        mean_overy_data_allscsans = {
            **mean_overy_data_allscsans,
            str(i_scan): mean_overy_data_allruns,
        }
        mean_overz_data_allscsans = {
            **mean_overz_data_allscsans,
            str(i_scan): mean_overz_data_allruns,
        }
        end = time.time()
        print(str(int(end - start)) + "sec")
    return (
        std_data_allscsans,
        mean_overy_data_allscsans,
        mean_overz_data_allscsans,
    )


def flatten_dict_scan_list_run(dict_):
    if type(dict_) is dict:
        a = [
            np.array(value)
            for value in np.array(list(dict_.items()), dtype=object)[:, 1]
        ]
    else:
        a = dict_
    a = [item for sublist in a for item in sublist]

    return np.array(list(set(np.array(a))))


def get_sphere_intsum_mode(int_ortho_scan_cut_max, range_r, radius_step):
    """get the mean intensity in spherical shells of increasing radius"""

    def xyz_to_thetaphir(x, y, z):
        hxy = np.hypot(x, y)

        r = np.hypot(hxy, z)
        theta = np.array(np.arctan2(z, hxy))
        phi = np.array(np.arctan2(y, x))
        return theta, phi, r

    center_ = (np.array(int_ortho_scan_cut_max.shape) / 2).astype(int)
    dim_x, dim_y, dim_z = center_
    frame = np.indices(int_ortho_scan_cut_max.shape)
    real_x_, real_y_, real_z_ = frame[0], frame[1], frame[2]
    theta, phi, r = xyz_to_thetaphir(
        real_x_ - int(dim_x), real_y_ - int(dim_x), real_z_ - int(dim_x)
    )
    a, b = range_r
    intensity_sum = []
    for r_i in range(a, b, radius_step):
        circle_matrix = np.zeros(int_ortho_scan_cut_max.shape)
        bool_d = np.where(((r >= r_i - radius_step) & (r <= r_i)))
        circle_matrix[bool_d] = 1
        int_ortho_scan_ = circle_matrix * int_ortho_scan_cut_max
        intensity_sum = np.append(
            intensity_sum, np.mean(int_ortho_scan_[int_ortho_scan_ != 0])
        )
    return intensity_sum


def get_max_or_com(ndata, off_set_y, max_or_com="com"):
    if max_or_com == "com":
        try:
            com_ = [int(i) for i in C_O_M(ndata)]  # type: ignore
            com_[1] += off_set_y
        except Exception:
            com_ = [0, 0, 0]
        return com_
    else:
        try:
            max_ = [int(i) for i in np.where(ndata == ndata.max())]
            max_[1] += off_set_y
        except Exception:
            max_ = [0, 0, 0]
        return max_


def load_reco_from_cxinpz(path, multiply_by_mask=False):
    """Load the specfile from the given path"""
    import silx.io
    import numpy as np

    #  return silx.io.specfile.SpecFile(path)
    if (path[-3:] == "cxi") or (path[-2:] == "h5"):
        with silx.io.open(path) as specfile:
            data = specfile
            # data.visititems(print_item)
            int_ = np.array(data["entry_1/image_1/data"])
        specfile.close()
        if multiply_by_mask:  # for reconstructions if available
            mask = np.array(data["entry_1/image_1/mask"])
    if path[-3:] == "npz":
        int_ = np.array(np.load(path)["obj"])
    if multiply_by_mask:
        return int_ * mask  # type: ignore
    else:
        return int_  # type: ignore


def get_max_cut_parametre(parametre, wanted_nb=25):
    to_sort_list = np.array(nan_to_zero(parametre))
    to_sort_list = to_sort_list[to_sort_list != 0]
    to_sort_list.sort()
    max_mtm = to_sort_list[wanted_nb]
    return max_mtm


def format_vector(vector, decimal_places=4):
    formatted = []
    for num in vector:
        num = float(num)
        rounded = round(num, decimal_places)
        if np.abs(rounded) < 0.001 or np.abs(rounded) >= 1000:
            str_num = f"{rounded:.{decimal_places}e}"
        else:
            str_num = f"{rounded:.{decimal_places}f}".rstrip("0").rstrip(".")
        formatted.append(str_num)
    return f"{', '.join(formatted)}"


def pad_to_shape(arr, target_shape, pad_value=0):
    # Calculate the difference in shape
    diff_shape = [(max(0, target_shape[i] - arr.shape[i])) for i in range(3)]

    # Calculate the padding for the beginning and the end
    pad_width = [(diff // 2, diff - diff // 2) for diff in diff_shape]

    # Pad the array
    padded_arr = np.pad(
        arr, pad_width, mode="constant", constant_values=pad_value
    )

    return padded_arr


def optimize_cropping(data, min_sym_crop=True):
    """
    Optimize cropping around the center of mass (COM) while avoiding zero cropping.

    Parameters:
    - data (numpy.ndarray): 3D array of shape (z, y, x).

    Returns:
    - crop_width (int): Maximum width for cropping in any direction.
    """

    # Calculate sum along each direction
    sum_z = np.sum(data, axis=(1, 2))
    sum_y = np.sum(data, axis=(0, 2))
    sum_x = np.sum(data, axis=(0, 1))

    # Find the index of the first and last non-zero elements
    first_nonzero_z = np.argmax(sum_z > 0)
    last_nonzero_z = len(sum_z) - np.argmax(sum_z[::-1] > 0) - 1
    first_nonzero_y = np.argmax(sum_y > 0)
    last_nonzero_y = len(sum_y) - np.argmax(sum_y[::-1] > 0) - 1
    first_nonzero_x = np.argmax(sum_x > 0)
    last_nonzero_x = len(sum_x) - np.argmax(sum_x[::-1] > 0) - 1

    # Calculate crop width for each direction
    crop_width_z = max(last_nonzero_z - first_nonzero_z + 1, 0)
    crop_width_y = max(last_nonzero_y - first_nonzero_y + 1, 0)
    crop_width_x = max(last_nonzero_x - first_nonzero_x + 1, 0)

    # Return the maximum crop width among the three directions
    if min_sym_crop:
        return max(crop_width_z, crop_width_y, crop_width_x)
    else:
        return (crop_width_z, crop_width_y, crop_width_x)


def get_size_3D_pixel(data):
    """
    Calculate the maximum width for cropping in any direction based on the non-zero elements along each axis.

    Parameters:
    - data (numpy.ndarray): 3D array of shape (z, y, x).

    Returns:
    - crop_width (tuple): Maximum width for cropping in each direction (z, y, x).
    """

    # Calculate sum along each direction
    sum_z = np.sum(data, axis=(1, 2))
    sum_y = np.sum(data, axis=(0, 2))
    sum_x = np.sum(data, axis=(0, 1))

    # Find the index of the first and last non-zero elements
    first_nonzero_z = np.argmax(sum_z > 0)
    last_nonzero_z = len(sum_z) - np.argmax(sum_z[::-1] > 0) - 1
    first_nonzero_y = np.argmax(sum_y > 0)
    last_nonzero_y = len(sum_y) - np.argmax(sum_y[::-1] > 0) - 1
    first_nonzero_x = np.argmax(sum_x > 0)
    last_nonzero_x = len(sum_x) - np.argmax(sum_x[::-1] > 0) - 1

    # Calculate crop width for each direction
    crop_width_z = max(last_nonzero_z - first_nonzero_z + 1, 0)
    crop_width_y = max(last_nonzero_y - first_nonzero_y + 1, 0)
    crop_width_x = max(last_nonzero_x - first_nonzero_x + 1, 0)

    # Return the maximum crop width among the three directions
    return (crop_width_z, crop_width_y, crop_width_x)


def fit_model_regression(method, X, y, **kwargs):
    """
    Fit a regression model specified by the method.

    Parameters:
    - method (str): The name of the method to use for fitting the model.
    - X (array-like): Input features.
    - y (array-like): Target variable.
    - **kwargs: Additional keyword arguments to pass to the model constructor.

    Returns:
    - model: Fitted regression model.
    - predictions: Predictions made by the model.
    """
    # Import necessary regression models
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor

    # Initialize the model based on the specified method
    if method == "LinearRegression":
        model = LinearRegression(**kwargs)
    elif method == "RandomForestRegressor":
        model = RandomForestRegressor(**kwargs)
    elif method == "DecisionTreeRegressor":
        model = DecisionTreeRegressor(**kwargs)
    elif method == "KNeighborsRegressor":
        model = KNeighborsRegressor(**kwargs)
    elif method == "SVR":
        model = SVR(**kwargs)
    elif method == "MLPRegressor":
        model = MLPRegressor(**kwargs)
    else:
        raise ValueError(
            "Invalid method specified. Supported methods: LinearRegression, RandomForestRegressor, DecisionTreeRegressor, KNeighborsRegressor, SVR, MLPRegressor."
        )

    # Fit the model
    model.fit(X, y)

    # Make predictions
    predictions = model.predict(X)

    return model, predictions
