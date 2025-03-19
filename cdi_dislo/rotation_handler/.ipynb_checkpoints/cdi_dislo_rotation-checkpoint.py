#####################################################################################################################
# SCRIPT SUMMARY
#####################################################################################################################
# This script provides a comprehensive set of functions for rotation-related utilities, including:
#
# 1. **Orthogonalization & Normalization:**
#    - `orthogonalize_basis`: Orthogonalizes a given 3x3 basis matrix using SVD or QR decomposition.
#    - `normalize_vector`: Ensures vectors have unit length.
#    - `normalize_vectors_3d`: Normalizes a set of 3D vectors.
#
# 2. **Geometric Transformations:**
#    - `cart2pol`, `xyz_to_thetaphir`, `spheric_to_cart`, `cartesian_to_cylind`: Convert between coordinate systems.
#    - `referent_rotzy_trans_xyz`, `referent_roty`, `referent_rotz_`: Perform 3D coordinate transformations.
#    - `transform_coordinates_to_crystallographic`: Maps given coordinates to the crystallographic basis.
#
# 3. **Rotation Matrix Computations:**
#    - `compute_rotation_matrix`: Computes a rotation matrix to align one vector with another.
#    - `compute_rotation_matrix_paraview_style`: Computes rotation matrices in ParaView’s ZYX order.
#    - `rotation_matrix_x/y/z`: Generates rotation matrices for rotations about the X, Y, and Z axes.
#    - `normalize_rotation_matrix`: Ensures rotation matrices remain valid.
#
# 4. **Grid & Data Transformations:**
#    - `apply_rotation_to_data`: Rotates a 3D numpy array with padding.
#    - `transform_data_paraview_style_with_new_axes`: Applies rotation to 3D data while preserving axes.
#    - `transform_grid_to_crystallographic`: Converts a 3D grid to the crystallographic basis.
#
# 5. **Dislocation Analysis:**
#    - `get_select_circle_data_2d` & `get_select_circle_data`: Extracts data within a defined radius around dislocation positions.
#    - `analyze_and_transform_vectors`: Analyzes and transforms crystallographic vectors.
#
# 6. **Utility & Debugging:**
#    - `test_rotation`: Tests alignment of predefined vectors.
#    - `calculate_rotation_angles`: Computes Euler angles to align vectors.
#    - `angle_between_vectors`: Calculates the angle between two vectors.
#####################################################################################################################
#####################################################################################################################
# SUGGESTIONS & IMPROVEMENTS
#####################################################################################################################
# 1. **Code Optimization:**
#    - Consider refactoring repeated code for coordinate transformations (`referent_roty`, `referent_rotz_`).
#    - Consolidate similar functions like `compute_rotation_matrix_paraview_style` and `compute_rotation_matrix`.
#
# 2. **Performance Enhancements:**
#    - `apply_rotation_to_data` could use **numba** or **scipy.ndimage.map_coordinates** for faster interpolation.
#    - Avoid redundant normalizations within functions that already assume unit vectors.
#
# 3. **Better Modularity & Readability:**
#    - Group related functions into separate modules: `rotation_utils.py`, `coordinate_transform.py`, `data_processing.py`.
#    - Add docstrings for missing function descriptions (some functions lack parameter descriptions).
#
# 4. **Validation & Error Handling:**
#    - Implement stricter input validation, e.g., ensure inputs to `compute_rotation_matrix` are non-zero vectors.
#    - Provide meaningful error messages when invalid transformations are attempted.
#
# 5. **Testing & Debugging:**
#    - Add unit tests for key functions (e.g., test cases for `compute_rotation_matrix`).
#    - Implement logging instead of `print()` statements in debugging functions.
#
# 6. **Vectorized Operations:**
#    - Some loops (`get_select_circle_data_2d`, `get_select_circle_data`) could be optimized using **NumPy vectorization**.
#
# 7. **Documentation:**
#    - Add a README with examples demonstrating how to use key functions.
#    - Provide visualization functions for rotation matrices and transformed data.
#####################################################################################################################








from cdi_dislo.common_imports import *

#####################################################################################################################
#####################################################################################################################

#####################################################################################################################
#####################################################################################################################
############################################rotation utility#########################################################
#####################################################################################################################
#####################################################################################################################

def orthogonalize_basis(basis, method='svd'):
    """
    Orthogonalizes a given 3x3 basis matrix using either SVD (default) or QR decomposition.

    :param basis: 3x3 numpy array where each row is a basis vector
    :param method: 'svd' (default) for Singular Value Decomposition, 'qr' for Gram-Schmidt (QR decomposition)
    :return: 3x3 orthonormalized basis matrix
    """
    basis = np.array(basis, dtype=float)
    
    if method == 'svd':
        # SVD-based orthogonalization (more robust)
        U, _, Vt = np.linalg.svd(basis)
        return U @ Vt
    elif method == 'qr':
        # QR-based orthogonalization (Gram-Schmidt)
        Q, _ = np.linalg.qr(basis.T)  # Transpose to work with columns
        return Q.T  # Transpose back so that rows represent basis vectors
    else:
        raise ValueError("Invalid method. Choose 'svd' (default) or 'qr'.")


def get_select_circle_data_2d(data_mask, dislo_position,radius,radius_deviation):
    dim_y, dim_x = data_mask.shape
    circle_matrix = np.zeros(data_mask.shape)
    d_dislo_circle= np.zeros(data_mask.shape)
    theta_c          = np.zeros(data_mask.shape)
    dislo_pos_     = [dislo_position[1],dislo_position[0]]
    for i_y in range(dim_y):
            for i_x in range(dim_x):
                x_d, y_d= referent_2d_trans(i_x, i_y, dislo_pos_)
                theta_c[ i_x, i_y], d_dislo_circle[ i_x, i_y] = cart2pol(x_d, y_d)
                #if (( (d_dislo_circle[i_z, i_y, i_x]<=radius) & (d_dislo_circle[i_z, i_y, i_x]>=radius-2)) & (zc_<10)):
                #    circle_matrix[i_x,i_y,i_z]=1
    #d_dislo_circle=np.rint(d_dislo_circle)
    bool_d = np.where(  ( (d_dislo_circle <= radius +radius_deviation) & (d_dislo_circle >= radius))  )
    circle_matrix[bool_d]        = 1
    circle_matrix                = circle_matrix*data_mask
    theta_c, d_dislo_circle = theta_c * circle_matrix, d_dislo_circle * circle_matrix
    return circle_matrix, d_dislo_circle, theta_c
def referent_2d_trans(x,y,dislo_pos):
    x0,y0=dislo_pos
    x_d     = (x-x0)
    y_d     = (y-y0)
    return x_d,y_d
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return phi,rho



def xyz_to_thetaphir(x,y,z):
    hxy = np.hypot(x, y)
    
    r=np.hypot(hxy, z)
    theta=np.array(np.arctan2(z, hxy))
    phi=np.array(np.arctan2(y, x))
    return theta,phi,r
def spheric_to_cart(r,theta,phi):
    x=r*np.cos(phi)*np.sin(theta)
    y=r*np.sin(phi)*np.sin(theta)
    z= r*np.cos(theta)
    return x, y, z
def cartesian_to_cylind(x,y,z):
    hxy = np.hypot(x, y)
    
    r=hxy
    theta=np.array(np.arctan2(y, x))
    z=z
    return theta,z,r
def angle_btw_two_vect2(vect1,vect2):
    vect_norm=[0,0,1]
    return np.arctan2(np.dot(np.cross(vect1,vect2),vect_norm ),np.dot(vect2,vect2))
def angle_btw_two_vect(vect1,vect2):
    return np.arccos(np.dot(vect1,vect2)/np.sqrt(np.dot(vect1,vect1)*np.dot(vect2,vect2)))
#------------------------------------------------------------------------------------------------------------
def get_select_circle_data(data_mask, dislo_position,angle, radius,radius_deviation,z_min):
    dim_z, dim_y, dim_x    = data_mask.shape
    circle_matrix          = np.zeros(data_mask.shape)
    d_dislo_circle         = np.zeros(data_mask.shape)
    theta_c                = np.zeros(data_mask.shape)
    z_c                    = np.zeros(data_mask.shape)
    dislo_pos_             = [dislo_position[2],dislo_position[1],dislo_position[0]]
    for i_z in range(dim_z):
        for i_y in range(dim_y):
            for i_x in range(dim_x):
                x_d, y_d, z_d= referent_rotzy_trans_xyz(i_x, i_y, i_z, dislo_pos_, angle,np.pi/2)
                theta_c[i_z, i_y, i_x], z_c[i_z, i_y, i_x], d_dislo_circle[i_z, i_y, i_x] = cartesian_to_cylind(x_d, y_d, z_d)
                #if (( (d_dislo_circle[i_z, i_y, i_x]<=radius) & (d_dislo_circle[i_z, i_y, i_x]>=radius-2)) & (zc_<10)):
                #    circle_matrix[i_x,i_y,i_z]=1
    bool_d = np.where(  ( (d_dislo_circle <= radius +radius_deviation) & (d_dislo_circle >= radius)) & (z_c<z_min) )
    circle_matrix[bool_d]        = 1
    circle_matrix                = circle_matrix*data_mask
    theta_c, z_c, d_dislo_circle = theta_c * circle_matrix, z_c * circle_matrix, d_dislo_circle * circle_matrix
    return circle_matrix, d_dislo_circle, theta_c, z_c
#------------------------------------------------------------------------------------------------------------
def referent_rotzy_trans_xyz(x,y,z,dislo_pos,angle_xd,angle_zd):
    x0,y0,z0=dislo_pos
    x_rotz     = (x-x0)*np.cos(angle_xd)  -(y-y0)*np.sin(angle_xd)
    y_rotz     = (y-y0)*np.cos(angle_xd)  +(x-x0)*np.sin(angle_xd)
    z_rotz     = (z-z0)
        
    x_d     = x_rotz*np.cos(angle_zd)  +z_rotz*np.sin(angle_zd)
    z_d     = z_rotz*np.cos(angle_zd)  -x_rotz*np.sin(angle_zd)
    y_d     = y_rotz
    
    return x_d,y_d,z_d
#------------------------------------------------------------------------------------------------------------
def referent_roty(x,y,z,angle_):
    x_d     = x*np.cos(angle_)  +z*np.sin(angle_)
    z_d     = z*np.cos(angle_)  -x*np.sin(angle_)
    y_d     = y
    return x_d,y_d,z_d
#------------------------------------------------------------------------------------------------------------
def referent_rotz_(x,y,z,angle_):
    x_rotz     = x*np.cos(angle_)  -y*np.sin(angle_)
    y_rotz     = y*np.cos(angle_)  +x*np.sin(angle_)
    z_rotz     = z
    return x_rotz,y_rotz,z_rotz

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


#------------------------------------------------------------------------------------------------------------
def normalize_rotation_matrix(R):
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
def test_rotation():
    v = np.array([1, 0, 0])  # Initial vector

    R1 = np.array([[0., -1., 0.],
                   [1., 0., 0.],
                   [0., 0., 1.]])
    
    R2 = np.array([[0., 0., -1.],
                   [0., 1., 0.],
                   [1., 0., 0.]])
    
    R3 = np.array([[0.57735027, -0.57735027, -0.57735027],
                   [0.57735027,  0.78867513, -0.21132487],
                   [0.57735027, -0.21132487,  0.78867513]])

    print("Align [1, 0, 0] with [0, 1, 0]:", R1 @ v)
    print("Align [1, 0, 0] with [0, 0, 1]:", R2 @ v)
    print("Align [1, 0, 0] with [1, 1, 1]:", R3 @ v)
#------------------------------------------------------------------------------------------------------------
def compute_rotation_matrix(v1, v2):
    """
    Compute the rotation matrix to align v1 with v2 using Rodrigues' formula.

    :param v1: The original vector to align (3D).
    :param v2: The target vector to align with (3D).
    :return: 3x3 rotation matrix.
    """
    # Normalize input vectors
    v1 = normalize_vector(v1)
    v2 = normalize_vector(v2)

    # Compute rotation axis
    axis = np.cross(v1, v2)
    axis_norm = np.linalg.norm(axis)
    
    # Handle the case when v1 and v2 are parallel or anti-parallel
    if axis_norm < 1e-6:
        dot_product = np.dot(v1, v2)
        if dot_product > 0:
            return np.eye(3)  # No rotation needed
        else:
            # 180-degree rotation around an arbitrary perpendicular axis
            arbitrary = np.array([1, 0, 0]) if abs(v1[0]) < abs(v1[1]) else np.array([0, 1, 0])
            axis = normalize_vector(np.cross(v1, arbitrary))
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            return 2 * np.outer(axis, axis) - np.eye(3)

    axis = axis / axis_norm

    # Compute angle
    cos_theta = np.dot(v1, v2)
    sin_theta = np.sqrt(1 - cos_theta**2)

    # Rodrigues' formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + sin_theta * K + (1 - cos_theta) * (K @ K)
    return R
#------------------------------------------------------------------------------------------------------------
def normalize_vector(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)
#------------------------------------------------------------------------------------------------------------
def align_x_to_z():
    """Align the vector [1, 0, 0] with [0, 0, 1]."""
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 0, 1])
    return compute_rotation_matrix(v1, v2)
#------------------------------------------------------------------------------------------------------------
def align_x_to_y():
    """Align the vector [1, 0, 0] with [0, 1, 0]."""
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    return compute_rotation_matrix(v1, v2)
#------------------------------------------------------------------------------------------------------------
def align_x_to_111():
    """Align the vector [1, 0, 0] with [1, 1, 1]."""
    v1 = np.array([1, 0, 0])
    v2 = np.array([1, 1, 1])
    return compute_rotation_matrix(v1, v2)
#------------------------------------------------------------------------------------------------------------
def rotation_matrix_z(angle_degrees):
    
    """
    Compute the rotation matrix for a rotation around the z-axis.

    :param angle_degrees: Rotation angle in degrees
    :return: 3x3 rotation matrix
    """
    angle_radians = np.radians(angle_degrees)  # Convert degrees to radians
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    R = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta,  0],
        [0,         0,          1]
    ])
    return R
    
#------------------------------------------------------------------------------------------------------------
def rotation_matrix_x(angle_degrees):
    """
    Compute the rotation matrix for a rotation around the x-axis.

    :param angle_degrees: Rotation angle in degrees
    :return: 3x3 rotation matrix
    """
    angle_radians = np.radians(angle_degrees)  # Convert degrees to radians
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    R = np.array([
        [1, 0,         0        ],
        [0, cos_theta, -sin_theta],
        [0, sin_theta, cos_theta]
    ])
    return R
#------------------------------------------------------------------------------------------------------------
def rotation_matrix_y(angle_degrees):
    """
    Compute the rotation matrix for a rotation around the y-axis.

    :param angle_degrees: Rotation angle in degrees
    :return: 3x3 rotation matrix
    """
    angle_radians = np.radians(angle_degrees)  # Convert degrees to radians
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    R = np.array([
        [cos_theta,  0, sin_theta],
        [0,          1, 0        ],
        [-sin_theta, 0, cos_theta]
    ])
    return R
#------------------------------------------------------------------------------------------------------------
def rotation_matrix_to_angles(rotation_matrix, order='xyz', degrees=True):
    """
    Convert a 3x3 rotation matrix to rotation angles.

    Parameters:
    rotation_matrix (numpy.ndarray): A 3x3 rotation matrix.
    order (str): The order of rotations (e.g., 'xyz', 'zyx'). Default is 'xyz'.
    degrees (bool): Whether to return angles in degrees. Default is True.

    Returns:
    numpy.ndarray: Rotation angles in the specified order.
    """
    if not isinstance(rotation_matrix, np.ndarray) or rotation_matrix.shape != (3, 3):
        raise ValueError("Input must be a 3x3 numpy array.")
    
    # Ensure the matrix is valid for a rotation (orthonormal and determinant == 1)
    if not np.allclose(np.linalg.det(rotation_matrix), 1.0, atol=1e-6):
        raise ValueError("Input matrix must be a valid rotation matrix with determinant 1.")
    
    if not np.allclose(rotation_matrix @ rotation_matrix.T, np.eye(3), atol=1e-6):
        raise ValueError("Input matrix must be orthonormal.")
    
    # Compute the rotation angles
    rotation = R.from_matrix(rotation_matrix)
    angles = rotation.as_euler(order, degrees=degrees)
    return angles

#------------------------------------------------------------------------------------------------------------
def compute_rotation_matrix_paraview_style(input_type='basis', vector=None, vector_direction_toalign_with=[0, 0, 1], current_basis=None, target_basis=None):
    """
    Compute the rotation matrix to transform data from one orthogonal basis to another,
    or to align a vector with a specific direction, considering ParaView's ZYX rotation sweep order.

    :param input_type: 'basis' for basis transformation, 'vector' for vector alignment
    :param vector: 1x3 vector to align with a direction (required for 'vector' mode)
    :param vector_direction_toalign_with: 1x3 direction vector to align with (default: [0, 0, 1])
    :param current_basis: 3x3 matrix where each row is a basis vector of the current orthogonal space
    :param target_basis: 3x3 matrix where each row is a basis vector of the target orthogonal space
    :return: 3x3 rotation matrix
    """
    if input_type == 'basis':
        # Check inputs for basis transformation
        if current_basis is None or target_basis is None:
            raise ValueError("For 'basis' input_type, both 'current_basis' and 'target_basis' must be provided.")
        current_basis = np.array(current_basis, dtype=float)
        target_basis = np.array(target_basis, dtype=float)
        if current_basis.shape != (3, 3) or target_basis.shape != (3, 3):
            raise ValueError("'current_basis' and 'target_basis' must both be 3x3 matrices.")
        
        # Orthogonalize the bases
        current_basis = orthogonalize_basis(current_basis)
        target_basis = orthogonalize_basis(target_basis)

        # Compute the rotation matrix: target_basis * current_basis^-1
        rotation_matrix = target_basis @ current_basis.T

    elif input_type == 'vector':
        # Check inputs for vector alignment
        if vector is None:
            raise ValueError("For 'vector' input_type, the 'vector' parameter must be provided.")
        if len(vector) != 3 or len(vector_direction_toalign_with) != 3:
            raise ValueError("'vector' and 'vector_direction_toalign_with' must both be 1x3 vectors.")
        
        # Normalize the input vector
        vector = np.array(vector, dtype=float)
        vector = vector / np.linalg.norm(vector)
        vector_direction_toalign_with = np.array(vector_direction_toalign_with, dtype=float)
        vector_direction_toalign_with = vector_direction_toalign_with / np.linalg.norm(vector_direction_toalign_with)

        # Define the new x-axis (aligned with the vector)
        new_x = vector

        # Create an arbitrary vector not parallel to the new x-axis
        if np.allclose(new_x, vector_direction_toalign_with):
            raise ValueError("The 'vector' and 'vector_direction_toalign_with' cannot be parallel.")

        arbitrary = vector_direction_toalign_with

        # Compute the new y-axis (orthogonal to the new x-axis)
        new_y = np.cross(new_x, arbitrary)
        new_y = new_y / np.linalg.norm(new_y)

        # Compute the new z-axis (orthogonal to both x and y)
        new_z = np.cross(new_x, new_y)

        # Construct the target basis (new x, y, z)
        target_basis = np.array([new_x, new_y, new_z])

        # Define the current basis (standard x, y, z)
        current_basis = np.eye(3)

        # Compute the rotation matrix: target_basis * current_basis^-1
        rotation_matrix = target_basis.T  # Since current_basis is the identity matrix
    else:
        raise ValueError("Invalid 'input_type'. Must be 'basis' or 'vector'.")
    
    return rotation_matrix
#------------------------------------------------------------------------------------------------------------
def apply_rotation_to_data(data, rotation_matrix, padding_factor=1.5):
    """
    Transform 3D numpy array using a given rotation matrix.

    :param data: 3D numpy array of shape (nx, ny, nz)
    :param rotation_matrix: 3x3 rotation matrix
    :param padding_factor: Factor by which to increase the grid size (default: 1.5)
    :return: Transformed 3D numpy array of the same shape as the input
    """
    original_shape = data.shape
    
    # Pad the data
    pad_width = tuple((int(s * (padding_factor - 1) / 2),) * 2 for s in original_shape)
    padded_data = np.pad(data, pad_width, mode='constant', constant_values=0)
    padded_shape = padded_data.shape
    
    # Create coordinate grid for padded data
    x, y, z = np.meshgrid(
        np.arange(padded_shape[0]),
        np.arange(padded_shape[1]),
        np.arange(padded_shape[2]),
        indexing='ij'
    )
    coords = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    
    # Center coordinates
    center = np.array([(s - 1) / 2 for s in padded_shape])
    coords_centered = coords - center

    # Apply rotation matrix
    rotated_coords = (rotation_matrix @ coords_centered.T).T + center

    # Interpolate data values at new coordinates
    interpolator = RegularGridInterpolator(
        (np.arange(padded_shape[0]), np.arange(padded_shape[1]), np.arange(padded_shape[2])),
        padded_data,
        bounds_error=False,
        fill_value=0
    )
    transformed_data = interpolator(rotated_coords).reshape(padded_shape)

    # Crop back to original size
    start = tuple(int((ps - os) / 2) for ps, os in zip(padded_shape, original_shape))
    end = tuple(s + os for s, os in zip(start, original_shape))
    transformed_data = transformed_data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    
    return transformed_data
#------------------------------------------------------------------------------------------------------------
def analyze_and_transform_vectors(dir111, dir11m1, dir100, loop_direction,max_index=10):
    def find_closest_integer_direction(vector, max_index=5, angle_threshold=1.0):
        closest_directions = []
        for x in range(-max_index, max_index + 1):
            for y in range(-max_index, max_index + 1):
                for z in range(-max_index, max_index + 1):
                    if x == y == z == 0:
                        continue
                    int_vec = np.array([x, y, z])
                    angle = np.arccos(np.clip(np.dot(vector, int_vec) / (np.linalg.norm(vector) * np.linalg.norm(int_vec)), -1.0, 1.0)) * 180 / np.pi
                    if angle < angle_threshold:
                        closest_directions.append((int_vec, angle))
        return sorted(closest_directions, key=lambda x: x[1])

    # Calculate other directions
    dir001 = np.sqrt(3) * (dir111 - dir11m1) / 2
    dir010 = (np.sqrt(3) * (dir111 + dir11m1) - 2 * dir100) / 2
    
    # Create the transformation matrix
    transformation_matrix = np.array([dir100, dir010, dir001]).T
    
    # Compute the inverse transformation matrix
    inverse_transformation_matrix = np.linalg.inv(transformation_matrix)
    
    # Transform the loop direction
    new_v = inverse_transformation_matrix @ loop_direction
    
    # Inverse transform to recover the original vector
    original_v = transformation_matrix @ new_v
    
    # Calculate angles
    deg = 180 / np.pi
    angle_btw_vectors = lambda v1, v2: np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    original_angle_with_111 = angle_btw_vectors(loop_direction, (1, 1, 1)) * deg
    new_angle_with_111 = angle_btw_vectors(new_v, (1, 1, 1)) * deg
    
    # Find closest integer directions for pb
    closest_directions = find_closest_integer_direction(new_v, max_index=max_index, angle_threshold=0.9)
    
    # Additional calculation for [3,1,1] direction
    pb311 = closest_directions[0][0]
    angle_311 = angle_between_vectors(pb311,(1,1,1))#np.arccos(np.dot(pb311, np.array([1, 1, 1])) / (linalg.norm(pb311)*linalg.norm(np.array([1, 1, 1])))) * 180 / np.pi
    
    # Compile results
    results = {
        "norms": {
            "dir100": np.linalg.norm(dir100),
            "dir010": np.linalg.norm(dir010),
            "dir001": np.linalg.norm(dir001)
        },
        "dot_products": {
            "dir100_dir010": np.dot(dir100, dir010),
            "dir100_dir001": np.dot(dir100, dir001),
            "dir001_dir010": np.dot(dir001, dir010)
        },
        "cross_product_dir100_dir010": np.cross(dir100, dir010),
        "dir001": dir001,
        "original_vector": loop_direction,
        "transformed_vector": new_v,
        "recovered_original_vector": original_v,
        "original_angle_with_111": original_angle_with_111,
        "new_angle_with_111": new_angle_with_111,
        "closest_integer_directions": closest_directions,
        "angle_311_ordinary_frame": angle_311,
        "angle_311_crystal_frame": new_angle_with_111
    }
    
    # Print results
    print("Norms and dot products:")
    print(f"Norm of dir100: {results['norms']['dir100']}")
    print(f"Norm of dir010: {results['norms']['dir010']}")
    print(f"Norm of dir001: {results['norms']['dir001']}")
    print(f"Dot product of dir100 and dir010: {results['dot_products']['dir100_dir010']}")
    print(f"Dot product of dir100 and dir001: {results['dot_products']['dir100_dir001']}")
    print(f"Dot product of dir001 and dir010: {results['dot_products']['dir001_dir010']}")
    print(f"Cross product of dir100 and dir010: {results['cross_product_dir100_dir010']}")
    print(f"dir001: {results['dir001']}")
    
    print(f"Original vector: {results['original_vector']} angle with [1, 1, 1] is: {results['original_angle_with_111']:.4f} degrees")
    print(f"Transformed vector: {results['transformed_vector']} angle with [1, 1, 1] is: {results['new_angle_with_111']:.4f} degrees")
    print(f"Recovered original vector: {results['recovered_original_vector']}")    
    
    print("\nClosest integer directions to transformed dirmyst (pb):")
    for direction, angle in results['closest_integer_directions']:
        print(f"Direction: {direction}, Angle: {angle:.4f} degrees")
    
    print(f"\nAngle between {pb311} and [1,1,1] in ordinary frame: {results['angle_311_ordinary_frame']:.4f} degrees")
    print(f"\nAngle between {pb311} and [1,1,1] in crystal frame: {results['angle_311_crystal_frame']:.4f} degrees")
    
    return results
#------------------------------------------------------------------------------------------------------------
def calculate_rotation_angles(vector, target=[1, 0, 0]):
    """
    Calculate rotation angles to align the given vector with the target vector.
    
    :param vector: A list or numpy array of 3 elements [x, y, z] representing the initial vector
    :param target: A list or numpy array of 3 elements [x, y, z] representing the target vector (default is [1, 0, 0])
    :return: A tuple (theta_x, theta_y, theta_z) with rotation angles in degrees around each axis
    """
    # Ensure inputs are numpy arrays and normalize them
    v1 = np.array(vector)
    v2 = np.array(target)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # Compute the rotation axis
    rotation_axis = np.cross(v1, v2)
    
    # If vectors are parallel, rotation axis will be zero. Handle this case:
    if np.allclose(rotation_axis, 0):
        if np.allclose(v1, v2):
            return 0, 0, 0  # Vectors are already aligned
        else:
            rotation_axis = np.array([1, 0, 0])  # Arbitrary axis for 180 degree rotation
    
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    # Compute the rotation angle
    cos_theta = np.dot(v1, v2)
    sin_theta = np.linalg.norm(np.cross(v1, v2))
    theta = np.arctan2(sin_theta, cos_theta)

    # Construct the quaternion
    q = np.array([np.cos(theta/2), *(np.sin(theta/2) * rotation_axis)])

    # Convert quaternion to Euler angles
    sinr_cosp = 2 * (q[0] * q[1] + q[2] * q[3])
    cosr_cosp = 1 - 2 * (q[1]**2 + q[2]**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (q[0] * q[2] - q[3] * q[1])
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2])
    cosy_cosp = 1 - 2 * (q[2]**2 + q[3]**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    # Convert to degrees
    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)

    return roll_deg, pitch_deg, yaw_deg
#------------------------------------------------------------------------------------------------------------
def angle_between_vectors(u, v):
    # Calculate dot product
    dot_product = sum(u_i * v_i for u_i, v_i in zip(u, v))
    
    # Calculate magnitudes
    magnitude_u = math.sqrt(sum(u_i**2 for u_i in u))
    magnitude_v = math.sqrt(sum(v_i**2 for v_i in v))
    
    # Calculate angle in radians and then convert to degrees
    angle_radians = math.acos(dot_product / (magnitude_u * magnitude_v))
    angle_degrees = math.degrees(angle_radians)
    
    return angle_degrees

#------------------------------------------------------------------------------------------------------------
def transform_data_paraview_style_with_new_axes(data, x, y, z, angle_x, angle_y, angle_z, padding_factor=1.5):
    """
    Transform a 3D numpy array using rotation angles in a way similar to ParaView,
    with padding to avoid data loss at edges, and compute new x, y, z axis values.
    
    :param data: 3D numpy array of any shape
    :param x, y, z: 1D numpy arrays representing the axis values along each dimension
    :param angle_x: Rotation angle around x-axis in degrees
    :param angle_y: Rotation angle around y-axis in degrees
    :param angle_z: Rotation angle around z-axis in degrees
    :param padding_factor: Factor by which to increase the grid size (default: 1.5)
    :return: Transformed 3D numpy array of the same shape as input, and new x, y, z axis values
    """
    original_shape = data.shape

    # Pad the data array
    pad_width = tuple((int(s * (padding_factor - 1) / 2),) * 2 for s in original_shape)
    padded_data = np.pad(data, pad_width, mode='constant', constant_values=0)
    padded_shape = padded_data.shape

    # Convert angles to radians and create rotation matrices (order: X -> Y -> Z)
    angle_x, angle_y, angle_z = np.radians([angle_x, angle_y, angle_z])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                   [0, 1, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z), np.cos(angle_z), 0],
                   [0, 0, 1]])
    R = Rx @ Ry @ Rz  # Combined rotation matrix in X -> Y -> Z order

    # --- First grid: for data transformation ---
    # Create an index-based grid for rotating the data
    x_idx, y_idx, z_idx = np.meshgrid(np.arange(padded_shape[0]), 
                                       np.arange(padded_shape[1]), 
                                       np.arange(padded_shape[2]), 
                                       indexing='ij')
    coords_idx = np.stack((x_idx, y_idx, z_idx), axis=-1).reshape(-1, 3)

    # Center coordinates around the middle of the padded data array
    center_idx = np.array([(s - 1) / 2 for s in padded_shape])
    coords_centered_idx = coords_idx - center_idx

    # Apply rotation
    rotated_coords_idx = (R @ coords_centered_idx.T).T + center_idx
    rotated_coords_idx = rotated_coords_idx.reshape(padded_shape + (3,))

    # Interpolate the data on the rotated index grid
    interpolator = RegularGridInterpolator((np.arange(padded_shape[0]), 
                                            np.arange(padded_shape[1]), 
                                            np.arange(padded_shape[2])), 
                                           padded_data, bounds_error=False, fill_value=0)
    transformed_data = interpolator(rotated_coords_idx.reshape(-1, 3)).reshape(padded_shape)

    # Crop back to the original size
    start = tuple(int((ps - os) / 2) for ps, os in zip(padded_shape, original_shape))
    end = tuple(s + os for s, os in zip(start, original_shape))
    transformed_data = transformed_data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

    # --- Second grid: for coordinate transformation ---
    # Use the actual physical grid defined by x, y, and z
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
    coords_real = np.stack((x_grid, y_grid, z_grid), axis=-1).reshape(-1, 3)

    # Center and rotate the physical grid
    center_real = np.array([np.mean(x), np.mean(y), np.mean(z)])
    coords_centered_real = coords_real - center_real
    rotated_coords_real = (R @ coords_centered_real.T).T + center_real
    rotated_coords_real = rotated_coords_real.reshape(original_shape + (3,))

    # Extract new axis values from the rotated real grid
    rotated_x = rotated_coords_real[:, 0, 0, 0]
    rotated_y = rotated_coords_real[0, :, 0, 1]
    rotated_z = rotated_coords_real[0, 0, :, 2]

    return transformed_data, rotated_x, rotated_y, rotated_z





#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
# Function to transform the original grid to the crystallographic basis using R_final
def transform_grid_to_crystallographic(grid_shape, R):
    """
    Transforms a 3D grid from the original frame to the crystallographic basis using a given rotation matrix.

    Args:
        grid_shape: Tuple representing the shape of the 3D grid (nx, ny, nz)
        R: 3x3 rotation matrix that maps the original frame to the crystallographic basis.

    Returns:
        - Transformed grid coordinates in the crystallographic basis.
    """
    import numpy as np

    # Generate the original coordinate grid
    nx, ny, nz = grid_shape
    x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')

    # Stack into coordinate array (flattened)
    original_coords = np.vstack((x.ravel(), y.ravel(), z.ravel()))

    # Apply transformation
    transformed_coords = R @ original_coords

    # Reshape back to the grid shape
    x_transformed = transformed_coords[0].reshape(grid_shape)
    y_transformed = transformed_coords[1].reshape(grid_shape)
    z_transformed = transformed_coords[2].reshape(grid_shape)

    return x_transformed, y_transformed, z_transformed
#------------------------------------------------------------------------------------------------------------
# Function to transform given (x, y, z) coordinates to the crystallographic basis
def transform_coordinates_to_crystallographic(x, y, z, R):
    """
    Transforms given (x, y, z) coordinates from the original frame to the crystallographic basis.

    Args:
        x: X-coordinate(s) in the original frame (can be scalar or array)
        y: Y-coordinate(s) in the original frame (can be scalar or array)
        z: Z-coordinate(s) in the original frame (can be scalar or array)
        R: 3x3 rotation matrix that maps the original frame to the crystallographic basis.

    Returns:
        - Transformed (x, y, z) in the crystallographic basis.
    """

    # Stack input coordinates into a matrix form
    original_coords = np.vstack((np.ravel(x), np.ravel(y), np.ravel(z)))  # Flatten inputs

    # Apply rotation matrix
    transformed_coords = R @ original_coords

    # Reshape to match input shape
    x_cryst = transformed_coords[0].reshape(np.shape(x))
    y_cryst = transformed_coords[1].reshape(np.shape(y))
    z_cryst = transformed_coords[2].reshape(np.shape(z))
    return x_cryst, y_cryst, z_cryst
#------------------------------------------------------------------------------------------------------------
# Function to transform a given vector (already known) to the crystallographic basis
def transform_known_vector_to_crystallographic(vx, vy, vz, R):
    """
    Transforms a given vector (vx, vy, vz) from the original frame to the crystallographic basis.

    Args:
        vx: X-component of the vector in the original frame (can be scalar or array)
        vy: Y-component of the vector in the original frame (can be scalar or array)
        vz: Z-component of the vector in the original frame (can be scalar or array)
        R: 3x3 rotation matrix that maps the original frame to the crystallographic basis.

    Returns:
        - Transformed vector components (vx_cryst, vy_cryst, vz_cryst) in the crystallographic basis.
    """
    import numpy as np

    # Stack vector components into a matrix form
    original_vector = np.array([vx, vy, vz]).reshape(3, -1)

    # Apply the rotation matrix (no translation)
    transformed_vector = R @ original_vector

    # Extract transformed components
    vx_cryst = transformed_vector[0].squeeze()
    vy_cryst = transformed_vector[1].squeeze()
    vz_cryst = transformed_vector[2].squeeze()

    return vx_cryst, vy_cryst, vz_cryst
#------------------------------------------------------------------------------------------------------------
# Updated function to normalize vectors (handling scalars and arrays correctly)
def normalize_vectors_3d(vx, vy, vz):
    """
    Normalizes a set of vectors given their X, Y, and Z components.

    Args:
        vx: X-component of vectors (array or scalar)
        vy: Y-component of vectors (array or scalar)
        vz: Z-component of vectors (array or scalar)

    Returns:
        - Normalized vector components (vx_norm, vy_norm, vz_norm)
    """
    import numpy as np

    # Convert to numpy arrays if inputs are scalars
    vx, vy, vz = np.asarray(vx), np.asarray(vy), np.asarray(vz)

    # Compute vector magnitudes
    magnitudes = np.sqrt(vx**2 + vy**2 + vz**2)

    # Avoid division by zero (if magnitude is 0, set to 1 to prevent NaN)
    magnitudes = np.where(magnitudes == 0, 1, magnitudes)

    # Normalize each component
    vx_norm = vx / magnitudes
    vy_norm = vy / magnitudes
    vz_norm = vz / magnitudes

    return vx_norm, vy_norm, vz_norm
#------------------------------------------------------------------------------------------------------------


