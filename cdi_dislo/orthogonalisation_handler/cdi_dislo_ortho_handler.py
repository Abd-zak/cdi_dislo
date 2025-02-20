#####################################################################################################################
###############################################   SUMMARY BLOCK   ###################################################
#####################################################################################################################

# This script provides a comprehensive set of utilities for phase retrieval, preprocessing, strain computation, 
# and visualization in Bragg Coherent Diffraction Imaging (BCDI). Key functionalities include:

# ✅ Phase Retrieval Setup (`setup_phase_retrieval`)
#    - Configures parameters for phase retrieval using RAAR, HIO, ER, and ML algorithms.
#    - Converts energy to wavelength and processes user-defined support, rebin, and other algorithm parameters.

# ✅ Preprocessing Setup (`setup_preprocessing`)
#    - Handles voxel binning, defect handling, cluster detection, and other preprocessing tasks before reconstruction.

# ✅ Phase Ramp Removal (`remove_phase_ramp`, `remove_phase_ramp_clement`)
#    - Removes unwanted phase ramps from 3D datasets using linear regression.

# ✅ Displacement Gradient & Strain Computation (`get_displacement_gradient`, `get_het_normal_strain`)
#    - Computes the displacement gradient of a dataset using hybrid second-order differentiation.
#    - Estimates strain from lattice parameter deviations.

# ✅ Lattice Parameter & Strain Calculation (`get_lattice_parametre`, `get_strain_from_lattice_parametre`)
#    - Extracts lattice parameters based on Bragg conditions and estimates heterogeneous strain.

# ✅ VTK Export & Visualization (`getting_strain_mapvti`)
#    - Exports strain maps to VTI format for visualization.
#    - Provides debug plots for inspecting different strain components.

#####################################################################################################################
###############################################   IMPROVEMENTS BLOCK   ###############################################
#####################################################################################################################

# 🔹 Refine Import Statements
#    - Instead of `import *`, explicitly import only the required functions for clarity & maintainability.

# 🔹 Ensure Missing Imports Exist
#    - Confirm that `plt`, `array()`, `gu`, and `en2lam()` are properly imported or defined.

# 🔹 Improve Exception Handling
#    - Handle missing dataset paths in HDF5 files gracefully in `getting_strain_mapvti()`.

# 🔹 Optimize `calculate_displacement_gradient()` Calls
#    - Reduce redundant calls to `get_displacement_gradient()` by storing computed values.

# 🔹 Improve `get_lattice_parametre()` Handling
#    - Use `np.linalg.norm()` correctly for handling nested lists of Miller indices (hkl).

# 🔹 Validate `nan_to_zero()` & `zero_to_nan()`
#    - Ensure these functions exist in `common_imports` and handle NaNs correctly.

# The script is well-structured and optimized, but minor refinements in imports, exception handling, and redundant 
# computations could further improve efficiency and robustness. 🚀











from cdi_dislo.common_imports import *
#from cdi_dislo.ewen_utilities.plot_utilities                      import *  #plot_2D_slices_middle_one_array3D
from cdi_dislo.ewen_utilities.plot_utilities                      import plot_3D_projections ,plot_2D_slices_middle_one_array3D
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
####################################ortogonalisation utility#########################################################
#####################################################################################################################
#####################################################################################################################
#------------------------------------------------------------------------------------------------------------
def setup_phase_retrieval(
    energy=8500,  # Default energy in eV
    distance=0.1,  # Default distance in meters
    data2cxi=True,support=None,support_size=None,support_threshold="0.10, 0.30",support_threshold_method="rms",support_only_shrink=False,
    support_update_period=0,    support_smooth_width_begin=2,    support_smooth_width_end=1,    support_post_expand="1,-2,1",
    psf="pseudo-voigt,0.5,0.1,10",    nb_raar=10000,    nb_hio=800,    nb_er=10000,    nb_ml=0,    nb_run=50,    nb_run_keep=200,
    nb_run_keep_max_obj2_out=200,    zero_mask=False,    crop_output=0,    positivity=False,    beta=0.9,    detwin=False,
    rebin="1,1,1",    pixel_size_detector=55e-6,    verbose=100,    output_format="cxi",    live_plot=False,    save_plot=True,
    mpi="run",    print_params=True  # New parameter to control whether to print parameters
):
    """
    Set up and run the phase retrieval process.
    
    Parameters:
        energy (float): Energy in eV (default: 8500).
        distance (float): Distance in meters (default: 0.1).
        print_params (bool): If True, print all parameters (default: True).
        All other parameters are as described in the original specification.
    """
    # Calculate wavelength in meters
    wavelength = float(en2lam(energy))
    
    # Parse support threshold
    support_threshold = [float(x) for x in support_threshold.split(',')]
    
    # Parse rebin
    rebin = [int(x) for x in rebin.split(',')]
    
    # Parse support post expand
    support_post_expand = [int(x) for x in support_post_expand.split(',')]
    
    # Setup phase retrieval parameters
    params = {
        'energy': energy,
        'distance': distance,
        'data2cxi': data2cxi,
        'support': support,
        'support_size': support_size,
        'support_threshold': support_threshold,
        'support_threshold_method': support_threshold_method,
        'support_only_shrink': support_only_shrink,
        'support_update_period': support_update_period,
        'support_smooth_width': (support_smooth_width_begin, support_smooth_width_end),
        'support_post_expand': support_post_expand,
        'psf': psf,
        'nb_raar': nb_raar,
        'nb_hio': nb_hio,
        'nb_er': nb_er,
        'nb_ml': nb_ml,
        'nb_run': nb_run,
        'nb_run_keep': nb_run_keep,
        'nb_run_keep_max_obj2_out': nb_run_keep_max_obj2_out,
        'zero_mask': zero_mask,
        'crop_output': crop_output,
        'positivity': positivity,
        'beta': beta,
        'detwin': detwin,
        'rebin': rebin,
        'pixel_size_detector': pixel_size_detector,
        'wavelength': wavelength,
        'verbose': verbose,
        'output_format': output_format,
        'live_plot': live_plot,
        'save_plot': save_plot,
        'mpi': mpi
    }
    
    # Print parameters if print_params is True
    if print_params:
        print("Phase Retrieval Setup:")
        for key, value in params.items():
            print(f"{key}: {value}")
        print("Phase retrieval setup complete. Ready to run algorithm.")
    
    return params

   
#------------------------------------------------------------------------------------------------------------
def setup_preprocessing(
    preprocess_shape=[200, 200],  # Required, 2 or 3 values. If 2, will take the whole RC
    energy=8500,  # Required, in eV
    hkl=[1, 1, 1],  # Required
    voxel_reference_methods=["max", "com", "com"],
    binning_factors=(1, 1, 1),  # Binning in the 3 directions
    apodize=True,    flip=False,    usetex=False,    show=False,    verbose=True,    debug=True,binning_along_axis0=None,  # Whether or not you want to bin in the RC direction
    light_loading=False,  # Load only the roi defined by the det_reference_voxel and preprocessing_output_shape
    voxel_size=[10.,10.,10.], size=[10.,10.,10.] ,  isosurface=0.05,    handle_defects=True,    jump_to_cluster=False,    clear_former_results=True,
    unwrap_before_orthogonalization= False,       hot_pixel_filter = False,        background_level = None,
):
    """
    Set up preprocessing parameters for data analysis.

    Args:
        preprocessing_output_shape (list): Shape of the output after preprocessing.
        energy (float): Energy in eV.
        hkl (list): Miller indices.
        det_reference_voxel_method (list): Methods for determining reference voxel.
        binning_factors (tuple): Binning factors in 3 directions.
        apodize (bool): Whether to apply apodization.
        flip (bool): Whether to flip the data.
        usetex (bool): Whether to use LaTeX for text rendering.
        show (bool): Whether to show plots.
        verbose (bool): Whether to print verbose output.
        debug (bool): Whether to print debug information.
        binning_along_axis0 (int or None): Binning factor along axis 0.
        light_loading (bool): Whether to use light loading.
        voxel_size (float): Size of voxels.
        isosurface (float): Isosurface value.
        handle_defects (bool): Whether to handle defects.

    Returns:
        dict: A dictionary containing all the preprocessing parameters.
    """
    
    # Create a dictionary with all parameters
    params = {
        'preprocess_shape': preprocess_shape,
        'energy': energy,
        'hkl': hkl,
        'voxel_reference_methods': voxel_reference_methods,
        'binning_factors': binning_factors,
        'apodize': apodize,
        'flip': flip,
        'usetex': usetex,
        'show': show,
        'verbose': verbose,
        'debug': debug,
        'binning_along_axis0': binning_along_axis0,
        'light_loading': light_loading,
        'voxel_size': voxel_size,
        'size': size,
        'isosurface': isosurface,
        'handle_defects': handle_defects,
        'jump_to_cluster':jump_to_cluster,
        'clear_former_results':clear_former_results,
        'unwrap_before_orthogonalization':unwrap_before_orthogonalization,
        'hot_pixel_filter':hot_pixel_filter,
        'background_level':background_level,  }

    # You can add any additional logic or validation here
    if verbose:
        print("Preprocessing parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")

    return params


def remove_phase_ramp(phase: np.array):
    """
    Remove the phase ramp of a 3D volume phase.                                                                                                                                                              

    :param phase: the 3D volume phase (np.ndarray)
    :return_ramp: whether to return the 3D phase ramp (bool)

    :return: the phase without the computed ramp. Return the ramp if 
    return_ramp is true.
    """
    # phase=phase=zero_to_nan(phase)
    x, y, z = np.indices(phase.shape)
    non_nan_coordinates = np.where(np.logical_not(np.isnan(phase)))

    non_nan_phase = phase[non_nan_coordinates]
    x = x[non_nan_coordinates]
    y = y[non_nan_coordinates]
    z = z[non_nan_coordinates]

    X = np.swapaxes(np.array([x, y, z]), 0, 1)
    reg = LinearRegression().fit(X, non_nan_phase)

    x, y, z = np.indices(phase.shape)

    ramp = (
        reg.coef_[0] * x
        + reg.coef_[1] * y
        + reg.coef_[2] * z
        + reg.intercept_
    )

    return phase - ramp, ramp
#------------------------------------------------------------------------------------------------------------
def get_het_normal_strain(displacement: np.ndarray,g_vector: np.ndarray or tuple or list,voxel_size: np.ndarray or tuple or list,gradient_method: str = "hybrid",) -> np.ndarray:
        """
        Compute the heterogeneous normal strain, i.e. the gradient of
        the displacement projected along the measured Bragg peak
        direction.

        Args:
            displacement (np.ndarray): the displacement array
            g_vector (np.ndarray or tuple or list): the position of the measured Bragg peak (com or max of the intensity).
            voxel_size (np.ndarray or tuple or list): voxel size of the array
            gradient_method (str, optional): the method employed to compute the gradient. "numpy" is the traditional gradient.
            "hybrid" compute first order gradient at the surface and second order within the bulk of the reconstruction. Defaults to "hybrid".
        Returns:
            np.ndarray: the heterogeneous normal strain
        """
        displacement_gradient = get_displacement_gradient(displacement,voxel_size,gradient_method)
        displacement_gradient = np.moveaxis(np.asarray(displacement_gradient),source=0,destination=3)
        return np.dot(displacement_gradient, g_vector / np.linalg.norm(g_vector))
#------------------------------------------------------------------------------------------------------------
def get_displacement_gradient(displacement: np.ndarray,voxel_size: np.ndarray or tuple or list,gradient_method: str = "hybrid") -> np.ndarray:
        """
        Calculate the gradient of the displacement.
        Args:
            displacement (np.ndarray): displacement array.
            voxel_size (np.ndarray or tuple or list): the voxel size of the array.
            gradient_method (str, optional): the method employed to compute the gradient. "numpy" is the traditional gradient. 
            "hybrid" compute first order gradient at the surface andsecond order within the bulk of the reconstruction.
            Defaults to "hybrid".
        Raises:
            ValueError: If parsed method is unknown.
        Returns:
            np.ndarray: the gradient of the volume in the three
            directions.
        """
        if gradient_method == "numpy":
            grad_function = np.gradient
        elif gradient_method in ("hybrid", "h"):
            grad_function = hybrid_gradient
        else:
            raise ValueError("Unknown method for normal strain computation.")
        return grad_function(displacement, *voxel_size)


#------------------------------------------------------------------------------------------------------------
def hybrid_gradient(data: np.ndarray, d0: float, d1: float, d2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the gradient of a 3D volume in the 3 directions, 2nd order 
    in the interior of the non-nan object, 1st order at the interface between
    the non-nan object and the surrounding nan values.
    """
   
    def compute_gradient(data, axis, d):
        """Compute gradient along a specific axis."""
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(1, None)
        grad = (data[tuple(slices)] - data[tuple(slice(None, -1) if i == axis else slice(None) for i in range(data.ndim))]) / d
        return grad
    
    
    def compute_mean_gradient(grad, axis):
        """Compute mean gradient along a specific axis."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            slices = [slice(None)] * grad.ndim
            slices[axis] = slice(1, None)
            mean_grad = np.nanmean([grad[tuple(slices)], grad[tuple(slice(None, -1) if i == axis else slice(None) for i in range(grad.ndim))]], axis=0)
        return mean_grad

    # Compute first order gradients in parallel
    grad_futures = [
        compute_gradient(data, 0, d0),
        compute_gradient(data, 1, d1),
        compute_gradient(data, 2, d2)
    ]
    grad_x, grad_y, grad_z = grad_futures

    # Compute mean gradients in parallel
    mean_grad_futures = [
        compute_mean_gradient(grad_x, 0),
        compute_mean_gradient(grad_y, 1),
        compute_mean_gradient(grad_z, 2)
    ]
    grad_x, grad_y, grad_z = mean_grad_futures

    # Pad the results
    result = (
        np.pad(grad_x, ((1, 1), (0, 0), (0, 0)), constant_values=np.nan),
        np.pad(grad_y, ((0, 0), (1, 1), (0, 0)), constant_values=np.nan),
        np.pad(grad_z, ((0, 0), (0, 0), (1, 1)), constant_values=np.nan)
    )


    return result






#------------------------------------------------------------------------------------------------------------
def get_max_without_num(data,num):
    data[data==num]=np.nan
    max_data= np.nanmax(data)
    return max_data  
#------------------------------------------------------------------------------------------------------------
def get_lattice_parametre(nrj,
                          gamma,
                          delta,
                          cch,
                          cy,
                          cz,
                          scale=1,
                          hkl=[1,1,1],
                          s_g=1,
                          s_d=1):
    """
    get the lattice parametre
    """
    vg=gamma+s_g*(cch[1]-cz)/(scale*cos(delta*np.pi/180.))
    vd=delta+s_d*(cy-cch[0])/scale
    tth=np.arccos(np.cos(vg*np.pi/180)*np.cos(vd*np.pi/180))*180/np.pi
    lamb=12.4/nrj
    a0=3.924
    d=lamb/(2*np.sin(tth*np.pi/360))
    if isinstance(hkl[0], list) :
        H= [np.linalg.norm(i) for i in hkl]
    else:
        H=np.linalg.norm(hkl)
    a=d*H
    return a 
#------------------------------------------------------------------------------------------------------------
def get_strain_from_lattice_parametre(a, a0=3.924):
    """
    Calculate the strain of a particle based on its lattice parameter.

    Parameters:
    a (float): The measured lattice parameter of the particle
    a0 (float): The reference lattice parameter (default is 3.924 for platinum)

    Returns:
    float: The calculated strain
    """
    return (a - a0) / a0
#------------------------------------------------------------------------------------------------------------
def remove_phase_ramp_clement(phase: np.ndarray) -> np.ndarray:
        """
        Remove the phase ramp of a 3D volume phase.

        Args:
            phase (np.ndarray): the 3D volume phase

        Returns:
            np.ndarray: the phase without the computed ramp.
        """
        i, j, k = np.indices(phase.shape)
        non_nan_coordinates = np.where(np.logical_not(np.isnan(phase)))

        non_nan_phase = phase[non_nan_coordinates]
        i = i[non_nan_coordinates]
        j = j[non_nan_coordinates]
        k = k[non_nan_coordinates]

        indices = np.swapaxes(np.array([i, j, k]), 0, 1)
        reg = LinearRegression().fit(indices, non_nan_phase)

        i, j, k = np.indices(phase.shape)

        ramp = (
            reg.coef_[0] * i
            + reg.coef_[1] * j
            + reg.coef_[2] * k
            + reg.intercept_
        )

        return phase - ramp  
#------------------------------------------------------------------------------------------------------------
def phase_offset_to_zero_clement(phase: np.ndarray,support: np.ndarray = None,) -> np.ndarray:
        """
        Set the phase offset to the mean phase value.
        """
        return phase - np.nanmean(phase * support if support else 1) 



#------------------------------------------------------------------------------------------------------------
def getting_strain_mapvti(path="", obj=None, voxel_size=[1,1,1], nb_of_phase_to_test=10,
                              path_to_save="", save_filename_vti='', plot_debug=False, output_shape=(40, 60,60)):
    
    def calculate_displacement_gradient(phase, voxel_size):
        return get_displacement_gradient(phase, voxel_size)
    
    
    def find_closest_to_zero(gradients):
        closest_to_zero_indices = np.argmin(np.abs(nan_to_zero(gradients)), axis=0)
        shape_data = gradients.shape
        i, j, k = np.meshgrid(np.arange(shape_data[1]), np.arange(shape_data[2]), np.arange(shape_data[3]), indexing='ij')
        return gradients[closest_to_zero_indices, i, j, k]
    
   
    def calculate_strain(displacement_gradient_min):
        strain_amp = ((displacement_gradient_min[0]**2 + displacement_gradient_min[1]**2 + displacement_gradient_min[2]**2)**0.5)
        strain_amp = strain_amp / np.nanmax(strain_amp)
        strain_mask = ((nan_to_zero(displacement_gradient_min)!=0.).astype(float).sum(axis=0)!=0.).astype(float)
        return strain_amp, strain_mask
    if path:
        obj = array(h5py.File(path)['entry_1/data_1/data'])[0]
    elif obj is not None:
        obj_list = obj
    else:
        print("no obj or path to mode are provided")
        return None, None

    if str(abs(obj_list).max()) == "nan":
        obj_list = nan_to_zero(abs(obj_list)) * np.exp(1j * nan_to_zero(np.angle(obj_list)))
    
    modulus = zero_to_nan(np.abs(obj_list))
    phase_0 = np.angle(np.exp(1j * zero_to_nan(np.angle(obj_list))))
    
    displacement_gradient_0 = calculate_displacement_gradient(phase_0, voxel_size)
    displacement_gradient_0 = np.asarray(displacement_gradient_0)
    
    all_gradients = [displacement_gradient_0]
    phase_futures = []
    for i_phase in np.linspace(-2*np.pi, 2*np.pi, nb_of_phase_to_test):
        phase_1 = np.angle(np.exp((phase_0+i_phase)*1j))
        phase_futures.append(calculate_displacement_gradient(phase_1, voxel_size))
    
    all_gradients.extend(phase_futures)
    
    all_gradients_x = [grad[0] for grad in all_gradients]
    all_gradients_y = [grad[1] for grad in all_gradients]
    all_gradients_z = [grad[2] for grad in all_gradients]
    
    closest_futures = [
        find_closest_to_zero(np.array(all_gradients_x)),
        find_closest_to_zero(np.array(all_gradients_y)),
        find_closest_to_zero(np.array(all_gradients_z))
    ]
    displacement_gradient_min = np.stack(closest_futures, axis=0)
    
    strain_amp, strain_mask = calculate_strain(displacement_gradient_min)
    
    shd_0, shd_x, sh_y, sh_z = displacement_gradient_min.shape
    print("Displacement Gradient Min shape:", displacement_gradient_min.shape)
    
    print(f"\nOriginal values at position ({shd_x//2,sh_y//2,sh_z//2}) for x direction:")
    print(f"Gradient 0: {displacement_gradient_0[0][shd_x//2,sh_y//2,sh_z//2]}")
    print(f"Min value:  {displacement_gradient_min[0][shd_x//2,sh_y//2,sh_z//2]}")
    
    print(f"\nOriginal values at position ({shd_x//2,sh_y//2,sh_z//2}) for y direction:")
    print(f"Gradient 0: {displacement_gradient_0[1][shd_x//2,sh_y//2,sh_z//2]}")
    print(f"Min value:  {displacement_gradient_min[1][shd_x//2,sh_y//2,sh_z//2]}")
    
    print(f"\nOriginal values at position ({shd_x//2,sh_y//2,sh_z//2}) for z direction:")
    print(f"Gradient 0: {displacement_gradient_0[2][shd_x//2,sh_y//2,sh_z//2]}")
    print(f"Min value:  {displacement_gradient_min[2][shd_x//2,sh_y//2,sh_z//2]}")

    if save_filename_vti:
        gu.save_to_vti(filename=save_filename_vti,voxel_size=list(voxel_size),
                           tuple_array=(nan_to_zero(modulus)                     ,nan_to_zero(phase_0)                   ,nan_to_zero(phase_1),
                                        nan_to_zero(displacement_gradient_min[0]),nan_to_zero(displacement_gradient_0[0]),
                                        nan_to_zero(displacement_gradient_min[1]),nan_to_zero(displacement_gradient_0[1]),
                                        nan_to_zero(displacement_gradient_min[2]),nan_to_zero(displacement_gradient_0[2]),strain_mask),
                       tuple_fieldnames=("modulus"                               ,"phase_0"                              ,"phase_1",
                                        "displacement_gradient_min_x"            ,"displacement_gradient_0_x"            ,
                                        "displacement_gradient_min_y"            ,"displacement_gradient_0_y"            ,
                                        "displacement_gradient_min_z"            ,"displacement_gradient_0_z"            ,"strain_mask"),
                       amplitude_threshold=0.1,    )
    if plot_debug:
        plot_2D_slices_middle_one_array3D(phase_0,vmin=-3,vmax=3,cmap="jet",fig_title="original phase midlle slice")
        if save_filename_vti:
            plt.savefig(path_to_save+"original_phase.png")
        plot_2D_slices_middle_one_array3D(phase_1,vmin=-3,vmax=3,cmap="jet",fig_title=f"phase + {np.round(i_phase,4)} midlle slice")
        if save_filename_vti:
            plt.savefig(path_to_save+"plus_phase.png")
        plot_2D_slices_middle_one_array3D(displacement_gradient_0[1],vmin=-0.3,vmax=0.3,cmap="jet",fig_title="original phase gradient midlle slice")
        if save_filename_vti:
            plt.savefig(path_to_save+"original_gradientphase.png")
        plot_2D_slices_middle_one_array3D(displacement_gradient_min[1],vmin=-0.3,vmax=0.3,cmap="jet",fig_title=f"phase  + {np.round(i_phase,4)} gradient midlle slice")
        if save_filename_vti:
            plt.savefig(path_to_save+"plus_gradientphase.png")
    return nan_to_zero(strain_mask), nan_to_zero(strain_amp)


