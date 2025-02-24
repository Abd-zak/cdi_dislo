#########################################################################################################
# Script Name: cdi_dislo_dislocation.py
# 
# Summary:
# This script provides a comprehensive set of utilities for analyzing dislocations in Bragg Coherent 
# Diffraction Imaging (BCDI) data. It includes functions for preprocessing, visualization, and 
# quantitative analysis of dislocation structures. Key functionalities include:
#
# - **Data Processing & Filtering**:
#   - Prepares BCDI data for analysis, including noise reduction, interpolation, and filtering.
#   - Computes dislocation density, Burgers vector components, and phase unwrapping.
# 
# - **Dislocation Analysis**:
#   - Detects dislocations using various methods, including phase gradients and topological charge.
#   - Computes and visualizes dislocation lines and network structures.
#   - Provides automated and interactive tools for analyzing dislocation properties.
#
# - **Visualization & Plotting**:
#   - 2D and 3D rendering of dislocation structures, phase maps, and diffraction data.
#   - Interactive plotting tools for volume slicing and dislocation tracing.
#   - Supports multiple visualization backends, including Matplotlib, PyVista, and Plotly.
#
# - **Machine Learning & Clustering**:
#   - Uses clustering techniques (KMeans, DBSCAN) for feature extraction and classification.
#   - Employs PCA for dimensionality reduction and anomaly detection.
#
# - **File Handling & Data Export**:
#   - Efficient loading and saving of large datasets in HDF5 format.
#   - Export of processed data and visualization results for further analysis.
#
# - **General Utilities**:
#   - Includes mathematical and statistical tools for signal processing and peak analysis.
#   - Supports custom coordinate transformations and rotation matrices.
#
# Dependencies:
# This script relies on a wide range of scientific computing libraries, including NumPy, SciPy, 
# Matplotlib, Scikit-learn, PyVista, and Xrayutilities, along with custom CDI analysis modules.
#
# Usage:
# The script is designed to be used interactively in Jupyter notebooks or as a standalone 
# module for batch processing of BCDI datasets.
#
# Author: Abdelrahman Zakaria
# Date: 19/02/2025
#########################################################################################################




from cdi_dislo.common_imports import *  # Load standard libraries
from cdi_dislo.ewen_utilities.plot_utilities                      import plot_3D_projections ,plot_2D_slices_middle_one_array3D,plot_2D_slices_middle ,plot_object_module_phase_2d
from cdi_dislo.diff_utils_handler.cdi_dislo_diffutils             import get_abc_direct_space_sixs2019,orth_sixs2019_gridder_def
from cdi_dislo.orthogonalisation_handler.cdi_dislo_ortho_handler  import remove_phase_ramp_abd , phase_offset_to_zero_clement , getting_strain_mapvti ,get_displacement_gradient
remove_phase_ramp=remove_phase_ramp_abd
from cdi_dislo.general_utilities.cdi_dislo_utils                  import  normalize_vector , project_vector, fill_up_support, pad_to_shape, find_max_and_com_3d, center_angles ,crop_3darray_pos ,crop_3d_obj_pos

from cdi_dislo.rotation_handler.cdi_dislo_rotation                import rotation_matrix_from_vectors

from cdi_dislo.plotutilities.cdi_dislo_plotutilities              import plot_3d_array

#####################################################################################################################
#####################################################################################################################
############################################Dislo utility############################################################
#####################################################################################################################
#####################################################################################################################

#------------------------------------------------------------------------------------------------------------
def cylindrical_to_cartesian(r, theta, z):
    """
    Convert cylindrical coordinates (r, theta, z) to Cartesian coordinates (x, y, z).

    Parameters:
        r (ndarray): Radial distance array.
        theta (ndarray): Angular array (in radians).
        z (ndarray): Height array.

    Returns:
        tuple: Arrays of (x, y, z) Cartesian coordinates.
    """
    # Convert to Cartesian coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, z


#------------------------------------------------------------------------------------------------------------
def compare_dislocations(Ux_iso, Uy_iso, Uz_iso, Ux_aniso, Uy_aniso, Uz_aniso, theta, R, r_values, suptitle,marker1="<",marker2=">", save_plot=None,mod_theta=False,remove_jump=True):
    """
    Compare the displacement fields (Ux, Uy, Uz) for isotropic and anisotropic media.

    Parameters:
        Ux_iso, Uy_iso, Uz_iso (ndarray): Isotropic displacement fields.
        Ux_aniso, Uy_aniso, Uz_aniso (ndarray): Anisotropic displacement fields.
        theta (ndarray): Angular array (theta).
        R (ndarray): Radial distance array.
        r_values (list): List of radial distances for comparison.
        suptitle (str): Title for the figure.
        save_plot (str, optional): File path to save the plot. If None, the plot is not saved.
    """

    # Define the middle z-slice index
    z_mid = Ux_iso.shape[2] // 2

    # Extract the middle z-slice
    Ux_iso_mid = Ux_iso[:, :, z_mid]
    Uy_iso_mid = Uy_iso[:, :, z_mid]
    Uz_iso_mid = Uz_iso[:, :, z_mid]

    Ux_aniso_mid = Ux_aniso[:, :, z_mid]
    Uy_aniso_mid = Uy_aniso[:, :, z_mid]
    Uz_aniso_mid = Uz_aniso[:, :, z_mid]

    # Define flags to check if each displacement field has non-zero values
    trig_plot_x = np.any(Ux_iso_mid != 0) or np.any(Ux_aniso_mid != 0)
    trig_plot_y = np.any(Uy_iso_mid != 0) or np.any(Uy_aniso_mid != 0)
    trig_plot_z = np.any(Uz_iso_mid != 0) or np.any(Uz_aniso_mid != 0)

    # Create R and Theta arrays for plotting
    R_mid, Theta_mid = R[:, :, z_mid], theta[:, :, z_mid]
    if mod_theta:
        Theta_mid_ani=np.unwrap(Theta_mid)


        
    # Create a figure with dynamic subplots based on the flags
    num_rows = sum([trig_plot_x, trig_plot_y, trig_plot_z])
    fig, axs = plt.subplots(num_rows, 1, figsize=(10, 4 * num_rows))

    # Ensure axs is always iterable
    if num_rows == 1:
        axs = [axs]

    row_idx = 0

    # Plot Ux comparison if trig_plot_x is True
    if trig_plot_x:
        ax = axs[row_idx]
        for r_val in r_values:
            mask = np.isclose(R_mid, r_val, atol=1)
            Ux_iso_masked = Ux_iso_mid[mask].flatten()
            Ux_aniso_masked = Ux_aniso_mid[mask].flatten()
            theta_masked = Theta_mid[mask].flatten()
            if mod_theta:
                Theta_masked_ani = Theta_mid_ani[mask].flatten()
            else:
                Theta_masked_ani = Theta_mid[mask].flatten()
            diff_min_theta=theta_masked.min()-Theta_masked_ani.min()
            print(f"shift theta: {rint(diff_min_theta*180/pi)} ")

            sorted_indices = np.argsort(theta_masked)
            sorted_indices_ani = np.argsort(Theta_masked_ani)



            sorted_theta_masked     =  theta_masked[sorted_indices]
            sorted_theta_ani_masked =  Theta_masked_ani[sorted_indices_ani]
            sorted_Ux_iso_masked    =  Ux_iso_masked  [sorted_indices]
            sorted_Ux_aniso_masked  =  Ux_aniso_masked[sorted_indices_ani]

            if remove_jump:
                jump_upos=np.where(np.diff(sorted_Ux_aniso_masked)>0.2)[0][0]
                modified_U=array(sorted_Ux_aniso_masked)
                modified_U[jump_upos+1:]=modified_U[jump_upos+1:]-(modified_U[jump_upos+1]-modified_U[jump_upos])
                sorted_Ux_aniso_masked=modified_U
            diff_min_U=sorted_Ux_iso_masked.min()-sorted_Ux_aniso_masked.min()
            sorted_Ux_aniso_masked+=diff_min_U
            
            ax.plot(sorted_theta_masked    , sorted_Ux_iso_masked  , marker1, label=f'Isotropic r={r_val}'  )
            ax.plot(sorted_theta_ani_masked, sorted_Ux_aniso_masked, marker2, label=f'Anisotropic r={r_val}')

        ax.set_title(r'$U_x$ Comparison')
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel(r'$U_x$')
        ax.legend()
        row_idx += 1

    # Plot Uy comparison if trig_plot_y is True
    if trig_plot_y:
        ax = axs[row_idx]
        for r_val in r_values:
            mask = np.isclose(R_mid, r_val, atol=1)
            theta_masked = Theta_mid[mask].flatten()
            Uy_iso_masked = Uy_iso_mid[mask].flatten()
            Uy_aniso_masked = Uy_aniso_mid[mask].flatten()

            sorted_indices = np.argsort(theta_masked)
            ax.plot(theta_masked[sorted_indices], Uy_iso_masked  [sorted_indices], marker1, label=f'Isotropic r={r_val}'  )
            ax.plot(theta_masked[sorted_indices], Uy_aniso_masked[sorted_indices], marker2, label=f'Anisotropic r={r_val}')

        ax.set_title(r'$U_y$ Comparison')
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel(r'$U_y$')
        ax.legend()
        row_idx += 1

    # Plot Uz comparison if trig_plot_z is True
    if trig_plot_z:
        ax = axs[row_idx]
        for r_val in r_values:
            mask = np.isclose(R_mid, r_val, atol=1)
            theta_masked = Theta_mid[mask].flatten()
            Uz_iso_masked = Uz_iso_mid[mask].flatten()
            Uz_aniso_masked = Uz_aniso_mid[mask].flatten()

            sorted_indices = np.argsort(theta_masked)
            ax.plot(theta_masked[sorted_indices], Uz_iso_masked  [sorted_indices], marker1, label=f'Isotropic r={r_val}'  )
            ax.plot(theta_masked[sorted_indices], Uz_aniso_masked[sorted_indices], marker2, label=f'Anisotropic r={r_val}')

        ax.set_title(r'$U_z$ Comparison')
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel(r'$U_z$')
        ax.legend()
        row_idx += 1

    # Set the main title for the figure
    plt.suptitle(suptitle, fontsize=16)

    # Tight layout to prevent overlap
    plt.tight_layout()

    # Save the plot if a file path is provided
    if save_plot:
        plt.savefig(save_plot, dpi=150)

    # Show the plot
    plt.show()
#------------------------------------------------------------------------------------------------------------
def compare_metals(Ux_dict, Uy_dict, Uz_dict, theta, r, r_values, dislocation_type, save_plot=None,mod_theta=False,metals_elastic_constants=None):
    """
    Compare the displacement fields Ux, Uy, and Uz for multiple metals at specific r-values.

    Parameters:
        Ux_dict (dict): Dictionary of Ux displacement fields for different metals.
        Uy_dict (dict): Dictionary of Uy displacement fields for different metals.
        Uz_dict (dict): Dictionary of Uz displacement fields for different metals.
        theta (ndarray): Angular array (theta).
        r (ndarray): Radial distance array.
        r_values (list): List of radial distances for which to plot U vs theta.
        dislocation_type (str): Type of dislocation ('Edge' or 'Screw').
        save_plot (str, optional): File path to save the plot. If None, the plot is not saved.
    """

    # Get the list of metals from the keys of the dictionaries
    list_metals = Ux_dict.keys()

    # Extract the middle z-slice index from the first entry in the dictionaries
    first_key = next(iter(Ux_dict))
    z_mid = Ux_dict[first_key].shape[2] // 2

    # Extract the middle z-slices for the first metal
    Ux_mid = Ux_dict[first_key][:, :, z_mid]
    Uy_mid = Uy_dict[first_key][:, :, z_mid]
    Uz_mid = Uz_dict[first_key][:, :, z_mid]

    # Define flags based on the first metal's displacement fields
    trig_plot_x = np.any(Ux_mid != 0)
    trig_plot_y = np.any(Uy_mid != 0)
    trig_plot_z = np.any(Uz_mid != 0)

    # Determine the number of rows based on which fields contain data
    num_rows = sum([trig_plot_x, trig_plot_y, trig_plot_z])

    # Create a figure with dynamic subplots
    fig, axs = plt.subplots(num_rows, 1, figsize=(10, 4 * num_rows))

    # Ensure axs is always iterable
    if num_rows == 1:
        axs = [axs]

    R_mid, Theta_mid = r[:, :, z_mid], theta[:, :, z_mid]
    if mod_theta:
        Theta_mid=np.unwrap(Theta_mid+pi/2)
    row_idx = 0

    # Plot Ux if trig_plot_x is True
    if trig_plot_x:
        ax = axs[row_idx]
        for i_metal in list_metals:
            Ux = Ux_dict[i_metal][:, :, z_mid]
            if metals_elastic_constants is not None:
                anis=metals_elastic_constants[i_metal]['anisotropy']
            for r_val in r_values:
                mask = np.isclose(R_mid, r_val, atol=1)
                sorted_indices = np.argsort(Theta_mid[mask].flatten())
                Ux_sorted   =Ux       [mask].flatten()[sorted_indices]
                theta_sorted=Theta_mid[mask].flatten()[sorted_indices]
                diff_U___=np.diff(Ux_sorted)
                if np.any(diff_U___>0.2):
                    jump_upos=np.where(np.diff(Ux_sorted)>0.2)[0][0]
                    modified_U=array(Ux_sorted)
                    modified_U[jump_upos+1:]=modified_U[jump_upos+1:]-(modified_U[jump_upos+1]-modified_U[jump_upos])
                    Ux_sorted=modified_U
                ax.plot(theta_sorted, Ux_sorted,
                        label=f'{i_metal} ({anis})')

        ax.set_title(rf'$U_x$ for {dislocation_type} Dislocation')
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel(r'$U_x$')
        ax.legend()
        row_idx += 1

    # Plot Uy if trig_plot_y is True
    if trig_plot_y:
        ax = axs[row_idx]
        for i_metal in list_metals:
            Uy = Uy_dict[i_metal][:, :, z_mid]

            if metals_elastic_constants is not None:
                anis=metals_elastic_constants[i_metal]['anisotropy']
            for r_val in r_values:
                mask = np.isclose(R_mid, r_val, atol=1)
                sorted_indices = np.argsort(Theta_mid[mask].flatten())
                ax.plot(Theta_mid[mask].flatten()[sorted_indices], Uy[mask].flatten()[sorted_indices],
                        label=f'{i_metal} ({anis})')

        ax.set_title(rf'$U_y$ for {dislocation_type} Dislocation')
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel(r'$U_y$')
        ax.legend()
        row_idx += 1

    # Plot Uz if trig_plot_z is True
    if trig_plot_z:
        ax = axs[row_idx]
        for i_metal in list_metals:
            Uz = Uz_dict[i_metal][:, :, z_mid]

            if metals_elastic_constants is not None:
                anis=metals_elastic_constants[i_metal]['anisotropy']
            for r_val in r_values:
                mask = np.isclose(R_mid, r_val, atol=1)
                sorted_indices = np.argsort(Theta_mid[mask].flatten())
                ax.plot(Theta_mid[mask].flatten()[sorted_indices], Uz[mask].flatten()[sorted_indices],
                        label=f'{i_metal} ({anis})')
        ax.set_title(rf'$U_z$ for {dislocation_type} Dislocation')
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel(r'$U_z$')
        ax.legend()
        row_idx += 1

    # Set the main title for the figure
    plt.suptitle(f'Comparison of {dislocation_type} Dislocation for Multiple Metals', fontsize=16)

    # Tight layout to prevent overlap
    plt.tight_layout()

    # Save the plot if a file path is provided
    if save_plot:
        plt.savefig(save_plot, dpi=150)

    # Show the plot
    plt.show()
#------------------------------------------------------------------------------------------------------------
def plot_displacement_fields(Ux_cyl, Uy_cyl, Uz_cyl, r, theta, z_mid, r_values,mod_theta=False, save_plot=None, marker=">", suptitle='Displacement Fields for Edge Dislocation in Isotropic Media'):
    """
    Plot the displacement fields Ux, Uy, and Uz for a fixed z-slice.

    Parameters:
        Ux_cyl (ndarray): Displacement field in the x-direction (cylindrical coordinates).
        Uy_cyl (ndarray): Displacement field in the y-direction (cylindrical coordinates).
        Uz_cyl (ndarray): Displacement field in the z-direction (cylindrical coordinates).
        r (ndarray): Radial distance array.
        theta (ndarray): Angular array (theta).
        z_mid (int): Index of the middle z-slice.
        r_values (list): List of radial distances for which to plot U vs theta.
        save_plot (str, optional): File path to save the plot. If None, the plot is not saved.
        suptitle (str): Title for the entire figure.
    """

    # Scale the displacements for visualization
    scale_factor = 1
    Ux_mid = Ux_cyl[:, :, z_mid] * scale_factor
    Uy_mid = Uy_cyl[:, :, z_mid] * scale_factor
    Uz_mid = Uz_cyl[:, :, z_mid] * scale_factor

    # Flags to check if each displacement field has non-zero values
    trig_plot_x = np.any(Ux_mid != 0)
    trig_plot_y = np.any(Uy_mid != 0)
    trig_plot_z = np.any(Uz_mid != 0)

    # Create R and Theta arrays for plotting
    R, Theta = r[:, :, z_mid], theta[:, :, z_mid]
    if mod_theta:
        Theta=np.unwrap(Theta)
    X,Y,Z=cylindrical_to_cartesian(R, Theta, z_mid)
    
    # Determine which plots are to be made
    plot_flags = [trig_plot_x, trig_plot_y, trig_plot_z]
    displacement_fields = [Ux_mid, Uy_mid, Uz_mid]
    titles = [r'$U_x$', r'$U_y$', r'$U_z$']
    ylabels = [r'$U_x$', r'$U_y$', r'$U_z$']

    # Number of rows to plot
    num_rows = sum(plot_flags)

    # Create a figure with dynamic subplots
    fig, axs = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))
    
    # Ensure axs is always a 2D list for consistency
    if num_rows == 1:
        axs = [axs]
    
    # Plot displacement fields
    row_idx = 0
    for i, (flag, U_mid, title, ylabel) in enumerate(zip(plot_flags, displacement_fields, titles, ylabels)):
        if not flag:
            continue

        # Contour plot for U as a function of r and theta
        ax = axs[row_idx][0]
        c = ax.contourf(X, Y, U_mid, levels=100, cmap='jet')
        ax.set_title(f'{title} as a function of $r$, $\\theta$ at fixed $z$')
        fig.colorbar(c, ax=ax)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # Line plot for U as a function of theta for different r values
        ax = axs[row_idx][1]
        for r_val in r_values:
            mask = np.isclose(R, r_val, atol=1)
            theta_masked = Theta[mask].flatten()
            U_masked = U_mid[mask].flatten()

            # Sort the theta and U values
            sorted_indices = np.argsort(theta_masked)
            theta_sorted = theta_masked[sorted_indices]
            U_sorted = U_masked[sorted_indices]
            if i==0:
                diff_U___=np.diff(U_sorted)
                if np.any(diff_U___>0.2):
                    jump_upos=np.where(np.diff(U_sorted)>0.2)[0][0]
                    modified_U=array(U_sorted)
                    modified_U[jump_upos+1:]=modified_U[jump_upos+1:]-(modified_U[jump_upos+1]-modified_U[jump_upos])
                    U_sorted=modified_U
            ax.plot(theta_sorted, U_sorted, marker, label=f'r={r_val}')

        ax.set_title(f'{title} as a function of $\\theta$')
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel(ylabel)
        ax.legend()

        row_idx += 1

    # Set the main title for the figure
    plt.suptitle(suptitle, fontsize=16)

    # Tight layout to prevent overlap
    fig.tight_layout()

    # Save the plot if a file path is provided
    if save_plot:
        plt.savefig(save_plot, dpi=150)

    # Show the plot
    plt.show()

######################################### Not used ##################################################################
# ------------------ Edge Dislocation in Isotropic Media ------------------
def edge_dislocation_isotropic(b, nu, r, theta):
    """
    Compute the displacement field for an edge dislocation in isotropic media.

    Parameters:
        b (float): Burgers vector magnitude.
        nu (float): Poisson's ratio.
        r (float): Radial distance.
        theta (float): Angle in radians.

    Returns:
        (U_r, U_theta, U_z): Tuple of radial, tangential, and axial displacements.
    """
    U_x = -b / (2 * np.pi) * (np.sin(theta) / (2 * (1 - nu)) + np.sin(2 * theta) / (4 * (1 - nu)))
    U_y = b / (2 * np.pi) * (np.cos(theta) / (2 * (1 - nu)) - (1 - 2 * nu) / (2 * (1 - nu)) * np.log(r) + np.cos(2 * theta) / (4 * (1 - nu)))
    U_z = np.zeros_like(r)
    return U_x, U_y, U_z
# ------------------ Screw Dislocation in Isotropic Media ------------------
def screw_dislocation_isotropic(b, theta):
    """
    Compute the displacement field for a screw dislocation in isotropic media.

    Parameters:
        b (float): Burgers vector magnitude.
        theta (float): Angle in radians.

    Returns:
        (U_r, U_theta, U_z): Tuple of radial, tangential, and axial displacements.
    """
    U_x = np.zeros_like(r)
    U_y = np.zeros_like(r)
    U_z = b * (theta+pi) / (2 * np.pi)
    return U_x, U_y, U_z
# ------------------ Edge Dislocation in Anisotropic Media ------------------
def edge_dislocation_anisotropic(bx, by, c11, c12, c44, r, theta):
    
    # Calculate c_0
    c0 = c11 - c12 - 2 * c44
    
    # Calculate h
    h = -c0
    
    # Calculate Anisotropy
    anisotropy = (2 * c44) / (c11 - c12)
    
    # Calculate c' values
    c11_prime = c11
    c12_prime = c12
    c55_prime = c44
    c66_prime = c44
    c22_prime = c11 + h / 2
    c23_prime = c12 - h / 2
    c44_prime = c44 - h / 2
    
    # Calculate \overline{c_{11}}'
    c11_bar_prime = np.sqrt(c11_prime * c22_prime)

    # Calculate lambda
    lambda_ = (c11_bar_prime / c22_prime) ** 0.25
    
    # Calculate phi
    phi = 0.5 * np.arccos((c12_prime + 2 * c12_prime * c66_prime - c11_bar_prime) / (2 * c11_bar_prime * c66_prime))
    
    # Calculate q^2 and t^2
    q_squared = (r**2) * (lambda_**2 + (1 - lambda_**2) * np.cos(theta)**2+lambda_*np.cos( phi)*np.sin(2* phi) )
    t_squared = (r**2) * (lambda_**2 + (1 - lambda_**2) * np.cos(theta)**2-lambda_*np.cos( phi)*np.sin(2* phi) )
    q = np.sqrt(q_squared)
    t = np.sqrt(t_squared)
    
    # Calculate theta_anis1
    theta_anis1 = np.arctan2(lambda_ * np.sin(2 * theta) * np.sin(phi), -lambda_ + np.cos(theta)**2 - (1 - lambda_))
    
    # Calculate theta_anis2
    theta_anis2 = np.arctan2(np.sin(2 * phi), lambda_**2 * np.tan(theta)**2 - np.cos(2 * phi))
    
    # Calculate theta_anis3
    theta_anis3 = np.arctan2(lambda_**2 * np.sin(2 * phi), 1 / np.tan(theta)**2 - lambda_**2 * np.cos(2 * phi))
    
    # Calculate A1
    A1 = (c11_bar_prime**2 - c12_prime**2) / (2 * c11_bar_prime * c66_prime * np.sin(2 * phi))
    
    # Calculate A2
    A2 = (c11_bar_prime - c12_prime) / (2 * c11_bar_prime * lambda_ * np.sin(phi))
    
    # Calculate A3
    A3 = (c11_bar_prime + c12_prime) / (2 * c11_bar_prime * lambda_ * np.cos(phi))
    
    # Calculate displacements u_x, u_y, u_z
    U_x = (-1 / (4 * np.pi)) * (bx * theta_anis1 -              by * A3 * theta_anis2) - (1 / (4 * np.pi)) * (             by * A2 * np.log(q * t) + bx * A1 * np.log(q / t))
    
    U_y = (-1 / (4 * np.pi)) * (by * theta_anis1 - lambda_**2 * A3 * bx * theta_anis3) - (1 / (4 * np.pi)) * (lambda_**2 * bx * A2 * np.log(q * t) - by * A1 * np.log(q / t))

    U_z = np.zeros_like(theta)

    return U_x, U_y, U_z
# ------------------ Screw Dislocation in Anisotropic Media ------------------
def screw_dislocation_anisotropic(b_z, c11, c12, c44, theta):
    """
    Compute the displacement field for a screw dislocation in anisotropic media.

    Parameters:
        b_z (float): Burgers vector component along z.
        c11 (float): Elastic constant C11.
        c12 (float): Elastic constant C12.
        c44 (float): Elastic constant C44.
        theta (float): Angle in radians.

    Returns:
        (U_x, U_y, U_z): Tuple of displacements in x, y, and z directions.
    """
    # Calculate c0 and the modified elastic constants (c' values)
    c0   = c11 - c12 - 2 * c44
    cp11 = c11 - c0 / 2
    cp12 = c12 + c0 / 3
    cp13 = c12 + c0 / 6
    cp44 = c44 + c0 / 3
    cp55 = c44 + c0 / 6
    cp16 = -c0 * np.sqrt(2) / 6
    cp22 = c11 - 2 * c0 / 3
    cp45 = -cp16

    # Displacement components
    U_x = np.zeros_like(theta)
    U_y = np.zeros_like(theta)
    numerator = np.sqrt(cp44 * cp55 - cp45**2) * np.tan(theta)
    denominator = cp44 - cp45 * np.tan(theta)
    
    U_z = -b_z / (2 * np.pi) * np.arctan2(numerator, denominator)

    return U_x, U_y, U_z
#------------------------------------------------------------------------------------------------------------
def calculate_mixed_displacement(R, Theta, Z, b, nu, alpha):
    """
    Calculate the displacement field for a mixed dislocation (edge + screw) in cylindrical coordinates.
    
    Parameters:
    r      : Radial distance array.
    theta  : Azimuthal angle array.
    z      : Vertical position array.
    b      : Burgers vector (array).
    nu     : Poisson's ratio.
    alpha  : Factor for mixed dislocation (0 <= alpha <= 1).
    
    Returns:
    Ux_cyl, Uy_cyl, Uz_cyl : Displacement components in cylindrical coordinates.
    """
    
    # Convert Burgers vector into cylindrical components
    b_r = np.linalg.norm(b[:2])  # Radial component
    b_z = b[2]  # Vertical component
        
    # Calculate displacement components for edge dislocation
    Ux_edge = (b_r / (2 * np.pi)) * (np.arctan(np.sin(Theta) / R) + (R * np.sin(Theta)) / (2 * (1 - nu) * (R**2 + np.sin(Theta)**2)))  # Ux for edge
    Uy_edge = -(b_r / (2 * np.pi)) * ((1 - 2 * nu) / (4 * (1 - nu))) * np.log(R**2 + np.sin(Theta)**2) + ((R**2 - np.sin(Theta)**2) / (4 * (1 - nu) * (R**2 + np.sin(Theta)**2)))  # Uy for edge
    Uz_edge = np.zeros_like(R)  # Uz for edge dislocation (zero)

    # Calculate displacement components for screw dislocation
    Ux_screw = np.zeros_like(R)  # Ux for screw dislocation (zero)
    Uy_screw = np.zeros_like(R)  # Uy for screw dislocation (zero)
    Uz_screw = (b_z / (2 * np.pi)) * np.arctan2(np.sin(Theta), R)  # Uz for screw dislocation

    # Combine the edge and screw dislocation components with the factor alpha
    Ux_cyl = alpha * Ux_edge + (1 - alpha) * Ux_screw
    Uy_cyl = alpha * Uy_edge + (1 - alpha) * Uy_screw
    Uz_cyl = alpha * Uz_edge + (1 - alpha) * Uz_screw
    
    return Ux_cyl, Uy_cyl, Uz_cyl

#####################################################################################################################
################################# Used    ###########################################################################
#####################################################################################################################

#------------------------------------------------------------------------------------------------------------
def edge_dislocation_isotropic_cart(b, nu, x, y):
    """
    Compute the displacement field for an edge dislocation in isotropic media.

    Parameters:
        b (float): Burgers vector magnitude.
        nu (float): Poisson's ratio.
        x, y, z (float or array): Cartesian coordinates.

    Returns:
        (U_x, U_y, U_z): Tuple of displacements in x, y, and z directions.
    """
    U_y = -b / (2 * np.pi*  (4 * (1 - nu))) * (
        (1 - 2*nu) * np.log(x**2 + y**2) +  
        (x**2 - y**2) / (4 * (1 - nu) * (x**2 + y**2))
    )
    U_x = b / (2 * np.pi) * (
        np.arctan2(y, x)+
        (x * y) / (2 * (1 - nu) * (x**2 + y**2))
    )
          
    U_z = np.zeros_like(x)
    
    return U_x, U_y, U_z
#------------------------------------------------------------------------------------------------------------
def screw_dislocation_isotropic_cart(b, x, y):
    """
    Compute the displacement field for a screw dislocation in isotropic media.

    Parameters:
        b (float): Burgers vector magnitude.
        x, y, z (float or array): Cartesian coordinates.

    Returns:
        (U_x, U_y, U_z): Tuple of displacements in x, y, and z directions.
    """
    theta = np.arctan2(y, x)
    
    U_x = np.zeros_like(x)
    U_y = np.zeros_like(y)
    U_z = b * theta / (2 * np.pi)
    
    return U_x, U_y, U_z

    
def mixed_dislocation_isotropic_cart(b_x, b_y, b_z, x, y, nu=0.3):
    """
    Compute the displacement field components for a mixed dislocation in isotropic material.

    Parameters:
        b_x (float): Burgers vector component in the x-direction.
        b_y (float): Burgers vector component in the y-direction.
        b_z (float): Burgers vector component in the z-direction.
        x (float): x-coordinate.
        y (float): y-coordinate.
        nu (float): Poisson's ratio (default: 0.3).

    Returns:
        tuple: Displacement components (Ux_cyl, Uy_cyl, Uz_cyl).
    """
    # Compute alpha
    alpha_angle = np.arctan2(b_z, (b_x**2 + b_y**2)**0.5)  # Mixing angle (in radians)
    alpha_factor = np.cos(alpha_angle)

    # Convert x, y to polar coordinates
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # Avoid division by zero for r
    r = r + 1e-10
    r_squared = r**2

    # Burgers vector magnitude
    b = (b_x**2 + b_y**2 + b_z**2)**0.5

    # Displacement components
    U_x = (-b_x / (2 * np.pi) * (np.arctan2(y, x) + (x * y) / (2 * (1 - nu) * r_squared)))
    U_y = -b_y / (2 * np.pi) * ((1 - 2 * nu) / (4 * (1 - nu)) * np.log(r_squared) + (x**2 - y**2) / (4 * (1 - nu) * r_squared))
    U_z = b * theta / (2 * np.pi)

    # Combine the edge and screw dislocation components with the factor alpha
    Ux_cyl = alpha_factor * U_x
    Uy_cyl = alpha_factor * U_y
    Uz_cyl = (1 - alpha_factor) * U_z

    return Ux_cyl, Uy_cyl, Uz_cyl



def edge_dislocation_anisotropic_carti_v0(bx, by, c11, c12, c44, x, y):
    # Calculate c_0
    c0 = c11 - c12 - 2 * c44
    # Calculate h
    h = -c0
    
    # Calculate Anisotropy
    anisotropy = (2 * c44) / (c11 - c12)
    
    # Calculate c' values
    c11_prime = c11
    c12_prime = c12
    c55_prime = c44
    c66_prime = c44
    c22_prime = c11 + h / 2
    c23_prime = c12 - h / 2
    c44_prime = c44 - h / 2
    
    # Calculate \overline{c_{11}}'
    c11_bar_prime = np.sqrt(c11_prime * c22_prime)

    
    # Calculate lambda
    lambda_ = (c11_bar_prime / c22_prime) ** 0.25
    
    # Calculate phi
    phi = 0.5 * np.arccos((c12_prime**2 + 2 * c12_prime * c66_prime - c11_bar_prime**2) / (2 * c11_bar_prime * c66_prime))
    
    # Calculate q^2 and t^2
    q_squared =  ( x**2 + ((lambda_**2) * (y**2))+ 2*x*y*lambda_*cos(phi) )
    t_squared =  ( x**2 + ((lambda_**2) * (y**2))- 2*x*y*lambda_*cos(phi) )
    q = np.sqrt(q_squared)
    t = np.sqrt(t_squared)
    
    # Calculate theta_anis1
    theta_anis1 = np.arctan2(2*(x)*(y)*lambda_*sin(phi),(x)**2-(y)**2*lambda_**2)+np.sign(x)*pi+2*pi
    # Calculate theta_anis2
    theta_anis2 = np.arctan2(np.sin(2 * phi)*x**2, lambda_**2 *y**2 - x**2 * np.cos(2 * phi))
    # Calculate theta_anis3
    theta_anis3 = np.arctan2(lambda_**2 * np.sin(2 * phi)*y**2, x**2- y**2* lambda_**2 * np.cos(2 * phi))
    # Calculate A1
    A1 = (c11_bar_prime**2 - c12_prime**2) / (2 * c11_bar_prime * c66_prime * np.sin(2 * phi))
    
    # Calculate A2
    A2 = (c11_bar_prime - c12_prime) / (2 * c11_bar_prime * lambda_ * np.sin(phi))
    
    # Calculate A3
    A3 = (c11_bar_prime + c12_prime) / (2 * c11_bar_prime * lambda_ * np.cos(phi))
    
    # Calculate displacements u_x, u_y, u_z
    U_x = (-1 / (4 * np.pi)) * (bx * theta_anis1 -              by * A3 * theta_anis2) - (1 / (4 * np.pi)) * (             by * A2 * np.log(q * t) + bx * A1 * np.log(q / t))
    
    U_y = (-1 / (4 * np.pi)) * (by * theta_anis1 - lambda_**2 * A3 * bx * theta_anis3) - (1 / (4 * np.pi)) * (lambda_**2 * bx * A2 * np.log(q * t) - by * A1 * np.log(q / t))

    U_z = np.zeros_like(np.arctan2(y, x))

    return U_x, U_y, U_z
#------------------------------------------------------------------------------------------------------------
def edge_dislocation_anisotropic_carti(bx, by, c11, c12, c44, x, y):
    # Calculate c_0
    c0 = c11 - c12 - 2 * c44
    # Calculate h
    h = -c0
    
    # Calculate Anisotropy
    anisotropy = (2 * c44) / (c11 - c12)
    
    # Calculate c' values
    c11_prime = c11
    c12_prime = c12
    c55_prime = c44
    c66_prime = c44
    c22_prime = c11 + h / 2
    c23_prime = c12 - h / 2
    c44_prime = c44 - h / 2
    
    # Calculate \overline{c_{11}}'
    c11_bar_prime = np.sqrt(c11_prime * c22_prime)

    
    # Calculate lambda
    lambda_ = (c11_bar_prime / c22_prime) ** 0.25
    
    # Calculate phi
    phi = 0.5 * np.arccos((c12_prime**2 + 2 * c12_prime * c66_prime - c11_bar_prime**2) / (2 * c11_bar_prime * c66_prime))
    
    # Calculate q^2 and t^2
    q_squared =  ( x**2 + ((lambda_**2) * (y**2))+ 2*x*y*lambda_*cos(phi) )
    t_squared =  ( x**2 + ((lambda_**2) * (y**2))- 2*x*y*lambda_*cos(phi) )
    q = np.sqrt(q_squared)
    t = np.sqrt(t_squared)
    
    # Calculate theta_anis1
    theta_anis1 = np.arctan2(2*(x)*(y)*lambda_*sin(phi),(x)**2-(y)**2*lambda_**2)+np.sign(x)*pi
    # Calculate theta_anis2
    theta_anis2 = np.arctan2(np.sin(2 * phi)*x**2, lambda_**2 *y**2 - x**2 * np.cos(2 * phi))
    # Calculate theta_anis3
    theta_anis3 = np.arctan2(lambda_**2 * np.sin(2 * phi)*y**2, x**2- y**2* lambda_**2 * np.cos(2 * phi))
    # Calculate A1
    A1 = (c11_bar_prime**2 - c12_prime**2) / (2 * c11_bar_prime * c66_prime * np.sin(2 * phi))
    
    # Calculate A2
    A2 = (c11_bar_prime - c12_prime) / (2 * c11_bar_prime * lambda_ * np.sin(phi))
    
    # Calculate A3
    A3 = (c11_bar_prime + c12_prime) / (2 * c11_bar_prime * lambda_ * np.cos(phi))
    
    # Calculate displacements u_x, u_y, u_z
    U_x = (-1 / (4 * np.pi)) * (bx * theta_anis1 -              by * A3 * theta_anis2) - (1 / (4 * np.pi)) * (             by * A2 * np.log(q * t) + bx * A1 * np.log(q / t))
    
    U_y = (-1 / (4 * np.pi)) * (by * theta_anis1 - lambda_**2 * A3 * bx * theta_anis3) - (1 / (4 * np.pi)) * (lambda_**2 * bx * A2 * np.log(q * t) - by * A1 * np.log(q / t))

    U_z = np.zeros_like(np.arctan2(y, x))

    return U_x, U_y, U_z
# ------------------ Screw Dislocation in Anisotropic Media ------------------
def screw_dislocation_anisotropic_carti(b_z, c11, c12, c44, x,y):
    """
    Compute the displacement field for a screw dislocation in anisotropic media.

    Parameters:
        b_z (float): Burgers vector component along z.
        c11 (float): Elastic constant C11.
        c12 (float): Elastic constant C12.
        c44 (float): Elastic constant C44.
        theta (float): Angle in radians.

    Returns:
        (U_x, U_y, U_z): Tuple of displacements in x, y, and z directions.
    """
    # Calculate c0 and the modified elastic constants (c' values)
    c0   = c11 - c12 - 2 * c44
    cp11 = c11 - c0 / 2
    cp12 = c12 + c0 / 3
    cp13 = c12 + c0 / 6
    cp44 = c44 + c0 / 3
    cp55 = c44 + c0 / 6
    cp16 = -c0 * np.sqrt(2) / 6
    cp22 = c11 - 2 * c0 / 3
    cp45 = -cp16
    coef= sqrt(cp44*cp55-cp45**2)
    # Displacement components
    U_x = np.zeros_like(np.arctan2(y, x))
    U_y = np.zeros_like(np.arctan2(y, x))
    numerator = coef*y
    denominator = cp44*x - cp45 *y
    
    U_z = b_z / (2 * np.pi) *( np.arctan2(numerator, denominator)+pi)

    return U_x, U_y, U_z

#####################################################################################################################
#------------------------------------------------------------------------------------------------------------
def calculate_alpha_factor_dislo(b_x, b_y, b_z):
    alpha_angle = np.arctan2(b_z, np.sqrt(b_x**2 + b_y**2))
    return np.cos(alpha_angle)
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
def center_object_list(obj_list):
    obj_list_centered = np.zeros(obj_list.shape, dtype='complex128')
    for n in range(len(obj_list)):
#         print(len(obj_list)-n,end=' ')
        obj_list_centered[n] += center_object(obj_list[n])
    return obj_list_centered
#------------------------------------------------------------------------------------------------------------
def center_of_mass_calculation_two_steps(data, crop = 50, plot=False):
    
    center = np.unravel_index(np.nanargmax(data), data.shape)

    cropping_dim = []
    for n in range(data.ndim):
        cropping_dim.append([max([0, int(center[n]-crop/2)]),  min(int(center[n]+crop//2), data.shape[n]-1)])


    s = [slice( cropping_dim[n][0],  cropping_dim[n][1] ) for n in range(data.ndim)]
    center2 = center_of_mass(data[tuple(s)])

    center = [int(round(cropping_dim[n][0]+center2[n])) for n in range(data.ndim)]
    
    if plot:
        if data.ndim==3:
            fig, ax = plt.subplots(1,3, figsize=(12,4))
            plot_3D_projections(data, fig=fig, ax=ax)
            ax[0].scatter(center[2], center[2], color='w')
            ax[1].scatter(center[2], center[0], color='w')
            ax[2].scatter(center[1], center[0], color='w')
        if data.ndim==2:
            fig = plt.figure(figsize=(10,10))
            plt.imshow(np.log(data), cmap='plasma', vmin=1)
            plt.colorbar()
            plt.scatter(center[1], center[0], color='w')
    return center
#------------------------------------------------------------------------------------------------------------
def automatic_object_roi(obj,threshold = .1, factor = .4,plot=False):
    module = np.abs(obj)
    
    if plot:
        fig,ax = plt.subplots(1,module.ndim, figsize=(5*module.ndim, 3))
    
    roi = np.zeros(2*module.ndim, dtype='int')
    for n in range(module.ndim):
        sum_axis = np.arange(module.ndim)
        sum_axis = np.delete(sum_axis, n)
        projection = np.nanmean(module,axis=tuple(sum_axis))
        projection -= np.nanmin(projection)
        projection = projection/np.nanmax(projection)

        start = np.nanmin(np.where(projection>threshold))
        end = np.nanmax(np.where(projection>threshold))

        size = end-start
        start = max(int(start-size*factor),0)
        end = int(end+size*factor)

        roi[2*n] += start
        roi[2*n+1] += end

        if plot:
            ax[n].plot(projection, '.-')
            ax[n].axvline(x=start, color='r')
            ax[n].axvline(x=end, color='r')
            
    return roi
#------------------------------------------------------------------------------------------------------------
def apply_roi(array, roi):
    s = [slice(roi[2*n], roi[2*n+1]) for n in range(array.ndim)]
    return array[tuple(s)]
#------------------------------------------------------------------------------------------------------------
def get_cropped_module_phase(obj,threshold_module = None, support = None,crop=False, apply_fftshift=False, unwrap=True):
    
    if apply_fftshift:
        obj = fftshift(obj)
        
    shape = obj.shape
    if crop:
        obj = crop_array_half_size(obj)
        if support is not None:
            support = crop_array_half_size(support)
        
    module = np.abs(obj)
    
    if support is None:
        if threshold_module is None:
            if obj.ndim ==3:
                threshold_module = .3 # Seems that 3D data need a smaller threshold. Nope in the end, .3 is fine
            if obj.ndim == 2:
                threshold_module = .3
        support = (module >= np.nanmax(module)*threshold_module)        
    
    phase = np.angle(obj)
    
    if unwrap:
        from skimage.restoration import unwrap_phase as unwrap_phase_skimage
        mask_unwrap = (1-support)
        
        if np.any(np.isnan(obj)): # Need to take nan's into account
            print('nan\'s are stil la problem, this freezes the unwrapping')
#             mask_nan = np.isnan(obj)
#             mask_unwrap = mask_unwrap + mask_nan
#             mask_unwrap[mask_unwrap != 0] = 1
#  Fail to take nan's into account...
            
        phase = np.ma.masked_array(phase, mask=mask_unwrap)
        phase = unwrap_phase_skimage(
                phase,
                wrap_around=False,
#                 seed=1
            ).data

#     if unwrap:
#         phase = unwrap_phase(obj)
#     else:
#         phase = np.angle(obj)
 
    phase[support==0] = np.nan 

    
#     # Badly written Not a good thing actually is the center is a nan
#     if phase.ndim==2:
#         phase -= phase[phase.shape[0]//2,phase.shape[1]//2]
#     if phase.ndim==3:
#         phase -= phase[phase.shape[0]//2,phase.shape[1]//2,phase.shape[2]//2]
        
    return module, phase
#####################################################################################################################
#####################################################################################################################
#------------------------------------------------------------------------------------------------------------
def process_and_merge_clusters_dislo_strain_map(data,amp,phase,save_path,voxel_sizes,threshold=0.35,min_cluster_size=10,distance_threshold=10.0,cylinder_radius=3.0,smoothing_param=2,num_spline_points=1000,
                                                apply_dilation=True,dilation_size=1,apply_smoothing=True,add_random_noise=False,noise_stddev=2.0,apply_polynomial_fit=False,polynomial_degree=3,save_output=True,debug_plot=True,):
    """
    Process the input 3D data to identify clusters, optionally add random noise,
    fit a spline and optionally a polynomial fit for each cluster, and replace with
    a single curved cylinder per cluster. Optionally save output and create a 3D debug plot.

    Parameters:
    - (same as before)
    - save_output: bool
        Whether to save the output data as a VTI file.
    - debug_plot: bool
        Whether to create a 3D GIF showing the final clustering result.

    Returns:
    - final_labeled_clusters: np.ndarray
        3D mask with consecutively relabeled clusters.
    - num_final_clusters: int
        The number of final clusters after relabeling.
    """
    # Step 1: Threshold the data to identify potential clusters
    binary_data = (data > threshold).astype(np.uint8)

    # Step 2: Label the clusters
    labeled_data, num_clusters = label(binary_data)
    print(f"Number of clusters identified: {num_clusters}")

    # Step 3: Filter clusters by size
    filtered_clusters = np.zeros_like(labeled_data)
    cluster_points = {}

    for cluster_id in range(1, num_clusters + 1):
        cluster_indices = np.argwhere(labeled_data == cluster_id)
        cluster_size = len(cluster_indices)

        if cluster_size >= min_cluster_size:
            filtered_clusters[labeled_data == cluster_id] = cluster_id
            cluster_points[cluster_id] = cluster_indices

    print(f"Filtered clusters: {np.unique(filtered_clusters)[1:]}")  # Exclude 0 (background)

    # Step 4: Merge clusters using minimum pairwise distances
    merge_mapping = {}
    cluster_ids = list(cluster_points.keys())

    for i, cluster_id_a in enumerate(cluster_ids):
        for j, cluster_id_b in enumerate(cluster_ids):
            if i >= j:
                continue

            points_a = cluster_points[cluster_id_a]
            points_b = cluster_points[cluster_id_b]

            # Compute minimum pairwise distance
            min_distance = np.min(cdist(points_a, points_b))
            print(min_distance,cluster_id_a,cluster_id_b)
            if min_distance < distance_threshold:
                merge_mapping[cluster_id_b] = cluster_id_a
                print("merged")

    # Apply the merge mapping
    merged_clusters = np.zeros_like(filtered_clusters)
    for cluster_id in np.unique(filtered_clusters):
        if cluster_id == 0:
            continue

        current_label = cluster_id
        while current_label in merge_mapping:
            current_label = merge_mapping[current_label]

        merged_clusters[filtered_clusters == cluster_id] = current_label

    # Step 5: Generate cylindrical mask based on splines or polynomial fits
    cylindrical_mask = np.zeros_like(merged_clusters)

    for cluster_id in range(1, np.max(merged_clusters) + 1):
        if cluster_id not in cluster_points:
            continue

        # Combine all points in the merged cluster
        cluster_indices = np.vstack(cluster_points[cluster_id])

        # Add random noise to points (if enabled)
        if add_random_noise:
            noise = np.random.normal(0, noise_stddev, size=cluster_indices.shape)
            cluster_indices = cluster_indices.astype(float) + noise

        # Sort points along the principal axis
        pca = PCA(n_components=3)
        pca.fit(cluster_indices)
        principal_axis = pca.components_[0]
        projections = np.dot(cluster_indices - pca.mean_, principal_axis)
        sorted_indices = cluster_indices[np.argsort(projections)]

        # Fit a spline through the sorted points
        tck, _ = splprep(sorted_indices.T, s=smoothing_param)
        u_new = np.linspace(0, 1, num_spline_points)
        spline_points = np.array(splev(u_new, tck)).T

        # Apply polynomial fit if enabled
        if apply_polynomial_fit:
            poly_fit_x = np.polyfit(u_new, spline_points[:, 0], polynomial_degree)
            poly_fit_y = np.polyfit(u_new, spline_points[:, 1], polynomial_degree)
            poly_fit_z = np.polyfit(u_new, spline_points[:, 2], polynomial_degree)

            poly_points_x = np.polyval(poly_fit_x, u_new)
            poly_points_y = np.polyval(poly_fit_y, u_new)
            poly_points_z = np.polyval(poly_fit_z, u_new)

            spline_points = np.vstack([poly_points_x, poly_points_y, poly_points_z]).T

        # Generate cylinders along the (possibly smoothed) spline points
        for point in spline_points:
            x_center, y_center, z_center = point

            # Define bounds
            x_min, x_max = int(np.floor(x_center - cylinder_radius)), int(np.ceil(x_center + cylinder_radius))
            y_min, y_max = int(np.floor(y_center - cylinder_radius)), int(np.ceil(y_center + cylinder_radius))
            z_min, z_max = int(np.floor(z_center - cylinder_radius)), int(np.ceil(z_center + cylinder_radius))

            # Clip bounds to array limits
            x_min, x_max = max(x_min, 0), min(x_max, cylindrical_mask.shape[0] - 1)
            y_min, y_max = max(y_min, 0), min(y_max, cylindrical_mask.shape[1] - 1)
            z_min, z_max = max(z_min, 0), min(z_max, cylindrical_mask.shape[2] - 1)

            # Compute distances
            x_range, y_range, z_range = np.meshgrid(
                np.arange(x_min, x_max + 1),
                np.arange(y_min, y_max + 1),
                np.arange(z_min, z_max + 1),
                indexing="ij",
            )
            distances = np.sqrt(
                (x_range - x_center) ** 2
                + (y_range - y_center) ** 2
                + (z_range - z_center) ** 2
            )
            within_radius = distances <= cylinder_radius

            cylindrical_mask[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1] |= within_radius

    print("Cylindrical mask constructed.")

    # Relabel clusters
    final_labeled_clusters, num_final_clusters = label(cylindrical_mask > 0)
    
    # Calculate cluster sizes
    cluster_sizes = [(label_id, np.sum(final_labeled_clusters == label_id)) 
                     for label_id in range(1, num_final_clusters + 1)]
    
    # Sort clusters by size in descending order
    sorted_clusters = sorted(cluster_sizes, key=lambda x: x[1], reverse=True)
    
    # Create a mapping from old labels to new sorted labels
    label_mapping = {old_label: new_label + 1 for new_label, (old_label, _) in enumerate(sorted_clusters)}
    
    # Apply the mapping to relabel clusters
    sorted_labeled_clusters = np.zeros_like(final_labeled_clusters)
    for old_label, new_label in label_mapping.items():
        sorted_labeled_clusters[final_labeled_clusters == old_label] = new_label
    
    final_labeled_clusters = sorted_labeled_clusters
    print(f"Final number of clusters after relabeling and sorting: {num_final_clusters}")

    # Save data (if enabled)
    if save_output:
        gu.save_to_vti(
            filename=save_path + "_amp-phase_bulk_nan.vti",
            voxel_size=voxel_sizes,
            tuple_array=(
                nan_to_zero(amp),  # Replace with your implementation
                nan_to_zero(phase),
                final_labeled_clusters,
            ),
            tuple_fieldnames=("density", "phase", "strain_amp"),
            amplitude_threshold=0.01,
        )
        print(f"Saved processed data to {save_path}_amp-phase_bulk_nan.vti")

    # Create a debug plot (if enabled)
    if debug_plot:
        frames = []
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_box_aspect([1, 1, 1])

        # Extract the coordinates of the final labeled clusters
        cluster_indices = np.argwhere(final_labeled_clusters > 0)
        scatter=ax.scatter(
            cluster_indices[:, 0],
            cluster_indices[:, 1],
            cluster_indices[:, 2],
            s=1,
            c=final_labeled_clusters[final_labeled_clusters > 0],
            cmap="jet", 
        )
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)  # Add colorbar
        cbar.set_label("Cluster Labels")  # Optional: Label the colorbar
        ax.set_title("Final Clustering")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        for angle in range(0, 360, 2):  # Create frames for the GIF
            ax.view_init(30, angle)
            plt.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8").reshape(
                fig.canvas.get_width_height()[::-1] + (3,)
            )
            frames.append(frame)

        gif_path = save_path + "_final_dislocation_clustering_and_processing.gif"
        imageio.mimsave(gif_path, frames, fps=15)
        print(f"Saved debug GIF to {gif_path}")

    return final_labeled_clusters, num_final_clusters
#------------------------------------------------------------------------------------------------------------
def refine_cluster_with_dbscan(cluster_indices, eps=2.0, min_samples=5):
    """
    Refine cluster points using DBSCAN for better connectivity.

    Parameters:
    - cluster_indices: np.ndarray
        Array of points in the cluster.
    - eps: float
        Maximum distance between points to be considered as neighbors.
    - min_samples: int
        Minimum number of points to form a dense region.

    Returns:
    - dbscan_labels: np.ndarray
        Labels assigned by DBSCAN for each point in the cluster.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(cluster_indices)
    return dbscan_labels


def fit_splines_to_dbscan_components(cluster_indices, dbscan_labels, smoothing_param, num_spline_points):
    """
    Fit splines to each DBSCAN component.

    Parameters:
    - cluster_indices: np.ndarray
        Array of points in the cluster.
    - dbscan_labels: np.ndarray
        Labels assigned by DBSCAN for each point.
    - smoothing_param: float
        Smoothing parameter for spline fitting.
    - num_spline_points: int
        Number of points for spline interpolation.

    Returns:
    - splines: list of np.ndarray
        List of spline points for each DBSCAN component.
    """
    splines = []
    unique_labels = np.unique(dbscan_labels)

    for label in unique_labels:
        if label == -1:  # Skip noise points
            continue

        component_points = cluster_indices[dbscan_labels == label]
        if len(component_points) < 3:
            continue  # Skip components too small for spline fitting

        # Sort points along principal axis
        pca = PCA(n_components=3)
        pca.fit(component_points)
        principal_axis = pca.components_[0]
        projections = np.dot(component_points - pca.mean_, principal_axis)
        sorted_indices = component_points[np.argsort(projections)]

        # Fit spline
        tck, _ = splprep(sorted_indices.T, s=smoothing_param)
        u_new = np.linspace(0, 1, num_spline_points)
        spline_points = np.array(splev(u_new, tck)).T
        splines.append(spline_points)

    return splines
#------------------------------------------------------------------------------------------------------------
def process_and_merge_clusters_dislo_strain_map_refined(data,amp,phase,save_path,voxel_sizes,threshold=0.35,min_cluster_size=10,distance_threshold=10.0,cylinder_radius=3.0,
                                                        num_spline_points=1000,smoothing_param=2,eps=2.0,min_samples=5,save_output=True,debug_plot=True,):
    """
    Process and refine clusters using DBSCAN and splines.

    Parameters:
    - data: np.ndarray
        3D input array representing the data.
    - amp: np.ndarray
        3D array of amplitude values.
    - phase: np.ndarray
        3D array of phase values.
    - save_path: str
        Path to save the output file.
    - voxel_sizes: list
        List of voxel sizes for the output file.
    - threshold: float
        Threshold to binarize the data.
    - min_cluster_size: int
        Minimum size of clusters to keep.
    - distance_threshold: float
        Maximum distance between closest points of clusters to allow merging.
    - cylinder_radius: float
        Radius of the cylindrical approximation.
    - num_spline_points: int
        Number of points to use for spline interpolation.
    - smoothing_param: float
        Smoothing parameter for spline fitting.
    - eps: float
        DBSCAN maximum distance between points for connectivity.
    - min_samples: int
        Minimum samples for DBSCAN clustering.
    - save_output: bool
        Whether to save the output data as a VTI file.
    - debug_plot: bool
        Whether to create a 3D GIF showing the final clustering result.

    Returns:
    - final_labeled_clusters: np.ndarray
        3D mask with consecutively relabeled clusters.
    - num_final_clusters: int
        Number of final clusters after relabeling.
    """
    # Step 1: Threshold the data to identify potential clusters
    binary_data = (data > threshold).astype(np.uint8)

    # Step 2: Label the clusters
    labeled_data, num_clusters = label(binary_data)
    print(f"Number of clusters identified: {num_clusters}")

    # Step 3: Filter clusters by size
    filtered_clusters = np.zeros_like(labeled_data)
    cluster_points = {}

    for cluster_id in range(1, num_clusters + 1):
        cluster_indices = np.argwhere(labeled_data == cluster_id)
        cluster_size = len(cluster_indices)

        if cluster_size >= min_cluster_size:
            filtered_clusters[labeled_data == cluster_id] = cluster_id
            cluster_points[cluster_id] = cluster_indices

    print(f"Filtered clusters: {np.unique(filtered_clusters)[1:]}")  # Exclude 0 (background)

    # Step 4: Merge clusters using minimum pairwise distances
    merge_mapping = {}
    cluster_ids = list(cluster_points.keys())

    for i, cluster_id_a in enumerate(cluster_ids):
        for j, cluster_id_b in enumerate(cluster_ids):
            if i >= j:
                continue

            points_a = cluster_points[cluster_id_a]
            points_b = cluster_points[cluster_id_b]

            # Compute minimum pairwise distance
            min_distance = np.min(cdist(points_a, points_b))
            if min_distance < distance_threshold:
                merge_mapping[cluster_id_b] = cluster_id_a

    # Apply the merge mapping
    merged_clusters = np.zeros_like(filtered_clusters)
    for cluster_id in np.unique(filtered_clusters):
        if cluster_id == 0:
            continue

        current_label = cluster_id
        while current_label in merge_mapping:
            current_label = merge_mapping[current_label]

        merged_clusters[filtered_clusters == cluster_id] = current_label

    # Step 5: Generate cylindrical mask with DBSCAN and splines
    cylindrical_mask = np.zeros_like(merged_clusters)

    for cluster_id in range(1, np.max(merged_clusters) + 1):
        if cluster_id not in cluster_points:
            continue

        cluster_indices = np.vstack(cluster_points[cluster_id])

        # Refine cluster with DBSCAN
        dbscan_labels = refine_cluster_with_dbscan(cluster_indices, eps=eps, min_samples=min_samples)

        # Fit splines to DBSCAN components
        splines = fit_splines_to_dbscan_components(cluster_indices, dbscan_labels, smoothing_param, num_spline_points)

        # Generate cylindrical mask for each spline
        for spline_points in splines:
            for point in spline_points:
                x_center, y_center, z_center = point

                x_min, x_max = int(np.floor(x_center - cylinder_radius)), int(np.ceil(x_center + cylinder_radius))
                y_min, y_max = int(np.floor(y_center - cylinder_radius)), int(np.ceil(y_center + cylinder_radius))
                z_min, z_max = int(np.floor(z_center - cylinder_radius)), int(np.ceil(z_center + cylinder_radius))

                x_min, x_max = max(x_min, 0), min(x_max, cylindrical_mask.shape[0] - 1)
                y_min, y_max = max(y_min, 0), min(y_max, cylindrical_mask.shape[1] - 1)
                z_min, z_max = max(z_min, 0), min(z_max, cylindrical_mask.shape[2] - 1)

                x_range, y_range, z_range = np.meshgrid(
                    np.arange(x_min, x_max + 1),
                    np.arange(y_min, y_max + 1),
                    np.arange(z_min, z_max + 1),
                    indexing="ij",
                )
                distances = np.sqrt(
                    (x_range - x_center) ** 2
                    + (y_range - y_center) ** 2
                    + (z_range - z_center) ** 2
                )
                within_radius = distances <= cylinder_radius
                cylindrical_mask[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1] |= within_radius

    print("Cylindrical mask constructed.")

    # Relabel clusters
    final_labeled_clusters, num_final_clusters = label(cylindrical_mask > 0)

    # Save and debug plot
    if save_output:
        gu.save_to_vti(
            filename=save_path + "_amp-phase_bulk_nan.vti",
            voxel_size=voxel_sizes,
            tuple_array=(
                nan_to_zero(amp),
                nan_to_zero(phase),
                final_labeled_clusters,
            ),
            tuple_fieldnames=("density", "phase", "strain_amp"),
            amplitude_threshold=0.01,
        )
        print("vti file save at"+save_path + "_amp-phase_bulk_nan.vti")

    if debug_plot:
        frames = []
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_box_aspect([1, 1, 1])

        cluster_indices = np.argwhere(final_labeled_clusters > 0)
        scatter = ax.scatter(
            cluster_indices[:, 0],
            cluster_indices[:, 1],
            cluster_indices[:, 2],
            s=1,
            c=final_labeled_clusters[final_labeled_clusters > 0],
            cmap="jet",
        )
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label("Cluster Labels")
        ax.set_title("Refined Clustering")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        for angle in range(0, 360, 2):
            ax.view_init(30, angle)
            plt.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8").reshape(
                fig.canvas.get_width_height()[::-1] + (3,)
            )
            frames.append(frame)

        gif_path = save_path + "_refined_dislocation_clustering_and_processing.gif"
        imageio.mimsave(gif_path, frames, fps=15)
        print(f"Saved debug GIF to {gif_path}")

    return final_labeled_clusters, num_final_clusters
#------------------------------------------------------------------------------------------------------------
def create_circular_mask_with_angles(data_shape, line_points, selected_point_index, r, dr):
    """
    Create a circular mask and compute polar angles around a selected point on a line.

    Parameters:
    - data_shape: tuple
        Shape of the original 3D data.
    - line_points: np.ndarray
        Array of (x, y, z) points defining the central line.
    - selected_point_index: int
        Index of the point on the line to use as the center of the circle.
    - r: float
        Inner radius of the circular region.
    - dr: float
        Thickness of the circular region.

    Returns:
    - circular_mask: np.ndarray
        Binary mask (3D array) with 1s in the circular region and 0s elsewhere.
    - polar_angles: np.ndarray
        Array of polar angles (in radians) for each voxel in the mask.
    """
    # Get the center point on the line
    center_point = line_points[selected_point_index]

    # Define the local Z-axis as the direction of the line
    if selected_point_index == 0:
        # If the first point, use the next point as a reference
        z_axis = line_points[1] - center_point
    elif selected_point_index == len(line_points) - 1:
        # If the last point, use the previous point as a reference
        z_axis = center_point - line_points[-2]
    else:
        # Otherwise, average the vectors from the previous and next points
        z_axis = (
            line_points[selected_point_index + 1] - line_points[selected_point_index - 1]
        ) / 2

    # Normalize the Z-axis
    z_axis = z_axis / np.linalg.norm(z_axis)

    # Define a random perpendicular vector to the Z-axis as the X-axis
    random_vector = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
    x_axis = np.cross(z_axis, random_vector)
    x_axis = x_axis / np.linalg.norm(x_axis)

    # Define the Y-axis as orthogonal to both Z and X
    y_axis = np.cross(z_axis, x_axis)

    # Generate a grid of all voxel indices
    grid_x, grid_y, grid_z = np.meshgrid(
        np.arange(data_shape[0]),
        np.arange(data_shape[1]),
        np.arange(data_shape[2]),
        indexing="ij",
    )
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    # Shift grid points relative to the center point
    shifted_points = grid_points - center_point

    # Convert the shifted points to the local cylindrical coordinate system
    local_x = np.dot(shifted_points, x_axis)
    local_y = np.dot(shifted_points, y_axis)
    local_z = np.dot(shifted_points, z_axis)

    # Compute the radial distances and polar angles
    radial_distances = np.sqrt(local_x**2 + local_y**2)
    polar_angles = np.arctan2(local_y, local_x)

    # Mask based on the radial distance
    circular_mask = np.zeros(data_shape, dtype=np.uint8)
    circular_mask_flat = (radial_distances >= r) & (radial_distances <= r + dr) & (
        np.abs(local_z) < 0.5
    )  # Limit to a thin slice around the center point
    circular_mask.flat[circular_mask_flat] = 1

    # Polar angles within the mask
    polar_angles_masked = np.zeros(data_shape, dtype=np.float32)
    polar_angles_masked.flat[circular_mask_flat] = polar_angles[circular_mask_flat]

    return circular_mask, polar_angles_masked
#------------------------------------------------------------------------------------------------------------
def create_cylindrical_mask_with_polar_angles(data_shape, line_points, r, dr):
    """
    Create a mask with 1s in the region defined by r and r+dr around the central line,
    and calculate the polar angles relative to the principal axis of the line.

    Parameters:
    - data_shape: tuple
        Shape of the original 3D data.
    - line_points: np.ndarray
        Array of shape (N, 3) representing the Cartesian coordinates of the central line.
    - r: float
        Inner radius of the cylindrical shell.
    - dr: float
        Thickness of the cylindrical shell.

    Returns:
    - mask: np.ndarray
        Binary 3D mask with 1s in the region of interest and 0s elsewhere.
    - polar_angle_mask: np.ndarray
        3D array with the polar angle values (in radians) for voxels within the mask.
    """
    # Create a KDTree for fast nearest-neighbor search
    tree = cKDTree(line_points)

    # Perform PCA to determine the principal axis of the line
    pca = PCA(n_components=3)
    pca.fit(line_points)
    principal_axis = pca.components_[0]
    line_midpoint = np.mean(line_points, axis=0)

    # Generate a grid of all voxel indices
    grid_x, grid_y, grid_z = np.meshgrid(
        np.arange(data_shape[0]),
        np.arange(data_shape[1]),
        np.arange(data_shape[2]),
        indexing="ij",
    )
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    # Find the distance from each voxel to the nearest line point
    distances, _ = tree.query(grid_points)

    # Reshape distances to match the 3D data shape
    distances_3d = distances.reshape(data_shape)

    # Create the mask based on the distance
    mask = np.zeros(data_shape, dtype=np.uint8)
    mask[(distances_3d >= r) & (distances_3d <= r + dr)] = 1

    # Calculate polar angles for voxels within the mask
    polar_angle_mask = np.zeros(data_shape, dtype=np.float32)

    # Shift grid points relative to the line midpoint
    shifted_points = grid_points - line_midpoint

    # Project onto the plane perpendicular to the principal axis
    radial_vectors = shifted_points - np.outer(
        np.dot(shifted_points, principal_axis), principal_axis
    )
    radial_vectors_3d = radial_vectors.reshape(data_shape + (3,))

    # Compute polar angles
    theta = np.arctan2(radial_vectors_3d[..., 1], radial_vectors_3d[..., 0])
    polar_angle_mask[mask == 1] = theta[mask == 1]

    return mask, polar_angle_mask
#------------------------------------------------------------------------------------------------------------
def automatic_obj_slice_dilo_finding(obj_slice,threshold_module=0.005,plot=False):
    module = np.abs(obj_slice)
    support = module > .2 * np.max(module)
    
    support_filled = np.zeros(support.shape)
    for axis in range(support.ndim):
        mask_1 = np.cumsum(support, axis=axis)
        mask_2 = np.flip(np.cumsum(np.flip(support,axis), axis=axis), axis)
        mask = (mask_1!=0) * (mask_2!=0) 
        support_filled[mask != 0] = 1
    dislo = support_filled - support
    dislo_position = np.array(center_of_mass_calculation_two_steps(dislo))
    
    if plot:
        fig,ax = plt.subplots(1,2, figsize=(8,4))
        ax[0].matshow(module, cmap='gray_r')
        ax[1].matshow(module, cmap='gray_r')
        ax[1].scatter([dislo_position[1]], [dislo_position[0]], color='r', s=30, label='dislo position')
        ax[1].legend(fontsize=15)
    return dislo_position
#------------------------------------------------------------------------------------------------------------
def dislocation_reorientation(obj, normal, dislo_position,plot=False):
    rotation_matrix = rotation_matrix_from_vectors([1,0,0], normal)

    pos = [np.arange(obj.shape[axis]) - dislo_position[axis] for axis in range(obj.ndim)]
    
    # Create interpolation function
    rgi = RegularGridInterpolator(
        (
            pos[0],
            pos[1],
            pos[2],
        ),
        obj,
        method="linear",
        bounds_error=False,
        fill_value=0,
    )
    
    # Find the largest distance to center before interpolating
    r = np.array(np.meshgrid(pos[0], pos[1], pos[2]))
    r = np.sqrt(np.sum(r**2., axis=0))
    radius_max = np.ceil(np.max(r))
    # Make the rotated grid
    x = np.arange(-radius_max, +radius_max)
    pos_rot  = np.array(np.meshgrid(x,x,x))
    shape = pos_rot.shape
    pos_rot = np.reshape(pos_rot, (3, np.prod(pos_rot.shape[1:])))
    pos_rot = np.dot(rotation_matrix, pos_rot)
    pos_rot = np.reshape(pos_rot, shape)
    
    # Create the rotated object
    obj_rot = rgi((pos_rot[0],pos_rot[1],pos_rot[2]), method='linear')
    obj_rot = center_object(obj_rot)
    obj_rot = apply_roi(obj_rot, automatic_object_roi(obj_rot,factor=.2))
    
    if plot:
        plot_2D_slices_middle(obj, crop=False, fig_title='original object')
        plot_2D_slices_middle(obj_rot, crop=False, fig_title='rotated object')
    
    return obj_rot
#------------------------------------------------------------------------------------------------------------
def phase_around_dislo(phase, dislo_position, r0,dr=1,eleminate_zero_values=True,plot=False):
    phase=nan_to_zero(phase)
    pos = np.indices(phase.shape)
    pos = pos - dislo_position[:,None,None]
    angle = np.arctan2(pos[0], pos[1])
    r = np.sqrt(np.sum(pos**2, axis=0))
    
    ring_mask = np.abs(r-r0) < dr
    
    
    phase_ring = np.copy(phase[ring_mask])
    angle_ring = np.copy(angle[ring_mask])
    if eleminate_zero_values:
        angle_ring = angle_ring[phase_ring!=0]
        phase_ring = phase_ring[phase_ring!=0]
    
    indices_sort = np.argsort(angle_ring)
    angle_ring = angle_ring[indices_sort]
    phase_ring = phase_ring[indices_sort]
    
    phase_ring = np.unwrap(phase_ring)
    
    if plot:
        fig,ax = plt.subplots(1,2, figsize=(8,4))
        ax[0].matshow(phase, cmap='hsv')
        mask_plot = np.copy(ring_mask)
        mask_plot[mask_plot != 0.] = 1.
        ax[0].imshow( np.dstack([mask_plot, np.zeros(mask_plot.shape), np.zeros(mask_plot.shape), mask_plot]), aspect='auto')
        
        ax[1].plot(angle_ring, phase_ring, '.')
        ax[1].set_xlabel('angle around dislocation (rad)', fontsize=15)
        ax[1].set_ylabel('phase (rad)', fontsize=15)
        
        fig.tight_layout()
        
    return angle_ring, phase_ring
#------------------------------------------------------------------------------------------------------------
def plot_compare_phase_radius(phase, dislo_position,radius_min=6, radius_max=25, nb_radius=6,fig_title=None):
    radius_list = np.linspace(radius_min, radius_max, nb_radius)

    fig,ax = plt.subplots(1,1, figsize=(7,5))
    for radius in radius_list:
        angle_ring, phase_ring = phase_around_dislo(phase, dislo_position, radius)
        ax.plot(angle_ring, phase_ring, '-', label=f'{round(radius,1)}')
    ax.legend(title='distance from\ndislocation center')
    ax.set_xlabel('angle around dislocation (rad)', fontsize=15)
    ax.set_ylabel('phase (rad)', fontsize=15)
    
    if fig_title is not None:
        ax.set_title(fig_title, fontsize=20)
    
    fig.tight_layout()
    return
#------------------------------------------------------------------------------------------------------------
def slice_coord_array_along_axis(array,axis,coord):
    s = [slice(None, None, None) for ii in range(array.ndim)]
    s[axis] = array.shape[axis]//2
    return tuple(s)
#------------------------------------------------------------------------------------------------------------
def phase_difference_fit_function(theta, alpha, beta):
    return alpha * sin(theta) + beta * cos(theta)
#------------------------------------------------------------------------------------------------------------
def remove_phase_ramp_dislo(phase, dislo_position,radius_1=6, radius_2=15,function_method="trigo",plot=True):
    '''
    Remove the phase ramp in an object slice. 
    This use the approximation that the dislocation displacement field as function of the angle around it 
    should be the same for any radius (distance from the dislocation center)
    :radius_1: first radius at which the dislo phase is taken
    :radius_2: second radius
    '''

    angle_ring_1, phase_ring_1 = phase_around_dislo(phase, dislo_position, radius_1, plot=plot)
    angle_ring_2, phase_ring_2 = phase_around_dislo(phase, dislo_position, radius_2, plot=plot)

    if plot:
        fig,ax = plt.subplots(2,3, figsize=(12,8))
        ax[0,0].plot(angle_ring_1, phase_ring_1, '.', label=f'{round(radius_1,1)}')
        ax[0,0].plot(angle_ring_2, phase_ring_2, '.', label=f'{round(radius_2,1)}')

        ax[0,0].set_xlabel('angle around dislocation (rad)', fontsize=15)
        ax[0,0].set_ylabel('phase (rad)', fontsize=15)
        ax[0,0].legend(title='distance from\ndislocation center', fontsize=12)
        ax[0,0].set_title('dislocation phase\naround center', fontsize=20)
        

    # Make interpolations to have phase at same angles
    angle_ring = np.linspace(array([angle_ring_1.min(),angle_ring_2.min()]).max(),array([angle_ring_1.max(),angle_ring_2.max()]).min(),100)
    phase_ring_1 = np.interp(angle_ring, angle_ring_1, phase_ring_1)
    phase_ring_2 = np.interp(angle_ring, angle_ring_2, phase_ring_2)
    # Now compute the difference (should be 0 if there's no phase ramp)
    difference = (phase_ring_2 - phase_ring_1) / (radius_2-radius_1)
    if function_method=="trigo":
        popt, pcov = curve_fit(phase_difference_fit_function, angle_ring, difference)
        fit = phase_difference_fit_function(angle_ring, *popt)
    else:
        # Perform Fourier Transform
        fft_data = fft.fft(difference)
        frequencies = fft.fftfreq(len(angle_ring), d=(angle_ring[1] - angle_ring[0]))
        
        # Keep only the first few terms (filter high frequencies)
        fft_filtered = fft_data.copy()
        fft_filtered[abs(frequencies) > 1] = 0
        
        # Inverse Fourier Transform to reconstruct the signal
        fit= fft.ifft(fft_filtered).real
        popt=fit


    if plot:
        ax[0,1].plot(angle_ring, phase_ring_1, '.', label=f'{round(radius_1,1)}')
        ax[0,1].plot(angle_ring, phase_ring_2, '.', label=f'{round(radius_1,1)}')
        
        ax[0,1].set_xlabel('angle around dislocation (rad)', fontsize=15)
        ax[0,1].set_ylabel('phase (rad)', fontsize=15)
        ax[0,1].legend(title='distance from\ndislocation center', fontsize=12)
        ax[0,1].set_title('dislocation phase\naround center\nafter interpolation', fontsize=20)
        
        ax[0,2].plot(angle_ring, difference, '.', label='difference')
        ax[0,2].plot(angle_ring, fit, 'r-', label='ramp fit')
        ax[0,2].legend(fontsize=12)
        ax[0,2].set_title('phase difference and fit', fontsize=20)
        
        
    # Remove the phase ramp
    pos = np.indices(phase.shape)
    pos = pos - dislo_position[:,None,None]
    ramp_fit = fit[0] * pos[0] + fit[1] * pos[1]
    phase_no_ramp = (zero_to_nan(phase) - ramp_fit)
    
    if plot:
        ax[1,0].matshow(phase, cmap='jet')
        ax[1,0].set_title('original phase', fontsize=20)
        ax[1,1].matshow(phase_no_ramp, cmap='jet')
        ax[1,1].set_title('phase after ramp removal', fontsize=20)
        
        angle_ring_1, phase_ring_1 = phase_around_dislo(phase_no_ramp, dislo_position, radius_1)
        angle_ring_2, phase_ring_2 = phase_around_dislo(phase_no_ramp, dislo_position, radius_2)
        
        ax[1,2].plot(angle_ring_1, phase_ring_1, '.', label=f'{round(radius_1,1)}')
        ax[1,2].plot(angle_ring_2, phase_ring_2, '.', label=f'{round(radius_2,1)}')

        ax[1,2].set_xlabel('angle around dislocation (rad)', fontsize=15)
        ax[1,2].set_ylabel('phase (rad)', fontsize=15)
        ax[1,2].legend(title='distance from\ndislocation center', fontsize=12)
        ax[1,2].set_title('dislocation phase\naround center\nafter ramp removal', fontsize=20)
        
        
        fig.tight_layout()
        
    return phase_no_ramp
#------------------------------------------------------------------------------------------------------------
def plot_compare_phase_radius_around_dislo(scan,amp,phase,selected_dislocation_data,selected_point_index=0,save_vti=False,fig_title=None,plot_debug=True,radius_min=2, radius_max=10, nb_radius=6,dr=1):
    radius_list = np.linspace(radius_min, radius_max, nb_radius)

    fig,ax = plt.subplots(1,1, figsize=(7,5))
    for radius in radius_list:
        phase_ring,angle_ring,circular_mask=plot_phase_around_dislo_one_radius_new(scan,amp,phase,selected_dislocation_data,radius,dr,selected_point_index=selected_point_index,save_vti=False,plot_debug=False);
        y=phase_ring[phase_ring!=0].flatten() 
        x=angle_ring[phase_ring!=0].flatten()
        ax.plot(x,y,'o', label=f'r:{round(radius,1)}'+rf" $\Delta\phi$ {np.round(-y.min()+y.max(),2)}rad")
    ax.legend(title='distance from\ndislocation center')
    ax.set_xlabel('angle around dislocation (rad)', fontsize=15)
    ax.set_ylabel('phase (rad)', fontsize=15)
    
    if fig_title is not None:
        ax.set_title(fig_title, fontsize=20)
    
    fig.tight_layout()
    return


#------------------------------------------------------------------------------------------------------------
def remove_phase_ramp_dislo_new(scan,amp,phase,selected_dislocation_data,selected_point_index=0,save_vti=False,fig_title=None,plot_debug=True,radius_1=2, radius_2=10,dr=1,function_method="trigo"):
    '''
    Remove the phase ramp in an object slice. 
    This use the approximation that the dislocation displacement field as function of the angle around it 
    should be the same for any radius (distance from the dislocation center)
    :radius_1: first radius at which the dislo phase is taken
    :radius_2: second radius
    '''
    phase_ring,angle_ring,circular_mask=plot_phase_around_dislo_one_radius_new (scan,amp,phase,selected_dislocation_data,selected_point_index,radius_1,dr,save_vti=False,plot_debug=False);
    angle_ring_1=phase_ring[phase_ring!=0].flatten() 
    phase_ring_1=angle_ring[phase_ring!=0].flatten()

    phase_ring,angle_ring,circular_mask=plot_phase_around_dislo_one_radius_new (scan,amp,phase,selected_dislocation_data,selected_point_index,radius_2,dr,save_vti=False,plot_debug=False);
    angle_ring_2=phase_ring[phase_ring!=0].flatten() 
    phase_ring_2=angle_ring[phase_ring!=0].flatten()


    if plot:
        fig,ax = plt.subplots(2,3, figsize=(12,8))
        ax[0,0].plot(angle_ring_1, phase_ring_1, '.', label=f'{round(radius_1,1)}')
        ax[0,0].plot(angle_ring_2, phase_ring_2, '.', label=f'{round(radius_2,1)}')

        ax[0,0].set_xlabel('angle around dislocation (rad)', fontsize=15)
        ax[0,0].set_ylabel('phase (rad)', fontsize=15)
        ax[0,0].legend(title='distance from\ndislocation center', fontsize=12)
        ax[0,0].set_title('dislocation phase\naround center', fontsize=20)
        

    # Make interpolations to have phase at same angles
    angle_ring = np.linspace(array([angle_ring_1.min(),angle_ring_2.min()]).max(),array([angle_ring_1.max(),angle_ring_2.max()]).min(),100)
    phase_ring_1 = np.interp(angle_ring, angle_ring_1, phase_ring_1)
    phase_ring_2 = np.interp(angle_ring, angle_ring_2, phase_ring_2)
    # Now compute the difference (should be 0 if there's no phase ramp)
    difference = (phase_ring_2 - phase_ring_1) / (radius_2-radius_1)
    if function_method=="trigo":
        popt, pcov = curve_fit(phase_difference_fit_function, angle_ring, difference)
        fit = phase_difference_fit_function(angle_ring, *popt)
    else:
        # Perform Fourier Transform
        fft_data = fft.fft(difference)
        frequencies = fft.fftfreq(len(angle_ring), d=(angle_ring[1] - angle_ring[0]))
        
        # Keep only the first few terms (filter high frequencies)
        fft_filtered = fft_data.copy()
        fft_filtered[abs(frequencies) > 1] = 0
        
        # Inverse Fourier Transform to reconstruct the signal
        fit= fft.ifft(fft_filtered).real
        popt=fit


    if plot:
        ax[0,1].plot(angle_ring, phase_ring_1, '.', label=f'{round(radius_1,1)}')
        ax[0,1].plot(angle_ring, phase_ring_2, '.', label=f'{round(radius_1,1)}')
        
        ax[0,1].set_xlabel('angle around dislocation (rad)', fontsize=15)
        ax[0,1].set_ylabel('phase (rad)', fontsize=15)
        ax[0,1].legend(title='distance from\ndislocation center', fontsize=12)
        ax[0,1].set_title('dislocation phase\naround center\nafter interpolation', fontsize=20)
        
        ax[0,2].plot(angle_ring, difference, '.', label='difference')
        ax[0,2].plot(angle_ring, fit, 'r-', label='ramp fit')
        ax[0,2].legend(fontsize=12)
        ax[0,2].set_title('phase difference and fit', fontsize=20)
        
        
    # Remove the phase ramp
    pos = np.indices(phase.shape)
    pos = pos - dislo_position[:,None,None]
    ramp_fit = fit[0] * pos[0] + fit[1] * pos[1]
    phase_no_ramp = (zero_to_nan(phase) - ramp_fit)
    
    if plot:
        ax[1,0].matshow(phase, cmap='jet')
        ax[1,0].set_title('original phase', fontsize=20)
        ax[1,1].matshow(phase_no_ramp, cmap='jet')
        ax[1,1].set_title('phase after ramp removal', fontsize=20)
        
        angle_ring_1, phase_ring_1 = phase_around_dislo(phase_no_ramp, dislo_position, radius_1)
        angle_ring_2, phase_ring_2 = phase_around_dislo(phase_no_ramp, dislo_position, radius_2)
        
        ax[1,2].plot(angle_ring_1, phase_ring_1, '.', label=f'{round(radius_1,1)}')
        ax[1,2].plot(angle_ring_2, phase_ring_2, '.', label=f'{round(radius_2,1)}')

        ax[1,2].set_xlabel('angle around dislocation (rad)', fontsize=15)
        ax[1,2].set_ylabel('phase (rad)', fontsize=15)
        ax[1,2].legend(title='distance from\ndislocation center', fontsize=12)
        ax[1,2].set_title('dislocation phase\naround center\nafter ramp removal', fontsize=20)
        
        
        fig.tight_layout()
        
    return phase_no_ramp
#------------------------------------------------------------------------------------------------------------
def process_dislocation_scan(scan, files_data_ortho, scan_list, save_path,threshold_strain_amp=0.3, threshold_module=0.005,dislocation_axis=2, radius_1=3, radius_2=5,radius_min=2, radius_max=8, nb_radius=6,normal=np.array([0, -0., 1]),midle_slice=None):
    """
    Processes a single scan for dislocation phase and ramp removal.

    Parameters:
    - scan (str): Scan name to process.
    - files_data_ortho (list): List of file paths for the scans.
    - scan_list (list): List of scan identifiers corresponding to the files.
    - save_path (str): Path to save the resulting figures.
    - threshold_strain_amp (float): Threshold for strain amplitude filtering.
    - threshold_module (float): Threshold for object module filtering.
    - dislocation_axis (int): Axis perpendicular to the dislocation plane.
    - radius_1 (float): First radius for ramp removal.
    - radius_2 (float): Second radius for ramp removal.
    - radius_min (float): Minimum radius for phase comparison.
    - radius_max (float): Maximum radius for phase comparison.
    - nb_radius (int): Number of radii for phase comparison plots.
    - normal (numpy.ndarray): Dislocation direction vector.

    Returns:
    - phase_no_ramp (numpy.ndarray): Phase data after ramp removal.
    """

    # Load data for the specified scan
    file = files_data_ortho[np.where(scan_list == scan)[0][0]]
    f = np.load(file)
    amp = nan_to_zero(f["amp"])
    phase = nan_to_zero(f["phase"])
    strain_amp = nan_to_zero(f["strain_amp"])
    strain_amp[strain_amp < threshold_strain_amp] = 0
    obj = amp * np.exp(1j * phase)
    voxel_sizes = f['voxel_size']

    # Plot and save original object
    plot_2D_slices_middle(obj, voxel_sizes=voxel_sizes, threshold_module=threshold_module)
    plt.suptitle(f"original obj {scan}", y=1.05)
    plt.savefig(f"{save_path}{scan}_original_obj.png")
    plt.show()

    # Select slice perpendicular to the dislocation axis
    if midle_slice is None:
        midle_slice = obj.shape[dislocation_axis] // 2
    coord_dislo_all = np.where(strain_amp > 0.35)
    coord_dislo_all_filtered = [
        [
            tuple(ii) for ii in np.array(coord_dislo_all)[:, coord_dislo_all[dislocation_axis] == i][
                :, np.where(np.logical_and.reduce((abs(np.diff(np.array(coord_dislo_all)
                    [:, coord_dislo_all[dislocation_axis] == i])) < 5), axis=0))[0]
            ]
        ]
        for i in range(obj.shape[dislocation_axis])
    ]
    coord_dislo_MIDLE_filtered = coord_dislo_all_filtered[midle_slice]
    dislo_position = np.array(coord_dislo_MIDLE_filtered)[:, 0]
    dislo_position_slice = np.delete(dislo_position, dislocation_axis)

    obj_slice = obj[slice_coord_array_along_axis(obj, dislocation_axis, midle_slice)]
    strain_amp_slice = strain_amp[slice_coord_array_along_axis(obj, dislocation_axis, midle_slice)]

    plot_object_module_phase_2d(obj_slice, crop=False, threshold_module=threshold_module)
    plt.suptitle(f"chosen slice obj {scan}", y=1.05)
    plt.savefig(f"{save_path}{scan}_slice_{midle_slice}_chosen_slice_obj.png")
    plt.show()

    # Get cropped module and phase
    module, phase = nan_to_zero(get_cropped_module_phase(obj_slice, threshold_module=threshold_module, crop=False))
    plt.show()

    # Plot phase around dislocation for a given radius
    angle_ring, phase_ring = phase_around_dislo(phase, dislo_position_slice, radius_1, dr=2, eleminate_zero_values=True, plot=True)
    plt.suptitle(f"phase around dislo {scan}", y=1.05)
    plt.savefig(f"{save_path}{scan}_slice_{midle_slice}_phase_around_dislo.png")
    plt.show()

    plot_compare_phase_radius(phase, dislo_position_slice, radius_min=radius_min, radius_max=radius_max, nb_radius=nb_radius,
                              fig_title='phase around dislocation before ramp removal')
    plt.suptitle(f"phase around dislo {scan}", y=1.05)
    plt.savefig(f"{save_path}{scan}_slice_{midle_slice}_phase_around_dislo_multipleradius.png")
    plt.show()

    # Remove phase ramp
    phase_no_ramp = remove_phase_ramp_dislo(phase, dislo_position_slice, radius_1=radius_1, radius_2=radius_2, plot=True)
    plt.suptitle(f"phase ramp around dislo {scan}", y=1.05)
    plt.savefig(f"{save_path}{scan}_slice_{midle_slice}_phase_ramp_around_dislo.png")
    plt.show()

    plot_compare_phase_radius(phase_no_ramp, dislo_position_slice, radius_min=radius_min, radius_max=radius_max, nb_radius=nb_radius,
                              fig_title='phase around dislocation after ramp removal')
    plt.suptitle(f"ramped phase around dislo {scan}", y=1.05)
    plt.savefig(f"{save_path}{scan}_slice_{midle_slice}_rampedphase_around_dislo_multipleradius.png")
    plt.show()

    return phase_no_ramp
#------------------------------------------------------------------------------------------------------------
def extract_structure(volume, threshold=0.5):
    """Extract points from the volume where the intensity exceeds a threshold."""
    indices = np.argwhere(volume > threshold)
    return indices
#------------------------------------------------------------------------------------------------------------
def fit_line_3d(points):
    """Fit a 3D line to the given points using SVD."""
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    _, _, vh = np.linalg.svd(centered_points)
    direction = vh[0]
    return centroid, direction
#------------------------------------------------------------------------------------------------------------
def generate_filled_cylinder_with_disks(shape, centroid, direction, radius, height, step=1):
    """Generate a 3D volume with a filled cylinder using disks along the fitted line."""
    direction = direction / np.linalg.norm(direction)
    volume = np.zeros(shape)

    # Generate points along the line within the specified height
    t_values = np.arange(-height / 2, height / 2, step)
    for t in t_values:
        # Compute the center of the current disk
        disk_center = centroid + t * direction

        # Create grid coordinates for the volume
        x, y, z = np.indices(shape)

        # Compute the distance of each grid point to the disk center
        distances = np.sqrt((x - disk_center[0])**2 + (y - disk_center[1])**2 + (z - disk_center[2])**2)

        # Set points within the disk radius to 1
        volume[distances <= radius] = 1

    return volume
#------------------------------------------------------------------------------------------------------------
def create_circular_mask_with_angles_new      (data_shape, centroid, direction, selected_point_index, r, dr, slice_thickness=2):
    """Create a circular mask and compute polar angles around a defined position along a direction vector.
    
    Args:
        data_shape (tuple): Shape of the 3D data (e.g., (100, 100, 100)).
        centroid (np.array): Central point of the fitted line (e.g., np.array([50, 50, 50])).
        direction (np.array): Direction vector of the line (must be normalized).
        selected_point_index (float): Scalar to move along the direction vector from the centroid.
        r (float): Inner radius of the circular mask.
        dr (float): Thickness of the circular mask.
        slice_thickness (float): Thickness of the slice along the direction vector.
    
    Returns:
        circular_mask (np.ndarray): 3D mask with the circular region marked (1s for the mask, 0s elsewhere).
        polar_angles_masked (np.ndarray): 3D array with polar angles where the mask is applied.
    """
    selected_point_index=selected_point_index/2
    # Normalize the direction vector
    direction = direction / np.linalg.norm(direction)
    
    # Compute the disk center based on the selected point index along the direction
    disk_center = centroid + selected_point_index * direction

    # Define the local Z-axis (parallel to the direction vector)
    z_axis = direction

    # Define a random perpendicular vector to the Z-axis as the X-axis
    random_vector = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
    x_axis = np.cross(z_axis, random_vector)
    x_axis = x_axis / np.linalg.norm(x_axis)

    # Define the Y-axis as orthogonal to both Z and X
    y_axis = np.cross(z_axis, x_axis)

    # Generate a grid of all voxel indices
    grid_x, grid_y, grid_z = np.meshgrid(
        np.arange(data_shape[0]),
        np.arange(data_shape[1]),
        np.arange(data_shape[2]),
        indexing="ij",
    )
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    # Shift grid points relative to the disk center
    shifted_points = grid_points - disk_center

    # Convert the shifted points to the local cylindrical coordinate system
    local_x = np.dot(shifted_points, x_axis)
    local_y = np.dot(shifted_points, y_axis)
    local_z = np.dot(shifted_points, z_axis)

    # Compute the radial distances and polar angles
    radial_distances = np.sqrt(local_x**2 + local_y**2)
    polar_angles = np.arctan2(local_y, local_x)

    # Create the circular mask within the specified radius range and slice thickness
    circular_mask = np.zeros(data_shape, dtype=np.uint8)
    circular_mask_flat = (radial_distances >= r) & (radial_distances <= r + dr) & (np.abs(local_z) <= slice_thickness)
    circular_mask.flat[circular_mask_flat] = 1

    # Polar angles within the mask
    polar_angles_masked = np.zeros(data_shape, dtype=np.float32)
    polar_angles_masked.flat[circular_mask_flat] = polar_angles[circular_mask_flat]

    return circular_mask, polar_angles_masked
#------------------------------------------------------------------------------------------------------------
def create_circular_mask_with_angles_and_vectors(data_shape, centroid, direction, selected_point_index, r, dr, slice_thickness=2):
    """Create a circular mask and compute polar angles and displacement vectors from the disk center.
    
    Args:
        data_shape (tuple): Shape of the 3D data (e.g., (100, 100, 100)).
        centroid (np.array): Central point of the fitted line (e.g., np.array([50, 50, 50])).
        direction (np.array): Direction vector of the line (must be normalized).
        selected_point_index (float): Scalar to move along the direction vector from the centroid.
        r (float): Inner radius of the circular mask.
        dr (float): Thickness of the circular mask.
        slice_thickness (float): Thickness of the slice along the direction vector.
    
    Returns:
        circular_mask (np.ndarray): 3D mask with the circular region marked (1s for the mask, 0s elsewhere).
        polar_angles_masked (np.ndarray): 3D array with polar angles where the mask is applied.
        displacement_vectors (np.ndarray): 3D array storing vectors from disk center to each masked point.
    """
    selected_point_index = selected_point_index / 2  # Adjust the index scaling

    # Normalize the direction vector
    direction = direction / np.linalg.norm(direction)
    
    # Compute the disk center based on the selected point index along the direction
    disk_center = centroid + selected_point_index * direction

    # Define the local Z-axis (parallel to the direction vector)
    z_axis = direction

    # Define a random perpendicular vector to the Z-axis as the X-axis
    random_vector = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
    x_axis = np.cross(z_axis, random_vector)
    x_axis = x_axis / np.linalg.norm(x_axis)

    # Define the Y-axis as orthogonal to both Z and X
    y_axis = np.cross(z_axis, x_axis)

    # Generate a grid of all voxel indices
    grid_x, grid_y, grid_z = np.meshgrid(
        np.arange(data_shape[0]),
        np.arange(data_shape[1]),
        np.arange(data_shape[2]),
        indexing="ij",
    )
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    # Shift grid points relative to the disk center
    shifted_points = grid_points - disk_center

    # Convert the shifted points to the local cylindrical coordinate system
    local_x = np.dot(shifted_points, x_axis)
    local_y = np.dot(shifted_points, y_axis)
    local_z = np.dot(shifted_points, z_axis)

    # Compute the radial distances and polar angles
    radial_distances = np.sqrt(local_x**2 + local_y**2)
    polar_angles = np.arctan2(local_y, local_x)

    # Create the circular mask within the specified radius range and slice thickness
    circular_mask = np.zeros(data_shape, dtype=np.uint8)
    circular_mask_flat = (radial_distances >= r) & (radial_distances <= r + dr) & (np.abs(local_z) <= slice_thickness)
    circular_mask.flat[circular_mask_flat] = 1

    # Polar angles within the mask
    polar_angles_masked = np.zeros(data_shape, dtype=np.float32)
    polar_angles_masked.flat[circular_mask_flat] = polar_angles[circular_mask_flat]

    # Compute displacement vectors from disk center to masked points
    displacement_vectors = np.zeros((*data_shape, 3), dtype=np.float32)  # 3D vector field
    displacement_vectors_flat = shifted_points[circular_mask_flat]  # Select only masked points
    displacement_vectors.reshape(-1, 3)[circular_mask_flat] = displacement_vectors_flat  # Assign vectors

    return circular_mask, polar_angles_masked, displacement_vectors,direction
#------------------------------------------------------------------------------------------------------------
def plot_phase_around_dislo_one_radius_new    (scan,amp,phase,selected_dislocation_data,r,dr, centroid, direction,  slice_thickness=1,selected_point_index=0,save_vti=False,fig_title=None,plot_debug=True,
                                               save_path=None,voxel_sizes=(1,1,1)):
    # Create the circular mask and polar angle map
    # phase=np.unwrap(phase)
    circular_mask, polar_angles,displacement_vectors,direction = create_circular_mask_with_angles_and_vectors(selected_dislocation_data.shape,centroid, direction, selected_point_index, r, dr, slice_thickness=slice_thickness)
    masked_region_phase=phase*circular_mask
    
    if save_vti:
        vect_x = displacement_vectors[..., 0]
        vect_y = displacement_vectors[..., 1]
        vect_z = displacement_vectors[..., 2]


        # Save or visualize the circular mask and polar angles
        gu.save_to_vti(filename=save_path +scan+ "_circular_region_with_angles.vti",voxel_size=tuple(voxel_sizes),
                       tuple_array=(nan_to_zero(amp),nan_to_zero(phase),selected_dislocation_data,circular_mask,polar_angles,vect_x,vect_y,vect_z),
                       tuple_fieldnames=("density", "phase","dislo", "circular_mask", "polar_angles", "vect_x", "vect_y", "vect_z" ),amplitude_threshold=0.01,)
    if plot_debug:
        # Debug: Plot the circular mask
        #plot_3d_array(circular_mask);
        fig,ax = plt.subplots(1,1, figsize=(7,5))
        
        angle_ring_1 =masked_region_phase[masked_region_phase!=0].flatten() 
        phase_ring_1 =polar_angles[masked_region_phase!=0].flatten()
        
        sort_indices_1=np.argsort(angle_ring_1)
        angle_ring_1=angle_ring_1[sort_indices_1]
        phase_ring_1=phase_ring_1[sort_indices_1]
        phase_ring_1=np.unwrap(phase_ring_1,discont=pi/3)
        phase_ring_1=phase_offset_to_zero_clement(phase_ring_1)
        angle_ring_1,phase_ring_1=remove_large_jumps(angle_ring_1, phase_ring_1, threshold_factor=1.5)

        y= np.unwrap(phase_ring_1[30:-40])
        x= angle_ring_1[30:-40]*180/pi
        plt.plot(x,y,'o', label=f'r:%.1f' % r + rf" $\Delta\phi$ %.2f rad" % (-y.min() + y.max()))
        ax.legend(title='distance from\ndislocation center')
        #ax.legend()
        ax.set_xlabel('angle around dislocation (rad)', fontsize=15)
        ax.set_ylabel('phase (rad)', fontsize=15)
        
        if fig_title is not None:
            ax.set_title(fig_title, fontsize=20)
        fig.tight_layout()    
        if save_path is not None:
            plt.savefig(save_path +scan+ "_plot_angle_vs_phase_polar.png")
    return masked_region_phase,polar_angles,circular_mask,displacement_vectors,direction
#------------------------------------------------------------------------------------------------------------
def plot_compare_phase_radius_around_dislo_new(scan,amp,phase,selected_dislocation_data,      centroid, direction,  slice_thickness=1,selected_point_index=0,save_vti=False,fig_title=None,plot_debug=True,radius_min=2, radius_max=10, nb_radius=6,dr=1):
    radius_list = np.linspace(radius_min, radius_max, nb_radius)

    fig,ax = plt.subplots(1,1, figsize=(7,5))
    for radius in radius_list:
        phase_ring,angle_ring,circular_mask=plot_phase_around_dislo_one_radius_new(scan,amp,phase,selected_dislocation_data,radius,dr, centroid, direction,  slice_thickness=slice_thickness,
                                                                               selected_point_index=selected_point_index,save_vti=False,fig_title=None,plot_debug=False)
        y=phase_ring[phase_ring!=0].flatten() 
        x=angle_ring[phase_ring!=0].flatten()
        ax.plot(x,y,'o', label=f'r:{round(radius,1)}'+rf" $\Delta\phi$ {np.round(-y.min()+y.max(),2)}rad")
    ax.legend(title='distance from\ndislocation center')
    ax.set_xlabel('angle around dislocation (rad)', fontsize=15)
    ax.set_ylabel('phase (rad)', fontsize=15)
    
    if fig_title is not None:
        ax.set_title(fig_title, fontsize=20)
    
    fig.tight_layout()
    return
#------------------------------------------------------------------------------------------------------------
def remove_large_jumps(x, y, threshold_factor=1.5):
    """
    Removes points with large jumps in the y-data based on a threshold.

    Args:
        x (np.ndarray): The x-values of the data.
        y (np.ndarray): The y-values of the data.
        threshold_factor (float): The factor for the threshold to detect large jumps.

    Returns:
        x_clean (np.ndarray): The x-values with large jumps removed.
        y_clean (np.ndarray): The y-values with large jumps removed.
        dy (np.ndarray): The computed differences for each point.
    """
    # Compute differences index-by-index, including the first and last points
    dy = np.zeros(len(y))

    # For the first point, difference with the next point

    # For the intermediate points, take the max difference with neighbors
    for i in range(1, len(y) - 1):
        dy[i] = max(np.abs(y[i + 1] - y[i]), np.abs(y[i - 1] - y[i]))

    # For the last point, difference with the previous point
    dy[-1] = np.max([np.abs(y[-1] - y[i]) for i in range(-4,-1)])
    dy[0] = np.max([np.abs(y[0] - y[i]) for i in range(1,3)])

    # Define a threshold for identifying large jumps
    threshold = threshold_factor * np.std(dy)

    # Create a mask for valid points (where the jump is below the threshold)
    valid_mask = dy < threshold

    # Filter the data to remove points with large jumps
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]

    return x_clean, y_clean

def remove_jumps_dbscan_algo(x, y, eps=1.5, min_samples=5, change_point_n=2, jump_expand=3, gradient_percentile=90, final_gradient_threshold=85):
    """
    Removes jumps from data using:
    1. Change-Point Detection (ruptures)
    2. DBSCAN Clustering
    3. Gradient-Based Filtering
    4. Final Gradient-Based Cleanup

    Parameters:
        x (array-like): X-axis values.
        y (array-like): Y-axis values.
        eps (float): DBSCAN neighborhood radius.
        min_samples (int): Minimum samples for DBSCAN clustering.
        change_point_n (int): Number of breakpoints for change-point detection.
        jump_expand (int): Number of points to expand around detected jumps.
        gradient_percentile (float): Percentile threshold for gradient-based filtering.
        final_gradient_threshold (float): Final threshold to remove last jumps.

    Returns:
        final_x (np.array): X-values after jump removal.
        final_y (np.array): Y-values after jump removal.
        final_indices (np.array): Indices of the selected points in the original dataset.
    """

    # Convert to NumPy if y is a Pandas Series
    if hasattr(y, "to_numpy"):
        y = y.to_numpy()
    
    # Remove NaNs and keep valid indices
    valid_indices = ~np.isnan(y)
    x, y = x[valid_indices], y[valid_indices]
    original_indices = np.arange(len(y))[valid_indices]  # Track original indices

    ### --- STEP 1: CHANGE-POINT DETECTION --- ###
    algo = rpt.Binseg(model="l2").fit(y.reshape(-1, 1))
    breakpoints = algo.predict(n_bkps=change_point_n)[:-1]  # Exclude last index to prevent errors

    # Expand jump regions (±jump_expand points)
    expanded_jump_indices = np.hstack([np.arange(max(0, bp - jump_expand), min(len(x), bp + jump_expand)) for bp in breakpoints])

    # Remove change-point detected points
    mask_cpd = np.ones(len(x), dtype=bool)
    mask_cpd[expanded_jump_indices] = False

    filtered_x_cpd = x[mask_cpd]
    filtered_y_cpd = y[mask_cpd]
    filtered_indices_cpd = original_indices[mask_cpd]  # Track indices

    ### --- STEP 2: APPLY DBSCAN --- ###
    data_cpd = np.vstack((filtered_x_cpd, filtered_y_cpd)).T
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data_cpd)

    # Keep only points classified as part of a cluster
    mask_dbscan = labels != -1
    filtered_x_dbscan = filtered_x_cpd[mask_dbscan]
    filtered_y_dbscan = filtered_y_cpd[mask_dbscan]
    filtered_indices_dbscan = filtered_indices_cpd[mask_dbscan]  # Track indices

    ### --- STEP 3: INITIAL GRADIENT-BASED CLEANUP --- ###
    dy = np.abs(np.diff(filtered_y_dbscan))
    threshold = np.percentile(dy, gradient_percentile)
    jump_indices = np.where(dy > threshold)[0]

    # Remove detected jumps
    mask_gradient = np.ones(len(filtered_x_dbscan), dtype=bool)
    mask_gradient[jump_indices] = False

    cleaned_x = filtered_x_dbscan[mask_gradient]
    cleaned_y = filtered_y_dbscan[mask_gradient]
    cleaned_indices = filtered_indices_dbscan[mask_gradient]  # Track indices

    ### --- STEP 4: FINAL GRADIENT FILTER --- ###
    dy_final = np.abs(np.diff(cleaned_y))
    final_threshold = np.percentile(dy_final, final_gradient_threshold)
    final_jump_indices = np.where(dy_final > final_threshold)[0]

    # Remove last jumps
    mask_final = np.ones(len(cleaned_x), dtype=bool)
    mask_final[final_jump_indices] = False

    final_x = cleaned_x[mask_final]
    final_y = cleaned_y[mask_final]
    final_indices = cleaned_indices[mask_final]  # Final tracked indices

    return final_x, final_y, final_indices
#------------------------------------------------------------------------------------------------------------
def save_results_to_h5_dislo(data, file_path):
    """
    Save a hierarchical dictionary to an HDF5 file.
    
    Parameters:
        data (dict): The hierarchical dictionary to save.
        file_path (str): Path to save the HDF5 file.
    """
    with h5py.File(file_path, "w") as h5file:
        for data_type, scans in data.items():  # data_processed, data_raw, data_smooth
            group = h5file.create_group(data_type)
            for scan, arrays in scans.items():
                scan_group = group.create_group(scan)
                for key, array in arrays.items():  # angle and phase
                    scan_group.create_dataset(key, data=array)
#------------------------------------------------------------------------------------------------------------
def load_results_from_h5_dislo(file_path):
    """
    Load a hierarchical dictionary from an HDF5 file.
    
    Parameters:
        file_path (str): Path to the HDF5 file.

    Returns:
        dict: The loaded hierarchical dictionary.
    """
    data = {}
    with h5py.File(file_path, "r") as h5file:
        for data_type in h5file.keys():
            data[data_type] = {}
            for scan in h5file[data_type].keys():
                data[data_type][scan] = {
                    key: h5file[data_type][scan][key][:]
                    for key in h5file[data_type][scan].keys()
                }
    return data
#------------------------------------------------------------------------------------------------------------
# Flatten results_phase_experiment into a DataFrame
def results_to_dataframe(results):
    """
    Convert the hierarchical results dictionary to a pandas DataFrame.

    Parameters:
        results (dict): The hierarchical dictionary.

    Returns:
        pd.DataFrame: Flattened DataFrame with columns for type, scan, angle, and phase.
    """
    flattened_data = []
    for data_type, scans in results.items():  # e.g., "data_processed", "data_raw", "data_smooth"
        for scan, arrays in scans.items():
            angles = arrays["angle"]
            phases = arrays["phase"]
            for angle, phase in zip(angles, phases):
                flattened_data.append({"type": data_type, "scan": scan, "angle": angle, "phase": phase})
    return pd.DataFrame(flattened_data)
#------------------------------------------------------------------------------------------------------------
def plot_phase_vs_angle(data=None,title="",ylabel="Phase (rad)",save_filename=None,dpi=300,linewidth=2,markersize=3,marker="o",alpha=0.6,linestyle=None,figsize=(12, 8),
                        highlight_scan=None,show_annotations=True,scans_to_annotate=None,scans_to_plot=None,normalise_phase=1,font_size=12):
    """
    Enhanced Plot Phase vs Angle with unique markers, configurable line styles, and additional features.

    Parameters:
        data (pd.DataFrame): DataFrame containing columns 'angle', 'phase', and 'scan'.
        title (str): Title of the plot.
        ylabel (str): Y-axis label. Default is "Phase (rad)".
        save_filename (str): Path to save the resulting plot.
        dpi (int): Dots per inch for saving the plot. Default is 300.
        linewidth (int): Line width for the plot. Default is 2.
        markersize (int): Size of markers in the plot. Default is 3.
        marker (str or list): Marker style(s) for the plot. Can be a list for unique markers per scan.
        alpha (float): Transparency for the plot lines and markers. Default is 0.6.
        linestyle (str): Line style for the plot (e.g., "-", "--", ":"). Default is None (no line).
        figsize (tuple): Tuple specifying the figure size. Default is (12, 8).
        highlight_scan (str): Highlight a specific scan by making it bold.
        show_annotations (bool): Whether to annotate peaks and valleys. Default is True.
        scans_to_annotate (list): List of specific scans to annotate.
        scans_to_plot (list or None): List of scans to plot. If None, all scans are plotted. Default is None.
        normalise_phase (float): Factor to normalize phase. Default is 1.

    Returns:
        None
    """
    # Validate data input
    if data is None or data.empty:
        raise ValueError("The 'data' parameter must be a non-empty DataFrame.")

    required_columns = {"angle", "phase", "scan"}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"The 'data' DataFrame must contain the following columns: {required_columns}")

    # Drop NaN values
    data = data.dropna(subset=["angle", "phase"])

    # Normalize phase
    data["phase"] *= normalise_phase

    # Filter data to include only selected scans
    if scans_to_plot is not None:
        data = data[data["scan"].isin(scans_to_plot)]
    else:
        scans_to_plot =np.unique(data["scan"])

    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)

    # Improved color palette and markers
    palette = sns.color_palette("husl", n_colors=len(data["scan"].unique()))
    markers = ["o", "s", "D", "v", "^", "<", ">", "p", "h", "*", "X", "P"]
    marker_map = {scan: markers[i % len(markers)] for i, scan in enumerate(data["scan"].unique())}

    # Track max-min values for table
    max_min_table = []
    # Automatically calculate the number of columns for the legend
    n_labels = len(data["scan"].unique())  # Number of unique scans
    ncol = max(1, int(np.ceil(n_labels / 5)))  # Set columns to fit 5 labels per row (adjust the 5 as needed)
    plt.subplots_adjust(left=0.1, right=0.85)

    # Plot data with enhancements
    for scan in data["scan"].unique():
        subset = data[data["scan"] == scan]

        # Skip empty subsets
        if subset.empty or subset["phase"].isna().all():
            continue

        # Calculate max-min for table
        max_phase = subset["phase"].max()
        min_phase = subset["phase"].min()
        max_angle = subset["angle"].max()
        min_angle = subset["angle"].min()
        max_min_table.append([scan, f"{max_phase - min_phase:.2f}", f"{max_angle - min_angle:.2f}"])

        # Highlight specific scan
        current_marker = marker_map.get(scan, marker)
        sns.lineplot(
            data=subset, 
            x="angle", 
            y="phase", 
            label=scan, 
            linewidth=linewidth if linestyle else 0, 
            markersize=markersize, 
            alpha=alpha, 
            linestyle=linestyle, 
            marker=current_marker,
            ax=ax
        )

        # Add annotations for specific scans
        if show_annotations and (scans_to_annotate is None or scan in scans_to_annotate):
            try:
                max_idx = subset["phase"].idxmax()
                min_idx = subset["phase"].idxmin()
                
                # Use .loc for label-based access
                max_angle = subset.loc[max_idx, "angle"]
                max_phase = subset.loc[max_idx, "phase"]
                min_angle = subset.loc[min_idx, "angle"]
                min_phase = subset.loc[min_idx, "phase"]

                # Annotate maximum and minimum with offsets
                ax.annotate("Max", (max_angle, max_phase + 0.2), fontsize=font_size, color="black", ha="center")
                ax.annotate("Min", (min_angle, min_phase - 0.2), fontsize=font_size, color="black", ha="center")
            except Exception as e:
                print(f"Error annotating scan: {scan}, {e}")

    # Add table for max-min info
    column_labels = ["Scan", "ΔY", "ΔX"]
    table = ax.table(
        cellText=max_min_table,
        colLabels=column_labels,
        loc="bottom",
        cellLoc="center",
        bbox=[1.05, 0.05+0.05*(11-len(scans_to_plot)), 0.4, 0.5*len(scans_to_plot)/6]  # Increased space between table and plot
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    # Style the table
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        if row == 0:  # Header row
            cell.set_facecolor("darkblue")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:  # Alternating row colors
            cell.set_facecolor("#f2f2f2")  # Light grey for alternate rows

    # Customizations
    ax.set_xlabel("Angle (degrees)", fontsize=font_size*1.5)
    ax.set_ylabel(ylabel, fontsize=font_size*1.5)
    ax.set_title(title, fontsize=font_size*2)

    # Modify font size and style for x and y ticks
    ax.tick_params(axis='x', labelsize=font_size)  # Change font size for x-axis ticks
    ax.tick_params(axis='y', labelsize=font_size)  # Change font size for y-axis ticks
    
    # Optionally set font family for the tick labels
    for label in ax.get_xticklabels():
        label.set_fontname('DejaVu Sans')  # Replace 'Arial' with your desired font
    for label in ax.get_yticklabels():
        label.set_fontname('DejaVu Sans')  # Replace 'Arial' with your desired font

    # Improved legend placement with dark blue background
    legend = ax.legend(title="Scan", fontsize=font_size, loc=1, ncol=ncol,frameon=True)

    # Set legend styling
    legend.get_frame().set_facecolor("darkblue")
    legend.get_frame().set_edgecolor("black")
    legend.get_title().set_color("white")  # Set legend title color to white
    legend.get_title().set_fontsize(font_size)    # Adjust legend title font size
    for text in legend.get_texts():
        text.set_color("white")

    ax.grid(alpha=0.4)  # Add light grid for better readability
    plt.tight_layout()
    # Save the plot if a filename is provided
    if save_filename:
        plt.savefig(save_filename, dpi=dpi)
        print(f"Plot saved to {save_filename}")

    plt.show()
#------------------------------------------------------------------------------------------------------------
def filter_phase_2pi_period(phase,angle):
    angle_min=angle.min()
    angle_threshold=angle_min+2*pi
    return phase[angle<=angle_threshold],angle[angle<=angle_threshold]
#####################################################################################################################
def deconvolute_1d(x, y, method="FFT", wavelet="db4", lowpass_filter=False, impulse_response=None):
    """
    Deconvolutes a 1D signal y = f(x) using different techniques.
    
    Parameters:
        - x: ndarray (1D array of x-values)
        - y: ndarray (1D array of y-values)
        - method: str ("FFT", "Wavelet", "Derivative", "ICA", "Wiener") - Choose deconvolution method
        - wavelet: str (default "db4") - Wavelet type for decomposition
        - lowpass_filter: bool (default False) - Apply Savitzky-Golay smoothing
        - impulse_response: ndarray (optional) - If using Wiener deconvolution, provide impulse response.

    Returns:
        - deconvoluted_signal: ndarray (Processed y-values)
    """

    y = np.array(y)

    if lowpass_filter:
        # Apply Savitzky-Golay filter for noise reduction
        y = signal.savgol_filter(y, window_length=11, polyorder=2)

    if method == "FFT":
        # Apply Fast Fourier Transform (FFT) for deconvolution
        Y = fft.fft(y)
        freq = fft.fftfreq(len(x), d=(x[1] - x[0]))  # Frequency domain

        # Remove unwanted high-frequency components
        Y[np.abs(freq) > 0.1 * max(freq)] = 0  # Low-pass filtering
        deconvoluted_signal = fft.ifft(Y).real  # Reconstruct signal

    elif method == "Wavelet":
        # Apply Discrete Wavelet Transform (DWT)
        coeffs = pywt.wavedec(y, wavelet, level=3)
        coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]  # Remove high frequencies
        deconvoluted_signal = pywt.waverec(coeffs, wavelet)

    elif method == "Derivative":
        # First derivative to extract high-frequency components
        deconvoluted_signal = np.gradient(y, x)

    elif method == "ICA":
        # Independent Component Analysis (for blind source separation)
        ica = FastICA(n_components=1)
        deconvoluted_signal = ica.fit_transform(y.reshape(-1, 1)).flatten()

    elif method == "Wiener":
        if impulse_response is None:
            raise ValueError("Impulse response required for Wiener deconvolution")
        deconvoluted_signal, _ = signal.deconvolve(y, impulse_response)

    else:
        raise ValueError("Unsupported method. Choose from 'FFT', 'Wavelet', 'Derivative', 'ICA', or 'Wiener'.")

    return deconvoluted_signal
#------------------------------------------------------------------------------------------------------------
def filter_outliers(x, y, method="zscore", threshold=3, window=11, polyorder=2,eps_dbscan=0.2):
    """
    Filters unwanted outlier signals from (x, y) data using various techniques.

    Parameters:
        - x: ndarray (1D array of x-values)
        - y: ndarray (1D array of y-values)
        - method: str ("zscore", "iqr", "dbscan", "savgol", "polyfit") - Filtering method
        - threshold: float (Default=3) - Sensitivity for outlier detection
        - window: int (Default=11) - Window size for smoothing filters
        - polyorder: int (Default=2) - Polynomial order for smoothing

    Returns:
        - x_filtered, y_filtered: ndarrays (Cleaned dataset)
    """

    y = np.array(y)

    if method == "zscore":
        # Compute Z-score and filter outliers
        z_scores = np.abs(stats.zscore(y))
        mask = z_scores < threshold  # Keep only inliers

    elif method == "iqr":
        # Compute Interquartile Range (IQR) to filter outliers
        q25, q75 = np.percentile(y, [25, 75])
        iqr = q75 - q25
        lower, upper = q25 - threshold * iqr, q75 + threshold * iqr
        mask = (y > lower) & (y < upper)

    elif method == "dbscan":
        # Clustering method: Detect anomalies as outliers
        xy = np.column_stack((x, y))
        clustering = DBSCAN(eps=eps_dbscan, min_samples=5).fit(xy)
        mask = clustering.labels_ != -1  # Keep only clustered points

    elif method == "savgol":
        # Apply Savitzky-Golay smoothing filter
        y_smooth = signal.savgol_filter(y, window_length=window, polyorder=polyorder)
        residuals = np.abs(y - y_smooth)
        mask = residuals < np.std(residuals) * threshold  # Keep smooth values

    elif method == "polyfit":
        # Fit a polynomial to remove high-residual points
        p = np.polyfit(x, y, polyorder)
        y_fit = np.polyval(p, x)
        residuals = np.abs(y - y_fit)
        mask = residuals < np.std(residuals) * threshold  # Keep inliers

    else:
        raise ValueError("Unsupported method. Choose from 'zscore', 'iqr', 'dbscan', 'savgol', or 'polyfit'.")

    # Filter out outliers
    x_filtered, y_filtered = x[mask], y[mask]

    return x_filtered, y_filtered
#------------------------------------------------------------------------------------------------------------
def process_phase_ring_ortho_old(angle,phase,factor_phase=1):

    # Process and sort phase and angle data
    angle_ring             = factor_phase * phase[phase != 0].flatten()
    phase_ring             =                angle[phase != 0].flatten()
    sort_indices           = np.argsort(angle_ring)
    angle_ring             = angle_ring[sort_indices]
    phase_ring             = phase_ring[sort_indices]
    phase_ring             = np.angle(np.exp(1j*phase_ring))
    # Raw data
    phase_raw, angle_raw   = phase_ring.copy(), angle_ring.copy()
    # Unwrap, center, and clean phase data
    phase_ring             = np.unwrap(phase_ring, discont=np.pi / 3)
    phase_ring             = phase_offset_to_zero_clement(phase_ring)
    angle_ring, phase_ring = angle_ring[30:-40], phase_ring[30:-40]
    angle_ring, phase_ring = remove_large_jumps(angle_ring, phase_ring, threshold_factor=1.5)

    # Remove points with large jumps
    for _ in range(3):  # Repeat cleaning process multiple times
        trigger_new_period = np.array(np.where(abs(np.diff(phase_ring)) > 0.5))[0]
        trigger_new_period = np.concatenate((trigger_new_period, trigger_new_period + 1, trigger_new_period - 1))
        phase_ring = np.delete(phase_ring, trigger_new_period)
        angle_ring = np.delete(angle_ring, trigger_new_period)

    # Final cleanup
    angle_ring, phase_ring = remove_large_jumps(angle_ring, phase_ring, threshold_factor=1.5)
    phase_ring = center_angles(np.unwrap(phase_ring))
    angle_ring = center_angles(angle_ring)

    #phase_final,angle_final=filter_phase_2pi_period(phase_ring_1,angle_ring_1)

    angle_final = center_angles(angle_ring)
    phase_final = center_angles(phase_ring)
    angle_final *= 180 / np.pi
    
    trigger_angle_rotation=np.where((angle_final<180) & (angle_final>-180))
    angle_final=angle_final[trigger_angle_rotation]
    phase_final=phase_final[trigger_angle_rotation]
    
    
    phase_ring_1_smooth = savgol_filter(phase_final, window_length=50, polyorder=4)
    
    phase_final *= 180 / np.pi
    phase_ring_1_smooth *= 180 / np.pi
    phase_raw *= 180 / np.pi
    angle_raw *= 180 / np.pi
    return angle_raw,phase_raw , angle_final ,phase_final ,phase_ring_1_smooth
#------------------------------------------------------------------------------------------------------------
def dislo_rotation_matrix_real_to_theo(t, b):
    """
    Compute the rotation matrix from real space to the dislocation (theoretical) frame.
    """
    # 1) ẑ = t̂ = t / ||t||
    t_hat = normalize_vector(t)  # new z-axis

    # 2) b_perp = b - (b·t̂) t̂  (the component of b perpendicular to t)
    b_perp = project_vector(b, t)
    b_perp_norm = np.linalg.norm(b_perp)

    # 3) x̂ = b_perp / ||b_perp||  (edge direction) unless b_perp=0 => pick any perpendicular
    if b_perp_norm < 1e-10:
        # Choose an arbitrary x-axis perpendicular to t
        temp = np.array([1.0, 0.0, 0.0])
        x_prime = temp - np.dot(temp, t_hat) * t_hat
        x_prime = normalize_vector(x_prime)
    else:
        x_prime = b_perp/b_perp_norm

    # 4) ŷ = ẑ × x̂  (right-hand rule)
    y_prime = normalize_vector(np.cross(t_hat, x_prime))

    # 5) R has rows = [x̂, ŷ, ẑ]
    R = np.array([x_prime, y_prime, t_hat])
    return R
#------------------------------------------------------------------------------------------------------------
def dislo_transform_vector(v, t, b, to_theo=True):
    R = dislo_rotation_matrix_real_to_theo(t, b)
    return (R @ v ) if to_theo else (R.T @ v )
#------------------------------------------------------------------------------------------------------------
def dislo_displacement_field(x, y, t, b, nu=0.3, frame='real'):
    r = np.sqrt(x**2 + y**2) + 1e-12  # Avoid log(0)
    theta = np.arctan2(y, x)
    
    b_perp = project_vector(b, t)
    b_perp_norm = np.linalg.norm(b_perp)
    b_screw = np.dot(b, t) / np.linalg.norm(t)
    
    # Edge component displacements
    u_x_theo = (b_perp_norm / (2 * np.pi)) * (theta + np.sin(2 * theta) / (4 * (1 - nu)))
    u_y_theo = -(b_perp_norm / (8 * np.pi * (1 - nu))) * (2 * (1 - 2 * nu) * np.log(r) + np.cos(2 * theta))
    
    # Screw component displacement
    u_z_theo = (b_screw / (2 * np.pi)) * theta
    
    u_theo = np.array([u_x_theo, u_y_theo, u_z_theo])
    
    if frame == 'theo':
        return u_theo
    elif frame == 'real':
        R = dislo_rotation_matrix_real_to_theo(t, b)
        return np.dot(R.T, u_theo)
    else:
        raise ValueError("frame must be 'real' or 'theo'.")
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
# Optimize Burgers vector fitting in real space
def dislo_fit_burgers_vector(theta_data, phase_data, t, G, nu=0.3, d_hkl=1.0, bounds=None):
    def dislo_loss_predict_b(b, theta_data, phase_data, t, G, nu=0.3, d_hkl=1.0):
        theta_array = np.array(theta_data)
        pred_phase = np.vectorize(lambda theta: dislo_phase_model(theta, t, G, b, nu, d_hkl))(theta_array)
        return np.sum((pred_phase - phase_data) ** 2)
    b0 = normalize_vector(np.array([1.0, 1.0, 0.0]))
    res = minimize(dislo_loss_predict_b, b0, args=(theta_data, phase_data, t, G, nu, d_hkl), method='L-BFGS-B', bounds=bounds, tol=1e-6)
    return res.x
#------------------------------------------------------------------------------------------------------------
def plot_dislocation_phase_analysis(theta_data, phase_data, phase_theo_3_cases, b_cases,center_angles, title_suffix="",
                                    zoom_factor=2, zoom_bbox=(-0.15, 0.9),d_hkl=0.39239, num_ticks=5, save_path=None,
                                    font_family="Liberation Serif", font_size=12,type_data_to_comp="Experimental"):
    """
    Plots the experimental vs theoretical phase analysis for dislocations, including:
      - Experimental vs Theoretical phase with a zoomed-in inset.
      - Error (Difference) between Experimental & Theoretical phase.
      - Phase with linear part removed (oscillatory part).
      - Phase difference with the linear component removed.

    Parameters:
    -----------
    theta_data : array-like
        The polar angle values (in radians).
    phase_data : array-like
        Experimental phase data.
    phase_theo_3_cases : list of array-like
        Theoretical phase data for different cases.
    b_cases : list of array-like
        Burgers vectors corresponding to each theoretical case.
    center_angles : function
        A function to center phase angles (used for wrapping).
    title_suffix : str, optional
        A suffix to append to plot titles for differentiation.
    zoom_factor : float, optional
        The zoom level for the inset (default: 2).
    zoom_bbox : tuple, optional
        The bbox position of the zoom inset (default: (-0.15, 0.9)).
    d_hkl : float, optional
        Interplanar spacing (default: 0.39239 Å).
    num_ticks : int, optional
        Number of ticks on each axis for better readability (default: 5).
    save_path : str, optional
        Path to save the figure (default: None, which means not saving).
    font_family : str, optional
        Font family for titles, axis labels, ticks, and legends (default: "Arial").
    font_size : int, optional
        Font size for all text elements in the plot (default: 12).

    Returns:
    --------
    None (Displays the plot).
    """

    # Set global font properties
    plt.rcParams.update({
        "font.family": font_family,
        "font.size": font_size
    })

    num_cases = len(b_cases)
    colors = matplotlib.cm.viridis(np.linspace(0, 1, num_cases))  # Use Viridis colormap for better contrast

    # Compute linear fits for both experimental and theoretical data
    coefficients_theo = [np.polyfit(theta_data, phase_theo_3_cases[i], 1) for i in range(num_cases)]
    coef_exp = np.polyfit(theta_data, phase_data, 1)

    # Compute oscillatory parts by removing linear trend
    phase_diff_oscillation = np.array([
        center_angles(phase_theo_3_cases[i] - (coefficients_theo[i][0] * theta_data + coefficients_theo[i][1]))
        for i in range(num_cases)
    ])
    phase_data_oscillation = center_angles(phase_data - (coef_exp[0] * theta_data + coef_exp[1]))
    phase_diff_oscillation_exp_based = np.array([
        center_angles(phase_theo_3_cases[i] - (coef_exp[0] * theta_data + coef_exp[1]))
        for i in range(num_cases)
    ])

    # Compute phase difference (error)
    phase_diff = np.array([center_angles(phase_theo_3_cases[i] - phase_data) for i in range(num_cases)])

    # Define zoomed-in region
    theta_min = np.min(theta_data)
    theta_max = theta_min + np.pi / 2
    relevant_indices = (theta_data >= theta_min) & (theta_data <= theta_max)
    local_phase_data = phase_data[relevant_indices]
    local_phase_theo = np.concatenate([phase_theo_3_cases[i][relevant_indices] for i in range(num_cases)])
    y_min = min(local_phase_data.min(), local_phase_theo.min())
    y_max = max(local_phase_data.max(), local_phase_theo.max())

    # Create a 2x2 subplot layout
    fig, axs = plt.subplots(2, 2, figsize=(18, 18), sharex=True)

    # --- Subplot 1: Experimental vs. Theoretical Phase ---
    ax0 = axs[0, 0]
    ax0.scatter(theta_data, phase_data, label='Exp', s=20, alpha=0.7,
                color='black', edgecolors='white', zorder=3)
    for i_b in range(num_cases):
        label = ''.join(map(str, (b_cases[i_b] /np.nanmin(zero_to_nan(abs(b_cases[i_b])))).astype(int)))
        ax0.plot(theta_data, phase_theo_3_cases[i_b], "-", label=f'Theory {label}', linewidth=2*num_cases-i_b*2, color=colors[i_b])

    ax0.set_ylabel(r"$\phi$")
    ax0.set_title(f"{type_data_to_comp} vs. Theoretical $\phi$ {title_suffix}")
    ax0.legend(fontsize=font_size-2, loc='best', frameon=False, markerscale=1.2)
    ax0.grid(True, linestyle='dotted', alpha=0.5)

    # Add zoomed inset to subplot 1
    axins = zoomed_inset_axes(ax0, zoom=zoom_factor, bbox_to_anchor=zoom_bbox, bbox_transform=ax0.transAxes)
    axins.scatter(theta_data, phase_data, s=20, alpha=0.7, color='black', edgecolors='white', zorder=3)
    for i_b in range(num_cases):        axins.plot(theta_data, phase_theo_3_cases[i_b], "-", linewidth=2*num_cases-i_b*2, color=colors[i_b])

    axins.set_xlim(theta_min, theta_max);    axins.set_ylim(y_min, y_max);    mark_inset(ax0, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # --- Subplot 2: Difference (Error) Between Experimental & Theoretical ---
    ax1 = axs[0, 1]
    for i_b in range(num_cases):    
        label = ''.join(map(str, (b_cases[i_b] /np.nanmin(zero_to_nan(abs(b_cases[i_b])))).astype(int)))
        ax1.plot(theta_data, phase_diff[i_b], "--", label=f'Error {label}', linewidth=2*num_cases-i_b*2, color=colors[i_b])

    ax1.set_ylabel(r"$\phi$ Difference (Error)");    ax1.set_title(f"$\phi$ Difference: Theory - {type_data_to_comp} {title_suffix}")
    ax1.axhline(0, color="gray", linestyle="dotted", linewidth=1.2)
    ax1.legend(fontsize=font_size-2, loc='best', frameon=False);    ax1.grid(True, linestyle='dotted', alpha=0.5)

    # --- Subplot 3: Phase - Linear Trend of Experimental Data ---
    ax2 = axs[1, 0]
    ax2.scatter(theta_data, phase_data_oscillation, label='Exp', s=20, alpha=0.7,color='black', edgecolors='white', zorder=3)
    for i_b in range(num_cases):        
        label = ''.join(map(str, (b_cases[i_b] /np.nanmin(zero_to_nan(abs(b_cases[i_b])))).astype(int)))
        ax2.plot(theta_data, phase_diff_oscillation_exp_based[i_b], "-", label=f'Theory {label}', linewidth=2*num_cases-i_b*2, color=colors[i_b])

    ax2.set_xlabel(r"$\theta$");    ax2.set_ylabel(r"$\phi - \alpha_{ref}\theta - \beta_{ref}$")
    ax2.set_title(f"$\phi -$ Linear Part of ref Data {title_suffix}")
    ax2.legend(fontsize=font_size-2, loc='best', frameon=False);    ax2.grid(True, linestyle='dotted', alpha=0.5)

    # --- Subplot 4: Phase - Linear Trend for Each Theoretical Case ---
    ax3 = axs[1, 1]
    ax3.scatter(theta_data, phase_data_oscillation, label='Exp', s=20, alpha=0.7,color='black', edgecolors='white', zorder=3)
    for i_b in range(num_cases):     
        label = ''.join(map(str, (b_cases[i_b] /np.nanmin(zero_to_nan(abs(b_cases[i_b])))).astype(int)))
        ax3.plot(theta_data, phase_diff_oscillation[i_b], "-", label=f'Theory {label}', linewidth=2*num_cases-i_b*2, color=colors[i_b])

    ax3.set_xlabel(r"$\theta$");    ax3.set_ylabel(r"$\phi - \alpha\theta - \beta$")
    ax3.set_title(f"$\phi -$ Linear Part of Each Case {title_suffix}")
    ax3.legend(fontsize=font_size-2, loc='best', frameon=False);    ax3.grid(True, linestyle='dotted', alpha=0.5)

    #plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()



#------------------------------------------------------------------------------------------------------------


def smooth_interpolate_common_x(x1, y1, x_common, sigma=2):
    """
    Interpolates and smooths y1 values onto a common x-axis using PCHIP interpolation
    and adaptive Gaussian smoothing.

    Parameters:
    -----------
    x1 : array-like
        Original x-values for y1.
    y1 : array-like
        Corresponding y-values to be interpolated.
    x_common : array-like
        Common x-values to interpolate onto.
    sigma : float, optional
        Standard deviation for Gaussian smoothing (default: 2).

    Returns:
    --------
    x_common : array
        The common x-axis values.
    y_smooth : array
        The smoothed and interpolated y-values.
    """

    # Ensure data is sorted
    sorted_indices = np.argsort(x1)
    x1_sorted = np.array(x1)[sorted_indices]
    y1_sorted = np.array(y1)[sorted_indices]

    # Use PCHIP for interpolation (monotonic cubic interpolation)
    interp_func = PchipInterpolator(x1_sorted, y1_sorted)

    # Interpolate to common x values
    y_interp = interp_func(x_common)

    # Apply Gaussian smoothing
    y_smooth = gaussian_filter1d(y_interp, sigma=sigma)

    return x_common, y_smooth
#------------------------------------------------------------------------------------------------------------
def remove_large_jumps_alter_unwrap(y, threshold=10):
    """
    Detects and removes large jumps in y based on a given threshold.
    
    Parameters:
        y (numpy array): Dependent variable (e.g., phase or measured value).
        threshold (float): Threshold for detecting large jumps.
        
    Returns:
        numpy array: Corrected y values.
    """
    y_fixed = y.copy()
    y_diff = np.diff(y)
    if np.abs(y_diff).max()<threshold:
        return y
    else:
        jumps = np.where(np.abs(y_diff) > threshold)[0]
        for j in jumps:
            y_fixed[j + 1:] -= y_diff[j]  # Shift the remaining data to remove jump
        
        return y_fixed
#------------------------------------------------------------------------------------------------------------
def dislo_process_phase_ring_ortho(angle, phase, displacement_vectors, factor_phase=1, poly_order=1, jump_filter_ML=True, jump_filter_gradient_only=False,gradient_percentile=90, final_gradient_threshold=90,
                                   plot_debug=False, save_path=None,period_jump=360):
    
    """
    Processes the phase and angle data to analyze dislocation properties in a phase ring.

    This function:
    1. Extracts and filters nonzero phase values.
    2. Sorts the phase and angle data.
    3. Removes phase jumps and outliers using an adaptive filtering method.
    4. Unwraps and centers the phase data to ensure phase continuity.
    5. Applies a Savitzky-Golay filter to smooth phase variations.
    6. Removes polynomial trends dynamically from the phase data.
    7. Tracks filtered data indices and visualizes the selection.
    8. Visualizes displacement vectors alongside phase data.

    Args:
        angle (np.ndarray): The angle data.
        phase (np.ndarray): The phase data.
        displacement_vectors (np.ndarray): The displacement vectors associated with the phase data.
        factor_phase (float, optional): Scaling factor applied to phase data. Defaults to 1.
        poly_order (int, optional): Order of polynomial fit for trend removal. Defaults to 1 (linear).
        jump_filter (bool, optional): If True, applies phase jump removal and outlier filtering.
        gradient_percentile (int, optional): Percentile threshold for filtering sharp gradients. Defaults to 90.
        final_gradient_threshold (int, optional): Threshold for final gradient-based outlier removal. Defaults to 90.
        plot_debug (bool, optional): If True, generates detailed debugging plots.
        save_path (str, optional): Path to save the debug plots.

    Returns:
        tuple: A tuple containing:
            - angle_raw (np.ndarray): Original angle data (before processing).
            - phase_raw (np.ndarray): Original phase data (before processing).
            - angle_final (np.ndarray): Processed angle data after jump removal.
            - phase_final (np.ndarray): Processed phase data after unwrapping and centering.
            - phase_ring_1_smooth (np.ndarray): Smoothed phase data.
            - phase_sinu (np.ndarray): Sinusoidal phase deviation after polynomial trend removal.
            - displacement_vectors_ring_sorted (np.ndarray): Sorted displacement vectors before filtering.
            - displacement_vectors_final (np.ndarray): Displacement vectors after filtering.
            - sel___ (np.ndarray): Boolean mask indicating selected (kept) data points.
    """
    def filter_phase_data(angle_ring, phase_ring, adaptive_threshold_factor=2.0, median_filter_sizes=(3, 7), zscore_threshold=2.8):
        """
        Filters phase data by removing large phase jumps, applying an adaptive median filter, 
        and filtering out statistical outliers. Also tracks the selected indices.
    
        Parameters:
        - angle_ring (numpy array): Angle values in degrees.
        - phase_ring (numpy array): Phase values in degrees.
        - adaptive_threshold_factor (float): Factor for detecting large jumps based on standard deviation.
        - median_filter_sizes (tuple): (small, large) filter sizes for adaptive filtering.
        - zscore_threshold (float): Threshold for filtering out extreme outliers.
    
        Returns:
        - angle_filtered (numpy array): Filtered angle values.
        - phase_filtered (numpy array): Filtered phase values.
        - selected_indices (numpy array): Indices of the selected data points in the original array.
        """
    
        original_indices = np.arange(len(angle_ring))  # Track original indices
    
        # Step 1: Identify Large Phase Jumps
        diff_phi = np.abs(np.diff(phase_ring, append=phase_ring[-1]))
        threshold_jump = np.median(diff_phi) + adaptive_threshold_factor * np.std(diff_phi)
    
        # Identify large jumps
        large_jump_indices = np.where(diff_phi > threshold_jump)[0]
    
        if len(large_jump_indices) > 0:
            # Correct only the largest discontinuity
            diff_phi_positionmax = np.argmax(diff_phi)
            phase_shift = phase_ring[diff_phi_positionmax] - phase_ring[diff_phi_positionmax - 1]
            phase_ring[diff_phi_positionmax:] -= phase_shift  # Adjust phase after the jump
    
        # Step 2: Apply Adaptive Median Filter
        phase_ring_smoothed = median_filter(phase_ring, size=median_filter_sizes[0])
    
        # Apply larger filtering only where large jumps occur
        for idx in large_jump_indices:
            if idx > 2 and idx < len(phase_ring) - 2:
                phase_ring_smoothed[idx] = np.median(phase_ring[idx-2:idx+3])
    
        # Step 3: Use an Adaptive Threshold for Filtering
        diff_phi = np.abs(np.diff(phase_ring_smoothed, append=phase_ring_smoothed[-1]))
        adaptive_threshold = np.median(diff_phi) + adaptive_threshold_factor * np.std(diff_phi)
        FILTER_DIFF_ = diff_phi < adaptive_threshold
    
        # Apply filtering
        angle_filtered, phase_filtered, selected_indices = angle_ring[FILTER_DIFF_], phase_ring_smoothed[FILTER_DIFF_], original_indices[FILTER_DIFF_]
    
        # Step 4: Final Cleanup with Z-Score Filtering
        z_scores = np.abs(zscore(phase_filtered))
        final_selection = z_scores < zscore_threshold  # Final mask after Z-score filtering
    
        return angle_filtered[final_selection], phase_filtered[final_selection], selected_indices[final_selection]

    # Extract indices where phase is nonzero
    nonzero_indices = np.nonzero(phase)
    displacement_vectors_ring = displacement_vectors[nonzero_indices]
    angle_ring =                angle[nonzero_indices].flatten()
    phase_ring = factor_phase * phase[nonzero_indices].flatten()

    # Sort by angle
    sort_indices = np.argsort(angle_ring)
    angle_ring = angle_ring[sort_indices]
    phase_ring = phase_ring[sort_indices]
    displacement_vectors_ring_sorted = displacement_vectors_ring[sort_indices]

    # Convert phase to degrees
    phase_ring = phase_ring * (180 / np.pi)
    angle_ring *= 180 / np.pi

    # Store raw data
    phase_raw, angle_raw = phase_ring.copy(), angle_ring.copy()
    if jump_filter_ML:
        
        # Remove jumps and outliers using DBSCAN-based method
        #angle_ring, phase_ring, filtered_indices = remove_jumps_dbscan_algo(
        #    angle_ring, phase_ring, eps=1.5, min_samples=2, change_point_n=2, 
        #    jump_expand=3, gradient_percentile=gradient_percentile, final_gradient_threshold=final_gradient_threshold
        #)
        # Select displacement vectors corresponding to filtered indices
        sel___ = np.zeros_like(angle_ring, dtype=bool)
        angle_ring, phase_ring, filtered_indices= filter_phase_data(angle_ring, phase_ring)
        displacement_vectors_final = displacement_vectors_ring_sorted[filtered_indices]
        # Create a mask for selected (kept) points
        sel___[filtered_indices] = True  # Mark selected indices as True
    elif jump_filter_gradient_only:
        phase_ring=remove_large_jumps_alter_unwrap(phase_ring)
        displacement_vectors_final=displacement_vectors_ring_sorted.copy()
    else:
        displacement_vectors_final=displacement_vectors_ring_sorted.copy()

    phase_final = np.unwrap(phase_ring, period=period_jump)
    phase_final = np.unwrap(phase_final, period=period_jump)
    print("Raw angle :", np.min(angle_raw), np.max(angle_raw))
    print("Raw phase :", np.min(phase_raw), np.max(phase_raw))
    
    print("unwrapped phase :", np.min(phase_final), np.max(phase_final))
    
    phase_final = center_angles(phase_final+angle_ring)-angle_ring


    # Apply Savitzky-Golay filter
    window_length = min(100, len(phase_final) - 1)
    if window_length % 2 == 0:
        window_length -= 1
    phase_ring_1_smooth = center_angles(savgol_filter(phase_final, window_length=window_length, polyorder=min(poly_order, window_length-1)))

    # Remove polynomial trend
    poly_coeffs = np.polyfit(angle_ring, phase_final, poly_order)
    poly_fit = np.polyval(poly_coeffs, angle_ring)
    phase_sinu = center_angles(phase_final - poly_fit)

    ### --- Debug Plotting --- ###
    if plot_debug:
        fig, axes = plt.subplots(6, 1, figsize=(12, 18), sharex=True)

        axes[0].plot(angle_raw, phase_raw, ">", label="Raw Phase", color="black", alpha=0.7, linewidth=3)
        if jump_filter_ML:
            axes[0].plot(angle_raw[~sel___], phase_raw[~sel___], ".", label="Filtered Out", color="red", alpha=0.7, linewidth=3,markersize=10)
        
        axes[0].set_title("Raw Phase Data")
        axes[0].legend()

        axes[1].plot(angle_ring, phase_final, ">", label="Processed Phase", color="blue", alpha=0.7, linewidth=3)
        axes[1].set_title("Processed Phase (Unwrapped & Centered)")
        axes[1].legend()

        axes[2].plot(angle_ring, phase_ring_1_smooth, ">", label="Smoothed Phase", color="red", alpha=0.7, linewidth=3)
        axes[2].set_title("Smoothed Phase (Savitzky-Golay)")
        axes[2].legend()

        axes[3].plot(angle_ring, phase_sinu, ">", label="Phase Sinusoidal Deviation", color="green", alpha=0.7, linewidth=3)
        poly_eq_str = " + ".join([f"{coef:.2f} $\\theta^{i}$" for i, coef in enumerate(poly_coeffs[::-1])])
        axes[3].set_title(f"Phase Sinusoidal Deviation (Trend Removed: {poly_eq_str})")
        axes[3].legend()

        # Overlay all plots
        axes[4].plot(angle_raw , phase_raw          ,">-", label="Raw Phase", color="black", alpha=0.5, linewidth=2)
        axes[4].plot(angle_ring, phase_final        ,">-", label="Processed Phase", color="blue", alpha=0.6, linewidth=3)
        axes[4].plot(angle_ring, phase_ring_1_smooth,">-", label="Smoothed Phase", color="red", alpha=0.7, linewidth=4)
        axes[4].set_title("All Phase Data Overlaid")
        axes[4].legend()

        # **NEW PLOT: Displacement Vectors as Quiver**
        displacement_magnitudes = np.linalg.norm(displacement_vectors_final, axis=1)
        axes[5].plot(angle_ring, displacement_vectors_final[...,0], ">", label="Displacement Vector X", alpha=0.7, linewidth=3)
        axes[5].plot(angle_ring, displacement_vectors_final[...,1], "<", label="Displacement Vector Y", alpha=0.7, linewidth=3)
        axes[5].plot(angle_ring, displacement_vectors_final[...,2], "^", label="Displacement Vector Z", alpha=0.7, linewidth=3)
        
        axes[5].set_title("Displacement Vector Magnitudes vs. Angle")
        axes[5].set_ylabel("Vector Magnitude")
        axes[5].set_xlabel("Angle (Degrees)")
        axes[5].legend()

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()

    return (angle_raw, phase_raw, angle_ring, phase_final, phase_ring_1_smooth, 
            phase_sinu, displacement_vectors_ring_sorted, displacement_vectors_final)










#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
# Optimize Burgers vector fitting in real space
def dislo_fit_to_predict_angle_ref(theta_data, phase_data, t, G, b, nu=0.3, d_hkl=0.39239,
                                   grid_points=1000,  # Number of points to scan between -π and π
                                   refine=True,       # Whether to fine-tune using optimization
                                   maxiter=5000,      # Maximum iterations for optimizer
                                   disp=False,        # Display optimization steps
                                   ftol=1e-15,        # Function tolerance for convergence
                                   gtol=1e-15,        # Gradient tolerance
                                   eps=1e-20,         # Small finite difference step
                                   maxls=200,# Maximum line search steps
                                   bounds = [(-np.pi, np.pi)]  # Keep the phase shift within -π to π
                                  ):        
    
    """
    Optimize the phase offset (delta_theta) for dislocation phase shift analysis.
    
    Instead of random initialization, this function:
      1. Evaluates the loss function at `grid_points` equally spaced points in [-π, π].
      2. Selects the value that minimizes the loss.
      3. (Optional) Fine-tunes the result using L-BFGS-B optimization.

    Parameters:
    -----------
    theta_data : array-like
        The polar angle values (radians).
    phase_data : array-like
        Experimental phase values corresponding to theta_data.
    t : array-like (3,)
        The dislocation line direction vector.
    G : array-like (3,)
        The reciprocal lattice vector.
    b : array-like (3,)
        The Burgers vector of the dislocation.
    nu : float, optional
        Poisson's ratio of the material (default: 0.3).
    d_hkl : float, optional
        Interplanar spacing in Ångströms (default: 0.39239 Å).
    grid_points : int, optional
        Number of points to sample between -π and π for initial search (default: 1000).
    refine : bool, optional
        If True, fine-tune the result using L-BFGS-B optimization (default: True).
    maxiter, disp, ftol, gtol, eps, maxls : Various optimizer settings.

    Returns:
    --------
    delta_theta_opt : float
        The optimized phase offset (delta_theta) in radians.
    """
    # Function to optimize delta_theta
    def dislo_loss_diff_angle_bdir(delta_theta, theta_data, phase_data, t, G, b, nu=0.3):
        theta_shifted = np.array(theta_data) + delta_theta  # Shifted angles
        predicted_phase = dislo_phase_model(theta_shifted, t, G, b, nu=nu)
        return np.mean((predicted_phase - phase_data) ** 2)  # Mean squared error

    # Generate grid of delta_theta values from -π to π
    delta_theta_grid = np.linspace(-np.pi, np.pi, grid_points)
    
    # Evaluate the loss function at each grid point
    loss_values = np.array([dislo_loss_diff_angle_bdir(dt, theta_data, phase_data, t, G, b, nu, d_hkl) 
                            for dt in delta_theta_grid])
    
    # Find the best initial guess with the lowest loss
    best_idx = np.argmin(loss_values)
    delta_theta_best = delta_theta_grid[best_idx]
    
    print("\n--- Grid Search Results ---")
    print(f"Best Initial Guess (delta_theta): {delta_theta_best:.6f} rad")
    print(f"Loss at Best Initial Guess: {loss_values[best_idx]:.6e}")

    if not refine:
        return delta_theta_best  # Return without refinement

    # Optimization step using L-BFGS-B with refined starting point
    
    options = {
        "maxiter": maxiter,
        "disp": disp,
        "ftol": ftol,
        "gtol": gtol,
        "eps": eps,
        "maxls": maxls
    }

    res = minimize(dislo_loss_diff_angle_bdir, [delta_theta_best], 
                   args=(theta_data, phase_data, t, G, b, nu, d_hkl),
                   method='L-BFGS-B', bounds=bounds, options=options)

    # Extract optimized phase offset (delta_theta)
    delta_theta_opt = res.x[0]
    final_loss = res.fun  # The final loss value after optimization

    # Normalize the Burgers vector for consistency
    b_normalized = b / np.linalg.norm(b)

    print("\n--- Final Optimization Results ---")
    print(f"Estimated Burgers vector (normalized): {b_normalized}")
    print(f"Optimal Offset (delta_theta): {delta_theta_opt:.6f} rad")
    print(f"Final Loss Value: {final_loss:.6e}")
    print(f"Optimization Successful: {res.success}")
    print(f"Stopping Reason: {res.message}")
    print("----------------------------\n")

    return delta_theta_opt










def process_phase_mode_ring(angle,phase,loop_cleaning_diff=6):
    # Process and sort phase and angle data
    angle_ring = phase[phase != 0].flatten()
    phase_ring = angle[angle != 0].flatten()

    sort_indices = np.argsort(angle_ring)
    angle_ring = angle_ring[sort_indices]
    phase_ring = phase_ring[sort_indices]
    phase_ring = np.angle(np.exp(1j*phase_ring))

    # Raw data
    phase_raw, angle_raw = phase_ring.copy(), angle_ring.copy()
    #phase_raw = savgol_filter(phase_raw, window_length=50, polyorder=4)
    

    # Unwrap, center, and clean phase data
    phase_ring = np.unwrap(phase_ring, discont=np.pi / 3)
    phase_ring = phase_offset_to_zero_clement(phase_ring)
    angle_ring, phase_ring = angle_ring[10:-10], phase_ring[10:-10]
    angle_ring, phase_ring = remove_large_jumps(angle_ring, phase_ring, threshold_factor=1.5)

    # Remove points with large jumps
    for _ in range(loop_cleaning_diff):  # Repeat cleaning process multiple times
        trigger_new_period = np.array(np.where(abs(np.diff(phase_ring)) > 0.5))[0]
        trigger_new_period = np.concatenate((trigger_new_period, trigger_new_period + 1, trigger_new_period - 1))
        phase_ring = np.delete(phase_ring, trigger_new_period)
        angle_ring = np.delete(angle_ring, trigger_new_period)

    # Final cleanup
    angle_ring, phase_ring = remove_large_jumps(angle_ring, phase_ring, threshold_factor=1.5)
    phase_ring = center_angles(np.unwrap(phase_ring))
    angle_ring = center_angles(angle_ring)

    if phase_ring[np.where(angle_ring== angle_ring.min())[0]][0]!=phase_ring.max():
        factor_phase=-1
    else:
        factor_phase=-1
    phase_ring*=factor_phase    

    angle_final = center_angles(angle_ring)
    phase_final = center_angles(phase_ring)
    
    phase_final_smooth = savgol_filter(phase_final, window_length=50, polyorder=4)
    phase_raw *= 180 / np.pi
    angle_raw *= 180 / np.pi
    angle_final *= 180 / np.pi
    phase_final *= 180 / np.pi
    phase_final_smooth *= 180 / np.pi   

    return angle_raw,phase_raw , angle_final ,phase_final , phase_final_smooth

    


#####################################################################################################################
#############################################testing   ##############################################################

# Cost function to minimize
def cost_function_0(params):
    # Unpack parameters
    b_x, b_y, b_z, A, C, B1, B2, k = params
    
    # Generate theoretical data
    x_predict, y_predict = get_predicted_theo_u(b_x=b_x, b_y=b_y, b_z=b_z, A=A, C=C, B1=B1, B2=B2, k=k)
    
    # Interpolate theoretical data to experimental x-values
    interp_theo = interp1d(x_predict, y_predict, kind='linear', bounds_error=False, fill_value="extrapolate")
    y_theo_interpolated = interp_theo(xexp_noisy)
    
    # Compute the difference
    diff = yexp_noisy - y_theo_interpolated
    
    # Return the mean squared error as the cost
    mse = np.mean(diff**2)
    return mse
#------------------------------------------------------------------------------------------------------------
def cost_function_1(params):
    # Unpack parameters
    b_x, b_y, b_z, C, B1, B2, k = params
    
    # Generate theoretical data
    x_predict, y_predict = get_predicted_theo_u(b_x=b_x, b_y=b_y, b_z=b_z, C=C, B1=B1, B2=B2, k=k)
    
    # Interpolate theoretical data to experimental x-values
    interp_theo = interp1d(x_predict, y_predict, kind='linear', bounds_error=False, fill_value="extrapolate")
    y_theo_interpolated = interp_theo(xexp_noisy)
    
    # Compute the difference
    diff = yexp_noisy - y_theo_interpolated
    
    # Return the mean squared error as the cost
    mse = np.mean(diff**2)
    return mse
#------------------------------------------------------------------------------------------------------------
def progress_callback(xk, convergence):
    # Elapsed time so far
    elapsed_time = time.time() - start_time
    # Number of generations completed
    generation = len(xk)  # Each call represents a generation
    # Calculate time per generation (based on elapsed time)
    time_per_generation = elapsed_time / generation
    # Estimate total time for optimization
    max_time_estimate = (time_per_generation * maxiter) / n_cpus
    # Estimate remaining time
    remaining_time = max_time_estimate - elapsed_time
    mse = cost_function(xk)
    generation_mse.append(mse)
    # Print progress, updating in the same line
    print(f"\rGeneration: {generation}, Elapsed Time: {elapsed_time:.2f}s, " f"Estimated Max Time: {max_time_estimate:.2f}s, " f"Estimated Remaining Time: {remaining_time:.2f}s",end="",flush=True,)
#------------------------------------------------------------------------------------------------------------
def fun_oscillation_part(B1,B2,k,x):
    periodic_adjustment = B1 * np.sin(k * x) + B2 * np.cos(k * x)
    return periodic_adjustment
#------------------------------------------------------------------------------------------------------------
def apply_noisy_sigma_todata(x,y,sigma_perc=0.5,overfactor_x=5,plot=True)  :
    # Calculate the standard deviation of the original experimental data
    sigma = np.std(y)
    # Define the noise level as 0.2 times the standard deviation
    error_std = sigma_perc* sigma/100
    # Generate a dense x_noisy with more points
    x_noisy = np.linspace(x.min(), x.max(), len(x) * overfactor_x)
    # Interpolate the original experimental data onto the dense x_noisy
    interp_func = interp1d(x, y, kind='linear', bounds_error=False, fill_value="extrapolate")
    y_noisy_interpolated = interp_func(x_noisy)
    
    # Add random noise to the interpolated data
    random_noise = np.random.normal(loc=0.0, scale=error_std, size=x_noisy.shape)
    # Apply a moving average to smooth the noisy data
    y_noisy = y_noisy_interpolated + random_noise
    if plot: 
        # Define the band
        upper_bound = y_noisy_interpolated + error_std
        lower_bound = y_noisy_interpolated - error_std
        # Plot the results
        plt.figure(figsize=(10, 6))
        # Original experimental data
        plt.plot(x, y, label="Original Data", linestyle='--')
        
        # Noisy data
        plt.plot(x_noisy, y_noisy, label="Noisy Data (More Points)", alpha=0.8)
        # Band
        plt.fill_between(x_noisy,lower_bound,upper_bound,color="lightgray",alpha=0.5,label="0.2σ Band")
        
        # Add labels and legend
        plt.xlabel("$\\theta$")
        plt.ylabel("Displacement")
        plt.title("Experimental Data with Dense Noisy Data and 0.2σ Band")
        plt.legend()
        plt.show()
        
    return x_noisy,y_noisy
#------------------------------------------------------------------------------------------------------------
def get_predicted_theo_u(b_x=1.0,b_y=1.0,b_z=1.0, A=1.0, C=0.0, B1=0.0, B2=0.0, k=1.0):
    # Define grid dimensions
    shape = (100, 100, 100)
    x, y, z = np.indices(shape)
    # Define the center of the grid
    cte_to_add=-0.001
    xc, yc, zc = shape[0] / 2+cte_to_add, shape[1] / 2+cte_to_add, shape[2] / 2+cte_to_add
    x, y, z=x-xc, y-yc, z- zc
    
    # Convert Cartesian coordinates to cylindrical coordinates
    r = np.sqrt((x)**2 + (y)**2) + 1e-6  # Add small value to avoid division by zero
    theta=np.arctan2(y,x)+pi
    z_grid = z - zc

    
    nu = 0.3       # Poisson's ratio for isotropic media
    # ------------------ mixed Dislocation in Isotropic Media ------------------
    Ux_mixed_iso, Uy_mixed_iso, Uz_mixed_iso = mixed_dislocation_isotropic_cart(b_x, b_y, b_z, x, y, nu=0.3)
    
    def get_ringdata_r(r,theta,Ux_cyl,Uy_cyl,Uz_cyl,z_mid=0,r_val=8):
        Ux_mid = Ux_cyl[:, :, z_mid]
        Uy_mid = Uy_cyl[:, :, z_mid]
        Uz_mid = Uz_cyl[:, :, z_mid]
        # Flags to check if each displacement field has non-zero values
        trig_plot_x = np.any(Ux_mid != 0)
        trig_plot_y = np.any(Uy_mid != 0)
        trig_plot_z = np.any(Uz_mid != 0)
        # Create R and Theta arrays for plotting
        R, Theta = r[:, :, z_mid], theta[:, :, z_mid]
        X,Y,Z=cylindrical_to_cartesian(R, Theta, z_mid)
        
        mask = np.isclose(R, r_val, atol=1)
        theta_masked = Theta[mask].flatten()
        U_maskedx    = Ux_mid[mask].flatten()
        U_maskedy    = Uy_mid[mask].flatten()
        U_maskedz    = Uz_mid[mask].flatten()
        
        # Sort the theta and U values
        sorted_indices = np.argsort(theta_masked)
        theta_sorted = theta_masked[sorted_indices]
        U_sortedx = U_maskedx[sorted_indices]
        U_sortedy = U_maskedy[sorted_indices]
        U_sortedz = U_maskedz[sorted_indices]
    
        diff_U___=np.diff(U_sortedx)
        if np.any(diff_U___>0.2):
            jump_upos=np.where(np.diff(U_sortedx)>0.2)[0][0]
            modified_U=array(U_sortedx)
            modified_U[jump_upos+1:]=modified_U[jump_upos+1:]-(modified_U[jump_upos+1]-modified_U[jump_upos])
            U_sortedx=modified_U
        return theta_sorted,U_sortedx,U_sortedy,U_sortedz
    
    theta_sorted__mixed_iso,U_sortedx__mixed_iso,U_sortedy__mixed_iso,U_sortedz__mixed_iso=get_ringdata_r(r,theta,Ux_mixed_iso, Uy_mixed_iso, Uz_mixed_iso,z_mid=0)

    x_predict___=theta_sorted__mixed_iso*180/pi
    y_predict___1=U_sortedx__mixed_iso    
    y_predict___2=U_sortedy__mixed_iso
    y_predict___3=U_sortedz__mixed_iso

    
    y_predict___t=y_predict___1+y_predict___2-y_predict___3
    y_predict___t = A * y_predict___t + C
    periodic_adjustment = fun_oscillation_part(B1,B2,k,x_predict___)


    y_predict___t+=periodic_adjustment
    return(x_predict___,y_predict___t)
#------------------------------------------------------------------------------------------------------------
def function_to_predict_mixed_dislo_isotropic(x,y,b,bz,alpha):
    nu=0.3
    bragg=[1,1,-1]
    U_y = -b / (2 * np.pi) * ((1 - 2*nu) / (4 * (1 - nu)) * np.log(x**2 + y**2) +(x**2 - y**2) / (4 * (1 - nu) * (x**2 + y**2)))
    U_x = (-b / (2 * np.pi) * (np.arctan2(y, x)+ (x * y) / (2 * (1 - nu) * (x**2 + y**2))))
    U_z = bz * np.arctan2(y, x) / (2 * np.pi)
    
    return alpha*(bragg[0]*U_x + bragg[1]*U_y) - bragg[2]*(1-alpha)* U_z
#------------------------------------------------------------------------------------------------------------
def get_predictiontheo_of_experimental(bx= 1, by=1,bz=1,alpha_screw_edge=1,bragg=[1,1,-1]):    
    
    a0=3.924
    bragg   = bragg/np.linalg.norm(bragg)
    #bragg   /=2*pi/a0
    norm_b  = np.linalg.norm(array([bx,by,bz]))
    bx/=norm_b
    by/=norm_b
    bz/=norm_b
    
    r_val=5
    # ------------------ Grid and Displacement Calculation ------------------
    i_metal="Platinum"
    selected_metal = metals_elastic_constants[i_metal]
    # Define grid dimensions
    shape = (100, 100, 100)
    x, y, z = np.indices(shape)
    # Define the center of the grid
    cte_to_add=-0.001
    xc, yc, zc = shape[0] / 2+cte_to_add, shape[1] / 2+cte_to_add, shape[2] / 2+cte_to_add
    x, y, z=x-xc, y-yc, z- zc
    # Convert Cartesian coordinates to cylindrical coordinates
    r = np.sqrt((x)**2 + (y)**2) + 1e-6  # Add small value to avoid division by zero
    theta=np.arctan2(y,x)+pi
    z_grid = z - zc
    # Set parameters
    b = 1.0        # Burgers vector magnitude
    nu = 0.3       # Poisson's ratio for isotropic media
    c11 = selected_metal["C11"]      # Elastic constant C11 in GPa
    c12 = selected_metal["C12"]      # Elastic constant C12 in GPa
    c44 = selected_metal["C44"]      # Elastic constant C44 in GPa
    z_mid=r.shape[-1]//2
    R, Theta = r[:, :, z_mid], theta[:, :, z_mid]
    mask = np.isclose(R, r_val, atol=2)
    Ux_edge_iso, Uy_edge_iso, Uz_edge_iso = edge_dislocation_isotropic(b_z, nu, r, theta)
    Ux_screw_iso, Uy_screw_iso, Uz_screw_iso = screw_dislocation_isotropic(b_z, theta)
    Ux_edge_aniso, Uy_edge_aniso, Uz_edge_aniso = edge_dislocation_anisotropic_carti(b_x, b_y, c11, c12, c44, x, y)
    Ux_screw_aniso, Uy_screw_aniso, Uz_screw_aniso = screw_dislocation_anisotropic_carti(b_z, c11, c12, c44, x,y)
    Theta_edge=np.unwrap(Theta+pi/2)
    #Theta_edge=(Theta)
    Ux_edge_aniso_masked_flatted, Uy_edge_aniso_masked_flatted, Uz_edge_aniso_masked_flatted    = Ux_edge_aniso [:, :,z_mid][mask].flatten(), Uy_edge_aniso [:, :,z_mid][mask].flatten(), Uz_edge_aniso [:, :,z_mid][mask].flatten()
    Ux_screw_aniso_masked_flatted, Uy_screw_aniso_masked_flatted, Uz_screw_aniso_masked_flatted = Ux_screw_aniso[:, :,z_mid][mask].flatten(), Uy_screw_aniso[:, :,z_mid][mask].flatten(), Uz_screw_aniso[:, :,z_mid][mask].flatten()
    theta_flatten           = array(Theta[mask]).flatten()
    thetaedge_flatten           = array(Theta_edge[mask]).flatten()
    # Sort the theta and U values
    sorted_indices                 = np.argsort(theta_flatten)
    sorted_indices_edge            = np.argsort(thetaedge_flatten)
    theta_flatten_sorted           = np.rint(theta_flatten    [sorted_indices]*180/pi)
    thetaedge_flatten_sorted       = np.rint(thetaedge_flatten[sorted_indices_edge]*180/pi)
    
    Ux_edge_aniso_masked_flatted_sorted  = Ux_edge_aniso_masked_flatted [sorted_indices_edge]
    Uy_edge_aniso_masked_flatted_sorted  = Uy_edge_aniso_masked_flatted [sorted_indices_edge]
    Uz_edge_aniso_masked_flatted_sorted  = Uz_edge_aniso_masked_flatted [sorted_indices_edge]
    Ux_screw_aniso_masked_flatted_sorted = Ux_screw_aniso_masked_flatted[sorted_indices]
    Uy_screw_aniso_masked_flatted_sorted = Uy_screw_aniso_masked_flatted[sorted_indices]
    Uz_screw_aniso_masked_flatted_sorted = Uz_screw_aniso_masked_flatted[sorted_indices]

    u_final= alpha_screw_edge*(
        bragg[0]*Ux_edge_aniso_masked_flatted_sorted + bragg[1]* Uy_edge_aniso_masked_flatted_sorted +bragg[2]* Uz_edge_aniso_masked_flatted_sorted 
    ) +(
        1-alpha_screw_edge)*(    
            bragg[0]*Ux_screw_aniso_masked_flatted_sorted + bragg[1]*  Uy_screw_aniso_masked_flatted_sorted+bragg[2]* Uz_screw_aniso_masked_flatted_sorted
                              )
    u_final=u_final-u_final.min()
    return u_final ,thetaedge_flatten_sorted
#####################################################################################################################
#####################################################################################################################
############################################ simu dislo utility######################################################
#####################################################################################################################
#####################################################################################################################


def plot_displacement_vectors_3D(angle_ring, displacement_vectors_final, save_path=None):
    """
    Plots the displacement vectors in 3D using both:
    1️⃣ **Matplotlib** (Quiver plot for vector field visualization).
    2️⃣ **Plotly** (Interactive 3D scatter for exploration).

    Args:
        angle_ring (np.ndarray): Array of angles corresponding to displacement vectors.
        displacement_vectors_final (np.ndarray): Nx3 array of displacement vectors.
        save_path (str, optional): Path to save the Matplotlib plot.

    Returns:
        None (Plots the 3D visualization)
    """

    ### --- 1️⃣ Matplotlib 3D Quiver Plot --- ###
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Define the displacement components
    X = angle_ring  # X-axis (Angle)
    Y = displacement_vectors_final[:, 0]  # Y-component of displacement
    Z = displacement_vectors_final[:, 1]  # Z-component of displacement
    U = np.zeros_like(X)  # Vector origin (no X-component)
    V = displacement_vectors_final[:, 2]  # Z displacement (vertical)
    W = np.zeros_like(X)  # No displacement in the W direction

    # Plot the 3D quiver plot
    ax.quiver(X, Y, Z, U, V, W, length=0.5, normalize=True, color="blue")

    ax.set_xlabel("Angle (Degrees)")
    ax.set_ylabel("X Displacement")
    ax.set_zlabel("Y Displacement")
    ax.set_title("3D Displacement Vector Field")

    if save_path:
        plt.savefig(save_path)
    plt.show()

    ### --- 2️⃣ Interactive Plotly 3D Scatter --- ###
    fig = go.Figure(data=[go.Scatter3d(
        x=angle_ring, y=displacement_vectors_final[:, 0], z=displacement_vectors_final[:, 1],
        mode='markers',
        marker=dict(size=4, color=displacement_vectors_final[:, 2], colorscale='Viridis', opacity=0.8)
    )])

    fig.update_layout(title="Interactive 3D Displacement Vector Plot",
                      scene=dict(
                          xaxis_title="Angle (Degrees)",
                          yaxis_title="X Displacement",
                          zaxis_title="Y Displacement"
                      ))

    fig.show()


def get_dataset_simu(file_name,group_name,dataset_name):

    f = h5py.File(file_name, 'r')
    data_field = f[group_name + '/' + dataset_name]
    
    return array(data_field)
 
#------------------------------------------------------------------------------------------------------------

def get_abc_sixs2019_for_simu(save__orth,scan="S472",wanted_shape=[240,256,256]):
    delta_scans=np.load(save__orth+"raw_int_delta_scans.npz" ) 
    gamma_scans=np.load(save__orth+"raw_int_gamma_scans.npz" ) 
    mu_scans=np.load(save__orth+"raw_int_mu_scans.npz" ) 
    data_diff=np.load(save__orth+"raw_int_data_diff.npz" )     
    
    intens                           = data_diff[scan]
    print(intens.shape)
    delta_value,gamma_value,mu_value = delta_scans[scan],gamma_scans[scan],mu_scans[scan]
    cch_value                        = [193, 201]
    shape_diffraction_raw            = intens.shape
    wanted_shape_cx_realspace = (512,512,512) #None
    
    cx,cy,cz=orth_sixs2019_gridder_def(shape_diffraction_raw,delta_value,gamma_value,mu_value ,cch =cch_value,
                                     wanted_shape=(512,512,512)
                                      )
    cx0_new, cy0_new, cz0_new = np.array(C_O_M(pad_to_shape(intens,cx.shape))).astype('int')
    a, b, c= get_abc_direct_space_sixs2019(cx, cy, cz, cx0_new, cy0_new, cz0_new,wanted_shape=wanted_shape,mu_range_trigger=False)
    return a,b,c


#------------------------------------------------------------------------------------------------------------

def transform_and_discretize(non_ortho_data, non_ortho_coords, target_voxel_size, width=None, normalize=True,estimate_ortho_shape=False):
    """
    Converts a non-orthogonal dataset into an orthogonal grid using FuzzyGridder3D, ensuring a specific voxel size.
    
    Parameters:
        non_ortho_data (numpy.ndarray): The data values in non-orthogonal space (shape: 240×512×512).
        non_ortho_coords (numpy.ndarray): The real-world coordinates of each point in non-orthogonal space.
                                          Shape: (3, 240, 512, 512).
        target_voxel_size (tuple): Desired voxel size in (vx, vy, vz) format.
        width (float or tuple): Controls the spread of each data point. Default is half the bin size.
        normalize (bool): Whether to normalize the gridder output.

    Returns:
        tuple:
            - (numpy.ndarray) Orthogonal grid with transformed data.
            - (tuple) Chosen voxel size (vx, vy, vz).
            - (numpy.ndarray) The new orthogonal data array.
            - (numpy.ndarray) Estimated error map per voxel.
    """
    
    def compute_voxel_grid(x_coords, y_coords, z_coords, target_voxel_size):
        """
        Computes the voxel grid size based on target voxel size.
        """
        nx_p = int((np.max(x_coords) - np.min(x_coords)) / target_voxel_size[0]/10)
        ny_p = int((np.max(y_coords) - np.min(y_coords)) / target_voxel_size[1]/10)
        nz_p = int((np.max(z_coords) - np.min(z_coords)) / target_voxel_size[2]/10)
        return nx_p, ny_p, nz_p
    # Extract x, y, z coordinates directly from non-orthogonal data
    x_coords, y_coords, z_coords = non_ortho_coords
    if estimate_ortho_shape:
        nx_p, ny_p, nz_p = compute_voxel_grid(x_coords, y_coords, z_coords, target_voxel_size)
    else:
        nx_p, ny_p, nz_p =  (279, 350, 310) 
        
    print(f"Target voxel sizes: {target_voxel_size}")
    print(f"Computed grid size: ({nx_p}, {ny_p}, {nz_p})")
    
    # Use xrayutilities FuzzyGridder3D for voxelization
    gridder = xu.FuzzyGridder3D(nx_p, ny_p, nz_p)
    gridder.dataRange(np.min(x_coords), np.max(x_coords), np.min(y_coords), np.max(y_coords), np.min(z_coords), np.max(z_coords))
    
    # Set width if not provided
    if width is None:
        width = (target_voxel_size[0] / 2, target_voxel_size[1] / 2, target_voxel_size[2] / 2)
    
    # Grid the original data
    gridder(x_coords.flatten(), y_coords.flatten(), z_coords.flatten(), non_ortho_data.flatten(), width=width)
    ortho_data = gridder.data.T  # Transpose to match expected output shape
    
    # Compute voxel grid coordinates
    x = (gridder.xaxis / 10.0).astype(float)
    y = (gridder.yaxis / 10.0).astype(float)
    z = (gridder.zaxis / 10.0).astype(float)
    x -= x[len(x) // 2]
    y -= y[len(y) // 2]
    z -= z[len(z) // 2]
    
    # Compute final voxel sizes
    voxel_size = np.round(np.mean(np.diff(x)),1), np.round(np.mean(np.diff(y)),1), np.round(np.mean(np.diff(z)),1)
    print(f'Final voxel size real space (nm): {voxel_size}')
    
    return ortho_data, voxel_size, x, y, z
#------------------------------------------------------------------------------------------------------------
#🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀
def dislo_phase_model(theta, t, G, b, nu=0.3, fact=1, r=1.0, print_debug=False,only_theta_dep=True):
    """
    Compute the theoretical phase shift due to a dislocation.

    Parameters:
    - theta: np.ndarray or float, polar angle(s) in radians
    - t: (3,) array, dislocation line direction
    - G: (3,) array, reciprocal lattice vector
    - b: (3,) array, Burgers vector
    - nu: float, Poisson's ratio (default = 0.3)
    - d_hkl: float, Interplanar spacing (default = 0.39239)
    - r: np.ndarray or float, radial distance(s) from dislocation core
    - print_debug: bool, whether to print debug information

    Returns:
    - u_final: np.ndarray, theoretical phase shift
    """

    # Convert inputs to NumPy arrays and ensure correct shape
    t = np.asarray(t, dtype=np.float64).reshape(-1)  
    G = np.asarray(G, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    theta = np.asarray(theta, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)

    # Compute perpendicular component of Burgers vector
    b_perp = project_vector(b, t)
    b_perp_norm = np.linalg.norm(b_perp)

    b_screw = np.dot(b, t) / np.linalg.norm(t)

    if print_debug:
        print(f"b_perp: {b_perp}, b_perp_norm: {b_perp_norm}, b_screw: {b_screw}")

    if np.isclose(b_perp_norm, 0) and print_debug:
        print("Warning: b_perp is zero, phase shift will be zero.")
    
    # Compute displacement fields
    if only_theta_dep:
        u_x_theo = (b_perp_norm / (2 * np.pi)) * (theta + np.sin(2 * theta) / (4 * (1 - nu)))
        u_y_theo = -(b_perp_norm / (8 * np.pi * (1 - nu))) * (np.cos(2 * theta))
        u_z_theo = (b_screw / (2 * np.pi)) * theta

    else:
        u_x_theo = (b_perp_norm / (2 * np.pi)) * (theta + np.sin(2 * theta) / (4 * (1 - nu)))
        u_y_theo = -(b_perp_norm / (8 * np.pi * (1 - nu))) * (2 * (1 - 2 * nu) * np.log(r) + np.cos(2 * theta))
        u_z_theo = (b_screw / (2 * np.pi)) * theta

    if print_debug:
        print(f"u_x_theo: {u_x_theo}, u_y_theo: {u_y_theo}, u_z_theo: {u_z_theo}")

    # Compute rotation matrix from real space to dislocation frame
    R = dislo_rotation_matrix_real_to_theo(t, b)

    if print_debug:
        print(f"Rotation matrix R:\n{R}")

    # Rotate G vector
    G_theo = np.dot(R, G)

    if print_debug:
        print(f"G_theo: {G_theo}")

    # Compute phase shift
    u_final = fact * (G_theo[0] * u_x_theo + G_theo[1] * u_y_theo + G_theo[2] * u_z_theo)

    if print_debug:
        print(f"Final Phase Shift: {u_final}")

    return u_final
#🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀
def get_phase_simu_gvector(h5_file, a, b, c, G=[1, -1, 1], path_to_save=None, file_name=None, h_w=45,
                           nb_of_phase_to_test=10, voxel_sizes=(1, 1, 1), stain_threshold=0.1,
                           debug_plot_step_1=False, save_output_step_1=True, min_cluster_size=20,
                           distance_threshold=2.0, cylinder_radius=2, num_spline_points=5000,
                           smoothing_param=4, eps=3.5, min_samples=20, final_radius__of_dislo=1,
                           height=100, step_along_dislo_line=1, orthogonalise_data=False,
                           wanted_voxel_sizes_ortho=(15, 15, 15), cut_strain_lower=0.01,
                           cut_strain_upper=0.5, plot_debug_centring_obj_from_diffraction=False, 
                           plot_debug=True,centering_method_for_fft="com",center_diffraction=True,shift_center=None):
    
    """
    Computes the simulated phase field based on a given diffraction vector G.
    
    This function extracts the displacement field from an HDF5 dataset, computes 
    the projected phase field, processes dislocation features, applies clustering, 
    and saves the results in a VTI file.

    ### **Processing Steps:**
    
    1️⃣ **Load and Normalize Data**
        - Extracts magnitude (`data_mod`) and displacement field (`data_disp`) from HDF5.
        - Normalizes the displacement field by the lattice parameter.

    2️⃣ **Compute Simulated Phase Projection**
        - Uses `G` to compute the phase projection from `data_disp`.

    3️⃣ **Preprocessing (Cropping and Normalization)**
        - Creates a support mask to isolate relevant data.
        - Crops data based on the region of interest (`h_w`).
    
    4️⃣ **Debug Plots for Initial Data (Optional)**
        - Plots raw phase data (before transformation).

    5️⃣ **Compute Diffraction Pattern**
        - Computes intensity and reciprocal space phase.
        - Applies inverse Fourier transform to realign phase.

    6️⃣ **Phase Processing**
        - Removes phase ramp artifacts.
        - Applies support mask filtering.

    7️⃣ **Compute Gradient-Based Masks**
        - Uses gradient operations to detect phase discontinuities.

    8️⃣ **Compute Strain Map**
        - Computes strain amplitude and masks invalid regions.

    9️⃣ **Dislocation Clustering and Detection**
        - Uses `DBSCAN` and strain clustering to detect dislocations.
        - Identifies key dislocation structures.

    🔟 **Dislocation Line Extraction**
        - Fits a 3D line to detected dislocations.
        - Generates a filled cylindrical dislocation mask.

    1️⃣1️⃣ **Optional: Orthogonalization**
        - Transforms data into a new coordinate system.
        - Adjusts voxel sizes for improved analysis.

    1️⃣2️⃣ **Final Cropping and Saving**
        - Crops data again for better focus.
        - Saves results as a VTI file for further visualization.

    1️⃣3️⃣ **Debug Visualization (Optional)**
        - Plots 3D isosurfaces to visualize dislocations.

    ---
    
    ### **Parameters:**
    #### **Input Data**
    - `h5_file` (str): Path to the HDF5 file containing the simulation data.
    - `G` (list): Diffraction vector for phase projection.

    #### **Processing Parameters**
    - `h_w` (int): Half-width of the cropping region of interest.
    - `nb_of_phase_to_test` (int): Number of phase values tested in strain analysis.
    - `voxel_sizes` (tuple): Voxel size in the dataset.
    - `stain_threshold` (float): Threshold for strain clustering.

    #### **Dislocation Detection Parameters**
    - `min_cluster_size` (int): Minimum size for dislocation clusters.
    - `distance_threshold` (float): Distance threshold for merging clusters.
    - `cylinder_radius` (int): Radius of the cylindrical mask for dislocations.
    - `final_radius__of_dislo` (int): Final radius for refined dislocation segmentation.
    - `height` (int): Height of the detected dislocation region.
    - `step_along_dislo_line` (int): Step size along the dislocation line.

    #### **DBSCAN Clustering Parameters**
    - `eps` (float): Epsilon for DBSCAN clustering.
    - `min_samples` (int): Minimum samples per cluster.

    #### **Smoothing Parameters**
    - `num_spline_points` (int): Number of spline points for smoothing.
    - `smoothing_param` (int): Smoothing parameter for spline fitting.

    #### **Orthogonalization (Optional)**
    - `orthogonalise_data` (bool): Whether to orthogonalize the dataset.
    - `wanted_voxel_sizes_ortho` (tuple): Voxel sizes for orthogonalized dataset.

    #### **Strain Map Parameters**
    - `cut_strain_lower` (float): Lower cutoff for strain values.
    - `cut_strain_upper` (float): Upper cutoff for strain values.

    #### **Plotting & Debugging**
    - `debug_plot_step_1` (bool): Debug plots for clustering step.
    - `save_output_step_1` (bool): Whether to save intermediate outputs.
    - `plot_debug_centring_obj_from_diffraction` (bool): Debug plots for centering diffraction patterns.
    - `plot_debug` (bool): Global toggle for all debug plots.

    ---
    
    ### **Returns:**
    - `simu_phase` (np.ndarray): Simulated phase field.
    - `obj` (np.ndarray): Complex object field (amplitude + phase).
    - `strain_amp` (np.ndarray): Strain amplitude map.
    - `selected_dislocation_data` (np.ndarray): Final dislocation segmentation mask.

    **If `orthogonalise_data=True`, returns additional orthogonalized versions:**
    - `ortho_simu_phase` (np.ndarray): Orthogonalized simulated phase.
    - `ortho_obj` (np.ndarray): Orthogonalized complex object field.
    - `ortho_strain_amp` (np.ndarray): Orthogonalized strain amplitude.
    - `ortho_selected_dislocation_data` (np.ndarray): Orthogonalized dislocation segmentation.

    ---
    
    ### **Example Usage:**
    ```python
    results = get_phase_simu_gvector(
        h5_file="simulation_data.h5",
        a=1, b=1, c=1,
        G=[1, -1, 1],
        path_to_save="output/",
        file_name="test_simulation",
        plot_debug=True
    )
    ```

    **If `orthogonalise_data=True`:**
    ```python
    (simu_phase, obj, strain_amp, selected_dislocation_data,
     ortho_simu_phase, ortho_obj, ortho_strain_amp, ortho_selected_dislocation_data) = results
    ```
    
    ---
    
    ### **Notes:**
    - This function is **highly optimized** for analyzing phase fields in **Bragg coherent diffraction imaging (BCDI)**.
    - The `DBSCAN` clustering and **cylinder-based dislocation segmentation** help refine the analysis.
    - The function supports **debug visualization** for all major processing steps.
    - **Results are saved in `.vti` format** for easy visualization in ParaView.

    ---
    **🚀 Optimized for High-Resolution BCDI Dislocation Analysis! 🚀**
    """
    
    start_time=time.time()
    a0 = 3.9239  # Lattice parameter
    normalize_factor =1/10 # A° to nm as G is in A°-1
    
    # Load datasets from HDF5 file
    data_mod = get_dataset_simu(h5_file, '', '|f|')  # Magnitude data
    data_mod /= data_mod.max()  # Normalize data
    data_disp = get_dataset_simu(h5_file, '', 'U')  # Displacement field
    
    # Swap axes to shape (3, 240, 256, 256) and normalize
    data_disp = np.transpose(data_disp, (3, 0, 1, 2)) * normalize_factor

    
    def compute_phase_projection(data_disp, G):
        return G[0] * data_disp[0] + G[1] * data_disp[1] + G[2] * data_disp[2]
    
    simu_phase = compute_phase_projection(data_disp, G)

    
    # Define support mask based on magnitude threshold
    supp = np.where(data_mod < 0.1, 0, 1)
    supp = fill_up_support(supp)
    
    if not orthogonalise_data:
        # Crop arrays to focus on the region of interest
        det_ref, supp                = crop_3darray_pos(supp      , output_shape=(2 * h_w, 2 * h_w, 2 * h_w), methods="com", det_ref_return=True)
        data_mod                     = crop_3darray_pos(data_mod  , output_shape=(2 * h_w, 2 * h_w, 2 * h_w), methods=det_ref, verbose=False)
        simu_phase                   = crop_3darray_pos(simu_phase, output_shape=(2 * h_w, 2 * h_w, 2 * h_w), methods=det_ref, verbose=False)

    if plot_debug:
        plot_2D_slices_middle_one_array3D(zero_to_nan(simu_phase),cmap='jet',fig_title=f"Raw phase {file_name}",)
        if path_to_save is not None:
            plt.savefig(path_to_save+file_name+"raw_phase_no_op.png")
        plt.show()
    if center_diffraction:
        # Compute diffraction pattern
        intensity, phase_reciprocal, reciprocal_space_object = compute_diffraction_pattern(data_mod, simu_phase)
        # Perform inverse transformation
        data_mod, simu_phase = inverse_diffraction_pattern(reciprocal_space_object,center_diffraction=center_diffraction,center_method=centering_method_for_fft,shift_center=shift_center)
        
    supp = fill_up_support(np.where(data_mod < 0.1, 0, 1))
    simu_phase=simu_phase*supp
    data_mod=data_mod*supp
    if plot_debug:
        plot_2D_slices_middle_one_array3D(zero_to_nan(simu_phase),cmap='jet',fig_title=f"Raw phase {file_name} \n after centring diffraction pattern",)
        if path_to_save is not None:
            plt.savefig(path_to_save+file_name+"raw_phase_roundtrip_real_recip.png")
        plt.show()
    #simu_phase, _ = nan_to_zero(remove_phase_ramp(zero_to_nan(simu_phase)))
    if plot_debug:
        plot_2D_slices_middle_one_array3D(zero_to_nan(simu_phase),cmap='jet',fig_title=f"Raw phase {file_name} \n after centring diffraction pattern & Ramp removal",)
        if path_to_save is not None:
            plt.savefig(path_to_save+file_name+"raw_phase_roundtrip_real_recip_plus_ramp_removal.png")
        plt.show()
    simu_phase= nan_to_zero(center_angles(zero_to_nan(simu_phase)))
    
    # Compute gradient-based masks
    supp_1 = fill_up_support(np.where(data_mod < 0.15, 0, 1))
    gradient_mask = 1 - (np.max(nan_to_zero(abs(array(get_displacement_gradient(supp_1, voxel_size=(1, 1, 1))))), axis=0) != 0).astype(float)
    
    # Construct complex object with phase information
    obj = data_mod * supp * np.exp(1j * simu_phase)

    def compute_strain_map(obj, voxel_size, nb_of_phase_to_test):
        return getting_strain_mapvti(obj=obj, voxel_size=voxel_size, nb_of_phase_to_test=nb_of_phase_to_test)
    
    strain_mask, strain_amp = compute_strain_map(obj, (1, 1, 1), nb_of_phase_to_test)

    _mask = fill_up_support(data_mod > 0.2)
    gradient_modes_mask = (np.max(nan_to_zero(abs(array(get_displacement_gradient(_mask, voxel_size=(1, 1, 1))))), axis=0) != 0).astype(float)
    strain_amp = ((1 - gradient_modes_mask) * strain_amp).astype(float)
    strain_amp[strain_amp < cut_strain_lower] = 0
    strain_amp[strain_amp > cut_strain_upper] = 0
    _mask_notfilled = (data_mod > 0.4).astype(float)
    strain_amp *= (1 - _mask_notfilled)


    # Process clusters in strain map to extract dislocations
    if path_to_save is not None:
    
        final_labeled_clusters, num_final_clusters = process_and_merge_clusters_dislo_strain_map_refined(
            data=strain_amp, amp=data_mod, phase=simu_phase,save_path=path_to_save + "segmentation_" + file_name,voxel_sizes=tuple(voxel_sizes), threshold=stain_threshold, 
            min_cluster_size=min_cluster_size,distance_threshold=distance_threshold, cylinder_radius=cylinder_radius,num_spline_points=num_spline_points, smoothing_param=smoothing_param,
            eps=eps, min_samples=min_samples, save_output=save_output_step_1,debug_plot=debug_plot_step_1)
    else:
        final_labeled_clusters, num_final_clusters = process_and_merge_clusters_dislo_strain_map_refined(
            data=strain_amp, amp=data_mod, phase=simu_phase,save_path=None,voxel_sizes=tuple(voxel_sizes), threshold=stain_threshold, 
            min_cluster_size=min_cluster_size,distance_threshold=distance_threshold, cylinder_radius=cylinder_radius,num_spline_points=num_spline_points, smoothing_param=smoothing_param,
            eps=eps, min_samples=min_samples, save_output=False,debug_plot=False)
    # Extract structure and fit a 3D line
    threshold = 0.5  # Remove void space clusters
    points = extract_structure(final_labeled_clusters, threshold)
    centroid, direction = fit_line_3d(points)
    
    # Generate a filled cylinder along detected dislocation
    filled_cylinder_volume = generate_filled_cylinder_with_disks(
        final_labeled_clusters.shape, centroid, direction, final_radius__of_dislo, height, step_along_dislo_line)
    selected_dislocation_data = filled_cylinder_volume * _mask

    if orthogonalise_data:
        ortho_data_mod                 , voxel_size_ortho, x, y, z = transform_and_discretize(data_mod                  , (a,b,c),wanted_voxel_sizes_ortho ,normalize=False,estimate_ortho_shape=True)
        ortho_simu_phase               , voxel_size_ortho, x, y, z = transform_and_discretize(simu_phase                , (a,b,c),wanted_voxel_sizes_ortho ,normalize=False,estimate_ortho_shape=True)
        ortho_selected_dislocation_data, voxel_size_ortho, x, y, z = transform_and_discretize(selected_dislocation_data , (a,b,c),wanted_voxel_sizes_ortho ,normalize=False,estimate_ortho_shape=True)
        ortho_strain_amp               , voxel_size_ortho, x, y, z = transform_and_discretize(strain_amp                , (a,b,c),wanted_voxel_sizes_ortho ,normalize=False,estimate_ortho_shape=True)
    # Crop arrays to focus on the region of interest
    det_ref, supp                = crop_3darray_pos(supp      , output_shape=(2 * h_w, 2 * h_w, 2 * h_w), methods="com", det_ref_return=True)
    data_mod                     = crop_3darray_pos(data_mod  , output_shape=(2 * h_w, 2 * h_w, 2 * h_w), methods=det_ref, verbose=False)
    simu_phase                   = crop_3darray_pos(simu_phase, output_shape=(2 * h_w, 2 * h_w, 2 * h_w), methods=det_ref, verbose=False)
    strain_amp                   = crop_3darray_pos(strain_amp, output_shape=(2 * h_w, 2 * h_w, 2 * h_w), methods=det_ref, verbose=False)
    selected_dislocation_data    = crop_3darray_pos(selected_dislocation_data, output_shape=(2 * h_w, 2 * h_w, 2 * h_w), methods=det_ref, verbose=False)

    if orthogonalise_data:
        # Define support mask based on magnitude threshold
        supp_ortho = np.where(ortho_data_mod < 0.1, 0, 1)
        supp_ortho = fill_up_support(supp_ortho)
        
        det_ref, supp_ortho             = crop_3darray_pos(supp_ortho                      , output_shape=(2 * h_w, 2 * h_w, 2 * h_w), methods="com", det_ref_return=True)
        ortho_data_mod                  = crop_3darray_pos(ortho_data_mod                  , output_shape=(2 * h_w, 2 * h_w, 2 * h_w), methods=det_ref, verbose=False)
        ortho_simu_phase                = crop_3darray_pos(ortho_simu_phase                , output_shape=(2 * h_w, 2 * h_w, 2 * h_w), methods=det_ref, verbose=False)
        ortho_selected_dislocation_data = crop_3darray_pos(ortho_selected_dislocation_data , output_shape=(2 * h_w, 2 * h_w, 2 * h_w), methods=det_ref, verbose=False)
        ortho_strain_amp                = crop_3darray_pos(ortho_strain_amp                , output_shape=(2 * h_w, 2 * h_w, 2 * h_w), methods=det_ref, verbose=False)
        
        # Extract structure and fit a 3D line
        threshold = 0.5  # Remove void space clusters
        points = extract_structure(ortho_selected_dislocation_data, threshold)
        centroid, direction = fit_line_3d(points)
        
        # Generate a filled cylinder along detected dislocation
        filled_cylinder_volume = generate_filled_cylinder_with_disks(
            ortho_selected_dislocation_data.shape, centroid, direction, final_radius__of_dislo, height, step_along_dislo_line)
        ortho_selected_dislocation_data = filled_cylinder_volume * supp_ortho
        ortho_obj = ortho_data_mod * supp_ortho * np.exp(1j * ortho_simu_phase)
        
        if path_to_save is not None:
            gu.save_to_vti(filename=path_to_save + "ortho_final_dislo_line_segmentation_mask" + file_name + ".vti",
                           voxel_size=tuple(voxel_size_ortho),
                           tuple_array=(nan_to_zero(ortho_data_mod), nan_to_zero(ortho_simu_phase), ortho_selected_dislocation_data, ortho_strain_amp),
                           tuple_fieldnames=("density", "phase", "mask_dislo", "strain_amp"),
                           amplitude_threshold=0.01)
    if path_to_save is not None:
        # Save results to a VTI file
        gu.save_to_vti(filename=path_to_save + "final_dislo_line_segmentation_mask" + file_name + ".vti",
                       voxel_size=tuple(wanted_voxel_sizes_ortho),
                       tuple_array=(nan_to_zero(data_mod), nan_to_zero(simu_phase), selected_dislocation_data, strain_amp),
                       tuple_fieldnames=("density", "phase", "mask_dislo", "strain_amp"),
                       amplitude_threshold=0.01)
    obj = data_mod * supp * np.exp(1j * simu_phase)
    end_time = time.time()
    print(f"Processing took {np.round((end_time - start_time) / 60, 1)} minutes")
    if plot_debug:
        plot_3d_dislo_amp_disloline(data_mod,selected_dislocation_data,iso_value = 0.3,elev=60, azim=-120,save_plot=path_to_save+file_name+"_raw_data_3D_plot_dislo_amp.png",
                                    title_fig=" Isosurface plot of the amplitude with the dislocation line ")
    if orthogonalise_data:
        if plot_debug:
            plot_3d_dislo_amp_disloline(nan_to_zero(ortho_data_mod),ortho_selected_dislocation_data,iso_value = 0.3,elev=60, azim=-120,save_plot=path_to_save+file_name+"_ortho_data_3D_plot_dislo_amp.png",
                                        title_fig=" Isosurface plot of the amplitude with the dislocation line ")
        return simu_phase, obj, strain_amp,selected_dislocation_data, ortho_simu_phase,ortho_obj, ortho_strain_amp,ortho_selected_dislocation_data
    else:
        return simu_phase, obj, strain_amp,selected_dislocation_data

#🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀
def calculate_phasetheo_for_dislo_particle(data_shape, centroid, direction, b_, t=np.array([8., 3., 8.]), G=np.array([1, -1, 1])):
    """Compute phase and displacement for a dislocation particle using vectorized operations."""
    
    # Ensure all vectors are properly formatted
    t = np.array(t, dtype=np.float64).reshape(-1)  
    G = np.array(G, dtype=np.float64).reshape(-1)
    b_ = np.array(b_, dtype=np.float64).reshape(-1)
    
    selected_point_index = 0
    r = 1
    dr = 1

    # Normalize direction
    direction = direction / np.linalg.norm(direction)

    # Compute disk center
    disk_center = centroid + selected_point_index * direction

    # Define coordinate system
    z_axis = direction
    random_vector = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
    x_axis = np.cross(z_axis, random_vector)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    # Generate voxel grid
    grid_x, grid_y, grid_z = np.meshgrid(
        np.arange(data_shape[0]),
        np.arange(data_shape[1]),
        np.arange(data_shape[2]),
        indexing="ij",
    )
    grid_points = np.stack([grid_x, grid_y, grid_z], axis=-1)  # Shape: (N, N, N, 3)

    # Shift grid points relative to disk center
    shifted_points = grid_points - disk_center

    # Convert to local coordinate system
    local_x = np.dot(shifted_points, x_axis)
    local_y = np.dot(shifted_points, y_axis)
    local_z = np.dot(shifted_points, z_axis)

    grid_points_dislo = np.stack([local_x, local_y, local_z], axis=-1)

    # Compute radial distances and polar angles
    radial_distances = np.sqrt(local_x**2 + local_y**2)
    polar_angles = np.arctan2(local_y, local_x)

    # Reshape angles & distances
    polar_angles_3d = polar_angles.reshape(data_shape)
    radial_distances_3d = radial_distances.reshape(data_shape)

    print("Polar Angles Shape:", polar_angles_3d.shape)
    print("Radial Distances Shape:", radial_distances_3d.shape)

    # **🚀 Vectorized computation of dislo_phase_model**
    predicted_phase_3d = np.zeros(data_shape, dtype=np.float32)

    # Only apply `dislo_phase_model` to the masked region (vectorized)
    predicted_phase_3d = dislo_phase_model(
        polar_angles_3d, t, G, b_, r=radial_distances_3d, print_debug=False
    )

    return (
        direction,
        grid_points,
        grid_points_dislo,
        predicted_phase_3d,
        polar_angles_3d,
        radial_distances_3d,
    )

#------------------------------------------------------------------------------------------------------------
#🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀
def compute_diffraction_pattern(amplitude, phase):
    # Construct the complex object in real space
    real_space_object = amplitude * np.exp(1j * phase)
    
    # Compute the 3D Fourier transform
    reciprocal_space_object = np.fft.ifftshift(
            np.fft.fftn(
                np.fft.fftshift(real_space_object)
            )
        )
    
    # Compute the intensity and phase in reciprocal space
    intensity = np.abs(reciprocal_space_object) ** 2
    phase_reciprocal = np.angle(reciprocal_space_object)
    
    return intensity, phase_reciprocal, reciprocal_space_object

#🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀
def inverse_diffraction_pattern(reciprocal_space_object, center_diffraction=True, center_method="com", shift_center=None):
    """
    Computes the inverse diffraction pattern by performing an inverse 3D Fourier transform 
    on a given reciprocal space object. Optionally, it recenters the diffraction pattern 
    using shifting instead of padding and cropping.

    Parameters:
    -----------
    reciprocal_space_object : np.ndarray
        A 3D complex-valued numpy array representing the reciprocal space data.
    
    center_diffraction : bool, optional (default=True)
        If True, the function will recenter the diffraction pattern based on 
        the center of mass (COM) of its intensity.
    
    shift_center : tuple or None, optional (default=None)
        A (z, y, x) tuple specifying an additional shift to apply to the computed 
        center of mass before shifting. If None, no additional shift is applied.

    Returns:
    --------
    amplitude : np.ndarray
        A 3D numpy array containing the amplitude of the real-space object.
    
    phase : np.ndarray
        A 3D numpy array containing the phase of the real-space object.
    
    """
    if center_diffraction:
        shape_data_init = np.array(reciprocal_space_object.shape)
        print(f"Initial shape of data: {shape_data_init}")
        
        # Compute intensity to find the center of mass
        intensity = np.abs(reciprocal_space_object) ** 2
        
        # Find max and center of mass positions
        max_pos, com_pos = find_max_and_com_3d(intensity, window_size=10)
        print(f"Max position: {max_pos}, Center of mass position: {com_pos}")
        
        if center_method == "com":
            center_pos = com_pos
        elif center_method == "max":
            center_pos = max_pos
        else:
            print("Wrong value for center_method. Using default (com).")
            center_pos = com_pos

        if shift_center is not None:
            try:
                center_pos = tuple(np.array(center_pos) + np.array(shift_center))
                print(f"Adjusted center position with shift: {center_pos}")
            except:
                print("Could not apply the shift.")

        # Compute the shift vector to bring center_pos to the middle of the array
        target_center = np.array(shape_data_init) // 2
        shift_vector =  target_center -np.array(center_pos) 
        print(f"Computed shift vector: {shift_vector}")
        
        plot_3D_projections(intensity, log_scale=True, cmap="jet")
        
        # Shift the object to align center_pos to the center
        reciprocal_space_object = np.roll(reciprocal_space_object, shift_vector, axis=(0, 1, 2))
        
        # Compute intensity after shifting
        intensity = np.abs(reciprocal_space_object) ** 2
        max_pos, com_pos = find_max_and_com_3d(intensity, window_size=10)
        plot_3D_projections(intensity, log_scale=True, cmap="jet")
        print(f"Max position: {max_pos}, Center of mass position: {com_pos}")
        
    # Compute the inverse 3D Fourier transform
    real_space_object = np.fft.ifftshift(
        np.fft.ifftn(
            np.fft.fftshift(reciprocal_space_object)
        )
    )

    # Extract amplitude and phase in real space
    amplitude = np.abs(real_space_object)
    phase = np.angle(real_space_object)
    
    print("Inverse Fourier transform computed.")
    return amplitude, phase
#------------------------------------------------------------------------------------------------------------
def inverse_diffraction_pattern_old(reciprocal_space_object, center_diffraction=True, center_method="com",shift_center=None):
    """
    Computes the inverse diffraction pattern by performing an inverse 3D Fourier transform 
    on a given reciprocal space object. Optionally, it recenters the diffraction pattern 
    based on the center of mass (COM).

    Parameters:
    -----------
    reciprocal_space_object : np.ndarray
        A 3D complex-valued numpy array representing the reciprocal space data.
    
    center_diffraction : bool, optional (default=True)
        If True, the function will recenter the diffraction pattern based on 
        the center of mass (COM) of its intensity.
    
    shift_center : tuple or None, optional (default=None)
        A (z, y, x) tuple specifying an additional shift to apply to the computed 
        center of mass before cropping. If None, no additional shift is applied.

    Returns:
    --------
    amplitude : np.ndarray
        A 3D numpy array containing the amplitude of the real-space object.
    
    phase : np.ndarray
        A 3D numpy array containing the phase of the real-space object.

    Notes:
    ------
    - The function first pads the reciprocal space data to twice its original size.
    - The intensity is computed as the squared magnitude of the reciprocal space object.
    - The center of mass (COM) of the intensity is computed and used to recenter the object.
    - If `shift_center` is provided, the computed COM is adjusted accordingly before cropping.
    - The function then applies an inverse 3D Fourier transform to obtain the real-space object.
    """
    if center_diffraction:
        shape_data_init = np.array(reciprocal_space_object.shape)
        shape_data_pad = shape_data_init * 2

        # Pad reciprocal space to double its size
        reciprocal_space_object = pad_to_shape(reciprocal_space_object, shape_data_pad)

        # Compute intensity to find the center of mass
        intensity = np.abs(reciprocal_space_object) ** 2

        # Crop the padded object based on center of mass (COM)
        max_pos, com_pos = find_max_and_com_3d(intensity, window_size=10)
        if center_method=="com":
            crop_pos__=com_pos
        elif center_method=="max":
            crop_pos__=max_pos
        else: 
            print("wrong value for center_method of fft intensity. the default value (com) will be taken" )
            crop_pos__=com_pos

        if shift_center is not None:
            try:
                crop_pos__ = tuple(np.array(crop_pos__) + np.array(shift_center))
            except:
                print("Could not apply the shift.")
    
        # Crop using COM position
        reciprocal_space_object = crop_3d_obj_pos(reciprocal_space_object, output_shape=shape_data_init, methods=crop_pos__)

    # Compute the inverse 3D Fourier transform
    real_space_object = np.fft.ifftshift(
        np.fft.ifftn(
            np.fft.fftshift(reciprocal_space_object)
        )
    )

    # Extract amplitude and phase in real space
    amplitude = np.abs(real_space_object)
    phase = np.angle(real_space_object)

    return amplitude, phase
#🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀
def plot_debug_difraction_from_obj(amplitude, phase, intensity, phase_reciprocal, reconstructed_amplitude, reconstructed_phase):
    N = amplitude.shape[0]
    mid_idx = N // 2
    
    amplitude_slice = amplitude[:, :, mid_idx]
    phase_slice = phase[:, :, mid_idx]
    intensity_slice = intensity[:, :, mid_idx]
    phase_reciprocal_slice = phase_reciprocal[:, :, mid_idx]
    reconstructed_amplitude_slice = reconstructed_amplitude[:, :, mid_idx]
    reconstructed_phase_slice = reconstructed_phase[:, :, mid_idx]
    
    intensity_sum = intensity.sum(axis=0)
    phase_sum = phase_reciprocal.sum(axis=0)
    
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    
    axes[0, 0].imshow(amplitude_slice, cmap='gray', extent=[-1, 1, -1, 1])
    axes[0, 0].set_title("Initial Amplitude (Central Slice)")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    
    axes[0, 1].imshow(phase_slice, cmap='twilight', extent=[-1, 1, -1, 1])
    axes[0, 1].set_title("Initial Phase (Central Slice)")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("y")
    
    axes[1, 0].imshow(np.log1p(intensity_slice), cmap='inferno', extent=[-1, 1, -1, 1])
    axes[1, 0].set_title("Log Diffracted Intensity (Central Slice)")
    axes[1, 0].set_xlabel("q_x")
    axes[1, 0].set_ylabel("q_y")
    
    axes[1, 1].imshow(phase_reciprocal_slice, cmap='twilight', extent=[-1, 1, -1, 1])
    axes[1, 1].set_title("Phase in Reciprocal Space (Central Slice)")
    axes[1, 1].set_xlabel("q_x")
    axes[1, 1].set_ylabel("q_y")
    
    axes[2, 0].imshow(reconstructed_amplitude_slice, cmap='gray', extent=[-1, 1, -1, 1])
    axes[2, 0].set_title("Reconstructed Amplitude (Central Slice)")
    axes[2, 0].set_xlabel("x")
    axes[2, 0].set_ylabel("y")
    
    axes[2, 1].imshow(reconstructed_phase_slice, cmap='twilight', extent=[-1, 1, -1, 1])
    axes[2, 1].set_title("Reconstructed Phase (Central Slice)")
    axes[2, 1].set_xlabel("x")
    axes[2, 1].set_ylabel("y")
    
    axes[3, 0].imshow(np.log1p(intensity_sum), cmap='inferno', extent=[-1, 1, -1, 1])
    axes[3, 0].set_title("Summed Intensity along z-axis")
    axes[3, 0].set_xlabel("q_x")
    axes[3, 0].set_ylabel("q_y")
    
    axes[3, 1].imshow(phase_sum, cmap='twilight', extent=[-1, 1, -1, 1])
    axes[3, 1].set_title("Summed Phase along z-axis")
    axes[3, 1].set_xlabel("q_x")
    axes[3, 1].set_ylabel("q_y")
    
    plt.tight_layout()
    plt.show()
#------------------------------------------------------------------------------------------------------------
#🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀
def plot_3d_dislo_amp_disloline(amp,dislo,iso_value = 0.3,elev=60, azim=-120,save_plot=None,title_fig=""):    
    data_mod,selected_dislocation_data=amp,dislo
    # Define isosurface level
    data_shape=array(amp.shape)
    
    # Compute isosurface using Marching Cubes
    verts1, faces1, _, _ = skm.marching_cubes(data_mod, level=iso_value)
    verts2, faces2, _, _ = skm.marching_cubes(selected_dislocation_data, level=0.5)
    
    # Create Matplotlib 3D figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the first isosurface (data_mod) with increased transparency
    mesh1 = Poly3DCollection(verts1[faces1], alpha=0.2, edgecolor='gray', facecolor='white')
    ax.add_collection3d(mesh1)
    
    # Plot the second isosurface (selected_dislocation_data) with different color
    mesh2 = Poly3DCollection(verts2[faces2], alpha=1.0, edgecolor='blue', facecolor='blue')
    ax.add_collection3d(mesh2)
    
    # Set axis limits and labels
    ax.set_xlim(0, data_shape[0])
    ax.set_ylim(0, data_shape[1])
    ax.set_zlim(0, data_shape[2])
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title(title_fig)
    
    # Adjust viewing angle for better perspective
    ax.view_init(elev=elev, azim=azim)
    if save_plot is not None:
        plt.savefig(save_plot)
    
    # Show the refined plot
    plt.show()

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------


#🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀
def plot_phase_subplots_allsimu(threecases_raw_results_simu_angle, threecases_raw_results_simu_phase, scan_list, title="Phase Data Subplots", save_path=None):
    """
    Plots phase data from multiple scans in a grid layout.
    
    Parameters:
    - threecases_raw_results_simu_angle: List of angle datasets
    - threecases_raw_results_simu_phase: List of phase datasets
    - all_raw_results_file_path_ortho_analysis_simu_ortho: Dictionary of scan file paths
    - title: Title of the plot (default: "Phase Data Subplots")
    - save_path: If provided, saves the figure to the given path.
    """
    num_scans = len(threecases_raw_results_simu_angle)  # Number of datasets
    num_cols = 2  # Set number of columns
    num_rows = (num_scans + 1) // num_cols  # Calculate required rows

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 4 * num_rows), sharey=True)
    fig.suptitle(title, fontsize=14)
    
    # Ensure axes is always a 2D array for easier indexing
    axes = np.array(axes).reshape(num_rows, num_cols)
    
    for i in range(num_scans):
        scan = scan_list[i]
        
        theta = threecases_raw_results_simu_angle[i]
        phi = threecases_raw_results_simu_phase[i]
        
        # Determine subplot position
        row, col = divmod(i, num_cols)
        ax = axes[row, col]

        # Plot filtered data
        ax.plot(theta, phi, "<", label=f"{scan}")
        ax.plot(theta, np.unwrap(phi), "-",linewidth=6,alpha=0.5, label=f"{scan} unwrap")
        ax.set_title(f"Phase: {scan}")
        ax.legend(fontsize=8)

    # Hide empty subplots if the number of scans is odd
    for j in range(num_scans, num_rows * num_cols):
        fig.delaxes(axes.flatten()[j])

    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()
#🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀
#------------------------------------------------------------------------------------------------------------
def plot_phase_data_comparison_exp_to_theo(exp_angle, exp_phase, theo_phases, labels, save_path=None,minus_theta=False,fact_minus=-1):
    """
    Plots experimental and theoretical phase data with a professional appearance.
    
    Parameters:
    exp_angle (array-like): Experimental angles.
    exp_phase (array-like): Experimental phase values.
    theo_phases (list of array-like): List of theoretical phase values for different cases.
    labels (list of str): Labels for the theoretical curves.
    save_path (str, optional): If provided, saves the plot to the specified path.
    """
    plt.figure(figsize=(8, 6))
    
    # Plot experimental data
    if minus_theta:
        y=center_angles(exp_phase+fact_minus*exp_angle)
    else:
        y=exp_phase
    plt.plot(exp_angle, y, "^", markersize=8, label="Experimental Data", color='black')
    
    # Plot theoretical data
    for i, theo_phase in enumerate(theo_phases):
        if minus_theta:
            y=center_angles(theo_phase+fact_minus*exp_angle)
        else:
            y=theo_phase
        plt.plot(exp_angle, y, "-", linewidth=3, label=labels[i])
    
    # Customizations
    plt.xlabel("Polar Angle (radians)", fontsize=14)
    plt.ylabel("Phase (radians)", fontsize=14)
    plt.title("Phase vs. Polar Angle", fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
#🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀
#------------------------------------------------------------------------------------------------------------
def plot_phase_data_comparison_combined_simu_exp_to_theo(exp_angle, exp_phase, theo_exp, 
                               simu_angles, simu_phases, theo_simu,
                               labels_exp, labels_simu, 
                               min_theta_exp=None, max_theta_exp=None, 
                               min_theta_simu=None, max_theta_simu=None,
                               save_path=None, minus_theta=False):
    """
    Plots experimental vs theoretical and simulation vs theoretical phase data in two subplots.
    If minus_theta is True, it removes the linear trend from each dataset and adds the trend equation to the legend.
    If min_theta and max_theta are provided, the cuts are applied separately (first min, then max).

    Theta cut is applied separately for experimental and simulation data.

    Parameters:
    exp_angle (array-like): Experimental angles.
    exp_phase (array-like): Experimental phase values.
    theo_exp (list of array-like): List of theoretical phase values for experimental cases.
    simu_angles (list of array-like): List of simulation angles for different cases.
    simu_phases (list of array-like): List of simulation phase values for different cases.
    theo_simu (list of array-like): List of theoretical phase values for simulation cases.
    labels_exp (list of str): Labels for the experimental theoretical curves.
    labels_simu (list of str): Labels for the simulation theoretical curves.
    min_theta_exp (float, optional): Minimum theta value for filtering experimental data.
    max_theta_exp (float, optional): Maximum theta value for filtering experimental data.
    min_theta_simu (float, optional): Minimum theta value for filtering simulation data.
    max_theta_simu (float, optional): Maximum theta value for filtering simulation data.
    save_path (str, optional): If provided, saves the plot to the specified path.
    minus_theta (bool, optional): If True, removes the linear fit from the data and adds the linear trend equation to the legend.
    """

    
    def remove_linear_fit(x, y):
        """Fits and removes a linear trend from the data while returning the trend equation."""
        coeffs = np.polyfit(x, y, 1)  # Linear fit (1st-degree polynomial)
        linear_trend = np.polyval(coeffs, x)  # Compute the trend
        equation = f"{coeffs[0]:.3f}x + {coeffs[1]:.3f}"  # Format equation as a string
        return y - linear_trend, equation  # Return detrended data and equation
    
    def apply_theta_cut_separately(x, y, min_theta, max_theta):
        """Applies separate cuts on min_theta and max_theta if they are not None."""
        if min_theta is not None:
            mask_min = x >= min_theta
            x, y = x[mask_min], y[mask_min]  # Apply min_theta cut
    
        if max_theta is not None:
            mask_max = x <= max_theta
            x, y = x[mask_max], y[mask_max]  # Apply max_theta cut
    
        return x, y  # Return filtered data
    fig, axs = plt.subplots(2, 1, figsize=(8, 12), sharex=True)
    colors = ["#8B0000", "#00008B", "#006400"]  # Dark red, dark blue, dark green

    # First subplot: Experimental vs Theoretical
    axs[0].set_title("Experimental vs Theoretical", fontsize=16, fontweight='bold')

    # Apply separate theta cuts for experimental data
    exp_angle_cut, exp_phase_cut = apply_theta_cut_separately(exp_angle, exp_phase, min_theta_exp, max_theta_exp)
    if minus_theta:
        y_exp, trend_exp_eq = remove_linear_fit(exp_angle_cut, exp_phase_cut)
    else:
        y_exp = exp_phase_cut
        trend_exp_eq = None

    axs[0].plot(exp_angle_cut, y_exp, "^", markersize=8, label=f"Experimental Data ({trend_exp_eq})", color='black')

    for i, label in enumerate(labels_exp):
        theo_angle_cut, theo_exp_cut = apply_theta_cut_separately(exp_angle, theo_exp[i], min_theta_exp, max_theta_exp)
        if minus_theta:
            y_theo_exp, trend_theo_exp_eq = remove_linear_fit(theo_angle_cut, theo_exp_cut)
        else:
            y_theo_exp = theo_exp_cut
            trend_theo_exp_eq = None
        axs[0].plot(theo_angle_cut, y_theo_exp, "-", linewidth=3, label=f"{label} (theo) [{trend_theo_exp_eq}]", color=colors[i])

    axs[0].set_ylabel("Phase (radians)", fontsize=14)
    axs[0].legend(fontsize=10, loc='best')
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # Second subplot: Simulation vs Theoretical
    axs[1].set_title("Simulation vs Theoretical", fontsize=16, fontweight='bold')

    for i, label in enumerate(labels_simu):
        simu_angle_cut, simu_phase_cut = apply_theta_cut_separately(simu_angles[i], simu_phases[i], min_theta_simu, max_theta_simu)
        theo_simu_angle_cut, theo_simu_cut = apply_theta_cut_separately(simu_angles[i], theo_simu[i], min_theta_simu, max_theta_simu)

        if minus_theta:
            y_simu, trend_simu_eq = remove_linear_fit(simu_angle_cut, simu_phase_cut)
            y_theo_simu, trend_theo_simu_eq = remove_linear_fit(theo_simu_angle_cut, theo_simu_cut)
        else:
            y_simu = simu_phase_cut
            y_theo_simu = theo_simu_cut
            trend_simu_eq = None
            trend_theo_simu_eq = None

        axs[1].plot(simu_angle_cut, y_simu, "+", markersize=6, label=f"{label} (simu) [{trend_simu_eq}]", color=colors[i])
        axs[1].plot(theo_simu_angle_cut, y_theo_simu, "-", linewidth=3, label=f"{label} (theo) [{trend_theo_simu_eq}]", color=colors[i])

    axs[1].set_xlabel("Polar Angle (radians)", fontsize=14)
    axs[1].set_ylabel("Phase (radians)", fontsize=14)
    axs[1].legend(fontsize=10, loc='best')
    axs[1].grid(True, linestyle='--', alpha=0.6)

    # Adjust layout
    plt.tight_layout()

    # Save plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
#------------------------------------------------------------------------------------------------------------
def fit_phase_correction_twodataset(grid_points, phase, predicted_phase_3d, radial_distances_3d,polar_angles_3d, grid_points_dislo,
                         r_range=(3.0, 5.0), z_range=(-5, 5.0), num_points=8, theta_exclude=(-0.2, 3),
                         plot_debug=True, save_path=None, show_plots=True,num_trials = 10,plot_points_positions=False):
    """
    Fit a linear model to the phase difference between simulated and theoretical phase data.
    Computes the correction needed to adjust the phase.
    """

    def select_non_collinear_indices(radial_distances_3d, z_prime_values, polar_angles_3d, 
                                     r_range, z_range, theta_exclude=(-0.2, 0.2), num_points=4, max_attempts=100):
        """
        Selects non-collinear random indices where:
        - Radial distance is within [rmin, rmax]
        - Z' is within [z_min, z_max]
        - Polar angle (θ) is outside the excluded range (e.g., not between -0.2 and 0.2)
        """
        r_min, r_max = r_range
        z_min, z_max = z_range
        theta_min, theta_max = theta_exclude
    
        # Create mask based on conditions
        mask = (radial_distances_3d >= r_min) & (radial_distances_3d <= r_max) & \
               (z_prime_values >= z_min) & (z_prime_values <= z_max) & \
               ~((polar_angles_3d >= theta_min) & (polar_angles_3d <= theta_max))  # Exclude theta range
    
        # Get indices where conditions are met
        valid_indices = list(zip(*np.where(mask)))
    
        # Ensure we have enough points to choose from
        if len(valid_indices) < num_points:
            raise ValueError("Not enough points available in the specified range to select from.")
    
        # Try multiple attempts to find non-collinear points
        for attempt in range(max_attempts):
            # Randomly select num_points indices
            selected_indices = np.random.choice(len(valid_indices), num_points, replace=False)
            selected_indices = [valid_indices[i] for i in selected_indices]
    
            # Extract coordinates
            points = np.array([grid_points[i, j, k] for i, j, k in selected_indices])  # Shape (num_points,3)
    
            # Check if the points are collinear by computing the volume of the tetrahedron formed
            if num_points >= 4:
                vec1 = points[1] - points[0]
                vec2 = points[2] - points[0]
                vec3 = points[3] - points[0]
    
                volume = np.abs(np.dot(vec1, np.cross(vec2, vec3))) / 6  # Volume of tetrahedron
    
                if volume > 1e-6:  # Threshold to avoid near-collinear points
                    return selected_indices
            else:
                return selected_indices  # If fewer than 4 points, return the selection directly
    
        raise ValueError("Failed to find sufficient non-collinear points after multiple attempts.")
    def plot_phase_for_selected_points(phase, selected_indices,show_plots, title_prefix,save_path):
        """
        Plots phase slices for each selected point in separate subplots arranged in multiple rows.
    
        Parameters:
        - phase: ndarray, the phase data to be visualized.
        - selected_indices: list of tuples, selected points (i, j, k) in the dataset.
        - title_prefix: str, prefix for subplot titles.
        """
        num_points = len(selected_indices)
        num_cols = 3  # Number of columns per row
        num_rows = (num_points // num_cols) + (1 if num_points % num_cols != 0 else 0)
    
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
        axes = np.array(axes).reshape(-1)  # Flatten axes for easy iteration
    
        for i, (i_idx, j_idx, k_idx) in enumerate(selected_indices):
            phase_slice = phase[i_idx, :, :]
    
            im = axes[i].imshow(phase_slice, cmap='jet', origin='lower', interpolation='nearest')
            axes[i].scatter(k_idx, j_idx, color='white', edgecolor='black', marker='o', s=100)
            axes[i].set_title(f"{title_prefix} {i_idx, j_idx, k_idx}")
            axes[i].set_xlabel("X Index")
            axes[i].set_ylabel("Y Index")
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
        # Hide any unused subplots
        for i in range(num_points, len(axes)):
            fig.delaxes(axes[i])
    
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path+"_seleceted_points_for_phasecorrection.png", dpi=300)
        if show_plots:
            plt.show()
        else:
            plt.close()

    
    
    from scipy.linalg import lstsq
    from sklearn.linear_model import RANSACRegressor

    coeffs_list = []
    
    for _ in range(num_trials):
        # Re-select different non-collinear points
        selected_indices = select_non_collinear_indices(radial_distances_3d, grid_points_dislo[..., 2],
                                                        polar_angles_3d, r_range, z_range,
                                                        theta_exclude=theta_exclude, num_points=num_points)
    
        X, Y, Z, Delta_Phi = [], [], [], []
    
        for i, j, k in selected_indices:
            x, y, z = grid_points[i, j, k]
            delta_phi = phase[i, j, k] - predicted_phase_3d[i, j, k]
            X.append(x)
            Y.append(y)
            Z.append(z)
            Delta_Phi.append(delta_phi)
        A = np.vstack([X, Y, Z, np.ones(len(X))]).T
        
        # Fit RANSAC robust regression model
        ransac = RANSACRegressor(min_samples=3, residual_threshold=0.05, max_trials=100)
        ransac.fit(A[:, :3], Delta_Phi)  # Fit only x, y, z (exclude bias term for now)
        # Retrieve robust coefficients
        coeffs = np.append(ransac.estimator_.coef_, ransac.estimator_.intercept_)
    
        #coeffs, _, _, _ = lstsq(A, Delta_Phi)
    
        coeffs_list.append(coeffs)
    
    # Compute the final averaged coefficients
    coeffs = np.mean(coeffs_list, axis=0)

    if plot_points_positions:
        # Plot phase images for selected points
        if save_path:
            save_path_0=save_path+"_Simulated_"
        plot_phase_for_selected_points(phase, selected_indices,show_plots, "Simulated Phase with Selected Points",save_path_0)
        if save_path:
            save_path_0=save_path+"_Theo_"
        plot_phase_for_selected_points(predicted_phase_3d, selected_indices,show_plots, "Theoretical Phase with Selected Points",save_path_0)

    # Compute residuals
    residuals = Delta_Phi - A @ coeffs
    sigma_squared = np.sum(residuals**2) / (len(X) - len(coeffs))

    # Compute covariance matrix with regularization
    cov_matrix = sigma_squared * np.linalg.inv(A.T @ A + np.eye(A.shape[1]) * 1e-6)
    errors = np.sqrt(np.diag(cov_matrix))

    # Compute phase correction across the entire dataset
    phase_correction = (coeffs[0] * grid_points[..., 0] + coeffs[1] * grid_points[..., 1] + 
                        coeffs[2] * grid_points[..., 2] + coeffs[3])

    # Print results as a table
    results_table = pd.DataFrame({
        "Parameter": ["a0", "b0", "c0", "d0"],
        "Value": np.round(coeffs,4),
        "Error": np.round(errors,4)
    })
    
    # Compute MAE and RMSE before correction
    mae_before = np.mean(np.abs(phase - predicted_phase_3d))
    rmse_before = np.sqrt(np.mean((phase - predicted_phase_3d) ** 2))
    
    # Compute MAE and RMSE after correction
    corrected_phase = phase - phase_correction
    mae_after = np.mean(np.abs(corrected_phase - predicted_phase_3d))
    rmse_after = np.sqrt(np.mean((corrected_phase - predicted_phase_3d) ** 2))
    
    # Print accuracy results as a table
    accuracy_table = pd.DataFrame({
        "Metric": ["MAE", "RMSE"],
        "Before Correction": [mae_before, rmse_before],
        "After Correction": [mae_after, rmse_after]
    })
    
    print("\nAccuracy Improvement:")
    print(accuracy_table)
    
    if plot_debug:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract phase values for the selected points
        simu_values = np.array([phase[i, j, k] for (i, j, k) in selected_indices])
        theo_values = np.array([predicted_phase_3d[i, j, k] for (i, j, k) in selected_indices])
        corrected_values = np.array([phase[i, j, k] - phase_correction[i, j, k] for (i, j, k) in selected_indices])

        x_labels = [str(idx) for idx in selected_indices]
        x_range = np.arange(len(selected_indices))

        # Plot all values in a single subplot
        ax.plot(x_range, simu_values, 'o-', color='blue', label="Simulated Phase")
        ax.plot(x_range, theo_values, 's-', color='green', label="Theoretical Phase")
        ax.plot(x_range, corrected_values, 'd-', color='red', label="Corrected Phase")

        ax.set_title("Phase Comparison at Selected Points")
        ax.set_xticks(x_range)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_ylabel("Phase Value")
        ax.legend()

        # Create a table next to the plot
        fig.subplots_adjust(right=1.)
        table_ax = fig.add_axes([1.05, 0.15, 0.5, 0.1])  # Position of the table
        table_ax.axis("tight")
        table_ax.axis("off")
        table = table_ax.table(cellText=results_table.values,
                               colLabels=results_table.columns,colWidths=[.25, .25, .25, .25, .25],
                               cellLoc="center",
                               loc="center")
        
        table.auto_set_column_width(False)
        table.auto_set_font_size(True)
        table.auto_set_font_size(False)
        # Increase font size
        table.scale(1.5,1.5)
        table.set_fontsize(20)

        
        plot_ax = fig.add_axes([1.1, 0.45, 0.5, 0.4])  # Position of the table

        metrics = ["MAE", "RMSE"]
        before_values = [mae_before, rmse_before]
        after_values = [mae_after, rmse_after]
    
        x_pos = np.arange(len(metrics))  # Get positions
        width = 0.4  # Bar width
        
        plot_ax.bar(x_pos - width/2, before_values, width, color="red", alpha=0.6, label="Before Correction")
        plot_ax.bar(x_pos + width/2, after_values, width, color="green", alpha=0.6, label="After Correction")
        
        plot_ax.set_xticks(x_pos)
        plot_ax.set_xticklabels(metrics)

        plot_ax.set_title("Phase Correction Accuracy")
        plot_ax.set_ylabel("Error Value")
        plot_ax.legend()
        

        
        if save_path:
            plt.savefig(save_path+"_summary_phase_correction.png", dpi=300)
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
    return coeffs, errors, phase_correction


