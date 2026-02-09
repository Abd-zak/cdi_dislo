import numpy as np

from scipy.interpolate import RegularGridInterpolator

from cdi_dislo.ewen_utilities.plot_utilities import (
    plot_2D_slices_middle_one_array3D,
)
from cdi_dislo.ewen_utilities.Orthogonalization_real_space import (
    compute_positions_inverse_matrix,
)

######################################################################################################################################
##################################          Transfer support function using interpolation           ##################################
######################################################################################################################################


def transfer_support(
    filename_obj_ortho_ref,
    preprocessed_datapath,
    threshold_module=0.3,
    plot=False,
):
    """
    Create a support from an orthogonalized reconstruction of the same particle (from any other Bragg if you want)
    :filename_obj_ortho_ref: file path of the reconstructed orthogonalized object
    :preprocessed_datapath: preprocessed BCDI data path that you want to reconstruct
    :threshold_module: between 0 and 1. Adjust this value to create support from the reconstructed object modulus.
    :plot: plot to check you support.
    """
    # Load reference support and voxel_size
    file = np.load(filename_obj_ortho_ref)
    obj = file["obj_ortho"]
    voxel_sizes_sup = file["voxel_sizes"]

    # Create orthogonal support from the object modulus
    module = np.abs(obj)
    support = module > threshold_module * np.max(module)

    # Create interpolation function
    rgi = RegularGridInterpolator(
        (
            np.arange(-support.shape[0] // 2, support.shape[0] // 2, 1)
            * voxel_sizes_sup[0],
            np.arange(-support.shape[1] // 2, support.shape[1] // 2, 1)
            * voxel_sizes_sup[1],
            np.arange(-support.shape[2] // 2, support.shape[2] // 2, 1)
            * voxel_sizes_sup[2],
        ),
        support,
        method="linear",
        bounds_error=False,
        fill_value=0,
    )

    # Create transferred support in non-orthogonal space of the preprocessed_datapath BCDI data
    file = np.load(preprocessed_datapath)
    file = dict(file)
    file["preprocessed_datapath"] = preprocessed_datapath
    R, _ = compute_positions_inverse_matrix(file)
    support_new = rgi(
        (R[0], R[1], R[2]), method="linear"
    )  # Your transferred support

    if plot:
        plot_2D_slices_middle_one_array3D(
            support_new, cmap="gray_r", fig_title="created support"
        )
    return support_new


######################################################################################################################################
##################################                     Support modifications                        ##################################
######################################################################################################################################


def fill_up_support(support, plot=False):
    """
    Modify the support by filling any hole inside.
    """

    support_convex = np.zeros(support.shape)
    for axis in range(support.ndim):
        support_cum = np.cumsum(support, axis=axis)
        support_cum_inv = np.flip(
            np.cumsum(np.flip(support, axis=axis), axis=axis), axis=axis
        )
        support_combine = support_cum * support_cum_inv
        support_convex[support_combine != 0] = 1

    if plot:
        plot_2D_slices_middle_one_array3D(
            support, cmap="gray_r", fig_title="original support"
        )
        plot_2D_slices_middle_one_array3D(
            support_convex, cmap="gray_r", fig_title="filled support"
        )

    return support_convex
