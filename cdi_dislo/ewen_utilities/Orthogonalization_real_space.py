import numpy as np
import pylab as plt

from cdi_dislo.ewen_utilities.plot_utilities import (
    plot_2D_slices_middle_one_array3D,
    plot_2D_slices_middle,
)
from cdi_dislo.ewen_utilities.Object_utilities import (
    automatic_object_roi,
    center_object,
)
from cdi_dislo.ewen_utilities.Global_utilities import (
    center_of_mass,
    rotation_matrix_from_vectors,
)
from scipy.interpolate import RegularGridInterpolator

######################################################################################################################################
###########################             Compute B natrix in reciprocal space               ###########################################
######################################################################################################################################


def crop_around_Bragg_center_compute_index_cen(
    data, qx, qy, qz, size=20, plot=False
):
    index_cen = np.round(center_of_mass(data)).astype(int)
    roi = [
        index_cen[0] - size,
        index_cen[0] + size + 1,
        index_cen[1] - size,
        index_cen[1] + size + 1,
        index_cen[2] - size,
        index_cen[2] + size + 1,
    ]

    if plot:
        from cdi_dislo.ewen_utilities.plot_utilities import plot_3D_projections

        fig, ax = plt.subplots(2, 3, figsize=(9, 6))
        plot_3D_projections(data, fig=fig, ax=ax[0])
        plot_3D_projections(
            data[roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]],
            fig=fig,
            ax=ax[1],
        )

    data = data[roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]]

    qx = qx[roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]]
    qy = qy[roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]]
    qz = qz[roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]]

    index_cen = np.round(center_of_mass(data)).astype(int)
    if plot:
        ax[1, 0].scatter(index_cen[2], index_cen[1], color="w")
        ax[1, 1].scatter(index_cen[2], index_cen[0], color="w")
        ax[1, 2].scatter(index_cen[1], index_cen[0], color="w")

    return data, qx, qy, qz, index_cen


def compute_Q_K_matrices(qx, qy, qz, index_cen):
    # Q matrix
    qx_cen = qx - qx[index_cen[0], index_cen[1], index_cen[2]]
    qy_cen = qy - qy[index_cen[0], index_cen[1], index_cen[2]]
    qz_cen = qz - qz[index_cen[0], index_cen[1], index_cen[2]]
    Q = np.array([qx_cen.flatten(), qy_cen.flatten(), qz_cen.flatten()])

    # K matrix
    i, j, k = np.indices(qx.shape)
    # Same as Q matrix, I need to center the indexes for the K matrix
    i = i - i[index_cen[0], index_cen[1], index_cen[2]]
    j = j - j[index_cen[0], index_cen[1], index_cen[2]]
    k = k - k[index_cen[0], index_cen[1], index_cen[2]]
    K = np.array([i.flatten(), j.flatten(), k.flatten()])

    return Q, K


def calculate_B_matrix_error(B_reciprocal, K, Q, data_shape):
    # Linear approximation of my Q matrix (Q was calculated with xrayutilities)
    Qb = np.dot(B_reciprocal, K)

    error = np.sqrt(
        np.sum((Q - Qb) ** 2.0, axis=0)
    )  # I just look at the module of the error vector.
    error = np.reshape(
        error, data_shape
    )  # Need to reshape that in the 3D format
    return error


def compute_reciprocal_space_B_matrix(file, rotation_matrix=None, plot=False):
    # q vector positions, calculated by xrayutilities
    qx = file["qx"]
    qy = file["qy"]
    qz = file["qz"]

    if rotation_matrix is not None:
        q = np.array([qx, qy, qz])
        q = np.moveaxis(q, 0, -1)
        q = np.dot(q, rotation_matrix.T)
        qx, qy, qz = q[..., 0], q[..., 1], q[..., 2]

    # diffracted intensity taken from recon
    data = np.load(str(file["preprocessed_datapath"]))["data"]

    # Get index center of mass (after cropping)
    #     data_crop, qx_crop, qy_crop, qz_crop, index_cen = crop_around_Bragg_center_compute_index_cen(data, qx, qy, qz,
    #                                                              size=20,
    #                                                              plot=plot)
    #     Q, K = compute_Q_K_matrices(qx_crop ,qy_crop, qz_crop, index_cen)

    index_cen = np.round(center_of_mass(data)).astype(int)
    Q, K = compute_Q_K_matrices(qx, qy, qz, index_cen)
    B_reciprocal = np.dot(Q, np.dot(K.T, np.linalg.inv(np.dot(K, K.T))))

    error = calculate_B_matrix_error(B_reciprocal, K, Q, data.shape)

    if plot:
        # plot the relative error (regarding the norm of q) in percent
        q_norm = np.sqrt(qx**2.0 + qy**2.0 + qz**2.0)
        plot_2D_slices_middle_one_array3D(
            100 * error / q_norm,
            fig_title="|q| linear approximation error (relative in %)",
        )

    return B_reciprocal, K, error


def compute_positions_inverse_matrix(
    file_ref, rotation_matrix=None, plot=False
):
    shape = file_ref["qx"].shape
    B_reciprocal, K, error = compute_reciprocal_space_B_matrix(
        file_ref, rotation_matrix=rotation_matrix, plot=plot
    )

    D = np.diag(2 * np.pi / np.array(shape))
    B_real = np.dot(np.linalg.inv(B_reciprocal).T, D)

    R = np.dot(B_real, K)
    R = np.reshape(R, (3,) + shape)

    B_real_inv = np.linalg.inv(B_real)
    return R, B_real_inv


######################################################################################################################################
###########################                 Orthogonalization function                     ###########################################
######################################################################################################################################


def compute_automatic_voxel_sizes(file_ref):
    voxel_sizes = np.zeros(3)
    for n, key in enumerate(["qx", "qy", "qz"]):
        q = file_ref[key]
        range_q = np.nanmax(q) - np.nanmin(q)
        voxel_sizes[n] += 2.0 * np.pi / range_q
    return voxel_sizes


def compute_matrix_to_interpolate(R, B_real_inv, voxel_sizes):
    R_grid = np.meshgrid(
        np.arange(np.min(R[0]), np.max(R[0]), voxel_sizes[0]),
        np.arange(np.min(R[1]), np.max(R[1]), voxel_sizes[1]),
        np.arange(np.min(R[2]), np.max(R[2]), voxel_sizes[2]),
        indexing="ij",
    )
    R_grid = np.array(R_grid)
    shape = R_grid.shape

    # Flatten it for the matrix product
    R_grid = np.reshape(R_grid, (3, np.prod(R_grid.shape[1:])))

    # "indices" matrices where the interpolation should be done (M is floats, maybe my explenation is not very clear)
    M = np.dot(B_real_inv, R_grid)
    M = np.reshape(M, shape)
    return M


def real_space_orthogonalization(
    obj,
    file_ref,
    rotate_bragg_to_last_axis=False,
    rotation_matrix=None,
    voxel_sizes=None,
    voxel_sizes_cubic=False,
    final_roi=True,
    final_centering=True,
    verbose=True,
    plot=False,
):
    qcen = file_ref["qcen"]
    if rotate_bragg_to_last_axis:
        print("rotate axes such that the Bragg is along the last axis")
        e_bragg = qcen / np.linalg.norm(qcen)
        ez = np.array([0, 0, 1])
        rotation_matrix = rotation_matrix_from_vectors(e_bragg, ez)
        qcen = np.dot(rotation_matrix, qcen)

    R, B_real_inv = compute_positions_inverse_matrix(
        file_ref, rotation_matrix=rotation_matrix, plot=False
    )

    # No need to interpolate far outside the object
    roi = automatic_object_roi(obj, plot=False)
    R = R[:, roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]]

    # Create interpolation function
    rgi = RegularGridInterpolator(
        (
            np.arange(-obj.shape[0] // 2, obj.shape[0] // 2, 1),
            np.arange(-obj.shape[1] // 2, obj.shape[1] // 2, 1),
            np.arange(-obj.shape[2] // 2, obj.shape[2] // 2, 1),
        ),
        obj,
        method="linear",
        bounds_error=False,
        fill_value=0,
    )

    # Compute automatic interpolation voxel sizes
    if voxel_sizes is None:
        voxel_sizes = compute_automatic_voxel_sizes(file_ref)
        if voxel_sizes_cubic:
            print("force voxel size to be the same along all direction")
            voxel_sizes = np.ones(3) * min(voxel_sizes)
    else:
        if not hasattr(voxel_sizes, "__len__"):
            voxel_sizes = [voxel_sizes for n in range(obj.ndim)]
    if verbose:
        print("voxel size (Angstrom) : ", voxel_sizes)

    # Calculate the "indices" matrix to interpolate
    M = compute_matrix_to_interpolate(R, B_real_inv, voxel_sizes)

    obj_ortho = rgi((M[0], M[1], M[2]), method="linear")

    if final_centering:
        obj_ortho = center_object(obj_ortho)

    if final_roi:
        roi = automatic_object_roi(obj_ortho, plot=False)
        obj_ortho = obj_ortho[
            roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]
        ]

    if plot:
        plot_2D_slices_middle(obj, fig_title="before orthogonalization", fw=3)
        plot_2D_slices_middle(
            obj_ortho,
            fig_title="after orthogonalization (uncropped)",
            fw=3,
            crop=False,
            voxel_sizes=voxel_sizes,
        )

    return obj_ortho, voxel_sizes, qcen


# def compute_real_space_positions(B_reciprocal, K, obj):
# #     D = np.diag([1./obj.shape[0], 1./obj.shape[1], 1./obj.shape[2]]) # maybe a 2*pi factor missing here
#     D = np.diag(2 * np.pi / np.array(obj.shape))
#     B_real = np.dot(np.linalg.inv(B_reciprocal).T, D)

#     R = np.dot(B_real,K) # real space position for each pixels in the object matrix
#     R = np.reshape(R, (3,)+obj.shape)
#     rx, ry, rz = R[0], R[1], R[2]
#     return rx, ry, rz

# ######################################################################################################################################
# ###########################                 Orthogonalization function                     ###########################################
# ######################################################################################################################################

# def real_space_orthogonalization(obj, file_ref,
#                                  plot=False):
#     B_reciprocal, K, error = compute_reciprocal_space_B_matrix(file_ref,
#                                           plot=plot)
#     rx, ry, rz = compute_real_space_positions(B_reciprocal, K, obj)

#     # Get module and phase without putting nan's out of the support
#     module, phase = get_cropped_module_phase(obj, crop=False, support = np.ones(obj.shape))

#     module_ortho, _, _, _ = interpolation_xrayutilities_gridder(rx,ry,rz, module)
#     phase_ortho, rx1d, ry1d, rz1d = interpolation_xrayutilities_gridder(rx,ry,rz, phase)
#     obj_ortho = module_ortho * np.exp(1.0j*phase_ortho)
#     voxel_sizes = [rx1d[1]-rx1d[0], ry1d[1]-ry1d[0], rz1d[1]-rz1d[0]]

#     if plot:
#         plot_2D_slices_middle(obj_ortho, voxel_sizes=voxel_sizes, fig_title='orthogonalized object')

#     return obj_ortho, voxel_sizes
