import numpy as np
import pylab as plt


from cdi_dislo.ewen_utilities.plot_utilities import (
    plot_2D_slices_middle_one_array3D,
    plot_symmetric_colorscale_2d,
    plot_global_2d,
)
from cdi_dislo.ewen_utilities.Object_utilities import get_cropped_module_phase

######################################################################################################################################
#########################            My custom gradient function (to keep surfaces)              #####################################
######################################################################################################################################


def EB_custom_gradient(array, voxel_sizes=None):
    """
    Should work for any array dimensions
    """

    grad = np.zeros((array.ndim,) + array.shape)
    for n in range(array.ndim):
        slice1 = [slice(None) for n in range(array.ndim)]
        slice1[n] = slice(1, None)

        slice2 = [slice(None) for n in range(array.ndim)]
        slice2[n] = slice(None, -1)

        grad_n = array[tuple(slice1)] - array[tuple(slice2)]
        grad_n = np.nanmean(
            [grad_n[tuple(slice1)], grad_n[tuple(slice2)]], axis=0
        )

        #         padding = np.zeros((array.ndim, array.ndim)).astype('int')
        padding = np.zeros((array.ndim, 2)).astype("int")
        padding[n] += 1
        padding = tuple(map(tuple, padding))
        grad_n = np.pad(grad_n, padding, "constant", constant_values=(np.nan))

        if voxel_sizes is not None:
            grad_n = grad_n / voxel_sizes[n]

        grad[n] += grad_n

    return grad


# def compute_strain(obj, path_reconstruction,
#                    threshold_module=.3,
#                    verbose=True,
#                    plot=False):

#     for file in sorted(os.listdir(path_reconstruction)):
#         if 'reconstruction' in file:
#             data = np.load(path_reconstruction+file)
#             if verbose :
#                 print('q and q_cen taken from : ', path_reconstruction+file)
#             break

#     if obj.ndim ==2 :
#         if verbose:
#             print('Careful !! The Bragg should be in the (x,y) plane otherwise this is wrong !!!')
#         direction_string = ['x', 'y']
#     if obj.ndim ==3 :
#         direction_string = ['x', 'y', 'z']

#     q = [data['q{}'.format(direction)] for direction in direction_string]
#     q_cen = [data['q{}_cen'.format(direction)] for direction in direction_string]

#     voxel_sizes = [np.fft.fftfreq(len(q[n]), d=q[n][1]-q[n][0]) for n in range(obj.ndim)]
#     voxel_sizes = [voxel[1] - voxel[0] for voxel in voxel_sizes]

#     if verbose :
#         for n in range(obj.ndim):
#             print('voxel size along {} :'.format(direction_string[n]),voxel_sizes[n])

#     module, phase = get_cropped_module_phase(obj, unwrap=True, threshold_module=threshold_module)

#     grad = EB_custom_gradient(phase, voxel_sizes = voxel_sizes)

#     strain = np.sum([q_cen[n] * grad[n] for n in range(obj.ndim)], axis=0)
#     strain = strain / np.sum([q_cen[n]**2. for n in range(obj.ndim)], axis=0)

#     if plot :

#         fig,ax = plt.subplots(figsize=(4,4))
#         plot_symmetric_colorscale(strain, fig=fig, ax=ax, cmap='bwr')
#         ax.set_title('strain', fontsize=20)

#     return strain, voxel_sizes


def compute_strain(
    obj_ortho,
    voxel_sizes,
    q_cen,
    crop=True,
    use_negative_phase=True,  # Careful to the sign of the strain regarding the phase. There's a minus sign !!!!!
    threshold_module=0.3,
    unwrap=True,
    defects=False,
    nb_shift=1,
    phase_shift=np.pi / 2.0,  # defect part
    verbose=True,
    plot=False,
):
    if verbose:
        print("voxel size : {}".format(voxel_sizes))

    if defects:
        print(
            f"defect_computation taken into account with\nphase shift : {phase_shift}\nnb_shift : {nb_shift}"
        )
        strain, d_spacing = compute_strain_defects(
            obj_ortho,
            voxel_sizes,
            q_cen,
            crop=crop,
            use_negative_phase=use_negative_phase,
            threshold_module=threshold_module,
            phase_shift=phase_shift,
            nb_shift=nb_shift,
            plot=plot,
        )
        return strain, d_spacing

    module, phase = get_cropped_module_phase(
        obj_ortho, unwrap=unwrap, threshold_module=threshold_module, crop=crop
    )

    if use_negative_phase:
        phase = -phase  # We need that due to the definition of numpy fft used in pynx !!!!!!!!!!!!

    grad = EB_custom_gradient(phase, voxel_sizes=voxel_sizes)

    strain = np.sum(
        [q_cen[n] * grad[n] for n in range(obj_ortho.ndim)], axis=0
    )
    strain = strain / np.sum(
        [q_cen[n] ** 2.0 for n in range(obj_ortho.ndim)], axis=0
    )

    d_cen = 2.0 * np.pi / np.sqrt(np.sum(q_cen**2.0, axis=0))
    d_spacing = d_cen * (1.0 + strain)

    if plot:
        plot_2D_slices_middle_one_array3D(
            100 * strain,
            symmetric_colorscale=True,
            fig_title="strain (%)",
            voxel_sizes=voxel_sizes,
            fw=4,
        )

        plot_2D_slices_middle_one_array3D(
            d_spacing,
            symmetric_colorscale=False,
            fig_title="d spacing",
            cmap="coolwarm",
            voxel_sizes=voxel_sizes,
            fw=4,
        )
    return strain, d_spacing


######################################################################################################################################
###################                         strain in the presence of defects                            #############################
######################################################################################################################################


def compute_strain_defects(
    obj_ortho,
    voxel_sizes,
    q_cen,
    crop=False,
    use_negative_phase=True,
    threshold_module=0.3,
    phase_shift=np.pi / 2.0,
    nb_shift=1,
    plot=False,
):
    kwargs = {
        "crop": crop,
        "use_negative_phase": use_negative_phase,
        "threshold_module": threshold_module,
        "unwrap": False,
        "verbose": False,
        "plot": False,
    }
    strain, _ = compute_strain(obj_ortho, voxel_sizes, q_cen, **kwargs)

    obj_ortho_shift = np.abs(obj_ortho) * np.exp(
        1.0j * (np.angle(obj_ortho) + phase_shift)
    )
    strain2, _ = compute_strain(obj_ortho_shift, voxel_sizes, q_cen, **kwargs)
    del obj_ortho_shift

    obj_ortho_shift = np.abs(obj_ortho) * np.exp(
        1.0j * (np.angle(obj_ortho) - phase_shift)
    )
    strain3, _ = compute_strain(obj_ortho_shift, voxel_sizes, q_cen, **kwargs)
    del obj_ortho_shift

    strain_clean = np.zeros(strain.shape)
    strain_clean[np.isnan(strain)] = np.nan

    mask = np.isclose(strain, strain2)
    strain_clean[mask == 1] = strain[mask == 1]

    mask = np.isclose(strain, strain3)
    strain_clean[mask == 1] = strain[mask == 1]

    mask = np.isclose(strain2, strain3)
    strain_clean[mask == 1] = strain2[mask == 1]

    d_cen = 2.0 * np.pi / np.sqrt(np.sum(q_cen**2.0, axis=0))
    d_spacing = d_cen * (1.0 + strain_clean)

    if plot:
        plot_2D_slices_middle_one_array3D(
            100 * strain_clean,
            symmetric_colorscale=True,
            fig_title="strain (%)",
            voxel_sizes=voxel_sizes,
            fw=4,
        )

        plot_2D_slices_middle_one_array3D(
            d_spacing,
            symmetric_colorscale=False,
            fig_title="d spacing",
            cmap="coolwarm",
            voxel_sizes=voxel_sizes,
            fw=4,
        )

    return strain_clean, d_spacing


######################################################################################################################################
#########################                         Zlotnikov 2D strain                            #####################################
######################################################################################################################################


def compute_strain_2D_Zlotnikov(
    obj_ortho,
    voxel_sizes,
    q_cen,
    crop=False,
    use_negative_phase=True,  # Careful to the sign of the strain regarding the phase. There's a minus sign !!!!!
    threshold_module=0.3,
    verbose=True,
    plot=False,
):
    if verbose:
        print("voxel size : {}".format(voxel_sizes))

    module, phase = get_cropped_module_phase(
        obj_ortho, unwrap=True, threshold_module=threshold_module, crop=crop
    )

    if use_negative_phase:
        phase = -phase  # We need that due to the definition of numpy fft used in pynx !!!!!!!!!!!!

    grad = EB_custom_gradient(phase, voxel_sizes=voxel_sizes)

    strain = np.sum(
        [q_cen[n] * grad[n] for n in range(obj_ortho.ndim)], axis=0
    )
    strain = strain / np.sum(
        [q_cen[n] ** 2.0 for n in range(obj_ortho.ndim)], axis=0
    )

    d_cen = 2.0 * np.pi / np.sqrt(np.sum(q_cen**2.0, axis=0))
    d_spacing = d_cen * (1.0 + strain)

    if plot:
        fw = 4
        fig, ax = plt.subplots(1, 2, figsize=(fw * 2, fw))
        plot_symmetric_colorscale_2d(
            100 * strain,
            fig_title="strain (%)",
            voxel_sizes=voxel_sizes,
            fig=fig,
            ax=ax[0],
        )
        plot_global_2d(
            d_spacing,
            fig_title="d spacing",
            voxel_sizes=voxel_sizes,
            fig=fig,
            ax=ax[1],
            cmap="coolwarm",
        )
    return strain, d_spacing
