import numpy as np
import pylab as plt


from cdi_dislo.ewen_utilities.plot_utilities import (
    plot_2D_slices_middle_one_array3D,
    plot_global_2d,
    plot_symmetric_colorscale_2d,
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
        grad_n = np.nanmean([grad_n[tuple(slice1)], grad_n[tuple(slice2)]], axis=0)

        #         padding = np.zeros((array.ndim, array.ndim)).astype('int')
        padding = np.zeros((array.ndim, 2)).astype("int")
        padding[n] += 1
        padding = tuple(map(tuple, padding))
        grad_n = np.pad(grad_n, padding, "constant", constant_values=(np.nan))

        if voxel_sizes is not None:
            grad_n = grad_n / voxel_sizes[n]

        grad[n] += grad_n

    return grad


######################################################################################################################################
#########################                  Gradient with topological defects                     #####################################
######################################################################################################################################


def EB_custom_gradient_defect(phase, qcen, voxel_sizes=None, phase_shift=np.pi / 2.0):
    grad_shift_list = []
    for n in [0, 1, 2]:
        shifted_phase_map = np.mod(phase + n * phase_shift, 2.0 * np.pi)
        grad_shift_list.append(
            EB_custom_gradient(shifted_phase_map, voxel_sizes=voxel_sizes)
        )

    grad_clean = np.zeros(grad_shift_list[0].shape)
    grad_clean[np.isnan(grad_shift_list[0])] = np.nan

    for indices in [[0, 1], [0, 2], [1, 2]]:
        mask = np.isclose(grad_shift_list[indices[0]], grad_shift_list[indices[1]])
        grad_clean[mask == 1] = grad_shift_list[indices[0]][mask == 1]

    if qcen is not None:
        grad_clean = grad_clean / np.linalg.norm(qcen)

    return grad_clean


######################################################################################################################################
#########################                       Displacement and gradient                        #####################################
######################################################################################################################################


def displacement_and_gradient(
    obj_ortho,
    qcen,
    voxel_sizes=None,
    unwrap=True,
    crop=False,
    threshold_module=0.3,
    use_negative_phase=True,
    defect=False,
    phase_shift=np.pi / 2.0,
    plot=True,
):
    _, phase = get_cropped_module_phase(
        obj_ortho, unwrap=unwrap, threshold_module=threshold_module, crop=crop
    )
    if use_negative_phase:
        phase = (
            -phase
        )  # We need that due to the definition of numpy fft used in pynx !!!!!!!!!!!!

    displacement = phase / np.linalg.norm(qcen)
    displacement = displacement - np.nanmean(
        displacement
    )  # To remove a constant in the displacement for later plots

    if not defect:
        grad = EB_custom_gradient(displacement, voxel_sizes=voxel_sizes)
    else:
        print("take into account defects")
        grad = EB_custom_gradient_defect(
            phase, qcen, voxel_sizes=voxel_sizes, phase_shift=phase_shift
        )

    if plot:
        if obj_ortho.ndim == 3:
            plot_2D_slices_middle_one_array3D(
                displacement, cmap="turbo", fig_title=r"displacement ($\AA$)"
            )
            plot_2D_slices_middle_one_array3D(
                grad[0],
                cmap="coolwarm",
                fig_title=r"gradient 1$^{st}$ component (no units)",
            )
            plot_2D_slices_middle_one_array3D(
                grad[1],
                cmap="coolwarm",
                fig_title=r"gradient 2$^{nd}$ component (no units)",
            )
            plot_2D_slices_middle_one_array3D(
                grad[2],
                cmap="coolwarm",
                fig_title=r"gradient 3$^{rd}$ component (no units)",
            )
        if obj_ortho.ndim == 2:
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            plot_global_2d(
                displacement,
                fig=fig,
                ax=ax[0],
                fig_title=r"displacement ($\AA$)",
                voxel_sizes=voxel_sizes,
                cmap="turbo",
            )
            plot_global_2d(
                grad[0],
                fig=fig,
                ax=ax[1],
                fig_title="gradient 1$^{st}$ component",
                voxel_sizes=voxel_sizes,
                cmap="coolwarm",
            )
            plot_global_2d(
                grad[1],
                fig=fig,
                ax=ax[2],
                fig_title="gradient 2$^{nd}$ component",
                voxel_sizes=voxel_sizes,
                cmap="coolwarm",
            )

    return displacement, grad


######################################################################################################################################
#########################                        Strain and d-spacing                            #####################################
######################################################################################################################################


def compute_strain(grad, qcen, plot=True, voxel_sizes=None):
    strain = np.sum(
        [qcen[n] * grad[n] for n in range(len(qcen))], axis=0
    ) / np.linalg.norm(qcen)

    d_cen = 2.0 * np.pi / np.sqrt(np.sum(qcen**2.0, axis=0))
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
#########################                         Zlotnikov 2D strain                            #####################################
######################################################################################################################################

# def compute_strain_2D_Zlotnikov(obj_ortho,
#                                 voxel_sizes,
#                                 q_cen,
#                                 crop=False,
#                                 use_negative_phase=True, # Careful to the sign of the strain regarding the phase. There's a minus sign !!!!!
#                                 threshold_module=.3,
#                                 verbose=True, plot=False):

#     if verbose :
#         print('voxel size : {}'.format(voxel_sizes))

#     module, phase = get_cropped_module_phase(obj_ortho, unwrap=True, threshold_module=threshold_module,
#                                              crop=crop)

#     if use_negative_phase:
#         phase = -phase # We need that due to the definition of numpy fft used in pynx !!!!!!!!!!!!

#     grad = EB_custom_gradient(phase, voxel_sizes = voxel_sizes)

#     strain = np.sum([q_cen[n] * grad[n] for n in range(obj_ortho.ndim)], axis=0)
#     strain = strain / np.sum([q_cen[n]**2. for n in range(obj_ortho.ndim)], axis=0)

#     d_cen = 2.*np.pi/np.sqrt(np.sum(q_cen**2.,axis=0))
#     d_spacing = d_cen * (1. + strain)

#     if plot :
#         fw=4
#         fig,ax = plt.subplots(1,2,figsize=(fw*2,fw))
#         plot_symmetric_colorscale_2d(100*strain, fig_title='strain (%)',voxel_sizes=voxel_sizes,
#                                  fig=fig,ax=ax[0])
#         plot_global_2d(d_spacing, fig_title='d spacing',voxel_sizes=voxel_sizes,
#                          fig=fig,ax=ax[1], cmap='coolwarm')
#     return strain, d_spacing


######################################################################################################################################
#########################                           Tilt functions                               #####################################
######################################################################################################################################


def orthogonal_vectors(qcen, check=False):
    """
    :qcen: Bragg vector (center of mass of the BCDI peak)
    """
    e_bragg = qcen / np.linalg.norm(qcen)

    guess = np.zeros(3)
    if np.argmax(qcen) == 2:
        guess[1] = 1
    else:
        guess[np.argmin(qcen)] = 1

    e1 = np.cross(guess, e_bragg)
    e1 = e1 / np.linalg.norm(e1)

    e2 = np.cross(e_bragg, e1)
    e2 = e2 / np.linalg.norm(e2)

    if check:
        print("e_bragg : ", e_bragg)
        print("e1 : ", e1)
        print("e2 : ", e2)

        print("all values below should be 1")
        print("check 1", np.dot(e_bragg, e_bragg))
        print("check 2", np.dot(e1, e1))
        print("check 3", np.dot(e2, e2))

        print("\nall values below should be 0")
        print("check 4", np.dot(e_bragg, e1))
        print("check 5", np.dot(e_bragg, e2))
        print("check 6", np.dot(e1, e2))
    return e_bragg, e1, e2


def compute_tilt(
    grad,
    qcen,
    polar_representation=True,
    check=False,
    plot=True,
    voxel_sizes=None,
):
    # Get 2 vectors perpendicular to the Bragg
    e_bragg, e1, e2 = orthogonal_vectors(qcen, check=check)

    # components of  the tilt along e1 and e2
    tilt_comp1 = np.sum([e1[n] * grad[n] for n in range(len(qcen))], axis=0)
    tilt_comp2 = np.sum([e2[n] * grad[n] for n in range(len(qcen))], axis=0)

    if plot:
        plot_2D_slices_middle_one_array3D(
            tilt_comp1,
            cmap="coolwarm",
            fig_title=f"tilt component along {np.round(e1, 2)}",
            voxel_sizes=voxel_sizes,
        )
        plot_2D_slices_middle_one_array3D(
            tilt_comp2,
            cmap="coolwarm",
            fig_title=f"tilt component along {np.round(e2, 2)}",
            voxel_sizes=voxel_sizes,
        )

    if polar_representation:
        # tilt angle (hope I'm not making a mistake choosing arctan2 instead of arctan)
        #     tilt_angle = np.arctan(tilt_comp2, tilt_comp1)
        tilt_angle = np.rad2deg(np.arctan2(tilt_comp2, tilt_comp1))

        # tilt magnitude
        tilt_magn = np.sqrt(tilt_comp1**2.0 + tilt_comp2**2.0)

        if plot:
            plot_2D_slices_middle_one_array3D(
                tilt_magn,
                cmap="gray_r",
                fig_title="tilt magnitude",
                voxel_sizes=voxel_sizes,
            )
            plot_2D_slices_middle_one_array3D(
                tilt_angle,
                cmap="coolwarm",
                fig_title="tilt angle (degrees)",
                voxel_sizes=voxel_sizes,
            )
        #             plot_2D_slices_middle_one_array3D(tilt_angle, cmap='coolwarm'
        #                                               , alpha=tilt_magn/np.nanmax(tilt_magn),
        #                                               fig_title='tilt angle (degrees) with magnitude transparency',
        #                                              voxel_sizes=voxel_sizes)
        return tilt_comp1, tilt_comp2, tilt_magn, tilt_angle, e1, e2
    else:
        return tilt_comp1, tilt_comp2, e1, e2


######################################################################################################################################
#########################                   Strain, d-spacing, Tilt in 2D                        #####################################
######################################################################################################################################


def Bragg_inplane_check(qcen, angle_error_limit=10, verbose=True):
    qcen_approx = np.copy(qcen)
    qcen_approx[-1] = 0
    angle_error = np.rad2deg(
        np.arccos(
            np.dot(
                qcen_approx / np.linalg.norm(qcen_approx),
                qcen / np.linalg.norm(qcen),
            )
        )
    )
    if verbose:
        print(
            "qcen in-plane orientation approximation error (degrees) :",
            angle_error,
        )
    if angle_error > angle_error_limit:
        raise ValueError(
            "qcen has a large component out of plane !\nThe in-plane approximation is off."
        )
    return


def orthogonal_vector_2D(qcen, verbose=False, plot=False):
    # Only keep the first 2 dimensions since we are looking at things in 2D along in the plane of the 2 first axes
    qcen_approx = np.copy(qcen[:2])
    u_bragg = qcen_approx / np.linalg.norm(
        qcen_approx
    )  # unit vector along Bragg direction

    # Create a vector perpendicular to u_bragg
    u_perp = np.array([+u_bragg[1], -u_bragg[0]])

    if verbose:
        print("safety check (should be 0): ", np.dot(u_bragg, u_perp))

    if plot:
        plt.figure(figsize=(3, 3))
        plt.arrow(
            0,
            0,
            u_bragg[1],
            u_bragg[0],
            color="b",
            width=0.03,
            length_includes_head=True,
        )
        plt.arrow(
            0,
            0,
            u_perp[1],
            u_perp[0],
            color="r",
            width=0.03,
            length_includes_head=True,
        )
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.text(
            u_bragg[1] / 2.0,
            u_bragg[0] / 2.0 + 0.1,
            "Bragg\ndirection",
            color="b",
        )
        plt.text(
            u_perp[1] / 2.0,
            u_perp[0] / 2.0 + 0.1,
            "perpendicular\ndirection",
            color="r",
        )
    return u_bragg, u_perp


def compute_strain_tilt_2D(
    grad, qcen, voxel_sizes=None, angle_error_limit=10, plot=True, verbose=True
):
    # First check that you're not doing something totally wrong
    Bragg_inplane_check(qcen, angle_error_limit=angle_error_limit, verbose=verbose)

    u_bragg, u_perp = orthogonal_vector_2D(qcen, verbose=verbose, plot=plot)

    strain = np.sum([grad[n] * u_bragg[n] for n in range(2)], axis=0)

    tilt = np.sum([grad[n] * u_perp[n] for n in range(2)], axis=0)
    tilt_angle = np.rad2deg(np.arctan(tilt))

    d_cen = 2.0 * np.pi / np.sqrt(np.sum(qcen**2.0, axis=0))
    d_spacing = d_cen * (1.0 + strain)

    if plot:
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        plot_symmetric_colorscale_2d(
            100 * strain,
            fig_title="strain (%)",
            voxel_sizes=voxel_sizes,
            fig=fig,
            ax=ax[0],
        )
        plot_global_2d(
            d_spacing,
            fig_title=r"d spacing ($\AA$)",
            voxel_sizes=voxel_sizes,
            fig=fig,
            ax=ax[1],
            cmap="coolwarm",
        )
        plot_global_2d(
            tilt_angle,
            fig_title="tilt (degrees)",
            voxel_sizes=voxel_sizes,
            fig=fig,
            ax=ax[2],
            cmap="coolwarm",
        )

    return strain, d_spacing, tilt_angle, u_bragg, u_perp
