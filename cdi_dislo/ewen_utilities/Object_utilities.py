import pylab as plt
import numpy as np
from numpy.fft import fftshift

# from pynx.utils.phase import unwrap_phase
from cdi_dislo.ewen_utilities.Global_utilities import center_the_center_of_mass

#################################################################################################################################
#################################          Center reconstructed object            ###############################################
#################################################################################################################################


def crop_array_half_size(array):
    """
    Works for any dimension
    """
    shape = array.shape
    s = [
        slice(shape[n] // 2 - shape[n] // 4, shape[n] // 2 + shape[n] // 4)
        for n in range(array.ndim)
    ]
    return array[tuple(s)]


def center_object(obj, standard_com=True, support=None):
    module = np.abs(obj)

    module_cen, offset = center_the_center_of_mass(
        module, return_offsets=True, standard_com=standard_com
    )

    # I make the centering twice, I use a cropping for the second one
    shape = obj.shape
    module_cen = crop_array_half_size(module_cen)
    module_cen2, offset2 = center_the_center_of_mass(
        module_cen, return_offsets=True, standard_com=standard_com
    )

    total_offset = np.array(offset) + np.array(offset2)

    obj_cen = np.roll(obj, total_offset, axis=range(len(shape)))

    if support is not None:
        support_cen = np.roll(support, total_offset, axis=range(len(shape)))
        return obj_cen, support_cen
    else:
        return obj_cen


def center_object_list(obj_list):
    obj_list_centered = np.zeros(obj_list.shape, dtype="complex128")
    for n in range(len(obj_list)):
        #         print(len(obj_list)-n,end=' ')
        obj_list_centered[n] += center_object(obj_list[n])
    return obj_list_centered


#################################################################################################################################
#################################          Get cropped module and phase            ##############################################
#################################################################################################################################

# def get_cropped_module_phase(obj,
#                              threshold_module = None, support = None,
#                              crop=True, apply_fftshift=False, unwrap=True):

#     if apply_fftshift:
#         obj = fftshift(obj)

#     shape = obj.shape
#     if crop:
#         obj = crop_array_half_size(obj)
#         if support is not None:
#             support = crop_array_half_size(support)

#     module = np.abs(obj)

#     if support is None:
#         if threshold_module is None:
#             if obj.ndim ==3:
#                 threshold_module=.01 # Seems that 3D data need a smaller threshold
#             if obj.ndim == 2:
#                 threshold_module = .3
#         support = (module > np.max(module)*threshold_module)

#     if unwrap:
#         phase = unwrap_phase(obj)
#     else:
#         phase = np.angle(obj)

#     phase[support==0] = np.nan # Maybe change this ?

#     # Badly written
#     if phase.ndim==2:
#         phase -= phase[phase.shape[0]//2,phase.shape[1]//2]
#     if phase.ndim==3:
#         phase -= phase[phase.shape[0]//2,phase.shape[1]//2,phase.shape[2]//2]

#     return module, phase


def get_cropped_module_phase(
    obj,
    threshold_module=None,
    support=None,
    crop=False,
    apply_fftshift=False,
    unwrap=True,
):
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
            if obj.ndim == 3:
                threshold_module = 0.3  # Seems that 3D data need a smaller threshold. Nope in the end, .3 is fine
            if obj.ndim == 2:
                threshold_module = 0.3
        support = module >= np.nanmax(module) * threshold_module

    phase = np.angle(obj)

    if unwrap:
        from skimage.restoration import unwrap_phase as unwrap_phase_skimage

        mask_unwrap = 1 - support

        if np.any(np.isnan(obj)):  # Need to take nan's into account
            print("nan's are stil la problem, this freezes the unwrapping")
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

    phase[support == 0] = np.nan

    #     # Badly written Not a good thing actually is the center is a nan
    #     if phase.ndim==2:
    #         phase -= phase[phase.shape[0]//2,phase.shape[1]//2]
    #     if phase.ndim==3:
    #         phase -= phase[phase.shape[0]//2,phase.shape[1]//2,phase.shape[2]//2]

    return module, phase


#################################################################################################################################
############################                 get object complex conjugate                   #####################################
#################################################################################################################################


def get_complex_conjugate(obj):
    obj_conj = np.conj(np.flip(obj, axis=range(obj.ndim)))
    return obj_conj


#################################################################################################################################
##############################                   Create support                     ########################################
#################################################################################################################################


def create_support(obj, threshold_module, fill_support=False):
    module = np.abs(obj)
    support = module > threshold_module * np.max(module)

    if fill_support:
        support_convex = np.zeros(support.shape)
        for axis in range(support.ndim):
            support_cum = np.cumsum(support, axis=axis)
            support_cum_inv = np.flip(
                np.cumsum(np.flip(support, axis=axis), axis=axis), axis=axis
            )
            support_combine = support_cum * support_cum_inv
            support_convex[support_combine != 0] = 1
        return support_convex
    else:
        return support


#################################################################################################################################
###########################                     automatic object ROI                       ######################################
#################################################################################################################################


def automatic_object_roi(obj, threshold=0.1, factor=0.4, plot=False):
    module = np.abs(obj)

    if plot:
        fig, ax = plt.subplots(1, module.ndim, figsize=(5 * module.ndim, 3))

    roi = np.zeros(2 * module.ndim, dtype="int")
    for n in range(module.ndim):
        sum_axis = np.arange(module.ndim)
        sum_axis = np.delete(sum_axis, n)
        projection = np.nanmean(module, axis=tuple(sum_axis))
        projection -= np.nanmin(projection)
        projection = projection / np.nanmax(projection)

        start = np.nanmin(np.where(projection > threshold))
        end = np.nanmax(np.where(projection > threshold))

        size = end - start
        start = max(int(start - size * factor), 0)
        end = int(end + size * factor)

        roi[2 * n] += start
        roi[2 * n + 1] += end

        if plot:
            ax[n].plot(projection, ".-")
            ax[n].axvline(x=start, color="r")
            ax[n].axvline(x=end, color="r")

    return roi


# def automatic_object_roi(obj,
#                          threshold = .1, factor = .05,
#                          plot=False):

#     module = np.abs(obj)
#     support = module > .1*np.max(module)
#     indices_support = np.where(support==1)
#     center = np.round(center_of_mass(support)).astype('int')

#     roi = np.zeros(2*module.ndim, dtype='int')
#     for axis in range(module.ndim):

#         start = np.nanmin(indices_support[axis])
#         end = np.nanmax(indices_support[axis])

#         size = end-start
#         start = max(round(start - factor * size), 0)
#         end = min(round(end + factor * size), module.shape[axis]-1)
#         start = max(int(start-size*factor),0)
#         end = int(end+size*factor)

#         roi[2*axis] += start
#         roi[2*axis+1] += end

#         # center roi by increasing side if necessary
#         difference = [center[axis] - roi[2*axis], roi[2*axis+1] - center[axis]]
#         shift = max(difference) - min(difference)
#         index_shift = np.argmin(difference)
#         roi[2*axis+index_shift] += shift * (2*index_shift-1)

#     return roi


######################################################################################################################################
##############################                 Compute oversampling ration                    ########################################
######################################################################################################################################


def compute_oversampling_ratio(obj, threshold_module=0.3, plot=False):
    module = np.abs(obj)
    support = module > threshold_module * np.max(module)

    indices_support = np.where(support == 1)
    size_per_dim = np.max(indices_support, axis=1) - np.min(
        indices_support, axis=1
    )
    oversampling = np.divide(np.array(support.shape), size_per_dim)

    if plot:
        fig, ax = plt.subplots(1, support.ndim, figsize=(5 * support.ndim, 4))
        for n in range(support.ndim):
            axes = tuple(np.delete(np.arange(3), n))
            proj = np.max(support, axis=axes)
            ax[n].plot(proj)
            title = f"oversampling along axis {n}\n{round(oversampling[n], 2)}"
            ax[n].set_title(title, fontsize=15)
    return oversampling


######################################################################################################################################
##############################                    Additional functions                        ########################################
######################################################################################################################################


def pad_to_higher_oversampling(obj, oversampling_final):
    oversampling = compute_oversampling_ratio(obj)
    padding = []
    for axe in range(obj.ndim):
        size = obj.shape[axe]
        final_size = int(
            np.ceil(size * oversampling_final[axe] / oversampling[axe])
        )
        pad = int(np.ceil((final_size - size) / 2.0))
        padding.append([pad, pad])
    obj_pad = np.pad(obj, padding)
    return obj_pad
