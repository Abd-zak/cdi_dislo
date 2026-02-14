import numpy as np

from cdi_dislo.ewen_utilities.Object_utilities import create_support
from skimage.registration import phase_cross_correlation
import scipy

##################################################################################################################################
#############################                Realign objects                  ####################################################
##################################################################################################################################


def force_same_shape(obj_list, constant_values=0, verbose=False):
    """
    In case objects in obj_list don't have the same shape,
    this function forces an identical shape by padding objects with 0s
    :obj_list: a list of objects of shape (number of objects, individual object shape)
    """

    shape_list = [obj.shape for obj in obj_list]

    if np.all([shape == shape_list[0] for shape in shape_list]):
        if verbose:
            print("All objects already have the same shape")
        return obj_list

    forced_shape = np.max(shape_list, axis=0)

    for n, obj in enumerate(obj_list):
        pad = np.array(forced_shape) - np.array(obj.shape)
        padding = [(p // 2, p // 2 + p % 2) for p in pad]
        obj_list[n] = np.pad(
            obj, padding, mode="constant", constant_values=(constant_values,)
        )

    if verbose:
        print(f"All objects have now the shape : {forced_shape}")

    return np.array(obj_list)


def realign_object_list(
    obj_list,
    integer_shift=True,
    ref_index=0,
    threshold_module=0.15,
    fill_support=False,
    force_shape=True,
    verbose=True,
):
    """
    Align all objects in obj_list using the supports and a phase_cross_correlation.
    Limited to integer pixels shift. No sub-pixel shifts.
    :integer_shift: if True make only integer shift using numpy.roll.
                    If False, use scipy.ndimage.shift to make subpixel shift with interpolation
    :ref_index: index of the reference object
                (for example with ref_index=0, the first object is the reference position)
    :threshold_module: threshold (between 0 and 1) used to create the support.
    :fill_support: If True, fill holes in the support (mostly for particles with dislocations)
    """

    if force_shape:
        obj_list = force_same_shape(obj_list, verbose=verbose)

    obj_ref = np.copy(obj_list[ref_index])
    support_ref = create_support(obj_ref, threshold_module, fill_support=fill_support)

    obj_list_shift = np.zeros(obj_list.shape).astype("complex128")
    for n, obj in enumerate(obj_list):
        support = create_support(obj, threshold_module, fill_support=fill_support)
        shift, error, diffphase = phase_cross_correlation(support_ref, support)

        if integer_shift:
            shift = np.round(shift).astype("int")
            obj_list_shift[n] += np.roll(obj, shift, axis=range(obj.ndim))
        else:
            obj_list_shift[n] += scipy.ndimage.shift(obj, shift)

    return obj_list_shift


def realign_amp_phase_list(
    amp_list,
    phase_list,
    ref_index=0,
    threshold_module=0.15,
    fill_support=False,
    force_shape=True,
    verbose=True,
):
    """
    Align all objects in obj_list using the supports and a phase_cross_correlation.
    Limited to integer pixels shift. No sub-pixel shifts.
    :ref_index: index of the reference object
                (for example with ref_index=0, the first object is the reference position)
    :threshold_module: threshold (between 0 and 1) used to create the support.
    :fill_support: If True, fill holes in the support (mostly for particles with dislocations)
    """

    if force_shape:
        amp_list = force_same_shape(amp_list, verbose=verbose)
        phase_list = force_same_shape(
            phase_list, verbose=verbose, constant_values=np.nan
        )

    amp_ref = np.copy(amp_list[ref_index])
    support_ref = create_support(amp_ref, threshold_module, fill_support=fill_support)

    amp_list_shift = np.zeros(amp_list.shape)
    phase_list_shift = np.zeros(phase_list.shape)
    for n, amp, phase in zip(range(amp_list_shift.shape[0]), amp_list, phase_list):
        support = create_support(amp, threshold_module, fill_support=fill_support)
        shift, error, diffphase = phase_cross_correlation(support_ref, support)

        #         amp_list_shift[n] += scipy.ndimage.shift(amp, shift)
        #         phase_list_shift[n] += scipy.ndimage.shift(phase, shift)

        shift = np.round(shift).astype("int")
        amp_list_shift[n] += np.roll(amp, shift, axis=range(amp.ndim))
        phase_list_shift[n] += np.roll(phase, shift, axis=range(amp.ndim))

    return amp_list_shift, phase_list_shift
