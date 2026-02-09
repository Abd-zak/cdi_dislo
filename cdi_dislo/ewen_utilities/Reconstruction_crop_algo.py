import numpy as np
import pylab as plt

import os
from cdi_dislo.ewen_utilities.plot_utilities import (
    plot_2D_slices_middle,
    plot_object_module_phase_2d,
    plot_2d_intensity_data,
)
from cdi_dislo.ewen_utilities.Reconstruction import (
    default_cdi_parameters,
    CDI_one_reconstruction,
    put_back_centering_ramp,
    dummy_file_dict,
)
from cdi_dislo.ewen_utilities.Global_utilities import (
    get_npz_files,
    check_path_create,
    center_the_center_of_mass,
)


def crop_diffraction_data(data, crop_factor):
    crop = (np.array(data.shape) * (1.0 - 1.0 / crop_factor)) / 2.0
    crop = crop.astype("int")
    #     data_crop = data[crop[0]:-crop[0], crop[1]:-crop[1], crop[2]:-crop[2]]
    s = [slice(crop[n], -crop[n]) for n in range(data.ndim)]
    return data[tuple(s)]


from scipy.interpolate import RegularGridInterpolator as rgi


def interpolate_object(obj, shape_end, plot=False):
    my_interpolating_function = rgi(
        tuple([np.linspace(-1, 1, shape) for shape in obj.shape]), obj
    )

    points = np.meshgrid(
        *[np.linspace(-1, 1, shape_end[n]) for n in range(obj.ndim)],
        indexing="ij",
    )
    points = np.moveaxis(np.array(points), 0, -1)
    obj_inter = my_interpolating_function(points)

    if plot:
        if obj.ndim == 3:
            plot_2D_slices_middle(obj)
            plot_2D_slices_middle(obj_inter)
        if obj.ndim == 2:
            plot_object_module_phase_2d(obj)
            plot_object_module_phase_2d(obj_inter)

    return obj_inter


def crop_scale_function(
    data,
    crop_factor_list=[2],
    data_centering=False,
    mask=None,
    params=None,
    show_cdi=None,
    plot=False,
    verbose=False,
):
    if params is None:
        params = default_cdi_parameters()

    if data_centering:
        data, centering_offsets = center_the_center_of_mass(
            data, return_offsets=True
        )
        if mask is not None:
            mask = np.roll(
                mask, centering_offsets, axis=range(len(mask.shape))
            )
        print("centering_offsets : ", centering_offsets)

    # Sorry but I need to force some parameters
    params["center_data"] = False  # necessary
    params["show_cdi"] = None  # could be removed
    params["calc_llk"] = 0  # could be removed
    params["plot_result"] = plot

    # Reconstruction
    params["obj_init"] = None
    for n, crop_factor in enumerate(crop_factor_list):
        if verbose:
            print("\ncrop_factor {}".format(crop_factor))

        data_crop = crop_diffraction_data(data, crop_factor)
        if mask is not None:
            mask_crop = crop_diffraction_data(mask, crop_factor)
        if plot:
            if data_crop.ndim == 3:
                from cdi_dislo.ewen_utilities.plot_utilities import (
                    plot_3D_projections,
                )

                plot_3D_projections(
                    data_crop, fig_title="crop_factor : {}".format(crop_factor)
                )
            if data_crop.ndim == 2:
                plot_2d_intensity_data(data_crop)

        if n == 0:
            if not type(params["support_init"]) == str:
                support_init_crop = interpolate_object(
                    params["support_init"], data_crop.shape, plot=False
                )
                params["support_init"] = support_init_crop
            else:
                params["support_init"] = "autocorrelation"

        if n == 1:
            params["support_threshold_relative"] = return_dict[
                "support_threshold_relative"
            ]
            # Fix the support threshold to be the same at all scale. Not sure if that's so important
            print(
                "support_threshold_relative {} :".format(
                    params["support_threshold_relative"]
                )
            )
        if n != 0:
            obj_inter = interpolate_object(
                obj_crop, data_crop.shape, plot=plot
            )
            support_inter = interpolate_object(
                support_crop, data_crop.shape, plot=False
            )
            params["obj_init"] = obj_inter
            #             params['support_init'] = 'gauss_conv'
            params["support_init"] = support_inter

        if mask is not None:
            params["mask"] = mask_crop

        if show_cdi is not None:  # Used for debugging mostly
            params["show_cdi"] = show_cdi

        obj_crop, _, support_crop, return_dict = CDI_one_reconstruction(
            data_crop, params
        )

    #     return obj_crop, support_crop, return_dict

    # Final reconstruction using whole data array
    if verbose:
        print("\nfinal reconstruction")
    if plot:
        if data.ndim == 3:
            from cdi_dislo.ewen_utilities.plot_utilities import (
                plot_3D_projections,
            )

            plot_3D_projections(
                data, fig_title="no cropping"
            )
        if data.ndim == 2:
            plot_2d_intensity_data(data)

    obj_inter = interpolate_object(obj_crop, data.shape, plot=plot)
    support_inter = interpolate_object(support_crop, data.shape, plot=False)
    params["obj_init"] = obj_inter
    params["support_init"] = support_inter
    # params['support_init'] = 'gauss_conv' I don't understand why this doesn't work at all...

    if mask is not None:
        params["mask"] = mask

    obj, llk, support, return_dict = CDI_one_reconstruction(data, params)

    if data_centering:
        obj = put_back_centering_ramp(obj, centering_offsets)

    return obj, llk, support, return_dict


def save_reconstruction_crop_algo(
    file_dict, obj, llk, support, n_reconstruction
):
    """
    Save the final object automatically.
    Not an important function> You could do that in another way if you want.
    """

    if file_dict["orthogonalization"]:
        ortho_string = "_ortho"
    else:
        ortho_string = ""

    path_save = "Reconstructions_CropAlgo/{}_scan{}{}{}/".format(
        file_dict["h5file"],
        file_dict["scan_nb"],
        ortho_string,
        file_dict["savename_add_string"],
    )

    check_path_create("Reconstructions_CropAlgo")
    check_path_create(path_save)

    savename = path_save + "reconstruction{}_{}".format(
        ortho_string, n_reconstruction
    )

    np.savez_compressed(
        savename, obj=obj, support=support, llk=llk, **file_dict
    )
    return


def make_several_reconstructions_crop_algo(
    data,
    file_dict,
    Nb_reconstruction,
    crop_factor_list,
    data_centering=False,
    mask=None,
    params=None,  # default reconstruction parameters
    dont_erase_previous_recon=True,
    plot=False,
    verbose=False,
):
    if file_dict is None:
        file_dict = dummy_file_dict()

    if dont_erase_previous_recon:
        if file_dict["orthogonalization"]:
            ortho_string = "_ortho"
        else:
            ortho_string = ""

        path_save = "Reconstructions_CropAlgo/{}_scan{}{}{}/".format(
            file_dict["h5file"],
            file_dict["scan_nb"],
            ortho_string,
            file_dict["savename_add_string"],
        )
        if os.path.exists(path_save):
            files = get_npz_files(path_save)
            if len(files) != 0:
                recon_nb = [int(f.split(".")[0].split("_")[-1]) for f in files]
                last_recon_nb = max(recon_nb)
            else:
                last_recon_nb = 0
        else:
            last_recon_nb = 0
    else:
        last_recon_nb = 0

    for n_reconstruction in range(Nb_reconstruction):
        if plot:
            plt.figure()
            plt.title(
                "Reconstruction {}".format(n_reconstruction), fontsize=20
            )
        nb_try = 0
        while 1:
            try:
                if params is not None:
                    params_copy = params.copy()
                else:
                    params_copy = None
                obj, llk, support, return_dict = crop_scale_function(
                    data,
                    mask=mask,
                    data_centering=data_centering,
                    crop_factor_list=crop_factor_list,
                    params=params_copy,
                    plot=plot,
                    verbose=verbose,
                )

                if plot:
                    if obj.ndim == 3:
                        plot_2D_slices_middle(obj)
                    else:
                        plot_object_module_phase_2d(obj)

                save_reconstruction_crop_algo(
                    file_dict,
                    obj,
                    llk,
                    support,
                    last_recon_nb + 1 + n_reconstruction,
                )
                print("\n\nsucessfull reconstruction\n\n")
                break
            except Exception :
                print("\n\nfailed reconstruction\n\n")
                nb_try += 1
                pass
    return
