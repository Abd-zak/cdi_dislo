import numpy as np
import pylab as plt
from numpy.fft import ifftn, fftn, fftshift, ifftshift
from pynx.cdi import (
    CDI,
    ScaleObj,
    InitPSF,
    AutoCorrelationSupport,
    SupportUpdate,
    HIO,
    RAAR,
    ER,
    DetwinHIO,
    DetwinRAAR,
)  # Might take a bit of time to import

import os
from matplotlib.colors import LogNorm
import scipy
from cdi_dislo.ewen_utilities.plot_utilities import (
    MIR_Colormap,
    plot_object_module_phase_2d,
    plot_2D_slices_middle_and_histogram,
)
from cdi_dislo.ewen_utilities.Global_utilities import (
    get_npz_files,
    check_path_create,
    center_the_center_of_mass,
)
from cdi_dislo.ewen_utilities.Object_utilities import center_object

# from cdi_dislo.ewen_utilities.Reconstruction_crop_algo                    import *


my_cmap = MIR_Colormap()


######################################################################################################################################
################################                      Dummy file_dict                       ##########################################
######################################################################################################################################


def dummy_file_dict(savename="dummy"):
    file_dict = {}
    file_dict["orthogonalization"] = False
    file_dict["h5file"] = savename
    file_dict["scan_nb"] = 0
    file_dict["savename_add_string"] = ""
    file_dict["sample"] = "dummy"
    return file_dict


######################################################################################################################################
################################            Load preprocessed diffraction data              ##########################################
######################################################################################################################################


def load_diffraction_data_Q_space(preprocessed_datapath, vmin=1, plot=False):
    file = np.load(preprocessed_datapath, allow_pickle=True)
    data = file["data"]

    if plot:
        if len(data.shape) == 2:
            fig = plt.figure(figsize=(10, 10))
            plt.imshow(data, cmap="gray", norm=LogNorm(vmin=vmin))
            plt.colorbar()

        if len(data.shape) == 3:
            from cdi_dislo.ewen_utilities.plot_utilities import (
                plot_3D_projections,
            )

            plot_3D_projections(data)

    mask = file["mask"]

    # load dictionary
    file_dict = {key: file[key] for key in file.files}
    file_dict["preprocessed_datapath"] = preprocessed_datapath
    file_dict.pop("data", None)
    file_dict.pop("mask", None)

    return data, mask, file_dict


######################################################################################################################################
###################################            Random object initialization             ##############################################
######################################################################################################################################


def CreateRandomInitialObject(shape, amin=0, amax=80, phirange=np.pi / 2.0):
    a = np.random.uniform(amin, amax, shape)
    phi = np.random.uniform(0, phirange, shape)
    obj_init = a * np.exp(1.0j * phi)
    return obj_init


######################################################################################################################################
###################################             Create support from object              ##############################################
######################################################################################################################################


def EB_custom_support_from_object(obj, sigma=3, threshold=0.1, plot=False, alpha=0.3):
    module = np.abs(obj)
    module_gauss = scipy.ndimage.gaussian_filter(module, sigma=sigma)

    support = np.zeros(module.shape)
    support[module_gauss > threshold * np.max(module_gauss)] = 1

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(module, cmap="gray_r")
        ax[1].imshow(module, cmap="gray_r")
        ax[1].imshow(
            np.dstack(
                [
                    support,
                    np.zeros(support.shape),
                    np.zeros(support.shape),
                    alpha * support,
                ]
            )
        )
        ax[0].set_title("module (gray)", fontsize=15)
        ax[1].set_title("module (gray)\nsupport (transparent red)", fontsize=15)
    return support


######################################################################################################################################
###########################            CDI reconstruction (works well for 2D data)             #######################################
######################################################################################################################################


def default_cdi_parameters():
    params = {}
    params["center_data"] = True  # center data during reconstruction
    params["show_cdi"] = 300  # None#10000#300
    # params['algo_string'] = 'HIO_600 RAAR_1000 ER_200 HIO_500'
    params["algo_string"] = "HIO_400 RAAR_1000 ER_300"
    params["support_update"] = 20
    params["fix_support"] = False
    params["init_psf"] = True
    params["update_psf"] = 20

    params["obj_init"] = None
    params["support_init"] = "gauss_conv"
    params["mask"] = None
    params["cdi"] = None

    # params['support_threshold_relative_min'] = .23
    # params['support_threshold_relative_max'] = .3
    params["support_threshold_relative"] = None
    #     params['support_threshold_relative_min'] = .2
    params["support_threshold_relative_min"] = 0.1
    params["support_threshold_relative_max"] = 0.3
    params["support_smooth_width"] = (2, 1, 600)
    params["post_expand"] = (1, -2, 1)
    params["support_update_method"] = "rms"
    params["update_border_n"] = 0

    params["plot_result"] = True
    params["return_cdi"] = False

    params["compute_free_llk"] = True
    params["calc_llk"] = None  # Don't calculate llk all the time

    return params


import warnings


def CDI_one_reconstruction(data, params, plot_result=True):
    """
    :support_init: 'gauss_conv' or 'autocorrelation'. gauss_conv only works if the obj_init_list is given
    :obj_init: careful, should not be fftshifted
    """

    if np.any(np.array(data.shape) % 2 != 0):
        params["center_data"] = False
        warnings.warn(
            "Centering removed. Centering is dangerous with odd array dimension. put_back_centering_ramp function might need changes.",
            UserWarning,
        )

    if params["center_data"]:
        data, centering_offsets = center_the_center_of_mass(data, return_offsets=True)
        if params["mask"] is not None:
            params["mask"] = np.roll(
                params["mask"],
                centering_offsets,
                axis=range(len(params["mask"].shape)),
            )
        print("centering_offsets : ", centering_offsets)

    if type(params["support_init"]) == np.ndarray:
        support = params["support_init"]
        print("using the support array given by user")
    elif type(params["support_init"]) == str:
        if params["support_init"] == "gauss_conv" and params["obj_init"] is not None:
            print("using gaussian convoluted and threshold as support")
            support = EB_custom_support_from_object(params["obj_init"], plot=False)
        else:
            support = None  # Autocorrelation support will be used
    else:
        raise ValueError("something is wrong in params['support_init']")

    if params["obj_init"] is None:
        obj_init = CreateRandomInitialObject(data.shape)
        if type(params["support_init"]) == str:
            print(
                "Using autocorrelation as an initial support. If you're not happy, give either a support or an obj_init"
            )
            params["support_init"] = "autocorrelation"
    else:
        obj_init = np.copy(params["obj_init"])
    obj_init = fftshift(obj_init)

    if support is not None:
        support = fftshift(support)

    if params["mask"] is not None:
        mask = fftshift(params["mask"])
    else:
        mask = np.zeros(data.shape)

    if params["cdi"] is None:
        cdi = CDI(fftshift(data), obj=obj_init, support=support, mask=mask)
    else:
        cdi = params["cdi"]
        cdi.set_obj(params["obj_init"])
        cdi.set_support(support)

    cdi = ScaleObj() * cdi
    #     cdi = InitPSF()*cdi
    if params["init_psf"]:
        cdi = (
            InitPSF(fwhm=0.5, eta=0.05) * cdi
        )  # Used to take partial coherence into account

    if type(params["support_init"]) == str:
        if params["support_init"] == "autocorrelation":
            print("using autocorrelation as support")
            cdi = AutoCorrelationSupport(threshold=0.1) * cdi

    if params["compute_free_llk"]:
        cdi.init_free_pixels()  # Used to compute the free log likehood. In the end it doesn't work very well

    if params["support_threshold_relative"] is None:
        support_threshold_relative = params[
            "support_threshold_relative_min"
        ] + np.random.rand() * (
            params["support_threshold_relative_max"]
            - params["support_threshold_relative_min"]
        )

    else:
        support_threshold_relative = np.copy(params["support_threshold_relative"])
    #     sup = SupportUpdate(threshold_relative=support_threshold_relative, smooth_width=params['support_smooth_width'],
    #                 force_shrink=False,method='max', post_expand=None)
    sup = SupportUpdate(
        threshold_relative=support_threshold_relative,
        smooth_width=params["support_smooth_width"],
        force_shrink=False,
        post_expand=params["post_expand"],
        method=params["support_update_method"],
        update_border_n=params["update_border_n"],
    )

    if params["show_cdi"] is not None:
        plt.figure()

    # Reconstruction algortihm
    for algo in params["algo_string"].split():
        method = algo.split("_")[0]
        iterations = int(algo.split("_")[1])

        if params["support_update"] == False or params["support_update"] == None:
            if method == "HIO":
                cdi = (
                    HIO(
                        beta=0.9,
                        calc_llk=params["calc_llk"],
                        show_cdi=params["show_cdi"],
                        update_psf=params["update_psf"],
                    )
                    ** iterations
                    * cdi
                )
            if method == "DetwinHIO":
                cdi = DetwinHIO(beta=0.9) ** iterations * cdi
            if method == "RAAR":
                cdi = (
                    RAAR(
                        beta=0.9,
                        calc_llk=params["calc_llk"],
                        show_cdi=params["show_cdi"],
                        update_psf=params["update_psf"],
                    )
                    ** iterations
                    * cdi
                )
            if method == "DetwinRAAR":
                cdi = DetwinRAAR(beta=0.9) ** iterations * cdi
            if method == "ER":
                cdi = (
                    ER(
                        calc_llk=params["calc_llk"],
                        show_cdi=params["show_cdi"],
                        update_psf=params["update_psf"],
                    )
                    ** iterations
                    * cdi
                )

        else:
            if method == "HIO":
                cdi = (
                    sup
                    * HIO(
                        beta=0.9,
                        calc_llk=params["calc_llk"],
                        show_cdi=params["show_cdi"],
                        update_psf=params["update_psf"],
                    )
                    ** params["support_update"]
                ) ** (iterations // params["support_update"]) * cdi
            if method == "DetwinHIO":
                cdi = DetwinHIO(beta=0.9) ** iterations * cdi
            if method == "RAAR":
                cdi = (
                    sup
                    * RAAR(
                        beta=0.9,
                        calc_llk=params["calc_llk"],
                        show_cdi=params["show_cdi"],
                        update_psf=params["update_psf"],
                    )
                    ** params["support_update"]
                ) ** (iterations // params["support_update"]) * cdi
            if method == "DetwinRAAR":
                cdi = DetwinRAAR(beta=0.9) ** iterations * cdi
            if method == "ER":
                cdi = (
                    sup
                    * ER(
                        calc_llk=params["calc_llk"],
                        show_cdi=params["show_cdi"],
                        update_psf=params["update_psf"],
                    )
                    ** params["support_update"]
                ) ** (iterations // params["support_update"]) * cdi

    #    # Reconstruction algortihm
    #     support_update = 20
    #     update_psf = 20
    #     cdi = (sup * HIO(beta=0.9, calc_llk=100, show_cdi=show_cdi, update_psf=update_psf)**support_update)**(400//support_update)* cdi
    #     cdi = (sup * RAAR(beta=0.9, calc_llk=100, show_cdi=show_cdi, update_psf=update_psf)**support_update)**(1000//support_update)* cdi
    #     cdi = (sup * ER(calc_llk=20, show_cdi=show_cdi, update_psf=update_psf) ** support_update)**(300//support_update) *cdi

    if params["calc_llk"] == 0:
        # I fucking hate this line. I'm struggling to calculate the llk at the end.
        # My dirty fix is to add one step of ER and calculate llk inside.
        cdi = (sup * ER(calc_llk=1)) * cdi

    if params["compute_free_llk"]:
        llk = cdi.get_llk()[3]  #  keep free poisson log likelihood
    else:
        llk = cdi.get_llk()[0]  #  keep poisson log likelihood

    obj = fftshift(cdi.get_obj())
    support = fftshift(cdi.get_support())
    obj, support = center_object(obj, support=support)

    if params["center_data"]:
        print("centering ramp is added back at the end of the reconstruction")
        obj = put_back_centering_ramp(obj, centering_offsets)

    if params["plot_result"]:
        if obj.ndim == 3:
            plot_2D_slices_middle_and_histogram(obj, support=support)
        if obj.ndim == 2:
            plot_object_module_phase_2d(obj)

    return_dict = {}
    return_dict["support_threshold_relative"] = support_threshold_relative
    if params["return_cdi"]:
        return_dict["cdi"] = cdi

    return obj, llk, support, return_dict


######################################################################################################################################
###########################                      Save one reconstruction                       #######################################
######################################################################################################################################


def save_reconstruction_best_recon_algo(file_dict, obj, llk, n_reconstruction):
    """
    Save the final object automatically.
    Not an important function> You could do that in another way if you want.
    """

    #     path_save = 'Reconstructions_BestReconAlgo/{}_h5file_{}_scan{}{}/'.format(file_dict['sample'],
    #                                                                            file_dict['h5file'], file_dict['scan_nb'],
    #                                                                            file_dict['savename_add_string'])

    if file_dict["orthogonalization"]:
        ortho_string = "_ortho"
    else:
        ortho_string = ""

    path_save = "Reconstructions_BestReconAlgo/{}_scan{}{}{}/".format(
        file_dict["h5file"],
        file_dict["scan_nb"],
        ortho_string,
        file_dict["savename_add_string"],
    )

    check_path_create("Reconstructions_BestReconAlgo")
    check_path_create(path_save)

    savename = path_save + "reconstruction_{}".format(n_reconstruction)

    np.savez_compressed(savename, obj=obj, llk=llk, **file_dict)
    return


######################################################################################################################################
###########################                    Make several reconstructions                     ######################################
######################################################################################################################################


def make_several_reconstructions(
    data, params, file_dict, Nb_reconstruction, dont_erase_previous_recon=True
):
    if dont_erase_previous_recon:
        if file_dict["orthogonalization"]:
            ortho_string = "_ortho"
        else:
            ortho_string = ""

        path_save = "Reconstructions_BestReconAlgo/{}_scan{}{}{}/".format(
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

    print(last_recon_nb)
    for n_reconstruction in range(Nb_reconstruction):
        #         plt.figure()
        #         plt.title('Reconstruction {}'.format(n_reconstruction), fontsize=20)
        while 1:
            try:
                obj, llk, support, return_dict = CDI_one_reconstruction(data, params)

                save_reconstruction_best_recon_algo(
                    file_dict, obj, llk, last_recon_nb + 1 + n_reconstruction
                )
                print("\n\nsucessfull reconstruction\n\n")
                break
            except:
                print("\n\nfailed reconstruction\n\n")
                pass
    return


######################################################################################################################################
###########################                Put back ramp due to the centering                   ######################################
######################################################################################################################################


def put_back_centering_ramp(obj, centering_offsets):
    Fexp = ifftshift(fftn(fftshift(obj)))
    Fexp = np.roll(Fexp, -np.array(centering_offsets), axis=range(len(Fexp.shape)))
    obj = fftshift(ifftn(ifftshift(Fexp)))
    return obj
