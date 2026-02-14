#####################################################################################################################
#####################################################################################################################
import numpy as np
import xrayutilities as xu
from numpy.fft import fftn, fftshift, ifftshift
import pylab as plt
import matplotlib
from matplotlib.colors import LogNorm
from ipywidgets import interact
from cdi_dislo.ewen_utilities.Global_utilities import (
    add_colorbar_subplot,
    apply_roi,
    slice_middle_array_along_axis,
)
from cdi_dislo.ewen_utilities.Object_utilities import (
    get_cropped_module_phase,
    automatic_object_roi,
)
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import warnings
import plotly.express as px
from IPython import get_ipython


# from matplotlib.colors import LogNorm
def MIR_Colormap():
    cdict = {
        "red": (
            (0.0, 1.0, 1.0),
            (0.11, 0.0, 0.0),
            (0.36, 0.0, 0.0),
            (0.62, 1.0, 1.0),
            (0.87, 1.0, 1.0),
            (1.0, 0.0, 0.0),
        ),
        "green": (
            (0.0, 1.0, 1.0),
            (0.11, 0.0, 0.0),
            (0.36, 1.0, 1.0),
            (0.62, 1.0, 1.0),
            (0.87, 0.0, 0.0),
            (1.0, 0.0, 0.0),
        ),
        "blue": (
            (0.0, 1.0, 1.0),
            (0.11, 1.0, 1.0),
            (0.36, 1.0, 1.0),
            (0.62, 0.0, 0.0),
            (0.87, 0.0, 0.0),
            (1.0, 0.0, 0.0),
        ),
    }
    my_cmap = matplotlib.colors.LinearSegmentedColormap("my_colormap", cdict, 256)
    return my_cmap


my_cmap = MIR_Colormap()
# my_cmap = 'gray'
# my_cmap = 'plasma'
###########################################################################################################################################
########################                  Plotly functions for a 3D scatter plot of the clusters                   ########################
###########################################################################################################################################


def get_colors(arr, cmap="viridis"):
    """
    transform an array of integers into an array of different colors

    Parameters
    ----------
    :arr: 1D array of integers
    :cmap: the colormap from which the colors are taken

    Returns
    ----------
    :colors: 1D array of same size as arr containing corresponding colors
    """
    cmap = cm.get_cmap(cmap)
    n = len(np.unique(arr))
    colornorm = matplotlib.colors.Normalize(vmin=1, vmax=n)
    hex_map = dict()
    for i, cl in enumerate(np.unique(arr)):
        hex_map[cl] = matplotlib.colors.to_hex(cmap(colornorm(i + 1)))
    colors = list(map(lambda x: hex_map[x], arr))
    return colors


def plotly_scatter3D_clusters(
    pixels, pixels_labels, Nb_pixels_partial=10000, marker_size=1.5
):
    """
    Make a clusters 3D scatter plot using plotly.express.scatter_3d and a different color for each cluster

    Parameters
    ----------
    :pixels: points positions with shape (number of points, 3)
    :pixels_labels: array of integer corresponding to the cluster label of each pixel. shape (number of points)
    :Nb_pixels_partial: number of pixels used in the plot. We don't use all of them for rapidity issues.
    :marker_size: size of the scattered points
    """

    if Nb_pixels_partial is not None:
        if len(pixels) < Nb_pixels_partial:
            Nb_pixels_partial = len(pixels)
        rand_indices = np.random.choice(np.arange(pixels.shape[0]), Nb_pixels_partial)
        pixels_partial = pixels[rand_indices]
        pixels_labels_partial = pixels_labels[rand_indices]

    colors = get_colors(pixels_labels_partial, cmap="viridis")

    fig = px.scatter_3d(
        x=pixels_partial[..., 0],
        y=pixels_partial[..., 1],
        z=pixels_partial[..., 2],
        color=colors,
    )
    fig.update_traces(marker_size=marker_size)
    fig.show()

    return


######################################################################################################################################
##################################          Plot with symmetric colorscale            ################################################
######################################################################################################################################


def plot_global_2d(
    array,
    fig=None,
    ax=None,
    fw=4,
    fig_title=None,
    voxel_sizes=None,
    vmin=None,
    vmax=None,
    cmap=None,
):
    if fig is None:
        fig, ax = plt.subplots(figsize=(fw, fw))

    if voxel_sizes is not None:
        voxel_sizes = (
            1e-3 * 0.1 * np.array(voxel_sizes)
        )  # Put the voxel_sizes in nanometers
        extent = [
            0,
            array.shape[1] * voxel_sizes[1],
            0,
            array.shape[0] * voxel_sizes[0],
        ]
    else:
        extent = None

    img = ax.matshow(array, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)

    # Just add the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax, orientation="vertical")

    if voxel_sizes is not None:
        ax.set_xlabel(r"$\mu$m", fontsize=15)
        ax.xaxis.set_ticks_position("bottom")

    if fig_title is not None:
        ax.set_title(fig_title, fontsize=20)
    fig.tight_layout()
    return


def plot_symmetric_colorscale_2d(
    array,
    fig=None,
    ax=None,
    fw=4,
    fig_title=None,
    voxel_sizes=None,
    cmap="coolwarm",
):
    vmax = np.nanmax(np.abs(array))
    vmin = -vmax

    plot_global_2d(
        array,
        fig=fig,
        ax=ax,
        fw=fw,
        fig_title=fig_title,
        voxel_sizes=voxel_sizes,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    return


def plot_subplot_colorbar(array, fig, ax, extent=None, cmap="turbo"):
    img = ax.matshow(array, cmap=cmap, extent=extent)
    # Just add the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax, orientation="vertical")
    fig.tight_layout()
    return


######################################################################################################################################
###################################          Plot projections of 3D data            ##################################################
######################################################################################################################################


def plot_3D_slice(data, mask=None, ax=None, fig=None, cmap=None, log_scale=False):
    """
    Same as plot_3D_projections but with only slices at the center of the data array instead of projections along axis.
    """

    if fig is None:
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    for n in range(3):
        s = data.shape[n] // 2
        if n == 0:
            data_ = data[s]
        if n == 1:
            data_ = data[:, s]
        if n == 2:
            data_ = data[:, :, s]
        if log_scale:
            ax[n].matshow(np.log(data_), cmap=cmap, aspect="auto")
        else:
            ax[n].matshow(data_, cmap=cmap, aspect="auto")

        if mask is not None:
            ax[n].matshow(mask[s], cmap="Reds", alpha=0.7, aspect="auto")

    fig.tight_layout()
    return


def plot_3D_projections(
    data,
    mask=None,
    alpha_mask=0.3,
    ax=None,
    fig=None,
    fw=4,
    fig_title=None,
    axes_labels=False,
    colorbar=False,
    log_scale=True,
    log_threshold=False,
    max_projection=False,
    vmin=None,
    vmax=None,
    fontsize=15,
    cmap=None,
    tight_layout=True,
):
    if cmap is None:
        cmap = my_cmap

    if fig is None:
        if colorbar:
            fig, ax = plt.subplots(1, 3, figsize=(3.5 * fw, fw))
        else:
            fig, ax = plt.subplots(1, 3, figsize=(3 * fw, fw))

    plots = []

    for n in range(3):
        if max_projection:
            img = np.nanmax(data, axis=n)
        else:
            img = np.nansum(data, axis=n)
        if log_scale:
            if log_threshold:
                plots.append(
                    ax[n].matshow(
                        xu.maplog(img, 5, 0),
                        cmap=cmap,
                        aspect="auto",
                        vmin=vmin,
                        vmax=vmax,
                    )
                )
            else:
                plots.append(
                    ax[n].matshow(
                        img,
                        cmap=cmap,
                        aspect="auto",
                        norm=LogNorm(vmin=vmin, vmax=vmax),
                    )
                )
        else:
            plots.append(
                ax[n].matshow(img, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
            )

        if mask is not None:
            mask_plot = np.nanmean(mask, axis=n)
            mask_plot[mask_plot != 0.0] = 1.0
            ax[n].imshow(
                np.dstack(
                    [
                        mask_plot,
                        np.zeros(mask_plot.shape),
                        np.zeros(mask_plot.shape),
                        alpha_mask * mask_plot,
                    ]
                ),
                aspect="auto",
            )

    if axes_labels:
        ax[0].set_xlabel("detector horizontal", fontsize=fontsize * fw / 4)
        ax[0].set_ylabel("detector vertical", fontsize=fontsize * fw / 4)

        ax[1].set_xlabel("detector horizontal", fontsize=fontsize * fw / 4)
        ax[1].set_ylabel("rocking curve", fontsize=fontsize * fw / 4)

        ax[2].set_xlabel("detector vertical", fontsize=fontsize * fw / 4)
        ax[2].set_ylabel("rocking curve", fontsize=fontsize * fw / 4)
    ax[0].tick_params(axis="y", labelsize=fontsize)
    ax[0].tick_params(axis="x", labelsize=fontsize)
    ax[1].tick_params(axis="y", labelsize=fontsize)
    ax[1].tick_params(axis="x", labelsize=fontsize)
    ax[2].tick_params(axis="y", labelsize=fontsize)
    ax[2].tick_params(axis="x", labelsize=fontsize)

    if colorbar:
        add_colorbar_subplot(fig, ax, plots, tick_fontsize=fontsize)

    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=fontsize * fw / 4)
    if tight_layout:
        fig.tight_layout()
    return fig


def plot_2d_intensity_data(
    data, ax=None, fig=None, fig_title=None, cmap=None, tight_layout=False
):
    if cmap is None:
        cmap = my_cmap

    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.imshow(np.log(data), cmap=my_cmap, vmin=1)

    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=20)
    if tight_layout:
        fig.tight_layout()
    return


######################################################################################################################################
################################            Plot object module histogram              ################################################
######################################################################################################################################


def plot_module_histogram(obj, crop=True, bins=None, fig=None, ax=None):
    module, phase = get_cropped_module_phase(obj, crop=crop)
    module[module < 0.01 * np.max(module)] = np.nan
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    if bins is None:
        if obj.ndim == 2:
            bins = 30
        if obj.ndim == 3:
            bins = 50
    n, bins, patches = ax.hist(module.flatten(), bins=bins)

    ax.set_xlabel("object module value", fontsize=15)
    ax.set_ylabel("number of pixels", fontsize=15)
    return


######################################################################################################################################
###########################            Plot slices in the middle of the array              ###########################################
######################################################################################################################################


def plot_2D_slices_middle_one_array3D(
    array,
    index=None,
    voxel_sizes=None,  # voxel size in Angstrom
    add_colorbar=True,
    cbar_position="right",
    cbar_label=None,
    cmap="gray_r",
    norm=None,
    ax=None,
    fig=None,
    fw=3,
    fig_title=None,
    xlabel=["nm", "nm", "nm"],
    ylabel=[None, None, None],
    alpha=1,
    vmin=None,
    vmax=None,
    symmetric_colorscale=False,
):
    if symmetric_colorscale:
        cmap = "coolwarm"

    shape = array.shape

    if fig is None:
        fig, ax = plt.subplots(1, 3, figsize=(3 * fw, fw))

    if voxel_sizes is not None:
        voxel_sizes = 0.1 * np.array(voxel_sizes)  # Put the voxel_sizes in nanometers
        extent = [
            [
                0,
                array.shape[2] * voxel_sizes[2],
                0,
                array.shape[1] * voxel_sizes[1],
            ],
            [
                0,
                array.shape[2] * voxel_sizes[2],
                0,
                array.shape[0] * voxel_sizes[0],
            ],
            [
                0,
                array.shape[1] * voxel_sizes[1],
                0,
                array.shape[0] * voxel_sizes[0],
            ],
        ]
    else:
        extent = [None, None, None]

    im = []
    for n in range(3):
        s = [slice(None, None, None) for ii in range(3)]
        if index is None:
            s[n] = shape[n] // 2
        else:
            s[n] = min(index, shape[n] - 1)
        arr = array[tuple(s)]
        if symmetric_colorscale:
            vmax = np.nanmax(np.abs(arr))
            vmin = -vmax
        if hasattr(alpha, "__len__"):
            alpha_plot = np.copy(alpha[tuple(s)])
            # Need to make the plot twice to avoid a matplotlib bug
            im.append(
                ax[n].imshow(
                    arr,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    extent=extent[n],
                    norm=norm,
                )
            )
            ax[n].cla()
            ax[n].imshow(
                arr,
                cmap=cmap,
                alpha=alpha_plot,
                vmin=vmin,
                vmax=vmax,
                extent=extent[n],
                norm=norm,
            )
        else:
            im.append(
                ax[n].imshow(
                    arr,
                    cmap=cmap,
                    alpha=alpha,
                    vmin=vmin,
                    vmax=vmax,
                    extent=extent[n],
                    norm=norm,
                )
            )

    if add_colorbar:
        for ii, img in enumerate(im):
            divider = make_axes_locatable(ax[ii])
            cax = divider.append_axes(cbar_position, size="5%", pad=0.05)
            #             cax = divider.append_axes('right', size='10%', pad=pad_colorbar)
            cbar = fig.colorbar(img, cax=cax, orientation="vertical")
            if cbar_label is not None:
                cbar.ax.set_ylabel(
                    cbar_label,
                    rotation=270,
                    fontsize=20 * fw / 4.0,
                    labelpad=10 * fw,
                )

    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=20 * fw / 4.0)

    if voxel_sizes is not None:
        for n in range(3):
            ax[n].set_xlabel(xlabel[n], fontsize=15 * fw / 4.0)
            ax[n].set_ylabel(ylabel[n], fontsize=15 * fw / 4.0)
            ax[n].xaxis.set_ticks_position("bottom")

    for axe in ax.flatten():
        axe.locator_params(nbins=4)

    fig.tight_layout()

    return


def plot_2D_slices_middle_only_module(
    obj,
    crop=False,
    voxel_sizes=None,
    cmap="gray_r",
    vmin=None,
    vmax=None,
    ax=None,
    fig=None,
    fw=3,
    fig_title=None,
    alpha=1,
):
    module, phase = get_cropped_module_phase(obj, unwrap=False, crop=crop)
    plot_2D_slices_middle_one_array3D(
        module,
        cmap=cmap,
        ax=ax,
        fig=fig,
        fw=fw,
        alpha=alpha,
        fig_title=fig_title,
        voxel_sizes=voxel_sizes,
        vmin=vmin,
        vmax=vmax,
    )
    return


def plot_2D_slices_middle_only_phase(
    obj,
    crop=False,
    threshold_module=None,
    support=None,
    voxel_sizes=None,
    cmap="hsv",
    vmin=None,
    vmax=None,
    unwrap=True,
    ax=None,
    fig=None,
    fw=3,
    fig_title=None,
):
    module, phase = get_cropped_module_phase(
        obj,
        threshold_module=threshold_module,
        support=support,
        crop=crop,
        apply_fftshift=False,
        unwrap=unwrap,
    )
    plot_2D_slices_middle_one_array3D(
        phase,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        fig=fig,
        fw=fw,
        fig_title=fig_title,
        voxel_sizes=voxel_sizes,
    )
    return


def plot_2D_slices_middle(
    obj,
    crop=False,
    support=None,
    threshold_module=None,
    unwrap=True,
    voxel_sizes=None,
    ax=None,
    fig=None,
    fw=3,
    fig_title=None,
    return_fig_ax=False,
):
    #     if not np.any(np.iscomplex(obj)):
    if not (
        isinstance(obj[0, 0, 0], complex)
        or isinstance(obj[0, 0, 0], np.complex64)
        or isinstance(obj[0, 0, 0], np.complex128)
    ):
        # Function was called maybe by mistake on a real (non-complex) array
        plot_2D_slices_middle_one_array3D(
            obj,
            voxel_sizes=voxel_sizes,
            ax=ax,
            fig=fig,
            fw=fw,
            fig_title=fig_title,
        )
        return

    if fig is None:
        fig, ax = plt.subplots(2, 3, figsize=(3 * fw, 2 * fw))

    plot_2D_slices_middle_only_module(
        obj, ax=ax[0], fig=fig, voxel_sizes=voxel_sizes, crop=crop
    )
    plot_2D_slices_middle_only_phase(
        obj,
        support=support,
        threshold_module=threshold_module,
        ax=ax[1],
        fig=fig,
        voxel_sizes=voxel_sizes,
        crop=crop,
        unwrap=unwrap,
    )

    ax[0, 0].set_ylabel("module", fontsize=20)
    ax[1, 0].set_ylabel("phase", fontsize=20)

    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=20)

    fig.tight_layout()

    if return_fig_ax:
        return fig, ax
    else:
        return


def plot_2D_slices_middle_object_list(obj_list, llk_list=None, crop=False, fw=3):
    nb_obj = len(obj_list)

    for n in range(nb_obj):
        obj = obj_list[n]

        fig, ax = plt.subplots(3, 3, figsize=(3 * fw, 3 * fw))
        plot_module_histogram(obj, fig=fig, ax=ax[0, 1], crop=crop)
        plot_2D_slices_middle(obj, fig=fig, ax=ax[1:], crop=crop)
        if llk_list is not None:
            fig.suptitle("llk : {}".format(llk_list[n]), fontsize=20)
        fig.delaxes(ax[0, 0])
        fig.delaxes(ax[0, 2])

        fig.tight_layout()

    return


def plot_2D_slices_middle_and_histogram(
    obj,
    crop=False,
    support=None,
    threshold_module=None,
    unwrap=True,
    ax=None,
    fig=None,
    fw=3,
    fig_title=None,
    return_fig_ax=False,
):
    if fig is None:
        fig, ax = plt.subplots(3, 3, figsize=(fw * 3, fw * 3))

    plot_module_histogram(obj, fig=fig, ax=ax[0, 1], crop=crop)
    fig.delaxes(ax[0, 0])
    fig.delaxes(ax[0, 2])

    plot_2D_slices_middle(
        obj,
        support=support,
        threshold_module=threshold_module,
        unwrap=unwrap,
        fig=fig,
        ax=ax[1:],
        fw=fw,
        crop=crop,
    )

    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=20)
    fig.tight_layout()
    if return_fig_ax:
        return fig, ax
    else:
        return


def plot_2D_slices_middle_only_module_and_histogram(
    obj,
    crop=False,
    ax=None,
    fig=None,
    fw=3,
    fig_title=None,
    return_fig_ax=False,
):
    if fig is None:
        fig, ax = plt.subplots(2, 3, figsize=(fw * 3, fw * 2))

    plot_module_histogram(obj, fig=fig, ax=ax[0, 1], crop=crop)
    fig.delaxes(ax[0, 0])
    fig.delaxes(ax[0, 2])

    plot_2D_slices_middle_only_module(obj, fig=fig, ax=ax[1], fw=fw, crop=crop)

    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=20)
    fig.tight_layout()

    if return_fig_ax:
        return fig, ax
    else:
        return


######################################################################################################################################
##############################                Interactive object plot                  ###############################################
######################################################################################################################################

# from ipywidgets import interact
# def interactive_object_slice(obj):
#     %matplotlib widget

#     module, phase = GetCroppedModulePhase(obj)
#     size = module.shape[0]

#     fig, ax = plt.subplots(1,2, figsize=(8,4))
#     img0 = ax[0].imshow(module[size//2], cmap='gray_r', vmin=np.nanmin(module), vmax=np.nanmax(module))
#     img1 = ax[1].imshow(phase[size//2], cmap='hsv', vmin=np.nanmin(phase), vmax=np.nanmax(phase))

#     for n, img in enumerate([img0, img1]):
#         divider = make_axes_locatable(ax[n])
#         cax = divider.append_axes('right', size='5%', pad=0.05)
#         fig.colorbar(img, cax=cax, orientation='vertical')

#     fig.tight_layout()

#     @interact(w=(0,size-1))
#     def update(w = size//2):
#         img0.set_data(module[w])
#         img1.set_data(phase[w])
#         fig.canvas.draw_idle()

#         return


######################################################################################################################################
##############################                    Plot 2D object                       ###############################################
######################################################################################################################################


def plot_object_module_phase_2d(
    obj,
    voxel_sizes=None,
    llk=None,
    fig=None,
    ax=None,
    fig_title=None,
    vmin=None,
    vmax=None,
    extent=None,
    threshold_module=0.3,
    crop=True,
    apply_fftshift=False,
    unwrap=True,
):
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(8, 5))

    if voxel_sizes is not None:
        extent = [
            0,
            obj.shape[1] * voxel_sizes[1] * 0.1,
            0,
            obj.shape[0] * voxel_sizes[0] * 0.1,
        ]

    # shape = obj.shape
    module, phase = get_cropped_module_phase(
        obj,
        threshold_module=threshold_module,
        crop=crop,
        apply_fftshift=apply_fftshift,
        unwrap=unwrap,
    )

    img1 = ax[0].matshow(module, cmap="gray_r", extent=extent)
    ax[0].set_title("module", fontsize=20)

    if llk is not None:
        fig.suptitle("llk : {}".format(llk), fontsize=15)

    img2 = ax[1].matshow(phase, cmap="hsv", vmin=vmin, vmax=vmax, extent=extent)
    ax[1].set_title("phase", fontsize=20)

    for ii, img in enumerate([img1, img2]):
        divider = make_axes_locatable(ax[ii])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(img, cax=cax, orientation="vertical")

    if voxel_sizes is not None:
        for n in range(2):
            ax[n].set_xlabel("nm", fontsize=15)
            ax[n].xaxis.set_ticks_position("bottom")

    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=20)

    fig.tight_layout()
    return


def plot_object_list_module_phase_2d(obj_list):
    fig, ax = plt.subplots(len(obj_list), 2, figsize=(8, 4 * len(obj_list)))
    for n in range(len(obj_list)):
        plot_object_module_phase_2d(obj_list[n], fig=fig, ax=ax[n])
    return


def plot_object_module_phase_and_histogram_2d(
    obj, threshold_module=0.3, voxel_sizes=None, fig=None, ax=None, bins=None
):
    if ax is None:
        fig, ax = plt.subplots(1, 3, figsize=(12, 5))

    plot_object_module_phase_2d(
        obj,
        fig=fig,
        ax=ax[:2],
        threshold_module=threshold_module,
        voxel_sizes=voxel_sizes,
    )
    plot_module_histogram(obj, bins=bins, fig=fig, ax=ax[2])
    fig.tight_layout()
    return


######################################################################################################################################
######################                Compare experimental data and reconstruction                   #################################
######################################################################################################################################


def compare_reconstuction_to_real_data(data, obj):

    I_recon = np.abs(ifftshift(fftn(fftshift(obj)))) ** 2.0

    if data.ndim == 3:
        plot_3D_projections(data, fig_title="experimental data")
        plot_3D_projections(I_recon, fig_title="reconstructed data")
    if data.ndim == 2:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].matshow(np.log(data), cmap="gray")
        ax[0].set_title("experimental", fontsize=20)
        ax[1].matshow(np.log(I_recon), cmap="gray")
        ax[1].set_title("reconstructed data", fontsize=20)
        fig.tight_layout()
    #         plot_2d_intensity_data(data, fig_title='experimental data')
    #         plot_2d_intensity_data(I_recon, fig_title='reconstructed data')
    return


######################################################################################################################################
###############################           Support projection for ROI selection              ##########################################
######################################################################################################################################


def plot_support_projection(obj, threshold_module=0.3, fw=3):
    module = np.abs(obj)
    support = module > threshold_module * np.max(module)

    if obj.ndim == 3:
        xlabels = ["axis 2", "axis 2", "axis 1"]
        ylabels = ["axis 1", "axis 0", "axis 0"]

    fig, ax = plt.subplots(1, obj.ndim, figsize=(fw * obj.ndim, fw))
    for axis in range(obj.ndim):
        projection = np.max(support, axis=axis)
        ax[axis].matshow(projection, cmap="gray_r", aspect="auto")
        ax[axis].set_xlabel(xlabels[axis], fontsize=15 * fw / 3.0)
        ax[axis].set_ylabel(ylabels[axis], fontsize=15 * fw / 3.0)
        ax[axis].set_title(f"projection along axis {axis}", fontsize=12 * fw / 3.0)
    fig.tight_layout()
    return


######################################################################################################################################
################################                  Final pretty figure                     ############################################
######################################################################################################################################


def final_figure(
    module_ortho,
    phase_ortho,
    strain,  # d_spacing,
    voxel_sizes=None,
    threshold=0.1,
    factor=0.2,
    fw=3,
    return_figure=False,
):
    fig, ax = plt.subplots(3, 3, figsize=(fw * 3, fw * 3))

    roi = automatic_object_roi(module_ortho, threshold=threshold, factor=factor)

    plot_2D_slices_middle_one_array3D(
        module_ortho[roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]],
        cmap="gray_r",
        voxel_sizes=voxel_sizes,
        add_colorbar=True,
        fig=fig,
        ax=ax[:, 0],
    )

    plot_2D_slices_middle_one_array3D(
        phase_ortho[roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]],
        cmap="hsv",
        voxel_sizes=voxel_sizes,
        add_colorbar=True,
        fig=fig,
        ax=ax[:, 1],
    )

    plot_2D_slices_middle_one_array3D(
        1e2 * strain[roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]],
        cmap="coolwarm",
        voxel_sizes=voxel_sizes,
        add_colorbar=True,
        fig=fig,
        ax=ax[:, 2],
    )

    ax[0, 0].set_title("module", fontsize=20 * fw / 3)
    ax[0, 1].set_title("phase", fontsize=20 * fw / 3)
    ax[0, 2].set_title("strain (%)", fontsize=20 * fw / 3)

    for axe in ax.flatten():
        axe.tick_params(axis="both", which="major", labelsize=15 * fw / 4)
        axe.xaxis.get_label().set_fontsize(20 * fw / 3)

    fig.tight_layout()

    if return_figure:
        return fig, ax
    else:
        return


def final_figure_version2(
    module_ortho,
    phase_ortho,
    strain,  # d_spacing,
    tilt_comp1,
    tilt_comp2,
    voxel_sizes=None,
    threshold=0.1,
    factor=0.2,
    fw=2.7,
    fig_title=None,
    return_figure=False,
):
    fig, ax = plt.subplots(3, 5, figsize=(fw * 5, fw * 3))

    roi = automatic_object_roi(module_ortho, threshold=threshold, factor=factor)
    xlabel = ["Z (nm)", "Z (nm)", "Y (nm)"]
    ylabel = ["Y", "X", "X"]

    plot_2D_slices_middle_one_array3D(
        module_ortho[roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]],
        cmap="gray_r",
        voxel_sizes=voxel_sizes,
        add_colorbar=True,
        xlabel=xlabel,
        ylabel=ylabel,
        fig=fig,
        ax=ax[:, 0],
    )

    plot_2D_slices_middle_one_array3D(
        phase_ortho[roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]],
        cmap="hsv",
        voxel_sizes=voxel_sizes,
        add_colorbar=True,
        fig=fig,
        ax=ax[:, 1],
    )

    plot_2D_slices_middle_one_array3D(
        1e2 * strain[roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]],
        cmap="coolwarm",
        voxel_sizes=voxel_sizes,
        add_colorbar=True,
        fig=fig,
        ax=ax[:, 2],
    )

    plot_2D_slices_middle_one_array3D(
        tilt_comp1[roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]],
        cmap="coolwarm",
        voxel_sizes=voxel_sizes,
        add_colorbar=True,
        fig=fig,
        ax=ax[:, 3],
    )

    plot_2D_slices_middle_one_array3D(
        tilt_comp2[roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]],
        cmap="coolwarm",
        voxel_sizes=voxel_sizes,
        add_colorbar=True,
        fig=fig,
        ax=ax[:, 4],
    )

    ax[0, 0].set_title("module", fontsize=20 * fw / 3)
    ax[0, 1].set_title("phase", fontsize=20 * fw / 3)
    ax[0, 2].set_title("strain (%)", fontsize=20 * fw / 3)
    ax[0, 3].set_title("tilt along X", fontsize=20 * fw / 3)
    ax[0, 4].set_title("tilt along Y", fontsize=20 * fw / 3)

    for axe in ax.flatten():
        axe.tick_params(axis="both", which="major", labelsize=15 * fw / 4)
        axe.xaxis.get_label().set_fontsize(20 * fw / 3)
        axe.yaxis.get_label().set_fontsize(20 * fw / 3)

    if fig_title is None:
        fig_title = ""
    fig_title += "\nBragg wavevector along Z"
    fig.suptitle(fig_title, fontsize=15)

    fig.tight_layout()

    if return_figure:
        return fig, ax
    else:
        return


def final_figure_version3(
    module_ortho,
    phase_ortho,
    strain,
    d_spacing,
    voxel_sizes=None,
    make_roi=True,
    threshold=0.1,
    factor=0.2,
    fw=2.7,
    fig_title=None,
    return_figure=False,
):
    fig, ax = plt.subplots(3, 4, figsize=(fw * 4, fw * 3))

    if make_roi:
        roi = automatic_object_roi(module_ortho, threshold=threshold, factor=factor)
    else:
        roi = [None, None, None, None, None, None]
    xlabel = ["Z (nm)", "Z (nm)", "Y (nm)"]
    ylabel = ["Y", "X", "X"]

    plot_2D_slices_middle_one_array3D(
        apply_roi(module_ortho, roi),
        cmap="gray_r",
        voxel_sizes=voxel_sizes,
        add_colorbar=True,
        xlabel=xlabel,
        ylabel=ylabel,
        fig=fig,
        ax=ax[:, 0],
    )

    plot_2D_slices_middle_one_array3D(
        apply_roi(phase_ortho, roi),
        cmap="hsv",
        voxel_sizes=voxel_sizes,
        add_colorbar=True,
        fig=fig,
        ax=ax[:, 1],
    )

    plot_2D_slices_middle_one_array3D(
        1e2 * apply_roi(strain, roi),
        cmap="coolwarm",
        voxel_sizes=voxel_sizes,
        add_colorbar=True,
        cbar_position="right",
        fig=fig,
        ax=ax[:, 2],
    )

    plot_2D_slices_middle_one_array3D(
        apply_roi(d_spacing, roi),
        cmap="coolwarm",
        voxel_sizes=voxel_sizes,
        add_colorbar=True,
        cbar_position="left",
        cbar_label=r"d spacing ($\AA$)",
        fig=fig,
        ax=ax[:, 3],
    )

    ax[0, 0].set_title("module", fontsize=20 * fw / 3)
    ax[0, 1].set_title("phase", fontsize=20 * fw / 3)
    ax[0, 2].set_title("strain (%)", fontsize=20 * fw / 3)
    ax[0, 3].set_title(r"d-spacing ($\AA$)", fontsize=20 * fw / 3)

    for axe in ax.flatten():
        axe.tick_params(axis="both", which="major", labelsize=15 * fw / 4)
        axe.xaxis.get_label().set_fontsize(20 * fw / 3)
        axe.yaxis.get_label().set_fontsize(20 * fw / 3)

    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=15)

    fig.tight_layout()

    for n in range(3):
        fig.delaxes(ax[n, 3])

    if return_figure:
        return fig, ax
    else:
        return


def final_figure_version4(
    module_ortho,
    phase_ortho,
    strain,
    d_spacing,
    tilt_magn,
    voxel_sizes=None,
    make_roi=True,
    threshold=0.1,
    factor=0.2,
    fw=2.7,
    fig_title=None,
    return_figure=False,
):
    fig, ax = plt.subplots(3, 5, figsize=(fw * 5, fw * 3))

    if make_roi:
        roi = automatic_object_roi(module_ortho, threshold=threshold, factor=factor)
    else:
        roi = [None, None, None, None, None, None]
    xlabel = ["Z (nm)", "Z (nm)", "Y (nm)"]
    ylabel = ["Y", "X", "X"]

    plot_2D_slices_middle_one_array3D(
        apply_roi(module_ortho, roi),
        cmap="gray_r",
        voxel_sizes=voxel_sizes,
        add_colorbar=True,
        xlabel=xlabel,
        ylabel=ylabel,
        fig=fig,
        ax=ax[:, 0],
    )

    plot_2D_slices_middle_one_array3D(
        apply_roi(phase_ortho, roi),
        cmap="hsv",
        voxel_sizes=voxel_sizes,
        add_colorbar=True,
        fig=fig,
        ax=ax[:, 1],
    )

    plot_2D_slices_middle_one_array3D(
        1e2 * apply_roi(tilt_magn, roi),
        cmap="gray_r",
        voxel_sizes=voxel_sizes,
        add_colorbar=True,
        fig=fig,
        ax=ax[:, 2],
    )

    plot_2D_slices_middle_one_array3D(
        1e2 * apply_roi(strain, roi),
        cmap="coolwarm",
        voxel_sizes=voxel_sizes,
        add_colorbar=True,
        cbar_position="right",
        fig=fig,
        ax=ax[:, 3],
    )

    plot_2D_slices_middle_one_array3D(
        apply_roi(d_spacing, roi),
        cmap="coolwarm",
        voxel_sizes=voxel_sizes,
        add_colorbar=True,
        cbar_position="left",
        cbar_label=r"d spacing ($\AA$)",
        fig=fig,
        ax=ax[:, 4],
    )

    ax[0, 0].set_title("module", fontsize=20 * fw / 3)
    ax[0, 1].set_title("phase", fontsize=20 * fw / 3)
    ax[0, 2].set_title("tilt magnitude (%)", fontsize=20 * fw / 3)
    ax[0, 3].set_title("strain (%)", fontsize=20 * fw / 3)
    ax[0, 4].set_title(r"d-spacing ($\AA$)", fontsize=20 * fw / 3)

    for axe in ax.flatten():
        axe.tick_params(axis="both", which="major", labelsize=15 * fw / 4)
        axe.xaxis.get_label().set_fontsize(20 * fw / 3)
        axe.yaxis.get_label().set_fontsize(20 * fw / 3)

    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=15)

    fig.tight_layout()

    for n in range(3):
        fig.delaxes(ax[n, 4])

    if return_figure:
        return fig, ax
    else:
        return


def plot_strain_histo(strain):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    _ = ax.hist(1e2 * strain.flatten(), bins=50)
    ax.set_xlabel("strain (%)", fontsize=15)
    ax.set_title("strain histogram", fontsize=20)
    return


######################################################################################################################################
############################                  Bragg planes schematic figure                  #########################################
######################################################################################################################################

# from matplotlib import cm
# def schematic_Bragg_planes_figure(displacement, qcen,
#                                   strain=None,
#                                   module_ortho=None,
#                                   axis=0, slice_index=None,
#                                   visual_factor=50,
#                                   close_roi=False,
#                                   fig=None, ax=None, fw=6):
#     angle_bragg_last_axis = np.rad2deg(np.arccos(np.dot(qcen, [0,0,1])/ np.linalg.norm(qcen)))
#     if angle_bragg_last_axis > 1.:
#         raise ValueError('qcen is not along the last axis ! My function is not ready for that.')
#         return

#     if axis==2 :
#         raise ValueError('axis should be 0 or 1!')

#     if slice_index is None:
#         s = slice_middle_array_along_axis(displacement, axis=axis)
#     else:
#         s = [slice(None, None, None) for ii in range(displacement.ndim)]
#         s[axis] = slice_index
#         s = tuple(s)

#     if fig is None:
#         fig,ax = plt.subplots(1,1, figsize=(fw,fw))

#     if strain is not None:
#         strain_color = strain-np.nanmin(strain)
#         strain_color = strain_color/np.nanmax(strain_color)
#         strain_color_slice = strain_color[s]

#     displacement_slice = displacement[s]
#     voxel_sizes_slice = np.delete(voxel_sizes,axis)

#     # Create a regular grid
#     pos = np.indices(displacement_slice.shape)
#     pos[0] = pos[0]*voxel_sizes_slice[0]*.1
#     pos[1] = pos[1]*voxel_sizes_slice[1]*.1

#     pos = pos + .1*displacement_slice[None] * np.array([0,1])[:,None,None] * visual_factor

# #     mean = np.nanmean(pos[0])
# #     pos[0] = -(pos[0]-mean) + mean
#     for n in range(displacement_slice.shape[1]):
#         x = pos[1,:,n]
#         y = pos[0,:,n]
#         strain_line = strain_color_slice[:,n]
#         color = cm.jet(strain_line)
#         for i in np.arange(len(x)-1):
#             ax.plot([x[i],x[i+1]], [y[i],y[i+1]], c=color[i], linewidth=3*fw/10)
# #         ax.plot(pos[1,:,n], pos[0,:,n], 'k-', linewidth=1, markersize=1, alpha=.3)


#         if strain is not None:
#             strain_line = strain_color_slice[:,n]
#             color = cm.coolwarm(strain_line)
# #             color = cm.bwr(strain_line)
#         else:
#             color = 'k'
# #         ax.scatter(pos[1,:,n], pos[0,:,n], s=60*fw/10., color=color)

#     ax.set_xlabel('nm', fontsize=15*fw/4.)

#     if not close_roi :
#         ax.set_xlim(0, displacement_slice.shape[1]*voxel_sizes_slice[1]*.1)
#         ax.set_ylim(0, displacement_slice.shape[0]*voxel_sizes_slice[0]*.1)
# #     else:
# #         ax.set_xlim(np.nanmin(pos[1]), np.nanmax(pos[1]))
# #         ax.set_ylim(np.nanmin(pos[0]), np.nanmax(pos[0]))


#     ax.invert_yaxis()

#     ax.set_title('Schematic Bragg\nplanes distortion', fontsize=12*fw/4.)
#     ax.set_aspect('equal', 'box')

#     return

# def final_figure_schematic_Bragg_planes(displacement, module_ortho, strain,
#                                         qcen,
#                                        threshold=0.1, factor=0.2,
#                                        visual_factor = 500,
#                                        fw = 10):
#     roi = automatic_object_roi(module_ortho, threshold=threshold, factor=factor)
#     module_ortho_roi = apply_roi(module_ortho, roi)

#     fig,ax = plt.subplots(1,2, figsize=(fw*2,fw))

#     extent = [[0,module_ortho_roi.shape[2]*voxel_sizes[2]*.1, module_ortho_roi.shape[1]*voxel_sizes[1]*.1, 0],
#               [0,module_ortho_roi.shape[2]*voxel_sizes[2]*.1, module_ortho_roi.shape[0]*voxel_sizes[0]*.1, 0]]
#     for axis in [0,1]:
#         module_ortho_slice = module_ortho_roi[slice_middle_array_along_axis(module_ortho_roi,axis=axis)]
#         ax[axis].matshow(module_ortho_slice, cmap='gray_r', extent=extent[axis], alpha=.5)

#         schematic_Bragg_planes_figure(apply_roi(displacement, roi), qcen,
#                                       strain = apply_roi(strain,roi),
#                                           axis=axis,
#                                           visual_factor=visual_factor,
#                                           fig=fig, ax=ax[axis])
#         ax[axis].set_xlabel('nm',fontsize=12*fw/4.)
#         ax[axis].xaxis.set_ticks_position('bottom')
#         ax[axis].xaxis.set_tick_params(labelsize=8*fw/4.)
#         ax[axis].yaxis.set_tick_params(labelsize=8*fw/4.)
#         ax[axis].set_title(f'slice along axis {axis}',fontsize=12*fw/4.)


#     fig.tight_layout()

#     return


def schematic_Bragg_planes_figure(
    displacement,
    qcen,
    voxel_sizes,
    strain=None,
    module_ortho=None,
    axis=0,
    slice_index=None,
    visual_factor=50,
    dislo_threshold=None,
    close_roi=False,
    fig=None,
    ax=None,
    fw=6,
    fig_title=None,
):
    angle_bragg_last_axis = np.rad2deg(
        np.arccos(np.dot(qcen, [0, 0, 1]) / np.linalg.norm(qcen))
    )
    if angle_bragg_last_axis > 1.0:
        raise ValueError(
            "qcen is not along the last axis ! My function is not ready for that."
        )
        return

    if axis == 2:
        raise ValueError("axis should be 0 or 1!")

    if slice_index is None:
        s = slice_middle_array_along_axis(displacement, axis=axis)
    else:
        s = [slice(None, None, None) for ii in range(displacement.ndim)]
        s[axis] = slice_index
        s = tuple(s)

    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(fw, fw))

    if strain is not None:
        strain_color_slice = np.copy(strain[s])
        strain_color_slice = strain_color_slice - np.nanmin(strain_color_slice)
        strain_color_slice = strain_color_slice / np.nanmax(strain_color_slice)

    displacement_slice = displacement[s]
    voxel_sizes_slice = np.delete(voxel_sizes, axis)

    # Create a regular grid
    pos = np.indices(displacement_slice.shape)
    pos[0] = pos[0] * voxel_sizes_slice[0] * 0.1
    pos[1] = pos[1] * voxel_sizes_slice[1] * 0.1

    pos = (
        pos
        + 0.1
        * displacement_slice[None]
        * np.array([0, 1])[:, None, None]
        * visual_factor
    )

    for n in range(displacement_slice.shape[1]):
        # Just a quick test
        if dislo_threshold is not None:
            grad_amp = np.abs(np.gradient(pos[1, :, n]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                indices = grad_amp > np.nanmedian(grad_amp) * 20
            pos[:, indices, n] = np.nan
        ax.plot(
            pos[1, :, n],
            pos[0, :, n],
            "k-",
            linewidth=1,
            markersize=1,
            alpha=1,
        )

        if strain is not None:
            strain_line = strain_color_slice[:, n]
            color = cm.coolwarm(strain_line)
            #             color = cm.jet(strain_line)
            ax.scatter(
                pos[1, :, n],
                pos[0, :, n],
                s=20 * fw / 10.0,
                color=color,
                alpha=1,
            )

    ax.set_xlabel("nm", fontsize=15 * fw / 4.0)

    if strain is not None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        norm = matplotlib.colors.Normalize(
            vmin=np.nanmin(1e2 * strain[s]), vmax=np.nanmax(1e2 * strain[s])
        )
        cbar = fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap="coolwarm"),
            cax=cax,
            orientation="vertical",
        )  # , label='Strain (%)')
        cbar.set_label("strain (%)", size=20 * fw / 6)

    if not close_roi:
        ax.set_xlim(0, displacement_slice.shape[1] * voxel_sizes_slice[1] * 0.1)
        ax.set_ylim(0, displacement_slice.shape[0] * voxel_sizes_slice[0] * 0.1)

    ax.invert_yaxis()

    if fig_title is None:
        fig_title = "Schematic Bragg\nplanes distortion"
    ax.set_title(fig_title, fontsize=12 * fw / 4.0)
    ax.set_aspect("equal", "box")

    return


def final_figure_schematic_Bragg_planes(
    displacement,
    qcen,
    voxel_sizes,
    strain=None,
    visual_factor=200,
    dislo_threshold=None,
    close_roi=True,
    fw=6,
):
    fig, ax = plt.subplots(1, 2, figsize=(2 * fw, fw))

    for n, axis in enumerate([0, 1]):
        schematic_Bragg_planes_figure(
            displacement,
            qcen,
            voxel_sizes,
            strain=strain,
            axis=axis,
            visual_factor=visual_factor,
            dislo_threshold=dislo_threshold,
            close_roi=True,
            fig=fig,
            ax=ax[n],
            fig_title=f"slice along axis {axis}",
        )

    fig.suptitle("Schematic Bragg planes", fontsize=20 * fw / 6.0)
    fig.tight_layout()
    return


######################################################################################################################################
#############################                  Final pretty figure 2D BCDI                  ##########################################
######################################################################################################################################


def Bragg_planes_schematic_figure_2D(
    displacement,
    u_bragg,
    voxel_sizes,
    visual_factor=50,
    fig=None,
    ax=None,
    title=True,
    fw=4,
):
    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(fw, fw))

    # Create a regular grid
    pos = np.indices(displacement.shape)
    pos[0] = pos[0] * voxel_sizes[0] * 0.1
    pos[1] = pos[1] * voxel_sizes[1] * 0.1

    pos = pos + displacement[None] * u_bragg[:, None, None] * visual_factor

    for n in range(displacement.shape[1]):
        ax.plot(pos[1, :, n], pos[0, :, n], "k.-")
    ax.set_xlabel("nm", fontsize=15 * fw / 4.0)

    if title:
        #         ax.set_title('Schematic Bragg\nplanes distortion', fontsize=15*fw/4.)
        ax.set_title("Schematic planes", fontsize=15 * fw / 4.0)

    ax.set_aspect("equal", "box")
    return


def final_figure_2d(
    obj,
    strain,
    d_spacing,
    tilt_angle,
    displacement,
    u_bragg,
    voxel_sizes,
    visual_factor=20,
    fw=3,
):
    module, phase = get_cropped_module_phase(obj, crop=False)

    fig, ax = plt.subplots(2, 3, figsize=(fw * 3.3, fw * 2))

    plot_global_2d(module, fig=fig, ax=ax[0, 0], cmap="gray_r", voxel_sizes=voxel_sizes)
    plot_global_2d(phase, fig=fig, ax=ax[0, 1], cmap="hsv", voxel_sizes=voxel_sizes)
    plot_symmetric_colorscale_2d(
        1e2 * strain,
        fig=fig,
        ax=ax[0, 2],
        cmap="coolwarm",
        voxel_sizes=voxel_sizes,
    )
    plot_global_2d(
        d_spacing,
        fig=fig,
        ax=ax[1, 0],
        cmap="coolwarm",
        voxel_sizes=voxel_sizes,
    )
    plot_global_2d(
        tilt_angle,
        fig=fig,
        ax=ax[1, 1],
        cmap="coolwarm",
        voxel_sizes=voxel_sizes,
    )

    Bragg_planes_schematic_figure_2D(
        displacement,
        u_bragg,
        voxel_sizes,
        visual_factor=visual_factor,
        fig=fig,
        ax=ax[1, 2],
        title=True,
    )

    ax[0, 0].set_title("module (a.u.)", fontsize=15 * fw / 3)
    ax[0, 1].set_title("phase (rad)", fontsize=15 * fw / 3)
    ax[0, 2].set_title("strain (%)", fontsize=15 * fw / 3)
    ax[1, 0].set_title(r"d-spacing ($\AA$)", fontsize=15 * fw / 3)
    ax[1, 1].set_title("tilt (degrees)", fontsize=15 * fw / 3)

    for axe in ax.flatten():
        axe.tick_params(axis="both", which="major", labelsize=12 * fw / 4)
        axe.xaxis.get_label().set_fontsize(15 * fw / 3)

    fig.tight_layout()

    return


######################################################################################################################################
##############################                 Surfaces projections                    ###############################################
######################################################################################################################################


def one_surface_projection(strain, axis, inverse):
    strain_used = np.copy(strain)
    if inverse:
        strain_used = np.flip(strain_used, axis=axis)

    # I have strain=nan outside the support so I can define the support here
    # If not, use your own support as input
    support = 1 - np.isnan(strain_used)
    support_surface = np.cumsum(support, axis=axis)
    support_surface[support_surface > 1] = 0

    surface_strain = np.copy(strain_used)
    surface_strain[support_surface == 0] = np.nan
    surface_strain = np.nanmean(surface_strain, axis=axis)
    return surface_strain


def plot_surface_projections(
    strain, voxel_sizes=None, fw=3, fig_title=None, vmin=None, vmax=None
):
    if vmin is None:
        vmin = np.nanmin(1e2 * strain)
    if vmax is None:
        vmax = np.nanmax(1e2 * strain)

    fig, ax = plt.subplots(2, 3, figsize=(fw * 3.3, fw * 2.2))

    if voxel_sizes is not None:
        extent = [
            [
                0,
                strain.shape[2] * voxel_sizes[2] * 0.1,
                0,
                strain.shape[1] * voxel_sizes[1] * 0.1,
            ],
            [
                0,
                strain.shape[2] * voxel_sizes[2] * 0.1,
                0,
                strain.shape[0] * voxel_sizes[0] * 0.1,
            ],
            [
                0,
                strain.shape[1] * voxel_sizes[1] * 0.1,
                0,
                strain.shape[0] * voxel_sizes[0] * 0.1,
            ],
        ]
    else:
        extent = [None, None, None]
    imgs = []
    for n1, axis in enumerate(range(3)):
        for n0, inverse in enumerate([False, True]):
            surface_strain = one_surface_projection(strain, axis, inverse)
            imgs.append(
                ax[n0, n1].matshow(
                    1e2 * surface_strain,
                    cmap="coolwarm",
                    extent=extent[n1],
                    vmin=vmin,
                    vmax=vmax,
                )
            )

    cax = fig.add_axes([1, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(imgs[-1], cax=cax, orientation="vertical")
    cbar.ax.tick_params(labelsize=20 * fw / 4.0)
    cbar.ax.locator_params(nbins=5)
    cbar.set_label("strain (%)", rotation=270, fontsize=30 * fw / 4.0, labelpad=15)

    for axe in ax.flatten():
        axe.locator_params(nbins=4)

    xlabel = ["Z (nm)", "Z (nm)", "Y (nm)"]
    ylabel = ["Y (nm)", "X (nm)", "X (nm)"]
    for n in range(3):
        for ii in range(2):
            ax[ii, n].set_xlabel(xlabel[n], fontsize=15 * fw / 4.0)
            ax[ii, n].xaxis.set_ticks_position("bottom")
            ax[ii, n].set_ylabel(ylabel[n], fontsize=15 * fw / 4.0)

    title_list = [
        ["along +X", "along +Y", "along +Z"],
        ["along -X", "along -Y", "along -Z"],
    ]
    for n in range(3):
        for ii in range(2):
            ax[ii, n].set_title(title_list[ii][n], fontsize=17 * fw / 4.0)

    if fig_title is not None:
        fig_title += "   surface projection"
        fig.suptitle(fig_title, fontsize=fw * 22 / 4.0)

    fig.tight_layout()
    return


######################################################################################################################################
##############################                Interactive slice plot                   ###############################################
######################################################################################################################################


def interactive_3d_object(obj, threshold_module=None, axis=0):
    ipython = get_ipython()
    if ipython is not None:
        ipython.magic("matplotlib widget")
    # %matplotlib widget

    module, phase = get_cropped_module_phase(
        obj, unwrap=True, threshold_module=threshold_module
    )
    shape = module.shape

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    im0 = ax[0].matshow(module.take(indices=shape[axis] // 2, axis=axis), cmap="gray_r")
    im1 = ax[1].matshow(phase.take(indices=shape[axis] // 2, axis=axis), cmap="hsv")

    @interact(w=(0, shape[axis] - 1))
    def update(w=0):
        im0.set_data(module.take(indices=w, axis=axis))
        im0.set_clim(np.nanmin(module), np.nanmax(module))

        im1.set_data(phase.take(indices=w, axis=axis))
        im1.set_clim(np.nanmin(phase), np.nanmax(phase))

        fig.canvas.draw_idle()

        return


def interactive_3d_array(
    array,
    axis=0,
    voxel_sizes=None,
    cmap="coolwarm",
    vmin=None,
    vmax=None,
    symmetric_colorscale=False,
):
    if symmetric_colorscale:
        cmap = "bwr"
        vmax = np.nanmax(np.abs(array))
        vmin = -vmax

    if voxel_sizes is not None:
        voxel_sizes = 0.1 * np.array(voxel_sizes)  # Put the voxel_sizes in nanometers
        extent = [
            [
                0,
                array.shape[2] * voxel_sizes[2],
                0,
                array.shape[1] * voxel_sizes[1],
            ],
            [
                0,
                array.shape[2] * voxel_sizes[2],
                0,
                array.shape[0] * voxel_sizes[0],
            ],
            [
                0,
                array.shape[1] * voxel_sizes[1],
                0,
                array.shape[0] * voxel_sizes[0],
            ],
        ]
    else:
        extent = [None, None, None]

    ipython = get_ipython()
    if ipython is not None:
        ipython.magic("matplotlib widget")
    # %matplotlib widget

    shape = array.shape

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    im0 = ax.matshow(
        array.take(indices=shape[axis] // 2, axis=axis),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent[axis],
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im0, cax=cax, orientation="vertical")

    @interact(w=(0, shape[axis] - 1))
    def update(w=0):
        im0.set_data(array.take(indices=w, axis=axis))
        #         im0.set_clim(vmin, vmax)

        fig.canvas.draw_idle()

        return
