import os
import numpy as np
import pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def check_path_create(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return


def get_npz_files(path):
    files_all = os.listdir(path)
    files = []
    for f in files_all:
        if ".npz" in f:
            files.append(path + f)
    return files


def get_numpy_files(path):
    files_all = os.listdir(path)
    files = []
    for f in files_all:
        if (".npz" in f) or (".npy" in f):
            files.append(path + f)
    return files


def center_of_mass_array_1D(x, array1d):
    proba = array1d / np.sum(array1d)
    return np.sum(x * proba)


def pearson_coef(img1, img2):
    x = img1 - np.nanmean(img1)
    y = img2 - np.nanmean(img2)
    numerator = np.sum(x * y)
    denominator = np.sqrt(np.sum(x**2.0) * np.sum(y**2.0))
    return numerator / denominator


# from scipy.ndimage.measurements import center_of_mass


def center_of_mass(array):
    cen = np.zeros(array.ndim)
    pos = np.indices(array.shape)
    proba = array / np.nansum(array)
    for n in range(array.ndim):
        cen[n] += np.nansum(pos[n] * proba)
    return cen


def center_of_mass_calculation_two_steps(data, crop=50, plot=False):
    center = np.unravel_index(np.nanargmax(data), data.shape)

    cropping_dim = []
    for n in range(data.ndim):
        cropping_dim.append(
            [
                max([0, int(center[n] - crop / 2)]),
                min(int(center[n] + crop // 2), data.shape[n] - 1),
            ]
        )

    s = [
        slice(cropping_dim[n][0], cropping_dim[n][1]) for n in range(data.ndim)
    ]
    center2 = center_of_mass(data[tuple(s)])

    center = [
        int(round(cropping_dim[n][0] + center2[n])) for n in range(data.ndim)
    ]

    if plot:
        from cdi_dislo.ewen_utilities.plot_utilities import plot_3D_projections

        if data.ndim == 3:
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            plot_3D_projections(data, fig=fig, ax=ax)
            ax[0].scatter(center[2], center[2], color="w")
            ax[1].scatter(center[2], center[0], color="w")
            ax[2].scatter(center[1], center[0], color="w")
        if data.ndim == 2:
            fig = plt.figure(figsize=(10, 10))
            plt.imshow(np.log(data), cmap="plasma", vmin=1)
            plt.colorbar()
            plt.scatter(center[1], center[0], color="w")
    return center


def center_the_center_of_mass(
    data,
    qx=None,
    qy=None,
    qz=None,
    standard_com=False,
    plot=False,
    vmin=None,
    return_offsets=False,
    cmap="plasma",
    norm=None,
    scatter_color="g",
    scatter_size=10,
):
    """
    Center the center of mass of a 3D matrix
    I use this to center the Bragg peak after the small random shift I did in the function "Createqxqyqz"
    """
    shape = data.shape

    data[~np.isfinite(data)] = 0

    # Calculate where is the center of mass
    if standard_com:
        com = center_of_mass(data)
    else:
        com = center_of_mass_calculation_two_steps(data)

    # Calculate what's the offset to put back the center of mass at the middle of the 3D matrix
    offset = [int(np.rint(shape[n] / 2.0 - com[n])) for n in range(len(shape))]

    # Put back the center of mass to the middle of the matrix
    data_cen = np.roll(data, offset, axis=range(len(shape)))
    if qx is not None:
        qx = np.roll(qx, offset, axis=range(len(shape)))
        qy = np.roll(qy, offset, axis=range(len(shape)))
        qz = np.roll(qz, offset, axis=range(len(shape)))
    if plot:
        if len(shape) == 2:
            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            ax[0].imshow(data, cmap=cmap, vmin=vmin, norm=norm)
            ax[0].scatter(com[1], com[0], c=scatter_color, s=scatter_size)
            ax[1].imshow(data_cen, cmap=cmap, vmin=vmin, norm=norm)
    #         if len(shape)==3:
    #             plot_3D_projections(data, fig_title='original data')
    #             plot_3D_projections(data_cen, fig_title='centered data')

    if return_offsets:
        return data_cen, offset
    else:
        if qx is not None:
            return data_cen, qx, qy, qz
        else:
            return data_cen


def interpolation_xrayutilities_gridder(x, y, z, data, fuzzy_gridder=False):
    import xrayutilities as xu

    maxbins = []
    for dim in (x, y, z):
        maxstep = max((abs(np.diff(dim, axis=j)).max() for j in range(3)))
        maxbins.append(int(abs(dim.max() - dim.min()) / maxstep))

    if fuzzy_gridder:
        gridder = xu.FuzzyGridder3D(*maxbins)
    else:
        gridder = xu.Gridder3D(*maxbins)

    gridder(x, y, z, data)
    data_ortho = gridder.data
    x1d, y1d, z1d = [gridder.xaxis, gridder.yaxis, gridder.zaxis]

    return data_ortho, x1d, y1d, z1d


# def crop_array_symmetric(array, crop_array):
#     array_crop = array[crop_array[0]:-crop_array[0],
#                                  crop_array[1]:-crop_array[1],
#                                  crop_array[2]:-crop_array[2]]
#     return array_crop


def crop_array_symmetric(array, crop_array, inverse_crop=False):
    s = []
    for n in range(array.ndim):
        if inverse_crop:
            s.append(
                slice(
                    array.shape[n] // 2 - crop_array[n] // 2,
                    array.shape[n] // 2 + crop_array[n] // 2,
                )
            )
        else:
            if crop_array[n] != 0:
                s.append(slice(crop_array[n], -crop_array[n]))
            else:
                s.append(slice(None))
    array_crop = array[tuple(s)]
    return array_crop


def apply_roi(array, roi):
    s = [slice(roi[2 * n], roi[2 * n + 1]) for n in range(array.ndim)]
    return array[tuple(s)]


from scipy.ndimage import median_filter


def hot_pixel_filter(data, threshold=1e2):
    """
    Remove hot pixels (using a median filter) that can mess up the data preprocessing
    """
    data_median = median_filter(data, size=2)
    mask = data < threshold * (data_median + 1)
    data_clean = data * mask
    return data_clean


def add_colorbar_subplot(
    fig, axes, imgs, size="5%", tick_fontsize=12, return_cbar=False
):
    if not isinstance(imgs, list):
        imgs = [imgs]
        axes = [axes]

    cbar_list = []
    for im, ax in zip(imgs, np.array(axes).flatten()):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=size, pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        cax.tick_params(labelsize=tick_fontsize)  # Set tick fontsize here
        cbar_list.append(cbar)

    if return_cbar:
        return cbar_list
    else:
        return


def subplots_numerous_images(
    img_list,
    fw=4,
    ncol=4,
    extent=None,
    norm=None,
    vmin=None,
    vmax=None,
    cmap=None,
    colorbar=True,
    title_list=None,
    suptitle=None,
    return_fig_ax=False,
):
    nax = len(img_list)
    nrow = nax // ncol + (nax % ncol != 0)
    fig, ax = plt.subplots(nrow, ncol, figsize=(ncol * fw, nrow * fw))
    axe = ax.flatten()
    mat = []

    for n in range(len(axe)):
        if n < len(img_list):
            mat.append(
                axe[n].matshow(
                    img_list[n],
                    cmap=cmap,
                    aspect="auto",
                    vmin=vmin,
                    vmax=vmax,
                    extent=extent,
                    norm=norm,
                )
            )
            if title_list is not None:
                axe[n].set_title(title_list[n], fontsize=15 * fw / 4.0)
        else:
            fig.delaxes(axe[n])

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=20 * fw / 4.0)

    if colorbar:
        add_colorbar_subplot(fig, ax[: len(img_list)], mat)

    for one_axe in axe:
        one_axe.locator_params(axis="both", nbins=5)
        one_axe.xaxis.set_ticks_position("bottom")
        one_axe.tick_params(
            axis="both", which="major", labelsize=12 * fw / 4.0
        )

    fig.tight_layout()

    if return_fig_ax:
        return fig, axe
    else:
        return


from numpy.fft import fftshift, ifftshift, fftn, ifftn


def create_diffracted_amplitude(obj):
    return ifftshift(fftn(fftshift(obj)))


def create_object(fexp):
    return ifftshift(ifftn(fftshift(fexp)))


def slice_middle_array_along_axis(array, axis):
    s = [slice(None, None, None) for ii in range(array.ndim)]
    s[axis] = array.shape[axis] // 2
    return tuple(s)


def force_even_dimension_one_array(array, verbose=True):
    s = []
    for n in range(array.ndim):
        if array.shape[n] % 2 == 0:
            s.append(slice(None))
        else:
            s.append(slice(1, None, None))

    array_even = array[tuple(s)]

    if verbose:
        print("shape changed :")
        print("array {} to {}".format(array.shape, array_even.shape))
    return array_even


def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (
        (vec1 / np.linalg.norm(vec1)).reshape(3),
        (vec2 / np.linalg.norm(vec2)).reshape(3),
    )
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix


def create_radius_array(array):
    r = np.indices(array.shape)
    r = r - np.mean(r, axis=tuple(range(1, array.ndim + 1)))[:, None, None]
    r = np.sqrt(np.sum(r**2.0, axis=0))
    return r
