"""
####################################################################################################################
This script provides various functions for analyzing intensity distributions in 3D data sets.
It includes methods for:
- Plotting 1D intensity distributions along different axes and computing key statistics (FWHM, barycenter, etc.).
- Fitting different peak profiles (Gaussian, Lorentzian, Voigt, Pseudo-Voigt, etc.) to the intensity data.
- Calculating full-width at half maximum (FWHM) using different geometric and fitting methods.
- Evaluating skewness, kurtosis, and peak fitting quality using R-squared values.
- Displaying intensity sum plots with fitted profiles, mean values, and FWHM indicators.
- Providing advanced functions for noise handling, Scherrer equation calculations, and Bragg angle determination.

The script is intended for use in Bragg Coherent Diffraction Imaging (BCDI) and related 3D diffraction experiments.
####################################################################################################################
####################################################################################################################
Suggestions for Improvements:
1. **Code Optimization**:
   - Some sections of the script repeat similar operations (e.g., sum along different axes). Consider modularizing them.

2. **Exception Handling**:
   - The script uses multiple `try-except` blocks when fitting models. It would be beneficial to log or display errors
     to understand fitting failures.

3. **Performance Enhancements**:
   - Use `numba` or `cython` to speed up computation-heavy functions, such as peak fitting and FWHM calculations.

4. **Visualization Enhancements**:
   - Improve the readability of plots by adding gridlines, annotations, and dynamic axis scaling.
   - Use `matplotlib` subplots for better organization of multiple plots.

5. **Configurable Parameters**:
   - Consider allowing users to specify noise levels, fitting methods, and peak detection thresholds via function arguments.

6. **Generalization**:
   - Support arbitrary dimensionality for the input data, making the script adaptable for different experimental setups.

7. **Save and Export Options**:
   - Add functionality to save fitted results (e.g., JSON, CSV) for further analysis.
   - Enable saving of figures in high resolution for publication purposes.

8. **Documentation and Readability**:
   - Improve docstrings with detailed parameter descriptions, expected input types, and return values.
   - Consider using `logging` instead of `print()` statements for better debugging.
####################################################################################################################
"""

# from scipy.signal import find_peaks
# from scipy.optimize import curve_fit
# from matplotlib.colors import LinearSegmentedColormap
# from sklearn.metrics import r2_score
# from IPython.display import display
# import matplotlib.pyplot as plt
# from matplotlib import gridspec
# from matplotlib.table import Table
# from matplotlib import ticker as mticker
# from scipy.stats import skew, kurtosis
# from tabulate import tabulate
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib.colors import LogNorm
# from matplotlib.ticker import MaxNLocator

# from scipy.stats import norm
# import xrayutilities as xu

#####################################################################################################################
# def plot_single_data_distr(data, i_scan):
#     # from scipy.stats import norm
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import xrayutilities as xu

#     data_0 = xu.maplog(data.sum(axis=0).flatten())
#     # data_0=data_0[data_0!=0]
#     mean_0 = np.mean(data_0)  # norm.mean(dddata)
#     std_0 = np.std(data_0)  # norm.std(dddata)
#     fwhm_0 = 2 * std_0 * np.sqrt(2 * np.log(2))
#     # intensity_at_fwhm_0 = norm.pdf(mean_0 - fwhm_0 / 2)
#     barycenter_0 = np.mean(data_0)
#     # intensity_at_barycenter_0 = norm.pdf(barycenter_0)

#     data_1 = xu.maplog(data.sum(axis=1).flatten())
#     # data_1=data_1[data_1!=0]
#     mean_1 = np.mean(data_1)  # norm.mean(dddata)
#     std_1 = np.std(data_1)  # norm.std(dddata)
#     fwhm_1 = 2 * std_1 * np.sqrt(2 * np.log(2))
#     # intensity_at_fwhm_1 = norm.pdf(mean_1 - fwhm_1 / 2)
#     barycenter_1 = np.mean(data_1)
#     # intensity_at_barycenter_1 = norm.pdf(barycenter_1)

#     data_2 = xu.maplog(data.sum(axis=2).flatten())
#     # data_2=data_2[data_2!=0]
#     mean_2 = np.mean(data_2)  # norm.mean(dddata)
#     std_2 = np.std(data_2)  # norm.std(dddata)
#     fwhm_2 = 2 * std_2 * np.sqrt(2 * np.log(2))
#     # intensity_at_fwhm_2 = norm.pdf(mean_2 - fwhm_2 / 2)
#     barycenter_2 = np.mean(data_2)
#     # intensity_at_barycenter_2 = norm.pdf(barycenter_2)

#     f_s = 16
#     fig2 = plt.figure(1, figsize=(20, 4))
#     ax = plt.subplot(1, 3, 1)
#     plt.hist(data_0, bins=100, density=True)
#     plt.xlabel("Norm Intensity [0]", fontsize=f_s)
#     plt.ylabel("", fontsize=f_s)
#     plt.axvline(mean_0 - fwhm_0 / 2, color="red", linestyle="dashed")
#     plt.axvline(mean_0 + fwhm_0 / 2, color="red", linestyle="dashed")
#     plt.axvline(barycenter_0, color="blue", linestyle="dashed", label="mean")  # type: ignore
#     plt.legend()
#     # plt.yscale('log')
#     plt.axis("tight")
#     ax.tick_params(labelsize=f_s)
#     plt.grid(alpha=0.01)

#     ax = plt.subplot(1, 3, 2)
#     plt.hist(data_1, bins=100, density=True)
#     plt.xlabel("Norm Intensity [1]", fontsize=f_s)
#     plt.axvline(mean_1 - fwhm_1 / 2, color="red", linestyle="dashed")
#     plt.axvline(mean_1 + fwhm_1 / 2, color="red", linestyle="dashed")
#     plt.axvline(barycenter_1, color="blue", linestyle="dashed", label="mean")  # type: ignore
#     plt.legend()
#     # plt.yscale('log')
#     plt.axis("tight")
#     plt.grid(alpha=0.01)
#     ax.tick_params(labelsize=f_s)
#     ax = plt.subplot(1, 3, 3)
#     plt.hist(data_2, bins=100, density=True)
#     plt.axvline(mean_2 - fwhm_2 / 2, color="red", linestyle="dashed")
#     plt.axvline(mean_2 + fwhm_2 / 2, color="red", linestyle="dashed")
#     plt.axvline(barycenter_2, color="blue", linestyle="dashed", label="mean")  # type: ignore
#     plt.xlabel("Norm Intensity[2]", fontsize=f_s)
#     # plt.yscale('log')
#     plt.axis("tight")
#     ax.tick_params(labelsize=f_s)
#     plt.grid(alpha=0.01)
#     plt.legend()
#     fig2.suptitle(r"Scan " + str(i_scan), fontsize=f_s)
#     plt.show()
#     return fig2


# def plot_Int(data, i_scan):
#     from scipy.optimize import curve_fit
#     import matplotlib.pyplot as plt
#     import numpy as np

#     data_sum_xy = (data.sum(axis=(0, 1))).astype(int)
#     z = np.arange(0, len(data_sum_xy))
#     data_sum_xz = (data.sum(axis=(0, 2))).astype(int)
#     y = np.arange(0, len(data_sum_xz))
#     data_sum_yz = (data.sum(axis=(2, 1))).astype(int)
#     x = np.arange(0, len(data_sum_yz))
#     f_s = 18
#     ax = plt.subplot(1, 1, 1)
#     plt.plot(z, data_sum_xy, "o", label="z dir")
#     plt.plot(y, data_sum_xz, "o", label="y dir")
#     plt.plot(x, data_sum_yz, "o", label="x dir")
#     plt.title(
#         "Sum of intensity in each direction for scan: " + str(i_scan),
#         fontsize=f_s,
#     )
#     plt.ylabel("Intensity", fontsize=f_s)
#     plt.xlabel("(x,y,z)", fontsize=f_s)
#     try:
#         popt_xy, pcov_xy = curve_fit(gaussian, z, data_sum_xy)
#         fwhm_xy = np.abs(2 * popt_xy[2] * np.sqrt(2 * np.log(2)))
#         fitted_data_sum_xy = gaussian(z, popt_xy[0], popt_xy[1], popt_xy[2])
#         mean_xy = popt_xy[1]
#         plt.plot(z, fitted_data_sum_xy, label="Fitted  z dir")
#         plt.axvline(mean_xy, color="r", linestyle="--", label="FWHM Bounds")
#     except Exception:
#         fwhm_xy, fitted_data_sum_xy, mean_xy = 0, 0, 0
#     try:
#         popt_xz, pcov_xz = curve_fit(gaussian, y, data_sum_xz)
#         fwhm_xz = np.abs(2 * popt_xz[2] * np.sqrt(2 * np.log(2)))
#         fitted_data_sum_xz = gaussian(y, popt_xz[0], popt_xz[1], popt_xz[2])
#         mean_xz = popt_xz[1]
#         plt.plot(y, fitted_data_sum_xz, label="Fitted  y dir")
#         plt.axvline(mean_xz, color="r", linestyle="--", label="FWHM Bounds")
#     except Exception:
#         fwhm_xz, fitted_data_sum_xz, mean_xz = 0, 0, 0
#     try:
#         popt_yz, pcov_yz = curve_fit(gaussian, x, data_sum_yz)
#         fwhm_yz = np.abs(2 * popt_yz[2] * np.sqrt(2 * np.log(2)))
#         fitted_data_sum_yz = gaussian(x, popt_yz[0], popt_yz[1], popt_yz[2])
#         mean_yz = popt_yz[1]
#         plt.plot(x, fitted_data_sum_yz, label="Fitted x dir")
#         plt.axvline(mean_yz, color="r", linestyle="--", label="FWHM Bounds")
#     except Exception:
#         fwhm_yz, fitted_data_sum_yz, mean_yz = 0, 0, 0

#     plt.legend(fontsize=f_s)
#     # plt.yscale('log')
#     plt.axis("tight")
#     ax.tick_params(labelsize=f_s)
#     plt.grid(alpha=0.01)
#     # plt.xlim(50,300)
#     fwhm_ = np.array([fwhm_yz, fwhm_xz, fwhm_xy])
#     return fwhm_


# def voigt(x, A, x0, sigma_g, gamma, alpha):
#     return (1 - alpha) * gaussian(x, A, x0, sigma_g) + alpha * lorentzian(
#         x, A, x0, gamma
#     )


# def exp_broadened_gaussian(x, A, x0, sigma, beta):
#     import numpy as np

#     return (
#         A
#         * np.exp(-beta * (x - x0))
#         * np.exp(-((x - x0) ** 2) / (2 * sigma**2))
#     )


# def doniach_sunjic(x, I0, gamma, beta, alpha):
#     import numpy as np

#     term1 = I0 * (1 + (x / gamma) ** 2) ** (-beta)
#     term2 = alpha * I0 * np.exp(-(x / gamma))
#     return term1 + term2


# Skewed Lorentzian profile
def skewed_lorentzian(x, A, x0, gamma, delta):
    import numpy as np

    term1 = (1 / (np.pi * gamma)) * (gamma**2 / ((x - x0 - delta) ** 2 + gamma**2))
    return A * term1


def pseudo_voigt_fwhm(sigma, gamma, eta):
    """Calculates the FWHM of a pseudo-Voigt distribution.

    Args:
      sigma: The standard deviation of the Gaussian distribution.
      eta: The mixing ratio of the Gaussian and Lorentzian distributions.
      gamma: The half-width at half-maximum (HWHM) of the Lorentzian distribution.

    Returns:
      The FWHM of the pseudo-Voigt distribution.
    """
    import numpy as np

    # Calculate the variance of the Gaussian distribution.
    sigma_variance = sigma**2

    # Calculate the weighted variance of the Lorentzian distribution.
    lorentzian_variance = 2 * eta**2 * gamma**2

    # Add the variance of the Gaussian distribution to the weighted variance of the Lorentzian distribution.
    total_variance = sigma_variance + lorentzian_variance

    # Take the square root of the sum of the variances.
    fwhm = np.sqrt(total_variance)
    fwhm *= 2.355
    # Multiply the square root by 2.355 to get the FWHM of the pseudo-Voigt distribution.
    return fwhm


def fwhm_calculation_geom_methode1(x_data, y_fit):
    import numpy as np
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(y_fit)
    half_max = np.max(y_fit) / 2
    left_peak = peaks[0]
    right_peak = peaks[-1]

    while y_fit[left_peak] > half_max:
        left_peak -= 1
    while y_fit[right_peak] > half_max:
        right_peak += 1

    fwhm = x_data[right_peak] - x_data[left_peak]
    return fwhm


# Fitting and evaluation
# def fit_best_profile(x_data, y_data, x_data_fit):
#     import numpy as np
#     from scipy.optimize import curve_fit

#     profiles = [
#         (gaussian, "Gaussian", [np.max(y_data), np.argmax(y_data), 5]),
#         (lorentzian, "Lorentzian", [np.max(y_data), np.argmax(y_data), 5]),
#         #  (voigt, "Voigt", [np.max(y_data), np.argmax(y_data), 10, 5, 0.5]),
#         (pearson_vii, "Pearson VII", [np.max(y_data), 6, 0.5, 3]),
#         (
#             pseudo_voigt,
#             "Pseudo-Voigt",
#             [np.max(y_data), np.argmax(y_data), 10, 5, 0.15],
#         ),
#     ]

#     best_r_squared = -1
#     best_profile = None
#     for profile, name, initial_guess in profiles:
#         np.random.seed(0)
#         popt, _ = curve_fit(
#             profile,
#             x_data,
#             y_data + 0.25 * np.random.normal(size=len(x_data)),
#             p0=initial_guess,
#             maxfev=4000000,
#         )
#         y_fit = profile(x_data_fit, *popt)
#         sse = np.sum((y_data - profile(x_data, *popt)) ** 2)
#         # mean = popt[1]  # Peak position as mean
#         # width = 2 * popt[2]  # FWHM as width
#         # half_max = popt[0] / 2
#         # Calculate FWHM based on the fit parameters
#         fwhm = find_fwhm_all(x_data_fit, y_fit)
#         r_squared = 1 - sse / np.sum((y_data - np.mean(y_data)) ** 2)
#         if r_squared > best_r_squared:
#             best_r_squared = r_squared

#             best_profile = (name, popt, r_squared, y_fit, np.abs(fwhm))

#     return best_profile


def pseudo_voigt_fwhm_Scherrer(lambda_, FWHM, theta, k=0.9):
    """Calculates the size of the crystallite in angstroms
    Args:

      FWHM (beta): is the full-width at half maximum of the X-ray diffraction peak in radians
      K: is a constant (typically 0.9)
      λ: is the wavelength of the X-rays in angstroms
      θ: is the Bragg angle in degrees

    Returns:
      D: is the size of the crystallite in angstroms
    """
    import numpy as np

    theta_rad = theta * np.pi / 180
    fact = (k * lambda_) / np.cos(theta_rad)
    return fact / (FWHM)


def theta_bragg_pt(lambda_, h, k, l_idx, a0=3.924):
    """
    Calculates the Bragg angle for a given wavelength.

    Args:
        lambda_: The wavelength of the X-rays in Angstroms.

    Returns:
        The Bragg angle in degrees.
    """
    import numpy as np

    # The Bragg angle for the (111) plane of platinum is 21.28 degrees at the
    # wavelength of X-rays used in most diffraction experiments.

    d = a0 / np.sqrt(
        h * h + k * k + l_idx * l_idx
    )  # Spacing between the (111) planes of platinum in Angstroms.

    return np.arcsin(lambda_ / (2 * d)) * 180 / np.pi


def fwhm_integral_1(x, y):
    """
    Compute the integral over the Full Width at Half Maximum (FWHM) range using raw data points.

    Parameters:
    x : numpy.ndarray
        The x-axis values.
    y : numpy.ndarray
        The corresponding y-axis values (intensity).

    Returns:
    float
        Integral of y over the FWHM range.
    """
    import numpy as np

    peak_max = np.max(y)
    if peak_max == 0:
        return 0.0

    half_max = peak_max / 2
    mask = y >= half_max  # Boolean mask of points above half-max

    if not np.any(mask):
        return 0.0

    # Get first and last indices where y >= half_max
    left_idx = np.argmax(mask)  # First True in mask
    right_idx = len(mask) - np.argmax(mask[::-1]) - 1  # Last True in mask

    # Extract the FWHM range
    x_fwhm = x[left_idx : right_idx + 1]
    y_fwhm = y[left_idx : right_idx + 1]

    return np.trapz(y_fwhm, x_fwhm)


# -------------------------------------------------------------------------------------------------------------
def get_plot_fwhm_and_skewness_kurtosis(
    data,
    plot_title="Sum of intensity along each direction",
    center_peak=False,
    save_fig=None,
    plot=True,
    eliminate_linear_background=False,
    log_distribution=False,
    plot_show=True,
    f_s=24,
    marker_size=60,
    linewidth=6,
    alpha_fit=0.5,
    f_s_table=20,
    scale_table=[4, 6],
    vmin=1e0,
    vmax=1e6,
    tight_layout=False,
    hspace_gridspec=0.05,
    wspace_gridspec=0.1,
    y_padding_factor=1.05,
    subplots=(4, 3),
    step_qxqyqz=None,
):
    """
    Analyze the intensity distribution in a 3D dataset along X, Y, and Z directions,
    computing statistical and shape metrics (FWHM, skewness, etc.) with visual output.

    Parameters:
    -----------
    data : numpy.ndarray
        3D array representing the intensity data (e.g., from Bragg Coherent Diffraction Imaging).

    plot_title : str, optional
        Title of the main plot. Default: "Sum of intensity in each direction".

    center_peak : bool, optional
        If True, centers the peak before fitting to improve robustness in small ROIs.

    save_fig : str or None, optional
        If provided, path to save the generated figure as an image.

    plot : bool, optional
        If True, generates plots of the summed profiles and fits.

    eliminate_linear_background : bool, optional
        If True, subtracts linear background before fitting along each direction.

    log_distribution : bool, optional
        If True, displays intensity profiles in log scale (y-axis).

    plot_show : bool, optional
        Whether to display the figure interactively with plt.show().

    f_s : int, optional
        Font size for all labels, titles, tick labels, and legend (default: 28).

    f_s_table : int, optional
        Font size for text inside the embedded result summary table.

    scale_table : list of float, optional
        Scaling factor [x, y] for the summary table’s size.

    vmin : float, optional
        Minimum intensity value (for color scale of 2D projections).

    vmax : float, optional
        Maximum intensity value (for color scale of 2D projections).

    tight_layout : bool, optional
        If True, uses figure.tight_layout() to optimize subplot spacing.

    hspace_gridspec : float, optional
        Vertical spacing between subplot rows (GridSpec hspace).

    wspace_gridspec : float, optional
        Horizontal spacing between subplot columns (GridSpec wspace).

    y_padding_factor : float, optional
        Factor to extend y-limit above maximum peak height (default: 1.35). Ensures headroom for table.

    Returns:
    --------
    fwhm_xyz : list of float
        Full-Width at Half Maximum (FWHM) for each axis (X, Y, Z).

    integral_fwhm_xyz : list of float
        Integral-based FWHM estimates for each axis.

    skewness_xyz : list of float
        Skewness of the summed intensity profiles for X, Y, Z.

    kurtosis_xyz : list of float
        Kurtosis of the summed intensity profiles for X, Y, Z.

    rsquared_xyz : list of float
        R² value of the best-fit profile along each direction.

    Notes:
    ------
    - Supports multiple fitting models (Gaussian, Lorentzian, Voigt, Pearson VII).
    - Auto-selects best fit by R² and overlays FWHM markers.
    - Embedded table summarizes all quantitative fit metrics.
    - Layout is fully managed with constrained_layout + manual spacing controls.

    Example:
    --------
    >>> get_plot_fwhm_and_skewness_kurtosis(data, center_peak=True, save_fig="output.png")
    """
    # === Required Imports ===
    try:
        from IPython.display import display  # type: ignore
    except Exception:  # pragma: no cover

        def display(x):  # type: ignore
            print(x)

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import gridspec
    from matplotlib import ticker as mticker
    from matplotlib.colors import LogNorm
    from matplotlib.table import Table
    from matplotlib.ticker import MaxNLocator
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy.stats import kurtosis, skew
    from tabulate import tabulate

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
        from matplotlib.colors import LinearSegmentedColormap

        my_cmap = LinearSegmentedColormap("my_colormap", cdict, 256)  # type: ignore
        return my_cmap

    my_cmap = MIR_Colormap()

    def fit_best_profile_with_noise(x_data, y_data, x_data_fit, noise_levels=None):
        """
        Fit the best peak profile (Gaussian, Lorentzian, Pearson VII, Pseudo-Voigt)
        to noisy data and compute both standard FWHM and Integral FWHM.

        Parameters:
        -----------
        x_data : numpy.ndarray
            The x-axis values (e.g., spatial coordinates, pixel positions).
        y_data : numpy.ndarray
            The corresponding y-axis intensity values.
        x_data_fit : numpy.ndarray
            The x-axis values for evaluating the fitted profile.
        noise_levels : list or numpy.ndarray, optional
            List of noise levels to iterate over (default: np.arange(0.01, 0.5, 0.01)).

        Returns:
        --------
        best_profile : tuple
            (profile_name, popt, r_squared, y_fit, fwhm, integral_fwhm)
            - profile_name : str : Name of the best fitting function.
            - popt : numpy.ndarray : Optimized parameters for the best fit.
            - r_squared : float : R-squared value for fit quality.
            - y_fit : numpy.ndarray : Best fitted data values.
            - fwhm : float : Full-Width at Half Maximum (geometric method).
            - integral_fwhm : float : Integral Full-Width at Half Maximum.
        """
        import numpy as np
        from scipy.optimize import curve_fit
        from sklearn.metrics import r2_score

        if noise_levels is None:
            noise_levels = np.arange(0.01, 0.5, 0.01)

        profiles = [
            (gaussian, "Gaussian", [np.max(y_data), np.argmax(y_data), 5]),
            (lorentzian, "Lorentzian", [np.max(y_data), np.argmax(y_data), 5]),
            (pearson_vii, "Pearson VII", [np.max(y_data), 6, 0.5, 3]),
            (
                pseudo_voigt,
                "Pseudo-Voigt",
                [np.max(y_data), np.argmax(y_data), 10, 5, 0.15],
            ),
        ]

        best_r_squared = -1
        best_profile = None

        # Loop over noise levels
        for noise_level in noise_levels:
            # Add Gaussian noise to data
            noisy_y_data = y_data + np.random.normal(
                scale=noise_level, size=len(y_data)
            )

            # Loop over peak profiles
            for profile_func, profile_name, initial_guess in profiles:
                try:
                    # Fit the profile to noisy data
                    popt, _ = curve_fit(
                        profile_func,
                        x_data,
                        noisy_y_data,
                        p0=initial_guess,
                        maxfev=5000,
                    )

                    # Evaluate fit quality
                    y_fit = profile_func(x_data_fit, *popt)
                    y_fit_ = profile_func(x_data, *popt)
                    r_squared = r2_score(noisy_y_data, y_fit_)

                    # Compute Standard FWHM using geometric method or model fitting
                    fwhm = find_fwhm_all(x_data_fit, y_fit)

                    # Compute Integral FWHM
                    integral_fwhm_value = integral_fwhm(x_data_fit, y_fit)

                    # Update best fit if current fit has higher R²
                    if r_squared > best_r_squared:
                        best_r_squared = r_squared
                        best_profile = (
                            profile_name,
                            popt,
                            r_squared,
                            y_fit,
                            fwhm,
                            integral_fwhm_value,
                        )

                except Exception:
                    continue

        return best_profile

    def get_figure_size(
        width: int | str = "default",
        scale: float = 1,
        subplots: tuple = (1, 1),
    ) -> tuple:
        """
        Get the figure dimensions to avoid scaling in LaTex.

        This function was taken from
        https://jwalton.info/Embed-Publication-Matplotlib-Latex/

        :param width: Document width in points, or string of predefined
        document type (float or string)
        :param fraction: fraction of the width which you wish the figure to
        occupy (float)
        :param subplots: the number of rows and columns of subplots

        :return: dimensions of the figure in inches (tuple)
        """

        if width == "default":
            width_pt = 420
        elif width == "thesis":
            width_pt = 455.30101
        elif width == "beamer":
            width_pt = 398.3386
        elif width == "nature":
            width_pt = 518.74
        else:
            width_pt = width

        # Width of figure (in pts)
        fig_width_pt = width_pt * scale  # type: ignore

        # Golden ratio to set aesthetic figure height
        # https://disq.us/p/2940ij3
        golden_ratio = (5**0.5 - 1) / 2

        # Convert from pt to inches
        inches_per_pt = 1 / 72.27

        # Figure width in inches
        fig_width_in = fig_width_pt * inches_per_pt
        # Figure height in inches
        fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

        return (fig_width_in, fig_height_in)

    # -------------------------------------------------------------------------------------------------------------
    def integral_fwhm(x, y, method="trapz"):
        """
        Compute the Integral Full-Width at Half Maximum (Integral FWHM).

        Parameters
        ----------
        x : numpy.ndarray
            The x-axis values (e.g., pixel positions or spatial coordinates).
        y : numpy.ndarray
            The corresponding y-axis values (intensity or function values).
        method : str, optional
            Integration method: 'trapz' (default) or 'simps'.

        Returns
        -------
        float
            The Integral FWHM value.
        """
        import numpy as np

        if method == "trapz":
            integral_intensity = np.trapz(y, x)  # Trapezoidal integration
        elif method == "simps":
            from scipy.integrate import simpson

            integral_intensity = simpson(y, x)  # Simpson's rule
        else:
            raise ValueError("method must be 'trapz' or 'simps'")

        peak_max = np.max(y)
        return integral_intensity / peak_max if peak_max != 0 else 0.0

    # -------------------------------------------------------------------------------------------------------------
    def find_fwhm_all(x_data_fit, y_fit):
        """
        Compute the Full-Width at Half Maximum (FWHM) of a given peak function
        using a geometric method.

        Parameters:
        -----------
        x_data_fit : numpy.ndarray
            The x-axis values of the fitted data (e.g., pixel or spatial coordinates).
        y_fit : numpy.ndarray
            The corresponding y-axis values (intensity or function values).

        Returns:
        --------
        fwhm : float
            The computed Full-Width at Half Maximum (FWHM). Returns 0 if computation fails.

        Notes:
        ------
        - This method finds the left and right points where the signal crosses half of its maximum value.
        - It uses `find_peaks` and thresholding to estimate the FWHM geometrically.
        """
        try:
            fwhm = fwhm_calculation_geom_methode2(x_data_fit, y_fit)
        except Exception as e:
            print(f"Warning: FWHM calculation failed with error: {e}")
            fwhm = 0
        return fwhm

    # Gaussian profile
    def gaussian(x, A, x0, sigma):
        import numpy as np

        return A * np.exp(-((x - x0) ** 2) / (2 * sigma**2))

    # Lorentzian profile
    def lorentzian(x, A, x0, gamma):
        return A / (1 + ((x - x0) / gamma) ** 2)

    # Pseudo-Voigt profile
    def pseudo_voigt(x, A, x0, sigma, gamma, alpha):
        """Returns the pseudo-Voigt distribution function for the given parameters.

        Args:
            x: A NumPy array of values to evaluate the function at.
            eta: The mixing ratio of the Gaussian and Lorentzian distributions.
            sigma: The standard deviation of the Gaussian distribution.
            gamma: The half-width at half-maximum (HWHM) of the Lorentzian distribution.

        Returns:
           A NumPy array of the values of the pseudo-Voigt distribution function at the given values of x.
        """
        import numpy as np

        gaussian = np.exp(-((x - x0) ** 2) / (2 * sigma**2))
        lorentzian = 1 / (1 + ((x - x0) / gamma) ** 2)
        return (1 - alpha) * gaussian + A * alpha * lorentzian

    # Pearson VII profile
    def pearson_vii(x, A, x0, gamma, m):
        import numpy as np

        gamma_safe = np.maximum(gamma, 1e-10)
        x_diff = (x - x0) / gamma_safe
        x_diff_sq = x_diff**2

        threshold = 1e6  # define a threshold for "large"

        # Masks
        mask_safe = x_diff_sq < threshold
        mask_large = ~mask_safe

        denom = np.empty_like(x_diff_sq)

        # Safe branch: moderate x_diff
        denom[mask_safe] = np.exp(m * np.log1p(x_diff_sq[mask_safe]))

        # Approximate branch: huge x_diff
        # We must be careful: extremely large x_diff_sq can still cause overflows in x_diff_sq**m
        with np.errstate(over="ignore"):  # Silence overflow temporarily
            denom[mask_large] = x_diff_sq[mask_large] ** m

        # Final protection against infinities/zero
        denom = np.maximum(denom, 1e-300)
        denom = np.where(np.isfinite(denom), denom, np.finfo(float).max)

        return A / denom

    def fwhm_calculation_geom_methode2(x_data, y_fit):
        import numpy as np
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(y_fit)
        half_max = np.max(y_fit) / 2
        indices_higher_than_hm = np.where(y_fit >= half_max)[0]
        left_peak = indices_higher_than_hm[0]
        right_peak = indices_higher_than_hm[-1]

        fwhm = x_data[right_peak] - x_data[left_peak]
        return fwhm

    def plot_3D_projections(
        data,
        mask=None,
        alpha_mask=0.3,
        ax=None,
        fig=None,
        fw=4,
        fig_title=None,
        axes_labels=True,
        colorbar=False,
        log_scale=True,
        log_threshold=False,
        max_projection=False,
        vmin=None,
        vmax=None,
        fontsize=15,
        cmap=None,
        tight_layout=True,
        step_qxqyqz=None,
    ):
        def format_unit_label(axis_label, exponent):
            if exponent is None or exponent == 0:
                return rf"{axis_label} (Å$^{{-1}}$)"
            return rf"{axis_label} (10$^{{{exponent}}}$×Å$^{{-1}}$)"

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
                cax.tick_params(labelsize=tick_fontsize)
                cbar_list.append(cbar)
            return cbar_list if return_cbar else None

        def apply_sci_format_imshow(ax, axis="both", fontsize=10):
            exponents = {"x": None, "y": None}

            if axis in ("x", "both"):
                ax.ticklabel_format(style="sci", axis="x", scilimits=(-1, 1))
                ax.xaxis.get_offset_text().set_fontsize(fontsize)
                ax.xaxis.get_offset_text().set_weight("normal")
                ax.figure.canvas.draw()
                offset_x = ax.xaxis.get_offset_text().get_text().replace("−", "-")
                if "e" in offset_x:
                    exponent = int(offset_x.split("e")[-1])
                    exponents["x"] = exponent  # type: ignore
                    # REMOVE this line:
                    # ax.ticklabel_format(style='plain', axis='x')
                    ax.xaxis.get_offset_text().set_visible(False)

            if axis in ("y", "both"):
                ax.ticklabel_format(style="sci", axis="y", scilimits=(-1, 1))
                ax.yaxis.get_offset_text().set_fontsize(fontsize)
                ax.yaxis.get_offset_text().set_weight("normal")
                ax.figure.canvas.draw()
                offset_y = ax.yaxis.get_offset_text().get_text().replace("−", "-")
                if "e" in offset_y:
                    exponent = int(offset_y.split("e")[-1])
                    exponents["y"] = exponent  # type: ignore
                    # REMOVE this line:
                    # ax.ticklabel_format(style='plain', axis='y')
                    ax.yaxis.get_offset_text().set_visible(False)

            return exponents

        if cmap is None:
            cmap = my_cmap

        if fig is None:
            fig, ax = plt.subplots(
                1, 3, figsize=(3.5 * fw, fw) if colorbar else (3 * fw, fw)
            )

        plots = []
        axis_titles = (
            [("$Q_y$", "$Q_z$"), ("$Q_x$", "$Q_z$"), ("$Q_x$", "$Q_y$")]
            if step_qxqyqz
            else [
                ("detector horizontal", "detector vertical"),
                ("detector horizontal", "rocking curve"),
                ("detector vertical", "rocking curve"),
            ]
        )

        for n in range(3):
            img = np.nanmax(data, axis=n) if max_projection else np.nansum(data, axis=n)

            if step_qxqyqz is not None:
                shape = img.shape
                if n == 0:
                    extent = [
                        (-shape[1] / 2) * step_qxqyqz[1],
                        (shape[1] / 2) * step_qxqyqz[1],
                        (-shape[0] / 2) * step_qxqyqz[2],
                        (shape[0] / 2) * step_qxqyqz[2],
                    ]
                elif n == 1:
                    extent = [
                        (-shape[1] / 2) * step_qxqyqz[0],
                        (shape[1] / 2) * step_qxqyqz[0],
                        (-shape[0] / 2) * step_qxqyqz[2],
                        (shape[0] / 2) * step_qxqyqz[2],
                    ]
                elif n == 2:
                    extent = [
                        (-shape[1] / 2) * step_qxqyqz[0],
                        (shape[1] / 2) * step_qxqyqz[0],
                        (-shape[0] / 2) * step_qxqyqz[1],
                        (shape[0] / 2) * step_qxqyqz[1],
                    ]
            else:
                extent = None

            if log_scale:
                if log_threshold:
                    img = np.log10(np.clip(img, 1e-5, None))
                    plots.append(
                        ax[n].imshow(
                            img,
                            cmap=cmap,
                            aspect="auto",
                            vmin=vmin,
                            vmax=vmax,
                            extent=extent,
                            origin="lower",
                        )
                    )  # type: ignore
                else:
                    norm = LogNorm(vmin=vmin, vmax=vmax)
                    plots.append(
                        ax[n].imshow(
                            img,
                            cmap=cmap,
                            aspect="auto",
                            norm=norm,
                            extent=extent,
                            origin="lower",
                        )
                    )  # type: ignore
            else:
                plots.append(
                    ax[n].imshow(
                        img,
                        cmap=cmap,
                        aspect="auto",
                        vmin=vmin,
                        vmax=vmax,
                        extent=extent,
                        origin="lower",
                    )
                )  # type: ignore

            if mask is not None:
                mask_plot = np.nanmean(mask, axis=n)
                mask_plot[mask_plot != 0.0] = 1.0
                overlay = np.dstack(
                    [
                        mask_plot,
                        np.zeros_like(mask_plot),
                        np.zeros_like(mask_plot),
                        alpha_mask * mask_plot,
                    ]
                )
                ax[n].imshow(
                    overlay, aspect="auto", extent=extent, origin="lower"
                )  # type: ignore

            ax[n].tick_params(axis="both", labelsize=fontsize)  # type: ignore
            ax[n].xaxis.set_major_locator(MaxNLocator(nbins=3))  # type: ignore
            ax[n].yaxis.set_major_locator(MaxNLocator(nbins=3))  # type: ignore

            if step_qxqyqz is not None:
                exp_dict = apply_sci_format_imshow(
                    ax[n], axis="both", fontsize=fontsize
                )  # type: ignore
                x_exp = exp_dict["x"]
                y_exp = exp_dict["y"]

            if axes_labels and step_qxqyqz is not None:
                xlabel = format_unit_label(axis_titles[n][0], x_exp)  # type: ignore
                ylabel = format_unit_label(axis_titles[n][1], y_exp)  # type: ignore
                ax[n].set_xlabel(
                    xlabel, fontsize=fontsize * fw / 4, fontweight="bold"
                )  # type: ignore
                ax[n].set_ylabel(
                    ylabel, fontsize=fontsize * fw / 4, fontweight="bold"
                )  # type: ignore

        if colorbar:
            add_colorbar_subplot(fig, ax, plots, tick_fontsize=fontsize)

        if fig_title is not None:
            fig.suptitle(fig_title, fontsize=fontsize * fw / 4)

        if tight_layout:
            fig.tight_layout()

        return fig

    def apply_sci_format(ax, fontsize=10):
        # Y-axis
        y_formatter = mticker.ScalarFormatter(useMathText=True)
        y_formatter.set_scientific(True)
        y_formatter.set_powerlimits((-2, 1))
        y_formatter.set_useOffset(False)
        ax.yaxis.set_major_formatter(y_formatter)

        # X-axis
        x_formatter = mticker.ScalarFormatter(useMathText=True)
        x_formatter.set_scientific(True)
        x_formatter.set_powerlimits((0, 0))  # Always sci-notation
        x_formatter.set_useOffset(False)
        ax.xaxis.set_major_formatter(x_formatter)

        # Styling offset text (if any)
        ax.xaxis.get_offset_text().set_fontsize(fontsize)
        ax.yaxis.get_offset_text().set_fontsize(fontsize)
        ax.xaxis.get_offset_text().set_weight("normal")
        ax.yaxis.get_offset_text().set_weight("normal")

    ##########################################################################################################################
    # === Main Function ===
    with plt.rc_context():
        plt.style.use("grayscale")
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        plt.rcParams["axes.titleweight"] = "bold"
        plt.rcParams.update(
            {
                "lines.linewidth": linewidth,  # Global line width
                "lines.markersize": marker_size,  # Global marker size
                "axes.labelsize": f_s,  # Label font size
                "axes.titlesize": f_s,  # Title font size
                "xtick.labelsize": f_s,  # Tick label size
                "ytick.labelsize": f_s,
                "legend.fontsize": f_s,
                "font.weight": "bold",
                "axes.labelweight": "bold",
                "axes.titleweight": "bold",
                "figure.dpi": 150,  # Save quality
            }
        )
        step_x_fit = 0.25
        unit_str = r"$(\mathrm{\AA}^{-1})$" if step_qxqyqz else "$_{(pixels)}$"
        color_list = ("#1f77b4", "#2ca02c", "#ff7f0e")
        directions = ["X", "Y", "Z"]
        background_degree = 1  # Adjust the degree of the polynomial as needed
        first_and_last_pixel = [0, -1]
        if plot:
            figsize = get_figure_size(scale=3, subplots=subplots)
            figure = plt.figure(figsize=figsize, dpi=150)
            figure.set_constrained_layout(True)  # type: ignore # Preferred for automatic handling
            # Define the grid layout with one row for the large subplot and three subplots in the second row
            gs = gridspec.GridSpec(
                nrows=4,
                ncols=3,
                wspace=wspace_gridspec,
                hspace=hspace_gridspec,
                figure=figure,
            )
            ax_table = figure.add_subplot(gs[0, :])
            ax_table.axis("off")  # This axis will hold only the table
            ax = figure.add_subplot(gs[1:3, :])  # Main plot
            ax1 = figure.add_subplot(gs[3, 0])
            ax2 = figure.add_subplot(gs[3, 1])
            ax3 = figure.add_subplot(gs[3, 2])

            plot_3D_projections(
                data,
                ax=[ax1, ax2, ax3],
                fig=figure,
                log_scale=True,
                cmap=my_cmap,
                vmin=vmin,
                vmax=vmax,
                colorbar=True,
                tight_layout=False,
                step_qxqyqz=step_qxqyqz,
                fontsize=f_s,
            )
            ax1.axis("tight")
            ax2.axis("tight")
            ax3.axis("tight")

            unit_label = r"$(\mathrm{\AA}^{-1})$" if step_qxqyqz else "$_{(pixels)}$"

            ax.set_title(plot_title, pad=20)
            ax.set_ylabel(
                "Integrated Intensity $_{(a.u.)}$",
                fontsize=f_s,
                fontweight="bold",
            )
            ax.set_xlabel(
                f"($Q_x,Q_y,Q_z$) {unit_label}",
                fontsize=f_s,
                fontweight="bold",
            )
            apply_sci_format(ax, fontsize=f_s)
        # figure.tight_layout()
        axes = [(1, 2), (0, 2), (0, 1)]
        data_sum_all = [data.sum(axis=axis) for axis in axes]  # Vectorized summation
        X_data_all = [np.arange(len(d)) for d in data_sum_all]  # Avoid recomputation
        X_fit_all = [
            np.linspace(0, len(d), int(len(d) / step_x_fit)) for d in data_sum_all
        ]  # More robust
        peak_positions = np.array([np.argmax(d) for d in data_sum_all], dtype=int)
        left_room = int(peak_positions.min())
        right_room = int(
            min(len(d) - int(p) for d, p in zip(data_sum_all, peak_positions))
        )
        h_len = max(min(left_room, right_room) - 1, 0)
        if center_peak:
            data_sum_all = [
                d[np.maximum(0, peak - h_len) : peak + h_len]
                for d, peak in zip(data_sum_all, peak_positions)
            ]
            X_data_all = [np.arange(len(d)) for d in data_sum_all]
            X_fit_all = [
                np.linspace(0, len(d), int(len(d) / step_x_fit)) for d in data_sum_all
            ]
        if eliminate_linear_background:
            background_coefficients_all = [
                np.polyfit(
                    X_data_all[d][first_and_last_pixel],
                    data_sum_all[d][first_and_last_pixel],
                    background_degree,
                )
                for d in range(len(data_sum_all))
            ]
            background_fit_all = [
                np.polyval(background_coefficients_all[d], X_data_all[d])
                for d in range(len(data_sum_all))
            ]
            data_sum_all = [
                data_sum_all[d] - background_fit_all[d]
                for d in range(len(data_sum_all))
            ]
        skewness_xyz = np.array([skew(d) for d in data_sum_all])
        kurtosis_xyz = np.array([kurtosis(d) for d in data_sum_all])
        fits = [
            fit_best_profile_with_noise(X_data_all[i], data_sum_all[i], X_fit_all[i])
            for i in range(len(data_sum_all))
        ]
        (
            names,
            popt_xyz,
            rsquared_xyz,
            fitted_data,
            fwhm_xyz,
            fwhm_integral_xyz,
        ) = zip(*fits)
        # Apply unit scaling if step_qxqyqz is provided
        if step_qxqyqz is not None:
            for i in range(3):
                X_data_all[i] = X_data_all[i].astype(float) * step_qxqyqz[i]
                X_fit_all[i] = X_fit_all[i].astype(float) * step_qxqyqz[i]
            fwhm_xyz = [f * step for f, step in zip(fwhm_xyz, step_qxqyqz)]
            fwhm_integral_xyz = [
                f * step for f, step in zip(fwhm_integral_xyz, step_qxqyqz)
            ]
        if plot and log_distribution:
            ax.set_yscale("log")  # type: ignore
        for i in range(len(data_sum_all)):
            if i == 0:
                _, direction = (1, 2), "X"
            if i == 1:
                _, direction = (0, 2), "Y"
            if i == 2:
                _, direction = (0, 1), "Z"
            popt = popt_xyz[i]
            try:
                print(f"fit along {direction} results :")  # type: ignore
                display(
                    f"A {popt[0]} {direction}0 {popt[1]} sigma {popt[2]} gamma  {popt[3]}  eta {popt[4]} "
                )  # type: ignore
            except Exception:
                print("Not pseudo-voigt")
            if plot:
                fit_safe = np.clip(fitted_data[i], 1e-12, None)
                scatter_safe = np.clip(data_sum_all[i], 1e-12, None)
                ax.scatter(
                    X_data_all[i],
                    scatter_safe,
                    s=marker_size,
                    color=color_list[i],
                )  # type: ignore
                ax.plot(
                    X_fit_all[i],
                    fit_safe,
                    color=color_list[i],
                    alpha=alpha_fit,
                    label=f"Fit {direction} {names[i]}",
                )  # type: ignore
        if plot:
            data_sum_fit_max = np.array(
                [np.max(fit) for fit in fitted_data]
            )  # Get max values for each fitted dataset
            data_sum_fit_max_max = (
                data_sum_fit_max.max()
            )  # Overall max value for normalization
            # ax.set_ylim(0,2 * data_sum_fit_max_max)  # 20% headroom above the tallest peak
            for i, (fit, color, popt) in enumerate(
                zip(fitted_data, color_list, popt_xyz)
            ):
                try:
                    half_max_norm = 0.51 * data_sum_fit_max[i] / data_sum_fit_max_max
                except ZeroDivisionError:
                    half_max_norm = 0.5  # Fallback to mid-range if fit is flat
                step = step_qxqyqz[i] if step_qxqyqz is not None else 1.0

                center_scaled = (
                    popt_xyz[i][1] * step
                )  # popt[1] is in pixels → convert if needed
                fwhm_scaled = fwhm_xyz[i]  # already scaled if you applied step_qxqyqz

                fwhm_line_m = center_scaled - fwhm_scaled / 2
                fwhm_line_p = center_scaled + fwhm_scaled / 2

                ax.axvline(
                    x=fwhm_line_m,
                    color=color,
                    linestyle="--",
                    ymin=0,
                    ymax=half_max_norm,
                )  # type: ignore
                ax.axvline(
                    x=fwhm_line_p,
                    color=color,
                    linestyle="--",
                    ymin=0,
                    ymax=half_max_norm,
                )  # type: ignore

            ax.tick_params(labelsize=f_s)  # type: ignore
            ax.grid(alpha=0.01)  # type: ignore
            ax.legend(loc="best")  # type: ignore
        if plot:
            # Stack FWHM and Integral FWHM values
            fwhm_xyz = np.array(fwhm_xyz)
            fwhm_integral_xyz = np.array(fwhm_integral_xyz)

            fwhm_values = np.concatenate([fwhm_xyz, fwhm_integral_xyz])
            exponents = np.floor(
                np.log10(np.abs(fwhm_values[np.nonzero(fwhm_values)]))
            ).astype(int)

            # Choose exponent closest to zero
            chosen_exp = exponents[np.abs(exponents).argmin()]
            scale_factor = 10 ** (-chosen_exp)

            # Format values scaled by this exponent
            scaled_fwhm = fwhm_xyz * scale_factor
            scaled_fwhm_integral = fwhm_integral_xyz * scale_factor

            # Replace original values in table row construction
            rounded_values = np.column_stack(
                (
                    np.round(scaled_fwhm, 2),
                    np.round(scaled_fwhm_integral, 2),
                    np.round(skewness_xyz, 1),
                    np.round(kurtosis_xyz, 1),
                    np.round(rsquared_xyz, 1),
                )
            )

            # Now format unit string with common exponent
            unit_str = (
                f"$\\times 10^{{{chosen_exp}}}\\;\\mathrm{{\\AA}}^{{-1}}$"
                if step_qxqyqz
                else "$_{(pixels)}$"
            )
            # Determine units
            directions = ["X", "Y", "Z"]
            # Build table data for plot
            table_data = [
                [
                    "Direction",
                    f"FWHM\n{unit_str}",
                    f"Integral FWHM\n{unit_str}",
                    "Skewness",
                    "Kurtosis",
                    "R-squared",
                ]
            ] + [[directions[i]] + list(map(str, rounded_values[i])) for i in range(3)]

            # Create matplotlib table on ax_table
            table = Table(ax_table, loc="upper left")  # type: ignore
            table.auto_set_font_size(False)
            table.set_fontsize(f_s_table)
            table.scale(scale_table[0], scale_table[1])

            for i, row in enumerate(table_data):
                for j, cell in enumerate(row):
                    table.add_cell(
                        i,
                        j,
                        width=1 / len(row),
                        height=0.3,
                        text=str(cell),
                        facecolor="darkblue",
                        loc="center",
                        edgecolor="darkblue",
                    )
                    text = table[i, j].get_text()
                    text.set_fontsize(f_s_table)
                    text.set_color("white")
                    text.set_weight("bold")  # type: ignore # type: ignore

            # Add table to axis
            ax_table.add_table(table)  # type: ignore # type: ignore
            ax_table.axis("off")  # type: ignore # type: ignore
        # Print summary table in console
        console_table = [
            ["Metric", "X", "Y", "Z"],
            [f"FWHM {unit_str}"] + [f"{x:.4f}" for x in fwhm_xyz],  # type: ignore
            [f"Integral FWHM {unit_str}"]
            + [f"{x:.4f}" for x in fwhm_integral_xyz],  # type: ignore
            ["Skewness"] + [f"{x:.4f}" for x in skewness_xyz],
            ["Kurtosis"] + [f"{x:.4f}" for x in kurtosis_xyz],
            ["R-squared"] + [f"{x:.4f}" for x in rsquared_xyz],
        ]
        print(tabulate(console_table, headers="firstrow", tablefmt="grid"))
        if plot:
            for ax_obj in figure.get_axes():  # type: ignore
                ax_obj.tick_params(labelsize=f_s)  # only label size allowed
                for label in ax_obj.get_xticklabels() + ax_obj.get_yticklabels():
                    label.set_fontweight("bold")
                ax_obj.title.set_fontsize(f_s)
                ax_obj.xaxis.label.set_size(f_s)  # type: ignore
                ax_obj.yaxis.label.set_size(f_s)  # type: ignore
                legend = ax_obj.get_legend()
                if legend:
                    for text in legend.get_texts():
                        text.set_weight("bold")  # type: ignore
                        text.set_fontsize(f_s)
        if plot:
            y_max = np.max([np.max(fit) for fit in fitted_data])
            x_max = np.max([np.max(fit) for fit in X_data_all])
            ax.set_ylim(0, y_padding_factor * y_max)  # type: ignore # Set bottom to 0, top with headroom
            ax.set_xlim(0, y_padding_factor * x_max)  # type: ignore # Set bottom to 0, top with headroom
        if tight_layout:
            figure.tight_layout()  # type: ignore
        if save_fig:
            figure.savefig(save_fig)  # type: ignore
            # plt.savefig(save_fig)
            print(f"saving figure to : {save_fig}")
        if plot:
            if plot_show:
                plt.show()
            else:
                plt.close()

    return fwhm_xyz, fwhm_integral_xyz, skewness_xyz, kurtosis_xyz
