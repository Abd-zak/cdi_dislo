#####################################################################################################################
# This script provides various functions for analyzing intensity distributions in 3D data sets. 
# It includes methods for:
# - Plotting 1D intensity distributions along different axes and computing key statistics (FWHM, barycenter, etc.).
# - Fitting different peak profiles (Gaussian, Lorentzian, Voigt, Pseudo-Voigt, etc.) to the intensity data.
# - Calculating full-width at half maximum (FWHM) using different geometric and fitting methods.
# - Evaluating skewness, kurtosis, and peak fitting quality using R-squared values.
# - Displaying intensity sum plots with fitted profiles, mean values, and FWHM indicators.
# - Providing advanced functions for noise handling, Scherrer equation calculations, and Bragg angle determination.
#
# The script is intended for use in Bragg Coherent Diffraction Imaging (BCDI) and related 3D diffraction experiments.
#####################################################################################################################
#####################################################################################################################
# Suggestions for Improvements:
# 1. **Code Optimization**:
#    - Some sections of the script repeat similar operations (e.g., sum along different axes). Consider modularizing them.
#
# 2. **Exception Handling**:
#    - The script uses multiple `try-except` blocks when fitting models. It would be beneficial to log or display errors 
#      to understand fitting failures.
#
# 3. **Performance Enhancements**:
#    - Use `numba` or `cython` to speed up computation-heavy functions, such as peak fitting and FWHM calculations.
#
# 4. **Visualization Enhancements**:
#    - Improve the readability of plots by adding gridlines, annotations, and dynamic axis scaling.
#    - Use `matplotlib` subplots for better organization of multiple plots.
#
# 5. **Configurable Parameters**:
#    - Consider allowing users to specify noise levels, fitting methods, and peak detection thresholds via function arguments.
#
# 6. **Generalization**:
#    - Support arbitrary dimensionality for the input data, making the script adaptable for different experimental setups.
#
# 7. **Save and Export Options**:
#    - Add functionality to save fitted results (e.g., JSON, CSV) for further analysis.
#    - Enable saving of figures in high resolution for publication purposes.
#
# 8. **Documentation and Readability**:
#    - Improve docstrings with detailed parameter descriptions, expected input types, and return values.
#    - Consider using `logging` instead of `print()` statements for better debugging.
#####################################################################################################################



from cdi_dislo.common_imports import *
from cdi_dislo.ewen_utilities.plot_utilities                      import plot_3D_projections ,plot_2D_slices_middle_one_array3D
from tabulate import tabulate

#####################################################################################################################
#####################################################################################################################


def plot_single_data_distr(data,i_scan ):
    data_0=xu.maplog(data.sum(axis=0).flatten())
   # data_0=data_0[data_0!=0]
    mean_0 = np.mean(data_0)#norm.mean(dddata)
    std_0 = np.std(data_0)#norm.std(dddata)
    fwhm_0 = 2 * std_0 * np.sqrt(2 * np.log(2))
    intensity_at_fwhm_0 = norm.pdf(mean_0 - fwhm_0 / 2)
    barycenter_0 = np.mean(data_0)
    intensity_at_barycenter_0 = norm.pdf(barycenter_0)

    data_1=xu.maplog(data.sum(axis=1).flatten())
    #data_1=data_1[data_1!=0]
    mean_1 = np.mean(data_1)#norm.mean(dddata)
    std_1 = np.std(data_1)#norm.std(dddata)
    fwhm_1 = 2 * std_1 * np.sqrt(2 * np.log(2))
    intensity_at_fwhm_1 = norm.pdf(mean_1 - fwhm_1 / 2)
    barycenter_1 = np.mean(data_1)
    intensity_at_barycenter_1 = norm.pdf(barycenter_1)

    data_2=xu.maplog(data.sum(axis=2).flatten())
    #data_2=data_2[data_2!=0]
    mean_2 = np.mean(data_2)#norm.mean(dddata)
    std_2 = np.std(data_2)#norm.std(dddata)
    fwhm_2 = 2 * std_2 * np.sqrt(2 * np.log(2))
    intensity_at_fwhm_2 = norm.pdf(mean_2 - fwhm_2 / 2)
    barycenter_2 = np.mean(data_2)
    intensity_at_barycenter_2 = norm.pdf(barycenter_2)
      
    f_s=16
    fig2=plt.figure(1,figsize=(20,4))
    ax=plt.subplot(1,3,1)
    im=plt.hist(data_0 , bins=100,density=True);
    plt.xlabel('Norm Intensity [0]', fontsize=f_s)
    plt.ylabel('', fontsize=f_s)
    plt.axvline(mean_0 - fwhm_0 / 2, color='red', linestyle='dashed')
    plt.axvline(mean_0 + fwhm_0/ 2, color='red', linestyle='dashed')
    plt.axvline(barycenter_0, color='blue', linestyle='dashed',label='mean')
    plt.legend()
    #plt.yscale('log')
    plt.axis('tight')
    ax.tick_params(labelsize=f_s)
    plt.grid(alpha=0.01)
    
    ax=plt.subplot(1,3,2)
    im=plt.hist(data_1, bins=100,density=True);
    plt.xlabel('Norm Intensity [1]', fontsize=f_s)
    plt.axvline(mean_1 - fwhm_1 / 2, color='red', linestyle='dashed')
    plt.axvline(mean_1 + fwhm_1/ 2, color='red', linestyle='dashed')
    plt.axvline(barycenter_1, color='blue', linestyle='dashed',label='mean')
    plt.legend()
    #plt.yscale('log')
    plt.axis('tight')
    plt.grid(alpha=0.01)
    ax.tick_params(labelsize=f_s)
    ax=plt.subplot(1,3,3)
    im=plt.hist(data_2, bins=100,density=True);
    plt.axvline(mean_2 - fwhm_2 / 2, color='red', linestyle='dashed')
    plt.axvline(mean_2 + fwhm_2/ 2, color='red', linestyle='dashed')
    plt.axvline(barycenter_2, color='blue', linestyle='dashed',label='mean')
    plt.xlabel('Norm Intensity[2]', fontsize=f_s)
    #plt.yscale('log')
    plt.axis('tight')
    ax.tick_params(labelsize=f_s)
    plt.grid(alpha=0.01)
    plt.legend()
    fig2.suptitle(r"Scan "+str(i_scan), fontsize=f_s)
    plt.show()
    return fig2
def plot_Int(data,i_scan ):
    data_sum_xy=(data.sum(axis=(0, 1)) ).astype(int) 
    z=np.arange(0,len(data_sum_xy))
    data_sum_xz=(data.sum(axis=(0, 2)) ).astype(int) 
    y=np.arange(0,len(data_sum_xz))
    data_sum_yz=(data.sum(axis=(2, 1)) ).astype(int) 
    x=np.arange(0,len(data_sum_yz))
    f_s=18
    ax=plt.subplot(1,1,1)
    plt.plot(z,data_sum_xy,"o",label="z dir")
    plt.plot(y,data_sum_xz,"o",label="y dir")
    plt.plot(x,data_sum_yz,"o",label="x dir")
    plt.title("Sum of intensity in each direction for scan: " + str(i_scan), fontsize=f_s)
    plt.ylabel("Intensity", fontsize=f_s)
    plt.xlabel("(x,y,z)", fontsize=f_s)
    try: 
        popt_xy, pcov_xy = curve_fit(gaussian, z, data_sum_xy)
        fwhm_xy = np.abs(2 * popt_xy[2] * np.sqrt(2 * np.log(2))    )
        fitted_data_sum_xy=gaussian(z, popt_xy[0], popt_xy[1],popt_xy[2])
        mean_xy=popt_xy[1]
        plt.plot(z, fitted_data_sum_xy, label='Fitted  z dir')
        plt.axvline(mean_xy, color='r', linestyle='--', label='FWHM Bounds')
    except:
        fwhm_xy,fitted_data_sum_xy,mean_xy=0,0,0
    try: 
        popt_xz, pcov_xz = curve_fit(gaussian, y, data_sum_xz)
        fwhm_xz = np.abs(2 * popt_xz[2] * np.sqrt(2 * np.log(2))    )
        fitted_data_sum_xz=gaussian(y, popt_xz[0], popt_xz[1],popt_xz[2])
        mean_xz=popt_xz[1]
        plt.plot(y, fitted_data_sum_xz, label='Fitted  y dir')
        plt.axvline(mean_xz, color='r', linestyle='--', label='FWHM Bounds')
    except:
        fwhm_xz,fitted_data_sum_xz,mean_xz=0,0,0
    try:         
        popt_yz, pcov_yz = curve_fit(gaussian, x, data_sum_yz)
        fwhm_yz = np.abs(2 * popt_yz[2] * np.sqrt(2 * np.log(2)) )
        fitted_data_sum_yz=    gaussian(x, popt_yz[0], popt_yz[1],popt_yz[2])
        mean_yz    = popt_yz[1]
        plt.plot(x, fitted_data_sum_yz, label='Fitted x dir')
        plt.axvline(mean_yz, color='r', linestyle='--', label='FWHM Bounds')
    except:
            fwhm_yz,fitted_data_sum_yz,mean_yz=0,0,0

    plt.legend(fontsize=f_s)
    #plt.yscale('log')
    plt.axis('tight')
    ax.tick_params(labelsize=f_s)
    plt.grid(alpha=0.01)
    #plt.xlim(50,300)
    fwhm_=np.array([fwhm_yz,fwhm_xz,fwhm_xy])
    return fwhm_
# Gaussian profile
def gaussian(x, A, x0, sigma):
    return A * np.exp(-(x - x0)**2 / (2 * sigma**2))

# Lorentzian profile
def lorentzian(x, A, x0, gamma):
    return A / (1 + ((x - x0) / gamma)**2)

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
    gaussian = np.exp(-(x - x0)**2 / (2 * sigma**2))
    lorentzian = 1 / (1 + ((x - x0) / gamma)**2)
    return  A *(1 - alpha) * gaussian + A *alpha * lorentzian
# Voigt profile
def voigt(x, A, x0, sigma_g, gamma, alpha):
    return A * (1 - alpha) * gaussian(x, A, x0, sigma_g) + alpha * lorentzian(x, A, x0, gamma)
# Exponentially Broadened Gaussian profile
def exp_broadened_gaussian(x, A, x0, sigma, beta):
    return A * np.exp(-beta * (x - x0)) * np.exp(-(x - x0)**2 / (2 * sigma**2))
# Doniach-Sunjic profile
def doniach_sunjic(x, I0, gamma, beta, alpha):
    term1 = I0 * (1 + (x / gamma)**2)**(-beta)
    term2 = alpha * I0 * np.exp(-(x / gamma))
    return term1 + term2
# Pearson VII profile
def pearson_vii(x, A, x0, gamma, m):
    gamma_safe = np.maximum(gamma, 1e-10)  # Prevent division by near-zero gamma
    x_diff = np.clip((x - x0) / gamma_safe, -1e10, 1e10)  # Clip extreme values

    term1 = A / ((1 + x_diff**2)**m)
    return term1

# Skewed Lorentzian profile
def skewed_lorentzian(x, A, x0, gamma, delta):
    term1 = (1 / (np.pi * gamma)) * (gamma**2 / ((x - x0 - delta)**2 + gamma**2))
    return A * term1


def pseudo_voigt_fwhm(sigma, gamma,eta ):
    """Calculates the FWHM of a pseudo-Voigt distribution.

  Args:
    sigma: The standard deviation of the Gaussian distribution.
    eta: The mixing ratio of the Gaussian and Lorentzian distributions.
    gamma: The half-width at half-maximum (HWHM) of the Lorentzian distribution.

  Returns:
    The FWHM of the pseudo-Voigt distribution.
  """

    # Calculate the variance of the Gaussian distribution.
    sigma_variance = sigma**2

    # Calculate the weighted variance of the Lorentzian distribution.
    lorentzian_variance = 2 * eta**2 * gamma**2

    # Add the variance of the Gaussian distribution to the weighted variance of the Lorentzian distribution.
    total_variance = sigma_variance + lorentzian_variance

    # Take the square root of the sum of the variances.
    fwhm = np.sqrt(total_variance)
    fwhm *=2.355
    # Multiply the square root by 2.355 to get the FWHM of the pseudo-Voigt distribution.
    return fwhm
def fwhm_calculation_geom_methode1(x_data,y_fit):
    peaks, _ = find_peaks(y_fit)
    half_max = max(y_fit) / 2
    left_peak = peaks[0]
    right_peak = peaks[-1]
    
    while y_fit[left_peak] > half_max:
        left_peak -= 1
    while y_fit[right_peak] > half_max:
        right_peak += 1
    
    fwhm = x_data[right_peak] - x_data[left_peak]
    return fwhm
def fwhm_calculation_geom_methode2(x_data,y_fit):    
    peaks, _ = find_peaks(y_fit)
    half_max = max(y_fit) / 2
    indices_higher_than_hm=np.where(y_fit>=half_max)[0]
    left_peak = indices_higher_than_hm[0]
    right_peak = indices_higher_than_hm[-1]   
    
    fwhm = x_data[right_peak] - x_data[left_peak]
    return fwhm

# Fitting and evaluation
def fit_best_profile(x_data, y_data,x_data_fit):
    profiles = [
        (gaussian, "Gaussian", [max(y_data), np.argmax(y_data), 5]),
        (lorentzian, "Lorentzian", [max(y_data), np.argmax(y_data), 5]),
      #  (voigt, "Voigt", [max(y_data), np.argmax(y_data), 10, 5, 0.5]),
        (pearson_vii, "Pearson VII", [max(y_data), 6, 0.5, 3]),
        (pseudo_voigt, "Pseudo-Voigt", [max(y_data), np.argmax(y_data), 10, 5, 0.15])]
    
    best_r_squared = -1
    best_profile = None
    for profile, name, initial_guess in profiles:
        np.random.seed(0)
        popt, _ = curve_fit(profile, x_data, y_data+ 0.25 * np.random.normal(size=len(x_data)), p0=initial_guess,maxfev=4000000)
        y_fit = profile(x_data_fit, *popt)
        sse = np.sum((y_data - profile(x_data, *popt))**2)
        mean = popt[1]  # Peak position as mean
        width = 2 * popt[2]  # FWHM as width
        half_max = popt[0] / 2
        # Calculate FWHM based on the fit parameters
        fwhm=find_fwhm_all(x_data_fit,y_fit)
        i_sigma=0
        
        r_squared = 1 - sse / np.sum((y_data - np.mean(y_data))**2)
        if r_squared > best_r_squared:
            best_r_squared = r_squared
            
            best_profile = (name, popt, r_squared, y_fit, np.abs(fwhm))

    return best_profile
def pseudo_voigt_fwhm_Scherrer(lambda_,FWHM ,theta,k=0.9 ):
    """Calculates the size of the crystallite in angstroms
  Args:

    FWHM (beta): is the full-width at half maximum of the X-ray diffraction peak in radians
    K: is a constant (typically 0.9)
    λ: is the wavelength of the X-rays in angstroms
    θ: is the Bragg angle in degrees

  Returns:
    D: is the size of the crystallite in angstroms
  """
    fact=(k*lambda_)/np.cos(np.cos(theta*np.pi/180))
    return fact/(FWHM)
def theta_bragg_pt(lambda_,h,k,l,a0=3.924):
    
  """Calculates the Bragg angle for a given wavelength.

  Args:
    lambda_: The wavelength of the X-rays in Angstroms.

  Returns:
    The Bragg angle in degrees.
  """

  # The Bragg angle for the (111) plane of platinum is 21.28 degrees at the
  # wavelength of X-rays used in most diffraction experiments.

  d=a0/np.sqrt(h*h+k*k+l*l)  # Spacing between the (111) planes of platinum in Angstroms.

  return np.arcsin(lambda_ / (2 * d)) * 180 / np.pi
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
def find_fwhm_all(x_data_fit, y_fit):
    """
    Compute the Full-Width at Half Maximum (FWHM) of a given peak function.

    This function attempts to determine the FWHM using multiple approaches:
    1. **Geometric Method (Primary Method)**:
       - Finds the two points where the intensity is at half the peak maximum.
       - Uses `fwhm_calculation_geom_methode2` to estimate the width.
    2. **Alternative Geometric Method**:
       - If the first method fails, it uses numerical differentiation (`np.sign(y_fit - half_max)`)
         to locate the half-max points and compute FWHM.
    3. **Analytical Method (If a Fitted Model Exists)**:
       - If a specific peak model (`Gaussian`, `Lorentzian`, `Pseudo-Voigt`) was fitted,
         it computes FWHM based on the model parameters:
         - **Gaussian:** `FWHM = 2.355 * sigma`
         - **Lorentzian:** `FWHM = 2 * gamma`
         - **Pseudo-Voigt:** Uses `pseudo_voigt_fwhm(sigma, gamma, alpha)`
    4. **Fallback Case**:
       - If all methods fail, it returns `0` as a default.

    Parameters:
    -----------
    x_data_fit : numpy.ndarray
        The x-axis values of the fitted data (e.g., pixel or spatial coordinates).
    y_fit : numpy.ndarray
        The corresponding y-axis values (intensity or function values).

    Returns:
    --------
    fwhm : float
        The computed Full-Width at Half Maximum (FWHM). Returns `0` if no valid FWHM is found.

    Notes:
    ------
    - The function prioritizes **geometric methods** for generality.
    - If a **named peak profile is fitted**, it prefers analytical solutions.
    - Ensures robustness by trying multiple methods before defaulting to `0`.

    Example:
    --------
    >>> x = np.linspace(-10, 10, 100)
    >>> y = np.exp(-x**2 / 2)  # Gaussian-like peak
    >>> fwhm = find_fwhm_all(x, y)
    >>> print(f"FWHM: {fwhm:.3f}")

    """
    try:
        fwhm=fwhm_calculation_geom_methode2(x_data_fit,y_fit)
        #print(name+"first methode")
    except:
        #print("not working")
        try:
                            idx = np.argwhere(np.diff(np.sign(y_fit - half_max))).flatten()
                            fwhm = x_data_fit[idx[-1]] - x_data_fit[idx[0]]
        except:
            try: 
                if name=="Pseudo-Voit": ##skiped for now the numerical solution doesn't work well on y direction
                    A, x0, sigma, gamma, alpha = popt
                    fwhm = pseudo_voigt_fwhm(sigma, gamma, alpha)       
                else: 
                        if name==  "Gaussian":    
                            A, x0, sigma= popt
                            fwhm = 2.355 *sigma
                        else:
                            if name==  "Lorentzian":    
                                A, x0, gamma= popt
                                fwhm = 2*gamma                    
            except:
                fwhm = 0
    return fwhm 
#-------------------------------------------------------------------------------------------------------------
def integral_fwhm(x, y):
    """
    Compute the Integral Full-Width at Half Maximum (Integral FWHM).

    Parameters:
    x : numpy.ndarray
        The x-axis values (e.g., pixel positions or spatial coordinates).
    y : numpy.ndarray
        The corresponding y-axis values (intensity or function values).

    Returns:
    float
        The Integral FWHM value.
    """
    integral_intensity = np.trapz(y, x)  # Numerical integration
    peak_max = np.max(y)
    
    return integral_intensity / peak_max if peak_max != 0 else 0
#-------------------------------------------------------------------------------------------------------------

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
    if noise_levels is None:
        noise_levels = np.arange(0.01, 0.5, 0.01)

    profiles = [
        (gaussian, "Gaussian", [max(y_data), np.argmax(y_data), 5]),
        (lorentzian, "Lorentzian", [max(y_data), np.argmax(y_data), 5]),
        (pearson_vii, "Pearson VII", [max(y_data), 6, 0.5, 3]),
        (pseudo_voigt, "Pseudo-Voigt", [max(y_data), np.argmax(y_data), 10, 5, 0.15])
    ]

    best_r_squared = -1
    best_profile = None

    # Loop over noise levels
    for noise_level in noise_levels:
        # Add Gaussian noise to data
        noisy_y_data = y_data + np.random.normal(scale=noise_level, size=len(y_data))

        # Loop over peak profiles
        for profile_func, profile_name, initial_guess in profiles:
            try:
                # Fit the profile to noisy data
                popt, _ = curve_fit(profile_func, x_data, noisy_y_data, p0=initial_guess, maxfev=5000)

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
                    best_profile = (profile_name, popt, r_squared, y_fit, fwhm, integral_fwhm_value)
                    
            except:
                continue  

    return best_profile

#-------------------------------------------------------------------------------------------------------------
def get_plot_fwhm_and_skewness_kurtosis(data,
                                        plot_title="Sum of intensity in each direction",
                                        center_peak=False,
                                        save_fig=None,
                                        plot=True,
                                        eliminate_linear_background=False,
                                        plot_show=True,
                                        f_s=14):
    """
    Analyze the intensity distribution in a 3D dataset along X, Y, and Z directions, 
    computing key statistical properties including:
    - Full-Width at Half Maximum (FWHM) and Integral FWHM.
    - Skewness and kurtosis for shape analysis.
    - R-squared value for the quality of peak fitting.
    - Visualizing the distribution through plots with peak fitting.

    Parameters:
    -----------
    data : numpy.ndarray
        3D array representing the intensity data (e.g., from Bragg Coherent Diffraction Imaging).
    
    plot_title : str, optional
        Title of the intensity sum plot. Default is `"Sum of intensity in each direction"`.
    
    center_peak : bool, optional
        If `True`, recenters the peak before fitting. Useful for analyzing small ROIs.
    
    save_fig : str or None, optional
        If a file path is provided, saves the generated plot instead of displaying it.

    plot : bool, optional
        Whether to generate plots of the data and fits. Default is `True`.

    eliminate_linear_background : bool, optional
        If `True`, subtracts a linear background before fitting to improve accuracy.

    plot_show : bool, optional
        If `True`, displays the plot. If `False`, suppresses display for batch processing.

    f_s : int, optional
        Font size for labels, titles, and legends. Default is `14`.

    Returns:
    --------
    fwhm_xyz : list of float
        Standard Full-Width at Half Maximum (FWHM) values for each axis.

    integral_fwhm_xyz : list of float
        Integral Full-Width at Half Maximum values for each axis.

    skewness_xyz : list of float
        Skewness values for intensity distributions along each axis.

    kurtosis_xyz : list of float
        Kurtosis values for intensity distributions along each axis.

    rsquared_xyz : list of float
        R² values indicating fit quality for each direction.

    Notes:
    ------
    - This function first sums the data along each axis, creating 1D distributions.
    - It fits each distribution with multiple peak profiles (Gaussian, Lorentzian, Pseudo-Voigt, Pearson VII) 
      and selects the best one based on R² score.
    - If peak fitting fails, fallback strategies ensure meaningful results.
    - It plots 1D projections and fits, including FWHM markers and a summary table.

    Example Usage:
    --------------
    >>> fwhm_xyz, integral_fwhm_xyz, skewness_xyz, kurtosis_xyz, rsquared_xyz = get_plot_fwhm_and_skewness_kurtosis(data)
    """
    plt.style.use('grayscale')
    step_x_fit=0.25
    color_list=('#1f77b4','#2ca02c','#ff7f0e')
    directions=["X","Y","Z"]
    h_len=20
    background_degree = 1  # Adjust the degree of the polynomial as needed
    first_and_last_pixel=[0,-1]
    
    if plot:
        figsize = get_figure_size(scale=3)
        figure = plt.figure( figsize=figsize, dpi=150)
        figure.set_constrained_layout(True)  # Preferred for automatic handling
        # Define the grid layout with one row for the large subplot and three subplots in the second row
        gs = gridspec.GridSpec(nrows=3, ncols=3, wspace=0.3, hspace=0.5)
        
        ax_dummy = figure.add_subplot(gs[1, -1])  # Add subplot to the last position (rightmost in the second row)
        ax_dummy.axis('off')  # Turn off visibility of the dummy subplot        
        ax = figure.add_subplot(gs[0:2, :])  # Colons (:) span all columns in the first row
        ax1 = figure.add_subplot(gs[2, 0])
        ax2 = figure.add_subplot(gs[2, 1])
        ax3 = figure.add_subplot(gs[2, 2])
        plot_3D_projections(data,ax=[ax1,ax2,ax3],fig=figure,log_scale=True,cmap="jet",vmin=1e0,vmax=1e6,colorbar=True,tight_layout=False)
        ax1.axis('tight')
        ax2.axis('tight')
        ax3.axis('tight')
        ax1.set_xlabel('Z')
        ax2.set_xlabel('Z')
        ax3.set_xlabel('Y')
        ax1.set_ylabel('Y')
        ax2.set_ylabel('X')
        ax3.set_ylabel('X')
        
        ax.set_title(plot_title)
        ax.set_ylabel("Intensity", fontsize=f_s)
        ax.set_xlabel("(x,y,z)", fontsize=f_s)
        #figure.tight_layout()
    axes = [(1,2), (0,2), (0,1)]
    data_sum_all                    = [data.sum(axis=axis)                                                     for axis in axes]  # Vectorized summation
    X_data_all                      = [np.arange(len(d))                                                       for d in data_sum_all]  # Avoid recomputation
    X_fit_all                       = [np.linspace(0, len(d), int(len(d) / step_x_fit))                        for d in data_sum_all]  # More robust
    if center_peak:
        loc_max_all                 = [np.where(d==d.max())[0][0]                                              for d in data_sum_all]      
        loc_max                     = [np.array(data_sum_all[d] [loc_max_all[d]-h_len:loc_max_all[d]+h_len])   for d in range(len(data_sum_all))]  
        X_data_all                  = [np.arange(len(d))                                                       for d in data_sum_all]  # Avoid recomputation
        X_fit_all                   = [np.linspace(0, len(d), int(len(d) / step_x_fit))                        for d in data_sum_all]  # More robust
         
    if eliminate_linear_background:
        background_coefficients_all = [np.polyfit(X_data_all[d][first_and_last_pixel], data_sum_all[d][first_and_last_pixel], background_degree)        for d in range(len(data_sum_all))]
        background_fit_all          = [np.polyval(background_coefficients_all[d], X_data_all[d] )                                                        for d in range(len(data_sum_all))]
        data_sum_all                = [data_sum_all[d] - background_fit_all[d]      for d in range(len(data_sum_all))]
        
    skewness_xyz                    = np.array([skew(d)                                 for d in data_sum_all])
    kurtosis_xyz                    = np.array([kurtosis(d)                             for d in data_sum_all])
    fits                            = [fit_best_profile_with_noise(X_data_all[i], data_sum_all[i], X_fit_all[i]) for i in range(len(data_sum_all))]
    names, popt_xyz, rsquared_xyz, fitted_data, fwhm_xyz, fwhm_integral_xyz = zip(*fits)
    for i in range(len(data_sum_all)):
        if i==0:            axis,direction=(1,2),"X"
        if (i==1):          axis,direction=(0,2),"Y"
        if (i==2):          axis,direction=(0,1),"Z"       
        popt=popt_xyz[i]
        try: 
            print(f"fit along {direction} results :")
            display(f"A {popt[0]} {direction}0 {popt[1]} sigma {popt[2]} gamma  {popt[3]}  eta {popt[4]} "  )
        except:
            print('Not pseudo-voigt')       
        if plot:
            ax.scatter(X_data_all[i],data_sum_all[i], s=10, color=color_list[i])
            ax.plot(X_fit_all[i], fitted_data[i]         , color=color_list[i] ,label=f"Fit {direction} {names[i]}")  


    
    if plot:
        data_sum_fit_max = np.array([max(fit) for fit in fitted_data])  # Get max values for each fitted dataset
        data_sum_fit_max_max = data_sum_fit_max.max()  # Overall max value for normalization
    
        for i, (fit, color, popt) in enumerate(zip(fitted_data, color_list, popt_xyz)):
            half_max_norm = 0.51 * data_sum_fit_max[i] / data_sum_fit_max_max  # Normalize half max value
    
            # Draw vertical lines for FWHM bounds
            ax.axvline(x=popt[1] - fwhm_xyz[i] / 2, color=color, linestyle='--', ymin=0, ymax=half_max_norm)
            ax.axvline(x=popt[1] + fwhm_xyz[i] / 2, color=color, linestyle='--', ymin=0, ymax=half_max_norm)
    
        ax.tick_params(labelsize=f_s)
        ax.grid(alpha=0.01)
        ax.legend(loc="best")

    if plot:
        # Round numerical values before constructing the table
        rounded_values = np.round(
            np.column_stack((fwhm_xyz, fwhm_integral_xyz, skewness_xyz, kurtosis_xyz, rsquared_xyz)), 4
        )
    
        # Create a list of lists to represent the table data
        table_data = np.vstack((
            ['Direction', 'FWHM', 'Integral FWHM', 'Skewness', 'Kurtosis', 'R-squared'],
            np.column_stack((directions, rounded_values))
        )).tolist()
    
        # Create table
        table = Table(ax, loc='upper left')
        table.auto_set_font_size(False)
        table.set_fontsize(18)
        table.scale(2, 2)  # Adjust scale as needed
    
        # Add table data
        for i, row in enumerate(table_data):
            for j, cell in enumerate(row):
                if isinstance(cell, float):  # Check if the cell value is a float
                    cell = round(cell, 4)  # Round float values to four decimal places
                table.add_cell(i, j, width=1, height=0.05, text=str(cell),
                               facecolor='darkblue', loc='center', edgecolor='darkblue')
                table[i, j].set_text_props(color='w', weight='bold')
    
        # Adjust column widths
        for col in range(len(table_data[0])):
            table.auto_set_column_width(col)
    
        ax.add_table(table)
        ax.axis('tight')
    
    
    # Prepare table data
    table_data = [
        ["Metric", "X", "Y", "Z"],
        ["FWHM", fwhm_xyz[0], fwhm_xyz[1], fwhm_xyz[2]],
        ["FWHM Integral", fwhm_integral_xyz[0], fwhm_integral_xyz[1], fwhm_integral_xyz[2]],
        ["Skewness", skewness_xyz[0], skewness_xyz[1], skewness_xyz[2]],
        ["Kurtosis", kurtosis_xyz[0], kurtosis_xyz[1], kurtosis_xyz[2]],
    ]
    
    # Print formatted table
    print(tabulate(table_data, headers="firstrow", tablefmt="fancy_grid"))

    #if plot:
    #    figure.tight_layout()
    if save_fig:
        plt.savefig(save_fig)            
    if plot:
        if plot_show:
            plt.show()
        else:
            plt.close()
    return fwhm_xyz,fwhm_integral_xyz,skewness_xyz,kurtosis_xyz

