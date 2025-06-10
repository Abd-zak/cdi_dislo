from cdi_dislo.common_imports import * # import os ,lam2en, glob ,C_O_M, h5py , numpy as np , pandas as pd , matplotlib.pyplot as plt ; from scipy.optimize import curve_fit ;from typing import List, Tuple, Union, Optional ;from math import log ;from numpy import array ; from sklearn.metrics import r2_score
from cdi_dislo.general_utilities.cdi_dislo_utils import (
    mask_clusters,
    crop_3darray_pos,
    zero_to_nan
)
from cdi_dislo.orthogonalisation_handler.cdi_dislo_ortho_handler  import get_lattice_parametre
from cdi_dislo.plotutilities.cdi_dislo_plotutilities              import plot_summary_difraction
from cdi_dislo.ewen_utilities.plot_utilities                      import plot_3D_projections



#####################################################################################################################
#####################################################################################################################
######################################### diffraction analysis utility    ###########################################
#####################################################################################################################
#####################################################################################################################
def plot_temperature_and_lattice(nov_2022_best_T_exp, nov_2022_best_T_theo, nov_2022_best_delta_temp, nov_2022_C_,save_plot=None):
    """
    Plot the comparison between experimental and theoretical temperatures,
    the temperature difference, the lattice parameter, and lattice parameter vs temperature
    in a single figure with four subplots.

    Parameters:
    nov_2022_best_T_exp (array): Experimental temperatures
    nov_2022_best_T_theo (array): Theoretical temperatures
    nov_2022_best_delta_temp (array): Temperature difference (exp - theo)
    nov_2022_C_ (array): Lattice parameter values
    """
    fig = plt.figure(figsize=(15, 20))
    gs = fig.add_gridspec(4, 2)

    # Plot experimental vs theoretical temperatures
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(nov_2022_best_T_exp, label='Experimental', marker='o')
    ax1.plot(nov_2022_best_T_theo, label='Theoretical', marker='s')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Experimental vs Theoretical Temperatures')
    ax1.legend()
    ax1.grid(True)

    # Plot temperature difference
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(nov_2022_best_delta_temp, label='Exp - Theo', color='red', marker='o')
    ax2.set_ylabel('Temperature Difference (°C)')
    ax2.set_title('Temperature Difference (Experimental - Theoretical)')
    ax2.legend()
    ax2.grid(True)

    # Plot lattice parameter
    ax3 = fig.add_subplot(gs[2, :])
    ax3.plot(nov_2022_C_, label='Lattice Parameter', color='green', marker='o')
    ax3.set_xlabel('Measurement Index')
    ax3.set_ylabel('Lattice Parameter (Å)')
    ax3.set_title('Sapphire Lattice Parameter')
    ax3.legend()
    ax3.grid(True)

    # Plot Lattice Parameter vs Temperature
    ax4 = fig.add_subplot(gs[3, :])
    ax4.scatter(nov_2022_best_T_exp, nov_2022_C_, label='Experimental', color='blue')
    ax4.scatter(nov_2022_best_T_theo, nov_2022_C_, label='Theoretical', color='red')
    ax4.set_xlabel('Temperature (°C)')
    ax4.set_ylabel('Lattice Parameter (Å)')
    ax4.set_title('Sapphire Lattice Parameter vs Temperature')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    if save_plot:
        plt.savefig(save_plot,dpi=150)
    plt.show()

def extract_coefficient_and_exponent(number):
    # Extract the exponent
    exponent = int(math.log10(np.abs(number)))
    # Calculate the coefficient
    coefficient = number / (10 ** exponent)
    return coefficient, exponent


def get_prediction_from_theo_sapphire_lattice(T_celsius, plot=False):
    """
    Predicts the lattice parameter of sapphire at a given temperature, ensuring it passes through 12.991 at 27°C (300.15K).
    
    Parameters:
        T_celsius (float): Temperature in Celsius.
        plot (bool): Whether to plot the results.
        
    Returns:
        prediction_a_0C (float): Lattice parameter at 0°C.
        predict_at_T_poly (float): Prediction at given temperature using adjusted polynomial fit.
        coefficients (array): Polynomial coefficients.
    """
    T_kelvin = T_celsius + 273.15
    
    # Temperature data and lattice parameter data
    T_data_sapphire = np.array([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 
                                900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000])

    def_theo_sapphire_Y = np.array([-5.33, -5.33, -5.21, -4.56, -3.20, -1.16, 1.44, 4.51, 7.94, 11.65, 15.58, 19.69, 23.94, 28.31, 32.77, 37.31,
                                    41.92, 51.29, 60.82, 70.48, 80.09, 89.76, 99.47, 109.2, 119.0, 128.7, 138.6, 148.4, 158.2]) * 10**-4

    # Function to calculate lattice parameter
    def calculate_lattice_param(base_param, T):
        coefficients = np.polyfit(T_data_sapphire, def_theo_sapphire_Y, 4)
        poly_function = np.poly1d(coefficients)
        return base_param * (poly_function(T) + 1)

    # Target temperature in Kelvin
    T_target = 300.15  # 27°C

    # Target lattice parameter
    target_param = 12.991

    # Initial guess for base parameter
    base_param = 12.989

    # Iterative adjustment
    max_iterations = 10000
    tolerance = 1e-9

    for _ in range(max_iterations):
        calculated_param = calculate_lattice_param(base_param, T_target)
        if np.abs(calculated_param - target_param) < tolerance:
            break
        base_param += (target_param - calculated_param) / 2

    print(f"Adjusted base parameter: {base_param:.6f}")
    print(f"Lattice parameter at 27°C: {calculate_lattice_param(base_param, T_target):.6f}")
    
    def_theo_sapphire = calculate_lattice_param(base_param, T_data_sapphire)
    
    # Calculate final coefficients
    coefficients = np.polyfit(T_data_sapphire, def_theo_sapphire, 4)
    poly_function = np.poly1d(coefficients)
    
    prediction_a_0C = poly_function(273.15)  # 0°C
    predict_at_T_poly = poly_function(T_kelvin)

    #print(f"Polynomial coefficients: {coefficients}")
    print(f"Value of lattice parameter at {T_celsius:.2f}°C fit poly 4 degree: {predict_at_T_poly:.6f}")

    # Plotting
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(T_data_sapphire, def_theo_sapphire, "+", label='Adjusted Theoretical Sapphire')
        
        T_range = np.linspace(min(T_data_sapphire), max(T_data_sapphire), 200)
        curve_fit_theo_par = poly_function(T_range)
        plt.plot(T_range, curve_fit_theo_par, "g--", label=f'Polynomial Fit | R2={r2_score(def_theo_sapphire, poly_function(T_data_sapphire)):.4f}')
        
        plt.plot(300.15, 12.991, 'ro', label='Constraint Point (27°C, 12.991)')
        
        plt.legend()
        plt.xlabel('Temperature (K)')
        plt.ylabel('Lattice Parameter')
        plt.title('Sapphire Lattice Parameter vs Temperature (Adjusted Fit)')
        plt.show()

    return prediction_a_0C, predict_at_T_poly, coefficients
def calculate_epsilon_and_prediction(T_data_exp, results_sapphire_d_spacing, fit_order=2, plot=True,save_plot=None,a0=12.991,experiment=""):
    def get_temperatureprediction_fromdeformation_and_fitparametre(curve_fit,popt,fit_order):
        if fit_order == "exp":
            T_predict=((1/popt[1])*log(popt[0]/curve_fit)) + popt[2]
        elif isinstance(fit_order, int):
            if fit_order == 1:
                T_predict= (curve_fit -popt[1])/popt[0]
            else:
                T_predict = []
                for i in curve_fit:
                    result = [popt[ii] - float(i * float(ii == fit_order)) for ii in range(fit_order + 1)]
                    roots = np.roots(result)
                    real_roots = [root.real for root in roots if np.isreal(root)]
                    if real_roots:
                        T_predict.append(max(real_roots))
        else:
            raise ValueError("Fit order should be an integer between 1 and 6 or 'exp'")
        return np.array(T_predict)
    # Define the fitting function based on the order

    # Define the fitting function based on the order
    if fit_order == "exp":
        def func_def(T_, a, b,c):
            return a * np.exp(b * (T_-c))
        fit_degree = "Exponential"
    elif isinstance(fit_order, int):
        if fit_order == 1:
            def func_def(T_, a, b):
                return a * T_ + b
            fit_degree = "1st"
        elif fit_order == 2:
            def func_def(T_, a, b, c):
                return a * T_ ** 2 + b * T_ + c
            fit_degree = "2nd"
        elif fit_order == 3:
            def func_def(T_, a, b, c, d):
                return a * T_ ** 3 + b * T_ ** 2 + c * T_ + d
            fit_degree = "3rd"
        elif fit_order == 4:
            def func_def(T_, a, b, c, d, e):
                return a * T_ ** 4 + b * T_ ** 3 + c * T_ ** 2 + d * T_ + e
            fit_degree = "4th"
        elif fit_order == 5:
            def func_def(T_, a, b, c, d, e, f):
                return a * T_ ** 5 + b * T_ ** 4 + c * T_ ** 3 + d * T_ ** 2 + e * T_ + f
            fit_degree = "5th"
        elif fit_order == 6:
            def func_def(T_, a, b, c, d, e, f, g):
                return a * T_ ** 6 + b * T_ ** 5 + c * T_ ** 4 + d * T_ ** 3 + e * T_ ** 2 + f * T_ + g
            fit_degree = "6th"
        else:
            raise ValueError("Fit order should be between 1 and 6 or 'exp'")
    else:
        raise ValueError("Fit order should be an integer between 1 and 6 or 'exp'")
    def func_def_theosapphire(T_, a, b, c, d, e):
                return a * T_ ** 4 + b * T_ ** 3 + c * T_ ** 2 + d * T_ + e
    # Theoretical data parameters
    T_data_sapphire   = array([  0. ,  50. , 100. , 150. , 200. , 250. , 300., 350. , 400. , 450. , 500. , 550.  , 600.  , 650.  , 700.  , 750.  , 800  , 900  ,1000  ,1100   , 1200    , 1300   , 1400    ,1500    , 1600   , 1700    ,   1800  , 1900   , 2000  
                               ]) - 273
    def_theo_sapphire = array([-5.33, -5.33, -5.21, -4.56, -3.20, -1.16, 1.44, 4.51 , 7.94 , 11.65, 15.58, 19.69 , 23.94 , 28.31 , 32.77 , 37.31 , 41.92, 51.29, 60.82, 70.48 , 80.09   , 89.76  , 99.47   ,  109.2 , 119.0  , 128.7   , 138.6   , 148.4  , 158.2]) * 10 ** -4
    def_theo_sapphire = 12.988461865901952 * (def_theo_sapphire + 1)
    T_data_plot = np.linspace(0, 900, 50)

    # Theoretical data fit
    popt_def, pcov_def = curve_fit(func_def_theosapphire, T_data_sapphire, def_theo_sapphire,maxfev=10**5)

    # Calculate corresponding temperature values for experimental data
    T_exp_predict = array(T_data_exp)#get_temperatureprediction_fromdeformation_and_fitparametre (results_sapphire_d_spacing,popt) 
    T_theo_predict = get_temperatureprediction_fromdeformation_and_fitparametre (results_sapphire_d_spacing,popt_def,4)
    T_data_exp=T_theo_predict
    
    # Experimental data fit
    popt, pcov = curve_fit(func_def, T_theo_predict, results_sapphire_d_spacing,maxfev=10**5)
    # Calculate theoretical and experimental fits

    curve_fit_exp_par = func_def(T_data_plot, *popt)
    curve_fit_theo_par = func_def_theosapphire(T_data_plot, *popt_def)


    # Calculate epsilon values
    epsilon_curve_fit_exp_par = np.round(100 * ((curve_fit_exp_par - a0) / a0), 2)
    epsilon_results_sapphire_d_spacing = np.round(100 * ((results_sapphire_d_spacing - a0) / a0), 2)
    epsilon_fit_theo_par = np.round(100 * ((curve_fit_theo_par - a0) / a0), 2)
    epsilon_theo_sapphire = np.round(100 * ((def_theo_sapphire - a0) / a0), 2)

    if plot:
        plt.figure(figsize=(20, 21))
        #plt.style.use('grayscale')
        # First subplot: Temperature Deviation between Experimental and Theoretical Values
        plt.subplot(3, 1, 1)
        plt.plot(T_exp_predict, T_exp_predict - T_theo_predict, 'o')
        plt.xticks(np.arange(0, 1000, step=50))
        plt.grid(alpha=0.25)
        plt.xlabel(r"$T_{\mathrm{exp}(°C)}$", fontsize=24)
        plt.ylabel("$ΔT_{(°C/K)} = T_{\mathrm{exp}} - T_{\mathrm{theo}}$", fontsize=24)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title('Temperature Deviation between Experimental and Theoretical Values')

        # Second subplot: Lattice Parameter (a) Values
        plt.subplot(3, 1, 2)
        plt.plot(T_data_plot, curve_fit_exp_par, '-', label='fit exp')
        plt.plot(T_data_exp, results_sapphire_d_spacing, 'o', label='data exp')
        plt.plot(T_data_plot, curve_fit_theo_par, 'r-', alpha=0.5, label='fit theo')
        plt.plot(T_data_sapphire, def_theo_sapphire, 'r*', alpha=0.5, label='data theo')

        plt.xlabel(r"$T_{(°C)}$", fontsize=24)
        plt.ylabel(r"$a_{(Å)}$", fontsize=24)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xticks(np.arange(-300, 1000, step=50))
        plt.grid(alpha=0.25)
        plt.title('Lattice Parameter Comparison')

        # Add a box for fit parameters
        if fit_order==1:
            text_annote_fit_res=r'$a_{\mathrm{fit(exp)}} = %.2f\,T \times 10^{%0.0f} + %.3f$' % (
                extract_coefficient_and_exponent(popt[0])[0], extract_coefficient_and_exponent(popt[0])[1],
                popt[1])
            text_annote_fit_res_epsilon=r'$\epsilon_{\mathrm{fit(exp)}} = %.2f\,T \times 10^{%0.0f} + %.3f\times 10^{%0.0f}$' % (
                extract_coefficient_and_exponent(popt[0]/a0)[0], extract_coefficient_and_exponent(popt[0]/a0)[1],
                extract_coefficient_and_exponent((popt[1]-a0)/a0)[0], extract_coefficient_and_exponent((popt[1]-a0)/a0)[1])
        if fit_order==2:
            text_annote_fit_res=r'$a_{\mathrm{fit(exp)}} = %.2f\,T^2 \times 10^{%0.0f} + %.2f\,T \times 10^{%0.0f} + %.3f$' % (
                extract_coefficient_and_exponent(popt[0])[0], extract_coefficient_and_exponent(popt[0])[1],
                extract_coefficient_and_exponent(popt[1])[0], extract_coefficient_and_exponent(popt[1])[1], popt[2])
            text_annote_fit_res_epsilon=r'$\epsilon_{\mathrm{fit(exp)}} = %.2f\,T^2 \times 10^{%0.0f} + %.2f\,T \times 10^{%0.0f} + %.3f$' % (
                extract_coefficient_and_exponent(popt[0]/a0)[0], extract_coefficient_and_exponent(popt[0]/a0)[1],
                extract_coefficient_and_exponent(popt[1]/a0)[0], extract_coefficient_and_exponent(popt[1]/a0)[1],
                (popt[2]-a0)/a0)
        if fit_order==3:
            text_annote_fit_res=r'$a_{\mathrm{(fit exp)}} = %.2f\,T^3 \times 10^{%0.0f} +%.2f\,T^2 \times 10^{%0.0f} + %.2f\,T \times 10^{%0.0f} + %.3f$' % (
                extract_coefficient_and_exponent(popt[0])[0], extract_coefficient_and_exponent(popt[0])[1],
                extract_coefficient_and_exponent(popt[1])[0], extract_coefficient_and_exponent(popt[1])[1],
                extract_coefficient_and_exponent(popt[2])[0], extract_coefficient_and_exponent(popt[2])[1],
                popt[3])
            text_annote_fit_res_epsilon=r'$\epsilon_{\mathrm{fit(exp)}} = %.2f\,T^3 \times 10^{%0.0f} +%.2f\,T^2 \times 10^{%0.0f} + %.2f\,T \times 10^{%0.0f} + %.3f$' % (
                extract_coefficient_and_exponent(popt[0]/a0)[0], extract_coefficient_and_exponent(popt[0]/a0)[1],
                extract_coefficient_and_exponent(popt[1]/a0)[0], extract_coefficient_and_exponent(popt[1]/a0)[1],
                extract_coefficient_and_exponent(popt[2]/a0)[0], extract_coefficient_and_exponent(popt[2]/a0)[1],
                (popt[3]-a0)/a0)
            
        if fit_order == 4:
            text_annote_fit_res = r'$a_{\mathrm{fit(exp)}} = %.2f\,T^4 \times 10^{%0.0f} + %.2f\,T^3 \times 10^{%0.0f} + %.2f\,T^2 \times 10^{%0.0f} + %.2f\,T \times 10^{%0.0f} + %.3f$' % (
                extract_coefficient_and_exponent(popt[0])[0], extract_coefficient_and_exponent(popt[0])[1],
                extract_coefficient_and_exponent(popt[1])[0], extract_coefficient_and_exponent(popt[1])[1],
                extract_coefficient_and_exponent(popt[2])[0], extract_coefficient_and_exponent(popt[2])[1],
                extract_coefficient_and_exponent(popt[3])[0], extract_coefficient_and_exponent(popt[3])[1],
                popt[4])
            text_annote_fit_res_epsilon = r'$\epsilon_{\mathrm{fit(exp)}} = %.2f\,T^4 \times 10^{%0.0f} + %.2f\,T^3 \times 10^{%0.0f} + %.2f\,T^2 \times 10^{%0.0f} + %.2f\,T \times 10^{%0.0f} + %.3f$' % (
                extract_coefficient_and_exponent(popt[0] / a0)[0],
                extract_coefficient_and_exponent(popt[0] / a0)[1],
                extract_coefficient_and_exponent(popt[1] / a0)[0],
                extract_coefficient_and_exponent(popt[1] / a0)[1],
                extract_coefficient_and_exponent(popt[2] / a0)[0],
                extract_coefficient_and_exponent(popt[2] / a0)[1],
                extract_coefficient_and_exponent(popt[3] / a0)[0],
                extract_coefficient_and_exponent(popt[3] / a0)[1],
                (popt[4] - a0) / a0)
        if fit_order == 5:
            text_annote_fit_res = r'$a_{\mathrm{fit(exp)}} = %.2f\,T^5 \times 10^{%0.0f} + %.2f\,T^4 \times 10^{%0.0f} + %.2f\,T^3 \times 10^{%0.0f} + %.2f\,T^2 \times 10^{%0.0f} + %.2f\,T \times 10^{%0.0f} + %.3f$' % (
                extract_coefficient_and_exponent(popt[0])[0], extract_coefficient_and_exponent(popt[0])[1],
                extract_coefficient_and_exponent(popt[1])[0], extract_coefficient_and_exponent(popt[1])[1],
                extract_coefficient_and_exponent(popt[2])[0], extract_coefficient_and_exponent(popt[2])[1],
                extract_coefficient_and_exponent(popt[3])[0], extract_coefficient_and_exponent(popt[3])[1],
                extract_coefficient_and_exponent(popt[4])[0], extract_coefficient_and_exponent(popt[4])[1],
                popt[5])
            text_annote_fit_res_epsilon = r'$\epsilon_{\mathrm{fit(exp)}} = %.2f\,T^5 \times 10^{%0.0f} + %.2f\,T^4 \times 10^{%0.0f} + %.2f\,T^3 \times 10^{%0.0f} + %.2f\,T^2 \times 10^{%0.0f} + %.2f\,T \times 10^{%0.0f} + %.3f$' % (
                extract_coefficient_and_exponent(popt[0] / a0)[0],
                extract_coefficient_and_exponent(popt[0] / a0)[1],
                extract_coefficient_and_exponent(popt[1] / a0)[0],
                extract_coefficient_and_exponent(popt[1] / a0)[1],
                extract_coefficient_and_exponent(popt[2] / a0)[0],
                extract_coefficient_and_exponent(popt[2] / a0)[1],
                extract_coefficient_and_exponent(popt[3] / a0)[0],
                extract_coefficient_and_exponent(popt[3] / a0)[1],
                extract_coefficient_and_exponent(popt[4] / a0)[0],
                extract_coefficient_and_exponent(popt[4] / a0)[1],
                (popt[5] - a0) / a0)

        if fit_order == 6:
            text_annote_fit_res = r'$a_{\mathrm{fit(exp)}} = %.2f\,T^6 \times 10^{%0.0f} + %.2f\,T^5 \times 10^{%0.0f} + %.2f\,T^4 \times 10^{%0.0f} + %.2f\,T^3 \times 10^{%0.0f} + %.2f\,T^2 \times 10^{%0.0f} + %.2f\,T \times 10^{%0.0f} + %.3f$' % (
                extract_coefficient_and_exponent(popt[0])[0], extract_coefficient_and_exponent(popt[0])[1],
                extract_coefficient_and_exponent(popt[1])[0], extract_coefficient_and_exponent(popt[1])[1],
                extract_coefficient_and_exponent(popt[2])[0], extract_coefficient_and_exponent(popt[2])[1],
                extract_coefficient_and_exponent(popt[3])[0], extract_coefficient_and_exponent(popt[3])[1],
                extract_coefficient_and_exponent(popt[4])[0], extract_coefficient_and_exponent(popt[4])[1],
                extract_coefficient_and_exponent(popt[5])[0], extract_coefficient_and_exponent(popt[5])[1],
                popt[6])
            text_annote_fit_res_epsilon = r'$\epsilon_{\mathrm{fit(exp)}} = %.2f\,T^6 \times 10^{%0.0f} + %.2f\,T^5 \times 10^{%0.0f} + %.2f\,T^4 \times 10^{%0.0f} + %.2f\,T^3 \times 10^{%0.0f} + %.2f\,T^2 \times 10^{%0.0f} + %.2f\,T \times 10^{%0.0f} + %.3f$' % (
                extract_coefficient_and_exponent(popt[0] / a0)[0],
                extract_coefficient_and_exponent(popt[0] / a0)[1],
                extract_coefficient_and_exponent(popt[1] / a0)[0],
                extract_coefficient_and_exponent(popt[1] / a0)[1],
                extract_coefficient_and_exponent(popt[2] / a0)[0],
                extract_coefficient_and_exponent(popt[2] / a0)[1],
                extract_coefficient_and_exponent(popt[3] / a0)[0],
                extract_coefficient_and_exponent(popt[3] / a0)[1],
                extract_coefficient_and_exponent(popt[4] / a0)[0],
                extract_coefficient_and_exponent(popt[4] / a0)[1],
                extract_coefficient_and_exponent(popt[5] / a0)[0],
                extract_coefficient_and_exponent(popt[5] / a0)[1],
                (popt[6] - a0) / a0)
        if fit_order == "exp":
            text_annote_fit_res = r'$a_{\mathrm{fit(exp)}} = %.2f \times e^{%.2f(T - %.2f)}$' % (popt[0], popt[1], popt[2])
            text_annote_fit_res_epsilon = r'$\epsilon_{\mathrm{fit(exp)}} = %.2f \times e^{%.2f(T - %.2f)}$' % (
                popt[0] / a0, popt[1], popt[2])

           
        plt.text(0.02, 0.95,text_annote_fit_res ,
            transform=plt.gca().transAxes,
            bbox=dict(facecolor='blue', alpha=0.9),
            fontsize=12, color='white')
        plt.legend(ncol=2,loc=4, fontsize=12)

        # Third subplot: Epsilon values
        plt.subplot(3, 1, 3)
        plt.plot(T_data_plot, epsilon_curve_fit_exp_par, label='epsilon fit exp')
        plt.plot(T_data_exp, epsilon_results_sapphire_d_spacing, 'o', label='epsilon data exp')
        plt.plot(T_data_plot, epsilon_fit_theo_par, 'r-', alpha=0.5, label='epsilon fit theo')
        plt.plot(T_data_sapphire, epsilon_theo_sapphire, 'r*', alpha=0.5, label='epsilon data theo')

        plt.xlabel(r"$T_{(°C)}$", fontsize=24)
        plt.ylabel(r"$\epsilon_{(\%)}$", fontsize=24)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xticks(np.arange(-300, 1000, step=50))
        #plt.yticks(np.linspace(epsilon_fit_theo_par.min(), epsilon_fit_theo_par.max(), 10))
        plt.grid(alpha=0.25)
        plt.title('Thermal deformation of Sapphire substrate')
        plt.text(0.02, 0.95,text_annote_fit_res_epsilon ,
            transform=plt.gca().transAxes,
            bbox=dict(facecolor='blue', alpha=0.9),
            fontsize=12, color='white')
        plt.legend(ncol=2,loc=4, fontsize=12)

        plt.tight_layout()
        plt.suptitle(f"Temperature calibration for experiment {experiment} ({fit_degree} degree polynomial fit)", y=1.01)
        if save_plot:
            plt.savefig(save_plot,dpi=150,facecolor='gray', edgecolor='gray')
        plt.show()

    return (T_data_plot, T_exp_predict, T_theo_predict, curve_fit_exp_par, curve_fit_theo_par, def_theo_sapphire, epsilon_curve_fit_exp_par, epsilon_results_sapphire_d_spacing, epsilon_fit_theo_par, epsilon_theo_sapphire)
def calculate_epsilon_and_prediction_2nd_degree(T_data_exp, results_sapphire_d_spacing, plot=True, save_plot=None, a0=12.991, experiment=""):

    # Define the 2nd-degree polynomial fitting function
    def func_def(T_, a, b, c):
        return a * T_ ** 2 + b * T_ + c

    # Theoretical data parameters
    T_data_sapphire = np.array([0., 50., 100., 150., 200., 250., 300., 350., 400., 450., 500., 
                                550., 600., 650., 700., 750., 800., 900., 1000., 1100., 1200.,
                                1300., 1400., 1500., 1600., 1700., 1800., 1900., 2000.]) - 273
    def_theo_sapphire = np.array([-5.33, -5.33, -5.21, -4.56, -3.20, -1.16, 1.44, 4.51, 7.94, 11.65, 15.58, 
                                  19.69, 23.94, 28.31, 32.77, 37.31, 41.92, 51.29, 60.82, 70.48, 80.09, 
                                  89.76, 99.47, 109.2, 119.0, 128.7, 138.6, 148.4, 158.2]) * 10**-4
    def_theo_sapphire = 12.988461865901952 * (def_theo_sapphire + 1)

    # Define a range for plotting
    Avril_2023_T_data_plot = np.linspace(0, 900, 50)

    # Fit experimental data with a 2nd-degree polynomial
    popt, pcov = curve_fit(func_def, T_data_exp, results_sapphire_d_spacing, maxfev=10**5)

    # Calculate fit values for plotting
    Avril_2023_curve_fit_exp_par = func_def(Avril_2023_T_data_plot, *popt)

    # Calculate theoretical lattice parameter prediction
    Avril_2023_curve_fit_theo_par = func_def(Avril_2023_T_data_plot, *popt)

    # Calculate epsilon values for experimental fit and theoretical values
    Avril_2023_epsilon_curve_fit_exp_par = np.round(100 * ((Avril_2023_curve_fit_exp_par - a0) / a0), 2)
    epsilon_results_sapphire_d_spacing = np.round(100 * ((results_sapphire_d_spacing - a0) / a0), 2)
    
    # Theoretical epsilon
    Avril_2023_def_theo_sapphire = def_theo_sapphire
    epsilon_theo_sapphire = np.round(100 * ((Avril_2023_def_theo_sapphire - a0) / a0), 2)
    
    # Temperature predictions for experimental and theoretical data
    Avril_2023_T_exp_predict = np.polyval(popt, T_data_exp)
    Avril_2023_T_theo_predict = np.polyval(popt, T_data_sapphire)

    # Calculate epsilon for theoretical fit
    epsilon_fit_theo_par = np.round(100 * ((Avril_2023_curve_fit_theo_par - a0) / a0), 2)

    if plot:
        plt.figure(figsize=(10, 10))

        # First subplot: Lattice Parameter (a) Values
        plt.subplot(3, 1, 1)
        plt.plot(Avril_2023_T_data_plot, Avril_2023_curve_fit_exp_par, '-', label='fit exp')
        plt.plot(T_data_exp, results_sapphire_d_spacing, 'o', label='data exp')

        plt.xlabel(r"$T_{(°C)}$", fontsize=14)
        plt.ylabel(r"$a_{(Å)}$", fontsize=14)
        plt.grid(alpha=0.25)
        plt.legend()

        # Second subplot: Epsilon values
        plt.subplot(3, 1, 2)
        plt.plot(Avril_2023_T_data_plot, Avril_2023_epsilon_curve_fit_exp_par, label='epsilon fit exp')
        plt.plot(T_data_exp, epsilon_results_sapphire_d_spacing, 'o', label='epsilon data exp')

        plt.xlabel(r"$T_{(°C)}$", fontsize=14)
        plt.ylabel(r"$\epsilon_{(\%)}$", fontsize=14)
        plt.grid(alpha=0.25)
        plt.legend()

        # Third subplot: Delta T values
        plt.subplot(3, 1, 3)
        delta_T = Avril_2023_T_exp_predict - T_data_exp
        plt.plot(T_data_exp, delta_T, 'o-', label=r'$\Delta T = T_{fit} - T_{exp}$')
        plt.axhline(0, color='gray', linestyle='--')
        plt.xlabel(r"$T_{(°C)}$", fontsize=14)
        plt.ylabel(r"$\Delta T_{(°C)}$", fontsize=14)
        plt.grid(alpha=0.25)
        plt.legend()

        plt.tight_layout()
        plt.suptitle(f"Temperature calibration for experiment {experiment} (2nd-degree polynomial fit)", y=1.02)
        
        if save_plot:
            plt.savefig(save_plot, dpi=150)
        plt.show()

    return (Avril_2023_T_data_plot, Avril_2023_T_exp_predict, Avril_2023_T_theo_predict, Avril_2023_curve_fit_exp_par, 
            Avril_2023_curve_fit_theo_par, Avril_2023_def_theo_sapphire, Avril_2023_epsilon_curve_fit_exp_par, 
            epsilon_results_sapphire_d_spacing, epsilon_fit_theo_par, epsilon_theo_sapphire)
def get_theo_strain_thermal_expansion(temp, material='pt', T0=23):
    """
    Calculate the theoretical strain due to thermal expansion for a given material.

    Parameters:
    temp (float): The temperature in Kelvin at which to calculate the strain.
    material (str): The material for which to calculate the strain ('pt' for platinum, 'sapphire_para' for sapphire parallel to c-axis, 'sapphire_perp' for sapphire perpendicular to c-axis).
    T0 (float): The reference temperature in Kelvin (default is 23°C converted to Kelvin).

    Returns:
    float: The calculated strain.
    """
    
    delta_T = temp - T0
    if material == 'pt':
        # Coefficients for platinum
        a1 = float(9.122e-6)
        a2 = float(7.467e-10)
        a3 = float(4.258e-13)
        return a1 * delta_T #+ a2 * delta_T**2 + a3 * delta_T**3
    
    elif material == 'sapphire_para':
        # Coefficient for sapphire parallel to c-axis
        alpha = 5.6e-6
        return alpha * delta_T
    
    elif material == 'sapphire_perp':
        # Coefficient for sapphire perpendicular to c-axis
        alpha = 5.0e-6
        return alpha * delta_T
    
    else:
        raise ValueError("Material not supported: {}".format(material))
def GET_theta_lambda_energy_latticeconstant_reference(results, temp_experimental, experiment_name):
    signe_delta, signe_gamma = 1, 1
    deg = np.pi / 180
    temp_rt=np.where(temp_experimental==temp_experimental.min())[0][0]
    # Handle temperature data
    if 'RT_Af_850' in str(temp_experimental[0]):
        temp_experimental = np.array([i.replace('RT_Af_850', "20") for i in temp_experimental]).astype(float)
    else:
        temp_experimental = temp_experimental.astype(float)

    com_y, com_z = results[:, 8].astype(float), results[:, 9].astype(float)
    gamma_scan, delta_scan = results[:, 3].astype(float), results[:, 4].astype(float)
    cchx, cchy = results[:, 12].astype(float), results[:, 13].astype(float)
    fact = results[:, 14].astype(float)

    com_y_ = com_y - cchx
    com_z_ = com_z - cchy

    delta = delta_scan - signe_delta * com_y_ / fact
    gamma = gamma_scan + signe_gamma * com_z_ / (fact * np.cos(delta_scan * deg))
    cos2o = np.cos(delta * deg) * np.cos(gamma * deg)
    theta = np.arccos(cos2o) * 0.5

    lambda_ = (2. * get_prediction_from_theo_sapphire_lattice(int(temp_experimental[temp_rt]), plot=False)[1] / 6.) * np.sin(theta[temp_rt])
    energy = lam2en(lambda_) / 1000

    C_ = 3. * lambda_ / np.sin(theta)

    print(f"\nResults for {experiment_name}:")
    print(f"theta: {theta/deg}°")
    print(f"Lambda: {lambda_}Å")
    print(f"Structure constant: {C_}Å")
    print(f"Energy: {12.4/lambda_} keV")

    return theta, lambda_, energy,C_
def process_scan_data_COM_CRISTAL_2022(
    path_master: str,
    path_results_code: str,
    path_results_saveplots: str,
    f_s: int = 20,
    vmin: int = 0,
    vmax: int = 10,
    check_a_scan: Optional[str] = None,check_a_particle: Optional[str] = None
) -> List[List[Union[str, float, int]]]:
    """
    Process scan data for Center of Mass (COM) analysis from CRISTAL beamline, collected in 2022.

    This function reads and processes scan data from the CRISTAL beamline, generates various plots,
    and returns processed information for each analyzed scan.

    Args:
        path_master (str): Path to the master directory containing the raw data files.
        path_results_code (str): Path to the directory containing analysis code and intermediate results.
        path_results_saveplots (str): Path to the directory where generated plots will be saved.
        f_s (int, optional): Font size for plot labels and titles. Defaults to 20.
        vmin (int, optional): Minimum value for color scale in plots. Defaults to 0.
        vmax (int, optional): Maximum value for color scale in plots. Defaults to 10.
        check_a_scan (Optional[str], optional): If provided, process only this specific scan. Defaults to None.

    Returns:
        List[List[Union[str, float, int]]]: A list of processed data for each analyzed scan. Each inner list contains:
            - Particle name (str)
            - Scan number (str)
            - Temperature (str)
            - Various angular positions (float): gamma, delta, omega
            - Center of Mass coordinates (int): x, y, z
            - Maximum intensity and total sum of data (float)
            - Other experimental parameters (float)

    Note:
        This function creates subdirectories in the path_results_saveplots directory for organizing output plots.
        It assumes specific file structures and naming conventions for the CRISTAL beamline data from 2022.
    """
    os.makedirs(path_results_saveplots, exist_ok=True)
    os.makedirs(os.path.join(path_results_saveplots, "original"), exist_ok=True)
    os.makedirs(os.path.join(path_results_saveplots, "cropped"), exist_ok=True)

    scan_list = glob.glob(path_master + '*/*.nxs')
    scan_list.sort()
    scan_list_det = np.array([str(int(i[-8:-4])) for i in scan_list])
    scan_list_det_datah5 = np.array(['exp_' + str(i[-8:-4]) for i in scan_list])
    file_name = glob.glob(path_results_code + '*.csv')[0]
    particles_scans = pd.read_csv(file_name, delimiter=";", encoding='unicode_escape')
    df = particles_scans.T
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    list_temp = np.array(particles_scans["Unnamed: 0"])

    particle_list_B12S1P1 = [i for i in particles_scans.keys()]
    particle_list_B12S1P1 = particle_list_B12S1P1[1:]
    list_scan_part_B12S1P1 = {}
    for i_part in particle_list_B12S1P1:
        list_s = str([i for i in list(particles_scans[i_part]) if str(i) != "-"])
        list_s = list_s.strip("[]").strip("'").replace("',", "-").replace("- '", "-").split("-")
        list_scan_part_B12S1P1 = {**list_scan_part_B12S1P1, i_part: np.array(list_s).astype(str)}

    list_scan_temp_B12S1P1 = {}
    for i_temp in list_temp:
        list_s = str([str(i) for i in list(df[i_temp]) if str(i) != "-"])
        list_s = list_s.strip("[]").strip("'").replace("',", "-").replace("- '", "-").split("-")
        list_scan_temp_B12S1P1 = {**list_scan_temp_B12S1P1, i_temp: np.array(list_s).astype(str)}

    mask = np.load(path_master + 'mask_hotpixels_nov2022_cristal_ter.npz')['data']

    res = []
    intensity_all=[]
    for i in range(0, len(scan_list_det)):
        if check_a_scan:
            if scan_list_det[i]  != check_a_scan:            
                continue
        filename = scan_list[i]
        check_part = [scan_list_det[i] in list_scan_part_B12S1P1[i_part] for i_part in list_scan_part_B12S1P1]
        try:
            scan_exist = True
            indice_part = np.where(check_part)[0][0]
        except:
            scan_exist = False
        if scan_exist:
            part_name = particle_list_B12S1P1[indice_part]
            check_scan_temp = [scan_list_det[i] in list_scan_temp_B12S1P1[i_temp] for i_temp in list_scan_temp_B12S1P1]
            indice_temp = np.where(check_scan_temp)[0][0]
            temp_scan = list_temp[indice_temp]
            if temp_scan in ['27 Af 950', '27 Af 800']:
                cch = np.array([113.3, 304.9])
                fact = 318.5
            else:
                cch = np.array([114.8, 300.6])
                fact = 330.8

            print(part_name, temp_scan, scan_list_det[i])
            f = h5py.File(filename)
            pref_key = scan_list_det_datah5[i] + "/scan_data/"
            key_path = [pref_key + i for i in h5py.File(filename)[pref_key].keys() if (len(h5py.File(filename)[pref_key][i].shape) == 3)][0]
            key_pos = scan_list_det_datah5[i] + '/CRISTAL/Diffractometer/'
            key_gamma = [key_pos + i for i in h5py.File(filename)[key_pos].keys() if ('gamma' in i)][0] + '/position'
            key_delta = [key_pos + i for i in h5py.File(filename)[key_pos].keys() if ('delta' in i)][0] + '/position'
            key_omega = [pref_key + i for i in h5py.File(filename)[pref_key].keys() if ('actuator_1_1' in i)][0]

            if scan_list_det[i] in ("668", "671", "703", "945"):
                hkl = [0, 0, 2]
            elif scan_list_det[i] in ("685", "959"):
                hkl = [0, 2, 0]
            else:
                hkl = [1, 1, 1]

            delta, gamma, omega = np.array(f[key_delta][()]).mean(), np.array(f[key_gamma][()]).mean(), np.array(f[key_omega][()])
            data_original = f[key_path][()]
            data_original = np.array([data_original[i] * (1 - mask) for i in range(len(data_original))])
            shape_data = np.array([i for i in data_original.shape])
            pic = data_original[int(shape_data[0] * 0.3):int(shape_data[0] * 0.7)]
            max_x, max_y, max_z = np.unravel_index(np.argmax(pic, axis=None), data_original.shape)
            max_x += int(shape_data[0] * 0.3)
            data_original_copy = np.array(data_original)
            if part_name == "A9":
                len__ = 80
                len__z = 120
            else:
                len__z = len__ = 80

            if scan_list_det[i] in np.array([188, 491, 495, 510, 526, 533, 544, 799, 703, 945, 959, 685, 668, 671]).astype(str):
                continue
            if part_name != "D7":
                if ((data_original_copy.max() == 118100) and (len(np.where(data_original_copy == data_original_copy.max())[0]) >= 15)):
                    continue
            data_original_copy[0:max_x - len__] = 0
            data_original_copy[:, 0:max_y - len__] = 0
            data_original_copy[:, :, 0:max_z - len__z] = 0
            data_original_copy[max_x + len__:] = 0
            data_original_copy[:, max_y + len__:] = 0
            data_original_copy[:, :, max_z + len__z:] = 0
            
            # Plotting
            fig_title = f"{part_name} S{scan_list_det[i]}"
            plot_3D_projections(data_original_copy[:, :, :], log_scale=True, fig_title=fig_title)
            plt.savefig(os.path.join(path_results_saveplots, f"{fig_title}_3D_projections.png"))
            plt.close()

            com_ = [round(i) for i in C_O_M(data_original_copy)]
            max_coords = get_max_coords(data_original_copy)
            data_cropped = crop_3darray_pos(data_original_copy, com_, (200, 200, 200))
            max_crop = get_max_coords(data_cropped)
            com_crop = [round(i) for i in C_O_M(data_cropped)]

            # Plot original data
            plot_summary_difraction(zero_to_nan(data_original),com_,max_coords,path_save=path_results_saveplots+"original/",
                                    fig_title=fig_title, fig_save_=f"{fig_title}_original.png", f_s=f_s, vmin=vmin, vmax=vmax)
            # Plot cropped data
            plot_summary_difraction(zero_to_nan(data_cropped),com_crop,max_crop,path_save=path_results_saveplots+"cropped/",
                                    fig_title=fig_title, fig_save_=f"{fig_title}_crop.png", f_s=20, vmin=0, vmax=10)

            ressss = [part_name, scan_list_det[i], temp_scan, gamma.mean(), delta.mean(),
                      omega[int(com_[0])], omega.mean(), com_[0], com_[1], com_[2], data_original.max(), data_original.sum(),
                      cch[0], cch[1], fact, hkl[0], hkl[1], hkl[2]]
            intensity_all.append(data_cropped)
            print(ressss)
            res.append(ressss)

    return res,intensity_all
def process_scan_data_COM_id1_avril_2023(
    path_master: str,
    path_results_code: str,
    path_results_saveplots: str,
    f_s: int = 20,
    vmin: int = 0,
    vmax: int = 10,
    len__: int = 30,
    check_a_scan: Optional[str] = None,
    check_a_particle: Optional[str] = None
) -> list:
    """
    Process scan data for COM (Center of Mass) analysis from ID1 beamline, collected in April 2023.

    This function reads scan data, processes it, generates various plots, and returns processed information.

    Args:
        path_master (str): Path to the master directory containing the raw data files.
        path_results_code (str): Path to the directory containing analysis code and intermediate results.
        path_results_saveplots (str): Path to the directory where generated plots will be saved.
        f_s (int, optional): Font size for plot labels and titles. Defaults to 20.
        vmin (int, optional): Minimum value for color scale in plots. Defaults to 0.
        vmax (int, optional): Maximum value for color scale in plots. Defaults to 10.
        len__ (int, optional): Length parameter for data cropping. Defaults to 30.

    Returns:
        list: A list of processed data for each analyzed scan, including particle information,
              scan parameters, and calculated metrics.

    Note:
        This function assumes specific file structures and naming conventions for the input data.
        It creates subdirectories in the path_results_saveplots directory for organizing output plots.
    """    
    os.makedirs(path_results_saveplots, exist_ok=True)
    os.makedirs(os.path.join(path_results_saveplots, "original"), exist_ok=True)
    os.makedirs(os.path.join(path_results_saveplots, "cropped"), exist_ok=True)

    data_location = [
        path_master + 'B12_S1P1_0001/B12-S1P1_0001.h5',
        path_master + 'B12_S1P1_BCDI/B12-S1P1_BCDI.h5',
        path_master + 'B12-S1P1_B12_B1P1_BCDI_1_0002/B12-S1P1_B12_B1P1_BCDI_1_0002.h5',
        path_master + 'B12_S1P1_B12_B1P1_BCDI_3/B12-S1P1_B12_B1P1_BCDI_3.h5'
    ]
    dataset_name = ['B12-S1P1_0001.h5', 'B12-S1P1_BCDI.h5', 'B12-S1P1_B12_B1P1_BCDI_1_0002.h5', 'B12-S1P1_B12_B1P1_BCDI_3.h5']
    
    scan_dict_all = []
    for i_sample, location in enumerate(data_location):
        print('********************************************************************')
        print('DataSet : ' + dataset_name[i_sample][:-3])
        f = h5py.File(location)
        all_scans_not_filtrerd = np.array([i for i in dict(f).keys()])
        all_scans_not_filtrerd_INT = np.array([i[:-2] for i in dict(f).keys()])
        all_scans_not_filtrerd_folders = np.array([f'{int(item):04d}' for item in all_scans_not_filtrerd_INT])
        scans_noteta = np.array([f'{int(i_scan[:-2]):04d}' for i_scan in all_scans_not_filtrerd if 'eta' not in f[i_scan]['title'][()].decode()])
        scans_eta = np.array([i_scan for i_scan in all_scans_not_filtrerd if ('eta' in f[i_scan]['title'][()].decode())])
        scans_eta_1d_scan_nmpoints = np.array([i_scan[:-2] for i_scan in scans_eta if ((float(f[i_scan]['title'][()].decode()[-2:]) >= 1.) and (float(f[i_scan]['title'][()].decode()[-5:-2]) >= 200))])
        scans_eta_1d_scan_nmpoints = scans_eta_1d_scan_nmpoints.astype(int)
        scans_eta_1d_scan_nmpoints.sort()
        scans_eta_1d_scan_nmpoints = scans_eta_1d_scan_nmpoints.astype(str)
        print('Scans on eta: ', scans_eta_1d_scan_nmpoints)
        scan_dict_all.append(scans_eta_1d_scan_nmpoints)

    file_name = glob.glob(path_results_code + '*.csv')[0]
    particles_scans = pd.read_csv(file_name, delimiter=";", encoding='unicode_escape')
    df = particles_scans.T
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    df = df.drop(df.index[0])
    list_temp = np.array(particles_scans["Unnamed: 0"])
    data_et_csv = [i.split("/") for i in np.array(particles_scans['data_set'])]

    particle_list_B12S1P1 = [i for i in particles_scans.keys()][2:]
    list_scan_part_B12S1P1 = {}
    for i_part in particle_list_B12S1P1:
        list_s = str([i for i in list(particles_scans[i_part]) if str(i) != "-"])
        list_s = list_s.strip("[]").strip("'").replace("',", "-").replace("- '", "-").split("-")
        list_scan_part_B12S1P1[i_part] = np.array(list_s).astype(str)

    temp_list_B12S1P1 = [i for i in df.keys()]
    list_scan_temp_B12S1P1 = {}
    for i_temp in list_temp:
        list_s = str([str(i) for i in list(df[i_temp]) if str(i) != "-"])
        list_s = list_s.strip("[]").strip("'").replace("',", "-").replace("- '", "-").split("-")
        list_scan_temp_B12S1P1[i_temp] = np.array(list_s).astype(str)

    res, scan_error_list = [], []
    intensity_all=[]
    for i_dataset in range(len(dataset_name)):
        data_set_sel = dataset_name[i_dataset][:-3]
        check_dataset = [data_set_sel in i_part for i_part in data_et_csv]
        indice_dataset = list(np.where(check_dataset)[0])
        temp_scanlist_sel = [list_scan_temp_B12S1P1[temp_list_B12S1P1[i]] for i in indice_dataset]
        temp_sel = [temp_list_B12S1P1[i] for i in indice_dataset]
        f = h5py.File(data_location[i_dataset])
        for i in range(len(scan_dict_all[i_dataset])):
            filename = scan_dict_all[i_dataset][i]
            check_temp = [filename in i_temp for i_temp in temp_scanlist_sel]
            try:
                scan_exist = True
                indice_temp = np.where(check_temp)[0][0]
            except:
                scan_exist = False
            if scan_exist:
                temp_scan = temp_sel[indice_temp]
                if filename == '77':
                    part_name = "C7" if temp_scan == "35" else "E8"
                elif filename == '136':
                    part_name = "A7" if temp_scan == "800" else "A9"
                elif filename == '174':
                    part_name = "D7" if temp_scan == "800" else "A9"
                else:
                    check_part = [filename in list_scan_part_B12S1P1[i_part] for i_part in list_scan_part_B12S1P1]
                    indice_part = np.where(check_part)[0][0]
                    part_name = particle_list_B12S1P1[indice_part]
                if check_a_scan:
                    if filename  != check_a_scan:            
                        continue
                if check_a_particle:
                    if part_name  != check_a_particle:            
                        continue
                                
                cch = np.array([169.047736, 185.517609])
                fact = 1.119673 * np.pi / 180 / 55 * 10**6

                pref_key = filename + '.1/measurement/'
                key_path = [pref_key + i for i in f[pref_key].keys() if (len(f[pref_key][i].shape) == 3)][0]
                key_pos = filename + '.1/instrument/positioners/'
                key_nu = [key_pos + i for i in f[key_pos].keys() if ('nu' in i)][0]
                key_delta = [key_pos + i for i in f[key_pos].keys() if ('delta' in i)][0]
                key_eta = [pref_key + i for i in f[pref_key].keys() if ('eta' in i)][0]

                delta, nu, eta = np.array(f[key_delta][()]).mean(), np.array(f[key_nu][()]).mean(), np.array(f[key_eta][()])

                data_original = f[key_path][()]
                shape_data = np.array([i for i in data_original.shape])
                pic = data_original[int(shape_data[0] * 0.3):int(shape_data[0] * 0.7)]
                max_x, max_y, max_z = np.unravel_index(np.argmax(pic, axis=None), data_original.shape)
                max_x += int(shape_data[0] * 0.3)
                data_original_copy = np.array(data_original)
                

                data_original_copy[:max_x - len__, :, :] = 0
                data_original_copy[:, :max_y - len__, :] = 0
                data_original_copy[:, :, :max_z - len__] = 0
                data_original_copy[max_x + len__:, :, :] = 0
                data_original_copy[:, max_y + len__:, :] = 0
                data_original_copy[:, :, max_z + len__:] = 0

                # Plotting section
                fig_title = f"{part_name} S{filename}"
                
                # Plot 3D projections
                plot_3D_projections(data_original_copy[:, :, :], log_scale=True, fig_title=fig_title)
                plt.savefig(os.path.join(path_results_saveplots, f"{fig_title}_3D_projections.png"))
                plt.close()

                # Get max coordinates and center of mass
                max_coords = get_max_coords(data_original_copy)
                com_ = [round(i) for i in C_O_M(data_original_copy)]

                # Crop data
                data_cropped = crop_3darray_pos(data_original_copy, com_, (200, 200, 200))
                max_crop = get_max_coords(data_cropped)
                com_crop = [round(i) for i in C_O_M(data_cropped)]

                # Plot original data
                plot_summary_difraction(zero_to_nan(data_original), com_, max_coords,
                                        path_save=path_results_saveplots+"original/",
                                        fig_title=fig_title, fig_save_=f"{fig_title}_original.png",
                                        f_s=f_s, vmin=vmin, vmax=vmax)

                # Plot cropped data
                plot_summary_difraction(zero_to_nan(data_cropped), com_crop, max_crop,
                                        path_save=path_results_saveplots+"cropped/",
                                        fig_title=fig_title, fig_save_=f"{fig_title}_crop.png",
                                        f_s=f_s, vmin=vmin, vmax=vmax)

                ressss = [part_name, filename, temp_scan, nu.mean(), delta.mean(), eta[int(com_[0])], eta.mean(),
                          com_[0], com_[1], com_[2], data_original.max(), data_original.sum(), cch[0], cch[1], fact]

                print(ressss)
                res.append(ressss)
                intensity_all.append(data_cropped)
                

    return res,intensity_all
def process_bcdi_data_B12S1P1_id1_Jan_2024(
    path_master: str,
    path_results_code: str,
    path_results_saveplots: str,
    f_s: int = 20,
    vmin: int = 0,
    vmax: int = 10,
    len__: int = 30,
    check_a_scan: Optional[str] = None,
    check_a_particle: Optional[str] = None
) -> list:
    """
    Process BCDI data for B12-S1P1 samples collected in January 2024.

    Args:
        path_master (str): Path to the master directory containing the data files.
        path_results_code (str): Path to the directory containing the CSV files with particle information.
        path_results_saveplots (str): Path to the directory where the plots will be saved.
        f_s (int, optional): Font size for plot titles. Defaults to 20.
        vmin (int, optional): Minimum value for plot color scale. Defaults to 0.
        vmax (int, optional): Maximum value for plot color scale. Defaults to 10.
        len__ (int, optional): Length for cropping the data. Defaults to 30.

    Returns:
        list: A list containing the processed data for each scan.
    """
    os.makedirs(path_results_saveplots, exist_ok=True)
    os.makedirs(os.path.join(path_results_saveplots, "original"), exist_ok=True)
    os.makedirs(os.path.join(path_results_saveplots, "cropped"), exist_ok=True)

    data_location = [
        os.path.join(path_master, 'P1_0001/P1_0001.h5'),
        os.path.join(path_master, 'TS_0001/TS_0001.h5')
    ]
    dataset_name = data_location

    # Process TS data
    file_name = glob.glob(os.path.join(path_results_code, '*TS.csv'))[0]
    particles_scans_TS = pd.read_csv(file_name, delimiter=";", encoding='unicode_escape')
    particle_list_TS = [i for i in particles_scans_TS.keys()]
    list_scan_part_TS = {}
    for i_part in particle_list_TS:
        list_s = str([i for i in list(particles_scans_TS[i_part]) if str(i) != "nan"])
        list_s = list_s.strip("[]").strip("'").replace("',", "-").replace("- '", "-").split("-")
        list_scan_part_TS[i_part] = np.array(list_s).astype(str)

    # Process P1 data
    file_name = glob.glob(os.path.join(path_results_code, '*P1.csv'))[0]
    particles_scans_P1 = pd.read_csv(file_name, delimiter=";", encoding='unicode_escape')
    particle_list_P1 = [i for i in particles_scans_P1.keys()]
    list_scan_part_P1 = {}
    for i_part in particle_list_P1:
        list_s = str([i for i in list(particles_scans_P1[i_part]) if str(i) != "nan"])
        list_s = list_s.strip("[]").strip("'").replace("',", "-").replace("- '", "-").split("-")
        list_scan_part_P1[i_part] = np.array(list_s).astype(str)

    i_sample = 0
    scan_list = [i for i in glob.glob(dataset_name[i_sample][:dataset_name[i_sample].find("0001/") + 5] + "sca*")]
    scan_list.sort()
    scan_list_fol = np.array([i[-4:] for i in scan_list])
    scan_list_det = np.array([str(int(i[-4:])) for i in scan_list])
    scan_list_det_datah5 = np.array([str(int(i[-4:])) + '.1' for i in scan_list])

    print('Working on data : ', data_location[i_sample])
    f = h5py.File(data_location[i_sample])
    res, scan_error_list = [], []
    intensity_all=[]
    for i in range(0, len(scan_list_det)):
        filename = scan_list_det[i]
        check_part = [scan_list_det[i] in list_scan_part_P1[i_part] for i_part in list_scan_part_P1]
        try:
            scan_exist = True
            indice_part = np.where(check_part)[0][0]
        except:
            scan_exist = False
        
        if scan_exist:
            part_name = particle_list_P1[indice_part]
            temp_scan = '27°C'
            if check_a_scan:
                if filename  != check_a_scan:            
                    continue
            if check_a_particle:
                if part_name  != check_a_particle:            
                    continue  
            print(part_name,file_name,temp_scan)
            key_pos = scan_list_det_datah5[i] + '/instrument/positioners/'
            delta, eta, nu = f[key_pos + 'delta'][()], f[key_pos + 'eta'][()], f[key_pos + 'nu'][()]
            print(f[scan_list_det_datah5[i]]['title'][()].decode())
            key_path = scan_list_det_datah5[i] + '/measurement/mpx1x4'
            data_original = f[key_path][()]
            shape_data = np.array([i for i in data_original.shape])
            pic = data_original[int(shape_data[0] * 0.3):int(shape_data[0] * 0.7)]
            max_x, max_y, max_z = np.unravel_index(np.argmax(pic, axis=None), data_original.shape)
            max_x += int(shape_data[0] * 0.3)
            data_original_copy = np.array(data_original)
            data_original_copy[0:max_x - len__] = 0
            data_original_copy[:, 0:max_y - len__] = 0
            data_original_copy[:, :, 0:max_z - len__] = 0
            data_original_copy[max_x + len__:] = 0
            data_original_copy[:, max_y + len__:] = 0
            data_original_copy[:, :, max_z + len__:] = 0

            # Plotting section
            fig_title = f"{part_name} S{filename}"
            
            # Plot 3D projections
            plot_3D_projections(data_original_copy[:, :, :], log_scale=True, fig_title=fig_title)
            plt.savefig(os.path.join(path_results_saveplots, f"{fig_title}_3D_projections.png"))
            plt.close()

            # Get max coordinates and center of mass
            max_coords = get_max_coords(data_original_copy)
            com_ = [round(i) for i in C_O_M(data_original_copy)]

            # Crop data
            data_cropped = crop_3darray_pos(data_original_copy, com_, (200, 200, 200))
            max_crop = get_max_coords(data_cropped)
            com_crop = [round(i) for i in C_O_M(data_cropped)]

            # Plot original data
            plot_summary_difraction(zero_to_nan(data_original), com_, max_coords,
                                    path_save=path_results_saveplots+"original/",
                                    fig_title=fig_title, fig_save_=f"{fig_title}_original.png",
                                    f_s=f_s, vmin=vmin, vmax=vmax)

            # Plot cropped data
            plot_summary_difraction(zero_to_nan(data_cropped), com_crop, max_crop,
                                    path_save=path_results_saveplots+"cropped/",
                                    fig_title=fig_title, fig_save_=f"{fig_title}_crop.png",
                                    f_s=f_s, vmin=vmin, vmax=vmax)

            ressss = [part_name, filename, temp_scan, nu.mean(), delta.mean(),
                      eta[int(com_[0])], eta.mean(), com_[0], com_[1], com_[2],
                      data_original.max(), data_original.sum(), 363.908464, 144.5]

            print(ressss)
            intensity_all.append(data_cropped)
            res.append(ressss)

    f.close()
    return res,intensity_all
def process_scan_data_COM_id1_june_2024(
    path_master: str,
    path_results_code: str,
    path_results_saveplots: str,
    f_s: int = 20,
    vmin: int = 0,
    vmax: int = 10,
    len__: int = 100,
    check_a_scan: Optional[str] = None,
    check_a_particle: Optional[str] = None
) -> Tuple[List[List[Union[str, int, float]]], List[Tuple[str, str, str, str]]]:
    """
    Process scan data for Center of Mass (COM) analysis from ID1 beamline, collected in June 2024.

    This function sets up the environment, reads the master file, processes each scan, 
    generates plots, and returns processed data and errors.

    Args:
        base_path (str): Base path for all directories.
        f_s (int, optional): Font size for plot labels and titles. Defaults to 20.
        vmin (int, optional): Minimum value for color scale in plots. Defaults to 0.
        vmax (int, optional): Maximum value for color scale in plots. Defaults to 10.
        len__ (int, optional): Length for data cropping. Defaults to 30.

    Returns:
        Tuple[List[List[Union[str, int, float]]], List[Tuple[str, str, str, str]]]: 
            - A list of processed data for each analyzed scan.
            - A list of errors encountered during processing.
    """
    def process_single_scan(f_scan, part_name, scan, temp, path_results_saveplots, f_s, vmin, vmax, len__):
        """Helper function to process a single scan."""
        pref_key = 'measurement/mpx1x4'
        key_pos = 'instrument/positioners/'
        
        # Extract position data
        delta, nu, eta = [np.array(f_scan[f"{key_pos}{pos}"][()]) for pos in ["delta", "nu", "eta"]]
        thx, thy, thz = [np.array(f_scan[f"{key_pos}{pos}"][()]).mean() for pos in ["thx", "thy", "thz"]]
        pix, piy, piz = [np.array(f_scan[f"{key_pos}{pos}"][()]).mean() for pos in ["pix", "piy", "piz"]]
        
        pos_x, pos_y, pos_z = thx - pix/1000, thy - piy/1000, thz - piz/1000
        fact = 1.079 * np.pi / 180 / 55 * 10**6
        cch = np.array([431.4, 191.41])
    
        data_original = f_scan[pref_key][()]
        data_copy = process_data(data_original, part_name, scan, len__)
    
        max_coords = get_max_coords(data_copy)
        com = [int(round(i)) for i in C_O_M(data_copy)]
        data_cropped = crop_3darray_pos(data_copy, com, (200,200,200))
        max_crop = get_max_coords(data_cropped)
        com_crop = [round(i) for i in C_O_M(data_cropped)]
        
        fig_title = f"{part_name} | {temp} °C | S{scan}"
        fig_save_ = f"{part_name}_{temp}_S{scan}.png"
        
        plot_summary_difraction(data_original, com, max_coords, 
                                path_save=path_results_saveplots+"original/",
                                fig_title=fig_title, fig_save_=fig_save_, f_s=f_s, vmin=vmin, vmax=vmax)
        plot_summary_difraction(data_cropped, com_crop, max_crop, 
                                path_save=path_results_saveplots+"cropped/",
                                fig_title=fig_title, fig_save_=fig_save_, f_s=f_s, vmin=vmin, vmax=vmax)
    
        return [part_name, scan, temp, nu.mean(), delta.mean(), eta[int(com[0])], eta.mean(), *com, 
                data_original.max(), data_original.sum(), *cch, fact, pos_x, pos_y, pos_z],data_cropped
    def should_skip_scan(part_name, scan):
        """Helper function to determine if a scan should be skipped."""
        skip_conditions = [
            part_name in ("II-D7", "II-B8","II-D9"),
            scan == '53' and part_name == "B12S1P1_0002",
            part_name == "II-D8" and scan in ("11","14","21","104","228","159"),
            part_name == "II-E8" and scan in ("57","75",'151','154','186',"222",'265',"415","424","428","438"),
            part_name == "II-G10" and scan in ("68",'80',"143", "144", "173", "216"),
            part_name == "VII-A6" and scan in ('93','94','97',"99"),
            part_name == "VII-A7" and scan in ("132","296","323","321"),
            part_name == "VII-A9" and scan in ("59","100","139")
        ]
        return any(skip_conditions)
    def process_data(data_original, part_name, scan, len__):
        
        """Helper function to process the data based on part name and scan number."""
        shape_data = np.array(data_original.shape)

        data_original_copy=np.array(data_original)
        if ((part_name=="II-D8") and ( int(scan)== 11)) :
                data_original_copy[:,:,:370]=0
                masked_data, data_original_copy, dilated_mask = mask_clusters(data_original_copy,threshold_factor=0.3,dilation_iterations=10,return_second_cluster=True)
                data_original_copy[:,:,:370]=data_original[:,:,:370]
                pic=data_original_copy
                max_x,max_y,max_z=np.unravel_index(np.argmax(pic,axis=None), pic.shape)
                max_=[max_x,max_y,max_z]
    
    
        if ((part_name=="II-D8") and ( int(scan)>= 87)) or ((part_name=="II-G10") and ( int(scan)>68)) :#or ((part_name=="VII-A7") and ( int(scan)<= 132)):
                masked_data, data_original_copy, dilated_mask = mask_clusters(data_original,threshold_factor=0.3,dilation_iterations=50,return_second_cluster=False)
                pic=data_original_copy
                max_x,max_y,max_z=np.unravel_index(np.argmax(pic,axis=None), pic.shape)
                max_=[max_x,max_y,max_z]
        elif ((part_name=="II-G10") and ( int(scan)== 67))  or ((part_name=="VII-A7") and ( int(scan)== 88)):
            masked_data, data_original_copy, dilated_mask = mask_clusters(data_original,threshold_factor=0.3,dilation_iterations=50,return_second_cluster=True)
            pic=data_original_copy
            max_x,max_y,max_z=np.unravel_index(np.argmax(pic,axis=None), pic.shape)
            max_=[max_x,max_y,max_z]
        elif ((part_name=="VII-A7") and ( int(scan)>= 254)) and (( int(scan)<321 )):
            print(scan)
            masked_data, data_original_copy, dilated_mask = mask_clusters(data_original,threshold_factor=0.6,dilation_iterations=10,return_second_cluster=False)
            pic=data_original_copy
            max_x,max_y,max_z=np.unravel_index(np.argmax(pic,axis=None), pic.shape)
            max_=[max_x,max_y,max_z]
        elif ((part_name=="VII-A7") and ( int(scan)>= 321)) :
            print(scan)
            masked_data, data_original_copy, dilated_mask = mask_clusters(data_original,threshold_factor=0.1,dilation_iterations=100,return_second_cluster=False)
            pic=data_original_copy
            max_x,max_y,max_z=np.unravel_index(np.argmax(pic,axis=None), pic.shape)
            max_=[max_x,max_y,max_z]
        elif (((part_name=="VII-A9") and ( int(scan)>= 139)) and ( int(scan)<258)) :
            print(scan)
            masked_data, data_original_copy, dilated_mask = mask_clusters(data_original,threshold_factor=0.6,dilation_iterations=8,return_second_cluster=False)
            pic=data_original_copy
            max_x,max_y,max_z=np.unravel_index(np.argmax(pic,axis=None), pic.shape)
            max_=[max_x,max_y,max_z]
        elif ((part_name=="VII-A9") and ( int(scan)== 59)) or ((part_name=="VII-A9") and ( int(scan)>= 100) and ( int(scan)< 139)) or ((part_name=="VII-A9") and ( int(scan)==333)) :
            masked_data, data_original_copy, dilated_mask = mask_clusters(data_original,threshold_factor=0.1,dilation_iterations=20,return_second_cluster=False)
            pic=data_original_copy
            max_x,max_y,max_z=np.unravel_index(np.argmax(pic,axis=None), pic.shape)
            max_=[max_x,max_y,max_z]
        elif ((part_name=="VII-A9") and ( int(scan)== 79))  or ((part_name=="VII-A9") and ( int(scan)== 404)) :
            masked_data, data_original_copy, dilated_mask = mask_clusters(data_original,threshold_factor=0.1,dilation_iterations=20,return_second_cluster=True)
            pic=data_original_copy
            max_x,max_y,max_z=np.unravel_index(np.argmax(pic,axis=None), pic.shape)
            max_=[max_x,max_y,max_z]
        else:
            pic=data_original_copy[int(shape_data[0]*0.3):int(shape_data[0]*0.7)]
            
            max_x,max_y,max_z=np.unravel_index(np.argmax(pic,axis=None), pic.shape)
            max_x+=int(shape_data[0]*0.3)
            max_=[max_x,max_y,max_z]
            half_w=80
            data_original_copy[:max_x-half_w, :, :] = 0 ;    data_original_copy[:, :max_y-half_w, :] = 0 ;
            data_original_copy[:, :, :max_z-half_w] = 0 ;    data_original_copy[max_x+half_w:, :, :] = 0 ; 
            data_original_copy[:, max_y+half_w:, :] = 0 ;    data_original_copy[:, :, max_z+half_w:] = 0 ;3
            

        return data_original_copy
    def apply_default_mask(data_original, shape_data, len__):
        """Apply default masking to the data."""
        data_copy = data_original[int(shape_data[0]*0.3):int(shape_data[0]*0.7)]
        max_x, max_y, max_z = get_max_coords(data_copy)
        max_x += int(shape_data[0]*0.3)
        data_copy = np.array(data_original)
        data_copy[:max_x-len__, :, :] = 0
        data_copy[:, :max_y-len__, :] = 0
        data_copy[:, :, :max_z-len__] = 0
        data_copy[max_x+len__:, :, :] = 0
        data_copy[:, max_y+len__:, :] = 0
        data_copy[:, :, max_z+len__:] = 0
        return data_copy
    # Setup paths
    master_file = os.path.join(path_master, 'ihhc4033_id01.h5')
    # Create directories
    os.makedirs(path_results_saveplots, exist_ok=True)
    os.makedirs(os.path.join(path_results_saveplots, "original"), exist_ok=True)
    os.makedirs(os.path.join(path_results_saveplots, "cropped"), exist_ok=True)

    # Setup data
    datasets = ['B12S1P1_0002', "B12S1P1_SXDM_0051", 'B12S1P1_BCDI']
    list_scans__ = np.array([3, 11, 14, 21, 22, 38, 46, 53, 57, 63, 67, 79, 88, 93, 94, 97, 98, 99, 127, 35,
                             46, 50, 53, 59, 75, 80, 81, 87, 90, 94, 97, 100, 101, 104,
                             113, 125, 132, 139, 143, 144, 151, 154, 159, 163, 169, 173, 175, 186, 187,
                             199, 203, 206, 215, 216, 222, 223, 228, 229, 233,
                             244, 254, 258, 263, 264, 265, 270, 275, 281, 296,
                             308, 311, 321, 323, 328, 333, 350, 354,
                             400, 404, 408, 415, 424, 428, 434, 438, 444])
    
    list_scans_part = np.array(['II-B8', 'II-D8', 'II-D8', 'II-D8', 'II-D8', 'II-D7', 'II-D7', 'II-D9', 'II-D9', 'II-E8', 'II-G10', 'VII-A9', 'VII-A7', 'VII-A6', 'VII-A6', 'VII-A6', 'VII-A6', 'VII-A6', 'II-E8', 'VII-A6',
                                'VII-A6', 'VII-A7', 'VII-A7', 'VII-A9', 'II-E8', 'II-G10', 'II-G10', 'II-D8', 'VII-A6', 'VII-A7', 'II-E8', 'VII-A9', 'VII-A9', 'II-D8',
                                'VII-A6', 'VII-A7', 'VII-A7', 'VII-A9', 'II-G10', 'II-G10', 'II-E8', 'II-E8', 'II-D8', 'II-D8', 'VII-A9', 'II-G10', 'II-G10', 'II-E8', 'II-E8',
                                'VII-A6', 'VII-A7', 'VII-A9', 'II-G10', 'II-G10', 'II-E8', 'II-E8', 'II-D8', 'II-D8', 'II-B8',
                                'VII-A6', 'VII-A7', 'VII-A9', 'II-G10', 'II-G10', 'II-E8', 'II-E8', 'II-D8', 'II-B8', 'VII-A7',
                                'VII-A6', 'VII-A6', 'VII-A7', 'VII-A7', 'VII-A7', 'VII-A9', 'II-G10', 'II-D8',
                                'VII-A7', 'VII-A9', 'II-G10', 'II-E8', 'II-E8', 'II-E8', 'II-D8', 'II-E8', 'II-E8'])
    
    list_scans_temp = np.array([23] * 20 + [100] * 14 + [200] * 15 + [300] * 10 + [400] * 10 + [500] * 8 + [700] * 9)

    data_set_scan_key_name_list = [f"{datasets[0]}_{scan}.1" if i < 18 else
                                   f"{datasets[1]}_{scan}.1" if i == 18 else
                                   f"{datasets[2]}_{scan}.1"
                                   for i, scan in enumerate(list_scans__)]

    res, scan_error_list = [], []
    intensity_all=[]
    with h5py.File(master_file, 'r') as f:
        for i, key in enumerate(data_set_scan_key_name_list):
            scan = str(list_scans__[i])
            temp = str(list_scans_temp[i])
            part_name = str(list_scans_part[i])
            if check_a_scan:
                if scan  != check_a_scan:            
                    continue
            if check_a_particle:
                if part_name  != check_a_particle:            
                    continue  
            if should_skip_scan(part_name, scan):
                continue

            print(f"scan : {scan} | temperature : {temp} | particle: {part_name}")            
            
            try:
                result,data_cropped = process_single_scan(f[key], part_name, int(scan), temp, 
                                             path_results_saveplots, f_s, vmin, vmax, len__)
                res.append(result)
                intensity_all.append(data_cropped)
                print(result)
            except Exception as e:
                print(f"Error processing scan {scan}: {str(e)}")
                scan_error_list.append((part_name, scan, temp, str(e)))
    
    return res, intensity_all
def adjust_temp_scan(T_theo_predicted, temp_scan_float, temp_scan):
    """
    Adjust the temp_scan array based on T_theo_predicted and temp_scan_float.
    
    Parameters:
    T_theo_predicted (array): Array of predicted temperature values
    temp_scan_float (array): Array of float temperature values
    temp_scan (array): Array of string temperature values to be adjusted
    
    Returns:
    array: Adjusted temp_scan array
    """
    value_to_modify = np.unique(temp_scan_float).astype(int)
    value_to_replaceby = [T_theo_predicted[np.where(temp_scan_float == i)[0][0]] for i in value_to_modify]
    
    temp_scan_new = np.array(temp_scan)
    
    for old_val, new_val in zip(value_to_modify, value_to_replaceby):
        print(f"Replacing {old_val} with {new_val}")
        temp_scan_new = np.array([elem.replace(str(old_val), str(new_val)) for elem in temp_scan_new])
        temp_scan_new = np.array([elem.replace(f"{old_val} RT", f"{new_val} RT") for elem in temp_scan_new])
    
    return temp_scan_new
def fit_temperature_model(T_exp, Ttheo, degree=1, debug_plot=None, plot_range=None):
    """
    Fit a polynomial model to temperature data and optionally create a debug plot.
    
    Parameters:
    T_exp (array-like): Experimental temperature values
    Ttheo (array-like): Theoretical temperature values
    degree (int): Degree of the polynomial to fit (default is 1 for linear fit)
    debug_plot (bool): If True, create and show a debug plot
    plot_range (tuple): (min, max) values for plotting the fitted curve. If None, uses data range.
    
    Returns:
    dict: A dictionary containing the polynomial function, coefficients, unique T_exp with mean Ttheo
    """
    T_exp = np.array(T_exp)
    Ttheo = np.array(Ttheo)
    
    # Calculate means for each unique T_exp
    unique_T_exp = np.unique(T_exp)
    mean_T_theo = []

    for temp in unique_T_exp:
        indices = np.where(T_exp == temp)[0]
        mean_T_theo.append(np.mean(Ttheo[indices]))

    mean_T_theo = np.array(mean_T_theo)

    # Fit a polynomial
    coeffs = np.polyfit(unique_T_exp, mean_T_theo, degree)

    # Create a polynomial function
    poly_func = np.poly1d(coeffs)

    if debug_plot:
        plt.figure(figsize=(10, 6))
        plt.scatter(T_exp, Ttheo, alpha=0.5, label='Original Data')
        plt.scatter(unique_T_exp, mean_T_theo, color='red', s=100, label='Mean Values')
        
        # Generate points for smooth curve
        if plot_range is None:
            T_exp_smooth = np.linspace(T_exp.min(), T_exp.max(), 200)
        else:
            T_exp_smooth = np.linspace(plot_range[0], plot_range[1], 200)
        
        plt.plot(T_exp_smooth, poly_func(T_exp_smooth), 'g-', label=f'Polynomial Fit (degree {degree})')
        
        plt.xlabel('T_exp')
        plt.ylabel('T_theo')
        plt.title('Temperature Model Fit')
        plt.legend()
        plt.grid(True)
        plt.savefig(debug_plot+"_calibration_temp.png")
        plt.show()

    return {
        "poly_func": poly_func,
        "coefficients": coeffs,
        "unique_T_exp": unique_T_exp,
        "mean_T_theo": mean_T_theo
    }
def apply_mask(data, threshold, iterations, second_cluster=False):
    masked_data, data_copy, _ = mask_clusters(data, threshold_factor=threshold, 
                                              dilation_iterations=iterations, 
                                              return_second_cluster=second_cluster)
    return data_copy
def get_max_coords(data):
    return np.unravel_index(np.argmax(data, axis=None), data.shape)
def optimize_energy_value_and_calibrate_temperature(original_energy, results_sapphire, Temp_experimentale, a_at_20TC=12.991, 
                    energy_range=2, energy_steps=100, signe_delta=-1, signe_gamma=-1,
                    minimize_method='sum', specific_index=None, plot=False, save_plot=None):
    
    """
    Optimize the energy to minimize the temperature difference (delta_temp).
    
    Parameters:
    - original_energy: The initial energy value
    - results_sapphire: The sapphire results array
    - Temp_experimentale: Experimental temperature data
    - a_at_20TC: Lattice parameter at 20°C
    - energy_range: Range around original energy to search (default 2)
    - energy_steps: Number of steps in the energy range (default 100)
    - signe_delta: Sign for delta (default -1)
    - signe_gamma: Sign for gamma (default -1)
    - minimize_method: 'sum' to minimize sum of delta_temp, 'index' to minimize specific index (default 'sum')
    - specific_index: Index to minimize if minimize_method is 'index' (default None)
    - plot: Boolean, whether to plot the results (default False)
    - save_plot: String, filename to save the plot (default None)
    
    Returns:
    - best_energy: The energy that minimizes delta_temp
    - min_metric: The minimum metric achieved (sum or specific index of delta_temp)
    - best_delta_temp: The delta_temp array for the best energy
    - best_T_exp: Experimental temperature data for the best energy
    - best_T_theo: Theoretical temperature predictions for the best energy
    """
    
    energy_values = np.linspace(original_energy - energy_range, 
                                original_energy + energy_range, 
                                energy_steps)
    
    metrics = []
    all_delta_temps = []
    all_T_exps = []
    all_T_theos = []

    for energy in energy_values:
        try:
            # Recalculate d_spacing with the new energy
            new_d_spacing = get_lattice_parametre(
                energy,
                np.array(results_sapphire[:,3]).astype(float),
                np.array(results_sapphire[:,4]).astype(float),
                [results_sapphire[:,12].astype(float), results_sapphire[:,13].astype(float)],
                np.array(results_sapphire[:,8]).astype(float),
                np.array(results_sapphire[:,9]).astype(float),
                scale=results_sapphire[:,14].astype(float),
                s_g=signe_delta, s_d=signe_gamma, hkl=[0, 0, 6]
            )
            
            # Check if new_d_spacing and Temp_experimentale have the same length
            if len(new_d_spacing) != len(Temp_experimentale):
                print(f"Mismatch in lengths: new_d_spacing ({len(new_d_spacing)}) and Temp_experimentale ({len(Temp_experimentale)})")
                continue

            # Define a function for thermal expansion
            def thermal_expansion(T, a0, alpha):
                return a0 * (1 + alpha * (T - 20))  # 20°C is the reference temperature

            # Fit the data
            params, _ = curve_fit(thermal_expansion, Temp_experimentale, new_d_spacing, p0=[a_at_20TC, 1e-6])

            # Calculate theoretical predictions
            T_exp_predict = (new_d_spacing / params[0] - 1) / params[1] + 20
            
            # Calculate delta_temp
            delta_temp = T_exp_predict - Temp_experimentale
            
            # Calculate metric based on chosen method
            if minimize_method == 'sum':
                metric = np.sum(np.abs(delta_temp))
            elif minimize_method == 'index':
                if specific_index is None or specific_index >= len(delta_temp):
                    raise ValueError("Invalid specific_index for minimize_method 'index'")
                metric = np.abs(delta_temp[specific_index])
            else:
                raise ValueError("Invalid minimize_method. Choose 'sum' or 'index'.")
            
            metrics.append(metric)
            all_delta_temps.append(delta_temp)
            all_T_exps.append(Temp_experimentale)
            all_T_theos.append(T_exp_predict)

        except Exception as e:
            print(f"Error occurred for energy {energy:.3f}: {str(e)}")
            metrics.append(np.inf)
            all_delta_temps.append(None)
            all_T_exps.append(None)
            all_T_theos.append(None)

    # Find the index of the minimum metric
    min_index = np.argmin(metrics)
    best_energy = energy_values[min_index]
    min_metric = metrics[min_index]
    best_delta_temp = all_delta_temps[min_index]
    best_T_exp = all_T_exps[min_index]
    best_T_theo = all_T_theos[min_index]

    if plot:
        # Create a new figure
        plt.figure(figsize=(12, 10))

        # First subplot: Lattice Parameter (a) Values
        plt.subplot(3, 1, 1)
        plt.plot(best_T_exp, new_d_spacing, 'o', label='Experimental data')
        plt.plot(best_T_exp, thermal_expansion(best_T_exp, *params), '-', label='Fitted curve')
        plt.xlabel(r"$T_{(°C)}$", fontsize=14)
        plt.ylabel(r"$a_{(Å)}$", fontsize=14)
        plt.grid(alpha=0.25)
        plt.legend()

        # Second subplot: Delta T values
        plt.subplot(3, 1, 2)
        plt.plot(best_T_exp, best_delta_temp, 'o', label=r'$\Delta T = T_{fit} - T_{exp}$')
        plt.axhline(0, color='gray', linestyle='--')
        plt.xlabel(r"$T_{(°C)}$", fontsize=14)
        plt.ylabel(r"$\Delta T_{(°C)}$", fontsize=14)
        plt.grid(alpha=0.25)
        plt.legend()

        # Third subplot: Energy optimization
        plt.subplot(3, 1, 3)
        plt.plot(energy_values, metrics, '-o')
        plt.axvline(best_energy, color='red', linestyle='--', label='Optimized Energy')
        plt.xlabel("Energy (keV)", fontsize=14)
        plt.ylabel("Optimization Metric", fontsize=14)
        plt.grid(alpha=0.25)
        plt.legend()

        plt.tight_layout()
        plt.suptitle(f"Energy Optimization Results (Best Energy: {best_energy:.3f} keV, Metric: {min_metric:.5f})", y=1.02)
        
        if save_plot:
            plt.savefig(save_plot, dpi=150)
        plt.show()

    return round(best_energy, 3), round(min_metric, 5), best_delta_temp, best_T_exp, best_T_theo
def process_scan_id1_june_2024(f_scan, part_name, scan, temp,plot_original=None,plot_crop=None):
    def __init__():
        pass
    
    pref_key = 'measurement/mpx1x4'
    key_pos = 'instrument/positioners/'
    
    # Extract position data
    delta, nu, eta = [np.array(f_scan[f"{key_pos}{pos}"][()]) for pos in ["delta", "nu", "eta"]]
    thx, thy, thz = [np.array(f_scan[f"{key_pos}{pos}"][()]).mean() for pos in ["thx", "thy", "thz"]]
    pix, piy, piz = [np.array(f_scan[f"{key_pos}{pos}"][()]).mean() for pos in ["pix", "piy", "piz"]]
    
    pos_x, pos_y, pos_z = thx - pix/1000, thy - piy/1000, thz - piz/1000
    fact = 1.079 * np.pi / 180 / 55 * 10**6
    cch = np.array([431.4, 191.41])

    data_original = f_scan[pref_key][()]
    shape_data = np.array(data_original.shape)
    data_copy = np.array(data_original)

    # Apply masks based on conditions
    if part_name == "II-D8" and scan == 11:
        data_copy[:,:,:370] = 0
        data_copy = apply_mask(data_copy, 0.3, 10, True)
        data_copy[:,:,:370] = data_original[:,:,:370]
    
    elif (part_name == "II-D8" and scan >= 87) or (part_name == "II-G10" and scan <= 408):
        data_copy = apply_mask(data_original, 0.3, 50)
    
    elif (part_name == "II-G10" and scan == 67) or (part_name == "VII-A7" and scan == 88):
        data_copy = apply_mask(data_original, 0.3, 50, True)
    
    elif part_name == "VII-A7":
        if 254 <= scan < 321:
            data_copy = apply_mask(data_original, 0.6, 10)
        elif scan >= 321:
            data_copy = apply_mask(data_original, 0.1, 100)
    
    elif part_name == "VII-A9":
        if 139 <= scan < 258:
            data_copy = apply_mask(data_original, 0.6, 8)
        elif scan == 59 or (100 <= scan < 139) or scan == 333:
            data_copy = apply_mask(data_original, 0.1, 20)
        elif scan in [79, 404]:
            data_copy = apply_mask(data_original, 0.1, 20, True)
        else:
            data_copy = data_original[int(shape_data[0]*0.3):int(shape_data[0]*0.7)]
            max_x, max_y, max_z = get_max_coords(data_copy)
            max_x += int(shape_data[0]*0.3)
            half_w = 80
            data_copy = np.array(data_original)
            data_copy[:max_x-half_w, :, :] = 0;data_copy[:, :max_y-half_w, :] = 0;data_copy[:, :, :max_z-half_w] = 0;
            data_copy[max_x+half_w:, :, :] = 0;data_copy[:, max_y+half_w:, :] = 0;data_copy[:, :, max_z+half_w:] = 0;
    else:
        data_copy = data_original[int(shape_data[0]*0.3):int(shape_data[0]*0.7)]
        max_x, max_y, max_z = get_max_coords(data_copy)
        max_x += int(shape_data[0]*0.3)
        half_w = 80
        data_copy = np.array(data_original)
        data_copy[:max_x-half_w, :, :] = 0;data_copy[:, :max_y-half_w, :] = 0;data_copy[:, :, :max_z-half_w] = 0;
        data_copy[max_x+half_w:, :, :] = 0;data_copy[:, max_y+half_w:, :] = 0;data_copy[:, :, max_z+half_w:] = 0;        
    max_coords = get_max_coords(data_copy)
    com = [int(round(i)) for i in C_O_M(data_copy)]
    print(com)
    data_cropped = crop_3darray_pos(data_copy, com, (200,200,200))
    max_crop = get_max_coords(data_cropped)
    com_crop = [round(i) for i in C_O_M(data_cropped)]
    
    fig_title = part_name+'  | S'+str(scan)+' | '+temp+" °C"
    fig_save_ = part_name+'_S'+str(scan)+'_' +temp+".png"
    if plot_original :
        plot_summary_difraction(data_original,com     ,max_coords ,path_save=plot_original+"original",fig_title=fig_title,fig_save_=fig_save_,f_s=20,vmin=0,vmax=10)
    if plot_crop :
        plot_summary_difraction(data_cropped ,com_crop,max_crop   ,path_save=plot_crop+"crop"        ,fig_title=fig_title,fig_save_=fig_save_,f_s=20,vmin=0,vmax=10)
   

    return [part_name, scan, temp, nu.mean(), delta.mean(), eta[int(com[0])], eta.mean(), *com, 
            data_original.max(), data_original.sum(), *cch, fact, pos_x, pos_y, pos_z]

#####################################################################################################################

