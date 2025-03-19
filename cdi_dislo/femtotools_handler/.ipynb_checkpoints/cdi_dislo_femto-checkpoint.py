"""
#####################################################################################################
# Issues and Areas for Improvement in the Script
#####################################################################################################

# 1. Missing Imports:
#    - The script relies on wildcard imports from `cdi_dislo.common_imports`, which can obscure dependencies.
#    - Ensure that all required functions and modules (e.g., numpy, matplotlib, pandas, seaborn, scipy, warnings, h5py) 
#      are explicitly imported if needed.

# 2. Outlier Handling:
#    - Some filtering conditions use hardcoded thresholds (e.g., `if np.min(Force_drops_max[i]) > 1e6`).
#    - Consider making these thresholds configurable or dynamically adjusting based on the data distribution.

# 3. Function Argument Validation:
#    - Some functions assume specific dictionary keys exist (`analyze_results`).
#    - Add checks to ensure the required keys are available before accessing dictionary elements.

# 4. Undefined Variables:
#    - `array()` is used without explicitly importing `numpy` (`slopes_elastic = array(data['slopes_elastic'])`).
#    - Ensure `numpy` is imported properly.

# 5. Hardcoded Parameters:
#    - Constants like lattice parameters (`a0 = 3.924e-10`), default area values, and drag coefficients (`B = 1e-4`)
#      are fixed. Consider allowing these as function arguments for better flexibility.

# 6. List Processing Issues:
#    - Flattening and processing lists can result in IndexError if they are empty (`distance_drops_flatten = array([...])`).
#    - Check for empty lists before indexing or performing operations on them.

# 7. Debugging and Exception Handling:
#    - Some `try-except` blocks swallow exceptions without proper logging (`velocity.append(get_dislocation_speed(...))`).
#    - Add detailed error messages for debugging.

# 8. File Handling:
#    - The script reads and processes `.npz` and `.hdf5` files but does not check for missing files.
#    - Add validation to ensure files exist before attempting to load them.

# 9. Code Efficiency:
#    - Some loops iterate over data multiple times (`for i in range(len(test))` inside `analyze_results`).
#    - Consider vectorized operations with NumPy for improved performance.

# 10. Plotting Performance:
#    - Some plots involve looping through large datasets (`plt.plot(f_corr_elastic_x[i], f_corr_elastic_y[i], '.')`).
#    - Reduce the number of points plotted for large datasets to improve rendering performance.

#####################################################################################################
# End of Issues and Areas for Improvement
#####################################################################################################
"""


"""
############################################################################################################
#                                    SCRIPT OVERVIEW                                                      #
############################################################################################################

# This script provides various functions and utilities for analyzing mechanical properties of 
# materials, particularly focusing on force-displacement analysis in nanoindentation experiments.
# It includes functionalities for data processing, denoising, regression analysis, and visualization.

# 1. **Issues and Areas for Improvement**
#    - Missing imports and explicit dependencies.
#    - Handling of outliers and hardcoded thresholds.
#    - Validation of function arguments.
#    - Efficiency improvements through vectorization.
#    - Better error handling in data processing.

# 2. **Mathematical & Utility Functions**
#    - `calculate_burgers_modulus()`: Computes the magnitude of the Burgers vector for FCC nanoparticles.
#    - `calculate_tau()`: Calculates the resolved shear stress.
#    - `get_dislocation_speed()`: Computes the speed of a dislocation based on force and drag coefficient.

# 3. **Dislocation & Force Analysis**
#    - `analyze_results()`: Performs comprehensive analysis of force-displacement data from nanoindentation.
#    - `analyze_force_displacement()`: Processes experimental data, extracts elastic and plastic phases, 
#      and performs regression to analyze mechanical properties.
#    - `dict_compare()`: Compares two dictionaries recursively, handling NumPy arrays.

# 4. **Data Loading & Handling**
#    - `load_and_save_as_dict_femtotools()`: Loads data from an `.npz` file and converts it into a dictionary.
#    - `save_data_hdf5()`: Saves structured data into an `.hdf5` file.

# 5. **Regression & Machine Learning Utilities**
#    - `fit_linear_regression()`: Performs linear regression using NumPy.
#    - `fit_and_plot_data()`: Fits data using constrained regression and visualizes results.
#    - `fit_with_min_residual()`: Finds the best regression model based on residuals.
#    - `get_x_fromy_linear()`: Computes the x-value for a given y-value in a linear regression.

# 6. **Nanoindentation Data Processing**
#    - `process_femtotools_data()`: Reads, calibrates, and denoises raw nanoindentation force-displacement data.
#    - `denoise_data_femtotools()`: Implements multiple denoising techniques (Moving Average, Median Filter, etc.).
#    - `get_xmin_nonzero_fromnoiseddata_femtotools()`: Determines the first displacement contact point.

# 7. **Plastic Deformation Analysis**
#    - `get_alldrops_fits_coeficients_and_functions()`: Identifies plastic drops and fits regression models.
#    - `get_x_y_of_drops_plastic()`: Extracts x and y values of force drops in plastic deformation.
#    - `constrained_polyfit()`: Fits data while constraining slope values.
#    - `get_reordering_indices()`: Computes reordering indices based on a target order.

# 8. **Visualization & Plotting**
#    - The script includes various Matplotlib-based plots to analyze mechanical behavior:
#      - Force vs Displacement
#      - Elastic and plastic deformation characteristics
#      - Dislocation velocity and stress distribution
#      - Histograms of slopes, force ratios, and dislocation behavior

# 9. **Execution & Main Script Logic**
#    - The script serves as a utility library, with modular functions that can be 
#      integrated into larger scientific analysis pipelines.

############################################################################################################
#                                   END OF SCRIPT OVERVIEW                                                #
############################################################################################################
"""











from cdi_dislo.common_imports                        import *
from cdi_dislo.general_utilities.cdi_dislo_utils     import     extract_coefficient_and_exponent
from cdi_dislo.plotutilities.cdi_dislo_plotutilities import get_color_list ,plot_mechanical_properties


#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
######################################### femtotools utility#########################################################
#####################################################################################################################
#####################################################################################################################
#------------------------------------------------------------------------------------------------------------
def calculate_burgers_modulus(a0, direction):
    """
    Calculate the modulus of the Burgers vector for an FCC nanoparticle.

    Parameters:
    - a0 (float): Lattice constant in meters.
    - direction (tuple): Direction vector (e.g., (1, 1, 1)).

    Returns:
    - b (float): The magnitude of the Burgers vector in meters.
    """
    import numpy as np

    direction = np.array(direction)
    b = a0 * np.sqrt(np.sum(direction**2)) / 2
    return b
#------------------------------------------------------------------------------------------------------------
def calculate_tau(delta_F, b, S_squared):
    """
    Calculate the resolved shear stress (tau).

    Parameters:
    - delta_F (float): The change in force in newtons.
    - b (float): The modulus of the Burgers vector in meters.
    - S_squared (float): The Schmid factor squared.

    Returns:
    - tau (float): The resolved shear stress in Pascals.
    """
    tau = delta_F / (b * S_squared)
    return tau
#------------------------------------------------------------------------------------------------------------
def get_dislocation_speed(delta_F_sq, S_squared_E_over_l, Area=None, B=1e-4):
    """
    Calculate the dislocation speed for an FCC nanoparticle.
    
    Parameters:
    - delta_F_sq (float): The change in force squared (µN**2).
    - S_squared_E_over_l (float): The Schmid factor squared times E/l (µN/nm).
    - Area (float): The area over which dislocation moves in square nano-meters. Default is None.
    - B (float): The drag coefficient in Pa·s. Default is 1e-4 Pa·s.
    
    Returns:
    - v (float): The dislocation speed in meters per second.
    """
    a0 = 3.924e-10  # Lattice constant for FCC in nanometers (0.3924 nm)

    # Calculate area if not provided (assuming {111} plane for FCC crystal)
    if Area is None:
        Area = (0.5*sqrt(3)/(a0)) * (500*400*300) * 1e-27 # m^2
    else:
        Area = Area * 1e-18  # Convert nm^2 to m^2

    # Calculate dislocation speed v = delta_F_sq / (2 * S_squared_E_over_l * Area * B)
    v = (1e-12) * delta_F_sq / (2 * S_squared_E_over_l * 1e3 * Area * B)
    
    return v

def analyze_results(data, path_pwd,Area_particles,S=(500*10**-9)**2,indenter_type='flat_punch'):
    """
    Analyze and visualize force vs. displacement test results for elastic and plastic deformation phases.

    This function performs comprehensive analysis and visualization of nanoindentation test data,
    including elastic and plastic deformation phases, force drops, and dislocation velocities.

    Parameters:
    -----------
    data : dict
        Dictionary containing the test results. Expected keys include:
        - 'Displacement_cal': List of displacement data for each test
        - 'ForceA': List of force data for each test
        - 'test_part': List of particle names or identifiers
        - 'test': List of test identifiers
        - 'elastic_part_all_x', 'elastic_part_all_y': Lists of elastic phase data
        - 'plastic_part_all_x', 'plastic_part_all_y': Lists of plastic phase data
        - 'Force_drops_min', 'Force_drops_max': Lists of force drop data
        - 'slopes_drops', 'slopes_elastic', 'slopes_release': Lists of slope data
        - 'elastic_limit': List of elastic limit forces
        - 'Volume_bcdi': Dictionary of particle volumes (optional)

    path_pwd : str
        Path to save output plots and files.

    S : float, optional
        Surface area in m^2. Default is (500*10^-9)^2 m^2.

    indenter_type : str, optional
        Type of indenter used. Either 'flat_punch' or 'cube_corner'. Default is 'flat_punch'.

    debug : bool, optional
        If True, print debug information during execution. Default is False.

    Returns:
    --------
    None
        The function saves plots to the specified path and prints analysis results.

    Notes:
    ------
    This function performs the following analyses:
    1. Extracts and analyzes first plastic drop characteristics
    2. Compares elastic phases across different tests
    3. Analyzes force drops and displacement changes during plastic deformation
    4. Calculates and plots dislocation velocities
    5. Generates various distribution plots (slopes, force ratios, etc.)
    6. Compares elastic, plastic, and release phase characteristics

    The function assumes specific data structures and may need modifications if the input data format changes.
    """
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    # Step 1: Extract data from the dictionary
    Displacement_cal     = (data['Displacement_cal']  )
    ForceA               = (data['ForceA']  )
    test_part            = (data['test_part']  )
    test                 = (data['test']  )
    
    slopes_drops         = (data['slopes_drops']  )
    Force_drops_min      = (data['Force_drops_min']  )
    Force_drops_max      = (data['Force_drops_max']  )
    plastic_part_all_x   = (data['plastic_part_all_x']  )
    plastic_part_all_y   = (data['plastic_part_all_y']  )
    distance_drops       = (data['distance_drops']  )
    elastic_part_all_x   = (data['elastic_part_all_x']  )
    elastic_part_all_y   = (data['elastic_part_all_y']  )
    slopes_elastic       = array(data['slopes_elastic']  )
    elastic_limit        = array(data['elastic_limit']  )
    slopes_release       = array(data['slopes_release']  )  
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    # Step 2: Perform the analysis as before
    # Extract the first plastic drop slope and forces
    first_plasticdrops_slope = np.array([i[0] for i in slopes_drops])
    first_plasticdrops_Force_min = np.array([np.array(i)[0] for i in Force_drops_min])
    first_plasticdrops_Force_max = np.array([np.array(i)[0] for i in Force_drops_max])
    # Extract the distance of the plastic drops 
    distance_drops_flatten = array([ii  for i in distance_drops for ii in (i)])
    distance_drops_flatten[distance_drops_flatten>10**4]=0
    # Apply threshold to remove unrealistic values
    first_plasticdrops_slope[first_plasticdrops_slope > 5000] = 0
    first_plasticdrops_Force_min[first_plasticdrops_Force_min > 5000] = 0
    first_plasticdrops_Force_max[first_plasticdrops_Force_max > 5000] = 0
    
    f_max_elastic_x, f_max_elastic_y = [], []
    f_corr_elastic_x, f_corr_elastic_y = [], []
    estimated_thickness_femto_=[]
    for i in range(len(test)):
        x, y, i_part, i_test = Displacement_cal[i] * 1000, ForceA[i], test_part[i], test[i]
        touching_point = np.where(elastic_part_all_y[i] > 1)[0].min() - 40
        delta_x = elastic_part_all_x[i][touching_point]
        f_max_elastic_x.append(elastic_part_all_x[i][-1] - delta_x)
        f_max_elastic_y.append(elastic_part_all_y[i][-1])
        f_corr_elastic_x.append(elastic_part_all_x[i][touching_point:] - delta_x)
        f_corr_elastic_y.append(elastic_part_all_y[i][touching_point:])
    slopes_dropsflattened_list = [item for sublist in slopes_drops for item in np.array(sublist) if item < 100000]
    tr_pls_def=np.where(first_plasticdrops_slope!=0)[0]
    trigger_release=np.where(array(slopes_release)<10000000)[0]
    ratio_forceelasticlimit_plastic_alldopsplastic,delta_f,delta_f_sq,delta_x_plastic=[],[],[],[]
    for i in range(len(Force_drops_min)):
        if np.min(Force_drops_max[i])>1e6:
            delta_f_sq.append([0])
            delta_f.append([0])
            ratio_forceelasticlimit_plastic_alldopsplastic.append([0])
            delta_x_plastic.append([0])  
        else:
            list_max=[elastic_limit[i]]+list(Force_drops_max[i])
            list_x_max=[elastic_part_all_x[i][-1]]+[ii[-1] for ii in plastic_part_all_x[i]]
            list_x_min=[ii[0] for ii in plastic_part_all_x[i]]+[[ii[0] for ii in plastic_part_all_x[i]][-1]]
            list_min=list(Force_drops_min[i])+[elastic_limit[i]]
            ratio=(array(list_min)/array(list_max))
            delta_fff=array(list_max)**2-array(list_min)**2
            ratio_forceelasticlimit_plastic_alldopsplastic.append(ratio)
            delta_f_sq.append(delta_fff)
            delta_f.append(array(list_max)-array(list_min))
            delta_x_plastic.append(array(list_x_max)-array(list_x_min))
    velocity,test_part_new=[],[]
    for i in range(len(delta_f)):
        try:
            #to rethink again about this method as the z direction is in 111 direction 
            #vol=Volume_bcdi[test_part[i]].min()
            #a0 = 3.9241e-1
            #Area=(vol/a0)*(0.5*3**0.5)
            # for now using this method as it's the best apraoch for now
            Area=Area_particles_L[test_part[i]].min()
        except:
            Area=None
        velocity.append(get_dislocation_speed(array(delta_f_sq[i]),slopes_elastic[i],Area=Area))
        test_part_new.append([test_part[i] for iii in range(len(delta_f[i]))] )
    velocity=(velocity)
    #distance_drops[array(distance_drops)>1e5]=0
    velocity_flatten,delta_x_plastic_flatten,delta_f_sq_flatten = [],[],[]

    for i in range(len(velocity)):
        for ii in range(len(velocity[i])):
            if velocity[i][ii] >= 0:
                delta_f_sq_flatten.append(delta_f_sq[i][ii])
                velocity_flatten.append(velocity[i][ii])
                delta_x_plastic_flatten.append(delta_x_plastic[i][ii])
    print(len(velocity_flatten),len(delta_x_plastic_flatten))

    max_elastic_saphire=np.max([i.max() for i in f_corr_elastic_y])
    max_elastic_nosaphire=np.max([f_corr_elastic_y[i].max() for i in range(len(f_corr_elastic_y)) if "Saphire" not in test_part[i]])
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################


    
    # Step 3: Compare the elastic phase across different tests
    # plot the force max elastic Vs the displacement= hauteur-hcontact
    plt.figure(figsize=(16, 5))
    for i in range(len(test)):
        x, y, i_part, i_test = Displacement_cal[i] * 1000, ForceA[i], test_part[i], test[i]
        if "Saphire" in i_part: continue
        plt.plot(f_max_elastic_x[i], f_max_elastic_y[i], "*", label=i_part)
        
    plt.title("Comparison of The elastic phase.\n The force max vs the displacement at max force",fontsize=18)
    plt.xlabel("Displacement (nm)",fontsize=18)
    plt.ylabel("Force (µN)",fontsize=18)
    plt.legend(ncols=3, loc='best',fontsize=12)
    plt.tight_layout()
    plt.savefig(path_pwd + "Forcemax_vs_displacementmax_elasticpart.png", dpi=100)
    plt.show()
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    
    # Step 4.0: Compare the elastic phase across different tests
    #plot the elastic part force vs displacement with saphire
    plt.figure(figsize=(16, 5))
    for i in range(len(test)):
        plt.plot(f_corr_elastic_x[i], f_corr_elastic_y[i], '.',markersize=5,label=test_part[i])
    plt.title("Comparison of The elastic phase.\n The force vs the shifted displacement",fontsize=18)
    plt.xlabel("Displacement (nm)",fontsize=18)
    plt.ylabel("Force (µN)",fontsize=18)
    plt.legend(ncols=3, loc='best',fontsize=12)
    plt.tight_layout()
    plt.savefig(path_pwd + "Force_vs_displacement_elasticpart.png", dpi=100)
    plt.show()
    
    # Step 4.1: Compare the elastic phase across different tests
    #plot the elastic part force vs displacement  without saphire
    plt.figure(figsize=(16, 5))
    for i in range(len(test)):
        if "Saphire" in test_part[i]:continue
        plt.plot(f_corr_elastic_x[i], f_corr_elastic_y[i], '.',markersize=5,label=test_part[i])
    plt.title("Comparison of The elastic phase.\n The force vs the shifted displacement",fontsize=18)
    plt.xlabel("Displacement (nm)",fontsize=18)
    plt.ylabel("Force (µN)",fontsize=18)
    plt.legend(ncols=3, loc='best',fontsize=12)
    plt.tight_layout()
    plt.savefig(path_pwd + "Force_vs_displacement_elasticpart_withoutsaphire.png", dpi=100)
    plt.show()

       
    # Step 4.2: Compare the elastic phase across different tests
    #plot the elastic part normalised force vs displacement with saphire
    plt.figure(figsize=(16, 5))
    for i in range(len(test)):
        plt.plot(f_corr_elastic_x[i], f_corr_elastic_y[i]/max_elastic_saphire, '.',markersize=5,label=test_part[i])
    plt.title("Comparison of The elastic phase.\n The normalised force vs the displacement",fontsize=18)
    plt.xlabel("Displacement (nm)",fontsize=18)
    plt.ylabel(f"Force (µN)/{int(max_elastic_saphire)}",fontsize=18)
    plt.legend(ncols=3, loc='best',fontsize=12)
    plt.tight_layout()
    plt.savefig(path_pwd + "normForce_vs_displacement_elasticpart.png", dpi=100)
    plt.show()
       
    # Step 4.3: Compare the elastic phase across different tests
    #plot the elastic part normalised force vs displacement without saphire
    plt.figure(figsize=(16, 5))
    for i in range(len(test)):
        if "Saphire" in test_part[i]:continue
        plt.plot(f_corr_elastic_x[i], f_corr_elastic_y[i]/max_elastic_nosaphire, '.',markersize=5,label=test_part[i])
    plt.title("Comparison of The elastic phase.\n The normalised force vs the displacement",fontsize=18)
    plt.xlabel("Displacement (nm)",fontsize=18)
    plt.ylabel(f"Force (µN)/{int(max_elastic_saphire)}",fontsize=18)
    plt.legend(ncols=3, loc='best',fontsize=12)
    plt.tight_layout()
    plt.savefig(path_pwd + "normForce_vs_displacement_elasticpartwithoutsaphire.png", dpi=100)
    plt.show()
    
   # Step 4.4: Compare the elastic phase across different tests +fit (force/presure)
    maxxxx_x,maxxxx_y=[],[]
    for i in range(len(test)):
        k = int(len(np.array(f_corr_elastic_x[i])) * 0.5)
        x = np.array(f_corr_elastic_x[i])[-k:] - np.array(f_corr_elastic_x[i])[-k]
        y = np.array(f_corr_elastic_y[i])[-k:] - np.array(f_corr_elastic_y[i])[-k]
        maxxxx_y.append(y.max())
        maxxxx_x.append(x.max())
    not_sapphire=np.where(maxxxx_y<0.5*array(maxxxx_y).max())[0]
    maxxxx_x=array(maxxxx_x)[not_sapphire].max()
    maxxxx_y=array(maxxxx_y)[not_sapphire].max()  
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    k_plot=1
    for i in range(len(test)):
        k = int(len(np.array(f_corr_elastic_x[i])) * 0.5)
        x = np.array(f_corr_elastic_x[i])[-k:] - np.array(f_corr_elastic_x[i])[-k]
        y = np.array(f_corr_elastic_y[i])[-k:] - np.array(f_corr_elastic_y[i])[-k]
        # Perform linear fit
        a, b = np.polyfit(x, y, 1)
        
        # Plot scatter points in the first subplot
        ax1.plot(x*k_plot, y*k_plot, '.', markersize=5, label=f'{test_part[i]} (data)')
        
        # Plot linear fit in the second subplot
        x_fit = np.linspace(x.min(), maxxxx_x, 100)
        fit_y = a * x_fit + b
        ax2.plot(x_fit*k_plot, fit_y*k_plot, '-', linewidth=2, label=f'{test_part[i]} (fit: y = {a:.2f}x + {b:.2f})')
    
    # Customize the first subplot
    ax1.set_title("Comparison of The Elastic Phase: Real Data", fontsize=18)
    ax1.set_ylabel("Force (µN)", fontsize=18)
    ax1.legend(ncols=3, loc='best', fontsize=10)
    ax1.set_ylim(0, maxxxx_y)
    
    # Customize the second subplot
    ax2.set_title("Comparison of The Elastic Phase: Linear Fits", fontsize=18)
    ax2.set_xlabel("Displacement (nm)", fontsize=18)
    ax2.set_ylabel("Force (µN)", fontsize=18)
    ax2.legend(ncols=3, loc='best', fontsize=10)
    ax2.set_ylim(0, maxxxx_y)
    plt.tight_layout()
    plt.savefig(path_pwd + "fit_and_data_normForce_vs_displacement_elasticpart.png", dpi=100)
    plt.show()
    
        
    maxxxx_x,maxxxx_y=[],[]
    for i in range(len(test)):
        part_name=test_part[i]
        if part_name not in Area_particles.keys():continue
        area_contact=Area_particles[part_name].mean()
        k = int(len(np.array(f_corr_elastic_x[i])) * 0.5)
        x = np.array(f_corr_elastic_x[i])[-k:] - np.array(f_corr_elastic_x[i])[-k]
        y = 1000*(np.array(f_corr_elastic_y[i])[-k:] - np.array(f_corr_elastic_y[i])[-k])/area_contact
        
        maxxxx_y.append(y.max())
        maxxxx_x.append(x.max())
    not_sapphire=np.where(maxxxx_y<0.5*array(maxxxx_y).max())[0]
    maxxxx_x=array(maxxxx_x)[not_sapphire].max()
    maxxxx_y=array(maxxxx_y)[not_sapphire].max()  
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    k_plot=1
    for i in range(len(test)):
        part_name=test_part[i]
        if part_name not in Area_particles.keys():continue
        area_contact=Area_particles[part_name].mean()
        k = int(len(np.array(f_corr_elastic_x[i])) * 0.5)
        x = np.array(f_corr_elastic_x[i])[-k:] - np.array(f_corr_elastic_x[i])[-k]
        y = 1000*(np.array(f_corr_elastic_y[i])[-k:] - np.array(f_corr_elastic_y[i])[-k])/area_contact
        # Perform linear fit
        a, b = np.polyfit(x, y, 1)
        
        # Plot scatter points in the first subplot
        ax1.plot(x*k_plot, y*k_plot, '.', markersize=5, label=f'{test_part[i]} (data)')
        
        # Plot linear fit in the second subplot
        x_fit = np.linspace(0, maxxxx_x, 100)
        fit_y = a * x_fit + b
        ax2.plot(x_fit*k_plot, fit_y*k_plot, '-', linewidth=2, label=f'{test_part[i]} (fit: y = {a:.2f}x + {b:.2f})')
    
    # Customize the first subplot
    ax1.set_title("Comparison of The Elastic Phase: Real Data", fontsize=18)
    ax1.set_ylabel("Pressure (GPa)", fontsize=18)
    ax1.legend(ncols=3, loc='best', fontsize=10)
    ax1.set_ylim(0, maxxxx_y)
    
    # Customize the second subplot
    ax2.set_title("Comparison of The Elastic Phase: Linear Fits", fontsize=18)
    ax2.set_xlabel("Displacement (nm)", fontsize=18)
    ax2.set_ylabel("Pressure (GPa)", fontsize=18)
    ax2.legend(ncols=3, loc='best', fontsize=10)
    ax2.set_ylim(0, maxxxx_y)
    
    plt.tight_layout()
    plt.savefig(path_pwd + "fit_and_data_normPressure_vs_displacement_elasticpart.png", dpi=100)
    plt.show()
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    
    # Step 5: Plot the distribution of slopes during plastic deformation
    num_bins = 20  # Increase the number of bins for finer steps
    
    # Determine the common range for both datasets
    min_edge = np.min([np.min(slopes_dropsflattened_list), slopes_elastic.min()])
    max_edge = np.max([np.max(slopes_dropsflattened_list), slopes_elastic.max()])
    
    # Create a common set of bin edges
    bin_edges = np.linspace(min_edge, max_edge, num_bins + 1)
    
    plt.figure(figsize=(16, 5))
    plt.hist(slopes_dropsflattened_list, bins=bin_edges, histtype='bar', edgecolor='black', alpha=0.7, label='Plastic Deformation')
    plt.hist(slopes_elastic, bins=bin_edges, histtype='bar', edgecolor='black', alpha=0.5, label='Elastic Deformation')
    
    plt.title("Slope distribution of plastic/elastic deformation of all the particles", fontsize=18)
    plt.xlabel("Slope (µN/nm)", fontsize=18)
    plt.ylabel("Frequency", fontsize=18)
    plt.xticks(bin_edges[::2])  # Adjust the step to control x-axis tick density
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_pwd + "slopes_distribution_plastic_elastic.png", dpi=100)
    plt.show()
   
    # Step 6.0: Plot the slope comparison for elastic and first plastic phase
    plt.figure(figsize=(16, 5))
    plt.plot(test_part                 ,array(slopes_elastic)                        ,'*',label='elastic')
    plt.plot(test_part[tr_pls_def]     ,array(first_plasticdrops_slope)[tr_pls_def]  ,"*",label=f'First plastic')
    plt.plot(test_part[trigger_release],array(slopes_release)[trigger_release]       ,"*",label=f'Release')
    plt.xticks(rotation=-45)
    plt.title("Slope plot of elastic & First plastic & release phases of all the particles",fontsize=18)
    plt.xlabel("Particle",fontsize=18)
    plt.ylabel("Slope (µN/nm)",fontsize=18)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(path_pwd + "slopes_plot_elastic_plastic_release.png", dpi=100)
    plt.show()


    
    # Step 6.1: Plot the slope comparison for elastic and first plastic phase for non-Sapphire particles
    plt.figure(figsize=(16, 5))
    
    # Create masks for non-Sapphire particles
    mask_non_sapphire_elastic = np.array(["Saphire" not in element for element in test_part])
    mask_non_sapphire_pls_def = np.array(["Saphire" not in element for element in test_part[tr_pls_def]])
    mask_non_sapphire_release = np.array(["Saphire" not in element for element in test_part[trigger_release]])
    
    # Plot for elastic phase
    plt.plot(test_part[mask_non_sapphire_elastic],
             np.array(slopes_elastic)[mask_non_sapphire_elastic],
             '*', label='elastic')
    
    # Plot for first plastic phase
    plt.plot(test_part[tr_pls_def][mask_non_sapphire_pls_def],
             np.array(first_plasticdrops_slope)[tr_pls_def][mask_non_sapphire_pls_def],
             "*", label='First plastic')
    
    # Plot for release phase
    plt.plot(test_part[trigger_release][mask_non_sapphire_release],
             np.array(slopes_release)[trigger_release][mask_non_sapphire_release],
             "*", label='Release')
    
    plt.xticks(rotation=-45)
    plt.title("Slope plot of elastic, First plastic & release phases", fontsize=18)
    plt.xlabel("Particle", fontsize=18)
    plt.ylabel("Slope (µN/nm)", fontsize=18)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(path_pwd + "slopes_plot_elastic_plastic_release_withoutsaphire.png", dpi=100)
    plt.show()

    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################


        
    # Step 7.0: Plot the force drop ratio during the first plastic event (max force ratio)
    plt.figure(figsize=(16, 5))
    ratio_el_min = np.array(first_plasticdrops_Force_max) / np.array(elastic_limit)
    ratio_el_max = np.array(first_plasticdrops_Force_min) / np.array(elastic_limit)
    plt.plot(test_part, ratio_el_max, '*', label="$F_{plastic\ max}\ /F_{elastic}$")
    plt.plot(test_part, ratio_el_min , '*', label="$F_{plastic\ min}\ /F_{elastic}$")
    plt.xticks(rotation=-45)
    plt.title("Force drop RATIO during the first plastic event all the particles",fontsize=18)
    plt.xlabel("Particle",fontsize=18)
    plt.ylabel("Force ratio",fontsize=18)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(path_pwd + "Forcemaxmin_ratio_plot_elastic_plastic.png", dpi=100)
    plt.show()
    
    
    # Step 7.1: Plot the force drop ratio during the first plastic event (max force ratio) WITHOUT SAPHIRE
    mask_nonplasticevent=np.where(ratio_el_min!=0)[0]
    plt.figure(figsize=(16, 5))
    ratio_el_min = np.array(first_plasticdrops_Force_max) / np.array(elastic_limit)
    ratio_el_max = np.array(first_plasticdrops_Force_min) / np.array(elastic_limit)
    plt.plot(test_part[mask_nonplasticevent], ratio_el_max[mask_nonplasticevent], '*', label="$F_{plastic\ max}\ /F_{elastic}$")
    plt.plot(test_part[mask_nonplasticevent], ratio_el_min[mask_nonplasticevent] , '*', label="$F_{plastic\ min}\ /F_{elastic}$")
    plt.xticks(rotation=-45)
    plt.title("Force drop during the first plastic event all the particles",fontsize=18)
    plt.xlabel("Particle",fontsize=18)
    plt.ylabel("Force ratio",fontsize=18)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(path_pwd + "Forcemaxmin_ratio_plot_elastic_plastic_WITHOUTSAPHIRE.png", dpi=100)
    plt.show()
    
    
    # Step 8: Plot the distribution of ratio elastic plastic for all elastic drops
    if indenter_type=='flat_punch':
        plt.figure(figsize=(16, 5))
        plt.plot(test_part,[i[0] for i in ratio_forceelasticlimit_plastic_alldopsplastic], '*',  label='$ \\frac{F_{\text{plastic max}}}{F_{\text{elastic}}}$')
        plt.title("Force ratio elastic plastic for all particles",fontsize=18)
        plt.xlabel("Particle",fontsize=18)
        plt.ylabel("$ \\frac{F_{\text{plastic max}}}{F_{\text{elastic}}}$",fontsize=18)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(path_pwd + "all_Ratioforce_elastic_plastic_distribution.png", dpi=100)
        plt.show()
    if indenter_type=='cube_corner':
        plt.figure(figsize=(16, 5))
        for iii_part in range(len(test_part)):
            y=array(ratio_forceelasticlimit_plastic_alldopsplastic[iii_part])
            x=np.arange(1,len(y)+1,1)
            if "Saphire" in test_part[iii_part]:
                        continue
            if y.max()==0:continue
            plt.plot(x,y, '*' ,label=test_part[iii_part] )#label='$ \\frac{F_{\text{plastic max}}}{F_{\text{elastic}}}$')
        plt.title("Force ratio elastic plastic for all particles",fontsize=18)
        plt.xlabel("Drops number",fontsize=18)
        plt.ylabel("$ \\frac{F_{\text{plastic max}}}{F_{\text{elastic}}}$",fontsize=18)
        plt.legend(ncols=3,fontsize=12)
        plt.tight_layout()
        plt.savefig(path_pwd + "all_Ratioforce_elastic_plastic_distribution.png", dpi=100)
        plt.show()   
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    
    # Step 9: Plot the distribution of drops length during plastic deformation
    num_bins = 20  # Increase the number of bins for finer steps
    hist_values, bin_edges = np.histogram(distance_drops_flatten[distance_drops_flatten!=0], bins=num_bins)
    plt.figure(figsize=(16, 5))
    plt.hist(distance_drops_flatten, bins=bin_edges, histtype='bar', edgecolor='black')
    plt.title("Drops length distribution of plastic deformation of all the particles",fontsize=18)
    plt.xlabel("Drops length (nm)",fontsize=18)
    plt.ylabel("#",fontsize=18)
    plt.xticks(bin_edges[::1])  # Adjust the step to control x-axis tick density
    plt.tight_layout()
    plt.savefig(path_pwd + "segment_drops_length_distribution_plastic.png", dpi=100)
    plt.show()
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################   
    # Step 10: Plot the force drop  gradient during the first plastic event (force ratio)
    plt.figure(figsize=(16, 5))
    plt.plot(test_part, (np.array(first_plasticdrops_Force_max)- np.array(first_plasticdrops_Force_min) ) , '*', label="$\Delta F_{plastic\ max}$")
    plt.xticks(rotation=-45)
    plt.title("Force drop difference during the first plastic event all the particles",fontsize=18)
    plt.xlabel("Particle",fontsize=18)
    plt.ylabel("Force ratio",fontsize=18)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(path_pwd + "DeltaForcemaxmin_plot_elastic_plastic.png", dpi=100)
    plt.show()
    # Step 11: Plot the delta force drop   vs the displacement
    plt.figure(figsize=(16, 5))
    trigger_plastic_event=np.where(first_plasticdrops_Force_min!=0)[0]
    y=array([i[-1] for i in elastic_part_all_y])-first_plasticdrops_Force_min
    x=array([i[-1] for i in elastic_part_all_x])
    plt.plot(x[trigger_plastic_event], y[trigger_plastic_event], '*')
    plt.xticks(rotation=-45)
    plt.title("Delta of Force drop VS the dispacement at the drop for all the particles",fontsize=18)
    plt.xlabel("Displacement before plastic event",fontsize=18)
    plt.ylabel("$F_{plastic\ max} - F_{elastic}$",fontsize=18)
    plt.tight_layout()
    plt.savefig(path_pwd + "DeltaForcemaxmin_plot_vs_displacement.png", dpi=100)
    plt.show()
    # Step 11 vis: Plot the delta force drop during  vs the delta displacement
    plt.figure(figsize=(16, 5))
    delta_x_elastic_plastic= array([ii for i in delta_x_plastic for ii in i])#array(plastic_x)-array([i[-1] for i in elastic_part_all_x])
    delta_y_elastic_plastic= array([ii for i in delta_f for ii in i])#array(plastic_x)-array([i[-1] for i in elastic_part_all_x])
    aranged_deltay=array(delta_y_elastic_plastic)
    aranged_deltay.sort()
    plt.plot(delta_x_elastic_plastic, delta_y_elastic_plastic, '*')
    plt.xticks(rotation=-45)
    plt.title("$\Delta$ of Force VS $\Delta$ dispacement between the elastic and plastic phase\n",fontsize=18)
    plt.xlabel("$\Delta$ Displacement",fontsize=18)
    plt.ylabel("$F_{plastic\ max} - F_{elastic}$",fontsize=18)
    #plt.ylim(0,aranged_deltay[-3]*1.5)
    plt.tight_layout()
    plt.savefig(path_pwd + "DeltaForcemaxmin_plot_vs_deltadisplacement.png", dpi=100)
    plt.show()
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    # Step 12.0: Plot the delta f**2 of dislocation vs delta displacement
    if indenter_type=='flat_punch':
        mask_non_sapphire= np.array(["Saphire" not in element for element in test_part])
        x=test_part[mask_non_sapphire]
        y=1e-6*array(delta_f_sq_flatten)[mask_non_sapphire]
        plt.figure(figsize=(16, 5))
        plt.plot(x,y, '*')
        plt.xticks(rotation=-45)
        plt.title("Dislocation $\Delta (F^2)$  for each the particles",fontsize=18)
        plt.xlabel("Particle name",fontsize=18)
        plt.ylabel(" $\Delta (F^2)_{N²}$  ",fontsize=18)
        plt.tight_layout()
        plt.savefig(path_pwd + "Particle_plot_vs_dislodelta_fsq_all.png", dpi=100)
        plt.show()  
        plt.figure(figsize=(16, 5))
        plt.plot(x[y!=0],y[y!=0], '*')
        plt.xticks(rotation=-45)
        plt.title("Dislocation $\Delta (F^2)$  for each the particles",fontsize=18)
        plt.xlabel("Particle name",fontsize=18)
        plt.ylabel(" $\Delta (F^2)_{N²}$  ",fontsize=18)
        plt.tight_layout()
        plt.savefig(path_pwd + "Particle_plot_vs_dislodelta_fsq_nonzero.png", dpi=100)
        plt.show()  
    if indenter_type=='cube_corner':

        plt.figure(figsize=(16, 5))
        # Plot each array as a separate line
        labels=test_part
        for i, arr in enumerate(delta_f_sq):
            arr=array(arr)
            if "Saph" in labels[i]: continue
            if len(arr) > 0:  # Only plot non-empty arrays
                if (arr>0).sum()!=0:
                    plt.plot(np.log10(arr[arr!=0]),'*', label=labels[i])
        plt.xticks(rotation=-45)
        plt.title("Dislocation $\Delta (F^2)$  for each the particles",fontsize=18)
        plt.xlabel("plastic event id",fontsize=18)
        plt.ylabel("$\Delta (F^2)$",fontsize=18)
        plt.legend(ncols=3,fontsize=12,loc="best")
        plt.tight_layout()
        plt.savefig(path_pwd + "Particle_plot_vs_dislo_deltafsq.png", dpi=100)
        plt.show()  
        
 
        f_s=18
        # Assuming velocity is your list of arrays and test_part is your list of labels
        # First, let's create a DataFrame that seaborn can use
        
        data = []
        for label, vel in zip(test_part, delta_f_sq):
            if "Saph" not in label:  # Skip "Saph" particles as in your previous plot
                for v in vel:
                    if v > 0:  # Skip zero values
                        data.append({'Particle': label, 'delta_f_sq': v})
        
        df = pd.DataFrame(data)
        
        # Create the plot
        plt.figure(figsize=(8, 5))
        
        # Create a box plot with individual points
        sns.boxplot(x='Particle', y='delta_f_sq', data=df, whis=[0, 100], width=0.1)
        sns.stripplot(x='Particle', y='delta_f_sq', data=df, color=".3", size=5, jitter=True)
        
        # Use log scale for y-axis
        plt.yscale('log')
        
        # Customize the plot
        plt.title("Dislocation $\Delta (F^2)$ Distribution for each Particle", fontsize=f_s)
        plt.xlabel("Particle", fontsize=f_s)
        plt.ylabel("$\Delta (F^2)$", fontsize=f_s)
        plt.xticks(rotation=-45, ha='left',fontsize=f_s)
        plt.yticks(rotation=-45 ,fontsize=f_s)
        
        # Adjust layout and display
        plt.tight_layout()
        plt.savefig(path_pwd + "Particle_plot_vs_dislo_deltafsq_byparticle.png", dpi=100)
        plt.show()
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    
    # Step 12.1: Plot the velocity of dislocation vs delta displacement
    plt.figure(figsize=(16, 5))
    plt.plot(delta_x_plastic_flatten,velocity_flatten, '*')
    plt.xticks(rotation=-45)
    plt.title("Dislocation Velocity VS $\Delta Displacement$",fontsize=18)
    plt.xlabel("$ \Delta Disp$",fontsize=18)
    plt.ylabel("Velocity (m/s)",fontsize=18)
    plt.tight_layout()
    plt.savefig(path_pwd + "Deltaxdrop_plot_vs_dislovelocity.png", dpi=100)
    plt.show()
    # Step 13: Plot the velocity of dislocation vs particle name
    if indenter_type=='flat_punch':
        mask_non_sapphire= np.array(["Saphire" not in element for element in test_part])
        x=test_part[mask_non_sapphire]
        y=array(velocity_flatten)[mask_non_sapphire]
        plt.figure(figsize=(16, 5))
        plt.plot(x,y, '*')
        plt.xticks(rotation=-45)
        plt.title("Dislocation Velocity  for each the particles",fontsize=18)
        plt.xlabel("Particle name",fontsize=18)
        plt.ylabel("Velocity (m/s)",fontsize=18)
        plt.tight_layout()
        plt.savefig(path_pwd + "Particle_plot_vs_dislovelocity_all.png", dpi=100)
        plt.show()  
        
        plt.figure(figsize=(16, 5))
        plt.plot(x[y!=0],y[y!=0], '*')
        plt.xticks(rotation=-45)
        plt.title("Dislocation Velocity  for each the particles",fontsize=18)
        plt.xlabel("Particle name",fontsize=18)
        plt.ylabel("Velocity (m/s)",fontsize=18)
        plt.tight_layout()        
        plt.savefig(path_pwd + "Particle_plot_vs_dislovelocity_nonzero.png", dpi=100)
        plt.show()  
        
        plt.figure(figsize=(16, 5))
        plt.plot(x[y!=0],y[y!=0], '*')
        plt.yscale('log')
        plt.xticks(rotation=-45)
        plt.title("Dislocation Velocity  for each the particles",fontsize=18)
        plt.xlabel("Particle name",fontsize=18)
        plt.ylabel("Velocity (m/s)",fontsize=18)
        plt.tight_layout()        
        plt.savefig(path_pwd + "Particle_plot_vs_dislovelocity_nonzero.png", dpi=100)
        plt.show()  
                
        
 
    if indenter_type=='cube_corner':

        plt.figure(figsize=(16, 5))
        # Plot each array as a separate line
        labels=test_part
        for i, arr in enumerate(velocity):
            if "Saph" in labels[i]: continue
            if len(arr) > 0:  # Only plot non-empty arrays
                if (arr>0).sum()!=0:
                    plt.plot(np.log10(arr[arr!=0]),'*', label=labels[i])
        plt.xticks(rotation=-45)
        plt.title("Dislocation Velocity  for each the particles",fontsize=18)
        plt.xlabel("plastic event id",fontsize=18)
        plt.ylabel("Log(Velocity) (m/s)",fontsize=18)
        plt.legend(ncols=3,fontsize=12,loc="best")
        plt.tight_layout()
        plt.savefig(path_pwd + "Particle_plot_vs_dislovelocity.png", dpi=100)
        plt.show()  
        
 
        f_s=18
        # Assuming velocity is your list of arrays and test_part is your list of labels
        # First, let's create a DataFrame that seaborn can use
        
        data = []
        for label, vel in zip(test_part, velocity):
            if "Saph" not in label:  # Skip "Saph" particles as in your previous plot
                for v in vel:
                    if v > 0:  # Skip zero values
                        data.append({'Particle': label, 'Velocity': v})
        
        df = pd.DataFrame(data)
        
        # Create the plot
        plt.figure(figsize=(8, 5))
        
        # Create a box plot with individual points
        sns.boxplot(x='Particle', y='Velocity', data=df, whis=[0, 100], width=0.1)
        sns.stripplot(x='Particle', y='Velocity', data=df, color=".3", size=5, jitter=True)
        
        # Use log scale for y-axis
        plt.yscale('log')
        
        # Customize the plot
        plt.title("Dislocation Velocity Distribution for each Particle", fontsize=f_s)
        plt.xlabel("Particle", fontsize=f_s)
        plt.ylabel("Log(Velocity) (m/s)", fontsize=f_s)
        plt.xticks(rotation=-45, ha='left',fontsize=f_s)
        plt.yticks(rotation=-45 ,fontsize=f_s)
        
        # Adjust layout and display
        plt.tight_layout()
        plt.savefig(path_pwd + "Particle_plot_vs_dislovelocity_byparticle.png", dpi=100)
        plt.show()
        
    plot_mechanical_properties(slopes_elastic, f_max_elastic_x, f_max_elastic_y, test_part, include_sapphire=False, plot_fit=True,save_plot=path_pwd + "plot_mechanical_properties_without_sapphire.png")
    plot_mechanical_properties(slopes_elastic, f_max_elastic_x, f_max_elastic_y, test_part, include_sapphire=True, plot_fit=True,save_plot=path_pwd + "plot_mechanical_properties_with_sapphire.png")



    return test_part,velocity
#------------------------------------------------------------------------------------------------------------
def analyze_force_displacement(Displacement_cal, ForceA, test_part, test, phase, path_pwd,indenter_type="cube_corner",font_size=24):
    """
    Analyze force vs. displacement data for multiple tests, identifying elastic and plastic phases, 
    and performing linear regressions to determine slopes.

    Parameters:
    -----------
    Displacement_cal : list of np.ndarray
        List of displacement values for each test, calibrated and scaled.
    ForceA : list of np.ndarray
        List of force values corresponding to the displacement values for each test.
    test_part : list of str
        List of test part identifiers for each test.
    test : list of int
        List of test numbers.
    phase : list of np.ndarray
        List of phase identifiers for each test (1: approach, 2: compression, 3: holding, 4: release).
    path_pwd : str
        Path to the directory where the analysis plot will be saved.

    Returns:
    --------
    results : dict
        Dictionary containing the results of the analysis:
        - distance_drops : list of list of float
            Distance drops during the plastic deformation phase.
        - Force_drops_max : list of list of float
            Maximum forces during the plastic deformation phase.
        - elastic_limit : list of float
            Elastic limit forces for each test.
        - Force_drops_min : list of list of float
            Minimum forces during the plastic deformation phase.
        - slopes_drops : list of list of float
            Slopes of the drops during the plastic deformation phase.
        - slopes_release : list of float
            Slopes during the release phase for each test.
        - elastic_part_all_x : list of np.ndarray
            Displacement values for the elastic part of each test.
        - elastic_part_all_y : list of np.ndarray
            Force values for the elastic part of each test.
        - last_touching_sample_point_allpart : list of tuple
            Last touching sample point (displacement, force) for each test.
        - slopes_elastic : list of float
            Slopes during the elastic phase for each test.
        - plastic_part_all_x : list of list of np.ndarray
            Displacement values for the plastic part of each test.
        - plastic_part_all_y : list of list of np.ndarray
            Force values for the plastic part of each test.

    Steps:
    ------
    1. Initialize lists to store results.
    2. Create a figure to hold subplots for each test.
    3. Loop through each test and process the data:
        a. Extract displacement, force, test part, and phase data for the current test.
        b. Identify indices for each phase (approach, compression, holding, release).
        c. Plot the data points for each phase with different markers and colors.
        d. Determine the inflection points and process the release phase:
            i. If the release phase data is not available or invalid, note the scan was aborted.
            ii. Otherwise, perform linear regression on the release phase data and plot the fit.
        e. Process the approach phase:
            i. If there are no inflection points, fit a line to the elastic phase and plot the fit.
            ii. If inflection points exist, split the approach phase into elastic and plastic parts:
                - Fit a line to the elastic part and plot the fit.
                - Identify and fit lines to the drops in the plastic part and plot the fits.
        f. Annotate the subplot with the linear regression equations.
    4. Adjust the layout and save the figure to the specified path.
    5. Compile results into a dictionary and return it.
    """
    # Initialize lists to store results
    distance_drops, Force_drops_max, elastic_limit, Force_drops_min = [], [], [], []
    slopes_drops, slopes_release = [], []
    elastic_part_all_x, elastic_part_all_y, last_touching_sample_point_allpart = [], [], []
    slopes_elastic,coeficents_elastic = [],[]
    plastic_part_all_x, plastic_part_all_y = [], []
    
    plt.figure(figsize=(50, 40))

    # Process each test
    for i in range(len(test)):
        ax = plt.subplot((len(test) // 3) + 1, 3, i + 1)
        i_real = i

        x, y, i_part, i_test = Displacement_cal[i_real] * 1000, ForceA[i_real], test_part[i_real], test[i_real]
        print(f"Particle: {i_part} ||| test: {i_test}")
        y_phase_1 = np.where(phase[i_real] == 1)[0]  # approach phase 
        y_phase_2 = np.where(phase[i_real] == 2)[0]  # compression phase
        y_phase_3 = np.where(phase[i_real] == 3)[0]  # holding phase
        y_phase_4 = np.where(phase[i_real] == 4)[0]  # release phase

        plt.xlim(get_xmin_nonzero_fromnoiseddata_femtotools(x, y)-1, x.max() + 1)
        plt.plot(x[y_phase_1], y[y_phase_1], ".", markersize=5, label="Approach"   )
        plt.plot(x[y_phase_2], y[y_phase_2], ".", markersize=2, label="Compression")
        plt.plot(x[y_phase_3], y[y_phase_3], ".", markersize=2, label="Holding"    )
        plt.plot(x[y_phase_4], y[y_phase_4], ".", markersize=1, label="Release"    )

        x_values_fit = np.linspace(min(x), max(x), 100)

        diff_y = np.diff(y[y_phase_1])
        if ((i_test == 6) and ( indenter_type=="flat_punch"   )):
            inflection_point = np.where(diff_y < -2)[0]
        else:  
            inflection_point = np.where(diff_y < -5)[0]

        fact_drop = 0.5

        ## Phase 4 analysis (Release phase)
        if len(y[y_phase_4]) == 0:
            print(" The scan was Aborted")
            equation_phase4 = 'The scan was Aborted'
        else:
            F_max_3 = y[y_phase_4].max()
            if F_max_3 == 0:
                print(" The scan was Aborted")
                equation_phase4 = 'The scan was Aborted'
            else:
                start_phase4 = np.where(y[y_phase_4] == F_max_3)[0][0]
                end_fit_4 = np.where(y[y_phase_4] <= F_max_3 * fact_drop)[0][0]
                y_phase4_for_fit = y[y_phase_4][start_phase4 + 1:end_fit_4]
                x_phase4_for_fit = x[y_phase_4][start_phase4 + 1:end_fit_4]
                poly_function_phase4, coefficients_phase4 = fit_linear_regression(x_phase4_for_fit, y_phase4_for_fit)
                y_values_phase4 = poly_function_phase4(x_values_fit)
                plt.plot(x_values_fit, y_values_phase4, linewidth=2)
                equation_phase4 = f'F = {coefficients_phase4[0]:.2f} ΔL + {coefficients_phase4[1]:.2f}'
                last_touching_sample_point = (x[y_phase_4][start_phase4 + 1], y[y_phase_4][start_phase4 + 1])
        
        ## Phase 1 analysis (Approach phase) - simple case: elastic deformation, no drops
        if len(inflection_point) == 0 or (len(y[y_phase_2]) == 0 and len(inflection_point) == 1):
            label_phase1, label_phase4 = "Approach", "Release"
            end_elasticphase_1 = np.where(y[y_phase_1] == y[y_phase_1].max())[0][0]
            start_fit_1 = np.where(y[y_phase_1] > y[y_phase_1].max() * fact_drop)[0][0]
            x_phase1_for_fit_elastic = x[y_phase_1][start_fit_1:end_elasticphase_1 - 1]
            y_phase1_for_fit_elastic = y[y_phase_1][start_fit_1:end_elasticphase_1 - 1]
            last_touching_sample_point=end_elasticphase_1
            poly_function_phase1_elastic, coefficients_phase1_elastic = fit_linear_regression(x_phase1_for_fit_elastic, y_phase1_for_fit_elastic)
            y_values_phase1_elastic = poly_function_phase1_elastic(x_values_fit)
            plt.plot(x_values_fit, y_values_phase1_elastic, linewidth=2)
            equation_phase1 = f'F = {coefficients_phase1_elastic[0]:.2f} ΔL + {coefficients_phase1_elastic[1]:.2f}'
            plt.text(-0.15, 1.24, f'{label_phase4}: {equation_phase4}\n{label_phase1}: {equation_phase1}',
                     fontsize=font_size, color='white', backgroundcolor='darkblue', transform=ax.transAxes, ha='left', va='top',
                     bbox=dict(facecolor='darkblue', alpha=1))
            elastic_part_all_x.append(x[:end_elasticphase_1])
            elastic_part_all_y.append(y[:end_elasticphase_1])
        ## Phase 1 analysis (Approach phase) - complex case: plastic deformation, one or several drops
        
        else:
            label_phase1_elastic, label_phase1_plastic, label_phase4 = "Approach elastic", "Approach plastic", "Release"

            ## Elastic part
            if ( indenter_type=="flat_punch"   ):
                indice_between_elastic_plastic = np.where(np.diff(x[y_phase_1]) == np.diff(x[y_phase_1]).max())[0][0]
            else:
                indice_between_elastic_plastic = inflection_point.min()
            end_elasticphase_1 = indice_between_elastic_plastic - 2
            start_plasticphase_1 = indice_between_elastic_plastic +3
            start_fit_1 = np.where(y[y_phase_1] > y[y_phase_1][:indice_between_elastic_plastic].max() * fact_drop)[0][0]
            y_phase1_for_fit_elastic = y[y_phase_1][start_fit_1:end_elasticphase_1]
            x_phase1_for_fit_elastic = x[y_phase_1][start_fit_1:end_elasticphase_1]
            poly_function_phase1_elastic, coefficients_phase1_elastic = fit_linear_regression(x_phase1_for_fit_elastic, y_phase1_for_fit_elastic)
            y_values_phase1_elastic = poly_function_phase1_elastic(x_values_fit)
            plt.plot(x_values_fit, y_values_phase1_elastic, linewidth=2)
            equation_phase1_elastic = f'F = {coefficients_phase1_elastic[0]:.2f} ΔL + {coefficients_phase1_elastic[1]:.2f}'
            elastic_part_all_x.append(x[:end_elasticphase_1])
            elastic_part_all_y.append(y[:end_elasticphase_1])
        
            ## Plastic part
            end_plastic = y_phase_1[-1]
            if ((i_test == 6) and ( indenter_type=="flat_punch"   )):
                y_phase1_for_fit_plastic_comb = y[y_phase_1][start_plasticphase_1:start_plasticphase_1 + 100]
                x_phase1_for_fit_plastic_comb = x[y_phase_1][start_plasticphase_1:start_plasticphase_1 + 100]
            else: 
                y_phase1_for_fit_plastic_comb = y[y_phase_1][start_plasticphase_1:end_plastic]
                x_phase1_for_fit_plastic_comb = x[y_phase_1][start_plasticphase_1:end_plastic]
                
            if len(y_phase_4) == 0:
                last_touching_sample_point = (x[y_phase_1][end_plastic], y[y_phase_1][end_plastic])
                checker_retour = False; start_plasticphase_1 += 1
            else:                    
                last_touching_sample_point=indice_between_elastic_plastic
                checker_retour = True   

            if indenter_type=="cube_corner":
                if i in (58, 11, 16):                    threshold = -2
                else:                                    threshold = -1.5
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=np.RankWarning)
                    coefficients_phase1_plastic_all,poly_function_phase1_plastic_all,x_values_phase1_plastic_all,y_values_phase1_plastic_all= get_alldrops_fits_coeficients_and_functions(
                        x_phase1_for_fit_plastic_comb,y_phase1_for_fit_plastic_comb,k=1, plot_debug=False, checker_retour=checker_retour, threshold=threshold)
                plt.text(-0.15, 1.24, f'{label_phase4}: {equation_phase4}\n{label_phase1_elastic}: {equation_phase1_elastic}',fontsize=font_size, color='white', backgroundcolor='darkblue', 
                         transform=ax.transAxes, ha='left', va='top',bbox=dict(facecolor='darkblue', alpha=1))
            elif     indenter_type=="flat_punch"     :
         
                poly_function_phase1_plastic, coefficients_phase1_plastic = fit_linear_regression(x_phase1_for_fit_plastic_comb, y_phase1_for_fit_plastic_comb)
                
                x_values_fit = np.linspace(get_x_fromy_linear(0, coefficients_phase1_plastic), x_phase1_for_fit_plastic_comb.max(), 20)        
                y_values_phase1_plasticurve_fit = poly_function_phase1_plastic(x_values_fit)

                # Plot the fitted lines
                plt.plot(x_values_fit, y_values_phase1_plasticurve_fit,color="red" ,linewidth=2)
                plt.plot(x_phase1_for_fit_plastic_comb, y_phase1_for_fit_plastic_comb,'.',markersize=1,color="red" ,linewidth=2)
                # Add the fit equations as text annotations
                equation_phase1_elastic = f'F = {coefficients_phase1_elastic[0]:.2f} 	ΔL + {coefficients_phase1_elastic[1]:.2f}'
                equation_phase1_plastic = f'F = {coefficients_phase1_plastic[0]:.2f} 	ΔL + {coefficients_phase1_plastic[1]:.2f}'
                plt.text(-0.15, 1.24, f'{label_phase4}: {equation_phase4}\n{label_phase1_elastic}: {equation_phase1_elastic}\n{label_phase1_plastic}: {equation_phase1_plastic}',
                         fontsize=font_size, color='white', backgroundcolor='darkblue', transform=ax.transAxes, ha='left', va='top',
                         bbox=dict(facecolor='darkblue', alpha=1))
                x_values_phase1_plastic_all=[x_phase1_for_fit_plastic_comb]
                y_values_phase1_plastic_all=[y_phase1_for_fit_plastic_comb]
                coefficients_phase1_plastic_all=[coefficients_phase1_plastic]

        plt.ylim(-y.max() * 0.1, y.max() * 1.05)
        plt.title(i_part + " test " + str(i_test),fontsize=font_size)
        plt.xlabel("Displacement (nm)",fontsize=font_size)
        plt.ylabel("Force ($\\mu$N)",fontsize=font_size)
        plt.legend(ncols=1, loc='best',fontsize=font_size)

        elastic_limit.append(y_phase1_for_fit_elastic.max())
        slopes_elastic.append(coefficients_phase1_elastic[0])
        coeficents_elastic.append(coefficients_phase1_elastic)
        last_touching_sample_point_allpart.append(last_touching_sample_point)
        try:  # Drops exist
            plastic_part_all_x.append(x_values_phase1_plastic_all)
            plastic_part_all_y.append(y_values_phase1_plastic_all)
            Force_drops_max.append([i.max() for i in y_values_phase1_plastic_all])
            Force_drops_min.append([i[5] for i in y_values_phase1_plastic_all])
            slopes_drops.append(np.array(coefficients_phase1_plastic_all)[:, 0])
            distance_drops.append([i.max() - i.min() for i in x_values_phase1_plastic_all])
        except:
            plastic_part_all_x.append([[]])
            plastic_part_all_y.append([[]])
            Force_drops_max.append([10**10])
            Force_drops_min.append([10**10])
            slopes_drops.append([10**10])
            distance_drops.append([10**10])
            print("Problem during extracting results for plastic part")
        try:
            slopes_release.append(coefficients_phase4[0])
        except:
            slopes_release.append(10**10)
            print("Problem during extracting results for release part")
        
        # Attempt to delete variables only if they are defined
        try:
            del coefficients_phase4
        except NameError:
            pass
        
        try:
            del coefficients_phase1_plastic_all
        except NameError:
            pass
        
        try:
            del x_values_phase1_plastic_all
        except NameError:
            pass
        
        try:
            del y_values_phase1_plastic_all
        except NameError:
            pass
    
    plt.tight_layout()
    plt.savefig(path_pwd + "Forece_vs_displacement.png", dpi=150)
    plt.show()
    # Construct the results dictionary
    results = {
        'Displacement_cal': Displacement_cal,
        'ForceA': ForceA,
        'test_part': test_part,
        'test': test,
        'phase': phase,
        'path_pwd': path_pwd,
        'elastic_limit': elastic_limit,
        'slopes_elastic': slopes_elastic,
        'last_touching_sample_point_allpart': last_touching_sample_point_allpart,
        'elastic_part_all_x': elastic_part_all_x,
        'elastic_part_all_y': elastic_part_all_y,
        'plastic_part_all_x': plastic_part_all_x,
        'plastic_part_all_y': plastic_part_all_y,
        'Force_drops_max': Force_drops_max,
        'Force_drops_min': Force_drops_min,
        'slopes_drops': slopes_drops,
        'distance_drops': distance_drops,
        'slopes_release': slopes_release,
        'coeficents_elastic': coeficents_elastic

        
    }    
    # Return the results as a dictionary
    return results  
#------------------------------------------------------------------------------------------------------------
def dict_compare(dict1, dict2):
    """
    Compare two dictionaries recursively, handling NumPy arrays and complex data structures correctly.
    Returns a list of keys where the dictionaries differ.

    Args:
        dict1 (dict): The first dictionary.
        dict2 (dict): The second dictionary.

    Returns:
        list: A list of keys where the dictionaries differ, or an empty list if they are identical.
    """
    differing_keys = []
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())

    # Check if the keys are the same
    if keys1 != keys2:
        differing_keys.extend(keys1 ^ keys2)  # Symmetric difference of keys

    # Compare the values for each key
    for key in keys1.intersection(keys2):
        value1 = dict1[key]
        value2 = dict2[key]

        # If both values are NumPy arrays, compare them element-wise
        if isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray):
            if not np.array_equal(value1, value2):
                differing_keys.append(key)

        # If both values are dictionaries, compare them recursively
        elif isinstance(value1, dict) and isinstance(value2, dict):
            sub_differing_keys = dict_compare(value1, value2)
            if sub_differing_keys:
                differing_keys.append((key, sub_differing_keys))

        # Otherwise, compare the values recursively using np.array_equal
        else:
            if not np.array_equal(value1, value2):
                differing_keys.append(key)

    return differing_keys
#------------------------------------------------------------------------------------------------------------
def load_and_save_as_dict_femtotools(npz_file_path):
    """
    Load data from an .npz file and save it as a dictionary.

    Args:
        npz_file_path (str): Path to the .npz file.
        output_dict_path (str): Path to save the dictionary.

    Returns:
        dict: The loaded data as a dictionary.
    """
    # Load the .npz file
    npz_file = np.load(npz_file_path, allow_pickle=True)

    # Get the keys of the NpzFile object
    keys = npz_file.keys()

    # Create a dictionary to store the data
    data_dict = {}

    # Iterate over the keys and load the data
    for key in keys:
        data = npz_file[key]
        if isinstance(data, np.ndarray) and data.ndim == 0:
            # If the data is a 0-dimensional array, convert it to a Python object
            data = data.item()
        data_dict[key] = data
    return data_dict["arr_0"]
# Function to fit a linear regression using np.polyfit
#------------------------------------------------------------------------------------------------------------
def fit_linear_regression(x, y):
    """
    Fits a linear regression using np.polyfit.

    Args:
    - x (array): Independent variable.
    - y (array): Dependent variable.

    Returns:
    - poly_function (np.poly1d): Polynomial function representing the linear regression.
    - coefficients (array): Coefficients of the linear regression.
    """
    coefficients = np.polyfit(x, y, 1)
    poly_function = np.poly1d(coefficients)
    return poly_function, coefficients
# Function to get x from y using linear regression coefficients
#------------------------------------------------------------------------------------------------------------
def get_x_fromy_linear(y, coef):
    """
    Calculates the x value corresponding to a given y value using linear regression coefficients.

    Args:
    - y (float): Dependent variable value.
    - coef (array): Coefficients of the linear regression.

    Returns:
    - x (float): Independent variable value corresponding to the given y.
    """
    # y = a*x + b, where coef = (a, b)
    return (y - coef[1]) / coef[0]
# Function to reorder a list based on a target order
#------------------------------------------------------------------------------------------------------------
def get_reordering_indices(original_list, target_order):
    """
    Reorders a list based on a target order.

    Args:
    - original_list (list): Original list to be reordered.
    - target_order (list): Target order specifying the desired order of elements.

    Returns:
    - reordering_indices (list): Indices for reordering the original list to match the target order.
    """
    original_array = np.array(original_list)
    target_order_array = np.array(target_order)
    value_to_index = {value: index for index, value in enumerate(original_array)}
    reordering_indices = [value_to_index[value] for value in target_order_array]
    return reordering_indices
# Function to fit and plot data
#------------------------------------------------------------------------------------------------------------
def fit_and_plot_data(x_data, y_data, x_fit, target_slope, slope_tolerance, color):
    """
    Fits the data with constrained slope and plots the results.

    Args:
    - x_data (array): Independent variable data.
    - y_data (array): Dependent variable data.
    - x_fit (array): X values for fitting (optional).
    - target_slope (float): Target slope around which to constrain.
    - slope_tolerance (float): Tolerance for slope.
    - color (str): Color for plotting.

    Returns:
    - slope_intercept (tuple): Tuple containing slope and intercept of the fitted line.
    - x_values_fit_res (array): X values for the fitted line.
    - y_values_fit_res (array): Y values for the fitted line.
    """
    # Fit the data with constrained slope
    results=fit_with_min_residual(x_data, y_data, target_slope, slope_tolerance) 
    
    best_method, slope,intercept, res_fit = fit_with_min_residual(x_data, y_data, target_slope, slope_tolerance)
    slope_intercept=(slope,intercept)
    #print("The best method is " + str(best_method))

    if slope is None:
        return None, None

    # Create a polynomial function with the fitted parameters
    poly_function = lambda x_val: slope * x_val + intercept
    
    # Generate x values for plotting
    if x_fit:
        x_values_fit = x_fit
    else:
        x_values_fit = np.linspace(get_x_fromy_linear(0, (slope, intercept)), x_data.max(), 20)
        
    y_values_fit = poly_function(x_values_fit)

    # Plot the fitted line
    plt.plot(x_values_fit, y_values_fit, linewidth=1, color=color)
    plt.plot(x_data, y_data, ".", markersize=1, color=color)
    return slope, intercept, x_values_fit, y_values_fit
#------------------------------------------------------------------------------------------------------------
def get_alldrops_fits_coeficients_and_functions(x_phase1_for_fit_plastic_comb,  
                                                y_phase1_for_fit_plastic_comb,
                                                x_plot=None, k=1, plot_debug=False,
                                                checker_retour=True, threshold=-3):
    """
    Get coefficients and functions for all drops in plastic based on the given data.
    
    Args:
    - x_phase1_for_fit_plastic_comb (array-like): Array of x values for fitting plastic drops.
    - y_phase1_for_fit_plastic_comb (array-like): Array of y values for fitting plastic drops.
    - x_plot (array-like): Array of x values for plotting.
    - k (int): Multiplication factor for x values.
    - plot_debug (bool): Whether to plot debug information.
    - checker_retour (bool): Checker return value.
    - threshold (float): Threshold for detecting rising and falling edges.
    
    Returns:
    - coefficients_phase1_plastic_all (list): List of tuples containing coefficients for all drops.
    - poly_function_phase1_plastic_all (list): List of polynomial functions for all drops.
    - x_phase1_plastic_all (list): List of x values for all drops.
    - y_phase1_plastic_all (list): List of y values for all drops.
    """
    # Adjust multiplication factor
    k = 1
    
    # Get x and y values of drops in plastic
    x_phase1_for_fit_plastic_all, y_phase1_for_fit_plastic_all = get_x_y_of_drops_plastic(
        k * x_phase1_for_fit_plastic_comb,
        k * y_phase1_for_fit_plastic_comb,
        plot_debug=plot_debug, checker_retour=checker_retour, threshold=-3
    )
    
    # Parameters
    target_slope = 5  # Target slope around which to constrain
    slope_tolerance = 6  # Tolerance for slope
    list_colors = get_color_list(len(x_phase1_for_fit_plastic_all))
    poly_function_phase1_plastic_all = []
    coefficients_phase1_plastic_all, y_phase1_plastic_all, x_phase1_plastic_all = [], [], []
    
    # Iterate over all drops
    for i_drop in range(len(x_phase1_for_fit_plastic_all)):
        x_data = x_phase1_for_fit_plastic_all[i_drop]
        y_data = y_phase1_for_fit_plastic_all[i_drop]
        
        # Skip drops with insufficient data points
        if len(x_data) <= 15:
            continue

        # Fit and plot data for the drop
        slope,intercept, x_fit_res,y_fit_res = fit_and_plot_data(x_data, y_data, x_plot, target_slope, slope_tolerance, list_colors[i_drop])
        slope_intercept=(slope,intercept)
        if slope_intercept[0]:
            
            coefficients_phase1_plastic_all.append(slope_intercept)
            poly_function_phase1_plastic_all.append(np.poly1d(slope_intercept))
            x_phase1_plastic_all.append(x_data)
            y_phase1_plastic_all.append(y_data)
    
    return coefficients_phase1_plastic_all, poly_function_phase1_plastic_all, x_phase1_plastic_all, y_phase1_plastic_all
#------------------------------------------------------------------------------------------------------------
def get_x_y_of_drops_plastic(x, y, plot_debug=False, checker_retour=True, threshold=-3):
    """
    Get the x and y values of drops in plastic based on the given data.
    
    Args:
    - x (array-like): Array of x values.
    - y (array-like): Array of y values.
    - plot_debug (bool): Whether to plot debug information.
    - checker_retour (bool): Checker return value.
    - threshold (float): Threshold for detecting rising and falling edges.
    
    Returns:
    - drops_results_x (list): List of arrays containing x values of drops.
    - drops_results_y (list): List of arrays containing y values of drops.
    """
    # Find minimum and maximum x values
    x_min, xmax = x.min(), x.max()
    
    # Find the index of the minimum x value
    trig_min = np.where(x == x_min)[0][0]
    
    try:
        list_slope_local, mean_slope_all = [], []
        
        # Iterate over a range of x values
        for i in np.arange(x_min + 1, x.max(), 1.8):
            # Find the maximum index where x is less than i
            trig_max = np.where(x < i)[0].max()
            
            # Extract the corresponding x and y values
            x_new, y_new = x[trig_min:trig_max], y[trig_min:trig_max]
            
            # Fit a linear regression
            coef = np.polyfit(x_new, y_new + random.normal(size=len(y_new), scale=0.50, loc=0), 1)
            
            # Calculate the local slope
            list_slope_local.append(coef[0])
            
            # Check if the slope exceeds a threshold
            if coef[0] > 4:
                if coef[0] > 1:
                    # Update trig_min and calculate the mean slope
                    trig_min = trig_max
                    mean_slope_all.append(array(list_slope_local).mean())
                    list_slope_local = []
                    continue
        
        # Calculate the new y values based on the mean slope
        new_y_phase1 = (y - x * mean_slope_all[0])
        
    except:
        # If an exception occurs, set new_y_phase1 to y - 5 * x
        new_y_phase1 = (y - x * 5.)
    
    # Find rising and falling edges
    threshold = threshold
    rising_edges = list(np.where(np.gradient((new_y_phase1).astype(int)) < threshold)[0])
    
    # Add the last index if checker_retour is False
    if not checker_retour:
        rising_edges.append((len(new_y_phase1) - 1))
    
    # Plot debug information if plot_debug is True
    if plot_debug:
        plt.figure(figsize=(15, 9))
        plt.figure(figsize=(15, 9))
        for i in range(len(rising_edges) - 1):
            start_index = rising_edges[i] + 1
            end_index = rising_edges[i + 1] - 2
            plt.plot(x[start_index:end_index], new_y_phase1[start_index:end_index], ".", markersize=1., label=i)
            plt.title("Debug plot ")
            plt.xlabel("Disp ")
            plt.ylabel(f"F- {np.round(mean_slope_all[0], 1)}Disp ")
        plt.legend(ncols=4)
        plt.show()
    
    # Extract x and y values of drops based on rising edges
    drops_results_x, drops_results_y = [], []
    for i in range(len(rising_edges) - 1):
        if rising_edges[i + 1] - rising_edges[i] < 5:
            continue
        
        start_index = rising_edges[i] + 1
        end_index = rising_edges[i + 1] - 2
        x_local, y_local = x[start_index:end_index], y[start_index:end_index]
        if len(x_local) == 0:
            continue
        drops_results_x.append(x_local)
        drops_results_y.append(y_local)
    
    return drops_results_x, drops_results_y
#------------------------------------------------------------------------------------------------------------
def constrained_polyfit(x, y, target_slope, slope_tolerance, method="poly"):
    """
    Fit the given data with constrained slope using either the "poly" method or the "scipy" method.
    
    Args:
    - x (array-like): Array of x values.
    - y (array-like): Array of y values.
    - target_slope (float): Target slope around which to constrain.
    - slope_tolerance (float): Tolerance for slope.
    - method (str): Method for fitting the data ("poly" or "scipy").
    
    Returns:
    - slope (float): Fitted slope.
    - intercept (float): Fitted intercept.
    - residuals (array-like): Residuals of the fitted data.
    """
    if method == "poly":
        # Fit the data with linear regression
        poly_function, coefficients = fit_linear_regression(x, y)
        residuals = ((y - poly_function(x)**2)).sum()
        slope, intercept = coefficients
        return slope, intercept, residuals
    
    if method == "scipy":
        # Define the objective function for fitting
        def objective(params):
            a, b = params
            residuals = y - (a * x + b)
            return np.sum(residuals ** 2)
    
        # Initial guess for the slope and intercept
        initial_guess = [target_slope, 0]
        
        # Define the bounds for the slope and intercept
        bounds = [(target_slope - slope_tolerance, target_slope + slope_tolerance), (-np.inf, 0.)]
        
        # Perform the minimization with constraints
        result = minimize(objective, initial_guess, bounds=bounds)
        
        # Get the fitted parameters
        slope, intercept = result.x
        residuals =( (y - (slope * x + intercept))**2).sum()
        return slope, intercept, residuals
#------------------------------------------------------------------------------------------------------------
def fit_with_min_residual(x, y, target_slope, slope_tolerance):
    """
    Fit the given data with the minimum residual method, using both "poly" and "scipy" methods.
    
    Args:
    - x (array-like): Array of x values.
    - y (array-like): Array of y values.
    - target_slope (float): Target slope around which to constrain.
    - slope_tolerance (float): Tolerance for slope.
    
    Returns:
    - best_method (str): The method that gives the minimum residual ("np.poly" or "scipy.minimize").
    - slope (float): Fitted slope.
    - intercept (float): Fitted intercept.
    - residuals (array-like): Residuals of the fitted data.
    """
    # Fit the data with "poly" method
    slope_poly, intercept_poly, residuals_poly = constrained_polyfit(x, y, target_slope, slope_tolerance, method="poly")
    
    # Fit the data with "scipy" method
    slope_scipy, intercept_scipy, residuals_scipy = constrained_polyfit(x, y, target_slope, slope_tolerance, method="scipy")
    
    # Check which method gives the minimum residual
    if np.sum(residuals_poly ** 2) < np.sum(residuals_scipy ** 2):
        return "np.poly", slope_poly, intercept_poly, residuals_poly
    else:
        return "scipy.minimize", slope_scipy, intercept_scipy, residuals_scipy
#------------------------------------------------------------------------------------------------------------
def process_femtotools_data(datadir,stifnessmachine=21400,force_max_platine=200,plot_alltest=True,plot_selected_test=True,noise_level=0.75,
                            save_figure=None,wanted_test_part=None,wanted_test=None,title=''
                           ):
    def process_data(del_test_line, data):
        scan_old,ForceA, Displacement,Phase,Time= 0,[],[],[],[]
        for scan in range(1,len(del_test_line)):
            a, b = del_test_line[scan_old], del_test_line[scan]
            print(a,b)
            new_data=data [a + 1:b-1]
            #display(new_data[0:-1])
            scan_old = int(scan)
            loc_ForceA          =array(new_data ["ForceA"      ]           .values.astype(float))
            loc_Displacement    =array(new_data ["Displacement"]           .values.astype(float))
            loc_Phase           =array(new_data ["Phase"       ]           .values.astype(float))
            loc_Time            =array(new_data ["Time"        ]           .values.astype(float))
            print(len(loc_ForceA),len(loc_Displacement),len(loc_Phase),len(loc_Time))
            if  (((len(loc_ForceA)!=len(loc_Displacement)) or (len(loc_ForceA)!=len(loc_Phase))) or (len(loc_ForceA)!=len(loc_Time))):
                print(f'error in reading between ligne {a} : {b}' )
            ForceA.append          (   loc_ForceA   )
            Displacement.append    (   loc_Displacement   )
            Phase.append           (   loc_Phase   )
            Time.append            (   loc_Time   )
        ForceA.append              ( array(data     ["ForceA"      ][b + 1:   ].values.astype(float))     )
        Displacement.append        ( array(data     ["Displacement"][b + 1:   ].values.astype(float))     )
        Phase.append               ( array(data     ["Phase"       ][b + 1:   ].values.astype(float))     )
        Time.append                ( array(data     ["Time"        ][b + 1:   ].values.astype(float))     )
        return ForceA, Displacement  ,Phase, Time    
    ############# Lecture d'un fichier lammps
    ii_file,len_data_all=0,[]
    for i_file in datadir:
        lstnom=["Index","Phase","Displacement","Time","PosX","PosY","PosZ","RotA","RotB","PiezoX","ForceA","ForceB","Gripper","VoltageA","VoltageB","Temperature","SampleDistance"]
        data = pd.read_csv(i_file, delim_whitespace=True, skiprows=3, encoding='unicode_escape', on_bad_lines='skip',names=lstnom,low_memory=False);
        display(data.head())
        if ii_file==0:
            del_test_line=np.where(data["ForceA"]=="[uN]")[0]
            ForceA_source1, Displacement_source1 ,Phase_source1 ,Time_source1 = process_data(del_test_line, data)
            ForceA, Displacement ,Phase ,Time =  ForceA_source1.copy(), Displacement_source1.copy(),Phase_source1.copy(),Time_source1.copy()
        else:
            del_test_line=np.where(data["ForceA"]=="[uN]")[0]
            ForceA_source1, Displacement_source1 ,Phase_source1 ,Time_source1 = process_data(del_test_line, data)            
            ForceA.extend(ForceA_source1)
            Displacement.extend(Displacement_source1)
            Phase.extend(Phase_source1)
            Time.extend(Time_source1)
        print(len(ForceA_source1),len(Displacement_source1),len(Phase_source1),len(Time_source1))
        len_data_all.append(len(ForceA_source1))
        print(f"File {ii_file} contain {len(ForceA_source1)} ")
        ii_file+=1
    data_l=len(ForceA)
    # Elimate the electronic noise
    F_denoised=list(ForceA)
    for  scan in range(len(Displacement)):
        
        try:
            F_denoised[scan]=denoise_data_femtotools(Displacement[scan],ForceA[scan],noise_level=0.05,method='Moving Average',debug_plot=False,times_denoise=2)

            indice_first_contact=np.where((ForceA[scan])>noise_level)[0]
            displcacement_first_contact=Displacement[scan][indice_first_contact][0]-0.005
            ForceA[scan][ Displacement[scan]<displcacement_first_contact]=0
        except:
            ForceA[scan]=np.zeros_like(ForceA[scan])
            F_denoised[scan]=np.zeros_like(ForceA[scan])
    
    # Calibration of the displcement based on the yound modulus of the Tip
    Displacement_cal=[ np.array(Displacement[i])-(1/stifnessmachine)*np.array(ForceA[i]) for i in range(len(ForceA))]
    
    # The test to be plotted 
    if isinstance(wanted_test_part, (list, tuple, np.ndarray)) and isinstance(wanted_test, (list, tuple, np.ndarray)) :
        print("The selection of tests is skiped as they are provided by user")
    else:
        print("The slection of tests is based on the maximum force provided by user.");        print(f" Fmax = {force_max_platine}")
        trigger_good_test=[]
        for scan in range(len(ForceA)):#wanted_test:
            if ( ForceA[scan].max()>2 and  ForceA[scan].max()<force_max_platine):
                trigger_good_test.append(scan)
        trigger_good_test=np.array(trigger_good_test)        
        
    if plot_alltest:
        plt.figure(figsize=(25,(len(Displacement_cal)//5)*3));        i=1
        for scan in range(len(Displacement_cal)):
            plt.subplot(len(Displacement_cal)//5,6,i);            plt.title("Test "+str(scan),fontsize=20);            plt.plot(Displacement[scan],ForceA[scan],'bo',markersize=1)
            if ForceA[scan].max()>2:
                plt.xlim(Displacement[scan][np.where(F_denoised[scan]>1)[0][0]]-0.005)
            plt.xlabel('Disp (µm)');plt.ylabel('Force (µN)');     i+=1
        plt.suptitle(title, y=1.01,fontsize=20)
        plt.tight_layout()
        if save_figure:
            plt.savefig(save_figure+'Test_all.png')
        plt.show()  
        
        
        plt.figure(figsize=(25,(len(Time)//5)*3));        i=1
        for scan in range(len(Time)):
            plt.subplot(len(Time)//5,6,i);            plt.title("Test "+str(scan),fontsize=20);            plt.plot(Time[scan],ForceA[scan],'bo',markersize=1)
            if ForceA[scan].max()>2:
                plt.xlim(Time[scan][np.where(F_denoised[scan]>1)[0][0]]-0.005)
            plt.xlabel('Disp (µm)');plt.ylabel('Force (µN)');     i+=1
        plt.suptitle(title, y=1.01,fontsize=20)
        plt.tight_layout()
        if save_figure:
            plt.savefig(save_figure+'Test_all_force_vs_Time.png')
        plt.show()      
    if plot_selected_test:
        
        if isinstance(wanted_test_part, (list, tuple, np.ndarray)) and isinstance(wanted_test, (list, tuple, np.ndarray)) :
            len_raw        =int(len(wanted_test_part)//4)
            plt.figure(figsize=(20,len_raw*5));            plt.subplots_adjust(wspace=0.4, hspace=0.4); i,ii_lab=0,0
            for scan in (wanted_test):
                true_scan=scan
                if scan>data_l:
                    true_scan=scan-data_l
                plt.subplot(len_raw,5,i+1);
                plt.plot(Displacement[scan],ForceA[scan],".",markersize=1,label='raw'); plt.plot(Displacement_cal[scan],ForceA[scan],".",markersize=1,label='calibrated')
                if ForceA[scan].max()>1:
                    plt.xlim(Displacement[scan][np.where(F_denoised[scan]>1)[0][0]]-0.005,Displacement[scan].max()+0.005)
                plt.xlabel('Disp (µm)');plt.ylabel('Force (µN)');plt.grid(alpha=0.1);plt.legend();plt.title("Test "+str(true_scan)+" "+wanted_test_part[ii_lab] ,fontsize=20) ; i+=1;ii_lab+=1
            plt.suptitle(title, y=1.01,fontsize=20)
            plt.tight_layout()
            if save_figure:
                plt.savefig(save_figure+'Test_seluser_raw_and_calibration.png')
            plt.show()  
            
            plt.figure(figsize=(20,len_raw*5));plt.subplots_adjust(wspace=0.4, hspace=0.4); i,ii_lab=0,0
            for scan in (wanted_test):
                true_scan=scan
                if scan>data_l:                    true_scan=scan-data_l
                plt.subplot(len_raw,5,i+1);  plt.plot(Displacement_cal[scan],ForceA[scan],".",markersize=1)
                if ForceA[scan].max()>1:
                    plt.xlim(Displacement[scan][np.where(F_denoised[scan]>1)[0][0]]-0.005,Displacement[scan].max()+0.005)
                plt.xlabel('Disp (µm)');plt.ylabel('Force (µN)');plt.grid(alpha=0.1);plt.title("Test "+str(true_scan)+" "+wanted_test_part[ii_lab] ,fontsize=20) ;  i+=1 ;ii_lab+=1;
            plt.suptitle(title, y=1.01,fontsize=20)
            plt.tight_layout()
            if save_figure:                plt.savefig(save_figure+'Test_seluser_calibration.png')
            plt.show()  
        else:
            len_raw        =int(len(trigger_good_test)//4)            
            plt.figure(figsize=(20,len_raw*5))
            i=1
            for scan in trigger_good_test:
                plt.subplot(len_raw,5,i);   plt.title("Test "+str(scan),fontsize=20)
                plt.plot(Displacement[scan],ForceA[scan],".",markersize=1,label='raw');   plt.plot(Displacement_cal[scan],ForceA[scan],".",markersize=1,label='calibrated')   
                if ForceA[scan].max()>1:
                    plt.xlim(Displacement[scan][np.where(F_denoised[scan]>1)[0][0]]-0.005,Displacement[scan].max()+0.005)
                plt.xlabel('Disp (µm)');  plt.ylabel('Force (µN)');  plt.legend();    i+=1
            plt.suptitle(title, y=1.01,fontsize=20)
            plt.tight_layout()
            if save_figure:
                plt.savefig(save_figure+'Test_selthresholdF_raw_and_calibration.png')
            plt.show()  
            plt.figure(figsize=(20,len_raw*5))
            i=1
            for scan in trigger_good_test:
                plt.subplot(len_raw,5,i);     plt.title("Test "+str(scan),fontsize=20)
                plt.plot(Displacement_cal[scan],ForceA[scan],".",markersize=1)   
                if ForceA[scan].max()>1:
                    plt.xlim(Displacement[scan][np.where(F_denoised[scan]>1)[0][0]]-0.005,Displacement[scan].max()+0.005)
                plt.xlabel('Disp (µm)'); plt.ylabel('Force (µN)');  i+=1
            plt.suptitle(title, y=1.01,fontsize=20)
            plt.tight_layout()
            if save_figure:
                plt.savefig(save_figure+'Test_selthresholdF__calibration.png')
            plt.show()  
    for i in range(len(ForceA)):
        i_real=i#wanted_test[i]    
        print(ForceA[i_real].shape,Displacement[i_real].shape,Phase[i_real].shape,Time[i_real].shape)
    data_summary={
        'len_data_all':len_data_all,
        'ForceA':ForceA,
        'Displacement':Displacement,
        'Displacement_cal':Displacement_cal,
        'Phase':Phase,
        'Time':Time
    }
    return data_summary
#------------------------------------------------------------------------------------------------------------
def denoise_data_femtotools(x,y,method='Moving Average',noise_level=1,debug_plot=False,times_denoise=1):
    # Define denoising methods
    window=5
    y_noisy= y+ np.random.normal(0, noise_level, len(y))
    denoising_methods = {
        'No Denoising': y_noisy,
        'Polynomial Regression': make_pipeline(PolynomialFeatures(degree=3), Ridge()),
        'Moving Average': np.convolve(y_noisy, np.ones(window) / window, mode='same'),
        'Median Filter': np.convolve(y_noisy, np.ones(window), mode='same') / window
    }
    
    # Evaluate denoising methods
    for method_name, denoising_method in denoising_methods.items():
        if method_name!=method:continue
        for i_times in range(times_denoise):
            if hasattr(denoising_method, 'fit'):
                y_denoised = denoising_method.fit(x.reshape(-1, 1), y_noisy).predict(x.reshape(-1, 1))
            else:
                y_denoised = denoising_method
        
            mse = mean_squared_error(y, y_denoised)    
            if debug_plot:
                print(f"{method_name} - MSE: {mse:.4f}")
                # Plot the results
                plt.figure(figsize=(10, 6))
                plt.plot(x, y,".",markersize=1, label='True Signal')
                plt.plot(x, y_denoised,".",markersize=1, label='Denoised Signal')
                plt.title(f"{method_name} Denoising")
                plt.xlabel("x");                    plt.ylabel("y");                    plt.legend()
                plt.xlim(x[np.where(y_denoised>1)[0][0]]-0.005)
                
                plt.show()
            y=array(y_denoised) 
    return y_denoised
#------------------------------------------------------------------------------------------------------------
def get_xmin_nonzero_fromnoiseddata_femtotools(x,y):
    y_denoised=denoise_data_femtotools(x,y,noise_level=0.05,method='Moving Average',debug_plot=False,times_denoise=2)
    indice_first_contact=np.where(y_denoised>1)[0]
    x_min=x[indice_first_contact][0]-0.005
    return x_min

#------------------------------------------------------------------------------------------------------------
def save_data_hdf5(data, filename):
    with h5py.File(filename, 'w') as f:
        for key, value in data.items():
            if isinstance(value, list):
                # Create a group for the list
                group = f.create_group(key)
                for i, element in enumerate(value):
                    if isinstance(element, str):
                        # Convert string to fixed-length string data type
                        element = np.string_(element)
                    group.create_dataset(f"{i}", data=element)
            else:
                if isinstance(value, str):
                    # Convert string to fixed-length string data type
                    value = np.string_(value)
                # Create a dataset for the single element
                f.create_dataset(key, data=value)
    f.close()
#------------------------------------------------------------------------------------------------------------
