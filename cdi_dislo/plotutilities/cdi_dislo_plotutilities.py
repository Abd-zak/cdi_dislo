"""
=================================================================================
Visualization Utilities for 3D Data, Diffraction, and Mechanical Properties
=================================================================================

This script provides a collection of visualization functions for analyzing 
and plotting various scientific datasets, including:

1. **3D Data Visualization:**
   - `visualize_3d_data()`: Interactive 3D volume rendering using PyVista.
   - `plot_interactive_slices()`: Interactive 2D slice visualization with widgets.
   - `plot_3d_array()`, `plot_3d_array_ipv()`: 3D scatter plots of array data.

2. **Diffraction and Phase Analysis:**
   - `plot_summary_difraction()`: Visualizes diffraction data in different projections.
   - `plot_qxqyqzI()`, `plotqxqyqzi_imshow()`: Diffraction intensity maps in Q-space.
   - `plot_def_()`: Displays deformation maps from phase data.

3. **Mechanical Properties and Strain Analysis:**
   - `plot_mechanical_properties()`: Plots stiffness vs. yield force/displacement.
   - `plot_data_disp()`, `plot_data_disp_projection()`: Strain visualization of nanoparticles.

4. **Statistical and Evolutionary Analysis:**
   - `plot_stast_evolution_id27()`: Evolution of statistical metrics over experiments.
   - `plot_xyz_com()`, `plot_combined_xyz()`: Tracks center-of-mass displacement over experiments.
   - `plot_phases()`, `plot_X_vs_Y_allpart_or_onebyone()`: Tracks phase evolution over pressure.

5. **Miscellaneous Functions:**
   - `format_vector()`: Formats numerical vectors for readable output.
   - `get_color_list()`: Generates distinct color palettes for plots.
   - `summary_slice_plot_abd()`: Produces summary slice plots with colorbar annotations.

=================================================================================
Usage:
Import the required functions into your script and provide appropriate data arrays 
as input. Most functions support optional parameters for customization.
=================================================================================
"""

"""
=================================================================================
Suggestions for Future Improvements
=================================================================================

1. **Performance Optimization:**
   - Consider optimizing large dataset visualizations using downsampling or parallel processing.
   - Use more efficient data structures (e.g., sparse matrices) for memory-intensive operations.

2. **Enhanced Interactivity:**
   - Integrate interactive visualization libraries like Dash or Bokeh for web-based plots.
   - Implement real-time updates for streaming data visualization.

3. **Improved Plot Customization:**
   - Allow users to specify colormaps, transparency, and annotation settings dynamically.
   - Add GUI-based controls (e.g., sliders, checkboxes) for enhanced user interaction.

4. **Standardization & Modularity:**
   - Convert frequently used plotting functions into a reusable Python package.
   - Implement configuration files for consistent styling and parameter settings.

5. **Error Handling & Robustness:**
   - Add exception handling for invalid inputs (e.g., non-3D data passed to a 3D plot).
   - Provide informative warnings or fallback mechanisms when data is missing or inconsistent.

6. **Documentation & Examples:**
   - Expand docstrings with usage examples and expected input/output formats.
   - Create Jupyter notebooks with sample data for reproducibility and demonstration.

7. **Automation & Batch Processing:**
   - Implement batch plotting functionality to process multiple datasets automatically.
   - Provide options to save all figures in a specified directory with standardized filenames.

=================================================================================
These suggestions aim to enhance the usability, efficiency, and versatility of the
visualization functions, making them more adaptable to different datasets and research needs.
=================================================================================
"""









# from cdi_dislo.common_imports                                     import *
from cdi_dislo.ewen_utilities.plot_utilities                      import plot_3D_projections ,plot_2D_slices_middle_one_array3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from ipywidgets import interact, widgets
import pyvista as pv
import math
import scipy.stats as stats
import warnings
import xrayutilities as xu
import time
from matplotlib.backends import backend_pdf as be_pdf
from cdi_dislo.general_utilities.cdi_dislo_utils import (  
    nan_to_zero,
    zero_to_nan,       
)
#####################################################################################################################
def mean_z_run(data):
    '''
    Docstring for mean_z_run
    
    :param data: Description
    '''
    sum_=data[0]
    for i in range(1,len(data)):
        sum_=sum_+ data[i]
        
    return sum_/len(data)
def mean_y_run(data):
    sum_=data[:,0,:]
    nb_=data.shape[1]
    for i in range(1,nb_):
        sum_=sum_+ data[:,i,:]     
    return sum_/nb_
def mean_x_run(data):
    sum_=data[:,:,0]
    nb_=data.shape[2]
    for i in range(1,nb_):
        sum_=sum_+ data[:,:,i]     
    return sum_/nb_
def std_data(data):
    
    data= data.flatten()
    data=data[data!=0]
    mean_d= np.sum(data)/len(data)
    std= np.sqrt(np.square(data-mean_d).sum() /len(data) )
    return std
def mean_data(data):
    data= data.flatten()
    data=data[data!=0]
    mean_d= np.sum(data)/len(data)
    return mean_d
def MIR_Colormap():
    cdict = {'red':  ((0.0, 1.0, 1.0),
                      (0.11, 0.0, 0.0),
                      (0.36, 0.0, 0.0),
                      (0.62, 1.0, 1.0),
                      (0.87, 1.0, 1.0),
                      (1.0, 0.0, 0.0)),
              'green': ((0.0, 1.0, 1.0),
                      (0.11, 0.0, 0.0),
                      (0.36, 1.0, 1.0),
                      (0.62, 1.0, 1.0),
                      (0.87, 0.0, 0.0),
                      (1.0, 0.0, 0.0)),
              'blue': ((0.0, 1.0, 1.0),
                      (0.11, 1.0, 1.0),
                      (0.36, 1.0, 1.0),
                      (0.62, 0.0, 0.0),
                      (0.87, 0.0, 0.0),
                      (1.0, 0.0, 0.0))}
    my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256) # type: ignore
    return my_cmap
#####################################################################################################################



def extract_coefficient_and_exponent(number):
    # Extract the exponent
    exponent = int(math.log10(np.abs(number)))
    # Calculate the coefficient
    coefficient = number / (10 ** exponent)
    return coefficient, exponent

def visualize_3d_data(
    data_array, 
    spacing=(1, 1, 1), 
    cmap="viridis", 
    colorbar_title="Data", 
    colorbar_orientation="vertical", 
    title_font_size=12, 
    label_font_size=10, 
    window_size=(800, 800), 
    plot_title="Interactive 3D Visualization"
):
    """
    Visualize 3D data with a customizable colorbar using PyVista.

    Parameters:
    - data_array (numpy.ndarray): 3D data array to visualize.
    - spacing (tuple): Grid spacing in x, y, z directions (default: (1, 1, 1)).
    - cmap (str): Colormap name (default: 'viridis').
    - colorbar_title (str): Title for the colorbar (default: 'Data').
    - colorbar_orientation (str): Orientation of the colorbar ('vertical' or 'horizontal').
    - title_font_size (int): Font size for the colorbar title.
    - label_font_size (int): Font size for the colorbar labels.
    - window_size (tuple): Window size for the plot (default: (800, 800)).
    - plot_title (str): Title of the plot window.
    """
    
    # Enable headless rendering if needed
    pv.start_xvfb()
    
    # Create the grid
    grid = pv.ImageData()
    grid.dimensions = tuple(np.array(data_array.shape) )  # Ensure correct point dimensions
    grid.spacing = spacing

    # Flatten and assign data
    grid.point_data["values"] = data_array.flatten(order="F")

    # Configure the plotter
    plotter = pv.Plotter()
    scalar_bar_args = {
        "title": colorbar_title,
        "vertical": colorbar_orientation == "vertical",
        "title_font_size": title_font_size,
        "label_font_size": label_font_size,
    }

    # Add the volume with the colorbar
    plotter.add_volume(grid, cmap=cmap, scalar_bar_args=scalar_bar_args)

    # Show the plot
    plotter.show(window_size=window_size, title=plot_title)


def plot_interactive_slices(data, default_axis="z", clim=None, cmap="Viridis"):
    """
    Create an interactive slicing plot for a 3D data array with options 
    for changing colormap, min/max values, and axis.
    
    Parameters:
    -----------
    data : numpy.ndarray
        A 3D NumPy array (e.g., 100x100x100).
    default_axis : str, optional
        The default axis along which to slice ('x', 'y', or 'z'), default is 'z'.
    clim : tuple, optional
        Tuple of (min, max) values for color scaling, default is None.
    cmap : str, optional
        The colormap to use for the plot, default is 'Viridis'.
    """
    if data.ndim != 3:
        raise ValueError("Input data must be a 3D array.")
    
    # Default color limits
    min_val, max_val = (data.min(), data.max()) if clim is None else clim

    def update(axis="z", slice_index=0, cmap=cmap, min_val=min_val, max_val=max_val):
        """Helper function to update the plot interactively."""
        if axis == "z":
            slice_data = data[:, :, slice_index]
            xlabel, ylabel = "X", "Y"
        elif axis == "y":
            slice_data = data[:, slice_index, :]
            xlabel, ylabel = "X", "Z"
        elif axis == "x":
            slice_data = data[slice_index, :, :]
            xlabel, ylabel = "Y", "Z"
        else:
            raise ValueError("Axis must be one of 'x', 'y', or 'z'.")

        # Create the plot
        fig = go.Figure(
            data=go.Heatmap(
                z=slice_data,
                colorscale=cmap,
                zmin=min_val,
                zmax=max_val,
            )
        )
        fig.update_layout(
            title=f"Slice along {axis.upper()} at index {slice_index}",
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            height=600,
            width=600,
        )
        fig.show()

    # Create interactive widgets
    interact(
        update,
        axis=widgets.Dropdown(
            options=["x", "y", "z"],
            value=default_axis,
            description="Axis",
            style={"description_width": "initial"}
        ),
        slice_index=widgets.IntSlider(
            min=0,
            max=data.shape[2 if default_axis == "z" else 1 if default_axis == "y" else 0] - 1,
            step=1,
            value=0,
            description="Slice Index",
            style={"description_width": "initial"}
        ),
        cmap=widgets.Dropdown(
            options=["Viridis", "Plasma", "Cividis", "Turbo", "Greys"],
            value=cmap,
            description="Colormap",
            style={"description_width": "initial"}
        ),
        min_val=widgets.FloatText(
            value=min_val,
            description="Min Value",
            style={"description_width": "initial"}
        ),
        max_val=widgets.FloatText(
            value=max_val,
            description="Max Value",
            style={"description_width": "initial"}
        ),
    )
    

def format_vector(vector, decimal_places=4):
    formatted = []
    for num in vector:
        num = float(num)
        rounded = round(num, decimal_places)
        if np.abs(rounded) < 0.001 or np.abs(rounded) >= 1000:
            str_num = f"{rounded:.{decimal_places}e}"
        else:
            str_num = f"{rounded:.{decimal_places}f}".rstrip('0').rstrip('.')
        formatted.append(str_num)
    return f"{', '.join(formatted)}"
def plot_summary_difraction(data_original,com_,max_,path_save=None,fig_title="",fig_save_="",f_s=20,vmin_sub=None,vmax_sub=None,vmin=None,vmax=None,box_comment_cord=[1.,1.3],eps = 0):
    """
    Plot three orthogonal log-mean projections with per-subplot colorbars,
    annotate COM and MAX, and optionally save separate 2D projection images.

    Parameters
    ----------
    data_original : np.ndarray
        3D array (Z, Y, X) or similar. Projections use np.nanmean along axes 0,1,2.
    com_, max_ : iterable of 3 floats/ints
        (X, Y, Z) or (indexing-consistent) coordinates. Used for scatter markers.
    path_save : str or None
        Base path (without extension). If provided, figures are saved.
    fig_title : str
        Suptitle for the main 3-panel figure and titles for 2D projections.
    fig_save_ : str
        Suffix added to saved filenames.
    f_s : int/float
        Font size for labels and ticks.
    vmin, vmax : float or None
        Color scaling passed to imshow for all three subplots (shared scaling).
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable  # needed by add_colorbar_subplot

    fig, ax = plt.subplots(1,3, figsize=(16,4)) 
    plot_3D_projections(data_original,cmap='jet', fig=fig,ax=ax,log_scale=True,log_threshold=True,fig_title=fig_title,vmin=vmin_sub,vmax=vmax_sub,colorbar=True)
    ax[0].scatter(com_[2],com_[1],marker="X",linewidth=1,s=50,alpha=0.5,color="red",label=f" COM: {com_[0]},{com_[1]},{com_[2]}")
    ax[0].scatter(max_[2],max_[1],marker="X",linewidth=1,s=50,alpha=0.5,color="black",label=f" MAX: {max_[0]},{max_[1]},{max_[2]}")
    ax[0].legend(bbox_to_anchor=box_comment_cord,fontsize=f_s/1.5)
    ax[0].set_xlabel('Z', fontsize=f_s)             # X-axis label font size
    ax[0].set_ylabel('Y', fontsize=f_s)             # Y-axis label font size
    # Set tick labels and their font sizes
    ax[0].tick_params(axis='x', labelsize=f_s)  # X-tick labels font size
    ax[0].tick_params(axis='y', labelsize=f_s)  # Y-tick labels font size
    
    ax[1].scatter(com_[2],com_[0],marker="X",s=50,color="red")
    ax[1].scatter(max_[2],max_[0],marker="X",s=50,color="black")
    ax[1].set_xlabel('Z', fontsize=f_s)             # X-axis label font size
    ax[1].set_ylabel('X', fontsize=f_s)             # Y-axis label font size
    ax[1].tick_params(axis='x', labelsize=f_s)  # X-tick labels font size
    ax[1].tick_params(axis='y', labelsize=f_s)  # Y-tick labels font size
    ax[2].scatter(com_[1],com_[0],marker="X",s=50,color="red")
    ax[2].scatter(max_[1],max_[0],marker="X",s=50,color="black")
    ax[2].set_xlabel('Y', fontsize=f_s)             # X-axis label font size
    ax[2].set_ylabel('X', fontsize=f_s)             # Y-axis label font size
    ax[2].tick_params(axis='x', labelsize=f_s)  # X-tick labels font size
    ax[2].tick_params(axis='y', labelsize=f_s)  # Y-tick labels font size

    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')

    [ax[i].set_axis_off() for i in range(3)]
    fig.suptitle(fig_title, fontsize=f_s,y=0.95)
    plt.tight_layout()
    if path_save:
        plt.savefig(path_save+"_3D_"+fig_save_,dpi=300)
    plt.show()


    # --- compute projections (log of mean along each axis) ---
    # Add a tiny epsilon to avoid log(0)
    proj_xy = np.log(np.nanmax(data_original, axis=0) + eps)  # (Y, X), label Z on x in your convention
    proj_xz = np.log(np.nanmax(data_original, axis=1) + eps)  # (Z, X)
    proj_yz = np.log(np.nanmax(data_original, axis=2) + eps)  # (Z, Y)

    
    # --- X projection ---
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(proj_xy, vmin=vmin, vmax=vmax, cmap='jet', aspect='auto')
    ax.set_title(fig_title, fontsize=f_s)
    ax.axis('off')
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=f_s*0.8)
    cbar.set_label("Intensity (a.u.)", fontsize=f_s)
    
    fig.tight_layout()
    plt.savefig(path_save + "_xproj_" + fig_save_, dpi=300, bbox_inches="tight") # type: ignore
    plt.show()

    # --- Y projection ---
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(proj_xz, vmin=vmin, vmax=vmax, cmap='jet', aspect='auto')
    ax.set_title(fig_title, fontsize=f_s)
    ax.axis('off')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=f_s*0.8)
    cbar.set_label("Intensity (a.u.)", fontsize=f_s)
    
    fig.tight_layout()
    plt.savefig(path_save + "_yproj_" + fig_save_, dpi=300) # type: ignore
    plt.show()
    
    # --- Z projection ---
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(proj_yz, vmin=vmin, vmax=vmax, cmap='jet', aspect='auto')
    ax.set_title(fig_title, fontsize=f_s)
    ax.axis('off')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=f_s*0.8)
    cbar.set_label("Intensity (a.u.)", fontsize=f_s)
    fig.tight_layout()
    plt.savefig(path_save + "_zproj_" + fig_save_, dpi=300) # type: ignore
    plt.show()


    return

def plot_mechanical_properties(slopes_elastic, f_max_elastic_x, f_max_elastic_y, test_part, 
                               include_sapphire=False, plot_fit=False, r_squared_threshold=0.85,
                               save_plot=False):
    """
    Plot mechanical properties of materials.
    
    Parameters:
    - slopes_elastic: array of stiffness values (µN/nm)
    - f_max_elastic_x: array of yield displacement values (nm)
    - f_max_elastic_y: array of yield force values (µN)
    - test_part: array of part names
    - include_sapphire: boolean, whether to include sapphire samples (default False)
    - plot_fit: boolean, whether to plot linear fit (default False)
    - r_squared_threshold: float, minimum R-squared value to plot fit (default 0.85)
    - save_plot: boolean, whether to save the plot as a PNG file (default False)
    - save_directory: string, directory to save the plot (default is current directory)
    """
    
    # Ensure all arrays are numpy arrays and convert units
    stiffness = np.array(slopes_elastic) * 1000  # µN/nm to N/m
    yield_displacement = np.array(f_max_elastic_x) / 1e9  # nm to m
    yield_force = np.array(f_max_elastic_y) / 1e6  # µN to N
    test_part = np.array(test_part)

    # Filter arrays if not including sapphire
    if not include_sapphire:
        mask = ['Saphire' not in str(p) for p in test_part]
        stiffness = stiffness[mask]
        yield_displacement = yield_displacement[mask]
        yield_force = yield_force[mask]
        test_part = test_part[mask]

    # Create a color map
    unique_parts = np.unique(test_part)
    color_map = plt.cm.get_cmap('tab20')
    colors = [color_map(i) for i in range(len(unique_parts))]
    color_dict = dict(zip(unique_parts, colors))

    # Create the figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    axes = [ax1, ax2, ax3]
    titles = ['Stiffness vs Yield Force',
              'Stiffness vs Yield Displacement',
              'Yield Displacement vs Yield Force']
    x_labels = ['Stiffness (N/m)', 'Stiffness (N/m)', 'Yield Displacement (m)']
    y_labels = ['Yield Force (N)', 'Yield Displacement (m)', 'Yield Force (N)']
    x_data = [stiffness, stiffness, yield_displacement]
    y_data = [yield_force, yield_displacement, yield_force]

    for ax, title, xlabel, ylabel, x, y in zip(axes, titles, x_labels, y_labels, x_data, y_data):
        for part in unique_parts:
            mask = test_part == part
            ax.scatter(x[mask], y[mask], label=part, color=color_dict[part], s=50)
        
        if plot_fit:
            # Perform linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            r_squared = r_value**2 # type: ignore
            
            # Only plot the fit if R-squared is above the threshold
            if r_squared > r_squared_threshold:
                line = slope * x + intercept
                ax.plot(x, line, color='r', 
                        label=f'Fit: y = {slope:.2e}x + {intercept:.2e}\nR² = {r_squared:.3f}')
            else:
                print(f"R-squared ({r_squared:.3f}) below threshold for {title}. Fit not plotted.")
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(title='Parts', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        
        # Use scientific notation for axes if numbers are very small
        ax.ticklabel_format(style='sci', scilimits=(0,0), axis='both')

    # Adjust layout
    plt.tight_layout()

    # Save the plot if save_plot is True
    if save_plot:
        # Save the figure
        plt.savefig(save_plot, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {save_plot}")

    plt.show()

    if plot_fit:
        # Print the fit parameters for each plot
        for i, (x, y, title) in enumerate(zip(x_data, y_data, titles), 1):
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            r_squared = r_value**2 # type: ignore
            print(f"\nFit parameters for {title}:")
            print(f"Slope: {slope:.4e}")
            print(f"Intercept: {intercept:.4e}")
            print(f"R-squared: {r_squared:.4f}")
            print(f"P-value: {p_value:.4e}")
            if r_squared <= r_squared_threshold:
                print(f"R-squared below threshold. Fit not plotted.")



def get_color_list(nb):
    from itertools import combinations
    import random as py_random

       
    # Get colors from tab10 and Paired colormaps
    tab10_colors = plt.cm.tab10(np.linspace(0, 0.8, 8))  # type: ignore # Use only 80% of the colormap
    paired_colors = plt.cm.Paired(np.linspace(0, 1, 6)) # type: ignore
    
    # Get XKCD color names
    xkcd_colors = list(mcolors.XKCD_COLORS.keys())
    
    # Custom muted colors in hex format
    custom_colors = ['#8B7D6B', '#556B2F', '#8B8386']  
    
    # Combine colors from different sources
    colors_comb = list(tab10_colors) + list(paired_colors) + xkcd_colors + custom_colors
    
    # Filter out light colors based on brightness threshold
    brightness_threshold = 0.6  # Adjust as needed
    dark_colors = [color for color in colors_comb if mcolors.to_rgba(color)[0] * 0.299 +
                                                   mcolors.to_rgba(color)[1] * 0.587 +
                                                   mcolors.to_rgba(color)[2] * 0.114 < brightness_threshold]
    
    # Convert hex colors to RGB tuples
    rgb_colors = [mcolors.to_rgb(color) for color in dark_colors]
    
    # Calculate the pairwise color differences
    color_diffs = np.zeros((len(rgb_colors), len(rgb_colors)))
    for i, j in combinations(range(len(rgb_colors)), 2):
        color_diffs[i, j] = np.linalg.norm(np.array(rgb_colors[i]) - np.array(rgb_colors[j]))
    
    # Filter out colors that are too similar
    threshold = 5  # Adjust as needed
    unique_colors = set()
    filtered_colors = []
    for i, color in enumerate(rgb_colors):
        if all(color_diffs[i, j] > threshold for j in range(i)):
            filtered_colors.append(dark_colors[i])
            unique_colors.add(color)

    colors =py_random.sample(filtered_colors, nb)
    return colors   
def get_x_y_of_drops_plastic(x,y,plot_debug=False):
    from numpy import linspace,random
    x_min,xmax=x.min(),x.max()
    trig_min=np.where(x==x_min)[0][0]
    list_slope_local,mean_slope_all=[],[]
    for i in linspace(x_min+1,x_min+50,50):
        trig_max=np.where(x<i)[0].max()
        x_new,y_new=x[trig_min:trig_max],(y[trig_min:trig_max] )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=np.RankWarning)
            coef=np.polyfit(x_new, y_new+random.normal(size=len(y_new),scale=0.250,loc=0), 1)
        poly_function = np.poly1d(coef)
        list_slope_local.append(coef[0])
        if coef[0]>4:
            if coef[0]>1:
                trig_min=trig_max
                mean_slope_all.append(np.array(list_slope_local).max())
                list_slope_local=[]
                continue
    new_y_phase1=(y-x*mean_slope_all[0])
    threshold=-4
    # Find rising and falling edges
    rising_edges  = np.where(np.diff((new_y_phase1 ).astype(int))< threshold)[0]
    #debug plot
    if plot_debug:
        plt.figure(figsize=(15,9))
        #plt.plot(x_phase1_for_fit_plastic_comb,new_y_phase1_for_fit_plastic_comb,".")
        plt.figure(figsize=(15,9))
        for i in range(len(rising_edges)-1):
            start_index = rising_edges[i]+1
            end_index = rising_edges[i+1]-2
            #print(f"Square wave {i+1}: Start index = {start_index}, End index = {end_index}")
            plt.plot(x[start_index:end_index],new_y_phase1[start_index:end_index],".",markersize=1.,label=i)
            plt.title("Debug plot ")
            plt.xlabel("Disp ")
            plt.ylabel(f"F- {np.round(mean_slope_all[0],1)}Disp ")
        plt.legend(ncols=4)
        plt.show()
    drops_results_x,drops_results_y=[],[]
    for i in range(len(rising_edges)-1):
        start_index = rising_edges[i]+1
        end_index = rising_edges[i+1]-2
        x_local,y_local=x[start_index:end_index],y[start_index:end_index]
        if len(x_local)==0:
            continue
        drops_results_x.append(x_local)
        drops_results_y.append(y_local)
    return drops_results_x,drops_results_y

def plot_qxqyqzI(qx_ ,qy_ ,qz_,Int,i_scan ):
    #%%affichage donnees interpolees
    fig1=plt.figure(1,figsize=(20,4))
    plt.subplot(1,3,1)
    
    plt.contourf(qx_,qy_,xu.maplog(Int.sum(axis=2)).T,150,cmap='jet')
    plt.xlabel(r"Q$_x$ ($1/\AA$)")
    plt.ylabel(r"Q$_y$ ($1/\AA$)")
    plt.colorbar()
    plt.axis('tight')
    plt.subplot(1,3,2)
    plt.contourf(qx_,qz_,xu.maplog(Int.sum(axis=1)).T,150,cmap='jet')
    plt.xlabel(r"Q$_X$ ($1/\AA$)")
    plt.ylabel(r"Q$_Z$ ($1/\AA$)")
    plt.colorbar()
    plt.axis('tight')
    plt.subplot(1,3,3)
    plt.contourf(qy_,qz_,xu.maplog(Int.sum(axis=0)).T,150,cmap='jet')
    plt.xlabel(r"Q$_Y$ ($1/\AA$)")
    plt.ylabel(r"Q$_Z$ ($1/\AA$)")
    plt.colorbar()
    plt.axis('tight')
    fig1.suptitle(r"Scan_"+str(i_scan))
    plt.show()
def plotqxqyqzi_imshow(Int,i_scan ,vmax=5):
    f_s=16
    fig2=plt.figure(1,figsize=(20,4))
    ax=plt.subplot(1,3,1)
    im=ax.imshow(xu.maplog(Int.sum(axis=2)),vmin=0,vmax=vmax,cmap='jet')
    plt.xlabel('indice[1]', fontsize=f_s)
    plt.ylabel('indice[0]', fontsize=f_s)
    plt.axis('tight')
    ax.invert_yaxis()
    ax.tick_params(labelsize=f_s)
    
    plt.grid(alpha=0.01)
    ax=plt.subplot(1,3,2)
    im=ax.imshow(xu.maplog(Int.sum(axis=1)),cmap='jet',vmin=0,vmax=vmax)
    plt.xlabel('indice[2]', fontsize=f_s)
    plt.ylabel('indice[0]', fontsize=f_s)
    plt.axis('tight')
    plt.grid(alpha=0.01)
    ax.invert_yaxis()
    ax.tick_params(labelsize=f_s)

    ax=plt.subplot(1,3,3)
    im=ax.imshow(xu.maplog(Int.sum(axis=0)),cmap='jet',vmin=0,vmax=vmax)
    plt.xlabel('indice[2]', fontsize=f_s)
    plt.ylabel('indice[1]', fontsize=f_s)
    cbar=plt.colorbar(im,ax=ax)
    cbar.ax.tick_params(labelsize=14)
    ax.invert_yaxis()
    plt.axis('tight')
    ax.tick_params(labelsize=f_s)

    plt.grid(alpha=0.01)
    fig2.suptitle(r"Scan "+str(i_scan), fontsize=f_s)
    return fig2
def plot_data_xyzprojcontour(x,y,z,data_, scan, mode,param):
    f_s=16
    n_frames = 3
    x_title_pos=0.5
    n_sbplots_cols, n_sbplots_row = 3, 1
    fig, axs = plt.subplots(nrows=n_sbplots_row,
                            ncols=n_sbplots_cols,
                            figsize=(7 * n_sbplots_cols, 5 * n_sbplots_row))

    i_plot = 0
    ax = plt.subplot(n_sbplots_row, n_sbplots_cols, i_plot + 1)
    plt.contourf(z,y,data_.sum(axis=2).T,150,cmap='jet') 
    plt.colorbar(ax=ax)
    ax.set_xlabel("Z", fontsize=f_s, labelpad=10)
    ax.set_ylabel("Y", fontsize=f_s, labelpad=10)
    ax.tick_params(labelsize=f_s)
    plt.grid(None)
    ax.set_title('Mean X ', fontsize=f_s)
    plt.axis('tight')

    i_plot = 1
    ax = plt.subplot(n_sbplots_row, n_sbplots_cols, i_plot + 1)
    plt.contourf(z,x,data_.sum(axis=1).T,150,cmap='jet') 
    plt.colorbar(ax=ax)
    ax.set_xlabel("Z", fontsize=f_s, labelpad=10)
    ax.set_ylabel("X", fontsize=f_s, labelpad=10)
    ax.tick_params(labelsize=f_s)
    plt.grid(None)
    ax.set_title('Mean Y ', fontsize=f_s)
    plt.axis('tight')

    i_plot = 2
    ax = plt.subplot(n_sbplots_row, n_sbplots_cols, i_plot + 1)
    plt.contourf(y,x,data_.sum(axis=0).T,150,cmap='jet') 
    plt.colorbar(ax=ax)
    ax.set_xlabel("Y", fontsize=f_s, labelpad=10)
    ax.set_ylabel("X", fontsize=f_s, labelpad=10)
    ax.tick_params(labelsize=f_s)
    plt.grid(None)
    ax.set_title('Mean Z ', fontsize=f_s)
    plt.axis('tight')

    fig.suptitle('scan: ' + scan + ' | Mode ' + str(mode)+ ' | ' +param,
                 fontweight='bold',
                 fontsize=f_s,
                 horizontalalignment='center',
                 y=0.95)
    return fig
def plot_data_xyzproj(data_, scan, mode,param):
    f_s=24
    n_frames = 3
    x_title_pos=0.5
    n_sbplots_cols, n_sbplots_row = 3, 1
    fig, axs = plt.subplots(nrows=n_sbplots_row,
                            ncols=n_sbplots_cols,
                            figsize=(7 * n_sbplots_cols, 7 * n_sbplots_row))

    i_plot = 0
    ax = plt.subplot(n_sbplots_row, n_sbplots_cols, i_plot + 1)
    im = ax.imshow(data_.sum(axis=2), cmap='jet')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Z", fontsize=f_s, labelpad=10)
    ax.set_ylabel("Y", fontsize=f_s, labelpad=10)
    ax.tick_params(labelsize=f_s)
    plt.grid(alpha=0.5)
    ax.invert_yaxis()
    ax.set_title('Mean X ', fontsize=f_s)
    plt.axis('tight')

    i_plot = 1
    ax = plt.subplot(n_sbplots_row, n_sbplots_cols, i_plot + 1)
    im = ax.imshow(data_.sum(axis=1), cmap='jet')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Z", fontsize=f_s, labelpad=10)
    ax.set_ylabel("X", fontsize=f_s, labelpad=10)
    ax.tick_params(labelsize=f_s)
    plt.grid(alpha=0.5)
    ax.invert_yaxis()
    ax.set_title('Mean Y ', fontsize=f_s)
    plt.axis('tight')

    i_plot = 2
    ax = plt.subplot(n_sbplots_row, n_sbplots_cols, i_plot + 1)
    im = ax.imshow(data_.sum(axis=0), cmap='jet')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Y", fontsize=f_s, labelpad=10)
    ax.set_ylabel("X", fontsize=f_s, labelpad=10)
    ax.tick_params(labelsize=f_s)
    plt.grid(alpha=0.5)
    ax.invert_yaxis()
    ax.set_title('Mean Z ', fontsize=f_s)
    plt.axis('tight')

    fig.suptitle('scan: ' + scan + ' | Mode ' + str(mode)+ ' | ' +param,
                 fontweight='bold',
                 fontsize=f_s,
                 horizontalalignment='center',
                 y=1)
    return fig             
def corr_plot(abs_data):
    fig, axes = plt.subplots(1,3)

    plt.tight_layout(pad=0.5, w_pad=0.01, h_pad=2.0)
    plt.grid(alpha=0.5)
    ax1 = plt.subplot(111) # creates first axis
    i1 = ax1.imshow(abs_data,cmap='hot'
    #                                ,extent=(xmin,xmax,ymin,ymax)               
                   )
    #plt.xticks(np.arange(0,180,25) )
    #plt.yticks(np.arange(0,180,25) )
    cb1=plt.colorbar(i1,ax=ax1)
    cb1.ax.tick_params(labelsize=18)
    plt.grid(alpha=0.5)    
    plt.show()
    return fig
def plot_selcted_runs_y(index_best_run_scans,data_scans,data_allscans_runs,data_allscans_LLK,param,file_save):
    pdf = be_pdf.PdfPages(file_save)
    for i_scan in index_best_run_scans.keys():
        start = time.time()
        n_frames = len(index_best_run_scans[i_scan])
        print('**********' + i_scan + ' with ' + str(n_frames) + 'selected runs' +'**********')
        n_sbplots_cols = 3
        if ((n_frames % n_sbplots_cols) == 0):           n_sbplots_row = (n_frames // n_sbplots_cols)
        else:                                            n_sbplots_row = (n_frames // n_sbplots_cols) + 1
        fig, axs = plt.subplots(nrows=n_sbplots_row,ncols=n_sbplots_cols,figsize=(16 * n_sbplots_cols, 10 * n_sbplots_row),)
        i_plot=0
        for i in index_best_run_scans[i_scan]:
            x_title_pos=0.5
            f_s=64
            font_size=34
            if n_frames==2:
                    x_title_pos=0.38
                    f_s=30
                    font_size=30
            ax = plt.subplot(n_sbplots_row, n_sbplots_cols, i_plot + 1)
            if param=='phi':
                im = ax.imshow(mean_y_run(data_scans[i_scan][i]),cmap='jet')
            else:
                im = ax.imshow(mean_y_run(data_scans[i_scan][i]),cmap='jet',vmin=0,vmax=6)
            plt.colorbar(im, ax=ax)
            ax.set_xlabel("X", fontsize=font_size, labelpad=10)
            ax.set_ylabel("Y", fontsize=font_size, labelpad=10)
            ax.tick_params(labelsize=font_size)
            ax.set_title('Run:' + str(int(data_allscans_runs[i_scan][i])) +'/ LLK:' + str(int(data_allscans_LLK[i_scan][i] * 10000) /10000) +
                         '/ mean '+param + ' :' + str(int(mean_data(data_scans[i_scan][i]) *10000) / 10000) + '/ $std_{\\rho}$ :' 
                         +str(int(std_data(data_scans[i_scan][i]) *10000) / 10000),fontsize=font_size,pad=40)
            plt.grid(0.1) # type: ignore

            i_plot+=1
        if n_sbplots_row*n_sbplots_cols!=n_frames:
            empty_fig_nb=n_sbplots_row*n_sbplots_cols-n_frames
            for i_empty in range(int(empty_fig_nb)):
                ax = plt.subplot(n_sbplots_row, n_sbplots_cols, n_frames+i_empty+1)
                ax.set_visible(False)
        fig.suptitle('scan: ' + i_scan +' | comparaison of '+ param+' for selected runs',fontweight='bold',fontsize=f_s,horizontalalignment='center',x = x_title_pos,y=1) # type: ignore
        plt.grid(None)
        pdf.savefig(fig, dpi=150)
        plt.show()
        end = time.time()
        print(str(int(((end - start) / 60) * 100) / 100) + 'min')
    pdf.close()
def plot_3d_array(data, step=2):
    z, y, x = np.where(data != 0)
    cdata = data[data != 0]
    
    # Downsample the coordinates and color data
    z = z[::step]
    y = y[::step]
    x = x[::step]
    cdata = cdata[::step]
    
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot scatter
    ax.scatter3D(x, y, z, c=cdata, cmap='jet', linewidth=1) # type: ignore
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z') # type: ignore
    
    # Add colorbar
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Value')
    
    plt.show()
    return fig
def plot_3d_array_ipv(data, norm=True):
    """
    Plots a 3D array using ipyvolume.

    Parameters:
    - data (ndarray): 3D array to be plotted.
    - norm (bool): Whether to normalize the data before plotting. Default is True.

    Returns:
    - None
    """
    import ipyvolume as ipv
    import matplotlib.pyplot as plt

    plt.figure()
    if norm:
        data = data / data.max()
    ipv.quickvolshow(data, level=[0.35], opacity=0.03, level_width=0.1, data_min=0, data_max=1)
    plt.show()
def plot_data_disp(data_,lim):
    f_s = 12
    n_frames = 6
    x_title_pos = 0.5
    cmap='jet'
    n_sbplots_cols, n_sbplots_row = 6,3
    fig, axs = plt.subplots(nrows=n_sbplots_row,ncols=n_sbplots_cols, figsize=(4 * n_sbplots_cols, 3 * n_sbplots_row))
    for i_frame in range(6):
        i_plot=0
        ax = plt.subplot(n_sbplots_row, n_sbplots_cols,i_frame*3+ i_plot + 1)
        im = ax.imshow(data_[i_frame].sum(axis=2), cmap=cmap,vmin=-lim,vmax=lim)
        ax.set_xlabel("Z", labelpad=5)
        ax.set_ylabel("Y", labelpad=5)
        plt.grid(None)
        ax.invert_yaxis()
        ax.set_title('Mean X ')
        plt.axis('tight')
        i_plot = 1
        ax = plt.subplot(n_sbplots_row, n_sbplots_cols,i_frame*3+ i_plot + 1)
        im = ax.imshow(data_[i_frame].sum(axis=1), cmap=cmap,vmin=-lim,vmax=lim)
        ax.set_xlabel(" Z", labelpad=5)
        ax.set_ylabel("X", labelpad=5)
        plt.grid(None)
        ax.invert_yaxis()
        ax.set_title('Mean Y ')
        plt.axis('tight')
        i_plot = 2
        ax = plt.subplot(n_sbplots_row, n_sbplots_cols,i_frame*3+ i_plot + 1)
        im = ax.imshow(data_[i_frame].sum(axis=0), cmap=cmap,vmin=-lim,vmax=lim)
        color_bar=plt.colorbar(im, ax=ax)
        ax.set_xlabel("Y", labelpad=5)
        ax.set_ylabel("X", labelpad=5)
        color_bar.set_label("$\\epsilon_{}$".format(i_frame+1), labelpad=5,fontsize=18)
        plt.grid(None)
        ax.invert_yaxis()
        ax.set_title('Mean Z ')
        plt.axis('tight')
    fig.suptitle("Strain of platinium nanoparticle",
                     fontweight='bold',
                     horizontalalignment='center',
                     y=1)
    return fig
def plot_data_disp_projection(data_,plan,max_v,param):
    f_s = 12
    n_frames = 6
    x_title_pos = 0.5
    cmap='jet'
    n_sbplots_cols, n_sbplots_row = 6,3
    fig, axs = plt.subplots(nrows=n_sbplots_row,ncols=n_sbplots_cols, figsize=(4 * n_sbplots_cols, 3 * n_sbplots_row))
    for i_frame in range(6):
        i_plot=0
        ax = plt.subplot(n_sbplots_row, n_sbplots_cols,i_frame*3+ i_plot + 1)
        im = ax.imshow(data_[i_frame][:,:,plan],cmap=cmap,vmin=-max_v,vmax=max_v)
        ax.set_xlabel("Z", labelpad=5)
        ax.set_ylabel("Y", labelpad=5)
        plt.grid(None)
        ax.invert_yaxis()
        ax.set_title('Mean X ')
        plt.axis('tight')
        i_plot = 1
        ax = plt.subplot(n_sbplots_row, n_sbplots_cols,i_frame*3+ i_plot + 1)
        im =  ax.imshow(data_[i_frame][:,plan],cmap=cmap,vmin=-max_v,vmax=max_v)
        ax.set_xlabel(" Z", labelpad=5)
        ax.set_ylabel("X", labelpad=5)
        plt.grid(None)
        ax.invert_yaxis()
        ax.set_title('Mean Y ')
        plt.axis('tight')
        i_plot = 2
        ax = plt.subplot(n_sbplots_row, n_sbplots_cols,i_frame*3+ i_plot + 1)
        im =ax.imshow(data_[i_frame][plan],cmap=cmap,vmin=-max_v,vmax=max_v)
        color_bar=plt.colorbar(im, ax=ax)
        ax.set_xlabel("Y", labelpad=5)
        ax.set_ylabel("X", labelpad=5)
        color_bar.set_label("$\\epsilon_{}$".format(i_frame+1), labelpad=5,fontsize=18)
        plt.grid(None)
        ax.invert_yaxis()
        ax.set_title('Mean Z ')
        plt.axis('tight')
    fig.suptitle("Strain of platinium nanoparticle " + param,
                     fontweight='bold',
                     horizontalalignment='center',
                     y=1)
    return fig
def plot_selcted_runs_x(index_best_run_scans,data_scans,data_allscans_runs,data_allscans_LLK,param,file_save):
    pdf = be_pdf.PdfPages(file_save)
    for i_scan in index_best_run_scans.keys():
        start = time.time()
        n_frames = len(index_best_run_scans[i_scan])
        print('**********' + i_scan + ' with ' + str(n_frames) + 'selected runs' +'**********')
        n_sbplots_cols = 3
        if ((n_frames % n_sbplots_cols) == 0):           n_sbplots_row = (n_frames // n_sbplots_cols)
        else:                                            n_sbplots_row = (n_frames // n_sbplots_cols) + 1
        fig, axs = plt.subplots(nrows=n_sbplots_row,ncols=n_sbplots_cols,figsize=(16 * n_sbplots_cols, 10 * n_sbplots_row),)
        i_plot=0
        for i in index_best_run_scans[i_scan]:
            x_title_pos=0.5
            f_s=64
            font_size=34
            if n_frames==2:
                    x_title_pos=0.38
                    f_s=30
                    font_size=30
            ax = plt.subplot(n_sbplots_row, n_sbplots_cols, i_plot + 1)
            if param=='phi':
                im = ax.imshow(mean_x_run(data_scans[i_scan][i]),cmap='jet')
            else:
                im = ax.imshow(mean_x_run(data_scans[i_scan][i]),cmap='jet',vmin=0,vmax=6)
            plt.colorbar(im, ax=ax)
            ax.set_xlabel("X", fontsize=font_size, labelpad=10)
            ax.set_ylabel("Y", fontsize=font_size, labelpad=10)
            ax.tick_params(labelsize=font_size)
            ax.set_title('Run:' + str(int(data_allscans_runs[i_scan][i])) +'/ LLK:' + str(int(data_allscans_LLK[i_scan][i] * 10000) /10000) + '/ mean '+
                         param + ' :' + str(int(mean_data(data_scans[i_scan][i]) *10000) / 10000) + '/ $std_{\\rho}$ :' +
                         str(int(std_data(data_scans[i_scan][i]) *10000) / 10000),fontsize=font_size,pad=40)
            plt.grid(0.1) # type: ignore

            i_plot+=1
        if n_sbplots_row*n_sbplots_cols!=n_frames:
            empty_fig_nb=n_sbplots_row*n_sbplots_cols-n_frames
            for i_empty in range(int(empty_fig_nb)):
                ax = plt.subplot(n_sbplots_row, n_sbplots_cols, n_frames+i_empty+1)
                ax.set_visible(False)
        fig.suptitle('scan: ' + i_scan +' | comparaison of '+ param+' for selected runs',fontweight='bold',fontsize=f_s,horizontalalignment='center',x = x_title_pos,y=1) # type: ignore
        plt.grid(None)
        pdf.savefig(fig, dpi=150)
        plt.show()
        end = time.time()
        print(str(int(((end - start) / 60) * 100) / 100) + 'min')
    pdf.close()
def plot_def_(d_delta_phi_fit,plan,cmap='hot'):
    fig, axs = plt.subplots(nrows=1,ncols=3, figsize=(5* 3, 3 ))
    ax = plt.subplot(1, 3, 1)
    im = ax.imshow(d_delta_phi_fit[:,:,plan],cmap=cmap)
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Z", labelpad=10)
    ax.set_ylabel("Y", labelpad=10)
    ax.invert_yaxis()
    plt.grid(alpha=0.5)    
    ax = plt.subplot(1, 3, 2)
    im0 = ax.imshow(d_delta_phi_fit[:,plan],cmap=cmap)
    plt.colorbar(im0, ax=ax)
    ax.set_xlabel("Z" , labelpad=10)
    ax.set_ylabel("X" , labelpad=10)
    ax.invert_yaxis()
    plt.grid(alpha=0.5)    
    ax = plt.subplot(1, 3, 3)
    im1 = ax.imshow(d_delta_phi_fit[plan],cmap=cmap)
    plt.colorbar(im1, ax=ax)
    ax.set_xlabel("Y", labelpad=10)
    ax.set_ylabel("X", labelpad=10)
    ax.invert_yaxis()
    plt.grid(alpha=0.5)
    plt.axis('tight')
def plot_phases(pressure___sel,pressure___sel1,pressure___sel2,z_max_sel,z_com_sel, 
                z_max_sel1, z_com_sel1,z_max_sel2 ,
                z_com_sel2,save_file_name ,ind
               ,max__=True,com___=False):
    f_s=10
    if ind==1 :        
        title__="Detector horizontal  axis VS Pressure"
        plot_='indice['+str(ind)+']'
    if ind==2:        
        title__="Detector vertical  axis VS Pressure"
        plot_='indice['+str(ind)+']'
    if ind==3:
        title__="Distance particle faisceaux VS Pressure"
        plot_="Distance"     
    fig=plt.figure(figsize=(16,7))
    ax=plt.subplot(1,3,1)
    if max__:
        plt.plot(pressure___sel,z_max_sel,"+",label='max')
    if com___:
        plt.plot(pressure___sel,z_com_sel,"+",label='com')
    plt.ylabel(plot_, fontsize=f_s) # type: ignore
    plt.xlabel('Pressure', fontsize=f_s)
    plt.axis('tight')
    ax.invert_yaxis()
    ax.tick_params(labelsize=f_s)
    plt.grid(alpha=0.01)
    plt.legend(fontsize=f_s)
    ax.set_title("First phase")    
    ax=plt.subplot(1,3,2)
    if max__:
        plt.plot(pressure___sel1,z_max_sel1,"+",label='max')
    if com___:
        plt.plot(pressure___sel1,z_com_sel1,"+",label='com')
    plt.xlabel('Pressure', fontsize=f_s)
    plt.ylabel(plot_, fontsize=f_s) # type: ignore
    plt.axis('tight')
    ax.invert_yaxis()
    ax.tick_params(labelsize=f_s)
    plt.grid(alpha=0.01)
    plt.legend(fontsize=f_s)
    ax.set_title("Second phase")    
    ax=plt.subplot(1,3,3)
    if max__:
        plt.plot(pressure___sel2,z_max_sel2,"+",label='max')
    if com___:
        plt.plot(pressure___sel2,z_com_sel2,"+",label='com')
    plt.ylabel(plot_, fontsize=f_s) # type: ignore
    plt.xlabel('Pressure', fontsize=f_s)
    plt.axis('tight')
    ax.invert_yaxis()
    ax.tick_params(labelsize=f_s)
    plt.grid(alpha=0.01)
    plt.legend(fontsize=f_s)
    ax.set_title("Third phase")    
    fig.suptitle(title__, fontsize=16) # type: ignore
    plt.savefig(save_file_name,dpi=100)
    plt.show()
def plot_xyz_com(com_values, scan, axis_labels, plot_norm=False, f_s=16,
                 rot_abs=-45, hspace=0.4, file_name_save=None, show_plot=True, filter_zero_value=False,
                 dpi=150, norm_values=None):
    num_rows = len(com_values[0])

    fig = plt.figure(figsize=(40, 80))
    plt.subplots_adjust(hspace=hspace)

    def plot_single_subplot(ax, com_values, axis_label,scan):
        for i_row in range(num_rows):
            scan_list_str = scan[i_row].tolist()
            com_row = com_values[i_row].tolist()
            # Filter non-zero coordinates
            if filter_zero_value:
                non_zero_indices = np.where(np.array(com_row) != 0)[0]
                com_values = np.array(com_row)[non_zero_indices]
                scan_list_str = np.array(scan_list_str)[non_zero_indices]
            plt.plot(scan_list_str, com_row, "o-", label=f'Pt {i_row + 1}')
        plt.xlabel('Pressure ', fontsize=f_s)
        plt.ylabel(f'{axis_label}', fontsize=f_s)
        plt.axis('tight')
        ax.tick_params(labelsize=f_s)
        plt.xticks(rotation=rot_abs)
        plt.grid(alpha=0.01)
        plt.legend(fontsize=f_s)
        plt.title(f"All phases {axis_label}_com", fontsize=f_s * 2)

    # Plot for x_com
    ax = plt.subplot(5, 1, 1)
    plot_single_subplot(ax, com_values[0], axis_labels[0],scan)

    # Plot for y_com
    ax = plt.subplot(5, 1, 2)
    plot_single_subplot(ax, com_values[1], axis_labels[1],scan)

    # Plot for z_com
    ax = plt.subplot(5, 1, 3)
    plot_single_subplot(ax, com_values[2], axis_labels[2],scan)

    # Plot for norm
    if plot_norm:
        ax = plt.subplot(5, 1, 4)
        if norm_values is None:
            norm_values_i_pt = np.sqrt(np.array(
                com_values[0])**2 + np.array(
                com_values[1])**2 + np.array(
                com_values[2])**2)
        else:
                # norm_values_i_pt=norm_values[i_pt]
                # if filter_zero_value:
                #     norm_values_i_pt = np.array(norm_values_i_pt)[non_zero_indices]  
                pass  # type: ignore
                    
        plot_single_subplot(ax, norm_values_i_pt, axis_labels[3],scan) # type: ignore

    if file_name_save:
        fig.savefig(file_name_save, dpi=dpi)

    if show_plot:
        plt.show()

    return fig

def plot_combined_xyz(com_values, scan_values, axis_labels, 
                      f_s=16, rot_abs=-45, fig_size=(20,150),plot_norm=False,
                      file_name_save=None,filter_zero_value=False,Normalize_parametre=False,
                      show_plot=True,dpi=150, norm_values=None):
    num_pts = len(com_values[0])

    fig, axes = plt.subplots( num_pts,1, figsize=fig_size)
    plt.subplots_adjust(hspace=0.9,wspace=0.2)

    for i_pt in range(num_pts):
        
        for i_coord in range(3):
            axis_labels_i_pt=axis_labels[i_coord]
            scan_row = scan_values[i_pt]
            com_row = com_values[i_coord][i_pt]
            if Normalize_parametre:
                max_val=np.round(com_row.max())
                com_row=com_row/max_val
                axis_labels_i_pt=axis_labels_i_pt+"/"+str(max_val)
                
            
            # Filter non-zero coordinates
            if filter_zero_value:
                non_zero_indices = np.where(np.array(com_row) != 0)[0]
                com_row = np.array(com_row)[non_zero_indices]
                scan_row = np.array(scan_row)[non_zero_indices]
            axes[i_pt].plot(scan_row, com_row, "o-",
                            label=f'{axis_labels_i_pt}')
        if plot_norm:
            if norm_values is None:
                norm_values_i_pt = np.sqrt(np.array(
                    com_values[0])**2 + np.array(
                    com_values[1])**2 + np.array(
                    com_values[2])**2)
            else:
                norm_values_i_pt=norm_values[i_pt]
                if filter_zero_value:
                    norm_values_i_pt = np.array(norm_values_i_pt)[non_zero_indices]    # type: ignore
            if Normalize_parametre:
                max_val=np.round(norm_values_i_pt.max())
                norm_values_i_pt=norm_values_i_pt/max_val
                axis_labels_i_pt=axis_labels[3]+"/"+str(max_val)
                
            axes[i_pt].plot(scan_row, norm_values_i_pt, "o-", # type: ignore
                            label=f'{axis_labels[3]}')
        axes[i_pt].set_xlabel('Scan', fontsize=f_s)
        axes[i_pt].set_ylabel('Coordinate', fontsize=f_s)
        axes[i_pt].tick_params(labelsize=f_s*0.5)
        axes[i_pt].grid(alpha=0.01)
        axes[i_pt].legend(fontsize=f_s)
        axes[i_pt].set_title(f'Pt {i_pt + 1}', fontsize=f_s )
        axes[i_pt].tick_params(axis='x', rotation=rot_abs)

    plt.tight_layout()
    if file_name_save:
        fig.savefig(file_name_save, dpi=dpi)

    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig

 
def plot_combined_single_coordinate(com_values, scan_values, axis_label, f_s=16, 
                                    rot_abs=-45, fig_size=(15, 7),plot_marker="--",marker_size=1,
                                    file_name_save=None, show_plot=True, dpi=150, filter_non_zero=True):
    num_pts = len(com_values)

    fig, ax = plt.subplots(figsize=fig_size)
    plt.subplots_adjust(hspace=0.9, wspace=0.2)

    for i_pt in range(num_pts):
        com_row = com_values[i_pt]
        scan_row = scan_values[i_pt]

        # Filter non-zero coordinates if required
        if filter_non_zero:
            non_zero_indices = np.where(np.array(com_row) != 0)[0]
            com_values_filtered = np.array(com_row)[non_zero_indices]
            scan_values_filtered = np.array(scan_row)[non_zero_indices]
        else:
            com_values_filtered = com_row
            scan_values_filtered = scan_row

        ax.scatter(scan_values_filtered, com_values_filtered,marker=plot_marker ,s=marker_size, label=f'Pt {i_pt + 1}')

    ax.set_xlabel('Scan', fontsize=f_s)
    ax.set_ylabel(axis_label, fontsize=f_s)
    ax.tick_params(labelsize=f_s * 0.5)
    ax.grid(alpha=0.01)
    ax.legend(fontsize=f_s*0.5)
    ax.set_title(f'Evolution of {axis_label} during compression  - All Particles', fontsize=f_s)
    ax.tick_params(axis='x', rotation=rot_abs)

    plt.tight_layout()

    if file_name_save:
        fig.savefig(file_name_save, dpi=dpi)

    if show_plot:
        plt.show()
    
    return fig
def plot_3darray_as_gif_animation(data,file_save,vmin,vmax,title_fig=''
                                 ):
    import matplotlib.animation as animation
    def update(frame
              ):
        ax[0].clear()  # Clear the previous plot for XY projection
        ax[1].clear()  # Clear the previous plot for XZ projection
        ax[2].clear()  # Clear the previous plot for YZ projection
    
        ax[0].imshow(data[:, :, frame], cmap='jet', origin='lower',vmin=-.5,vmax=.5)
        ax[0].set_title(f'XY Projection, Z={frame}')
        ax[0].axis('off')
    
        ax[1].imshow(data[:, frame, :], cmap='jet', origin='lower',vmin=-.5,vmax=.5)
        ax[1].set_title(f'XZ Projection, Y={frame}')
        ax[1].axis('off')
    
        ax[2].imshow(data[frame, :, :], cmap='jet', origin='lower',vmin=-.5,vmax=.5)
        ax[2].set_title(f'YZ Projection, X={frame}')
        ax[2].axis('off')
        
        #plt.colorbar()
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title_fig, fontsize=16)
    # Function to update the plot for each frame

    
    # Set up the animation
    ani = animation.FuncAnimation(fig, update, frames=data.shape[2], interval=100) # type: ignore
    
    # Save the animation as a gif
    ani.save(file_save, writer='imagemagick',dpi=80, fps=10)
    plt.close()

def summary_slice_plot_abd(
        save: str = None, # type: ignore
        title: str = None, # type: ignore
        dpi: int = 200,
        show: bool = False,
        voxel_size: np.ndarray or list or tuple = None, # type: ignore
        isosurface: float = None, # type: ignore
        averaged_dspacing: float = None, # type: ignore
        averaged_lattice_parameter: float = None, # type: ignore
        det_reference_voxel: np.ndarray or list or tuple = None, # type: ignore
        respect_aspect=False,
        support: np.ndarray = None, # type: ignore
        single_vmin: float = None, # type: ignore
        single_vmax: float = None, # type: ignore
        phase_min_max=None,displacement_min_max=None,het_strain_min_max=None ,
        **kwargs) -> matplotlib.figure.Figure: # type: ignore
    
    """
    Summary plot of 3D arrays with central slices in xy, xz, and zy planes.     
    Parameters
    ----------
    save : str, optional
        File path to save the figure. If None, the figure is not saved. 
    title : str, optional

        Title of the figure. If None, no title is set.
    dpi : int, optional     


        Dots per inch for the saved figure. Default is 200.
    show : bool, optional
        Whether to display the figure. Default is False.
    voxel_size : array-like, optional
        Voxel size in each dimension to respect aspect ratios. If None, aspect ratios are set to 'auto'.
    isosurface : float, optional    
        Isosurface value for reference (not used in this function).
    averaged_dspacing : float, optional
        Averaged d-spacing value for reference (not used in this function).
    averaged_lattice_parameter : float, optional
        Averaged lattice parameter for reference (not used in this function).   
    det_reference_voxel : array-like, optional
        Detector reference voxel (not used in this function).
    respect_aspect : bool, optional
        Whether to respect aspect ratios based on voxel size. Default is False.
    support : ndarray, optional
        Support mask to apply to the arrays before plotting. If None, no mask is applied.
    single_vmin : float, optional
        Single minimum value for all plots.
    single_vmax : float, optional
        Single maximum value for all plots. 
    phase_min_max : tuple, optional
        Minimum and maximum values for phase plotting.
    displacement_min_max : tuple, optional
        Minimum and maximum values for displacement plotting.
    het_strain_min_max : tuple, optional
        Minimum and maximum values for heterogenous strain plotting.
    **kwargs : dict
        Key-value pairs where keys are parameter names (e.g., 'amplitude', 'phase', etc.) and values are 3D numpy arrays to be plotted.
    Returns
    -------
    matplotlib.figure.Figure
        The created matplotlib figure.
    """
    import matplotlib.pyplot as plt
    from cdiutils.plot import (get_figure_size,set_plot_configs,white_interior_ticks_labels)
    from cdiutils.plot.slice import plot_contour

    ANGSTROM_SYMBOL, _, PLOT_CONFIGS = set_plot_configs()

    # take care of the aspect ratios:
    if voxel_size is not None and respect_aspect:
        aspect_ratios = {
            "zy": voxel_size[0]/voxel_size[1],
            "zx": voxel_size[0]/voxel_size[2],
            "yx":  voxel_size[1]/voxel_size[2]

        }
    else:
        aspect_ratios = {"zy": "auto", "zx": "auto", "yx": "auto"}
    k=[1.25,1.25,1.25]
    subplots = (5, len(kwargs))
    figsize = [i for i in get_figure_size(subplots=subplots) ]
    figsize=[figsize[i]*k[i] for i in range(len(figsize))]
    figure, axes = plt.subplots(subplots[0]-2, subplots[1], figsize=figsize)
    figure.subplots_adjust(hspace=0.01, wspace=0.01, bottom=0.1)
    cb_font=12
    cb_font_2=8
    axes[0, 0].annotate(r"(xy)$_{cxi}$ slice",xy=(0.2, 0.6),xytext=(-axes[0, 0].yaxis.labelpad - 2, 0),
                        xycoords=axes[0, 0].yaxis.label,textcoords="offset points",ha="right",va="center", size=cb_font  ,rotation=90 )
    axes[1, 0].annotate(r"(xz)$_{cxi}$ slice",xy=(0.2, 0.6),xytext=(-axes[1, 0].yaxis.labelpad - 2, 0),xycoords=axes[1, 0].yaxis.label,
                textcoords="offset points",ha="right",va="center",size=cb_font,rotation=90)
    axes[2, 0].annotate(r"(zy)$_{cxi}$ slice",xy=(0.2, 0.6),xytext=(-axes[2, 0].yaxis.labelpad - 2, 0),xycoords=axes[2, 0].yaxis.label,
                textcoords="offset points",ha="right",va="center",size=cb_font,rotation=90)
    mappables = {}
    if support is not None:
        support = zero_to_nan(support)
    for i, (key, array) in enumerate(kwargs.items()):
        if support is not None and key != "amplitude":
            array = support * array
        if key in PLOT_CONFIGS.keys():
            cmap = PLOT_CONFIGS[key]["cmap"]
            # check if vmin and vmax are given or not
            if single_vmin is None or single_vmax is None:
                if support is not None:
                    if key in ("dspacing", "lattice_parameter"):
                        vmin,vmax=3.8, 4.2
                        # vmin = np.nanmin(array)
                        # vmax = np.nanmax(array)
                    elif key == "amplitude":
                        vmin = 0
                        vmax = np.nanmax(array)
                    elif key == 'phase':           
                        if phase_min_max:
                            vmin,vmax=phase_min_max
                        else:
                            vmax = np.nanmax(np.abs(array))
                            vmin = -vmax  
                    elif key == 'displacement': 
                        if displacement_min_max:
                            vmin,vmax=displacement_min_max
                        else:
                            vmax = np.nanmax(np.abs(array))
                            vmin = -vmax                           
                    elif key == 'het_strain': 
                        if het_strain_min_max:
                            vmin,vmax=het_strain_min_max
                        else: 
                            vmax = np.nanmax(np.abs(array))
                            vmin = -vmax                            
                    else:
                        vmax = np.nanmax(np.abs(array))
                        vmin = -vmax
                else:
                    vmin = PLOT_CONFIGS[key]["vmin"]
                    vmax = PLOT_CONFIGS[key]["vmax"]
            else:
                vmin = single_vmin
                vmax = single_vmax
        shape = array.shape
        axes[0, i].matshow(array[shape[0] // 2],vmin=vmin,vmax=vmax,cmap=cmap,origin="lower",aspect=aspect_ratios["yx"]) # type: ignore
        axes[1, i].matshow(array[:, shape[1] // 2, :],vmin=vmin,vmax=vmax,cmap=cmap,origin="lower",aspect=aspect_ratios["zx"]) # type: ignore
        mappables[key] = axes[2, i].matshow(np.swapaxes(array[..., shape[2] // 2], axis1=0, axis2=1),vmin=vmin,vmax=vmax, # type: ignore
                                            cmap=cmap,origin="lower",aspect=aspect_ratios["zy"]) # type: ignore
        axes[0, i].tick_params(axis='both', which='both', width=0., labelsize=8)
        axes[1, i].tick_params(axis='both', which='both', width=0., labelsize=8)
        axes[2, i].tick_params(axis='both', which='both', width=0., labelsize=8)


        if key == "amplitude":
            plot_contour(axes[0, i], support[shape[0] // 2], color="k")
            plot_contour(axes[1, i], support[:, shape[1] // 2, :], color="k")
            plot_contour(axes[2, i],np.swapaxes(support[..., shape[2] // 2], axis1=0, axis2=1),color="k")
    table_ax = figure.add_axes([0.25, -0.175, 0.5, 0.1]) # type: ignore
    table_ax.axis("tight")
    table_ax.axis("off")
    # format the data
    isosurface = round(isosurface, 3)
    averaged_dspacing = round(averaged_dspacing, 4)
    averaged_lattice_parameter = round(averaged_lattice_parameter, 4)
    voxel_size= [ round(i, 4) for i in  voxel_size]
    det_reference_voxel= [ round(i, 4) for i in  det_reference_voxel]
    col_wid=0.25
    table = table_ax.table(cellText=np.transpose([[str(voxel_size).replace("[","").replace("]","")],
            [format_vector(det_reference_voxel)],[isosurface],[averaged_dspacing],[averaged_lattice_parameter]]),
        colLabels=(
            "Voxel size (nm)",
            "Detector voxel reference",
            "Isosurface",
            f"Averaged dspacing ({ANGSTROM_SYMBOL})",
            f"Averaged lattice ({ANGSTROM_SYMBOL})"
        ),
        colWidths=[col_wid, col_wid, col_wid, col_wid, col_wid],
        loc="center",
        cellLoc="center"
    )
    table.scale(1.6, 1.6)
    table.auto_set_font_size(False)
    table.set_fontsize(cb_font_2)

    for i, key in enumerate(kwargs.keys()):
        l, _, w, _ = axes[0, i].get_position().bounds
        cax = figure.add_axes([
            l + (0.015+0.03*(i-2) if i > 1 else -0.05 if i == 0 else -0.02),
            0.93,
            w - 0.01,
            0.03
        ]) # type: ignore
        cax.set_title(PLOT_CONFIGS[key]["title"],fontsize=cb_font_2)
        figure.colorbar(mappables[key], cax=cax, extend="both", orientation="horizontal" )
        cax.tick_params(axis='x', which='major', pad=0.5,labelsize=cb_font_2)

    figure.canvas.draw()
    for i, ax in enumerate(axes.ravel()):
        #ax.set_aspect("equal")
        if ( i % len(kwargs) == 0 and list(kwargs.keys())[i % len(kwargs.keys())] == "amplitude"        ):
            white_interior_ticks_labels(ax, -7, -0)
            ax.tick_params(axis='both', which='both', labelsize=cb_font_2)
        else:
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])

    figure.suptitle(title, y=1.05,fontsize=cb_font)
    
    if show:
        plt.show()
    # save the figure
    if save:
        figure.savefig(save, dpi=dpi)
    
    return figure
def plot_3darray_slices_as_subplots(data,file_prefix, vmin, vmax,title_subplots='', title_fig='' ,proj=1
                                   ):
    import matplotlib.animation as animation
    def update_xy(frame
                 ):
        for i, ax_subplot in enumerate(ax.flat):
            ax_subplot.clear()  # Clear the previous plot for XY projection
            if proj==0:ax_subplot.imshow(data[i, frame, :, :], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
            if proj==1:ax_subplot.imshow(data[i, :,frame,  :], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
            if proj==2:ax_subplot.imshow(data[i, :,:,frame], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
            
            ax_subplot.set_title(f'Slice {frame}, Z={title_subplots[i]}')
            ax_subplot.axis('off')
            plt.axis('tight')
        plt.tight_layout()  

    fig, ax = plt.subplots(len(data)//8, 8, figsize=(16, 8))
    fig.suptitle(title_fig, fontsize=16)

    # Set up the animation
    ani_xy = animation.FuncAnimation(fig, update_xy, frames=data.shape[1], interval=100) # type: ignore

    # Save the animation as a gif
    ani_xy.save(file_prefix + '_xy_subplots.gif', writer='pillow', dpi=150, fps=5)
    #ani_xy.save(file_prefix + '_xy_subplots.gif', writer='imagemagick', dpi=100, fps=2)

    plt.close()
def plot_single_3darray_slices_as_subplots(data,file_prefix, vmin, vmax,title_fig='' ,proj=1,dpi=150, fps=5
                                          ):
    import matplotlib.animation as animation
    def update_xy(frame
                 ):
            ax.clear()  # Clear the previous plot for XY projection
            if proj==0:ax.imshow(data[frame, :, :], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
            if proj==1:ax.imshow(data[:,frame,  :], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
            if proj==2:ax.imshow(data[:,:,frame], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
            
            #ax.set_title(f'Slice {frame}, Z={title_subplots[i]}')
            ax.axis('off')
            plt.axis('tight')
            #plt.tight_layout()  

    fig, ax = plt.subplots(1,1, figsize=(6, 4))
    fig.suptitle(title_fig, fontsize=16)

    # Set up the animation
    ani_xy = animation.FuncAnimation(fig, update_xy, frames=data.shape[0], interval=100) # type: ignore

    # Save the animation as a gif
    ani_xy.save(file_prefix + '_xy_subplots.gif', writer='pillow', dpi=dpi, fps=fps)
    #ani_xy.save(file_prefix + '_xy_subplots.gif', writer='imagemagick', dpi=100, fps=2)

    plt.close()
def plot_X_vs_Y_allpart_or_onebyone(X, Y, part_name_list, save_dir_plot=None, x_label="", y_label="",
                                    fig_title="", all_in_one=True, unique_particles=None, marker="+", show_plt=True
                                   ):
    """
    Plot X vs Y for all particles in one figure or one particle per figure.

    Args:
        X (array-like): Data for the x-axis.
        Y (array-like): Data for the y-axis.
        part_name_list (array-like): List of particle names corresponding to each data point.
        save_dir_plot (str, optional): Directory to save the plot. Defaults to None.
        x_label (str, optional): Label for the x-axis. Defaults to "".
        y_label (str, optional): Label for the y-axis. Defaults to "".
        fig_title (str, optional): Title for the figure. Defaults to "".
        all_in_one (bool, optional): Whether to plot all particles in one figure (True) or plot each particle separately (False).
            Defaults to True.
        unique_particles (array-like, optional): List of unique particle names. Defaults to None (auto-determined from part_name_list).
        marker (str, optional): Marker style for plotting. Defaults to '+'.
        show_plt (bool, optional): Whether to display the plot. Defaults to True.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    if unique_particles is None:
        unique_particles = np.unique(part_name_list)

    if all_in_one:
        fig = plt.figure(figsize=(8, 5))
        for particle in unique_particles:
            x_plot = X[part_name_list == particle]
            y_plot = Y[part_name_list == particle]
            plt.plot(x_plot, y_plot, marker, label=particle)
        
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(fig_title)
        plt.legend(loc='best', ncols=3) 
        plt.tight_layout()
        if save_dir_plot:
            plt.savefig(save_dir_plot + y_label.replace(" ", "")+"_vs_"+x_label.replace(" ", "")+"_all_part.png")
        if show_plt:
            plt.show()
        return fig
    else:
        for particle in unique_particles:
            fig = plt.figure(figsize=(9, 5))
            x_plot = X[part_name_list == particle]
            y_plot = Y[part_name_list == particle]
            plt.plot(x_plot, y_plot, marker, label=particle)

            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(particle+" "+fig_title)
            plt.legend(loc='best', ncols=3) 
            plt.tight_layout()
            if save_dir_plot:
                plt.savefig(save_dir_plot + "lattice_parameter_" + particle + ".png")
            if show_plt:
                plt.show()    
        return fig # type: ignore

def plot_data_lattice_parametre_multidata(x_label, y_label, fig_title, subtitles, a_min, a_max,
              nov_2022_temp_scan, nov_2022_lattice_parametre, nov_2022_part_name_list,
              Avril_2023_temp_scan, Avril_2023_lattice_parametre, Avril_2023_part_name_list,
              Jan_2024_temp_scan, Jan_2024_lattice_parametre, Jan_2024_part_name_list,
              marker='+', show_plt=True, save_dir_plot=None
                                         ):
    
    # Define color map for particles
    color_map = {}
    all_part_name_lists = [nov_2022_part_name_list, Avril_2023_part_name_list, Jan_2024_part_name_list]
    all_part_name_array = np.concatenate(all_part_name_lists)
    unique_particles = np.unique(all_part_name_array)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_particles))) # type: ignore
    for particle, color in zip(unique_particles, colors):
        color_map[particle] = color

    # Plotting for the first dataset
    plt.figure(figsize=(16, 5))
    for i in range(3):
        ax = plt.subplot(1, 3, i+1)
        all_part_temp_lists = [nov_2022_temp_scan, Avril_2023_temp_scan, Jan_2024_temp_scan]
        all_part_y_lists = [nov_2022_lattice_parametre, Avril_2023_lattice_parametre, Jan_2024_lattice_parametre]
        
        for particle in np.unique(all_part_name_lists[i]):
            x_plot = [x for x, p in zip(all_part_temp_lists[i], all_part_name_lists[i]) if p == particle]
            y_plot = [y for y, p in zip(all_part_y_lists[i], all_part_name_lists[i]) if p == particle]
            plt.plot(x_plot, y_plot, marker, label=particle, color=color_map[particle])
        if i <= 1:
            desired_order = [['27', '100', '370', '370 RT', '800 RT', '950 RT'],
                             ['35', '300', '600', '800', '850', '850 RT']]
            ax.set_xticks(range(len(desired_order[i])))
            ax.set_xticklabels(desired_order[i])
        
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(fig_title)
        plt.legend(loc='best', ncols=3)
        plt.ylim(a_min, a_max)
        if subtitles:
            plt.suptitle(subtitles[i])
        else:
            plt.suptitle("Subplot {}".format(i+1))

    plt.tight_layout()
    if save_dir_plot:
        plt.savefig(save_dir_plot + y_label.replace(" ", "") + "_vs_" + x_label.replace(" ", "") + "_all_part.png")
    if show_plt:
        plt.show()

def conf_plot__(x_label,y_label,fig_tilte,a_min,a_max
               ):
     
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(fig_tilte)
    plt.legend(loc='best', ncols=3)
    plt.ylim(a_min, a_max)
def plot_data_single_or_multiple(x_label, y_label, fig_title, subtitles, a_min=None, a_max=None, data_sets=[], desired_order=None, marker_size=10,
              marker='+', show_plt=True, save_dir_plot=None, all_in_one=True,xyz_plot=False,norm=False,linestyle="--",linewidth=4,rotation_xtick=-45,file_type='pdf',
              f_s=16, f_s_legend=12,figsize_UNIT=(9,9)):
    
    """
    Plot data for comparison.

    Args:
        x_label (str or list of str): Label(s) for the x-axis.
        y_label (str or list of str): Label(s) for the y-axis.
        fig_title (str): Title for the entire figure.
        subtitles (list of str): List of subtitles for each subplot.
        a_min (float): Minimum value for the y-axis.
        a_max (float): Maximum value for the y-axis.
        data_sets (list of tuples, optional): List of datasets. Each tuple contains x_data, y_data, and part_name_list.
            Defaults to [[]].
        desired_order (list of lists, optional): Desired order for x-axis ticks for each subplot. 
            Each sublist corresponds to the desired order for the x-axis ticks in the corresponding subplot.
            Defaults to None.
        marker (str, optional): Marker style for plotting. Defaults to '+'.
        show_plt (bool, optional): Whether to display the plot. Defaults to True.
        save_dir_plot (str, optional): Directory to save the plot. Defaults to None.
        all_in_one (bool, optional): Whether to plot all particles in one figure (True) or plot each particle separately (False).
            Defaults to True.
        xyz_plot (bool, optional): Whether to treat the data as XYZ (True) or not (False). 
            If True, the common prefix of x_label and y_label will be used as axis labels in the plots.
            Defaults to False.
        norm (bool, optional): Whether to normalize the y-axis data. Defaults to False.

    Raises:
        ValueError: If number of subtitles does not match the number of plots, or if the length of desired_order
            does not match the number of subplots.
    """
    plt.rcParams.update({
        'font.size': f_s,
        'font.weight': 'bold',
        'axes.titlesize': f_s,
        'axes.titleweight': 'bold',
        'axes.labelsize': f_s,
        'axes.labelweight': 'bold',
        'xtick.labelsize': f_s,
        'ytick.labelsize': f_s,
        'xtick.major.width': 4.5,
        'ytick.major.width': 4.5,
        'legend.fontsize': f_s_legend,
        'legend.title_fontsize': f_s_legend,
        'figure.titlesize': f_s
    })
    from matplotlib.ticker import ScalarFormatter
    def format_ticks_scientific(ax, axis='y', font_size=16):
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))  # Use scientific notation for numbers outside these limits
    
        if axis == 'y':
            ax.yaxis.set_major_formatter(formatter)
            ax.tick_params(axis='y', labelsize=font_size)
        elif axis == 'x':
            ax.xaxis.set_major_formatter(formatter)
            ax.tick_params(axis='x', labelsize=font_size)
    # Ensure subtitles list matches the number of plots
    if xyz_plot:
        num_plots = len(data_sets[0])
    else:
        num_plots = len(data_sets)
    if (len(subtitles) != num_plots ) :
        raise ValueError("Number of subtitles must match the number of plots.")
    if norm:
        max_for_norm=(np.array([ data_sets[i_dataset][i_exp][1].max() for i_dataset in range(len(data_sets)) for i_exp in range(len(data_sets[i_dataset]))]).max())
        print()
        max_for_norm=np.round(extract_coefficient_and_exponent(max_for_norm)[0],2)*10**extract_coefficient_and_exponent(max_for_norm)[1]
    # Define color map for particles
    from itertools import combinations
    # Get colors from tab10 and Paired colormaps
    tab10_colors = plt.cm.tab10(np.linspace(0, 0.8, 8))  # type: ignore # Use only 80% of the colormap
    paired_colors = plt.cm.Paired(np.linspace(0, 1, 6)) # type: ignore
    # Get XKCD color names
    xkcd_colors = list(mcolors.XKCD_COLORS.keys())
    # Custom muted colors in hex format
    custom_colors = ['#8B7D6B', '#556B2F', '#8B8386']  
    # Combine colors from different sources
    colors_comb = list(tab10_colors) + list(paired_colors) + xkcd_colors + custom_colors
    # Filter out light colors based on brightness threshold
    brightness_threshold = 0.6  # Adjust as needed
    dark_colors = [color for color in colors_comb if mcolors.to_rgba(color)[0] * 0.299 +mcolors.to_rgba(color)[1] * 0.587 +
                                                   mcolors.to_rgba(color)[2] * 0.114 < brightness_threshold]
    # Convert hex colors to RGB tuples
    rgb_colors = [mcolors.to_rgb(color) for color in dark_colors]
    # Calculate the pairwise color differences
    color_diffs = np.zeros((len(rgb_colors), len(rgb_colors)))
    for i, j in combinations(range(len(rgb_colors)), 2):
        color_diffs[i, j] = np.linalg.norm(np.array(rgb_colors[i]) - np.array(rgb_colors[j]))
    
    # Filter out colors that are too similar
    threshold = 5  # Adjust as needed
    unique_colors = set()
    filtered_colors = []
    for i, color in enumerate(rgb_colors):
        if all(color_diffs[i, j] > threshold for j in range(i)):
            filtered_colors.append(dark_colors[i])
            unique_colors.add(color)
    if not xyz_plot:
        color_map = {}
        all_part_name_lists = []
        for data in data_sets:
            all_part_name_lists.append(data[2])
        all_part_name_array = np.concatenate(all_part_name_lists)
        unique_particles = np.unique(all_part_name_array)
        colors = filtered_colors[:len(unique_particles)] 
        for particle, color in zip(unique_particles, colors):
            color_map[particle] = color
    else:
        color_map = {}
        all_part_name_lists = []
        for i_data in range(3):
            for data in (data_sets[i_data]):
                all_part_name_lists.append(data[2])
        all_part_name_array = np.concatenate(all_part_name_lists)
        unique_particles = np.unique(all_part_name_array)
        colors = filtered_colors[:len(unique_particles)] 
        for particle, color in zip(unique_particles, colors):
            color_map[particle] = color  
    print(len(unique_particles))        
    # Additional logic for setting x-axis ticks if desired_order is provided
    # Convert single labels to lists if xyz_plot is True
    if xyz_plot:
        #if not isinstance(x_label, list):
        #    x_label = [x_label] * 3
        #if not isinstance(y_label, list):
        y_labels = ["X","Y","Z"]
        #else:
        #    y_labels=y_label
    try:
        y_label_,unit_y= y_label.split(" ")
    except:
        print("Unit of Y lable not provided. Note the format is  ' Y unit' ")
        y_label_,unit_y= y_label,None
    if unit_y:
        if norm:
            y_label_for_saving= y_label_+f'_over_{max_for_norm}{unit_y}' # type: ignore
        else:
            y_label_for_saving= y_label_+unit_y
    else:
        if norm:
            y_label_for_saving= y_label_+f'_over_{max_for_norm}' # type: ignore
        else:
            y_label_for_saving= y_label_
    
    # Plotting
    if all_in_one:
        if xyz_plot:
            plt.figure(figsize=(figsize_UNIT[0]*3, figsize_UNIT[1]*3))
            for i_datasets_pos in range(3):
                for i, data_set in enumerate(data_sets[i_datasets_pos]):
                    ax = plt.subplot(3, num_plots, num_plots*(i_datasets_pos)+1+i)
                    X, Y, part_name_list = data_set
                    if desired_order:
                            X_new = np.array([desired_order[i].index(m) for m in X])
                            ax.set_xticks(range(len(desired_order[i])))
                            ax.set_xticklabels(desired_order[i])
                    else:
                        X_new=X
                    for particle in np.unique(part_name_list):
                        x_plot = np.array([x for x, p in zip(X_new, part_name_list) if p == particle])
                        y_plot = np.array([y for y, p in zip(Y, part_name_list) if p == particle])
                        if norm:
                            y_plot/=max_for_norm # type: ignore
                        plt.plot(x_plot,y_plot, marker=marker, ms=marker_size,label=particle, color=color_map[particle],linestyle=linestyle,linewidth=linewidth,)
                        plt.xticks(rotation=rotation_xtick, fontsize=f_s, fontweight='bold')

                    
                    plt.xlabel(x_label)
                    if norm:
                        if unit_y:
                            plt.ylabel(y_label_+ ' ' +y_labels[i_datasets_pos]+f"/({max_for_norm}"+f"$_{{(\mathrm{{{unit_y}}})}}$)") # type: ignore
                        else:
                            plt.ylabel(y_label_+ ' ' +y_labels[i_datasets_pos]+f"/({max_for_norm}") # type: ignore
  
                    else:
                        if unit_y:
                            plt.ylabel(y_label_+ ' ' +y_labels[i_datasets_pos] +f"$_{{(\mathrm{{{unit_y}}})}}$") # type: ignore
                        else:
                            plt.ylabel(y_label_+ ' ' +y_labels[i_datasets_pos]) # type: ignore
                    plt.legend(loc='best', ncols=3  )
                    if (a_min and a_max):
                        plt.ylim(a_min, a_max)
                    format_ticks_scientific(ax, axis='y', font_size=f_s)            
                    if subtitles:
                        plt.title(subtitles[i])
                    else:
                        plt.title("{}".format(i+1))
                plt.suptitle(fig_title)
                plt.tight_layout()
                if save_dir_plot:
                    plt.savefig(save_dir_plot +( y_label_for_saving )  + "XYZ_vs_" + x_label    + f"_all_part.{file_type}",) 
            if show_plt:
                plt.show()    
            else:
                plt.close()
        else:
            plt.figure(figsize=(figsize_UNIT[0]*num_plots, figsize_UNIT[1]))
            for i, data_set in enumerate(data_sets):
                ax = plt.subplot(1, num_plots, i+1)
                X, Y, part_name_list = data_set
                if desired_order:
                        X_new = np.array([desired_order[i].index(m) for m in X])
                        ax.set_xticks(range(len(desired_order[i])))
                        ax.set_xticklabels(desired_order[i])
                else:
                    X_new=X
                for particle in np.unique(part_name_list):
                    x_plot = [x for x, p in zip(X_new, part_name_list) if p == particle]
                    y_plot = [y for y, p in zip(Y, part_name_list) if p == particle]
                    # Sort x_plot and y_plot together based on x_plot values
                    sorted_pairs = sorted(zip(x_plot, y_plot), key=lambda pair: pair[0])
                    
                    # Unzip the sorted pairs back into separate lists
                    x_plot_sorted, y_plot_sorted = zip(*sorted_pairs)
                    
                    # Convert back to lists if needed
                    x_plot = list(x_plot_sorted)
                    y_plot = list(y_plot_sorted)
                    plt.plot(x_plot, y_plot, marker, ms=marker_size, label=particle, color=color_map[particle],linestyle=linestyle,linewidth=linewidth)
                    plt.xticks(rotation=rotation_xtick, fontsize=f_s, fontweight='bold')
                
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                plt.legend(loc='best', ncols=3)
                if (a_min and a_max):
                    plt.ylim(a_min, a_max)
                format_ticks_scientific(ax, axis='y', font_size=f_s)
        
                if subtitles:
                    plt.title(subtitles[i])
                else:
                    plt.title("{}".format(i+1))
            plt.suptitle(fig_title)
            plt.tight_layout()
            if save_dir_plot:
                plt.savefig(save_dir_plot + y_label_for_saving+ "_vs_" + x_label.replace(" ", "") + f"_all_part.{file_type}",)
            if show_plt:
                plt.show()    
            else:
                plt.close()
    else:
        if xyz_plot:
            colors = plt.cm.tab10(np.linspace(0, 0.5, 3)) # type: ignore
            for particle in np.unique(unique_particles):
                trigger_subplots=np.zeros(num_plots, dtype=bool)  # One for X, Y, Z
                for i_datasets_pos in range(3):
                    for i, data_set in enumerate(data_sets[i_datasets_pos]):
                        X, Y, part_name_list = data_set
                        if particle  in part_name_list:
                            trigger_subplots[i]+=1  
                trigger_subplots= np.array(trigger_subplots)
                num_plots_loc=(trigger_subplots.sum()).astype(int)
                print(trigger_subplots,num_plots_loc)
                fig, axes = plt.subplots(nrows=1, ncols=num_plots_loc, figsize=(figsize_UNIT[0]*num_plots_loc, figsize_UNIT[1]))
                for i_datasets_pos in range(3):
                    i_subplots=0
                    for i, data_set in enumerate(data_sets[i_datasets_pos]):   
                        X, Y, part_name_list = data_set
                        if (trigger_subplots[i]==0) or (particle not in part_name_list):
                            continue
                        if num_plots_loc==1:
                            ax_loc=axes
                        else:
                            ax_loc=axes[i_subplots]
                        if desired_order:
                                X_new = np.array([desired_order[i].index(m) for m in X])
                                ax_loc.set_xticks(range(len(desired_order[i])))
                                ax_loc.set_xticklabels(desired_order[i])
                        else:
                            X_new=X
                        ax_loc.tick_params(axis='x', labelrotation=rotation_xtick)
                        x_plot = np.array([x for x, p in zip(X_new, part_name_list) if p == particle])
                        y_plot = np.array([y for y, p in zip(Y, part_name_list) if p == particle])

                        if norm:
                            y_plot/=max_for_norm # type: ignore

                        
                        ax_loc.plot(x_plot, y_plot, marker,  ms=marker_size,color=colors[i_datasets_pos],label=y_labels[i_datasets_pos],linestyle=linestyle,linewidth=linewidth) # type: ignore
                        ax_loc.set_xlabel( x_label  )
                        if norm:
                            if unit_y:
                                ax_loc.set_ylabel(y_label_+ ' ' +f"/({max_for_norm}"+f"$_{{(\mathrm{{{unit_y}}})}}$)") # type: ignore
                            else:
                                ax_loc.set_ylabel(y_label_+ ' ' +f"/({max_for_norm}") # type: ignore
      
                        else:
                            if unit_y:
                                ax_loc.set_ylabel(y_label_+ ' ' +f"$_{{(\mathrm{{{unit_y}}})}}$") # type: ignore
                            else:
                                ax_loc.set_ylabel(y_label_)
                        format_ticks_scientific(ax_loc, axis='y', font_size=f_s)

                        ax_loc.legend(loc='best', ncols=3)
                        if (a_min and a_max):
                            ax_loc.set_ylim(a_min, a_max)
                
                        if subtitles:
                            ax_loc.set_title(subtitles[i])
                        else:
                            ax_loc.set_title("{}".format(i+1))
                        i_subplots+=1
                suptitle=plt.suptitle(fig_title + " "+particle ,ha='center')
                # try:
                #     suptitle.set_y(y_coords.max()+0.08 )
                # except:
                #     pass
                plt.tight_layout()
                
                if save_dir_plot:
                        plt.savefig(save_dir_plot + y_label_for_saving  + "_vs_" +( x_label )  + "_"+particle+f".{file_type}",)
                if show_plt:
                    plt.show()    
                else:
                    plt.close()
        else:
            for particle in np.unique(unique_particles):
                fig=plt.figure(figsize=(figsize_UNIT[0]*num_plots, figsize_UNIT[1]))
                i_subplots=0
                for i, data_set in enumerate(data_sets):   
                    ax = plt.subplot(1, num_plots, i_subplots+1)
                    X, Y, part_name_list = data_set
                    if desired_order:
                            X_new = np.array([desired_order[i].index(m) for m in X])
                            ax.set_xticks(range(len(desired_order[i])))
                            ax.set_xticklabels(desired_order[i])
                    else:
                        X_new=X
                    x_plot = [x for x, p in zip(X_new, part_name_list) if p == particle]
                    y_plot = [y for y, p in zip(Y, part_name_list) if p == particle]
                    if len(y_plot)==0:
                        #ax.set_visible(False)
                        #ax.set_axis_off()
                        fig.delaxes(ax)
                    else:
                        # Sort x_plot and y_plot together based on x_plot values
                        sorted_pairs = sorted(zip(x_plot, y_plot), key=lambda pair: pair[0])
                        
                        # Unzip the sorted pairs back into separate lists
                        x_plot_sorted, y_plot_sorted = zip(*sorted_pairs)
                        
                        # Convert back to lists if needed
                        x_plot = list(x_plot_sorted)
                        y_plot = list(y_plot_sorted)
                        plt.plot(x_plot, y_plot, marker, ms=marker_size, label=particle, color=color_map[particle],linestyle=linestyle,linewidth=linewidth)
                        plt.xticks(rotation=rotation_xtick, fontsize=f_s, fontweight='bold')

                        
                        plt.xlabel(x_label)
                        plt.ylabel(y_label)
                        plt.legend(loc='best', ncols=3)
                        if (a_min and a_max):
                            plt.ylim(a_min, a_max)
                        format_ticks_scientific(ax, axis='y', font_size=f_s)
                        if subtitles:
                            plt.title(subtitles[i])
                        else:
                            plt.title("{}".format(i+1))
                        i_subplots+=1
                plt.suptitle(fig_title + " "+particle, x=i_subplots*0.5/(i+1)) # type: ignore
                plt.tight_layout()
                if save_dir_plot:
                    plt.savefig(save_dir_plot + y_label_for_saving+ "_vs_" +( x_label )   + "_"+particle+f".{file_type}",)
            
                if show_plt:
                    plt.show()    
                else:
                    plt.close()

def plot_stast_evolution_id27(x_absis, stats_x, stats_y, stats_z, pressure_allscan_list, y_label="", y_label_unit="", label_rot=-70, fontsize_ticks=60, figsize=(20, 42), n=4, m=1, line_width=8,marker_size=15,linestyle="-",marker="^",
                              save_path=None,f_s_labels=50,labelpad=50,prime_ref=''
                             ):
    import matplotlib.ticker as mticker
    from matplotlib.ticker import ScalarFormatter
    import matplotlib.gridspec as gridspec
    
    def figure_axes_desidn(ax, label_rot):
        plt.axis('tight')
        plt.xticks(rotation=label_rot)
        ax.grid(alpha=0.01)
        
        # Tick colors
        ax.tick_params('y', colors='b')
        ax.tick_params('x', colors='b')
    
        # Unified tick font sizes
        ax.tick_params(axis='x', labelsize=fontsize_ticks)
        ax.tick_params(axis='y', labelsize=int(fontsize_ticks * 1.15))
    
        # Scientific notation formatting on Y-axis (if needed)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))  # Show scientific notation if values are very small/large
        ax.yaxis.set_major_formatter(formatter)
    
        # Move offset text to the top-left
        ax.yaxis.get_offset_text().set_fontsize(int(fontsize_ticks * 0.9))
        ax.yaxis.set_offset_position('left')
    
    stat_all = np.array([stats_x, stats_y, stats_z])
    min_value = np.min(stat_all)

    y_min = np.where(min_value >= 0, min_value * 0.95, min_value * 1.05)
    y_max=np.max(stat_all)*1.05
    fig = plt.figure(figsize=figsize)
    # Create GridSpec
    gs = gridspec.GridSpec(n, m, height_ratios=[1, 1, 1, 0.2], hspace=0.1)
    list_to_replace,list_indices_to_replace=[],[]
    uniques_sorted_presure=np.unique(pressure_allscan_list)
    for i in range(len(uniques_sorted_presure)):
        list_indices_pre=np.where(pressure_allscan_list==uniques_sorted_presure[i])[0]
        print(list_indices_pre,list_indices_pre[len(list_indices_pre)//2])
        list_to_replace.append(x_absis[i])
        list_indices_to_replace.append(list_indices_pre[len(list_indices_pre)//2])
        
    #list_to_replace = [x_absis[i] for i in range(len(x_absis)) if " S0" in x_absis[i]]
    #list_indices_to_replace = [i for i in range(len(x_absis)) if " S0" in x_absis[i]]   
    list_presure_to_show_p = pressure_allscan_list[list_indices_to_replace]
    list_presure_to_show= np.array(['0.3', '1.6', '4.5', '4.6','', '5', '5.0', '', '5.6','6.7'])
    #list_presure_to_show= np.array(['0.3', '1.6', '4.5', '4.6', '4.77', '5', '5.0', '5.2', '5.6','6.7'])
    def get_color(i_pres):
        try:
            if float(i_pres) < 3:
                return "yellow"
            elif i_pres == "5":
                return "green"
            elif i_pres == "6.7":
                return "blue"
            else:
                return "red"
        except ValueError:
            return "gray"  # Default color if conversion to float fails

    def add_colored_bands(ax):
        for i in range(len(x_absis)):
            i_pres = x_absis[i].split()[0][1:]  # Assuming pressure is the first part before a space
            color = get_color(i_pres)
            ax.axvspan(i, i+1, facecolor=color, alpha=0.3)

    # Skewness X
    ax1 = fig.add_subplot(gs[0, 0])
    add_colored_bands(ax1)
    ax1.plot(range(len(x_absis)), stats_x, linestyle=linestyle,marker=marker, ms=marker_size,linewidth=line_width, markerfacecolor='red', markeredgecolor='black',)
    ax1.set_ylabel(    f"{y_label}\n$Q_X{{{prime_ref}}}$ {y_label_unit}",    fontsize=f_s_labels,    labelpad=labelpad)
    ax1.set_title("Evolution of " + str(y_label),fontsize=f_s_labels)
    figure_axes_desidn(ax1, label_rot)  # Assuming this is your custom function
    ax1.get_xaxis().set_visible(False)  # Hide x-axis for the first subplot

    # Skewness Y
    ax2 = fig.add_subplot(gs[1, 0])
    add_colored_bands(ax2)
    ax2.plot(range(len(x_absis)), stats_y, linestyle=linestyle,marker=marker, ms=marker_size,linewidth=line_width, markerfacecolor='red', markeredgecolor='black',)
    ax2.set_ylabel(    f"{y_label}\n$Q_Y{{{prime_ref}}}$ {y_label_unit}",    fontsize=f_s_labels,    labelpad=labelpad)
    figure_axes_desidn(ax2, label_rot)
    ax2.get_xaxis().set_visible(False)  # Hide x-axis for the second subplot

    # Skewness Z
    ax3 = fig.add_subplot(gs[2, 0])
    add_colored_bands(ax3)
    ax3.plot(range(len(x_absis)), stats_z, linestyle=linestyle,marker=marker, ms=marker_size,linewidth=line_width, markerfacecolor='red', markeredgecolor='black',)
    ax3.set_ylabel(f"{y_label}\n$Q_Z{{{prime_ref}}}$ {y_label_unit}",    fontsize=f_s_labels,    labelpad=labelpad)
    ax3.set_xlabel("")  # remove default
    ax3.annotate('Pressure (GPa)', 
                 xy=(0.25, -0.45), xycoords='axes fraction', 
                 fontsize=f_s_labels, ha='left', va='center', fontweight='bold')

    #ax3.set_xlabel('Pressure (GPa)',fontsize=f_s_labels,labelpad=labelpad)
    ax3.set_xlabel('')
    ax3.set_xticks(list_indices_to_replace)
    ax3.set_xticklabels(list_presure_to_show, rotation=label_rot)
    
    ax3.tick_params(axis='x', which='major', length=20, width=4, color='b', pad=5, labelsize=fontsize_ticks)    
    ax3.tick_params(axis='y', which='major', length=20, width=4, color='b', pad=5, labelsize=fontsize_ticks*1.15)
    ax2.tick_params(axis='y', which='major', length=20, width=4, color='b', pad=5, labelsize=fontsize_ticks*1.15)
    ax1.tick_params(axis='y', which='major', length=20, width=4, color='b', pad=5, labelsize=fontsize_ticks*1.15)

    
    figure_axes_desidn(ax3, label_rot)

    # Create zoomed labels
    zoom_values = np.array([ '4.77', '5.2'])
    zoom_indices = np.where(np.isin(list_presure_to_show_p, zoom_values))[0]
    zoom_indices[1]-=2
    print(zoom_indices)
    

    
    if len(zoom_indices) > 0:
        ax_zoom = fig.add_axes([0.64, .14, 0.15, 0.0005]) # type: ignore
        ax_zoom.set_xlim(ax3.get_xlim())
        ax_zoom.set_xticks(zoom_indices)
        figure_axes_desidn(ax_zoom,label_rot)
        
        # Convert labels to float and format them
        float_labels = ["4.77","5.2"]#array([f"{float(label):.2f}" for label in list_presure_to_show_p[zoom_indices]]).astype(str)
        ax_zoom.set_xticklabels(float_labels, rotation=label_rot, fontsize=fontsize_ticks)
        
        ax_zoom.tick_params(axis='y', which='both', left=False, labelleft=False)
        ax_zoom.spines['top'].set_visible(False)
        ax_zoom.spines['right'].set_visible(False)
        ax_zoom.spines['left'].set_visible(False)
        ax_zoom.spines['bottom'].set_visible(False)
        ax_zoom.set_ylim(0, 0.01)  # Set y-limits to create some space
        ax_zoom.set_xlim(4., 9.)  # Set y-limits to create some space
        ax_zoom.axis('tight')
        # Set background color to light gray
        #ax_zoom.set_facecolor('#E6E6E6')  # Light gray color

    
    ax1.set_ylim(y_min,y_max) # type: ignore
    ax2.set_ylim(y_min,y_max) # type: ignore
    ax3.set_ylim(y_min,y_max) # type: ignore

    
    # Adjust the layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def anealing_plot_stat_multiple(temperatures, stat_params_groups, particle_names, 
                                stat_param_names=None, desired_order=None, 
                                line_style="-",linewidth=4,markersize=15,rotation_x=45,
                                font_size=24, xtick_fontsize=18,
                                list_of_particle_to_plot=None,rect =[0, 0, 1, 0.97],
                                save_fig=None, x_label="Stage"):
    """
    Creates a figure with multiple rows of subplots for statistical parameters.

    Parameters
    ----------
    temperatures : array-like
        X-axis values (e.g., temperature or scan labels).
    stat_params_groups : list of 3-element lists
        Each group contains [X, Y, Z] arrays for one stat (e.g., FWHM).
    particle_names : array-like
        Names of particles for each datapoint.
    stat_param_names : list of str, optional
        Names for the statistical parameter groups.
    desired_order : array-like, optional
        Custom x-axis order (e.g., sorted temperatures).
    line_style : str or None, optional
        Line style (e.g., "--" or None for markers only).
    font_size : int, optional
        Font size for all text elements (bold).
    xtick_fontsize : int, optional
        Font size (bold) for x-tick labels only.
    list_of_particle_to_plot : list of str, optional
        Subset of particle names to plot.
    save_fig : str or None, optional
        If provided, saves the plot to this filename.
    x_label : str, optional
        X-axis label to use.

    Returns
    -------
    None
    """
    plt.rcParams.update({
        'font.weight': 'bold',
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
    })
    temperatures = np.array(temperatures)
    particle_names = np.array(particle_names)
    stat_params_groups = [[np.array(arr) for arr in group] for group in stat_params_groups]
    
    if list_of_particle_to_plot is not None:
        mask = np.isin(particle_names, list_of_particle_to_plot)
        temperatures = temperatures[mask]
        particle_names = particle_names[mask]
        stat_params_groups = [[arr[mask] for arr in group] for group in stat_params_groups]
    
    if desired_order is not None:
        indices = []
        for t in desired_order:
            matching = np.where(temperatures == t)[0]
            if matching.size > 0:
                indices.extend(matching.tolist())
        temperatures = temperatures[indices]
        particle_names = particle_names[indices]
        stat_params_groups = [[arr[indices] for arr in group] for group in stat_params_groups]
    
    if desired_order is not None:
        unique_temps = np.array([t for t in desired_order if t in temperatures])
        mapping = {temp: i for i, temp in enumerate(unique_temps)}
        x_numeric = np.array([mapping[t] for t in temperatures])
    else:
        unique_temps, x_numeric = np.unique(temperatures, return_inverse=True)
    
    n_groups = len(stat_params_groups)
    n_coords = 3
    
    if stat_param_names is None or len(stat_param_names) != n_groups:
        stat_param_names = [f"StatParam {i+1}" for i in range(n_groups)]
    
    unique_particles = np.unique(particle_names)
    n_particles = len(unique_particles)
    color_map = matplotlib.cm.get_cmap("tab10", n_particles) # type: ignore
    particle_to_color = {particle: color_map(i) for i, particle in enumerate(unique_particles)}
    marker_list = ['o', 's', '^', 'd', 'v', '<', '>', 'P', 'X', '*']
    particle_to_marker = {particle: marker_list[i % len(marker_list)] for i, particle in enumerate(unique_particles)}
    
    fig, axes = plt.subplots(nrows=n_groups, ncols=n_coords, figsize=(18, 10*n_groups), sharex=True, sharey='row')
    if n_groups == 1:
        axes = np.array([axes])
    
    legend_dict = {}
    
    for i_group in range(n_groups):
        for i_coord in range(n_coords):
            ax = axes[i_group, i_coord]
            for particle in unique_particles:
                idx = np.where(particle_names == particle)[0]
                x_vals = x_numeric[idx]
                y_vals = stat_params_groups[i_group][i_coord][idx]
                ls = line_style if line_style is not None else 'None'
                
                line, = ax.plot(x_vals, y_vals,
                                marker=particle_to_marker[particle],
                                linestyle=ls,
                                color=particle_to_color[particle],linewidth=linewidth,ms=markersize,
                                label=particle)
                if particle not in legend_dict:
                    legend_dict[particle] = line
            
            coord = ['X', 'Y', 'Z'][i_coord]
            ax.set_title(f"{stat_param_names[i_group]} {coord}",
                         fontsize=font_size, fontweight='bold')
            ax.grid(True)
            
            if i_coord == 0:
                ax.set_ylabel(stat_param_names[i_group],
                              fontsize=font_size, fontweight='bold')
            
            if i_group == n_groups - 1:
                ax.set_xlabel(x_label, fontsize=font_size, fontweight='bold')
                ax.set_xticks(np.arange(len(unique_temps)))
                ax.set_xticklabels(unique_temps, rotation=rotation_x, ha="right",
                                   fontsize=xtick_fontsize, fontweight='bold')
            
            ax.tick_params(axis='y', labelsize=font_size)
            ax.tick_params(axis='x', labelsize=xtick_fontsize)
            ax.xaxis.get_offset_text().set_fontsize(xtick_fontsize)
            ax.yaxis.get_offset_text().set_fontsize(font_size)
    
    fig.legend(legend_dict.values(), legend_dict.keys(),
               loc='upper center', ncol=n_particles,
               bbox_to_anchor=(0.5, 1.0),
               fontsize=font_size, frameon=False)
    
    plt.tight_layout(rect=rect)
    
    if save_fig is not None:
        plt.savefig(save_fig, bbox_inches='tight')
    
    plt.show()

