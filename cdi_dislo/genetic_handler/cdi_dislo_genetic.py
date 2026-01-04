#####################################################################################################################
#####################################################################################################################
###############################################   SUMMARY BLOCK    ##################################################
#####################################################################################################################
#####################################################################################################################

"""
This script provides essential utility functions for coherent diffraction imaging (CDI) reconstruction, 
including multiple reconstructions, object sorting based on sharpness, and genetic optimization of object lists.

### Key Features:
- **make_several_reconstruction**: Performs multiple CDI reconstructions with error handling.
- **sharpness_metric**: Computes a sharpness score for ranking reconstructed objects.
- **sort_objects**: Sorts reconstructed objects based on sharpness for selection.
- **genetic_update_object_list**: Applies a genetic approach to refine reconstructions.
- **automatic_crop**: Determines optimal cropping around the center of mass (COM) while avoiding zero cropping.

### Modules Utilized:
- CDI reconstruction utilities from `cdi_dislo.ewen_utilities`
- NumPy for numerical and array manipulations
- Time module for execution time tracking

### Potential Use Cases:
- Batch processing of CDI data
- Automated object ranking for selection
- Optimization of reconstruction accuracy through genetic algorithms
"""

#####################################################################################################################
#####################################################################################################################
###############################################   SUGGESTIONS BLOCK    ##############################################
#####################################################################################################################
#####################################################################################################################

"""
1️⃣ **Enhance Error Handling:**
   - Replace the general `except:` with specific exceptions like `except ValueError:` or `except KeyError:` 
     to improve debugging clarity.
   - Implement structured logging (e.g., `logging` module) to track failures and debugging information.

2️⃣ **Optimize Performance:**
   - Utilize **parallel processing** (e.g., `multiprocessing` or `joblib`) to speed up `make_several_reconstruction`.
   - Implement **NumPy vectorization** where possible to reduce loop overhead and improve computational efficiency.
   - Consider **GPU acceleration** using libraries like CuPy or TensorFlow for heavy matrix computations.

3️⃣ **Improve Sorting & Selection Criteria:**
   - Instead of only using sharpness, introduce **multi-metric ranking** (e.g., phase consistency, reconstruction stability).
   - Store object metadata (e.g., reconstruction time, error metrics) for **better analysis and reproducibility**.

4️⃣ **Refine Genetic Algorithm Approach:**
   - Extend `genetic_update_object_list` by:
     - Implementing **mutation operations** to introduce variability in the updated objects.
     - Using **adaptive parameter tuning** to dynamically adjust selection pressure.
     - Exploring **crossover techniques** for enhanced object refinement.

5️⃣ **Increase Code Modularity & Testing:**
   - Convert frequently used utility functions into **standalone modules** for better maintainability.
   - Implement **unit tests** (e.g., using `pytest`) for robustness and error detection.
   - Add **input validation checks** to handle unexpected parameter values gracefully.

6️⃣ **Improve User Experience & Visualization:**
   - Add **verbose mode** for detailed execution logs while keeping the default output minimal.
   - Introduce **progress bars** (e.g., `tqdm`) for tracking reconstruction progress.
   - Enhance **plotting functions** with overlays or annotations for better visualization of results.
"""




from cdi_dislo.ewen_utilities.plot_utilities                      import plot_2D_slices_middle
from cdi_dislo.ewen_utilities.Reconstruction                      import CDI_one_reconstruction
from cdi_dislo.ewen_utilities.PostProcessing                      import force_same_complex_conjugate_object_list
from cdi_dislo.ewen_utilities.Object_utilities                    import center_object_list

import numpy as np
import time 

#####################################################################################################################
#####################################################################################################################

#####################################################################################################################
#####################################################################################################################
###############################################   generic algo utility    ###########################################
#####################################################################################################################
#####################################################################################################################
def make_several_reconstruction(data, nb_recon, params_init, obj_list=None, support_list=None):
    '''
    Perform several reconstructions.

    Parameters:
        data (ndarray): The input data for reconstruction.
        nb_recon (int): Number of reconstructions to perform.
        params_init (dict): Initial parameters for the reconstruction algorithm.
        obj_list (ndarray, optional): Initial object list. Default is None.
        support_list (ndarray, optional): Initial support list. Default is None.

    Returns:
        tuple: Tuple containing the new object list and support list.
    '''
    obj_list_new = np.zeros((nb_recon,)+data.shape).astype('complex128')
    support_list_new = np.zeros((nb_recon,)+data.shape)
    
    for n in range(nb_recon):
        start_time_reco=time.time()
        print(f"Reconstruction #{n}") 
        params = dict(params_init)
        if obj_list is not None:
            params['obj_init'] = obj_list[n]
        
        fail = True
        while fail:
            try:
                obj, llk, support, return_dict = CDI_one_reconstruction(data, params)
                obj_list_new[n] += obj
                support_list_new[n] += support
                fail = False
            except:
                pass
        print(f"reconstruction took {time.time()-start_time_reco} sec")
        
    return obj_list_new, support_list_new
def sharpness_metric(obj, support):
    '''
    Calculate the sharpness metric for an object.

    Parameters:
        obj (ndarray): The reconstructed object.
        support (ndarray): The support for the object.

    Returns:
        float: The sharpness metric.
    '''
    module = np.abs(obj) * support
    return np.mean(module ** 4.)
def sort_objects(obj_list, support_list, plot=False):
    '''
    Sort the object list based on sharpness metric.

    Parameters:
        obj_list (ndarray): List of reconstructed objects.
        support_list (ndarray): List of supports for the objects.
        plot (bool, optional): Whether to plot the results. Default is False.

    Returns:
        tuple: Tuple containing the sorted metric list, object list, and support list.
    '''
    metric_list = np.zeros(len(obj_list))
    for n in range(len(obj_list)):
        metric_list[n] += sharpness_metric(obj_list[n], support_list[n])
    
    # Sort the objects and supports
    indices = np.argsort(metric_list)
    
    metric_list = metric_list[indices]
    obj_list = obj_list[indices]
    support_list = support_list[indices]
    
    if plot:
        for n in range(len(obj_list)):
            plot_2D_slices_middle(obj_list[n], fig_title=f'sharpness : {metric_list[n]}')
    
    return metric_list, obj_list, support_list
def genetic_update_object_list(obj_list, check_inv_complex_conjugate=True):
    '''
    Update the object list using the best one.

    Parameters:
        obj_list (ndarray): List of reconstructed objects.
        check_inv_complex_conjugate (bool, optional): Whether to check for inverse complex conjugate. Default is True.

    Returns:
        ndarray: Updated object list.
    '''
    if check_inv_complex_conjugate:
        obj_list = force_same_complex_conjugate_object_list(obj_list)
        obj_list = center_object_list(obj_list)
        
    for n in range(1, len(obj_list)):
        obj_list[n] = np.sqrt(obj_list[n] * obj_list[0])
    
    return obj_list

    """
    Optimize cropping around the center of mass (COM) while avoiding zero cropping.

    Parameters:
    - data (numpy.ndarray): 3D array of shape (z, y, x).

    Returns:
    - crop_width (int): Maximum width for cropping in any direction.
    """

    # Calculate sum along each direction
    sum_z = np.sum(data, axis=(1, 2))
    sum_y = np.sum(data, axis=(0, 2))
    sum_x = np.sum(data, axis=(0, 1))

    # Find the index of the first and last non-zero elements
    first_nonzero_z = np.argmax(sum_z > 0)
    last_nonzero_z = len(sum_z) - np.argmax(sum_z[::-1] > 0) - 1
    first_nonzero_y = np.argmax(sum_y > 0)
    last_nonzero_y = len(sum_y) - np.argmax(sum_y[::-1] > 0) - 1
    first_nonzero_x = np.argmax(sum_x > 0)
    last_nonzero_x = len(sum_x) - np.argmax(sum_x[::-1] > 0) - 1

    # Calculate crop width for each direction
    crop_width_z = max(last_nonzero_z - first_nonzero_z + 1, 0)
    crop_width_y = max(last_nonzero_y - first_nonzero_y + 1, 0)
    crop_width_x = max(last_nonzero_x - first_nonzero_x + 1, 0)

    # Return the maximum crop width among the three directions
    return max(crop_width_z, crop_width_y, crop_width_x)


