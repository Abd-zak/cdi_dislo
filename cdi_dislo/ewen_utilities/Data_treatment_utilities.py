import numpy as np
import pylab as plt

import sys
from skimage.registration import phase_cross_correlation
import scipy
from cdi_dislo.common_imports import *
from cdi_dislo.ewen_utilities.plot_utilities                      import *
from cdi_dislo.ewen_utilities.Object_utilities                    import *
#################################################################################################################################
#################################                Align objects                    ###############################################
################################################################################################################################# 

def force_same_shape(obj_list,
                     verbose = False) :
    
    '''
    In case objects in obj_list don't have the same shape, 
    this function forces an identical shape by padding objects with 0s
    :obj_list: a list of objects of shape (number of objects, individual object shape)
    '''
    
    shape_list = [obj.shape for obj in obj_list]
    
    if np.all([shape == shape_list[0] for shape in shape_list]) :
        if verbose :
            print('All objects already have the same shape')
        return obj_list
    
    forced_shape = np.max(shape_list, axis=0)
    
    for n, obj in enumerate(obj_list):
        pad = (np.array(forced_shape) - np.array(obj.shape))
        padding = [(p//2, p//2 + p%2) for p in pad]
        obj_list[n] = np.pad(obj, padding,mode='constant',constant_values=(0,))
        
    if verbose :
        print(f'All objects have now the shape : {forced_shape}')
        
    return np.array(obj_list)



def realign_object_list(obj_list,
                        ref_index=0,
                        threshold_module=.15, fill_support=False) :
    
    '''
    Align all objects in obj_list using the supports and a phase_cross_correlation.
    Limited to integer pixels shift. No sub-pixel shifts.
    :ref_index: index of the reference object 
                (for example with ref_index=0, the first object is the reference position)
    :threshold_module: threshold (between 0 and 1) used to create the support.
    :fill_support: If True, fill holes in the support (mostly for particles with dislocations)
    '''
    
    obj_ref = np.copy(obj_list[ref_index])
    support_ref = create_support(obj_ref, threshold_module, fill_support=fill_support)
        
    obj_list_shift = np.zeros(obj_list.shape)
    for n, obj in enumerate(obj_list):
        support = create_support(obj, threshold_module, fill_support=fill_support)
        shift, error, diffphase = phase_cross_correlation(support_ref, support)
        obj_list_shift[n] += scipy.ndimage.shift(np.abs(obj), shift)
        
    return obj_list_shift


