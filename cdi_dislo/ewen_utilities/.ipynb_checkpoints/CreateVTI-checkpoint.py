
from cdi_dislo.common_imports import *
from cdi_dislo.ewen_utilities.plot_utilities                      import *
from cdi_dislo.ewen_utilities.Global_utilities                    import check_path_create

import vtk
from vtk.util import numpy_support
import numpy as np
import scipy

import vtk
from vtk.util import numpy_support

###########################################################################################################################################
################               Create custom strain array for surface interpolation in paraview                ############################
###########################################################################################################################################

def cross_kernel():
    '''
    3x3x3 kernel in form of a cross (with 0 at the center but this doesn't matter)
    '''
    # very badly written but at least it works
    kernel = np.zeros((3,3,3))
    x,y,z = np.indices(kernel.shape)
    x = x - np.mean(x)
    y = y - np.mean(y)
    z = z - np.mean(z)
    r = np.sqrt(x**2. + y**2. + z**2.)
    kernel[r == np.nanmin(r[r!=0])] = 1   
    return kernel

# Gaussian kernel wasn't working so great when zooming at the surface
# I don't use it in the end
def gaussian_kernel(sigma = 1.):
    '''
    create a 3x3x3 gaussian kernel. I dodn't use it in the end.
    '''
    kernel = np.zeros((3,3,3))
    x,y,z = np.indices(kernel.shape)
    x = x - np.mean(x)
    y = y - np.mean(y)
    z = z - np.mean(z)
    r = np.sqrt(x**2. + y**2.+ z**2.)
    kernel = np.exp(-r**2./(.5*sigma**2.))
    return kernel


def get_distance_to_array_borders(support):
    '''
    get the shortest distance between the cristal surface and the border of the array along each direction
    '''
    distance_border_array = np.zeros(3)
    indices_support = np.where(support==1)
    for n in range(3):
        distance_border_array[n] += min( np.min(indices_support[n]), (support.shape[n]-1)-np.max(indices_support[n]) )
    return distance_border_array

def padding_avoid_array_border_problems(strain, support, Nb_surface_pixels_added):
    '''
    Pad the strain and support arrays in case you want to add too many additional surface pixels
    '''
    distance_border_array = get_distance_to_array_borders(support)
    
    padding = np.zeros(3, dtype='int')
    for n in range(3):
        if distance_border_array[n] < Nb_surface_pixels_added - 2 : # I add this -2 for safety
            padding[n] = Nb_surface_pixels_added - distance_border_array[n]
            
    strain_pad = np.pad(strain,((padding[0],padding[0]), (padding[1],padding[1]), (padding[2],padding[2]))
                        ,constant_values=(np.nan,))
    support_pad = np.pad(support,((padding[0],padding[0]), (padding[1],padding[1]), (padding[2],padding[2])))

    return strain_pad, support_pad

def one_run(strain, support, kernel):
    '''
    Create additional pixels right outside the surface
    '''
    strain_frontier = scipy.ndimage.convolve(strain, kernel)
    weight = scipy.ndimage.convolve(support, kernel) 
    weight = weight.astype('float64')
    
    strain_frontier[support.astype('int')==1] = np.nan # remove all pixels inside the cristal
    strain_frontier[weight==0] = np.nan # remove all pixels not touching the surface.
    weight[support.astype('int')==1] = np.nan
    weight[weight==0] = np.nan
    
    # divide by the kernel "sum" (making an of average of each touching pixels)
    strain_frontier = np.divide(strain_frontier,weight) 
    
    strain_update = np.nansum([strain, strain_frontier], axis=0) # add these outside pixels to the already exsiting ones.
    
    support_update = np.copy(support)
    support_update += 1-np.isnan(weight) # Update the support by adding these new pixels right outside the surface
    
    return strain_update, support_update

def several_run(strain, support, kernel, Nb_surface_pixels_added):
    '''
    Run several the creation of the frontier pixels
    '''
    strain_update = np.copy(strain)
    strain_update[np.isnan(strain)] = 0. # I need to remove all nan's for my convolution. 
                                         # Those 0 shouldn't matter if I'm not doing anything stupid 
                                         # since they're not in the support
    support_update = np.copy(support)
    for n in range(Nb_surface_pixels_added):
        print(Nb_surface_pixels_added-n, end=' ')
        strain_update, support_update = one_run(strain_update, support_update, kernel)
        
    strain_update[support_update==0] = np.nan
    return strain_update, support_update

def add_surface_pixels(strain,
                       Nb_surface_pixels_added = None,
                       kernel = None,
                       verbose=False, plot=False):

    if kernel is None:
        kernel = cross_kernel() # I will use a kernel in form of a cross (the central pixel doesn't matter)
        # kernel = gaussian_kernel() # This one was giving me weird stuff
        
    support = 1-np.isnan(strain)

    # Check Nb_surface_pixels_added
    distance_border_array = get_distance_to_array_borders(support)
    max_pixels_added = np.min(distance_border_array.astype('int'))
    if Nb_surface_pixels_added is None:
        # Number of pixels added at the surface. You can write something lower than that.
        Nb_surface_pixels_added =  max_pixels_added
    elif Nb_surface_pixels_added > max_pixels_added :
            print('problem, Nb_surface_pixels_added should be less than {}. I overwrite it.'.format(max_pixels_added))
    if verbose:
        print('Nb_surface_pixels_added :', Nb_surface_pixels_added)
     
    # Add the pixels at the surface
    strain_update, support_update = several_run(strain, support, kernel, Nb_surface_pixels_added)
    
    if plot:
        plot_2D_slices_middle_one_array3D(strain, cmap='bwr', fig_title='original strain')
        #plot_2D_slices_middle_one_array3D(support, cmap='gray_r', fig_title='original support')
        
#         fig,ax = plt.subplots(1,3, figsize=(12,4))
#         for n in range(3):
#             ax[n].matshow(kernel[n], cmap='gray_r', vmin=0, vmax=1)
#             ax[n].set_title('kernel[{}]'.format(n), fontsize=15)
#         fig.suptitle('kernel', fontsize=20)
#         fig.tight_layout()
    
        plot_2D_slices_middle_one_array3D(strain_update, cmap='bwr', fig_title='strain with additional pixels')
    return strain_update


###########################################################################################################################################
#####################################               Save vti file                ##########################################################
###########################################################################################################################################

# import vtk
# from vtk.util import numpy_support
# def save_vti(module_ortho, phase_ortho, strain_hetero, d_spacing, 
#              voxel_sizes, path_reconstruction,
#              additional_vti_files=None,
#              origin=(0, 0, 0)):
    

#     tuple_array = (module_ortho, phase_ortho, strain_hetero, d_spacing)
#     tuple_fieldnames = ('module_ortho', 'phase_ortho', 'strain_hetero', 'd_spacing')
    
#     if additional_vti_files is not None:
#         for key in additional_vti_files.keys():
#             tuple_array = tuple_array + (additional_vti_files[key],)
#             tuple_fieldnames = tuple_fieldnames + (key,)
    
#     nbz, nby, nbx = tuple_array[0].shape
#     image_data = vtk.vtkImageData()
#     image_data.SetOrigin(origin[0], origin[1], origin[2])
#     image_data.SetSpacing(voxel_sizes[0], voxel_sizes[1], voxel_sizes[2])
#     image_data.SetExtent(0, nbz - 1, 0, nby - 1, 0, nbx - 1)

#     first_array = np.copy(tuple_array[0])
#     first_array = first_array / first_array.max()
    
#     nb_arrays = len(tuple_array)
#     index_first = 0
    
#     savename = path_reconstruction + 'final_obj/{}.vti'.format(path_reconstruction.split('/')[-2])
    
#     first_arr = np.transpose(np.flip(first_array, 2)).reshape(first_array.size)
#     first_arr = numpy_support.numpy_to_vtk(first_arr)
#     pd = image_data.GetPointData()
#     pd.SetScalars(first_arr)
#     pd.GetArray(0).SetName(tuple_fieldnames[index_first])
#     counter = 1
#     for idx in range(nb_arrays):
#         if idx == index_first:
#             continue
#         temp_array = tuple_array[idx]
#     #     if is_amp:
#     #         temp_array[
#     #             first_array == 0
#     #         ] = 0  # use the thresholded amplitude as a support
#     #         # in order to save disk space
#         temp_array = np.transpose(np.flip(temp_array, 2)).reshape(temp_array.size)
#         temp_array = numpy_support.numpy_to_vtk(temp_array)
#         pd.AddArray(temp_array)
#         pd.GetArray(counter).SetName(tuple_fieldnames[idx])
#         pd.Update()
#         counter = counter + 1

#     # export data to file
#     writer = vtk.vtkXMLImageDataWriter()
#     writer.SetFileName(savename)
#     writer.SetInputData(image_data)
#     writer.Write()
    
#     print('vti file saved in : ', savename)
    
#     return



def save_vti(np_arrays, 
             voxel_sizes, path_reconstruction,
             output_path=None,
             origin=(0, 0, 0)):
    
    # A safety check
    if path_reconstruction[-1] != '/' :
        path_reconstruction += '/'
    
    if output_path is None:
        output_path = path_reconstruction + 'final_obj/{}.vti'.format(path_reconstruction.split('/')[-2])
        nb_arrays = len(np_arrays)

    is_init = False
    for i, (key, array) in enumerate(np_arrays.items()):
        array = np.swapaxes(array, 0,2)
        if not is_init:
            # I play on the axis to have the same result orientation as cdiutils
            voxel_size = (.1*voxel_sizes[0], .1*voxel_sizes[1], .1*voxel_sizes[2])
            shape = array.shape

            
            shape = (shape[2], shape[1], shape[0])
            image_data = vtk.vtkImageData()
            image_data.SetOrigin(origin)
            image_data.SetSpacing(voxel_size)
            image_data.SetExtent(
                0, shape[0] - 1,
                0, shape[1] - 1,
                0, shape[2] - 1
            )
            point_data = image_data.GetPointData()
            is_init = True

        vtk_array = numpy_support.numpy_to_vtk(array.ravel())
        point_data.AddArray(vtk_array)
        point_data.GetArray(i).SetName(key)
        point_data.Update()

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(image_data)
    writer.Write()
    
    print('vti file saved in : ', output_path)
    return

def save_arrays_in_vti(np_arrays_dict,
             savename='default_vti_name.vti',
             origin=(0, 0, 0)):
    
    if savename[-4:] != '.vti':
        savename += '.vti'

    is_init = False
    for i, (key, array) in enumerate(np_arrays_dict.items()):
        array = np.swapaxes(array, 1,2)
        if not is_init:
            shape = array.shape[::-1]
            image_data = vtk.vtkImageData()
            image_data.SetOrigin(origin)
            image_data.SetExtent(
                0, shape[0] - 1,
                0, shape[1] - 1,
                0, shape[2] - 1
            )
            point_data = image_data.GetPointData()
            is_init = True

        vtk_array = numpy_support.numpy_to_vtk(array.ravel())
        point_data.AddArray(vtk_array)
        point_data.GetArray(i).SetName(key)
        point_data.Update()

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(savename)
    writer.SetInputData(image_data)
    writer.Write()
    
    print('vti file saved in : ', savename)
    return



