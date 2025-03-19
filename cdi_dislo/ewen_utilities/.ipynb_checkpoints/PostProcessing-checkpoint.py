import numpy as np
import pylab as plt
from numpy.fft import fftshift, fft, ifft
from numpy import pi

from pynx.cdi import * # Might take a bit of time to import

from cdi_dislo.common_imports import *
from cdi_dislo.ewen_utilities.plot_utilities                      import *
from cdi_dislo.ewen_utilities.Object_utilities                    import *
from cdi_dislo.ewen_utilities.Global_utilities                    import *
from cdi_dislo.ewen_utilities.Single_peak_gaussian_fit                    import *
from scipy.stats import gaussian_kde

from pynx.utils.math import ortho_modes
from sklearn.linear_model import LinearRegression
import os

my_cmap = MIR_Colormap()
###########################################################################################################################################
#####################################            Load reconstructions            ##########################################################
###########################################################################################################################################

def load_reconstructions(path_reconstruction):
    
    # A safety check
    if path_reconstruction[-1] != '/' :
        path_reconstruction += '/'
    
    obj_list = []
    llk_list = []
    file_list = []
    # Just to sort the files. Not great since it relies on my hardcoded reconstructions names
    files = get_npz_files(path_reconstruction)
    index_recon = [int(file[:-4].split('_')[-1]) for file in files]
    files = [files[index] for index in np.argsort(index_recon)]
    
    for file in files:
        if 'reconstruction' in file:
            try:
                data = np.load(file)
                obj_list.append(data['obj'])
                try:
                    llk_list.append(data['llk'])
                except : 
                    llk_list.append(0)
                file_list.append(file)
            except:
                print('failed to load {}'.format(file))
                pass
    obj_list = np.array(obj_list)
    llk_list = np.array(llk_list)
    file_list = np.array(file_list)
    return obj_list, llk_list, file_list

###########################################################################################################################################
###############################            Sort object list using a metric            #####################################################
###########################################################################################################################################

def compute_module_max_position_metric(obj, 
                                       bins=None):
    if bins is None:
        if obj.ndim == 2:
            bins = 30
        if obj.ndim == 3:
            bins = 50
            
    module, phase = get_cropped_module_phase(obj)
    module[module<.01*np.max(module)] = np.nan
    hist, bins_pos = np.histogram(module[np.logical_not(np.isnan(module))], bins=bins)
    x =  bins_pos[:-1] + (bins_pos[1:] - bins_pos[:-1])/2.
    metric = np.nanmax(module) - x[hist.argmax()]
    return metric

def compute_module_histogram_peak_width(obj,
                                        plot=False):
    module, phase = get_cropped_module_phase(obj)
    module = module/np.max(module)
    # module = normalize(module)
    module[module<.01*np.max(module)] = np.nan
    module_no_nan = module[np.logical_not(np.isnan(module))]

    # fit the amplitude distribution
    kernel = gaussian_kde(module_no_nan)
    x= np.linspace(0, 1, 100)
    fitted_counts = kernel(x)
    
    max_index = np.argmax(fitted_counts)
    right_gaussian_part = np.where(x >= x[max_index], fitted_counts, 0)

    # find the closest indexes
    right_HM_index = np.argmin(  np.abs(right_gaussian_part - fitted_counts.max() / 2) )  
    
    left_gaussian_part = np.where(x < x[max_index], fitted_counts, 0)

#     left_HM_index = max_index - (right_HM_index - max_index)
    left_HM_index = np.argmin( np.abs(left_gaussian_part - fitted_counts.max() / 2) )  

    fwhm = x[right_HM_index] - x[left_HM_index]
#     sigma_estimate = fwhm / 2*np.sqrt(2*np.log(2))
    
    if plot:
        plt.figure()
        plt.plot(x, fitted_counts, '-b')
        plt.axvline(x=x[left_HM_index], color='r')
        plt.axvline(x=x[right_HM_index], color='r')
        plt.title('FWHM : {}'.format(round(fwhm,3)), fontsize=15)
    return fwhm
    
def sort_object_list_metric(obj_list, 
                            llk_list = None, file_list=None,
                            metric_string = 'module max position'):
    metric = np.zeros(len(obj_list))
    for n in range(len(obj_list)):
        if metric_string == 'llk':
            metric[n] += llk_list[n]
        if metric_string == 'module max position':
            metric[n] += compute_module_max_position_metric( obj_list[n] )
        if metric_string == 'fwhm':
            metric[n] += compute_module_histogram_peak_width( obj_list[n] )
    indexes = np.argsort(metric)
    if (llk_list is not None) and (file_list is not None):
        return obj_list[indexes], llk_list[indexes], file_list[indexes]
    elif llk_list is not None:
        return obj_list[indexes], llk_list[indexes]
    else :
        return obj_list[indexes]

###########################################################################################################################################
#################################            Select best reconstructions            #######################################################
###########################################################################################################################################
def automatic_best_objects_selection(path_reconstruction,
                                     nb_keep_llk = 6,
                                     nb_keep = 4,
                                     gaussian_background_fit=False,
                                     plot=True):

    obj_list, llk_list = load_reconstructions(path_reconstruction)

    # keep only lowest llk
    indexes_to_keep = np.argsort(llk_list)[:nb_keep_llk]
    obj_list = obj_list[indexes_to_keep]
    llk_list = llk_list[indexes_to_keep]
    
    std_fit_list = np.zeros(len(obj_list))

    ax_title = []
    
    for n in range(len(obj_list)):
        obj = obj_list[n]
        
        try :
            
            module, phase= get_cropped_module_phase(obj) 
            module[module<.01*np.max(module)] = np.nan
            
            n_hist, bins = np.histogram(module[np.logical_not(np.isnan(module))], bins=100)
            
            x = (bins[1:] + bins[:-1])/2.
            
            fit, popt, pcov = GaussianAndLinearBackgroundFit(n_hist, x=x,
                                                             sig=None,
                                                             return_popt_pcov=True,
                                                             background=gaussian_background_fit,
                                                             plot=False)
            std_fit_list[n] += popt[2]
            
            if plot :
                fig, ax = plt.subplots(3,3, figsize=(8,8))
                plot_2D_slices_middle(obj, fig=fig,ax=ax[1:])
                ax[0,1].hist(module.flatten(), bins=100)
                ax[0,1].plot(x,fit, 'r-')
                fig.delaxes(ax[0,0])
                fig.delaxes(ax[0,2])
                ax[0,1].set_title('index : {}'.format(n), fontsize=20)
                fig.tight_layout()
                ax_title.append(ax[0,1])
        except :
            std_fit_list[n] += np.nan
        
      
    indices_keep = np.sort(np.argsort(std_fit_list)[:nb_keep])
    print('indices of kept object : ', indices_keep)
    
    for index in indices_keep:
        title = ax_title[index].get_title()
        ax_title[index].set_title(title, color='red', fontsize=20)
        
    return obj_list[indices_keep]

######################################################################################################################################
##############################         Put all object in list in same complex conjugate       ########################################
###################################################################################################################################### 

# Careful, doesn't work very well unfortunatly...

def pearson_correlation_coef(img1, img2):
    img_corr = np.zeros((2,)+img1.shape)
    for n,img in enumerate((img1,img2)):
        img_corr[n] = img-np.mean(img)
    denominator = np.sqrt(np.sum(img_corr[0]**2.))*np.sqrt(np.sum(img_corr[1]**2.))
    numerator = np.sum(img_corr[0]*img_corr[1])
    return numerator/denominator

def force_same_complex_conjugate_object_list(obj_list):
    obj_ref = obj_list[0] # I take the first object as the reference
    module_ref, phase_ref = get_cropped_module_phase(obj_ref)
    
    obj_list_updated = np.zeros(obj_list.shape, dtype='complex128')
    
    for n, obj in enumerate(obj_list):
        module, phase = get_cropped_module_phase(obj)
        p_coef = pearson_correlation_coef(module_ref, module)

        obj_conj = get_complex_conjugate(obj)
        module_conj, phase_conj = get_cropped_module_phase(obj_conj)
        p_coef_conj = pearson_correlation_coef(module_ref, module_conj)

        if p_coef>p_coef_conj:
            obj_list_updated[n] += obj
        else:
            obj_list_updated[n] += obj_conj
    return obj_list_updated

# Other option. Make combination of mode decomposition and select the index having largest 1st mode
def automatic_conj_index_finding(file_list, index_best_recon,
                                conj_index_start=0):
    conj_index = [conj_index_start]
    obj_ref = np.load(file_list[index_best_recon[0]])['obj']
    if conj_index_start==1:
#         obj_ref = np.conj(obj_ref[::-1,::-1,::-1])
        obj_ref = np.conj(np.flip(obj_ref,axis=range(obj_ref.ndim)))

    for n in range(1,len(index_best_recon)):
        obj2 = np.load(file_list[index_best_recon[n]])['obj']
#         obj2_conj = np.conj(obj2[::-1,::-1,::-1])
        obj2_conj = np.conj(np.flip(obj2,axis=range(obj2.ndim)))
        mode1 = []
        
        conjugate=0
        for obj in [obj2, obj2_conj]:
            obj_list = np.array([obj_ref, obj])
            obj_list = center_object_list(obj_list)
            obj_ref, weights = mode_decomposition(obj_list)
            mode1.append(100*weights[0])
            print(f'object {n} conj_index {conjugate} : {round(1e2*weights[0])} %')
            conjugate += 1

        conj_index.append(np.argmax(mode1))
    print('\nconj_index : ', conj_index)
    return conj_index


#########################################################################################################################################
################################                    Modes decomposition                  ################################################
#########################################################################################################################################


def mode_decomposition(obj_list,
                       plot=False):
    obj_modes, weights = ortho_modes(obj_list, return_weights=True)
    
    if plot:
        Nb_modes = obj_modes.shape[0]
        
        # For 2D object
        if obj_modes[0].ndim ==2 :
            fig,ax = plt.subplots(2,Nb_modes, figsize=(4*Nb_modes,8))
            for n in range(Nb_modes):
                plot_object_module_phase_2d(obj_modes[n], ax=ax[:,n], fig=fig)

                ax[0,n].set_title('mode {}\n{}%'.format(n, round(100*weights[n],1)), fontsize=20)
            ax[0,0].set_ylabel('module', fontsize=20)
            ax[1,0].set_ylabel('phase', fontsize=20)
            
        # For 3D object 
        if obj_modes[0].ndim ==3 :
            fig,ax = plt.subplots(Nb_modes,3, figsize=(4*Nb_modes,8))
            for n in range(Nb_modes):
                plot_2D_slices_middle_only_module(obj_modes[n], ax=ax[n], fig=fig)
                ax[n,0].set_ylabel('mode {}\n{}%'.format(n, round(100*weights[n],1)), fontsize=20)
            
        fig.tight_layout()
        
    return obj_modes[0], weights

#########################################################################################################################################
#################################                    Remove linear phase ramp                  ##########################################
#########################################################################################################################################

def linear_fit(array):

    pos = np.indices(array.shape)
    f = array[np.logical_not(np.isnan(array))]
    pos = [ p[np.logical_not(np.isnan(array))] for p in pos]

    X = np.zeros((len(f),len(pos)))
    for n in range(len(pos)):
        X[:,n] += pos[n]
    reg = LinearRegression().fit(X, f)
    
    return reg 

def remove_phase_linear_fit(phase):
    reg = linear_fit(phase)
    pos = np.indices(phase.shape)
#     ramp = np.sum([reg.coef_[n]*pos[-n] for n in range(len(pos))], axis=0) + reg.intercept_
#     ramp = -ramp
    ramp = np.sum([reg.coef_[n]*pos[+n] for n in range(len(pos))], axis=0) + reg.intercept_
    return ramp

def remove_phase_ramp_gradient_average(phase):
    # Get the slope
    grad = np.array(np.gradient(phase))#EB_custom_gradient(phase)
    slope = np.nanmean(grad, axis=(1,2,3))
    
    pos = np.indices(phase.shape).astype('float64')
    ramp = np.sum(pos * slope[:,None,None,None], axis=0)    
    return ramp

def remove_phase_ramp(obj,
                      threshold_module=.3,
                      crop=False,
                      return_ramp=False,
                      method='fit', # 'gradient'
                      plot=False):
    
    module, phase = get_cropped_module_phase(obj, crop=crop, unwrap=True, threshold_module=threshold_module)
    
    if method=='fit':
        ramp = remove_phase_linear_fit(phase)
    elif method=='gradient':
        ramp = remove_phase_ramp_gradient_average(phase)
    else:
        raise ValueError('no ramp computation method given')
        
    _, phase_full = get_cropped_module_phase(obj, crop=crop, unwrap=True, threshold_module=0.)
    phase_no_ramp = phase_full - ramp
    phase_no_ramp -= np.nanmean(phase_no_ramp) # Just remove a phase constant

    obj_no_ramp = np.abs(obj)*np.exp(1.0j*phase_no_ramp)
    if plot:
        
        if obj.ndim==2:
            fig, ax = plt.subplots(2,2, figsize=(8,8))            
            plot_object_module_phase_2d(obj, fig=fig, ax=ax[:,0], crop=crop,
                                       threshold_module=threshold_module)
            plot_object_module_phase_2d(obj_no_ramp, fig=fig, ax=ax[:,1], crop=crop,
                                       threshold_module=threshold_module)

            ax[0,0].set_title('object', fontsize=20)
            ax[0,1].set_title('object without phase ramp', fontsize=20)
            ax[0,0].set_ylabel('module', fontsize=20)
            ax[1,0].set_ylabel('phase', fontsize=20)
            fig.tight_layout()
            
            module, phase = get_cropped_module_phase(obj, crop=crop, 
                                                     threshold_module=threshold_module)
            module_no_ramp, phase_no_ramp = get_cropped_module_phase(obj_no_ramp, crop=crop,
                                                                    threshold_module=threshold_module)
            plt.figure()
            plt.matshow(phase-phase_no_ramp, cmap='hsv')
            plt.colorbar()
            plt.title('phase ramp', fontsize=20)
            
        if obj.ndim==3:
            fig, ax = plt.subplots(4,3, figsize=(3*4,3*3))
            plot_2D_slices_middle_only_module(obj, fig=fig, ax=ax[0], crop=crop)
            plot_2D_slices_middle_only_phase(obj, fig=fig, ax=ax[1], threshold_module=threshold_module, crop=crop)
            plot_2D_slices_middle_only_phase(obj_no_ramp, fig=fig, ax=ax[2], threshold_module=threshold_module, crop=crop)
            
            fake_obj_ramp = np.abs(obj)*np.exp(1.0j*(ramp))
            plot_2D_slices_middle_only_phase(fake_obj_ramp, fig=fig, ax=ax[3], threshold_module=threshold_module, crop=crop)
            
            ax[0,0].set_ylabel('module', fontsize=20)
            ax[1,0].set_ylabel('phase', fontsize=20)
            ax[2,0].set_ylabel('phase - ramp', fontsize=20)
            ax[3,0].set_ylabel('ramp', fontsize=20)
            fig.tight_layout()           
    if return_ramp:
        return obj_no_ramp, ramp
    else:
        return obj_no_ramp
    
#########################################################################################################################################
##############              Remove very large linear phase ramp (on purposed off-centered Bragg peak)             #######################
#########################################################################################################################################

def FT_remove_large_ramp(obj, 
                         offsets = None,
                         plot=False):
    F_recon = ifftshift(fftn(fftshift(obj)))
    
    I_recon = np.abs(F_recon)**2.

    if plot:
        plot_3D_projections(I_recon)

    if offsets is None:
        I_recon, offsets = center_the_center_of_mass(I_recon, return_offsets=True)
    else:
        I_recon = np.roll(I_recon, offsets, axis=range(I_recon.ndim))
    
    if plot:
        plot_3D_projections(I_recon)

    F_recon = np.roll(F_recon, offsets, axis=range(F_recon.ndim)) 

    obj = ifftshift(ifftn(fftshift(F_recon)))
    return obj, offsets

#########################################################################################################################################
########################################                    Apodization                  ################################################
#########################################################################################################################################

# def blackman_window_2D(shape, normalization=1):
#     """
#     Create a 2d Blackman window based on shape.

#     :param shape: tuple, shape of the 2d window
#     :param normalization: value of the integral of the backman window
#     :return: the 2d Blackman window
#     """
#     nby, nbx = shape
#     array_y = np.blackman(nby)
#     array_x = np.blackman(nbx)
#     blackman2 = np.ones((nby, nbx))
#     for idy in range(nby):
#         blackman2[idy, :] = array_y[idy] * array_x
#     blackman2 = blackman2 / blackman2.sum() * normalization
#     return blackman2

# import bcdi.utils.validation as valid
# import bcdi.utils.utilities as util
# import gc
# import bcdi.graph.graph_utils as gu
# from numpy.fft import fftn, ifftn, ifftshift
# def apodize2D(obj, initial_shape, window_type, debugging=False, **kwargs):

#     amp, phase = np.abs(obj), np.angle(obj)
#     # calculate the diffraction pattern of the reconstructed object
#     nb_y, nb_x = amp.shape
#     nby, nbx = initial_shape
# #     myobj = crop_pad2D(amp * np.exp(1j * phase), (nby, nbx))
#     myobj = amp * np.exp(1j * phase)

#     del amp, phase
#     gc.collect()

#     my_fft = fftshift(fftn(myobj))
#     del myobj
#     gc.collect()
#     fftmax = abs(my_fft).max()
# #     print("Max FFT=", fftmax)
    
#     if window_type == "blackman":
# #         print("Apodization using a 3d Blackman window")
#         window = blackman_window_2D(initial_shape)
#     else:
#         raise ValueError("Invalid window type")

#     my_fft = np.multiply(my_fft, window)
#     del window
#     gc.collect()
#     my_fft = my_fft * fftmax / abs(my_fft).max()
# #     print("Max apodized FFT after normalization =", abs(my_fft).max())

#     myobj = ifftn(ifftshift(my_fft))
#     del my_fft
#     gc.collect()

#     return myobj

# def EB_apodization(obj, data2d_shape,
#                    plot=False):

#     obj_apodized = apodize2D(obj, data2d_shape, window_type='blackman')
    
#     if plot:
#         fig,ax = plt.subplots(2,2, figsize=(8,8))
#         plot_object_module_phase_2d(obj, ax=ax[:,0], fig=fig, vmin=None, vmax=None, unwrap=True)
#         plot_object_module_phase_2d(obj_apodized, ax=ax[:,1], fig=fig, vmin=None, vmax=None, unwrap=True)

#         ax[0,0].set_title('Original object', fontsize=20)
#         ax[0,1].set_title('Apodized object', fontsize=20)

#         ax[0,0].set_ylabel('module', fontsize=20)
#         ax[1,0].set_ylabel('phase', fontsize=20)
#         fig.tight_layout()
        
#     return obj_apodized


def blackman_window(shape):
    index = np.indices(shape)
    blackman = np.ones(shape)
    for ii,n in enumerate(index):
        blackman = blackman * (.42-.5*np.cos(2.*pi*n/(shape[ii]-1)) + .08*np.cos(4.*pi*n/(shape[ii]-1)) )
    blackman = blackman/np.sum(blackman)
    return blackman

def apodize(obj,
            window_type='blackman',
            plot=False):
    shape = obj.shape
    if window_type=='blackman':
        window = blackman_window(shape)
    else:
        raise ValueError('only \'blackman\' available for window_type right now...')
    
    Fexp = ifftshift(fftn(fftshift(obj)))
    Fexp_max = np.abs(Fexp).max()
    Fexp = np.multiply(Fexp, window)
    Fexp = Fexp * Fexp_max / np.abs(Fexp).max()
    
    obj_apodized = ifftshift(ifftn(fftshift(Fexp)))
    
    if plot:
        if obj_apodized.ndim==3:
            plot_2D_slices_middle(obj_apodized, fig_title='apodized object')
        if obj_apodized.ndim==2:
            plot_object_module_phase_2d(obj_apodized, fig_title='apodized object')
    return obj_apodized


######################################################################################################################################
##############################                    Combine objects function                    ########################################
###################################################################################################################################### 

def combine_reconstructions(file_list, index_best_recon,
                            conj_index='auto', conj_index_start=0,
                            apodization = True,
                            large_phase_ramp = False,
                            oversampling_check = True,
                            plot=False, plot_recon=True, plot_result=True):

    file_ref = np.load(file_list[index_best_recon[0]]) # file as reference use for later strain calculation
    
    obj_list = []
    for file in file_list[index_best_recon]:
        data = np.load(file)
        obj_list.append(data['obj'])
    obj_list = np.array(obj_list)
    
    # Remove large phase ramp in case of Bragg off-centered on purpose
    if large_phase_ramp:
        obj_list[0], offsets = FT_remove_large_ramp(obj_list[0])
        for n in range(1,len(obj_list)):
            obj_list[n], _ = FT_remove_large_ramp(obj_list[n], offsets=offsets)
        
    # center all objects
    obj_list = center_object_list(obj_list)
    
    # Force same complex conjugate inverse for all objects
    if conj_index == 'auto':
        conj_index = automatic_conj_index_finding(file_list, index_best_recon, conj_index_start=conj_index_start)

    if conj_index is None:
        obj_list = force_same_complex_conjugate_object_list(obj_list) # Not working very well for 3D data
    else:
        for n in range(len(obj_list)):
            if conj_index[n] ==1 :
#                 obj_list[n] = np.conj(obj_list[n][::-1,::-1,::-1])
                obj_list[n] = np.conj(np.flip(obj_list[n],axis=range(obj_list[n].ndim)))
                
    # center again all objects (just to feel safe)
    obj_list = center_object_list(obj_list)
    
    # mode decomposition
    if len(obj_list)>1:
        obj, weights = mode_decomposition(obj_list, plot=plot)
    else:
        # no decomposition if we selected only 1 object
        obj = obj_list[0]
        weights = [1]
        
    for n in range(len(weights)):
        print('mode {} : {} %'.format(n+1, round(1e2*weights[n], 2)))
        
    if plot_recon or plot:
        I_exp = np.load(str(file_ref['preprocessed_datapath']))['data']
        compare_reconstuction_to_real_data(I_exp, obj)

    if apodization :
        obj = apodize(obj,
                    window_type='blackman',
                    plot=plot)

    if oversampling_check:
        print('oversampling ratio :', compute_oversampling_ratio(obj))
        
        
    if plot_result or plot:
        if obj.ndim==3:
            plot_2D_slices_middle(obj, threshold_module=.3)
        elif obj.ndim==2:
            plot_object_module_phase_2d(obj)
    
    return obj, file_ref, weights

#########################################################################################################################################
############################                    Remove constant phase at the COR                  #######################################
#########################################################################################################################################

# def remove_phase_constant_center_of_mass(obj,
#                                           plot=False):
#     module, phase = get_cropped_module_phase(obj, unwrap=True)
#     index = center_of_mass(module)
#     phase_offset = phase[round(index[0]), round(index[1])]
#     obj_corr = np.abs(obj)*np.exp(1.0j*(np.angle(obj)-phase_offset))
    
#     if plot:
#         plot_object_module_phase_2d(obj, unwrap=True)
#         plot_object_module_phase_2d(obj_corr, unwrap=True)
    
#     return obj_corr

#########################################################################################################################################
############################                  Save post-processed reconstruction                  #######################################
#########################################################################################################################################

def save_final_object(obj_ortho, strain, d_spacing, voxel_sizes,
                      file_ref, 
                      path_reconstruction,
                      additional_dict = {},
                      obj_non_ortho=None,
                      verbose=False):
    # A safety check
    if path_reconstruction[-1] != '/' :
        path_reconstruction += '/'
    
    file_ref.allow_pickle = True
    dico = dict(file_ref)
    for key in ['obj', 'support']:
        if key in dico.keys():
            del(dico[key])
    for key in additional_dict.keys():
        if key in dico.keys():
            del(dico[key])
    
    if obj_non_ortho is not None:
        additional_dict['obj_non_ortho'] = obj_non_ortho
        
    path_final_obj = path_reconstruction + '/final_obj/'
    check_path_create(path_final_obj)
    savename = path_final_obj + 'final_object'
    
    print('final object saved at : ', savename+'.npz')
    np.savez(savename, obj_ortho=obj_ortho, 
             strain=strain, d_spacing=d_spacing, voxel_sizes=voxel_sizes,
             **dico, **additional_dict)
    return

#########################################################################################################################################
###########################                  Remove bad reconstruction at the end                  ######################################
#########################################################################################################################################

def remove_bad_reconstructions(file_list,
                               index_best_recon,
                               verbose=False):
    for n in range(len(file_list)):
        if n not in index_best_recon:
            file = file_list[n]
            os.remove(file)
            if verbose:
                print('removed ',file)
    return
