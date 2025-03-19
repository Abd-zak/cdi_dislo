import numpy as np
import pylab as plt


from cdi_dislo.common_imports import *
from cdi_dislo.ewen_utilities.Reconstruction                    import *
from cdi_dislo.ewen_utilities.PostProcessing                    import force_same_complex_conjugate_object_list
def make_several_reconstruction(data, nb_recon, params_init,
                                obj_list=None, support_list=None):
    '''
    My reconstruction function. It's calling my CDI_one_reconstruction() custom function.
    '''
    
    obj_list_new = np.zeros((nb_recon,)+data.shape).astype('complex128')
    support_list_new = np.zeros((nb_recon,)+data.shape)
    
    for n in range(nb_recon):
        
        params = dict(params_init)
        if obj_list is not None:
            # not used during the 1st reconstruction. This is only used starting the 2nd step of the genetic algo
            params['obj_init'] = obj_list[n]
        
        fail=True
        while fail:
            try: # I use this try in case pynx reconstruction get lost and crashes
                obj, llk, support, return_dict = CDI_one_reconstruction(data, params)
                obj_list_new[n] += obj
                support_list_new[n] += support
                fail=False
            except:
                pass
        
    return obj_list_new, support_list_new

def sharpness_metric(obj, support):
    '''
    David Yang sharpness metric. The object having the minimum sharpness is considered as the best.
    '''
    module = np.abs(obj) * support
    return np.mean(module ** 4.)

def sort_objects(obj_list, support_list,
                 plot=False):
    '''
    Sort object list such that the best object is put first
    '''
    metric_list = np.zeros(len(obj_list))
    for n in range(len(obj_list)):
        metric_list[n] += sharpness_metric(obj_list[n], support_list[n])
    
    # Sort the object and supports
    indices = np.argsort(metric_list)
    
    metric_list = metric_list[indices]
    obj_list = obj_list[indices]
    support_list = support_list[indices]
    
    if plot:
        for n in range(len(obj_list)):
            plot_2D_slices_middle(obj_list[n], fig_title=f'sharpness : {metric_list[n]}')
    return metric_list, obj_list, support_list

def genetic_update_object_list(obj_list,
                               check_inv_complex_conjugate=True):
    '''
    Update the object list using the best one (first object in the list, obj_list[0])
    '''
    
    if check_inv_complex_conjugate:
        obj_list = force_same_complex_conjugate_object_list(obj_list)
        obj_list = center_object_list(obj_list)
        
    for n in range(1,len(obj_list)):
        obj_list[n] = np.sqrt(obj_list[n] * obj_list[0])
    return obj_list

def press_button_genetic(data, nb_recon, nb_genetic,
                         mask=None,
                         params_init=None,
                         plot=True):
    obj_list = None
    support_list = None
    
    if params_init is None:
        params_init = default_cdi_parameters()
        print('parameters for BCDI :\n')
        params_init['show_cdi'] = 0
        params_init['plot_result'] = False
        params_init['center_data'] = False
#         for key in params_init.keys():
#             print(f'{key} :  {params_init[key]}')

    if mask is not None:
        params_init['mask'] = mask
    
    for n_gen in range(nb_genetic):
        print(f'\nGENETIC STEP : {n_gen+1}\n')
        if n_gen!=0:
            obj_list = genetic_update_object_list(obj_list)

        obj_list, support_list = make_several_reconstruction(data, nb_recon, params_init,
                                    obj_list=obj_list, support_list=support_list)
        metric_list, obj_list, support_list = sort_objects(obj_list, support_list, plot=False)
        obj_result = obj_list[0]
    if plot:
        plot_2D_slices_middle(obj_result, threshold_module=.3, fig_title='Result of genetic algorithm')
    return obj_result

def save_reconstruction_genetic(file_dict, obj, llk, support, n_reconstruction):
    '''
    Save the final object automatically.
    Not an important function> You could do that in another way if you want.
    '''
    
    if file_dict['orthogonalization']:
        ortho_string = '_ortho'
    else:
        ortho_string = ''
        
    path_save = 'Reconstructions_Genetic/{}_scan{}{}{}/'.format(file_dict['h5file'], file_dict['scan_nb'],ortho_string,
                                                                       file_dict['savename_add_string'])

    check_path_create('Reconstructions_Genetic')
    check_path_create(path_save)
        
    savename = path_save + 'reconstruction{}_{}'.format(ortho_string, n_reconstruction)
    
    np.savez_compressed( savename, obj=obj, support=support, llk=llk, **file_dict )
    return

def make_several_reconstructions_genetic(data, file_dict, nb_recon, nb_genetic, Nb_reconstruction,  
                                           data_centering=False,
                                           mask=None,
                                           params_init=None,  # default reconstruction parameters
                                           dont_erase_previous_recon=True,
                                           plot=False, verbose=False):
    
    if file_dict is None:
        file_dict = dummy_file_dict()
    
    if dont_erase_previous_recon:
        if file_dict['orthogonalization']:
            ortho_string = '_ortho'
        else:
            ortho_string = ''

        path_save = 'Reconstructions_Genetic/{}_scan{}{}{}/'.format(file_dict['h5file'], file_dict['scan_nb'],ortho_string,
                                                                           file_dict['savename_add_string'])
        if os.path.exists(path_save):
            files = get_npz_files(path_save)
            if len(files) != 0:
                recon_nb = [int(f.split('.')[0].split('_')[-1]) for f in files]
                last_recon_nb = max(recon_nb)
            else:
                last_recon_nb = 0
        else:
            last_recon_nb = 0 
    else:
        last_recon_nb = 0
        
    for n_reconstruction in range(Nb_reconstruction):
        if plot:
            plt.figure()
            plt.title('Reconstruction {}'.format(n_reconstruction), fontsize=20)
            
        obj = press_button_genetic(data, nb_recon, nb_genetic,
                         mask=mask,
                         params_init=params_init,
                         plot=True)
        llk = None
        module = np.abs(obj)
        threshold_module = .3
        support = module > threshold_module * np.max(module)
        return_dict = {}

        save_reconstruction_genetic(file_dict, obj, llk, support, last_recon_nb+1+n_reconstruction)
    return
