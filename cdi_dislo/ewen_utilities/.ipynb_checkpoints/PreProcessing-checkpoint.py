import pylab as plt
import numpy as np
import xrayutilities as xu

from cdi_dislo.common_imports import *
from cdi_dislo.ewen_utilities.plot_utilities                      import *
from cdi_dislo.ewen_utilities.FilterBackground                    import *
from cdi_dislo.ewen_utilities.Global_utilities                    import *
from scipy.ndimage.measurements import center_of_mass
import h5py

my_cmap = MIR_Colormap()


###########################################################################################################################################
#################################            Check detector saturation            #########################################################
###########################################################################################################################################

def check_detector_saturation(data, scan):
    maxi = np.nanmax(data)
    print('maximum counts : ', maxi)
    if scan.data_type=='PETRA':
        print('I don\'t remember the detector saturation level at PETRA...')
    else:
        if scan.detector == 'mpx1x4':
            if maxi>150000:
                print(f'{scan.detector} detector in non-linear dynamic range')
    return

###########################################################################################################################################
#######################################            Create q arrays            #############################################################
###########################################################################################################################################

def create_Q_array(scan, 
                   roi=None,
                   det_calib=None,
                   energy=None,
                   switch_x_y_direct_beam_position=False,
                   chi=0, eta=None, phi=None, delta=None, nu=None,
                   cxi_convention=False,
                   verbose=False):
    
    if det_calib is None :
        det_calib = scan.getDetCalibInfo()
    
    if 'detrot' not in det_calib.keys():
        det_calib['detrot'] = 0
    if 'tiltazimuth' not in det_calib.keys():
        det_calib['tiltazimuth'] = 0
    if 'tilt' not in det_calib.keys():
        det_calib['tilt'] = 0
    if det_calib['distance'] < 0:
        print('detector calibration has a negative distance. Forcing it positive')
        det_calib['distance'] = abs(det_calib['distance'])
    
    if roi is None:
        roi = [0,None, 0,scan.detector_shape[0], 0, scan.detector_shape[1]]
    

        
    if eta is None:
        eta = scan.getMotorPosition('eta')
    if phi is None:
        phi = scan.getMotorPosition('phi')
    if delta is None:
        if 'spec' in str(scan.__class__) or scan.data_type=='PETRA':
            delta = scan.getMotorPosition('del')
        else:
            delta = scan.getMotorPosition('delta')
       
    if nu is None:
        if scan.data_type=='PETRA':
            nu = -scan.getMotorPosition('gam')
        else:
            nu = scan.getMotorPosition('nu')
        
    if energy is None:
        energy = scan.getEnergy()
        
    beam_center_y = det_calib['beam_center_y']
    beam_center_x = det_calib['beam_center_x'] 
    if switch_x_y_direct_beam_position:
        beam_center_x, beam_center_y = beam_center_y, beam_center_x
        
        
    qconv = xu.experiment.QConversion(['y-','z-','x+'],['z-','y-'],[1,0,0]) # 2S+2D goniometer (simplified ID01 goniometer, sample: eta, phi detector nu,del
    # convention for coordinate system: x downstream; z upwards; y to the "outside" (righthanded)
    hxrd = xu.HXRD([1,0,0],[0,0,1], en=energy, qconv=qconv)

    if scan.detector == 'CITIUS':
        vertical_orientation = 'z+'
    else:
        vertical_orientation = 'z-'
    hxrd.Ang2Q.init_area(vertical_orientation, 'y+', 
                     cch1=beam_center_y-roi[2], cch2=beam_center_x-roi[4],
                     Nch1=roi[3]-roi[2], Nch2=roi[5]-roi[4],
                     pwidth1=det_calib['y_pixel_size'], pwidth2=det_calib['x_pixel_size'], distance=det_calib['distance'],
                        detrot=det_calib['detrot'],
                        tiltazimuth=det_calib['tiltazimuth'],
                        tilt=det_calib['tilt'])

    qx,qy,qz = hxrd.Ang2Q.area(eta,
                               phi,
                               chi,
                               nu,
                               delta)
    
    # Apply the ROI along the rocking curve angle
    qx = qx[roi[0]:roi[1]]
    qy = qy[roi[0]:roi[1]]
    qz = qz[roi[0]:roi[1]]
    
    if verbose:
        if switch_x_y_direct_beam_position:
            print("\x1b[31m beam_center_x and beam_center_y were switched when loading the detector calibration parameters ! \x1b[0m")
            print('\n')
            
        print('phi : {}'.format(phi))
        print('eta : {}'.format(eta))
        print('chi : {}'.format(chi))
        print('delta : {}'.format(delta))
        print('nu : {}'.format(nu))

        print('\nenergy (eV) :', energy)
        print('detector_distance :{} m'.format(det_calib['distance']))
        print('beam_center_x :{} '.format(beam_center_x))
        print('beam_center_y :{} '.format(beam_center_y))
        print('detector pixel size x : {}m'.format(det_calib['x_pixel_size']))
        print('detector pixel size y : {}m'.format(det_calib['y_pixel_size']))

    if cxi_convention:
        qy,qz = qz,qy
        print('qy and qz were switched in order to end up with the good orientation for the VTI')
    return qx,qy,qz

###########################################################################################################################################
###################################            q space transformation            ##########################################################
###########################################################################################################################################


def Q_space_transformation(data,
                           qx,qy,qz,
                           return_3D_q=True,
                           mask=None,
                           plot=False):
    maxbins = []
    for dim in (qx, qy, qz):
        maxstep = max((abs(np.diff(dim, axis=j)).max() for j in range(3)))
        maxbins.append(int(abs(dim.max() - dim.min()) / maxstep))
#     print(f'Maximum number of bins based on the sampling in q: {maxbins}')
    
    if mask is not None: 
        mask_for_grid = np.ones(mask.shape)
        
        gridder_mask = xu.FuzzyGridder3D(*maxbins)  
        gridder_mask(qx,qy,qz, mask)
        mask = gridder_mask.data
        
        gridder_mask = xu.FuzzyGridder3D(*maxbins)  
        gridder_mask(qx,qy,qz, mask_for_grid)
        mask_for_grid = gridder_mask.data
        
        mask[mask_for_grid==0] = 1
        
    gridder = xu.FuzzyGridder3D(*maxbins)  

    gridder(qx,qy,qz,data)
    qx, qy, qz = [gridder.xaxis, gridder.yaxis, gridder.zaxis]
    data_q_space = gridder.data
    
    if return_3D_q:
        qx, qy, qz = np.meshgrid(qx,qy,qz, indexing='ij')
    
    if plot :
        plot_3D_projections(data, fig_title='original data', colorbar=True)
        plot_3D_projections(data_q_space, fig_title='orthogonalized data', colorbar=True)
    
    if mask is not None:
        return data_q_space, mask, qx, qy, qz
    else:
        return data_q_space, qx, qy, qz


def center_of_mass_calculation_two_steps(data, 
                                         crop = 50, 
                                         plot=False):
    
    
#     piz,piy,pix = center_of_mass(data)
    piz,piy,pix = np.unravel_index(data.argmax(), data.shape)
    
    
    
    cropping_dim0 =( max([0, int(piz-crop/2)]),  min(int(piz+crop//2), data.shape[0]-1)) 
    cropping_dim1 =( max([0, int(piy-crop/2)]),  min(int(piy+crop//2), data.shape[1]-1)) 
    cropping_dim2 =( max([0, int(pix-crop/2)]),  min(int(pix+crop//2), data.shape[2]-1)) 
    
    piz2,piy2,pix2 = center_of_mass(data[cropping_dim0[0]:cropping_dim0[1],
                                         cropping_dim1[0]:cropping_dim1[1],
                                         cropping_dim2[0]:cropping_dim2[1]])

    piz,piy,pix = int(round(cropping_dim0[0]+piz2)), int(round(cropping_dim1[0]+piy2)), int(round(cropping_dim2[0]+pix2))
    
    if plot:
        fig, ax = plt.subplots(1,3, figsize=(12,4))
        plot_3D_projections(data, fig=fig, ax=ax)
        ax[0].scatter(pix, piy, color='w')
        ax[1].scatter(pix, piz, color='w')
        ax[2].scatter(piy, piz, color='w')
    return piz,piy,pix

###########################################################################################################################################
####################################            Automatic ROI selection            ########################################################
###########################################################################################################################################

def automatic_roi_selection(data,
                            roi_init=[0,-1,0,-1,0,-1],
                            crop=False, crop_array=[0,0,0],
                            crop_with_final_size=False, final_size_array=None,
                            plot=False):
    
    data = data[roi_init[0]:roi_init[1], roi_init[2]:roi_init[3], roi_init[4]:roi_init[5]]
    
    piz,piy,pix = center_of_mass_calculation_two_steps(data)
    
    minz = min(piz, data.shape[0]-piz)
    miny = min(piy, data.shape[1]-piy)
    minx = min(pix, data.shape[2]-pix)
    roi = [piz-minz,piz+minz+1, 
           piy-miny,piy+miny+1, pix-minx,pix+minx+1]
    
    if plot:
        # Check center of mass
        fig, ax = plt.subplots(1,3, figsize=(12,4))
        ax[0].matshow(np.log(np.sum(data,axis=0)))
        ax[0].scatter(pix,piy, color='r')
        ax[1].matshow(np.log(np.sum(data,axis=1)))
        ax[1].scatter(pix,piz, color='r')
        ax[2].matshow(np.log(np.sum(data,axis=2)))
        ax[2].scatter(piy,piz, color='r')
        
        # Check ROI
        plot_3D_projections(data[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]])
        
    if crop:
        roi = crop_roi(data, roi, crop_array, plot=plot, fig_title='cropped')
    if crop_with_final_size:
        roi = crop_roi_given_final_size(data, roi, final_size_array, plot=plot, fig_title='cropped')
        
    roi[0] += roi_init[0]
    roi[1] += roi_init[0]
    roi[2] += roi_init[2]
    roi[3] += roi_init[2]
    roi[4] += roi_init[4]
    roi[5] += roi_init[4]
    return np.array(roi)

def crop_roi(data, roi,
             crop_array,
             plot=False, fig_title=None):
 
    roi_crop = [roi[0]+crop_array[0]//2, roi[1] - crop_array[0]//2, 
                roi[2]+crop_array[1]//2,roi[3]-crop_array[1]//2,
                roi[4]+crop_array[2]//2,roi[5]-crop_array[2]//2]
    if plot:
        plot_3D_projections(data[roi_crop[0]:roi_crop[1], roi_crop[2]:roi_crop[3], roi_crop[4]:roi_crop[5]],
                            fig_title=fig_title)
    
    return roi_crop

def crop_roi_given_final_size(data,roi,
                              final_size_array,
                              plot=False, fig_title=None):
    
    for n in range(data.ndim):
        if final_size_array[n] is None:
            final_size_array[n] = roi[2*n + 1]-roi[2*n]
    
    roi_crop = [roi[0]+(roi[1]-roi[0] - final_size_array[0])//2, roi[1] - (roi[1]-roi[0] - final_size_array[0])//2, 
                roi[2]+(roi[3]-roi[2] - final_size_array[1])//2,roi[3]-(roi[3]-roi[2] - final_size_array[1])//2,
                roi[4]+(roi[5]-roi[4] - final_size_array[2])//2,roi[5]-(roi[5]-roi[4] - final_size_array[2])//2]
    
    for n in range(data.ndim):
        roi_crop[2*n] = max(0,roi_crop[2*n])

    if plot:
        plot_3D_projections(data[roi_crop[0]:roi_crop[1], roi_crop[2]:roi_crop[3], roi_crop[4]:roi_crop[5]],
                            fig_title=fig_title)
    
    return roi_crop

###########################################################################################################################################
##########################         More involved automatic ROI selection for 2D data          #############################################
###########################################################################################################################################


def find_1D_center_custom(array1d, x=None,
                          remove_min=True):
    
    if x is None :
        x = np.arange(len(array1d))
    
    argmax = np.argmax(array1d)
    array1d = array1d[argmax-15:argmax+15]
    x = x[argmax-15:argmax+15]
    
    if remove_min:
        array1d = array1d - np.min(array1d)
    
    cen = np.sum(array1d*x)/np.sum(array1d)
    
    return cen

def find_center_custom(detector_sum, roi_init=None):
    
    if roi_init is None:
        roi_init = [0, detector_sum.shape[0]-1,
                    0, detector_sum.shape[1]-1]
    
    # Find center in vertical
    projection = np.sum(detector_sum[roi_init[0]:roi_init[1], roi_init[2]:roi_init[3]],axis=1)
    cen_vert = find_1D_center_custom(projection)
    cen_vert += roi_init[0]
    
    # Find center in horizontal 
    projection = np.sum(detector_sum[roi_init[0]:roi_init[1], roi_init[2]:roi_init[3]],axis=0)
    cen_hori = find_1D_center_custom(projection)
    cen_hori += roi_init[2]
    
    return cen_vert, cen_hori

def automatic_roi_selection_2D_BCDI_data(detector_sum, roi_size_vertical, roi_size_horizontal,
                                         roi_init = None):
    
    cen_vert, cen_hori = find_center_custom(detector_sum, roi_init=roi_init)
    roi = [round(cen_vert)-roi_size_vertical//2, round(cen_vert)+roi_size_vertical//2, 
           round(cen_hori)-roi_size_horizontal//2, round(cen_hori)+roi_size_horizontal//2]
    
    return roi

def check_custom_detector_ROI(scan, data, roi):
    extent = [ [roi[2],roi[3],roi[0],roi[1]],
               [roi[2],roi[3],0,len(data)],
               [roi[0],roi[1],0,len(data)]]

    xlabels = ['detector horizontal', 'detector horizontal', 'detector vertical']
    ylabels = ['detector vertical', scan.motor_name, scan.motor_name]

    fig, ax = plt.subplots(2,3, figsize=(12,8))
    for n in range(3):
        ax[0,n].matshow(xu.maplog(data.sum(axis=n),5,0),cmap=my_cmap, aspect='auto')

        ax[1,n].matshow(xu.maplog(data[:,roi[0]:roi[1],roi[2]:roi[3]].sum(axis=n),5,0),cmap=my_cmap, aspect='auto',
                        extent=extent[n])
        for ii in range(2):
            ax[ii,n].set_xlabel(xlabels[n], fontsize=15)
            ax[ii,n].set_ylabel(ylabels[n], fontsize=15)

    fig.tight_layout()
    
    return

# ###########################################################################################################################################
# ####################################                     Mask                      ########################################################
# ###########################################################################################################################################

# import fabio
# def standard_mask(scan, data, plot=False):
#     '''
#     Careful, not ready for the eiger. Need to be modified.
#     '''
#     if scan.detector == 'mpx1x4':
#         mask = fabio.open('/data/id01/inhouse/bellec/software/sharedipynb/gitlab/bcdi_eb/masks_flatfields/mpx4_hotmask.edf').data
#     else:
#         mask = np.ones(data.shape[1:]) 
        
#     if scan.detector == 'mpx1x4' or scan.detector == 'mpxgaas':
#         # Remove the gaps
#         mask[255:261] = 0
#         mask[:,255:261] = 0
    
#     if plot :
#         plt.matshow(mask, cmap='gray')
#         plt.title('mask', fontsize=20)
#         plt.colorbar()

#         if scan is not None:
#             fig, ax = plt.subplots(1,3, figsize=(12,4))
#             for n in range(3):
#                 ax[n].matshow(xu.maplog((data*mask).sum(axis=n),5,0),cmap=my_cmap)
#             fig.suptitle('data with mask', fontsize=20)
#             fig.tight_layout()
        
#     return mask

###########################################################################################################################################
###############################                     Saving in npz                      ####################################################
###########################################################################################################################################

def save_preprocessed_data(scan, data, qx,qy,qz,
                           orthogonalization=False,
                           mask=None,
                           path_save=None,
                           qcen=None, qmax=None,
                           additional_dict = {},
                           savename_add_string='',
                           compress=False,
                           verbose=False):
    if path_save is None:
        path_save = 'preprocessed_data_{}/'.format(scan.sample) 
        check_path_create(path_save)

    if scan.data_type=='PETRA':
        dataset_name = 'dummy'
        scan_nb=scan.scan_nb
    else:
        dataset_name = scan.h5file.split('/')[-2]
        scan_nb = int(scan.scan_string.split('.')[0])
    
    if orthogonalization:
        ortho_string = '_ortho'
    else:
        ortho_string = ''
    
    save_name = path_save + 'dataset_{}_scan_{}{}{}'.format(dataset_name, scan_nb,ortho_string, savename_add_string)
    
    # overwrite_safety
    if os.path.exists(save_name+'.npz'):
        overwrite = input(f'{save_name}.npz already exist ! This is an overwriting safety.'
                         + '\ndo you want to overwrite the file (y/n)' )
        if overwrite=='y':
            print('overwriting the file')
            pass
        else:
            print('saving stopped to avoid overwriting the file.')
            return
    
    print('preprocessed data saved in : {}.npz'.format(save_name))

    if compress:
        saving_function = np.savez_compressed
    else:
        saving_function = np.savez
        
    saving_function(save_name, data = data, mask=mask, qx=qx, qy=qy, qz=qz,
             orthogonalization=orthogonalization,
             qcen=qcen, qmax=qmax,
             scan_nb=scan_nb, sample=scan.sample, h5file=scan.h5file.split('/')[-1][:-3], 
             savename_add_string=savename_add_string,
             **additional_dict) 
#     if verbose:
#         print('data saved in : ',save_name)
    return

###########################################################################################################################################
########################         calculate q center of mass and at maximum intensity          #############################################
###########################################################################################################################################


def plot_3D_projections_qcen(data,
                             qx,qy,qz,
                             qcen,
                             suptitle=None):

    q = np.array([qx,qy,qz])
    rough_cen_index = np.unravel_index(np.argmin(np.sum( (q-qcen[:,None,None,None])**2.,axis=0)), data.shape)

    fig, ax = plt.subplots(1,3, figsize=(12,4))
    plot_3D_projections(data, fig=fig, ax=ax, cmap='gray')
    ax[0].scatter(rough_cen_index[2], rough_cen_index[1], color='r')
    ax[1].scatter(rough_cen_index[2], rough_cen_index[0], color='r')
    ax[2].scatter(rough_cen_index[1], rough_cen_index[0], color='r')
    
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=20)

    fig.tight_layout()

    return

def calculate_Q_center_of_mass(data, 
                               qx,qy,qz,
                               remove_min=True,
                               plot=False):
    if remove_min:
        data -= np.nanmin(data)

    proba = data/np.nansum(data)

    qcen = np.array([ np.nansum(proba*q) for q in (qx,qy,qz)])

    if plot:
        plot_3D_projections_qcen(data,
                             qx,qy,qz,
                             qcen,
                                suptitle='q center of mass')
    return qcen

def calculate_qmax(data, 
                   qx,qy,qz,
                   plot=False):
    index_max = np.unravel_index(data.argmax(), data.shape)
    qmax = np.array([ qx[index_max], qy[index_max], qz[index_max] ])

    if plot:
        plot_3D_projections_qcen(data,
                             qx,qy,qz,
                             qmax,
                                 suptitle='q maximum intensity')
    return qmax

###########################################################################################################################################
################################                 flatfield correction                   ###################################################
###########################################################################################################################################

def correct_flatfield(data, roi,
                      flatfield_file,
                      plot=False):
    if flatfield_file.split('.')[-1] == 'h5':
        h5f = h5py.File(flatfield_file, 'r')
        flatfield = h5f['ff/ff'][()]
    elif flatfield_file.split('.')[-1] == 'npz':
        try:
            flatfield = np.load(flatfield_file)['flatfield']
        except:
            flatfield = np.load(flatfield_file)['arr_0']
    elif flatfield_file.split('.')[-1] == 'npy':
        flatfield = np.load(flatfield_file)
    else:
        raise ValueError('flatfield problem')
    flatfield = flatfield[roi[2]:roi[3], roi[4]:roi[5]]
    data_corrected = data*flatfield[None]
        
    # Take nan's into account
    data_corrected[np.isnan(data_corrected)] = 0
    
    if plot:
        plt.figure(figsize=(8,8))
        plt.imshow(np.log(flatfield))
        plt.colorbar()
        plt.title('log flatfield (in ROI)', fontsize=20)
        
        plot_3D_projections(data, fig_title='before flatfield correction')
        plot_3D_projections(data_corrected, fig_title='After flatfield correction')
        
    return data_corrected


###########################################################################################################################################
##########################                        Rebinning guess                            ##############################################
###########################################################################################################################################

def compute_oversampling_ratio(support,
                               plot=False):
    
    '''
    Compute the oversampling ratio from the real space support.
    :support: real space support
    '''
        
    indices_support = np.where(support==1)
    size_per_dim = np.max(indices_support,axis=1) - np.min(indices_support,axis=1)
    oversampling = np.divide( np.array(support.shape), size_per_dim)
    
    if plot:
        fig,ax = plt.subplots(1,support.ndim, figsize=(5*support.ndim, 4))
        for n in range(support.ndim):
            axes = tuple(np.delete(np.arange(support.ndim), n))
            proj = np.max(support,axis=axes)
            ax[n].plot(proj)
            title = f'oversampling along axis {n}\n{round(oversampling[n],2)}'
            ax[n].set_title(title, fontsize=15)
    return oversampling

def oversampling_from_diffraction(data,
                                  support_threshold=.1,
                                  plot=False, verbose = True):
    '''
    Compute a guess of the oversampling ratio from the object auto-correlation (FT of the diffraction data).
    This way, you can have a guess of the possible rebinning before making any reconstruction.
    This works well for low strain data but not for high-strained particle.
    In practice, rebin for low-strain but not for high-strain (EB rule of thumb)
    :data: diffraction data
    :support_threshold: auto-correlation support threshold. Leave it to .1 like in pynx
    '''
    obj_autocor = np.abs(ifftshift(fftn(fftshift(data))))
    support_autocor = (obj_autocor > support_threshold * np.max(obj_autocor))
    
    if plot:
        if data.ndim==3:
            plot_3D_projections(obj_autocor, log_scale=False, cmap='gray_r', 
                                fig_title='object autocorrelation (FT of the diffraction)')
            plot_3D_projections(support_autocor, log_scale=False, cmap='gray_r', max_projection=True,
                                fig_title='support from autocorrelation')
        elif data.ndim==2:
            fig,ax = plt.subplots(1,2,figsize=(8,4))
            ax[0].matshow(obj_autocor, cmap='gray_r')
            ax[0].set_title('object autocorrelation\n(FT of the diffraction)', fontsize=15)
            ax[1].matshow(support_autocor, cmap='gray_r')
            ax[1].set_title('support from autocorrelation', fontsize=15)
        
    oversampling = compute_oversampling_ratio(support_autocor, plot=plot)
    rebinning_possibility = (oversampling//2).astype('int')
    
    if verbose:
        print('current calculated oversampling ratio :', oversampling)
        print('Possibility to rebin : ', rebinning_possibility)
        print('oversampling after rebinning : ', np.divide(oversampling, rebinning_possibility))
    return oversampling, rebinning_possibility

###########################################################################################################################################
################################                        Mask                            ###################################################
###########################################################################################################################################

def load_mask(scan,
              data, roi=None,
              plot=False):
    if hasattr(scan, 'mask'):
        if scan.mask is not None: # PETRA data
            mask = np.zeros(data.shape)
            mask += scan.mask[None,:,:]
    
    else:
        path_mask = '/data/id01/inhouse/bellec/software/sharedipynb/gitlab/bcdi_eb/saved_masks/'
        path_mask_array = path_mask+'mask_{}.npy'.format(scan.detector)
        if os.path.isfile(path_mask_array):
            mask2d = np.load(path_mask_array)
        else:
            mask2d = np.zeros(scan.detector_shape)

        if roi is not None:
            mask2d = mask2d[roi[0]:roi[1],roi[2]:roi[3]]

        mask = np.zeros(data.shape)
        mask += mask2d[None]
    
    if plot :
        fig, ax = plt.subplots(1,3, figsize=(14,6))
        for n in range(3):
            ax[n].matshow(mask.sum(axis=n),cmap='gray_r')

        # plot the projection along the 3 directions
        fig, ax = plt.subplots(1,3, figsize=(12,4))
        for n in range(3):
            ax[n].matshow(xu.maplog(((1-mask)*data).sum(axis=n),5,0),cmap=my_cmap)
        
    return mask

###########################################################################################################################################
##############            Force dimensions to be even in order to avoid future fftshift - ifftshift problems                 ##############
###########################################################################################################################################

def force_even_dimension(data, 
                         qx,qy,qz,
                         mask, 
                         verbose=False):
    s = []
    for n in range(data.ndim):
        if data.shape[n] %2 == 0:
            s.append(slice(None))
        else:
            s.append(slice(1,None,None))
        
    if qx.ndim==1: # If orthogonalization was already done
        qx_even, qy_even, qz_even = qx[s[0]], qy[s[1]], qz[s[2]]
    else:
        qx_even, qy_even, qz_even = qx[tuple(s)], qy[tuple(s)], qz[tuple(s)]

        
    data_even = data[tuple(s)]
    mask_even = mask[tuple(s)]
    
    if verbose:
        print('shape changed :\n')
        print('qx {} to {}'.format(qx.shape, qx_even.shape))
        print('qy {} to {}'.format(qy.shape, qy_even.shape))
        print('qz {} to {}'.format(qz.shape, qz_even.shape))
        print('data {} to {}'.format(data.shape, data_even.shape))
        print('mask {} to {}'.format(mask.shape, mask_even.shape))

    return data_even, qx_even,qy_even,qz_even , mask_even
