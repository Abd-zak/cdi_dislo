import numpy as np
import pylab as plt

from numpy import pi

from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from cdi_dislo.common_imports import *
from cdi_dislo.ewen_utilities.plot_utilities                      import *
from cdi_dislo.ewen_utilities.Object_utilities                    import *
from cdi_dislo.ewen_utilities.Single_peak_gaussian_fit                    import *
from cdi_dislo.ewen_utilities.Orthogonalization_real_space                    import *
from numpy import cos,sin

from scipy.signal import fftconvolve
#########################################################################################################################################
###############################                    Using the sample surface fit                  ########################################
#########################################################################################################################################

def select_border_roi2D(obj_ortho, voxel_sizes,
                        axis,roi):
    module = np.abs(obj_ortho)
    module = module[slice_middle_array_along_axis(module, axis)]
    border2D = apply_roi(module,roi)

    axes = np.arange(3)
    axes = np.delete(axes, axis)
    voxel_sizes2D = voxel_sizes[axes]

    fig,ax = plt.subplots(1,2, figsize=(8,4))
    ax[0].matshow(module,cmap='gray_r')
    ax[0].add_patch(Rectangle((roi[2], roi[1]), roi[3]-roi[2],roi[0]-roi[1], color='r',fill=False,linewidth=2))
    extent = [roi[2],roi[3],roi[1],roi[0]]
    ax[1].matshow(border2D,cmap='gray_r', extent=extent)
    
    return border2D, voxel_sizes2D

def sigmoid_plus_background_2D(xy, A, x0, theta, sig, back):
    x,y = xy
    u = [cos(theta), sin(theta)]
    factor = x * u[0] + (y-x0) * u[1]
    return back + A * 1./(1.+np.exp(-factor/sig))


def fit_surface(border2D,
                voxel_sizes2D,
                verbose=True, plot=False):
    x,y = np.indices(border2D.shape)
    x = x * voxel_sizes2D[0]
    y = y * voxel_sizes2D[1]
    border2D = border2D - np.min(border2D)
    border2D = border2D/np.max(border2D)
    popt, pcov = curve_fit(sigmoid_plus_background_2D, (x.flatten(),y.flatten()), border2D.flatten())
    
    surface_width = 2.*np.abs(popt[-2])
    print('voxel sizes (nm) : ', .1*voxel_sizes2D)
    print(f'surface width : {round(.1*surface_width,3)} nm')
    
    if plot:
        fit = sigmoid_plus_background_2D((x,y), *popt)
        fig, ax = plt.subplots(1,2, figsize=(8,4))
        ax[0].matshow(border2D, cmap='gray_r')
        ax[0].set_title('border', fontsize=15)
        ax[1].matshow(fit, cmap='gray_r')
        ax[1].set_title('sigmoid fit', fontsize=15)
        fig.tight_layout()
    return surface_width


#########################################################################################################################################
################                    Using Fourier correaltion shell on non-orthogonalized objects                  ######################
#########################################################################################################################################

def Fourier_shell_correlation(fexp1, fexp2, q,
                              N = 30,
                              plot=False):
    dq = (np.max(q) - np.min(q)) / N
    q1d = np.linspace(np.min(q), np.max(q)-dq, N) + dq/2.
    dq = q1d[1] - q1d[0]
    fsc = np.zeros(N)

    for n in range(N):
#         print(n,end=' ')
        if plot:
            if n==int(N/4):
                plot_2D_slices_middle_one_array3D(indices)
        indices = np.abs(q-q1d[n]) <= dq
        numerator = np.nansum(fexp1[indices] * np.conj(fexp2[indices]))
        denominator = np.sqrt(np.nansum(np.abs(fexp1[indices])**2.) * np.nansum(np.abs(fexp2[indices])**2.))
        fsc[n] += np.abs(numerator/denominator)
    return fsc, q1d

def Fourier_correlation_shell_plot(fsc, q1d, threshold, resolution=None,
                                   prtf_plot=False):
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(111)

    ax1.plot(q1d,fsc, '.-')
    ax1.axhline(y=threshold, color='r', linestyle='--')
    ax2 = ax1.twiny()
    ax2.plot(q1d,fsc, '.-')
    ax2.set_xticks(ax1.get_xticks())
    ax2.set_xticklabels(np.round(.1*2.*pi/ax1.get_xticks()))
    ax2.set_xlim(0, ax1.get_xticks()[-2] + (ax1.get_xticks()[-1]-ax1.get_xticks()[-2])/2.)
    ax1.set_xlim(0, ax1.get_xticks()[-2]+ (ax1.get_xticks()[-1]-ax1.get_xticks()[-2])/2.)

    ax2.set_xlabel('resolution (nm)',fontsize=15)
    ax1.set_xlabel(r'q-space ring distance (nm$^{-1}$)',fontsize=15)
    if prtf_plot:
        ax1.set_ylabel('PRTF', fontsize=15)
        ax1.set_title('PRTF', fontsize=20)
    else:
        ax1.set_ylabel('Fourier shell correlation', fontsize=15)
        ax1.set_title('Fourier shell correlation', fontsize=20)
    ax1.text(ax1.get_xticks()[1], threshold+.05, f'threshold : {round(threshold,4)}', fontsize=15, color='r')
    if resolution is not None:
        if prtf_plot:
            ypos = .5
        else:
            ypos = .9
        ax1.text(ax1.get_xticks()[len(ax1.get_xticks())//2], ypos,
                 f'resolution : {round(.1*resolution)} nm', fontsize=15, color='k')
    return

def compute_Fourier_shell_correlation_two_object(obj1,obj2, qx,qy,qz, qcen,
                                                 N =30,
                                                  apodizing = False,
                                                 threshold = 0.143,
                                                  verbose=True, plot=False):

    qx = qx - qcen[0]
    qy = qy - qcen[1]
    qz = qz - qcen[2]
    q = np.sqrt(qx**2. + qy**2. + qz**2.)
    if plot:
        plot_2D_slices_middle_one_array3D(q)
        
    if apodizing:
        from PostProcessing import apodize
        obj1 = apodize(obj1,window_type='blackman')
        obj2 = apodize(obj2,window_type='blackman')
        
    if plot:
        plot_2D_slices_middle(obj1, threshold_module=.3, fig_title=r'1$^{st}$ object', crop=False)
        plot_2D_slices_middle(obj2, threshold_module=.3, fig_title=r'2$^{nd}$ object', crop=False)
        
    fexp1 = create_diffracted_amplitude(obj1)
    fexp2 = create_diffracted_amplitude(obj2)
    
    fsc, q1d = Fourier_shell_correlation(fexp1, fexp2, q, N=N, plot=plot)
    
    if np.all(fsc >threshold):
        resolution = 2.*pi/q1d[-1]
        print(f'\nFSC didn\'t reach the threshold. Not sure you can trust this calculated resolution.')
    else:
        resolution = 2.*pi/q1d[np.where(fsc <threshold)[0][0]]
    print(f'\nreal space resolution : {round(.1*resolution,2)} nm')
    
    if plot:
        Fourier_correlation_shell_plot(fsc, q1d, threshold, resolution=resolution)
    
    return fsc, q1d, resolution

def compute_resolution_Fourier_shell_correlation(obj_list, file_ref,
                                                 index1=0, index2=1,
                                                 N =30,
                                                 apodizing = False,
                                                 threshold = 0.143,
                                                 return_fsc=False,
                                                 verbose=True, plot=False, plot_fsc=True):
    qx = file_ref['qx']
    qy = file_ref['qy']
    qz = file_ref['qz']
    qcen = file_ref['qcen']
    obj1 = obj_list[index1]
    obj2 = obj_list[index2]
    
    fsc, q1d, resolution = compute_Fourier_shell_correlation_two_object(obj1,obj2, qx,qy,qz,qcen,
                                                                    N =N,
                                                  apodizing = apodizing,
                                                 threshold = threshold,
                                                  verbose=verbose, plot=plot)
    if not plot and plot_fsc:
        Fourier_correlation_shell_plot(fsc, q1d, threshold, resolution=resolution)
    if return_fsc:
        return resolution, fsc, q1d
    else:
        return resolution

#########################################################################################################################################
##################                    Using Fourier correaltion shell on orthogonalized objects                  ########################
#########################################################################################################################################


def compute_resolution_Fourier_shell_correlation_orthogonalized_objects(obj_list, file_ref,
                                                 index1=0, index2=1,
                                                 N =30,
                                                 apodizing = False,
                                                 threshold = 0.143,
                                                                        return_fsc=False,
                                                 verbose=True, plot=False, plot_fsc=True):
    
    obj1 = obj_list[index1]
    obj2 = obj_list[index2]
    
    if apodizing:
        from PostProcessing import apodize
        obj1 = apodize(obj1,window_type='blackman')
        obj2 = apodize(obj2,window_type='blackman')
        
    obj_ortho1, voxel_sizes = real_space_orthogonalization(obj1, file_ref, final_roi=False,plot=False, verbose=False)
    obj_ortho2, voxel_sizes = real_space_orthogonalization(obj2, file_ref, final_roi=False,plot=False, verbose=False)
    obj_ortho1 = force_even_dimension_one_array(obj_ortho1, verbose=False)
    obj_ortho2 = force_even_dimension_one_array(obj_ortho2, verbose=False)
    
    qx,qy,qz = np.indices(obj_ortho1.shape)
    qx = qx * 2.*pi/(voxel_sizes[0] * obj_ortho1.shape[0])
    qy = qy * 2.*pi/(voxel_sizes[1] * obj_ortho1.shape[1])
    qz = qz * 2.*pi/(voxel_sizes[2] * obj_ortho1.shape[2])  
    qcen = [np.mean(qx), np.mean(qy), np.mean(qz)]
    
    fsc, q1d, resolution = compute_Fourier_shell_correlation_two_object(obj_ortho1,obj_ortho2, qx,qy,qz, qcen,
                                                     N =N,
                                                      apodizing = False,
                                                     threshold = threshold,
                                                      verbose=verbose, plot=plot)
    if not plot and plot_fsc:
        Fourier_correlation_shell_plot(fsc, q1d, threshold, resolution=resolution)
    if return_fsc:
        return resolution, fsc, q1d
    else:
        return resolution

#########################################################################################################################################
############################################                    PRTF                  ###################################################
#########################################################################################################################################

def Fourier_shell_average(array3D_qspace, q,
                          N = 30,
                          plot=False):
    dq = (np.max(q) - np.min(q)) / N
    q1d = np.linspace(np.min(q), np.max(q)-dq, N) + dq/2.
    dq = q1d[1] - q1d[0]
    
    average = np.zeros(N)

    for n in range(N):
#         print(n,end=' ')
        if plot:
            if n==int(N/4):
                plot_2D_slices_middle_one_array3D(indices)
        indices = np.abs(q-q1d[n]) <= dq
        average[n] += np.nanmean(array3D_qspace[indices])
    return average, q1d

def PTRF_resolution(obj_list, file_ref,
                    threshold=1./np.exp(1),
                    verbose=False, plot=False, plot_prtf=False):
    
    if plot:
        plot_prtf = True
        
    data = np.load(str(file_ref['preprocessed_datapath']))['data']
    
    data_recon = [np.abs(create_diffracted_amplitude(obj_list[n]))**2. for n in range(len(obj_list))]
    data_recon = np.nanmean(data_recon, axis=0)
    data_recon = data_recon*np.max(data)/np.max(data_recon) # Is it a good idea to do this here instead of on prtf?
    
    prtf3d = np.sqrt(data_recon)/np.sqrt(data)
    prtf3d[np.isinf(prtf3d)] = np.nan
#     prtf3d = prtf3d/np.max(prtf3d) # Doesn't work well

    if plot:
        plot_3D_projections(data, fig_title='experimental data', axes_labels=True)
        plot_3D_projections(data_recon, fig_title='reconstructed data')
        plot_3D_projections(prtf3d, log_scale=False, fig_title='PRTF')

    qx = file_ref['qx']
    qy = file_ref['qy']
    qz = file_ref['qz']
    qcen = file_ref['qcen']
    qx = qx - qcen[0]
    qy = qy - qcen[1]
    qz = qz - qcen[2]
    q = np.sqrt(qx**2. + qy**2. + qz**2.)
    
    prtf, q1d = Fourier_shell_average(prtf3d, q,  N = 30, plot=False)
#     prtf[0] = 1
    
    if np.all(prtf >threshold):
        resolution = 2.*pi/q1d[-1]
        print(f'\nFSC didn\'t reach the threshold. Not sure you can trust this calculated resolution.')
    else:
        resolution = 2.*pi/q1d[np.where(prtf <threshold)[0][0]]
    print(f'\nreal space resolution : {round(.1*resolution,2)} nm')
    
    if plot_prtf:
        Fourier_correlation_shell_plot(prtf, q1d, threshold, resolution=resolution, prtf_plot=True)
    
    return resolution

#########################################################################################################################################
###############################                         Cherukara's method                       ########################################
#########################################################################################################################################

def Gaussian3D(N,sig):
    x = np.linspace(-1,1,N)
    y = np.linspace(-1,1,N)
    z = np.linspace(-1,1,N)

    x,y,z = np.meshgrid(x,y,z)
    r = np.sqrt(x**2. + y**2. + z**2.)

    gaussian = np.exp(-.5*r**2./sig**2.)
    return gaussian

def richardson_lucy_mine(image, psf, iterations=50, clip=True, filter_epsilon=None):
    float_type = np.promote_types(image.dtype, np.float32)
    image = image.astype(float_type, copy=False)
    psf = psf.astype(float_type, copy=False)
    im_deconv = np.full(image.shape, 0.5, dtype=float_type)
    psf_mirror = np.flip(psf)

    deconv_error = np.zeros(iterations)
    for n in range(iterations):
        if n%100 ==0 :
            print(iterations-n, end=' ')
            
        im_deconv_start = np.copy(im_deconv)
#         conv = convolve(im_deconv, psf, mode='same')
        conv = fftconvolve(im_deconv, psf, mode='same')
        if filter_epsilon:
            relative_blur = np.where(conv < filter_epsilon, 0, image / conv)
        else:
            relative_blur = image / conv
#         im_deconv *= convolve(relative_blur, psf_mirror, mode='same')
            im_deconv *= fftconvolve(relative_blur, psf_mirror, mode='same')
        
        deconv_error[n] += np.sum(np.abs(im_deconv - im_deconv_start) )/ np.sum(im_deconv_start)

    if clip:
        im_deconv[im_deconv > 1] = 1
        im_deconv[im_deconv < -1] = -1
        

    return im_deconv, deconv_error

def fit_PSF_width(psf_deconv, voxel_sizes):
    resolution = np.zeros(3)

    fig,ax = plt.subplots(1,3,figsize=(16,4))
    for n in range(3):
        axes = np.arange(3)
        axes = np.delete(axes, n)
        proj = np.nanmean(psf_deconv,axis=tuple(axes))
        ax[n].plot(proj,'bo')

        fit, popt, pcov = GaussianAndLinearBackgroundFit(proj, x=None,
                                           sig=None,
                                           return_popt_pcov=True,
                                           background=True,
                                           plot=False)
        ax[n].plot(fit,'r.-')
        resolution[n] += popt[2] * voxel_sizes[n]
        title = f'width (in pixels) : {round(popt[2],2)}'
        title += f'\n resolution (nm) : {round(.1*resolution[n],1)}'
        ax[n].set_title(title, fontsize=15)
        fig.suptitle(r'width is given by the gaussian variance ($\sigma$ parameter)', fontsize=20,y=1.2)
        
    return resolution

def resolution_cherukara(obj, voxel_sizes,
                         threshold=.3,
                         sig_init=.2, size_psf=40,
                         iterations = 600, clip=True, filter_epsilon=None,
                         verbose=False, plot=False):
    module = np.abs(obj)
    support = (module>threshold * np.max(module))
    if plot:
        plot_2D_slices_middle_one_array3D(module, fig_title='object module')
        plot_2D_slices_middle_one_array3D(support, fig_title='support')
        
    psf_init = Gaussian3D(size_psf,sig_init)
    support_conv = fftconvolve(support, psf_init, mode='same')
    if plot:
        plot_2D_slices_middle_one_array3D(psf_init, fig_title='initial start for the psf')
        plot_2D_slices_middle_one_array3D(support_conv, fig_title=r'$\it{Convolution}$[ support, psf initial ]')
    
    psf_deconv, deconv_error = richardson_lucy_mine(psf_init, module, iterations=iterations,
                                                    clip=clip, filter_epsilon=filter_epsilon)
    if plot:
        plt.figure()
        plt.plot(deconv_error,'.-')
        plt.yscale('log')
        plt.xlabel('iteration', fontsize=15)
        plt.ylabel(r'$\sum$|psf$_{n+1}$ - psf$_n$| / $\sum$psf$_n$', fontsize=15)
        plt.title('deconvolution error', fontsize=20)
    
        plot_3D_projections(psf_deconv, log_scale=False, cmap='gray_r', fw=3, fig_title='PSF result')

        support_conv = fftconvolve(support, psf_deconv, mode='same')
        plot_2D_slices_middle_one_array3D(support_conv, fig_title=r'$\it{Convolution}$[ support, psf result ]')
        plot_2D_slices_middle_one_array3D(module, fig_title='object module')
        
    resolution = fit_PSF_width(psf_deconv, voxel_sizes)
    return resolution


