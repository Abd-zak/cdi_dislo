import numpy as np
import pylab as plt
import xrayutilities as xu


from cdi_dislo.common_imports import *
from cdi_dislo.ewen_utilities.plot_utilities                      import *
from cdi_dislo.ewen_utilities.Object_utilities                    import *

import ipywidgets as widgets

from numpy.fft import fftn, ifftn, ifftshift, fftshift
def QSpaceTransformation_orthogonalization(data,
                         qx,qy,qz,
                         return_qs=False):
    maxbins = []
    for dim in (qx, qy, qz):
        maxstep = max((abs(np.diff(dim, axis=j)).max() for j in range(3)))
        maxbins.append(int(abs(dim.max() - dim.min()) / maxstep))
        
    gridder = xu.Gridder3D(*maxbins) # Do not use FuzzyGridder here !!!!!

    gridder(qx,qy,qz,data)
    qx, qy, qz = [gridder.xaxis, gridder.yaxis, gridder.zaxis]
    data_ortho = gridder.data
    
    if return_qs:
        return data_ortho, qx, qy, qz
    else:
        return data_ortho
    
def compute_automatic_voxel_sizes(qx, qy, qz):
    voxel_sizes = np.zeros(3)
    for n,q in enumerate([qx,qy,qz]):
        range_q = np.nanmax(q) - np.nanmin(q)
        voxel_sizes[n] += 2.*np.pi/range_q
    return voxel_sizes

def OrthogonalizeObject(obj,
                        path_reconstruction,
                        plot=False):
    
    filepath = get_npz_files(path_reconstruction)[0]
    print('q\'s taken from : ',filepath)
    file = np.load(filepath)
    
    # q vector positions, calculated by xrayutilities
    qx = file['qx']
    qy = file['qy']
    qz = file['qz']
    
    # diffracted intensity taken from recon
    I_exp = np.load(str(file['preprocessed_datapath']))['data'] # experimental intensity
    data_recon = fftshift(fftn(ifftshift(obj))) # reconstructed amplitude
    
    data_module_ortho, qx1d_ortho, qy1d_ortho, qz1d_ortho = QSpaceTransformation_orthogonalization(np.abs(data_recon),
                                                                                                   qx,qy,qz, return_qs=True)
    data_phase_ortho = QSpaceTransformation_orthogonalization(np.angle(data_recon), qx,qy,qz)
    data_ortho = data_module_ortho*np.exp(1.0j*data_phase_ortho)
    
    q = [qx1d_ortho, qy1d_ortho, qz1d_ortho]
#     voxel_sizes = [np.fft.fftfreq(len(q[n]), d=q[n][1]-q[n][0]) for n in range(obj.ndim)] 
#     voxel_sizes = [2.*np.pi*np.fft.fftfreq(len(q[n]), d=q[n][1]-q[n][0]) for n in range(obj.ndim)] 
#     voxel_sizes = [voxel[1] - voxel[0] for voxel in voxel_sizes] # voxel size in real space
    voxel_sizes = compute_automatic_voxel_sizes(qx, qy, qz)
    
    if plot:
        fig,ax = plt.subplots(3,3, figsize=(12,12))
        plot_3D_projections(I_exp, fig=fig, ax=ax[0])
        plot_3D_projections(np.abs(data_recon)**2., fig=fig, ax=ax[1])
        plot_3D_projections(np.abs(data_ortho)**2., fig=fig, ax=ax[2])
        ax[0,0].set_ylabel('experimental diffraction', fontsize=20)
        ax[1,0].set_ylabel('reconstructed diffraction', fontsize=20)
        ax[2,0].set_ylabel('orthogonalized diffraction', fontsize=20)
        
    obj_ortho = ifftshift(ifftn(fftshift(data_ortho)))
    obj_ortho = center_object(obj_ortho)
    if plot:
        fig,ax = plot_2D_slices_middle(obj, return_fig_ax=True)
        fig.suptitle('object', fontsize=20)
        fig.tight_layout()
        fig,ax = plot_2D_slices_middle(obj_ortho, return_fig_ax=True, threshold_module=.3)
        fig.suptitle('object orthogonalized', fontsize=20)
        fig.tight_layout()

    return obj_ortho, qx1d_ortho, qy1d_ortho, qz1d_ortho, voxel_sizes

def interactive_threshold_selection(obj,
                                    axis=0):
    
    module, phase = get_cropped_module_phase(obj)
    
    ipython = get_ipython()
    if ipython is not None:
        ipython.magic("matplotlib widget")
    #%matplotlib widget
    
    shape = module.shape
    
    fig, ax = plt.subplots(2,3, figsize=(6,4))
    im_module = []
    im_phase = []
    for axis in range(3):
        im_module.append(ax[0,axis].matshow(module.take(indices=shape[axis]//2, axis=axis), cmap='gray_r'))
        im_phase.append(ax[1,axis].matshow(phase.take(indices=shape[axis]//2, axis=axis), cmap='hsv'))

    @interact(threshold=widgets.FloatSlider(min=0, max=.5, step=.005))
    def update(threshold = 0):
        
        module_clean = np.copy(module)
        phase_clean = np.copy(phase)
        
        module_clean[module<threshold*np.max(module)] = 0.
        phase_clean[module<threshold*np.max(module)] = np.nan
        
        fig.suptitle('threshold : {}'.format(threshold))
        for axis in range(3):
            im_module[axis].set_data(module_clean.take(indices=shape[axis]//2, axis=axis))
            im_phase[axis].set_data(phase_clean.take(indices=shape[axis]//2, axis=axis))
            
        fig.canvas.draw_idle() 
