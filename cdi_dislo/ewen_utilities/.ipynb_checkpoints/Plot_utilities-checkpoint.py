import matplotlib
import pylab as plt
# import xrayutilities as xu
import os
import numpy as np

from numpy.fft import ifftn, fftn, fftshift, ifftshift

from mpl_toolkits.axes_grid1 import make_axes_locatable

import warnings

from matplotlib.colors import LogNorm

from cdi_dislo.common_imports import *
from cdi_dislo.ewen_utilities.Object_utilities                    import *
from ipywidgets import interact





######################################################################################################################################
##############################                 Surfaces projections                    ###############################################
######################################################################################################################################

def one_surface_projection(strain, axis, inverse):
    strain_used = np.copy(strain)
    if inverse:
        strain_used = np.flip(strain_used, axis=axis)
        
    # I have strain=nan outside the support so I can define the support here
    # If not, use your own support as input
    support = 1-np.isnan(strain_used) 
    support_surface = np.cumsum(support,axis=axis)
    support_surface[support_surface>1] = 0

    surface_strain = np.copy(strain_used)
    surface_strain[support_surface==0] = np.nan
    surface_strain = np.nanmean(surface_strain, axis=axis)
    return surface_strain

def plot_surface_projections(strain, 
                             voxel_sizes=None,
                             fw=3, fig_title=None,
                             vmin=None, vmax=None):
    
    if vmin is None:
        vmin=np.nanmin(1e2*strain)
    if vmax is None:
        vmax=np.nanmax(1e2*strain)
        
    fig,ax = plt.subplots(2,3, figsize=(fw*3.3,fw*2.2))
    
    if voxel_sizes is not None:
        extent = [[0, strain.shape[2]*voxel_sizes[2]*.1, 0, strain.shape[1]*voxel_sizes[1]*.1],
                  [0, strain.shape[2]*voxel_sizes[2]*.1, 0, strain.shape[0]*voxel_sizes[0]*.1],
                  [0, strain.shape[1]*voxel_sizes[1]*.1, 0, strain.shape[0]*voxel_sizes[0]*.1]]
    else:
        extent= [None, None, None]
    imgs = []
    for n1, axis in enumerate(range(3)):
        for n0,inverse in enumerate([False, True]):
            surface_strain = one_surface_projection(strain, axis, inverse)
            imgs.append(ax[n0,n1].matshow(1e2*surface_strain, cmap='coolwarm', extent=extent[n1],
                                          vmin=vmin, vmax=vmax))
            
    cax = fig.add_axes([1, .1, .02, .8])
    cbar = fig.colorbar(imgs[-1], cax=cax, orientation='vertical')
    cbar.ax.tick_params(labelsize=20*fw/4.)
    cbar.ax.locator_params(nbins=5)
    cbar.set_label('strain (%)', rotation=270, fontsize=30*fw/4., labelpad=15)
    
    for axe in ax.flatten():
        axe.locator_params(nbins=4)
        
    xlabel = ['Z (nm)', 'Z (nm)', 'Y (nm)']
    ylabel = ['Y (nm)', 'X (nm)', 'X (nm)']
    for n in range(3):
        for ii in range(2):
            ax[ii,n].set_xlabel(xlabel[n],fontsize=15*fw/4.)
            ax[ii,n].xaxis.set_ticks_position('bottom')
            ax[ii,n].set_ylabel(ylabel[n],fontsize=15*fw/4.)
    
    title_list = [['along +X', 'along +Y', 'along +Z'], ['along -X', 'along -Y', 'along -Z']]
    for n in range(3):
        for ii in range(2):
            ax[ii,n].set_title(title_list[ii][n],fontsize=17*fw/4.)
        
    if fig_title is not None:
        fig_title += '   surface projection'
        fig.suptitle(fig_title, fontsize=fw*22/4.)
            
    fig.tight_layout()
    return



######################################################################################################################################
##############################                Interactive slice plot                   ###############################################
######################################################################################################################################


def interactive_3d_object(obj, 
                          threshold_module=None,
                          axis=0):
    
    ipython = get_ipython()
    if ipython is not None:
        ipython.magic("matplotlib widget")
    #%matplotlib widget
    
    module, phase = get_cropped_module_phase(obj, unwrap=True, threshold_module=threshold_module)
    shape = module.shape

    fig, ax = plt.subplots(1,2, figsize=(8,4))
    im0 = ax[0].matshow(module.take(indices=shape[axis]//2, axis=axis), cmap='gray_r')
    im1 = ax[1].matshow(phase.take(indices=shape[axis]//2, axis=axis), cmap='hsv')

    @interact(w=(0,shape[axis]-1))
    def update(w = 0):
        
        im0.set_data(module.take(indices=w, axis=axis))
        im0.set_clim(np.nanmin(module), np.nanmax(module))
        
        im1.set_data(phase.take(indices=w, axis=axis))
        im1.set_clim(np.nanmin(phase), np.nanmax(phase))
        
        fig.canvas.draw_idle() 

        return   
    
    
def interactive_3d_array(array, 
                          axis=0,
                         voxel_sizes=None,
                         cmap='coolwarm',
                         vmin=None, vmax=None,
                         symmetric_colorscale=False):
    
    if symmetric_colorscale:
        cmap='bwr'
        vmax = np.nanmax(np.abs(array))
        vmin = -vmax
        
    if voxel_sizes is not None:
        voxel_sizes = .1*np.array(voxel_sizes) # Put the voxel_sizes in nanometers
        extent = [[0, array.shape[2]*voxel_sizes[2], 0, array.shape[1]*voxel_sizes[1]], 
                   [0, array.shape[2]*voxel_sizes[2], 0, array.shape[0]*voxel_sizes[0]],
                   [0, array.shape[1]*voxel_sizes[1], 0, array.shape[0]*voxel_sizes[0]]]
    else:
        extent= [None, None, None]
        
    
    ipython = get_ipython()
    if ipython is not None:
        ipython.magic("matplotlib widget")
    #%matplotlib widget
    
    shape = array.shape
    
    fig, ax = plt.subplots(1,1, figsize=(4,4))
    im0 = ax.matshow(array.take(indices=shape[axis]//2, axis=axis), cmap=cmap, vmin=vmin, vmax=vmax, 
                    extent=extent[axis])
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')


    @interact(w=(0,shape[axis]-1))
    def update(w = 0):
        
        im0.set_data(array.take(indices=w, axis=axis))
#         im0.set_clim(vmin, vmax)
        
        fig.canvas.draw_idle() 

        return
    
    
