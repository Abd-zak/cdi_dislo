import numpy as np
import pylab as plt

import gif

import sys


from PIL import Image
from matplotlib.colors import LogNorm

from cdi_dislo.common_imports import *
from cdi_dislo.ewen_utilities.plot_utilities                      import *
from cdi_dislo.ewen_utilities.Global_utilities                    import *
######################################################################################################################################
##################################             module phase strain gif                ################################################
###################################################################################################################################### 


# @gif.frame
# def Gif_one_slice_plot(module,phase,strain, axis, indice,
#                        voxel_sizes=None,
#                        vmin_phase=None, vmax_phase=None,
#                        vmax_strain=None,
#                        fw=4):
#     fig,ax = plt.subplots(1,3, figsize=(fw*3,fw))
#     imgs = []

#     shape = module.shape

#     vmin_mod = np.nanmin(module)
#     vmax_mod = np.nanmax(module)

#     if vmin_phase is None:
#         vmin_phase = np.nanmin(phase)
#     if vmax_phase is None:
#         vmax_phase = np.nanmax(phase)
        
#     if vmax_strain is None:
#         vmax_strain = np.nanmax(np.abs(strain))
#     vmin_strain = -vmax_strain
    
#     if voxel_sizes is not None:
#         voxel_sizes = .1*np.array(voxel_sizes) # Put the voxel_sizes in nanometers
#         extent = [[0, module.shape[2]*voxel_sizes[2], 0, module.shape[1]*voxel_sizes[1]], 
#                    [0, module.shape[2]*voxel_sizes[2], 0, module.shape[0]*voxel_sizes[0]],
#                    [0, module.shape[1]*voxel_sizes[1], 0, module.shape[0]*voxel_sizes[0]]]
#     else:
#         extent=[None,None,None]


#     imgs.append(ax[0].matshow(module.take(indices=indice, axis=axis), cmap='gray_r', vmin=vmin_mod, vmax=vmax_mod,
#                               extent=extent[axis]))
#     imgs.append(ax[1].matshow(phase.take(indices=indice, axis=axis), cmap='hsv', vmin=vmin_phase, vmax=vmax_phase,
#                               extent=extent[axis]))
#     imgs.append(ax[2].matshow(strain.take(indices=indice, axis=axis), cmap='coolwarm', vmin=vmin_strain, vmax=vmax_strain,
#                               extent=extent[axis]))

#     add_colorbar_subplot(fig,ax,imgs)
    
#     if voxel_sizes is not None:
#         for axe in ax:
#             axe.xaxis.set_ticks_position('bottom')
#             axe.set_xlabel('nm',fontsize=15)

#     fig.tight_layout()
#     return

# def make_gif_module_phase_strain(module,phase,strain, axis,
#             voxel_sizes=None,
#             vmin_phase=None,vmax_phase=None,
#             vmax_strain=None,
#             fw=4, name='gif_default_name', duration=10):
    
#     frames = [Gif_one_slice_plot(module,phase,strain,axis, indice,voxel_sizes=voxel_sizes, fw=fw,
#                                  vmin_phase=vmin_phase, vmax_phase=vmax_phase, vmax_strain=vmax_strain) for indice in range(module.shape[axis])]
#     gif.save(frames, name+'.gif', duration=duration, unit='s', between='startend')
#     print('gif saved in : ',name+'.gif')
#     return


@gif.frame
def Gif_one_slice_plot(module_ortho, phase_ortho, strain,#d_spacing,
                          tilt_comp1, tilt_comp2, tilt_magn,
                       index,
                 voxel_sizes=None,
                 threshold=.1, factor=.2,
                       vmin_list=None,vmax_list=None,
                 fw=2.7,
                 return_figure=False,
                       verbose=True):
    
    if verbose:
        print(max(module_ortho.shape)-index, end=' ')
    
    fig,ax = plt.subplots(3,6,figsize=(fw*6, fw*3))

    xlabel = ['Z (nm)', 'Z (nm)', 'Y (nm)']
    ylabel = ['Y', 'X', 'X']
    
    if vmin_list is None:
        vmin_list = [None for n in range(6)]
    if vmax_list is None:
        vmax_list = [None for n in range(6)]

    plot_2D_slices_middle_one_array3D(module_ortho, index=index,
                                      cmap='gray_r', voxel_sizes=voxel_sizes, add_colorbar=True,
                                      vmin=vmin_list[0], vmax=vmax_list[0],
                                      xlabel=xlabel, ylabel=ylabel,
                                      fig=fig,ax=ax[:,0])

    plot_2D_slices_middle_one_array3D(phase_ortho, index=index,
                                      cmap='hsv', voxel_sizes=voxel_sizes, add_colorbar=True,
                                      vmin=vmin_list[1], vmax=vmax_list[1],
                                      fig=fig,ax=ax[:,1])

    plot_2D_slices_middle_one_array3D(1e2*strain, index=index,
                                      cmap='coolwarm', voxel_sizes=voxel_sizes, add_colorbar=True,
                                      vmin=vmin_list[2], vmax=vmax_list[2],
                                      fig=fig,ax=ax[:,2])
    
    plot_2D_slices_middle_one_array3D(tilt_comp1, index=index,
                                      cmap='coolwarm', voxel_sizes=voxel_sizes, add_colorbar=True,
                                      vmin=vmin_list[3], vmax=vmax_list[3],
                                      fig=fig,ax=ax[:,3])
    
    plot_2D_slices_middle_one_array3D(tilt_comp2, index=index,
                                      cmap='coolwarm', voxel_sizes=voxel_sizes, add_colorbar=True,
                                      vmin=vmin_list[4], vmax=vmax_list[4],
                                      fig=fig,ax=ax[:,4])
    
    plot_2D_slices_middle_one_array3D(1e2*tilt_magn, index=index,
                                      cmap='gray_r', voxel_sizes=voxel_sizes, add_colorbar=True,
                                      vmin=vmin_list[5], vmax=vmax_list[5],
                                      fig=fig,ax=ax[:,5])
    

    ax[0,0].set_title('module', fontsize=20*fw/3)
    ax[0,1].set_title('phase', fontsize=20*fw/3)
    ax[0,2].set_title('strain (%)', fontsize=20*fw/3)
    ax[0,3].set_title('tilt along X', fontsize=20*fw/3)
    ax[0,4].set_title('tilt along Y', fontsize=20*fw/3)
    ax[0,5].set_title('tilt magnitude (%)', fontsize=20*fw/3)    
    
    for axe in ax.flatten():
        axe.tick_params(axis='both', which='major', labelsize=15*fw/4)
        axe.xaxis.get_label().set_fontsize(20*fw/3)
        axe.yaxis.get_label().set_fontsize(20*fw/3)
        
    fig.suptitle('Bragg wavevector along Z', fontsize=15)

    fig.tight_layout()
    
    if return_figure:
        return fig,ax
    else:
        return
    
    
def make_gif_module_phase_strain_tilt(module_ortho, phase_ortho, strain,
                          tilt_comp1, tilt_comp2,
            voxel_sizes=None,
            vmin_phase=None,vmax_phase=None,
            vmax_strain=None,
            fw=4, name='gif_default_name', duration=10):
    
    vmin_list = [np.nanmin(module_ortho),
                 np.nanmin(phase_ortho),
                 1e2*np.nanmin(strain),
                 np.nanmin(tilt_comp1),
                 np.nanmin(tilt_comp2)]
    vmax_list = [np.nanmax(module_ortho),
                 np.nanmax(phase_ortho),
                 1e2*np.nanmax(strain),
                 np.nanmax(tilt_comp1),
                 np.nanmax(tilt_comp2)]
    
    frames = [Gif_one_slice_plot(module_ortho, phase_ortho, strain,
                          tilt_comp1, tilt_comp2,
                       index,
                 voxel_sizes=voxel_sizes,
                 vmin_list=vmin_list,vmax_list=vmax_list)
              for index in range(max(module_ortho.shape))]
    gif.save(frames, name+'.gif', duration=duration, unit='s', between='startend')
    print('gif saved in : ',name+'.gif')
    return
    
    




######################################################################################################################################
##################################             Rotating isosurface gif                ################################################
######################################################################################################################################

from skimage import measure

@gif.frame
def gif_one_rotation(verts , faces,  angle):
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2],
                    cmap='gray', lw=1)
    # Hide grid lines
    ax.grid(False)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.view_init(20, angle)
    
    fig.tight_layout()
    return

def make_gif_pixels_3d(module_ortho,
                       iso_val=0.2,
                       name='gif_default_name', duration=10, nb_angles=30):
    
    verts , faces, normals, values = measure.marching_cubes(module_ortho, iso_val, spacing=(0.1, 0.1, 0.1))
    angle_list = np.linspace(0,360,nb_angles) 
    frames = [gif_one_rotation(verts , faces,  angle) for angle in angle_list]
    gif.save(frames, name+'.gif', duration=duration, unit='s', between='startend')
    print('gif saved in : ',name+'.gif')
    
    return

######################################################################################################################################
#################################             Gif schematic Bragg planes                ##############################################
######################################################################################################################################

def schematic_Bragg_planes_figure(displacement, qcen, voxel_sizes,
                                  strain=None,
                                  module_ortho=None,
                                  axis=0, slice_index=None,
                                  visual_factor=50,
                                  dislo_threshold = None,
                                  close_roi=False,
                                  fig=None, ax=None, fw=6, fig_title=None):
    angle_bragg_last_axis = np.rad2deg(np.arccos(np.dot(qcen, [0,0,1])/ np.linalg.norm(qcen)))
    if angle_bragg_last_axis > 1.:
        raise ValueError('qcen is not along the last axis ! My function is not ready for that.')
        return
    
    if axis==2 :
        raise ValueError('axis should be 0 or 1!')
    
    if slice_index is None:
        s = slice_middle_array_along_axis(displacement, axis=axis)
    else:
        s = [slice(None, None, None) for ii in range(displacement.ndim)]
        s[axis] = min(slice_index, displacement.shape[axis]-1)#slice_index
        s = tuple(s)
        
    if fig is None:
        fig,ax = plt.subplots(1,1, figsize=(fw,fw))
        
    if strain is not None:
#         strain_color_slice = np.copy(strain[s])
#         strain_color_slice = strain_color_slice - np.nanmin(strain_color_slice)
#         strain_color_slice = strain_color_slice/ np.nanmax(strain_color_slice)
        
        strain_color = np.copy(strain)
        strain_color -= np.nanmin(strain_color)
        strain_color = strain_color/np.nanmax(strain_color)
        strain_color_slice = np.copy(strain_color[s])
    
    displacement_slice = displacement[s]
    voxel_sizes_slice = np.delete(voxel_sizes,axis)

    # Create a regular grid
    pos = np.indices(displacement_slice.shape)
    pos[0] = pos[0]*voxel_sizes_slice[0]*.1
    pos[1] = pos[1]*voxel_sizes_slice[1]*.1

    pos = pos + .1*displacement_slice[None] * np.array([0,1])[:,None,None] * visual_factor

    for n in range(displacement_slice.shape[1]):
        # Just a quick test
        if dislo_threshold is not None:
            grad_amp = np.abs(np.gradient(pos[1,:,n]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                indices = grad_amp>np.nanmedian(grad_amp)*20
            pos[:,indices,n] = np.nan
        ax.plot(pos[1,:,n], pos[0,:,n], 'k-', linewidth=1, markersize=1, alpha=1)
        
        if strain is not None:
            strain_line = strain_color_slice[:,n]
            color = cm.coolwarm(strain_line)
#             color = cm.jet(strain_line)
            sc = ax.scatter(pos[1,:,n], pos[0,:,n], s=20*fw/10., color=color, alpha=1)        
    
    ax.set_xlabel('nm', fontsize=15*fw/4.)
        
    if strain is not None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        norm = mpl.colors.Normalize(vmin=np.nanmin(1e2*strain), vmax=np.nanmax(1e2*strain))
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='coolwarm'),
                     cax=cax, orientation='vertical')#, label='Strain (%)')
        cbar.set_label('strain (%)', size=20*fw/6)
    
    if not close_roi :
        ax.set_xlim(0, displacement_slice.shape[1]*voxel_sizes_slice[1]*.1)
        ax.set_ylim(0, displacement_slice.shape[0]*voxel_sizes_slice[0]*.1)
    
    ax.invert_yaxis()

    if fig_title is None:
        fig_title = 'Schematic Bragg\nplanes distortion'
    ax.set_title(fig_title, fontsize=12*fw/4.)
    ax.set_aspect('equal', 'box')
    
    return

@gif.frame
def Gif_one_slice_schematic_Bragg_planes(displacement, qcen, voxel_sizes, slice_index,
                                  strain=None,
                                  visual_factor=200, dislo_threshold=None,
                                  close_roi=True, fw=7):
    fig,ax = plt.subplots(1,2, figsize=(2*fw,fw))
    

    for n, axis in enumerate([0,1]):
        schematic_Bragg_planes_figure(displacement, qcen, voxel_sizes,
                                          strain=strain,
                                          axis=axis, slice_index=slice_index,
                                          visual_factor=visual_factor,
                                      dislo_threshold=dislo_threshold,
                                          close_roi=False,
                                          fig=fig, ax=ax[n], fig_title=f'slice along axis {axis}')
        
    fig.suptitle('Schematic Bragg planes', fontsize=20*fw/6.)
    fig.tight_layout()
    
def make_gif_schematic_Bragg_planes(displacement, qcen, voxel_sizes,
                                    strain=None,
                                    visual_factor=200, dislo_threshold=None,
                                    fw=7,
                                    name='gif_default_name', duration=10):
    
    
    frames = [Gif_one_slice_schematic_Bragg_planes(displacement, qcen, voxel_sizes, slice_index,
                                  strain=strain,
                                  visual_factor=visual_factor, dislo_threshold=dislo_threshold,
                                  close_roi=False, fw=fw)
              for slice_index in range(max(displacement.shape[:2]))]
    gif.save(frames, name+'.gif', duration=duration, unit='s', between='startend')
    print('gif saved in : ',name+'.gif')
    return


######################################################################################################################################
##################################                 Bragg peak gif                     ################################################
###################################################################################################################################### 


@gif.frame
def Gif_one_image(array, n,
                  log_scale=False,
                  fw=4):
    if n%10==0:
        print(array.shape[0]-n,end=' ')
    fig,ax = plt.subplots(1,figsize=(fw,fw))
    if log_scale:
#         norm = LogNorm(vmin=np.min(array), vmax=np.max(array))
        norm = LogNorm(vmin=1, vmax=np.max(array))
#         norm = LogNorm()
    else:
        norm=None
    img = ax.matshow(array[n], norm=norm, cmap='gray_r')
    ax.set_title(f'slice {n}', fontsize=15)
    add_colorbar_subplot(fig,ax,img)
    return

def make_gif_array3d(array, name='gif_default_name', duration=10,
                     log_scale=False,
                         fw=6):
    
    frames = [Gif_one_image(array,n, fw=fw,log_scale=log_scale) for n in range(array.shape[0])]
    gif.save(frames, name+'.gif', duration=duration, unit='s', between='startend')
    print('gif saved in : ',name+'.gif')
    return

######################################################################################################################################
#################################             Gif from a list of images                ###############################################
######################################################################################################################################

from PIL import Image

@gif.frame
def Gif_one_image(file,
                  fw=6):
    
    img = Image.open(file)
    ratio = img.size[0]/img.size[1]
    fig,ax = plt.subplots(1,figsize=(fw*ratio,fw))
    ax.imshow(img)
    ax.axis('off')
    return

def make_gif_list_images(files, name='gif_default_name', duration=10,
                         fw=6):
    
    frames = [Gif_one_image(file,fw=fw) for file in files]
    gif.save(frames, name+'.gif', duration=duration, unit='s', between='startend')
    print('gif saved in : ',name+'.gif')
    return
