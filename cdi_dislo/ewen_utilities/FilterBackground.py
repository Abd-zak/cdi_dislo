import numpy as np
import pylab as plt

def filter_background_zlotnikov(data, 
                                nb_points_rocking=10,
                                nb_points_start_end=13,
                                const_relative_shift=1,
                                plot=False):
    profile = (np.nanmean(data[:nb_points_rocking], axis=(0,1))+np.nanmean(data[-nb_points_rocking:], axis=(0,1)))/2.
    
    x = np.arange(len(profile))
    indices = np.concatenate((np.arange(nb_points_start_end), -np.arange(nb_points_start_end)))
    profile_to_fit = profile[indices]
    x_to_fit = x[indices]
    p = np.polyfit(x_to_fit, profile_to_fit, deg=1)
    p[1] = p[1] * const_relative_shift #  not great !
    fit = x*p[0] + p[1]
    
    y2d,x2d = np.indices(data.shape[1:])
    background = x2d*p[0] + p[1]
    
    if plot:
        fig,ax = plt.subplots(2,2, figsize=(12,6))
        
        x_rock = np.arange(data.shape[0])
        rocking = np.nanmean(data, axis=(1,2))
        ax[0,0].plot(np.log(rocking), '.-')
        ax[0,0].plot(x_rock[:nb_points_rocking], np.log(rocking[:nb_points_rocking]), 'r.-')
        ax[0,0].plot(x_rock[-nb_points_rocking:], np.log(rocking[-nb_points_rocking:]), 'r.-')
        ax[0,0].set_ylabel('rocking curve (log scale)', fontsize=15)
        
        ax[0,1].plot(x,profile, '.-')
        ax[0,1].plot(x,fit, 'r')
        
        ax[1,0].matshow(background)
        
        ax[1,1].matshow(np.nanmean(data, axis=(0)), vmin=background.min(), vmax=background.max())
        
        fig.tight_layout()
        
    return background
