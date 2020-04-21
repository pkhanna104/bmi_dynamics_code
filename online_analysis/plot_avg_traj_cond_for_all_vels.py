import pickle
import numpy as np; 
import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap

cmap_list = ['maroon', 'orangered', 'darkgoldenrod', 'olivedrab', 'teal', 'steelblue', 'midnightblue', 'darkmagenta', 'black']
cmap_list = [['darkgreen', '', '', '', 'seagreen', 'seagreen', 'darkgreen', 'darkgreen'],
             ['slateblue', '', '', '', 'slateblue', 'darkblue', 'blue', 'darkslateblue', 'darkblue']]

fig_dir = '/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/BMI_co_obs_paper/figures/data_figs/'


co_obs_cmap = [np.array([0, 103, 56])/255., np.array([46, 48, 146])/255., ]
co_obs_cmap_cm = []; 

for _, (c, cnm) in enumerate(zip(co_obs_cmap, ['co', 'obs'])):
    colors = [[1, 1, 1], c]  # white --> color
    cmap_name = 'my_list'
    cm = LinearSegmentedColormap.from_list(
        cnm, colors, N=1000)
    co_obs_cmap_cm.append(cm)

import subspace_overlap
### #Make a plot for all velocity commands showing the avg. place its used in the workspace

def make_plots(animal = 'grom', i_d = 0, min_obs = [30, 65]):
    mag_boundaries = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/radial_boundaries_fit_based_on_perc_feb_2019.pkl'))
    model_dict = pickle.load(open('/Users/preeyakhanna/fa_analysis/tuning_models_'+animal+'_model_set%d.pkl' %3, 'rb'))

    tsk  = model_dict[i_d, 'task']
    targ = model_dict[i_d, 'trg']
    push = model_dict[i_d, 'np']
    pos = model_dict[i_d, 'pos']
    bin_num = model_dict[i_d, 'bin_num']

    commands_disc = subspace_overlap.commands2bins([push], mag_boundaries, animal, i_d, vel_ix = [0, 1])[0]
    col_tsk = ['g', 'b']

    n = 5

    ### For each command, grab 
    f, ax = plt.subplots(ncols = 3, nrows = 1, figsize=(6, 2))

    for i_mag in range(3, 4):
        for i_ang in range(2, 5): 
            axi = ax[i_ang - 2]
            
            for i_tsk in range(2):
                cmap = co_obs_cmap_cm[i_tsk]
                for i_trg in range(8):
                    col = cmap(int((i_trg+9)/16.*1000.))

                    ix = (tsk == i_tsk) & (targ == i_trg) & (commands_disc[:, 0] == i_mag) & (commands_disc[:, 1] == i_ang)
                    ix = np.nonzero(ix == True)[0]

                    if len(ix) > min_obs[i_tsk]:

                        z = np.zeros((len(ix), 2*n + 1, 2)) + np.nan

                        for ii, i in enumerate(ix): 
                            if i + 7 < len(bin_num):
                                if np.logical_and(bin_num[i] >= n, bin_num[i+n] >=(2*n + 1)):
                                    z[ii, :, :] = pos[i-n:i+n+1, :]

                        ## Now plot: 
                        z_mn = np.nanmean(z, axis=0)
                        z_mean_pos = z_mn[n, :]
                        z_mn_repos = z_mn - z_mean_pos[np.newaxis, :]

                        axi.plot(z_mn_repos[:, 0], z_mn_repos[:, 1], '.-', color = cmap_list[i_tsk][i_trg], linewidth = 3., alpha=.8)
            axi.set_xlim([-2.5, 2.5])
            axi.set_ylim([-2.5, 2.5])
            axi.set_xticks([])
            axi.set_yticks([])
            
    f.tight_layout()
    f.savefig(fig_dir+'grom_day0_eg_traj.svg')

def quantify_neural_var(): 

    ### variance of neural activity  
    ### 