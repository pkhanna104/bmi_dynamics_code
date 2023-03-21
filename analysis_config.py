#########################################
############ Repo CONFIG ################
#########################################
import numpy as np 
from matplotlib import colors

###### Has configuration of codebase, pointers to datafiles and temporary datafiles #####
config = dict(); 
config['sub_df_path'] = '/Users/preeyakhanna/fa_analysis/grom_data/'
config['grom_pref'] = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/'
config['jeev_pref'] = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/'
config['home_pref'] = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/home2020/'

config['BMI_DYN'] = '/Users/preeyakhanna/bmi_dynamics_code/'
config['fig_dir'] = '/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/BMI_co_obs_paper/figures/data_figs/'
config['fig_dir2'] = '/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/BMI_co_obs_paper/data/'
config['fig_dir3'] = '/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/BMI_co_obs_paper/presentations/6_2020_presentation_figs/'
config['shuff_fig_dir'] = '/Volumes/Elements/shuffle_data/w_intc/'
config['shuff_fig_dir_nointc'] = '/Volumes/Elements/shuffle_data/wo_intc/'
config['shuff_fig_dir_latentLDS'] = '/Volumes/Elements/shuffle_data/latent/'
config['lqr_sim_saved_dyn'] = '/Users/preeyakhanna/bmi_dynamics_code/dynamics_sims/exp_dynamics_models/'

#########################################
############ Data paths  ################
#########################################
#### Monkey G, datafiles (task entries from BMI 3D) #####
#### Ordered by [CO, OBS] for each day
data_params = dict();
data_params['grom_input_type'] = [[[4377], [4378, 4382]], # co -- 4375 used to train the decoder (3/2/2016)
								 [[4395], [4394]], # co -- 4387 (3/4/2016)
								 [[4411], [4412, 4415]], # co -- (3/7/2016)
								 [[4499], [4497, 4498, 4504]], # obs -- (3/15/2016)
								 [[4510], [4509, 4514]],  # obs -- (3/16/2016)
								 [[4523, 4525], [4520, 4522]], # obs -- (3/17/2016)
								 [[4536], [4532, 4533]], # obs -- (3/18/2016)
								 [[4553], [4549]], # obs -- (3/19/2016)
								 [[4558], [4560]]] # co -- (3/19/2016)

data_params['grom_ordered_input_type'] = [[[0], [1, 2]], [[1], [0]], [[0], [1, 2]], [[2], [0, 1, 3]], [[1],
    [0, 2]], [[2, 3], [0, 1]], [[2], [0, 1]], [[1], [0]], [[0], [1]]]

#### Dates used: ###
data_params['grom_names'] = ['3-2', '3-4', '3-7', '3-15', '3-16', '3-17', '3-18', '3-19', '3-19_2']
data_params['grom_ndays'] = 9
data_params['gbins'] = np.linspace(-3., 3., 20)

#### Monkey J, datafiles (task entries from mat files from Amy/Maryam) #####
data_params['jeev_input_type'] = [
				 [['jeev082413_VFB_PPF_B100_NS5_NU20_Z1_assist_ofc_cont_fixData.mat'], # 24d
				 ['jeev082413_VFB_PPF_B100_NS5_NU20_Z1_assist_ofc_cont_Barrier1fixData']], #24e

				 [['jeev082413_VFB_PPF_B100_NS5_NU20_Z1_assist_ofc_cont_cont_assist_ofc_fixData'], #24h, #assist is on until index 55003
				 ['jeev082413_VFB_PPF_B100_NS5_NU20_Z1_assist_ofc_cont_cont_Barrier2fixData']], #24g
				 
				 [['jeev082813_VFB_PPF_B100_NS5_NU20_Z1_swap49a50a_97a58a_assist_ofc_cont_fixData'], # 29c
				 ['jeev082813_VFB_PPF_B100_NS5_NU20_Z1_swap49a50a_97a58a_assist_ofc_cont_Barrier1fixData']], # 29d
				 
				 [['jeev082813_VFB_PPF_B100_NS5_NU20_Z1_swap49a50a_97a58a_assist_ofc_cont_swap121a97a_cont_cont_add121_cont_cont_fixData'], # 31e
				 ['jeev082813_VFB_PPF_B100_NS5_NU20_Z1_swap49a50a_97a58a_assist_ofc_cont_swap121a97a_cont_cont_add121_cont_cont_Barrier1fixData']]] # 31f

data_params['jeev_ordered_input_type'] = [[[0], [1]], 
						 [[1], [0]], 
						 [[0], [1]], 
						 [[0], [1]],
						 ]
data_params['jeev_ndays'] = 4
data_params['jeev_names'] = ['8-24a', '8-24b', '8-29', '8-31']
data_params['jbins'] = np.linspace(-9., 9., 40)

##### Monkey H 
data_params['home_input_type'] = [[3471, 3472, 3473, 3474], 
								  [3481, 3482, 3483],
								  [3516, 3517, 3519, 3521, 3522], ### No kinarm, no video 
								  [3562, 3563, 3564, 3565, 3566], 
								  [3590, 3591, 3592, 3593, 3594]] ### No kinarm, no video
### could also use data from 3/10 --> (3455, 3456, 3457), no to video (162 BMI rew)
### could also use data from 3/17 --> (3545, 3546, 3547), yes to video (218 BMI rew)



data_params['home_ordered_input_type'] = [[0, 1,2, 3], [0, 1, 2], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]

data_params['home_names'] = ['3-11', '3-12', '3-16', '3-18', '3-19']
data_params['home_ndays'] = 5


# Whether or not to use the 95th percentile? 
ninty_fifth_perc = True
if ninty_fifth_perc:
	data_params['mag_bound_file'] = config['grom_pref'] + 'radial_boundaries_fit_based_on_perc_oct_2020_95th_max.pkl'
else:
	data_params['mag_bound_file'] = config['grom_pref'] + 'radial_boundaries_fit_based_on_perc_feb_2019.pkl'

############# Which targets to analyze ####
min_targix = dict()

### For CO / OBS --> Jeev only analyze targs >= 2
min_targix['jeev'] = [0, 2]
min_targix['grom'] = [0, 0]
min_targix['home'] = [0]

############ Jeev Cw/CCW ########
jeev_cw_ccw_dict = {}
jeev_cw_ccw_dict[1] = 1; 
jeev_cw_ccw_dict[2] = 1;  
jeev_cw_ccw_dict[3] = 0; 
jeev_cw_ccw_dict[4] = 0; 
jeev_cw_ccw_dict[5] = 1; 
jeev_cw_ccw_dict[8] = 0; 
jeev_cw_ccw_dict[9] = 1; 

##################################
####### Useful formating #########
##################################
pref_colors = ['maroon', 'orangered', 'darkgoldenrod', 'olivedrab', 
		'teal', 'steelblue', 'midnightblue', 'darkmagenta', 'black', 'gray']

pref_colors_rgb = [colors.to_rgba(col) for col in pref_colors]

blue_rgb = np.array([39, 169, 225])/255.

def print_G_params(te=data_params['grom_input_type']):
	for i_d, day in enumerate(te):
		for i_t, tsk in enumerate(day):
			for i_te, tei in enumerate(tsk):

				tedb = dbfn.TaskEntry(tei)

				try:
					print('TE %d: Center Target = %.2f, Cursor Rad = %.2f, \
					Target Rad = %.2f, Target Dist = %.2f, Tiemout Time = %.2f, \
					Hold Time = %.2f' %(tei, tedb.target_radius, tedb.cursor_radius,
						tedb.target_radius, 10, tedb.timeout_time, tedb.hold_time))
				except:
					print('pass te %d'%(tei))

########### number of eigenvalues w/ TD > 0.1 sec in each day ####
########## calculated in notebookes/sup-eigenvalues.ipynb: plot_pred_fr_diffs.get_data_EVs()
num_dims = dict(grom=dict(), jeev=dict())
num_dims['grom'][0] = 4
num_dims['grom'][1] = 3
num_dims['grom'][2] = 3
num_dims['grom'][3] = 3
num_dims['grom'][4] = 4
num_dims['grom'][5] = 3
num_dims['grom'][6] = 3
num_dims['grom'][7] = 2
num_dims['grom'][8] = 2

num_dims['jeev'][0] = 3
num_dims['jeev'][1] = 3
num_dims['jeev'][2] = 3
num_dims['jeev'][3] = 2
