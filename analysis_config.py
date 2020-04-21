#########################################
############ Repo CONFIG ################
#########################################

###### Has configuration of codebase, pointers to datafiles and temporary datafiles #####
config = dict(); 
config['sub_df_path'] = '/Users/preeyakhanna/fa_analysis/grom_data/'
config['grom_pref'] = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/'
config['jeev_pref'] = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/'

#########################################
############ Data paths  ################
#########################################
#### Monkey G, datafiles (task entries from BMI 3D) #####
#### Ordered by [CO, OBS] for each day
data_params = dict();
data_params['grom_input_type'] = [[[4377], [4378, 4382]], [[4395], [4394]], [[4411], [4412, 4415]], [[4499], 
[4497, 4498, 4504]], [[4510], [4509, 4514]], [[4523, 4525], [4520, 4522]], [[4536], 
[4532, 4533]], [[4553], [4549]], [[4558], [4560]]]

#### Dates used: ###
data_params['grom_names'] = ['3-2', '3-4', '3-7', '3-15', '3-16', '3-17', '3-18', '3-19', '3-19_2']

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

##################################
####### Useful formating #########
##################################
pref_colors = ['maroon', 'orangered', 'darkgoldenrod', 'olivedrab', 
		'teal', 'steelblue', 'midnightblue', 'darkmagenta']
