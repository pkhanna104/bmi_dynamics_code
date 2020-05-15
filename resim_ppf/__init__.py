################################################
###### Methods for processing Jeev's Data ######
################################################

# Facts: EC_trialEndErrs = [4, 8, 12, 9]; %errors for not holding at center/target and reach time-outs, and code for reward that marks successful trial. Trial is over after an error

########################
### USING PLX NAMES ###
########################
### File lists from giant backupdrive 
filelist = ['/media/lab/My Book/pk/jeev_2013/jeev081513c_all.mat',
'/media/lab/My Book/pk/jeev_2013/jeev081513d_all.mat',
'/media/lab/My Book/pk/jeev_2013/jeev081513e_all.mat',
'/media/lab/My Book/pk/jeev_2013/jeev081613c_all.mat',
'/media/lab/My Book/pk/jeev_2013/jeev081613d_all.mat',
'/media/lab/My Book/pk/jeev_2013/jeev081613e_all.mat',
'/media/lab/My Book/pk/jeev_2013/jeev082313d_all.mat',
'/media/lab/My Book/pk/jeev_2013/jeev082313e_all.mat',
'/media/lab/My Book/pk/jeev_2013/jeev082413d_all.mat',
'/media/lab/My Book/pk/jeev_2013/jeev082413e_all.mat',
'/media/lab/My Book/pk/jeev_2013/jeev082413g_all.mat',
'/media/lab/My Book/pk/jeev_2013/jeev082413h_all.mat',
'/media/lab/My Book/pk/jeev_2013/jeev082813d_all.mat',
'/media/lab/My Book/pk/jeev_2013/jeev082813e_all.mat',
'/media/lab/My Book/pk/jeev_2013/jeev082913c_all.mat',
'/media/lab/My Book/pk/jeev_2013/jeev082913d_all.mat',
'/media/lab/My Book/pk/jeev_2013/jeev083113e_all.mat',
'/media/lab/My Book/pk/jeev_2013/jeev083113f_all.mat',
'/media/lab/My Book/pk/jeev_2013/jeev090113f_all.mat',
'/media/lab/My Book/pk/jeev_2013/jeev090113g_all.mat']

decoderlist = ['/media/lab/My Book/pk/jeev_2013/jeev080713_VFB_PPF_B100_NS5_NU18_Z1_assist_ofc_cont_cont_cont_swap50a15a_cont_swap58a50a113ab114ab_cont_swap124a125b_cont_cont_swap125ba_cont_cont.mat', 
'/media/lab/My Book/pk/jeev_2013/jeev080713_VFB_PPF_B100_NS5_NU18_Z1_assist_ofc_cont_cont_cont_swap50a15a_cont_swap58a50a113ab114ab_cont_swap124a125b_cont_cont_swap125ba_cont_cont.mat', 
'/media/lab/My Book/pk/jeev_2013/jeev080713_VFB_PPF_B100_NS5_NU18_Z1_assist_ofc_cont_cont_cont_swap50a15a_cont_swap58a50a113ab114ab_cont_swap124a125b_cont_cont_swap125ba_cont_cont.mat', 
'/media/lab/My Book/pk/jeev_2013/jeev080713_VFB_PPF_B100_NS5_NU18_Z1_assist_ofc_cont_cont_cont_swap50a15a_cont_swap58a50a113ab114ab_cont_swap124a125b_cont_cont_swap125ba_cont_cont_cont_cont_swap6758_12582_cont_cont.mat',
'/media/lab/My Book/pk/jeev_2013/jeev080713_VFB_PPF_B100_NS5_NU18_Z1_assist_ofc_cont_cont_cont_swap50a15a_cont_swap58a50a113ab114ab_cont_swap124a125b_cont_cont_swap125ba_cont_cont_cont_cont_swap6758_12582_cont_cont.mat',
'/media/lab/My Book/pk/jeev_2013/jeev080713_VFB_PPF_B100_NS5_NU18_Z1_assist_ofc_cont_cont_cont_swap50a15a_cont_swap58a50a113ab114ab_cont_swap124a125b_cont_cont_swap125ba_cont_cont_cont_cont_swap6758_12582_cont_cont.mat',
'/media/lab/My Book/pk/jeev_2013/jeev082313_VFB_PPF_B100_NS5_NU19_Z1_assist_ofc_cont.mat',
'/media/lab/My Book/pk/jeev_2013/jeev082313_VFB_PPF_B100_NS5_NU19_Z1_assist_ofc_cont.mat',
'/media/lab/My Book/pk/jeev_2013/jeev082413_VFB_PPF_B100_NS5_NU20_Z1_assist_ofc_cont.mat',
'/media/lab/My Book/pk/jeev_2013/jeev082413_VFB_PPF_B100_NS5_NU20_Z1_assist_ofc_cont.mat',
'/media/lab/My Book/pk/jeev_2013/jeev082413_VFB_PPF_B100_NS5_NU20_Z1_assist_ofc_cont_cont.mat',
'/media/lab/My Book/pk/jeev_2013/jeev082413_VFB_PPF_B100_NS5_NU20_Z1_assist_ofc_cont_cont.mat',
'/media/lab/My Book/pk/jeev_2013/jeev082613_VFB_PPF_B100_NS5_NU20_Z1_assist_ofc_cont.mat',
'/media/lab/My Book/pk/jeev_2013/jeev082613_VFB_PPF_B100_NS5_NU20_Z1_assist_ofc_cont.mat',
'/media/lab/My Book/pk/jeev_2013/jeev082613_VFB_PPF_B100_NS5_NU20_Z1_assist_ofc_cont.mat',
'/media/lab/My Book/pk/jeev_2013/jeev082613_VFB_PPF_B100_NS5_NU20_Z1_assist_ofc_cont.mat',
'/media/lab/My Book/pk/jeev_2013/jeev082813_VFB_PPF_B100_NS5_NU20_Z1_swap49a50a_97a58a_assist_ofc_cont_swap121a97a_cont_cont_add121_cont_cont.mat',
'/media/lab/My Book/pk/jeev_2013/jeev082813_VFB_PPF_B100_NS5_NU20_Z1_swap49a50a_97a58a_assist_ofc_cont_swap121a97a_cont_cont_add121_cont_cont.mat',
'/media/lab/My Book/pk/jeev_2013/jeev082813_VFB_PPF_B100_NS5_NU20_Z1_swap49a50a_97a58a_assist_ofc_cont_swap121a97a_cont_cont_add121_cont_rmv97_105_121_add81_cont_cont.mat',
'/media/lab/My Book/pk/jeev_2013/jeev082813_VFB_PPF_B100_NS5_NU20_Z1_swap49a50a_97a58a_assist_ofc_cont_swap121a97a_cont_cont_add121_cont_rmv97_105_121_add81_cont_cont.mat']

# input_type = [[['jeev081513e'], ['jeev081513c', 'jeev081513d']], [['jeev081613e'], ['jeev081613c', 'jeev081613d']], 
# 	[['jeev082313d'], ['jeev082313e']], [['jeev082413d'], ['jeev082413e']], [['jeev082413h'], ['jeev082413g']], 
# 	#[['jeev082813e'], ['jeev082813d']], 
# 	[['jeev082913c'], ['jeev082913d']], [['jeev083113e'], ['jeev083113f']]]

# input_type = [[['jeev081613e'], ['jeev081613c', 'jeev081613d']], 
# 	[['jeev082313d'], ['jeev082313e']], [['jeev082413d'], ['jeev082413e']], [['jeev082413h'], ['jeev082413g']], 
# 	#[['jeev082813e'], ['jeev082813d']], 
# 	[['jeev082913c'], ['jeev082913d']], [['jeev083113e'], ['jeev083113f']]]


#input_names= ['8-15', '8-16', '8-23', '8-24a', '8-24b', '8-29', '8-31']
#input_names= ['8-16', '8-23', '8-24a', '8-24b', '8-29', '8-31']


########################
### USING TASK NAMES ###
########################
task_directory = '/home/lab/preeya/jeev_data_tmp/'
task_directory_mbp = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/'
#'/Volumes/TimeMachineBackups/jeev2013/'


#### File we actually end up analyzing since have both CO / OBS on the same day w/o CLDA ####
task_filelist = [#[['jeev080713_VFB_PPF_B100_NS5_NU18_Z1_assist_ofc_cont_cont_cont_swap50a15a_cont_swap58a50a113ab114ab_cont_swap124a125b_cont_cont_swap125ba_cont_cont_fixData'], # 15e -- no rewards :(
				 #['jeev080713_VFB_PPF_B100_NS5_NU18_Z1_assist_ofc_cont_cont_cont_swap50a15a_cont_swap58a50a113ab114ab_cont_swap124a125b_cont_cont_swap125ba_cont_cont_Barrier1fixData.mat']], #15c
				 
				 #[['jeev080713_VFB_PPF_B100_NS5_NU18_Z1_assist_ofc_cont_cont_cont_swap50a15a_cont_swap58a50a113ab114ab_cont_swap124a125b_cont_cont_swap125ba_cont_cont_contData.mat'], # 16e, wrong file -- no fixed option
				 #['jeev080713_VFB_PPF_B100_NS5_NU18_Z1_assist_ofc_cont_cont_cont_swap50a15a_cont_swap58a50a113ab114ab_cont_swap124a125b_cont_cont_swap125ba_cont_cont_cont_Barrier1fixData.mat', #16c
				 #'jeev080713_VFB_PPF_B100_NS5_NU18_Z1_assist_ofc_cont_cont_cont_swap50a15a_cont_swap58a50a113ab114ab_cont_swap124a125b_cont_cont_swap125ba_cont_cont_cont_Barrier2fixData.mat']], #16d
				 
				 #[['jeev082313_VFB_PPF_B100_NS5_NU19_Z1_assist_ofc_cont_fixData.mat'], #23d
				 #['jeev082313_VFB_PPF_B100_NS5_NU19_Z1_assist_ofc_cont_Barrier1fixData.mat']], #23e -- only 16 trials
				
				 [['jeev082413_VFB_PPF_B100_NS5_NU20_Z1_assist_ofc_cont_fixData.mat'], # 24d
				 ['jeev082413_VFB_PPF_B100_NS5_NU20_Z1_assist_ofc_cont_Barrier1fixData']], #24e

				 [['jeev082413_VFB_PPF_B100_NS5_NU20_Z1_assist_ofc_cont_cont_assist_ofc_fixData'], #24h, #assist is on until index 55003
				 ['jeev082413_VFB_PPF_B100_NS5_NU20_Z1_assist_ofc_cont_cont_Barrier2fixData']], #24g
				 
				 [['jeev082813_VFB_PPF_B100_NS5_NU20_Z1_swap49a50a_97a58a_assist_ofc_cont_fixData'], # 29c
				 ['jeev082813_VFB_PPF_B100_NS5_NU20_Z1_swap49a50a_97a58a_assist_ofc_cont_Barrier1fixData']], # 29d
				 
				 [['jeev082813_VFB_PPF_B100_NS5_NU20_Z1_swap49a50a_97a58a_assist_ofc_cont_swap121a97a_cont_cont_add121_cont_cont_fixData'], # 31e
				 ['jeev082813_VFB_PPF_B100_NS5_NU20_Z1_swap49a50a_97a58a_assist_ofc_cont_swap121a97a_cont_cont_add121_cont_cont_Barrier1fixData']]] # 31f


# task_input_type = [[['jeev081513e'], ['jeev081513c']], [['jeev081613e'], ['jeev081613c', 'jeev081613d']], 
# 	[['jeev082313d'], ['jeev082313e']], [['jeev082413d'], ['jeev082413e']], [['jeev082413h'], ['jeev082413g']], 
# 	#[['jeev082813e'], ['jeev082813d']], 
# 	[['jeev082913c'], ['jeev082913d']], [['jeev083113e'], ['jeev083113f']]]

task_input_type = [#[['jeev081613e'], ['jeev081613c', 'jeev081613d']], 
	#[['jeev082313d'], ['jeev082313e']], 
	[['jeev082413d'], ['jeev082413e']], 
	[['jeev082413h'], ['jeev082413g']], 
	#[['jeev082813e'], ['jeev082813d']], 
	[['jeev082913c'], ['jeev082913d']], 
	[['jeev083113e'], ['jeev083113f']]]

ordered_task_filelist = [[[0], [1]], 
						 [[1], [0]], 
						 [[0], [1]], 
						 [[0], [1]],
						 ]

#task_input_names= ['8-15', '8-16', '8-23', '8-24a', '8-24b', '8-29', '8-31']
task_input_names= [#'8-16', 
				   #'8-23',
				   '8-24a', 
				   '8-24b', 
				   '8-29', 
				   '8-31']

number_names= [[[240], [241]], [[242], [243]], [[290], [291]], [[310], [311]]]


##################################################
##### Close vs. Far XTask Block Comparisons ######
##################################################
jeev_close = 	[[[3], [1]], 
				    [[1], [3]], 
				    [[3], [1]], 
				    [[3], [1]] ]

jeev_far = [[[1], [3]], 
					[[3], [1]],
					[[1], [3]],
					[[1], [3]] ]

grom_close = [[[3], [1, 0]], 
				   [[1], [3]], 
				   [[3], [1, 0]], 
				   [[1], [0, 3, 0]],
				   [[1], [3, 0]], 
				   [[1, 0], [0, 3]],
				   [[1], [0, 3]], 
				   [[1], [3]], 
				   [[3], [1]] ]

grom_far = [[[1], [0, 3]], 
				   [[3], [1]], 
				   [[1], [0, 3]], 
				   [[3], [1, 0, 0]],
				   [[3], [1, 0]], 
				   [[0, 3], [1, 0]],
				   [[3], [1, 0]], 
				   [[3], [1]], 
				   [[1], [3]] ]

##################################################
##### Close vs. Far w/in task Comparisons ########
##################################################

# Jeevs, assess (first 16 vs. last 16 trials) compared to middle 32 trials (divided by 2)

grom_win_close = [[[13], [3, 1]], 
				   [[13], [13]], 
				   [[13], [3, 1]], 
				   [[13], [3, 1, 0]],
				   [[13], [3, 1]], 
				   [[3, 1], [3, 1]],
				   [[13], [3, 1]], 
				   [[13], [13]], 
				   [[13], [13]] ]

grom_win_far = [[[13], [1, 3]], 
				   [[13], [13]], 
				   [[13], [1, 3]], 
				   [[13], [1, 0, 3]],
				   [[13], [1, 3]], 
				   [[1, 3], [1, 3]],
				   [[13], [1, 3]], 
				   [[13], [13]], 
				   [[13], [13]] ]
