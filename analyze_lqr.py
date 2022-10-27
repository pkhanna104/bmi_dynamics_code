import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import scipy.stats

import os
import pickle
import sys
import copy
import timeit

from bmi_dynamics_code import util as bmi_util
from bmi_dynamics_code import behavior_co_obs as bmi_b



def pool_lqr_sim_across_dm(dm_list, sim_dm_paths):
	'''
	We perform lqr simulations using different dynamics models ('dm').
	We will pool sims for each dm into a dictionary

	We will add a column to the data frames for 'dynamics' (will have the entry 'full' or 'decoder-null'
	We will stack the dataframes for the different simulations into one big dataframe.

	Relevant data frames for analysis:
	norm_u_df_n - contains the norm of u (input in the lqr simulations) for each condition (task, target)
	df_lqr_n - contains the moment-by-moment data for the lqr simulations

	dm_list: 
		list of dynamics types (typical: ['full', 'decoder_null'])

	'''

	lqr_dm = {}
	for dm in dm_list: 
		load_path = sim_dm_paths[dm]
		with open(load_path, 'rb') as f: 
			lqr_dm[dm] = pickle.load(f)

	#Pool data frames relevant for analysis:
	df_labels = ['norm_u_df_n', 'df_lqr_n']
	df_pool = {}
	for i,dm in enumerate(dm_list): #loop dynamics
		for dfl in df_labels: #loop data frames
			df = copy.deepcopy(lqr_dm[dm][dfl])
			df['dynamics'] = dm #add a column for 'dynamics'
			if i==0:
				#Initialize the pooled df: 
				df_pool[dfl] = df
			else: 
				#stack dfs
				df_stack = pd.concat([df_pool[dfl],df],ignore_index=True)
				df_pool[dfl] = df_stack
	return lqr_dm, df_pool       


def analyze_u_norm(lqr_dm, model_list, model_pairs, target_list, task_rot_list):
	'''
	analyze 'u_norm' of lqr sim across dynamics models

	model_list: 
		list of models
			each model is a tuple of (dyn_type, dyn_subset) 
				dyn_type is a str, an element of {'full', 'decoder_null'}
				dyn_subset is a str, an element of {'n_do', 'n_o', 'n_null', 'n_d'})

	model_pairs: 
		a list of pairs
			a pair is a tuple with two models


	'''
	r_u = {}
	for m in model_list:
		r_u[m]=[]
	
	for target in target_list:
		for task in task_rot_list:
			for i,m in enumerate(model_list):
				dyn_type = m[0]
				dyn_subset = m[1]

				df = lqr_dm[dyn_type]['norm_u_df_n']
				
				sel = \
				(df['target']==target)\
				&(df['task']==task)\
				&(df['model']==dyn_subset)
				y_i = float(df.loc[sel,'norm_u'])
				r_u[m].append(y_i)  
	
	#Stats: 
	for pair in model_pairs:
		r_wilcoxon = scipy.stats.wilcoxon(r_u[pair[0]], r_u[pair[1]])
		r_u[pair, 'wilcoxon'] = r_wilcoxon

	return r_u

def def_var_lists(num_targets, task_rot_list, num_mag_bins_analyze, num_angle_bins, num_neurons):
	#Useful especially when splitting an obstacle movement into cw vs ccw.
	move_list = []
	for target in range(num_targets):
		for task in task_rot_list:
			move_list.append((target,task))
	num_move = len(move_list)        
	#List of commands in case it's useful    
	c_list = [] #(bm,ba)
	for bm in range(num_mag_bins_analyze):
		for ba in range(num_angle_bins):
			c_list.append((bm,ba))        
	num_c = len(c_list)
	#List of neurons:
	n_list = ['n_'+str(i) for i in range(num_neurons)]
	
	return move_list, c_list, n_list

def main_compute_neural_command_diff(df, Kn, model_list, m_list, c_list, n_list, p_sig_match, shuffle_bool, num_shuffle):
	'''
	model_list: 
		list of models
			each model is a tuple of (dyn_type, dyn_subset) 
				dyn_type is a str, an element of {'full', 'decoder_null'}
				dyn_subset is a str, an element of {'n_do', 'n_o', 'n_null', 'n_d'})

	'''

	#
	model_cm = compute_command_sel(df, model_list, m_list, c_list)
	model_cm = match_pool_activity_to_command_movement(model_cm, df, model_list, c_list, m_list, p_sig=p_sig_match)
	model_cm = def_shuffle_mat(model_cm, model_list, c_list, m_list, num_shuffle)
	model_cm = compute_neural_command_diff(model_cm, df, Kn, model_list, m_list, c_list, n_list, shuffle_bool, num_shuffle)
	
	return model_cm

def compute_command_sel(df, model_list, m_list, c_list):
	'''
	Note: re-computing sel_m, sel_c on every loop is faster than computing it once, saving it, and accessing it from memory

	'''
	#Collect the selection 
	t_start = timeit.default_timer()
	model_cm = {} #model,command,movement data
	#key: model,command | model,command,movement
	# for each model: 
	# make an xarray data array for each command-movement and for the command
	# variables should be all the columns of the df
	for model in model_list: #model
		dyn_type = model[0]
		dyn_subset = model[1]
		sel_model = (df['model'] == dyn_subset)&(df['dynamics'] == dyn_type)

		for ic, c in enumerate(c_list): #command
			bm = c[0]
			ba = c[1]
			sel_ba = (df.loc[:,'u_v_angle_bin']==ba)
			sel_bm = (df.loc[:,'u_v_mag_bin']==bm) 
			#------------------------------------------------------------------
			sel_c = sel_model&sel_ba&sel_bm   #includes model
			#------------------------------------------------------------------
			model_cm[model,c,'sel'] = sel_c
			model_cm[model,c,'num_obs'] = sum(sel_c)
			for im, m in enumerate(m_list): #movement
				print(model, c, m)
				target = m[0]
				task = m[1]
				sel_m = (df.loc[:,'target']==target)&(df.loc[:,'task_rot']==task)
				#------------------------------------------------------------------
				sel_cm = sel_c&sel_m
				#------------------------------------------------------------------
				model_cm[model,c,m,'sel'] = sel_cm
				model_cm[model,c,m,'num_obs'] = sum(sel_cm)
	t_elapsed = timeit.default_timer()-t_start
	print(t_elapsed)
	return model_cm

def match_pool_activity_to_command_movement(model_cm, df, model_list, c_list, m_list, p_sig):
	#---------------------------------------------------------------------------------------------------------------------
	t_start = timeit.default_timer()
	#key: model,command | model,command,movement
	# for each model: 
	# make an xarray data array for each command-movement and for the command
	# variables should be all the columns of the df
	var = ['u_vx', 'u_vy']
	match_var = var
	for model in model_list: #model
		for ic, c in enumerate(c_list): #command    
			sel_c = model_cm[model,c,'sel']
			c_idx = df[sel_c].index.values
			c_da = bmi_b.df_idx2da(df,c_idx,var)

			for im, m in enumerate(m_list): #movement
				#data array of var of interest
				print(model, c, m)
				sel_cm = model_cm[model,c,m,'sel']
				cm_idx = df[sel_cm].index.values
				cm_da = bmi_b.df_idx2da(df,cm_idx,var)

				success, kept_list, discard_list, df_match, ttest_r, mean_r = \
						bmi_b.subsample_dataset_to_match_mean_target_dataset(match_var, d_ss=c_da, d_target=cm_da, p_sig=p_sig, frac_data_exclude_per_iter=0.05, min_frac_remain=0.1)            
				model_cm[model,c,m,'pool_match_idx'] = kept_list[0]
				model_cm[model,c,m,'pool_match_success'] = success
				model_cm[model,c,m,'pool_match_discard'] = discard_list[0]
				model_cm[model,c,m,'pool_match_ttest'] = ttest_r
				model_cm[model,c,m,'pool_match_mean'] = mean_r                    
	t_elapsed = timeit.default_timer()-t_start 
	print(t_elapsed)
	return model_cm

def def_shuffle_mat(model_cm, model_list, c_list, m_list, num_shuffle):
	#---------------------------------------------------------------------------------------------------------------------
	#make a shuffle mat for each command.
	#shuffle mat contains the idxs chosen for each shuffle 
	#For each movement, pick K samples at random from the 'movement-pooled'
	t_start = timeit.default_timer()
	for model in model_list:
		for c in c_list:
			for m in m_list:
				c_idxs = model_cm[model,c,m,'pool_match_idx']
				num_obs = model_cm[model,c,m,'num_obs']
				shuffle_mat = np.ones((num_obs, num_shuffle))*np.nan

				if len(c_idxs)<num_obs:
					for s in range(num_shuffle):
						shuffle_mat[:,s] = np.random.choice(c_idxs,num_obs,replace=True)                
				else:
					for s in range(num_shuffle):
						shuffle_mat[:,s] = np.random.choice(c_idxs,num_obs,replace=False)
				#ASSIGN:
				model_cm[model,c,m,'shuffle_mat'] = shuffle_mat
	t_elapsed = timeit.default_timer()-t_start       
	print(t_elapsed)
	return model_cm

def compute_neural_command_diff(model_cm, df, Kn, model_list, m_list, c_list, n_list, shuffle_bool, num_shuffle):
	#---------------------------------------------------------------------------------------------------------------------
	t_start = timeit.default_timer()
	n_df = np.array(df[n_list])

	mean_var = copy.copy(n_list)
	for model in model_list: #model
		for ic, c in enumerate(c_list): #command    
			for im, m in enumerate(m_list): #movement

				if model_cm[model,c,m,'pool_match_success']:    
					print(model,c,m)
					c_idxs = model_cm[model,c,m,'pool_match_idx']#use these idxs for average
	#                 mu_c = df.loc[c_idxs, mean_var].mean()
					mu_c = n_df[c_idxs, :].mean(axis=0) 

					#MOVE:
					cm_sel = model_cm[model,c,m,'sel']
					cm_idxs = cm_sel[cm_sel].index.values
	#                 mu_cm = df.loc[cm_idxs, mean_var].mean()       
					mu_cm = n_df[cm_idxs, :].mean(axis=0) 

					if shuffle_bool: 
						#SHUFFLE:
						nan_mat = np.ones((len(mean_var), num_shuffle))*np.nan
						s_mean = xr.DataArray(nan_mat, 
										  coords={'v':mean_var,'shuffle':range(num_shuffle)},
										  dims=['v','shuffle'])
						for s in range(num_shuffle):
							s_idxs = model_cm[model,c,m,'shuffle_mat'][:,s].astype(int)
	#                         mu_s = df.loc[s_idxs, mean_var].mean()
							mu_s = n_df[s_idxs, :].mean(axis=0)
							s_mean.loc[:,s] = mu_s

					#DIFFERENCE: 

					#ASSIGN:
	#                 model_cm[model,c,m,'mat_df'] = df.loc[cm_idxs, mean_var]
					model_cm[model,c,m,'n_c'] = mu_c
					model_cm[model,c,m,'n_cm'] = mu_cm

					diff_i = mu_c-mu_cm   
					diff_potent_i, diff_null_i, _ = bmi_util.proj_null_potent(Kn[2:4,:].T, np.array(diff_i).reshape((-1,1)))
					model_cm[model,c,m,'n_diff_true'] = diff_i             
					model_cm[model,c,m,'n_diff_norm_true'] = np.linalg.norm(diff_i)
					model_cm[model,c,m,'n_diff_norm_potent'] = np.linalg.norm(diff_potent_i)
					model_cm[model,c,m,'n_diff_norm_null'] = np.linalg.norm(diff_null_i)

					if shuffle_bool: 
						model_cm[model,c,m,'n_s'] = s_mean              
						n_c_rep = np.array(mu_c)[...,None]
						model_cm[model,c,m,'n_diff_s'] = n_c_rep-s_mean
						model_cm[model,c,m,'n_diff_s_norm'] = np.linalg.norm(model_cm[model,c,m,'n_diff_s'], axis=0)
						model_cm[model,c,m,'n_diff_s_norm_mean'] = model_cm[model,c,m,'n_diff_s_norm'].mean()

	t_elapsed = timeit.default_timer()-t_start
	print(t_elapsed)
	
	return model_cm

def collect_neural_command_diff(model_cm, Kn, model_list, c_list, m_list, min_obs, shuffle_bool):
	#---------------------------------------------------------------------------------------------------------------------
	#loop over all conditions, collect a list of differences:
	
	model_diff = {}
	for model in model_list: #model
		model_diff[model] = {'total':[], 'potent':[],'null':[], 'shuffle':[]}
		for ic, c in enumerate(c_list): #command    
			for im, m in enumerate(m_list): #movement
				if model_cm[model,c,m,'pool_match_success']:  
					#project the diff

					diff_i = model_cm[model,c,m,'n_diff_true']

					if model_cm[model,c,m,'num_obs'] > min_obs:
						if not np.isnan(diff_i).any():
							diff_potent_i, diff_null_i, _ = bmi_util.proj_null_potent(Kn[2:4,:].T, np.array(diff_i).reshape((-1,1)))
							norm_potent_i = np.linalg.norm(diff_potent_i)
							norm_null_i = np.linalg.norm(diff_null_i)

							model_diff[model]['total'].append(model_cm[model,c,m,'n_diff_norm_true'])
							model_diff[model]['potent'].append(norm_potent_i)
							model_diff[model]['null'].append(norm_null_i)

							if shuffle_bool: 
								model_diff[model]['shuffle'].append(model_cm[model,c,m,'n_diff_s_norm_mean']) 
	return model_diff

def analyze_n_diff(model_diff, model_list, model_pairs):
	'''
	models in 'model_list' are compared against the shuffle
	models in model_pairs are compared against each other

	'''

	r_n = {}
	#1. collect individual models' data: 
	for m in model_list:
		r_n[m, 'total'] = np.array(model_diff[m]['total'])

	#2. compare models
	for pair in model_pairs:
		r_n[pair, 'model_pair_ks'] = scipy.stats.ks_2samp(np.array(r_n[pair[0], 'total']), np.array(r_n[pair[1], 'total']))

	#3. compare model to its own shuffle
	for m in model_list:
		m_i = r_n[m,'total']
		s_i = np.array(model_diff[m]['shuffle'])
		r_n[m, 'shuffle'] = s_i
		r_n[m, 'model_shuffle_ks'] = scipy.stats.ks_2samp(m_i, s_i)
	return r_n

#---------------------------------------------------------------------------------------------------------------------

def compute_command_sel_attempt(df, model_list, m_list, c_list):
	'''
	This was an attempt to make code run faster, but it is actually slower...
	I precomputed selections, but accessing from memory seems to be slower than just re-computing on the fly
	model_list: 
	list of models
		each model is a tuple of (dyn_type, dyn_subset) 
			dyn_type is a str, an element of {'full', 'decoder_null'}
			dyn_subset is a str, an element of {'n_do', 'n_o', 'n_null', 'n_d'})


	'''
	#Collect the selection 
	t_start = timeit.default_timer()
	model_cm = {} #model,command,movement data
	#key: model,command | model,command,movement
	# for each model: 
	# make an xarray data array for each command-movement and for the command
	# variables should be all the columns of the df

	b_sel = compute_behavior_sel(df, m_list, c_list)
	model_sel = compute_model_sel(df, model_list)

	#Loop behavior: 
	for ic, c in enumerate(c_list): #command
		sel_c = b_sel['command', c]
		for im, m in enumerate(m_list): #movement
			sel_m = b_sel['move', m]
			for model in model_list: #model
				print(model, c, m)
				sel_model = model_sel[model]

				#-------------------------------------------------------------------------------------------
				sel_c_model = sel_c&sel_model
				#-------------------------------------------------------------------------------------------
				model_cm[model,c,'sel'] = sel_c_model
				model_cm[model,c,'num_obs'] = sum(sel_c)				

				#-------------------------------------------------------------------------------------------
				sel_c_m_model = sel_c_model&sel_m 
				#-------------------------------------------------------------------------------------------
				model_cm[model,c,m,'sel'] = sel_c_m_model
				model_cm[model,c,m,'num_obs'] = sum(sel_c_m_model)

	t_elapsed = timeit.default_timer()-t_start
	print(t_elapsed)
	return model_cm

def compute_model_sel(df, model_list):
	'''
	model_list: 
	list of models
		each model is a tuple of (dyn_type, dyn_subset) 
			dyn_type is a str, an element of {'full', 'decoder_null'}
			dyn_subset is a str, an element of {'n_do', 'n_o', 'n_null', 'n_d'})
	'''
	model_sel = {}
	for model in model_list:
		dyn_type = model[0]
		dyn_subset = model[1]
		sel_model = (df['model'] == dyn_subset)&(df['dynamics'] == dyn_type)
		model_sel[model] = sel_model
	return model_sel

def compute_behavior_sel(df, m_list, c_list):
	'''
	identify samples of the relevant behavior in the lqr simulation dataframe
	behavior refers to command and movement
	command: angle bin, magnitude bin
	movement: task, target
	'''
	#behavior selection
	b_sel = {}

	#command
	for ic, c in enumerate(c_list): #command
		bm = c[0]
		ba = c[1]
		#------------------------------------------------------------
		sel_ba = (df.loc[:,'u_v_angle_bin']==ba)
		sel_bm = (df.loc[:,'u_v_mag_bin']==bm)
		sel_c = sel_ba&sel_bm
		#------------------------------------------------------------
		#assign:
		# b_sel['angle', ba] = sel_ba
		# b_sel['mag', bm] = sel_bm
		b_sel['command', c] = sel_ba&sel_bm
	
	#movement:
	for im, m in enumerate(m_list): #movement
		target = m[0]
		task = m[1]
		#------------------------------------------------------------
		sel_target = (df.loc[:,'target']==target)
		sel_task = (df.loc[:,'task_rot']==task)
		sel_m = sel_target&sel_task
		#------------------------------------------------------------
		#assign:
		# b_sel['target', target]	= sel_target
		# b_sel['task_rot', task]	= sel_task
		b_sel['move', m] = sel_m
	return b_sel