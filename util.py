import scipy.io as sio
import scipy.stats as sio_stat
import scipy.linalg as slinalg
from sklearn.decomposition import FactorAnalysis
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import pickle
import sys
import copy
import pandas as pd

def verbose_print(input, verbose_bool=1):
	if verbose_bool:
		print(input)

def invert_dic(dic):
	#Note: will only work if the dictionary is one-to-one, is invertible.  
	dic_inverse = {v: k for k, v in dic.items()}
	return dic_inverse

def mean_var_std_sem(data_vec):
	result = {'mean':np.mean(data_vec), 
	'var':np.var(data_vec), 
	'std':np.std(data_vec), 
	'sem':sio_stat.sem(data_vec)}
	return result

def make_df_idx(index_df, task_idx, tc, target, trial, pre, num_data):
	"""
	makes a dataframe containing indexing information, typically for a trial of data
	(task, tc, target, trial, bin (index wtihin the trial))
	input:
	pre - how many "prefix" samples are in the data
	num_data - number of samples
	"""
	label = ['task', 'tc', 'target', 'trial', 'bin', 'trial_start', 'trial_stop']
	m_task = np.ones(num_data)*task_idx
	m_tc = np.ones(num_data)*tc
	m_target = np.ones(num_data)*target
	m_trial = np.ones(num_data)*trial
	m_bin = np.arange(num_data)-pre
	m_trial_start = np.zeros(num_data)
	m_trial_start[pre] = 1
	m_trial_stop = np.zeros(num_data)
	m_trial_stop[-1] = 1

	m_idx = np.stack((m_task, m_tc, m_target, m_trial, m_bin, m_trial_start, m_trial_stop), axis=1) #num_data X 5
	df_idx = pd.DataFrame(m_idx, index=index_df, columns=label)
	return df_idx

def make_df_size(index_df, target_size, obs_size, num_data): 
	"""
	makes a dataframe containing target and obstacle size information, typically for a trial of data
	(target_loc, target_size, obs_loc, obs_size)
	input:
	target_size - 
	obs_size - 
	num_data - number of samples (typically in the trail)	
	"""
	label = ['target_size', 'obs_size']
	m_target_size = np.ones(num_data)*target_size
	m_obs_size = np.ones(num_data)*obs_size
	m_size = np.stack((m_target_size, m_obs_size), axis=1)
	df_size = pd.DataFrame(m_size, index=index_df, columns=label)
	return df_size

def def_target_over_trials(data, task_codes):
	target_over_trials = {}
	for task_i, (task_key, code_list) in enumerate(task_codes.items()):
		#print("index: {}, key: {}, value: {}".format(task_i, task_key, task_val))
		for code_i in code_list:
			code_targets = []
			for i_trial, trial in enumerate(data[code_i, 'cursor_pos']):
				trl = np.nonzero(data[code_i, 'trial_ix']==i_trial)[0]
				# target number of this index: 
				target_num = int(data[code_i, 'target_index'][trl[0]])
				code_targets.append(target_num)
			target_over_trials[code_i] = np.array(code_targets)
	return target_over_trials

def def_train_test_trials(task_codes, target_over_trials, targets, train_frac, num_folds, verbose=0):
	train_test_fold = {}
	for i_fold in np.arange(0,num_folds):
		train_test_fold[(i_fold, 'train')] = {}
		train_test_fold[(i_fold, 'test')] = {}
		for i_task, (task_key, code_list) in enumerate(task_codes.items()):
			for code_i in code_list:
				if verbose:
					print(str(code_i) + ' ' + task_key)        
			#     print(task_key)
				for i_target in targets:
				#         print('target ' + str(i_target))
					trials = np.where(target_over_trials[code_i] == i_target)[0]
					#Shuffle: 
					trial_shuffle = copy.copy(trials)
					np.random.shuffle(trial_shuffle)
					#Select trials
					num_i = trials.shape[0]
					num_train_i = np.round(train_frac*num_i).astype(int)        
					train_trials = trial_shuffle[:num_train_i]
					test_trials = trial_shuffle[num_train_i:]
				#         print(trial_shuffle)
				#         print(num_train_i)
				#         print(train_trials)
				#         print(test_trials)
					train_test_fold[(i_fold, 'train')][(code_i, i_target)] = train_trials
					train_test_fold[(i_fold, 'test')][(code_i, i_target)] = test_trials
	return train_test_fold


def calc_n_mean_var(data, task_codes, targets, sample_rate, target_over_trials):
	#TODO: can add which trials to use to calculate
	n_pool_list = []
	for task_i, (task_key, code_list) in enumerate(task_codes.items()):
		for code_i in code_list:
			print(str(code_i) + ' ' + task_key)
			for target_i in targets:
				print('target:')
				print(target_i)

				trial_idxs_plot = np.nonzero(target_over_trials[code_i]==target_i)[0]
				for i, trial_i in enumerate(trial_idxs_plot):
					if sample_rate == 60:
						n_i = data[(code_i, 'binned_spk_cnts_60Hz')][trial_i]
					elif sample_rate == 10:
						n_i = data[(code_i, 'binned_spk_cnts')][trial_i]
					n_pool_list.append(n_i) 
	
	n_pool = np.concatenate(n_pool_list,axis=0)
	n_pool_mean = np.mean(n_pool, axis=0)
	n_pool_mean.shape
	n_pool_var = np.var(n_pool, axis=0)

	return n_pool_mean, n_pool_var

def enum_past_future_lags(lag_range):
	#INPUT: lag_range is an integer
	#m = minus, past
	#p = plus, future
	dic_lag = {'mp':{}, 'p':{}, 'm':{}}
	for lag_i in np.arange(0,lag_range+1):
		dic_lag['mp'][lag_i] = np.arange(-lag_i, lag_i+1)
		dic_lag['m'][lag_i] = np.arange(-lag_i, 0+1)
		dic_lag['p'][lag_i] = np.arange(0, lag_i+1)
		
	return dic_lag

def mat_lag(X, num_lag):
	#INPUT:
	#X - num_var X num_samples
	#OUTPUT: 
	#X_lag - (num_var*num_lag) X (num_samples - num_lag)
	num_var = X.shape[0]
	num_samples = X.shape[1]
#     print(num_var)
#     print(num_samples)
	num_var_lag = num_var*num_lag
	num_samples_lag = num_samples - (num_lag-1)
	X_lag = np.zeros((num_var_lag, num_samples_lag))
	for i in np.arange(0,num_lag):
		X_i = X[:,i:(num_samples_lag+i)]
		
		offset = i*(num_var)
		X_lag[offset:(offset+num_var),:] = X_i
	return X_lag

def data_X_Y_lag(X, Y, x_num_lag, y_lag_pred):
	#Generates data matrices for predicting a particular lag of Y using lags in X
	#y_lag_pred: relative to the most recent lag in X.  
	#Examples:
	#0: predict Y using 'x_num_lag' into past
	#1: predict Y one lag into future 
	#-1: predict Y using x_num_lag-2 lags into past, current lag, and 1 lag into future.
	#
	#INPUT:
	#X - num_var by num_samples, data used to predict
	#Y - num_var by num_samples, data that is predicted
	#x_num_lag - number of X lags to use
	#y_lag_pred - lag of Y to predict, relative to most recent lag in X
	#OUTPUT:
	#X_lag_pred, Y_pred
	
	num_samples = X.shape[1]
	
	X_lag = mat_lag(X,x_num_lag)
#     #--
#     print('X_lag:')
#     print(X_lag)
	num_samples_lag = X_lag.shape[0] #should be: num_samples-x_num_lag
	x_idxs = np.arange(0,num_samples-x_num_lag+1)
	x_idxs_lag = np.arange(0,num_samples-x_num_lag+1)+x_num_lag-1
	#--
#     print(x_idxs)
	y_pred_idxs = x_idxs_lag + y_lag_pred
	#--
#     print(y_pred_idxs)
	y_invalid = np.where(np.logical_or(y_pred_idxs<0, y_pred_idxs>=num_samples))[0]
	#--
#     print('y invalid:')
#     print(y_invalid)
	#Remove invalid idxs:
	if y_invalid.shape[0] != 0:
		#--        
#         print(x_idxs[y_invalid])
		X_lag_pred = np.delete(X_lag, x_idxs[y_invalid], axis=1)
		#--    
#         print(X_lag_pred)
		y_pred_idxs_valid = np.delete(y_pred_idxs, y_invalid)
		Y_pred = Y[:,y_pred_idxs_valid]
	else:
		X_lag_pred = X_lag
		#--
#         print(y_pred_idxs)
		Y_pred = Y[:,y_pred_idxs]
	return X_lag_pred, Y_pred

# def lag_vec2

def reg_R2(reg, X,Y):
	#INPUT: 
	#reg: regression object (from sklearn.linear_model import LinearRegression)
	#X: num_samples X num_var
	#Y: num_samples X num_var
	#OUTPUT:
	#R2_ind - average the R2 of each output variable
	#R2_pop - sum variance predicted / sum total variance
	
	Y_pred = reg.predict(X)
	Y_pred_e = Y-Y_pred
	#Individual var:
	R2_ind = np.nanmean(1-np.var(Y_pred_e,axis=0)/np.var(Y,axis=0))
	#Population var:
	R2_pop = 1-np.nansum(np.var(Y_pred_e,axis=0))/np.nansum(np.var(Y,axis=0))    
	
	return R2_ind, R2_pop		    	

def accum_data_a_s_lag(data, a_bool, s_bool, task_code, trials, x_num_lag, y_lag_pred, X, Y):
	#INPUT:
	#data
	#a_bool - use action = neural_push
	#s_bool - use state = cursor_state
	#task_code - for co or obs
	#trials - trial idxs 
	#
	#OUTPUT:
	#X_lag_pred
	#Y_pred
	a_idxs = np.array([2,3]) #Only use velocity idxs of neural push
	#Reminder: position idxs of neural push: are just 1/10th of velocity idxs of push, are redundant
	
	task_val = task_code
	for i_trial in trials: #[train_trials[0]] #trials:
		#neural activity
		n_i = data[(task_val, 'binned_spk_cnts')][i_trial].T
		if a_bool and not s_bool:
			x_i = data[(task_val, 'neural_push')][i_trial][:,a_idxs].T #num_dim by num_samples
		elif s_bool and not a_bool:
			x_i = data[(task_val, 'cursor_state')][i_trial].T #num_dim by num_samples
		elif s_bool and a_bool:
			a_i = data[(task_val, 'neural_push')][i_trial][:,a_idxs].T #num_dim by num_samples
			s_i = data[(task_val, 'cursor_state')][i_trial].T #num_dim by num_samples
			x_i = np.vstack((a_i, s_i))                
		X_lag_pred, Y_pred = data_X_Y_lag(x_i, n_i, x_num_lag, y_lag_pred)
		X.append(X_lag_pred)
		Y.append(Y_pred)
	return X, Y

def finite_diff(data_mat):
	#returns finite difference, and corresponding data
	#input: 
	#data_mat: num_samples x num_dim
	
	diff_filter = 0.5*np.array([1, 0, -1])
	num_samples, num_dim = data_mat.shape
	diff_mat = np.zeros((num_samples-2,num_dim))
	data_mat_trunc = np.zeros((num_samples-2,num_dim))
	for dim_i in range(0,num_dim):
		diff_mat[:,dim_i] = np.convolve(data_mat[:,dim_i], diff_filter, 'valid')
		data_mat_trunc[:,dim_i] = data_mat[1:-1,dim_i]
	return diff_mat, data_mat_trunc

def tangling(data):
	#input: 
	#data_mat: num_samples x num_dim
	#0.1 times the average squared magnitude
	eps = 0.1*np.mean(np.linalg.norm(data,axis=1))

	#calculate finite_diff: 
	data_d, data = finite_diff(data)
	num_samples = data.shape[0]
	result = np.zeros((num_samples,1))
	for i in range(num_samples):
		#for each sample, subtract sample from mat
		data_i = data[i,:]
		data_d_i = data_d[i,:]
		state_diff_vec = np.linalg.norm(data-data_i, axis=1)
		d_diff_vec = np.linalg.norm(data_d-data_d_i, axis=1)
		tangling_vec = d_diff_vec/(state_diff_vec+eps)
		result[i] = np.max(tangling_vec)
	
	return result, data_d, data




def inflate_cursor_state(cursor_state):
	#input: cursor_state is of size: num_samples X num_cursor_dim
	#idx: quantity
	#0: px
	#1: irrelevant p
	#2: py
	#3: vx
	#4: irrelevant v
	#5: vy
	#6: offset

	num_samples = cursor_state.shape[0]
	zeros_vec = np.zeros((num_samples,1))
	ones_vec = np.ones((num_samples,1))
	col0 = cursor_state[:,0].reshape((num_samples,1))
	col1 = zeros_vec
	col2 = cursor_state[:,1].reshape((num_samples,1))
	col3 = cursor_state[:,2].reshape((num_samples,1))
	col4 = zeros_vec
	col5 = cursor_state[:,3].reshape((num_samples,1))
	col6 = ones_vec
	inflate_state = np.hstack((col0, col1, col2, col3, col4, col5, col6))    
	return inflate_state    

def decompose_decoder(F, K):
	"""
	decoder performs: 
	x(t)=Fx(t-1)+Ky(t)
	where
	x is 'cursor_state_in'
	y is 'spk_cnts'
	K is kalman gain, also called 'KG'

	F can be split into: 
	#Position Output:
	F_po: offset->position
	F_pp: position->position
	F_pv: velocity->position
	
	#Velocity Output
	F_vo: velocity->offset
	F_vp: position->velocity
	F_vv: velocity->velocity

	#Offset Output
	F_vo: velocity->offset
	F_vp: position->velocity
	F_vv: velocity->velocity	

	KG can be split into: 
	K_pn: neural->position
	K_vn: neural->velocity
	"""

	output_var = ['o', 'p', 'v']
	var_idxs = {}
	var_idxs['o'] = [6]
	var_idxs['p'] = [0,2]
	var_idxs['v'] = [3,5]
	var_idxs['n'] = range(0, K.shape[1])
	output_idxs = [0,2,3,5,6]

	#Decompose F: 
	F_d = {}
	F_hat = np.zeros(F.shape) #to confirm the decomposition makes sense
	for var_out in output_var: 		
		F_d[var_out] = {}
		for var_in in output_var: 	
				sel = np.ix_(var_idxs[var_out], var_idxs[var_in])
				F_d[var_out][var_in] = np.zeros(F.shape)
				F_d[var_out][var_in][sel] = F[sel]
				F_hat += F_d[var_out][var_in]
	
	sel = np.ix_(output_idxs, output_idxs)
	if np.allclose(F_hat[sel], F[sel]):
		print('F_hat close to F')
	else:
		print('F_hat not close to F')

	#Decompose K:
	K_d = {}
	K_hat = np.zeros(K.shape)
	for var_out in output_var: 
		sel = np.ix_(var_idxs[var_out], var_idxs['n'])
		K_d[var_out] = np.zeros(K.shape)
		K_d[var_out][sel] = K[sel]
		K_hat += K_d[var_out]

	if np.allclose(K_hat, K):
		print('K_hat close to K')
	else:
		print('K_hat not close to K')

	decoder = {}
	decoder['F'] = F
	decoder['K'] = K
	decoder['F_d'] = F_d
	decoder['F_hat'] = F_hat
	decoder['K_d'] = K_d
	decoder['K_hat'] = K_hat
	decoder['out_var'] = output_var
	decoder['out_idxs'] = output_idxs
	decoder['var_idxs'] = var_idxs

	return decoder

def plot_targets(locations, colors, radii): 
	"""
	targets are circles.  
	plots the targets at desired location with desired radius 
	locations, colors, sizes are all lists of the same length
	"""
	num_theta = 100
	for l,c,r in zip(locations, colors, radii):
		# print(l)
		plot_circle(l,r,num_theta,c)

def plot_circle(ctr, radius, num_theta, color):
	r = radius
	theta = np.linspace(0, 2*np.pi, num_theta)
	plt.plot(r*np.cos(theta)+ctr[0], r*np.sin(theta)+ctr[1], color)

def plot_obstacles(locations, colors, sizes):
	"""
	obstacles are squares.  
	plots the obstacles at desired location with desired size (side length of obstacle)
	locations, colors, sizes are all lists of the same length
	"""
	#NOTES: 
	#center is 1/2 way between origin (0,0) and the target location.
	#they are 3cm square
	# num_obs = locations.shape[]
	for l,c,s in zip(locations, colors, sizes):
		# print(l)
		plot_rect(l,s,s,c)

def plot_rect(ctr, width, height, color):
	d = np.array([width/2, height/2])
	#pts: 4 x 2
	# NN, PN, PP, NP, NN 
	pts = np.array([[ctr[0]-d[0], ctr[1]-d[1]],
		[ctr[0]+d[0], ctr[1]-d[1]],
		[ctr[0]+d[0], ctr[1]+d[1]],
		[ctr[0]-d[0], ctr[1]+d[1]],
		[ctr[0]-d[0], ctr[1]-d[1]]])
	plt.plot(pts[:,0],pts[:,1],color, linewidth=4.5)	
	plt.plot(pts[:,0],pts[:,1],'w', linewidth=1.5)

def sim_bmi(cursor_state_in, spk_cnts, F, K): 
	"""
	Extract the individual contribution of each input variable to the output variable
	Group contribution by "kinematic contribution", "neural contribution"

	cursor_state_in: 
		num_samples X 4, or num_samples X 7
	spk_cnts: 
		num_samples X num_neurons
	F: 
		7 x 7
		cursor dynamics matrix
	K: 
		7 x num_neurons
		neural -> cursor matrix

	OUTPUT: 
	r: 
		simulated cursor output
		num_samples-1 X num
	r_kn: decompose cursor update into terms due to kinematics and those due to current neural activity
	r_d: decompose cursor update into terms due to each type of input: ['p', 'v', 'o', 'n'], where 'o' is offset
	"""
	len_cursor_state = 7
	d = decompose_decoder(F,K)

	if cursor_state_in.shape[1] !=  len_cursor_state:
		cursor_state = inflate_cursor_state(cursor_state_in)
	elif cursor_state_in.shape[1] ==  len_cursor_state:
		cursor_state = cursor_state_in
	else:
		print('error in cursor_state_in dimensionality!')
		print(cursor_state.shape)
		return []
	c_tm1 = cursor_state[:-1,:]
	c_t = cursor_state[1:,:]
	n_t = spk_cnts[1:,:]	

	#SIMULATION
	#---------------------------------------------------------------------
	r_kn = {}
	r_d = {}
	r = np.zeros(c_tm1.shape)
	for o in d['out_var']: 
		r_d[o] = {}
		r_kn[o] = {}
		#Loop kinematics: 
		r_k = np.zeros(c_tm1.shape)
		for i in d['out_var']: 
			F_sel = d['F_d'][o][i]
			r_d[o][i] = np.dot(F_sel, c_tm1.T).T
			print(r_d[o][i].shape)
			r += r_d[o][i]
			r_k += r_d[o][i]
		# print(r.shape)
		r_kn[o]['k'] = copy.copy(r_k)
		#Neural:
		K_sel = d['K_d'][o]
		r_d[o]['n'] = np.dot(K_sel, n_t.T).T
		r_kn[o]['n'] = copy.copy(r_d[o]['n'])
		r += r_d[o]['n']
	#Confirm reconstruction: 
	if np.allclose(r, c_t):
		print('SUCCESSFUL reconstruction')
	else:
		print('FAIL reconstruction')

	# #Contribution Magnitude
	# #---------------------------------------------------------------------
	# #4 x num_samples
	# #Compute the the contribution size of the delta
	# mag = {}
	# for o in d['out_var']: 
	# 	mag[o] = {}
	# 	#Loop kinematics: 
	# 	for i in d['out_var']: 
	# 		mag[o][i] = np.linalg.norm(r_d[o][i],axis=1)

	# #Relative Contribution
	# #---------------------------------------------------------------------
	# #position: won't consider contribution of last position
	# #velocity: WILL consider contribution of last velocity
	# frac = {'p':{}, 'v':{}}
	# denom = {'p':np.zeros((c_t.shape[0],1)), 'v':np.zeros((c_t.shape[0],1))}

	# #Position:
	# #---------------------------------------------------------------------
	# o = 'p'
	# for i in d['out_var']:
	# 	if i is not 'p':
	# 		print(mag[o][i].shape)
	# 		denom[o] += mag[o][i]
	# for i in d['out_var']: 
	# 	if i is not 'p':
	# 		frac[o][i] = mag[o][i]/denom[o]

	# #Velocity:
	# #---------------------------------------------------------------------
	# v_denom = np.zeros((c_t.shape[0],1))
	# o = 'v'
	# for i in d['out_var']:
	# 	denom[o] += mag[o][i]
	# for i in d['out_var']: 
	# 	frac[o][i] = mag[o][i]/v_denom

	#ASSIGN:
	#---------------------------------------------------------------------
	result = {}
	result['decoder'] = d
	result['c_tm1'] = c_tm1
	result['c_t'] = c_t
	result['n_t'] = n_t
	result['r'] = r
	result['r_d'] = r_d
	result['r_kn'] = r_kn
	return result


def sim_bmi_old(cursor_state_in, spk_cnts, F, KG):
	"""
	Performs x(t)=Fx(t-1)+KGy(t)
	where
	x is 'cursor_state_in'
	y is 'spk_cnts'

	figures out the contribution of offset, position, velocity, 
	"""
	#INPUT:
	#cursor_state: (num_samples x 4)
	#-- 0:px, 1:py, 2:vx, 3:vy
	#spk_cnts: (num_samples x num_neurons)
	#F: (7x7)
	#KG: (7 x num_neurons)
	#simulate BMI, and return position contributions and velocity contributions

	if cursor_state_in.shape[1] !=  7:
		cursor_state = inflate_cursor_state(cursor_state_in)
	elif cursor_state_in.shape[1] ==  7:
		cursor_state = cursor_state_in
	else:
		print('error in cursor_state_in dimensionality!')
		print(cursor_state.shape)
		return []

	o_i = [6]
	p_i = [0,2]
	v_i = [3,5]
	pv_i = [0,2,3,5]
	n_i = range(0,KG.shape[1])

	#offset->position
	F_po = np.zeros(F.shape)
	po_sel = np.ix_(p_i,o_i)
	F_po[po_sel] = F[po_sel]
	#position->position
	F_pp = np.zeros(F.shape)
	pp_sel = np.ix_(p_i, p_i)
	F_pp[pp_sel] = F[pp_sel]
	#velocity->position
	F_pv = np.zeros(F.shape)
	pv_sel = np.ix_(p_i, v_i)
	F_pv[pv_sel] = F[pv_sel]
	#neural->position
	K_pn = np.zeros(KG.shape)
	pn_sel = np.ix_(p_i,n_i)
	K_pn[pn_sel] = KG[pn_sel]

	#velocity
	#offset->velocity
	F_vo = np.zeros(F.shape)
	vo_sel = np.ix_(v_i, o_i)
	F_vo[vo_sel] = F[vo_sel]
	#(position->velocity is 0)
	F_vp = np.zeros(F.shape)
	vp_sel = np.ix_(v_i,p_i)
	F_vp[vp_sel] = F[vp_sel]
	#velocity->velocity
	F_vv = np.zeros(F.shape)
	vv_sel = np.ix_(v_i,v_i)
	F_vv[vv_sel]=F[vv_sel]
	#neural->velocity
	K_vn = np.zeros(KG.shape)
	vn_sel = np.ix_(v_i,n_i)
	K_vn[vn_sel] = KG[vn_sel]

	F_hat = F_po + F_pp + F_pv + F_vo + F_vp + F_vv
	print('F_hat close to F')
	print(np.allclose(F_hat[pv_i,:], F[pv_i,:]))

	K_hat = K_pn + K_vn 
	print('K_hat close to K')
	print(np.allclose(K_hat[pv_i,:], KG[pv_i,:]))
	#Do bmi sim using above submatrices and check that they reconstruct: 
	#task as input cursor state and neural, cutting off the last saved time point:
	c_tm1 = cursor_state[:-1,:]
	# n_last = spk_cnts[:-1,:]
	n_t = spk_cnts[1:,:]

	#recon:
	#extract simulated dimensions: pos (2d), vel (2d)
	r_po = np.dot(F_po, c_tm1.T)[pv_i,:].T
	r_pp = np.dot(F_pp, c_tm1.T)[pv_i,:].T
	r_pv = np.dot(F_pv, c_tm1.T)[pv_i,:].T
	r_pn = np.dot(K_pn, n_t.T)[pv_i,:].T
	r_vo = np.dot(F_vo, c_tm1.T)[pv_i,:].T
	r_vp = np.dot(F_vp, c_tm1.T)[pv_i,:].T
	r_vv = np.dot(F_vv, c_tm1.T)[pv_i,:].T
	r_vn = np.dot(K_vn, n_t.T)[pv_i,:].T

	r = r_po + r_pp + r_pv + r_pn + r_vo + r_vp + r_vv + r_vn

	result = {'r':r,
	'r_po':r_po,
	'r_pp':r_pp,
	'r_pv':r_pv,
	'r_pn':r_pn,
	'r_vo':r_vo,
	'r_vp':r_vp,
	'r_vv':r_vv,
	'r_vn':r_vn}

	#4 x num_samples
	#Compute the the contribution size of the delta
	#position: won't consider contribution of last position
	#velocity: WILL consider contribution of last velocity
	diff_frac = {}
	p_denom = np.linalg.norm(r_po,axis=1) + np.linalg.norm(r_pv,axis=1) + np.linalg.norm(r_pn,axis=1)
	v_denom = np.linalg.norm(r_vo,axis=1) + np.linalg.norm(r_vp,axis=1) + np.linalg.norm(r_vv,axis=1) + np.linalg.norm(r_vn,axis=1) 
	diff_frac['p_denom'] = p_denom
	diff_frac['po'] = np.linalg.norm(r_po,axis=1)/p_denom
	diff_frac['pv'] = np.linalg.norm(r_pv,axis=1)/p_denom
	diff_frac['pn'] = np.linalg.norm(r_pn,axis=1)/p_denom

	diff_frac['v_denom'] = v_denom
	diff_frac['vo'] = np.linalg.norm(r_vo,axis=1)/v_denom
	diff_frac['vv'] = np.linalg.norm(r_vv,axis=1)/v_denom
	diff_frac['vn'] = np.linalg.norm(r_vn,axis=1)/v_denom

	return result, diff_frac
	

def draw_boxplot(data, offset,edge_color, fill_color):
	pos = np.arange(data.shape[1])+offset 
	bp = plt.boxplot(data, positions= pos, widths=0.3, patch_artist=True, manage_xticks=False)
	for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
		plt.setp(bp[element], color=edge_color)
	for patch in bp['boxes']:
		patch.set(facecolor=fill_color)

def hist_step_plot(bins, H, color, alpha=1):
	num_bins = bins.shape[0]-1
	print(num_bins)
	bin_edge_plot = np.zeros(2*num_bins+2)
	H_edge_plot = np.zeros(2*num_bins+2)
	print(H_edge_plot.shape)
	#First and Last bins: 
	bin_edge_plot[0]=bins[0]
	bin_edge_plot[-1]=bins[-1]
	for i in range(num_bins):
		bin_edge_plot[(2*i+1):(2*i+2+1)] = bins[i:(i+2)]
		H_edge_plot[(2*i+1):(2*i+2+1)] = H[i]
	plt.step(bin_edge_plot, H_edge_plot, color=color, alpha=alpha) 

def proj_null_potent(mat, data_mat):
	#mat: num_dim x num_col.  Wnat to find projection of data_mat into column space of mat.
	#data_mat: num_dim x num_data_pts
	#returns: data_null, data_potent
	
	#step 1: take svd of mat: 
	u,s,vh = np.linalg.svd(mat, full_matrices=False)
	data_potent = np.dot(u, np.dot(u.T, data_mat))
	data_null = data_mat - data_potent
	return data_potent, data_null, u

def norm_diff(x,y):
	return np.linalg.norm(x-y)

def compare_data_pairs_in_mat(data_mat, compare_fn):
	#assume:
	#data_mat is of size num_data x num_dim
	num_data = data_mat.shape[0]
	# result_mat = np.zeros((num_data, num_data))
	result_vec = []
	for i in np.arange(0,num_data): 
		for j in np.arange(i,num_data): 
			d_i = data_mat[i,:]
			d_j = data_mat[j,:]
			r_ij = compare_fn(d_i, d_j)
			# result_mat[i,j] = r_ij
			# result_mat[j,i] = r_ij
			result_vec.append(r_ij)
			
	result_vec = np.array(result_vec)           
	#, result_mat           
	return result_vec

def compare_data_pairs_across_mat(data_mat1, data_mat2, compare_fn):
	#input:
	#data_mat1 is of size num_data1 x num_dim
	#data_mat2 is of size num_data

	result_vec = []
	num_data1 = data_mat1.shape[0]
	num_data2 = data_mat2.shape[0]
	for i_d1 in np.arange(0,num_data1):
		for j_d2 in np.arange(0, num_data2):
			d1_i = data_mat1[i_d1,:]
			d2_j = data_mat2[j_d2,:]
			r_ij = compare_fn(d1_i,d2_j)
			result_vec.append(r_ij)
	result_vec = np.array(result_vec)
	return result_vec

class MidpointNormalize(colors.Normalize):
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y))