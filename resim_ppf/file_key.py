
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import gc
import scipy.stats
import test_full_no_adapt as tfna
from resim_ppf import *
import pickle
import analysis_config

trialList = np.zeros((11, 3))
trialList[1,:]  = [5, 2, 11]
trialList[2,:]  = [11, 2, 5]
trialList[3,:]  = [3, 13, 11]
trialList[4,:]  = [11, 13, 3]
trialList[5,:]  = [5, 16, 9]
trialList[6,:]  = [9, 16, 5]
trialList[7,:]  = [3, 21, 9]
trialList[8,:]  = [9, 21, 3]
trialList[9,:]  = [23, 20, 24]
trialList[10,:] = [24, 20, 23]
trialList = trialList + 63
obstrialList = trialList[1:, :]
obstrialList_types = dict(hard= [6, 7, 8, 9], easy = [0, 1, 2, 3, 4, 5], all = range(10))

cotrialList = np.arange(64, 73)
centerPos = np.array([ 0.0377292,  0.1383867])
targetPos = np.array([[ 0.1027292 ,  0.1383867 ],
			         [ 0.08369114,  0.18434864],
			         [ 0.0377292 ,  0.2033867 ],
			         [-0.00823274,  0.18434864],
			         [-0.0272708 ,  0.1383867 ],
			         [-0.00823274,  0.09242476],
			         [ 0.0377292 ,  0.0733867 ],
			         [ 0.08369114,  0.09242476]])
reps = 100
ix = []
for i in range(reps):
	ix.extend(np.random.permutation(len(targetPos)))
cotargetgen = targetPos[ix, :]
cotargetgen = np.hstack((cotargetgen[:, 0][:, np.newaxis], np.zeros((len(cotargetgen), 1)), cotargetgen[:, 1][:, np.newaxis]))


# Check each mat file and make sure it's a BMI file by plotting
# the BMI channels vs. MC channels 
def check_bmi(mat_list=filelist):
	for i, fn in enumerate(mat_list):
		dat = sio.loadmat(fn)

		f, ax = plt.subplots(nrows=3)
		ax[0].plot(dat['AD33'])
		ax[0].set_title('AD33 - xpos')

		ax[1].plot(dat['AD34'])
		ax[1].set_title('AD34 - ypos')

		ax[2].plot(dat['AD39'], dat['AD40'])
		ax[2].set_title('BMI channels')
		ax[2].set_xlabel(fn)

def targ_ix_to_loc(targ_ix):
	''' From '/backup2/jeev/ppf_data/jeev_spk/adaptive_ppf_new/jeev_center_out_bmi_targets_post012813.mat' '''
	# centerPos = np.array([ 0.0377292,  0.1383867])
	# targetPos = np.array([[ 0.1027292 ,  0.1383867 ],
	# 			         [ 0.08369114,  0.18434864],
	# 			         [ 0.0377292 ,  0.2033867 ],
	# 			         [-0.00823274,  0.18434864],
	# 			         [-0.0272708 ,  0.1383867 ],
	# 			         [-0.00823274,  0.09242476],
	# 			         [ 0.0377292 ,  0.0733867 ],
	# 			         [ 0.08369114,  0.09242476]])

	tix = []
	for it, tg in enumerate(targ_ix):
		tix.append(targetPos[tg, :])
	return np.vstack((tix))

def targ_ix_to_loc_obs(targ_ix):
	# try:
	# 	jdat = sio.loadmat('/home/lab/preeya/fa_analysis/resim_ppf/jeev_obs_positions_from_amy.mat')
	# except:
	# 	jdat = sio.loadmat('/Users/preeyakhanna/fa_analysis/resim_ppf/jeev_obs_positions_from_amy.mat')
	
	#### Within same directory ####
	jdat = sio.loadmat(analysis_config.config['BMI_DYN'] + 'resim_ppf/jeev_obs_positions_from_amy.mat')
	tix = []
	for it, tg in enumerate(targ_ix):
		final_targ_num = jdat['trialList'][tg, -1] - 1 #Python indexing
		tix.append(jdat['targPos'][final_targ_num, :])
	return np.vstack((tix))

def obs_target_gen(trial_types='all'):
	# Already in CM	
	# try:
	# 	jdat = sio.loadmat('/home/lab/preeya/fa_analysis/resim_ppf/jeev_obs_positions_from_amy.mat')
	# except:
	# 	jdat = sio.loadmat('/Users/preeyakhanna/fa_analysis/resim_ppf/jeev_obs_positions_from_amy.mat')
	
	#### Within same directory ####
	jdat = sio.loadmat(analysis_config.config['BMI_DYN'] + 'resim_ppf/jeev_obs_positions_from_amy.mat')
	hard = [6, 7, 8, 9]
	easy = [0, 1, 2, 3, 4, 5]

	if trial_types == 'all':
		tgs = range(10)
	elif trial_types == 'easy':
		tgs = easy
	elif trial_types == 'hard':
		tgs = hard

	reps = 100
	ix = []
	for i in range(reps):
		ix.extend(np.random.permutation(tgs))

	# Now take indices and add start, final, obstacle, major rad, minor rad, orient
	targs = []
	for i in ix:
		trl = jdat['trialList'][i, :] - 1 #Python indexing
		tmp = [jdat['targPos'][trl[0]], jdat['targPos'][trl[-1]], jdat['targPos'][trl[1]], jdat['targRad_maj'][trl[1]], jdat['targRad_min'][trl[1]], jdat['targOrient'][trl[1]]]
		targs.append(tmp)
	return targs

def confirm_channels(dat):
	# Lowpass filter pos data, try to make velocity data:
	# X / Y VEL: AD 37, 38
	# X / Y POS: AD 39, 40

	d = np.squeeze(dat['AD40']) #Positions
	b, a = sig.butter(6, 10/500., btype='lowpass')
	d2 = sig.filtfilt(b, a, d)

	d_d2 = np.diff(d2)/.001
	plt.plot(d_d2)
	plt.plot(dat['AD38']) # Velocities

	d = np.squeeze(dat['AD39']) #Positions
	b, a = sig.butter(6, 10/500., btype='lowpass')
	d2 = sig.filtfilt(b, a, d)

	d_d2 = np.diff(d2)/.001
	plt.plot(d_d2)
	plt.plot(dat['AD37']) # Velocities

def check_if_strobed_events_are_on_5ms(dat):
	events_based_on_cursor = [9]
	ts = []
	for i, ie in enumerate(events_based_on_cursor):
		ix = np.nonzero(dat['Strobed'][:, 1] == ie)[0]
		ts.append(dat['Strobed'][ix, 0])
	TS = np.hstack((ts))
	ts0 = np.min(TS)
	TS = TS - ts0

def unit_list(dat):
	units = []
	for k in dat.keys():
		if k[:3] == 'sig' and k[-2:] not in ['wf', 'ts']:
			units.append(k)
	return units

def sim_all(filelist=filelist, decoderlist=decoderlist):
	for i, (fn, dn) in enumerate(zip(filelist, decoderlist)):
		if i > -1:
			RSP = ReSimPPF(dn, fn)
			try:
				RSP.init_run_decoder()
				cont_flag = 1
			except:
				print 'cant init decoder: ', fn
				cont_flag = 0
			if cont_flag:
				RSP.run_decoder()
				RSP.plot_dir = '/home/lab/preeya/fa_analysis/resim_ppf/vel_plots/'
				RSP.plot_pred_vel(save=True)
				gc.collect()
		else:
			print 'done: ', fn

def surajs_sim_with_task_data(filelist=task_filelist, save_neural_push_fn=None):
	directory = '/home/lab/preeya/jeev_data_tmp/'
	directory2 = '/Volumes/TimeMachineBackups/jeev2013/'
	
	R2 = []
	cnt = 0
	neural_push_dict = {}
	for i, fns in enumerate(filelist):
		#fname_decoder0 = directory + fns[0][0]
		#fname_decoder1 = directory + fns[1][0]
		for j, fn in enumerate(fns):
			for k, fname0 in enumerate(fn):
				fname = directory + fname0
				fname2 = directory2 + fname0
				
				if cnt == 2:
					start_ix = 55003
					n = 33800
				else:
					start_ix = 0
					n = None
			
				try:
					T, K = tfna.run_sim(data_fname=fname, decoder_fname=fname, n_iter2 = n, start_ix = start_ix)
				except:
					T, K = tfna.run_sim(data_fname=fname2, decoder_fname=fname2, n_iter2 = n, start_ix = start_ix)
				

				# f, ax = plt.subplots(nrows=2)
				# ax[0].plot(T.decoder_state[3, :])
				# ax[0].plot(K[2, :])
				# ax[1].plot(T.decoder_state[5, :])
				# ax[1].plot(K[3, :])
				# slp, intc, rv_x, pv, ste = scipy.stats.linregress(T.decoder_state[3, :], K[2, :])
				# slp, intc, rv_y, pv, ste = scipy.stats.linregress(T.decoder_state[5, :], K[3, :])
				# R2.append([rv_x**2, rv_y**2])
				
				if save_neural_push_fn is not None:
					neural_push_dict[filelist[i][j][k]] = T.neural_push
				cnt += 1
	if save_neural_push_fn is not None:
		pickle.dump(neural_push_dict, open(save_neural_push_fn, 'w'))
	return R2

class ReSimPPF(object):
	def __init__(self, decoder_mat_file, dat_file):
		self.decoder = sio.loadmat(decoder_mat_file)
		self.unit_names = [self.decoder['decoder'][0]['predSig'][0][0][i][0] for 
			i in range(len(self.decoder['decoder'][0]['predSig'][0][0]))]
		self.n_units = len(self.unit_names)
		self.dat = sio.loadmat(dat_file)
		self.T = 0.005 # 5 ms loops
		self.Fs = 1000
		try:
			self.n_iter = int(np.ceil(self.dat['AD39'].shape[0]/(self.T*self.Fs)))+1
		except:
			print 'pass on n_iter'

		# TODO: beta is 3 x n_units. First is offset? 
		self.beta = self.decoder['decoder'][0]['beta'][0]
		self.beta_sub = self.beta[1:, :]

		# Save Filename: 
		ix = dat_file.find('/', -22, -1)
		self.fname = dat_file[ix+1:-4]

	def init_run_decoder(self):
		## Initialize Decoder Params ##: 
		self.pred_vel = np.zeros((self.n_iter, 2))
		self.pred_P = np.zeros((self.n_iter, 2, 2))
		
		self.bin_edges = np.arange(0.0, (self.n_iter+2)*self.T, self.T)
		self.spk_cnts = self.bin(self.dat, self.unit_names, self.bin_edges)
		self.P = np.mat(np.zeros((2, 2)))

		## Initialize SSM ##
		self.a_ppf = 0.8**(.05) # see suraj's thesis appendix for details
		w_kf = .0007; #Based on Amy's data? In BMI3D it's '7' for cm, so .07 in m
		N = 1./(1/20.);
		self.w_ppf = w_kf/((1-self.a_ppf**N)/(1-self.a_ppf) - 1)
		self.A = np.diag(np.array([self.a_ppf, self.a_ppf]))
		self.W = np.diag(np.array([self.w_ppf, self.w_ppf]))

	def bin(self, dat, unit_names, bin_edges):
		''' function to bin spike counts -- returns time x units'''
		spk_cnts = np.zeros((len(bin_edges), len(self.unit_names)))		
		for iu, un in enumerate(unit_names):
			dt = np.squeeze(dat[un])
			dt_dig = np.digitize(dt, bin_edges)

			for i, d in enumerate(dt_dig):
				spk_cnts[d, iu] += 1
		return spk_cnts

	def run_decoder(self):
		# Based off of Maryam's 'adaptive_ppf_new/offline_ppf_loop.m'
		 
		for n in range(1, self.n_iter):
			x_est, P_est = self._cycle(n)
			self.pred_vel[n, :] = np.squeeze(np.array(x_est))
			self.pred_P[n, :, :] = np.squeeze(np.array(P_est))

	def _cycle(self, n):
		# Based off of Maryam's 'adaptive_ppf_new/PPF_fixed_beta.m'
		x_prev = np.mat(self.pred_vel[n-1, :]).T
		x_pred = self.A*x_prev
		P_pred = self.A*self.P*self.A.T + self.W;

		#Keep track of blown up units ? 
		i_d = np.zeros(self.n_units)
		x_pred_stoch = np.vstack((np.mat(1),  x_pred))
		Loglambda_predict = x_pred_stoch.T * self.beta
		lambda_predict = np.exp(Loglambda_predict)/self.T

		invalid_pred_prob = np.nonzero((lambda_predict*self.T) > 1)[0]
		if len(invalid_pred_prob) > 0:
			lambda_predict[invalid_pred_prob] = 1./self.T
			i_d[invalid_pred_prob] = 1

		L = np.mat(np.diag(np.squeeze(np.array(lambda_predict*self.T))))

		if np.linalg.cond(P_pred) > 1e5:
			P_est = P_pred;
			print 'not updating P'
		else:
			P_est = np.linalg.inv(np.linalg.inv(P_pred) + self.beta_sub*L*self.beta_sub.T)

		unpred_spikes = self.spk_cnts[n, :] - lambda_predict*self.T
		x_est = x_pred + P_est*self.beta_sub*unpred_spikes.T

		#Bound the velocity: none, just position
		return x_est, P_est

	def plot_pred_vel(self, save=False):
		''' after decoding plot pred velocities'''
		xvel = self.pred_vel[:, 0]
		yvel = self.pred_vel[:, 1]

		xvel_5 = np.tile(xvel[:, np.newaxis], [1, 5]).reshape(-1)
		yvel_5 = np.tile(yvel[:, np.newaxis], [1, 5]).reshape(-1)

		N = np.min([len(xvel_5), self.dat['AD37'].shape[0], 300000])
		
		s, i, rv, pv, ste = scipy.stats.linregress(xvel_5[:N]*100, self.dat['AD37'][:N, 0])
		s2, i2, rv2, pv2, ste2 = scipy.stats.linregress(yvel_5[:N]*100, self.dat['AD38'][:N, 0])

		print self.fname
		print 'xvel match: ', rv**2
		print 'yvel match: ', rv2**2

		f, ax = plt.subplots(nrows=2)
		ax[0].plot(xvel_5[:N]*100) # m to cm
		ax[0].plot(self.dat['AD37'][:N, 0])
		ax[0].set_title('xvel match: '+str(rv**2))

		ax[1].plot(yvel_5[:N]*100) #m to cm
		ax[1].plot(self.dat['AD38'][:N, 0])
		ax[1].set_title('yvel match: '+str(rv2**2))

		if save:
			plt.savefig(self.plot_dir+self.fname)

class TaskFileSim(ReSimPPF):
	def __init__(self, decoder_mat_file, dat_file):
		super(TaskFileSim, self).__init__(decoder_mat_file, dat_file)

	def init_run_decoder(self):
		self.spk_cnts = self.dat['spike_counts'].T # Time x Units: 
		self.n_iter = self.spk_cnts.shape[0]
		self.pred_vel = np.zeros((self.n_iter, 2))
		self.pred_P = np.zeros((self.n_iter, 2, 2))		
		self.A = self.dat['A'][-2:, -2:] #Only velocity components
		self.W = self.dat['W']
		self.P = np.mat(np.zeros((2, 2)))

def make_sskg_ppf(filelist=task_filelist, save_KG_fn = None):
	cnt = 0

	for i, fns in enumerate(filelist):
		T_all = {}
		for k, fname0 in enumerate(fns):
			for fname in fname0:
				if cnt == 2:
					start_ix = 55003
				else:
					start_ix = 0

				n_iter2 = 'max';
				fname_full = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/' + fname
				T, K = tfna.run_sim(data_fname=fname_full, decoder_fname=fname_full, start_ix = start_ix, n_iter2=n_iter2)

				KGs = []
				#N = T.P_est.shape[2] 
				N = T.idx
				for j in range(N):
					# Test neural push: 
					neural_push = T.neural_push[:, j]
					sc = T.spike_counts[:, j]
					
					### This is from the PPF filter -- x_est = x_pred + np.dot(P, C)
					KG_est = np.dot(T.P_est[:, :, j], T.decoder.filt.C.T)
					assert np.allclose(neural_push*100, np.dot(KG_est, sc))
					KGs.append(np.dot(T.P_est[:, :, j], T.decoder.filt.C.T))
					T_all[fname] = KGs
				cnt += 1
		
		pickle.dump(T_all, open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/jeev_KG_approx_feb_2019_day'+str(i)+'.pkl', 'wb'))

def get_sskf(P_est_array, CT, tol = 1e-15):
	K_last = np.ones_like(P_est_array[:, :, 0]*CT)
	K = P_est_array[:, :, 0]*CT
	cnt = 1
	err = []
	while np.linalg.norm(K-K_last) > tol:
		if cnt >= P_est_array.shape[2]:
			print 'did not converge: ', np.linalg.norm(K-K_last)
			break
		else:
			K_last = K.copy()
			K = P_est_array[:, :, cnt]*CT
			err.append(np.linalg.norm(K_last-K))
			cnt += 1
	return K, err




















		


