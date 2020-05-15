from __future__ import division
from online_analysis import co_obs_tuning_matrices, subspace_overlap, behav_neural_PSTH
from online_analysis import test_fa_ov_vs_u_cov_ov as tfo
import analysis_config
import prelim_analysis as pa
from resim_ppf import ppf_pa 
from resim_ppf import file_key as fk
import scipy
import scipy.signal

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from pybasicbayes.util.text import progprint_xrange
from pylds.models import DefaultLDS
from pylds.states import kalman_filter
from sklearn.linear_model import LinearRegression
import sklearn.decomposition as skdecomp
import pickle, tables
import fcns, os
import math
import scipy
import pickle

cmap_list = ['maroon', 'orangered', 'darkgoldenrod', 'olivedrab', 'teal', 'steelblue',
    'midnightblue', 'darkmagenta', 'black', 'blue',]

ax_to_targ = {}
ax_to_targ[0] = [1, 2]
ax_to_targ[1] = [0, 2]
ax_to_targ[2] = [0, 1]
ax_to_targ[3] = [0, 0]
ax_to_targ[4] = [1, 0]
ax_to_targ[5] = [2, 0]
ax_to_targ[6] = [2, 1]
ax_to_targ[7] = [2, 2]
marker = ['.', '*', 'o']

input_test = [analysis_config.data_params['grom_input_type'][0]]
input_test_j = [fk.task_filelist[0]]

from matplotlib import cm
import matplotlib as mpl

try:
    info_grom = pickle.load(open('/Volumes/TimeMachineBackups/grom2016/co_obs_neuron_importance_dict.pkl'))
    info_jeev = pickle.load(open('/Volumes/TimeMachineBackups/jeev2013/jeev_co_obs_neuron_importance_dict.pkl'))
except:
    info_grom = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/co_obs_neuron_importance_dict.pkl'))
    info_jeev = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/jeev_co_obs_neuron_importance_dict.pkl'))

info_now = info_grom[input_test[0][0][0]]

min_ = np.min(info_now)
max_ = np.max(info_now)
minj = np.min(info_jeev[input_test_j[0][0][0]])
maxj = np.max(info_jeev[input_test_j[0][0][0]])
mymap = mpl.colors.LinearSegmentedColormap.from_list('mycolors',['blue', 'red'], N = 50)

# Using contourf to provide my colorbar info, then clearing the figure
Z = [[0,0],[0,0]]
levels = np.linspace(min_,max_, 50)
levels_j = np.linspace(minj, maxj, 50)
#CS3 = plt.contourf(Z, levels_j, cmap=mymap)


def fit_LDS(bin_spk_train, bin_spk_test, n_dim_latent, nEMiters=30,
    return_model=False, seed_w_FA = False, mfr = None, stdfr = None, **kwargs):
    '''
    Summary: fits an LDS with EM and returns the model, predicted data, 
        smoothed data, log-likelihood (final)

    Input param: bin_spk_train: spks to train on
    Input param: bin_spk_test: spks to test on (can be same as above if needed)
    Input param: n_dim_latent: latent dimensionality
    Input param: nEMiters: How many EM iterations to do before stopping
    Input param: return_model: whether to return the model or not
    Input param: seed_w_FA: Start LDS A, W, C, Q matrices by fitting FA w/ same number of 
        latent dim (as in Kao et. al) and estimating A, W, Q from FA's U
    Input param: mfr --> mean firing rate, stdfr --> std firing rate to be used in R2 computation, 
    Input param: **kwargs:
        - seed_pred_x0_with_smoothed_x0 = use the smoothed prediction of x0 (inital state)
             to seed the forward prediction. 
        - get_ratio = get dynamics / innovations ratios
        - pre_go_bins = how many bins are not to be assessed as part of R2
        - fa_thresh = DEPRECATED
        - task_ix_trials = which trials go with which task (len(task_ix_trials) == len(bin_spk_test))
        - input_train / input_test --> inputs into decoder
        - eval_model --> use model to evaluate
        - include_pre_go_data --> return all data, including the pre-go period

    Output param: 
    '''
    eval_model = kwargs.pop('eval_model', None)
    include_pre_go_data = kwargs.pop('include_pre_go_data', False)

    eps = 10e-10
    seed_pred_x0_with_smoothed_x0 = kwargs.pop('seed_pred_x0_with_smoothed_x0', False)
    
    get_ratio = kwargs.pop('get_ratio', False)
    pre_go_bins = kwargs.pop('pre_go_bins', 0)

    if include_pre_go_data: 
        pre_go_bins_return = 0; 
    else:
        pre_go_bins_return = pre_go_bins

    D_obs = bin_spk_train[0].shape[1]
    D_latent = n_dim_latent
    print 'fitting n_dim: ', D_latent

    input_train = kwargs.pop('input_train', None)
    input_test = kwargs.pop('input_test', None)

    if input_train is not None:
        D_input = input_train[0].shape[1]
    else:
        D_input = 0

    ntrials = len(bin_spk_train)

    ##################
    # Initialize LDS #
    ##################
    if eval_model is None:
        model = DefaultLDS(D_obs, n_dim_latent, D_input)

        for nt in range(ntrials):
            if input_train is not None:
                model.add_data(bin_spk_train[nt], inputs=input_train[nt])
            else:
                model.add_data(bin_spk_train[nt])

        if seed_w_FA:
            FA = skdecomp.FactorAnalysis(n_components=n_dim_latent)
            dat = np.vstack((bin_spk_train))
            FA.fit(dat)
            x_hat = FA.transform(dat)

            # Do main shared variance to solve this issue: 
            A = np.mat(np.linalg.lstsq(x_hat[:-1, :], x_hat[1:, :])[0])
            err = x_hat[1:, :].T - A*x_hat[:-1, :].T
            err_obs = dat.T - np.mat(FA.components_).T*x_hat.T

            model.C = FA.components_.T
            model.A = A
            if n_dim_latent == 1:
                model.sigma_states = np.array([[np.cov(err)]])
                model.sigma_obs = np.array([[np.cov(err_obs)]])
            else:
                model.sigma_states = np.cov(err)
                model.sigma_obs = np.cov(err_obs)

        #############
        # Train LDS #
        ############# 
        def update(model):
            model.EM_step()
            return model.log_likelihood()
        try:
            lls = [update(model) for i in progprint_xrange(nEMiters)]
        except:
            print 'adding reg to cov matrices:'
            dd = np.diag(model.sigma_states).copy()
            ix = np.mean(dd[np.nonzero(dd)[0]])
            ix2 = np.nonzero(dd==0)[0]
            for i in ix2:
                model.sigma_states[i, i] = ix

            try:
                lls = [update(model) for i in progprint_xrange(nEMiters)]
            except:
                print 'adding reg to cov matrices again'
                model.sigma_states = np.eye(model.sigma_states.shape[0])
                for i in range(len(model.sigma_states)):
                    model.sigma_states[i, i] = ix
    else:
        model = eval_model
        lls = [None]

    #################
    # Predict Stuff #
    #################

    pred_data = []
    smooth_data = []
    smooth_state = []

    filt_state = []
    pred_state = []

    filt_wt = []
    pred_wt = []    
    dynamics_norm = []
    innov_norm = []
    
    data_real = []
    filt_sigma = []
    
    kg = []

    ntrials_test = len(bin_spk_test)

    for nt in range(ntrials_test):
        if input_test is None:
            model.add_data(bin_spk_test[nt])
        else:
            model.add_data(bin_spk_test[nt], inputs=input_test[nt])

        g = model.states_list.pop()
        
        # Smoothed (y_t | y0...yT)
        smoothed_trial = g.smooth()
        smooth_data.append(smoothed_trial[pre_go_bins_return:, :])
        smooth_state.append(g.smoothed_mus[pre_go_bins_return:, :])

        if seed_pred_x0_with_smoothed_x0:
        # Smoothed (x_t | x0...xT)
            x0 = g.smoothed_mus[0, :] # Time x ndim
            P0 = g.smoothed_sigmas[0, :, :]
        
        else:
        # Use standard
            x0 = g.mu_init
            P0 = g.sigma_init

        # Filtered states (x_t | y0...y_t)
        _, filtered_mus, filtered_sigmas = kalman_filter(
        x0, P0,
        g.A, g.B, g.sigma_states,
        g.C, g.D, g.sigma_obs,
        g.inputs, g.data)
        
        filt_state.append(filtered_mus[pre_go_bins_return:, :])
        filt_sigma.append(filtered_sigmas[pre_go_bins_return:, :, :])

        # Predicted state (one time step fwd from filtering)
        if pre_go_bins_return == 0:
            ps = np.mat(g.A)*np.vstack(( np.zeros((1, filtered_mus.shape[1])), filtered_mus[pre_go_bins_return:-1, :])).T
        else:
            ps = np.mat(g.A)*filtered_mus[pre_go_bins_return-1:-1, :].T
        pred_state.append(ps)

        # Predicted data: (y_t | y0..y_(t-1))
        pred_trial = g.C*ps #np.mat(g.A)*filtered_mus[:-1, :].T
        pred_data.append(pred_trial) #pred_trial[:, pre_go_bins-1:])

        data_real.append(g.data[pre_go_bins_return:, :])

        if get_ratio:
            if pre_go_bins_return == 0:
                # Norm of dynamics process: 
                fm1 = np.vstack(( filtered_mus[0, :], filtered_mus[:-1, :] )).T
                dynamics_norm.append(np.linalg.norm(np.array(np.mat(g.A)*fm1 - fm1), axis=0))
            else:
                # Select the bins before the bins you want to evaluate: 
                fm1 = np.vstack(( filtered_mus[pre_go_bins_return-1:-1, :] )).T

                # Propogate these forward and assess the norm difference: 
                dynamics_norm.append(np.linalg.norm(np.array(np.mat(g.A)*fm1 - fm1), axis=0))

            # Norm of innovations process:
            inn_, kg_list = get_innov_list_simple(g, filtered_sigmas, filtered_mus, pre_go_bins_return)
            inn_2, kg_list2 = get_innov_list(g, filtered_sigmas, filtered_mus, pre_go_bins_return)

            #import pdb; pdb.set_trace()

            innov_norm.append(inn_)
            kg.append(kg_list)

        else:
            dynamics_norm = None
            innov_norm = None

    ## Predict R2 of pred_data vs. data
    ix_trials = kwargs.pop('task_ix_trials', np.zeros((len(bin_spk_test))))
    ix_trials = np.array(ix_trials).astype(int)

    R2_smooth = []
    R2_pred = []
    R2_filt = []
    for i in range(3):
        ix0 = np.nonzero(ix_trials==i)[0]

        if len(ix0) > 0:
            # Full data w/o pre_go_bins
            D0 = np.vstack(([bs[pre_go_bins:, :] for ib, bs in enumerate(bin_spk_test) if ib in ix0]))    
            
            # Full data w/ pre_go_bins
            #D_real = np.vstack(([dm for ib, dm in enumerate(data_real) if ib in ix0]))
            
            D_smooth0 = np.vstack(([sd for ib, sd in enumerate(smooth_data) if ib in ix0]))
            D_pred0 = np.hstack(([pd for ib, pd in enumerate(pred_data) if ib in ix0])).T
            D_filt0 = np.vstack(([fd*np.mat(model.C.T) for i_f, fd in enumerate(filt_state) if i_f in ix0]))

            ### Non-normalized R2
            R2_smooth.append(get_R2(D0, D_smooth0))
            R2_pred.append(get_R2(D0, D_pred0))
            R2_filt.append(get_R2(D0, D_filt0))

            ### Normalized R2
            R2_smooth.append(get_R2(D0, D_smooth0, mfr, stdfr))
            R2_pred.append(get_R2(D0, D_pred0, mfr, stdfr))
            R2_filt.append(get_R2(D0, D_filt0, mfr, stdfr))

    if return_model:
        return R2_smooth, R2_pred, lls, dynamics_norm, innov_norm, model, pred_data, smooth_data, filt_state, pred_state, filt_sigma, kg, R2_filt, smooth_state
    else:
        return R2_smooth, R2_pred, lls[-1], dynamics_norm, innov_norm

def get_main_FA_model(FA, thresh):
    # Return 1) dimensionality, 2) 
    U = np.mat(FA.components_).T
    i = np.diag_indices(U.shape[0])
    Psi = np.mat(np.zeros((U.shape[0], U.shape[0])))
    Psi[i] = FA.noise_variance_
    A = U*U.T
    B = np.linalg.inv(U*U.T + Psi)
    sharL = A*B

    #Get main shared space: 
    u, s, v = np.linalg.svd(A)
    s_red = np.zeros_like(s)
    s_hd = np.zeros_like(s)

    ix = np.nonzero(np.cumsum(s**2)/float(np.sum(s**2))>thresh)[0]
    if len(ix) > 0:
        n_dim_main_shared = ix[0]+1
    else:
        n_dim_main_shared = len(s)
    if n_dim_main_shared < 2:
        n_dim_main_shared = 2
    print "main shared: n_dim: ", n_dim_main_shared#, np.cumsum(s**2)/float(np.sum(s**2))
    return n_dim_main_shared

def demean_bin_spk(bin_spk, mean=None):
    '''
    Summary: remove mean from binned spikes
    Input param: bin_spk: list, each entry is an (t x n_neurons) trial
    Input param: mean: mean to subtract, if any
    Output param: bin_spk_demean, same format as bin_spk but with mean removed
    '''

    if mean is None:
        mean = np.mean(np.vstack((bin_spk)), axis=0)

    bin_spk_demean = []
    for bs in bin_spk:
        bin_spk_demean.append(bs - mean[np.newaxis, :])
    return bin_spk_demean, mean 

def get_innov_list_simple(state, filtered_sigmas, filtered_mus, pre_go_bins_return):
    ### Simpler way to get innovations --> just subtract out the dynamics
    ### Take the state_t and subtract the dynamics norm: 
    if pre_go_bins_return == 0:
        fm1 = np.vstack(( filtered_mus[0, :], filtered_mus[:-1, :] )).T
        fm2 = np.vstack(( filtered_mus[:, :] )).T
    else:
        fm1 = np.vstack(( filtered_mus[pre_go_bins_return-1:-1] )).T
        fm2 = np.vstack(( filtered_mus[pre_go_bins_return:, :] )).T

    ### For each dynamics norm state, get the diff: 
    nT = fm1.shape[1]

    inn_list = []; 
    kg_list = []; 

    ### Iterate through states: 
    for n in range(nT):
        x_t = fm1[:, n]
        x_t1 = fm2[:, n]

        # Norm diff between these: 
        state_diff = np.linalg.norm(x_t1 - x_t)

        # Dyn diff: 
        dyn_diff = np.linalg.norm(np.dot(state.A, x_t) - x_t)

        # Inn diff is the delta between the two: 
        inn_diff = state_diff - dyn_diff; 

        inn_list.append(inn_diff)
        kg_list.append(None)

    return inn_list, kg_list

def get_innov_list(state, filtered_sigmas, filtered_mus, pre_go_bins):
    s = []
    kg = [np.nan]
    for fs in filtered_sigmas:
        fs_t1 = np.dot(np.dot(state.A, fs), state.A.T)+ state.sigma_states
        s = state.C*np.mat(fs_t1)*state.C.T + state.sigma_obs
        kg.append(np.mat(fs_t1)*state.C.T*np.linalg.inv(s))
    
    innov = []
    for ik, k in enumerate(kg):
        if np.logical_and(ik > 0, ik < len(kg) - 1):
            innov.append(np.linalg.norm(np.array(k*(np.mat(state.data[ik]).T - state.C*np.mat(state.A)*np.mat(filtered_mus[ik-1]).T))))

    # predict next state: 
    # state_t = filtered_mus[ik]
    # dyn = np.dot(state.A, filtered_mus[ik-1])
    # inn  =np.dot(k, state.data[ik].T - np.dot(np.dot(state.C, state.A), filtered_mus[ik-1]));

    innov_pre_go_rm = []
    kg_rm = []
    if pre_go_bins == 0: 
        pregb = 1; 
        innov_pre_go_rm.append(innov[0])
        kg_rm.append(kg[0])
    else:
        pregb = pre_go_bins
    
    for i in range(pregb-1, len(innov)):
        innov_pre_go_rm.append(innov[i])
        kg_rm.append(kg[i])

    return innov_pre_go_rm, kg_rm

def get_R2(D_actual, D_predicted, mfr = None, stdfr = None):
    '''
    Summary: get R2 between two multivariate data sets
    Input param: D_actual: data1, (nobs, nfeatures)
    Input param: D_predicted: data2, (nobs, nfeatures)
    Output param: R2 (nfeatures x 1)
    '''
    ix_nan, ix_ft = np.nonzero(D_predicted==np.isnan)
    ix_use = np.array([ i for i in range(D_actual.shape[0]) if i not in ix_nan])

    D_actual_sub = D_actual[ix_use, :]
    D_pred_sub = D_predicted[ix_use, :]

    if mfr is not None: 
        D_actual_sub = (D_actual_sub - mfr[np.newaxis, :]) / stdfr[np.newaxis, :]
        D_pred_sub = (D_pred_sub - mfr[np.newaxis, :]) / stdfr[np.newaxis, :]

    SS_res = np.sum((np.array(D_actual_sub)-np.array(D_pred_sub))**2, axis=0)
    zsc = (np.array(D_actual_sub) - np.squeeze(np.array(np.mean(D_actual_sub, axis=0)))[np.newaxis, :])
    SS_tot = np.sum(zsc**2, axis=0)
    SS_tot[np.abs(SS_tot)<= 1e-15] = 0
    
    # Prevent divide by zeros: If SS_tot is zero, then R2 ought to be zero
    ix = np.nonzero(SS_tot==0)[0]
    SS_tot[ix] = 1.
    SS_res[ix] = 1.

    R2 = 1 - (SS_res/SS_tot)
    return R2

def find_optimal_nstate(bin_spk, nstates_list=None, nAttempts=2, nSplits=2, 
    pre_go_bins=0, **kwargs):
    '''
    Summary: Find optimal number of hidden states using x validated LL
    by fitting on nSplits-1 / nSplits of the data, and testing on 1/ nSplits of the 
    data each nAttempts times (total of nAttempts*nSplits fits per number of hiden states)

    Input param: bin_spk: training bin_spk
    Input param: nstates_list: np.array of states to sweep (e.g. np.arange(1, 20))
    Input param: nAttempts: How many times to fit each subset of data
    Input param: nSplits: How many splits of the data to do
    Input param: pre_go: how many seconds before go cue are included (dont' use in R2 assessment)
    
    Output param: 
    '''

    if nstates_list is None:
        nstates_list = np.arange(1, 20, 8)

    print 'nstates list: ', nstates_list
    n0 = nstates_list[0]
    nneurons = bin_spk[0].shape[1]

    kwargs['pre_go_bins'] = pre_go_bins
    LL = np.zeros((len(nstates_list), nAttempts, nSplits))
    R2_smooth = np.zeros((nneurons, len(nstates_list), nAttempts, nSplits))
    R2_pred = np.zeros((nneurons, len(nstates_list), nAttempts, nSplits))
    dyn = {}
    inn = {}

    N = len(bin_spk)
    N_test = N
    ixs = get_ixs(N, nSplits)

    for st, nf in enumerate(nstates_list):
        for n in range(nAttempts): 
            for i, u in enumerate(ixs.keys()):
                test = ixs[u]
                train = list(set(range(N)).difference(set(test)))

                # Only test on actual trials, not the pre-go data
                bs_test = [bin_spk[j] for j in test]
                bs_train = [bin_spk[j] for j in train]
                
                R2_smooth[:, st, n, i], R2_pred[:, st, n, i], LL[st, n, i], dyn[st, n, i], inn[st, n, i] = fit_LDS(bs_train, bs_test, nf, **kwargs)
    
    return R2_smooth, R2_pred, LL, nstates_list, dyn, inn

def get_ixs(N, nSplits):
    stgs = 'abcdefghijk'
    X = dict()
    third = int(np.floor(N/float(nSplits)))
    rand = np.random.permutation(N)
    for i, k in enumerate(stgs[:nSplits]):
        X[k] = []
        for ii in range(i*third, (i+1)*third):
            X[k].append(rand[ii])
    return X

def compute_neuron_importance(input_type, animal):
    '''
    Summary: for each BMI neuron, compute how important it is defined as:
        norm ( variance during successful trials * [vel_x_KG, vel_y_KG] )
    Input param: input_type:
    Output param: 
    '''
    importance_dict = {}
    for i_d, day in enumerate(input_type):
        for i_t, tsk in enumerate(day):
            for _, te_num in enumerate(tsk):
                bin_spk, targ_ix, trial_ix_all, KG = pull_data(te_num, animal)
                BS = np.vstack((bin_spk))
                n_var = np.var(BS, axis=0)
                importance = np.linalg.norm(n_var*KG[[3, 5], :], axis=0)
                importance_dict[te_num] = importance
    if animal == 'grom':
        pickle.dump(importance_dict, open(co_obs_tuning_matrices.pref+'co_obs_neuron_importance_dict.pkl', 'wb'))
    elif animal == 'jeev':
        pickle.dump(importance_dict, open('/Volumes/TimeMachineBackups/jeev2013/jeev_co_obs_neuron_importance_dict.pkl', 'wb'))

def pull_data(te_num, animal, pre_go=0, binsize_ms=100, keep_units='all'):

    ''' here pre_go is in seconds '''

    if animal == 'grom':
        # Get SSKF for animal: 
        try:
            # if on arc: 
            te = dbfn.TaskEntry(te_num)
            hdf = te.hdf
            decoder = te.decoder
        
        except:
            # elif on preeyas MBP
            co_obs_dict = pickle.load(open(co_obs_tuning_matrices.pref+'co_obs_file_dict.pkl'))

            hdf = co_obs_dict[te_num, 'hdf']
            hdfix = hdf.rfind('/')
            hdf = tables.openFile(co_obs_tuning_matrices.pref+hdf[hdfix:])

            dec = co_obs_dict[te_num, 'dec']
            decix = dec.rfind('/')
            decoder = pickle.load(open(co_obs_tuning_matrices.pref+dec[decix:]))
            F, KG = decoder.filt.get_sskf()

        # Get trials: 
        drives_neurons_ix0 = 3
        key = 'spike_counts'
        
        rew_ix = np.array([t[1] for it, t in enumerate(hdf.root.task_msgs[:]) if t[0]=='reward'])
        if pre_go > 0:
            go_ix = np.array([hdf.root.task_msgs[it-3][1] for it, t in enumerate(hdf.root.task_msgs[:]) if scipy.logical_and(t[0] == 'reward', t[1] in rew_ix)])
            pre_go_bins = pre_go*60
            exclude_ix = np.array([i for i, r in enumerate(go_ix) if r < pre_go_bins])
            print 'exclude ix: ', exclude_ix

        # decoder_all is now KG*spks
        # decoder_all units (aka 'neural push') is in cm / sec
        bin_spk, targ_i_all, targ_ix, trial_ix_all, decoder_all = pa.extract_trials_all(hdf, rew_ix, neural_bins = binsize_ms,
            drives_neurons_ix0=drives_neurons_ix0, hdf_key=key, keep_trials_sep=True,
            reach_tm_is_hdf_cursor_pos=False, reach_tm_is_kg_vel=True, include_pre_go= pre_go, **dict(kalman_gain=KG))
        
        ### Extract full cursor state (pos, vel)
        _, _, _, _, cursor_state = pa.extract_trials_all(hdf, rew_ix, neural_bins = binsize_ms,
            drives_neurons_ix0=drives_neurons_ix0, hdf_key=key, keep_trials_sep=True,
            reach_tm_is_hdf_cursor_pos=False, reach_tm_is_hdf_cursor_state=True, 
            reach_tm_is_kg_vel=False, include_pre_go= pre_go, **dict(kalman_gain=KG))

        if os.path.exists('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/targ_locs.pkl'):
            pass
        else:
            target_loc = dict(targ_i_all = targ_i_all, targ_ix = targ_ix)
            pickle.dump(target_loc, open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/targ_locs.pkl', 'wb'))

    elif animal == 'jeev':

        # Which day are we: 
        for i_d in range(4):
            for i_t in range(2):
                if te_num == fk.task_filelist[i_d][i_t][0]:
                    day = i_d; 

        kg_approx_dict = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/jeev_KG_approx_feb_2019_day'+str(day)+'.pkl', 'rb'))

        bin_spk, targ_i_all, targ_ix, trial_ix_all, decoder_all, cursor_state, unbinned, exclude_ix = ppf_pa.get_jeev_trials_from_task_data(te_num, 
            binsize=binsize_ms/1000., pre_go=pre_go,  include_pos = True)
        KG = np.array(kg_approx_dict[te_num][0])

    if keep_units is 'all':
        bin_spk_nonz = bin_spk
    
    else:
        bin_spk_nonz = []
        for bs in bin_spk:
            bin_spk_nonz.append(bs[:, keep_units])

    if pre_go > 0:
        return bin_spk_nonz, targ_ix, trial_ix_all, KG, decoder_all, cursor_state, exclude_ix
    else:
        return bin_spk_nonz, targ_ix, trial_ix_all, KG, decoder_all, cursor_state

def simult_PSTH_vs_single_trial_sweep_nstates(input_type, animal='grom', nAttempts=1, nSplits=5, 
    nstates_list=np.arange(10), pre_go=0, binsize_ms=100, **kwargs):

    pre_go_bins = int(np.round(pre_go*(1000./(binsize_ms))))
    colors = kwargs.pop('colors', ['b', 'g'])
    nms = kwargs.pop('nms', ['Spks','PSTH'])
    plot = kwargs.pop('plot', True)

    if plot:
        ncols = int(np.ceil(len(input_type)/3.))
        f, ax = plt.subplots(nrows = 3, ncols = ncols)
        f2, ax2 = plt.subplots(nrows = 3, ncols = ncols)
        f3, ax3 = plt.subplots(nrows = 3, ncols = ncols)

        if ncols == 1:
            ax = ax[:, np.newaxis]
            ax2 = ax2[:, np.newaxis]
            ax3 = ax3[:, np.newaxis] 
    
    r2_dict = {}

    for i_d, day in enumerate(input_type):  
        BS = []
        TK = []
        TG = []
        EX = []
        
        if plot:
            axi = ax[int(i_d%3), int(i_d/3)]
            axi.set_title('R2 smooth, Day: '+str(i_d))
            
            axi2 = ax2[int(i_d%3), int(i_d/3)]
            axi2.set_title('R2 pred, Day: '+str(i_d))

            axi3 = ax3[int(i_d%3), int(i_d/3)]
            axi3.set_title('LL, Day: '+str(i_d))     

        te_list_tmp = np.hstack((day))
        keep_units = subspace_overlap.get_keep_units(te_list_tmp, 10000, animal, binsize_ms)
   
        for i_t, tsk in enumerate(day):

            for t, te_num in enumerate(tsk):
                if pre_go == 0:
                    bs, tg, trl, _ = pull_data(te_num, animal, pre_go=pre_go, binsize_ms=binsize_ms)
                    exclude = []
                else:
                    bs, tg, trl, _, exclude = pull_data(te_num, animal, pre_go=pre_go, binsize_ms=binsize_ms)
                

                BS.extend(bs[i][:, keep_units] for i in range(len(bs)) if i not in exclude)
                TK.extend(np.zeros((len(bs) - len(exclude)))  + i_t)
                trl_ix = [i for i, j in enumerate(trl) if np.logical_and(i not in exclude, np.logical_and(j!= trl[i-1], i!= 0))]
                if 0 not in exclude:
                    trl_ix2 = [0]
                else:
                    trl_ix2 = []
                trl_ix2.extend(trl_ix)
                TG.extend(tg[trl_ix2])
                EX.append(exclude)

        PSTHs, TSK_TGs = PSTHify(BS, np.hstack((TK)), np.hstack((TG)))
        
        for ts, type_signal in enumerate(nms):

            if type_signal == 'PSTH':
                bin_spk_train = PSTHs
            elif type_signal == 'Spks':
                bin_spk_train = BS

            R2_smooth, R2_pred, LL, nstates, dynamics_norm, innov_norm = find_optimal_nstate(bin_spk_train, nstates_list=nstates_list,
                nAttempts=nAttempts, nSplits=nSplits, pre_go_bins=pre_go_bins, **kwargs)

            r2_dict[i_d, type_signal, 'smooth'] = R2_smooth
            r2_dict[i_d, type_signal, 'pred'] = R2_pred
            r2_dict[i_d, type_signal, 'LL'] = LL
            r2_dict[i_d, type_signal, 'nstates'] = nstates
            r2_dict[i_d, type_signal, 'nunits'] = bin_spk_train[0].shape[1]
            r2_dict[i_d, type_signal, 'dyn_norm'] = dynamics_norm
            r2_dict[i_d, type_signal, 'innov_norm'] = innov_norm

            if plot:
                R2_flat_smooth = R2_smooth.reshape(len(nstates), nAttempts*nSplits)
                R2_flat_predict = R2_pred.reshape(len(nstates), nAttempts*nSplits)
                LL_flat = LL.reshape(len(nstates), nAttempts*nSplits)
            
                fcns.plot_mean_and_sem(nstates, R2_flat_smooth, axi, color=colors[ts], label=nms[ts])
                fcns.plot_mean_and_sem(nstates, R2_flat_predict, axi2, color=colors[ts], label=nms[ts])
                fcns.plot_mean_and_sem(nstates, LL_flat, axi3, color=colors[ts], label=nms[ts])
                for axiii in [axi, axi2]:
                    axiii.set_ylim([0., 1.])
    return r2_dict

def PSTHify(BS, TK, TG, exclude_ix=None, nbins=10):
    if nbins == 'max':
        fit_bins = True
    PSTH = []
    tsk_tg = []
    for tsk in range(2):
        for tg in range(10):
            if exclude_ix is None:
                trl_ix = np.nonzero(np.logical_and(TK == tsk, TG == tg))[0]
            else:
                trl_ix = np.hstack(([i for i, (t_k, t_g) in enumerate(zip(TK, TG)) if np.logical_and(np.logical_and(t_k == tsk, t_g==tsk), i not in exclude_ix)]))

            if len(trl_ix) > 0:
                if fit_bins:
                    mx = 100
                    for ii, i in enumerate(trl_ix):
                        mx = np.min([mx, BS[i].shape[0]])
                    print ' max bins, tg ', tg, mx
                    nbins = mx
                bs = [BS[i][np.newaxis, :nbins, :] for ii, i in enumerate(trl_ix) if BS[i].shape[0] >= nbins]
                bs_mean = np.mean(np.vstack((bs)), axis=0)
                PSTH.append(bs_mean)
                tsk_tg.append([tsk, tg])
    return PSTH, tsk_tg

def plot_eigs_from_LDS():
    n_states = 15; 
    pref = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/'
    ndays = dict(grom=9, jeev=4)

    f, ax = plt.subplots(ncols = 2, figsize = (8, 4))
    fh, axh = plt.subplots(ncols = 2, figsize = (8, 4))

    hist_bins = np.arange(0., 1.6, .1)
    hist_bins_plt = hist_bins[:-1] + .05; 

    for ia, animal in enumerate(['grom', 'jeev']):
        fn = pref + animal + 'LDSmodels_nstates_'+str(n_states)+'_combined_models_w_dyn_inno_norms.pkl'
        dat = pickle.load(open(fn, 'rb'))

        hist_ = np.zeros_like(hist_bins)

        for i_d in range(ndays[animal]):
            model = dat[i_d]
            A = model.A; 
            ev, _ = np.linalg.eig(A)

            angs = np.array([ np.arctan2(np.imag(ev[i]), np.real(ev[i])) for i in range(len(ev))])
            hz = (angs)/(2*np.pi*.1)
            tau = -1/np.log(np.abs(ev))*.1
            ax[ia].plot(tau, hz, '.', color=cmap_list[i_d], label='Day'+str(i_d))
        
            ### histogram: 
            dig_hist = np.digitize(tau, hist_bins)
            dig_hist = dig_hist - 1; 

            for d in dig_hist:
                hist_[d] += 1; 

        ax[ia].legend(fontsize=10)
        ax[ia].set_title('Monk '+animal[0], fontsize=10)
        ax[ia].set_xlabel('Tau: -1/ln(|eigvalue|) * dt', fontsize=10)
        ax[ia].set_ylabel('Hz: angle(eigvalue) / (2*pi*dt)', fontsize=10)

        ax[ia].set_xlim([0., 1.5])
        ax[ia].set_ylim([-5.5, 5.5])

        axh[ia].bar(hist_bins_plt, hist_[:-1]/ float(np.sum(hist_)), width=.1, color='lightgray')
        axh[ia].bar(hist_bins_plt[-1] + .1, hist_[-1] / float(np.sum(hist_)), width=.1,color='lightgray')
        axh[ia].set_xlabel('Eigenvalue Tau (secs)')
        axh[ia].set_ylabel('Proportion of Eigs')
        axh[ia].set_xticks(np.arange(0., 1.6, .2))
        xlab = ['%.1f' %a for a in np.arange(0., 1.6, .2)]
        axh[ia].set_xticklabels(xlab, fontsize=10)
    
    f.tight_layout()
    fh.tight_layout()

### Method to fit combo CO / OBS model
def fit_LDS_CO_Obs(input_type, animal='grom', separate_models = False, 
    save_model = True, pre_go_secs = 1., **kwargs):

    pre_go_bins = int(pre_go_secs/.1)

    ### Updated 7-27-19 --> using this for computing and saving LDS models fit on all day from a day
    ### plus neural push outputs plus target info. 
    ### updated 9-12-19 --> also add cursor state: 
    print('')
    print('')
    print("This does not have mean subtraction implemetned for separate models. Does for combo models")
    print('')
    print('')

    pref = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/'

    #n_states_all = kwargs.pop('n_states_all', [5, 10, 15, 20])
    ### Update here -- nstates = 15
    n_states_all = [15]

    kwargs['get_ratio'] = True

    for n_states in n_states_all:
        # Begin by accruing data:
        models = {}
        #input_type = [[[4377],[4378, 4382]]]

        for i_d, day in enumerate(input_type):  
            if not separate_models:
                ### Binned spike counts
                BS1 = []; BS_for_mn = []; 

                ### Not sure what this is for...
                #BS_m_1 = []

                ## Task Label
                TK = []

                #TK_m_1 = []

                ## Target 
                TG = []

                ### 
                #tk = []

                ### Also track the neural push: 
                NP = []; 

                ### Also track teh cursor state: 
                ST = []; 

                ### And the decoder KalmanGain
                KG = []; 

            for i_t, tsk in enumerate(day):
                if separate_models:
                    BS1 = []; BS_for_mn = [];
                    TK = []
                    #TK_m_1 = []
                    TG = []; 
                    NP = []; 
                    KG = []; 

                for t, te_num in enumerate(tsk):

                    ### Get the data
                    ### Pre-go is in SECONDS
                    bs, tg, trl, kalm_gain, decoder_all, cursor_state, exclude_ix = pull_data(te_num, animal, pre_go = pre_go_secs)
                    
                    ### Add the binned spike counts: 
                    for i_b, b in enumerate(bs):
                        if i_b not in exclude_ix:
                            BS1.append(b); 
                            BS_for_mn.append(b);
                        else:
                            print('Excluding index %d, task %d, day %d' %(i_b, i_t, i_d))

                    TK.extend(np.zeros((len(bs) - len(exclude_ix))) + i_t)

                    trl_ix = [i for i, j in enumerate(trl) if np.logical_and(j!= trl[i-1], i!= 0)]
                    trl_ix2 = [0]
                    trl_ix2.extend(trl_ix)
                    trl_ix3 = [t for i, t in enumerate(trl_ix2) if i not in exclude_ix]
                    TG.extend(tg[trl_ix3])
                    ### Add the neural push and KG:
                    dec = [d for i, d in enumerate(decoder_all) if i not in exclude_ix]
                    st = [np.squeeze(s) for i, s in enumerate(cursor_state) if i not in exclude_ix]

                    NP.extend(dec)
                    KG.append(kalm_gain) 
                    ST.extend(st)

                if separate_models:
                    #R2, LL, model_BS, pred_state, pred_data_bs = fit_LDS(BS, BS, n_states, return_model=True, **kwargs)
                    R2_smooth, R2_pred, LL, _, _, model_BS, pred_data, _, filt_state, pred_state, filt_sigma, _, R2_filt, smooth_state = fit_LDS(BS, 
                        BS, n_states, return_model=True, **kwargs)

                    if save_model:
                        models[i_d, i_t] = model_BS

                    models[i_d, i_t, 'n'] = len(BS)
                    models[i_d, i_t, 'R2_smooth'] = R2_smooth
                    models[i_d, i_t, 'R2_pred'] = R2_pred
                    models[i_d, i_t, 'LL'] = LL

                    models[i_d, i_t, 'filt_state'] = filt_state
                    models[i_d, i_t, 'pred_state'] = pred_state
                    models[i_d, i_t, 'filt_sigma'] = filt_sigma
                    models[i_d, i_t, 'R2_filt'] = R2_filt
                    models[i_d, i_t, 'smooth_state'] = smooth_state

                    ### Save the params ###
                    models[i_d, i_t, 'binned_spks'] = BS; 
                    models[i_d, i_t, 'target'] = TG; 
                    models[i_d, i_t, 'neural_push'] = NP;
                    models[i_d, i_t, 'decoder_KG'] = KG; 
            
            ### Remove the mean across day: 
            BS_for_mnx = np.mean(np.vstack((BS_for_mn)), axis=0); 
            BS_for_stx = np.std(np.vstack((BS_for_mn)), axis=0); 

            ix_keep = np.nonzero(BS_for_mnx != 0.)[0]

            if len(ix_keep) < len(BS_for_mnx):
                print('getting rid of some zero units: total: %d, keep %d' %(len(BS_for_mnx), len(ix_keep)))

            BS = [b[:, ix_keep] - BS_for_mnx[np.newaxis, ix_keep] for b in BS1]
            #BS = [b for b in BS1]

            if not separate_models:
                kwargs['task_ix_trials'] = TK
                kwargs['seed_pred_x0_with_smoothed_x0'] = True; 
                kwargs['get_ratio'] = True; 
                #kwargs['pre_go_bins'] = pre_go_bins

                ### Fit a model with neural 
                #R2_smooth, R2_pred, LL, dynamics_norm, innov_norm, model, pred_data, smooth_data, filt_state, pred_state, filt_sigma, kg, R2_filt, smooth_state = fit_LDS(BS, BS, n_states, return_model=True, seed_w_FA=True, nEMiters=30, pre_go_bins = pre_go_bins, include_pre_go_data = True, mfr = BS_for_mnx, stdfr = BS_for_stx , **kwargs)

                R2_smooth, R2_pred, LL, dynamics_norm, innov_norm, model, pred_data, smooth_data, filt_state, pred_state, filt_sigma, kg, R2_filt, smooth_state = fit_LDS(
                    BS, BS, 
                    n_states, return_model=True, seed_w_FA=True, nEMiters=30, mfr = BS_for_mnx[ix_keep], stdfr = BS_for_stx[ix_keep],
                    **dict(seed_pred_x0_with_smoothed_x0= True, get_ratio=True, 
                        pre_go_bins=pre_go_bins, task_ix_trials=TK))

                if save_model:
                    models[i_d] = model
                    models[i_d, 'filt_state'] = filt_state
                    models[i_d, 'pred_state'] = pred_state
                    models[i_d, 'filt_sigma'] = filt_sigma
                    models[i_d, 'R2_filt'] = R2_filt
                    models[i_d, 'R2_smooth'] = R2_smooth
                    models[i_d, 'R2_pred'] = R2_pred
                    models[i_d, 'smooth_state'] = smooth_state

                    ### Save the params ###
                    models[i_d, 'binned_spks'] = BS; 
                    models[i_d, 'target'] = TG; 
                    models[i_d, 'neural_push'] = NP;
                    models[i_d, 'decoder_KG'] = KG; 
                    models[i_d, 'cursor_state'] = ST; 

                ### Fit a model with state as input
                R2_smooth, R2_pred, LL, dynamics_norm, innov_norm, model_BS, pred_data, smooth_data, filt_state, pred_state, filt_sigma, kg, R2_filt, smooth_state = fit_LDS(
                    BS, BS, 
                    n_states, return_model=True, seed_w_FA=True, nEMiters=30, mfr = BS_for_mnx[ix_keep], stdfr = BS_for_stx[ix_keep],
                    **dict(seed_pred_x0_with_smoothed_x0= True, get_ratio=True, 
                        pre_go_bins=pre_go_bins, task_ix_trials=TK, input_train = ST, input_test = ST))

                # R2_smooth, R2_pred, LL, _, _, model_BS, pred_data, _, filt_state, pred_state, filt_sigma, _, R2_filt, smooth_state = fit_LDS(BS,
                #     BS, n_states, return_model=True, pre_go_bins = pre_go_bins, include_pre_go_data = True, **cs_kwargs)                
                
                if save_model:
                    models[i_d, 'cs'] = model_BS
                    models[i_d, 'R2_filt_cs'] = R2_filt
                    models[i_d, 'R2_smooth_cs'] = R2_smooth
                    models[i_d, 'R2_pred_cs'] = R2_pred
                    models[i_d, 'filt_state_cs'] = filt_state
                    models[i_d, 'filt_sigma_cs'] = filt_sigma
                    
                    ### Add the emperical input covariance for use in later analyses. 
                    models[i_d, 'cs_input_cov'] = np.cov(np.vstack((ST)).T)
                    

                # models[i_d, 0, 'R2_smooth'] = R2_smooth[0]
                # models[i_d, 1, 'R2_smooth'] = R2_smooth[1]
                # models[i_d, 0, 'R2_pred'] = R2_pred[0]
                # models[i_d, 1, 'R2_pred'] = R2_pred[1]
                models[i_d, 'tsk'] = TK

        if separate_models:
            fn = pref + animal + 'LDSmodels_nstates_'+str(n_states)+'_separate_models_w_dyn_inno_norms.pkl'
        else:
            fn = pref + animal + 'LDSmodels_nstates_'+str(n_states)+'_combined_models_w_dyn_inno_norms.pkl'
        
        print('Done with %s' %fn)
        pickle.dump(models, open(fn, 'wb'))

def plot_features_LDS_separate_CO_OBS(nstates=[5, 10, 15, 20]):
    #########################
    #### Load Everything ####
    #########################

    models_all = {}
    for n_s in nstates:
        models_all[n_s] = pickle.load(open('/Volumes/TimeMachineBackups/grom2016/LDSmodels_nstates_'+str(n_s)+'.pkl'))
    
    f, ax = plt.subplots()
    sizes = [10, 20, 30, 40]
    marker = ['.', '*']

    for i_n, n_s in enumerate(nstates):
        models = models_all[n_s]

        # Plot N vs. R2, color = task: 
        for i_d in range(9):
            for i_t, tsk in enumerate(['co', 'obs']):
                n = models[(i_d, i_t, 'n')]
                r2 = models[(i_d, i_t, 'R2')]
                ax.plot(n+10*np.random.rand(), r2, marker[i_t], color=cmap_list[i_d], label='tsk: '+tsk, markersize=sizes[i_n])
    plt.title('Circle - CO, Stars - Obs, Size - nstates')
    plt.ylabel('R2')
    plt.xlabel('Ntrials')

    # Plot eigenvalues of A: 
    nst = 20
    models = models_all[nst]
    f2, ax2 = plt.subplots(nrows = 3, ncols = 3)
    f3, ax3 = plt.subplots(ncols = 2)

    real = {}
    mag = {}
    theta = {}
    for i_d in range(9):
        axi = ax2[int(i_d/3), int(i_d%3)]
        for i_t in range(2):
            mod = models[i_d, i_t]
            A = mod.A
            w, v = np.linalg.eig(A)
            axi = plot_eigs(w, axi, color=cmap_list[i_t*4])

            try:
                real[i_t].append(np.real(w))
                mag[i_t].append(np.abs(w))
                theta[i_t].append([math.atan2(y, x) for i, (x, y) in enumerate(zip(np.real(w), np.imag(w)))]) 
            except:
                print 'except: '
                real[i_t]=[np.real(w)]
                mag[i_t]=[np.abs(w)]
                theta[i_t]=[math.atan2(y, x) for i, (x, y) in enumerate(zip(np.real(w), np.imag(w)))]             
    
    magbins = np.arange(0, 1., 1/7.)
    angbins = np.arange(-np.pi, np.pi, np.pi/14.) 

    # Magnitude: 
    tsks = ['CO', 'OBS']
    for i_t in range(2):
        mag_ = np.hstack((mag[i_t]))
        ang_ = np.hstack((theta[i_t]))
        H, x, y= np.histogram2d(mag_, ang_, [magbins, angbins])
        im = ax3[i_t].pcolormesh(x, y, H.T/np.sum(H), vmin = 0.0, vmax = .1, cmap='jet')
        f3.colorbar(im, ax=ax3[i_t])
        ax3[i_t].set_xlabel('Magnitude')
        ax3[i_t].set_ylabel('Osc. Frequency')
        ax3[i_t].set_title('Task: '+tsks[i_t])
        ax3[i_t].axis('tight')
    plt.tight_layout()

def separate_states(fname):
    nsts = 20
    models = pickle.load(open('/Volumes/TimeMachineBackups/grom2016/LDSmodels_nstates_'+str(nsts)+'_separate_models_0.pkl'))

    for i_d in range(9): 
        ### Combined model: 
        mod = models[i_d]

        ### Get the A matrix: 
        A = mod.A

        ### Now get its eigenvalue decomp:
        w, T  = np.linalg.eig(A)
        sqrt_w = np.sqrt(w)
        inv_T = np.linalg.inv(T)
        ey = np.mat(inv_T)*T
        rl_sm, im_sm = np.real(np.sum(ey)), np.imag(np.sum(ey))

        assert np.abs(rl_sm - float(ey.shape[0])) < 1e-10
        assert np.abs(float(im_sm)) < 1e-10

        ### Get states 
        state_hat = models[i_d, 'pred_x']
        trg_ = models[i_d, 'targets']
        tsk_ = models[i_d, 'task']

        ### Now characterize the eigenvalues: 
        wix = char_eig(w, mag_thresh = 0.3, theta_thresh = np.pi/4.)
        wix_ = dict()
        wix_[0] = np.nonzero(np.array(wix)==0)[0]
        wix_[1] = np.nonzero(np.array(wix)==1)[0]
        wix_[2] = np.nonzero(np.array(wix)==2)[0]

        ### Now look at activation of each eigenmode: 
        # Now, do the substitution x_new = v^{-1}x_old
        state_hat_new = [np.array(np.mat(inv_T)*st.T).T for st in state_hat]

        # PSTHify: 
        PSTH_state_hat_new, _ = PSTHify(state_hat_new, np.array(tsk_).astype(int), np.array(trg_).astype(int))

        # Plot PSTH:
        f, ax = plt.subplots(nrows=3, ncols=3)
        co_ = np.mean(np.abs(np.vstack(([p for i, p in enumerate(PSTH_state_hat_new) if i < 8]))), axis=0)
        obs_ = np.mean(np.abs(np.vstack(([p for i, p in enumerate(PSTH_state_hat_new) if i >= 8]))), axis=0)
        tg_means = [co_, obs_]
        for j in range(2):
            for i in range(8):
                axi = ax[ax_to_targ[i][0], ax_to_targ[i][1]]
                for cat in range(3):
                    axi.plot(wix_[cat], np.abs(np.mean(PSTH_state_hat_new[i + j*8], axis=0))[wix_[cat]], marker[cat], color=cmap_list[j*4])
            axi = ax[1, 1]

            for cat in range(3):
                axi.plot(wix_[cat], tg_means[j][wix_[cat]], marker[cat], color=cmap_list[j*4])

        # Plot bar plots: 
        f, ax = plt.subplots()
        for i in range(8):
            for cat in range(3):
                axi.plot(wix_[cat], np.abs(np.mean(PSTH_state_hat_new[i + j*8], axis=0))[wix_[cat]], marker[cat], color=cmap_list[j*4])

            for cat in range(3):
                axi.plot(wix_[cat], tg_means[j][wix_[cat]], marker[cat], color=cmap_list[j*4])

def char_eig(w, mag_thresh = 0.2, theta_thresh = 0.2):
    ix = []
    for i, iw in enumerate(w):
        mg = np.abs(iw)
        th = np.abs(math.atan2(np.imag(iw), np.real(iw)))
        if mg > mag_thresh:
            if th < theta_thresh:
                ix.append(1)
            else:
                raise

        elif mg < mag_thresh:
            if th < theta_thresh:
                ix.append(0)
            else:
                ix.append(2)
    assert len(ix) == len(w)
    return ix

def plot_eigs(w, ax, color):
    for i in w:
        ax.plot([0, np.real(i)], [0, np.imag(i)], '.-', color=color)
    return ax
               
def kg_vel_to_traj(KG, spks):
    '''
    Summary: method to turn velocities (K*y_t) to trajectories
    Input param: KG: Kalman Gain (7 x nneurons)
    Input param: spks: list of spikes, spks[0].shape = T x nneurons
    Output param: list of trajectories
    '''
    vels = []
    for s in spks:
        vels.append(np.mat(KG)*s.T) # 7 x T

    traj = []
    for dat in vels:
        p = []
        pos = np.array([[0., 0.]]).T
        p.append(pos)
        for i in range(10):
            pos = pos + .1*(dat[[3, 5], i])
            p.append(pos.copy())
        traj.append(np.array(np.hstack((p)).T))
    return traj

def plot_separate_vs_single_models(input_type, animal, nstates=15):

    colmap = ['r', 'b']
    marker = ['^', 'o']
    if animal == 'grom':
        dat_sep = pickle.load(open('/Volumes/TimeMachineBackups/grom2016/LDSmodels_nstates_'+str(nstates)+'_separate_models_w_dyn_inno_norms.pkl'))
        dat_combo = pickle.load(open('/Volumes/TimeMachineBackups/grom2016/LDSmodels_nstates_'+str(nstates)+'_combined_models_w_dyn_inno_norms.pkl'))

    # Some dat_combo pre-processing: 
    for i_d, day in enumerate(input_type):
        ix = dat_combo[i_d, 'tsk'][0]
        ix0 = np.nonzero(np.array(ix).astype(int)==0)[0]
        ix1 = np.nonzero(np.array(ix).astype(int)==1)[0]

        dat_combo[i_d, 0, 'dynamics_norm'] = [dn for i_, dn in enumerate(dat_combo[i_d, 'dynamics_norm']) if i_ in ix0]
        dat_combo[i_d, 1, 'dynamics_norm'] = [dn for i_, dn in enumerate(dat_combo[i_d, 'dynamics_norm']) if i_ in ix1]
        dat_combo[i_d, 0, 'innov_norm'] = [dn for i_, dn in enumerate(dat_combo[i_d, 'innov_norm']) if i_ in ix0]
        dat_combo[i_d, 1, 'innov_norm'] = [dn for i_, dn in enumerate(dat_combo[i_d, 'innov_norm']) if i_ in ix1]

    metrics = ['R2_smooth', 'R2_pred']
    metrics_dict = {}
    for met in metrics:
        for i in range(2):
            metrics_dict[i, met] = []
            metrics_dict[i, 'rat'] = []

    fax = [plt.subplots() for i in range(len(metrics))]
    ax = [a[1] for a in fax]

    # Get data predictions: 
    for i_d, day in enumerate(input_type):  
        for i_t, tsk in enumerate(day):
            for i_m, met in enumerate(metrics):
                axi = ax[i_m]
                r2_comb = dat_combo[i_d, i_t, met]
                r2_sep = dat_sep[i_d, i_t, met]
                axi.plot(i_d+(.5*i_t), r2_comb, colmap[i_t]+'^')
                axi.plot(i_d+(.5*i_t)+.2, r2_sep, colmap[i_t]+'o')
                metrics_dict[i_t, met].append(r2_sep)

    for m, axi in enumerate(ax):
        axi.set_title('Red: CO, Blue: Obs, Triangle: Combined Models, Circle: Separate: '+metrics[m])
            
    # Dynamics vs. Innovation Process: 
    f, ax = plt.subplots()
    for i_d, day in enumerate(input_type):
        for i_t, tsk in enumerate(day):
            for i, dat in enumerate([dat_combo, dat_sep]):
                dyn = np.hstack(( dat[i_d, i_t, 'dynamics_norm'] ))
                innov = np.hstack(( dat[i_d, i_t, 'innov_norm'] ))
                rat = dyn / (dyn + innov)
                ax.errorbar(i_d+(.5*i_t)+(.2*i), np.mean(rat), fmt=marker[i]+'-',
                    yerr=np.std(rat)/np.sqrt(len(rat)), color=colmap[i_t])
                if i == 1:
                    metrics_dict[i_t, 'rat'].append(np.mean(rat))

    ax.set_ylabel('Ratio of Dynamics Proc / (Dyn + Innov)')
    ax.set_title('Red: CO, Blue: Obs, Triangle: Combined Models, Circle: Separate, Nstates='+str(nstates))
    return metrics_dict

def plot_ntrials_nunits_nbins_vs_R2(input_type, metrics, nstates=20):
    '''
    Summary: 
    Input param: input_type: analysis_config.data_params['grom_input_type']
    Input param: metrics: output from 'plot_separate_vs_single_models'
    Input param: nstates:
    Output param: plots of R2, etc. vs. # neurons, # trials, # bins / trial
    '''

    colmap = ['b', 'r']
    dat_sep = pickle.load(open('/Volumes/TimeMachineBackups/grom2016/LDSmodels_nstates_'+str(nstates)+'_separate_models_w_dyn_inno_norms.pkl'))
    for i_t in range(2):
        metrics[i_t, 'n'] = []
        metrics[i_t, 'neur'] = []
        metrics[i_t, 'avg_trial'] = []
        
    for i_d, day in enumerate(input_type):
        for i_t, tsk in enumerate(day):
            BS = []
            for t, te_num in enumerate(tsk):
                bs, tg, trl, _ = pull_data(te_num, 'grom')
                BS.extend(bs)
            metrics[i_t, 'n'].append(dat_sep[i_d, i_t, 'n'])
            metrics[i_t, 'neur'].append(bs[0].shape[1])
            metrics[i_t, 'avg_trial'].append(np.mean([len(bs) for bs in BS]))

    # Now plot metrics: 
    mets = ['n', 'neur', 'avg_trial']
    vs = ['rat', 'R2_pred', 'R2_smooth']
    f, ax = plt.subplots(nrows=3, ncols=3)

    for im, met in enumerate(mets):
        for iv, vss in enumerate(vs):
            axi = ax[im, iv]
            for i_t in range(2):
                axi.plot(metrics[i_t, met], metrics[i_t, vss], '.', color=colmap[i_t])
            axi.set_xlabel(met)
            axi.set_ylabel(vss)
    plt.tight_layout()

def to_run_overnight_v0():
    from online_analysis import fit_LDS
    from online_analysis import co_obs_tuning_matrices
    input_type = analysis_config.data_params['grom_input_type']

    try:
        r2_dict = fit_LDS.simult_PSTH_vs_single_trial_sweep_nstates(input_type, animal='grom',
            nAttempts=1, nSplits=5,nstates_list=np.arange(1, 20, 3))         
        fnm = '/Volumes/TimeMachineBackups/grom2016/grom_r2_sweep_nstates_demean_xval5_niterEM30.pkl'
        pickle.dump(r2_dict, open(fnm, 'wb'))
    except:
        print 'failed sweep states thing'


    kwargs = dict(n_states_all=[20])
    try:
        fit_LDS.fit_LDS_CO_Obs(analysis_config.data_params['grom_input_type'], separate_models=True, **kwargs)
    except:
        print 'failed separate models '

    try:
        fit_LDS.fit_LDS_CO_Obs(analysis_config.data_params['grom_input_type'], separate_models=False, **kwargs)
    except:
        print 'failed same models'

def to_run_overnight_6_7_17_day0_only():
    from online_analysis import fit_LDS
    from online_analysis import co_obs_tuning_matrices
    input_type = [analysis_config.data_params['grom_input_type'][0]]
    import pickle

    #########################
    ##### STATE SWEEP #######
    #########################

    # Run state sweep with fancy new additions: a) seed LDS w/ FA, b) pre-go = 1 sec, 
    # c) seed_x0_w_smooth_x0, d) return R2 for each neuron, then plot by importance
    kwargs = dict(seed_pred_x0_with_smoothed_x0=True, seed_w_FA=True, nms=['Spks'], plot=False, get_ratio=True)
    r2_dict = fit_LDS.simult_PSTH_vs_single_trial_sweep_nstates(input_type, animal='grom',
        nAttempts=1, nSplits=2, nstates_list=np.arange(2, 43, 4), pre_go=1., **kwargs)         
    fnm = '/Volumes/TimeMachineBackups/grom2016/grom_r2byfeature_sweep_nstates_demean_xval5_niterEM30_seedwFA_seedx0wsmoothx0_100msbins_day0.pkl'
    pickle.dump(r2_dict, open(fnm, 'wb'))
    
    ### Plot normal R2
    plt.clf()

    nstates = dat[0, 'Spks', 'nstates']
    nneurons, ns, natt, nit = dat[0, 'Spks', 'smooth'].shape
    f, ax = plt.subplots(nrows = 2)
    for i, j in enumerate(['smooth', 'pred']):
        axi = ax[i]
        d = dat[0, 'Spks', j]
        for n in range(nneurons):
            c = axi.plot(nstates, np.mean(d[n, :, 0, :],axis=1), '.-',color = cm.jet(info_now[n]))
        axi.set_xlabel('# States')
        axi.set_ylabel('R^2 per neuron, '+j)
        axi.set_title(j)
        plt.colorbar(CS3, ax = axi) # using the colorbar info I got from contourf
    plt.tight_layout()

    mn_std = {}
    f, ax = plt.subplots()
    for n in range(ns):
        dyn = dat[0, 'Spks', 'dyn_norm'][n, 0, 0]
        inn = dat[0, 'Spks', 'innov_norm'][n, 0, 0]
        for i in range(1, nit):
            dyn.extend(dat[0, 'Spks', 'dyn_norm'][n, 0, i])
            inn.extend(dat[0, 'Spks', 'innov_norm'][n, 0, i])

        mn_std[n, 'dyn'] = [np.mean(np.hstack((dyn))), np.std(np.hstack((dyn)))]
        mn_std[n, 'inn'] = [np.mean(np.hstack((inn))), np.std(np.hstack((inn)))]
        ax.bar(nstates[n], mn_std[n, 'dyn'][0], color='b', width=1)
        ax.errorbar(nstates[n]+.5, mn_std[n, 'dyn'][0], yerr=mn_std[n, 'dyn'][1], color='b')
        ax.bar(nstates[n]+1, mn_std[n, 'inn'][0], color='r', width=1)
        ax.errorbar(nstates[n]+1.5, mn_std[n, 'inn'][0], yerr=mn_std[n, 'dyn'][1], color='r')

    #########################
    ##### Bin SWEEP #######
    #########################

def overnight_bin_sweep(animal):
    kwargs = dict(seed_pred_x0_with_smoothed_x0=True, seed_w_FA=True, nms=['Spks'], plot=False, get_ratio=True)
    if animal == 'grom':
        binsweep = np.arange(2*1000/60., 1000*12/60., 2*1000/60.)
        direct = 'grom2016'
        input_type = analysis_config.data_params['grom_input_type']
    elif animal == 'jeev':
        binsweep = np.arange(.005, .15, .025)*1000.
        direct = 'jeev2013'
        input_type = fk.task_filelist

    master_r2_dict = {}
    nSplits = 2
    #nstates = [5, 10, 15, 20]
    nstates = [10]
    for bs in binsweep:
        if LDS_or_FA is 'LDS':
            master_r2_dict[bs] = simult_PSTH_vs_single_trial_sweep_nstates(input_type, animal=animal,
                nAttempts=1, nSplits=nSplits, nstates_list=nstates, pre_go=1., binsize_ms=bs, **kwargs)         
        elif LDS_or_FA is 'FA':
            master_r2_dict[bs] = simult_PSTH_vs_single_trial_sweep_nstates(input_type, animal=animal,
                nAttempts=1, nSplits=nSplits, nstates_list=nstates, pre_go=1., binsize_ms=bs, FA_only = True,
                **kwargs)              

    fnm = '/Volumes/TimeMachineBackups/'+direct+'/'+animal+'_r2byfeature_sweep_bins_nstates_many_nodemean_xval5_niterEM30_seedwFA_seedx0wsmoothx0_day0.pkl'
    pickle.dump(master_r2_dict, open(fnm, 'wb'))

def overnight_stats_sweep(animal, day_list = range(3), n_trials_base = 56, n_models = 3):
    # Dimensionality, main dimensionality, R2 for separate vs. together trial-matched models
    # 
    mod_nms = ['FA', 'LDS', 'LPF25', 'LPF100', 'LPF200']
    mod_types = ['_sep_fit', '_comb_fit']
    mets = ['R2', 'R2_pred', 'main_dim', 'dim']
    models = {}
    n_trials_base = int(n_trials_base)

    for m in mod_nms:
        for mt in mod_types:
            for met in mets:
                for i_d in day_list:
                    for i_t in range(3): # CO, OBS, Combo
                        models[i_d, i_t, m+mt, met] = []

    if animal == 'jeev':
        input_type = fk.task_filelist
    elif animal == 'grom':
        input_type = analysis_config.data_params['grom_input_type']

    for i_d, day in enumerate(input_type):       
        if i_d in day_list:
            # Get non-zero units; 
            d = np.hstack((day))
            keep_units = subspace_overlap.get_keep_units(d, 10000, animal, 100.)
            models[i_d, 'keep_units'] = keep_units
            day2 = [day[0], day[1], [0]]
            
            for i_t, tsk in enumerate(day2):
                
                if i_t in [0, 1]:
                    # List of within-task TEs
                    te_list = np.hstack((tsk))

                    # Pull data from non-zero units: 
                    bs = []
                    tg_ix = []
                    for te in te_list:
                        bin_spk_nonz, targ_ix, trial_ix_all, _, exclude_ix = pull_data(te, animal, 
                            pre_go=1., binsize_ms=100, keep_units=keep_units)
                        bs.extend([bin_spk_nonz[i] for i in range(len(bin_spk_nonz)) if i not in exclude_ix])
                        sub_trl_ix = [int(targ_ix[np.nonzero(trial_ix_all==i)[0][15]]) for i in range(len(bin_spk_nonz)) if i not in exclude_ix]
                        tg_ix.extend(sub_trl_ix)
                    models[i_d, i_t, 'bin_spks'] = bs
                    models[i_d, i_t, 'targ_ix'] = tg_ix

                    model_type = 'sep'
                    n_trials = n_trials_base

                elif i_t == 2:
                    bs = models[i_d, 0, 'bin_spks']
                    bs.extend(models[i_d, 1, 'bin_spks'])
                    tg_ix = models[i_d, 0, 'targ_ix']
                    tg_ix.extend(models[i_d, 1, 'targ_ix'])

                    model_type = 'comb'
                    n_trials = int(n_trials_base*2)

                for n in range(int(n_models)):
                    # FA Separate: 
                    R2_smooth, R2_pred, main_dim, num_factors = get_model_stats(bs, tg_ix, animal, n_trials, 'FA', 10)
                    models[i_d, i_t, 'FA_'+model_type+'_fit', 'R2'].append(R2_smooth)
                    models[i_d, i_t, 'FA_'+model_type+'_fit', 'main_dim'].append(main_dim)
                    models[i_d, i_t, 'FA_'+model_type+'_fit', 'dim'].append(num_factors)

                    #LDS Separate: 
                    R2_smooth, R2_pred, main_dim, num_factors = get_model_stats(bs, tg_ix, animal, n_trials, 'LDS', 10)
                    models[i_d, i_t, 'LDS_'+model_type+'_fit', 'R2'].append(R2_smooth)
                    models[i_d, i_t, 'LDS_'+model_type+'_fit', 'R2_pred'].append(R2_pred)
                    models[i_d, i_t, 'LDS_'+model_type+'_fit', 'main_dim'].append(main_dim)
                    models[i_d, i_t, 'LDS_'+model_type+'_fit', 'dim'].append(num_factors)

                    # Low pass filter window 500 MS: Y_t = aY_{t-1} + ... bY_{t-5}
                    R2_25_pred, R2_25_smooth = get_LPF(bs, tg_ix, animal, n_trials, 25., 10)
                    R2_100_pred, R2_100_smooth = get_LPF(bs, tg_ix, animal, n_trials, 100., 10)
                    R2_200_pred, R2_200_smooth = get_LPF(bs, tg_ix, animal, n_trials, 200., 10)

                    models[i_d, i_t, 'LPF25_'+model_type+'_fit', 'R2_pred'].append(R2_25_pred)
                    models[i_d, i_t, 'LPF25_'+model_type+'_fit', 'R2'].append(R2_25_smooth)
                    models[i_d, i_t, 'LPF100_'+model_type+'_fit', 'R2_pred'].append(R2_100_pred)
                    models[i_d, i_t, 'LPF100_'+model_type+'_fit', 'R2'].append(R2_100_smooth)
                    models[i_d, i_t, 'LPF200_'+model_type+'_fit', 'R2_pred'].append(R2_200_pred)
                    models[i_d, i_t, 'LPF200_'+model_type+'_fit', 'R2'].append(R2_200_smooth)
            pickle.dump(models, open('test_model_stats_'+animal+'_day'+str(i_d)+'.pkl', 'wb'))   
            send_email('Successfully saved '+animal+' day_'+str(i_d)+'!')     
    return models

def plot_stats(fname, animal, input_type):
    models = pickle.load(open(fname))
    color = ['r', 'g', 'lightgrey', 'grey', 'darkgray']
    hatchi = ['//', '', '*']
    mod_nms = ['FA', 'LDS', 'LPF25', 'LPF100', 'LPF200']
    mod_types = ['_sep_fit', '_comb_fit']
    mets = ['R2', 'R2_pred', 'main_dim', 'dim']

    if animal == 'grom':
        info = info_grom

    elif animal == 'jeev':
        info = info_jeev

    AX = []
    for a in range(4):
        f, ax = plt.subplots()
        AX.append(ax)

    for i_m, met in enumerate(mets):
        ax = AX[i_m]

        for i_d, day in enumerate(input_type):
            keep_units = models[i_d, 'keep_units']

            for imod, mod in enumerate(mod_nms):

                # Sep CO: 
                m0 = np.array(models[i_d, 0, mod+'_sep_fit', met])
                i0 = info[day[0][0]][keep_units]

                # Sep OBS: 
                m1 = np.array(models[i_d, 1, mod+'_sep_fit', met])
                i1 = info[day[1][0]][keep_units]

                # Combo: 
                m2 = np.array(models[i_d, 2, mod+'_comb_fit', met])
                i2 = info[day[0][0]][keep_units]

                tmp = ax.bar(i_d+(.2*imod), np.mean(m0), yerr=np.std(m0.reshape(-1))/np.sqrt(len(m0.reshape(-1))),
                    color = color[imod], width=1/15.) 
                tmp[0].set_hatch(hatchi[0])

                tmp = ax.bar(i_d+(.2*imod)+(.2/3.), np.mean(m1), yerr=np.std(m1.reshape(-1))/np.sqrt(len(m1.reshape(-1))),
                    color = color[imod], width=1/15.)
                tmp[0].set_hatch(hatchi[1])
                
                tmp = ax.bar(i_d+(.2*imod)+(.4/3.), np.mean(m2), yerr=np.std(m2.reshape(-1))/np.sqrt(len(m2.reshape(-1))),
                    color = color[imod], width=1/15.) 
                tmp[0].set_hatch(hatchi[2])

        ax.set_title(met)

def send_email(message_text):
    import smtplib
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("pkhanna104@gmail.com", "wrlyltbmcjwnpxil")
    server.sendmail("pkhanna104@gmail.com", "pkhanna104@gmail.com", message_text)
    server.quit()

def get_LPF(bs, tg_ix, animal, n_trials, s_d, pre_go_bins):
    assert pre_go_bins == 10

    # Count trials:
    N = len(bs)
    if N > n_trials:
        targs = np.unique(tg_ix)
        trl_per_targ = int(n_trials/len(targs))
        print 'n_trials: ', n_trials, 'n_targs:', len(targs), ' trl_per_targ: ', trl_per_targ

        IX = []
        for t in targs:
            ix = np.nonzero(tg_ix == t)[0]
            ix2 = np.random.permutation(len(ix))[:trl_per_targ]
            IX.append(ix[ix2])
        IX = np.hstack((IX))
    else:
        IX = range(n_trials)

    # Always 500 ms window:
    bs_subselect = [ bs[i][5:, :] for i in IX]
    bs_corr = np.vstack(([bs[i][10:, :] for i in IX]))
    pred_bs, smooth_bs = fit_LPF(bs_subselect, s_d)

    # R2: 
    R2_pred = get_R2(bs_corr, pred_bs)
    R2_smooth = get_R2(bs_corr, smooth_bs)
    return R2_pred, R2_smooth

def fit_LPF(bs_subselect, s_d, sep_trials = False):
    window = scipy.signal.gaussian(10., s_d/100.)

    # Causal Gaussian filter
    window[5:] = 0.
    window = window / float(np.sum(window))

    # Convolve all points with this filter:
    smooth_bs = []
    pred_bs = []
    for bs in bs_subselect:
        s_bs = []
        # Bs = time x units:
        for n in range(bs.shape[1]):
            # Inludes y_smooth_{-1}:y_smooth{T}
            s_bs.append(scipy.signal.convolve(bs[:, n], window, mode='same'))
        # T+1 x units
        s_bs = np.vstack((s_bs)).T

        # {t-1} --> {T-1}
        pred_bs.append(s_bs[:-1, :]) ### Prediction means can't use curernt time point -- at t = 0, doesn't use current tiem bc fo window zeroing
        smooth_bs.append(s_bs[1:, :]) ### Smooth says you can. 
    if sep_trials:
        return pred_bs, smooth_bs
    else:
        return np.vstack((pred_bs)), np.vstack((smooth_bs))

    # 

def get_model_stats(bs, tg_ix, animal, n_trials, model, pre_go_bins):

    # Count trials: 
    N = len(bs)
    if N > n_trials:
        targs = np.unique(tg_ix)
        trl_per_targ = int(n_trials/len(targs))
        print 'n_trials: ', n_trials, 'n_targs:', len(targs), ' trl_per_targ: ', trl_per_targ

        IX = []
        for t in targs:
            ix = np.nonzero(tg_ix == t)[0]
            ix2 = np.random.permutation(len(ix))[:trl_per_targ]
            IX.append(ix[ix2])
        IX = np.hstack((IX))
    else:
        IX = range(n_trials)
    
    if model is 'FA':
        bs_subselect = np.vstack(( [bs[i][pre_go_bins:, :] for i in IX] ))
        zscore_X, mu = pa.zscore_spks(bs_subselect)
        log_lik, ax = pa.find_k_FA(zscore_X, iters = 3, max_k = 15, plot=False)
        sum_LL = np.nanmean(log_lik, axis=0)
        num_factors = np.argmax(sum_LL) + 1
        FA_full = skdecomp.FactorAnalysis(n_components = num_factors)
        FA_full.fit(zscore_X)

        # Compute FA 
        reconst_ = np.array(FA_full.transform(zscore_X)*np.mat(FA_full.components_))
        R2_smooth = get_R2(zscore_X, reconst_)
        R2_pred = 0.

        # Get main shared dim: 
        U_B = np.matrix(FA_full.components_.T)
        UUT_B = U_B*U_B.T
        main_dim = get_main_shared_dim(UUT_B)

    elif model is 'LDS':
        bs_subselect =  [bs[i] for i in IX]
        kwargs = dict(seed_pred_x0_with_smoothed_x0=True, 
            seed_w_FA=True, nms=['Spks'], plot=False, get_ratio=False, pre_go_bins=pre_go_bins)
        num_factors = 10
                
        R2_smooth, R2_pred, LL, _, _, model, _, _, _, _, _, _ = fit_LDS(bs_subselect, 
            bs_subselect, num_factors, nEMiters=30, return_model=True, **kwargs)

        x0 = np.vstack(([model.states_list[i].smoothed_mus for i in range(len(bs_subselect))]))                
        C_fit = np.mat(model.C)
        Z_cov_0 = C_fit*np.cov(x0.T)*C_fit.T
        main_dim = get_main_shared_dim(Z_cov_0)

    return R2_smooth, R2_pred, main_dim, num_factors

def get_main_shared_dim(UU_T):
    v, s, vt = np.linalg.svd(UU_T)
    s_cum = np.cumsum(s)/np.sum(s)
    red_s = np.zeros((v.shape[1], ))

    #Find shared space that occupies > 90% of var:
    ix = np.nonzero(s_cum>0.9)[0]
    nf = ix[0] + 1
    return nf

def plot_bin_sweep(master_r2_dict, input_type, nstates, nSplits=2, animal='grom'):
    ######################
    ### Plot Bin Sweep ###
    ######################
    if animal == 'grom':
        CS3 = plt.contourf(Z, levels, cmap=mymap)
        lev = levels.copy()
    elif animal == 'jeev':
        CS3 = plt.contourf(Z, levels_j, cmap=mymap)
        lev = levels_j.copy()
    keys = np.sort(master_r2_dict.keys())
    nneurons, ns, natt, nit = master_r2_dict[keys[0]][0, 'Spks', 'smooth'].shape
    plt.clf()
    nms = ['smooth', 'pred']
    f, ax = plt.subplots(nrows = 2, ncols = len(nstates))
    if len(nstates) == 1:
        ax = ax[:, np.newaxis]
    if animal == 'jeev':
        info = info_jeev
    elif animal == 'grom':
        info = info_grom
    info_now = info[input_type[0][0][0]]

    for n in range(nneurons):
        for ns, n_s in enumerate(nstates):
            r2smooth = []
            r2pred = []
            for ik, ky in enumerate(keys):
                r2smooth.append(np.mean(master_r2_dict[ky][0, 'Spks', 'smooth'][n, ns, 0, :]))
                r2pred.append(np.mean(master_r2_dict[ky][0, 'Spks', 'pred'][n, ns, 0, :]))
            #c = ax[0].plot(n, info_now[n], '.', color=mymap(info_now[n]))
            c = ax[0, ns].plot(keys, r2smooth, '.-',color = mymap(np.digitize(info_now[n], lev)))
            c = ax[1, ns].plot(keys, r2pred, '.-',color = mymap(np.digitize(info_now[n], lev)))

    # Choose 100 ms: Show R2 plots: 
    f, ax2 = plt.subplots()
    for i_d in range(len(input_type)):
        info_now = info[input_type[i_d][0][0]]
        nneurons, _, _, _ = master_r2_dict[keys[0]][i_d, 'Spks', 'smooth'].shape
        pred_r2 = []
        imp_pred_r2 = []
        for n in range(nneurons):
            ns = 0
            r2pred= []
            ky = [k for k in keys if 99. < k < 106.]
            ky = ky[0]
            ax2.plot(i_d+.1*np.random.randn(), np.mean(master_r2_dict[ky][i_d, 'Spks', 'pred'][n, ns, 0, :]), '.', color=mymap(np.digitize(info_now[n], lev)))
            pred_r2.append(np.mean(master_r2_dict[ky][i_d, 'Spks', 'pred'][n, ns, 0, :]))
            if info_now[n] > .5:
                imp_pred_r2.append(np.mean(master_r2_dict[ky][i_d, 'Spks', 'pred'][n, ns, 0, :]))
        ax2.plot(i_d, np.mean(pred_r2), 'k.', markersize=20)
        ax2.errorbar(i_d, np.mean(pred_r2), yerr=np.std(pred_r2)/np.sqrt(nneurons), marker='o', color='k')
        ax2.plot(i_d, np.mean(imp_pred_r2), 'r.', markersize=20)
        ax2.errorbar(i_d, np.mean(imp_pred_r2), yerr=np.std(imp_pred_r2)/np.sqrt(len(imp_pred_r2)), marker='o', color='r')
    ax2.set_ylabel('R2 of Predicted Neurons')
    ax2.set_xlabel('Days')
    ax2.set_title('R2 for LDS: nstates=10, Black:All Neurons, Red:Important Neurons')

    i = 1
    ii = 0
    for j in range(len(nstates)):
        ax[i, j].set_xlabel('Bin Size (ms)')
        ax[ii, j].set_title('Nstates: '+str(nstates[j]))

    j = 0
    jj = len(nstates) - 1
    for i in range(2):
        ax[i, j].set_ylabel('R2: '+nms[i])
        plt.colorbar(CS3, ax=ax[i, jj])

            
    ### Dynamics vs. Dynamics / Total
    mn_std = {}
    f, ax = plt.subplots(nrows=len(nstates))
    if len(nstates) == 1:
        ax = [ax]
    for ns, n_s in enumerate(nstates):
        axi = ax[ns]
        for ik, ky in enumerate(keys):
            for i_d in range(len(input_type)):
                dyn = master_r2_dict[ky][i_d, 'Spks', 'dyn_norm'][ns, 0, 0]
                inn = master_r2_dict[ky][i_d, 'Spks', 'innov_norm'][ns, 0, 0]
                for i in range(1, nSplits):
                    dyn.extend(master_r2_dict[ky][i_d, 'Spks', 'dyn_norm'][ns, 0, i])
                    inn.extend(master_r2_dict[ky][i_d, 'Spks', 'innov_norm'][ns, 0, i])
                ratio = np.hstack((dyn)) / (np.hstack((dyn)) + np.hstack((inn)))
                mn_std[ky, 'rat'] = [np.mean(ratio), np.std(ratio)/float(np.sqrt(len(ratio)))]
                #mn_std[ky, 'inn'] = [np.mean(ratio), np.std(ratio)]
                axi.bar(ky+(2*i_d), mn_std[ky, 'rat'][0], color=cmap_list[i_d], width=2)
                axi.errorbar(ky+1+(2*i_d), mn_std[ky, 'rat'][0], yerr=mn_std[ky, 'rat'][1], color=cmap_list[i_d])
        axi.set_xlabel('Bin Size')
        axi.set_ylabel('Dyn / (Innov + Dyn)')
            # axi.bar(ky+3, mn_std[ky, 'inn'][0], color='r', width=6)
            # axi.errorbar(ky+6, mn_std[ky, 'inn'][0], yerr=mn_std[ky, 'inn'][1], color='r')


    ####################################
    ##### Test Initial BIN SWEEP #######
    ####################################
    # bin_sweep = pickle.load(open('test_bin_sweep_day0.pkl'))
    # keys = np.sort(bin_sweep.keys())
    # f, ax = plt.subplots(nrows=3)
    # for ik, ky in enumerate(keys):
    #     bs = bin_sweep[ky]
    #     ax[0].errorbar(ky, bs[0, 0, 'pred'].mean(), yerr=bs[0, 0, 'pred'].std(), marker='.')
    #     ax[1].errorbar(ky, bs[0, 0, 'smooth'].mean(), yerr=bs[0, 0, 'smooth'].std(), marker='.')
    #     ax[2].errorbar(ky, bs[0, 0, 'LL'].mean(), yerr=bs[0, 0, 'LL'].std(), marker='.')

def to_run_overlaps(epoch_size=32, mx = 10, skip_days = [], pre_go=1, animal='grom', only_FA=False):

    pre_go_bins = pre_go*10 
    # Assuming 100 ms
    
    ####################################
    ### Run Subspace Overlap on Day 0 ##
    ####################################
    #input_type = [[[4377], [4378, 4382]]]
    if animal == 'grom':
        input_type = analysis_config.data_params['grom_input_type']
    elif animal == 'jeev':
        input_type = fk.task_filelist
        input_names = fk.task_input_type

    day_dict = {}
    for i_d, day in enumerate(input_type):
        if i_d not in skip_days:

            te_list = list(np.hstack((day)))

            if animal == 'grom':
                #Will cycle through FAs and choose best one! yay
                data_agg, te_mod = subspace_overlap.targ_vs_all_subspace_align(te_list, cycle_FAs=None, epoch_size = epoch_size, 
                    include_mn=False, include_training_data = True, ignore_zero_units = True)

            elif animal == 'jeev':
                tsk_ix = []
                for i_t, tsk in enumerate(day):
                    for i_f, fn in enumerate(tsk):
                        tsk_ix.append(i_t)

                te_nm_list = list(np.hstack((input_names[i_d])))
                data_agg, te_mod = subspace_overlap.targ_vs_all_subspace_align_jeev(te_list, te_nm_list, 
                    cycle_FAs=None, epoch_size = epoch_size, include_mn=False, tsk_ix = tsk_ix,
                    include_training_data = True)                

            day_dict[i_d, 'FA_sep_overlap_dict'] = np.squeeze(data_agg['overlab_mat'])
            day_dict[i_d, 'te_mod'] = te_mod
            if animal == 'grom':
                day_dict[i_d, 'keep_units'] = data_agg['keep_units']
        
            FA_combo_ov = np.zeros((len(te_mod), len(te_mod), 2))
            REP_FA_combo_ov = np.zeros((len(te_mod), len(te_mod), 2 ))

            for it, te in enumerate(te_mod):
                for it2, te2 in enumerate(te_mod[it:]):
                    
                    ### Get data ###
                    zscore_X = np.vstack((data_agg[te, 0].training_data, data_agg[te2, 0].training_data))
                    ix0 = np.arange(len(np.vstack((data_agg[te, 0].training_data))))
                    ix1 = np.arange(len(ix0), len(ix0) + len(np.vstack((data_agg[te2, 0].training_data))))
                    
                    ### Fit Combo FA model ###
                    log_lik, ax = pa.find_k_FA(zscore_X, iters = 1, max_k = mx, plot=False)
                    mn_log_like = np.nanmean(log_lik, axis=0)
                    ix = np.argmax(mn_log_like)
                    num_factors = ix + 1
                    FA_full = skdecomp.FactorAnalysis(n_components = num_factors)
                    FA_full.fit(zscore_X)
                    print 'done fitting combo FA: ', te, te2, num_factors

                    #### Now fit individual U matrices for each task ####
                    z_hat = FA_full.transform(zscore_X)
                    z_hat_0 = z_hat[ix0, :]
                    z_hat_1 = z_hat[ix1, :]

                    U_fit = np.mat(FA_full.components_).T

                    Z_cov_0 = U_fit*np.cov(z_hat_0.T)*U_fit.T
                    Z_cov_1 = U_fit*np.cov(z_hat_1.T)*U_fit.T

                    Z_cov_0_mn, Z_cov_0_norm = tfo.get_mn_shar(Z_cov_0)
                    Z_cov_1_mn, Z_cov_1_norm = tfo.get_mn_shar(Z_cov_1)

                    proj_1_0_comb = np.trace(Z_cov_0_norm*Z_cov_1_mn*Z_cov_0_norm.T)/float(np.trace(Z_cov_1_mn))
                    proj_0_1_comb = np.trace(Z_cov_1_norm*Z_cov_0_mn*Z_cov_1_norm.T)/float(np.trace(Z_cov_0_mn))
                    print proj_0_1_comb, proj_1_0_comb
                    FA_combo_ov[it, it2, :] = np.array([proj_0_1_comb, proj_1_0_comb])
                    ov0, ov1 = get_repertoire_ov(z_hat_0, z_hat_1)
                    REP_FA_combo_ov[it, it2, :] = np.array([ov0, ov1])

            day_dict[i_d, 'FA_combo_overlap_dict'] = FA_combo_ov
            day_dict[i_d, 'REP_FA_combo_ov'] = REP_FA_combo_ov

            if not only_FA:
                #### Now try LDS separate first:
                bin_spk_dict = {}
                for te in te_list:
                    # We need to exclude indices because some trials don't have enough time for the pre-go
                    if animal == 'grom':
                        bin_spk, _, _, _, exclude_ix = pull_data(te, animal, pre_go=1, keep_units=data_agg['keep_units'])
                    elif animal == 'jeev':
                        bin_spk, _, _, _, exclude_ix = pull_data(te, animal, pre_go=1)
                    
                    bin_spk_dict[te] = bin_spk
                    bin_spk_dict[te, 'exclude'] = exclude_ix

                if animal == 'grom':
                    for te in te_mod:
                        ix = data_agg[te, 0].training_data_trl_ix
                        # Reuced number of trials:
                        bin_spk_dict[te] = [bin_spk_dict[int(te)][i] for i in ix if i not in bin_spk_dict[int(te), 'exclude']]
                elif animal == 'jeev':
                    for iii, (te_nm, te_fl) in enumerate(zip(te_nm_list, np.hstack((te_list)))):
                        pref_te_nm = te_nm[4:]

                        for iv, te in enumerate(te_mod):
                            ix = data_agg[te, 0].training_data_trl_ix
                            
                            # Find right file_name: Does this te match the pref_te_nm?
                            if te.find(pref_te_nm) == 0:
                                #Then use this current file name:
                                # Reuced number of trials:
                                bin_spk_dict[te] = [bin_spk_dict[te_fl][i] for i in ix if i not in bin_spk_dict[te_fl, 'exclude']]                    

                #### Separate LDS
                kwargs = dict(seed_pred_x0_with_smoothed_x0=True, 
                    seed_w_FA=True, nms=['Spks'], plot=False, get_ratio=False, pre_go_bins=pre_go_bins)

                nstates = 10
                LDS_dict = {}
                for te in te_mod:
                    R2_smooth, R2_pred, LL, dynamics_norm, innov_norm, model, _, _, _, _, _, _ = fit_LDS(bin_spk_dict[te], 
                        bin_spk_dict[te], nstates, nEMiters=30, return_model=True, **kwargs)
                    LDS_dict[te] = model
                    LDS_dict[te, 'xpred'] = np.vstack(([model.states_list[i].smoothed_mus for i in range(len(model.states_list))]))

                #### Overlap of C matrics: 
                LSD_sep_ov = np.zeros((len(te_mod), len(te_mod), 2))
                for it, te in enumerate(te_mod):
                    C0 = np.mat(LDS_dict[te].C)
                    x0 = np.mat(LDS_dict[te, 'xpred'])
                    cov0 = C0*np.cov(x0.T)*C0.T

                    for it2, te2 in enumerate(te_mod[it:]):
                        C1 = np.mat(LDS_dict[te2].C)
                        x1 = np.mat(LDS_dict[te2, 'xpred'])
                        cov1 = C1*np.cov(x1.T)*C1.T

                        ov01 = subspace_overlap.get_overlap(0, 0, first_UUT=np.mat(cov0), second_UUT = np.mat(cov1))
                        ov10 = subspace_overlap.get_overlap(0, 0, first_UUT=np.mat(cov1), second_UUT = np.mat(cov0))
                        
                        LSD_sep_ov[it, it2, :] = np.array([ov01, ov10])
                day_dict[i_d, 'LDS_sep_overlap_dict'] = LSD_sep_ov

                ### Combo LDS ###
                LSD_comb_ov = np.zeros((len(te_mod), len(te_mod), 2))
                REP_LDS_comb_ov = np.zeros((len(te_mod), len(te_mod), 2))
                for it, te in enumerate(te_mod):
                    BS = bin_spk_dict[te]
                    ix0 = np.arange(len(BS))
                    for it2, te2 in enumerate(te_mod[it:]):
                        bs1 = bin_spk_dict[te2]
                        ix1 = np.arange(len(BS), len(BS)+len(bs1))
                        BS.extend(bs1)
                        R2_smooth, R2_pred, LL, dynamics_norm, innov_norm, model,_, _, _, _, _, _ = fit_LDS(BS, 
                            BS, nstates, nEMiters=30, return_model=True, **kwargs)
                        
                        LDS_dict[te, te2] = model
                        x0 = np.vstack(([model.states_list[i].smoothed_mus for i in ix0]))
                        LDS_dict[te, 'xpred'] = x0
                        d0 = np.vstack(([BS[i] for i in ix0]))

                        x1 = np.vstack(([model.states_list[i].smoothed_mus for i in ix1]))
                        d1 = np.vstack((BS[i] for i in ix1))
                        LDS_dict[te2, 'xpred'] = x1
                        C_fit = np.mat(model.C)

                        Z_cov_0 = C_fit*np.cov(x0.T)*C_fit.T
                        Z_cov_1 = C_fit*np.cov(x1.T)*C_fit.T

                        Z_cov_0_mn, Z_cov_0_norm = tfo.get_mn_shar(Z_cov_0)
                        Z_cov_1_mn, Z_cov_1_norm = tfo.get_mn_shar(Z_cov_1)

                        proj_1_0_comb = np.trace(Z_cov_0_norm*Z_cov_1_mn*Z_cov_0_norm.T)/float(np.trace(Z_cov_1_mn))
                        proj_0_1_comb = np.trace(Z_cov_1_norm*Z_cov_0_mn*Z_cov_1_norm.T)/float(np.trace(Z_cov_0_mn))
                        
                        LSD_comb_ov[it, it2, :] = np.array([proj_0_1_comb, proj_1_0_comb])
                        
                        ### Now try this repertoire thing with combo LDS, combo FA: 
                        ov0, ov1 = get_repertoire_ov(x0, x1)
                        REP_LDS_comb_ov[it, it2, :] = np.array([ov0, ov1])

                day_dict[i_d, 'LDS_comb_overlap_dict'] = LSD_comb_ov
                day_dict[i_d, 'REP_LDS_comb_ov'] = REP_LDS_comb_ov
            pickle.dump(day_dict, open(animal+'test_day_dict_'+str(i_d)+'.pkl', 'wb'))
            print 'dumped day_dict, day: ', i_d
    return day_dict

def to_run_dyn_predictions_from_bin_vel(input_type, skip_days, pre_go, animal, input_names=None):
    pre_go_bins = int(pre_go*10) # asssuming bins size is 100 ms
    day_dict = {}
    MASTER = {}
    for i_d, day in enumerate(input_type):
        if i_d not in skip_days:
            te_list = list(np.hstack((day)))

            if animal == 'grom':
                # Get zero units: 
                data_agg, te_mod = subspace_overlap.targ_vs_all_subspace_align(te_list, cycle_FAs=None, epoch_size = 500, 
                    include_mn = False, include_training_data = True, ignore_zero_units = True)
            elif animal == 'jeev':
                if input_names is None:
                    raise Exception
                
                tsk_ix = []
                for i_t, tsk in enumerate(day):
                    for i_f, fn in enumerate(tsk):
                        tsk_ix.append(i_t)

                te_nm_list = list(np.hstack((input_names[i_d])))
                data_agg, te_mod = subspace_overlap.targ_vs_all_subspace_align_jeev(te_list, te_nm_list, 
                    cycle_FAs=None, epoch_size = 1000, include_mn=False, tsk_ix = tsk_ix,
                    include_training_data = True)                

            
            #### Now try LDS separate first:
            bin_spk_dict = {}
            for te in te_list:
                # We need to exclude indices because some trials don't have enough time for the pre-go
                bin_spk, targ_ix, _, _, exclude_ix = pull_data(te, animal, pre_go=pre_go, keep_units=data_agg['keep_units'])
                bin_spk_dict[te] = bin_spk
                bin_spk_dict[te, 'exclude'] = exclude_ix
                bin_spk_dict[te, 'target'] = targ_ix

                _, mag_thresh, bin_spk_dict[te, 'vel_ix'] = behav_neural_PSTH.get_snips(te, animal=animal, mag_thresh=[.5, 1.0, 1.5], 
                    beh_or_neural='beh', return_ind_ix= True, exclude_ix = exclude_ix)

            kwargs = dict(seed_pred_x0_with_smoothed_x0=True, 
                seed_w_FA=True, nms=['Spks'], plot=False, get_ratio=False, pre_go_bins=pre_go_bins, fa_thresh = .9)

            nstates = 10

            #################
            ### Combo LDS ###
            #################

            BS = []
            IX = []
            cnt = 0
            for it, te in enumerate(te_list):
                BS.extend(bin_spk_dict[te][i] for i in range(len(bin_spk_dict[te])) if i not in bin_spk_dict[te, 'exclude'])
                IX.extend([it, i, cnt+i] for i in range(len(bin_spk_dict[te]) - len(bin_spk_dict[te, 'exclude'])))
                cnt += len(bin_spk_dict[te]) - len(bin_spk_dict[te, 'exclude'])

            R2_smooth, R2_pred, LL, dynamics_norm, innov_norm, model, pred_data, smooth_data, filt_state, pred_state, filt_sigma, kg_ = fit_LDS(BS, 
                BS, nstates, nEMiters=30, return_model=True, **kwargs)

            # Compute SSKF: 
            KG = []
            df = []
            kg_last = np.zeros((nstates, BS[0].shape[1]))
            for it, trl in enumerate(filt_sigma):
                for ib in range(trl.shape[0]):
                    S = model.C*np.mat(trl[ib, :, :])*np.mat(model.C).T + model.sigma_obs
                    kg = np.mat(trl[ib, :, :])*model.C.T*np.linalg.inv(S)
                    KG.append(kg)
                    df.append(np.linalg.norm(kg - kg_last))
                    kg_last = kg.copy()
            df = np.array(df)
            df = df[5:]
            assert np.max(df) < .1
            KalmanGain = KG[6]

            real_dat = [BS[i][pre_go_bins:, :] for i in range(len(BS))]

            #### Accrue all velocity commands ####
            cnt = 0
            vel_cmd_keys = {}
            tsk_ix_keys = {}

            for i_t, tsk in enumerate(day):
                for t, te_num in enumerate(tsk):
                    ix_ = np.vstack((IX))

                    ### velocity data ###
                    dat = bin_spk_dict[te_num, 'vel_ix']
                    ix = np.nonzero(ix_[:, 0] == cnt)[0]
                    sub_ix = ix_[ix, :]
                    cnt += 1

                    for trl_ix in dat.keys():
                        ident = np.nonzero(sub_ix[:, 1]==trl_ix)[0]

                        ### Each bin in each trial ###
                        for bin in range(len(dat[trl_ix])):
                            
                            # 0: Trial, 1: Big trial, 2: Bin, 3: Task Ix, 4: X_filt_t-1, 5: X_pred_t, 6: X_filt_t, 7: Y_real_t, 8: target_num
                            if bin > 0:
                                big_ix = sub_ix[int(ident), 2]
                                pre_state_filt = filt_state[big_ix][bin-1, :]
                                try:
                                    vel_cmd_keys[tuple(dat[trl_ix][bin, :])].append([trl_ix, big_ix, bin, i_t, pre_state_filt,
                                        pred_state[big_ix][:, bin], filt_state[big_ix][bin, :], real_dat[big_ix][bin, :]])

                                    tsk_ix_keys[tuple(dat[trl_ix][bin, :])].append([i_t])

                                except:
                                    vel_cmd_keys[tuple(dat[trl_ix][bin, :])] = [[trl_ix, big_ix, bin, i_t, pre_state_filt,
                                        pred_state[big_ix][:, bin], filt_state[big_ix][bin, :], real_dat[big_ix][bin, :]]]
                                    tsk_ix_keys[tuple(dat[trl_ix][bin, :])] = [[i_t]]

            # Now, for every bin, compare to other bins: 
            self_co = {}
            self_obs = {}
            co_co = {}
            obs_obs = {}
            co_obs = {}
            obs_co = {}

            for k in vel_cmd_keys.keys():
                ixs = np.squeeze(np.hstack((tsk_ix_keys[k])))
                dat = vel_cmd_keys[k]
                co_ix = np.nonzero(ixs == 0)[0]
                obs_ix = np.nonzero(ixs == 1)[0]

                for dict_ in [self_co, self_obs, co_co, obs_obs, co_obs, obs_co]:
                    dict_[k] = []

                for i in co_ix:
                    # Self:
                    di = dat[i] 
                    dyn = np.linalg.norm(np.mat(di[5]) - np.mat(di[4]).T)
                    inn = np.linalg.norm(KalmanGain*(np.mat(di[7]).T - model.C*np.mat(model.A)*np.mat(di[4]).T))
                    self_co[k].append([dyn, inn])

                    # Other CO
                    rand_ix = np.random.permutation(len(co_ix))
                    co_rnd_ix = co_ix[rand_ix[:20]]
                    for j in co_rnd_ix:
                        do = dat[j] 
                        dyn = np.linalg.norm(np.mat(di[5]) - np.mat(di[4]).T)
                        inn = np.linalg.norm(KalmanGain*(np.mat(do[7]).T - model.C*np.mat(model.A)*np.mat(di[4]).T))
                        co_co[k].append([dyn, inn])                        

                    # Other Obs
                    rand_ix = np.random.permutation(len(obs_ix))
                    obs_rnd_ix = obs_ix[rand_ix[:20]]
                    for j in obs_rnd_ix:
                        do = dat[j] 
                        dyn = np.linalg.norm(np.mat(di[5]) - np.mat(di[4]).T)
                        inn = np.linalg.norm(KalmanGain*(np.mat(do[7]).T - model.C*np.mat(model.A)*np.mat(di[4]).T))
                        co_obs[k].append([dyn, inn])      

                for i in obs_ix:
                    # Self:
                    di = dat[i] 
                    dyn = np.linalg.norm(np.mat(di[5]) - np.mat(di[4]).T)
                    inn = np.linalg.norm(KalmanGain*(np.mat(di[7]).T - model.C*np.mat(model.A)*np.mat(di[4]).T))
                    self_obs[k].append([dyn, inn])

                    # Other CO
                    rand_ix = np.random.permutation(len(co_ix))
                    co_rnd_ix = co_ix[rand_ix[:20]]
                    for j in co_rnd_ix:
                        do = dat[j] 
                        dyn = np.linalg.norm(np.mat(di[5]) - np.mat(di[4]).T)
                        inn = np.linalg.norm(KalmanGain*(np.mat(do[7]).T - model.C*np.mat(model.A)*np.mat(di[4]).T))
                        obs_co[k].append([dyn, inn])                        

                    # Other Obs
                    rand_ix = np.random.permutation(len(obs_ix))
                    obs_rnd_ix = obs_ix[rand_ix[:20]]
                    for j in obs_rnd_ix:
                        do = dat[j] 
                        dyn = np.linalg.norm(np.mat(di[5]) - np.mat(di[4]).T)
                        inn = np.linalg.norm(KalmanGain*(np.mat(do[7]).T - model.C*np.mat(model.A)*np.mat(di[4]).T))
                        obs_obs[k].append([dyn, inn])   

            master = {}
            for dict_ in ['self_co', 'self_obs', 'co_co', 'obs_obs', 'co_obs', 'obs_co']:
                master[dict_] = []

            for k in vel_cmd_keys.keys():
                for i, (nm, dict_) in enumerate(zip(['self_co', 'self_obs', 'co_co', 'obs_obs', 'co_obs', 'obs_co'],
                    [self_co, self_obs, co_co, obs_obs, co_obs, obs_co])):
                    if len(dict_[k]) > 0:
                        dat = np.vstack((dict_[k]))
                        #dyn_ratio = dat[:, 0]/(np.sum(dat, axis=1))
                        #dyn_ratio = dat[:, 1]
                        master[nm].append(dat)
            MASTER[i_d] = master
            pickle.dump(MASTER, open('dynamics_grom_by_tsk_bin_'+str(i_d)+'.pkl', 'wb'))

def plot_dynamics_predictions(input_type):
    color = ['k', 'b', 'r']
    task_nms={}
    task_nms[0]=['self_co', 'co_co', 'co_obs']
    task_nms[1]=['self_obs', 'obs_obs', 'obs_co']
    MASTER={}
    tasks = ['co', 'obs']

    for i in range(2):
        for i_n, nm in enumerate(task_nms[i]):
            MASTER[tasks[i], nm] = [] 
            MASTER[i_n] = []
    f, ax = plt.subplots(nrows=2)

    for i_d, day in enumerate(input_type):
        master = pickle.load(open('dynamics_grom_by_tsk_bin_'+str(i_d)+'.pkl'))

        for i_t in range(2):
            for t, tsks in enumerate(task_nms[i_t]):
                # Innovation Terms
                dt = np.vstack((master[i_d][tsks]))
                master[i_d][tsks] = dt[:, 0]/(dt[:, 0]+dt[:, 1])
                ax[i_t].bar(i_d+(.2*t), np.median(master[i_d][tsks]), width=.2, color=color[t])
                # ax[i_t].errorbar(i_d+(.2*t), np.mean(master[i_d][tsks]),yerr=np.std(master[i_d][tsks])/np.sqrt(len(master[i_d][tsks])),
                #     marker='.', color=color[t])
                MASTER[tasks[i_t], tsks].append(master[i_d][tsks])
    
    print 'pvalues by ind task aggregated over days: '
    for i_t in range(2):
        for t, tsks in enumerate(task_nms[i_t]):
            MASTER[tasks[i_t], tsks] = np.hstack((MASTER[tasks[i_t], tsks]))
            ax[i_t].bar(i_d+3+.2*t, np.median(MASTER[tasks[i_t], tsks]), width=.2, color=color[t])
            MASTER[t].append(MASTER[tasks[i_t], tsks])

        print 'task: ', tasks[i_t]
        kw, kwp = scipy.stats.kruskal(MASTER[tasks[i_t], task_nms[i_t][0]], MASTER[tasks[i_t], task_nms[i_t][1]], MASTER[tasks[i_t], task_nms[i_t][2]], )
        print 'kw: ', kwp
        u01, p01 = scipy.stats.mannwhitneyu(MASTER[tasks[i_t], task_nms[i_t][0]], MASTER[tasks[i_t], task_nms[i_t][1]])
        u12, p12 = scipy.stats.mannwhitneyu(MASTER[tasks[i_t], task_nms[i_t][1]], MASTER[tasks[i_t], task_nms[i_t][2]])
        u02, p02 = scipy.stats.mannwhitneyu(MASTER[tasks[i_t], task_nms[i_t][0]], MASTER[tasks[i_t], task_nms[i_t][2]])
        print 'mw: ', p01, p12, p02

        ax[i_t].set_title(tasks[i_t]+': dynamics ratio')
        ax[i_t].set_title(tasks[i_t]+': dynamics ratio')

    for i in range(3):
        MASTER[i] = np.hstack((MASTER[i]))
    print 'pvalues agg over task aggregated over days: '
    print 'kw: ', kwp

    kw, kwp = scipy.stats.kruskal(MASTER[0], MASTER[1], MASTER[2])

    # 0 - 1
    u01, p01 = scipy.stats.mannwhitneyu(MASTER[0], MASTER[1])
    u12, p12 = scipy.stats.mannwhitneyu(MASTER[1], MASTER[2])
    u02, p02 = scipy.stats.mannwhitneyu(MASTER[0], MASTER[2])
    print 'mw: ', p01, p12, p02

def plot_overlaps(i_d):
    day_dict = pickle.load(open('test_day_dict_'+str(i_d)+'.pkl'))
    day = analysis_config.data_params['grom_input_type'][i_d]
    f, ax = plt.subplots(nrows = 1, ncols = 4)
    ax_dict = {}
    ax_dict[0, 0] = 0 # Co-Co
    ax_dict[1, 1] = 1 # Obs-Obs

    ax_dict[0, 1] = 2 # Co--> Obs
    ax_dict[1, 0] = 3 # Obs --> Co

    cmap = ['r', 'b', 'g', 'k']

    te_mod = day_dict[i_d, 'te_mod']
    f2, ax2 = plt.subplots(nrows = 1, ncols = 3)

    metrics = ['FA_sep_overlap_dict', 'FA_combo_overlap_dict', 'LDS_sep_overlap_dict', 'LDS_comb_overlap_dict']
    metrics2 = ['REP_FA_combo_ov', 'REP_LDS_comb_ov']
    titles = ['CO<->CO', 'Obs<->Obs', 'Co--> Obs', 'Obs--> Co']

    for i_t, te in enumerate(te_mod):
        for i_t2, te2 in enumerate(te_mod[i_t:]):

            if te != te2:
                # Decide which category this combo is in: 
                int_te = int(te)
                int_te2 = int(te2)

                te_cat = 0 if int_te in day[0] else 1
                te_cat2 = 0 if int_te2 in day[0] else 1

                # Now plot stuff: 
                axi_num = ax_dict[te_cat, te_cat2]
                assert axi_num != 3 # should never go obs -- co

                axi = ax[axi_num]

                for im, met in enumerate(metrics):
                    mat = day_dict[i_d, met]

                    if axi_num in [0, 1]:
                        axi2 = [axi, axi]
                    elif axi_num == 2:
                        axi2 = [axi, ax[3]]

                    for i in range(2):
                        sub_mat = mat[:, :, i]
                        sub_ax = axi2[i]

                        if im == 0:
                            ix2 = i_t + i_t2
                            pt = sub_mat[i_t, ix2]
                        else:
                            pt = sub_mat[i_t, i_t2]

                        sub_ax.plot(im + .3*np.random.rand(), pt, cmap[im]+'.')

                for im, met in enumerate(metrics2):
                    mat = day_dict[i_d, met]
                    axi = ax2[axi_num]

                    if axi_num in [0, 1]:
                        axi.plot(im+.3*np.random.rand(2), mat[i_t, i_t2, :], cmap[im]+'.')
                    else:
                        axi.plot(im+.15*np.random.rand(), mat[i_t, i_t2, 0], cmap[im]+'.')
                        axi.plot(im+.4+.05*np.random.rand(), mat[i_t, i_t2, 1], cmap[im]+'.')

    for axj in [ax, ax2]:
        for i, axi in enumerate(axj):
            axi.set_ylim([0.0, 1.1])
            axi.set_xlim([-.5, 3.5])
            axi.set_title(titles[i])
    ax[0].text(0, 0.25, 'Red: FA Sep, Blue: FA Comb, \nGrn: LDS Sep, Blk: LDS Comb')
    ax2[0].text(0, 0.25, 'Red: FA Rep Fraction,\nBlue: LDS Rep Fraction')
    ax2[2].text(0, 0.25, 'Red1: FA CO Rep, Red2: FA Obs Rep')
    
def plot_ov_all(animal, day_dict):
    if animal == 'grom':
        input_type = analysis_config.data_params['grom_input_type'] #[i] for i in range(4)]
        #day_dict = pickle.load(open('test_day_dict_'+str(8)+'.pkl'))
    elif animal == 'jeev':
        input_type = fk.task_input_type
        #day_dict = pickle.load(open('test_day_dict_'+str(3)+'.pkl'))



    f, axall = plt.subplots(nrows=3, ncols = 2)
    metrics = [['FA_sep_overlap_dict', 'FA_combo_overlap_dict','REP_FA_combo_ov'],
        ['LDS_sep_overlap_dict','LDS_comb_overlap_dict','REP_LDS_comb_ov']]

    master_met = {}
    for mm in metrics:
        for m in mm:
            master_met[m, 0] = []
            master_met[m, 1] = []

    for md in range(2):
        for im, metric in enumerate(metrics[md]):
            ax = axall[im, md]

            for i_d in range(len(input_type)):
                day = input_type[i_d]
                met_dict = {}
                met_dict[0] = []
                met_dict[1] = []

                te_mod = day_dict[i_d, 'te_mod']
                for i_t, te in enumerate(te_mod):
                    for i_t2, te2 in enumerate(te_mod[i_t:]):
                        if te != te2:
                            # Decide which category this combo is in: 
                            if animal == 'grom':
                                int_te = int(te)
                                int_te2 = int(te2)
                                te_cat = 0 if int_te in day[0] else 1
                                te_cat2 = 0 if int_te2 in day[0] else 1

                            elif animal == 'jeev':
                                sub_te = te[:7]
                                sub_te2 = te2[:7]
                                te_cat = 0 if sub_te in day[0][0] else 1
                                te_cat2 = 0 if sub_te2 in day[0][0] else 1

                            if te_cat != te_cat2:
                                bin = 1
                            else:
                                bin = 0

                            mat = day_dict[i_d, metric]
                            if metric == 'FA_sep_overlap_dict':
                                met_dict[bin].append(mat[i_t, i_t2+i_t, :])
                            else:
                                met_dict[bin].append(mat[i_t, i_t2, :])
                for bin in range(2):
                    met_dict[bin] = np.hstack((met_dict[bin]))
                
                ax.plot(np.zeros((len(met_dict[0])))+i_d+.6, met_dict[0], 'r.', markersize=4)
                ax.bar(i_d+.4, np.mean(met_dict[0]), width=.4, color='b')
                ax.errorbar(i_d+.6, np.mean(met_dict[0]), yerr=np.std(met_dict[0])/np.sqrt(len(met_dict[0])), color='b')
                
                ax.plot(np.zeros((len(met_dict[1])))+i_d+.2, met_dict[1], 'r.', markersize=4)
                ax.bar(i_d, np.mean(met_dict[1]), width=.4, color='k')
                ax.errorbar(i_d+.2, np.mean(met_dict[1]), yerr=np.std(met_dict[1])/np.sqrt(len(met_dict[1])), color='k')

                master_met[metric, 0].append(met_dict[0])
                master_met[metric, 1].append(met_dict[1])

            z = np.hstack((master_met[metric, 0]))
            z2 = np.hstack((master_met[metric, 1]))

            if animal == 'grom':
                endx = 11.
            elif animal == 'jeev':
                endx = 6.

            ax.bar(endx+.4, np.mean(z), width = .4, color = 'b')
            ax.bar(endx, np.mean(z2), width=.4, color = 'k')
            t, p = scipy.stats.ttest_ind(z, z2)
            ax.set_title(metric+', p = '+str(p)+' b:win, k:x', dict(fontsize=12))
    plt.tight_layout()

def get_repertoire_ov(x0, x1):
    ''' Rectangle Method '''
    ranges_ = {}
    overlap = {}
    area0 = 1.
    area1 = 1.
    areaov = 1.

    ndim = x0.shape[1]
    assert x1.shape[1] == ndim
    for i in range(ndim):
        ranges_[i, 0] = [np.mean(x0[:, i]) - 2*np.std(x0[:, i]), np.mean(x0[:, i]+ 2*np.std(x0[:, i]))]
        area0 *= ranges_[i, 0][1] - ranges_[i, 0][0]
        ranges_[i, 1] = [np.mean(x1[:, i]) - 2*np.std(x1[:, i]), np.mean(x1[:, i]+ 2*np.std(x0[:, i]))]
        area1 *= ranges_[i, 1][1] - ranges_[i, 1][0]
        ranges_[i, 'ov'] = [np.max([ranges_[i, 0][0], ranges_[i, 1][0]]), np.min([ranges_[i, 0][1], ranges_[i, 1][1]])]
        areaov *= ranges_[i, 'ov'][1] - ranges_[i, 'ov'][0]

    return areaov/area0, areaov/area1

def get_repertoire_ov_numerical(x0, x1, ndiv=25):
    ''' 
    Takes mean +/- 2*std of each axis for x0 and x1 -- at least 1000 subsections of biggest dimension
    Matrix spanning this
    '''

    ranges_ = {}
    ndim = x0.shape[1]
    assert x1.shape[1] == ndim
    mx = 0.
    for i in range(ndim):
        ranges_[i, 0] = [np.mean(x0[:, i]) - 2*np.std(x0[:, i]), np.mean(x0[:, i]+ 2*np.std(x0[:, i]))]
        ranges_[i, 1] = [np.mean(x1[:, i]) - 2*np.std(x1[:, i]), np.mean(x1[:, i]+ 2*np.std(x0[:, i]))]

        mx = np.max([mx, ranges_[i, 0][1] - ranges_[i, 0][0]])
        mx = np.max([mx, ranges_[i, 1][1] - ranges_[i, 1][0]])

    # Take mx: 
    delta = mx/float(ndiv)

    #Make matrix of zeros:
    x = np.zeros((ndim))
    ruler = {}
    for i in range(ndim):
        mn = np.min(np.hstack(( ranges_[i, 0], ranges_[i, 1])))
        mxx = np.max(np.hstack(( ranges_[i, 0], ranges_[i, 1])))
        di = np.arange(mn, mxx, delta)
        ruler[i] = di
        x[i] = len(di)

    A = np.zeros(tuple(x.astype(int)))
    X0 = []
    X1 = []
    for i in range(ndim):
        x0_range = np.nonzero(np.logical_and(ruler[i] >= ranges_[i, 0][0], ruler[i] <= ranges_[i, 0][1]))[0]
        x1_range = np.nonzero(np.logical_and(ruler[i] >= ranges_[i, 1][0], ruler[i] <= ranges_[i, 1][1]))[0]
        X0.append(x0_range)
        X1.append(x1_range)
    ix0 = np.ix_(*X0)
    ix1 = np.ix_(*X1)
 
    A[ix0] += 1
    A[ix1] += 1
    del ruler
    del ranges_
    del x
    # Overlap over total non-zero area: 
    return np.sum(A==2)/np.float(np.sum(A!=0))

def extract_PSTH(animal, binsize_ms=100, pre_go=0):
    pre_go_bins = int(np.round(pre_go*(1000./(binsize_ms))))
    if animal == 'grom':
        input_type = analysis_config.data_params['grom_input_type']
        pre = '/Volumes/TimeMachineBackups/grom2016/grom_PSTH_per_targ_max_bins_possible.pkl'
    elif animal == 'jeev':
        input_type = fk.task_filelist
        pre = '/Volumes/TimeMachineBackups/jeev2013/jeev_PSTH_per_targ_max_bins_possible.pkl'

    PSTH_dict = {}
    tasks = ['co', 'obs']

    for i_d, day in enumerate(input_type):  
        te_list_tmp = np.hstack((day))

        # Decide which units to keep (rm zero sum over full day)
        keep_units = subspace_overlap.get_keep_units(te_list_tmp, 10000, animal, binsize_ms)
        
        for i_t, tsk in enumerate(day):
            BS = []
            TK = []
            TG = []
            EX = []

            for t, te_num in enumerate(tsk):
                if pre_go == 0:
                    bs, tg, trl, kg = pull_data(te_num, animal, pre_go=pre_go, binsize_ms=binsize_ms)
                    exclude = []
                else:
                    bs, tg, trl, kg, exclude = pull_data(te_num, animal, pre_go=pre_go, binsize_ms=binsize_ms)
                
                BS.extend(bs[i][:, keep_units] for i in range(len(bs)) if i not in exclude)
                TK.extend(np.zeros((len(bs) - len(exclude)))  + i_t)
                trl_ix = [i for i, j in enumerate(trl) if np.logical_and(i not in exclude, np.logical_and(j!= trl[i-1], i!= 0))]
                
                if 0 not in exclude:
                    trl_ix2 = [0]
                else:
                    trl_ix2 = []
                trl_ix2.extend(trl_ix)
                TG.extend(tg[trl_ix2])
                EX.append(exclude)

            PSTHs, TSK_TGs = PSTHify(BS, np.hstack((TK)), np.hstack((TG)), nbins='max')
            tgs = [t[1] for t in TSK_TGs]
            PSTH_dict[i_d, tasks[i_t]] = dict(psth=PSTHs, target=tgs, kg=kg[:, keep_units])
            print 'done w/ task: ', tasks[i_t], ' day: ', i_d
    if pre_go > 0:
        pickle.dump(PSTH_dict, open(pre[:-4]+'_prego_'+str(pre_go)+'.pkl', 'wb'))
    else:
        pickle.dump(PSTH_dict, open(pre, 'wb'))

def dyn_ratio_of_PSTH(animal, pre_go, binsize_ms=100):
    import model_aggregate

    ################
    ### Open PSTH ##
    ################
    pre_go_bins = int(np.round(pre_go*(1000./(binsize_ms))))
    if animal == 'grom':
        input_type = analysis_config.data_params['grom_input_type']
        pre = '/Volumes/TimeMachineBackups/grom2016/grom_PSTH_per_targ_max_bins_possible.pkl'
        save = '/Volumes/TimeMachineBackups/grom2016/'
    elif animal == 'jeev':
        input_type = fk.task_filelist
        pre = '/Volumes/TimeMachineBackups/jeev2013/jeev_PSTH_per_targ_max_bins_possible.pkl'
        save = '/Volumes/TimeMachineBackups/jeev2013/'
    if pre_go > 0:
        pre = pre[:-4]+'_prego_'+str(pre_go)+'.pkl'

    dat_PSTH = pickle.load(open(pre))
    dyn_ratio = {}
    task_names = ['co', 'obs']
    ################
    ### Open LDS ###
    ################
    # for all days, folds:
    dynamics_norm = {}
    innov_norm = {}
    bin_cnt = {}

    for task in range(2): 
        A = model_aggregate.get_metric(animal, 'data_LDS', 'LDS_model_A', task)
        C = model_aggregate.get_metric(animal, 'data_LDS', 'LDS_model_C', task)
        W = model_aggregate.get_metric(animal, 'data_LDS', 'LDS_model_W', task)
        Q = model_aggregate.get_metric(animal, 'data_LDS', 'LDS_model_Q', task)

        for i_d in range(len(A)):

            dynamics_norm[i_d] = []
            innov_norm[i_d] = []
            bin_cnt[i_d] = []

            n_folds, D_obs, n_dim_latent = C[i_d].shape
            D_input = 0

            for n in range(n_folds):
                model = DefaultLDS(D_obs, n_dim_latent, D_input)
                model.A = A[i_d][n, :, :]
                model.C = C[i_d][n, :, :]
                model.sigma_obs = Q[i_d][n, :, :]
                model.sigma_states = W[i_d][n, :, :]

                psth_list = dat_PSTH[i_d, task_names[task]]['psth']
                targ_list = dat_PSTH[i_d, task_names[task]]['target']

                for i_t, trg in enumerate(targ_list):
                    psth = psth_list[i_t]
                    model.add_data(psth)
                    g = model.states_list.pop()
        
                    try:
                        # Smoothed (y_t | y0...yT)
                        smoothed_trial = g.smooth()

            
                        # Smoothed (x_t | x0...xT)
                        x0 = g.smoothed_mus[0, :] # Time x ndim
                        P0 = g.smoothed_sigmas[0, :, :]
            

                        # Filtered states (x_t | y0...y_t)
                        _, filtered_mus, filtered_sigmas = kalman_filter(
                        x0, P0,
                        g.A, g.B, g.sigma_states,
                        g.C, g.D, g.sigma_obs,
                        g.inputs, g.data)
            

                        # Norm of dynamics process: 
                        dynamics_norm[i_d].append(np.linalg.norm(np.array(np.mat(g.A)*filtered_mus[pre_go_bins-1:-1, :].T - filtered_mus[pre_go_bins:, :].T), axis=0))

                        # Norm of innovations process:
                        inn_, kg_list = get_innov_list(g, filtered_sigmas, filtered_mus, pre_go_bins)
                        innov_norm[i_d].append(inn_)

                        bin_cnt[i_d].append(np.arange(len(inn_)))
                    except:
                        print 'missing: target %d, fold: %d, day: %d' %(i_t, n, i_d)

    total = dict(dyn = dynamics_norm, inn = innov_norm, bin_cnt = bin_cnt)
    pickle.dump(total, open(save+animal+'_PSTH_dyn_ratio.pkl', 'wb'))

def plot_dyn_ratio(animal, bin_cnt_max = 'all'):
    if animal == 'grom':
        input_type = analysis_config.data_params['grom_input_type']
        save = '/Volumes/TimeMachineBackups/grom2016/'
    elif animal == 'jeev':
        input_type = fk.task_filelist
        save = '/Volumes/TimeMachineBackups/jeev2013/'
    dat = pickle.load(open(save+animal+'_PSTH_dyn_ratio.pkl'))

    for i_d in range(len(input_type)):
        try:
            inn = np.hstack(( dat['inn'][i_d] ))
            dyn = np.hstack(( dat['dyn'][i_d] ))
            cnt = np.hstack(( dat['bin_cnt'][i_d] ))

            if bin_cnt_max == 'all':
                rat = dyn / (dyn + inn)

            else:
                ix = np.nonzero(cnt <= bin_cnt_max)[0]
                rat = dyn[ix] / (dyn[ix] + inn[ix])
            plt.bar(i_d, np.mean(rat))
            plt.errorbar(i_d+.5, np.mean(rat), np.std(rat)/np.sqrt(len(rat)), fmt='.')            
        except:
            print 'skip day %d' %(i_d)

