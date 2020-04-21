
### Methods to analyze smoothness in LDS ###
### Also to analyze mFR / covFR difference ###
import subspace_overlap, co_obs_tuning_curves_fig3
import prelim_analysis as pa
import pickle
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from pylds.states import kalman_filter
from sklearn import mixture
import math
import tables;

pref = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/'

def extract_data(animal='grom', min_observations = 15):

    ### analyze mFR / covFR difference ###
    ### Animal
    if animal == 'grom':
        dat = pickle.load(open(pref+'gromLDSmodels_nstates_15_combined_models_w_dyn_inno_norms.pkl', 'rb'))
        ndays = 9; 
    elif animal == 'jeev':
        dat = pickle.load(open(pref+'jeevLDSmodels_nstates_15_combined_models_w_dyn_inno_norms.pkl', 'rb'))
        ndays = 4; 

    ### Bin commands into Mag / Ang
    mag_boundaries = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/radial_boundaries_fit_based_on_perc_feb_2019.pkl'))

    WIN = []; 
    X = [];

    WIN_cov = []; 
    X_cov = []; 

    for i_d in range(ndays):

        ### Neural push
        commands = dat[i_d, 'neural_push']
        ntrls = len(commands)
        command_bins = subspace_overlap.commands2bins(commands, mag_boundaries, animal, i_d)

        ### Task: 
        tsk = dat[i_d, 'tsk']

        ### State info: 
        lds_state = dat[i_d, 'smooth_state']

        ### For each command compute mean / covariance; 
        state_dict = dict(); # Collect state for each binned mag / ang given task
        state_dict[0] = dict(); 
        state_dict[1] = dict(); 

        ### Go through each trial, sort out the states: 
        for i_t, (tsk_trl, lds_trl) in enumerate(zip(dat[(i_d, 'tsk')], dat[(i_d, 'smooth_state')])): 

            ### Command bins: 
            cb = command_bins[i_t]
            
            for i_b, (bin, lds_bin) in enumerate(zip(cb, lds_trl)): 

                if tuple(bin) not in state_dict[int(tsk_trl)].keys(): 
                    state_dict[int(tsk_trl)][tuple(bin)] = []; 

                ### Now add state info to this
                state_dict[int(tsk_trl)][tuple(bin)].append(lds_bin)

        ### Now go through and stack everything
        for i_t in range(2):
            for i_k, key in enumerate(state_dict[i_t].keys()):
                state_dict[i_t][key] = np.vstack(( state_dict[i_t][key] )) ### n_instances x 15: 

        ### Then go through each key and if enough in task 1 and task 2, then compute sub-index mean and cov: 
        keys = state_dict[0].keys()

        for i_k, key in enumerate(keys):
            n0 = state_dict[0][key].shape[0]

            if key in state_dict[1].keys():
                n1 = state_dict[1][key].shape[0]

                if np.logical_and(n0 > min_observations, n1 > min_observations):
                    
                    ### Proceed with comparisons
                    ix0 = np.random.permutation(n0)
                    ix0_0 = ix0[:int(np.floor(n0/2))]
                    ix0_1 = ix0[int(np.floor(n0/2)):]

                    ix1 = np.random.permutation(n1)
                    ix1_0 = ix1[:int(np.floor(n1/2))]
                    ix1_1 = ix1[int(np.floor(n1/2)):]

                    ### Get MEAN DIFF ###
                    co0 = np.mean(state_dict[0][key][ix0_0], axis=0) # avg across instances
                    co1 = np.mean(state_dict[0][key][ix0_1], axis=0)

                    ob0 = np.mean(state_dict[1][key][ix1_0], axis=0)
                    ob1 = np.mean(state_dict[1][key][ix1_1], axis=0)

                    WIN.append(np.mean(np.abs(co0 - co1)))
                    WIN.append(np.mean(np.abs(ob0 - ob1))) ## Take the mean across the states

                    X.append(np.mean(np.abs(co0 - ob0)))
                    X.append(np.mean(np.abs(co1 - ob1)))

                    ### GET COV DIFF ###
                    co_cov0 = np.cov(state_dict[0][key][ix0_0].T)
                    co_cov1 = np.cov(state_dict[0][key][ix0_1].T)

                    ob_cov0 = np.cov(state_dict[1][key][ix1_0].T)
                    ob_cov1 = np.cov(state_dict[1][key][ix1_1].T)

                    ### Get cov-overlaps ###
                    WIN_cov.append(subspace_overlap.get_overlap(None, None, 
                    first_UUT=co_cov0, second_UUT=co_cov1, main=True))
                    
                    WIN_cov.append(subspace_overlap.get_overlap(None, None, 
                    first_UUT=ob_cov0, second_UUT=ob_cov1, main=True))

                    X_cov.append(subspace_overlap.get_overlap(None, None, 
                    first_UUT=co_cov1, second_UUT=ob_cov1, main=True))

                    X_cov.append(subspace_overlap.get_overlap(None, None, 
                    first_UUT=co_cov0, second_UUT=ob_cov0, main=True))


    return WIN, X, WIN_cov, X_cov

def plot_mFR_diff(WIN, X):
    win = np.hstack((WIN))[:, np.newaxis]
    x = np.hstack((X))[:, np.newaxis] 
    co_obs_tuning_curves_fig3.plot_diff_cov(None, X =x, WIN = win, actually_mean = True, 
        ylim = [0., .05], ystar = .04, ylabel = 'Mean Neural State Difference')

def plot_cov_diff(WIN_cov, X_cov):
    co_obs_tuning_curves_fig3.plot_diff_cov(None, X = X_cov, WIN = WIN_cov)

#############################
### Looking at smoothness ###

def smoothness_PSTH(animal='grom', shuffle_x = 10, compare_to_real = True): 

    ### Taking data, compute a) R2-pred, b) likelihood, etc. PSTH for all trials
    if animal == 'grom':
        dat = pickle.load(open(pref+'gromLDSmodels_nstates_15_combined_models_w_dyn_inno_norms.pkl', 'rb'))
        ndays = 3; 
    elif animal == 'jeev':
        dat = pickle.load(open(pref+'jeevLDSmodels_nstates_15_combined_models_w_dyn_inno_norms.pkl', 'rb'))
        ndays = 4; 

    day_color = ['maroon', 'orangered', 'goldenrod', 'olivedrab', 'teal',
    'steelblue', 'midnightblue', 'darkmagenta', 'brown']

    ### for all trials get mean LL, R2
    for i_d in range(ndays):
        f, ax = plt.subplots(nrows = 4, ncols = 2, figsize = (5, 5))
        PSTH = dict(); 
        PSTH[0] = dict(); ### CO
        PSTH[1] = dict(); ### OBS

        SHUFF = dict()
        SHUFF[0] = dict()
        SHUFF[1] = dict()

        for tsk in range(2):
            PSTH[tsk]['LL'] = []

            ### Used for R2
            PSTH[tsk]['R2_parts'] = [];
            PSTH[tsk]['R2_parts_obs'] = []; 
            PSTH[tsk]['R2_parts_beh'] = []; 

            SHUFF[tsk]['R2_parts'] = []; 
            SHUFF[tsk]['R2_parts_obs'] = []; 
            SHUFF[tsk]['R2_parts_beh'] = []; 
            SHUFF[tsk]['LL'] = [];

        tsk_trls = dat[(i_d, 'tsk')]
        n_trls = len(tsk_trls)
        model = dat[(i_d)]
        day_kg = dat[(i_d, 'decoder_KG')][0]

        ### For each trial: 
        for i_n, tsk in enumerate(tsk_trls):

            ### Get each trial LL: 
            mus = dat[(i_d, 'filt_state')][i_n]
            sigs = dat[(i_d, 'filt_sigma')][i_n]
            npush = dat[(i_d, 'neural_push')][i_n]

            ### observations still have the pre-go in the beginning so need to remove it: 
            obs = dat[(i_d, 'binned_spks')][i_n]
            A = model.A
            C = model.C
            W = model.sigma_states

            PSTH[int(tsk)]['LL'].append(get_likelihood(mus, sigs, A, W))

            ### Now get predictions: 
            R2_st, R2_obs, R2_beh = get_R2_parts(mus, obs, A, C, day_kg, real_npush=npush)
            PSTH[int(tsk)]['R2_parts'].append(R2_st)
            PSTH[int(tsk)]['R2_parts_obs'].append(R2_obs)
            PSTH[int(tsk)]['R2_parts_beh'].append(R2_beh)

            ### Get shuffle data: 
            LL_shuff, R2_shuff, R2_obs_shuff, R2_beh_shuff = get_shuffle_data(shuffle_x, obs, npush, model, day_kg, 
                compare_to_real = compare_to_real, mus = mus)
            SHUFF[int(tsk)]['R2_parts'].append(R2_shuff)
            SHUFF[int(tsk)]['R2_parts_obs'].append(R2_obs_shuff)
            SHUFF[int(tsk)]['R2_parts_beh'].append(R2_beh_shuff)
            SHUFF[int(tsk)]['LL'].append(LL_shuff)

        for tsk in range(2):
            max_T = np.max(np.array([len(L) for L in PSTH[tsk]['LL']]))

            for di, dictz in enumerate([PSTH, SHUFF]):
                col = 'gray'

                for i_a, MET in enumerate([dictz[tsk]['LL'], dictz[tsk]['R2_parts'], dictz[tsk]['R2_parts_obs'],
                    dictz[tsk]['R2_parts_beh']]):

                    ### R2 metrics ###
                    if i_a in [1, 2, 3]:
                        if di == 0:
                            met = np.zeros((n_trls, max_T, 2))
                            met[:] = np.nan
                            for il, L in enumerate(MET): met[il, :len(L), :] = L
                            mean_met = 1 - (np.nansum(met[:, :, 0], axis=0) / np.nansum(met[:, :, 1], axis=0))
                        else:
                            met = np.zeros((n_trls, max_T, 2, shuffle_x,))
                            met[:] = np.nan
                            for il, L in enumerate(MET): met[il, :len(L), :, :] = L
                            ssr = (np.nansum(np.nansum(met[:, :, 0, :], axis=2), axis=0))
                            sst = (np.nansum(np.nansum(met[:, :, 1, :], axis=2), axis=0))
                            mean_met = 1 - (ssr/sst)

                    ### LL metrics ###
                    else:
                        if di == 0:
                            met = np.zeros((n_trls, max_T))
                            met[:] = np.nan
                            for il, L in enumerate(MET): met[il, :len(L)] = L
                            mean_met = np.nanmean(met, axis=0)
                        else:
                            met = np.zeros((n_trls, max_T, shuffle_x))
                            met[:] = np.nan
                            for il, L in enumerate(MET): met[il, :L.shape[1], :] = L.T
                            mean_met = np.nanmean(np.nanmean(met, axis=2), axis=0)

                    if di == 0:
                        ax[i_a, tsk].plot(mean_met, color=day_color[i_d])
                    else:
                        ax[i_a, tsk].plot(mean_met, color=col)

        ax[0, 0].set_title('CO', fontsize=8)
        ax[0, 1].set_title('OBS', fontsize=8)
        ax[0, 0].set_ylabel('LL', fontsize=8)
        ax[1, 0].set_ylabel('R2 - state real ('+str(compare_to_real)+')', fontsize=8)
        ax[2, 0].set_ylabel('R2 - obs real ('+str(compare_to_real)+')', fontsize=8)
        ax[3, 0].set_ylabel('R2 - beh real ', fontsize=8)

        for i in range(2):
            for a in range(1, 4):
                if a in [1, 2]:
                    ax[a, i].set_ylim([0., 1.])
                else:
                    ax[a, i].set_ylim([-1, 1.])

        f.tight_layout()

def smoothness_swap(animal='grom', behav_diff = 'norm', bix = [3, 5], use_state_model = False, ncommands = 1.):
    ### behav_diff can be 'norm' or 'ang'. Norm takes the norm difference, ang takes the acos(dot(a, b))

    ### For each command (32) swap state with another from
    ###     own target/own task
    ###     nearby targets/own task
    ###     other task /own target
    ###     other task/nearby target
    ###     own task far targets
    ###     other task far targets
    mag_boundaries = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/radial_boundaries_fit_based_on_perc_feb_2019.pkl'))

    ### Load a model that has already been fit: 
    if animal == 'grom':
        dat = pickle.load(open(pref+'gromLDSmodels_nstates_15_combined_models_w_dyn_inno_norms.pkl', 'rb'))
        ndays = 1; 
    elif animal == 'jeev':
        dat = pickle.load(open(pref+'jeevLDSmodels_nstates_15_combined_models_w_dyn_inno_norms.pkl', 'rb'))
        ndays = 4; 
    
    SWAP_MASTER = dict()

    for i_d in range(ndays):

        SWAP_METS = dict()
        SWAP_METS['self'] =   dict(LL = [], SSR = [], SST = [], mnt = [], state_diff = [], pred_beh_err = [], pred_beh_diff = [], prev_beh_diff = [], info = [])
        SWAP_METS['own_targ_own_task'] =   dict(LL = [], SSR = [], SST = [], mnt = [], state_diff = [], pred_beh_err = [], pred_beh_diff = [], prev_beh_diff = [], info = [])
        SWAP_METS['near_targ_own_task'] =  dict(LL = [], SSR = [], SST = [], mnt = [], state_diff = [], pred_beh_err = [], pred_beh_diff = [], prev_beh_diff = [], info = [])
        SWAP_METS['far_targ_own_task'] =   dict(LL = [], SSR = [], SST = [], mnt = [], state_diff = [], pred_beh_err = [], pred_beh_diff = [], prev_beh_diff = [], info = [])
        SWAP_METS['own_targ_other_task'] = dict(LL = [], SSR = [], SST = [], mnt = [], state_diff = [], pred_beh_err = [], pred_beh_diff = [], prev_beh_diff = [], info = [])
        SWAP_METS['near_targ_other_task']= dict(LL = [], SSR = [], SST = [], mnt = [], state_diff = [], pred_beh_err = [], pred_beh_diff = [], prev_beh_diff = [], info = [])
        SWAP_METS['far_targ_other_task'] = dict(LL = [], SSR = [], SST = [], mnt = [], state_diff = [], pred_beh_err = [], pred_beh_diff = [], prev_beh_diff = [], info = [])

        ### Info has [own target / task / mag*ang / trl_num]
        tsk = np.hstack((dat[(i_d, 'tsk')]))
        trg = np.hstack((dat[(i_d, 'target')]))

        ### From data: 
        neural_push = [npi[10:, :] for npi in dat[(i_d, 'neural_push')]]
        cursor_state = [cs[10:, :] for cs in dat[(i_d, 'cursor_state')]]

        if use_state_model:
            model = dat[(i_d, 'cs')]
            dyn_B = model.B;
            dyn_D = model.D; 

            state = dat[(i_d, 'filt_state_cs')]
            sigma = dat[(i_d, 'filt_sigma_cs')]
            sigma_ut = dat[(i_d, 'cs_input_cov')]
            
        else:
            model = dat[i_d]
            state = dat[(i_d, 'filt_state')]
            sigma = dat[(i_d, 'filt_sigma')]

        dyn_A = model.A
        dyn_C = model.C
        dyn_W = model.sigma_states
        dyn_Q = model.sigma_obs; 

        day_KG = dat[(i_d, 'decoder_KG')][0]
        
        ### These have already been mean subtracted
        binned_spks = [b[10:, :] for b in dat[(i_d, 'binned_spks')]]

        ### Get magnitude / etc. 
        command_bins = []
        for push in neural_push:

            ### Get the trial: 
            command_bins.append(subspace_overlap.commands2bins(push, mag_boundaries, animal, i_d))

        ### Go through trials: 
        for i_t, (binned, trl_state) in enumerate(zip(command_bins, state)):
            print('Starting Trial %d'%i_t)

            ### ### ### ### ### ### ### ### ####
            ### Get own task and own target: ###
            ### ### ### ### ### ### ### ### ####

            bin_tsk = tsk[i_t]
            bin_targ = trg[i_t]
            mu_trl = state[i_t]
            sig_trl = sigma[i_t]
            curs_st = cursor_state[i_t]

            ### Make sure that this is the correct numer of ts:
            trl_ix = dict(); 

            ### Get trials that are in specific categories: 
            for T, (key, match_task) in enumerate(zip(['own', 'other'], [bin_tsk, np.mod(bin_tsk + 1, 2)])):

                ###### Own trial
                trl_ix['own_targ_'+key+'_task'] = np.nonzero(np.logical_and(tsk == match_task, trg == bin_targ))[0]

                ###### Remove own trial: 
                trl_ix['own_targ_'+key+'_task'] = trl_ix['own_targ_'+key+'_task'][trl_ix['own_targ_'+key+'_task'] != i_t]

                ###### Nearby targets
                targs = np.array([bin_targ - 1, bin_targ + 1])
                targs[targs < 0] = targs[targs < 0] + 8
                targs[targs > 7] = targs[targs > 7] - 8

                trl_ix['near_targ_'+key+'_task'] = np.nonzero(np.logical_and(tsk == match_task, np.logical_or(trg == targs[0], trg == targs[1])))[0]

                ###### Far targs
                ix = []
                for t in range(8): 
                    if t not in targs: 
                        ix.append(np.nonzero(np.logical_and(tsk == match_task, trg == t))[0])
                trl_ix['far_targ_'+key+'_task'] = np.sort(np.hstack((ix)))

        
            ### Iterate through the bins of this particular trial 
            for ib, (st, bn) in enumerate(zip(trl_state, binned)):
                print('Starting Bin %d'%ib)

                # Get a code for this bin: 
                bn = np.squeeze(bn)
                
                ### Avoid the pre-go period [already done above], and the last time point. 
                if np.logical_and(ib >= 0, ib < len(trl_state) - 1): 

                    ### This is the important part!
                    ### Using mu_{t-1}, propogate with dynamics and get prob. distribution associated with propogated state 
                    obs_t_true = binned_spks[i_t][ib,:]

                    mu_tm1_bin = mu_trl[ib - 1, :]
                    sig_tm1_bin = sig_trl[ib - 1, :, :]
                    
                    ### States: 
                    u_t_true_bin = curs_st[ib, :]
                    u_tm1_true_bin = curs_st[ib - 1, :]

                    ### Go through each type of matching:
                    for i_k, key in enumerate(trl_ix.keys()):

                        ### Trial indices we can compare for this type of matching
                        indices = trl_ix[key]

                        ### Shuffled indices
                        indices = indices[np.random.permutation(len(indices))]

                        #keep_searching = True
                        
                        ### Go through each of the indices and look through the trial and if you find a match, stop
                        for ind in indices:
                            #if keep_searching:

                            bn_trl = np.vstack(( command_bins[ind] ))
                            ix = np.nonzero(np.logical_and(bn_trl[:, 0] == bn[0], bn_trl[:, 1] == bn[1]))[0]
                            
                            ### Make sure the found index is in the trial
                            if len(ix) > 0:
                                ix = ix[np.random.permutation(len(ix))]
                                select_nix = int(np.min([len(ix), ncommands]))
                                ix = ix[:select_nix]

                                import pdb; pdb.set_trace()

                                for ix_select in ix:
                                    ### Get the observation associated with this trial
                                    obs_new = binned_spks[ind][ix_select, :]

                                    if use_state_model:
                                        
                                        ### Get the state at this tiem: 
                                        u_t   = cursor_state[ind][ix_select, :]
                                        #u_tm1 = cursor_state[ind][ix[-1]-1, :]

                                        ### Make a prediction for x_t | u_t: using the new swapped u_ts
                                        # x_t = Ax_{t-1} + Bu_{t-1}
                                        mu_tm1_prop = np.dot(dyn_A, mu_tm1_bin)  + np.dot(dyn_B, u_tm1_true_bin) # Use the trials true t-1 u_t, 
                                        ob_tm1_prop = np.dot(dyn_C, mu_tm1_prop) + np.dot(dyn_D, u_t) #only swap the feedthrough u_t

                                        # get real data propogation:L 
                                        mu_t_true_prop = np.dot(dyn_A, mu_tm1_bin) + np.dot(dyn_B, u_tm1_true_bin)
                                        ob_t_true_prop = np.dot(dyn_C, mu_t_true_prop) + np.dot(dyn_D, u_t_true_bin)

                                        ### update the sigma: 
                                        sig_tm1_prop = np.dot(dyn_A, np.dot(sig_tm1_bin, dyn_A.T)) + np.dot(dyn_B, np.dot(sigma_ut, dyn_B.T)) + dyn_W
                                        sig_ob_tm1_prop = np.dot(dyn_C, np.dot(sig_tm1_prop, dyn_C.T)) + np.dot(dyn_D, np.dot(sigma_ut, dyn_D.T)) + dyn_Q; 
                                        
                                        ### using the propagated previous state PLUS SWAPPED u_ts
                                        rv = scipy.stats.multivariate_normal(mean=ob_tm1_prop, cov=sig_ob_tm1_prop);

                                        ### using the progated previous state PLUS OWN u_ts; 
                                        rv_true = scipy.stats.multivariate_normal(mean=ob_t_true_prop, cov=sig_ob_tm1_prop);

                                        ### Mu new: 
                                        ll = rv.logpdf(obs_new); 
                                        ll_self = rv_true.logpdf(obs_t_true)

                                    else:
                                        mu_tm1_prop = np.dot(dyn_A, mu_tm1_bin)
                                        sig_tm1_prop = np.dot(dyn_A, np.dot(sig_tm1_bin, dyn_A.T)) + dyn_W

                                        ob_tm1_prop = np.dot(dyn_C, mu_tm1_prop); 
                                        sig_ob_tm1_prop = np.dot(dyn_C, np.dot(sig_tm1_prop, dyn_C.T)) + dyn_Q; 
                                        
                                        ### In observation space: 
                                        rv = scipy.stats.multivariate_normal(mean=ob_tm1_prop, 
                                            cov=sig_ob_tm1_prop); 

                                        ll = rv.logpdf(obs_new)

                                    ### Add this to the LL pile: 
                                    SWAP_METS[key]['LL'].append(ll)
                                    
                                    ### GET the R2 parts: Difference between the dynamically predicted obs and the actual swapped obs: 
                                    SWAP_METS[key]['SSR'].append((ob_tm1_prop - obs_new)**2)
                                    SWAP_METS[key]['mnt'].append(obs_new)

                                    if use_state_model:
                                        SWAP_METS['self']['SSR'].append((ob_t_true_prop - obs_t_true)**2)
                                        SWAP_METS['self']['mnt'].append(obs_t_true)
                                        SWAP_METS['self']['LL'].append(ll_self)
                                    
                                    else:
                                        SWAP_METS['self']['LL'].append(rv.logpdf(obs_t_true))
                                        SWAP_METS['self']['SSR'].append((ob_tm1_prop - obs_t_true)**2)
                                        SWAP_METS['self']['mnt'].append(obs_t_true)

                        # if keep_searching:
                        #     print('No match, mag %d ang %d, key %s, trl %d bin %d'% (bn[0], bn[1], key, i_t, ib))
        SWAP_MASTER[i_d] = SWAP_METS
    return SWAP_MASTER

### Bar plot of smoothness ###
def plot_SWAP_smoothness(SWAP_METS):
    ### LL and R2

    for i_d in np.sort(SWAP_METS.keys()):

        SWAP_METS_day = SWAP_METS[i_d]

        f, ax = plt.subplots(nrows = 2, figsize = (5, 8))

        nms = ['self', 'own_targ_own_task', 'near_targ_own_task', 'own_targ_other_task', 'near_targ_other_task',
            'far_targ_own_task', 'far_targ_other_task']

        colors = ['purple', 'k', 'b', 'k', 'b', 'r', 'r']
        alpha =  [1., 1. ,  1.,  .5,  .5,  1., .5 ]

        for i_n, (nm, col, alph) in enumerate(zip(nms, colors, alpha)):
            LL = np.hstack((SWAP_METS_day[nm]['LL']))
            ax[0].bar(i_n, np.mean(LL), color = col, alpha=alph)
            ax[0].errorbar(i_n, np.mean(LL), np.std(LL)/np.sqrt(len(LL)), marker='|', color='k')
            
            # Sum over observations. 
            SSR = np.sum(np.vstack((SWAP_METS_day[nm]['SSR'])), axis=0)

            # Get teh mean: 
            mnt = np.vstack((SWAP_METS_day[nm]['mnt']))
            mu = np.mean(mnt, axis=0)

            SST = np.sum((mnt - mu[np.newaxis, :])**2, axis=0)

            #### individual neuron mean ###
            R2 = 1 - (SSR / SST)
            ax[1].bar(i_n, np.mean(R2), color = col, alpha=alph)

        ax[1].set_xticks(np.arange(len(nms)))
        ax[1].set_xticklabels(nms, rotation=90)

        ax[0].set_ylabel('Likelihood')
        ax[1].set_ylabel('R2 of swapped state vs. dyn prediction from t-1')

### Correlation plot of state diff vs. behavior prediction error
def plot_SWAP_state_diff(SWAP_METS, plot_tasks = [1], use_gmm_to_sep = False, 
    perc_thresh = 90):
    
    #for i_d in np.sort(SWAP_METS.keys()):
    for i_d in [0]:

        SWAP_METS_day = SWAP_METS[i_d]
        f, ax = plt.subplots(ncols = 3, figsize = (8, 8), nrows = 6)

        nms = ['own_targ_own_task', 'near_targ_own_task', 'own_targ_other_task', 'near_targ_other_task',
            'far_targ_own_task', 'far_targ_other_task']

        colors = ['k', 'b', 'k', 'b', 'r', 'r']
        alpha =  [1. ,  1.,  .5,  .5,  1., .5 ]

        # X = []; 
        # Y = []; 
        # Y2 = []; 
        # Y3 = []; 
        SWAP = SWAP_METS_day['own_targ_own_task']
        swap_info_LL = np.vstack(( SWAP['info'] ))[:, -1]

        if use_gmm_to_sep:
            ### Find the swapped points that have high LL ###
            ftest, axtest = plt.subplots()
            cnt, binz = np.histogram(swap_info_LL, 40)

            ### Plot histogram of data: 
            axtest.plot(binz[1:] + 0.5*(binz[1] - binz[0]), cnt / float(np.sum(cnt)), 'k-')
            gmm = mixture.GaussianMixture(n_components=2, covariance_type='full')
            gmm.fit(swap_info_LL.reshape(-1, 1))

            x = np.linspace(np.min(swap_info_LL), np.max(swap_info_LL), 1000.)
            rv0 = scipy.stats.multivariate_normal(gmm.means_[0], gmm.covariances_[0, :, :])
            rv1 = scipy.stats.multivariate_normal(gmm.means_[1], gmm.covariances_[1, :, :])
            axtest.plot(x, rv0.pdf(x), 'r-')
            axtest.plot(x, rv1.pdf(x), 'b-')
            ftest.legend(['Real', 'GMM1', 'GMM2'])

            ### Which mean is higher? 
            if gmm.means_[0] > gmm.means_[1]:
                kix = 0 
            else:
                kix = 1

            ylabs = gmm.predict(swap_info_LL.reshape(-1, 1))
            keep_ix = np.nonzero(ylabs == kix)[0]
        else: 
            thresh = np.percentile(swap_info_LL, perc_thresh)
            keep_ix = np.nonzero(swap_info_LL >= thresh)[0]
            print("Thresh: %f" %thresh)

        for i_n, (nm, col, alph) in enumerate(zip(nms, colors, alpha)):
            X = []; X2 = []; 
            Y = []; 
            Y2 = []; 
            Y3 = []; 

            SWAP = SWAP_METS_day[nm]
            prev_beh_diff = np.hstack(( SWAP['prev_beh_diff'] ))
            state_diff = np.hstack(( SWAP['state_diff'] ))
            pred_beh_err = np.hstack(( SWAP['pred_beh_err'] ))
            pred_beh_diff = np.hstack(( SWAP['pred_beh_diff'] ))
            swap_info = np.vstack(( SWAP['info'] )) ### own task / target / mag*ang / trl_num / likelihood of next step

            for i_tsk in plot_tasks:
                ix = np.nonzero(swap_info[:, 0] == i_tsk)[0]
                targs = np.unique(swap_info[ix, 1])

                for i_targ in targs:
                    ix2 = np.nonzero(swap_info[ix, 1] == i_targ)[0]

                    mags = np.unique(swap_info[ix[ix2], 2])

                    for i_m in mags: 
                        ix3 = np.nonzero(swap_info[ix[ix2], 2] == i_m)[0]

                        ### Indices for task / target / mag bin
                        ix_avg = ix[ix2[ix3]]
                        ix_avg_keep = np.array([i for i in ix_avg if i in keep_ix])

                        mn_prev_beh_diff = np.mean(prev_beh_diff[ix_avg])
                        mn_state_diff = np.mean(state_diff[ix_avg])
                        mn_pred_beh_diff = np.mean(pred_beh_diff[ix_avg])

                        ### Only use the keep ix: 
                        if len(ix_avg_keep) > 0:
                            mn_state_diff_keep = np.mean(state_diff[ix_avg_keep])
                            mn_pred_beh_err = np.mean(pred_beh_err[ix_avg_keep])
                            
                            X2.append(mn_state_diff_keep)
                            Y.append(mn_pred_beh_err)
                            ax[i_n, 0].plot(mn_state_diff_keep, mn_pred_beh_err, '.', color=col, alpha = alph)

                        X.append(mn_state_diff)                        
                        Y2.append(mn_prev_beh_diff)
                        Y3.append(mn_pred_beh_diff)

                        ax[i_n, 1].plot(mn_state_diff, mn_prev_beh_diff, '.', color=col, alpha = alph)
                        ax[i_n, 2].plot(mn_state_diff, mn_pred_beh_diff, '.', color=col, alpha = alph)

            X = np.hstack((X))
            Y2 = np.hstack((Y2))
            Y3 = np.hstack((Y3))

            X2 = np.hstack((X2))
            Y = np.hstack((Y))

            for iy, (y, ynm) in enumerate(zip([Y, Y2, Y3], ['Prop Beh Error\nOnly high LL', 'Prev Beh Diff', 'Prop Beh Diff'])):
                
                if iy == 0:
                    slp, intc, rv, pv, err = scipy.stats.linregress(X2, y)
                    xmin = np.min(X2)
                    xmax = np.max(X2)
                else:
                    slp, intc, rv, pv, err = scipy.stats.linregress(X, y)
                    xmin = np.min(X)
                    xmax = np.max(X)

                xint = np.arange(xmin, xmax, .1)
                ax[i_n, iy].plot(xint, slp*xint + intc, 'k--')
                ax[i_n, iy].set_title('R: %.2f, PV: %.4f' %(rv, pv), fontsize=8)
                ax[i_n, iy].set_ylabel(ynm, fontsize=8)
                ax[i_n, iy].set_xlim([0., 1.8])
                if iy == 0:
                    ax[i_n, iy].set_ylim([-.75, .75])
                else:
                    ax[i_n, iy].set_ylim([0, 3])

                if i_n == 5:
                    ax[i_n, iy].set_xlabel('Mean State Diff', fontsize=8)

### Plot the distributions of propogated commands for tasks / command  ###
def prediction_beh_dist(animal='grom', bix = [3, 5], testing_A_prop = False, plot_distribution_disks = False,
    plot_joint_dist = True, ang_thresh = 2*np.pi/8., error_thresh = 2*np.pi/64.):

    mag_boundaries = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/radial_boundaries_fit_based_on_perc_feb_2019.pkl'))

    useful_points = dict(task = [], targ = [], trl_num_overall = [], trl_num_task = [], bin = [], dig_ang = [], dig_mag = [],
        next_t_ang_true = [], next_t_ang_pred = [], t_ang_true = [])

    if animal == 'grom':
        dat = pickle.load(open(pref+'gromLDSmodels_nstates_15_combined_models_w_dyn_inno_norms.pkl', 'rb'))
        ndays = 1; 

    elif animal == 'jeev':
        dat = pickle.load(open(pref+'jeevLDSmodels_nstates_15_combined_models_w_dyn_inno_norms.pkl', 'rb'))
        ndays = 1; 

    SWAP_MASTER = dict()

    for i_d in range(ndays):

        ### Task
        tsk = np.hstack((dat[(i_d, 'tsk')]))

        ### Target
        trg = np.hstack((dat[(i_d, 'target')]))

        ### Filtered State / Sigma
        state = dat[(i_d, 'filt_state')]
        sigma = dat[(i_d, 'filt_sigma')]

        ### Neural Push
        neural_push = dat[(i_d, 'neural_push')]

        ### Model
        model = dat[i_d]
        dyn_A = model.A
        dyn_C = model.C
        dyn_W = model.sigma_states
        day_KG = dat[(i_d, 'decoder_KG')][0]
        
        ### Get magnitude / etc. 
        command_bins = []
        for push in neural_push:
            command_bins.append(subspace_overlap.commands2bins(push, mag_boundaries, animal, i_d))

        ### Distribution count: 
        dist_cnt = dict()
        dist_rl = dict()

        for task in range(2):
            for targ in range(8):
                for mag in range(4):
                    for ang in range(8):
                        dist_cnt[task, targ, mag, ang] = np.zeros((4, 8))
                        dist_rl[task, targ, mag, ang] = np.zeros((4, 8))

        ### Setup a plot for each bin: 
        if not testing_A_prop:
            ax_dict = dict()
            f_dict = dict()
            for mag in range(4):
                for ang in range(8):

                    if plot_distribution_disks:
                        f_, ax_ = plt.subplots(ncols = 8, nrows = 4, figsize = (10, 5))
                    
                    elif plot_joint_dist:
                        f_, ax_ = plt.subplots(ncols = 8, nrows = 2, figsize = (10, 2.5))
                        task_colors = ['k', 'b']

                    ax_dict[mag, ang] = ax_
                    f_dict[mag, ang] = f_

        ### For each of these trials
        for i_t, (binned, trl_state) in enumerate(zip(command_bins, state)):

            ###################################
            ### Get own task and own target: ##
            ###################################
            bin_tsk = tsk[i_t]
            bin_targ = trg[i_t]

            ### State for this trial ###
            mu_trl = state[i_t]

            ### For each bin, 
            for ib, (st, bn) in enumerate(zip(trl_state, binned)):
                bn = np.squeeze(bn)

                ### Avoid the pre-go period: 
                if np.logical_and(ib >= 10, ib < len(trl_state) - 1): 

                    ### Mu/sigmas for this bin
                    mu_t_bin = mu_trl[ib, :]

                    ### Propogate fwd
                    mu_tp1_prop = np.dot(dyn_A, mu_t_bin)

                    ### Get observations 
                    obs_tp1_prop = np.dot(dyn_C, mu_tp1_prop)

                    ### Real data [interlude]
                    np_tp1 = np.squeeze(np.array(neural_push[i_t][ib+1, :]))
                    np_tp1 = [np.hstack((np_tp1))[np.newaxis, :]]
                    rl_bins = subspace_overlap.commands2bins(np_tp1, mag_boundaries, animal, i_d)
                    rl_bins = rl_bins[0]
                    dist_rl[bin_tsk, bin_targ, bn[0], bn[1]][rl_bins[0, 0], rl_bins[0, 1]] += 1

                    ### Decode with decoder.
                    np_tp1_prop = np.dot(day_KG, obs_tp1_prop)
                    np_tp1_prop = [np.hstack((np_tp1_prop))[np.newaxis, :]]

                    ## Bin this: 
                    tmp_bins = subspace_overlap.commands2bins(np_tp1_prop, mag_boundaries, animal, i_d)
                    tmp_bins = tmp_bins[0]
                    dist_cnt[bin_tsk, bin_targ, bn[0], bn[1]][tmp_bins[0, 0], tmp_bins[0, 1]] += 1

                    if plot_joint_dist: 

                        ### Plot real next angle: 
                        ang = math.atan2(neural_push[i_t][ib+1, 5], neural_push[i_t][ib+1, 3])

                        ### Is this real angle different from the same bin we're already in? 
                        A_mod = np.linspace(-np.pi/8, 2*np.pi - np.pi/8, 9)
                        mn_ang = np.mean([A_mod[bn[1]], A_mod[bn[1]+1]])
                        if mn_ang > np.pi: 
                            mn_ang = mn_ang - 2*np.pi
                        t_diff = ang_difference(np.array([ang]), np.array([mn_ang]))

                        ### Plot prop next angle minus real angel; 
                        ang_prop = math.atan2(np_tp1_prop[0][0, 5], np_tp1_prop[0][0, 3])

                        ### Get the angular difference: 
                        diff = ang_difference(np.array([ang]), np.array([ang_prop]))


                        ### Is this a good example of angle change + good prediciton? 
                        if np.logical_and(np.abs(t_diff) > ang_thresh, np.abs(diff) < error_thresh):

                            if bn[0] > 1:
                                useful_points['task'].append(bin_tsk)
                                useful_points['targ'].append(bin_targ)
                                useful_points['trl_num_overall'].append(i_t)
                                useful_points['bin'].append(ib)
                                useful_points['dig_ang'].append(bn[1])
                                useful_points['dig_mag'].append(bn[0])
                                useful_points['next_t_ang_true'].append(ang)
                                useful_points['next_t_ang_pred'].append(ang_prop)

                            print 'Adding a useful point!'

                        ### Plot real angle, then plot real next step: 
                        ax_dict[bn[0], bn[1]][int(bin_tsk), int(bin_targ)].plot(ang, diff, '.', 
                            color = task_colors[int(bin_tsk)], markersize=2.5)

                        #import pdb; pdb.set_trace()
                    if testing_A_prop:
                        ### Check if magnitude jumps up compared to real bin -- 
                        #if np.linalg.norm(np_tp1_prop[0][0, :]) > np.linalg.norm(neural_push[i_t][ib, :]):
                        '''
                        trying to figure out how velocities can increase in mag if state is primarily using 
                        decay axes

                        answer: imagine decaying axes where x2 (yaxis) decays faster than x1 (xaxis). Then consider
                            if the decoder is the y = -x axis. start at (x1,x2) = (1,1). Then proceed to 
                            (.9, .7), (.8, .4), (.7, .1) and look at projections on the x axis. 
                        '''

                        ### Do a change of axes: 
                        ### Get the eigenvalues: 
                        ev, evect = np.linalg.eig(dyn_A)
                        ix_ev_sort = np.argsort(np.abs(ev))[::-1]
                        ev_sort = ev[ix_ev_sort]

                        ### Get the columns of T in the correct order: 
                        T = evect[:, ix_ev_sort]
                        TI = np.linalg.inv(T)

                        ### Get propogation of x_t
                        x_t = np.dot(TI, mu_t_bin)
                        x_t1 = np.dot(np.dot(TI, np.dot(dyn_A, T)), x_t)

                        ### Also check if any cases where norm x_t1 > norm x_t
                        if np.linalg.norm(x_t) < np.linalg.norm(x_t1):
                            import pdb; pdb.set_trace()



        ### Plot the distribution: 
        A = np.linspace(0., 2*np.pi, 9) + np.pi/8 
        A_mod = np.linspace(-np.pi/8, 2*np.pi - np.pi/8, 9)
        R = np.array([0]+mag_boundaries[animal, i_d]+[mag_boundaries[animal, i_d][2]+1])

        if plot_distribution_disks:
            for mag in range(4):
                for ang in range(8):
                    ax_plot = ax_dict[mag, ang]

                    for task in range(2):
                        for targ in range(8):
                            pco = dist_cnt[task, targ, mag, ang] / float(np.sum(dist_cnt[task, targ, mag, ang]))
                            prl = dist_rl[task, targ, mag, ang] / float(np.sum(dist_rl[task, targ, mag, ang]))
                        
                            ### Distribution: 
                            co_obs_tuning_curves_fig3.polar_plot(R, A, pco, ax=ax_plot[task, targ], vmin=0, vmax=.25, cmap='Greys')
                            
                            ## Plot the red dot: 
                            mn_R = np.mean([R[mag], R[mag+1]])
                            mn_ang = np.mean([A[ang], A[ang+1]])
                            ax_plot[task, targ].plot(mn_R*np.cos(mn_ang), mn_R*np.sin(mn_ang), 'r.')
                            ax_plot[task, targ].set_title('Task %d, Targ, %d, Prop ' %(task, targ), fontsize=6)

                            co_obs_tuning_curves_fig3.polar_plot(R, A, prl, ax=ax_plot[2 + task, targ], vmin=0, vmax=.25, cmap='Greys')
                            ax_plot[2 + task, targ].plot(mn_R*np.cos(mn_ang), mn_R*np.sin(mn_ang), 'r.')
                            ax_plot[2 + task, targ].set_title('Task %d, Targ, %d, Real ' %(task, targ), fontsize=6)
                    
                    f_dict[mag, ang].tight_layout()
                    f_dict[mag, ang].savefig('/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/BMI_co_obs_paper/data/real_vs_prop_dist_disks_mag_%d_ang_%d.png' %(mag, ang))
                    plt.close(f_dict[mag, ang])

        elif plot_joint_dist:
            for mag in range(4):
                for ang in range(8):
                    mn_ang = np.mean([A_mod[ang], A_mod[ang+1]])

                    ### Remap from [0, 2*np.pi] --> [-np.pi, np.pi]
                    if mn_ang > np.pi: 
                        mn_ang = mn_ang - 2*np.pi; 

                    for task in range(2):
                        for targ in range(8):
                            axi = ax_dict[mag, ang][task, targ]
                            axi.vlines(mn_ang, -np.pi, np.pi, 'g', linewidth=.5)
                            axi.hlines(0, -np.pi, np.pi, 'g', linewidth=.5)
                            axi.set_title('Task %s, Targ %s' %(task, targ), fontsize=8)
                            if task == 0 and targ == 0:
                                axi.set_title('Mag %d, Ang %d' %(mag, ang), fontsize=8)
                    f_dict[mag, ang].tight_layout()
                    f_dict[mag, ang].savefig('/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/BMI_co_obs_paper/data/real_prop_jt_dist_mag_%d_ang_%d.png' %(mag, ang))
                    
    return useful_points

def plot_useful_pts(useful_points):
        
    ### Go through each trial, make a plot!
    dat = pickle.load(open(pref+'gromLDSmodels_nstates_15_combined_models_w_dyn_inno_norms.pkl', 'rb'))
    co_obs_dict = pickle.load(open(pref+'co_obs_file_dict.pkl'))

    day = 0; 
    binsize_ms = 100.
    drives_neurons_ix0 = 3
    pre_go = 1.
    key = 'spike_counts'

    model = dat[day]
    dyn_A = model.A; 
    dyn_C = model.C; 

    cursor_pos = []; 
    cursor_state = []; 

    for te_num in [4377, 4378, 4382]: 

        ### Open task data file
        hdf = co_obs_dict[te_num, 'hdf']
        hdfix = hdf.rfind('/')
        hdf = tables.openFile(pref+hdf[hdfix:])

        ### Open decoder file
        dec = co_obs_dict[te_num, 'dec']
        decix = dec.rfind('/')
        decoder = pickle.load(open(pref+dec[decix:]))

        ### Get steady state kalman filter matrices
        F, KG = decoder.filt.get_sskf()
    
        ### Get task indices for rewarded trials
        rew_ix = np.array([t[1] for it, t in enumerate(hdf.root.task_msgs[:]) if t[0]=='reward'])
    
        ### Extract only cursor pos
        _, _, _, _, cursor_p = pa.extract_trials_all(hdf, rew_ix, neural_bins = binsize_ms,
            drives_neurons_ix0=drives_neurons_ix0, hdf_key=key, keep_trials_sep=True,
            reach_tm_is_hdf_cursor_pos=True, reach_tm_is_kg_vel=False, include_pre_go= pre_go, **dict(kalman_gain=KG))

        ### Extract full cursor state (pos, vel)
        _, _, _, _, cursor_s = pa.extract_trials_all(hdf, rew_ix, neural_bins = binsize_ms,
            drives_neurons_ix0=drives_neurons_ix0, hdf_key=key, keep_trials_sep=True,
            reach_tm_is_hdf_cursor_pos=False, reach_tm_is_hdf_cursor_state=True, 
            reach_tm_is_kg_vel=False, include_pre_go= pre_go, **dict(kalman_gain=KG))

        cursor_pos.extend([p for p in cursor_p if len(p) > 0])
        cursor_state.extend([s for s in cursor_s if len(s) > 0])

    ### Double check trials
    for i in range(len(cursor_pos)):
        assert(len(cursor_pos[i]) == len(cursor_state[i]) == len(dat[0, 'neural_push'][i]))
    neural_push = dat[0, 'neural_push']

    ### Which trials have these points? 
    unique_trials = np.unique(useful_points['trl_num_overall'])

    max_plots = np.min([len(unique_trials), 20])
    ix_to_plot = unique_trials[np.random.permutation(len(unique_trials))[:max_plots]]

    ### Get neural state space: 
    neural_state = dat[day, 'filt_state']

    for trl in ix_to_plot:
        cursor_state_trl = cursor_state[trl]
        neural_push_trl = neural_push[trl]
        neural_state_trl = neural_state[trl]

        ix_trl = np.nonzero(useful_points['trl_num_overall'] == trl)[0]
        bins = np.hstack((useful_points['bin']))[ix_trl]

        # Which bins are remarkable
        plot_trial_data(cursor_state_trl, neural_push_trl, bins, trl)

        plot_state_spaci(dyn_A, dyn_C, KG, neural_state_trl, bins)

def distribution_of_neural_var_exp_by_eigs(dt = .1):
    dat = pickle.load(open(pref+'gromLDSmodels_nstates_15_combined_models_w_dyn_inno_norms.pkl', 'rb'))
    i_d = 0; 
    cols = ['steelblue', 'midnightblue']
    marker = ['.', '*']

    ff, axf = plt.subplots()
    ndays = dict(grom=9, jeev=4)

    for i_a, animal in enumerate(['grom', 'jeev']):
        for i_d in range(ndays[animal]):

            ### Task
            tsk = np.hstack((dat[(i_d, 'tsk')]))

            ### Target
            trg = np.hstack((dat[(i_d, 'target')]))

            ### Filtered State / Sigma
            state = dat[(i_d, 'filt_state')]
            sigma = dat[(i_d, 'filt_sigma')]

            ### Neural Push
            neural_push = dat[(i_d, 'neural_push')]

            ### Model
            model = dat[i_d]
            dyn_A = model.A
            dyn_C = model.C
            day_KG = dat[(i_d, 'decoder_KG')][0]
            day_KG_vel = day_KG[[3, 5], :]
            nstates = dyn_A.shape[0]

            ### Get the eigenvalues: 
            ev, evect = np.linalg.eig(dyn_A)
            ix_ev_sort = np.argsort(np.abs(ev))[::-1]
            ev_sort = ev[ix_ev_sort]

            ### Get the columns of T in the correct order: 
            T = evect[:, ix_ev_sort]
            TI = np.linalg.inv(T)

            ### Get labels for each ev: 
            Tau = -1./np.log(np.abs(ev_sort))*dt*1000. 
            
            ## Get angs of A matrix eigenvalues
            angs = np.array([ np.arctan2(np.imag(ev_sort[i]), np.real(ev_sort[i])) for i in range(len(ev_sort))])
            angs[angs==np.pi] = 0.

            ## Get the frequency: rad / dt --> cycle / dt --> cycle / sec
            Hz = angs/(2*np.pi*dt)

            axf.plot(Tau, Hz, marker[i_a], color = cols[i_a])

            ### Make sure that this matches: 
            assert np.allclose(np.dot(T, TI), np.eye(nstates))
            assert np.allclose(np.dot(TI, T), np.eye(nstates))

            ### Confirm this is correct: 
            Lambda = np.diag(ev_sort)
            assert np.allclose(np.dot(dyn_A, T), np.dot(T, Lambda))

            ### PLot this: 
            f, ax = plt.subplots(ncols = 2)

            cols = ['k', 'b']

            for task in range(2):
                
                ### Get the task index
                ix = np.nonzero(tsk == task)[0]

                ### Get the states: 
                states_all = np.vstack(( [s for i_s, s in enumerate(state) if tsk[i_s] == task] )) # T x nstates

                ### Now transform these states: 
                state_trans = np.dot(TI, states_all.T).T # T x nstates

                ### Now get covariance: 
                cov_state_trans = np.cov(state_trans.T)

                ### Now get the neural covariance: 
                L = np.dot(dyn_C, T)
                R = np.dot(T.T, dyn_C.T)
                y_cov = np.dot(L, np.dot(cov_state_trans, R))
                tot_y = np.trace(np.abs(y_cov))

                ### Vel cov: 
                Lk = np.dot(day_KG_vel, L)
                Rk = np.dot(R, day_KG_vel.T)
                v_cov = np.dot(Lk, np.dot(cov_state_trans, Rk))
                tot_v = np.trace(np.abs(v_cov))

                ### Get variance for each state individually: 
                y_cov_list = []; v_cov_list = []; 

                xlab = []; 

                for n in range(nstates):
                    cov_zerod = np.zeros_like(cov_state_trans)
                    cov_zerod[n, n] = cov_state_trans[n, n]
                    y_cov_i = np.trace(np.dot(L, np.dot(cov_zerod, R)))
                    v_cov_i = np.trace(np.dot(Lk, np.dot(cov_zerod, Rk)))

                    y_cov_list.append(y_cov_i)
                    v_cov_list.append(v_cov_i)

                tot_y = np.sum(y_cov_list); 
                tot_v = np.sum(v_cov_list)

                for n in range(nstates):
                    ax[0].plot(n, np.abs(y_cov_list[n])/tot_y, '.', color=cols[task])
                    ax[1].plot(n, np.abs(v_cov_list[n])/tot_v, '.', color=cols[task])  
                    xlab.append('T %.2f, Hz %.2f' %(Tau[n], Hz[n]))

                ax[0].set_xticks(np.arange(nstates))
                ax[0].set_xticklabels(xlab, fontsize=10, rotation=90)
                ax[0].set_title('Neural Variance')
                ax[1].set_xticks(np.arange(nstates))
                ax[1].set_xticklabels(xlab, fontsize=10, rotation=90)
                ax[1].set_title('Neural Push Variance')

            f.tight_layout()

#############################
### Looking at Utils ###

def get_behav_diff(a, b, method):
    if method == 'norm':
        return np.linalg.norm(a - b)
    
    elif method == 'ang':
        norma = a / np.linalg.norm(a)
        normb = b / np.linalg.norm(b)
        return np.arccos(np.dot(norma,normb))

    elif method == 'mag':
        return np.abs(np.linalg.norm(a) - np.linalg.norm(b))

def get_likelihood(mus, sigmas, A, W):
    T = len(mus) - 1; 
    LL = []; 

    for t in range(T): 

        mu_t = mus[t, :]
        mu_t1 = mus[t+1, :]
        sig_t = sigmas[t, :, :]

        ### Propagate forward: 
        mu_t_prop = np.dot(A, mu_t.T).T
        sig_t_prop = np.dot(A, np.dot(sig_t, A.T)) + W

        ## Create a random variable, assess the next step: 
        rv = scipy.stats.multivariate_normal(mean=mu_t_prop, cov=sig_t_prop);
        
        ## Get LL of this guy: 
        ll = rv.logpdf(mu_t1)
        LL.append(ll)

    return np.hstack((LL))

def get_R2_parts(mus, obs, A, C, day_kg, real_mu = None, 
    real_obs = None, real_npush = None):
    ''' get R2 for state and for neural activity '''

    T = len(mus) - 1; 
    R2 = []; 
    R2_obs = []; 
    R2_beh = []; 

    for t in range(T): 
        mu_t = mus[t, :]

        ### Real mu ###
        if real_mu is None: 
            mu_t1 = mus[t+1, :]
        else:
            mu_t1 = real_mu[t+1, :]

        ### Real observations ###
        if real_obs is None:
            obs_t1 = obs[t+1, :]
        else:
            obs_t1 = real_obs[t+1, :]

        ### Real Behavior ###
        if real_npush is None:
            raise Exception('No npush here')
        else:
            npush_t1 = np.squeeze(np.array(real_npush[t+1, [3, 5]]))

        ### Propagate forward: 
        mu_t_prop = np.dot(A, mu_t.T).T
        obs_t_prop = np.dot(C, mu_t_prop)
        beh_t_prop = np.dot(day_kg, obs_t_prop)
        beh_t_prop = beh_t_prop[[3, 5]]

        ### What is error of mu_t_prop? 
        ssr = np.sum((mu_t_prop - mu_t1)**2)
        sst = np.sum(mu_t1**2)
        R2.append([ssr, sst])

        ### do for observations: 
        ssr_obs = np.sum((obs_t_prop - obs_t1)**2)
        sst_obs = np.sum(obs_t1**2)
        R2_obs.append([ssr_obs, sst_obs])

        ### do for behavior: 
        ssr_beh = np.sum((npush_t1 - beh_t_prop)**2)
        sst_beh = np.sum(npush_t1**2)
        R2_beh.append([ssr_beh, sst_beh])
        
    return np.vstack((R2)), np.vstack((R2_obs)), np.vstack((R2_beh))

def get_shuffle_data(shuffle_x, binned_spks, npush, model, day_kg, compare_to_real = True, mus = None):
    ''' 
    goal is to shuffle the neurons within their own time bins. 
    This tells us whether the dynamics are actually predicting
    anything more than just a consistent value. Keep shuffle only
    in the time dimension, otherwise probably off-manifold 
    '''
    LL_shuff = []; 
    R2_shuff = []; 
    R2_obs_shuff = []; 
    R2_beh_shuff = []; 

    ### For each shuffle:
    for x in range(shuffle_x):

        ### Shuffle: 
        N = binned_spks.shape[0]
        bs_shuff = binned_spks[np.random.permutation(N), :]

        ### Add this to the model, then pop it off:
        model.add_data(bs_shuff)
        g = model.states_list.pop()

        smoothed_trial = g.smooth()
        x0 = g.smoothed_mus[0, :] # Time x ndim
        P0 = g.smoothed_sigmas[0, :, :]

        ### Get filtered mus / sigmas of this 
        _, filtered_mus, filtered_sigmas = kalman_filter(
            x0, P0,
            g.A, g.B, g.sigma_states,
            g.C, g.D, g.sigma_obs,
            g.inputs, g.data)

        ### Get smoothness metrics: 
        LL_shuff.append(get_likelihood(filtered_mus, filtered_sigmas, g.A, g.sigma_states))
        
        if compare_to_real:
            r2_real, r2_obs_real, r2_beh_real = get_R2_parts(filtered_mus, bs_shuff, g.A, g.C, day_kg,
                real_mu = mus, real_obs = binned_spks, real_npush = npush)
            R2_obs_shuff.append(r2_obs_real)
            R2_shuff.append(r2_real)
        else:
            r2, r2_obs, r2_beh_real = get_R2_parts(filtered_mus, bs_shuff, g.A, g.C, day_kg, 
                real_npush = npush)
            R2_obs_shuff.append(r2_obs)
            R2_shuff.append(r2)

        R2_beh_shuff.append(r2_beh_real)

    ### Now stack and take the mean: 
    return np.vstack((LL_shuff)), np.dstack((R2_shuff)), np.dstack((R2_obs_shuff)), np.dstack((R2_beh_shuff))

def plot_trial_data(cursor_state_trl, neural_push_trl, bins, trl_num):

    f, ax = plt.subplots(figsize = (8, 8))
    ax.set_title('Trl %d' %trl_num)
    ax.plot(cursor_state_trl[10:, 0], cursor_state_trl[10:, 1], '-', color='gray', linewidth=.5)

    for b in bins: 
        ax.plot(cursor_state_trl[b, 0], cursor_state_trl[b, 1], 'g*', alpha=0.5, markersize=30)


    pos_offset = np.array([ 0.01767333, -0.01161837])

    ### Setup the trial: 
    ### Plot from 10 --> end of trial
    curs_pos = cursor_state_trl[10:-1, [0, 1], 0] ### Plot at T, what happens at T+1
    curs_vel = cursor_state_trl[9:-2, [2, 3], 0] ### Lagged by 1
    npsh_vel = neural_push_trl[10:-1, [3, 5]] ### Normal
    npsh_pos = neural_push_trl[11:, [0, 2]] ### Plus 1 because directly added



    kwargs = dict(scale_units = 'inches', scale=.7, angles='uv', headwidth=3.0, width=.003)

    arrows = []; 

    for vi, (cs_p, cs_v, npsh_v, npsh_p) in enumerate(zip(curs_pos, curs_vel, npsh_vel, npsh_pos)):

        x = cs_p[0]; y = cs_p[1]; 

        ### Plot integrated, decayed vel from t-1 part of F*F*v_{t-1}
        plt.quiver(cs_p[0], cs_p[1], 0.07*0.5*cs_v[0], 0.07*0.5*cs_v[1], color='k', **kwargs)
        x+= 0.07*0.5*cs_v[0]
        y+= 0.07*0.5*cs_v[1]

        ### Plot neural push: integrated KG from t: F*KGv*y_t
        plt.quiver(cs_p[0], cs_p[1], 0.07*npsh_v[0,0], 0.07*npsh_v[0,1], color='b', **kwargs)
        x+=  0.07*npsh_v[0,0]
        y+= 0.07*npsh_v[0,1]

        ### Plot position (backed up in time): KGp*y_{t+1}
        plt.quiver(cs_p[0], cs_p[1], npsh_p[0,0], npsh_p[0,1], color='r', **kwargs)
        x+= npsh_p[0,0]
        y+= npsh_p[0,1]

        ### Plot offset? 
        plt.quiver(cs_p[0], cs_p[1], pos_offset[0], pos_offset[1], color='maroon', **kwargs)
        x+= pos_offset[0]
        y+= pos_offset[1]

        #print x - curs_pos[vi+1, 0], y- curs_pos[vi+1, 1]

    ax.set_aspect('equal', 'datalim')

def resim_trial_data():
    curs_state_est = [] 
    curs = cursor_state_trl[9, :, 0]
    curs = np.mat([curs[0], 0., curs[1], curs[2], 0., curs[3], 1.]).T

    for i in range(10, 55):
        curs = np.squeeze(np.array(np.dot(F, curs))) + np.dot(KG, spks_trl[i])
        curs_state_est.append(curs)
    curs_state_est = np.vstack((curs_state_est))

    plt.plot(curs_state_est[:, 0], curs_state_est[:, 2])
    plt.plot(cursor_state_trl[10:, 0, 0], cursor_state_trl[10:, 1, 0] )

def plot_state_spaci(dyn_A, dyn_C, KG, neural_state_trl, bins):

    color_pairs = [['forestgreen', 'black'], ['limegreen','grey'], ['lightgreen','lightgray'], ['purple', 'k'],['mediumorchid','grey'],['slateblue','lightgray']]

    ### First just plot the state space: 
    f, ax = plt.subplots(ncols = 3, nrows = 3, figsize = (9, 9))
    #f, ax = plt.subplots()

    ev, evect = np.linalg.eig(dyn_A)

    ### Sort eigenvectors: 
    ev_ix = np.argsort(ev)[::-1]

    ### Re-order by sorting
    ev = ev[ev_ix]
    T = evect[:, ev_ix]
    TI = np.linalg.inv(T)

    A_mod = np.dot(TI, np.dot(dyn_A, T))
    C_mod = np.dot(dyn_C, T)
    x2np = np.dot(KG, C_mod)[[3, 5], :]

    nb_points   = 20
    x = np.linspace(-1, 1, nb_points)
    y = np.linspace(-1, 1, nb_points)
    X1 , Y1  = np.meshgrid(x, y)                       # create a grid
    
    for i_d, dim in enumerate(range(0, 14, 2)):

        axi = ax[i_d/3, i_d%3]

        ### Get changes on grid: 
        DX, DY = compute_dX(X1, Y1, A_mod, dim, dim+1)  
        M = (np.hypot(DX, DY))           
        
        ### Quiver changes
        Q = axi.quiver(X1, Y1, DX, DY, pivot='mid')

        ### Get Readout space: 
        axi.plot([0, x2np[0, dim]], [0, x2np[0, dim+1]], 'b-')
        axi.plot([0, -1*x2np[0, dim]], [0, -1*x2np[0, dim+1]], '-', color='lightblue')

        axi.plot([0, x2np[1, dim]], [0, x2np[1, dim+1]], 'r-')
        axi.plot([0, -1*x2np[1, dim]], [0, -1*x2np[1, dim+1]], '-', color='lightcoral')

    ### Plot the points: 
    for i_b, bin in enumerate(bins):
        if i_b > 0:
            pass
        else:
            x0 = np.dot(TI, neural_state_trl[bin, :])
            x_prop = np.dot(A_mod, x0)
            x_prop1 = np.dot(A_mod, x_prop)
            x_prop2 = np.dot(A_mod, x_prop1)

            ### Get out neural state: 

            x1 = np.dot(TI, neural_state_trl[bin+1, :])

            for i_d, dim in enumerate(range(0, 14, 2)):
                axi = ax[i_d/3, i_d%3]

                axi.plot([x0[dim], x_prop[dim]], [x0[dim+1], x_prop[dim+1]], '-', color=color_pairs[i_b][0])
                axi.plot([x0[dim], x1[dim]], [x0[dim+1], x1[dim+1]], '-', color=color_pairs[i_b][1])

    f.tight_layout()
    import pdb; pdb.set_trace()

def compute_dX(X, Y, A, dim1, dim2):
    newX = np.zeros_like(X)
    newY = np.zeros_like(Y)

    nrows, ncols = X.shape

    for nr in range(nrows):
        for nc in range(ncols):
            st = np.zeros((len(A), 1))
            st[dim1] = X[nr, nc]; 
            st[dim2] = Y[nr, nc];

            st_nx = np.squeeze(np.dot(A, st))
            newX[nr, nc] = st_nx[dim1]
            newY[nr, nc] = st_nx[dim2]

    ### Now to get the change, do new - old: 
    DX = newX - X; 
    DY = newY - Y; 

    return DX, DY

#############################
### Angle Utils ###
### From https://github.com/jhamrick/python-snippets/blob/master/snippets/circstats.py

def ang_difference(a1, a2):
    """Compute the smallest difference between two angle arrays.
    Parameters
    ----------
    a1, a2 : np.ndarray
        The angle arrays to subtract
    deg : bool (default=False)
        Whether to compute the difference in degrees or radians
    Returns
    -------
    out : np.ndarray
        The difference between a1 and a2
    """

    diff = a1 - a2
    return wrapdiff(diff)

def wrapdiff(diff, deg=False):
    """Given an array of angle differences, make sure that they lie
    between -pi and pi.
    Parameters
    ----------
    diff : np.ndarray
        The angle difference array
    deg : bool (default=False)
        Whether the angles are in degrees or radians
    Returns
    -------
    out : np.ndarray
        The updated angle differences
    """

    base = np.pi * 2
    i = np.abs(diff) > (base / 2.0)
    out = diff.copy()
    out[i] -= np.sign(diff[i]) * base
    return out

