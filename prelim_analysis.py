import numpy as np
from collections import defaultdict
import sklearn.decomposition as skdecomp
from db import dbfunctions as dbfn
import tables
#import test_reconst_bmi_traj as trbt
import matplotlib.pyplot as plt
# import seaborn
# seaborn.set_context('talk',font_scale=1)
# seaborn.set_style('whitegrid')
import pickle
import scipy
import os, copy
import scipy.io as sio
# te = 3138

# if te==3136:
#     hdf = tables.openFile('grom_data/grom20150427_02.hdf')
# elif te ==3138:
#     hdf = tables.openFile('grom_data/grom20150428_02.hdf')
#hdf = tables.openFile('grom_data/grom20151129_19_te3725.hdf')
#dec = pickle.load(open('grom_data/grom20150422_13_PM04221738.pkl'))
#dec = pickle.load(open('grom_data/grom20151129_16_RMLC11291807.pkl'))
#hdf = tables.openFile(os.path.expandvars('$FA_GROM_DATA/grom20151201_03_te3729.hdf'))
#dec = pickle.load(open(os.path.expandvars('$FA_GROM_DATA/grom20151201_01_RMLC12011916.pkl')))

#ReDecoder = trbt.RerunDecoding(hdf, dec, task='point_mass', drives_neurons=3)
#ReDecoder = trbt.RerunDecoding(hdf, dec, task='bmi_multi')#

########################
########################
#### ANALYSIS FCNS #####
########################
########################
# def FA_k_500ms(): 
#     #Function to determine optimal number of factors for first 500 ms of trial
#     #Separates FA model for each target
#     rew_ix = get_trials_per_min(hdf)
#     spk, targ_pos, targ_ix, reach_time = extract_trials(hdf, rew_ix, ms=500)
#     LL = fit_all_targs(spk, targ_ix,iters=10, max_k = 10)
hdf = []

def FA_k_ALLms(hdf):
    #Function to determine optimal number of factors for full trial
    #Separates FA model for each target
    drives_neurons_ix0 = 3
    rew_ix = get_trials_per_min(hdf)
    internal_state = hdf.root.task[:]['internal_decoder_state']
    update_bmi_ix = np.nonzero(np.diff(np.squeeze(internal_state[:, drives_neurons_ix0, 0])))[0]+1

    bin_spk, targ_pos, targ_ix, z, zz = extract_trials_all(hdf, rew_ix, update_bmi_ix=update_bmi_ix)
    LL, ax = fit_all_targs(bin_spk, targ_ix, proc_spks = False, iters=20, max_k = 10, return_ax=True) 
    ax[3,0].set_xlabel('Num Factors')
    ax[3,1].set_xlabel('Num Factors')
    for i in range(4):
        ax[i, 0].set_ylabel('Log Lik')
    return LL, ax

def FA_all_targ_ALLms(hdf, max_k=10, iters=20):
    #Function to determine optimal number of factors for full trial
    drives_neurons_ix0 = 3
    rew_ix = get_trials_per_min(hdf)
    internal_state = hdf.root.task[:]['internal_decoder_state']
    update_bmi_ix = np.nonzero(np.diff(np.squeeze(internal_state[:, drives_neurons_ix0, 0])))[0]+1
    bin_spk, targ_pos, targ_ix, z, zz = extract_trials_all(hdf, rew_ix, update_bmi_ix=update_bmi_ix)
    zscore_X, mu = zscore_spks(bin_spk)
    log_lik, psv, ax = find_k_FA(zscore_X, iters=iters, max_k = max_k, plot=True)
    return log_lik, ax

def SVR_slow_fast_ALLms(hdf, factors=5):
    rew_ix = get_trials_per_min(hdf)
    bin_spk, targ_pos, targ_ix, trial_ix, reach_time = extract_trials_all(hdf, rew_ix)
    fast_vs_slow_trials(bin_spk, targ_ix, reach_time, factors=factors,plot=True, all_trial_tm=True)

def ind_vs_all_targ_SSAlign(factors=5, hdf=hdf):
    rew_ix = get_trials_per_min(hdf)
    bin_spk, targ_pos, targ_ix, trial_ix, reach_time = extract_trials_all(hdf, rew_ix)
    overlap = targ_vs_all_subspace_align(bin_spk, targ_ix, factors=factors)
    f, ax = plt.subplots()
    c = ax.pcolormesh(overlap)
    plt.colorbar(c)
    return overlap

def learning_curve_metrics(hdf_list, epoch_size=56, n_factors=5):
    #hdf_list = [3822, 3834, 3835, 3840]
    #obstacle learning: hdf_list = [4098, 4100, 4102, 4104, 4114, 4116, 4118, 4119]
    rew_ix_list = []
    te_refs = []
    rpm_list = []
    hdf_dict = {}
    perc_succ = []
    time_list = []
    offs = 0

    #f, ax = plt.subplots()
    for te in hdf_list:
        hdf_t = dbfn.TaskEntry(te)
        hdf = hdf_t.hdf
        hdf_dict[te] = hdf

        rew_ix, rpm = pa.get_trials_per_min(hdf, nmin=2,rew_per_min_cutoff=0, 
            ignore_assist=True, return_rpm=True)
        ix = 0
        #ax.plot(rpm)

        trial_ix = np.array([i for i in hdf.root.task_msgs[:] if 
            i['msg'] in ['reward','timeout_penalty','hold_penalty','obstacle_penalty'] ], dtype=hdf.root.task_msgs.dtype)


        while (ix+epoch_size) < len(rew_ix):
            start_rew_ix = rew_ix[ix]
            end_rew_ix = rew_ix[ix+epoch_size]
            msg_ix_mod = np.nonzero(scipy.logical_and(trial_ix['time']<=end_rew_ix, trial_ix['time']>start_rew_ix))[0]
            all_msg = trial_ix[msg_ix_mod]
            perc_succ.append(len(np.nonzero(all_msg['msg']=='reward')[0]) / float(len(all_msg)))

            rew_ix_list.append(rew_ix[ix:ix+epoch_size])
            rpm_list.append(np.mean(rpm[ix:ix+epoch_size]))
            te_refs.append(te)
            time_list.append((0.5*(start_rew_ix+end_rew_ix))+offs)

            ix += epoch_size
        offs = offs+len(hdf.root.task)

    #For each epoch, fit FA model (stick w/ 5 factors for now):
    ratio = []
    for te, r_ix in zip(te_refs, rew_ix_list):
        print te, len(r_ix)

        update_bmi_ix = np.nonzero(np.diff(np.squeeze(hdf.root.task[:]['internal_decoder_state'][:, 3, 0])))[0] + 1
        bin_spk, targ_pos, targ_ix, z, zz = pa.extract_trials_all(hdf_dict[te], r_ix, time_cutoff=1000, update_bmi_ix=update_bmi_ix)
        zscore_X, mu = pa.zscore_spks(bin_spk)
        FA = skdecomp.FactorAnalysis(n_components=n_factors)
        FA.fit(zscore_X)

        #SOT Variance Ratio by target
        #Priv var / mean
        Cov_Priv = np.sum(FA.noise_variance_)
        U = np.mat(FA.components_).T
        Cov_Shar = np.trace(U*U.T)

        ratio.append(Cov_Shar/(Cov_Shar+Cov_Priv))


########################
########################
###### HELPER FCNS #####
########################
########################

#Get behavior
def get_trials_per_min(hdf,nmin=2, rew_per_min_cutoff=0, ignore_assist=False, return_rpm=False, 
    return_per_succ=False, plot=False):
    '''
    Summary: Getting trials per minute from hdf file
    Input param: hdf: hdf file to use
    Input param: nmin: number of min to use a rectangular window
    Input param: rew_per_min_cutoff: ignore rew_ix after a 
        certain rew_per_min low threshold is passed
    Output param: rew_ix = rewarded indices in hdf file
    '''

    rew_ix = np.array([t[1] for it, t in enumerate(hdf.root.task_msgs[:]) if t[0]=='reward'])
    tm = np.zeros((np.max(rew_ix)+1))
    tm[rew_ix] += 1
    
    if hasattr(hdf.root.task, 'assist_level'):
        assist_ix = np.nonzero(hdf.root.task[:]['assist_level']==0)[0]
    else:
        assist_ix = np.zeros((len(hdf.root.task)))

    #Each row occurs ~1/60 sec, so: 
    minute = 60*60;
    min_wind = np.ones((nmin*minute))/float(nmin)
    rew_per_min_tmp = np.convolve(min_wind, tm, mode='same')

    #Now smooth please: 
    smooth_wind = np.ones((3*minute))/float(3*minute)
    rew_per_min = pk_convolve(smooth_wind, rew_per_min_tmp)

    if rew_per_min_cutoff > 0:
        ix = np.nonzero(rew_per_min < rew_per_min_cutoff)[0]
        if len(ix)>0:
            cutoff_ix = ix[0]
        else:
            cutoff_ix = rew_ix[-1]
    
    else:
        cutoff_ix = rew_ix[-1]

    if ignore_assist:
        try:
            beg_zer_assist_ix = assist_ix[0]
        except:
            print 'No values w/o assist for filename: ', hdf.filename
            beg_zer_assist_ix = rew_ix[-1]+1
    else:
        beg_zer_assist_ix = 0

    if plot:
        plt.plot(np.arange(len(tm))/float(minute), rew_per_min)
        plt.show()
    ix_final = scipy.logical_and(rew_ix <= cutoff_ix, rew_ix >= beg_zer_assist_ix)
    if return_rpm:
        return rew_ix[ix_final], rew_per_min[rew_ix[ix_final]]
    else:
        return rew_ix[ix_final]

def pk_convolve(window, arr):
    '''
    Summary: Same as NP convolve but no edge effects (edges padded with zeros+start value and zeros+end value )
    Input param: window: window to convolve
    Input param: arr: array (longer than window)
    Output param: 
    '''
    win_len = len(window)
    start = arr[0]
    end = arr[-1]

    arr_pad = np.hstack(( np.zeros((win_len, )) + start, arr, np.zeros((win_len, )) + end ))
    tmp = np.convolve(window, arr_pad, mode='same')
    return tmp[win_len:-win_len]

def extract_trials_all(hdf, rew_ix, neural_bins = 100, time_cutoff=None, hdf_ix=False, 
    update_bmi_ix=None, rew_pls=False, step_dict=None, drives_neurons_ix0=None, first_n_sec_of_trial=None,
    use_ITI=False, hdf_key='spike_counts', keep_trials_sep=True, divide_by_6=False, reach_tm_is_hdf_cursor_pos=False,
    reach_tm_is_hdf_cursor_state=False, reach_tm_is_kg_vel=False, include_pre_go=0, **kwargs):
    '''
    Summary: method to extract all time points from trials
    Input param: hdf: task file input
    Input param: rew_ix: rows in the hdf file corresponding to reward times
    Input param: neural_bins: ms per bin
    Input param: time_cutoff: time in minutes, only extract trials before this time
    Input param: hdf_ix: bool, whether to return hdf row corresponding to time of decoder 
    update (and hence end of spike bin)
    Input param: update_bmi_ix: bins for updating bmi

    Input param: rew_pls : If rew_ix is actually a N x 2 array with error trials included (default = false)
    input param: step_dict: The number of steps to go backward from each type of trial in rew_ix (only 
        if rew_pls is True)
    Input param: drives_neurons_ix0: Index in decoder that is driven by neurons (usually 3 or 6 for velx, vely in bmi3d)
    Input param: first_n_sec_of_trial: only use the first 'n' sec of trial (go --> go + n) , not go--> reward (default=None), if 
        negative then use reward - n --> reward
    Input param: use_ITI: use reward --> target onset
    Input param: hdf_key: key of what to extract from the HDF file (usually 'spike_counts')
    Input param: keep_trials_sep: keep binned spikes into seperate trials (each trial is an array w/in a list of many arrays)
    Input param: divide_by_6: since many of the FA HDF files list the same value for each point in 100 ms bin (e.g. [3.5, 3.5, 3.5, 3.5, 3.5], 
        instead of just [3.5, 0 0 0 0 0], we can divide the binned spike counts by 6 to account for this). So, only do this if analyzing 
        FA HDF files like shar_scaled
    Input param: reach_tm_is_hdf_cursor_pos: Instead of returning 'reach time' (aka time to target, or t2t), instead return the cursor positions from 
        the HDF file
    Input param: reach_tm_is_kg_vel:  Instead of returning 'reach time' (aka time to target, or t2t), instead return the kalman gain*spk count values
        from each bin (requires kwarg input of 'kalman_gain')
    Input param: include_pre_go: Number of seconds to include before go cue

    Output param: bin_spk -- binned spikes in time x units
                  targ_i_all -- target location at each update
                  targ_ix -- target index 
                  trial_ix -- trial number
                  reach_time -- reach time for trial (or other depending on input params -- a) reach_tm_is_hdf_cursor_pos, b) reach_tm_is_kg_vel
                  hdf_ix -- end bin in units of hdf rows
    '''
    if update_bmi_ix is None:
        internal_state = hdf.root.task[:]['internal_decoder_state']
        if drives_neurons_ix0 is None:
            print 'Must define either update_bmi_ix OR drives_neurons_ix0'
            raise NameError('drives_neurons_ix0 must be defined')

        update_bmi_ix = np.nonzero(np.diff(np.squeeze(internal_state[:, drives_neurons_ix0, 0])))[0]+1

    #### Index cutoff #### 
    if time_cutoff is None:
        it_cutoff = len(hdf.root.task)
    else:
        it_cutoff = time_cutoff*60*60

    ### If rew is supplied (unlikely)
    if rew_pls:
        print('Using rew pls option -- this is unusual, may want to check out prelim_analysis fcn')
        go_ix = np.array([hdf.root.task_msgs[it - step_dict[m['msg']]]['time'] for it, m in enumerate(hdf.root.task_msgs[:]) 
            if m['msg'] in step_dict.keys()])

        rew_ix = np.array([hdf.root.task_msgs[it]['time'] for it, m in enumerate(hdf.root.task_msgs[:]) 
            if m['msg'] in step_dict.keys()])

        outcome_ix = np.array([hdf.root.task_msgs[it]['msg'] for it, m in enumerate(hdf.root.task_msgs[:]) 
            if m['msg'] in step_dict.keys()])

    ### Get rewards from the HDF file ####
    else:
        go_ix = np.array([hdf.root.task_msgs[it-3][1] for it, t in enumerate(hdf.root.task_msgs[:]) if 
            scipy.logical_and(t[0] == 'reward', t[1] in rew_ix)])

        ### Make sure the mesaged for all of these are ...?
        assert(np.all(s[0] == 'target' for s in hdf.root.task_msgs[:] if s[1] in go_ix))
        assert(len([s[0] for s in hdf.root.task_msgs[:] if s[1] in go_ix]) == len(go_ix))
        assert(np.all(s[0] == 'reward' for s in hdf.root.task_msgs[:] if s[1] in rew_ix))
        assert(len([s[0] for s in hdf.root.task_msgs[:] if s[1] in rew_ix]) == len(rew_ix))

    go_ix = go_ix[go_ix<it_cutoff]
    rew_ix = rew_ix[go_ix<it_cutoff]

    if first_n_sec_of_trial is not None:
        if first_n_sec_of_trial > 0:
            n = 60*first_n_sec_of_trial #Fs (it / sec) * sec = ix
            print 'fwd from go: ', n
            rew_ix = go_ix + n

        elif first_n_sec_of_trial < 0:
            n = 60*np.abs(first_n_sec_of_trial)
            print 'back from reward: ', n
            go_ix = rew_ix - n

    if include_pre_go > 0:
        go_ix = go_ix - (60.*include_pre_go)
        i = np.nonzero(go_ix<0)[0]

        ### Keep it but set it to -1 #####
        go_ix[i] = -1. 
        rew_ix = rew_ix*1. ## Int to float 
        rew_ix[i] = -1.

    if use_ITI: 
        print ' using ITI '
        go_ix = rew_ix.copy()
        rew_ix = np.array([hdf.root.task_msgs[it+3][1] for it, t in enumerate(hdf.root.task_msgs[:-3]) if 
            scipy.logical_and(t[0] == 'reward', t[1] in go_ix)])

    ### Approach here, from 2016 preeya is to initialize a bunch of arrays that could 
    ### be added to, approach from 2020 preeya is to use a default dict that is a list-like thing ###
    all_data = defaultdict(list)

    for ig, (g, r) in enumerate(zip(go_ix, rew_ix)):
        g = int(g)
        r = int(r)
        
        ### If we want to analyze you ###
        if g >= 0:

            ### T x N spks 
            spk_i = hdf.root.task[g:r][hdf_key][:,:,0]

            #Sum spikes in neural_bins:
            bin_spk_i, nbins, hdf_ix_i = bin_spks(spk_i, g, r, neural_bins, update_bmi_ix, divide_by_6)

        else:
            nn = hdf.root.task[0]['spike_counts'].shape[0]
            bin_spk_i = np.zeros((1, nn))
            nbins = 0
            hdf_ix_i = []

        ### Append all now, vstack later 
        nT = bin_spk_i.shape[0]
        all_data['bin_spk'].append(bin_spk_i)

        ### Add back the pre_go so get the right target ####
        targ = hdf.root.task[int(g) + 1 + int(include_pre_go*60)]['target'][[0,2]]
        nT_targ = np.tile(targ, (nT, 1))
        all_data['targ_i_all'].append(nT_targ)
        all_data['trial_ix_all'].append(np.zeros((nT, )) + ig)

        if reach_tm_is_hdf_cursor_pos:
            sub_cursor_i = hdf.root.task[hdf_ix_i]['cursor'][:, [0, 2]]
            #reach_tm_all.append(sub_cursor_i)
            all_data['cursor_pos'].append(sub_cursor_i)

        
        elif reach_tm_is_hdf_cursor_state:
            sub_cursor_i = hdf.root.task[hdf_ix_i]['decoder_state'][:, [0, 2, 3, 5]]
            #reach_tm_all.append(sub_cursor_i)
            all_data['decoder_state'].append(sub_cursor_i)
            
        elif reach_tm_is_kg_vel:
            kg = kwargs['kalman_gain']
            #reach_tm_all.append(hdf.root.task[g:r]['internal_decoder_state'][:, :, 0])
            reach_tm_all.append(bin_spk_i*np.mat(kg).T)

            ### pos/vel x time 
            all_data['kg_vel'].append(np.dot(bin_spk_i, kg).T)
        else:
            reach_tm_all = np.hstack((reach_tm_all, np.zeros(( bin_spk_i.shape[0] ))+((r-g)*1000./60.) ))
            all_data['rch_tm'].append(np.zeros((nT, )) + (r-g)*1000./60.)
        
    print go_ix.shape, rew_ix.shape, bin_spk_i.shape, nbins, hdf_ix_i.shape
    targ_ix = get_target_ix(np.vstack((all_data['targ_i_all']))) #targ_i_all[1:,:])
    print np.unique(targ_ix)

    if hdf_ix:
        raise Exception('Deprecated')
    else:
        nTrls = len(all_data['bin_spk'])
        nT = np.vstack((all_data['bin_spk'])).shape[0]
        assert(len(all_data['bin_spk']) == nTrls)
        
        ### Array regardless: 
        assert(len(all_data['targ_i_all']) == nTrls)
        targ_i_all = np.vstack((all_data['targ_i_all']))
        assert(len(targ_i_all) == nT)

        assert(len(targ_ix) == nT)
        
        assert(len(all_data['trial_ix_all']) == nTrls)
        trl_i_all = np.hstack((all_data['trial_ix_all']))
        assert(len(trl_i_all) == nT)

        if reach_tm_is_hdf_cursor_pos:
            mystery_meat = all_data['cursor_pos']
            assert(len(mystery_meat) == nTrls)
        elif reach_tm_is_hdf_cursor_state:
            mystery_meat = all_data['decoder_state']
            assert(len(mystery_meat) == nTrls)
        elif reach_tm_is_kg_vel:
            mystery_meat = all_data['kg_vel']
            assert(len(mystery_meat) == nTrls)
        else:
            mystery_meat = np.hstack(( all_data['rch_tm'] ))
            assert(len(mystery_meat) == nT)

        if type(mystery_meat) is list:
            assert(np.vstack((mystery_meat)).shape[0] == nT)

        if keep_trials_sep:
            return all_data['bin_spk'], targ_i_all, targ_ix, trl_i_all, mystery_meat
        else:
            return np.vstack(( all_data['bin_spk'])), targ_i_all, targ_ix, trl_i_all, mystery_meat

def bin_spks(spk_i, g_ix, r_ix, binsize_ms, update_bmi_ix, divide_by_6):
    ''' 
    update_bmi_ix has time points of updates
    '''

    assert r_ix - g_ix == spk_i.shape[0]

    trial_inds = np.arange(g_ix, r_ix+1)
    nbins = binsize_ms/(1000/60.)  ### For BMI3D --> task sampling is 60 hz

    # Make sure binsize is a multiple of 60Hz bins:
    assert np.abs(np.round(nbins)-nbins) < 1e-3
    nbins = int(np.round(nbins))

    # Use actual decoder bins, Need to use 'update_bmi_ix' from ReDecoder to get bin edges correctly
    # need this to be > g_ix + 5 ? if not then don't have enough bins beforehand #
    end_bin100 = np.array([(j,i) for j, i in enumerate(trial_inds) if np.logical_and(i in update_bmi_ix, i>=(g_ix+5))])
    if binsize_ms == 100: 
        nbins_total = len(end_bin100)
        bin_spk_i = np.zeros((nbins_total, spk_i.shape[1]))
        end_bin_final = copy.deepcopy(end_bin100); 
        bef_endbin = 5
        aft_endbin = 1
    
    else:
        print('Doing some sort of subsampling -- this hasnt been tested')
        ref = end_bin100[0, :]
        nstep_before = ref[0]/nbins
        end_bin_final = np.arange(ref[0] - (nstep_before*nbins), end_bin100[-1, 0]+1, nbins)
        end_bin_final = np.hstack((end_bin_final[:, np.newaxis], trial_inds[end_bin_final][:, np.newaxis]))
        bin_spk_i = np.zeros((len(end_bin_final), spk_i.shape[1]))
        bef_endbin = nbins - 1
        aft_endbin = 1

    hdf_ix_i = []
    for ib, (i_ix, hdf_ix) in enumerate(end_bin_final):
        #Inclusive of EndBin
        if divide_by_6:
            bin_spk_i[ib,:] = np.sum(spk_i[i_ix - bef_endbin:i_ix+aft_endbin,:], axis=0)/6.
        else:
            bin_spk_i[ib,:] = np.sum(spk_i[i_ix - bef_endbin:i_ix+aft_endbin,:], axis=0)
        hdf_ix_i.append(hdf_ix)
    return bin_spk_i, nbins, np.array(hdf_ix_i)
    
def extract_trials(hdf, rew_ix, ms=500, time_cutoff=40):
    it_cutoff = time_cutoff*60*60
    nsteps = int(ms/(1000.)*60)

    #Get Go cue and 
    go_ix = np.array([hdf.root.task_msgs[it-3][1] for it, t in enumerate(hdf.root.task_msgs[:]) if t[0] == 'reward'])
    go_ix = go_ix[go_ix<it_cutoff]

    spk = np.zeros((len(go_ix), nsteps, hdf.root.task[0]['spike_counts'].shape[0]))

    targ_pos = np.zeros((len(go_ix), 2))
    reach_time = np.zeros((len(go_ix), ))
    for ig, g in enumerate(go_ix):
        spk[ig, :, :] = hdf.root.task[g:g+nsteps]['spike_counts'][:,:,0]
        targ_pos[ig, :] = hdf.root.task[g+1]['target'][[0,2]]
        reach_time[ig] = (rew_ix[ig]- g)/(60.) #In seconds
    
    targ_ix = get_target_ix(targ_pos)
    return spk, targ_pos, targ_ix, reach_time

def get_target_ix(targ_pos):
    #Target Index: 
    # b = np.ascontiguousarray(targ_pos).view(np.dtype((np.void, targ_pos.dtype.itemsize * targ_pos.shape[1])))
    # _, idx = np.unique(b, return_index=True)
    # unique_targ = targ_pos[idx,:]

    # # Dont' count initial: 
    # rm_ix = np.nonzero(np.logical_and(unique_targ[:, 0]==0, unique_targ[:, 1]==0))[0]
    # keep_ix = [i for i in range(len(idx)) if i not in list(rm_ix)]
    # unique_targ = unique_targ[keep_ix, :]

    # #Order by theta: 
    # theta = np.arctan2(unique_targ[:,1],unique_targ[:,0])
    # thet_i = np.argsort(theta)
    # unique_targ = unique_targ[thet_i, :]

    # unique_targ = np.array([[-7.07106781e+00, -7.07106781e+00],
    #        [-1.83697020e-15, -1.00000000e+01],
    #        [ 7.07106781e+00, -7.07106781e+00],
    #        [ 1.00000000e+01,  0.00000000e+00],
    #        [ 7.07106781e+00,  7.07106781e+00],
    #        [ 6.12323400e-16,  1.00000000e+01],
    #        [-7.07106781e+00,  7.07106781e+00],
    #        [-1.00000000e+01,  1.22464680e-15]])

    dats = sio.loadmat('/Users/preeyakhanna/fa_analysis/online_analysis/unique_targ.mat')
    unique_targ = dats['unique_targ']

    targ_ix = np.zeros((targ_pos.shape[0]), )
    for ig, (x,y) in enumerate(targ_pos):
        tmp_ix = np.nonzero(np.sum(targ_pos[ig,:]==unique_targ, axis=1)==2)[0]
        if len(tmp_ix) > 0:
            targ_ix[ig] = tmp_ix
        else:
            targ_ix[ig] = -1

    return targ_ix

def proc_spks(spk, targ_ix, targ_ix_analysis=0, neural_bins = 100, return_unshapedX=False):
    '''
    Summary: processes bin_spikes (time x units)
    Input param: spk: time x units (in binned spikes)
    Input param: targ_ix: array lenth of bins
    Input param: targ_ix_analysis: 
    Input param: neural_bins:
    Input param: return_unshapedX:
    Output param: 
    '''

    if targ_ix_analysis == 'all':
        spk_trunc = spk.copy()
    else:
        ix = np.nonzero(targ_ix==targ_ix_analysis)[0]
        spk_trunc = spk[ix, :, :]

    bin_ix = int(neural_bins/1000.*60.)
    spk_trunc_bin = np.zeros((spk_trunc.shape[0], spk_trunc.shape[1]/bin_ix, spk_trunc.shape[2]))
    for i_start in range(spk_trunc.shape[1]/bin_ix):
        spk_trunc_bin[:, i_start, :] = np.sum(spk_trunc[:, (i_start*bin_ix):(i_start+1)*bin_ix, :], axis=1)
    resh_spk_trunc_bin = spk_trunc_bin.reshape((spk_trunc_bin.shape[0]*spk_trunc_bin.shape[1], spk_trunc_bin.shape[2]))
    if return_unshapedX:
        return spk_trunc_bin
    else:
        return resh_spk_trunc_bin

def zscore_spks(proc_spks):
    '''
    proc_spks in time x neurons
    zscore_X in time x neurons
    '''

    mu = np.tile(np.mean(proc_spks, axis=0), (proc_spks.shape[0], 1))
    zscore_X = proc_spks - mu
    return zscore_X, mu

def find_k_FA(zscore_X, xval_test_perc = .1, iters=100, max_k = 20, plot=True):
    ntrials = zscore_X.shape[0]
    ntrain = ntrials*(1-xval_test_perc)
    log_lik = np.zeros((iters, max_k))
    perc_shar_var = np.zeros((iters, max_k))

    for i in range(iters):
        #print 'iter: ', i
        ix = np.random.permutation(ntrials)
        train_ix = ix[:int(ntrain)]
        test_ix = ix[int(ntrain):]

        for k in range(max_k):
            #print 'factor n : ', k
            FA = skdecomp.FactorAnalysis(n_components=k+1)
            FA.fit(zscore_X[train_ix,:])
            LL = np.sum(FA.score(zscore_X[test_ix,:]))
            log_lik[i, k] = LL

            if (k+1) == max_k:
                cov_shar = np.matrix(FA.components_) * np.matrix(FA.components_.T)
                perc_shar_var[i, :] = np.cumsum(np.diag(cov_shar)) 

    log_lik[log_lik<-1e4] = np.nan
    if plot:
        fig, ax = plt.subplots(nrows=2)
        mu = 1/float(log_lik.shape[0])*np.nansum(log_lik,axis=0)
        
        for it in range(log_lik.shape[0]):
            for nf in range(log_lik.shape[1]):
                if np.isnan(log_lik[it, nf]):
                    log_lik[it, nf] = mu[nf]

        fin = np.tile(np.array([perc_shar_var[:,-1]]).T, [1, perc_shar_var.shape[1]])
        perc = perc_shar_var/fin.astype(float)

        ax[0].errorbar(np.arange(1,log_lik.shape[1]+1), np.mean(log_lik,axis=0), yerr=np.std(log_lik,axis=0)/np.sqrt(iters), fmt='o')
        #plt.plot(np.arange(1,log_lik.shape[1]+1), np.mean(log_lik,axis=0),'.-')
        ax[0].set_ylabel('Log Likelihood of Held Out Data')

        ax[1].errorbar(np.arange(1,perc.shape[1]+1), np.mean(perc,axis=0), yerr=np.std(perc,axis=0)/np.sqrt(iters), fmt='o')
        ax[1].set_xlabel('Number of Factors')
        ax[1].set_ylabel('Perc. Shar. Var. ')
        return log_lik, perc_shar_var, ax
    else:

        return log_lik, perc_shar_var

def fit_all_targs(spk, targ_ix, proc_spks=True, iters=10, max_k = 10, return_ax=False):
    LL = dict()
    tg = np.unique(targ_ix)
    fig, ax = plt.subplots(nrows=4, ncols=2)

    for it, t in enumerate(tg):
        if proc_spks:
            resh_spk_trunc_bin = proc_spks(spk, targ_ix, targ_ix_analysis=t)
        else:
            ix = np.nonzero(targ_ix==t)[0]
            resh_spk_trunc_bin = spk[ix,:]
            print 'ix: ', str(len(ix))
        zscore_X, mu = zscore_spks(resh_spk_trunc_bin)
        log_lik, psv = find_k_FA(zscore_X, iters=iters, max_k =max_k, plot=False)
        LL[t] = log_lik
        #TODO: insert way to 
        ax[it%4, it/4].plot(np.arange(10)+1, np.mean(log_lik, axis=0), '.-', label='Targ '+str(t))
        ax[it%4, it/4].errorbar(np.arange(1,log_lik.shape[1]+1), np.mean(log_lik,axis=0), yerr=np.std(log_lik,axis=0)/np.sqrt(iters), fmt='o')
        ax[it%4, it/4].set_title('Targ '+str(t))
    
    plt.axis('tight')
    plt.tight_layout()
    if return_ax:
        return LL, ax
    else:
        return LL

def shared_vs_total_var(spk, targ_ix, reach_time, factors=5):
    #For each target, fit a FA analysis model:
    tg = np.unique(targ_ix)
    ratio = np.zeros((len(tg), ))
    beh = ratio.copy()
    for it, t in enumerate(tg):
        zscore_X, z, m = proc_spks(spk, targ_ix, targ_ix_analysis=t)
        FA = skdecomp.FactorAnalysis(n_components=factors)
        FA.fit(zscore_X)

        Cov_Priv = np.sum(FA.noise_variance_)
        U = np.mat(FA.components_).T
        Cov_Shar = np.trace(U*U.T)

        ratio[it] = Cov_Shar/(Cov_Shar+Cov_Priv)

        ix = np.nonzero(targ_ix==t)[0]
        beh[it] = np.mean(reach_time[ix])

    f, ax1 = plt.subplots()
    ax1.plot(ratio, label='Ratio')
    ax1.set_ylabel('Shared to Total Variance')
    ax2 = ax1.twinx()
    ax2.plot(beh, 'k-', label='Reach Time (sec)')
    ax2.set_ylabel('Reach Time')
    ax2.set_title('N Factors: '+str(factors))
    ax1.set_xlabel('Target Number')
    plt.legend()
    return ratio, beh

def fast_vs_slow_trials(spk, targ_ix, reach_time, factors=5,plot=True, all_trial_tm=False):
    tg = np.unique(targ_ix)

    fast_ratio = np.zeros((len(tg), ))
    slow_ratio = fast_ratio.copy()
    beh_fast = fast_ratio.copy()
    beh_slow = fast_ratio.copy()

    for it, t in enumerate(tg):
        ix = np.nonzero(targ_ix==t)[0]
        mid = np.percentile(reach_time[ix], 50)
        fast_ix_ix = np.nonzero(reach_time[ix]<= mid)[0]
        slow_ix_ix = np.nonzero(reach_time[ix]> mid)[0]

        #Fast FA: 
        def get_ratio(ix_ix):
            if all_trial_tm:
                X = spk[ix[ix_ix], :]
                zscore_X, mu = zscore_spks(X)
            else:
                X = proc_spks(spk, targ_ix, targ_ix_analysis=t,return_unshapedX=True)
                X = X[ix_ix, :, :]
                zscore_X = zscore_spks(X.reshape((X.shape[0]*X.shape[1], X.shape[2])))
                
            FA = skdecomp.FactorAnalysis(n_components=factors)
            FA.fit(zscore_X)

            Cov_Priv = np.sum(FA.noise_variance_)
            U = np.mat(FA.components_).T
            Cov_Shar = np.trace(U*U.T)
            return Cov_Shar/(Cov_Shar+Cov_Priv)
        
        fast_ratio[it] = get_ratio(fast_ix_ix)
        slow_ratio[it] = get_ratio(slow_ix_ix)
        beh_fast[it] = np.mean(reach_time[ix][fast_ix_ix])
        beh_slow[it] = np.mean(reach_time[ix][slow_ix_ix])

    if plot:
        f, ax1 = plt.subplots(nrows=4, ncols=2)
        for it in range(len(tg)):
            ii = it%4
            jj = it/4

            ax1[ii, jj].plot(tg[it]+np.array([-.33, .33]), [fast_ratio[it], slow_ratio[it]], 'r*-')
            #ax1[ii, jj].plot(tg[it]+.33, slow_ratio[it], 'b*')
            ax1[ii, jj].set_ylabel('Var. Ratio')
            #ax1[ii, jj].set_ylim([.35, .55])
            ax2 = ax1[ii, jj].twinx()
            ax2.plot(tg[it]+np.array([-.33, .33]), [beh_fast[it], beh_slow[it]], 'bo-', label='Reach Time (sec)')
            #ax2.plot(tg[it]+.33, beh_slow[it], 'bo', label='Slow Reach Time (sec)')

            ax2.set_ylabel('Reach Time')
            ax2.set_title('N Factors: '+str(factors))
            #ax2.set_ylim([2.5, 4.5])
            ax1[ii, jj].set_xlabel('Target Number '+str(tg[it]))
            #plt.legend()
    plt.tight_layout()
    return fast_ratio, slow_ratio, beh_fast, beh_slow

def FA_subspace_align(FA1, FA2):
    U_A = np.mat(FA1.components_.T)
    U_B = np.mat(FA2.components_).T
    v, s, vt = np.linalg.svd(U_B*U_B.T)
    P_B = v*vt
    S_A_shared = U_A*U_A.T
    return np.trace(P_B*S_A_shared*P_B.T)/np.trace(S_A_shared)

def targ_vs_all_subspace_align(bin_spk, targ_ix, factors=5):
    X_zsc, mu = zscore_spks(bin_spk)
    FA_full = skdecomp.FactorAnalysis(n_components = factors)
    FA_full.fit(X_zsc)

    unique_targ = np.unique(targ_ix)
    Overlap = np.zeros((len(unique_targ)+1, len(unique_targ)+1))
    
    FA_targ = dict()
    for it, t in enumerate(unique_targ):
        FA = skdecomp.FactorAnalysis(n_components = factors)
        ix = np.nonzero(targ_ix==t)[0]
        X_zsc, mu = zscore_spks(bin_spk[ix,:])
        FA.fit(X_zsc)
        FA_targ[t] = FA
        Overlap[it, len(unique_targ)] = FA_subspace_align(FA, FA_full)
        Overlap[len(unique_targ), it] = FA_subspace_align(FA_full, FA)

    Overlap[len(unique_targ), len(unique_targ)] = FA_subspace_align(FA_full, FA_full)
        
    for it, t in enumerate(unique_targ):
        for iitt, tt in enumerate(unique_targ):
            Overlap[it, iitt] = FA_subspace_align(FA_targ[t], FA_targ[tt])

    return Overlap





