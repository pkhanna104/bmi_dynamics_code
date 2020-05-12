import prelim_analysis as pa
from matplotlib import mlab, cm
import matplotlib.pyplot as plt

import os
import numpy as np
import pickle
import math
import scipy.ndimage

from resim_ppf import ppf_pa
import resim_ppf

import scipy.io as sio
import sklearn.decomposition as skdecomp
import tables

import analysis_config

# gbins = np.linspace(-3., 3., 20)
# jbins = np.linspace(-9., 9., 40)
# ordered_input_type = [[[0], [1, 2]], [[1], [0]], [[0], [1, 2]], [[2], [0, 1, 3]], [[1],
#     [0, 2]], [[2, 3], [0, 1]], [[2], [0, 1]], [[1], [0]], [[0], [1]]]

#pref = '/Volumes/TimeMachineBackups/grom2016/'
#pref = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/'

def get_tuning_hists(te_num, binx=None, biny=None, sample_within_task_dist=False, 
    sample_n_trials = None, animal='grom',
    radial_binning=False, mag_thresh=None, pre_go = None):
    
    '''
    Summary: Method to extract tuning histograms
    Input param: te_num: nested list: inner most is co, obs TEs, next is within day
    Input param: binx: same as biny
    Input param: biny: bin edges to use for the histogram ( binx/biny defines right edges of bins:

         0,  0, ... ||     0   || 1,  1,  ... ||    1    || ... || len(binx) -1, || len(binx), len(binx)
        ai, aj, ... || binx[0] || bi, bj, ... || binx[1] || ... || binx[-1]      || di, dj, ...

    Input param: sample_within_task_dist: Whether or not to get many random distributions of the within task histograms
    Input param: animal: 'grom' or 'jeev' only
    Input param: radial_binning: instead of binning by vx, vy grid, bin by angle and magnitude (16 bins -- inner 8 
        and outer 8)
    Input param: mag_thresh: threshold used to distinguish inner and outer ring in radial binning

    Output param:
        tuning_hist -- outputs of get_tun
        tuning_hist_dist -- outputs of get_tun when only using a random 1/3 of the task data
    '''

    if animal == 'grom':

        pref = analysis_config.config['grom_pref']

        # Get SSKF for animal: 
        try:
            # if on arc: 
            te = dbfn.TaskEntry(te_num)
            hdf = te.hdf
            decoder = te.decoder
        
        except:
            # elif on preeyas MBP
            co_obs_dict = pickle.load(open(pref+'co_obs_file_dict.pkl'))
            hdf = co_obs_dict[te_num, 'hdf']
            hdfix = hdf.rfind('/')
            hdf = tables.openFile(pref+hdf[hdfix:])

            dec = co_obs_dict[te_num, 'dec']
            decix = dec.rfind('/')
            decoder = pickle.load(open(pref+dec[decix:]))
            F, KG = decoder.filt.get_sskf()

        # Get trials: 
        drives_neurons_ix0 = 3
        key = 'spike_counts'
        
        rew_ix = np.array([t[1] for it, t in enumerate(hdf.root.task_msgs[:]) if t[0]=='reward'])

        # decoder_all is now KG*spks
        # decoder_all units (aka 'neural push') is in cm / sec
        bin_spk, targ_i_all, targ_ix, trial_ix_all, decoder_all = pa.extract_trials_all(hdf, rew_ix, 
            drives_neurons_ix0=drives_neurons_ix0, hdf_key=key, keep_trials_sep=True,
            reach_tm_is_hdf_cursor_pos=False, reach_tm_is_kg_vel=True, 
            include_pre_go = pre_go, **dict(kalman_gain=KG))

        _, _, _, _, cursor_state = pa.extract_trials_all(hdf, rew_ix, neural_bins = 100.,
                drives_neurons_ix0=drives_neurons_ix0, hdf_key=key, keep_trials_sep=True,
                reach_tm_is_hdf_cursor_pos=False, reach_tm_is_hdf_cursor_state=True, 
                reach_tm_is_kg_vel=False, include_pre_go= pre_go, **dict(kalman_gain=KG))
        exclude = []
        for i, deci in enumerate(decoder_all):
            if deci.shape[0] == 1:
                exclude.append(i)

    elif animal == 'jeev':
        bin_spk, targ_i_all, targ_ix, trial_ix_all, decoder_all, unbinned, exclude = ppf_pa.get_jeev_trials_from_task_data(te_num, binsize=.1)
        cursor_state = []

    bins_all = np.vstack((bin_spk))

    sub_targ_ix = []
    for trl in np.unique(trial_ix_all):
        ix = np.nonzero(trial_ix_all==trl)[0]
        if pre_go is None:
            sub_targ_ix.append(targ_ix[ix[0]])
        else:
            sub_targ_ix.append(targ_ix[ix[0] + int(pre_go*10) + 2])
    sub_targ_ix_all = np.hstack((sub_targ_ix))

    print 'mag thresh: ', mag_thresh, exclude

    neural_push = decoder_all
    
    tuning_hist, mag_thresh, MAG, ANG = get_tun(binx, biny, bin_spk, 
        decoder_all, animal, radial_binning=radial_binning, mag_thresh=mag_thresh, 
        targ_ix=sub_targ_ix_all, exclude = exclude )

    if sample_within_task_dist:
        select_ix = np.random.permutation(len(bin_spk))
        
        for i in [0, 1, 2, 3]:
            if sample_n_trials is None:
                select_ix2 = select_ix[int(i/3.*len(select_ix)):int((i+1)/3.*len(select_ix))]
            else:
                if i == 0:
                    if len(bin_spk) < sample_n_trials:
                        select_ix2 = np.arange(len(bin_spk))
                    else:
                        select_ix2 = np.arange(sample_n_trials)
                elif i ==3:
                    select_ix2 = np.arange(np.max([0, len(bin_spk)- sample_n_trials]), len(bin_spk))
                elif i ==1:
                    if len(bin_spk) >= 2*sample_n_trials:
                        ii = int(np.floor(len(bin_spk) - 2*sample_n_trials)/2.)
                    else: 
                        ii = 0
                    select_ix2 = np.arange(ii, ii+sample_n_trials)
                elif i ==2: 
                    if len(bin_spk) >= 2*sample_n_trials:
                        ii = int(np.floor(len(bin_spk) - 2*sample_n_trials)/2.)
                        select_ix2 = np.arange(ii+sample_n_trials, ii+2*sample_n_trials)
                    else:
                        ii = len(bin_spk) - sample_n_trials
                        select_ix2 = np.arange(ii, len(bin_spk))

            sub_targ_ix = []
            for trl in select_ix2:
                ix = np.nonzero(trial_ix_all==trl)[0]
                sub_targ_ix.append(targ_ix[ix[0]])
            sub_targ_ix = np.hstack((sub_targ_ix))
            
            N, mag_thresh, _, _ = get_tun(binx, biny, bin_spk, decoder_all, animal, ix = select_ix2, 
                radial_binning=radial_binning, mag_thresh=mag_thresh, targ_ix = sub_targ_ix,
                exclude = exclude)
            if i == 0:
                tuning_hist_dist = {}
                for targI in np.unique(targ_ix):
                    if targI in N.keys():
                        tuning_hist_dist[targI] = [N[targI]]
                #tuning_hist_dist = N[:, :, :, :, np.newaxis]
                # For i = 1:
                #tuning_hist_dist = np.concatenate( (( tuning_hist_dist, np.zeros_like(N[:, :, :, :, np.newaxis]))), 4)
            else:
                for targI in range(len(np.unique(targ_ix))):
                    if targI in N.keys():
                        try:
                            tuning_hist_dist[targI].append(N[targI])
                        except:
                            tuning_hist_dist[targI] = [N[targI]]

                #tuning_hist_dist = np.concatenate( (( tuning_hist_dist, N[:, :, :, :, np.newaxis])), 4)
            
    if sample_within_task_dist:
        return tuning_hist, tuning_hist_dist, mag_thresh, bins_all, targ_ix, MAG, ANG, cursor_state, neural_push
    else:
        return tuning_hist, mag_thresh

def get_tun(binx, biny, bin_spk, decoder_all, animal, ix = None, radial_binning=False, 
    mag_thresh=None, targ_ix = None, exclude = None):
    '''
    Summary: Method to get histogram of binned velocities (usually neural pushes), and neural spike counts given velocities 
    
    Input param: binx: same as biny
    Input param: biny: bin edges to use for the histogram ( binx/biny defines right edges of bins:

         0,  0, ... ||     0   || 1,  1,  ... ||    1    || ... || len(binx) -1, || len(binx), len(binx)
        ai, aj, ... || binx[0] || bi, bj, ... || binx[1] || ... || binx[-1]      || di, dj, ...
    
    Input param: bin_spk: binned spike counts, should be the same format as decoder_all (below)
    Input param: decoder_all: continuous velocities to bin, input format is list, each list item is a np.array
        for each trial

    Input param: animal: 'jeev' or 'grom'
    Input param: ix: only use select trials, not all trials

    Output param: 
        tuning_hist: array (velx, vely, neurons, spk counts), indicating number events with specific spikes counts / bin

    '''

    if exclude is None:
        exclude = []

    if ix is None:
        select_ix = np.arange(len(decoder_all))
    else:
        select_ix = ix

    n_neurons = bin_spk[0].shape[1]

    #For each trial: 
    MAG = []; ANG = []; tuning_hist = {}

    for i in np.unique(targ_ix):
        # Histogram of spike counts of neurons given velocity commands: 
        if radial_binning:
            tuning_hist[int(i)] = np.zeros((8, len(mag_thresh)+1, n_neurons, 40))
        else:
            tuning_hist[int(i)] = np.zeros((len(binx)+1, len(biny)+1, n_neurons, 40))
        
    cnt = 0
    for _, (i, targ) in enumerate(zip(select_ix, targ_ix)):
        if i not in exclude:
            dec = np.array(decoder_all[i])
            targ_number = np.array(targ_ix[cnt:cnt + dec.shape[0]])[1]
            bs = bin_spk[i]

            if radial_binning:
                mag = np.sqrt(dec[:, 3]**2 + dec[:, 5]**2)
                ang = np.array([math.atan2(yi, xi) for i, (xi, yi) in enumerate(zip(dec[:, 3], dec[:, 5]))])
                ang[ang < 0] = ang[ang < 0] + 2*np.pi

                ### THIS CHANGED AS OF 2-7-19 --> NOW RADIAL TUNING IS [[-22.5, 22.5], [22.5, 45.], [45, 67.5], [67.5, 90], etc.]
                ### Changed to be easier on 7-25-19. Same radial tuning, easier code: 

                #boundaries = np.linspace(0, 2*np.pi, 9) - np.pi/8.
                boundaries = np.linspace(0, 2*np.pi, 9) + np.pi/8.

                ### Then subtract minus np.pi
                #ang[ang > boundaries[-1]] = ang[ang > boundaries[-1]] - (2*np.pi)
                #dig = np.digitize(ang, boundaries) - 1 # will vary from 0 - 7

                ## Will vary from 0-7
                dig = np.digitize(ang, boundaries)
                dig[dig == 8] = 0;

                if mag_thresh is None:
                    mag_thresh = np.array([np.percentile(mag, 50)])
                mag_dig = np.digitize(mag, mag_thresh)

                for ii, (ai, mi) in enumerate(zip(dig, mag_dig)):
                    for k in range(n_neurons):
                        tuning_hist[targ][ai, int(mi), k, int(bs[ii, k])] += 1
                MAG.append(mag_dig)
                ANG.append(dig)

            else:
                # Digitize velocities: 
                vx = np.digitize(np.squeeze(np.array(dec[:, 3])), binx)
                vy = np.digitize(np.squeeze(np.array(dec[:, 5])), biny)

                # Add digitized velocities to tuning hist: 
                for i, (ix, iy) in enumerate(zip(vx, vy)):
                    for k in range(n_neurons):
                            tuning_hist[ix, iy, k, int(bs[i, k])] += 1

    return tuning_hist, mag_thresh, MAG, ANG

def get_all_tuning(input_type, binx, biny, fname_pref=None, animal='grom',
    radial_binning=False, mag_thresh=None, sample_n_trials=None, pre_go = None):
    '''
    Summary: 
    Input param: input_type: 
        for grom: -- input_type is nested list of te_nums 
        for jeev: -- input_type is nested list of task filenames (fk.task_filelist)
    
    Input param: binx: see get_tuning_hists
    Input param: biny: see get_tuning_hists
    
    Input param: fname: filename to save data in
    Input param: animal: 'grom' or 'jeev'
    Output param: 
    '''

    # Within each day: 
    for i_d, day_te in enumerate(input_type):
        hist_dict = {}
        hist_dict_dist = {}
                
        if type(mag_thresh) is str:
            mg_thr = pickle.load(open(mag_thresh))
            mt = np.array(mg_thr[animal, i_d])
            print('Day %d, Plot Mag Thresh: %s' %(i_d, str(mt)))

        elif type(mag_thresh) is np.ndarray:
            mt = mag_thresh
        else:
            mt = mag_thresh

        # Within each task: 
        for i_t, task_te in enumerate(day_te):
            
            # For each task entry
            for i, te in enumerate(task_te):
                
                tuning_hist, tuning_hist_dist, _, bins_all, targ_ix, MAG, ANG, cursor_state, neural_push = get_tuning_hists(te, 
                    binx, biny, sample_within_task_dist=True, sample_n_trials=sample_n_trials,
                    animal=animal, radial_binning=radial_binning, mag_thresh=mt, pre_go = pre_go)
                key = 'day'+str(i_d), 'tsk'+str(i_t), 'n'+str(i)

                hist_dict[key] = tuning_hist
                # if tuning_hist[0, 0, 0, 0] > 10e10:
                #     import pdb; pdb.set_trace()
                hist_dict_dist[key] = tuning_hist_dist
                hist_dict[key, 'binned_spikes'] = bins_all
                hist_dict[key, 'targ_ix'] = targ_ix
                hist_dict[key, 'mag'] = MAG
                hist_dict[key, 'ang'] = ANG
                hist_dict[key, 'cursor_state'] = cursor_state
                hist_dict[key, 'neural_push'] = neural_push
                        
        if fname_pref is not None:
            if binx is not None:
                hist_dict['bins'] = binx
            if biny is not None:
                hist_dict_dist['bins'] = binx
            if mt is not None:
                hist_dict_dist['mag_thresh'] = mt
            
            pref = analysis_config.config[animal+'_pref']
            pickle.dump(hist_dict, open(pref+fname_pref+'day'+str(i_d)+'.pkl', 'wb'))
            pickle.dump(hist_dict_dist, open(pref+fname_pref+'day'+str(i_d)+'dist.pkl', 'wb'))
            
            print 'Saved to : ', pref+fname_pref+'day'+str(i_d)+'.pkl'

    if fname_pref is None:
        return hist_dict

def get_mn_fr_REWARD(input_type, animal='grom', fname=None):
    '''
    Summary: Method to extract binned spike count during reward period + input_type_dict 
    Input param: input_type: list of filenames (fk.task_filelist or input_type)
    Input param: animal:
    Output param: 
    '''
    
    # Within each day: 
    ITI_dist = {}
    
    for i_d, day_te in enumerate(input_type):

        # Within each task: 
        for i_t, task_te in enumerate(day_te):
            
            # For each task entry
            for ii, te_num in enumerate(task_te):

                if animal == 'grom':

                    try:
                        # if on arc: 
                        te = dbfn.TaskEntry(te_num)
                        hdf = te.hdf
                    
                    except:
                        # elif on preeyas MBP
                        co_obs_dict = pickle.load(open(pref+'co_obs_file_dict.pkl'))
                        hdf = co_obs_dict[te_num, 'hdf']
                        hdfix = hdf.rfind('/')
                        hdf = tables.openFile(pref+hdf[hdfix:])

                    # Get trials: 
                    drives_neurons_ix0 = 3
                    key = 'spike_counts'
                    rew_ix = np.array([t[1] for it, t in enumerate(hdf.root.task_msgs[:]) if t[0]=='reward'])

                    bin_spk, targ_i_all, targ_ix, trial_ix_all, reach_tm = pa.extract_trials_all(hdf, rew_ix, 
                        drives_neurons_ix0=drives_neurons_ix0, hdf_key=key, keep_trials_sep=True, use_ITI=True)

                elif animal == 'jeev':
                    bin_spk, targ_i_all, targ_ix, trial_ix_all, decoder_all, unbinned, exclude = ppf_pa.get_jeev_trials_from_task_data(te_num,
                        binsize=.1, use_ITI=True)

                trl = np.hstack(([[i]*j.shape[0] for i, j in enumerate(bin_spk)]))
                ITI_bs = np.vstack((bin_spk))

                key = 'day'+str(i_d), 'tsk'+str(i_t), 'n'+str(ii)
                ITI_dist[key, 'trl'] = trl
                ITI_dist[key, 'bs'] = ITI_bs
    if fname is None:
        return ITI_dist
    else:
        sio.savemat(fname, ITI_dist)

def get_unit_metrics(input_type, animal='grom', fname=None):
    '''
    Summary: Used to get units metrics such as: 
        - trial-to-trial variability by target
        - trial-to-trial variability over targets
        - SNR: Gromit: C_mag / Q_mag, PPF: No noise model --> C_mag
        - Kalman Gain / PPF Gain, PPF: P*C.T
        - mod_depth (max - min) of mean to targets
        - mean
        - PSTH by target
    
    Input param: input_type: 
    Input param: animal:
    Input param: fname:

    Output param: 
    '''

    # Within each day: 
    Metrics = {}

    for i_d, day_te in enumerate(input_type):

        # Within each task: 
        for i_t, task_te in enumerate(day_te):
            
            bin_spk_all = []
            bin_spk_by_trl = []
            bin_spk_by_trl_ix = []

            targ_ix_all = []

            # For each task entry
            for ii, te_num in enumerate(task_te):

                if animal == 'grom':

                    try:
                        # if on arc: 
                        te = dbfn.TaskEntry(te_num)
                        hdf = te.hdf
                        decoder = te.decoder
                    
                    except:
                        # elif on preeyas MBP
                        co_obs_dict = pickle.load(open(pref+'co_obs_file_dict.pkl'))
                        hdf = co_obs_dict[te_num, 'hdf']
                        hdfix = hdf.rfind('/')
                        hdf = tables.openFile(pref+hdf[hdfix:])

                        dec = co_obs_dict[te_num, 'dec']
                        decix = dec.rfind('/')
                        decoder = pickle.load(open(pref+dec[decix:]))
                        F, KG = decoder.filt.get_sskf()

                    # Get trials: 
                    drives_neurons_ix0 = 3
                    key = 'spike_counts'
                    rew_ix = np.array([t[1] for it, t in enumerate(hdf.root.task_msgs[:]) if t[0]=='reward'])

                    bin_spk, targ_i_all, targ_ix, trial_ix, reach_tm = pa.extract_trials_all(hdf, rew_ix, 
                        drives_neurons_ix0=drives_neurons_ix0, hdf_key=key, keep_trials_sep=True, use_ITI=False)

                elif animal == 'jeev':

                    bin_spk, targ_i_all, targ_ix, trial_ix, decoder_all, unbinned, exclude = ppf_pa.get_jeev_trials_from_task_data(te_num,
                        binsize=.1, use_ITI=False)

                    data = sio.loadmat('/Volumes/TimeMachineBackups/jeev2013/'+te_num)
                    decoder = data['decoder']
                    KG = decoder['beta'][0][0].T*.01 # offset, X, Y

                #### Aggregate Binned spikes ###
                valid_ix = np.nonzero(targ_ix >= 0)[0]
                b = np.vstack((bin_spk))[valid_ix, :]
                bin_spk_all.append(b)
                targ_ix_all.append(targ_ix[valid_ix])

                for v in np.unique(trial_ix[valid_ix]):
                    ix = np.nonzero(trial_ix[valid_ix]== v)[0][0]

                    bin_spk_by_trl.append(bin_spk[int(v)])
                    bin_spk_by_trl_ix.append(targ_ix[valid_ix][ix])

            ##### Analysis ##### 
            bs_stack = np.vstack((bin_spk_all))
            nneurons = bs_stack.shape[1]
            targ_ix = np.hstack((targ_ix_all))
            trl_ix = np.hstack((bin_spk_by_trl_ix))

            ###################
            ### Variability ###
            ###################
            tmp = {}
            psth = {}
            for i in range(10):
                tmp[i] = []
                psth[i] = []

            for i in range(bs_stack.shape[0]):
                if targ_ix[i] >= 0:
                    tmp[targ_ix[i]].append(bs_stack[i, :])

            assert len(bin_spk_by_trl) == len(trl_ix)

            for i, j in enumerate(trl_ix):
                xx = bin_spk_by_trl[i]
                T = np.min([xx.shape[0], 10])
                if T < 10:
                    supp = np.zeros((10-T, xx.shape[1])) + np.nan
                    psth[j].append(np.vstack((bin_spk_by_trl[i], supp))[np.newaxis, :, :])
                else:
                    psth[j].append(bin_spk_by_trl[i][np.newaxis, :T, :])


            bs_var_by_tg = np.zeros((10, nneurons))
            bs_mn_by_tg = np.zeros((10, nneurons))
            psth_all = np.zeros((10, 10, nneurons))
            
            for i in range(10):
                try:
                    tmp2 = np.vstack((tmp[i]))
                    x = tmp[i]
                    cont = True
                except:
                    # Remove targets that don't have any trials
                    tmp2 = tmp.pop(i)
                    cont = False
                    print 'removing target: ', i

                if cont:
                    tmp[i] = np.vstack((x))
                    bs_var_by_tg[i, :] = np.var(x, axis=0)
                    bs_mn_by_tg[i, :] = np.mean(x, axis=0)
                    psth_all[i, :, :] = np.nanmean(psth[i], axis=0)
                
            Metrics[i_d, i_t, 'bs_var_by_tg'] = bs_var_by_tg
            Metrics[i_d, i_t, 'bs_mn_by_tg'] = bs_mn_by_tg
            Metrics[i_d, i_t, 'psth_by_tg'] = psth_all

            bs_var_across_tg = np.var(bs_stack, axis=0)
            Metrics[i_d, i_t, 'bs_var_across_tg'] = bs_var_across_tg
            
            ################
            ### SNR      ###
            ################

            if animal == 'grom':
                C = decoder.filt.C[:, [3, 5]]
                c = np.sum(np.array(C)**2, axis=1)**.5
                Q = np.diag(decoder.filt.Q)
                q = np.abs(Q)
                SNR = c/q
                K = np.sum((KG[[3, 5], :]**2), axis=0)**.5
            
                Metrics[i_d, i_t, 'SNR'] = SNR
                Metrics[i_d, i_t, 'KG'] = K
                Metrics[i_d, i_t, 'KG_all'] = KG

            elif animal == 'jeev':
                Metrics[i_d, i_t, 'KG'] = KG


            #################
            ### Mod Depth ###
            #################
            mod_depth = np.max(bs_mn_by_tg, axis=0) - np.min(bs_mn_by_tg, axis=0)
            mn = np.mean(bs_stack,axis=0)

            Metrics[i_d, i_t, 'mod_depth'] = mod_depth
            Metrics[i_d, i_t, 'mn'] = mn

    if fname is not None:
        sio.savemat(fname, Metrics)
    else:
        return Metrics

def plot_tuning_of_across_task_vs_within_task(input_type, fname, hist_or_cnt='hist', names=None):
    dat = pickle.load(open(fname))
    
    if hist_or_cnt == 'hist':
        hist = dat['hist_dict']
    elif hist_or_cnt == 'cnt':
        hist = dat['cnt_dict']

    f, ax = plt.subplots()
    x_task_diff = dict()
    co_co_task_diff = dict()
    obs_obs_task_diff = dict()

    #Compute x-task distribution: 
    for i, i_d in enumerate(input_type):
        corr_co = len(i_d[0])
        corr_obs = len(i_d[1])

        co = hist[i, 0]/float(corr_co)
        obs = hist[i, 1]/float(corr_obs)
        if hist_or_cnt == 'cnt':
            norm_diff = np.sum(np.sum(np.sqrt((co[:, :, 0]/float(co[:, :, 0].sum()) - obs[:, :, 0]/float(obs[:, :, 0].sum()))**2), axis=0), axis=0)
        elif hist_or_cnt == 'hist':
            norm_diff = np.sum(np.sum(np.sqrt((co - obs)**2), axis=0), axis=0)
        
        x_task_diff[i] = norm_diff

        co_co = hist[i, 0, 'dist']/float(corr_co)
        obs_obs = hist[i, 1, 'dist']/float(corr_obs)

        co_co_task_diff[i] = []
        obs_obs_task_diff[i] = []

        for j in range(2):
            for k in range(1, 3):
                if k > j:
                    if hist_or_cnt == 'hist':
                        norm_diff_co = np.sum(np.sum(np.sqrt((co_co[:, :, :, k]- co_co[:, :, :, j])**2), axis=0), axis=0)
                        norm_diff_obs = np.sum(np.sum(np.sqrt((obs_obs[:, :, :, k]- obs_obs[:, :, :, j])**2), axis=0), axis=0)
                    
                    elif hist_or_cnt == 'cnt':
                        norm_diff_co = np.sum(np.sum(np.sqrt((co_co[:, :, 0, k]/float(co_co[:, :, 0, k].sum())- 
                            co_co[:, :, 0, j]/float(co_co[:, :, 0, j].sum()))**2), axis=0), axis=0)

                        norm_diff_obs = np.sum(np.sum(np.sqrt((obs_obs[:, :, 0, k]/float(obs_obs[:, :, 0, k].sum())- 
                            obs_obs[:, :, 0, j]/float(obs_obs[:, :, 0, j].sum()))**2), axis=0), axis=0)

                    co_co_task_diff[i].append(norm_diff_co)
                    obs_obs_task_diff[i].append(norm_diff_obs)
        dat = [np.hstack(co_co_task_diff[i]), x_task_diff[i], np.hstack(obs_obs_task_diff[i])]
        
        if hist_or_cnt == 'hist':
            ax.boxplot(dat, positions=[i+.25, i+.5, i+.75])
            ax.set_ylim([0, 800])
        elif hist_or_cnt == 'cnt':
            for jj, j in enumerate([i+.25, i+.5, i+.75]):
                if jj == 1:
                    ax.plot(j, dat[jj], 'r.')
                else:
                    ax.plot(np.zeros(len(dat[jj]))+j, dat[jj], 'b.')
                
            ax.set_ylim([0, 1])

    ax.set_xlim([-1, 9])
    ax.set_ylabel('Norm Difference Between Tuning Plots')
    ax.set_title('Plot Norm Tuning Differences Across Tasks vs. Within Task')
    ax.set_xticks(np.arange(1,10)-0.5)
    if names is not None:
        ax.set_xticklabels(names)

def plot_PD_mag_differences(input_type, dat):
    '''
    Summary: Method to asses preferred direction based on mnFR at each bin
    Input param: input_type: same as above in tuning methods
    Input param: dat: dat file -- OLD -- MUST BE UPDATED
    Output param: 
    '''

    hist  = dat['hist_dict']
    x_task_diff = dict()
    co_co_task_diff = dict()
    obs_obs_task_diff = dict()
    mets  = {}
    mets_null = {}
    for i, i_d in enumerate(input_type):
        corr_co = len(i_d[0])
        corr_obs = len(i_d[1])

        co = hist[i, 0]/float(corr_co)
        obs = hist[i, 1]/float(corr_obs)

        for m in range(2):
            mets[i, m, 'pd'] = []
            mets[i, m, 'mag'] = []

        for j in range(co.shape[2]):
            for m, mat in enumerate([co, obs]):
                filt = scipy.ndimage.filters.gaussian_filter(mat[:, :, j], 2)
                x, y = np.nonzero(filt == np.max(filt))
                if len(x) > 2:
                    proceed = False
                    mets[i, m, 'pd'].append(np.nan)
                    mets[i, m, 'mag'].append(np.nan)  
                    print 'no data: day: ', str(i), ', task: ', str(m), ', unit: ', str(j)                  
                else:
                    x = x[0]
                    y = y[0]
                    proceed = True

                if proceed:
                    if x == 30:
                        x = 29
                    if y == 30:
                        y = 29
                    mets[i, m, 'pd'].append(math.atan2(bins[y], bins[x]))
                    mets[i, m, 'mag'].append(np.sqrt(np.sum((np.array([bins[y], bins[x]])**2))))
    return mets

def plot_mets(mets, tuning_dat=None, save_fname=None, inp_type=None):
    if inp_type is None:
        inp_type = input_type

    f, ax = plt.subplots()#ncols=3, nrows=3)
    #f.set_figheight(5)
    #f.set_figwidth(10)
    ang_bins = np.linspace(-np.pi, np.pi, 20)
    AA = []
    BB = []
    for i in range(len(inp_type)):
        #axi = ax[i/3, i%3]
        #axi.set_title('Day: '+str(i))

        co = mets[i, 0, 'pd']
        obs = mets[i, 1, 'pd']
        A= []
        B = []

        ### For individual units ###
        for j, (c, o) in enumerate(zip(co, obs)):
            proceed = True

            ### ONLY USE SIG MOD UNITS ###
            if tuning_dat is not None: 
                kyz = np.sort(tuning_dat.keys())
                ky = kyz[i]

                if j in tuning_dat[ky]['not_task_mod']:
                    proceed = False
                    print 'skipping ix: ', j, ' on day: ', i, ' for untunedness'
                
            ### DONT use units that don't fire ###
            if np.logical_or(np.isnan(c), np.isnan(o)):
                proceed = False
                print 'skipping ix: ', j, ' on day: ', i
            
            ### COMPUTE dPD ###
            else:
                tmp = c - o
                if tmp > np.pi:
                    tmp = -1*(2*np.pi - tmp)
                elif tmp < -1*np.pi:
                    tmp = 2*np.pi + tmp

                A.append(tmp)
                B.append([mets[i, 0, 'mag'][j], mets[i, 1, 'mag'][j]])

        AA.append(A)
        BB.append(B)

    ######################################
    ##### DISTRIBUTION OF PD CHANGES #####
    ######################################

    plt.boxplot(AA)#, color=(.75, .75, .75, 1))
    ax.set_ylabel('Angle (Radians)')
    ax.set_xticks([])
    ax.set_xlabel('Days')
    ax.set_xlim([-.2, len(inp_type)+1])


    ######################################
    ## DISTRIBUTION OF PD CHANGES vs MD ##
    ######################################
    f2, ax2 = plt.subplots(ncols = 2)
    import plot_factor_tuning_curves
    cmap = plot_factor_tuning_curves.cmap_list + ['k']
    # CO M.D.:
    for k in [0, 1]:
        axi = ax2[k]
        for i in range(len(inp_type)):
            col = cmap[i]
            md = np.vstack((BB[i]))
            axi.plot(np.abs(AA[i]), md[:, k], '.', color=cmap[i])
        axi.set_xlabel(' | Change PD |')
        axi.set_ylabel(' Modulation Depth ')

    if save_fname is not None:
        plt.savefig(save_fname, bbox_inches='tight', pad_inches=1)

def correlate_ssm(cnts, names, thresh_perc = 50, save_fname=None):
    f, ax = plt.subplots()
    f.set_figheight(5)
    f.set_figwidth(2)

    #names =  ['3-2', '3-4', '3-7','3-15', '3-16', '3-17', '3-18', '3-19', '3-19_2']
    for day in range(len(names)):

        cnt_co = cnts[day, 0][:, :, 0].copy()
        cnt_co[cnt_co==.1] = 0 #Remove zero placeholders
        cnt_co = cnt_co /np.sum(cnt_co)

        cnt_obs0 = cnts[day, 1][:, :, 0].copy()
        cnt_obs0[cnt_obs0==.1] = 0
        cnt_obs0 = cnt_obs0 /np.sum(cnt_obs0)

        cnt_co_flat = cnt_co.reshape(-1)
        cnt_co_flat_nonz = cnt_co_flat[cnt_co_flat>0]
        thresh = np.percentile(cnt_co_flat_nonz, thresh_perc)
        cnt_co[cnt_co>=thresh] = 1
        cnt_co[cnt_co<thresh] = 0

        cnt_obs = np.zeros_like(cnt_co)
        cnt_obs[cnt_obs0>=thresh] = 1 

        skin_co = cnt_co.reshape(-1)
        skin_obs = cnt_obs.reshape(-1)

        #Compute overlap: 
        z = np.logical_xor(skin_co, skin_obs)

        ax.plot(day, (len(z) - np.sum(z))/float(len(z)), 'k.', markersize=15)

    ax.set_ylim([0., 1])
    ax.set_xlim([-.5, 0.5+len(names)])
    ax.set_xlabel('Days')
    ax.set_xticks([])
    #ax.set_xticklabels(names)
    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.set_ylabel('Fraction Similarity')
    
    if save_fname is not None:
        plt.savefig(save_fname, bbox_inches='tight', pad_inches=1)

def ex_hists(cnts, save_fname=None):
    f, ax = plt.subplots(ncols=2)
    f.set_figheight(5)
    f.set_figwidth(10)
    bins = np.linspace(-4, 4, 31)

    c_co = cnts[0, 0][:, :, 0]
    c_obs = cnts[0, 1][:, :, 0]

    ax[0].pcolormesh(bins, bins, c_co/np.sum(c_co), vmin=0, vmax=.010)
    ax[1].pcolormesh(bins, bins, c_obs/np.sum(c_obs), vmin=0, vmax=.01)

    for i in [0, 1]:
        ax[i].set_xlabel('X Velocity')
        ax[i].set_ylabel('Y Velocity')
    ax[0].set_title('Day 1: Center Out Task')
    ax[1].set_title('Day 1: Obstacle Task')
    plt.tight_layout()
    plt.savefig('/home/lab/preeya/fa_analysis/cosyne_figs/grom_day0_hist_corr.pdf', bbox_inches='tight', pad_inches=1)

def plot_cont(ax, FR, cursor_vel, bar=False, mesh_plot=False):
    delta = 2/30.
    extent = (-1, 1, -1, 1)
    xvel = zvel = np.linspace(-1, 1, 30)
    
    x_ix = np.digitize(cursor_vel[:,0], xvel)
    z_ix = np.digitize(cursor_vel[:,1], zvel)

    x_ix[x_ix==30] = 29
    z_ix[z_ix==30] = 29

    if mesh_plot:
        x_ix[x_ix==0] = 1
        x_ix = x_ix - 1

        z_ix[z_ix==0] = 1
        z_ix = z_ix - 1

    F = np.zeros((len(xvel), len(zvel)))
    if mesh_plot:
        F = np.zeros((len(xvel) - 1, len(zvel) - 1))

    for x in np.unique(x_ix):
        for z in np.unique(z_ix):
            xx = set(np.nonzero(x_ix==x)[0])
            zz = set(np.nonzero(z_ix==z)[0])

            ix = np.array(list(xx.intersection(zz)))
            if len(ix) > 0:
                F[x, z] = np.mean(FR[ix])

    levels = np.arange(-1, 1, .2)
    norm = cm.colors.Normalize(vmax=1, vmin=-1)
    cmap = cm.PRGn

    if mesh_plot:
        cset = ax.pcolormesh(xvel, zvel, F - np.mean(F), vmin=-1.5, vmax=1.5)

    else:
        cset = ax.contourf(xvel, zvel, F, 10, cmap=cmap, levels=levels, extent=extent)
        CS2 = ax.contour(cset, levels=cset.levels[::2], colors='k',
            hold='on')

    if bar:
        cbar = plt.colorbar(cset)
        cbar.ax.set_ylabel('mean FR')
        cbar.add_lines(CS2)

def plot_all(animal):
    ### Jeev ###

    #histogram data file: 
    if animal == 'jeev':
        from resim_ppf import file_key
        #fname = '/storage/preeya/grom_data/jeev_co_obs_tuning_w_neural_push.pkl'
        fname = '/storage/preeya/grom_data/jeev_co_obs_tuning_to_neural_push_4daysonly.pkl'
        input_ = file_key.task_filelist
        names_ = file_key.task_input_names
        tuning_vel_only = '/storage/preeya/grom_data/jeev_co_obs_tuning_OLS_velocity_only.pkl'
        tuning_vel_pos = '/storage/preeya/grom_data/jeev_co_obs_tuning_OLS_velocity_position.pkl'

    elif animal == 'grom':
        fname = '/storage/preeya/grom_data/grom_co_obs_tuning_w_neural_push.pkl'
        input_ = input_type
        names_ = grom_names
        tuning_vel_only = '/storage/preeya/grom_data/grom_co_obs_tuning_OLS_velocity_only.pkl'
        tuning_vel_pos = '/storage/preeya/grom_data/grom_co_obs_tuning_OLS_velocity_position.pkl'


    # difference b/w w/in task tuning diffs vs. across task tuning diffs
    plot_tuning_of_across_task_vs_within_task(input_, fname, names=names_)

    # plot_PD_mag_differences
    dat = pickle.load(open(fname))
    mets = plot_PD_mag_differences(input_, dat)

    ###############################
    ## Jeev Specific Exploration ##
    ###############################

    pd_max2 = []
    for i in range(len(mets)/4):
        dpd = np.array(mets[i, 0, 'pd']) - np.array(mets[i, 1,'pd'])
        j = np.argmax(np.abs(dpd))
        dpd[j] = 0
        j2 = np.argmax(np.abs(dpd))
        dpd[j2] = 0
        j3 = np.argmax(np.abs(dpd))
        pd_max2.append([j, j2, j3])
    hists = dat['hist_dict']
    bins = np.linspace(-.0025, .0025, 31)*100
    f, ax = plt.subplots(ncols=2)
    f.set_figheight(5)
    f.set_figwidth(10)
    ax[0].pcolormesh(bins, bins, hists[0, 0][:, :, 1]/np.sum(hists[0,0][:, :, 1]), vmin=0, vmax=.01)
    ax[1].pcolormesh(bins, bins, hists[0, 1][:, :, 1]/np.sum(hists[0,1][:, :, 1]), vmin=0, vmax=.01)
    for i in [0, 1]:
        ax[i].set_xlabel(' X Velocity ')
        ax[i].set_ylabel(' Y Velocity ')
    ax[0].set_title('Example Unit Tuning \n Center Out Task')
    ax[1].set_title('Example Unit Tuning \n Obstacle Task')
    plt.tight_layout()
    plt.savefig('/home/lab/preeya/fa_analysis/cosyne_figs/jeev_ex_tuned_unit.pdf', bbox_inches='tight', pad_inches=1)



    dat_both = [pickle.load(open(tuning_vel_only)), pickle.load(open(tuning_vel_pos))]
    nm = ['Vel Tuning Only', 'Vel + Pos Tuning']
    for i_d, d in enumerate(dat_both):
        plot_mets(mets, tuning_dat=d, inp_type=input_,)
        plt.title(nm[i_d])

    # correlate SSM: 
    cnts = dat['cnt_dict']
    correlate_ssm(cnts, names_, thresh_perc=50)

    import plot_factor_tuning_curves
    reload(plot_factor_tuning_curves)
    # Plot tuning model results: 
    for i_d, tun_fn in enumerate([tuning_vel_only, tuning_vel_pos]):
        plot_factor_tuning_curves.plot_model_results(tun_fn)
        plt.title(nm[i_d])

def summary_stats_files(animal):
    import prelim_analysis
    from resim_ppf import file_key
    import subspace_overlap

    if animal == 'jeev':
        input_ = file_key.task_filelist
        names_ = file_key.task_input_type

    elif animal == 'grom':
        input_ = input_type
        names_ = input_type

    for i_d, day in enumerate(input_):
        for i_t, tsk in enumerate(day):
            for i_f, te_num in enumerate(tsk):

                if animal == 'grom':
                    te = dbfn.TaskEntry(te_num)
                    hdf = te.hdf

                    # Get trials: 
                    drives_neurons_ix0 = 3
                    key = 'spike_counts'
                    rew_ix = np.array([t[1] for it, t in enumerate(hdf.root.task_msgs[:]) if t[0]=='reward'])

                    #Note: 'decoder_all' is internal state of decoder, NOT KalmanGain * Neurons
                    bin_spk, targ_i_all, targ_ix, trial_ix_all, decoder_all = pa.extract_trials_all(hdf, rew_ix, 
                        drives_neurons_ix0=drives_neurons_ix0, hdf_key=key, keep_trials_sep=True,
                        reach_tm_is_hdf_cursor_pos=True)
                    print_name = str(te_num)
                    print_len = len(hdf.root.task[:]['cursor'])*(1./60.)*(1/60.) # minutes

                elif animal == 'jeev':
                    bin_spk, targ_i_all, targ_ix, trial_ix_all, decoder_all = ppf_pa.get_jeev_trials_from_task_data(te_num, binsize=.1)
                    print_name = names_[i_d][i_t][i_f]
                    data = sio.loadmat(file_key.task_directory+te_num)
                    print_len = (data['idx'][0, 0]/200.)*(1/60.)

                # GET SOT: 
                FA, mu, num_factors = get_SOT_from_bin_spk(bin_spk)
                SOT = subspace_overlap.get_sot(FA)
                SOT_neuron = subspace_overlap.get_sot(FA, by_neuron=True)


                ### PRINT SUMMARY ###
                # 0) Name: 1) N trials, 2) N units, 2.5) Distribution of MFR, sdFR, 3) minutes, 4) SOT 5) Optimal Number of factors
                print '####### New File: '+print_name+ ' ######## '
                print ' n_units: ', bin_spk[0].shape[1]
                print ' n_trials: ', len(bin_spk)
                print ' minutes: ', print_len
                print ' SOT: ', SOT
                print ' SOT by Neuron: ', SOT_neuron
                print ' Num factors: ', num_factors
                print ' MFR: ', mu

def get_SOT_from_bin_spk(bin_spk):
    zscore_X, mu = pa.zscore_spks(np.vstack((bin_spk)))
    mx = np.min([mu.shape[1], 10])
    log_lik, ax = pa.find_k_FA(zscore_X, iters = 5, max_k = mx, plot=False)
    mn_log_like = np.zeros((log_lik.shape[1], ))

    #Np.nanmean
    for ii in range(log_lik.shape[1]):
        cnt_ = 0
        sum_ = 0
        for jj in range(log_lik.shape[0]):
            if np.isnan(log_lik[jj, ii]):
                pass
            else:                        
                cnt_ += 1
                sum_ += log_lik[jj, ii]
        if cnt_ == 0.:
            cnt_ = .1
        mn_log_like[ii] = sum_/float(cnt_)

    ix = np.argmax(mn_log_like)
    num_factors = ix + 1
    FA_full = skdecomp.FactorAnalysis(n_components = num_factors)
    FA_full.fit(zscore_X)
    return FA_full, mu[0, :]*10, num_factors

def extract_radial_bin_edges_grom_jeev():
    ''' Method to extract and save bin edges for magnitude in scheme
    that bins velocity commands by angle (8 bins: edges = np.linspace(0, 2*np.pi, 9.)) and 
    magnitude: edges = np.linspace(0, 25thperc, 50thperc, 75thperc, 100thperc) fit separately 
    for each day

    Saves data for each animal
    '''
    input_type2 = {}
    input_type2['grom'] = analysis_config.data_params['grom_input_type']
    input_type2['jeev'] = analysis_config.data_params['jeev_input_type']

    boundaries = {}

    for animal in ['grom', 'jeev']:
        inp = input_type2[animal]

        for i_d, day in enumerate(inp):

            # For each day get all commands:
            day_dict = []
            for i_t, task_te in enumerate(day):
            
                # For each task entry
                for i, te in enumerate(task_te):
                    day_dict.append(get_spks(animal, te))
                    
            # Squish all bins into an array: 
            day_dict = np.array(np.vstack((day_dict)))

            # Get angle and magnitude of each
            mag = np.sqrt(day_dict[:, 3]**2 + day_dict[:, 5]**2)

            boundaries[animal, i_d] = [np.percentile(mag, 25), np.percentile(mag, 50), np.percentile(mag, 75)]
    
    pref = analysis_config.config['grom_pref']
    pickle.dump(boundaries, open(pref+'radial_boundaries_fit_based_on_perc_feb_2019.pkl', 'wb'))
    return pref+'radial_boundaries_fit_based_on_perc_feb_2019.pkl', boundaries

def get_spks(animal, te, keep_trls_sep = False):
    if animal == 'grom':

        pref = analysis_config.config['grom_pref']

        co_obs_dict = pickle.load(open(pref+'co_obs_file_dict.pkl'))
        hdf = co_obs_dict[te, 'hdf']
        hdfix = hdf.rfind('/')
        hdf = tables.openFile(pref+hdf[hdfix:])

        dec = co_obs_dict[te, 'dec']
        decix = dec.rfind('/')
        decoder = pickle.load(open(pref+dec[decix:]))
        F, KG = decoder.filt.get_sskf()

        # Get trials: 
        drives_neurons_ix0 = 3
        key = 'spike_counts'
        
        rew_ix = np.array([t[1] for it, t in enumerate(hdf.root.task_msgs[:]) if t[0]=='reward'])

        # decoder_all is now KG*spks
        # decoder_all units (aka 'neural push') is in cm / sec
        bin_spk, targ_i_all, targ_ix, trial_ix_all, decoder_all = pa.extract_trials_all(hdf, rew_ix, 
            drives_neurons_ix0=drives_neurons_ix0, hdf_key=key, keep_trials_sep=True,
            reach_tm_is_hdf_cursor_pos=False, reach_tm_is_kg_vel=True, **dict(kalman_gain=KG))

    elif animal == 'jeev':
        bin_spk, targ_i_all, targ_ix, trial_ix_all, decoder_all, unbinned, exclude = ppf_pa.get_jeev_trials_from_task_data(te, binsize=.1)

    if keep_trls_sep:
        return decoder_all
    else:
        #Squish all bin_spk together:
        return np.vstack((decoder_all))



