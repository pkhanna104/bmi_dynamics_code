import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from . import file_key as fk
import pickle
import os
import analysis_config
from online_analysis import util_fcns

import sys
py_ver = sys.version

if '3.6.15' in py_ver:
    pkl_kw = dict(encoding='latin1')
elif '2.7.8' in py_ver: 
    pkl_kw = dict()


def get_jeev_trials(filename, binsize=.1):
    try:
        dat = sio.loadmat(filename)
    except:
        ix = [j for i, j in enumerate(fk.filelist) if filename in j]
        assert(len(ix)==1)
        dat = sio.loadmat(ix[0])

    strobed = dat['Strobed']

    rew_ix = np.nonzero(strobed[:, 1]==9)[0]
    go_ix = rew_ix - 3
    ix = np.nonzero(strobed[go_ix, 1] == 5)[0]
    ix2 = np.nonzero(strobed[go_ix-1, 1] == 15)[0]
    ix_f = np.intersect1d(ix, ix2)

    rew_ix = rew_ix[ix_f]
    go_ix = go_ix[ix_f]

    # Make sure all 'go indices' 5s. 
    assert np.sum(np.squeeze(strobed[go_ix, 1] - 5)) == 0

    times = list(zip(strobed[go_ix, 0], strobed[rew_ix, 0]))

    # Get decoder: 
    ix = [i for i, j in enumerate(fk.filelist) if filename in j]

    # Get units
    assert(len(ix) == 1)
    decname = fk.decoderlist[ix[0]]
    dec = sio.loadmat(decname)
    unitlist = dec['decoder'][0]['predSig'][0][0]
    dat_units = [dat[k[0]] for i, k in enumerate(unitlist)]

    # Binning: 
    bin_spk = _bin(times, dat_units, binsize)
    units_per = np.array([bs.shape[0] for i, bs in enumerate(bin_spk)])

    # Get Target info
    start_ix = strobed[go_ix - 3, 1]
    start_ix[start_ix == 400] = 2;
    start_ix[start_ix == 15] = 2;

    if np.sum(np.squeeze(start_ix - 2)) == 0:
        task = 'co'
    else:
        task = 'obs'

    if task == 'co':
        targ = strobed[go_ix - 2, 1]
        targ_ix = np.digitize(targ, fk.cotrialList) - 1
        assert np.sum(fk.cotrialList[targ_ix] - targ) == 0

    elif task == 'obs':
        targ = [strobed[g-4:g-1, 1] for i, g in enumerate(go_ix)]
        targ_ix = []
        for i, tg in enumerate(targ):
            tmp = np.tile(tg[np.newaxis, :], [len(fk.obstrialList), 1])
            ix = np.nonzero(np.sum(np.abs(fk.obstrialList - tmp), 1) == 0)[0]
            assert(len(ix)==1)
            targ_ix.append(ix[0])
        targ_ix = np.array(targ_ix)

    # Targ_ix, trial_ix
    targ_IX = []
    trial_IX = []
    for b, nb in enumerate(units_per):
        targ_IX.extend([targ_ix[b]]*nb)
        trial_IX.extend([b]*nb)

    # Decoder velocity outputs from AD 39/40 
    decoder_all = _bin_ad(times, dat, binsize)
    targ_i_all = []
    return bin_spk, targ_i_all, np.array(targ_IX), np.array(trial_IX), decoder_all

def get_jeev_trials_from_task_data(filename, include_pos = False, include_vel = False, binsize=.1, 
    use_ITI=False, pre_go=0., get_ixs = False):

    if 'jeev082413_VFB_PPF_B100_NS5_NU20_Z1_assist_ofc_cont_cont_assist_ofc_fixData' in filename:
        start_index_overall = 55003
    else:
        start_index_overall = 0

    unbinned = dict()

    filelist_task = []
    for i, j in enumerate(fk.task_filelist):
        for k, l in enumerate(j):
            filelist_task.extend(l)
    try:
        dat = sio.loadmat(filename)
    except:
        ix = [j for i, j in enumerate(filelist_task) if filename in j]
        assert(len(ix)==1)
        try:
            dat = sio.loadmat(fk.task_directory+ix[0])
        except:
            dat = sio.loadmat(fk.task_directory_mbp+ix[0])

    strobed = make_strobed(dat, start_index_overall)

    rew_ix = np.nonzero(strobed[:, 1]==9)[0]
    go_ix = rew_ix - 3
    ix = np.nonzero(strobed[go_ix, 1] == 5)[0]
    ix2 = np.nonzero(strobed[go_ix-1, 1] == 15)[0]
    ix_f = np.intersect1d(ix, ix2)

    rew_ix = rew_ix[ix_f]
    go_ix = go_ix[ix_f]

    if use_ITI:
        go_ix = rew_ix.copy()
        rew_ix = rew_ix.copy() + 3 

    # Make sure all 'go indices' 5s. 
    if use_ITI:
        assert np.sum(np.squeeze(strobed[go_ix, 1] - 9)) == 0
    else:
        assert np.sum(np.squeeze(strobed[go_ix, 1] - 5)) == 0
    
    ixs_og = list(zip(strobed[go_ix, 0], strobed[rew_ix, 0]))
    ixs = []

    # Ensure only indices > start_index_overall: 
    for ii, (ji, ki) in enumerate(ixs_og):
        if np.logical_and(ji > start_index_overall, ki > start_index_overall):
            ixs.append((ji, ki))

    # Binning: 
    spk_counts = dat['spike_counts'] # changed from 'spike_counts_all' to 'spike_counts', 2-7-19
    spk_counts_dt = float(dat['Delta_PPF'])
    assert spk_counts_dt == 0.005
    
    bin_spk, bin_spk_ub, exclude = _bin_spike_counts(ixs, spk_counts, spk_counts_dt, binsize, pre_go)
    unbinned['spike_counts'] = bin_spk_ub # changed from 'spike_counts_all' to 'spike_counts'
    units_per = np.array([bs.shape[0] for i, bs in enumerate(bin_spk)])
    unbinned_units_per = np.array([ix[1]-ix[0] for ix in ixs])

    # Get Target info
    start_ix = strobed[go_ix - 3, 1]
    start_ix[start_ix == 400] = 2;
    start_ix[start_ix == 15] = 2;

    if np.sum(np.squeeze(start_ix - 2)) == 0:
        task = 'co'
    else:
        task = 'obs'

    if task == 'co':
        targ = strobed[go_ix - 2, 1]
        targ_ix = targ - 64
        assert np.sum(fk.cotrialList[targ_ix] - targ) == 0

    elif task == 'obs':
        targ = [strobed[g-4:g-1, 1] for i, g in enumerate(go_ix)]
        targ_ix = []
        for i, tg in enumerate(targ):
            tmp = np.tile(tg[np.newaxis, :], [len(fk.obstrialList), 1])
            ix = np.nonzero(np.sum(np.abs(fk.obstrialList - tmp), 1) == 0)[0]
            if len(ix)==1:
                targ_ix.append(ix[0])
            else:
                targ_ix.append(-1)
        targ_ix = np.array(targ_ix)

    # Targ_ix, trial_ix
    targ_IX = []
    trial_IX = []
    for b, nb in enumerate(units_per):
        targ_IX.extend([targ_ix[b]]*nb)
        trial_IX.extend([b]*nb)

    targ_IX_UB = []
    for b, nb in enumerate(unbinned_units_per):
        targ_IX_UB.extend([targ_ix[b]]*nb)

    unbinned['target_ix'] = np.hstack((targ_IX_UB))

    if task == 'co':
        unbinned['target_loc'] = fk.targ_ix_to_loc(np.hstack((targ_IX_UB)))
        targ_i_all = fk.targ_ix_to_loc(np.array(targ_IX))
    elif task == 'obs':
        unbinned['target_loc'] = fk.targ_ix_to_loc_obs(np.hstack((targ_IX_UB)))
        targ_i_all = fk.targ_ix_to_loc_obs(np.array(targ_IX))

    # Decoder velocity outputs from AD 39/40 
    #try:
    decoder_all, decoder_all_ub = _bin_neural_push(ixs, filename, binsize, start_index_overall, pre_go)
    unbinned['neural_push'] = decoder_all_ub
    # #except:
    #     import pdb; pdb.set_trace()
    #     decoder_all = 0
    #     unbinned['neural_push'] = 0
    unbinned['ixs'] = ixs
    unbinned['start_index_overall'] = start_index_overall
    
    if include_pos:
        cursor_kin = dat['cursor_kin']
        #import pdb; pdb.set_trace()
        print('Xlims: %.2f, %.2f'%(dat['horiz_min'][0, 0], dat['horiz_max'][0, 0]))
        print('Ylims: %.2f, %.2f'%(dat['vert_min'][0, 0], dat['vert_max'][0, 0]))
        bin_ck, ck = _bin_cursor_kin(ixs, cursor_kin, binsize, pre_go)
        print(len(bin_ck), bin_ck[0].shape)
        unbinned['cursor_kin'] = ck
        return bin_spk, targ_i_all, np.array(targ_IX), np.array(trial_IX), decoder_all, bin_ck, unbinned, exclude
    else:
        return bin_spk, targ_i_all, np.array(targ_IX), np.array(trial_IX), decoder_all, unbinned, exclude

def _bin(times, dat_units, binsize):
    bin_spk = []
    for i, (t0, t1) in enumerate(times):
        binedges = np.arange(t0, t1+binsize, binsize)
        X =  np.zeros((len(binedges), len(dat_units) ))
        for u, unit in enumerate(dat_units):
            rel_ix = np.nonzero(np.logical_and(unit[:, 0] >= t0, unit[:, 0] < binedges[-1]))[0]
            ts = unit[rel_ix, 0]
            ts_dig = np.digitize(ts, binedges)
            for t, tsi_dig in enumerate(ts_dig):
                X[tsi_dig, u] += 1
        bin_spk.append(X)
    return bin_spk

def _bin_spike_counts(ixs, spike_counts, spk_counts_dt, binsize, pre_go):
    ''' pre_go_in seconds here...'''
    n_per_bin = int(binsize/spk_counts_dt)
    if pre_go is not None:
        pre_go_bins = int(pre_go/spk_counts_dt)
    else:
        pre_go_bins = 0

    exclude = []

    n_units = spike_counts.shape[0]
    bin_spk = []
    bin_spk_ub = []
    for i, (ix0, ix1) in enumerate(ixs):
        if ix0 > pre_go_bins:
            binedges = np.arange(ix0-pre_go_bins, ix1+n_per_bin, n_per_bin)
            X = np.zeros((len(binedges)-1, spike_counts.shape[0]))
            Xub = spike_counts[:, ix0-pre_go_bins:ix1]
            for b, bn in enumerate(binedges[:-1]):
                X[b, :] = np.sum(spike_counts[:, bn:binedges[b+1]], 1)
        else:
            exclude.append(i)
            X = np.zeros((1, spike_counts.shape[0]))
        bin_spk.append(X)
        bin_spk_ub.append(Xub)
    return bin_spk, bin_spk_ub, exclude

def _bin_cursor_kin(ixs, cursor_kin, binsize, pre_go = 0.):
    n_per_bin = int(binsize/.005)
    print(('Bin Curson kin Size %d' %(n_per_bin)))
    pre_go_bins = int( pre_go / .005 ); 

    bin_ck = []
    ck = []
    z = np.zeros_like(cursor_kin[0, :])
    pos_vel = np.vstack((cursor_kin[0, :], z, cursor_kin[1, :], cursor_kin[2, :], z, cursor_kin[3, :], z+1.)).T
    for i, (ix0, ix1) in enumerate(ixs):
        binedges = np.arange(ix0-pre_go_bins, ix1+n_per_bin, n_per_bin)
        X = np.zeros((len(binedges)-1, pos_vel.shape[1]))
        Xub = pos_vel[ix0 - pre_go_bins:ix1, :]
        for b, bn in enumerate(binedges[:-1]):
            X[b, :] = np.sum(pos_vel[bn:binedges[b+1], :], 0)
        bin_ck.append(X)
        ck.append(Xub)
    return bin_ck, ck

def _bin_neural_push(ixs, filename, binsize, start_index_overall, pre_go = 0.):
    # First load neural push
    #neuralpush_fn = os.path.expandvars('$FA_GROM_DATA/jeev_neural_push.pkl')
    
    neuralpush_fn = '/Volumes/TimeMachineBackups/jeev2013/jeev_neural_push_apr2017.pkl'
    try:
        neuralpush = pickle.load(open(neuralpush_fn, 'rb'), **pkl_kw)
    except:
        neuralpush = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/jeev_neural_push_apr2017.pkl', 'rb'), **pkl_kw)
    
    # Get correct tag

    for i, dayfn in enumerate(fk.task_filelist):
        for j, blkfn in enumerate(dayfn):
            for k, fn in enumerate(blkfn):
                if fn == filename:
                    tix = [i, j, k]
    
    neuralpush_spec = neuralpush[fk.task_input_type[tix[0]][tix[1]][tix[2]]]

    #Convert m to cm:
    neuralpush_spec = 100*neuralpush_spec

    #assert binsize == 0.1
    dt_per_bin = 0.005
    n_per_bin = int(binsize/dt_per_bin)
    pre_go_bins = int(pre_go/dt_per_bin)

    bin_ck = []
    ck = []
    for i, (ix0, ix1) in enumerate(ixs):
        binedges = np.arange(ix0 - start_index_overall - pre_go_bins, 
            ix1+n_per_bin - start_index_overall, n_per_bin)
        X = np.zeros((len(binedges)-1, neuralpush_spec.shape[0]))
        Xub = neuralpush_spec[:, ix0 - pre_go_bins:ix1]
        for b, bn in enumerate(binedges[:-1]):
            ''' Get mean within 100 ms bins'''
            X[b, :] = np.sum(neuralpush_spec[:, bn:binedges[b+1]], 1)
        bin_ck.append(X)
        ck.append(Xub)
    return bin_ck, ck

def _bin_ad(times, dat, binsize):
    z = np.zeros_like(dat['AD37'])
    pos_vel = np.hstack((dat['AD39'], z, dat['AD40'], dat['AD37'], z, dat['AD38'], z+1.)) # Pos and velocity
    bin_ad = []
    for i, (t0, t1) in enumerate(times):
        binedges = np.arange(t0, t1+binsize, binsize)
        X = np.zeros((len(binedges)-1, pos_vel.shape[1]))
        for i, b in enumerate(binedges[:-1]):
            X[i, :] = np.mean(pos_vel[int(b*1000):int(binedges[i+1]*1000), :], 0)
        bin_ad.append(X)
    return bin_ad

def get_targ_ix(strobed, go_ix, task):
    if task == 'co':
        targ = strobed[go_ix - 2, 1]
        targ_ix = targ - 64
        assert np.sum(fk.cotrialList[targ_ix] - targ) == 0

    elif task == 'obs':
        targ = [strobed[g-4:g-1, 1] for i, g in enumerate(go_ix)]
        targ_ix = []
        for i, tg in enumerate(targ):
            tmp = np.tile(tg[np.newaxis, :], [len(fk.obstrialList), 1])
            ix = np.nonzero(np.sum(np.abs(fk.obstrialList - tmp), 1) == 0)[0]
            if len(ix)==1:
                targ_ix.append(ix[0])
            else:
                targ_ix.append(-1)
        targ_ix = np.array(targ_ix)
    return targ_ix

def make_strobed(dat, start_index_overall):
    strobed = []
    events = dat['task_events']

    for i, e in enumerate(events):
        ### Added 10-3-19 -- only keep if > start_index_overall; 
        if i >= start_index_overall:
            if np.any(np.array(e[0].shape) == 0):
                skip = 1
            else:
                for j, ee in enumerate(e[0]):
                    strobed.append([i, ee[0]])
    return np.array(strobed)

def plot_jeev_trials(task = 'obs', targ_only = 3, day_ix = 0, binsize = 0.1):
    input_type = fk.task_filelist
    
    if task == 'co':
        te_num = input_type[day_ix][0][0]
    
    elif task == 'obs':
        te_num = input_type[day_ix][1][0]
    
    bin_spk, targ_i_all, targ_ix, trial_ix_all, decoder_all, cursor_state, unbinned, exclude = get_jeev_trials_from_task_data(te_num,
        include_pos=True, binsize=binsize)

    colors = ['maroon', 'orangered', 'goldenrod','olivedrab','teal', 'steelblue', 'midnightblue', 'darkmagenta', 'k', 'brown']

    targs = np.unique(targ_ix)
    targs = targs[targs >= 0] # skip -1

    if targ_only is not None:
        targs = targs[targs == targ_only]

    if task == 'co':
        f, ax = plt.subplots(ncols = len(targs), figsize=(20, 3))
    
    elif task == 'obs':
        f, ax = plt.subplots(ncols = len(targs), nrows = 2, figsize=(20, 6))
        ax[0, 0].set_ylabel('CW')
        ax[1, 0].set_ylabel('CCW')
        
    if len(targs) == 1:
        ax = [ax]

    for ti, i in enumerate(targs):


        ### Get trials with the right target number: 
        ix = np.nonzero(targ_ix == i)[0]

        ### Now figure out which trial: 
        trl_ix = np.unique(trial_ix_all[ix])

        ### Plot the target: 
        if task == 'co':
            ax[ti].set_xlim([-1.5, 3.0])
            ax[ti].set_ylim([1,  4.5])

            ### Targ
            tmp = np.linspace(0, 2*np.pi, 1000)
            tmp_x = .013*np.cos(tmp) + targ_i_all[ix[0], 0]
            tmp_y = .013*np.sin(tmp) + targ_i_all[ix[0], 1]

            ### Center
            tmp2_x = .013*np.cos(tmp) + 0.0377292
            tmp2_y = .013*np.sin(tmp) + 0.1383867       


            if binsize == 0.1:
                ax[ti].plot(tmp_x*20, tmp_y*20, 'k-')
                ax[ti].plot(tmp2_x*20, tmp2_y*20, 'k-')
                centerPos = np.array([0.0377292, 0.1383867])*20
                targetPos = targ_i_all[ix[0], [0, 1]]*20
            
            elif binsize == 0.005:
                ax[ti].plot(tmp_x*1, tmp_y*1, 'k-')
                ax[ti].plot(tmp2_x*1, tmp2_y*1, 'k-')         
                centerPos = np.array([0.0377292, 0.1383867])*1    
                targetPos = targ_i_all[ix[0], [0, 1]]*1   
        
        else:
            #tg = targ_i_all[ix[0], :]

            ### Subtract obstacle target center; 
            #tg = tg - np.array([0., 2.5])

            ### Convert cm --> m 
            #tg = tg / 100.

            ### Add back the other center; 
            #tg = tg + np.array([.04, .14])
            for axrow in range(2):
                ax[axrow, ti].set_xlim([-1.5, 3.0])
                ax[axrow, ti].set_ylim([1,  4.5])

            #### Test obstacles ####
            x = sio.loadmat('resim_ppf/jeev_obs_positions_from_amy.mat')
            targ_list = fk.obstrialList
            targ_series = targ_list[i, :] - 63
            TC0 = x['targObjects'][:, :, int(targ_series[0])-1]
            TC1 = x['targObjects'][:, :, int(targ_series[1])-1]
            TC2 = x['targObjects'][:, :, int(targ_series[2])-1]

            for t in [TC0, TC1, TC2]:

                ### Subtract center;
                ### Update 10/6/20 --> this isn't necessary;  
                #t_dem = t - np.array([0., 2.5])[:, np.newaxis]

                ### Convert from cm --> m 
                t_m = t / 100.

                ### Add other center; 
                centerpos = np.array([ 0.0377292,  0.1383867])
                t_ = t_m + centerpos[:, np.newaxis]

                if binsize == 0.005:
                    for axrow in range(2):
                        ax[axrow, ti].plot(t_[0, :], t_[1, :], 'k-')

                elif binsize == 0.1:
                    for axrow in range(2):
                        ax[axrow, ti].plot(t_[0, :]*20, t_[1, :]*20, 'k-')
            
            #### Get cneterpos; 
            centerPos = np.mean(TC0, axis=1)/100. + centerpos
            targetPos = np.mean(TC2, axis=1)/100. + centerpos
            
            if binsize == 0.1:
                centerPos = centerPos*20
                targetPos = targetPos*20

        ######### Plot each trial; ############
        for trl in trl_ix:

            ### This a CW or CCW trial  ?
            if task == 'obs':
                
                if i in [1, 3, 4, 5, 8, 9]: 
                    axros = analysis_config.jeev_cw_ccw_dict[i]
                elif day_ix == 3 and i == 2:
                    axrow, s = CW_CCW_obs(centerPos, targetPos, cursor_state[trl][:, [0, 2]].T)
                    if s < 0.1: 
                        axrow = 1
                else:
                    axrow, _ = CW_CCW_obs(centerPos, targetPos, cursor_state[trl][:, [0, 2]].T)
                
                ax[axrow, ti].plot(cursor_state[trl][:, 0], cursor_state[trl][:, 2], '-', color = colors[i], linewidth=1.0)
                ax[axrow, ti].plot(cursor_state[trl][0, 0], cursor_state[trl][0, 2], 'r.')
                ax[axrow, ti].set_title("Targ %d" %(i))
            else:
                ax[ti].plot(cursor_state[trl][:, 0], cursor_state[trl][:, 2], '-', color = colors[i], linewidth=1.0)
                ax[ti].plot(cursor_state[trl][0, 0], cursor_state[trl][0, 2], 'r.')
                ax[ti].set_title("Targ %d" %(i))
    f.tight_layout()

def plot_percent_correct_t2t(plot=True, min_obs_targ = 2): 
    input_type = fk.task_filelist
    
    metrics = dict()
    metrics['co_pc'] = []
    metrics['obs_pc'] = []

    metrics['co_tt'] = []
    metrics['co_tt_mn'] = []

    metrics['obs_tt'] = []
    metrics['obs_tt_mn'] = []   

    metrics['perc_fulfill_obs'] = []
    metrics['perc_fulfill_obs_mn'] = []
     
    
    tsk_keys = ['co', 'obs']

    f, ax = plt.subplots(figsize = (3, 4))
    f2, ax2 = plt.subplots(figsize = (3, 4))
    f3, ax3 = plt.subplots(figsize = (3, 4))

    for i_d, day in enumerate(input_type):
        day_perc_fulfill = []; 

        for i_tsk, tsk in enumerate(day):
            for i_te, filename in enumerate(tsk):

                tsk_key = tsk_keys[i_tsk]

                ####### Get percent correct #############
                if 'jeev082413_VFB_PPF_B100_NS5_NU20_Z1_assist_ofc_cont_cont_assist_ofc_fixData' in filename:
                    start_index_overall = 55003
                else:
                    start_index_overall = 0

                ####### Load the correct file #######
                filelist_task = []
                for i, j in enumerate(fk.task_filelist):
                    for k, l in enumerate(j):
                        filelist_task.extend(l)
                try:
                    dat = sio.loadmat(filename)
                except:
                    ix = [j for i, j in enumerate(filelist_task) if filename in j]
                    assert(len(ix)==1)
                    
                    try:
                        dat = sio.loadmat(fk.task_directory+ix[0])
                    except:
                        dat = sio.loadmat(fk.task_directory_mbp+ix[0])
                
                ###### Get strobed #########
                strobed = make_strobed(dat, start_index_overall)
                cursor_kin = dat['cursor_kin']

                go_ix = np.nonzero(strobed[:, 1] == 5)[0]
                print(('1. Total Go Ix: %d' %(len(go_ix))))

                ###### Successfully get out of the center #####
                keep_ix = np.nonzero(strobed[go_ix+1, 1] == 6)[0]
                go_ix = go_ix[keep_ix]
                go_ix = go_ix[go_ix < len(strobed) - 3]
                print(('2. Go Ix, out of center: %d' %(len(go_ix))))
                
                ##### Remove targets 0, 1 for obstacle; 
                ### Get target index: 
                targ_ix = get_targ_ix(strobed, go_ix, tsk_key)

                if tsk_key == 'obs':
                    keep_ix = np.nonzero(targ_ix >= min_obs_targ)[0]
                else:
                    keep_ix = np.arange(len(targ_ix))

                go_ix = go_ix[keep_ix]
                print(('3. Go Ix, rm targs obs only: %d' %(len(go_ix))))
                
                ##### Obstacle collision = 300 ########
                obs_coll_ix = np.nonzero(strobed[go_ix+2, 1] == 300)[0]

                ##### Target timeout = 12 ##########
                timeout_ix = np.nonzero(strobed[go_ix+2, 1] == 12)[0]

                ###### Target hold error = 8 #######
                the_ix = np.nonzero(strobed[go_ix + 3, 1] == 8)[0]

                ##### Rew Ix 
                rew_ix = np.nonzero(strobed[go_ix + 3, 1] == 9)[0]

                #### Are all trials accounted for ####? 
                ### Make sure no overlap 
                assert len(np.unique(np.hstack((rew_ix, the_ix, timeout_ix, obs_coll_ix)))) == len(np.hstack((rew_ix, the_ix, timeout_ix, obs_coll_ix)))
                
                ### Make sure all trials accounted for ###
                if len(np.unique(np.hstack((rew_ix, the_ix, timeout_ix, obs_coll_ix)))) == len(go_ix):
                    N_trls = len(go_ix)
                else:
                    ### These notes for for go_ix with all obs targets; May be different if ignore obs target 0, 1; 
                    ### this happened once in day 0 --> 5, 6, 15, 4, go_ix[205]
                    ### with 25 sec in between 6 --> 15 
                    ### Maybe kinarm task restarted? 

                    ### happened once day 1 --> go_ix[185], seems like skipped 9? 

                    ### Day 2 --> [5, 6, 9] at go_ix[132], seems like skipped 7; 
                    print(('Day %d, Task %d, Discrepancy %d' %(i_d, i_tsk, len(go_ix) - len(np.hstack((rew_ix, the_ix, timeout_ix, obs_coll_ix))))))
                    N_trls = len(np.unique(np.hstack((rew_ix, the_ix, timeout_ix, obs_coll_ix))))
                    
                #### Add percent correct 
                metrics[tsk_key + '_pc'].append(float(len(rew_ix)) / float(N_trls))

                #### Trial time #####
                trl_time = strobed[go_ix[rew_ix] + 3, 0] - strobed[go_ix[rew_ix], 0]
                metrics[tsk_key + '_tt'].append(trl_time*.005)
                metrics[tsk_key + '_tt_mn'].append(np.mean(trl_time*.005))

                ##### For each rewarded trial, get the 3 targets: 
                if i_tsk == 0: 

                    for i_trl, trl_go in enumerate(go_ix[rew_ix]): 
                        targ_ = strobed[trl_go - 2, 1]
                        go_ = strobed[trl_go, 0]
                        rew_ = strobed[trl_go + 3, 0]

                        ### Get cursor trajectories ###
                        trl_cursor = cursor_kin[[0, 1], go_:rew_] - fk.centerPos[:, np.newaxis]

                        ### Which target are we in ####
                        targ_ix = np.nonzero(fk.cotrialList == targ_)[0]
                        tp = fk.targetPos[targ_ix, :] - fk.centerPos; 
                        tp_norm = tp / np.linalg.norm(tp)

                        ### Rotate the cursor #### 
                        angle = np.arctan2(tp_norm[0, 1], tp_norm[0, 0])
                        Rot = np.array([[np.cos(-1*angle), -np.sin(-1*angle)], [np.sin(-1*angle), np.cos(-1*angle)]])
    
                        ### Does the path go through any of the obstacle targets? How many? 
                        trl_cursor_rot = np.dot(Rot, trl_cursor)
                        perc_obs_viol = test_perc_obs_viol(trl_cursor_rot, min_obs_targ, plot=plot)
                        
                        metrics['perc_fulfill_obs'].append(1. - perc_obs_viol)
                        day_perc_fulfill.append(1. - perc_obs_viol)

                        if plot:
                            import pdb; pdb.set_trace()

        day_perc_fulfill = np.hstack((day_perc_fulfill))
        metrics['perc_fulfill_obs_mn'].append(np.mean(day_perc_fulfill))

    ##### plot these guys #####
    ########### PERCENT CORRECT ##############
    util_fcns.draw_plot(0, np.hstack((metrics['co_pc'])), 'g', [1., 1., 1., 0.], ax, width = .5)
    util_fcns.draw_plot(1, np.hstack((metrics['obs_pc'])),'b', [1., 1., 1., 0.], ax, width = .5)
    print(('mean perc correct CO %.2f, OBS %.2f' %(np.mean(np.hstack((metrics['co_pc']))), 
        np.mean(np.hstack((metrics['obs_pc']))))))

    for _, (c, o) in enumerate(zip(metrics['co_pc'], metrics['obs_pc'])):
        ax.plot([0, 1], [c, o], '-', color='gray', linewidth=0.5)
    ax.set_xlim([-0.8, 1.8])
    ax.set_ylim([0., 1.])
    ax.set_ylabel('Percent Correct')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['CO', 'OBS'], rotation=45)

    ########### Target TIME ##############
    util_fcns.draw_plot(0, np.hstack((metrics['co_tt'])), 'k', [1., 1., 1., 0.], ax2, width = .5)
    util_fcns.draw_plot(1, np.hstack((metrics['obs_tt'])),'k', [1., 1., 1., 0.], ax2, width = .5)
    for _, (c, o) in enumerate(zip(metrics['co_tt_mn'], metrics['obs_tt_mn'])):
        ax2.plot([0, 1], [c, o], '-', color='gray', linewidth=0.5)
    ax2.set_xlim([-0.8, 1.8])
    ax2.set_ylim([0., 7.5])
    ax2.set_ylabel('Time to Target (sec)')
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['CO', 'OBS'], rotation=45)
    f2.tight_layout()
    util_fcns.savefig(f2, 't2t_jeev')

    ########### Target TIME ##############
    util_fcns.draw_plot(0, np.hstack((metrics['perc_fulfill_obs'])), 'g', [1., 1., 1., 0.], ax3, width = .5)
    for _, o in enumerate(metrics['perc_fulfill_obs_mn']):
        ax3.plot(0, o, '.', color='gray')
    ax3.set_xlim([-0.8, 0.8])
    ax3.set_ylim([0., 1.0])
    ax3.set_ylabel('% Fulfill Obs.')
    ax3.set_xticks([0])
    ax3.set_xticklabels(['CO'], rotation=45)

def test_perc_obs_viol(trl_curs, min_obs_targ, plot=False): 

    ### Rotate and stretch all the targets to line up to 
    x = sio.loadmat('resim_ppf/jeev_obs_positions_from_amy.mat')
    targ_list = fk.obstrialList
    targ_list = targ_list[min_obs_targ:, :]
    rad = 0.065

    if plot:
        f, ax = plt.subplots()
        ax.plot(trl_curs[0, :], trl_curs[1, :], 'k-')
        ax.plot(trl_curs[0, 0], trl_curs[1, 0], 'r.')

    violate = np.zeros((targ_list.shape[0]))

    for i in range(targ_list.shape[0]):
        targ_series = targ_list[i, :] - 63
        TC0 = np.mean(x['targObjects'][:, :, int(targ_series[0])-1], axis=1)
        TC2 = np.mean(x['targObjects'][:, :, int(targ_series[2])-1], axis=1) - TC0
        
        ### Center by TC0
        ### Rotate by TC2; 
        ### What is the angle with respect to center point; 
        TC2_norm = TC2 / np.linalg.norm(TC2)
        angle = np.arctan2(TC2_norm[1], TC2_norm[0])
        R = np.array([[np.cos(-1*angle), -np.sin(-1*angle)], [np.sin(-1*angle), np.cos(-1*angle)]])
        
        ### Scaling on x-axis ###
        TC2_rot = np.dot(R, TC2[:, np.newaxis])
        scale_all = rad/TC2_rot[0]
        
        ### Now get the obstacle target; 
        targ = targ_series[1]
        TC1 = x['targObjects'][:, :, int(targ)-1] # 2 x 100 
        T_dem = TC1-TC0[:, np.newaxis]
        T_rot = np.dot(R, T_dem)
        T_rot_scale = T_rot*scale_all; 

        if plot: 
            #### Plot obstacle ###
            ax.plot(T_rot_scale[0, :], T_rot_scale[1, :], 'b-')

        ### Does the cursor pass through this?
        xrad = 0.5*(np.max(T_rot_scale[0, :]) - np.min(T_rot_scale[0, :]))
        yrad = 0.5*(np.max(T_rot_scale[1, :]) - np.min(T_rot_scale[1, :]))

        ### Get the center point; 
        xmn = np.mean(T_rot_scale[0, :])
        ymn = np.mean(T_rot_scale[1, :])

        val = ((trl_curs[0, :] - xmn)**2 / (xrad**2)) + ((trl_curs[1, :] - ymn)**2 / (yrad**2))
        if np.any(val <= 1): 
            ix = np.nonzero(val <= 1)[0]
            violate[i] = 1; 

            if plot: 
                ax.plot(trl_curs[0, ix], trl_curs[1, ix], 'm.')

    return np.sum(violate) / float(len(violate))

def CW_CCW_obs(centerPos, targPos, trialPos, plot=False):
    """
    Return 0: if CW to get to target, return 1 if CCW to get to target; 
    Steps: 
        1. Center by the centerPos
        2. Rotate by targPos angle such that aligned with (0, 1) axis; 
        3. Compute integral unneath curve; 

    Args:
        centerPos (np.array): (x, y) of centerPos (or start target for Jeev)
        targPos (np.array): (x, y) of targetPos (or end target for Jeev)
        trialPos (np.array): (2 x T) of trial trajectory (position)
    
    Returns:
        integer: 0 for CW, 1 for CCW
    """

    targCentered = targPos - centerPos
    targCentered_norm = targCentered / np.linalg.norm(targCentered)

    assert(trialPos.shape[0] == len(centerPos) == 2)
    trialCentered = trialPos - centerPos[:, np.newaxis]

    ### Rotate Target / trial; 
    angle = np.arctan2(targCentered_norm[1], targCentered_norm[0])

    ### Rotate by negative of angle: 
    Rot = np.array([[np.cos(-1*angle), -np.sin(-1*angle)], [np.sin(-1*angle), np.cos(-1*angle)]])

    ### Apply this rotation to the trial: 
    trialCentRot = np.dot(Rot, trialCentered)

    ### Get step size; 
    dx = np.hstack(([0], np.diff(trialCentRot[0, :])))

    ### Get the height: 
    y = trialCentRot[1, :]

    assert(len(dx) == len(y))

    if np.dot(dx, y) > 0:
        rot = 'CW'
        return 0, np.dot(dx, y)

    elif np.dot(dx, y) < 0:
        rot = 'CCW'
        return 1, np.dot(dx, y)

    elif np.dot(dx, y) == 0:
        raise Exception('Perfect Straight Line?')

    if plot:
        plot_CW_CCW(trialPos, trialCentRot, rot)
            
def plot_CW_CCW(trialPos, trialCentRot, rot):
    f, ax = plt.subplots(ncols = 2)
    ax[0].plot(trialPos[0, :], trialPos[1, :])
    ax[1].plot(trialCentRot[0, :], trialCentRot[1, :])
    ax[1].set_title('Rot %s'%(rot))




