import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import file_key as fk
import pickle
import os

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

    times = zip(strobed[go_ix, 0], strobed[rew_ix, 0])

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
    
    ixs_og = zip(strobed[go_ix, 0], strobed[rew_ix, 0])
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
    try:
        decoder_all, decoder_all_ub = _bin_neural_push(ixs, filename, binsize, start_index_overall, pre_go)
        unbinned['neural_push'] = decoder_all_ub
    except:
        decoder_all = 0
        unbinned['neural_push'] = 0
    unbinned['ixs'] = ixs
    unbinned['start_index_overall'] = start_index_overall
    
    if include_pos:
        cursor_kin = dat['cursor_kin']
        bin_ck, ck = _bin_cursor_kin(ixs, cursor_kin, binsize, pre_go)
        print len(bin_ck), bin_ck[0].shape
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
    n_per_bin = 20
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
        neuralpush = pickle.load(open(neuralpush_fn))
    except:
        neuralpush = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/jeev_neural_push_apr2017.pkl'))
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

def plot_jeev_trials(task = 'obs'):
    input_type = fk.task_filelist
    
    if task == 'co':
        te_num = input_type[0][0][0]
    elif task == 'obs':
        te_num = input_type[0][1][0]
    
    bin_spk, targ_i_all, targ_ix, trial_ix_all, decoder_all, cursor_state, unbinned, exclude = get_jeev_trials_from_task_data(te_num,
        include_pos=True, binsize=.1)

    colors = ['maroon', 'orangered', 'goldenrod','olivedrab','teal', 'steelblue', 'midnightblue', 'darkmagenta', 'k', 'brown']

    f, ax = plt.subplots()

    targs = np.unique(targ_ix)
    targs = targs[targs >= 0]

    for i in targs:

        ### Get trials with the right target number: 
        ix = np.nonzero(targ_ix == i)[0]

        ### Now figure out which trial: 
        trl_ix = np.unique(trial_ix_all[ix])

        for trl in trl_ix:
            ax.plot(cursor_state[trl][:, 0], cursor_state[trl][:, 2], '-', color = colors[i], linewidth=1.0)

        ### Plot the target: 
        if task == 'co':
            ax.plot(np.array([.04, targ_i_all[ix[0], 0]])*20, np.array([.14, targ_i_all[ix[0], 1]])*20, 'k-.')
        else:
            tg = targ_i_all[ix[0], :]

            ### Subtract obstacle target center; 
            tg = tg - np.array([0., 2.5])

            ### Convert cm --> m 
            tg = tg / 100.

            ### Add back the other center; 
            tg = tg + np.array([.04, .14])

            ### Plot the whole thing
            ax.plot(np.array([.04, tg[0]])*20, np.array([.14,tg[1]])*20, 'k-.')




