import numpy as np
co = [4377]
obs = [4378, 4382]

def get_cursor_state_and_neural_push(co=[4377], obs=[4378, 4382], animal='grom', 
    binsize_ms=100., savename='for_v_v5.pkl', pre_go=0.):

    import analysis_config
    import prelim_analysis as pa
    import pickle
    import tables

    ### Check with old data #####
    dat_check = pickle.load(open('for_v_v4.pkl', 'rb'))

    data = {}
    for te_num in np.hstack((co, obs)):
        
        ### Get cursor state and neural push
        co_obs_dict = pickle.load(open(analysis_config.config['grom_pref']+'co_obs_file_dict.pkl'))

        ### Open task data file
        hdf = co_obs_dict[te_num, 'hdf']
        hdfix = hdf.rfind('/')
        hdf = tables.openFile(analysis_config.config['grom_pref']+hdf[hdfix:])


        ### Open decoder file
        dec = co_obs_dict[te_num, 'dec']
        decix = dec.rfind('/')
        decoder = pickle.load(open(analysis_config.config['grom_pref']+dec[decix:]))

        ### Get steady state kalman filter matrices
        F, KG = decoder.filt.get_sskf()

        # Get trials: 
        drives_neurons_ix0 = 3
        key = 'spike_counts'
    
        ### Get task indices for rewarded trials
        rew_ix = np.array([t[1] for it, t in enumerate(hdf.root.task_msgs[:]) if t[0]=='reward'])
        
        ## Extract all trial data
        bin_spk_16_67ms, _, _, _, _ = pa.extract_trials_all(hdf, rew_ix, neural_bins = 1000/60.,
            drives_neurons_ix0=drives_neurons_ix0, hdf_key=key, keep_trials_sep=True,
            reach_tm_is_hdf_cursor_pos=False, reach_tm_is_kg_vel=True, include_pre_go= pre_go, **dict(kalman_gain=KG))

        ## Extract all trial data
        bin_spk, targ_i_all, targ_ix, trial_ix_all, decoder_all = pa.extract_trials_all(hdf, rew_ix, neural_bins = binsize_ms,
            drives_neurons_ix0=drives_neurons_ix0, hdf_key=key, keep_trials_sep=True,
            reach_tm_is_hdf_cursor_pos=False, reach_tm_is_kg_vel=True, include_pre_go= pre_go, **dict(kalman_gain=KG))

        ### Extract only cursor pos
        _, _, _, _, cursor_pos = pa.extract_trials_all(hdf, rew_ix, neural_bins = binsize_ms,
            drives_neurons_ix0=drives_neurons_ix0, hdf_key=key, keep_trials_sep=True,
            reach_tm_is_hdf_cursor_pos=True, reach_tm_is_kg_vel=False, include_pre_go= pre_go, **dict(kalman_gain=KG))

        ### Extract full cursor state (pos, vel)
        _, _, _, _, cursor_state = pa.extract_trials_all(hdf, rew_ix, neural_bins = binsize_ms,
            drives_neurons_ix0=drives_neurons_ix0, hdf_key=key, keep_trials_sep=True,
            reach_tm_is_hdf_cursor_pos=False, reach_tm_is_hdf_cursor_state=True, 
            reach_tm_is_kg_vel=False, include_pre_go= pre_go, **dict(kalman_gain=KG))

        # Decoder matrices: 
        data[te_num, 'decA'] = decoder.filt.A
        data[te_num, 'decW'] = decoder.filt.W
        data[te_num, 'decC'] = decoder.filt.C
        data[te_num, 'decQ'] = decoder.filt.Q
        data[te_num, 'dec_KG'] = KG
        data[te_num, 'decF'] = F
        
        # Binned spike counts:    
        data[te_num, 'binned_spk_cnts'] = bin_spk
        data[te_num, 'binned_spk_cnts_60Hz'] = bin_spk_16_67ms

        # Pos x, pos y, vel x, vel y -- each list entry is a trial
        data[te_num, 'neural_push'] = [d[:, [0, 2, 3, 5]] for d in decoder_all]

        # Pos x, pos y -- each list entry is a trial
        data[te_num, 'cursor_pos'] = cursor_pos

        # Pos x, pos y, vel x, vel y (state space) -- pos x, pos y will match cursor pos above
        # each list entry is a trial
        data[te_num, 'cursor_state'] = [c[:, :, 0] for c in cursor_state]

        # target number for time bin. Length of targ_ix is length of array when all 
        # trials are concatenated together (e.g. if only 2 trials with trl 1 --> 10 time bins, 
        # trl2 --> 41 time bins, len(targ_ix) will be 51 bins
        data[te_num, 'target_index'] = targ_ix

        # Target position. Same format as target_index
        data[te_num, 'target_pos'] = targ_i_all

        # Trial number of each time bin, same format as target_index. 
        data[te_num, 'trial_ix'] = trial_ix_all

        if te_num in obs:

            ### Add obstacle size and location 
            data[te_num, 'obstacle_size']  = hdf.root.task[rew_ix]['obstacle_size']
            data[te_num, 'obstacle_location'] = hdf.root.task[rew_ix]['obstacle_location'][:, [0, 2]]

            #### Will be diff of fisrt trial, make srue end entries are good ####
            assert(np.allclose(dat_check[te_num, 'obstacle_size'][-90:], data[te_num, 'obstacle_size'][-90:]))
            assert(np.allclose(dat_check[te_num, 'obstacle_location'][-90:], data[te_num, 'obstacle_location'][-90:]))
    
    # list of task entries 
    data['task_entries'] = [co, obs]

    # save data dictionary. 
    pickle.dump(data, open(savename, 'wb'))


####### PLOT #############

import matplotlib.pyplot as plt
import pickle

def open_data_for_v(data_name='for_v.pkl'):
    cmap_list = ['maroon', 'orangered', 'darkgoldenrod', 'olivedrab',
    'teal', 'steelblue', 'midnightblue', 'darkmagenta']

    data = pickle.load(open(data_name))

    # For each task entry, plot cursor pos:
    for te_num in data['task_entries']:

        f, ax = plt.subplots()

        for i_trial, trial in enumerate(data[te_num, 'cursor_pos']):
            
            # color it based on target number: 

            #indices that correpond to this trial number: 
            trl = np.nonzero(data[te_num, 'trial_ix']==i_trial)[0]

            # First trial index
            trl_0 = trl[0]

            # target number of this index: 
            targ_num = int(data[te_num, 'target_index'][trl_0])

            ax.plot(trial[:, 0], trial[:, 1], '.-', color=cmap_list[targ_num])
            # ax.axis('equal')

        ax.set_title('Task Entry Number:'+str(te_num))
        ax.axis('equal')
