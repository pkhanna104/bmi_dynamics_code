import numpy as np
co = [4377]
obs = [4378, 4382]

def get_cursor_state_and_neural_push(co, obs, animal='grom', 
    binsize_ms=100., savename='for_v.pkl', pre_go=1.):

    from online_analysis import fit_LDS, co_obs_tuning_matrices
    import prelim_analysis as pa
    import pickle
    import tables

    data = {}
    for te_num in [co, obs]:
        
        ### Get cursor state and neural push
        co_obs_dict = pickle.load(open(co_obs_tuning_matrices.pref+'co_obs_file_dict.pkl'))

        ### Open task data file
        hdf = co_obs_dict[te_num, 'hdf']
        hdfix = hdf.rfind('/')
        hdf = tables.openFile(co_obs_tuning_matrices.pref+hdf[hdfix:])


        ### Open decoder file
        dec = co_obs_dict[te_num, 'dec']
        decix = dec.rfind('/')
        decoder = pickle.load(open(co_obs_tuning_matrices.pref+dec[decix:]))

        ### Get steady state kalman filter matrices
        F, KG = decoder.filt.get_sskf()

        # Get trials: 
        drives_neurons_ix0 = 3
        key = 'spike_counts'
    
        ### Get task indices for rewarded trials
        rew_ix = np.array([t[1] for it, t in enumerate(hdf.root.task_msgs[:]) if t[0]=='reward'])
        
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
