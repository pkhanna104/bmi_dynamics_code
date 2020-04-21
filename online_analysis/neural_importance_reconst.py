### Method for deciding which neurons are needed for control ###
### If remove unimporant neurons how well can you reconstruct the 
### Vx / Vy trajectory? ###
### Assess if that's true -- use R2? Implies there may need to be a 
### linear transform to get the full extent 
import pickle
import co_obs_tuning_matrices
import numpy as np
import fit_LDS
import prelim_analysis as pa


binsize_ms = 100.
import_dict = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/co_obs_neuron_importance_dict.pkl'))
datafiles = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/co_obs_file_dict.pkl'))

important_neuron_ix = dict()

for day in range(9):

    # Get bin spikes for this task entry:
    bin_spk_all = []
    cursor_KG_all = []
    
    for task in range(2):
        te_nums = co_obs_tuning_matrices.input_type[day][task]
        important_neurons = np.mean(np.vstack(([import_dict[te] for te in te_nums])), axis=0)
        order_to_drop = np.argsort(important_neurons)
        N_neur = len(order_to_drop)

        for te_num in te_nums:
            bin_spk_nonz, targ_ix, trial_ix_all, KG, exclude_ix = fit_LDS.pull_data(te_num, animal,
                pre_go=1, binsize_ms=binsize_ms, keep_units='all')

            # Let's get the actual 
            hdf = datafiles[te_num, 'hdf']
            name = hdf.split('/')[-1]
            hdf = tables.openFile('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/'+name)

            # decoder: 
            dec = datafiles[te_num, 'dec']
            dec = dec.split('/')[-1]
            decoder = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/'+dec))

            rew_ix = np.array([t[1] for it, t in enumerate(hdf.root.task_msgs[:]) if t[0]=='reward'])

            _, _, _, _, cursor_KG = pa.extract_trials_all(hdf, rew_ix, neural_bins = binsize_ms,
                drives_neurons_ix0=3, hdf_key='spike_counts', keep_trials_sep=True,
                reach_tm_is_hdf_cursor_pos=False, reach_tm_is_hdf_cursor_state=False, 
                reach_tm_is_kg_vel=True, include_pre_go= 1, **dict(kalman_gain=KG))

            bin_spk_all.extend([bs for ib, bs in enumerate(bin_spk_nonz) if ib not in exclude_ix])
            cursor_KG_all.extend([kg for ik, kg in enumerate(cursor_KG) if ik not in exclude_ix])

    # Get accuracy of 
    bin_spk_all = np.vstack((bin_spk_all))
    cursor_KG_all = np.vstack((cursor_KG_all))[:, [3, 5]].reshape(-1)

    R2 = 1.
    inc = 0

    while R2 > 0.9: 
        ix_to_zero = order_to_drop[:inc]
        bin_spk_all[:, ix_to_zero] = 0
        est_kg = np.array(bin_spk_all*np.mat(KG.T))

        _, _, rv, _, _, = scipy.stats.linregress(est_kg[:, [3, 5]].reshape(-1),
            cursor_KG_all)
        R2 = rv**2
        inc += 1

    ix_to_zero = order_to_drop[:inc-1]
    ix_to_important = [i for i in range(bin_spk_all.shape[1]) if i not in ix_to_zero]
    important_neuron_ix[day, 'ix'] = np.sort(ix_to_important)
    important_neuron_ix[day, 'units'] = decoder.units[important_neuron_ix[day, 'ix'], :]
pickle.dump(important_neuron_ix, open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/import_neur_ix.pkl', 'wb'))



