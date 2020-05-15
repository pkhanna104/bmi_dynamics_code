import numpy as np
import prelim_analysis as pa
import tables
import scipy
import datetime
from db import dbfunctions as dbfn
import pickle
import analysis_config

######### Methods for extracting trial metrics, Rew per min etc. used to analyze online results #########

storage_loc = analysis_config.config['grom_pref']

def targ_ix_to_3x3_subplot(ix):
    if ix == 0:
        return 2, 0
    elif ix == 1:
        return 2, 1
    elif ix == 2:
        return 2, 2
    elif ix == 3: 
        return 1, 2
    elif ix == 4:
        return 0, 2
    elif ix == 5:
        return 0, 1
    elif ix == 6:
        return 0, 0
    elif ix == 7: 
        return 1, 0

class Trial_Metrics(tables.IsDescription):
    trial_number = tables.IntCol() #
    target_index = tables.IntCol()#
    target_loc = tables.Float64Col(shape=(2,))#
    start_time = tables.Float64Col()#
    input_type = tables.StringCol(20)#
    task_entry = tables.IntCol() #
    obstacle_size = tables.IntCol()

    time2targ = tables.Float64Col() #
    path_length = tables.Float64Col() #
    path_error = tables.Float64Col() #
    avg_path_error = tables.Float64Col()
    avg_speed = tables.Float64Col() # 
    avg_dist2targ = tables.Float64Col()
    rew_per_min = tables.Float64Col() #

    timeout_time = tables.Float64Col()
    timeout_penalty_time = tables.Float64Col()

class Meta_Metrics(tables.IsDescription):
    targ_percent_success = tables.Float64Col(shape=(8,))
    all_percent_success = tables.Float64Col()
    all_perc_succ_lte10sec = tables.Float64Col(shape=(8,))
    task_entry = tables.IntCol()

class RPM(tables.IsDescription):
    trials_per_min_2min_chunks = tables.Float64Col()
    task_entry = tables.IntCol()

def process_targets(te_list, new_hdf_name, add_spk_cnts=False):
    '''
    Summary: Yields an array of trial information in 
        pytables with columns seen above in Trial_Metrics table class

    Input param: hdf: hdf for either fa_bmi or bmi_resetting
    Output param: trial dictionary
    '''
    h5file = tables.openFile(new_hdf_name, mode="w", title='FA Trial Analysis')
    trial_mets_table = h5file.createTable("/", 'trial_metrics', Trial_Metrics, "Trial Mets")
    meta_table = h5file.createTable("/", 'meta_metrics', Meta_Metrics, "Meta Mets")
    rpm_table = h5file.createTable("/", 'rpm', RPM, "RPM in 2 min chunks, aligned to next rew")

    extra = {}    

    row_cnt = -1

    for te in te_list:

        te_obj = dbfn.TaskEntry(te)
        task_entry = te
        
        try:
            hdf = te_obj.hdf
        except:
            hdf_fname = te_obj.hdf_filename.split('/')
            hdf = tables.openFile(storage_loc + hdf_fname[-1])

        #Extract trials ignoring assist: 
        rew_ix, rew_per_min = pa.get_trials_per_min(hdf, nmin=2, rew_per_min_cutoff=0, 
            ignore_assist=True, return_rpm=True)

        if add_spk_cnts:
            bin_spk, targ_i_all, targ_ix, trial_ix_all, reach_tm_all = pa.extract_trials_all(hdf, rew_ix, 
                neural_bins = 100, time_cutoff=60, drives_neurons_ix0=3, keep_trials_sep=True)

            t_loc = []
            t_ix = []
            tmp = 2
            for i in range(len(bin_spk)):
                t_loc.append(targ_i_all[tmp])
                t_ix.append(targ_ix[tmp])
                tmp += len(bin_spk[i])

            assert len(t_loc) == len(t_ix) == len(bin_spk)
            extra['bin_spk'] = bin_spk
            extra['targ_loc'] = t_loc
            extra['targ_ix'] = t_ix

            pickle.dump(extra, open(new_hdf_name[:-4]+'_'+str(te)+'_spike_stuff.pkl', 'wb'))

        #Go backwards 3 steps (rew --> targ trans --> hold --> target (onset))
        go_ix = np.array([hdf.root.task_msgs[it-3][1] for it, t in enumerate(hdf.root.task_msgs[:]) if 
            scipy.logical_and(t[0] == 'reward', t[1] in rew_ix)])

        assert len(rew_ix) == len(go_ix)#: raise Exception("Rew ix and Go ix are unequal")

        #Time to target is diff b/w target onset and reward time
        time2targs = (rew_ix - go_ix)/60.

        #Add buffer of 5 task steps (80 ms)
        target_locs = hdf.root.task[go_ix.astype(int) + 5]['target'][:, [0, 2]]

        try:
            obstacle_sz = hdf.root.task[go_ix.astype(int) + 5]['obstacle_size'][:, 0]
        except:
            obstacle_sz = np.zeros_like(go_ix) 

        #Targ ix -- should be 8
        targ_ixs = pa.get_target_ix(target_locs)
        assert len(np.unique(targ_ixs)) == 8#: raise Exception("Target Ix has more or less than 8 targets :/ ")

        #Input type: 
        try:
            input_types = hdf.root.task[go_ix.astype(int)+5]['fa_input']
        except:
            input_types = ['all']*len(go_ix)

        rew_time_2_trial = {}

        for ri, r_ix in enumerate(rew_ix):
            trl = trial_mets_table.row
            row_cnt += 1

            trl['trial_number'] = ri
            trl['task_entry'] = task_entry
            trl['rew_per_min'] = rew_per_min[ri]
            trl['target_index'] = targ_ixs[ri]
            trl['target_loc'] = target_locs[ri,:]
            trl['start_time'] = go_ix[ri]/60. # In seconds
            trl['input_type'] = input_types[ri]
            trl['time2targ'] = time2targs[ri]
            rew_time_2_trial[r_ix] = time2targs[ri]

            trl['obstacle_size'] = obstacle_sz[ri]


            path = hdf.root.task[int(go_ix[ri]):int(r_ix)]['cursor'][:, [0, 2]]
            path_length, path_error, avg_speed, avg_path_error, avg_dist2targ = path_metrics(path, target_locs[ri, :])

            trl['path_length'] = path_length
            trl['path_error'] = path_error
            trl['avg_path_error'] = avg_path_error
            trl['avg_speed'] = avg_speed*60
            trl['avg_dist2targ'] = avg_dist2targ

            trl['timeout_time'] = hdf.root.task.attrs.timeout_time
            trl['timeout_penalty_time'] = hdf.root.task.attrs.timeout_penalty_time

            trl.append()

        trial_mets_table.flush()

        #Rew Per Minute: 
        rpm_window = 2*60*60 # min --> sec --> hdf row iterations
        rpms = []
        ri = rew_ix[0]
        while ri + rpm_window <= rew_ix[-1]:
            # Add # of rewards: 
            rpms.append(len(np.nonzero(np.logical_and(rew_ix >= ri, rew_ix < ri + rpm_window))[0]))
            
            # Update head to ri + 2min
            ri = ri + rpm_window

            # Bump to next reward: 
            ix = np.nonzero(rew_ix >= ri)[0]
            
            # Set at next reward: 
            ri = rew_ix[ix[0]]

        rpm = rpm_table.row
        for i in range(len(rpms)):
            rpm['trials_per_min_2min_chunks'] = rpms[i]
            rpm['task_entry'] = task_entry
            rpm.append()
        rpm_table.flush()

        #Meta metrics for hdf: 
        block_len = hdf.root.task.shape[0]
        wind_len = 5*60*60
        wind_step = 2.5*60*60

        meta_ix = np.arange(0, block_len - wind_len, wind_step)
        trial_ix = np.array([i for i in hdf.root.task_msgs[:] if 
            i['msg'] in ['reward','timeout_penalty','hold_penalty','obstacle_penalty'] ], dtype=hdf.root.task_msgs.dtype)

        for m in meta_ix:
            meta_trl = meta_table.row
            meta_trl['task_entry'] = te
            end_meta_ix = m + wind_len

            trial_ix_time = trial_ix[:,]
            msg_ix = np.nonzero(np.logical_and(trial_ix['time']<=end_meta_ix, trial_ix['time']>m))[0]

            targ = hdf.root.task[trial_ix[msg_ix]['time'].astype(int)]['target'][:,[0,2]]
            targ_ix = pa.get_target_ix(targ)

            #Only mark as correct if it's the first correct trial
            targ_ix_ix = [0]
            tg_prev = targ_ix[0]
            for ii, tg_ix in enumerate(targ_ix[1:]):
                if tg_prev != tg_ix:
                    targ_ix_ix.append(ii+1)
                    tg_prev = tg_ix

            targ_ix_mod = targ_ix[targ_ix_ix]
            msg_ix_mod = msg_ix[targ_ix_ix]

            targ_percent_success = np.zeros((8, )) - 1
            all_perc_succ_lte10sec = np.zeros((8, )) - 1

            #time2targs for rew_ix:
            rew_ix_meta = np.array([i['time'] for i in trial_ix[msg_ix] if i['msg']=='reward'])

            for t in range(8):
                t_ix = np.nonzero(targ_ix_mod==t)[0]
                msgs = trial_ix[msg_ix_mod[t_ix]]

                if len(msgs) > 0:
                    targ_percent_success[t] = len(np.nonzero(msgs['msg']=='reward')[0]) / float(len(msgs))
                    msgs_copy = msgs.copy()
                    for mc in msgs_copy:
                        if mc['msg'] == 'reward':
                            r_tm = mc['time']
                            if r_tm in rew_time_2_trial.keys():
                                if rew_time_2_trial[r_tm] > 10.:
                                    msgs_copy['msg'] = 'not_reward'
                                    print 'not reward: ', m, t, mc
                            else:
                                print 'asissted: ', m, t, mc
                                msgs_copy['msg'] = 'not_rewarded'
                    all_perc_succ_lte10sec[t] = len(np.nonzero(msgs_copy['msg'] == 'reward')[0]) / float(len(msgs_copy))
                    if all_perc_succ_lte10sec[t] < 0:
                        print 'error: ', m, t, mc
                else: print 'no msgs for this epoch: ', m
            meta_trl['targ_percent_success'] = targ_percent_success
            meta_trl['all_perc_succ_lte10sec'] = all_perc_succ_lte10sec

            all_msg = trial_ix[msg_ix_mod]
            meta_trl['all_percent_success'] = len(np.nonzero(all_msg['msg']=='reward')[0]) / float(len(all_msg))
            meta_trl.append()

        meta_table.flush()

        #How many 5 min segments in 2.5 min steps


    h5file.close()
    

    return new_hdf_name

def path_metrics(single_path, target_loc):
    '''
    Summary: returns path length, path error, avg speed, 
        and hist speed +bins with 20 divisions
    
    Input param: 
        single_path: format: L x 2 of cursor
            over course of trial where L = number of task 
            loops between start and end of trial
        target_loc: np.array([x_targ, y_targ])

    Output param: dict with keys for path_length, 
        path_error, avg_speed
    '''

    # import matplotlib.pyplot as plt
    # f, ax = plt.subplots()
    # ax.plot(single_path[:, 0], single_path[:, 1])
    L = single_path.shape[0]

    ### PATH LENGTH ###

    #Take the diff of the path
    dpath = np.diff(single_path, axis=0)

    #Take distance between each point using x, y pos and hypotenus func
    hypot = np.hypot(dpath[:,0], dpath[:,1]) #This ends up being speed ()
    path_length = np.sum(hypot)

    # Also get the component in the target direction: 
    tg_norm = target_loc / float(np.sqrt(target_loc[0]**2 + target_loc[1]**2))

    efficiency = []
    for i in range(L-1):
        p = np.dot(dpath[i, :], tg_norm)
        efficiency.append(p)
        # ax.arrow(single_path[i, 0], single_path[i, 1], 3*p*tg_norm[0], 3*p*tg_norm[1])
    avg_dist2targ = np.mean(efficiency)

    ### PATH ERROR ###
    #The distance of each point from the straigth line from origin to target
    # Assume origin is (0, 0)

    #Calc each point's angular error from straight line: 
    #Make v_pt, v_targ unit vectors:
    hyp = np.sqrt(single_path[:,0]**2 + single_path[:,1]**2)
    v_pt = single_path / np.tile(hyp[:,np.newaxis], [1, 2])

    v_tg = np.tile(tg_norm[np.newaxis, :], [v_pt.shape[0], 1])

    # for i in range(L):
    #     ax.arrow(single_path[i, 0], single_path[i, 1], v_pt[i, 0], v_pt[i, 1])
    #     ax.arrow(single_path[i, 0], single_path[i, 1], v_tg[i, 0], v_tg[i, 1])

    #Dot product: 
    dt_prd = np.nansum(v_pt*v_tg, axis=1)
    angle = np.arccos(dt_prd)

    # Now for each point, let's take
    path_err = hyp*np.sin(angle)
    path_error = np.nansum(np.abs(path_err))
    avg_path_error = np.nanmean(np.abs(path_err))

    # pe = []
    # # plot dist to center: 
    # for i in range(L):
    #     # Get proj onto straight line: 
    #     proj = np.dot(single_path[i, :], tg_norm)

    #     ax.plot([single_path[i, 0], proj*tg_norm[0]],
    #         [single_path[i, 1], proj*tg_norm[1]], 'r-')
    #     pe_ = np.sqrt(np.sum(((tg_norm*proj) - single_path[i, :])**2))
    #     pe.append(pe_)

    # import pdb; pdb.set_trace()

    ### AVG SPEED ###
    #For path, calc speed of 
    avg_speed = np.mean(hypot)

    return path_length, path_error, avg_speed, avg_path_error, avg_dist2targ


