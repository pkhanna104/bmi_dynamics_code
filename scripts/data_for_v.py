import numpy as np
#from online_analysis import fit_LDS
import analysis_config
import prelim_analysis as pa
import pickle
import tables

co = [4377]
obs = [4378, 4382]

def get_cursor_state_and_neural_push(co=co, obs=obs, animal='grom', 
    binsize_ms=100., savename='for_v_v4.pkl', pre_go=1.):

    data = {}
    for i_tsk, te_nums in enumerate([co, obs]):
        for te_num in te_nums:

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

            # Get obstacle size
            if i_tsk == 1:

                ### Get trial start indices; 
                rew_ix = np.nonzero(hdf.root.task_msgs[:]['msg'] == 'reward')[0]
                assert(len(rew_ix) == len(bin_spk))

                ### Get table ix; 
                tbl_rew_ix = hdf.root.task_msgs[rew_ix]['time']; 

                ### get obstacle size / loc 
                data[te_num, 'obstacle_size'] = hdf.root.task[tbl_rew_ix]['obstacle_size']
                data[te_num, 'obstacle_location'] = hdf.root.task[tbl_rew_ix]['obstacle_location'][:, [0, 2]]

                # tg_sz = []
                # for i in range(len(rew_ix)):
                #     ix = np.nonzero(trial_ix_all == i)[0]
                #     tg_sz.append(targ_i_all[ix[0], :])


    # list of task entries 
    data['task_entries'] = [co, obs]

    # save data dictionary. 
    pickle.dump(data, open(savename, 'wb'))


####### PLOT #############

import matplotlib.pyplot as plt
import pickle, os

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

        ax.set_title('Task Entry Number:'+str(te_num))

def activity_plot_by_targ(data_name='for_v.pkl', unit_index=38.):

    cmap_list = ['maroon', 'orangered', 'darkgoldenrod', 'olivedrab',
    'teal', 'steelblue', 'midnightblue', 'darkmagenta']

    data = pickle.load(open(data_name))

    # choose the right subplot for target number: 
    # 3 | 2 | 1
    # 4 | _ | 0
    # 5 | 6 | 7

    axi_ref = dict()
    axi_ref[0] = [1, 2]
    axi_ref[1] = [0, 2]
    axi_ref[2] = [0, 1]
    axi_ref[3] = [0, 0]
    axi_ref[4] = [1, 0]
    axi_ref[5] = [2, 0]
    axi_ref[6] = [2, 1]
    axi_ref[7] = [2, 2]

    tasknms = ['CO', 'OBS']
    for it, te_num in enumerate(data['task_entries']):

        # Print decoder matrices (just to show you where they're stored):
        print 'A: ', data[te_num, 'decA']
        print ''; print '';
        print 'W: ', data[te_num, 'decW']
        print ''; print '';
        print 'C: ', data[te_num, 'decC']
        print ''; print '';
        print 'Q: ', data[te_num, 'decQ']
        print ''; print '';
        print 'Kalman Gain: ', data[te_num, 'dec_KG']
        print ''; print '';
        print 'F: ', data[te_num, 'decF']
        print ''; print '';

        # Now make a raster plot for one unit:
        f, ax = plt.subplots(nrows=3, ncols=3)

        # Targets x trial number for target x bins 
        data_tg = np.zeros((8, 8, 20))

        # Trial number for each target: 
        trl_cnt = np.zeros((8, ))

        for i_trial, trial in enumerate(data[te_num, 'binned_spk_cnts']):
            # "trial" is a bins x neurons matrix

            # Figure out which target this is
            #indices that correpond to this trial number: 
            trl = np.nonzero(data[te_num, 'trial_ix']==i_trial)[0]
            
            # First trial index
            trl_0 = trl[0]

            # target number of this index: 
            targ_num = int(data[te_num, 'target_index'][trl_0])
            trl_i = int(trl_cnt[targ_num])

            T = np.min([20, trial.shape[0]])
            try:
                data_tg[targ_num, trl_i, :T] = trial[:T, int(unit_index)]
            except:
                pass
            # increment index: 
            trl_cnt[targ_num] += 1

        for i_tg, tg in enumerate(range(8)):
            axi_r = axi_ref[tg]
            axi = ax[axi_r[0], axi_r[1]]
            axi.pcolormesh(data_tg[tg, :, :], vmin=0, vmax=10)
            axi.set_title('Targ #: '+str(tg))

        for i in range(3):
            ax[i, 0].set_ylabel('Trials')
            ax[2, i].set_xlabel('Time (100 ms Bins)')
        
        ax[0, 1].set_title('Task: '+tasknms[it])

        plt.tight_layout()


###### Export decoders ######
from online_analysis import util_fcns

def save_decoders(): 

    decoder_dict = {}
    dynamics_dict = {}

    for i_a, animal in enumerate(['grom', 'jeev']):

        model_fname = analysis_config.config[animal+'_pref']+'tuning_models_'+animal+'_model_set'+str(6)+'_.pkl'
        model_data = pickle.load(open(model_fname, 'rb'))

        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):

            ### get decoder: 
            if animal == 'grom':

                F, KG = util_fcns.get_grom_decoder(day_ix)

            elif animal == 'jeev':

                F = util_fcns.get_jeev_F(day_ix)

                ### Returns a 2 x Nneurons estimated Kalman Gain matrix 
                KG_sub = util_fcns.get_jeev_decoder(day_ix)

                ### Populate the 7 x Nneurons matrix
                ### Confirmed from previous analysis that neural push only 
                ### ever effects velocity (on current timestep), not position 
                KG = np.zeros((7, KG_sub.shape[1]))
                KG[[3, 5], :] = KG_sub.copy()

            #### Format of matrices is [Xpos,Zpos,Ypos, Xvel,Zvel,Yvel, offset]: 7 dim
            #### This is a 2D task, so Zpos/Zvel are irrelevant, so removing them:
            keep_ix = [0, 2, 3, 5, 6] 
            KG_xy = KG[keep_ix, :]
            F_xy = F[np.ix_(keep_ix, keep_ix)]

            ### Shoudl yield a 5 x Nneurons and 5x5 matrix: 
            assert(KG_xy.shape[0] == F_xy.shape[0] == F_xy.shape[1] == 5)

            decoder_dict[animal, day_ix, 'KG'] = KG_xy.copy()
            decoder_dict[animal, day_ix, 'F'] = F_xy.copy()

            ### Get dynamics 
            fold = 0
            mod = model_data[day_ix, 'hist_1pos_0psh_2spksm_1_spksp_0', fold, 0, 'model']
            dynamics_dict[animal, day_ix, 'A'] =  mod.coef_.copy()
            dynamics_dict[animal, day_ix, 'b'] =  mod.intercept_.copy()

    pickle.dump(decoder_dict, open(os.path.join(analysis_config.config['BMI_DYN'], 'scripts', 'decoders_for_v.pkl'), 'wb'))
    pickle.dump(dynamics_dict, open(os.path.join(analysis_config.config['BMI_DYN'], 'scripts', 'dynamics_for_v.pkl'), 'wb'))

# def print_EVs():
#     dat = pickle.load(open(os.path.join(analysis_config.config['BMI_DYN'], 'scripts', 'dynamics_for_v.pkl'), 'rb'))
#     for day in range(9):
#         A = dat['grom', day, 'A']
#         AI = A - np.eye(A.shape[0])

#         ev, evect = np.linalg.eig(AI)

#         ix_sort = np.argsort(ev)
#         ev = ev[ix_sort[::-1]]
#         evect = evect[ix_sort[::-1]]

#         time_decay = -1/np.real(ev)*.1

#         f, ax = plt.subplots()
#         ax.plot(time_decay, 'k.')
#         ax.set_title(day)
#         ax.set_ylim([0., .5])

#         KG = util_fcns.get_decoder('grom', day)
#         s, v, d = np.linalg.svd(KG)
#         print('Day %d' %(day))
#         for i in range(8):
#             print('ev %d: %.2f, %.2f'% (i, np.dot(np.real(evect[:, i]), d[0, :]),
#                 np.dot(np.real(evect[:, i]), d[0, :])))
        


# tes = [4377, 4395, 4411, 4499, 4510, 4523, 4536, 4553, 4558]
# for te in tes: 
#     TE = dbfn.TaskEntry(te)
#     clda_seed = TE.decoder_record.entry_id
#     TEclda = dbfn.TaskEntry(clda_seed)
#     print(TEclda)


# f, ax = plt.subplots(figsize=(3, 3))

# for i_ in range(9):
#     print('session %d' %i_) 
#     F, KG, decoder = util_fcns.get_grom_decoder(i_)
#     #print(decoder.filt.A)
#     #print(decoder.filt.W)
#     #print(np.round(F[np.ix_([0, 2, 3, 5, 6], [0, 2, 3, 5, 6])], 2))
#     #print(np.squeeze(np.round(decoder.filt.C[0:5, :], 2)))
#     #print(np.trace(decoder.filt.Q)/float(KG.shape[1]))
#     #Fsub = np.dot(KG[[3, 5], :], decoder.filt.C)
#     #print(np.sum(decoder.filt.Q)/(float(KG.shape[1])**2))
#     #print(Fsub[:, [3, 5, 6]])
#     #print(np.linalg.norm(decoder.filt.C[:, [3, 5]], axis=1))
#     #print(np.mean(np.linalg.norm(decoder.filt.C[:, [3, 5]], axis=1)))
#     if i_ in [0, 1, 2, 8]:
#         mew = 1.
#         mec = 'r'
#     else:
#         mew = 0.
#         mec = 'r'

#     ax.plot(F[3, 3], np.mean(np.linalg.norm(decoder.filt.C[:, 3])), 'k.', mew=mew, mec=mec)
#     ax.plot(F[5, 5], np.mean(np.linalg.norm(decoder.filt.C[:, 5])), 'b.', mew=mew, mec=mec)

# ax.set_xlabel('Cursor dyn. value (F[3,3] and F[5,5])')
# ax.set_ylabel('np.linalg.norm(C[3, :] or C[5, :])')
# plt.title('Blue=yvel, Black=xvel, Red Edge=CO-trained')

