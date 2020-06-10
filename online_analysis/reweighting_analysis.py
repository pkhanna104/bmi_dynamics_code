import analysis_config
from online_analysis import generate_models, util_fcns

import matplotlib.pyplot as plt
import pickle 
import numpy as np

def PC_proj_plot(animal, day, model_set_number = 7, min_obs = 30, min_obs2 = 20, dat = None, skip_null = False):
    tsk_mark = dict()
    tsk_mark[0] = 'o'
    tsk_mark[1] = 's'

    ### Get the KG / KG_null projection matrices; 
    if animal == 'grom':
        _, KG_null, _ = generate_models.get_KG_decoder_grom(day)
    
    elif animal == 'jeev':
        _, KG_null, _ = generate_models.get_KG_decoder_jeev(day)
    
    if skip_null:
        print('Null is identity')
        KG_null = np.eye(KG_null.shape[0])

    ### Load maybe file 7 ### 
    if dat is None:
        dat = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen.pkl' %(model_set_number), 'rb'))
    
    ### Now for ang/mag combos
    neural_push = dat[day, 'np']
    spks = dat[day, 'spks']

    ### General dynamics 
    pred_spks = dat[day, 'hist_1pos_0psh_0spksm_1_spksp_0'][:, :, 2]
    targ = dat[day, 'trg']
    task = dat[day, 'task']
    bin_num = dat[day, 'bin_num']
    
    ### Now get discretized; 
    mag_boundaries = pickle.load(open(analysis_config.config['grom_pref'] + 'radial_boundaries_fit_based_on_perc_feb_2019.pkl'))
    command_bins = util_fcns.commands2bins([neural_push], mag_boundaries, animal, day, vel_ix = [0, 1], ndiv=8)[0]
    
    ### Get mean FR for each transition observed; 
    ### Enough observations to count as a commonly observed transition
    mFR_trans, trans_list = get_transition_mFR(command_bins, spks, bin_num, min_obs, past = False)

    ### For each mag / angle, get all instances; 
    for m in range(4):
        for a in range(8):
            
            ### Indices; 
            ix = np.nonzero(np.logical_and(command_bins[:, 0] == m, command_bins[:, 1] == a))[0]
            
            ### Enough total observations of this command; 
            if len(ix) > min_obs:
                
                ### Get neural actiivty; 
                spks_tmp = spks[ix, :]
                spks_null_tmp = np.dot(KG_null, spks_tmp.T).T
                
                ### Do PCA here; 
                trans_data, pc_model, ev = util_fcns.PCA(spks_null_tmp, 2, mean_subtract = False)
                #assert(np.sum(np.abs(pc_model['x_mn'])) == 0.)
                
                ### Ok, now for each task and condition, plot mean in PC space
                f, ax = plt.subplots()
                vaf = (ev[0]+ev[1]) / np.sum(ev)
                ax.set_title('Mag %d, Ang %d, VAF %.2f' %(m, a, vaf))
                ax.set_xlabel('PC 1 VAF %.2f' %(ev[0]/np.sum(ev)))
                ax.set_ylabel('PC 2 VAF %.2f' %(ev[1]/np.sum(ev)))
                
                for i_tsk in range(2):
                    for i_trg in range(10):
                        #print('task %d, targ %d' %(i_tsk, i_trg))

                        ### Get the condition ####
                        ix2 = np.nonzero(np.logical_and(targ[ix] == i_trg, task[ix] == i_tsk))[0]
                        
                        assert(np.all(targ[ix[ix2]] == i_trg))
                        assert(np.all(task[ix[ix2]] == i_tsk))
                        assert(np.all(command_bins[ix[ix2], 0] == m))
                        assert(np.all(command_bins[ix[ix2], 1] == a))

                        ### Enough "next step" observations 
                        if len(ix2) > min_obs2:

                            # ### if enough plot the mean; 
                            # tmp_mn = np.mean(trans_data[ix2, :], axis=0)
                            # ax.plot(tmp_mn[0], tmp_mn[1], marker = tsk_mark[i_tsk], 
                            #         color = analysis_config.pref_colors[i_trg])
                            
                            ### Ok, now want to plot mean estimated from transitions and distribution 
                            distribution = get_distribution(ix[ix2], command_bins, spks, pred_spks, bin_num, targ, task, trans_list[m, a], m, a)
                            
                            ### For each of the entries of "distribution", estimate the mean FR; 
                            true_tmp_spks = []
                            pred_tmp_spksA = []
                            reweighted_mn = []
                            reweighted_N = []

                            for i_k, key in enumerate(distribution.keys()): 
                                assert(tuple(key) in mFR_trans[m, a].keys())

                                ### Plot this guy ###
                                true_tmp_spks.append(distribution[key]['true'])
                                pred_tmp_spksA.append(distribution[key]['pred'])

                                ### Number of spikes ###
                                N = distribution[key]['true'].shape[0]
                                N1 = distribution[key]['pred'].shape[0]
                                assert(N == N1)

                                reweighted_N.append(N)

                                ### Mean of spikes ###
                                reweighted_mn.append(mFR_trans[m, a][tuple(key)])

                            #import pdb; pdb.set_trace()
                            ### Ok compute the mean FR from the true tmp spks;
                            if len(true_tmp_spks) > 0:
                                true_tmp_spks = np.vstack((true_tmp_spks))
                                pred_tmp_spksA = np.vstack((pred_tmp_spksA))

                                true_tmp_pcs = spk2pc(true_tmp_spks, KG_null, pc_model)
                                pred_tmp_pcsA = spk2pc(pred_tmp_spksA, KG_null, pc_model)

                                ### Get the true mean 
                                tmp_mn_sub = np.mean(true_tmp_pcs, axis=0)
                                ax.plot(tmp_mn_sub[0], tmp_mn_sub[1], marker = tsk_mark[i_tsk], 
                                        color = analysis_config.pref_colors[i_trg], alpha=1., markersize = 10)

                                tmp_pred_mn = np.mean(pred_tmp_pcsA, axis=0)
                                ax.plot(tmp_pred_mn[0], tmp_pred_mn[1], marker = tsk_mark[i_tsk],
                                    color = analysis_config.pref_colors[i_trg], alpha=1., mec='gray', mew=4, markersize = 12)

                                ### get estimated;
                                reweighted_N = np.hstack((reweighted_N))
                                reweighted_N = reweighted_N / float(np.sum(reweighted_N))
                                est_reweighted_N = []
                                
                                ### Make sure matching; 
                                assert(len(reweighted_N) == len(reweighted_mn))

                                for i_n, tmp_n in enumerate(reweighted_N):
                                    est_reweighted_N.append(float(tmp_n)*reweighted_mn[i_n])
                                est_reweighted_N = np.sum(np.vstack((est_reweighted_N)), axis=0)
                                est_reweighted_N = est_reweighted_N[np.newaxis, :] ### 1 x N 
                                est_rew_pc = np.squeeze(np.array(spk2pc(est_reweighted_N, KG_null, pc_model)))

                                ax.plot(est_rew_pc[0], est_rew_pc[1], marker = tsk_mark[i_tsk], 
                                        color = analysis_config.pref_colors[i_trg], alpha=.8, mec='k', mew=3, markersize = 12)

                            ### Ok how do we add A*x_t plot to this; take all the times where this command occurs; 
                            pred_spks_tmp = pred_spks[ix, :]
                

def get_distribution(ix_all, command_bins, spks, pred_spks, bin_num, targ, task, sub_trans_list, m, a):
    '''
    for each task / target, get a command bin with enough representation, 
        then get the breakdown using the trans_list
    '''

    distribution = dict()

    ### Next step ###
    ix_all_tp1 = ix_all + 1
    ix_all_tp1 = ix_all_tp1[ix_all_tp1 < len(task)]
    ix_all_tp1 = ix_all_tp1[bin_num[ix_all_tp1] > np.min(bin_num)]

    ### Keep this guy; 
    ### Go through the available transition list; 
    for i_trans, next in enumerate(sub_trans_list):

        ### Now see how many of these are present;  
        prez = np.nonzero(np.logical_and(command_bins[ix_all_tp1, 0] == next[0], command_bins[ix_all_tp1, 1] == next[1]))[0]
        ix_tp1_sub = ix_all_tp1[prez]

        if len(prez) > 0:
            ### present spikes; 
            distribution[next[0], next[1]] = dict(true=spks[ix_tp1_sub-1, :])
            distribution[next[0], next[1]]['pred'] = pred_spks[ix_tp1_sub-1, :]
            assert(np.all(command_bins[ix_tp1_sub-1, 0] == m))
            assert(np.all(command_bins[ix_tp1_sub-1, 1] == a))

    return distribution

def get_transition_mFR(command_bins, spks, bin_num, min_obs, past = False):
    ''' 
    for each command, get next command and mean associated iwth this 
    '''
    min_bin_num = np.min(bin_num)
    N = len(bin_num)

    mFR_trans = dict()
    trans_list = dict()

    for m in range(4):
        for a in range(8):

            ### Get the indices ###
            ix = np.nonzero(np.logical_and(command_bins[:, 0] == m, command_bins[:, 1] == a))[0]

            if len(ix) > min_obs:
                mFR_trans[m, a] = dict()
                trans_list[m, a] = []

                if past:
                    ix = ix[bin_num[ix] > min_bin_num]

                    ### Now get the next steps; 
                    ix_tp1 = ix - 1; 

                else:
                    ### Now get the next steps; 
                    ix_tp1 = ix + 1; 

                    ### Remove last entry if over the lenght of the set; 
                    ix_tp1 = ix_tp1[ix_tp1 < N]

                    ### remove anything that rolled over from the last trial 
                    ix_tp1 = ix_tp1[bin_num[ix_tp1] > min_bin_num]

                for mtp1 in range(4):
                    for atp1 in range(8):

                        #### Get the indices #####
                        ix2 = np.nonzero(np.logical_and(command_bins[ix_tp1, 0] == mtp1, command_bins[ix_tp1, 1] == atp1))[0]

                        if len(ix2) > min_obs:
                            mFR_trans[m, a][mtp1, atp1] = np.mean(spks[ix[ix2], :], axis=0)

                            ### Add to the transitions ####
                            trans_list[m, a].append([mtp1, atp1])

    return mFR_trans, trans_list

def spk2pc(spks_, kg_nul_, pc_model_):
    assert(spks_.shape[1] == kg_nul_.shape[0])
    spks_nul_ = np.dot(kg_nul_, spks_.T).T
    spks_z_ = spks_nul_ - pc_model_['x_mn'][np.newaxis, :]
    spks_pc_ = np.dot(pc_model_['proj_mat'].T, spks_z_.T).T
    assert(spks_pc_.shape[0] == spks_.shape[0])
    return spks_pc_