'''

Methods for fitting LDS / FA / tuning / LPF models on the same exact data, 
    and getting R2, Beh Read-out, etc. 

For LDS figures in paper

Data is segmented as follows: 
    - 80 percent used for training (from entire task)
    - 20 percent used to test remaineder of task_filelist

Ignore units that are zero w/in 80 percent sections
Ignore trials that don't have 1 second of history (for LDS init state)

'''

import numpy as np
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
import matplotlib.pyplot as plt

from online_analysis import co_obs_tuning_matrices, fit_LDS, subspace_overlap, plot_factor_tuning_curves, behav_neural_PSTH
from online_analysis import test_fa_ov_vs_u_cov_ov as tfo
from resim_ppf import file_key as fk
from resim_ppf import ppf_pa
import scipy, scipy.signal, scipy.stats
import prelim_analysis as pa
import numpy as np
from pybasicbayes.util.text import progprint_xrange
import math
import pickle
import sklearn.decomposition as skdecomp
from fa_analysis import fcns
import seaborn
import os, fcns
import tables

import statsmodels.api as sm
import pandas
mag_boundaries = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/radial_boundaries_fit_based_on_perc_feb_2019.pkl'))

data_agg = dict(grom='/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/grom2019_data_seg_9_6_19.pkl',
                jeev='/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/jeev2019_data_seg_9_6_19.pkl')
nfolds = 5
n_dim_latent = 15
pre_go_bins = 10

### Main function ###
def main(animal, models = ['LDS', 'FA', 'max_tuning']):#, 'LPF150', 'LPF300', 'LPF450', 'LPF600']):

    '''
    Things to compute: 
        For [all neurons, imporant neurons]:
            a. R2 of all models 
            b. R2 of neural push, all models 
            c. LDS -- R2-smooth, R2-filt, R2-pred for neural
            d. Dimensionality (main and absolute of FA / LDS)
    '''
    
    ### load data structs ###
    data_master = pickle.load(open(data_agg[animal]))

    if animal == 'grom':
        input_type = co_obs_tuning_matrices.input_type
        model_order = co_obs_tuning_matrices.ordered_input_type
        fname_pref = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/test_'
    
    elif animal == 'jeev':
        input_type = fk.task_filelist
        model_order = fk.ordered_task_filelist
        fname_pref = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/test_'
    
    ### run models ###
    for i_d in range(len(input_type)):
        big_data = {}
        data_LDS = {}
        data_FA = {}
        data_tuning = {}
        data_LPF = {}   

        ### runs models for important neurons only ###
        for i_o, only_important in enumerate([False, True]):

            ### Which neurons are we removing / keeeping? 
            if only_important: 

                ### These are zero points ###
                rm = data_master['dat'][i_d, 'rm_units_imp']
                N_units = data_master['dat'][i_d, 0, 'bs_imp'][0].shape[1]
                keep_units = np.array([i for i in range(N_units) if i not in rm])
                day_mu = data_master['dat'][i_d, 'mfr_imp'][keep_units]
                day_st = data_master['dat'][i_d, 'std_imp'][keep_units]
                bs_key = 'bs_imp'

            else:
                rm = data_master['dat'][i_d, 'rm_units']
                if len(rm) > 0:
                    print('----------')
                    print('----------')
                    print('Removing some neurons -- need to deal with KG issues -- making rm = [] for now')
                    print('----------')
                    print('----------')
                #rm = []
                N_units = data_master['dat'][i_d, 0, 'bs'][0].shape[1]
                keep_units = np.array([i for i in range(N_units) if i not in rm])
                day_mu = data_master['dat'][i_d, 'mfr'][keep_units]
                day_st = data_master['dat'][i_d, 'std'][keep_units]
                bs_key = 'bs'

            ### For each xval: 
            for i_nf in range(nfolds):

                ### Task number 2 is a combination of both tasks for training and testing -- focus on this only for now:
                for i_t in range(3): 

                    if i_t in [0, 1]:
                        ### For task i_t:    
                        data_train_ix = data_master['xval'][i_d, i_t, i_nf, 'train']
                        data_test_ix = data_master['xval'][i_d, i_t, i_nf, 'test']

                        # Gather the data; 
                        bs_train = [data_master['dat'][i_d, i_t, bs_key][i][:, keep_units] for i in data_train_ix]
                        bs_test =  [data_master['dat'][i_d, i_t, bs_key][i][:, keep_units] for i in data_test_ix]
                        
                        # Get the neural push (will have 10 less bins)
                        np_test =  [data_master['dat'][i_d, i_t, 'np'][i][pre_go_bins:, :] for i in data_test_ix]

                        # Index to test
                        test_ix = np.zeros((len(bs_test))) + i_t

                        ### INCLDUE OFF TASK ###
                        non_i_t = np.mod(i_t+1, 2)

                        ### Add non-task data -- not just test/train. 
                        bs_test_off_tsk = [data_master['dat'][i_d, non_i_t, bs_key][i][:, keep_units] for i in range(len(data_master['dat'][i_d, non_i_t, bs_key]))]
                        np_test_off_tsk = [data_master['dat'][i_d, non_i_t, 'np'][i][pre_go_bins:, :] for i in range(len(data_master['dat'][i_d, non_i_t, 'np']))]

                        test_ix_nonit = np.zeros((len(bs_test_off_tsk))) + non_i_t

                        ### Combine testing: 
                        bs_testing_all = bs_test + bs_test_off_tsk
                        np_testing_all = np_test + np_test_off_tsk

                        ### Task index for testing
                        task_ix_trials = np.hstack((test_ix, test_ix_nonit))
                        
                        ### Task index for individual bins: 
                        pt_ix_trials = np.hstack(([task_ix_trials[i]]*bs_testing_all[i][pre_go_bins:, :].shape[0] for i in range(len(bs_testing_all))))
                        
                        ### Get velocity from cursor for  testing and training 
                        data_vel_train = [data_master['dat'][i_d, i_t, 'vl'][i] for i in data_train_ix]
                        data_vel_test =  [data_master['dat'][i_d, i_t, 'vl'][i] for i in data_test_ix] + [data_master['dat'][i_d, non_i_t, 'vl'][i] for i in range(len(data_master['dat'][i_d, non_i_t, 'vl']))]

                    ### If want to train using both tasks for comparison to other models: 
                    elif i_t in [2]:
                        bs_train = []; 
                        bs_testing_all = []; 

                        task_ix_trials = []; 
                        pt_ix_trials = []; 

                        data_vel_train = []; 
                        data_vel_test = []; 

                        np_testing_all = [] 

                        non_i_t = 2; 

                        # Cycle through both tasks to use
                        for i_t2 in range(2):

                            ### For task i_t:    
                            data_train_ix = data_master['xval'][i_d, i_t2, i_nf, 'train']
                            data_test_ix =  data_master['xval'][i_d, i_t2, i_nf, 'test']

                            # Aggregate the data: 
                            for i in data_train_ix:
                                bs_train.append(data_master['dat'][i_d, i_t2, bs_key][i][:, keep_units])

                            for i in data_test_ix:
                                bs_testing_all.append(data_master['dat'][i_d, i_t2, bs_key][i][:, keep_units])
                                np_testing_all.append(data_master['dat'][i_d, i_t2, 'np'][i][pre_go_bins:, :])


                            test_ix = np.zeros((len(data_test_ix))) + i_t2
                            task_ix_trials.append(test_ix)

                            data_vel_train.append(data_master['dat'][i_d, i_t2, 'vl'][i] for i in data_train_ix)
                            data_vel_test.append( data_master['dat'][i_d, i_t2, 'vl'][i] for i in data_test_ix)
                            
                        tmp = []
                        task_ix_trials = np.hstack((task_ix_trials))
                        task_ix_trials[:] = 2. # set these equal to zero else wont work later

                        for i, (bs, ix) in enumerate(zip(bs_testing_all, np.hstack(( task_ix_trials )))):
                            tmp.append([ix]*bs[pre_go_bins:, :].shape[0])
                        pt_ix_trials = np.hstack((tmp))

                    ### STart model fitting ####
                    for i_m, mod in enumerate(models):
                        print 'starting model %s, day %d, only_important %s, task index %d' %(mod, i_d, str(only_important), i_t)

                        if mod == 'LDS':

                            ### Return metrics on testing data only ###
                            n_dim_latent_i = np.min([n_dim_latent, bs_train[1].shape[1] ])

                            # ### must mean subtract
                            # dat = bs_train + bs_testing_all; 
                            #bs_train_z = [bs - day_mu[np.newaxis, :] for bs in bs_train]

                            R2_smooth, R2_pred, LL, dynamics_norm, innov_norm, model, pred_data, smooth_data, filt_state, pred_state, filt_sigma, kg, R2_filt, smooth_state = fit_LDS.fit_LDS(
                                bs_train, bs_testing_all, 
                                #bs_train_z, bs_train_z,
                                n_dim_latent_i, return_model=True, seed_w_FA=True, nEMiters=30, mfr = day_mu, stdfr = day_st,
                                **dict(seed_pred_x0_with_smoothed_x0= True, get_ratio=True, 
                                    pre_go_bins=pre_go_bins, task_ix_trials=task_ix_trials))

                            import pdb; pdb.set_trace()

                            ### R2s return tasks as [0, 1] no matter what the order, CO / OBS; 
                            ### So use i_t (task 1) and non_i_t (task 2) to get the right things in the right places. 
                            ### So for i_t == 2: 

                            if i_t in [0, 1]:
                                ### Non-normalized and normalized R2 (units with global mean and std subtracted in real and predicted prior to R2)
                                data_LDS[i_nf, i_t, i_o, 'R2_pred_tsk'] = [R2_pred[i_t*2], R2_pred[i_t*2 + 1]]
                                data_LDS[i_nf, i_t, i_o, 'R2_pred_nontsk'] = [R2_pred[non_i_t*2], R2_pred[non_i_t*2 + 1]]

                                data_LDS[i_nf, i_t, i_o, 'R2_filt_tsk'] = [R2_filt[i_t*2], R2_filt[i_t*2 + 1]]
                                data_LDS[i_nf, i_t, i_o, 'R2_filt_nontsk'] = [R2_filt[non_i_t*2], R2_filt[non_i_t*2 + 1]] 
                                
                                data_LDS[i_nf, i_t, i_o, 'R2_smooth_tsk'] = [R2_smooth[i_t*2], R2_smooth[i_t*2 + 1]]
                                data_LDS[i_nf, i_t, i_o, 'R2_smooth_nontsk'] = [R2_smooth[non_i_t*2], R2_smooth[non_i_t*2 + 1]]

                            elif i_t in [2]:
                                ### Non-normalized and normalized R2 (units with global mean and std subtracted in real and predicted prior to R2)
                                data_LDS[i_nf, i_t, i_o, 'R2_pred_test'] = [R2_pred[0], R2_pred[1]]
                                data_LDS[i_nf, i_t, i_o, 'R2_filt_test'] = [R2_filt[0], R2_filt[1]]
                                data_LDS[i_nf, i_t, i_o, 'R2_smooth_test'] = [R2_smooth[0], R2_smooth[1]]

                            data_LDS[i_nf, i_t, i_o, 'LDS_model_LL'] = model.log_likelihood()
                            data_LDS[i_nf, i_t, i_o, 'LDS_model_A'] = model.A
                            data_LDS[i_nf, i_t, i_o, 'LDS_model_C'] = model.C
                            data_LDS[i_nf, i_t, i_o, 'LDS_model_W'] = model.sigma_states
                            data_LDS[i_nf, i_t, i_o, 'LDS_model_Q'] = model.sigma_obs

                            ### For smoothed LDS: 
                            data_LDS[i_nf, i_t, i_o, 'LDS_main_shared'] = []
                        
                            for iii in range(3):
                                ixx = np.nonzero(task_ix_trials==iii)[0]

                                if iii == 2:
                                    ixx = np.nonzero(task_ix_trials == 2)[0]

                                if len(ixx) > 0:
                                    cov_x0 = []
                                    for nt in ixx:
                                        model.add_data(bs_testing_all[nt])
                                        g = model.states_list.pop()
                                        tmp = g.smooth()
                                        cov_x0.append(g.smoothed_mus)
                                    x0 = np.vstack(( cov_x0 ))               
                                    C_fit = np.mat(model.C)
                                    Z_cov_0 = C_fit*np.cov(x0.T)*C_fit.T

                                    ### This is for smoothed LDS
                                    data_LDS[i_nf, i_t, i_o, 'LDS_main_shared'].append(fit_LDS.get_main_shared_dim(Z_cov_0))

                            ############# Neural push predictions ############
                            if i_o == 0: 
                                ### Can only do this if we have all neurons: 
                                # Compare neural_pred * KG, neural_smooth * KG to neural_push
                                ### Same decoder regardless: 
                                kg_decoder = np.mat(data_master['kg'][i_d, 0, input_type[i_d][0][0]])[:, keep_units]
                                
                                filt_data = [np.dot(model.C, filt_state[i].T).T for i in range(len(filt_state))]

                                ### All of these guys are in trials: T x nneurons
                                pred_data0 = [pred_data[i].T for i in range(len(pred_data))]
                                lds_d = dict(smooth = smooth_data, filt = filt_data, pred = pred_data0)

                                for key in ['smooth', 'filt', 'pred']:
                                    key_data = lds_d[key]

                                    kg_from_lds = np.vstack(( [np.dot(kg_decoder, key_data[i].T).T for i in range(len(key_data))]))
                                    kg_real = np.vstack((np_testing_all))

                                    if i_t == 2:
                                        ix0 = np.nonzero(pt_ix_trials == 2)[0]
                                        R2_np0 = fit_LDS.get_R2(kg_real[np.ix_(ix0, [3, 5])], kg_from_lds[np.ix_(ix0, [3, 5])])
                                        data_LDS[i_nf, i_t, i_o, 'R2_'+key+'_vx_vy_test'] = R2_np0; 

                                    else:
                                        ix_tsk = np.nonzero(pt_ix_trials == i_t)[0]
                                        ix_nontsk = np.nonzero(pt_ix_trials == non_i_t)[0]

                                        R2_np0 = fit_LDS.get_R2(kg_real[np.ix_(ix_tsk, [3, 5])], kg_from_lds[np.ix_(ix_tsk, [3, 5])])
                                        data_LDS[i_nf, i_t, i_o, 'R2_'+key+'_vx_vy_tsk'] = R2_np0; 

                                        R2_np1 = fit_LDS.get_R2(kg_real[np.ix_(ix_nontsk, [3, 5])], kg_from_lds[np.ix_(ix_nontsk, [3, 5])])
                                        data_LDS[i_nf, i_t, i_o, 'R2_'+key+'_vx_vy_nottsk'] = R2_np1; 

                        elif mod == 'FA':

                            #Find optimal number of factors: 
                            bs_train_stack = np.vstack(([bs_train[x][pre_go_bins:, :] for x in range(len(bs_train))]))
                            bs_test_stack = np.vstack(([bs_testing_all[x][pre_go_bins:, :] for x in range(len(bs_testing_all))]))
                            
                            # zscore_X = bs_train_stack - day_mu[np.newaxis, :]
                            # zscore_X_test = bs_test_stack - day_mu[np.newaxis, :]
                            
                            #LL, psv = pa.find_k_FA(zscore_X, iters=3, max_k = 10, plot=False)
                            #Np.nanmean:
                            #num_factors = 1+np.argmax(np.nanmean(LL, axis=0))
                            #print 'optimal LL factors: ', num_factors

                            FA = skdecomp.FactorAnalysis(n_components=n_dim_latent)

                            #Samples x features:
                            FA.fit(bs_train_stack)

                            # Predicted test data: 
                            dat_pred = np.dot(FA.components_.T, FA.transform(bs_test_stack).T).T +FA.mean_[np.newaxis, :]

                            if i_t in [0, 1]:                            
                                ### Task and non-task indices: 
                                ix_tsk = np.nonzero(pt_ix_trials==i_t)[0]
                                non_ix_tsk = np.nonzero(pt_ix_trials==non_i_t)[0]

                                ### First normalized, then not: 
                                R2_tsk = [fit_LDS.get_R2(bs_test_stack[ix_tsk, :], dat_pred[ix_tsk, :])]
                                R2_tsk.append(fit_LDS.get_R2(bs_test_stack[ix_tsk, :], dat_pred[ix_tsk, :], day_mu, day_st))
                                
                                data_FA[i_nf, i_t, i_o, 'R2_tsk'] = R2_tsk

                                R2_nontsk = [fit_LDS.get_R2(bs_test_stack[non_ix_tsk, :], dat_pred[non_ix_tsk, :])]
                                R2_nontsk.append(fit_LDS.get_R2(bs_test_stack[non_ix_tsk, :], dat_pred[non_ix_tsk, :], day_mu, day_st))
                                data_FA[i_nf, i_t, i_o, 'R2_nontsk'] = R2_nontsk

                            elif i_t in [2]:
                                ix_test0 = np.nonzero(pt_ix_trials == 2)[0]

                                ### First normalized, then not: 
                                R2_tsk = [fit_LDS.get_R2(bs_test_stack[ix_test0, :], dat_pred[ix_test0, :])]
                                R2_tsk.append(fit_LDS.get_R2(bs_test_stack[ix_test0, :], dat_pred[ix_test0, :], day_mu, day_st))
                                data_FA[i_nf, i_t, i_o, 'R2_test'] = R2_tsk

                            if not only_important:

                                ### Get neural push estimates: 
                                kg_real = np.vstack((np_testing_all))[:, [3, 5]]
                                kg_FA = np.dot(kg_decoder, dat_pred.T).T
                                kg_FA = kg_FA[:, [3, 5]]

                                if i_t in [0, 1]:
                                    R2_tsk = fit_LDS.get_R2(kg_real[ix_tsk, :], kg_FA[ix_tsk, :])
                                    R2_nontsk = fit_LDS.get_R2(kg_real[non_ix_tsk, :], kg_FA[non_ix_tsk, :])
                                    
                                    data_FA[i_nf, i_t, i_o, 'R2_vx_vy_tsk'] = R2_tsk
                                    data_FA[i_nf, i_t, i_o, 'R2_vx_vy_nottsk'] = R2_nontsk
                                elif i_t in [2]:
                                    R20 = fit_LDS.get_R2(kg_real[ix_test0, :], kg_FA[ix_test0, :])
                                    data_FA[i_nf, i_t, i_o, 'R2_vx_vy_test'] = R20                               

                            U_B = np.matrix(FA.components_.T)
                            UUT_B = U_B*U_B.T
                            main_dim = fit_LDS.get_main_shared_dim(UUT_B)
                            data_FA[i_nf, i_t, i_o, 'FA_main_shared'] = main_dim

                        elif mod == 'max_tuning':
                            # Get Model #
                            model_nms, model_str = plot_factor_tuning_curves.lag_ix_2_var_nm(np.arange(-4, 5), 'vel')
                            model_nms_p, model_str = plot_factor_tuning_curves.lag_ix_2_var_nm(np.arange(-4, 5), 'pos')
                            
                            if i_t == 2: 

                                pt_ix_trials = []; 

                                ### Do something: 
                                ### TRAINING DATA: 
                                data1, data_temp1, spikes1, sub_spk_temp_all1, push1 = plot_factor_tuning_curves.get_spike_kinematics(animal, [input_type[i_d][0]], 
                                    [model_order[i_d][0]], 4, **dict(trial_ix = data_master['xval'][i_d, 0, i_nf, 'train']))
                                data_temp_dict1 = plot_factor_tuning_curves.panda_to_dict(data_temp1)
                                data_temp_dict1['spks'] = sub_spk_temp_all1[:, keep_units]

                                data2, data_temp2, spikes2, sub_spk_temp_all2, push2 = plot_factor_tuning_curves.get_spike_kinematics(animal, [input_type[i_d][1]], 
                                    [model_order[i_d][1]], 4, **dict(trial_ix = data_master['xval'][i_d, 1, i_nf, 'train']))
                                data_temp_dict2 = plot_factor_tuning_curves.panda_to_dict(data_temp2)
                                data_temp_dict2['spks'] = sub_spk_temp_all2[:, keep_units]
                                
                                #### TESTING DATA: 
                                data3, data_temp3, spikes3, sub_spk_temp_all3, push3 = plot_factor_tuning_curves.get_spike_kinematics(animal, [input_type[i_d][0]], 
                                    [model_order[i_d][0]], 4, **dict(trial_ix = data_master['xval'][i_d, 0, i_nf, 'test']))
                                data_temp_dict3 = plot_factor_tuning_curves.panda_to_dict(data_temp3)
                                data_temp_dict3['spks'] = sub_spk_temp_all3[:, keep_units]
                                pt_ix_trials.append([2]*sub_spk_temp_all3.shape[0])

                                data4, data_temp4, spikes4, sub_spk_temp_all4, push4 = plot_factor_tuning_curves.get_spike_kinematics(animal, [input_type[i_d][1]], 
                                    [model_order[i_d][1]], 4, **dict(trial_ix = data_master['xval'][i_d, 1, i_nf, 'test']))
                                data_temp_dict4 = plot_factor_tuning_curves.panda_to_dict(data_temp4)
                                data_temp_dict4['spks'] = sub_spk_temp_all4[:, keep_units]
                                pt_ix_trials.append([2]*sub_spk_temp_all4.shape[0])

                                spks_train = np.vstack((data_temp_dict1['spks'], data_temp_dict2['spks']))
                                model_, pred, pred_nontask = plot_factor_tuning_curves.fit_ridge(spks_train, data_temp_dict1, np.hstack((model_nms, model_nms_p)), 
                                    test_data=data_temp_dict3, test_data2=data_temp_dict4, train_data2 = data_temp_dict2)
                            
                                # Predict w/ model #
                                pred_out = np.vstack((pred, pred_nontask))
                                spks_true = np.vstack((data_temp_dict3['spks'], data_temp_dict4['spks']))
                                np_out = np.vstack((push3, push4))

                                pt_ix_trials = np.hstack((pt_ix_trials))
                                pt_ix_ix = np.nonzero(pt_ix_trials == 2)[0]
                                R2_ = [fit_LDS.get_R2(spks_true[pt_ix_ix, :], pred_out[pt_ix_ix, :])]
                                R2_.append(fit_LDS.get_R2(spks_true[pt_ix_ix, :], pred_out[pt_ix_ix, :], day_mu, day_st))
                                data_tuning[i_nf, i_t, i_o, 'R2_test'] = R2_
                                
                                ### Get neural pushses: 
                                if only_important is not True: 
                                    np_pred = np.dot(kg_decoder, pred_out.T).T

                                    R2 = fit_LDS.get_R2(np_out, np_pred[:, [3, 5]])
                                    data_tuning[i_nf, i_t, i_o, 'R2_vx_vy_test'] = R2; 

                            else:
                                ### This is to get all of the parameters
                                data, data_temp, spikes, sub_spk_temp_all, push_train = plot_factor_tuning_curves.get_spike_kinematics(animal, [input_type[i_d][i_t]], 
                                    [model_order[i_d][i_t]], 4, **dict(trial_ix = data_master['xval'][i_d, i_t, i_nf, 'train']))
                                
                                data_temp_dict = plot_factor_tuning_curves.panda_to_dict(data_temp)

                                ### Keep units only: 
                                ### This is the data for training data: 
                                data_temp_dict['spks'] = sub_spk_temp_all[:, keep_units]

                                ### Same task, test data: 
                                _, data_temp_test, _, sub_spk_temp_test, push_test = plot_factor_tuning_curves.get_spike_kinematics(animal, [input_type[i_d][i_t]], 
                                    [model_order[i_d][i_t]], 4, **dict(trial_ix = data_master['xval'][i_d, i_t, i_nf, 'test']))
                                
                                data_temp_dict_test = plot_factor_tuning_curves.panda_to_dict(data_temp_test)
                                data_temp_dict_test['spks'] = sub_spk_temp_test[:, keep_units]

                                # Non-task Data: 
                                _, data_temp_test_nontask, _, sub_spk_temp_test_nontask, push_test_nontask = plot_factor_tuning_curves.get_spike_kinematics(animal, [input_type[i_d][non_i_t]], 
                                    [model_order[i_d][non_i_t]], 4)
                                data_temp_dict_test_nontask = plot_factor_tuning_curves.panda_to_dict(data_temp_test_nontask)
                                data_temp_dict_test_nontask['spks'] = sub_spk_temp_test_nontask[:, keep_units]

                                ### Fit and Predict ###
                                model_, pred, pred_nontask = plot_factor_tuning_curves.fit_ridge(data_temp_dict['spks'], data_temp_dict, np.hstack((model_nms, model_nms_p)), 
                                    test_data=data_temp_dict_test, test_data2=data_temp_dict_test_nontask)

                                # Predict w/ model #
                                R2_tsk = [fit_LDS.get_R2(data_temp_dict_test['spks'], pred)]
                                R2_tsk.append(fit_LDS.get_R2(data_temp_dict_test['spks'], pred, day_mu, day_st))
                                data_tuning[i_nf, i_t, i_o, 'R2_tsk'] = R2_tsk

                                R2_nontsk = [fit_LDS.get_R2(data_temp_dict_test_nontask['spks'], pred_nontask)]
                                R2_nontsk.append(fit_LDS.get_R2(data_temp_dict_test_nontask['spks'], pred_nontask, day_mu, day_st))
                                data_tuning[i_nf, i_t, i_o, 'R2_nontsk']  = R2_nontsk

                                ### Get neural pushses: 
                                if only_important is not True: 
                                    np_pred_tsk = np.dot(kg_decoder, pred.T).T
                                    np_pred_nontsk = np.dot(kg_decoder, pred_nontask.T).T

                                    R2 = fit_LDS.get_R2(push_test, np_pred_tsk[:, [3, 5]])
                                    data_tuning[i_nf, i_t, i_o, 'R2_vx_vy_tsk'] = R2; 

                                    R22 = fit_LDS.get_R2(push_test_nontask, np_pred_nontsk[:, [3, 5]])
                                    data_tuning[i_nf, i_t, i_o, 'R2_vx_vy_nontsk'] = R22; 

                        elif 'LPF' in mod:

                            # Just fit test subset w/ LPF: 
                            s_d = float(mod[3:])
                            bs_train_sub = [d[5:,:] for d in bs_train]
                            bs_test_sub =  [d[5:, :] for d in bs_testing_all]
                            bs_test_sub1 = np.vstack(([d[5:-1, :] for d in bs_testing_all]))

                            np_test_all = np.vstack(( [d[:-1, :] for d in np_testing_all] ))[:, [3, 5]]

                            pred_bs, smooth_bs = fit_LDS.fit_LPF(bs_test_sub, s_d, sep_trials = True)

                            # 4 not 5 because pred and smoo offset: 
                            pred = np.vstack((pred_bs))
                            smoo = np.vstack((smooth_bs))

                            # R2: 
                            data_LPF[i_nf, i_t, i_o, s_d, 'R2_pred'] = fit_LDS.get_R2(bs_test_sub1, pred)
                            data_LPF[i_nf, i_t, i_o, s_d, 'R2_smooth'] = fit_LDS.get_R2(bs_test_sub1, smoo)

                            if not only_important:
                                pred_np = np.dot(kg_decoder, np.vstack(( [pred_bs[i][4:, :] for i in range(len(pred_bs))] )).T).T
                                smoo_np = np.dot(kg_decoder, np.vstack(( [smooth_bs[i][4:, :] for i in range(len(smooth_bs))] )).T).T

                                data_LPF[i_nf, i_t, i_o, s_d, 'R2_pred_vx_vy_tsk'] = fit_LDS.get_R2(np_test_all, pred_np[:, [3, 5]])
                                data_LPF[i_nf, i_t, i_o, s_d, 'R2_smooth_vx_vy_tsk'] = fit_LDS.get_R2(np_test_all, smoo_np[:, [3, 5]])
      
                print 'done w/ fold: ', i_nf
        big_data['data_LDS'] = data_LDS
        big_data['data_FA'] = data_FA
        big_data['data_tuning'] = data_tuning
        big_data['data_LPF'] = data_LPF
        pickle.dump(big_data, open(fname_pref+'big_data_'+str(i_d)+'_9_6_19.pkl', 'wb'))
        print 'done w/ ', animal, ' all models, all folds'
        #fcns.send_email('done with monkey: '+animal+', day: '+str(i_d), 'data_alert')

### Main linear regression function ###
def main_lin_regression(animal, only_important = False):

    '''
    Things to compute: 
        For [all neurons, imporant neurons, neural push]:
            a. y_t | np_t 
            b. y_t | np_t, s_t
            c. y_t | np_t, s_t, task (diff models for tasks)
            d. y_t | y_t-1
            e. y_t | y_t-1, np_t
            f. y_t | y_t-1, np_t, s_t
            e. y_t | (a_t-T,..a_t+T), (np_t-T,..np_t+T)

    Compute 1) general performance when training on both sets of data. 2) generalization performance. 

    '''
    
    models = [['np'], ['np', 'st'], ['np', 'st', 'tsk'], ['ytm1'], ['ytm1', 'np'], ['ytm1','np','st']]

    # For each bin / task get a_t, s_t, y_t, y_{t-1}
    gather_means = dict(); # day / task / stuff

    ### load data structs ###
    data_master = pickle.load(open(data_agg[animal]))

    if animal == 'grom':
        input_type = co_obs_tuning_matrices.input_type
        model_order = co_obs_tuning_matrices.ordered_input_type
        fname_pref = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/test_'
    
    elif animal == 'jeev':
        input_type = fk.task_filelist
        model_order = fk.ordered_task_filelist
        fname_pref = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/test_'
    
    ### run models ###
    for i_d in range(len(input_type)):
        data_tuning = {}
        gather_means[i_d] = dict()
        gather_means[i_d][0] = dict();
        gather_means[i_d][1] = dict()

        ### runs models for important neurons only ###
        # for i_o, only_important in enumerate([False, True]):#False, True]):

        ### Which neurons are we removing / keeeping? 
        if only_important: 

            ### These are zero points ###
            rm = data_master['dat'][i_d, 'rm_units_imp']

            ### New vs. old important
            new_important = subspace_overlap.get_important_units(animal, i_d, est_type = 'est', thresh = .8)
            old_important = subspace_overlap.get_important_units(animal, i_d, est_type = 'est')

            N_units = data_master['dat'][i_d, 0, 'bs_imp'][0].shape[1]

            ### Get the new units: 
            assert(N_units == len(old_important)); 
            new_keep_units = np.array([i for i, j in enumerate(old_important) if j in new_important])

            #### Make sure they arent ones we wanted to throw away (unlikely)
            keep_units = np.array([i for i in new_keep_units if i not in rm])

            day_mu = data_master['dat'][i_d, 'mfr_imp'][keep_units]
            day_st = data_master['dat'][i_d, 'std_imp'][keep_units]
            bs_key = 'bs_imp'
            i_o = 1; 
        else:
            rm = data_master['dat'][i_d, 'rm_units']
            N_units = data_master['dat'][i_d, 0, 'bs'][0].shape[1]
            keep_units = np.array([i for i in range(N_units) if i not in rm])
            day_mu = data_master['dat'][i_d, 'mfr'][keep_units]
            day_st = data_master['dat'][i_d, 'std'][keep_units]
            bs_key = 'bs'
            i_o = 0; 

        ### For each xval: 
        for i_nf in range(nfolds):
            print('Startign fold %d, Day %s, Only important %d' %(i_nf, i_d, i_o))

            ### Task number 2 is a combination of both tasks for training and testing -- focus on this only for now:
            for i_t in range(3): 

                params = dict()
                
                for p in ['np', 'st', 'tsk', 'yt', 'ytm1']:
                    params[p + '_train'] = []
                    params[p + '_test_wi'] = []
                    params[p + '_test_x'] = []
                    params[p + '_test'] = []

                if i_t in [0, 1]:
                    ### For task i_t:    
                    data_train_ix = data_master['xval'][i_d, i_t, i_nf, 'train']
                    data_test_ix = data_master['xval'][i_d, i_t, i_nf, 'test']
                    non_i_t = np.mod(i_t+1, 2)

                    # Gather the data -- binned spike counts test / train
                    bs_train = [data_master['dat'][i_d, i_t, bs_key][i][pre_go_bins:, keep_units] for i in data_train_ix]
                    bs_test =  [data_master['dat'][i_d, i_t, bs_key][i][pre_go_bins:, keep_units] for i in data_test_ix]
                    params['yt_train'].append(np.vstack((bs_train)))
                    params['yt_test_wi'].append(np.vstack((bs_test)))

                    # Gather y_t minus 1 for testing dynamics:
                    bs_train_tm1 = np.vstack(([data_master['dat'][i_d, i_t, bs_key][i][pre_go_bins-1:-1, keep_units] for i in data_train_ix]))
                    bs_test_tm1 =  np.vstack(([data_master['dat'][i_d, i_t, bs_key][i][pre_go_bins-1:-1, keep_units] for i in data_test_ix]))

                    n_train = bs_train_tm1.shape[0]
                    n_test = bs_test_tm1.shape[0]
                    
                    params['ytm1_train'].append(np.vstack((bs_train_tm1)))
                    params['ytm1_test_wi'].append(np.vstack((bs_test_tm1)))

                    ## Get the neural push
                    params['np_train'].append(np.vstack(([data_master['dat'][i_d, i_t, 'np'][i][pre_go_bins:, [3, 5]] for i in data_train_ix])))
                    params['np_test_wi'].append(np.vstack(([data_master['dat'][i_d, i_t, 'np'][i][pre_go_bins:, [3, 5]] for i in data_test_ix])))
                    
                    params['st_train'].append(np.vstack(([data_master['dat'][i_d, i_t, 'st'][i][pre_go_bins:, :] for i in data_train_ix])))
                    params['st_test_wi'].append(np.vstack(([data_master['dat'][i_d, i_t, 'st'][i][pre_go_bins:, :] for i in data_test_ix])))
                    
                    params['tsk_train'].append( np.zeros((n_train)) + i_t )
                    params['tsk_test_wi'].append( np.zeros((n_test)) + i_t )
                    
                    ### INCLDUE OFF TASK ###
                    non_i_t = np.mod(i_t+1, 2)

                    ### Add non-task data -- not just test/train. 
                    bs_test_off_tsk = [data_master['dat'][i_d, non_i_t, bs_key][i][pre_go_bins:, keep_units] for i in range(len(data_master['dat'][i_d, non_i_t, bs_key]))]
                    bs_test_off_tsk_tm1 = [data_master['dat'][i_d, non_i_t, bs_key][i][pre_go_bins-1:-1, keep_units] for i in range(len(data_master['dat'][i_d, non_i_t, bs_key]))]
                    np_test_off_tsk = [data_master['dat'][i_d, non_i_t, 'np'][i][pre_go_bins:, [3, 5]] for i in range(len(data_master['dat'][i_d, non_i_t, 'np']))]
                    st_test_off_tsk = [data_master['dat'][i_d, non_i_t, 'st'][i][pre_go_bins:, :] for i in range(len(data_master['dat'][i_d, non_i_t, 'st']))]
                    n_test_x = np.vstack((st_test_off_tsk)).shape[0]
                    tsk_test_off_tsk = np.zeros((n_test_x)) + non_i_t; 

                    params['yt_test_x'].append(np.vstack((bs_test_off_tsk)))
                    params['ytm1_test_x'].append(np.vstack((bs_test_off_tsk_tm1)))
                    params['np_test_x'].append(np.vstack((np_test_off_tsk)))
                    params['st_test_x'].append(np.vstack((st_test_off_tsk)))
                    params['tsk_test_x'].append(np.vstack((tsk_test_off_tsk)))
                    
                ### If want to train using both tasks for comparison to other models: 
                elif i_t in [2]:
                    
                    # Cycle through both tasks to use
                    for i_t2 in range(2):

                        bs = []; at = []; bs_tm1 = []; st = []; 

                        ### For task i_t:    
                        data_train_ix = data_master['xval'][i_d, i_t2, i_nf, 'train']
                        data_test_ix =  data_master['xval'][i_d, i_t2, i_nf, 'test']

                        for _, (dat_ix, key) in enumerate(zip([data_train_ix, data_test_ix], ['train', 'test'])):

                            bs.extend([data_master['dat'][i_d, i_t2, bs_key][i][pre_go_bins:, keep_units] for i in dat_ix])
                            bs_tm1.extend([data_master['dat'][i_d, i_t2, bs_key][i][pre_go_bins-1:-1, keep_units] for i in dat_ix])
                            at.extend([data_master['dat'][i_d, i_t2, 'np'][i][pre_go_bins:, [3, 5]] for i in dat_ix])
                            st.extend([data_master['dat'][i_d, i_t2, 'st'][i][pre_go_bins:, :] for i in dat_ix])

                            # Aggregate the data: 
                            for i in data_train_ix:
                                params['yt_'+key].append(data_master['dat'][i_d, i_t2, bs_key][i][pre_go_bins:, keep_units])
                                params['ytm1_'+key].append(data_master['dat'][i_d, i_t2, bs_key][i][pre_go_bins-1:-1, keep_units])
                                params['np_'+key].append(data_master['dat'][i_d, i_t2, 'np'][i][pre_go_bins:, [3, 5]])
                                params['st_'+key].append(data_master['dat'][i_d, i_t2, 'st'][i][pre_go_bins:, :])
                            x = np.vstack(([data_master['dat'][i_d, i_t2, 'st'][i][pre_go_bins:,:] for i in data_train_ix]))
                            params['tsk_'+key].append(np.zeros((x.shape[0])) + i_t2)

                        ### Get the task-related means here: 
                        vel_bins = np.vstack(( subspace_overlap.commands2bins(at, mag_boundaries, animal, i_d, vel_ix = [0, 1])))
                        bs = np.vstack((bs))
                        bs_tm1 = np.vstack((bs_tm1))
                        at = np.vstack((at))
                        st = np.vstack((st))

                        # If in fold 1: 
                        if i_nf == 0: 
                            for mag in range(4):
                                m_ix = np.nonzero(vel_bins[:, 0] == mag)[0]

                                for ang in range(8):
                                    a_ix = np.nonzero(vel_bins[m_ix, 1] == ang)[0]

                                    if len(a_ix) > 15:
                                        ix = m_ix[a_ix]

                                        gather_means[i_d][i_t2][mag, ang] = dict()
                                        gather_means[i_d][i_t2][mag, ang]['bs'] = np.mean(bs[ix, :], axis=0)
                                        gather_means[i_d][i_t2][mag, ang]['bs_tm1'] = np.mean(bs_tm1[ix, :], axis=0)
                                        gather_means[i_d][i_t2][mag, ang]['at'] = np.mean(at[ix, :], axis=0)
                                        gather_means[i_d][i_t2][mag, ang]['st'] = np.mean(st[ix, :], axis=0)


                ### STart model fitting ####
                for i_m, model_params in enumerate(models):

                    if 'tsk' in model_params: 

                        if i_t == 2:

                            ### Train a diff linear regression model for the task
                            X = dict(); X_test = dict();  
                            
                            for mp in model_params:
                                if 'tsk' not in mp:
                                    ### Training:
                                    X = mult_to_dict(np.vstack((params[mp+'_train'])), X, mp)
                                    X_test = mult_to_dict(np.vstack((params[mp+'_test'])), X_test, mp); 
                            
                            tsk_train = np.hstack((params['tsk_train']))
                            tsk_test = np.hstack((params['tsk_test']))

                            ### Now train a model for each neuron at a time: 
                            Y_pred = np.zeros(( (len(tsk_test), len(keep_units)))) 

                            for nneur in range(len(keep_units)):

                                for model_task in range(2):

                                    ### Training
                                    ix_tsk = np.nonzero(tsk_train == model_task)[0]
                                    ix_tsk_test = np.nonzero(tsk_test == model_task)[0]

                                    ytrain =  pandas.DataFrame(data = dict(yt=np.vstack((params['yt_train']))[:, nneur]))
                                    ytrain_sub = ytrain.iloc[ix_tsk]
                                    X = pandas.DataFrame(data=X)
                                    X = sm.add_constant(X)

                                    ### Multiple linear regression: 
                                    X_sub = X.iloc[ix_tsk]
                                    model = sm.OLS(ytrain_sub, X_sub).fit()

                                    ### Testing: 
                                    ytest = pandas.DataFrame(data = dict(yt = np.vstack((params['yt_test']))[:, nneur]))
                                    ytest_sub = ytest.iloc[ix_tsk_test]

                                    X_test = pandas.DataFrame(data=X_test)
                                    X_test = sm.add_constant(X_test)
                                    X_test_sub = X_test.iloc[ix_tsk_test]

                                    ### Predict: 
                                    Y_pred[ix_tsk_test, nneur] = np.array(model.predict(X_test_sub).tolist())

                            ytest = np.vstack((params['yt_test']))
                            x2 = [fit_LDS.get_R2(ytest, Y_pred), fit_LDS.get_R2(ytest, Y_pred, mfr=day_mu, stdfr=day_st)]
                            data_tuning[i_nf, i_t, i_o, i_m] = x2; 


                    else:
                        X = dict(); X_test_w = dict(); X_test_x = dict();  X_test = dict(); 
                        for mp in model_params:
                            ### Trainign:
                            X = mult_to_dict(np.vstack((params[mp+'_train'])), X, mp)

                            if i_t in [0, 1]:
                                X_test_w = mult_to_dict(np.vstack((params[mp+'_test_wi'])), X_test_w, mp); 
                                yt_w = np.vstack((params['yt_test_wi']))

                                X_test_x = mult_to_dict(np.vstack((params[mp+'_test_x'])), X_test_x, mp); 
                                yt_x = np.vstack((params['yt_test_x']))
                            else:
                                X_test = mult_to_dict(np.vstack((params[mp+'_test'])), X_test, mp); 
                                ytest = np.vstack((params['yt_test']))

                        ### Now train a model for each neuron at a time: 
                        Y_pred_w = []; Y_pred_x = []; Y_pred = []; 

                        for nneur in range(len(keep_units)):
                            ytrain =  pandas.DataFrame(data = dict(yt=np.vstack((params['yt_train']))[:, nneur]))
                            X = pandas.DataFrame(data=X)
                            X = sm.add_constant(X)

                            ### Multiple linear regression: 
                            model = sm.OLS(ytrain, X).fit()

                            if i_t in [0, 1]:
                                ### Test: 
                                x0 = pandas.DataFrame(data=X_test_w)
                                x0 = sm.add_constant(x0)
                                Y_pred_w.append(np.array(model.predict(x0).tolist())[:, np.newaxis])

                                x1 = pandas.DataFrame(data=X_test_x)
                                x1 = sm.add_constant(x1)
                                Y_pred_x.append(np.array(model.predict(x1).tolist())[:, np.newaxis])

                            else:
                                x2 = pandas.DataFrame(data=X_test); 
                                x2 = sm.add_constant(x2)
                                Y_pred.append(np.array(model.predict(x2).tolist())[:, np.newaxis])

                                if i_m == 5:
                                    gather_means[i_d]['model', nneur, i_nf] = model; 

                        ### Save R2: 
                        if i_t in [0, 1]:
                            x = [fit_LDS.get_R2(yt_w, np.hstack((Y_pred_w))), fit_LDS.get_R2(yt_w, np.hstack((Y_pred_w)), mfr=day_mu, stdfr=day_st)]
                            data_tuning[i_nf, i_t, i_o, i_m, 'w'] = x

                            x1 = [fit_LDS.get_R2(yt_x, np.hstack((Y_pred_x))), fit_LDS.get_R2(yt_x, np.hstack((Y_pred_x)), mfr=day_mu, stdfr=day_st)]
                            data_tuning[i_nf, i_t, i_o, i_m, 'x'] = x1; 

                        else:
                            x2 = [fit_LDS.get_R2(ytest, np.hstack((Y_pred))), fit_LDS.get_R2(ytest, np.hstack((Y_pred)), mfr=day_mu, stdfr=day_st)]
                            data_tuning[i_nf, i_t, i_o, i_m] = x2; 

        pickle.dump(data_tuning, open(fname_pref+'linear_tuning_'+str(i_d)+'_9_10_19_thresh_0.8.pkl', 'wb'))

    return gather_means

def plot_main_lin_regression(only_important=0, norm_vals = 0, thresh = None):
    models = [['np'], ['np', 'st'], ['np', 'st', 'tsk'], ['ytm1'], ['ytm1', 'np'], ['ytm1','np','st']]

    f1, ax1 = plt.subplots() # General performance plot
    f2, ax2 = plt.subplots() # Generalizartion plot

    for i_a, animal in enumerate(['grom', 'jeev']):

        if animal == 'grom':
            input_type = co_obs_tuning_matrices.input_type
            model_order = co_obs_tuning_matrices.ordered_input_type
            fname_pref = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/test_'
            ndays = 4; 
        
        elif animal == 'jeev':
            input_type = fk.task_filelist
            model_order = fk.ordered_task_filelist
            fname_pref = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/test_'
            ndays = 4; 

        ### For each day
        model_dict = dict(); gen_dict = dict()
        for m in models: model_dict[tuple(m)] = []  
        for m in models:
            gen_dict[tuple(m)] = dict(w=[], x=[]) 

        for i_d in range(ndays):
            ### load the data: fold, task, important, model 
            if thresh is None:
                dat = pickle.load(open(fname_pref+'linear_tuning_'+str(i_d)+'_9_10_19.pkl', 'rb'))
            elif thresh == 0.8:
                dat = pickle.load(open(fname_pref+'linear_tuning_'+str(i_d)+'_9_10_19_thresh_0.8.pkl', 'rb'))

            for i_m, model in enumerate(models): 

                if 'tsk' not in model:
                    ### Get general performance first: 
                    model_dict[tuple(model)].append(np.vstack(( [dat[f, 2, only_important, i_m][norm_vals] for f in range(5)] )))

                    ### Now get generalization: 
                    for i in range(2): 
                        gen_dict[tuple(model)]['w'].append(np.vstack(( [dat[f, i, only_important, i_m, 'w'][norm_vals] for f in range(5)])))
                        gen_dict[tuple(model)]['x'].append(np.vstack(( [dat[f, i, only_important, i_m, 'x'][norm_vals] for f in range(5)])))

        xlab = []; 
        for i_m, model in enumerate(models):
            if 'tsk' not in model:
                val = np.mean(np.hstack(( model_dict[tuple(model)] )), axis=0)

                ax1.bar(i_m + 8*i_a, np.mean(val), width=1.,edgecolor='k', linewidth=4., color='w')
                ax1.errorbar(i_m + 8*i_a, np.mean(val), np.std(val)/np.sqrt(len(val)), color='k', marker='|')
                
                val_w = np.mean(np.hstack(( gen_dict[tuple(model)]['w'])), axis=0)
                val_x = np.mean(np.hstack(( gen_dict[tuple(model)]['x'])), axis=0)

                ax2.bar(i_m + 8*i_a, np.mean(val_w), width=.4,edgecolor='k', linewidth=4., color='w')
                ax2.errorbar(i_m + 8*i_a, np.mean(val_w), np.std(val_w)/np.sqrt(len(val_w)), color='k', marker='|')

                ax2.bar(i_m+.4 + 8*i_a, np.mean(val_x), width=.4,edgecolor='k', linewidth=4., color='w')
                ax2.errorbar(i_m+.4 + 8*i_a, np.mean(val_x), np.std(val_x)/np.sqrt(len(val_x)), color='k', marker= '|')

            xlab.append(list_2_lab(model))

        for axi in [ax1, ax2]:
            axi.set_xticks(np.arange(len(models)))
            axi.set_xticklabels(xlab, rotation=90)
    f1.tight_layout()
    f2.tight_layout()

def list_2_lab(a):
    s = ''
    for ia in a:
        s = s+ia+','
    return s

def mult_to_dict(X, dic, nm_base):
    N = X.shape[1]

    for n in range(N):
        nm = nm_base + '_'+str(n)
        if X[:,n].shape[0] == 1:
            dic[nm] = X[:, n]
        else:
            dic[nm] = np.squeeze(np.array(X[:, n]))
    return dic; 

### Method to get mean i) neural activity pattern (t), (t-1) ii) action iii) state for each bin / task
### Then, get model fit for y_t | y_{t-1}, a_t, s_t on single trials
def plot_gather_means(gather_means):

    f0, ax0 = plt.subplots(ncols = 3, nrows = 3)
    f1, ax1 = plt.subplots(ncols = 3, nrows = 3)

    ### gather_means has means for all above: 
    for i_d in gather_means.keys():

        for bin in gather_means[i_d][0].keys():

            ### If other task has this bin too: 
            if bin in gather_means[i_d][1].keys():

                ### individual points:  
                y_t_0 = gather_means[i_d][0][bin]['bs']
                y_t_1 = gather_means[i_d][1][bin]['bs']

                y_tm1_0 = gather_means[i_d][0][bin]['bs_tm1'][np.newaxis, :]
                y_tm1_1 = gather_means[i_d][1][bin]['bs_tm1'][np.newaxis, :]

                nneur = len(y_t_0)

                ### Then can plot a point: 
                y_t_0_pred_from_1 = np.zeros((nneur, 5)) 
                y_t_1_pred_from_0 = np.zeros((nneur, 5))
                y_t_0_pred_from_0 = np.zeros((nneur, 5)) 
                y_t_1_pred_from_1 = np.zeros((nneur, 5))


                at_0 = np.squeeze(np.array(gather_means[i_d][0][bin]['at']))[np.newaxis, :]
                at_1 = np.squeeze(np.array(gather_means[i_d][1][bin]['at']))[np.newaxis, :]
                
                st_0 = np.squeeze(np.array(gather_means[i_d][0][bin]['st']))[np.newaxis, :]
                st_1 = np.squeeze(np.array(gather_means[i_d][1][bin]['st']))[np.newaxis, :]

                X0 = dict(); 
                X0 = mult_to_dict(y_tm1_1, X0, 'ytm1') ### note adding opposite
                X0 = mult_to_dict(at_0, X0, 'np')
                X0 = mult_to_dict(st_0, X0, 'st')
                X0['const'] = 1;
                X0 = pandas.DataFrame(data=X0)

                X0_true = dict(); 
                X0_true = mult_to_dict(y_tm1_0, X0_true, 'ytm1') ### note same
                X0_true = mult_to_dict(at_0, X0_true, 'np')
                X0_true = mult_to_dict(st_0, X0_true, 'st')
                X0_true['const'] = 1;
                X0_true = pandas.DataFrame(data=X0_true)

                X1 = dict(); 
                X1 = mult_to_dict(y_tm1_0, X1, 'ytm1') ### note adding opposite
                X1 = mult_to_dict(at_1, X1, 'np')
                X1 = mult_to_dict(st_1, X1, 'st')
                X1['const'] = 1;
                X1 = pandas.DataFrame(data=X1)

                X1_true = dict(); 
                X1_true = mult_to_dict(y_tm1_1, X1_true, 'ytm1') ### note same
                X1_true = mult_to_dict(at_1, X1_true, 'np')
                X1_true = mult_to_dict(st_1, X1_true, 'st')
                X1_true['const'] = 1;
                X1_true = pandas.DataFrame(data=X1_true)

                for i in range(5):
                    for n in range(nneur): 
                        mod = gather_means[i_d][('model', n, i)]
                        y_t_0_pred_from_1[n, i] = mod.predict(X0)
                        y_t_0_pred_from_0[n, i] = mod.predict(X0_true)

                        y_t_1_pred_from_0[n, i] = mod.predict(X1)
                        y_t_1_pred_from_1[n, i] = mod.predict(X1_true)

                ### Now take mean over folds: 
                y_t_0_pred_mn = np.mean(y_t_0_pred_from_1, axis=1)
                y_t_1_pred_mn = np.mean(y_t_1_pred_from_0, axis=1)

                y_t_0_true_mn = np.mean(y_t_0_pred_from_0, axis=1)
                y_t_1_true_mn = np.mean(y_t_1_pred_from_1, axis=1)

                ax0[i_d / 3, i_d % 3].plot(np.linalg.norm(y_t_0 - y_t_0_true_mn), 
                    np.linalg.norm(y_t_0 - y_t_0_pred_mn), '.')

                ax1[i_d / 3, i_d % 3].plot(np.linalg.norm(y_t_1 - y_t_1_true_mn), 
                    np.linalg.norm(y_t_1 - y_t_1_pred_mn), '.')

        ax0[i_d / 3, i_d % 3].set_xlim([0., 3.])
        ax0[i_d / 3, i_d % 3].set_ylim([0., 3.])
        ax0[i_d / 3, i_d % 3].plot([0., 3.], [0., 3.], 'k--')

        ax1[i_d / 3, i_d % 3].set_xlim([0., 3.])
        ax1[i_d / 3, i_d % 3].set_ylim([0., 3.])
        ax1[i_d / 3, i_d % 3].plot([0., 3.], [0., 3.], 'k--')

    f0.tight_layout()
    f1.tight_layout()

def get_dyn_ratio(kg_list, bs_testing_all, data_vel_test, pred_state, filt_state, key, C, A, task):
    ''' 
    Function to get the dynamics ratio associated with each bin 
    '''
    dyn_ratio_master = []
    for it, trl in enumerate(bs_testing_all):
        if data_vel_test[it][0,0] == task:
            for ib in range(trl.shape[0]-10):
                if ib > 0:
                    ### Each Bin ###: 
                    if key == 'self':
                        dyn = np.linalg.norm(np.mat(pred_state[it][:, ib]) - np.mat(filt_state[it][ib-1, :]).T)
                        inn = np.linalg.norm(kg_list[it][ib]*(np.mat(bs_testing_all[it][ib+10, :]).T - C*np.mat(A)*np.mat(filt_state[it][ib-1, :]).T ))
                        dyn_ratio = [dyn/(dyn+inn)]

                    else:
                        if data_vel_test[it].shape[0] == trl.shape[0] - 10:
                            tsk, trg, ang, mag = data_vel_test[it][ib]
                            dyn_ratio = []
                            
                            for it2 in np.random.permutation(len(bs_testing_all)):
                                if it2!=it:
                                    if key == 'same_task_targ':
                                        key_min = 4
                                        ix = (data_vel_test[it2][:, 0]==tsk).astype(int)
                                        ix += (data_vel_test[it2][:, 1]==trg).astype(int)
                                        ix += (data_vel_test[it2][:, 2] == ang).astype(int)
                                        ix += (data_vel_test[it2][:, 3] == mag).astype(int)
                                    
                                    elif key == 'same_task_diff_targ':
                                        key_min = 4
                                        ix = (data_vel_test[it2][:, 0]==tsk).astype(int)
                                        ix += (data_vel_test[it2][:, 1]!=trg).astype(int)
                                        ix += (data_vel_test[it2][:, 2] == ang).astype(int)
                                        ix += (data_vel_test[it2][:, 3] == mag).astype(int)                                
                                    elif key == 'diff_task':
                                        key_min = 3
                                        ix = (data_vel_test[it2][:, 0]!=tsk).astype(int)
                                        ix += (data_vel_test[it2][:, 2] == ang).astype(int)
                                        ix += (data_vel_test[it2][:, 3] == mag).astype(int)                                 

                                    ix_use = np.nonzero(ix == key_min)[0]
                                    
                                    if len(ix_use) > 0:
                                        for ixi in ix_use:
                                            dyn = np.linalg.norm(np.mat(pred_state[it][:, ib]) - np.mat(filt_state[it][ib-1, :]).T)
                                            inn = np.linalg.norm(kg_list[it][ib]*(np.mat(bs_testing_all[it2][ixi+10, :]).T - C*np.mat(A)*np.mat(filt_state[it][ib-1, :]).T ))
                                            dyn_ratio.append(dyn/(dyn+inn))
                                
                                if len(dyn_ratio) > 5:
                                    break

                    dyn_ratio_master.append(dyn_ratio)
    return np.hstack((dyn_ratio_master))

def main_lds_16(animal):
    d = pickle.load(open(data_agg[animal]))

    if animal == 'grom':
        input_type = co_obs_tuning_matrices.input_type
        model_order = co_obs_tuning_matrices.ordered_input_type
        #fname_pref = '/Volumes/TimeMachineBackups/grom2016/test_'
        fname_pref = '/Volumes/TimeMachineBackups/grom2016/test_'
        ss = pickle.load(open('/Users/preeyakhanna/fa_analysis/grom_data/grom2017_aug_subspace_ov_CO_vs_Obs_same_day_many_factors_w_epochs16.pkl'))
    
    elif animal == 'jeev':
        input_type = fk.task_input_type
        model_order = fk.ordered_task_filelist
        #fname_pref = '/Volumes/TimeMachineBackups/jeev2013/test_'
        fname_pref = '/Volumes/TimeMachineBackups/jeev2013/test_'
        ss = pickle.load(open('/Users/preeyakhanna/fa_analysis/grom_data/jeev2017_aug_subspace_ov_CO_vs_Obs_same_day_many_factors_w_epochs_16all.pkl'))
    
    LDS_master = {}
    for i_d in range(len(input_type)):
        rm_units = np.hstack(( data_master['dat'][i_d, 'rm_units'] ))
        N = data_master['dat'][i_d, 0, 'bs'][0].shape[1]
        keep_units = np.array([i for i in range(N) if i not in rm_units])
        te_list = ss[i_d, 'te']

        subspace_ov = np.zeros((len(te_list), len(te_list), 2))
        repertoire_ov = np.zeros((len(te_list), len(te_list), 2))
        LDS = {}
        data_array = {}

        for i_t, te in enumerate(te_list):
            if animal == 'grom':
                tmp = str(te)
                te_master = int(tmp[:tmp.find('.')])
                te_master_name = te_master
            
            elif animal == 'jeev':
                te_master_name = te[:te.find('_')]
                for i in range(2):
                    for j in range(len(input_type[i_d][i])):
                        if te_master_name in input_type[i_d][i][j]:
                            te_master = fk.task_filelist[i_d][i][j]

            if te_master_name not in data_array.keys():

                bin_spk_nonz, targ_ix, trial_ix_all, KG, exclude_ix = fit_LDS.pull_data(te_master, animal, pre_go=pre_go_bins/10.,
                    binsize_ms=100, keep_units=keep_units)
                data_array[te_master_name] = dict(bin_spk_nonz=bin_spk_nonz, targ_ix=targ_ix, trial_ix_all=trial_ix_all, KG=KG, exclude_ix=exclude_ix)

            
            trl_ixs = ss[i_d][te, 0].training_data_trl_ix
            data_sub = [data_array[te_master_name]['bin_spk_nonz'][i] for i in trl_ixs if i not in data_array[te_master_name]['exclude_ix']]
            # Use trial subsets from subspace overlap dictionary:  
            try:       
                R2_smooth, R2_pred, LL, dynamics_norm, innov_norm, model, pred_data, smooth_data, filt_state, pred_state, filt_sigma, kg, R2_filt = fit_LDS.fit_LDS(
                    data_sub, data_sub, n_dim_latent, return_model=True, seed_w_FA=True, nEMiters=30, **dict(seed_pred_x0_with_smoothed_x0= True, get_ratio=False, 
                        pre_go_bins=pre_go_bins))

                LDS[te] = model
                LDS[te, 'smooth_x'] = np.vstack(( [model.states_list[i].smoothed_mus for i in range(len(data_sub))]))
            except:
                print 'SKIPPING TE: ', te

        LDS_master[i_d] = LDS
    try:
        pickle.dump(LDS_master, open(fname_pref+'_LDS_16_epoch.pkl', 'w'))
        x=0
    except:
        x = LDS_master
    return x

def get_LDS_ov_and_rep_ov():
    for ia, animal in enumerate(['jeev', 'grom']):
        if animal == 'grom':
            dat = pickle.load(open('/Volumes/TimeMachineBackups/grom2016/test__LDS_16_epoch.pkl'))
            fn = '/Volumes/TimeMachineBackups/grom2016/'
        elif animal == 'jeev':
            dat = pickle.load(open('/Volumes/TimeMachineBackups/jeev2013/test__LDS_16_epoch.pkl'))
            fn = '/Volumes/TimeMachineBackups/jeev2013/'

        ov_master = {}
        for i_d in dat.keys():
            if animal == 'jeev':
                te_list = np.array([i for i in dat[i_d] if type(i) is str])
            elif animal == 'grom':
                te_list = np.array([i for i in dat[i_d] if type(i) is float])

            overlap_mat = np.zeros((len(te_list), len(te_list), 2))
            dim_dict = {}

            for i_t, te in enumerate(te_list):
                x0 = np.cov(dat[i_d][te, 'smooth_x'].T)
                C0 = np.mat(dat[i_d][te].C)
                
                Z_cov_0 = C0*x0*C0.T
                main_dim = fit_LDS.get_main_shared_dim(Z_cov_0)
                dim_dict[te] = main_dim
                Z_cov_0_mn, Z_cov_0_norm = tfo.get_mn_shar(Z_cov_0)

                for i_t2, te2 in enumerate(te_list[i_t:]):
                    x1 = np.cov(dat[i_d][te2, 'smooth_x'].T)
                    C1 = np.mat(dat[i_d][te2].C)
                    Z_cov_1 = C1*x1*C1.T
                    Z_cov_1_mn, Z_cov_1_norm = tfo.get_mn_shar(Z_cov_1)

                    proj_1_0_comb = np.trace(Z_cov_0_norm*Z_cov_1_mn*Z_cov_0_norm.T)/float(np.trace(Z_cov_1_mn))
                    proj_0_1_comb = np.trace(Z_cov_1_norm*Z_cov_0_mn*Z_cov_1_norm.T)/float(np.trace(Z_cov_0_mn))
                    print proj_0_1_comb, proj_1_0_comb
                    overlap_mat[i_t, i_t+i_t2, :] = np.array([proj_0_1_comb, proj_1_0_comb])
            ov_master[i_d] = overlap_mat
            ov_master[i_d, 'te'] = te_list
        pickle.dump(ov_master, open(fn+'LDS_16epoch_overlap.pkl', 'wb'))
        print 'done w/ ', animal

def predict_neural_push(real_dat, task_ix_trials, i_t, kg_decoder, keep_units, pred_data, day_mu):
    ix_ = np.ix_([3, 5], keep_units)
    kg_decoder = kg_decoder[ix_]
    pred_xy = []
    real_xy = []

    nt_pred_xy = []
    nt_real_xy = []

    for i in range(len(real_dat)):
        if task_ix_trials is not None:
            if task_ix_trials[i] == i_t:
                pred_xy.append(kg_decoder*(pred_data[i]+day_mu[np.newaxis, keep_units]).T)
                real_xy.append(kg_decoder*(real_dat[i]+day_mu[np.newaxis, keep_units]).T)
            else:
                nt_pred_xy.append(kg_decoder*(pred_data[i]+day_mu[np.newaxis, keep_units]).T) 
                nt_real_xy.append(kg_decoder*(real_dat[i]+day_mu[np.newaxis, keep_units]).T)
        else:
            pred_xy.append(kg_decoder*(pred_data[i]+day_mu[np.newaxis, keep_units]).T)
            real_xy.append(kg_decoder*(real_dat[i]+day_mu[np.newaxis, keep_units]).T)
    return pred_xy, real_xy, nt_pred_xy, nt_real_xy

### Methods to aggregate data into giant structures ###
def run_all_data_structs(animals=['jeev'], ndays = [4]): #, 'grom']):

    for ia, (animal, nday) in enumerate(zip(animals, ndays)):
        dat, xval, kg = get_data_struct(animal)
        d = dict(dat=dat, xval=xval, kg=kg)

        ### Run some checks ####
        check_x_vals(d, nday)

        if animal == 'grom':
            fn = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/grom2019_data_seg_9_6_19.pkl'
        elif animal == 'jeev':
            fn = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/jeev2019_data_seg_9_6_19.pkl'
        pickle.dump(d, open(fn, 'wb'))

def get_data_struct(animal, binsize_ms=100, pre_go_secs = 1.):
    ''' reviewed on 9/5/19, added neural push and important neurons'''

    data = {}
    xval = {}
    kg = {}
    vel_ix = {}
    targ_loc = {}
    neural_push = {}
    
    if animal == 'grom':
        input_type = co_obs_tuning_matrices.input_type
        datafiles = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/co_obs_file_dict.pkl'))

    elif animal == 'jeev':
        input_type = fk.task_filelist

    for i_d, day in enumerate(input_type):

        ###############################
        # Extract data from entire day: 
        ###############################

        BS = []
        BS_imp = []; 
        TSK = []
        TRG = []
        VEL = []
        NP = []
        CSTATE = []

        important_neurons = subspace_overlap.get_important_units(animal, i_d, est_type = 'est')

        for i_t, tsk in enumerate(day):
            for ii_tt, te_num in enumerate(tsk):

                bin_spk_nonz, targ_ix, trial_ix_all, KG, decoder_all, exclude_ix = fit_LDS.pull_data(te_num, animal,
                    pre_go=pre_go_secs, binsize_ms=binsize_ms, keep_units='all')


                ### Get the target per trial: 
                slm_targ_ix = [] 
                for iii in np.unique(trial_ix_all):
                    ix = np.nonzero(trial_ix_all==iii)[0]
                    slm_targ_ix.append(targ_ix[ix[0]+5]) #Use a later bin to get the target just in case

                # Which velocity bin are you in? 
                vel_bins = subspace_overlap.commands2bins(decoder_all, mag_boundaries, animal, i_d, vel_ix = [3, 5])

                if animal == 'grom':
                    hdf = datafiles[te_num, 'hdf']
                    name = hdf.split('/')[-1]
                    hdf = tables.openFile('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/'+name)

                    rew_ix = np.array([t[1] for it, t in enumerate(hdf.root.task_msgs[:]) if t[0]=='reward'])
                    drives_neurons_ix0 = 3
                    key = 'spike_counts'

                    ### Also get neural push? 
                    _, _, _, _, cursor_state = pa.extract_trials_all(hdf, rew_ix, neural_bins = binsize_ms,
                        drives_neurons_ix0=drives_neurons_ix0, hdf_key=key, keep_trials_sep=True,
                        reach_tm_is_hdf_cursor_pos=False, reach_tm_is_hdf_cursor_state=True, 
                        reach_tm_is_kg_vel=False, include_pre_go= pre_go_secs, **dict(kalman_gain=KG))

                elif animal == 'jeev':
                    _, _, _, _, _, cursor_state, _, _ = ppf_pa.get_jeev_trials_from_task_data(te_num, 
                        binsize=binsize_ms/1000., pre_go=pre_go_secs, include_pos = True)

                kg[i_d, i_t, te_num] = KG

                ### Go through all the trials: 
                for bi, bs in enumerate(bin_spk_nonz):

                    # If we want to keep this trial: 
                    if bi not in exclude_ix:
                        
                        ### add the neurons
                        BS.append(bs)
                        BS_imp.append(bs[:, important_neurons])

                        ### Ad teh task: 
                        TSK.append(i_t)

                        ### Add the target: 
                        ix = np.nonzero(trial_ix_all==bi)[0][0]+1
                        TRG.append(targ_ix[ix])

                        ### Get the discretized velocity bins: 
                        vels = vel_bins[bi]

                        # Task, Target, Radial, Mag
                        velix = np.hstack((np.zeros((len(vels), 1)) + i_t, np.zeros((len(vels), 1)) + slm_targ_ix[bi], vels))
                        VEL.append(velix)

                        # Cursor state: 
                        CSTATE.append(cursor_state[bi])

                        ### Add the neural push: 
                        NP.append(decoder_all[bi])
 
        ######################
        ### Divide up data ### 
        ######################
        zero_units = []
        mfr_units = []; 
        std_units = []; 

        zero_imp_units = []; 

        X = []; X_imp = []; 
        for i_t in range(2):
            tsk_ix = np.nonzero(np.hstack((TSK))==i_t)[0]
            data[i_d, i_t, 'bs'] = [BS[i] for i in tsk_ix]
            data[i_d, i_t, 'bs_imp'] = [BS_imp[i] for i in tsk_ix]
            data[i_d, i_t, 'tg'] = np.hstack((TRG))[tsk_ix]
            data[i_d, i_t, 'vl'] = [VEL[i] for i in tsk_ix]
            data[i_d, i_t, 'st'] = [CSTATE[i] for i in tsk_ix]
            data[i_d, i_t, 'np'] = [NP[i] for i in tsk_ix]

            ### Get the important units: 
            X.append(np.vstack(([BS[i] for i in tsk_ix])))
            X_imp.append(np.vstack(([BS_imp[i] for i in tsk_ix])))

            ### Number of trials per fold: 
            nperfold = int(np.floor(len(tsk_ix) / nfolds))
            ix_shuffle = np.random.permutation(len(tsk_ix))

            for i in range(nfolds):
                ix_test = ix_shuffle[(i*nperfold):((i+1)*nperfold)]
                ix_train = np.array([j for j in ix_shuffle if j not in ix_test])

                xval[i_d, i_t, i, 'test'] = ix_test
                xval[i_d, i_t, i, 'train'] = ix_train

                ##############################################
                ### Rm units that are zero in 80% sections ### 
                ##############################################
                bs_train_fold = np.sum(np.vstack(( data[i_d, i_t, 'bs'][j] for j in ix_train)), axis=0)
                bs_train_fold_imp = np.sum(np.vstack(( data[i_d, i_t, 'bs_imp'][j] for j in ix_train)), axis=0)

                zero_units.append(np.nonzero(bs_train_fold==0)[0])
                zero_imp_units.append(np.nonzero(bs_train_fold_imp == 0)[0])

        data[i_d, 'mfr'] = np.mean(np.vstack((X)), axis=0)
        data[i_d, 'mfr_imp'] = np.mean(np.vstack((X_imp)), axis=0)

        data[i_d, 'std'] = np.std(np.vstack((X)), axis=0)
        data[i_d, 'std_imp'] = np.std(np.vstack((X_imp)), axis=0)
        
        data[i_d, 'rm_units'] = np.unique(np.hstack((zero_units)))
        data[i_d, 'rm_units_imp'] = np.unique(np.hstack((zero_imp_units)))

    return data, xval, kg

def check_x_vals(dat_grom, ndays = 9):
    ''' test if xval / KG makes sense '''

    KG = []; 
    for i in range(ndays):
        for j in range(2):

            ### Check xvals
            for x in range(5):
                train = dat_grom['xval'][i, j, x, 'train']
                test = dat_grom['xval'][i, j, x, 'test']

                all_ = np.hstack(( train, test ))
                assert(len(all_) == len(np.unique(all_)))

            ### Check KGs
            kg = []
            for k in dat_grom['kg'].keys():
                if np.logical_and(k[0] == i, k[1] == j):
                    kg.append(dat_grom['kg'][k])

            for x in range(len(kg)):
                for y in range(x, len(kg)):
                    assert np.allclose(kg[x], kg[y]) 
            if j == 0:
                KG.append(kg[0])

            ### Check data lengths make sense: 
            nneur = dat_grom['dat'][i, j, 'bs'][0].shape[1]

            for itrl, trl in enumerate(dat_grom['dat'][i, j, 'bs']):
                assert(trl.shape[1] == nneur)
                assert(trl.shape[0] == dat_grom['dat'][i, j, 'vl'][itrl].shape[0])
                assert(trl.shape[0] == dat_grom['dat'][i, j, 'st'][itrl].shape[0])
                assert(trl.shape[0] == dat_grom['dat'][i, j, 'np'][itrl].shape[0])

    for x in range(len(KG)):
        for y in range(x+1, len(KG)):
            if KG[x].shape[1] == KG[y].shape[1]:
                if np.allclose(KG[x], KG[y]):
                    raise Exception

### Plotting everything!###
def plot_all_R2(animal, input_type, only_important=False):
    ''' 
    Method to plot R2 of all models that were fit above in main: 
    Plots: 1) R2 on own task
           2) other task
           3) behavioral vx/vy
    '''

    # if animal == 'grom':
    #     input_type = co_obs_tuning_matrices.input_type
    fname_pref = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/'
    #     imp = pickle.load(open('/Volumes/TimeMachineBackups/grom2016/co_obs_neuron_importance_dict.pkl'))

    # elif animal == 'jeev':
    #     input_type = fk.task_filelist
    #     fname_pref = '/Volumes/TimeMachineBackups/jeev2013/'
    #     imp = pickle.load(open('/Volumes/TimeMachineBackups/jeev2013/jeev_co_obs_neuron_importance_dict.pkl'))

    R2_neural_LDS = dict(fit_co_own=[], fit_co_val_obs=[], fit_obs_own=[], fit_obs_val_co=[])
    R2_neural_FA = dict(fit_co_own=[], fit_co_val_obs=[], fit_obs_own=[], fit_obs_val_co=[])
    R2_neural_tuning = dict(fit_co_own=[], fit_co_val_obs=[], fit_obs_own=[], fit_obs_val_co=[])
    R2_neural_LPF150 = dict(fit_co_own=[], fit_obs_own=[])
    R2_neural_LPF300 = dict(fit_co_own=[], fit_obs_own=[])
    R2_neural_LPF450 = dict(fit_co_own=[], fit_obs_own=[])
    R2_neural_LPF600 = dict(fit_co_own=[], fit_obs_own=[])

    for i_d in range(len(input_type)):
        data = pickle.load(open(fname_pref+'test_big_data_'+str(i_d)+'_8_4_17.pkl'))
        egte = input_type[i_d][0][0]
        if only_important is False:
            #imp_ix = np.arange(len(imp[egte]))
            imp_ix = np.arange(len(data['data_LDS'][0, 0, 'R2_filt_tsk']))

        else:
            egte = input_type[i_d][0][0]
            imp_ix = np.nonzero(imp[egte] >= only_important)[0]

        imp_ix = np.arange(len(data['data_LDS'][0, 0, 'R2_filt_tsk']))

        # R2 Neural -- LDS
        R2_neural_LDS['fit_co_own'].append(np.mean(np.array([ data['data_LDS'][f, 0, 'R2_filt_tsk'][imp_ix] for f in range(nfolds)]), axis=1))
        R2_neural_LDS['fit_co_val_obs'].append(np.mean(np.array([ data['data_LDS'][f, 0, 'R2_filt_nontsk'][imp_ix] for f in range(nfolds)]), axis=1))
        R2_neural_LDS['fit_obs_own'].append(np.mean(np.array([ data['data_LDS'][f, 1, 'R2_filt_tsk'][imp_ix] for f in range(nfolds)]), axis=1))
        R2_neural_LDS['fit_obs_val_co'].append(np.mean(np.array([ data['data_LDS'][f, 1, 'R2_filt_nontsk'][imp_ix] for f in range(nfolds)]), axis=1))

        # R2 Neural -- FA: 
        R2_neural_FA['fit_co_own'].append(np.mean(np.array([ data['data_FA'][f, 0, 'R2_tsk'][imp_ix] for f in range(nfolds)]), axis=1))
        R2_neural_FA['fit_co_val_obs'].append(np.mean(np.array([ data['data_FA'][f, 0, 'R2_nontsk'][imp_ix] for f in range(nfolds)]), axis=1))
        R2_neural_FA['fit_obs_own'].append(np.mean(np.array([ data['data_FA'][f, 1, 'R2_tsk'][imp_ix] for f in range(nfolds)]), axis=1))
        R2_neural_FA['fit_obs_val_co'].append(np.mean(np.array([ data['data_FA'][f, 1, 'R2_nontsk'][imp_ix] for f in range(nfolds)]), axis=1))

        # R2 Neural -- Tuning: 
        R2_neural_tuning['fit_co_own'].append(np.mean(np.array([ data['data_tuning'][f, 0, 'R2_tsk'][imp_ix] for f in range(nfolds)]), axis=1))
        R2_neural_tuning['fit_co_val_obs'].append(np.mean(np.array([ data['data_tuning'][f, 0, 'R2_nontsk'][imp_ix] for f in range(nfolds)]), axis=1))
        R2_neural_tuning['fit_obs_own'].append(np.mean(np.array([ data['data_tuning'][f, 1, 'R2_tsk'][imp_ix] for f in range(nfolds)]), axis=1))
        R2_neural_tuning['fit_obs_val_co'].append(np.mean(np.array([ data['data_tuning'][f, 1, 'R2_nontsk'][imp_ix] for f in range(nfolds)]), axis=1))

        # R2 Neural -- LPF / Predicted: 
        R2_neural_LPF150['fit_co_own'].append(np.mean(np.array([data['data_LPF'][f, 0, 150., 'R2_pred'][imp_ix] for f in range(nfolds)]), axis=1))
        R2_neural_LPF150['fit_obs_own'].append(np.mean(np.array([data['data_LPF'][f, 1, 150., 'R2_pred'][imp_ix] for f in range(nfolds)]), axis=1))

        R2_neural_LPF300['fit_co_own'].append(np.mean(np.array([data['data_LPF'][f, 0, 300., 'R2_pred'][imp_ix] for f in range(nfolds)]), axis=1))
        R2_neural_LPF300['fit_obs_own'].append(np.mean(np.array([data['data_LPF'][f, 1, 300., 'R2_pred'][imp_ix] for f in range(nfolds)]), axis=1))

        R2_neural_LPF450['fit_co_own'].append(np.mean(np.array([data['data_LPF'][f, 0, 450., 'R2_pred'][imp_ix] for f in range(nfolds)]), axis=1))
        R2_neural_LPF450['fit_obs_own'].append(np.mean(np.array([data['data_LPF'][f, 1, 450., 'R2_pred'][imp_ix] for f in range(nfolds)]), axis=1))

        R2_neural_LPF600['fit_co_own'].append(np.mean(np.array([data['data_LPF'][f, 0, 600., 'R2_pred'][imp_ix] for f in range(nfolds)]), axis=1))
        R2_neural_LPF600['fit_obs_own'].append(np.mean(np.array([data['data_LPF'][f, 1, 600., 'R2_pred'][imp_ix] for f in range(nfolds)]), axis=1))

    f, ax = plt.subplots()
    color = ['green', 'lightgreen', 'blue', 'lightblue']

    for i_m, model in enumerate([R2_neural_LDS, R2_neural_FA, R2_neural_tuning, R2_neural_LPF150, R2_neural_LPF300, R2_neural_LPF450,R2_neural_LPF600]):
        for m, met in enumerate(['fit_co_own', 'fit_obs_val_co', 'fit_obs_own', 'fit_co_val_obs']):
            try:
                x = np.hstack((model[met]))
                x = x[np.nonzero(np.abs(x) != np.inf)]
                ax.bar(i_m + .15*(m), np.nanmean(x), width=.2, color=color[m])
            except:
                print 'skip model: ', i_m, ' variable', met

def plot_all_R2_behav(animal, only_important=False):
    ''' 
    Method to plot R2 of all models that were fit above in main: 
    Plots: 1) R2 on own task
           2) other task
           3) behavioral vx/vy
    '''

    if animal == 'grom':
        input_type = co_obs_tuning_matrices.input_type
        fname_pref = '/Volumes/TimeMachineBackups/grom2016/'
        imp = pickle.load(open('/Volumes/TimeMachineBackups/grom2016/co_obs_neuron_importance_dict.pkl'))

    elif animal == 'jeev':
        input_type = fk.task_filelist
        fname_pref = '/Volumes/TimeMachineBackups/jeev2013/'
        imp = pickle.load(open('/Volumes/TimeMachineBackups/jeev2013/jeev_co_obs_neuron_importance_dict.pkl'))

    R2_beh_LDS = dict()
    R2_beh_FA = dict()
    R2_beh_tun = dict()
    R2_beh_lpf150 = dict()
    R2_beh_lpf300 = dict() 
    R2_beh_lpf450 = dict() 
    R2_beh_lpf600 = dict() 
    for dxt in [R2_beh_LDS, R2_beh_FA, R2_beh_tun, R2_beh_lpf150, R2_beh_lpf300, R2_beh_lpf450,R2_beh_lpf600]:
        dxt['co'] = []
        dxt['obs'] = []

    for i_d in range(len(input_type)):
        data = pickle.load(open(fname_pref+'test_big_data_'+str(i_d)+'_8_4_17.pkl'))

        # R2 Neural -- LDS
        for ix, nm in enumerate(['co', 'obs']):
            R2_beh_LDS[nm].append( np.array([ data['data_LDS'][f, ix, 'R2_filt_vx_vy_tsk'] for f in range(nfolds)]) )
            R2_beh_FA[nm].append( np.array([ data['data_FA'][f, ix, 'R2_pred_vx_vy_tsk'] for f in range(nfolds)]) )
            R2_beh_tun[nm].append( np.array([ data['data_tuning'][f, ix, 'R2_pred_vx_vy_tsk'] for f in range(nfolds)]) )
            R2_beh_lpf150[nm].append( np.array([ data['data_LPF'][f, ix, 150., 'R2_pred_vx_vy_tsk'] for f in range(nfolds)]) )
            R2_beh_lpf300[nm].append( np.array([ data['data_LPF'][f, ix, 300., 'R2_pred_vx_vy_tsk'] for f in range(nfolds)]) )
            R2_beh_lpf450[nm].append( np.array([ data['data_LPF'][f, ix, 450., 'R2_pred_vx_vy_tsk'] for f in range(nfolds)]) )
            R2_beh_lpf600[nm].append( np.array([ data['data_LPF'][f, ix, 600., 'R2_pred_vx_vy_tsk'] for f in range(nfolds)]) )
    f, ax = plt.subplots()
    color = ['green', 'blue']

    for i_m, model in enumerate([R2_beh_LDS, R2_beh_FA, R2_beh_lpf150, R2_beh_lpf300, R2_beh_lpf450, R2_beh_lpf600]):
        for m, met in enumerate(['co', 'obs']):
            try:
                x = np.hstack((model[met]))
                #x = x[np.nonzero(np.abs(x) != np.inf)]
                ax.bar(i_m + .15*(m), np.mean(x), width=.2, color=color[m])
            except:
                print 'skip model: ', i_m, ' variable', met

def plot_figure_6(save_figs=False):
    ''' 
    First plot: R2 of LDS filt vs. FA vs. tuning filt (result R2 LDS Filt >> R2 FA >> R2 Tuning)
    Second plot: R2 filt vs. LPF filt (and dynamics from 'plot_figure_5'), R2 filt beh vs. LPF filt beh
    Third plot: LDS main shared overlap x-task vs. w-in task, possible repertoire overlap too
    Fourth plot: R2 of non-task better for LDS than FA (neural and beh)

    '''

    ##################
    ### FIRST PLOT ###
    ##################

    ysim = dict()
    ysim[0] = [[.23, .24, .13], ['***', '***', '*']]
    ysim[1] = [[.43, .45, .28], ['***', '***', '***']]

    for ia, animal in enumerate(['grom', 'jeev']):
        f, ax = plt.subplots()
        f.set_figheight(5)
        f.set_figwidth(6)
        co_dyn = [np.mean(co) for co in get_metric(animal, 'data_LDS', 'R2_filt_tsk', 0, True)]
        obs_dyn = [np.mean(obs) for obs in get_metric(animal, 'data_LDS', 'R2_filt_tsk', 1, True)]
        dyn = co_dyn + obs_dyn

        # Mean of data across folds: 
        co_tun = [np.mean(co) for co in get_metric(animal, 'data_tuning', 'R2_tsk', 0, True)]
        obs_tun = [np.mean(obs) for obs in get_metric(animal, 'data_tuning', 'R2_tsk', 1, True)]
        
        co_fa = [np.mean(co) for co in get_metric(animal, 'data_FA', 'R2_tsk', 0, True)]
        obs_fa = [np.mean(co) for co in get_metric(animal, 'data_FA', 'R2_tsk', 1, True)]

        tun = co_tun + obs_tun
        fa = co_fa + obs_fa

        # Mean across folds: 
        ax.bar(0, np.mean(dyn), width=1.,edgecolor='k', linewidth=4., color='w')
        ax.errorbar(0+.5, np.mean(dyn), np.std(dyn)/np.sqrt(len(dyn)), marker='.', color='k')

        ax.bar(1, np.mean(fa), width=1., edgecolor='k', linewidth=4., color='w')
        ax.errorbar(1.5, np.mean(fa), np.std(fa)/np.sqrt(len(fa)), marker='.', color='k')

        ax.bar(2, np.mean(tun), width=1.,edgecolor='k', linewidth=4., color='w')
        ax.errorbar(2.5, np.mean(tun), np.std(tun)/np.sqrt(len(tun)), marker='.', color='k')

        # scipy.stats.f_oneway(dyn, fa, tun)
        # from statsmodels.stats.multicomp import MultiComparison
        # grp = np.zeros((len(dyn)*3, ))
        # grp[len(dyn):2*len(dyn)] = 1
        # grp[2*len(dyn):] = 3
        # mc = MultiComparison(np.hstack((dyn, fa, tun)), grp)
        # print mc.tukeyhsd(0.001)
        # print mc.tukeyhsd(0.01)
        # print mc.tukeyhsd(0.05)
        print 'n : ', len(dyn), len(fa), len(lpf_dat)
        print scipy.stats.kruskal(dyn, fa, lpf_dat)
        print 'mann whitney testss: '
        print scipy.stats.mannwhitneyu(dyn, fa).pvalue * 2
        print scipy.stats.mannwhitneyu(dyn, lpf_dat).pvalue * 2


        ax.set_ylabel('R2 Filtered')
        ax.set_xticks(np.arange(.5, 3., 1.))
        ax.set_xticklabels(['LDS, Filt', 'FA', 'Linear Tuning'])
        ax.set_xlim([-.1, 3.1])

        s = animal[0]
        ax.set_title('Monkey '+s.capitalize())
        
        ax.plot([.5, 1.5], [ysim[ia][0][0], ysim[ia][0][0]], 'k-')
        ax.plot([.5, 2.5], [ysim[ia][0][1], ysim[ia][0][1]], 'k-')
        ax.plot([1.5, 2.5], [ysim[ia][0][2], ysim[ia][0][2]], 'k-')
        
        ax.text(1, ysim[ia][0][0], ysim[ia][1][0])
        ax.text(1.5, ysim[ia][0][1], ysim[ia][1][1])
        ax.text(2, ysim[ia][0][2], ysim[ia][1][2])
        ax.set_ylim([0., np.max(ysim[ia][0])+.03])
        if save_figs:
            f.savefig(save_pre+'monkey_'+s+'LDS_vs_FA_vs_tuning_R2_filt.svg', transparent=True)

    ###################
    ### SECOND PLOT ###
    ###################

    ###################
    ### Third PLOT ###
    ###################

    ###################
    ### Fourth PLOT ###
    ###################

def plot_figure_5(old=False, save_figs=False):
    from matplotlib import rcParams
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']

    save_pre = '/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/BMI_co_obs_paper/figures/data_figs/'
    if old:
        ################
        # Dynamics ratio
        ################
        f, ax = plt.subplots()
        f.set_figheight(5)
        f.set_figwidth(6)
        for ia, animal in enumerate(['grom', 'jeev']):
            co_dyn = get_metric(animal, 'data_LDS', 'dynamics_ratio_mn', 0)
            obs_dyn = get_metric(animal,'data_LDS', 'dynamics_ratio_mn', 1)

            # Mean across folds: 
            co_dyn = [np.mean(cod[:, 0]) for cod in co_dyn]
            obs_dyn = [np.mean(cod[:, 1]) for cod in obs_dyn]

            ax.bar(ia, np.mean(np.vstack((co_dyn, obs_dyn))), width=1.,edgecolor='k', linewidth=4., color='w')
        ax.set_ylabel('Dynamics Ratio', )
        ax.set_xticks(np.arange(.5, 2., 1.))
        ax.set_xticklabels(['Monkey G', 'Monkey J'])
        ax.set_xlim([-.1, 2.1])
        if save_figs:
            f.savefig(save_pre+'dyn_ratio_MG_MJ.svg', transparent=True)


        ysim = [.23, .42]

        #############################
        # LDS vs. Tuning: Neural R2 #
        #############################

        for ia, animal in enumerate(['grom', 'jeev']):
            f, ax = plt.subplots()
            f.set_figheight(5)
            f.set_figwidth(6)
            co_dyn = [np.mean(co) for co in get_metric(animal, 'data_LDS', 'R2_smooth_tsk', 0, True)]
            obs_dyn = [np.mean(obs) for obs in get_metric(animal, 'data_LDS', 'R2_smooth_tsk', 1, True)]
            dyn = co_dyn + obs_dyn

            co_tun = [np.mean(co) for co in get_metric(animal, 'data_tuning', 'R2_tsk', 0, True)]
            obs_tun = [np.mean(obs) for obs in get_metric(animal, 'data_tuning', 'R2_tsk', 1, True)]
            tun = co_tun + obs_tun
            
            # Mean across folds: 
            ax.bar(0, np.mean(dyn), width=1.,edgecolor='k', linewidth=4., color='w')
            ax.errorbar(0+.5, np.mean(dyn), np.std(dyn)/np.sqrt(len(dyn)), marker='.', color='k')
            
            ax.bar(0+1, np.mean(tun), width=1.,edgecolor='k', linewidth=4., color='w')
            ax.errorbar(0+1.5, np.mean(tun), np.std(tun)/np.sqrt(len(tun)), marker='.', color='k')

            #print scipy.stats.ttest_rel(dyn, tun), len(dyn), len(tun)
            print 'kruskal; ', animal
            print scipy.stats.kruskal(dyn, tun), len(dyn), len(tun)
            
            # Ttest_relResult(statistic=18.143496500121113, pvalue=1.4614143318891041e-12)
            # Ttest_relResult(statistic=13.438837299079841, pvalue=2.9637659237019646e-06)

            ax.set_ylabel('R2 LDS, smoothed vs. R2 Tuning (-400 ms : 400 ms) w/ position')
            ax.set_xticks(np.arange(.5, 2., 1.))
            ax.set_xticklabels(['LDS, Smooth', 'Linear Tuning'])
            ax.set_xlim([-.1, 2.1])

            s = animal[0]
            ax.set_title('Monkey '+s.capitalize())
            ax.plot([.5, 1.5], [ysim[ia], ysim[ia]], 'k-')
            ax.text(1, ysim[ia], '***')    
            if save_figs:
                f.savefig(save_pre+'monkey_'+s+'LDS_vs_tuning_R2.svg', transparent=True)

        ###################################################
        # R2 beh FA vs. R2 beh filt LDS vs. R2 beh tuning #
        ###################################################
        for ia, animal in enumerate(['grom', 'jeev']):
            f, ax = plt.subplots()
            f.set_figheight(5)
            f.set_figwidth(6)
            print 'WHAT WE WANT: ', animal
            co_dyn = [np.mean(co) for co in get_metric(animal, 'data_LDS', 'R2_filt_vx_vy_tsk', 0, True)]
            obs_dyn = [np.mean(obs) for obs in get_metric(animal, 'data_LDS', 'R2_filt_vx_vy_tsk', 1, True)]
            dyn = co_dyn + obs_dyn

            co_FA = [np.mean(co) for co in get_metric(animal, 'data_FA', 'R2_pred_vx_vy_tsk', 0, True)]
            obs_FA = [np.mean(obs) for obs in get_metric(animal, 'data_FA', 'R2_pred_vx_vy_tsk', 1, True)]
            fa = co_FA + obs_FA
        
            lpf = 300.
            co_lpf = [np.mean(co) for co in get_metric(animal, 'data_LPF', 'R2_smooth_vx_vy_tsk', 0, True, LPF=lpf)]
            obs_lpf = [np.mean(obs) for obs in get_metric(animal, 'data_LPF', 'R2_smooth_vx_vy_tsk', 1, True, LPF=lpf)]
            lpf_dat = co_lpf + obs_lpf
        
            # co_tun = [np.mean(co) for co in get_metric(animal, 'data_tuning', 'R2_pred_vx_vy_tsk', 0, True)]
            # obs_tun = [np.mean(obs) for obs in get_metric(animal, 'data_tuning', 'R2_pred_vx_vy_tsk', 1, True)]
            # tun = co_tun + obs_tun
            
            # Mean across folds: 
            ax.bar(0, np.mean(dyn), width=1.,edgecolor='k', linewidth=4., color='w')
            ax.errorbar(0+.5, np.mean(dyn), np.std(dyn)/np.sqrt(len(dyn)), marker='.', color='k')
            
            ax.bar(1, np.mean(fa), width=1.,edgecolor='k', linewidth=4., color='w')
            ax.errorbar(1+.5, np.mean(fa), np.std(fa)/np.sqrt(len(fa)), marker='.', color='k')
            
            ax.bar(2, np.mean(lpf_dat), width=1.,edgecolor='k', linewidth=4., color='w')
            ax.errorbar(2+.5, np.mean(lpf_dat), np.std(lpf_dat)/np.sqrt(len(lpf_dat)), marker='.', color='k')

            # scipy.stats.f_oneway(dyn, fa, lpf_dat)
            # print 'ANOVA + TUKEY: LDS VS. FA vs. LPF300, BEH: ', animal
            # from statsmodels.stats.multicomp import MultiComparison
            # grp = np.zeros((len(dyn)*3, ))
            # grp[len(dyn):2*len(dyn)] = 1
            # grp[2*len(dyn):] = 3
            # mc = MultiComparison(np.hstack((dyn, fa, lpf_dat)), grp)
            # print mc.tukeyhsd(0.001)
            # print mc.tukeyhsd(0.01)
            # print mc.tukeyhsd(0.05)
            print 'n : ', len(dyn), len(fa), len(lpf_dat)
            print scipy.stats.kruskal(dyn, fa, lpf_dat)
            print 'mann whitney testss: '
            print scipy.stats.mannwhitneyu(dyn, fa).pvalue * 2
            print scipy.stats.mannwhitneyu(dyn, lpf_dat).pvalue * 2


            ax.set_ylabel('R2 of Vx,Vy')
            ax.set_xticks(np.arange(.5, 3., 1.))
            ax.set_xticklabels(['LDS, Filt', 'Factor Analysis', 'LPF, 300ms'], rotation=45)
            ax.set_xlim([-.1, 3.1])

            s = animal[0]
            ax.set_title('Monkey '+s.capitalize())
            
            if animal == 'jeev':
                ax.plot([.5, 1.4], [1., 1.], 'k-')
                ax.text(1., 1, '*')   
            
                ax.plot([1.6, 2.5], [1., 1.], 'k-')
                ax.text(2., 1, '*')   

            elif animal == 'grom':
                ax.plot([.5, 2.5], [1.2, 1.2], 'k-')
                ax.text(1.5, 1.2, '***')               

                ax.plot([.5, 1.4], [1.1, 1.1], 'k-')
                ax.text(1., 1.1, '***')        

                ax.plot([1.6, 2.5], [1.1, 1.1], 'k-')
                ax.text(2., 1.1, '***')        
            ax.set_ylim([0., 1.25])
            if save_figs:
                f.savefig(save_pre+'monkey_'+s+'LDS_vs_FA_vs_LPF_R2_beh.svg', transparent=True)

    ##################################
    #R2 LDS filt vs. FA vs. LPF smooth neural
    ##################################
    for ia, animal in enumerate(['grom', 'jeev']):
        f, ax = plt.subplots()
        f.set_figheight(5)
        f.set_figwidth(6)

        co_dyn = [np.mean(co) for co in get_metric(animal, 'data_LDS', 'R2_filt_tsk', 0, True)]
        obs_dyn = [np.mean(obs) for obs in get_metric(animal, 'data_LDS', 'R2_filt_tsk', 1, True)]
        dyn = co_dyn + obs_dyn

        co_FA = [np.mean(co) for co in get_metric(animal, 'data_FA', 'R2_tsk', 0, True)]
        obs_FA = [np.mean(obs) for obs in get_metric(animal, 'data_FA', 'R2_tsk', 1, True)]
        fa = co_FA + obs_FA

        ax.bar(0, np.mean(dyn), width=1.,edgecolor='k', linewidth=4., color='w')
        ax.errorbar(.5, np.mean(dyn), np.std(dyn)/np.sqrt(len(dyn)), marker='.', color='k')

        ax.bar(1, np.mean(fa), width=1.,edgecolor='k', linewidth=4., color='w')
        ax.errorbar(1.5, np.mean(fa), np.std(fa)/np.sqrt(len(fa)), marker='.', color='k')

        for il, lpf in enumerate([300.]):
            co_tun = [np.mean(co) for co in get_metric(animal, 'data_LPF', 'R2_smooth', 0, True, LPF=lpf)]
            obs_tun = [np.mean(obs) for obs in get_metric(animal, 'data_LPF', 'R2_smooth', 1, True, LPF=lpf)]
            tun = co_tun + obs_tun
        
            ax.bar(il+2, np.mean(tun), width=1.,edgecolor='k', linewidth=4., color='w')
            ax.errorbar(il+2.5, np.mean(tun), np.std(tun)/np.sqrt(len(tun)), marker='.', color='k')

        print 'ANOVA + TUKEY: LDS VS. FA vs. LPF300 NEURAL, ', animal
        from statsmodels.stats.multicomp import MultiComparison
        grp = np.zeros((len(dyn)*3, ))
        grp[len(dyn):2*len(dyn)] = 1
        grp[2*len(dyn):] = 3
        mc = MultiComparison(np.hstack((dyn, fa, tun)), grp)
        print mc.tukeyhsd(0.001)
        print mc.tukeyhsd(0.01)
        print mc.tukeyhsd(0.05)

        ax.set_ylabel('R2')
        ax.set_xticks(np.arange(.5, 3., 1.))
        ax.set_xticklabels(['LDS, Filt', 'FA', 'LPF 300'])
        ax.set_xlim([-.1, 3.1])

        s = animal[0]
        ax.set_title('Monkey '+s.capitalize())

        if animal == 'jeev':
            ax.plot([0.5, 1.5], [.43, .43], 'k-')
            ax.text(1.0, .43, '***')
            ax.plot([1.5, 2.5], [.38, .38], 'k-')
            ax.text(2.0, .38, '***')
        elif animal == 'grom':
            ax.plot([0.5, 1.5], [.243, .243], 'k-')
            ax.text(1.0, .243, '***')
            ax.plot([1.5, 2.5], [.28, .28], 'k-')
            ax.text(2.0, .28, '***')
            ax.plot([.5, 2.5], [.3, .3], 'k-')
            ax.text(1.5, .3, '*')
            ax.set_ylim([0, .32])

        if save_figs:
            f.savefig(save_pre+'monkey_'+s+'LDS_filt_vs_FA_vs_LPF_filt.svg', transparent=True)

        ##################################        
        #R2 LDS beh filt vs. FA beh vs. LPF beh pred
        ##################################
        f, ax = plt.subplots()
        f.set_figheight(5)
        f.set_figwidth(6)

        co_dyn = [np.mean(co) for co in get_metric(animal, 'data_LDS', 'R2_pred_vx_vy_tsk', 0, True)]
        obs_dyn = [np.mean(obs) for obs in get_metric(animal, 'data_LDS', 'R2_pred_vx_vy_tsk', 1, True)]
        dyn = co_dyn + obs_dyn

        ax.bar(0, np.mean(dyn), width=1.,edgecolor='k', linewidth=4., color='w')
        ax.errorbar(.5, np.mean(dyn), np.std(dyn)/np.sqrt(len(dyn)), marker='.', color='k')

        for il, lpf in enumerate([300., 450., 600.]):
            co_tun = [np.mean(co) for co in get_metric(animal, 'data_LPF', 'R2_smooth_vx_vy_tsk', 0, True, LPF=lpf)]
            obs_tun = [np.mean(obs) for obs in get_metric(animal, 'data_LPF', 'R2_smooth_vx_vy_tsk', 1, True, LPF=lpf)]
            tun = co_tun + obs_tun
        
            ax.bar(il+1, np.mean(tun), width=1.,edgecolor='k', linewidth=4., color='w')
            ax.errorbar(il+1.5, np.mean(tun), np.std(tun)/np.sqrt(len(tun)), marker='.', color='k')

        ax.set_ylabel('R2 Beh LDS, pred vs. R2 Beh LPF')
        ax.set_xticks(np.arange(.5, 4., 1.))
        ax.set_xticklabels(['LDS, Pred', 'LPF 300', 'LPF 450', 'LPF 600'])
        ax.set_xlim([-.1, 4.1])

        s = animal[0]
        ax.set_title('Monkey '+s.capitalize())
        if save_figs:
            f.savefig(save_pre+'monkey_'+s+'LDS_filt_vs_LPF_filt_BEH.svg', transparent=True)

def plot_r2_win_task_vs_xtask(save_figs=False):
    from collections import OrderedDict

    save_pre = '/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/BMI_co_obs_paper/figures/data_figs/'
    for ia, animal in enumerate(['grom', 'jeev']):

        dyn = OrderedDict()
        for r2 in ['R2_filt_tsk', 'R2_filt_nontsk', 'R2_filt_vx_vy_tsk', 'R2_filt_vx_vy_nottsk']:
            co_dyn = [np.mean(co) for co in get_metric(animal, 'data_LDS', r2, 0, True)]
            obs_dyn = [np.mean(obs) for obs in get_metric(animal, 'data_LDS', r2, 1, True)]
            task_beh_dyn = co_dyn + obs_dyn
            dyn[r2] = task_beh_dyn
        
        fa = OrderedDict()
        for r2 in ['R2_tsk', 'R2_nontsk', 'R2_pred_vx_vy_tsk', 'R2_pred_vx_vy_nottsk']:
            co_fa = [np.mean(co) for co in get_metric(animal, 'data_FA', r2, 0, True)]
            obs_fa = [np.mean(obs) for obs in get_metric(animal, 'data_FA', r2, 1, True)]
            fa[r2] = co_fa + obs_fa

        tun = OrderedDict()
        for r2 in ['R2_tsk', 'R2_nontsk', 'R2_pred_vx_vy_tsk', 'R2_pred_vx_vy_nottsk']:
            co_tun = [np.mean(co) for co in get_metric(animal, 'data_tuning', r2, 0, True)]
            obs_tun = [np.mean(obs) for obs in get_metric(animal, 'data_tuning', r2, 1, True)]
            tun[r2] = co_tun + obs_tun
        
        # Neural data: 
        f, ax = plt.subplots()
        f.set_figheight(5)
        f.set_figwidth(6)

        f2, ax2 = plt.subplots()
        f2.set_figheight(5)
        f2.set_figwidth(6)


        dtsk_notsk = []
        dbeh = []

        for im, model in enumerate([dyn, fa]):#, tun]):
            keys = model.keys()
            
            # Task, neural: 
            ax.bar(im, np.mean(model[keys[0]]), width=.4, edgecolor='k', linewidth=4., color='w')
            ax.errorbar(im+.2, np.mean(model[keys[0]]), np.std(model[keys[0]])/np.sqrt(len(model[keys[0]])), marker='.', color='k')

            ax.bar(im+.4, np.mean(model[keys[1]]), width=.4,edgecolor='k', linewidth=4., color='w', hatch='/')
            ax.errorbar(im+.4+.2, np.mean(model[keys[1]]), np.std(model[keys[1]])/np.sqrt(len(model[keys[1]])), marker='.', color='k')

            dtsk_notsk.append(np.array(model[keys[0]]) - np.array(model[keys[1]]))

            # Task, Beh: 
            if im < 2: # Skip plotting tuning: 
                ax2.bar(im, np.mean(model[keys[2]]), width=.4,edgecolor='k', linewidth=4., color='w')
                ax2.errorbar(im+.2, np.mean(model[keys[2]]), np.std(model[keys[2]])/np.sqrt(len(model[keys[2]])), marker='.', color='k')

                ax2.bar(im+.4, np.mean(model[keys[3]]), width=.4,edgecolor='k', linewidth=4., color='w', hatch='/')
                ax2.errorbar(im+.4+.2, np.mean(model[keys[3]]), np.std(model[keys[3]])/np.sqrt(len(model[keys[3]])), marker='.', color='k')
                dbeh.append(np.array(model[keys[2]]) - np.array(model[keys[3]]))


        ax.set_title('Monkey '+animal[0].capitalize())
        ax2.set_title('Monkey '+animal[0].capitalize())
        ax.set_ylabel('R2')
        ax2.set_ylabel('R2 of Vx,Vy')
        
        ax.set_xticks(np.arange(.4, 2, 1.))
        ax.set_xticklabels(['LDS', 'FA'])#, 'Tuning'])
        
        ax2.set_xticks(np.arange(.4, 2, 1.))
        ax2.set_xticklabels(['LDS', 'FA'])

        if ia == 0:
            #ax.plot([.2, .6], [.25, .25], 'k-')
            #ax.plot([1.2, 1.6], [.14, .14], 'k-')
            ax.plot([.4, 1.4], [.27, .27], 'k-')
            ax.text(.9, .27, '**')
            ax.set_ylim([-.2, .31])
            
            #ax2.plot([.2, .6], [.9, .9], 'k-')
            #ax2.plot([1.2, 1.6], [.65, .65], 'k-')
            ax2.plot([.4, 1.4], [.92, .92], 'k-')
            ax2.text(.9, .92, '***')
            ax2.set_ylim([0, .96])

        elif ia == 1:
            #ax.plot([.2, .6], [.44, .44], 'k-')
            #ax.plot([1.2, 1.6], [.3, .3], 'k-')
            ax.text(.9, .46, '**')
            ax.plot([.4, 1.4], [.46, .46], 'k-')
            ax.set_ylim([0, .50])

            #ax2.plot([.2, .6], [.92, .92], 'k-')
            #ax2.plot([1.2, 1.6], [.8, .8], 'k-')            
            ax2.plot([.4, 1.4], [.94, .94], 'k-')
            ax2.text(.9, .94, '***')
            ax2.set_ylim([0, .98])
        ax.set_xlim([-.1, 1.9])
        ax2.set_xlim([-.1, 1.9])
        
        ### STATS ### 
        print 'STATS: animal ', animal
        #print 'TTEST neural: ', scipy.stats.ttest_rel(dtsk_notsk[0], dtsk_notsk[1])
        print ' neural kruskal: ', scipy.stats.kruskal(dtsk_notsk[0], dtsk_notsk[1])
        print ' n = ', len(dtsk_notsk[0])


        ### TTEST ###
        #print 'TTEST beh: ', scipy.stats.ttest_rel(dbeh[0], dbeh[1])
        print ' beh kruskal: ', scipy.stats.kruskal(dbeh[0], dbeh[1])
        print ' n = ', len(dbeh[0])

        if save_figs:
            f.savefig(save_pre+'monkey_'+animal[0]+'LDS_vs_FA_task_vs_nontask.svg', transparent=True)
            f2.savefig(save_pre+'monkey_'+animal[0]+'LDS_vs_FA_task_vs_nontask_behavior.svg', transparent=True)

def plot_main_dim_FA_and_LDS():
    save_pre = '/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/BMI_co_obs_paper/figures/data_figs/'
    f, ax = plt.subplots()
    f.set_figheight(5)
    f.set_figwidth(6)

    for ia, animal in enumerate(['grom', 'jeev']):
        f2, ax2 = plt.subplots()
        r = 'LDS_main_shared'
        co_dyn = [np.mean(co) for co in get_metric(animal, 'data_LDS', r, 0, False)]
        obs_dyn = [np.mean(obs) for obs in get_metric(animal, 'data_LDS', r, 1, False)]
        LDS = co_dyn + obs_dyn
        ax2.bar(0, np.mean(co_dyn))
        ax2.bar(1, np.mean(obs_dyn))


        r = 'FA_main_shared'
        co_dyn = [np.mean(co) for co in get_metric(animal, 'data_FA', r, 0, False)]
        obs_dyn = [np.mean(obs) for obs in get_metric(animal, 'data_FA', r, 1, False)]
        FA = co_dyn + obs_dyn

        ax.bar(0+ia, np.mean(LDS), width=.4,edgecolor='k', linewidth=4., color='w')
        ax.errorbar(.2+ia, np.mean(LDS), np.std(LDS)/np.sqrt(len(LDS)), marker='.', color='k')

        ax.bar(0.4+ia, np.mean(FA), width=.4,edgecolor='k', linewidth=4., color='w')
        ax.errorbar(.6+ia, np.mean(FA), np.std(FA)/np.sqrt(len(FA)), marker='.', color='k')

        print animal, ' ttest: '
        print scipy.stats.ttest_rel(LDS, FA)

    ax.plot([.2, .6], [6.5, 6.5], 'k-')
    ax.text(.4, 6.5, '***')
    
    ax.plot([1.2, 1.6], [5.5, 5.5], 'k-')
    ax.text(1.4, 5.5, '*')
    ax.set_ylabel('Main Shared Dimensionality')

    ax.set_xticks([.2, .6, 1.2, 1.6])
    ax.set_xticklabels(['LDS', 'FA', 'LDS', 'FA'])
    f.savefig(save_pre+'LDS_vs_FA_main_dim.svg', transparent=True)

def plot_dynamics_ratio_analysis():
    save_pre = '/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/BMI_co_obs_paper/figures/data_figs/'
    for ia, animal in enumerate(['grom', 'jeev']):
        f, ax = plt.subplots()
        f.set_figheight(5)
        f.set_figwidth(6)

        R = ['dyn_ratio_self', 'dyn_ratio_same_task_targ', 'dyn_ratio_same_task_nottarg', 'dyn_ratio_diff_task']
        x = {}
        for ir, r in enumerate(R):
            co_dyn = [np.mean(co) for co in get_metric(animal, 'data_LDS', r, 0, False)]
            obs_dyn = [np.mean(obs) for obs in get_metric(animal, 'data_LDS', r, 1, False)]
            LDS = co_dyn + obs_dyn
            ax.bar(ir, np.mean(LDS), width=1.,edgecolor='k', linewidth=4., color='w')
            ax.errorbar(.5+ir, np.mean(LDS), np.std(LDS)/np.sqrt(len(LDS)), marker='.', color='k')
            x[r] = LDS

        # print scipy.stats.kruskal(x['dyn_ratio_self'], x['dyn_ratio_same_task_targ'], x['dyn_ratio_same_task_nottarg'], x['dyn_ratio_diff_task'])
        # print 'self, sametasktarg', scipy.stats.mannwhitneyu(x['dyn_ratio_self'], x['dyn_ratio_same_task_targ'] ).pvalue * 3
        # print 'self, sametaskdifftarg', scipy.stats.mannwhitneyu(x['dyn_ratio_self'], x['dyn_ratio_same_task_nottarg'] ).pvalue
        # print 'self, diff', scipy.stats.mannwhitneyu(x['dyn_ratio_self'], x['dyn_ratio_diff_task'] ).pvalue
        print animal, 'cuzicks test for ordering: ', 
        print fcns.cuzicks_test([x['dyn_ratio_self'], x['dyn_ratio_same_task_targ'], x['dyn_ratio_same_task_nottarg'], x['dyn_ratio_diff_task']])
        print 'n : ', len(x['dyn_ratio_self']), len(x['dyn_ratio_same_task_targ']), len(x['dyn_ratio_same_task_nottarg']), len(x['dyn_ratio_diff_task'])
        ax.set_xticks(np.arange(.5, 4., 1.))
        ax.set_xticklabels(['self', 'same task, target', 'same task, diff target', 'diff task'], rotation=45)
        #f.savefig(save_pre+animal+'_dyn_ratio.svg', transparent=True)

def get_metric(animal, dict_nm, name, i_t, imp, mean_across_neurons=False, LPF=None, min_day = 0, max_day = 3, norm = 0):
    if animal == 'grom':
        input_type = co_obs_tuning_matrices.input_type
        #fname_pref = '/Volumes/TimeMachineBackups/grom2016/'
        fname_pref = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/'
        min_day = 0; max_day = 3; 
    elif animal == 'jeev':
        input_type = fk.task_filelist
        #fname_pref = '/Volumes/TimeMachineBackups/jeev2013/'
        fname_pref = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/'
        min_day = 0; max_day  = 3; 
    X = []

    for i_d in range(len(input_type)):
        if np.logical_and(i_d <= max_day, i_d >= min_day):
            data = pickle.load(open(fname_pref+'test_big_data_'+str(i_d)+'_9_6_19.pkl'))
            try:
                if len(data[dict_nm][0, i_t, imp, name]) == 2:
                    tmp = np.array([ data[dict_nm][f, i_t, imp, name][norm] for f in range(nfolds)])
                else:
                    tmp = np.array([ data[dict_nm][f, i_t, imp, name] for f in range(nfolds)])
            except:
                if len(data[dict_nm][0, i_t, imp, LPF, name]) == 2:
                    tmp = np.array([ data[dict_nm][f, i_t, imp, LPF, name][norm] for f in range(nfolds)])
                else:
                    tmp = np.array([ data[dict_nm][f, i_t, imp, LPF, name] for f in range(nfolds)])
            
            assert tmp.shape[0] == nfolds

            if len(tmp.shape) == 1:
                tmp = tmp[:, np.newaxis]

            if np.logical_and(mean_across_neurons, tmp.shape[1] >= 10):
                X.append(np.nanmean(tmp, axis=1))
            else:
                X.append(tmp)

    return X

def accrue_dynamics_ratio_analysis(model, bs_testing_all, i_t, 
    filt_sigma_from_LDS, data_vel_test, filt_state, pred_state,):
    real_dat = [bs_testing_all[x][pre_go_bins:, :] for x in range(len(bs_testing_all))]
    KalmanGain = get_sskg_ldf(model.D_latent, model.D_obs, filt_sigma_from_LDS, model)
    A = np.mat(model.A)
    C = np.mat(model.C)

    X = dict(self = [], same_task_same_targ = [], same_task = [], other_task = [])

    for i_tsk, trl in enumerate(real_dat):
        for i_b in range(1, len(trl)):

            # Velocity profile of bin: 
            # tsk, trg, rad, mag = data_vel_test[i_tsk][i_b, :]

            # First, own self: 
            dyn = np.linalg.norm(pred_state[i_tsk][:, i_b] - np.mat(filt_state[i_tsk][i_b-1, :]).T)
            inn = np.linalg.norm(KalmanGain*(np.mat(real_dat[i_tsk][i_b, :]).T - C*A*np.mat(filt_state[i_tsk][i_b-1, :]).T))

            X['self'].append(dyn/(dyn+inn))

            # Next, own task, own target:
            ix = np.zeros((3, ))
            i = 0
            j = -1
            while not np.all(ix >= 10):
                j+= 1
                if j >= len(real_dat[i]):
                    j = 0
                    i += 1
                if i >= len(real_dat):
                    ix = np.zeros((3,))+100
                    print 'done!'
                    break
                if np.logical_and(i!= i_tsk, j!=i_b):
                    try:
                        # Same target, same task, same vel command:
                        if np.all(data_vel_test[i][j, :]==data_vel_test[i_tsk][i_b, :]):
                            key = 'same_task_same_targ'
                            ix[0] += 1
                        
                        # Same task, same vel command:
                        elif np.all(data_vel_test[i][j, [0, 2, 3]] == data_vel_test[i_tsk][i_b, [0, 2, 3]]):
                            key = 'same_task'
                            ix[1] += 1

                        # Same vel command:
                        elif np.all(data_vel_test[i][j, [2, 3]] == data_vel_test[i_tsk][i_b, [2, 3]]):
                            key = 'other_task'
                            ix[2] += 1

                        else:
                            key = None

                        if key is not None:
                            inn = np.linalg.norm(KalmanGain*(np.mat(real_dat[i][j, :]).T - C*A*np.mat(filt_state[i_tsk][i_b-1, :]).T))
                            X[key].append(dyn/(dyn+inn))
                    except:
                        print 'failed on trial: ', i, 'bin ', j, 'len real_dat[i]: ', len(real_dat[i]), ' len data_vel_test[i]: ', len(data_vel_test[i])
    return X

def get_sskg_ldf(nstates, nneurons, filt_sigma, model):
    KG = []
    df = []
    kg_last = np.zeros((nstates, nneurons))
    for it, trl in enumerate(filt_sigma):
        for ib in range(trl.shape[0]):
            S = model.C*np.mat(trl[ib, :, :])*np.mat(model.C).T + model.sigma_obs
            kg = np.mat(trl[ib, :, :])*model.C.T*np.linalg.inv(S)
            KG.append(kg)
            df.append(np.linalg.norm(kg - kg_last))
            kg_last = kg.copy()
    df = np.array(df)
    df = df[5:]
    assert np.max(df) < .1
    KalmanGain = KG[6]
    return KalmanGain

def get_FA_rep_OV_data(animal):
    import pickle
    if animal == 'grom':
        fname = os.path.expandvars('$FA_GROM_DATA/grom2017_aug_subspace_ov_CO_vs_Obs_same_day_many_factors_w_epochs'+str(16)+'.pkl')
        input_type = co_obs_tuning_matrices.input_type
    elif animal == 'jeev':
        fname = os.path.expandvars('$FA_GROM_DATA/jeev2017_aug_subspace_ov_CO_vs_Obs_same_day_many_factors_w_epochs_'+str(16)+'all'+'.pkl')
        input_type = fk.task_filelist

    dat = pickle.load(open(fname))
    ndays = len(input_type)
    
    FA_OV = {}

    for i_d in range(ndays):
        FA_OV[i_d] = {}
        rep_ov = np.zeros_like(dat[i_d]['overlab_mat'])
        te_list = dat[i_d, 'te']

        # Fit an LDS model for each set: 
        # Then use own data to predict x0
        # compute overlap
        for i_t, te in enumerate(te_list):
            for i_t2, te2 in enumerate(te_list[i_t:]):
                
                # First use first FA model: 
                data_FA1 = dat[i_d][te, 0]
                mu_FA1 = dat[i_d]['mu', te]

                data_FA2 = dat[i_d][te2, 0]
                mu_FA2 = dat[i_d]['mu', te2]

                zhat1 = data_FA1.transform(data_FA1.training_data)
                zhat2 = data_FA1.transform(data_FA2.training_data + mu_FA2 - mu_FA1[0, :][np.newaxis, :])
                rep_ov[i_t, i_t2, 0] = fit_LDS.get_repertoire_ov_numerical(zhat1, zhat2)

                zhat1 = data_FA2.transform(data_FA1.training_data + mu_FA1 - mu_FA2[0, :][np.newaxis, :])
                zhat2 = data_FA2.transform(data_FA2.training_data)
                rep_ov[i_t, i_t2, 1] = fit_LDS.get_repertoire_ov_numerical(zhat1, zhat2)
                print 'done with :', i_d, i_t, i_t2
        FA_OV[i_d]['rep_ov'] = rep_ov
        FA_OV[i_d]['te_list'] = te_list
        print animal, 'done w/ day: ', i_d
    fname = fname[:-4]+'_new_FA_rep_ov.pkl'
    pickle.dump(FA_OV, open(fname, 'wb'))

##### Updated FCNs #####
def plot_all_R2_updated2019(save_figs=False, norm = 0):
    ''' 
    Plotting: All of this for normalized neurons vs. unnormalized neurons
        Model Comparison -- Linear dynamics
        Train all tasks, test all tasks: 
            All neurons 
                1. R2 -- filt, smooth, pred for LDS
                2. R2-filt-LDS, R2-FA, R2-LPF
            Important neurons 
                1. R2 -- filt, smooth, pred for LDS
                2. R2-filt-LDS, R2-FA, R2-LPF
            NP
                3. R2NP -- filt, smooth, pred for LDS
                4. R2NP-filt-LDS, R2NP-FA, R2NP-LFP


        Linear dynamics vs. Behavioral dynamics: 
            All neurons
                1. R2--smooth for LDS, R2-linear tuning
            Important neurons
                2. R2NP--smooth LDS, R2NP-linear tuning
            NP
                2. R2NP--smooth LDS, R2NP-linear tuning


        Generalization Pairs: Train one task/[test other task  vs. test on self]
            All neurons: 
                1. R2-filt-LDS, R2-FA, R2-LPF, R2-linear tuning
            Important neurons:
                1. R2-filt-LDS, R2-FA, R2-LPF, R2-linear tuning
            NP:
                1. R2-filt-LDS, R2-FA, R2-LPF, R2-linear tuning
    '''

    animals = ['grom', 'jeev']
    for impor in range(2):

        for pni, pn in enumerate(['', 'vx_vy_']):

            if np.logical_and(impor == 1, pni == 1):
                ### no neural push for only-imoprtnat: 
                pass
            else:
                ylabel = ''
                if impor == 1:
                    ylabel = 'R2 '+pn+', Top Neurons'
                else:
                    ylabel = 'R2 '+pn+', All Neurons'

                f, ax = plt.subplots(figsize=(6, 5))
                f1, ax1 = plt.subplots(figsize = (6, 5))
                f2, ax2 = plt.subplots(figsize = (6, 5))
                f3, ax3 = plt.subplots(figsize = (6, 5))

                #### For plots showing diff R2 of the LDS ####
                for ia, animal in enumerate(animals):
                    dyn_pred = [np.nanmean(co) for co in get_metric(animal, 'data_LDS', 'R2_pred_'+pn+'test', 2, impor, norm = norm, mean_across_neurons=True)]
                    dyn_filt = [np.nanmean(co) for co in get_metric(animal, 'data_LDS', 'R2_filt_'+pn+'test', 2, impor, norm = norm, mean_across_neurons=True)]
                    dyn_smooth = [np.nanmean(co) for co in get_metric(animal, 'data_LDS', 'R2_smooth_'+pn+'test', 2, impor, norm = norm, mean_across_neurons=True)]
                    
                    # Mean across folds: 
                    ax.bar(0 + ia*4, np.nanmean(dyn_pred), width=1.,edgecolor='k', linewidth=4., color='w')
                    ax.errorbar(0 + ia*4, np.nanmean(dyn_pred), np.nanstd(dyn_pred)/np.sqrt(len(dyn_pred)), marker='.', color='k')

                    ax.bar(1 + ia*4, np.nanmean(dyn_filt), width=1., edgecolor='k', linewidth=4., color='w')
                    ax.errorbar(1+ ia*4, np.nanmean(dyn_filt), np.nanstd(dyn_filt)/np.sqrt(len(dyn_filt)), marker='.', color='k')

                    ax.bar(2 + ia*4, np.nanmean(dyn_smooth), width=1.,edgecolor='k', linewidth=4., color='w')
                    ax.errorbar(2+ ia*4, np.nanmean(dyn_smooth), np.nanstd(dyn_smooth)/np.sqrt(len(dyn_smooth)), marker='.', color='k')

                    ax.set_xticks([0, 1, 2, 4, 5, 6])
                    ax.set_xticklabels(['LDS pre', 'LDS filt', 'LDS smooth', 'LDS pre', 'LDS filt', 'LDS smooth'], rotation=90)
                    ax.set_ylabel(ylabel)
                    f.tight_layout()

                #### For plots showing LDS vs. FA vs. LPF ####
                for ia, animal in enumerate(animals):
                    dyn_filt = [np.mean(co) for co in get_metric(animal, 'data_LDS', 'R2_filt_'+pn+'test', 2, impor, norm = norm, mean_across_neurons=True)]
                    FA = [np.mean(co) for co in get_metric(animal, 'data_FA', 'R2_'+pn+'test', 2, impor, norm = norm, mean_across_neurons=True)]
                    
                    # if len(pn) > 0:
                    #     pnn = '_'+pn+'tsk'; 
                    # else:
                    #     pnn = ''
                    # LPF150 = [np.mean(co) for co in get_metric(animal, 'data_LPF', 'R2_smooth'+pnn, 2, impor, LPF = 150., norm = norm, mean_across_neurons=True)]
                    # LPF300 = [np.mean(co) for co in get_metric(animal, 'data_LPF', 'R2_smooth'+pnn, 2, impor, LPF = 300., norm = norm, mean_across_neurons=True)]
                    # LPF450 = [np.mean(co) for co in get_metric(animal, 'data_LPF', 'R2_smooth'+pnn, 2, impor, LPF = 450., norm = norm, mean_across_neurons=True)]
                    # LPF600 = [np.mean(co) for co in get_metric(animal, 'data_LPF', 'R2_smooth'+pnn, 2, impor, LPF = 600., norm = norm, mean_across_neurons=True)]
                    
                    for i_s, sig in enumerate([dyn_filt, FA]):#, LPF150, LPF300, LPF450, LPF600]):
                        ax1.bar(i_s + ia*7, np.nanmean(sig), width=1.,edgecolor='k', linewidth=4., color='w')
                        ax1.errorbar(i_s + ia*7, np.nanmean(sig), np.nanstd(sig)/np.sqrt(len(sig)), marker='.', color='k')

                    ax1.set_xticks(np.hstack(( np.arange(6), np.arange(7, 7+6))))
                    ax1.set_xticklabels(['LDS filt', 'FA']*2, rotation=90)#, 'LPF150', 'LPF300', 'LFP450', 'LFP600']*2, rotation=90)
                    ax1.set_ylabel(ylabel)
                    f1.tight_layout()

                ### Plots showing neural dynamics vs. tuning: 
                for ia, animal in enumerate(animals):
                    dyn_smooth = [np.mean(co) for co in get_metric(animal, 'data_LDS', 'R2_smooth_'+pn+'test', 2, impor, norm = norm, mean_across_neurons=True)]
                    tuning = [np.mean(co) for co in get_metric(animal, 'data_tuning', 'R2_'+pn+'test', 2, impor, norm = norm, mean_across_neurons = True)]
                   
                    for i_s, sig in enumerate([dyn_smooth, tuning]):
                        ax2.bar(i_s + ia*3, np.nanmean(sig), width=1.,edgecolor='k', linewidth=4., color='w')
                        ax2.errorbar(i_s + ia*3, np.nanmean(sig), np.nanstd(sig)/np.sqrt(len(sig)), marker='.', color='k')

                    ax2.set_xticks(np.hstack(( np.arange(2), np.arange(3, 3+2))))
                    ax2.set_xticklabels(['LDS smooth', 'Linear Tuning']*2, rotation=90)
                    ax2.set_ylabel(ylabel)
                    f2.tight_layout()

                #Within task vs. across task R2-filt-LDS, R2-FA, R2-LPF, R2-linear tuning
                for ia, animal in enumerate(animals):

                    for wi, winx in enumerate(['', 'non']):

                        ### First within task: 
                        if len(pn) > 0:
                            winxx = 'not'
                        else:
                            winxx = winx; 

                        dyn_co = [np.mean(co) for co in get_metric(animal, 'data_LDS', 'R2_filt_'+pn+winxx+'tsk', 0, impor, norm = norm, mean_across_neurons=True)]
                        dyn_ob = [np.mean(co) for co in get_metric(animal, 'data_LDS', 'R2_filt_'+pn+winxx+'tsk', 1, impor, norm = norm, mean_across_neurons=True)]
                        dyn = dyn_co + dyn_ob; 

                        if len(pn) > 0:
                            prf = '_pred_'
                            sfx = '_task'
                        else:
                            prf = '_'
                            sfx = ''

                        fa_co = [np.mean(co) for co in get_metric(animal, 'data_FA', 'R2_'+pn+winxx+'tsk', 0, impor, norm = norm, mean_across_neurons=True)]
                        fa_ob = [np.mean(co) for co in get_metric(animal, 'data_FA', 'R2_'+pn+winxx+'tsk', 1, impor, norm = norm, mean_across_neurons=True)]
                        fa = fa_co + fa_ob; 

                        ### Choose a single LPF -- 
                        LPF = 300. 
                        if len(pn) > 0:
                            prf = '_pred_'
                            sfx = 'tsk'
                            us = '_'
                        else:
                            sfx = ''
                            us = ''

                        # lpf_co = [np.mean(co) for co in get_metric(animal, 'data_LPF','R2_pred'+us+pn+sfx, 0, impor, LPF = LPF, norm=norm, mean_across_neurons = True)]
                        # lpf_ob = [np.mean(co) for co in get_metric(animal, 'data_LPF','R2_pred'+us+pn+sfx, 1, impor, LPF = LPF, norm=norm, mean_across_neurons = True)]
                        # lpf = lpf_co + lpf_ob; 

                        if len(pn) > 0:
                            tuning_co = [np.mean(co) for co in get_metric(animal, 'data_tuning', 'R2_vx_vy_'+winx+'tsk', 0, impor, norm = norm, mean_across_neurons = True)]
                            tuning_ob = [np.mean(co) for co in get_metric(animal, 'data_tuning', 'R2_vx_vy_'+winx+'tsk', 1, impor, norm = norm, mean_across_neurons = True)]
                        else:
                            tuning_co = [np.mean(co) for co in get_metric(animal, 'data_tuning', 'R2_'+winx+'tsk', 0, impor, norm = norm, mean_across_neurons = True)] 
                            tuning_ob = [np.mean(co) for co in get_metric(animal, 'data_tuning', 'R2_'+winx+'tsk', 1, impor, norm = norm, mean_across_neurons = True)] 
                        tun = tuning_co + tuning_ob

                        for i_s, sig in enumerate([dyn, fa, tun]):
                            sig = np.hstack(( sig)) 
                            ax3.bar(i_s + ia*4 + 0.4*wi,  np.nanmean(sig), width=0.4,edgecolor='k', linewidth=2., color='w')
                            ax3.errorbar(i_s + ia*4 + 0.4*wi, np.nanmean(sig), np.nanstd(sig)/np.sqrt(len(sig)), marker='.', color='k')

                        ax3.set_xticks(np.hstack(( np.arange(4) ,np.arange(5, 5+4))))
                        ax3.set_xticklabels(['LDS filt', 'FA', 'Linear Tuning']*2, rotation=90)
                        ax3.set_ylabel(ylabel)
                        f3.tight_layout()

def three_bar_stats(x1, x2, x3):
    scipy.stats.f_oneway(x1, x2, x3)
    from statsmodels.stats.multicomp import MultiComparison
    grp = np.zeros((len(x1)*3, ))
    grp[len(dyn):2*len(dyn)] = 1
    grp[2*len(dyn):] = 2
    mc = MultiComparison(np.hstack((dyn, fa, tun)), grp)

    print 'MC hsd .001, ', mc.tukeyhsd(0.001)
    print 'MC hsd .01, ', mc.tukeyhsd(0.01)
    print 'MC hsd .05, ', mc.tukeyhsd(0.05)
    
    print 'n : ', len(x1), len(x2), len(x3)
    print scipy.stats.kruskal(x1, x2, x3)
    print 'mann whitney testss: '
    print scipy.stats.mannwhitneyu(x1, x2).pvalue * 2
    print scipy.stats.mannwhitneyu(x1, x3).pvalue * 2


