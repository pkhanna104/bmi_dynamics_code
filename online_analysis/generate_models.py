#### Methods to generate models used to predict neural activity ####
from analysis_config import config 
import analysis_config
import generate_models_utils

import numpy as np
import datetime
import gc
import tables 


def sweep_alpha_all(run_alphas=True, model_set_number = 3):

    max_alphas = dict(grom=[], jeev=[])

    alphas = []; 
    for i in range(-4, 7):
        alphas.append((1./4)*10**i)
        alphas.append((2./4.)*10**i)
        alphas.append((3./4.)*10**i)
        alphas.append(1.*10**i)

    ndays = dict(grom=9, jeev=4) # for testing only; 

    for animal in ['jeev','grom']:
        if run_alphas:
            h5_name = sweep_ridge_alpha(animal=animal, alphas = alphas, model_set_number = model_set_number, ndays = ndays[animal])
        
        else:
            if animal == 'grom':
                h5_name = config['grom_pref'] + 'grom_sweep_alpha_days_models_set%d.h5' %model_set_number
            elif animal == 'jeev':
                h5_name = config['jeev_pref'] + 'jeev_sweep_alpha_days_models_set%d.h5' %model_set_number

        max_alpha = plot_sweep_alpha(animal, alphas=alphas, model_set_number=model_set_number, ndays=ndays[animal])
        max_alphas[animal].append(max_alpha)

    pickle.dump(max_alphas, open(config['grom_pref'] + 'max_alphas_ridge_model_set%d.pkl' %model_set_number, 'wb'))
    print('Done with max_alphas_ridge')

def sweep_ridge_alpha(alphas, animal='grom', n_folds = 5, history_bins_max = 4, model_set_number = 1, ndays=None):

    '''
    Summary: 
        goal to sweep alphas for ridge regression
        do this for different model types
    '''

    ### Models to sweep ###
    model_var_list = []
    include_action_lags = True;

    if model_set_number == 1:
        ### Lags, name, include state?, include y_{t-1}, include y_{t+1} ###
        model_var_list.append([np.array([0]),     'prespos_0psh_1spksm_0_spksp_0',      0, 0, 0])     ### t=0, only action
        model_var_list.append([np.array([-1]),    'hist_1pos_-1psh_1spksm_0_spksp_0',  -1, 0, 0])     ### t=-1, action and position only
        model_var_list.append([np.array([-1]),    'hist_1pos_1psh_1spksm_0_spksp_0',    1, 0, 0])     ### t=-1, action and prev state
        model_var_list.append([np.array([-1]),    'hist_1pos_2psh_1spksm_0_spksp_0',    2, 0, 0])     ### t=-1, action and prev state + target
        model_var_list.append([np.array([-1]),    'hist_1pos_3psh_1spksm_0_spksp_0',    3, 0, 0])     ### t=-1, action and prev state + target + task
        model_var_list.append([np.array([-1]),    'hist_1pos_3psh_1spksm_1_spksp_0',    3, 1, 0])     ### t=-1, action and prev state + target + task plus neural dynamics
        # model_var_list.append([np.arange(-history_bins_max, history_bins_max+1), 'hist_fut_4pos_1psh_1spksm_0_spksp_0', 1, 0, 0]) ### t=-4,4, action and state
        # model_var_list.append([np.arange(-history_bins_max, history_bins_max+1), 'hist_fut_4pos_1psh_1spksm_1_spksp_0', 1, 1, 0]) ### t=-4,4, action and state, y_{t-1}
        # model_var_list.append([np.arange(-history_bins_max, history_bins_max+1), 'hist_fut_4pos_1psh_1spksm_1_spksp_1', 1, 1, 1]) ### t=-4,4, action and state, y_{t-1}, y_{t+1}
        predict_key = 'spks'
        include_action_lags = False; ### Only action at t = 0; 

    elif model_set_number == 2:
        ### Lags, name, include state?, include y_{t-1}, include y_{t+1} ###
        #nclude_state, include_past_yt, include_fut_yt

        ### Also want only neural, only history of neural, and a_t plus history of neural. 
        model_var_list.append([np.array([0]),                 'prespos_0psh_0spksm_1_spksp_0',         0, 1, 0])     #### Model 1: a_{t+1} | y_{t+1} --> should be 100%
        model_var_list.append([np.array([-1]),                'hist_1pos_0psh_0spksm_1_spksp_0',       0, 1, 0])     #### Model 1: a_{t+1} | y_{t}
        model_var_list.append([np.array([-1]),                'hist_1pos_0psh_1spksm_0_spksp_0',       0, 0, 0])     #### Model 1: a_{t+1} | a_t
        model_var_list.append([np.array([-1]),                'hist_1pos_0psh_1spksm_1_spksp_0',       1, 0, 0])     #### Model 2: a_{t+1} | a_t, y_t
        model_var_list.append([np.array([-4, -3, -2, -1]),    'hist_4pos_0psh_1spksm_0_spksp_0',       0, 0, 0])     #### Model 3: a_{t+1} | a_t, a_{t-1},...
        model_var_list.append([np.array([-4, -3, -2, -1]),    'hist_4pos_0psh_1spksm_1_spksp_0',       0, 1, 0])     #### Model 4: a_{t+1} | a_t, a_{t-1},..., y_t; 
        model_var_list.append([np.array([-4, -3, -2, -1]),    'hist_4pos_0psh_1spksm_4_spksp_0',       0, 4, 0])     #### Model 5: a_{t+1} | a_t, a_{t-1},..., y_t, y_{t-1},...;
        predict_key = 'psh'

    elif model_set_number == 3:
        ### Same as 2, but with spks: 
        model_var_list.append([np.array([0]),                 'prespos_0psh_0spksm_1_spksp_0',         0, 1, 0])     #### Model 1: y_{t+1} | y_{t+1} --> should be 100%
        model_var_list.append([np.array([-1]),                'hist_1pos_0psh_0spksm_1_spksp_0',       0, 1, 0])     #### Model 1: y_{t+1} | y_{t}
        model_var_list.append([np.array([-1]),                'hist_1pos_0psh_1spksm_0_spksp_0',       0, 0, 0])     #### Model 1: y_{t+1} | a_t
        # model_var_list.append([np.array([-1]),                'hist_1pos_0psh_1spksm_1_spksp_0',       1, 0, 0])     #### Model 2: y_{t+1} | a_t, y_t
        # model_var_list.append([np.array([-4, -3, -2, -1]),    'hist_4pos_0psh_1spksm_0_spksp_0',       0, 0, 0])     #### Model 3: y_{t+1} | a_t, a_{t-1},...
        # model_var_list.append([np.array([-4, -3, -2, -1]),    'hist_4pos_0psh_1spksm_1_spksp_0',       0, 1, 0])     #### Model 4: y_{t+1} | a_t, a_{t-1},..., y_t; 
        # model_var_list.append([np.array([-4, -3, -2, -1]),    'hist_4pos_0psh_1spksm_4_spksp_0',       0, 4, 0])     #### Model 5: y_{t+1} | a_t, a_{t-1},..., y_t, y_{t-1},...;
        predict_key = 'spks'

    elif model_set_number == 4:
        #### Here we only want action at time T, and only state at the given lag, not all lags;
        model_var_list.append([np.array([0]), 'prespos_0psh_1spksm_0_spksp_0',       0, 0, 0])     ### t=0, only action
        model_var_list.append([np.array([0]), 'prespos_0psh_1spksm_1_spksp_0',       0, 1, 0])     ### t=0, only action & n_t 
        model_var_list.append([np.array([0]), 'prespos_1psh_1spksm_0_spksp_0',       1, 0, 0])     ### t=0,  action and state
        model_var_list.append([np.array([0]), 'prespos_1psh_1spksm_1_spksp_0',       1, 1, 0])     ### t=0, only action and state and n_t 
        model_var_list.append([np.array([-1]), 'hist_1pos_1psh_1spksm_0_spksp_0',     1, 0, 0])     ### t=0,  action and state @ -1
        model_var_list.append([np.array([-1]),  'hist_1pos_1psh_1spksm_1_spksp_0',    1, 1, 0])     ### t=0, only action and state t-1 & n_t 
        model_var_list.append([np.array([-2]), 'hist_2pos_1psh_1spksm_0_spksp_0',     1, 0, 0])     ### t=0,  action and state @ -1
        model_var_list.append([np.array([-2]), 'hist_2pos_1psh_1spksm_1_spksp_0',     1, 1, 0])     ### t=0, only action and state t-2 adn n_t 
        predict_key = 'spks'
        history_bins_max = 2; 
        include_action_lags = False ### only add action at current time point; 

    elif model_set_number == 5:
        model_var_list.append([np.array([-1]), 'hist_1pos_1psh_0spks_0_spksp_0', 0, 0, 0]) ### state only for regression again.  
        model_var_list.append([np.array([-1]), 'hist_1pos_0psh_1spks_1_spksp_0', 0, 1, 0]) ### everything except state: (spks / psh)
        model_var_list.append([np.array([-1]), 'hist_1pos_0psh_0spks_1_spksp_0', 0, 1, 0]) ### everything excpet state: (spks)
        model_var_list.append([np.array([-1]), 'hist_1pos_1psh_0spks_1_spksp_0', 1, 1, 0]) ### Include both for comparison to jointly fitting. 
        model_var_list.append([np.array([-1]), 'hist_1pos_1psh_1spks_1_spksp_0', 1, 1, 0]) ### Include both for comparison to jointly fitting. 
        predict_key = 'spks'

    elif model_set_number == 6:
        model_var_list.append([np.array([-1]), 'hist_1pos_1psh_0spks_0_spksp_0', 1, 0, 0]) ### Only state; 
        predict_key = 'psh'

    elif model_set_number == 7:
        model_var_list.append([np.array([-1]), 'hist_1pos_0psh_0spksm_1_spksp_0', 0, 1, 0]) ### only previous neural activity; 
        model_var_list.append([np.array([-1]), 'hist_1pos_0psh_1spksm_1_spksp_0', 0, 1, 0]) ### only previous neural activity & action; 
        predict_key = 'spks'

    elif model_set_number == 8:
        ### Dissecting previous state encoding vs. neural encoding. 
        model_var_list.append([np.array([-1]), 'hist_1pos_0psh_0spksm_1_spksp_0', 0, 1, 0]) ### only previous neural activity; 
        model_var_list.append([np.array([-1]), 'hist_1pos_1psh_0spksm_0_spksp_0', 1, 0, 0]) ### previous state
        model_var_list.append([np.array([-1]), 'hist_1pos_3psh_0spksm_0_spksp_0', 3, 0, 0]) #### previous state + targ + task
        predict_key = 'spks'


    elif model_set_number == 9:
        ### Summary of all models -- R2 instead of just mean diffs ###
        model_var_list.append([np.arary([-1]), 'hist_1pos_0psh_0spksm_1_spksp_0', 0, 1, 0])
        model_var_list.append([np.arary([-1]), 'hist_1pos_0psh_1spksm_0_spksp_0', 0, 0, 0])
        model_var_list.append([np.arary([0]),   'prespos_0psh_1spksm_0_spksp_0',  0, 0, 0])
        model_var_list.append([np.arary([0]),   'prespos_1psh_0spksm_0_spksp_0',  1, 0, 0])
        model_var_list.append([np.arary([-1]), 'hist_1pos_1psh_0spksm_0_spksp_0', 1, 0, 0])

    x = datetime.datetime.now()
    hdf_filename = animal + '_sweep_alpha_days_models_set%d.h5' %model_set_number

    pref = config[animal + '_pref']
    hdf_filename = pref + hdf_filename; 

    for obj in gc.get_objects():   # Browse through ALL objects
        if isinstance(obj, tables.File):   # Just HDF5 files
            try:
                obj.close()
            except:
                pass #

    tuning_dict = {}
    if animal == 'grom':
        order_dict = analysis_config.data_params['grom_ordered_input_type']
        input_type = analysis_config.data_params['grom_input_type']

    elif animal == 'jeev':
        order_dict = analysis_config.data_params['jeev_ordered_input_type']
        input_type = analysis_config.data_params['jeev_input_type']

    if ndays is None:
        pass
    else:
        print 'Only using %d days' %ndays
        input_type = [input_type[i] for i in range(ndays)]
        order_dict = [order_dict[i] for i in range(ndays)]

    h5file = tables.openFile(hdf_filename, mode="w", title=animal+'_tuning')
    vxvy0 = ['velx_tm0', 'vely_tm0']

    mFR = {}
    sdFR = {}

    for i_d, day in enumerate(input_type):
        
        # Get spike data from data fcn
        data, data_temp, spikes, sub_spk_temp_all, sub_push_all = generate_models_utils.get_spike_kinematics(animal,
            day, order_dict[i_d], history_bins_max)

        ########################################################################################################
        ### Ok, time to amp up model-wise : Fit all neurons simulataneously, and add many more model options ###
        ########################################################################################################
        ### Get training and testing datasets: 
        N_pts = [];

        ### List which points are from which task: 
        for tsk in range(2):
            ix = np.nonzero(data_temp['tsk'] == tsk)[0]

            ### Shuffle the task indices
            N_pts.append(ix[np.random.permutation(len(ix))])
        
        train_ix = dict();
        test_ix = dict(); 

        ### Get training and testing data
        for i_f, fold_perc in enumerate(np.arange(0, 1., 1./n_folds)):
            test_ix[i_f] = []
            train_ix[i_f] = []; 

            ### Task -- pull test/train points 
            for tsk in range(2):
                ntmp = len(N_pts[tsk])
                tst = N_pts[tsk][int(fold_perc*ntmp):int((fold_perc+(1./n_folds))*ntmp)]
                trn = np.array([j for i, j in enumerate(N_pts[tsk]) if j not in tst])
                
                test_ix[i_f].append(tst)
                train_ix[i_f].append(trn)

            test_ix[i_f] = np.hstack((test_ix[i_f]))
            train_ix[i_f] = np.hstack((train_ix[i_f]))

            tmp = np.unique(np.hstack((test_ix[i_f], train_ix[i_f])))

            ### Make sure that unique point is same as all data
            assert len(tmp) == len(data_temp)

        ### For training and testing data, setup the dictionaries
        for i_fold in range(n_folds):

            ### TEST DATA ####
            data_temp_dict_test = panda_to_dict(data_temp.iloc[test_ix[i_fold]])
            data_temp_dict_test['spks'] = spikes[test_ix[i_fold]]
            data_temp_dict_test['pshy'] = sub_push_all[test_ix[i_fold], 0]
            data_temp_dict_test['pshx'] = sub_push_all[test_ix[i_fold], 1]
            data_temp_dict_test['psh'] = np.hstack((data_temp_dict_test['pshx'], data_temp_dict_test['pshy']))

            ### TRAIN DATA ####
            data_temp_dict = panda_to_dict(data_temp.iloc[train_ix[i_fold]])
            data_temp_dict['spks'] = spikes[train_ix[i_fold]]
            data_temp_dict['pshy'] = sub_push_all[train_ix[i_fold], 0]
            data_temp_dict['pshx'] = sub_push_all[train_ix[i_fold], 1]
            data_temp_dict['psh'] = np.hstack((data_temp_dict['pshx'], data_temp_dict['pshy']))

            nneur = sub_spk_temp_all.shape[2]

            ### list of: [lags, model_nm]
            ### For each variable in the model: 
            for im, (model_vars, model_nm, include_state, include_past_yt, include_fut_yt) in enumerate(model_var_list):
                
                if include_state == 1:
                    vel_model_nms, vel_model_str = lag_ix_2_var_nm(model_vars, 'vel')
                    pos_model_nms, pos_model_str = lag_ix_2_var_nm(model_vars, 'pos')
                    tg_model_nms = tsk_model_nms = []; 
                    
                #### ONLY POSITION
                elif include_state == -1:
                    vel_model_nms = []; 
                    pos_model_nms, pos_model_str = lag_ix_2_var_nm(model_vars, 'pos')
                    tg_model_nms = tsk_model_nms = []; 

                elif include_state == 2:
                    vel_model_nms, vel_model_str = lag_ix_2_var_nm(model_vars, 'vel')
                    pos_model_nms, pos_model_str = lag_ix_2_var_nm(model_vars, 'pos')

                    ### Also include target_info: 
                    tg_model_nms, tg_model_str = lag_ix_2_var_nm(model_vars, 'tg')    
                    tsk_model_nms = [];                 

                elif include_state == 3:
                    vel_model_nms, vel_model_str = lag_ix_2_var_nm(model_vars, 'vel')
                    pos_model_nms, pos_model_str = lag_ix_2_var_nm(model_vars, 'pos')

                    ### Also include target_info: 
                    tg_model_nms, tg_model_str = lag_ix_2_var_nm(model_vars, 'tg')
                    tsk_model_nms, tsk_model_str = ['tsk', '']

                elif include_state == 0:
                    vel_model_nms = pos_model_nms = []; tg_model_nms = tsk_model_nms = []; 
                
                ### Add push always -- this push uses the model_vars; 
                push_model_nms, push_model_str = lag_ix_2_var_nm(model_vars, 'psh', include_action_lags = include_action_lags) 

                ### Past neural activity
                if include_past_yt != 0:
                    # Do we only want 1 lag? 
                    if include_past_yt == 0:
                        neur_nms1, neur_model_str1 = lag_ix_2_var_nm(model_vars, 'neur', nneur, [0])
                    else:
                        neur_nms1, neur_model_str1 = lag_ix_2_var_nm(model_vars, 'neur', nneur, np.arange(-1*include_past_yt, 0))
                else:
                    neur_nms1 = []; 

                ### Next neural activity; 
                if include_fut_yt != 0:
                    neur_nms2, neur_model_str2 = lag_ix_2_var_nm(model_vars, 'neur', nneur, np.arange(1, 1*include_fut_yt + 1))
                else:
                    neur_nms2 = []; 

                ### Get all the variables together ###
                variables = np.hstack(( [vel_model_nms, pos_model_nms, tg_model_nms, tsk_model_nms, push_model_nms, neur_nms1, neur_nms2] ))

                #############################
                ### Model with parameters ###
                #############################

                for ia, alpha in enumerate(alphas):

                    ### Model ###
                    model_ = fit_ridge(data_temp_dict[predict_key], data_temp_dict, variables, alpha=alpha)
                    str_alpha = str(alpha)
                    str_alpha = str_alpha.replace('.','_')
                    name = model_nm + '_alpha_' + str_alpha

                    ### add info ###
                    h5file, model_, _ = h5_add_model(h5file, model_, i_d, first=i_d==0, model_nm=name, 
                        test_data = data_temp_dict_test, fold = i_fold, xvars = variables, predict_key = predict_key)

    h5file.close()
    print 'H5 File Done: ', hdf_filename
    return hdf_filename