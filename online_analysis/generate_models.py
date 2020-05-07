#### Methods to generate models used to predict neural activity ####
from analysis_config import config 
import analysis_config
import generate_models_utils

import numpy as np
import matplotlib.pyplot as plt
import datetime
import gc
import tables, pickle

from sklearn.linear_model import Ridge

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


    ####### STATE VARIABLE: 
    #######    1 -- velocity & position 
    #######    -1 --  position 
    #######    2 -- velocity & position & target
    #######    3 -- velocity & position & target & task
    #######    0 -- none; 
    
    ####### Include action_lags --> whether to include push_{t-1} etc. 
    ###### Include past_y{t-1} if zero -- nothing. If > 1, include that many past lags


    if model_set_number == 1:
        ###                     Lags,               name,                   include state?, include y_{t-1}, include y_{t+1} ###
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
        ### Model predicting spikes with a) current spikes b) previous spikes, c) previous actions 
        ###                                                                              include state // y_{t-1} // y_{t+1} ###
        model_var_list.append([np.array([0]),                 'prespos_0psh_0spksm_1_spksp_0',         0, 1, 0])     #### Model 1: y_{t+1} | y_{t+1} --> should be 100%
        model_var_list.append([np.array([-1]),                'hist_1pos_0psh_0spksm_1_spksp_0',       0, 1, 0])     #### Model 1: y_{t+1} | y_{t}
        model_var_list.append([np.array([-1]),                'hist_1pos_0psh_1spksm_0_spksp_0',       0, 0, 0])     #### Model 1: y_{t+1} | a_t
        predict_key = 'spks'
        include_action_lags = False; ### Only action at t = 0; 

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

    ###### Clear out all HDF files that may be been previously created #####
    for obj in gc.get_objects():   # Browse through ALL objects
        if isinstance(obj, tables.File):   # Just HDF5 files
            try:
                obj.close()
            except:
                pass #

    tuning_dict = {}

    ### Get data files 
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

    #### Open file 
    h5file = tables.openFile(hdf_filename, mode="w", title=animal+'_tuning')
    vxvy0 = ['velx_tm0', 'vely_tm0']

    mFR = {}
    sdFR = {}

    #### For each day 
    for i_d, day in enumerate(input_type):
        
        # Get spike data
        ####### data is everything; 
        ###### data_temp / sub_pikes / sub_spk_temp_all / sub_push_all --> all spks / actions 
        ##### but samples only using history_bins:nT-history_bins within trial 
        data, data_temp, sub_spikes, sub_spk_temp_all, sub_push_all = generate_models_utils.get_spike_kinematics(animal,
            day, order_dict[i_d], history_bins_max)

        #### Assertion to match length #####
        assert(len(data_temp) == len(sub_spikes) == len(sub_spk_temp_all) == len(sub_push_all))

        ### Get trianing / testing sets; 
        test_ix, train_ix = generate_models_utils.get_training_testings(n_folds, data_temp)

        ### For training and testing data, setup the dictionaries
        for i_fold in range(n_folds):

            ### TEST DATA ####
            data_temp_dict_test = panda_to_dict(data_temp.iloc[test_ix[i_fold]])
            data_temp_dict_test['spks'] = sub_spikes[test_ix[i_fold]]
            data_temp_dict_test['pshy'] = sub_push_all[test_ix[i_fold], 0]
            data_temp_dict_test['pshx'] = sub_push_all[test_ix[i_fold], 1]
            data_temp_dict_test['psh'] = np.hstack((data_temp_dict_test['pshx'], data_temp_dict_test['pshy']))

            ### TRAIN DATA ####
            data_temp_dict = panda_to_dict(data_temp.iloc[train_ix[i_fold]])
            data_temp_dict['spks'] = sub_spikes[train_ix[i_fold]]
            data_temp_dict['pshy'] = sub_push_all[train_ix[i_fold], 0]
            data_temp_dict['pshx'] = sub_push_all[train_ix[i_fold], 1]
            data_temp_dict['psh'] = np.hstack((data_temp_dict['pshx'], data_temp_dict['pshy']))

            ### Get number of neruons you expect ####
            nneur = sub_spk_temp_all.shape[2]

            ### For each variable in the model: 
            for im, (model_vars, model_nm, include_state, include_past_yt, include_fut_yt) in enumerate(model_var_list):
                
                ### Include state --> 
                if include_state == 1:
                    vel_model_nms, vel_model_str = generate_models_utils.lag_ix_2_var_nm(model_vars, 'vel')
                    pos_model_nms, pos_model_str = generate_models_utils.lag_ix_2_var_nm(model_vars, 'pos')
                    tg_model_nms = tsk_model_nms = []; 
                    
                #### ONLY POSITION
                elif include_state == -1:
                    vel_model_nms = []; 
                    pos_model_nms, pos_model_str = generate_models_utils.lag_ix_2_var_nm(model_vars, 'pos')
                    tg_model_nms = tsk_model_nms = []; 

                elif include_state == 2:
                    vel_model_nms, vel_model_str = generate_models_utils.lag_ix_2_var_nm(model_vars, 'vel')
                    pos_model_nms, pos_model_str = generate_models_utils.lag_ix_2_var_nm(model_vars, 'pos')

                    ### Also include target_info: 
                    tg_model_nms, tg_model_str = generate_models_utils.lag_ix_2_var_nm(model_vars, 'tg')    
                    tsk_model_nms = [];                 

                elif include_state == 3:
                    vel_model_nms, vel_model_str = generate_models_utils.lag_ix_2_var_nm(model_vars, 'vel')
                    pos_model_nms, pos_model_str = generate_models_utils.lag_ix_2_var_nm(model_vars, 'pos')

                    ### Also include target_info: 
                    tg_model_nms, tg_model_str = generate_models_utils.lag_ix_2_var_nm(model_vars, 'tg')
                    tsk_model_nms, tsk_model_str = ['tsk', '']

                elif include_state == 0:
                    vel_model_nms = pos_model_nms = []; tg_model_nms = tsk_model_nms = []; 
                
                ### Add push always -- this push uses the model_vars; 
                push_model_nms, push_model_str = generate_models_utils.lag_ix_2_var_nm(model_vars, 'psh', include_action_lags = include_action_lags) 

                ### Past neural activity
                if include_past_yt > 0:
                    neur_nms1, neur_model_str1 = generate_models_utils.lag_ix_2_var_nm(model_vars, 'neur', nneur, np.arange(-1*include_past_yt, 0))
                else:
                    neur_nms1 = []; 

                ### Next neural activity; 
                if include_fut_yt > 0:
                    neur_nms2, neur_model_str2 = generate_models_utils.lag_ix_2_var_nm(model_vars, 'neur', nneur, np.arange(1, 1*include_fut_yt + 1))
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
                    ##### h5file is an HDF table, model_ is the output model, i_d is day, first day 
                    ##### model name 
                    ##### data_temp_dict_test 
                    h5file, model_, _ = generate_models_utils.h5_add_model(h5file, model_, i_d, first=i_d==0, model_nm=name, 
                        test_data = data_temp_dict_test, fold = i_fold, xvars = variables, predict_key = predict_key)

    h5file.close()
    print 'H5 File Done: ', hdf_filename
    return hdf_filename

#### UTILS #####
def panda_to_dict(D):
    d = dict()
    for k in D.keys():
        d[k] = np.array(D[k][:])
    return d

def fit_ridge(y_train, data_temp_dict, x_var_names, alpha = 1.0,
    test_data=None, test_data2=None, train_data2=None,
    only_potent_predictor = False, KG_pot = None, 
    fit_task_specific_model_test_task_spec = False):

    ''' fit Ridge regression using alpha = ridge parameter
        only_potent_predictor --> multiply KG_pot -- which I think should be N x N by data;  '''

    ### Initialize model 
    model_2 = Ridge(alpha=alpha)

    ### Aggregate the variable name
    x = []
    for vr in x_var_names:
        x.append(data_temp_dict[vr][: , np.newaxis])
    X = np.hstack((x))
    assert(X.shape[1] == len(x_var_names))

    ### Aggregate the training data 2 if needed: 
    X2 = []
    if train_data2 is not None: 
        for vr in x_var_names:
            X2.append(train_data2[vr][:, np.newaxis])
        X2 = np.hstack((X2))

        ### Append the training data 1 to training data 2
        print('Appending training_data2 to training_data1')
        X = np.vstack((X, X2))

    if only_potent_predictor:
        assert KG_pot is not None

        #### Assuuming that the trianing data is some sort of spiking thing
        assert(KG_pot.shape[1] == X.shape[1])
        pre_X_shape = X.shape
        X = np.dot(KG_pot, X.T).T
        
        ### Assuming KG_pot is the 
        assert(X.shape == pre_X_shape)
        print 'only potent used to train ridge'

    #### Fit two models one on each task ####
    if fit_task_specific_model_test_task_spec:
        tsk = data_temp_dict['tsk']
        ix0 = np.nonzero(tsk == 0)[0]
        ix1 = np.nonzero(tsk == 1)[0]

        X0 = X[ix0, :]; X1 = X[ix1, :]; 
        Y0 = y[ix0, :]; Y1 = y[ix1, :];
        
        #### Model 1 ####
        model_2.fit(X0, Y0); 
        
        #### Model 2 ####
        model_2_ = Ridge(alpha=alpha)
        model_2_.fit(X1, Y1); 

        ######## Get relevant params #######
        model_2.nneurons = Y0.shape[1]
        model_2.nobs = X0.shape[0]

        model_2_.nneurons = Y1.shape[1]
        model_2_.nobs = X1.shape[0]

        ######## Put the models together #######
        model_2 = [model_2, model_2_]

    else:
        model_2.fit(X, y_train)
        ### Add the data for some reason? ####
        ### Maybe this is what is making the HDF files so big?###
        # model_2.X = X
        # model_2.y = y_train

        ##### Add relevant parameters; 
        model_2.nneurons = y_train.shape[1]
        model_2.nobs = X.shape[0]

        model_2.coef_names = x_var_names

    if test_data is None:
        return model_2
    
    else:
        y = []
        z = []
        
        for vr in x_var_names:
            y.append(test_data[vr][:, np.newaxis])
        
            ### Test second dataset ###
            if test_data2 is not None:
                z.append(test_data2[vr][:, np.newaxis])
        
        Y = np.hstack((y))
        if test_data2 is not None:
            Z = np.hstack((z))
        
        pred = model_2.predict(Y)
        if test_data2 is not None:
            pred2 = model_2.predict(Z)
        else:
            pred2 = None
        
        return model_2, pred, pred2

def plot_sweep_alpha(animal, alphas = None, model_set_number = 1, ndays=None, skip_plots = True, r2_ind_or_pop = 'pop'):

    if model_set_number == 1:
        # model_names = ['prespos_0psh_1spksm_0_spksp_0', 'prespos_1psh_1spksm_0_spksp_0', 'hist_fut_4pos_1psh_1spksm_0_spksp_0',
        # 'hist_fut_4pos_1psh_1spksm_1_spksp_0','hist_fut_4pos_1psh_1spksm_1_spksp_1']
        model_names = ['prespos_0psh_1spksm_0_spksp_0', 
                            'hist_1pos_-1psh_1spksm_0_spksp_0', 
                            'hist_1pos_1psh_1spksm_0_spksp_0',
                            'hist_1pos_2psh_1spksm_0_spksp_0', 
                            'hist_1pos_3psh_1spksm_0_spksp_0', 
                            'hist_1pos_3psh_1spksm_1_spksp_0']
    elif model_set_number in [2, 3]:
        # model_names = ['hist_1pos_0psh_1spksm_0_spksp_0', 'hist_1pos_0psh_1spksm_1_spksp_0', 'hist_4pos_0psh_1spksm_0_spksp_0', 'hist_4pos_0psh_1spksm_1_spksp_0',
        # 'hist_4pos_0psh_1spksm_4_spksp_0', 'prespos_0psh_0spksm_1_spksp_0', 'hist_1pos_0psh_0spksm_1_spksp_0']
        model_names = ['prespos_0psh_0spksm_1_spksp_0','hist_1pos_0psh_0spksm_1_spksp_0', 'hist_1pos_0psh_1spksm_0_spksp_0']

    elif model_set_number in [4]:
        model_names = ['prespos_0psh_1spksm_0_spksp_0', 
                       'prespos_0psh_1spksm_1_spksp_0',

                       'prespos_1psh_1spksm_0_spksp_0', 
                       'prespos_1psh_1spksm_1_spksp_0',
                       
                       'hist_1pos_1psh_1spksm_0_spksp_0', 
                       'hist_1pos_1psh_1spksm_1_spksp_0', 
                       
                       'hist_2pos_1psh_1spksm_0_spksp_0', 
                       'hist_2pos_1psh_1spksm_1_spksp_0']

    elif model_set_number in [5]:
        model_names = ['hist_1pos_1psh_0spks_0_spksp_0',
                       'hist_1pos_0psh_1spks_1_spksp_0',
                       'hist_1pos_0psh_0spks_1_spksp_0',
                       'hist_1pos_1psh_0spks_1_spksp_0',
                       'hist_1pos_1psh_1spks_1_spksp_0']

    elif model_set_number in [6]:
        model_names = ['hist_1pos_1psh_0spks_0_spksp_0'] ### Only state; 

    elif model_set_number == 7:
        model_names = ['hist_1pos_0psh_0spksm_1_spksp_0', 'hist_1pos_0psh_1spksm_1_spksp_0']

    elif model_set_number == 8:
        model_names = ['hist_1pos_0psh_0spksm_1_spksp_0', 'hist_1pos_1psh_0spksm_0_spksp_0', 'hist_1pos_3psh_0spksm_0_spksp_0']

    if animal == 'grom':
        hdf = tables.openFile('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/grom_sweep_alpha_days_models_set%d.h5' %model_set_number)
        if ndays is None:
            ndays = 9; 
    
    elif animal == 'jeev':
        hdf = tables.openFile('/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/jeev_sweep_alpha_days_models_set%d.h5' %model_set_number)
        if ndays is None:
            ndays = 4; 

    if alphas is None:
        alphas = np.hstack(([[10**float(i), 5*10**float(i)] for i in np.arange(-3, 7)]))
    
    str_alphas = [];
    for ia, alpha in enumerate(alphas):
        tmp = str(alpha)
        tmp = tmp.replace('.','_')
        str_alphas.append(tmp)

    max_alpha = dict()

    #### days #####
    for i_d in range(ndays):

        #### model names ####
        for im, model_nm in enumerate(model_names):

            if skip_plots:
                pass
            else:
                f, ax = plt.subplots(figsize = (5, 3)) ### Separate fig for each day/
            max_alpha[i_d, model_nm] = []

            alp = []; 
            #### alphas ####
            for ia, (alpha, alpha_float) in enumerate(zip(str_alphas, alphas)):
                alp_i = []; 

                for fold in range(5):
                    tbl = getattr(hdf.root, model_nm+'_alpha_'+alpha+'_fold_'+str(fold))
                    day_index = np.nonzero(tbl[:]['day_ix'] == i_d)[0]

                    if r2_ind_or_pop == 'ind':
                        #### Add day index to alphas
                        #### Uses alpha of individual neurons to decide optimal alpha
                        alp_i.append(tbl[day_index]['r2'])
                    
                    elif r2_ind_or_pop == 'pop':
                        import pdb; pdb.set_trace()
                        alp_i.append(tbl[day_index]['r2_pop'][0])

                ### plot alpha vs. mean day: 
                if skip_plots:
                    pass
                else:
                    ax.bar(np.log10(alpha_float), np.nanmean(np.hstack((alp_i))), width=.3)
                    ax.errorbar(np.log10(alpha_float), np.nanmean(np.hstack((alp_i))), np.nanstd(np.hstack((alp_i)))/np.sqrt(len(np.hstack((alp_i)))),
                        marker='|', color='k')

                ### max_alpha[i_d, model_nm] = []
                import pdb; pdb.set_trace()
                alp.append([alpha_float, np.nanmean(np.hstack((alp_i)))])

            if skip_plots:
                pass
            else:
                ax.set_title('Animal %s, Model %s, Day %d' %(animal, model_nm, i_d), fontsize = 10)
                ax.set_ylabel('R2')
                ax.set_xlabel('Log10(alpha)')
                f.tight_layout()

            ### Get the max to use: 
            alp = np.vstack((alp))
            ix_max = np.argmax(alp[:, 1])
            max_alpha[i_d, model_nm] = alp[ix_max, 0]

    return max_alpha
