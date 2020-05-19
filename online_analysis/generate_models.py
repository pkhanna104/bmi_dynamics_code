#### Methods to generate models used to predict neural activity ####

from analysis_config import config 
import analysis_config
import generate_models_utils, generate_models_list, util_fcns

import numpy as np
import matplotlib.pyplot as plt
import datetime
import gc
import tables, pickle
from collections import defaultdict

from sklearn.linear_model import Ridge
import scipy

######### STEP 1 -- Get alpha value #########
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

    #### Get models from here ###
    model_var_list, predict_key, include_action_lags, _ = generate_models_list.get_model_var_list(model_set_number)

    x = datetime.datetime.now()
    hdf_filename = animal + '_sweep_alpha_days_models_set%d.h5' %model_set_number

    pref = config[animal + '_pref']
    hdf_filename = pref + hdf_filename

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

            variables_list = return_variables_associated_with_model_var(model_var_list, include_action_lags, nneur)
            
            
            ### For each variable in the model: 
            for _, (variables, model_var_list_i) in enumerate(zip(variables_list, model_var_list)):

                ### Unpack model_var_list; 
                _, model_nm, _, _, _ = model_var_list_i; 

                #import pdb; pdb.set_trace()
                ####### HERE ######
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

######## STEP 2 -- Fit the models ###########
### Main tuning function -- run this for diff animals; 
def model_individual_cell_tuning_curves(hdf_filename='_models_to_pred_mn_diffs', 
    animal='grom', 
    n_folds = 5, 
    norm_neur = False, 
    return_models = True, 
    model_set_number = 8, 
    ndays = None,
    include_null_pot = False,
    only_potent_predictor = False,

    fit_task_specific_model_test_task_spec = False,
    fit_task_spec_and_general = False,
    fit_condition_spec_no_general = False):
    
    ### Deprecated variables 
    only_vx0_vy0_tsk_mod=False; 
    task_prediction = False; 
    compute_res = False; 
    ridge=True; 

    '''
    Summary: 
    
    Modified 9/13/19 to add neural tuning on top of this; 
    Modified 9/16/19 to add model return on top of this; 
    Modiefied 9/17/19 to use specialized alphas for each model, and change notation to specify y_{t-1} and/or y_{t+1}
    Modified 5/7/20 to use "generate models list" to determine model parameters instead of looping throuhg ;

    Input param: hdf_filename: name to save params to 
    Input param: animal: 'grom' or jeev'
    Input param: history_bins_max: number of bins to use into the past and 
        future to model spike counts (binsize = 100ms)

    Input param: only_vx0_vy0_tsk_mod: True if you only want to model res ~ velx_tm0:tsk + vely_tm0:tsk instead
        of all variables being task-modulated
        --- update 5/7/20 -- removing this option, will raise exception; 

    Input param: task_prediction: True if you want to include way to predict task (0: CO, 1: Obs) from kinematics
    Input param: **kwargs: 
        task_pred_prefix -- what is this? 
        --- appears to be deprecated; 

        use_lags: list of lags to actually use (e.g. np.arange(-4, 5, 2))
        normalize_neurons: whether to compute a within-day mFR and sdFR so that high FR neurons don't dominate R2 
            calculations

    Task specificity: 
        -- fit_task_specific_model_test_task_spec --- leftover from sept 2019 -- fit on task specific data / test on it 
        -- fit_task_spec_and_general -- fit on task spec data and general data and use models to reconstruct all data; 
        -- fit_condition_spec_no_general -- fit data on condition specific data, reconstruct all data. 

    Output param: 
    '''
    model_var_list, predict_key, include_action_lags, history_bins_max = generate_models_list.get_model_var_list(model_set_number)

    ### Get model params from the models list: 
    pref = analysis_config.config[animal + '_pref']

    #### Generate final HDF name ######
    if hdf_filename is None: 
        x = datetime.datetime.now()
        hdf_filename = animal + '_' + x.strftime("%Y_%m_%d_%H_%M_%S") + '.h5'
        hdf_filename = pref + hdf_filename; 

    else:
        if fit_task_specific_model_test_task_spec:
            hdf_filename = pref + animal + hdf_filename + '_model_set%d_task_spec.h5' %model_set_number
        else:
            hdf_filename = pref + animal + hdf_filename + '_model_set%d.h5' %model_set_number

    ### Place to save models: 
    model_data = dict(); 

    ### Get the ridge dict: 
    ridge_dict = pickle.load(open(analysis_config.config['grom_pref'] + 'max_alphas_ridge_model_set%d.pkl' %model_set_number, 'rb')); 

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
        order_dict = [order_dict[i] for i in range(ndays)]
        input_type = [input_type[i] for i in range(ndays)]

    h5file = tables.openFile(hdf_filename, mode="w", title=animal+'_tuning')

    if task_prediction:
        task_dict_file = {} 
        if 'task_pred_prefix' not in kwargs.keys():
            raise Exception

    mFR = {}
    sdFR = {}

    ##### For each day ####
    for i_d, day in enumerate(input_type):
        
        # Get spike data from data fcn
        data, data_temp, sub_spikes, sub_spk_temp_all, sub_push_all = generate_models_utils.get_spike_kinematics(animal, day, 
            order_dict[i_d], history_bins_max)

        models_to_include = []
        for m in model_var_list:
            models_to_include.append(m[1])

        ### Get kalman gain etc. 
        if animal == 'grom':
            KG, KG_null_proj, KG_potent_orth = get_KG_decoder_grom(i_d)

        elif animal == 'jeev':
            KG, KG_null_proj, KG_potent_orth = get_KG_decoder_jeev(i_d)

        ### Make sure the KG_potent decoders are 2D ####
        assert np.linalg.matrix_rank(KG_potent_orth) == 2

        ###############################################################
        ################ CODE TO REVIEW / RESTRUCTURE #################
        ###############################################################

        ### Seems like we want to save variables ?? ####
        ### Replicate datastructure for saving later: 
        if return_models:

            ### Want to save neural push, task, target 
            model_data[i_d, 'spks'] = sub_spikes.copy();
            model_data[i_d, 'task'] = np.squeeze(np.array(data_temp['tsk']))
            model_data[i_d, 'trg'] = np.squeeze(np.array(data_temp['trg']))
            model_data[i_d, 'np'] = np.squeeze(np.array(sub_push_all))
            model_data[i_d, 'bin_num'] = np.squeeze(np.array(data_temp['bin_num']))
            model_data[i_d, 'pos'] = np.vstack((np.array(data_temp['posx_tm0']), np.array(data_temp['posy_tm0']))).T
            model_data[i_d, 'vel'] = np.vstack((np.array(data_temp['velx_tm0']), np.array(data_temp['vely_tm0']))).T
            model_data[i_d, 'vel_tm1'] = np.vstack((np.array(data_temp['velx_tm1']), np.array(data_temp['vely_tm1']))).T
            model_data[i_d, 'pos_tm1'] = np.vstack((np.array(data_temp['posx_tm1']), np.array(data_temp['posy_tm1']))).T
            model_data[i_d, 'trl'] = np.squeeze(np.array(data_temp['trl']))
            
            ### Models -- save predictions
            for mod in models_to_include:

                ### Models to save ##########
                ### Keep spikes ####
                if fit_task_spec_and_general:
                    nT, nn = sub_spikes.shape
                    model_data[i_d, mod] = np.zeros((nT, nn, 3)) 
                
                elif fit_condition_spec_no_general:
                    ### Use whatever keys are available
                    model_data[i_d, mod] = defaultdict(list)
                
                else:
                    model_data[i_d, mod] = np.zeros_like(sub_spikes) 

                if include_null_pot:
                    ### if just use null / potent parts of predictions and propogate those guys
                    model_data[i_d, mod, 'null'] = np.zeros_like(sub_spikes)
                    model_data[i_d, mod, 'pot'] = np.zeros_like(sub_spikes)

                elif model_set_number == 2:
                    ##### 
                    model_data[i_d, mod] = np.zeros_like(np.squeeze(np.array(sub_push_all)))

        if norm_neur:
            print 'normalizing neurons!'
            mFR[i_d] = np.mean(sub_spikes, axis=0)
            sdFR[i_d] = np.std(sub_spikes, axis=0)
            sdFR[i_d][sdFR[i_d]==0] = 1
            sub_spikes = ( sub_spikes - mFR[i_d][np.newaxis, :] ) / sdFR[i_d][np.newaxis, :]

        #### Get training / testing sets split up --- test on 80% one task, test on 20% same tasks 20% other task
        if fit_task_spec_and_general:
            test_ix, train_ix, type_of_model = generate_models_utils.get_training_testings_generalization(n_folds, data_temp)
        
        elif fit_condition_spec_no_general:
            test_ix, train_ix, type_of_model = generate_models_utils.get_training_testings_condition_spec(n_folds, data_temp)
            model_data[i_d, 'type_of_model'] = type_of_model
        ####### Get the test / train indices balanced over both tasks; 
        else:
            test_ix, train_ix = generate_models_utils.get_training_testings(n_folds, data_temp)
            type_of_model = np.zeros((n_folds, ))

        for i_fold, type_of_model_index in enumerate(type_of_model):

            if type_of_model_index < 0:
                pass
            else:
                ### TEST DATA ####
                data_temp_dict_test = panda_to_dict(data_temp.iloc[test_ix[i_fold]])
                data_temp_dict_test['spks'] = sub_spikes[test_ix[i_fold]]
                data_temp_dict_test['pshy'] = sub_push_all[test_ix[i_fold], 1]
                data_temp_dict_test['pshx'] = sub_push_all[test_ix[i_fold], 0]
                data_temp_dict_test['psh'] = np.hstack(( data_temp_dict_test['pshx'], data_temp_dict_test['pshy']))

                ### TRAIN DATA ####
                data_temp_dict = panda_to_dict(data_temp.iloc[train_ix[i_fold]])
                data_temp_dict['spks'] = sub_spikes[train_ix[i_fold]]
                data_temp_dict['pshy'] = sub_push_all[train_ix[i_fold], 1]
                data_temp_dict['pshx'] = sub_push_all[train_ix[i_fold], 0]
                data_temp_dict['psh'] = np.hstack(( data_temp_dict['pshx'], data_temp_dict['pshy']))

                nneur = sub_spk_temp_all.shape[2]

                variables_list = return_variables_associated_with_model_var(model_var_list, include_action_lags, nneur)
                
                ### For each variable in the model: 
                for _, (variables, model_var_list_i) in enumerate(zip(variables_list, model_var_list)):

                    ### These are teh params; 
                    _, model_nm, _, _, _ = model_var_list_i

                    ## Store the params; 
                    model_data[model_nm, 'variables'] = variables
                    #import pdb; pdb.set_trace()

                    if ridge:
                        alpha_spec = ridge_dict[animal][0][i_d, model_nm]
                        model_ = fit_ridge(data_temp_dict[predict_key], data_temp_dict, variables, alpha=alpha_spec, 
                            only_potent_predictor = only_potent_predictor, KG_pot = KG_potent_orth, 
                            fit_task_specific_model_test_task_spec = fit_task_specific_model_test_task_spec)
                        save_model = True
                    else:
                        raise Exception('Need to figure out teh stirng business again -- removed for clarity')
                        model_ = ols(st, data_temp_dict).fit()
                        save_model = False

                    if save_model:
                        h5file, model_, pred_Y = generate_models_utils.h5_add_model(h5file, model_, i_d, first=i_d==0, model_nm=model_nm, 
                            test_data = data_temp_dict_test, fold = i_fold, xvars = variables, predict_key=predict_key, 
                            only_potent_predictor = only_potent_predictor, KG_pot = KG_potent_orth,
                            fit_task_specific_model_test_task_spec = fit_task_specific_model_test_task_spec)

                        ### Save models, make predictions ####
                        ### Need to figure out which spikes are where:
                        if fit_task_specific_model_test_task_spec:
                            ix0 = np.nonzero(data_temp_dict_test['tsk'] == 0)[0]
                            ix1 = np.nonzero(data_temp_dict_test['tsk'] == 1)[0]
                            
                            model_data[i_d, model_nm][test_ix[i_fold][ix0], :] = np.squeeze(np.array(pred_Y[0]))
                            model_data[i_d, model_nm][test_ix[i_fold][ix1], :] = np.squeeze(np.array(pred_Y[1]))
                        
                        elif fit_task_spec_and_general:
                            model_data[i_d, model_nm][test_ix[i_fold], :, type_of_model_index] = np.squeeze(np.array(pred_Y))
                        
                        elif fit_condition_spec_no_general:
                            ### List the indices and the prediction and the fold: 
                            model_data[i_d, model_nm][type_of_model_index, 'ix'].append(test_ix[i_fold])
                            model_data[i_d, model_nm][type_of_model_index, 'pred'].append(np.squeeze(np.array(pred_Y)))
                            
                        else:
                            model_data[i_d, model_nm][test_ix[i_fold], :] = np.squeeze(np.array(pred_Y))
                           
                            ### Save model -- for use later. 
                            if model_set_number == 8:
                                model_data[i_d, model_nm, i_fold, 'model'] = model_; 
                                model_data[i_d, model_nm, i_fold, 'model_testix'] = test_ix[i_fold]; 


                        #### Add / null potent? 
                        if include_null_pot:

                            ### Get predictors together: 
                            x_test = [];
                            for vr in variables:
                                x_test.append(data_temp_dict_test[vr][: , np.newaxis])
                            X = np.mat(np.hstack((x_test)))

                            ### Get null and potent -- KG_null_proj, KG_potent_orth
                            X_null = np.dot(KG_null_proj, X.T).T
                            X_pot =  np.dot(KG_potent_orth, X.T).T

                            assert np.allclose(X, X_null + X_pot)

                            ### Need to divide the intercept into null / potent: 
                            intc = model_.intercept_
                            intc_null = np.dot(KG_null_proj, intc)
                            intc_pot = np.dot(KG_potent_orth, intc)

                            assert np.allclose(np.sum(np.abs(np.dot(KG, intc_null))), 0)
                            assert np.allclose(np.sum(np.abs(np.dot(KG, X_null.T))), 0)

                            pred_null = np.mat(X_null)*np.mat(model_.coef_).T + intc_null[np.newaxis, :]
                            pred_pot = np.mat(X_pot)*np.mat(model_.coef_).T + intc_pot[np.newaxis, :]

                            assert np.allclose(pred_Y, pred_null + pred_pot)

                            ### This just propogates the identity; 
                            # model_data[i_d, giant_model_name][test_ix[i_fold], :] = X.copy()
                            model_data[i_d, model_nm, 'null'][test_ix[i_fold], :] = pred_null.copy()
                            model_data[i_d, model_nm, 'pot'][test_ix[i_fold], :] = pred_pot.copy()
                            
    h5file.close()
    print 'H5 File Done: ', hdf_filename

    ### ALSO SAVE MODEL_DATA: 
    if only_potent_predictor:
        pickle.dump(model_data, open(analysis_config.config[animal + '_pref'] + 'tuning_models_'+animal+'_model_set%d_only_pot.pkl' %model_set_number, 'wb'))
    else:
        if fit_task_specific_model_test_task_spec:
            pickle.dump(model_data, open(analysis_config.config[animal + '_pref'] + 'tuning_models_'+animal+'_model_set%d_task_spec.pkl' %model_set_number, 'wb'))
        elif fit_task_spec_and_general:
            pickle.dump(model_data, open(analysis_config.config[animal + '_pref'] + 'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen.pkl' %model_set_number, 'wb'))
        elif fit_condition_spec_no_general:
            pickle.dump(model_data, open(analysis_config.config[animal + '_pref'] + 'tuning_models_'+animal+'_model_set%d_cond_spec.pkl' %model_set_number, 'wb'))
        else:
            pickle.dump(model_data, open(analysis_config.config[animal + '_pref'] + 'tuning_models_'+animal+'_model_set%d.pkl' %model_set_number, 'wb'))

######## Possible STEP 2 -- fit the residuals #####
def model_state_encoding(animal, model_set_number = 7, state_vars = ['pos_tm1', 'vel_tm1', 'trg', 'tsk'],
    model = 'hist_1pos_0psh_0spksm_1_spksp_0', n_folds = 5):
    '''
    method to open up a model fit above, compute the residules, and then fit the residules to state encoding? 
        -- need to make sure all the correct variables are in the dictionary; 
    
    inputs; 
        animal == jeev or grom ; 
        model_set_number = file to pull from ; 
        state_vars = list of state variables to pull from ; 
        fit_task_spec_and_general = [CO task / OBS task / General ]
        fit_condition_spec_no_general = [all keys with indices and own target]
        model = name of dynamics model -- this should be relatively constant; 

    '''

    ### Open up the model, compute the residuals from the previous model; 
    ### then set them as true; 
    ### save the original data, the dynamics prediction, the state prediction so all types of R2 can be computed; 
    suffx = ''

    ### Load this animal's data;
    suffx1 = '_task_spec_pls_gen'
    suffx2 = '_cond_spec'
    
    dat = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d%s.pkl' %(model_set_number, suffx1), 'rb'))
    dat_cond = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d%s.pkl' %(model_set_number, suffx2), 'rb'))

    ### Go through all the days; 
    ndays = analysis_config.data_params[animal + '_ndays']
    order_dict = analysis_config.data_params[animal+'_ordered_input_type']
    input_type = analysis_config.data_params[animal+'_input_type']

    ### Start a model dictionary; ###
    rez_dict = {}; pred_dict = {}

    ### Go through all the days ###
    for i_d, day in enumerate(input_type):

        rez_dict[i_d] = {}
        pred_dict[i_d] = {}
        data_temp_rez = {}

        true_spks = dat[i_d, 'spks']

        ### Now compute the residuals ###
        #### Get the general thing: 
        rez_dict[i_d]['res', 'gen'] = true_spks - dat[i_d, model][:, :, 2]

        ### Compile the 'within task' 
        tsk_spec = np.zeros_like(true_spks)
        tsk0 = np.nonzero(dat[i_d, 'task'] == 0)[0]
        tsk1 = np.nonzero(dat[i_d, 'task'] == 1)[0]

        ### Get 'within' task for CO and OBS ###
        tsk_spec = np.zeros_like(true_spks)
        tsk_spec[tsk0, :] = dat[i_d, model][tsk0, :, 0]
        tsk_spec[tsk1, :] = dat[i_d, model][tsk1, :, 1]

        ### copy the tsk spec one: 
        rez_dict[i_d]['res', 'tsk'] = true_spks - tsk_spec.copy()

        ### Now get the condition specific model ### 
        cond_spec = np.zeros_like(true_spks)
        ix_all = []; 
        for trg in range(20):
            if len(dat_cond[i_d, model][trg, 'ix']) > 0:
                IX = np.hstack((dat_cond[i_d, model][trg, 'ix'])); 
                PRED = np.vstack((dat_cond[i_d, model][trg, 'pred'])); 
                cond_spec[IX, :] = PRED.copy()
                ix_all.append(IX)
            else:
                pass

        ix_all = np.hstack((ix_all))
        assert(len(ix_all) == tsk_spec.shape[0])

        ### Save the residuals ###
        rez_dict[i_d]['res', 'cond'] = true_spks - cond_spec.copy()
        
        _, data_temp, _, _, _ = generate_models_utils.get_spike_kinematics(animal,
            day, order_dict[i_d], 1)

        data_temp = util_fcns.add_targ_locs(data_temp, animal)

        #### Add the residuals 
        data_temp_rez['res_cond'] = rez_dict[i_d]['res','cond']
        data_temp_rez['res_tsk'] = rez_dict[i_d]['res', 'tsk']
        data_temp_rez['res_gen'] = rez_dict[i_d]['res', 'gen']


        variables = []
        for i_s, sv in enumerate(state_vars):
            if 'pos' in sv:
                variables.append(['posx_tm1', 'posy_tm1'])
            elif 'vel' in sv:
                variables.append(['velx_tm1', 'vely_tm1'])
            elif 'trg' in sv:
                variables.append(['trgx', 'trgy', 'centx', 'centy', 'obsx', 'obsy'])
            elif 'tsk' in sv:
                variables.append(['tsk'])

        variables = np.hstack((variables))
        import pdb; pdb.set_trace()
        ###### Add relevant stuff to rez dict
        rez_dict[i_d]['trg'] = data_temp['trg']
        rez_dict[i_d]['tsk'] = data_temp['tsk']
        rez_dict[i_d]['true_spks'] = true_spks.copy()
        rez_dict[i_d]['pred_spks_cond'] = cond_spec.copy()
        rez_dict[i_d]['pred_spks_tsk'] = tsk_spec.copy()
        rez_dict[i_d]['pred_spks_gen'] = dat[i_d, model][:, :, 2].copy()

        ################################################################################
        ############ For each of these --- fit a gen B, task B, cond specific B; #######
        for i_M, rez_mod in enumerate(['gen', 'tsk', 'cond']):
            predict_key = 'res_' + rez_mod
            pred_dict[i_d][predict_key] = {}
            
            for i_m, mod_type in enumerate(['gen', 'tsk', 'cond']):
                pred_dict[i_d][predict_key]['B'+mod_type] = np.zeros_like(data_temp_rez[predict_key])

                ### Get out the training / testing indices #####
                if mod_type in ['gen', 'tsk']:
                    test_ix, train_ix, mtype = generate_models_utils.get_training_testings_generalization(n_folds, data_temp)            
                else:
                    test_ix, train_ix, mtype = generate_models_utils.get_training_testings_condition_spec(n_folds, data_temp)

                IX_tot = [] 

                #### Aggregate state ####
                for i_fold, type_of_model_index in enumerate(mtype):
                    
                    if type_of_model_index < 0:
                        pass
                    
                    else:
                    
                        alpha_spec = 0; 
                        model_ = fit_ridge(data_temp_rez[predict_key][train_ix[i_fold], :], data_temp.iloc[train_ix[i_fold]], variables, alpha=alpha_spec)
                        
                        ### Plot predictions 
                        model_, predY = generate_models_utils.sklearn_mod_to_ols(model_, data_temp.iloc[test_ix[i_fold]], 
                            variables, predict_key, testY = data_temp_rez[predict_key][test_ix[i_fold], :])

                        ### Ok, now figure out how to store these guys ####
                        if mod_type == 'gen' and type_of_model_index == 2:
                            pred_dict[i_d][predict_key]['B'+mod_type][test_ix[i_fold], :] = predY.copy()
                            IX_tot.append(test_ix[i_fold])
                        
                        ### CO model: only fill in the CO task; 
                        elif mod_type == 'tsk' and type_of_model_index == 0:
                            ix_co = np.nonzero(data_temp['tsk'][test_ix[i_fold]] == 0)[0]
                            pred_dict[i_d][predict_key]['B'+mod_type][test_ix[i_fold][ix_co], :] = predY[ix_co, :].copy()
                            IX_tot.append(test_ix[i_fold][ix_co])
                        
                        ### OBS model -- only fill in the obstacle task 
                        elif mod_type == 'tsk' and type_of_model_index == 1:
                            ix_ob = np.nonzero(data_temp['tsk'][test_ix[i_fold]] == 1)[0]
                            pred_dict[i_d][predict_key]['B'+mod_type][test_ix[i_fold][ix_ob], :] = predY[ix_ob, :].copy()
                            IX_tot.append(test_ix[i_fold][ix_ob])
                        
                        elif mod_type == 'cond':
                            assert(np.all(data_temp['trg'][test_ix[i_fold]] == np.mod(type_of_model_index, 10)))
                            assert(np.all(data_temp['tsk'][test_ix[i_fold]] == type_of_model_index / 10))
                            pred_dict[i_d][predict_key]['B'+mod_type][test_ix[i_fold], :] = predY.copy()
                            IX_tot.append(test_ix[i_fold])

                ### Now aggregate together -- check taht all indices were covered ####
                IX_tot = np.hstack((IX_tot))
                if mod_type == 'gen':
                    assert(len(np.unique(IX_tot)) == data_temp_rez[predict_key].shape[0])
                else:
                    assert(len(IX_tot) == data_temp_rez[predict_key].shape[0])
                    assert(len(np.unique(IX_tot)) == len(IX_tot))

    #### Save stuff for later; 
    D = dict(pred_dict=pred_dict, rez_dict = rez_dict)
    pickle.dump(D, open(analysis_config.config[animal+'_pref'] + 'res_model_fit_state.pkl', 'wb'))

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
        Y0 = y_train[ix0, :]; Y1 = y_train[ix1, :];
        
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

def return_variables_associated_with_model_var(model_var_list, include_action_lags, nneur):
    variables_list = []
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
        
        ### Add push -- if include_action_lags, push is added at lag in model_vars (lag ix), else if 'psh_1' in model name, push at time 0 is added
        ### else, no push is added
        push_model_nms, push_model_str = generate_models_utils.lag_ix_2_var_nm(model_vars, 'psh', include_action_lags = include_action_lags, model_nm = model_nm) 

        ### Past neural activity

        if include_past_yt > 0:
            neur_nms1, neur_model_str1 = generate_models_utils.lag_ix_2_var_nm(model_vars, 'neur', nneur)#, np.arange(-1*include_past_yt, 0))
        else:
            neur_nms1 = []; 

        ### Next neural activity; 
        if include_fut_yt > 0:
            neur_nms2, neur_model_str2 = generate_models_utils.lag_ix_2_var_nm(model_vars, 'neur', nneur)#, np.arange(1, 1*include_fut_yt + 1))
        else:
            neur_nms2 = []; 

        ### Get all the variables together ###
        variables = np.hstack(( [vel_model_nms, pos_model_nms, tg_model_nms, tsk_model_nms, push_model_nms, neur_nms1, neur_nms2] ))
        variables_list.append(variables)
    return variables_list

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
                        ### Only take the first value -- all are the same; 
                        tmp = tbl[day_index]['r2_pop']
                        mn = np.mean(tmp)
                        assert(np.allclose(tmp-mn, np.zeros((len(tmp), 1))))
                        alp_i.append(mn)

                ### plot alpha vs. mean day: 
                if skip_plots:
                    pass
                else:
                    ax.bar(np.log10(alpha_float), np.nanmean(np.hstack((alp_i))), width=.3)
                    ax.errorbar(np.log10(alpha_float), np.nanmean(np.hstack((alp_i))), np.nanstd(np.hstack((alp_i)))/np.sqrt(len(np.hstack((alp_i)))),
                        marker='|', color='k')

                ### max_alpha[i_d, model_nm] = []
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


#### Decoder UTILS ####
def get_KG_decoder_grom(day_ix):
    co_obs_dict = pickle.load(open(analysis_config.config['grom_pref']+'co_obs_file_dict.pkl'))
    input_type = analysis_config.data_params['grom_input_type']

    ### First CO task for that day: 
    te_num = input_type[day_ix][0][0]
    dec = co_obs_dict[te_num, 'dec']
    decix = dec.rfind('/')
    decoder = pickle.load(open(analysis_config.config['grom_pref']+dec[decix:]))
    F, KG = decoder.filt.get_sskf()
    KG_potent = KG[[3, 5], :]; # 2 x N
    KG_null = scipy.linalg.null_space(KG_potent) # N x (N-2)
    KG_null_proj = np.dot(KG_null, KG_null.T)

    ## Get KG potent too; 
    U, S, Vh = scipy.linalg.svd(KG_potent); #[2x2, 2, 44x44]
    Va = np.zeros_like(Vh)
    Va[:2, :] = Vh[:2, :]
    KG_potent_orth = np.dot(Va.T, Va)

    return KG_potent, KG_null_proj, KG_potent_orth

def generate_KG_decoder_jeev():
    KG_approx = dict() 

    ### Task filelist ###
    filelist = file_key.task_filelist
    days = len(filelist)
    binsize_ms = 5.
    important_neuron_ix = dict()

    for day in range(days):

        # open the thing: 
        Ps = pickle.load(open(analysis_config.config['jeev_pref']+'/jeev_KG_approx_feb_2019_day'+str(day)+'.pkl'))
        
        # Get bin spikes for this task entry:
        bin_spk_all = []
        cursor_KG_all = []
        P_mats = []

        for task in range(2):
            te_nums = filelist[day][task]
            
            for te_num in te_nums:
                bin_spk, targ_i_all, targ_ix, trial_ix_all, decoder_all, unbinned, exclude = ppf_pa.get_jeev_trials_from_task_data(te_num, 
                    binsize=binsize_ms/1000.)

                indices = []
                for j, (i0, i1) in enumerate(unbinned['ixs']):
                    indices.append(np.arange(i0, i1) - unbinned['start_index_overall'])
                
                for j in np.hstack((indices)):
                    P_mats.append(np.array(Ps[te_num][j]))

                print unbinned['start_index_overall'], i1, i1 -  unbinned['start_index_overall']

                bin_spk_all.extend([bs for ib, bs in enumerate(bin_spk) if ib not in exclude])
                cursor_KG_all.extend([kg for ik, kg in enumerate(decoder_all) if ik not in exclude])

        bin_spk_all = np.vstack((bin_spk_all))
        P_mats = np.dstack((P_mats))

        cursor_KG_all = np.vstack((cursor_KG_all))[:, [3, 5]]
        
        # De-meaned cursor: 
        cursor_KG_all_demean = (cursor_KG_all - np.mean(cursor_KG_all, axis=0)[np.newaxis, :]).T
        
        # Fit a matrix that predicts cursor KG all from binned spike counts
        KG = np.mat(np.linalg.lstsq(bin_spk_all, cursor_KG_all)[0]).T
        
        cursor_KG_all_reconst = KG*bin_spk_all.T
        vx = np.sum(np.array(cursor_KG_all[:, 0] - cursor_KG_all_reconst[0, :])**2)
        vy = np.sum(np.array(cursor_KG_all[:, 1] - cursor_KG_all_reconst[1, :])**2)
        R2_best = 1 - ((vx + vy)/np.sum(cursor_KG_all_demean**2))
        print 'KG est: shape: ', KG.shape, ', R2: ', R2_best

        KG_approx[day] = KG.copy();
        KG_approx[day, 'R2'] = R2_best;

    ### Save this: 
    pickle.dump(KG_approx, open(analysis_config.config['jeev_pref']+'jeev_KG_approx_fit.pkl', 'wb'))

def get_KG_decoder_jeev(day_ix):
    kgs = pickle.load(open(analysis_config.config['jeev_pref']+'jeev_KG_approx_fit.pkl', 'rb'))
    KG = kgs[day_ix]
    KG_potent = KG.copy(); #$[[3, 5], :]; # 2 x N
    KG_null = scipy.linalg.null_space(KG_potent) # N x (N-2)
    KG_null_proj = np.dot(KG_null, KG_null.T)
    
    U, S, Vh = scipy.linalg.svd(KG_potent); #[2x2, 2, 44x44]
    Va = np.zeros_like(Vh)
    Va[:2, :] = Vh[:2, :]
    KG_potent_orth = np.dot(Va.T, Va)

    return KG_potent, KG_null_proj, KG_potent_orth

def get_decomp_y(KG_null, KG_pot, y_true, only_null = True, only_potent=False):
    if only_null:
        y_proj = np.dot(KG_null, y_true.T).T 

    elif only_potent:
        y_proj = np.dot(KG_pot, y_true.T).T

    ## Make sure kg*y_null is zero
    #np_null = np.dot(KG, y_null.T).T
    #f,ax = plt.subplots()
    #plt.plot(np_null[:, 0])
    #plt.plot(np_null[:, 1])
    #plt.title('Null Proj throuhg KG')

    # Do SVD on KG potent and ensure match to KG*y_true? 
    #N = y_true.shape[1]
    #U, S, Vh = scipy.linalg.svd(KG.T); #[2x2, 2, 44x44]

    # Take rows to Vh to make a projection basis: 
    #y_test = np.dot(KG, np.dot(U, np.dot(U.T, y_true.T)))
    #y_test2 = np.dot(KG, y_true.T)

    # f, ax = plt.subplots(ncols = 2)
    # ax[0].plot(y_test[0, :], y_test2[0,:], '.')
    # ax[1].plot(y_test[1, :], y_test2[1,:], '.')

    return y_proj; 
