#### Methods to generate models used to predict neural activity ####

from analysis_config import config 
import analysis_config
from online_analysis import generate_models_utils, generate_models_list, util_fcns
import resim_ppf
from resim_ppf import file_key, ppf_pa

import numpy as np
import matplotlib.pyplot as plt
import datetime
import gc
import tables, pickle
from collections import defaultdict

import sklearn.decomposition as skdecomp
from sklearn.linear_model import Ridge
import scipy
import scipy.io as sio



import sys, os
py_ver = sys.version

if '3.6.15' in py_ver: 
    from online_analysis import slds_tools
    from autograd.scipy.special import logsumexp

else:
    pass
    #from pylds.models import DefaultLDS
    #from pylds.states import kalman_filter



######### STEP 1 -- Get alpha value #########
def sweep_alpha_all(run_alphas=True, model_set_number = 3,
    fit_intercept = True, within_bin_shuffle = False,
    normalize_ridge_vars = False, keep_bin_spk_zsc = False, 
    null = False):

    """ Sweep different alphas for Ridge regression --> see which one is best
    
    Args:
        run_alphas (bool, optional): default true -- Whether or not the alphas actually need to be run via sweep_ridge_alpha
        model_set_number (int, optional): Model number -- see generate_models_list for model options
        fit_intercept (bool, optional): whether to fit the intercept of the ridge regression; 
        within_bin_shuffle (bool, optional): fit alpha on shuffled data (shuffle activity for commands)
    
    Raises:
        Exception: if run_alphas is false
    """
    max_alphas = dict(grom=[], jeev=[], home=[])

    alphas = []; 
    for i in range(-4, 7):
        if i == 4:
            for j in range(2, 11):
                alphas.append(float(j)*10**3)
        else:
            alphas.append((1./4)*10**i)
            alphas.append((2./4.)*10**i)
            alphas.append((3./4.)*10**i)
            alphas.append(1.*10**i)

    

    ndays = dict(grom=9, jeev=4, home=5) # for testing only; 

    for animal in ['jeev', 'grom']: #home
        if run_alphas:
            h5_name = sweep_ridge_alpha(animal=animal, alphas = alphas, model_set_number = model_set_number, 
                ndays = ndays[animal], fit_intercept = fit_intercept, within_bin_shuffle = within_bin_shuffle,
                normalize_ridge_vars = normalize_ridge_vars, keep_bin_spk_zsc = keep_bin_spk_zsc, null=null)
        else:
            #raise Exception('deprecated')
            h5_name = config['%s_pref'%animal] + '%s_sweep_alpha_days_models_set%d.h5' %(animal, model_set_number)

        max_alpha = plot_sweep_alpha(animal, alphas=alphas, model_set_number=model_set_number, ndays=ndays[animal],
            fit_intercept = fit_intercept, within_bin_shuffle = within_bin_shuffle, null=null)
        
        max_alphas[animal].append(max_alpha)

    if fit_intercept:
        if within_bin_shuffle:
            sff = '_shuff'
        else:
            sff = ''

        if normalize_ridge_vars: 
            sff1 = '_ridge_norm'
        else:
            sff1 = ''

        if keep_bin_spk_zsc:
            sff2 = '_zsc'
        else:
            sff2 = ''

        if null: 
            sff3 = '_null'
        else:
            sff3 = ''

        pickle.dump(max_alphas, open(config['grom_pref'] + 'max_alphas_ridge_model_set%d%s%s%s%s.pkl' %(model_set_number, sff, sff1, sff2, sff3), 'wb'))
        
    else:
        pickle.dump(max_alphas, open(config['grom_pref'] + 'max_alphas_ridge_model_set%d_no_intc.pkl' %model_set_number, 'wb'))
    
    print('Done with max_alphas_ridge')

def sweep_ridge_alpha(alphas, animal='grom', n_folds = 5, history_bins_max = 1, 
    model_set_number = 1, ndays=None, fit_intercept = True, within_bin_shuffle = False,
    normalize_ridge_vars = False, keep_bin_spk_zsc = False, null = False):
    """Summary
    
    Args:
        alphas (list): list of alphas to test
        animal (str, optional): animal name; 
        n_folds (int, optional): default = 5
        history_bins_max (int, optional): default = 1; dictates how lags to use to fit; 
        model_set_number (int, optional): form generate_models_list; 
        ndays (int, optional): how many days to fit (if None, fits all)
        fit_intercept (bool, optional): whether to fit intercept in Ridge; 
        within_bin_shuffle (bool, optional): whther to fit on shuffled data (shuffle activity for commands)
    
    Returns:
        hdf_filename: name of file where alphas are stored; 
    """

    #### Get models from here ###
    model_var_list, predict_key, include_action_lags, _ = generate_models_list.get_model_var_list(model_set_number)

    x = datetime.datetime.now()

    if fit_intercept:
        if within_bin_shuffle:
            hdf_filename = animal + '_sweep_alpha_days_models_set%d_shuff.h5' %model_set_number
        else:
            hdf_filename = animal + '_sweep_alpha_days_models_set%d.h5' %model_set_number
    else:
        hdf_filename = animal + '_sweep_alpha_days_models_set%d_no_intc.h5' %model_set_number

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
    order_dict = analysis_config.data_params['%s_ordered_input_type'%animal]
    input_type = analysis_config.data_params['%s_input_type'%animal]

    if ndays is None:
        pass
    else:
        print(('Only using %d days' %ndays))
        input_type = [input_type[i] for i in range(ndays)]
        order_dict = [order_dict[i] for i in range(ndays)]

    #### Open file 
    h5file = tables.openFile(hdf_filename, mode="w", title=animal+'_tuning')
    mFR = {}
    sdFR = {}

    #### For each day 
    for i_d, day in enumerate(input_type):
        
        # Get spike data
        ####### data is everything; 
        ###### data_temp / sub_pikes / sub_spk_temp_all / sub_push_all --> all spks / actions 
        ##### but samples only using history_bins:nT-history_bins within trial 
        if animal == 'home':
            day = [day]
            order_d = [order_dict[i_d]]
        else:
            order_d = order_dict[i_d]

        data, data_temp, sub_spikes, sub_spk_temp_all, sub_push_all = generate_models_utils.get_spike_kinematics(animal,
            day, order_d, history_bins_max, day_ix = i_d, within_bin_shuffle = within_bin_shuffle, 
            keep_bin_spk_zsc = keep_bin_spk_zsc, null = null)
        
        if animal == 'home':
            KG, dec_mFR, dec_sdFR = util_fcns.get_decoder(animal, i_d)
        else:
            dec_mFR = None 
            dec_sdFR = None
            KG = util_fcns.get_decoder(animal, i_d)


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

                print(model_var_list_i)
                print('Variables; ')
                print(variables)

                ### Unpack model_var_list; 
                _, model_nm, _, _, _ = model_var_list_i; 

                if model_nm in ['identity_dyn', 'diagonal_dyn']:
                    pass
                else:
                    ####### HERE ######
                    #############################
                    ### Model with parameters ###
                    #############################
                    for ia, alpha in enumerate(alphas):

                        ### Model ###
                        model_ = fit_ridge(data_temp_dict[predict_key], data_temp_dict, variables, alpha=alpha,
                            fit_intercept = fit_intercept, model_nm= model_nm, normalize_vars = normalize_ridge_vars)
                        str_alpha = str(alpha)
                        str_alpha = str_alpha.replace('.','_')
                        name = model_nm + '_alpha_' + str_alpha

                        ### add info ###
                        ##### h5file is an HDF table, model_ is the output model, i_d is day, first day 
                        ##### model name 
                        ##### data_temp_dict_test 
                        h5file, model_, _ = generate_models_utils.h5_add_model(h5file, model_, i_d, first=i_d==0, model_nm=name, 
                            test_data = data_temp_dict_test, fold = i_fold, xvars = variables, predict_key = predict_key, KG = KG,
                            fit_intercept = fit_intercept, keep_bin_spk_zsc=keep_bin_spk_zsc, decoder_params=dict(dec_mFR=dec_mFR,
                                dec_sdFR = dec_sdFR))

    h5file.close()
    print(('H5 File Done: ', hdf_filename))
    return hdf_filename

######## IF LDS or SLDS -- step 1 get x-validated dimensionality #####
def sweep_dim_all(model_set_number = 11, history_bins_max = 1, within_bin_shuffle = False,
    n_folds = 5, highDsweep = False):
    '''
    here the number of dimensions swept for LDS is # latent dimensions, 
    SLDS is # of discrete dynamical systems fit (always try to fit full dimensionality)
    '''

    model_var_list, predict_key, include_action_lags, _ = generate_models_list.get_model_var_list(model_set_number)
    
    if highDsweep:
        animals = ['grom']
    else:
        animals = ['jeev', 'grom']

    for animal in animals:
        max_LL_dim = dict()

        if animal == 'grom':
            order_dict = analysis_config.data_params['grom_ordered_input_type']
            input_type = analysis_config.data_params['grom_input_type']

        elif animal == 'jeev':
            order_dict = analysis_config.data_params['jeev_ordered_input_type']
            input_type = analysis_config.data_params['jeev_input_type']

        ##### For each day ####
        for i_d, day in enumerate(input_type):
            
            print('##############################')
            print(('########## DAY %d ##########' %(i_d) ))
            print('##############################')
            
            # Get spike data from data fcn
            data, data_temp, sub_spikes, sub_spk_temp_all, sub_push_all = generate_models_utils.get_spike_kinematics(animal, day, 
                order_dict[i_d], history_bins_max, within_bin_shuffle = within_bin_shuffle,
                day_ix = i_d)

            #### This gets trials instead of time points ###
            trl_test_ix, trl_train_ix, type_of_model = generate_models_utils.get_training_testings_generalization_LDS_trial(n_folds, data_temp,
                match_task_spec_n = True)

            #### CO/ OBS / GEN vs. folds 
            nneur = sub_spk_temp_all.shape[2]

            if model_set_number == 11: 
                if highDsweep:
                    maxDim = np.min([nneur - 2, 36])
                    dims = np.arange(22, maxDim, 2)
                    LL = np.zeros((3, n_folds, len(dims) )) + np.nan
                else:
                    maxDim = nneur - 4; 
                    if maxDim > 50: 
                        dims = np.hstack((np.arange(2, 20, 2), np.arange(20, 5, 50), np.arange(50, 10, maxDim)))
                    elif maxDim > 20: 
                        dims = np.hstack((np.arange(2, 20, 2), np.arange(20, 5, maxDim)))
                    else: 
                        dims = np.arange(2, maxDim, 2)

            elif model_set_number == 13: 
                ## Number of states to sweep -- 
                dims = [1, 3, 5, 7, 9, 11, 13, 15]
    
            #LL = np.zeros((3, n_folds, len(dims) )) + np.nan
            ### pk, 4-2022 removed task 0, task 1, just care about mixed; 
            LL = np.zeros((1, n_folds, len(dims) )) + np.nan

            for i_fold, type_of_model_index in enumerate(type_of_model):

                if type_of_model_index == 2: 
                    #### Get variable names #####
                    variables_list = return_variables_associated_with_model_var(model_var_list, include_action_lags, nneur)
                    
                    ### For each variable in the model: 
                    for _, (variables, model_var_list_i) in enumerate(zip(variables_list, model_var_list)):

                        check_variables_order(variables, nneur)

                        ### These are teh params; 
                        _, model_nm, _, _, _ = model_var_list_i

                        print(('Start dims fold %d, type %d' %(i_fold, type_of_model_index)))
                        for i_n, n_dim in enumerate(dims):

                            if model_set_number == 11: 
                                model_ = fit_LDS(data_temp, variables, trl_train_ix[i_fold], n_dim_latent = n_dim)
                                ### Add to log likelihood
                                #LL[type_of_model_index, i_fold % 5, i_n] = model_.lls[-1]
                                LL[0, i_fold % 5, i_n] = model_.lls[-1]

                            elif model_set_number == 13: 
                                exp_log_prob = slds_tools.fit_slds(data_temp, variables, trl_train_ix[i_fold], n_dim)
                                LL[0, i_fold%5, i_n] = exp_log_prob
                            
            maxll = np.nanmean(LL, axis=1)

            ##### max sure ####
            assert(np.all(np.isnan(maxll) == False))
            max_dim = np.argmax(maxll, axis=1)

            max_LL_dim[i_d, 'LLs'] = LL.copy()
            max_LL_dim[i_d, 'dims'] = dims.copy()

            #assert(len(max_dim) == 3)
            max_LL_dim[i_d] = dims[max_dim]

        ### Save data 
        #if highDsweep:
        if model_set_number == 11:
            pickle.dump(max_LL_dim, open(analysis_config.config[animal+'_pref'] + 'LDS_maxL_ndims.pkl', 'wb'))
        elif model_set_number == 13: 
            pickle.dump(max_LL_dim, open(analysis_config.config[animal+'_pref'] + 'SLDS_sweep_K.pkl', 'wb'))



######## STEP 2 -- Fit the models ###########
### Main tuning function -- run this for diff animals; 
def model_individual_cell_tuning_curves(hdf_filename='_models_to_pred_mn_diffs', 
    animal='grom', 
    n_folds = 5, 
    norm_neur = False, 
    normalize_ridge_vars = False,
    return_models = True, 
    model_set_number = 8, 
    ndays = None,
    include_null_pot = False,
    only_potent_predictor = False,
    only_command = False,
    fit_intercept = True,

    fit_task_specific_model_test_task_spec = False,
    fit_task_spec_and_general = False,
    match_task_spec_n = False,
    fit_condition_spec_no_general = False, 

    full_shuffle = False,
    within_bin_shuffle = False, 
    shuff_id = 0,
    add_model_to_datafile = True,
    task_demean = False,
    gen_demean = False,
    alpha_always_zero = False,
    latent_dim = 'full', 
    keep_bin_spk_zsc = False, 
    null_predictor = False,
    null_predictor_w_null_alpha = False,

    LDS_skip_task_specific = True,
    window_size_LDS = 'full'):
    
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
    Modified 6/10/20 to include "add model to datafile" as an optional thing; think this is teh thing messing up loading sometimes; 

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
        -- task_demean -- option to fit task demeaned spikes instead of full spiking activity; 

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
            hdf_filename = pref + animal + hdf_filename + '_model_set%d_task_spec' %model_set_number
        else:
            hdf_filename = pref + animal + hdf_filename + '_model_set%d' %model_set_number

        if full_shuffle:
            hdf_filename = hdf_filename + '_full_shuff%d' %shuff_id
        
        elif within_bin_shuffle:
            hdf_filename = hdf_filename + '_within_bin_shuff%d' %shuff_id

        if task_demean:
            hdf_filename = hdf_filename + '_tsk_demean'
            assert(not gen_demean)

        if gen_demean:
            hdf_filename = hdf_filename + '_gen_demean'
            assert(not task_demean)

        if alpha_always_zero:
            hdf_filename = hdf_filename + '_alphazero'

        if null_predictor: 
            hdf_filename = hdf_filename + '_null'

        elif null_predictor_w_null_alpha:
            hdf_filename = hdf_filename + '_null_alpha'


        if fit_intercept:
            hdf_filename = hdf_filename + '.h5'
        else:
            hdf_filename = hdf_filename + '_no_intc.h5'

    ### Place to save models: 
    model_data = dict(); 

    ### Get the ridge dict: 
    if model_set_number == 11: 
        ### LDS model w/ latent ###
        if latent_dim == 'full':
            pass
        elif type(latent_dim) is int:
            pass
        else:
            dim_dict = pickle.load(open(analysis_config.config[animal+'_pref'] + 'LDS_maxL_ndims_lowD.pkl', 'rb'))

    else:
        if model_set_number == 12 and normalize_ridge_vars:
            ridge_dict = pickle.load(open(analysis_config.config['grom_pref'] + 'max_alphas_ridge_model_set%d_ridge_norm.pkl' %model_set_number, 'rb'))

        else:
            if fit_intercept:
                if within_bin_shuffle:
                    ridge_dict = pickle.load(open(analysis_config.config['grom_pref'] + 'max_alphas_ridge_model_set%d_shuff.pkl' %model_set_number, 'rb')); 
                else:   
                    if null_predictor: 
                        print(' SKIP Load null dict')
                        ridge_dict = pickle.load(open(analysis_config.config['grom_pref'] + 'max_alphas_ridge_model_set%d.pkl' %model_set_number, 'rb')); 
                    elif null_predictor_w_null_alpha:
                        print('Loading null alha')
                        ridge_dict = pickle.load(open(analysis_config.config['grom_pref'] + 'max_alphas_ridge_model_set%d_null.pkl' %model_set_number, 'rb')); 
                        
                    else:
                        ridge_dict = pickle.load(open(analysis_config.config['grom_pref'] + 'max_alphas_ridge_model_set%d.pkl' %model_set_number, 'rb')); 

                ##### HOMER specific alphas #########
                if keep_bin_spk_zsc: 
                    if within_bin_shuffle:
                        homer_ridge_dict = pickle.load(open(analysis_config.config['grom_pref'] + 'max_alphas_ridge_model_set%d_shuff_zsc.pkl' %model_set_number, 'rb'))
                    else:
                        homer_ridge_dict = pickle.load(open(analysis_config.config['grom_pref'] + 'max_alphas_ridge_model_set%d_zsc.pkl' %model_set_number, 'rb'))
                else:
                    if within_bin_shuffle:
                        homer_ridge_dict = pickle.load(open(analysis_config.config['grom_pref'] + 'max_alphas_ridge_model_set%d_shuff.pkl' %model_set_number, 'rb'))
                    else:
                        homer_ridge_dict = pickle.load(open(analysis_config.config['grom_pref'] + 'max_alphas_ridge_model_set%d.pkl' %model_set_number, 'rb')); 

            else:
                ridge_dict = pickle.load(open(analysis_config.config['grom_pref'] + 'max_alphas_ridge_model_set%d_no_intc.pkl' %model_set_number, 'rb')); 

    for obj in gc.get_objects():   # Browse through ALL objects
        if isinstance(obj, tables.File):   # Just HDF5 files
            try:
                obj.close()
            except:
                pass #

    tuning_dict = {}
    order_dict = analysis_config.data_params['%s_ordered_input_type'%animal]
    input_type = analysis_config.data_params['%s_input_type'%animal]

    if ndays is None:
        pass
    else:
        order_dict = [order_dict[i] for i in range(ndays)]
        input_type = [input_type[i] for i in range(ndays)]

    h5file = tables.openFile(hdf_filename, mode="w", title=animal+'_tuning')

    if task_prediction:
        task_dict_file = {} 
        if 'task_pred_prefix' not in list(kwargs.keys()):
            raise Exception

    mFR = {}
    sdFR = {}

    ##### For each day ####
    for i_d, day in enumerate(input_type):
        
        print('##############################')
        print(('########## DAY %d ##########' %(i_d) ))
        print('##############################')
        if animal == 'home':
            day = [day]
            order_d = [order_dict[i_d]]
        else:
            order_d = order_dict[i_d]

        if animal != 'home':
            assert(not keep_bin_spk_zsc)

        null_pred = np.logical_or(null_predictor, null_predictor_w_null_alpha)

        ### By default returns standard binned spike counts
        data, data_temp, sub_spikes, sub_spk_temp_all, sub_push_all = generate_models_utils.get_spike_kinematics(animal, day, 
            order_d, history_bins_max, within_bin_shuffle = within_bin_shuffle, keep_bin_spk_zsc = keep_bin_spk_zsc,
            day_ix = i_d, null = null_pred)
        
        print(('R2 again, %.2f' %generate_models_utils.quick_reg(sub_spikes, sub_push_all)))
        print(('R2 est spks dyn, %.2f' %generate_models_utils.quick_reg(sub_spikes[1:, :], sub_push_all[:-1, :])))
        
        models_to_include = []
        for m in model_var_list:
            models_to_include.append(m[1])

        ### Get kalman gain etc. 
        dec_sdFR = None
        dec_mFR = None
        if animal == 'grom':
            KG, KG_null_proj, KG_potent_orth = get_KG_decoder_grom(i_d)
            if null_pred:
                assert(np.allclose(np.zeros_like(sub_push_all), np.dot(KG, sub_spikes.T).T))
        elif animal == 'home':
            KG, KG_null_proj, KG_potent_orth, dec_mFR, dec_sdFR = get_KG_decoder_home(i_d)
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

            ##### Model data / task #####
            model_data[i_d, 'task'] = np.squeeze(np.array(data_temp['tsk']))

            #### Task Demean ######
            if task_demean:
                ix_co = np.nonzero(model_data[i_d, 'task'] == 0)[0]
                ix_ob = np.nonzero(model_data[i_d, 'task'] == 1)[0]
                
                ### Get the means ###
                co_spks_mn = np.mean(sub_spikes[ix_co, :], axis=0)
                ob_spks_mn = np.mean(sub_spikes[ix_ob, :], axis=0)

                ### save the means ### 
                model_data[i_d, 'spks_co_mn'] = co_spks_mn
                model_data[i_d, 'spks_ob_mn'] = ob_spks_mn

                ### subtract the means 
                sub_spikes[ix_co, :] = sub_spikes[ix_co, :] - co_spks_mn[np.newaxis, :]
                sub_spikes[ix_ob, :] = sub_spikes[ix_ob, :] - ob_spks_mn[np.newaxis, :]
            
            elif gen_demean:
                spks_mn = np.mean(sub_spikes, axis=0)
                sub_spikes = sub_spikes - spks_mn[np.newaxis, :]
                model_data[i_d, 'spks_mn'] = spks_mn

            ### Want to save neural push, task, target 
            model_data[i_d, 'spks'] = sub_spikes.copy();
            model_data[i_d, 'trg'] = np.squeeze(np.array(data_temp['trg']))
            model_data[i_d, 'np'] = np.squeeze(np.array(sub_push_all))
            model_data[i_d, 'bin_num'] = np.squeeze(np.array(data_temp['bin_num']))
            model_data[i_d, 'pos'] = np.vstack((np.array(data_temp['posx_tm0']), np.array(data_temp['posy_tm0']))).T
            model_data[i_d, 'vel'] = np.vstack((np.array(data_temp['velx_tm0']), np.array(data_temp['vely_tm0']))).T
            model_data[i_d, 'vel_tm1'] = np.vstack((np.array(data_temp['velx_tm1']), np.array(data_temp['vely_tm1']))).T
            model_data[i_d, 'pos_tm1'] = np.vstack((np.array(data_temp['posx_tm1']), np.array(data_temp['posy_tm1']))).T
            model_data[i_d, 'trl'] = np.squeeze(np.array(data_temp['trl']))
            model_data[i_d, 'day_bin_ix'] = np.squeeze(np.array(data_temp['day_bin_ix']))
            #model_data[i_d, 'day_bin_ix_shuff'] = np.squeeze(np.array(data_temp['day_bin_ix_shuff']))
            
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
                
                elif only_command: 
                    model_data[i_d, mod] = np.zeros((sub_spikes.shape[0], 2)) 
                else:
                    model_data[i_d, mod] = np.zeros_like(sub_spikes) 

                if include_null_pot:
                    if fit_condition_spec_no_general:
                        model_data[i_d, mod, 'null'] = defaultdict(list)
                        model_data[i_d, mod, 'pot'] = defaultdict(list)
                    
                    elif fit_task_spec_and_general:
                        nT, nn = sub_spikes.shape 
                        model_data[i_d, mod, 'null'] = np.zeros((nT, nn, 3))
                        model_data[i_d, mod, 'pot'] = np.zeros((nT, nn, 3))    
                    else:
                        ### if just use null / potent parts of predictions and propogate those guys
                        model_data[i_d, mod, 'null'] = np.zeros_like(sub_spikes)
                        model_data[i_d, mod, 'pot'] = np.zeros_like(sub_spikes)

        if norm_neur:
            print('normalizing neurons!')
            mFR[i_d] = np.mean(sub_spikes, axis=0)
            sdFR[i_d] = np.std(sub_spikes, axis=0)
            sdFR[i_d][sdFR[i_d]==0] = 1
            sub_spikes = ( sub_spikes - mFR[i_d][np.newaxis, :] ) / sdFR[i_d][np.newaxis, :]


        #### 
        #### Get training / testing sets split up --- test on 80% one task, test on 20% same tasks 20% other task
        if fit_task_spec_and_general:

            #### LDS -- keep the trials together; 
            if model_set_number == 11:
                #### This gets trials instead of time points ###
                trl_test_ix, trl_train_ix, type_of_model = generate_models_utils.get_training_testings_generalization_LDS_trial(n_folds, data_temp,
                    match_task_spec_n = match_task_spec_n, skip_task_specific = LDS_skip_task_specific)
                test_confirm = {}
                for tmp in np.unique(type_of_model):
                    test_confirm[tmp] = []

            else:
                ### Regular regression ###
                test_ix, train_ix, type_of_model = generate_models_utils.get_training_testings_generalization(n_folds, data_temp,
                    match_task_spec_n = match_task_spec_n)
        
        elif fit_condition_spec_no_general:
            test_ix, train_ix, type_of_model = generate_models_utils.get_training_testings_condition_spec(n_folds, data_temp)
            model_data[i_d, 'type_of_model'] = type_of_model
        
        ####### Get the test / train indices balanced over both tasks; 
        else:
            test_ix, train_ix = generate_models_utils.get_training_testings(n_folds, data_temp)
            type_of_model = np.zeros((n_folds, ))

        ############### Iterate through the folds ###################
        for i_fold, type_of_model_index in enumerate(type_of_model):

            if type_of_model_index < 0:
                pass
            else:
                nneur = sub_spk_temp_all.shape[2]

                if model_set_number == 11: 
                    assert(type_of_model_index in [0, 1, 2])
                    if latent_dim == 'full':
                        ndims = 'full'
                    elif type(latent_dim) is int: 
                        ndims = latent_dim
                    else:
                        ndims = dim_dict[i_d][type_of_model_index]
                else:
                    ### TEST DATA ####
                    data_temp_dict_test = panda_to_dict(data_temp.iloc[test_ix[i_fold]])
                    data_temp_dict_test['spks'] = sub_spikes[test_ix[i_fold]]
                    for ntmp in range(nneur): assert(np.allclose(data_temp_dict_test['spks'][:, ntmp],data_temp_dict_test['spk_tm0_n%d'%ntmp]))

                    data_temp_dict_test['pshy'] = sub_push_all[test_ix[i_fold], 1]
                    data_temp_dict_test['pshx'] = sub_push_all[test_ix[i_fold], 0]
                    data_temp_dict_test['psh'] = np.hstack(( data_temp_dict_test['pshx'], data_temp_dict_test['pshy']))

                    ### TRAIN DATA ####
                    data_temp_dict = panda_to_dict(data_temp.iloc[train_ix[i_fold]])
                    data_temp_dict['spks'] = sub_spikes[train_ix[i_fold]]
                    for ntmp in range(nneur): assert(np.allclose(data_temp_dict['spks'][:, ntmp],data_temp_dict['spk_tm0_n%d'%ntmp]))
                    data_temp_dict['pshy'] = sub_push_all[train_ix[i_fold], 1]
                    data_temp_dict['pshx'] = sub_push_all[train_ix[i_fold], 0]
                    data_temp_dict['psh'] = np.hstack(( data_temp_dict['pshx'][:, np.newaxis], data_temp_dict['pshy'][:, np.newaxis]))

                    print(('R2 train ix: %.2f' %(generate_models_utils.quick_reg(data_temp_dict['spks'], data_temp_dict['psh']))))

                variables_list = return_variables_associated_with_model_var(model_var_list, include_action_lags, nneur)
                
                ### For each variable in the model: 
                for _, (variables, model_var_list_i) in enumerate(zip(variables_list, model_var_list)):

                    check_variables_order(variables, nneur)

                    ### These are teh params; 
                    _, model_nm, _, _, _ = model_var_list_i

                    if 'psh_2' in model_nm and null_pred:
                        print(('SKIPPING %s, null=%s'%(model_nm, str(null_pred))))
                    elif 'psh_2' in model_nm and only_command: 
                        print(('SKIPPING %s, null=%s'%(model_nm, str(only_command))))
                    else: 

                        ## Store the params; 
                        model_data[model_nm, 'variables'] = variables

                        if ridge:
                            if model_nm in ['identity_dyn']:
                                pass
                            
                            elif model_nm in ['hist_1pos_0psh_0spksm_1_spksp_0_latentLDS', 'hist_1pos_0psh_0spksm_1_spksp_0_smoothLDS']: 
                                #assert(fit_intercept == False)
                                assert(only_potent_predictor == False)
                                model_ = fit_LDS(data_temp, variables, trl_train_ix[i_fold], n_dim_latent = ndims,
                                    fit_intercept= fit_intercept)
                            
                            elif 'hist_1pos_0psh_0spksm_1_spksp_0_rsLDS' in model_nm:
                                posterior = slds_tools.fit_rslds(data_temp, variables, trl_train_ix[i_fold], n_dim_latent = ndims)
                            
                            else:
                                if model_nm == 'diagonal_dyn':
                                    alpha_spec = 0.
                                else:
                                    alpha_spec = ridge_dict[animal][0][i_d, model_nm]

                                    if null_predictor: 
                                        print(('Adjusting alpha by n-2 %.3f'%(float(nneur)/float(nneur-2))))
                                        alpha_spec = alpha_spec*(float(nneur)/float(nneur-2))
                                    
                                    elif null_predictor_w_null_alpha: 
                                        print('keeping null alpha')

                                    elif only_command: 
                                        alpha_spec = 0.

                                    if animal == 'home':
                                        alpha_spec = homer_ridge_dict[animal][0][i_d, model_nm]
                                        print(('Homer alpha %.1f' %(alpha_spec)))

                                if alpha_always_zero:
                                    alpha_spec = 0.
                                    print('alpha zero')

                                model_ = fit_ridge(data_temp_dict[predict_key], data_temp_dict, variables, alpha=alpha_spec, 
                                    only_potent_predictor = only_potent_predictor, only_command = only_command, KG = KG, KG_pot = KG_potent_orth, 
                                    fit_task_specific_model_test_task_spec = fit_task_specific_model_test_task_spec,
                                    fit_intercept = fit_intercept, model_nm = model_nm, nneur=nneur, 
                                    normalize_vars = normalize_ridge_vars, keep_bin_spk_zsc=keep_bin_spk_zsc)
                            
                            save_model = True

                        else:
                            raise Exception('Need to figure out teh stirng business again -- removed for clarity')
                            model_ = ols(st, data_temp_dict).fit()
                            save_model = False

                        if save_model:
                            if model_nm == 'identity_dyn':
                                pred_Y = identity_dyn(data_temp_dict_test, nneur)

                            elif model_nm == 'diagonal_dyn': 
                                pred_Y = pred_diag_cond(data_temp_dict_test, model_, nneur, KG)

                            elif model_nm == 'hist_1pos_0psh_0spksm_1_spksp_0_latentLDS':
                                pred_Y, test_ix = pred_LDS(data_temp, model_, variables, trl_test_ix[i_fold], i_fold, 
                                    window_size=window_size_LDS)
                                test_confirm[type_of_model_index].append(test_ix[i_fold])
                            
                            elif model_nm == 'hist_1pos_0psh_0spksm_1_spksp_0_smoothLDS':
                                ### setting variable name as "pred_Y" even though really mean "smooth Y"
                                pred_Y, test_ix = smooth_LDS(data_temp, model_, variables, trl_test_ix[i_fold], i_fold)
                                test_confirm[type_of_model_index].append(test_ix[i_fold])
                            
                            elif 'hist_1pos_0psh_0spksm_1_spksp_0_rsLDS' in model_nm: 
                                pred_Y, test_ix = slds_tools.smooth_data(data_temp, posterior, variables, trl_test_ix[i_fold], i_fold)

                            else:
                                h5file, model_, pred_Y = generate_models_utils.h5_add_model(h5file, model_, i_d, first=i_d==0, model_nm=model_nm, 
                                    test_data = data_temp_dict_test, fold = i_fold, xvars = variables, predict_key=predict_key, 
                                    only_potent_predictor = only_potent_predictor, only_command = only_command, KG_pot = KG_potent_orth, KG = KG,
                                    fit_task_specific_model_test_task_spec = fit_task_specific_model_test_task_spec,
                                    fit_intercept = fit_intercept, keep_bin_spk_zsc = keep_bin_spk_zsc, decoder_params = dict(dec_mFR=dec_mFR,
                                        dec_sdFR=dec_sdFR))

                            ### Save models, make predictions ####
                            ### Need to figure out which spikes are where:
                            if fit_task_specific_model_test_task_spec:
                                ix0 = np.nonzero(data_temp_dict_test['tsk'] == 0)[0]
                                ix1 = np.nonzero(data_temp_dict_test['tsk'] == 1)[0]
                                
                                model_data[i_d, model_nm][test_ix[i_fold][ix0], :] = np.squeeze(np.array(pred_Y[0]))
                                model_data[i_d, model_nm][test_ix[i_fold][ix1], :] = np.squeeze(np.array(pred_Y[1]))
                            
                            elif fit_task_spec_and_general:
                                model_data[i_d, model_nm][test_ix[i_fold], :, type_of_model_index] = np.squeeze(np.array(pred_Y))
                                
                                if model_nm == 'prespos_0psh_1spksm_0_spksp_0':
                                    r2tmp = util_fcns.get_R2(data_temp_dict_test['spks'], np.squeeze(np.array(pred_Y)))
                                    print(('R2 from model: %.4f' %(r2tmp)))

                            elif fit_condition_spec_no_general:
                                ### List the indices and the prediction and the fold: 
                                model_data[i_d, model_nm][type_of_model_index, 'ix'].append(test_ix[i_fold])
                                model_data[i_d, model_nm][type_of_model_index, 'pred'].append(np.squeeze(np.array(pred_Y)))
                                
                            else:
                                model_data[i_d, model_nm][test_ix[i_fold], :] = np.squeeze(np.array(pred_Y))
                                
                            ### Save model -- for use later.
                            
                            if add_model_to_datafile:
                                if model_nm != 'identity_dyn':
                                    if model_nm == 'hist_1pos_0psh_0spksm_1_spksp_0_latentLDS':
                                        model_data[i_d, model_nm, i_fold, type_of_model_index, 'modelA'] = model_.A;
                                        model_data[i_d, model_nm, i_fold, type_of_model_index, 'modelW'] = model_.sigma_states;
                                        model_data[i_d, model_nm, i_fold, type_of_model_index, 'modelC'] = model_.C;
                                        model_data[i_d, model_nm, i_fold, type_of_model_index, 'modelQ'] = model_.sigma_obs; 
                                        model_data[i_d, model_nm, i_fold, type_of_model_index, 'modelkeepix'] = model_.keep_ix; 
                                        model_data[i_d, model_nm, i_fold, type_of_model_index, 'model_spikemn'] = model_.spike_mn; 
                                        
                                    else:
                                        print('Adding model and test_indices') 
                                        model_data[i_d, model_nm, i_fold, type_of_model_index, 'model'] = model_; 
                                        model_data[i_d, model_nm, i_fold, type_of_model_index, 'test_ix'] = test_ix[i_fold]; 
                                
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
                                if fit_condition_spec_no_general:
                                    model_data[i_d, model_nm, 'null'][type_of_model_index, 'ix'].append(test_ix[i_fold])
                                    model_data[i_d, model_nm, 'pot'][type_of_model_index, 'ix'].append(test_ix[i_fold])

                                    ### Save the null / potent predictions 
                                    model_data[i_d, model_nm, 'null'][type_of_model_index, 'pred'].append(pred_null)
                                    model_data[i_d, model_nm, 'pot'][type_of_model_index, 'pred'].append(pred_pot)
                                
                                elif fit_task_spec_and_general:
                                    model_data[i_d, model_nm, 'null'][test_ix[i_fold], :, type_of_model_index] = np.squeeze(np.array(pred_null))
                                    model_data[i_d, model_nm, 'pot'][test_ix[i_fold], :, type_of_model_index] = np.squeeze(np.array(pred_pot))
                                
                                else:
                                    model_data[i_d, model_nm, 'null'][test_ix[i_fold], :] = pred_null.copy()
                                    model_data[i_d, model_nm, 'pot'][test_ix[i_fold], :] = pred_pot.copy()
                                
        #### Confirm test_confirm matches ####
        if model_set_number == 11:
            for k in list(test_confirm.keys()):
                tmp = np.unique(np.hstack((test_confirm[k])))
                assert(len(tmp) == data_temp.shape[0])

    h5file.close()
    print(('H5 File Done: ', hdf_filename))

    if match_task_spec_n:
        sff = '_match_tsk_N'
    else:
        sff = ''

    if fit_intercept:
        sff2 = ''
    else:
        sff2 = '_no_intc'

    if full_shuffle:
        sff3 = '_full_shuff%d' %shuff_id
    
    elif within_bin_shuffle:
        sff3 = '_within_bin_shuff%d' %shuff_id
    else:
        sff3 = ''

    if task_demean:
        sff4 = '_task_demean'
    elif gen_demean:
        sff4 = '_gen_demean'
    else:
        sff4 = ''

    if alpha_always_zero:
        sff5 = '_alphazero'
    else:
        sff5 = ''

    if model_set_number == 11 and type(latent_dim) is int:
        sff6 = '_ndim%d'%(latent_dim)
    else:
        sff6 = ''

    if normalize_ridge_vars:
        sff7 = 'ridge_norm'
    else: 
        sff7 = ''

    if keep_bin_spk_zsc: 
        sff8 = '_zsc'
    else:
        sff8 = ''

    if null_predictor:
        sff9 = '_null'
    elif null_predictor_w_null_alpha:
        sff9 = '_null_alpha'
    else:
        sff9 = ''

    ### ALSO SAVE MODEL_DATA: 
    if only_potent_predictor:
        pickle.dump(model_data, open(analysis_config.config[animal + '_pref'] + 'tuning_models_'+animal+'_model_set%d_only_pot.pkl' %model_set_number, 'wb'))
    
    elif only_command:
        pickle.dump(model_data, open(analysis_config.config[animal + '_pref'] + 'tuning_models_'+animal+'_model_set%d_only_com.pkl' %model_set_number, 'wb'))
    
    else:
        if fit_task_specific_model_test_task_spec:
            pickle.dump(model_data, open(analysis_config.config[animal + '_pref'] + 'tuning_models_'+animal+'_model_set%d_task_spec%s.pkl' %(model_set_number, sff2), 'wb'))
        
        elif fit_task_spec_and_general:
            pickle.dump(model_data, open(analysis_config.config[animal + '_pref'] + 'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen%s%s%s%s%s%s.pkl' %(model_set_number, sff, sff2, sff3, sff4, sff5, sff6), 'wb'))
        
        elif fit_condition_spec_no_general:
            pickle.dump(model_data, open(analysis_config.config[animal + '_pref'] + 'tuning_models_'+animal+'_model_set%d_cond_spec%s%s.pkl' %(model_set_number, sff2, sff3), 'wb'))
        
        else:
            pickle.dump(model_data, open(analysis_config.config[animal + '_pref'] + 'tuning_models_'+animal+'_model_set%d_%s%s%s%s%s%s%s.pkl' %(model_set_number, sff2, sff3, sff4, sff5, sff7, sff8, sff9), 'wb'))

def model_ind_cell_tuning_SHUFFLE(fit_intercept = True, latent_LDS = False, latent_dim = 'full',
    nshuffs = 1000, shuff_type = 'beh_maint', keep_bin_spk_zsc = False):
    '''
    general model tuning curves for shuffled data
    
    Parameters
    ----------
    fit_intercept : bool, optional
        Description
    latent_LDS : bool, optional
        Description
    latent_dim : str, optional
        Description
    nshuffs : int, optional
        number of shuffles to generate 
    shuff_type : str, 
        "beh_maint" --> shuffle from figure 3, 4; shuffle activity across commands 
        "mn_diff_maint" --> Shuffle actiivty across commands, but keep it so that mean diffs are still present
            -- have a movement transition matrix; 
            -- permute activity from movemnet to movement 
            -- if not enough commands --> resample 
        "within_mov_shuff" --> shuffle same command within movment 
        "null_roll" --> shuffle for figure 6; roll null with respect to potent 
            -- each shuffle is a random roll integer sampled from 0-? 
        "null_roll_pot_beh_maint" --> shuffle for figure 6; roll null w.r.t potent 
            -- each shuffle is random roll intenger samples from 100 - T
            -- plus behavior maintaining potent shuffle; 
    keep_bin_spk_zsc: only matters for homer
        if true --> then load (for homer) the alphas associated with zsc. 
    Raises
    ------
    Exception
        Description
    '''
    if latent_LDS:
        model_set_number = 11
        assert(fit_intercept == False)
        save_directory = analysis_config.config['shuff_fig_dir_latentLDS']
    
    else:
        model_set_number = 6
        if fit_intercept:
            save_directory = analysis_config.config['shuff_fig_dir']
        else:
            save_directory = analysis_config.config['shuff_fig_dir_nointc']
    
    n_folds = 5
    model_var_list, predict_key, include_action_lags, history_bins_max = generate_models_list.get_model_var_list(model_set_number)
    models_to_include = [m[1] for m in model_var_list]

    ### Get magnitude boundaries ####
    mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))

    ### Place to save models: 
    model_data = dict(); 

    #### Get the dim dict ###
    if latent_LDS:
        if latent_dim == 'full':
            pass
        else:
            dim_dict = {}
            for animal in ['grom','jeev']:
                dim_dict[animal] = pickle.load(open(analysis_config.config[animal+'_pref'] + 'LDS_maxL_ndims_lowD.pkl', 'rb'))
    
    ### Get the ridge dict:
    else:   
        ridge_dict = pickle.load(open(analysis_config.config['grom_pref'] + 'max_alphas_ridge_model_set%d_shuff.pkl' %model_set_number, 'rb')); 

    for obj in gc.get_objects():   # Browse through ALL objects
        if isinstance(obj, tables.File):   # Just HDF5 files
            try:
                obj.close()
            except:
                pass #

    for animal in ['home', 'grom', 'jeev']:

        if animal == 'home':
            if keep_bin_spk_zsc:
                ridge_dict = pickle.load(open(analysis_config.config['grom_pref'] + 'max_alphas_ridge_model_set%d_shuff_zsc.pkl' %model_set_number, 'rb'))
                print('loading zscore shuff dict for homer')

        order_dict = analysis_config.data_params['%s_ordered_input_type'%animal]
        input_type = analysis_config.data_params['%s_input_type'%animal]

        ##### For each day ####
        for i_d, day in enumerate(input_type):
            if animal in ['grom', 'jeev']:
                pass
            else:
                print('##############################')
                print(('########## DAY %d ##########' %(i_d) ))
                print('##############################')
            
                ### Get kalman gain etc. 
                if animal == 'grom':
                    KG, KG_null_proj, KG_potent_orth = get_KG_decoder_grom(i_d)
                    ord_d = order_dict[i_d]
                elif animal == 'jeev':
                    KG, KG_null_proj, KG_potent_orth = get_KG_decoder_jeev(i_d)
                    ord_d = order_dict[i_d]
                elif animal == 'home': 
                    KG, KG_null_proj, KG_potent_orth, dec_mFR, dec_sdFR = get_KG_decoder_home(i_d)
                    day = [day]
                    ord_d = [order_dict[i_d]]

                
                # Get spike data from data fcn
                if shuff_type == 'beh_maint':
                    Data, Data_temp, Sub_spikes, Sub_spk_temp_all, Sub_push, Shuff_ix = generate_models_utils.get_spike_kinematics(animal, day, 
                        ord_d, history_bins_max, within_bin_shuffle = True, keep_bin_spk_zsc = keep_bin_spk_zsc,
                        day_ix = i_d, nshuffs = nshuffs)

                elif shuff_type == 'mn_diff_maint':
                    Data, Data_temp, Sub_spikes, Sub_spk_temp_all, Sub_push, Shuff_ix = generate_models_utils.get_spike_kinematics(animal, day, 
                        ord_d, history_bins_max, within_bin_shuffle = False, mn_maint_within_bin_shuffle = True, keep_bin_spk_zsc = keep_bin_spk_zsc,
                        day_ix = i_d, nshuffs = nshuffs)
                
                elif shuff_type ==  'within_mov_shuff':
                    Data, Data_temp, Sub_spikes, Sub_spk_temp_all, Sub_push, Shuff_ix = generate_models_utils.get_spike_kinematics(animal, day, 
                        ord_d, history_bins_max, within_bin_shuffle = False, mn_maint_within_bin_shuffle = False, keep_bin_spk_zsc = keep_bin_spk_zsc,
                        within_mov_bin_shuffle = True, 
                        day_ix = i_d, nshuffs = nshuffs)
                
                elif shuff_type == 'null_roll': 
                    Data, Data_temp, Sub_spikes, Sub_spk_temp_all, Sub_push, Shuff_ix = generate_models_utils.get_spike_kinematics(animal, day, 
                        ord_d, history_bins_max, within_bin_shuffle = False, mn_maint_within_bin_shuffle = False, roll_shuff = True,
                        keep_bin_spk_zsc = keep_bin_spk_zsc,
                        day_ix = i_d, nshuffs = nshuffs)

                elif shuff_type == 'null_roll_pot_beh_maint':
                    ### Ge the rolling indices ######
                    Data, Data_temp, Sub_spikes, Sub_spk_temp_all, Sub_push, Shuff_ix_roll = generate_models_utils.get_spike_kinematics(animal, day, 
                        ord_d, history_bins_max, within_bin_shuffle = False, mn_maint_within_bin_shuffle = False, roll_shuff = True,
                        keep_bin_spk_zsc = keep_bin_spk_zsc,
                        day_ix = i_d, nshuffs = nshuffs)

                    #### Get the behavior maintaining indices ####
                    _, _, _, _, _, Shuff_ix = generate_models_utils.get_spike_kinematics(animal, day, 
                        ord_d, history_bins_max, within_bin_shuffle = True, keep_bin_spk_zsc = keep_bin_spk_zsc,
                        day_ix = i_d, nshuffs = nshuffs)

                    #### For each shuffle add teh shuff_ix_roll to the shuff_ix: 
                    for nshuf in range(nshuffs):
                        Shuff_ix[nshuf, 'null_roll'] = Shuff_ix_roll[nshuf]

                print(('Animal %s, Day %d data extraction done' %(animal, i_d)))

                #### Save shuffle indices ####### 
                if animal == 'home':
                    if keep_bin_spk_zsc:
                        zstr = 'zsc'
                    else:
                        zstr = ''
                else:
                    zstr = ''

                if shuff_type == 'beh_maint':
                    shuff_ix_fname = save_directory + '%s_%d_%s_shuff_ix.pkl' %(animal, i_d, zstr)
                else:
                    shuff_ix_fname = save_directory + '%s_%d_%s_shuff_ix_%s.pkl' %(animal, i_d, zstr, shuff_type)

                ### Re make data to be smaller 
                Data2 = dict(spks=Data['spks'], push=Data['push'], bin_num=Data['bin_num'], targ=Data['targ'], task=Data['tsk'])
                Data2['animal'] = animal 
                Data2['day_ix'] = i_d 


                #### Null rolling ####
                if 'null_roll' in shuff_type:

                    #### Get null vs. potent spkes #### 
                    null, pot = decompose_null_pot(Data['spks'], Data['push'], KG, KG_null_proj, KG_potent_orth)

                    #### Save these in Data2 ####
                    Data2['spks_null'] = null.copy()
                    Data2['spks_pot'] = pot.copy()
                    Data2['KG'] = KG.copy()

                #### Save data in this guy too ####
                Shuff_ix['Data'] = Data2

                #### Get training / testing sets split up --- test on 80% one task, test on 20% same tasks 20% other task
                test_ix, train_ix = generate_models_utils.get_training_testings(n_folds, Data_temp)
                type_of_model = np.zeros((5, ))

                #### Save the test and train indices ####
                Shuff_ix['test_ix'] = test_ix
                Shuff_ix['train_ix'] = train_ix
                Shuff_ix['temp_n']  = Sub_push.shape[0]

                command_bins_disc = util_fcns.commands2bins([Sub_push], mag_boundaries, animal, i_d, vel_ix = [0, 1], ndiv=8)[0]

                #### Save these guys ###
                pickle.dump(Shuff_ix, open(shuff_ix_fname, 'wb'))

                for shuffle in range(nshuffs):

                    ##### Shuffles and get trial starts #####
                    if shuff_type == 'null_roll':
                        sub_spikes, sub_spikes_tm1, sub_push, tm0ix, tm1ix, _ = get_temp_spks_null_pot_roll(Data2, Shuff_ix[shuffle])
                    
                    elif shuff_type == 'null_roll_pot_beh_maint':
                        sub_spikes, sub_spikes_tm1, sub_push, tm0ix, tm1ix = get_temp_spks_null_roll_pot_shuff(Data2, Shuff_ix[shuffle], Shuff_ix[shuffle, 'null_roll'])
                    
                    else:
                        sub_spikes, sub_spikes_tm1, sub_push, tm0ix, tm1ix = get_temp_spks(Data2, Shuff_ix[shuffle])

                    assert(np.allclose(Sub_spikes, Data2['spks'][tm0ix, :]))
                    assert(np.allclose(Sub_spk_temp_all[:, 0, :], Data2['spks'][tm1ix]))

                    #### Split up #####
                    shuff_com_bins_disc = util_fcns.commands2bins([sub_push], mag_boundaries, animal, i_d, vel_ix = [3, 5], ndiv=8)[0]

                    #### Check that these are linked still; 
                    if animal == 'grom':
                        assert(np.allclose(sub_push[:, [3, 5]], np.dot(KG, sub_spikes.T).T))
                    elif animal == 'jeev':
                        assert(generate_models_utils.quick_reg(  np.array(np.dot(KG, sub_spikes.T).T), 
                            np.array(sub_push[:, [3, 5]])) > .98)
                    elif animal == 'home':
                        if keep_bin_spk_zsc:
                            assert(np.allclose(sub_push[:, [3, 5]], np.dot(KG, sub_spikes.T).T))
                        else:
                            sub_spks_z = (sub_spikes - dec_mFR[np.newaxis, :]) / dec_sdFR[np.newaxis, :]
                            assert(np.allclose(sub_push[:, [3, 5]], np.dot(KG, sub_spks_z.T).T))

                    ### Now make sure the discretized version are ok: 
                    assert(np.allclose(command_bins_disc, shuff_com_bins_disc))

                    ### Make sure not exact match ###
                    assert(np.sum(np.sum(sub_spikes, axis=1) != np.sum(Sub_spikes, axis=1)) > 0)
                    
                    nneur = sub_spikes.shape[1]
                    variables_list = return_variables_associated_with_model_var(model_var_list, include_action_lags, nneur)
                    
                    ### Models -- save predictions
                    for i_m, (model_nm, variables) in enumerate(zip(models_to_include, variables_list)):

                        check_variables_order(variables, nneur)

                        ### Models to save ##########
                        ### Keep spikes ####
                        nT, nneur = sub_spikes.shape
                        model_data = np.zeros((nT, nneur, 3)) 
                    
                        ######### Ridge / LDS #########
                        if latent_LDS: 
                            #### This gets trials instead of time points ###
                            trl_test_ix, trl_train_ix, type_of_model = generate_models_utils.get_training_testings_generalization_LDS_trial(n_folds, data_temp,
                                match_task_spec_n = True)
                            test_confirm = {}
                            for tmp in np.unique(type_of_model):
                                test_confirm[tmp] = []
                            save_dat = dict()
                            save_dat['modelA'] = []
                            save_dat['modelC'] = []
                            save_dat['modelW'] = []
                            save_dat['modelQ'] = []
                            save_dat['modelkeepix'] = []
                            save_dat['type_of_model_index'] = []

                        ######### Ridge #########
                        else:
                            save_dat = dict()
                        
                        ###### Go through the all the models #####
                        for i_fold, type_of_model_index in enumerate(type_of_model):

                            if type_of_model_index < 0:
                                pass
                            else:
                                if latent_LDS:
                                    raise Exception('Need to re-order spk_tm1')

                                    assert(type_of_model_index in [0, 1, 2])
                                    if latent_dim == 'full':
                                        ndims = 'full'
                                    else:
                                        ndims = dim_dict[animal][i_d][type_of_model_index]
                                    
                                    ####### Fit the model ##########
                                    model_ = fit_LDS(data_temp, variables, trl_train_ix[i_fold], n_dim_latent = ndims)
                                    
                                    ####### predict with the model ##########
                                    pred_Y, test_ix = pred_LDS(data_temp, model_, variables, trl_test_ix[i_fold], i_fold)
                                    test_confirm[type_of_model_index].append(test_ix[i_fold])
                                    if type_of_model_index == 2:
                                        save_dat['modelA'].append(model_.A)
                                        save_dat['modelC'].append(model_.C)
                                        save_dat['modelQ'].append(model_.sigma_obs)
                                        save_dat['modelW'].append(model_.sigma_states)
                                        save_dat['modelkeepix'].append(model_.keep_ix)
                                        save_dat['type_of_model_index'].append(type_of_model_index)
                                                
                                else:
                                    ### TEST DATA ####
                                    # data_temp_dict_test = panda_to_dict(data_temp.iloc[test_ix[i_fold]])
                                    # data_temp_dict_test['spks'] = sub_spikes[test_ix[i_fold]]
                                    # data_temp_dict_test['pshy'] = sub_push_all[test_ix[i_fold], 1]
                                    # data_temp_dict_test['pshx'] = sub_push_all[test_ix[i_fold], 0]
                                    # data_temp_dict_test['psh'] = np.hstack(( data_temp_dict_test['pshx'], data_temp_dict_test['pshy']))
                                    # data_temp_dict_test['spks_all'] = sub_spikes_all[test_ix[i_fold], :, :]

                                    ### TRAIN DATA ####
                                    #data_temp_dict = panda_to_dict(Data_temp.iloc[train_ix[i_fold]])
                                    data_temp_dict = {}
                                    data_temp_dict['spks'] = sub_spikes[train_ix[i_fold], :]
                                    #data_temp_dict['pshy'] = sub_push[train_ix[i_fold], 1]
                                    #data_temp_dict['pshx'] = sub_push[train_ix[i_fold], 0]
                                    #data_temp_dict['psh'] = np.hstack(( data_temp_dict['pshx'], data_temp_dict['pshy']))
                                    data_temp_dict['spks_tm1'] = sub_spikes_tm1[train_ix[i_fold], :]
                                    
                                    #### Replace spk_tm1 and spk_tp1 with sub_spks_all --> Shuffled 
                                    for i_n in range(nneur):
                                        data_temp_dict['spk_tm1_n%d'%i_n] = data_temp_dict['spks_tm1'][:, i_n]
                                        
                                    ## Store the params; 
                                    alpha_spec = ridge_dict[animal][0][i_d, model_nm]

                                    model_ = fit_ridge(data_temp_dict[predict_key], data_temp_dict, variables, alpha=alpha_spec, 
                                        only_potent_predictor = False, KG_pot = KG_potent_orth, 
                                        fit_task_specific_model_test_task_spec = False,
                                        fit_intercept = fit_intercept, model_nm = model_nm)
                                    
                                    # h5file, model_, pred_Y = generate_models_utils.h5_add_model(None, model_, i_d, first=i_d==0, model_nm=model_nm, 
                                    #     test_data = data_temp_dict_test, fold = i_fold, xvars = variables, predict_key=predict_key, 
                                    #     only_potent_predictor = False, KG_pot = KG_potent_orth, KG = KG,
                                    #     fit_task_specific_model_test_task_spec = False,
                                    #     fit_intercept = fit_intercept)

                                    # if type_of_model_index == 2:
                                    #     save_dat['model_coef'].append(model_.coef_)

                                    #     if fit_intercept:
                                    #         save_dat['model_intc'].append(model_.intercept_)

                                    ### Save the model 
                                    save_dat[i_fold, 'coef_'] = model_.coef_
                                    save_dat[i_fold, 'intc_'] = model_.intercept_

                                    if 'psh_2' in model_nm:
                                        save_dat[i_fold, 'W'] = model_.W

                                ####### Save model data #############
                                #model_data[test_ix[i_fold], :, type_of_model_index] = np.squeeze(np.array(pred_Y))
                                
                        #### Save Animal/Day/Shuffle/Model Name ###
                        shuff_str = str(shuffle)
                        shuff_str = shuff_str.zfill(3)

                        ### Only save the general model for space; 
                        #save_dat['model_data'] = model_data[:, :, 2]
                        plt.close('all')
                    
                        if latent_LDS:
                            #### Confirm test_confirm matches ####
                            for k in list(test_confirm.keys()):
                                tmp = np.unique(np.hstack((test_confirm[k])))
                                assert(len(tmp) == data_temp.shape[0])
                            
                            #### Save as PKL instead of .mat ###
                            pickle.dump(save_dat, open(save_directory+'%s_%d_shuff%s_%s.pkl' %(animal, i_d, shuff_str, model_nm), 'wb'))

                        else:
                            save_dat['shuff_num'] = shuffle

                            if animal == 'home':
                                if keep_bin_spk_zsc:
                                    zstr = 'zsc'
                                else:
                                    zstr = ''
                            else:
                                zstr = ''

                            if shuff_type == 'beh_maint':
                                fname = save_directory+'%s_%d_shuff%s_%s_%s_models.mat' %(animal, i_d, shuff_str, model_nm, zstr)
                            else:
                                fname = save_directory+'%s_%d_shuff%s_%s_%s_%s_models.mat' %(animal, i_d, shuff_str, model_nm, shuff_type, zstr)

                            sio.savemat(fname, save_dat)                        
                            print(('file %s' %(fname)))

def get_temp_spks_ix(Data2):
    ##### Shuffles and get trial starts #####
    bin_num = np.hstack((Data2['bin_num']))

    ### Get ones that are > first bin; 
    spks_t1 = np.nonzero(bin_num > 0)[0]

    ### Get the zero bins 
    spks_not_t2 = np.nonzero(bin_num == 0)[0]

    ### Remove the first one 
    spks_not_t2 = spks_not_t2[spks_not_t2 > 0]

    ### Add teh last bine 
    spks_not_t2 = np.hstack((spks_not_t2, len(bin_num)))

    ### subtract by 1 to get the last bin
    spks_not_t2 = spks_not_t2 - 1

    assert(np.all(bin_num[spks_not_t2[:-1] + 1] == 0))
    spks_t2 = np.array([i for i in range(len(bin_num)) if i not in spks_not_t2])

    ##### Keep these guys 
    spks_keep = np.intersect1d(spks_t1, spks_t2)

    return spks_keep, spks_keep - 1

def get_temp_spks(Data2, shuff_ix):
    ##### Shuffles and get trial starts #####
    bin_num = np.hstack((Data2['bin_num']))

    ### Get ones that are > first bin; 
    spks_t1 = np.nonzero(bin_num > 0)[0]

    ### Get the zero bins 
    spks_not_t2 = np.nonzero(bin_num == 0)[0]

    ### Remove the first trial 
    spks_not_t2 = spks_not_t2[spks_not_t2 > 0]

    ### Add teh last bine 
    spks_not_t2 = np.hstack((spks_not_t2, len(bin_num)))

    ### subtract by 1 to get the last bin
    spks_not_t2 = spks_not_t2 - 1

    assert(np.all(bin_num[spks_not_t2[:-1] + 1] == 0))
    spks_t2 = np.array([i for i in range(len(bin_num)) if i not in spks_not_t2])

    ##### Keep these guys --> anything that is NOT the first or the last bin in the trial 
    spks_keep = np.intersect1d(spks_t1, spks_t2)

    spks_shuff = Data2['spks'][shuff_ix, :]
    push_shuff = Data2['push'][shuff_ix, :]

    return spks_shuff[spks_keep], spks_shuff[spks_keep - 1], push_shuff[spks_keep], spks_keep, spks_keep - 1

def get_temp_spks_null_roll_pot_shuff(Data2, shuff_ix, shuff_int_roll):
    '''
    method to get spikes with rolling the null, shuffling the potent 
    
    Parameters
    ----------
    Data2 : dict with all the data 
    shuff_ix : np.array, behavior maintianing potetn shuffle --> needs to be applied to full data (not subselected)
    shuff_int_roll : integer to roll null 
    '''
    ##### Shuffles and get trial starts #####
    bin_num = np.hstack((Data2['bin_num']))

    ### Get ones that are > first bin; 
    spks_t1 = np.nonzero(bin_num > 0)[0]

    ### Get the zero bins 
    spks_not_t2 = np.nonzero(bin_num == 0)[0]

    ### Remove the first trial 
    spks_not_t2 = spks_not_t2[spks_not_t2 > 0]

    ### Add teh last bine 
    spks_not_t2 = np.hstack((spks_not_t2, len(bin_num)))

    ### subtract by 1 to get the last bin
    spks_not_t2 = spks_not_t2 - 1
    assert(np.all(bin_num[spks_not_t2[:-1] + 1] == 0))
    spks_t2 = np.array([i for i in range(len(bin_num)) if i not in spks_not_t2])

    ##### Keep these guys --> anything that is NOT the first or the last bin in the trial 
    spks_keep = np.intersect1d(spks_t1, spks_t2)

    #### We need to keep spks_keep and spks_keep -1 together: 
    spks_null_tm0 = Data2['spks_null'][spks_keep, :]
    nN = spks_null_tm0.shape[1]

    spks_null_tm1 = np.vstack(( np.zeros((1, nN)) + np.nan, Data2['spks_null'][:-1, :] ))[spks_keep, :]
    assert(spks_null_tm0.shape == spks_null_tm1.shape)
    assert(np.sum(np.isnan(spks_null_tm1)) == 0)

    #### Potent ####
    spks_pot_shuff = Data2['spks_pot'][shuff_ix, :]
    spks_pot_shuff_tm0  = spks_pot_shuff[spks_keep, :]
    spks_pot_shuff_tm1  = np.vstack(( np.zeros((1, nN)) + np.nan, spks_pot_shuff[:-1, :] ))[spks_keep, :]
    assert(np.sum(np.isnan(spks_pot_shuff_tm1)) == 0)

    #### Now that these have been subselected,roll the null activiyt 
    ix_roll = np.roll(np.arange(len(spks_keep)), shuff_int_roll)
    spks_null_shuff_tm0 = spks_null_tm0[ix_roll]
    spks_null_shuff_tm1 = spks_null_tm1[ix_roll]

    spks_shuff_tm0 = spks_null_shuff_tm0 + spks_pot_shuff_tm0
    spks_shuff_tm1 = spks_null_shuff_tm1 + spks_pot_shuff_tm1

    ### make sure the command bins match #####
    mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file'], 'rb'))

    ### This is true push (for jeev ~= kg*spks_pot)
    command_bins_push = util_fcns.commands2bins([Data2['push'][spks_keep, :]], mag_boundaries, Data2['animal'], 
        Data2['day_ix'], vel_ix = [3, 5], ndiv=8)[0]

    #### bins associated with estimated push ####
    command_bins_kg_shuff_spks = util_fcns.commands2bins([np.dot(Data2['KG'], Data2['spks'][shuff_ix[spks_keep], :].T).T], 
        mag_boundaries, Data2['animal'], Data2['day_ix'], vel_ix = [0, 1], ndiv=8)[0]
    
    ### get 
    push_shuff = Data2['push'][shuff_ix, :]
    push_shuff = push_shuff[spks_keep, :]

    #### This is estimated push given shuffels of potent activity ### 
    command_bins_kg_shuff_spks2 = util_fcns.commands2bins([np.dot(Data2['KG'], spks_shuff_tm0.T).T], mag_boundaries, Data2['animal'], 
        Data2['day_ix'], vel_ix = [0, 1], ndiv=8)[0]
    
    #### This is actual same shuffling applied to push #### 
    command_bins_push_shuff = util_fcns.commands2bins([push_shuff], mag_boundaries, Data2['animal'], 
        Data2['day_ix'], vel_ix = [3, 5], ndiv=8)[0]

    if Data2['animal'] == 'grom':
        ### This works here bc KG * spks = push exactly, so should match bins fo real push
        assert(np.allclose(command_bins_push, command_bins_push_shuff))
        assert(np.allclose(command_bins_push, command_bins_kg_shuff_spks2))
        assert(np.allclose(command_bins_kg_shuff_spks, command_bins_kg_shuff_spks2))
        assert(np.allclose(command_bins_kg_shuff_spks, command_bins_push_shuff))
        assert(np.allclose(command_bins_kg_shuff_spks2, command_bins_push_shuff))
    
    elif Data2['animal'] == 'jeev':
        ### Here want to say that actual push bisn == shuffled push bins; 
        assert(np.allclose(command_bins_push, command_bins_push_shuff))

        ### This may not be true; 
        ### The shuffle is doen to maintain the push as is, so if KG*spks != push commadn bins 
        ### Then this may not be true; 
        diff_c = command_bins_kg_shuff_spks - command_bins_kg_shuff_spks2
        ix,iy = np.nonzero(diff_c)
        assert(len(iy)/float(diff_c.shape[0]) < .1)
        #assert(np.allclose(command_bins_kg_shuff_spks, command_bins_kg_shuff_spks2))
        
    return spks_shuff_tm0, spks_shuff_tm1, push_shuff, spks_keep, spks_keep - 1, ix_roll

def get_temp_spks_null_pot_roll(Data2, shuff_ix):
    """
    method to get spikes according to rolling the null activity with respsect to the potent
    
    Parameters
    ----------
    Data2 : dict with all the data
    shuff_ix : indices to roll the null activity 
        np.array
    
    Returns
    -------
    TYPE
        Description
    """
    ##### Shuffles and get trial starts #####
    bin_num = np.hstack((Data2['bin_num']))

    ### Get ones that are > first bin; 
    spks_t1 = np.nonzero(bin_num > 0)[0]

    ### Get the zero bins 
    spks_not_t2 = np.nonzero(bin_num == 0)[0]

    ### Remove the first trial 
    spks_not_t2 = spks_not_t2[spks_not_t2 > 0]

    ### Add teh last bine 
    spks_not_t2 = np.hstack((spks_not_t2, len(bin_num)))

    ### subtract by 1 to get the last bin
    spks_not_t2 = spks_not_t2 - 1

    assert(np.all(bin_num[spks_not_t2[:-1] + 1] == 0))
    spks_t2 = np.array([i for i in range(len(bin_num)) if i not in spks_not_t2])

    ##### Keep these guys --> anything that is NOT the first or the last bin in the trial 
    spks_keep = np.intersect1d(spks_t1, spks_t2)

    #### We need to keep spks_keep and spks_keep -1 together: 
    spks_null_tm0 = Data2['spks_null'][spks_keep, :]
    nN = spks_null_tm0.shape[1]

    spks_null_tm1 = np.vstack(( np.zeros((1, nN)) + np.nan, Data2['spks_null'][:-1, :] ))[spks_keep, :]
    assert(spks_null_tm0.shape == spks_null_tm1.shape)
    assert(np.sum(np.isnan(spks_null_tm1)) == 0)

    #### Potent ####
    spks_pot_tm0  = Data2['spks_pot'][spks_keep, :]
    spks_pot_tm1  = np.vstack(( np.zeros((1, nN)) + np.nan, Data2['spks_pot'][:-1, :] ))[spks_keep, :]
    assert(np.sum(np.isnan(spks_pot_tm1)) == 0)

    #### Now that these have been subselected,roll the null activiyt 
    ix_roll = np.roll(np.arange(len(spks_keep)), shuff_ix)
    spks_null_shuff_tm0 = spks_null_tm0[ix_roll]
    spks_null_shuff_tm1 = spks_null_tm1[ix_roll]

    spks_shuff_tm0 = spks_null_shuff_tm0 + spks_pot_tm0
    spks_shuff_tm1 = spks_null_shuff_tm1 + spks_pot_tm1

    if nN > 21: # Grom proxy 
        assert(np.allclose(np.dot(Data2['KG'], spks_shuff_tm0.T).T, Data2['push'][np.ix_(spks_keep, [3, 5])]))
    else:
        assert(generate_models_utils.quick_reg(np.array(np.dot(Data2['KG'], spks_shuff_tm0.T).T), Data2['push'][np.ix_(spks_keep, [3, 5])]) > .98)

    push_shuff = Data2['push'][spks_keep, :]

    return spks_shuff_tm0, spks_shuff_tm1, push_shuff, spks_keep, spks_keep - 1, ix_roll

def decompose_null_pot(spks, push, KG, KG_null_proj, KG_potent_orth):
    '''
    decompose null potent 
    '''

    assert(len(spks.shape) == 2)

    null_spks = np.dot(KG_null_proj, spks.T).T
    pot_spks = np.dot(KG_potent_orth, spks.T).T

    assert(np.allclose(null_spks + pot_spks, spks))
    assert(np.allclose(np.dot(KG, null_spks.T).T, np.zeros(( null_spks.shape[0], 2))))

    nneur = pot_spks.shape[1]
    if nneur <= 21:
        ### Jeevs
        assert(generate_models_utils.quick_reg(np.array(np.dot(KG, pot_spks.T).T), push[:, [3, 5]]) > .98)

    else:
        assert(np.allclose(np.dot(KG, pot_spks.T).T, push[:, [3, 5]]))

    return null_spks, pot_spks

######## Possible STEP 2 -- fit the residuals #####
def model_state_encoding(animal, model_set_number = 7, state_vars = ['pos_tm1', 'vel_tm1', 'trg', 'tsk'],
    model = 'hist_1pos_0psh_0spksm_1_spksp_0', n_folds = 5, fit_intercept = True):
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
            day, order_dict[i_d], 1, day_ix = i_d)

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
                        model_ = fit_ridge(data_temp_rez[predict_key][train_ix[i_fold], :], data_temp.iloc[train_ix[i_fold]], variables, 
                            alpha=alpha_spec, fit_intercept = fit_intercept)
                        
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

######## SLDS sweeping ###################
def sweep_slds_params(animal, day, n_folds=5, ver=1, mean_sub=True, n_dim_latent_try=None):
    ## ver: 
    ### model version: 1 == gaussian/gaussian 
    ### model version: 2 == diag gaussian / gaussian_ortho

    ### Posterior fitting params 
    window_size = [5] #2, 5, 10]#, 20] # window size for posterior 
    window_size_max = 5
    niters_fit_post = [5]#2, 5, 10, 50] # number of iters to fit posterior 
    
    ### model fitting params 
    niters_fit_model = [10] # number of iteratinos to fit the model 
    K = [1, 2, 3, 4, 5]#, 3, 5, 9, 12] # Number of discrete LDSs 
    alphas = [0.01]#, .01, .01]

    history_bins_max = 1; 
    within_bin_shuffle = False; 
    keep_bin_spk_zsc = False; 
    null_pred = False 

    ### order
    ndays = analysis_config.data_params['%s_ndays'%animal]
    order_dict = analysis_config.data_params['%s_ordered_input_type'%animal]
    order_dict = [order_dict[i] for i in range(ndays)]
    order_d = order_dict[day]

    input_type = analysis_config.data_params['%s_input_type'%animal]
    input_type = [input_type[i] for i in range(ndays)]

    ### pull spike kinematics
    data, data_temp, sub_spikes, sub_spk_temp_all, sub_push_all = generate_models_utils.get_spike_kinematics(animal, 
        input_type[day], order_d, history_bins_max, within_bin_shuffle = within_bin_shuffle, 
        keep_bin_spk_zsc = keep_bin_spk_zsc,
        day_ix = day, null = null_pred)
    
    print('n spks in data %d'%(data['spks'].shape[1]))
        
    #### Which trials are test vs. training ###
    ### type of model 2 doesn't care about task specificity ####
    trl_test_ix, trl_train_ix, type_of_model = generate_models_utils.get_training_testings_generalization_LDS_trial(n_folds, data_temp,
        match_task_spec_n = False)

    save_params_model = {}
    save_params_post = {}

    ### For each fold and set of parameters
    for i_fold in range(n_folds): 

        trls_train = get_slds_data_train(data, trl_train_ix, type_of_model, i_fold, window_size_max)
        print('n spks in trls_train %d'%(trls_train[0].shape[1]))

        for ik, k in enumerate(K): 
        
            for i_nit, nit in enumerate(niters_fit_model): 

                for i_a, alph in enumerate(alphas): 
            
                    ### fit model ####
                    print('training model ver %d'%ver)
                    rslds, elbos = fit_slds(trls_train, k, nit, alph, ver=ver, mean_sub=mean_sub, n_dim_latent_try=n_dim_latent_try) 

                    save_params_model[i_fold, k, nit, alph, 'elbo'] = elbos 
                    save_params_model[i_fold, k, nit, alph, 'model'] = rslds 

                    ### fit posterior 
                    for i_w, ws in enumerate(window_size): 

                        ### Get test data w/ different window sizes ###
                        pts_post, pts_true_test, pts_true_test_tm1, pts_id = get_slds_data_test(data, trl_test_ix, 
                            type_of_model, i_fold, ws, window_size_max)

                        for i_nit_post, nit_post in enumerate(niters_fit_post): 

                            ### get held out data prediction ###
                            xhat_post, elbos2 = fit_slds_posteriors(rslds, pts_post, nit_post, mean_sub=mean_sub)

                            ### predict it forward
                            yt_pred_all = pred_fwd_slds(rslds, xhat_post, mean_sub=mean_sub)

                            ### going forward ### 
                            r2_fwd = util_fcns.get_R2(np.vstack((pts_true_test)), np.vstack((yt_pred_all)))
                            
                            save_params_post[i_fold, k, nit, alph, ws, nit_post, 'r2_pred'] = r2_fwd
                            save_params_post[i_fold, k, nit, alph, ws, nit_post, 'elbos'] = elbos2
                            save_params_post[i_fold, k, nit, alph, ws, nit_post, 'pred_spks'] = yt_pred_all
                            save_params_post[i_fold, k, nit, alph, ws, nit_post, 'true_spks'] = pts_true_test

                            ## Trl / bin_num predicted in pred_spks
                            save_params_post[i_fold, k, nit, alph, ws, nit_post, 'pred_spks_params'] = pts_id
                            
                            ### smoothed ###
                            yt_pred_tm1 = pred_tm1_slds(rslds, xhat_post, mean_sub=mean_sub)
                            r2_smooth = util_fcns.get_R2(np.vstack((pts_true_test_tm1)), np.vstack((yt_pred_tm1)))

                            save_params_post[i_fold, k, nit, alph, ws, nit_post, 'r2_smooth'] = r2_smooth

                            ### Save these each time (over writting so we can get intermediate plots)
                            d = dict(save_params_post=save_params_post, save_params_model=save_params_model)

                            if n_dim_latent_try is not None: 
                                ndlt = '_ndlt_'+str(n_dim_latent_try)
                            else: 
                                ndlt = ''

                            if mean_sub: 
                                pickle.dump(d, open(os.path.join(analysis_config.config['BMI_DYN'], 'slds_sweep_%s_%d_%d_mean_sub%s.pkl'%(animal, day, ver, ndlt)), 'wb'))
                            else: 
                                pickle.dump(d, open(os.path.join(analysis_config.config['BMI_DYN'], 'slds_sweep_%s_%d_%d%s.pkl'%(animal, day, ver, ndlt)), 'wb'))
    return True

def get_slds_data_train(data, trl_train_ix, type_of_model, i_fold, window_size_max):

    ### Get out only the index 2 (doesn't balance across tasks -- no longer relevant)
    ix_valid = np.nonzero(type_of_model == 2)[0]

    train_trls = None; 
    for i in ix_valid: 

        ### This fold should match i_fold 
        if (i%5) == i_fold: 

            train_trls = trl_train_ix[i]

    ### Make sure they got assigned 
    assert(train_trls is not None) 

    trls_train = []; 

    for t in train_trls: 
        ix = np.nonzero(data['trl'] == t)[0]

        if len(ix) > (window_size_max+1):
        
            ## add spikes for this trial ###
            trls_train.append(data['spks'][ix[window_size_max:], :])

    return trls_train

def get_slds_data_test(data, trl_test_ix, type_of_model, i_fold, window_size, window_size_max):

    ### Get out only the index 2 (doesn't balance across tasks -- no longer relevant)
    ix_valid = np.nonzero(type_of_model == 2)[0]

    test_trls = None; 
    for i in ix_valid: 

        ### This fold should match i_fold 
        if (i%5) == i_fold: 

            test_trls = trl_test_ix[i]

    ### Make sure they got assigned 
    assert(test_trls is not None) 

    pts_post = []; 
    pts_test = []; 
    pts_test_tm1 = []; 
    pts_id = []; 

    for t in test_trls: 
        ix = np.nonzero(data['trl'] == t)[0]

        ### add a set of windows used to predict next data point 
        if len(ix) > (window_size + 1): 

            ### only include data points accessible to all; 
            for pt in range(window_size_max, len(ix) - 1): 
                ix_ = ix[np.arange(pt-window_size, pt)]

                pts_post.append(data['spks'][ix_, :])
                pts_test.append(data['spks'][ix[pt], :])
                
                ### Make sure last point in spks ix is actually the last point ### 
                assert(np.all(data['spks'][ix_, :][-1, :] == data['spks'][ix[pt-1], :]))

                pts_test_tm1.append(data['spks'][ix[pt-1], :])
                pts_id.append([t, pt]) # trial // predicted point 

    return pts_post, pts_test, pts_test_tm1, pts_id

def get_slds_data_test_full_trial(data, trl_test_ix, type_of_model, i_fold): 
    ### Get out only the index 2 (doesn't balance across tasks -- no longer relevant)
    ix_valid = np.nonzero(type_of_model == 2)[0]

    test_trls = None; 
    for i in ix_valid: 

        ### This fold should match i_fold 
        if (i%5) == i_fold: 

            test_trls = trl_test_ix[i]
   
    ### Make sure they got assigned 
    assert(test_trls is not None) 

    pts_test = []; 

    for t in test_trls: 
        ix = np.nonzero(data['trl'] == t)[0]

        ### add a set of windows used to predict next data point 
        if len(ix) > 0: 

            ### add the trial ### 
            pts_test.append(data['spks'][ix, :])
            
    return pts_test

def fit_slds(trls_train, k, nit, alpha, ver=1, mean_sub=False, n_dim_latent_try = None): 
    D_obs = trls_train[0].shape[1]
     

    if mean_sub: 
        X = np.vstack((trls_train))
        mean_sub = np.mean(X, axis=0)
        keep_ix = np.nonzero(10*mean_sub > 0.5)[0]
        D_obs = len(keep_ix)
        trls_train2 = []
        for trl in trls_train: 
            trls_train2.append(trl[:, keep_ix] - mean_sub[keep_ix][np.newaxis, :])

    else: 
        D_obs = trls_train[0].shape[1]
        trls_train2 = [trl for trl in trls_train]
        keep_ix = np.arange(D_obs)
        mean_sub = np.zeros((D_obs))
    
    if n_dim_latent_try is None: 
        n_dim_latent_try = D_obs - 1;


    import ssm 
    if ver == 1: 
        rslds_lem = ssm.SLDS(D_obs, k, n_dim_latent_try,
                 transitions="recurrent_only",
                 dynamics="gaussian",
                 emissions="gaussian",
                 single_subspace=True)

    elif ver == 2: 
        rslds_lem = ssm.SLDS(D_obs, k, n_dim_latent_try,
                 transitions="recurrent_only",
                 dynamics="diagonal_gaussian", #"gaussian",
                 emissions="gaussian_orthog", #"gaussian",
                 single_subspace=True)

    rslds_lem.initialize(trls_train2)

    q_elbos_lem, q_lem = rslds_lem.fit(trls_train2, method="laplace_em",
        variational_posterior="structured_meanfield",initialize=False, 
        num_iters=nit, alpha=alpha)

    rslds_lem.keep_ix = keep_ix.copy()
    rslds_lem.mean_sub = mean_sub.copy()

    return rslds_lem, q_elbos_lem

def fit_slds_posteriors(rslds, pts_post, nit_post, mean_sub=False): 

    if mean_sub: 
        pts_post2 = []
        for trl in pts_post: 
            pts_post2.append(trl[:, rslds.keep_ix] - rslds.mean_sub[rslds.keep_ix][np.newaxis, :])
    else: 
        pts_post2 = [trl for trl in pts_post]

    elbos, post_test = rslds.approximate_posterior(pts_post2, num_iters=nit_post, alpha=0.0)
    
    return post_test.mean_continuous_states, elbos

def pred_fwd_slds(rslds, xhat_all, mean_sub = False): 

    yt_pred_all = []

    for xhat in xhat_all: 

        ### Get p(discrete state) based on estimate of x; 
        #Ptm1 = rslds.transitions.log_transition_matrices(xhat, np.zeros((xhat.shape[0], 0)), 
        #                                                   None, None)
        # ## Get z_tm1
        #z_tm1 = np.argmax(Ptm1[-1, 0, :])

        ### get Log PS on own calcs; 
        log_Ps =  np.dot(xhat, rslds.transitions.Rs.T)[:, None, :]     # past observations
        log_Ps = log_Ps + rslds.transitions.r                                       # bias
        log_Ps = np.tile(log_Ps, (1, rslds.transitions.K, 1))                       # expand
        log_Ps = log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)       # normalize

        # log_Ps is just tiled in K dimensions; 
        z_tm1 = np.argmax(log_Ps[-1, 0, :])

        ## x_{t-1}
        x_tm1 = xhat[-1, :]

        # A_{z_t-1} 
        A = rslds.dynamics.As[z_tm1, :, :]
        b = rslds.dynamics.bs[z_tm1, :]
        xt = np.dot(A, x_tm1) + b

        C = rslds.emissions.Cs[0, :, :]
        d = rslds.emissions.ds[0, :]
        yt_pred = np.dot(C, xt) + d

        if mean_sub: 
            ## set all values to mean ###
            yt_pred_2 = rslds.mean_sub.copy()

            ## Updated the values that have FR > 0.5 Hz; ### 
            yt_pred_2[rslds.keep_ix] = yt_pred + rslds.mean_sub[rslds.keep_ix]
            yt_pred_all.append(yt_pred_2)

        else: 
            yt_pred_all.append(yt_pred)

    return yt_pred_all

def pred_tm1_slds(rslds, xhat_all, mean_sub=False): 
    
    yt_pred_all = []

    for xhat in xhat_all: 

        ## x_{t-1}
        x_tm1 = xhat[-1, :]

        
        C = rslds.emissions.Cs[0, :, :]
        d = rslds.emissions.ds[0, :]
        ytm1_pred = np.dot(C, x_tm1) + d

        if mean_sub: 

            ## Mean 
            mn = rslds.mean_sub
            ix = rslds.keep_ix

            ### set all to mean and update relevant ones 
            yt_p = mn.copy()
            yt_p[ix] = ytm1_pred + mn[ix]

            yt_pred_all.append(yt_p)
        else: 
            yt_pred_all.append(ytm1_pred)

    return yt_pred_all

#### UTILS #####
def panda_to_dict(D):
    d = dict()
    for k in list(D.keys()):
        d[k] = np.array(D[k][:])
    return d

def fit_ridge(y_train, data_temp_dict, x_var_names, alpha = 1.0,
    test_data=None, test_data2=None, train_data2=None,
    only_potent_predictor = False, only_command = False, KG = None, KG_pot = None, 
    fit_task_specific_model_test_task_spec = False,
    fit_intercept = True, model_nm=None, nneur=None,
    normalize_vars = False, keep_bin_spk_zsc = False):

    ''' fit Ridge regression using alpha = ridge parameter
        only_potent_predictor --> multiply KG_pot -- which I think should be N x N by data;  '''

    if model_nm == 'diagonal_dyn':
        return fit_diag(data_temp_dict, nneur)
    else:
        ### Initialize model 
        model_2 = Ridge(alpha=alpha, fit_intercept=fit_intercept)

        ### Aggregate the variable name
        X, mnX, stX = aggX(data_temp_dict, x_var_names, normalize_vars)
        if train_data2 is not None: 
            X2, _, _ = aggX(train_data2, x_var_names, normalize_vars)
            if normalize_vars:
                print('Different norm values for X1 and X2 ')
                import pdb; pdb.set_trace()
            X = np.vstack((X, X2))

        if only_potent_predictor:
            assert KG_pot is not None

            #### Assuuming that the trianing data is some sort of spiking thing
            assert(KG_pot.shape[1] == X.shape[1])
            pre_X_shape = X.shape
            X = np.dot(KG_pot, X.T).T
            
            ### Assuming KG_pot is the 
            assert(X.shape == pre_X_shape)
            print('only potent used to train ridge')

        elif only_command:
            assert KG is not None

            #### Assuuming that the trianing data is some sort of spiking thing
            assert(KG.shape[1] == X.shape[1])
            pre_X_shape = X.shape
            
            X = np.dot(KG, X.T).T
            y_train = np.dot(KG, y_train.T).T
            assert(X.shape == y_train.shape)

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

            ##### Add normalize: 
            model_2.normalize_vars = normalize_vars
            if normalize_vars:
                model_2.normalize_vars_mn = mnX
                model_2.normalize_vars_std = stX

            if 'psh_2' in model_nm:
                ###### Conditioning needs an error covariance; 
                pred_x = model_2.predict(X)
                W = np.cov(y_train.T - pred_x.T)
                model_2.W = W 

        if test_data is None:
            return model_2
        
        else:
            if normalize_vars:
                print('havent dealt with usingnrom vars to predict yet')
                import pdb; pdb.set_trace()

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

def aggX(data, var_nms, normalize_vars):
    x = []; mn = []; std = []; 
    for vr in var_nms:
        if vr == 'mov':
            mov_hot = getonehotvar(data[vr][:, np.newaxis])
            x.append(mov_hot)
            mn.append(np.zeros((mov_hot.shape[1])) )
            std.append(np.ones((mov_hot.shape[1])) )
        else:
            if normalize_vars:
                tmp = data[vr][: , np.newaxis]
                
                ### Avoid divide by zero errors
                if np.std(tmp) == 0:
                    sd = 1.
                else:
                    sd = np.std(tmp)
                                
                x.append((tmp - np.mean(tmp)) / sd)
                mn.append(np.mean(tmp))
                std.append(sd)
                
            else:
                x.append(data[vr][: , np.newaxis])
                mn.append(0)
                std.append(1)

    X = np.hstack((x))

    if 'mov' in var_nms:
        assert(X.shape[1] == len(var_nms) - 1 + 28)
    else:
        assert(X.shape[1] == len(var_nms))
    assert(len(np.hstack((mn))) == X.shape[1])
    assert(len(np.hstack((std)) == X.shape[1]))

    return X, np.hstack((mn)), np.hstack((std))

def getonehotvar(var):

    CO = np.arange(8)
    OBS = np.arange(10, 20)
    OBS2 = np.arange(10, 20) + 0.1

    cols = list(np.hstack((CO, OBS, OBS2)))
    X = np.zeros((var.shape[0], len(cols)))
    for iv, v in enumerate(np.squeeze(var)):
        tmp = cols.index(v)
        X[iv, tmp] = 1
    assert(np.all(np.sum(X, axis=1)) == 1)
    return X

def fit_LDS(data_temp_dict, x_var_names, trl_train_ix,
    nEMiters = 30, n_dim_latent = 'full', 
    fit_intercept = False):

    ### Aggregate the variable name
    x = [] #### So this is X_{t-1}, so t=0:end_trial(-1)
    for vr in x_var_names:
        x.append(data_temp_dict[vr][: , np.newaxis])
    
    X = np.hstack((x))

    if fit_intercept: 
        ### subtract mean 
        spike_mn = np.mean(X, axis=0)
        X = X - spike_mn[np.newaxis, :]
    else:
        spike_mn = np.zeros((X.shape[1]))

    assert(X.shape[1] == len(x_var_names))
    numN = X.shape[1]

    ### Only neurons with FR > 0.5 ### 
    if fit_intercept: 
        keep_ix = np.nonzero(10*spike_mn > 0.5)[0]
    else: 
        keep_ix = np.nonzero(10*np.sum(X, axis=0)/float(X.shape[0]) > 0.5)[0]

    #### Retrospectively check variable order ####
    check_variables_order(x_var_names, numN)

    #### Make trials #####
    trls = data_temp_dict['trl']
    bin_num = data_temp_dict['bin_num']

    ##### Append to list ####
    train_trls = []
    for i_t in trl_train_ix: 
        ix = np.nonzero(trls==i_t)[0]
        assert(np.all(np.diff(bin_num[ix]) == 1))
        train_trls.append(X[np.ix_(ix, keep_ix)])

    #### Now get LDS based on these trials ###
    D_obs = len(keep_ix)
    D_input = 0; 
    if n_dim_latent == 'full':
        n_dim_latent = D_obs
        try_dims = np.arange(n_dim_latent, 15, -1)
    elif type(n_dim_latent) is np.int64:
        try_dims = [n_dim_latent]
    
    for n_dim_latent_try in try_dims:
        print(('starting to try to fit Dim %d' %(n_dim_latent_try)))
        model = DefaultLDS(D_obs, n_dim_latent_try, D_input)

        for trl in train_trls:
            model.add_data(trl) ### T x N
        
        ######## Initialize matrices w/ FA #########
        FA = skdecomp.FactorAnalysis(n_components=n_dim_latent_try)
        dat = np.vstack((train_trls))
        FA.fit(dat)
        x_hat = FA.transform(dat)
        # Do main shared variance to solve this issue: 
        A = np.mat(np.linalg.lstsq(x_hat[:-1, :], x_hat[1:, :])[0])
        err = x_hat[1:, :].T - A*x_hat[:-1, :].T
        err_obs = dat.T - np.mat(FA.components_).T*x_hat.T

        model.C = FA.components_.T
        model.A = A
        model.spike_mn = spike_mn ### mean that was subtracted

        if n_dim_latent == 1:
            model.sigma_states = np.array([[np.cov(err)]])
            model.sigma_obs = np.array([[np.cov(err_obs)]])
        else:
            model.sigma_states = np.cov(err)
            model.sigma_obs = np.cov(err_obs)

        #############
        # Train LDS #
        ############# 
        def update(model):
            model.EM_step()
            return model.log_likelihood()
        
        try:
            lls = [update(model) for i in range(nEMiters)]
            model.keep_ix = keep_ix
            model.lls = lls
            return model
        except:
            pass

    ### If you've gotten here, you've failed to fit any 
    raise Exception('Cant fit any models above dim 15')

def smooth_LDS(data_temp, model, variables, trl_test_ix, i_f): 
    trls = data_temp['trl']
    bin_num = data_temp['bin_num']
    keep_ix = model.keep_ix
    spike_mn = model.spike_mn

    ### Aggregate the variable name
    x = [] #### So this is X_{t-1}, so t=0:end_trial(-1)
    for vr in variables:
        x.append(data_temp[vr][: , np.newaxis])
    X = np.hstack((x))
    X = X - spike_mn[np.newaxis, :]
    assert(X.shape[1] == len(variables))

    ########### Append to list ###########
    test_trls = []; 
    test_ix = [ [] for i in range(i_f+1)]
    
    for i_t in trl_test_ix: 
        ix = np.nonzero(trls==i_t)[0]
        assert(np.all(np.diff(bin_num[ix]) == 1))
        test_trls.append(X[np.ix_(ix, keep_ix)])
        test_ix[i_f].append(ix)

    #import pdb; pdb.set_trace()

    ########## Make predictions #########
    smooth_Y = [] 
    for trl in test_trls:

        ##### Add data, pop it off again ####
        model.add_data(trl)
        g = model.states_list.pop()
        
        # Smoothed (y_t | y0...yT)
        smoothed_trial = g.smooth()

        assert(smoothed_trial.shape == trl.shape)

        smooth_Y.append(smoothed_trial + spike_mn[np.newaxis, keep_ix])
    
    #### Stack up ####
    test_ix[i_f] = np.hstack((test_ix[i_f]))
    smooth_Y = np.vstack((smooth_Y))


    ### Set estimate of all neurons equal to their mean 
    smooth_Y_all = np.tile(spike_mn, [smooth_Y.shape[0], 1])
    assert(smooth_Y_all.shape[0] == smooth_Y.shape[0])
    assert(smooth_Y_all.shape[1] == X.shape[1])

    ## Update the relevant ones; 
    for i, ix in enumerate(keep_ix):
        smooth_Y_all[:, ix] = smooth_Y[:, i] 

    return smooth_Y_all, test_ix 

def pred_LDS(data_temp, model, variables, trl_test_ix, i_f, 
    window_size = 'full'):

    trls = data_temp['trl']
    bin_num = data_temp['bin_num']
    keep_ix = model.keep_ix
    spike_mn = model.spike_mn

    ### Aggregate the variable name
    x = [] #### So this is X_{t-1}, so t=0:end_trial(-1)
    for vr in variables:
        x.append(data_temp[vr][: , np.newaxis])
    X = np.hstack((x))
    X = X - spike_mn[np.newaxis, :]
    assert(X.shape[1] == len(variables))

    ########### Append to list ###########
    test_trls = []; 
    test_ix = [ [] for i in range(i_f+1)]
    
    for i_t in trl_test_ix: 
        ix = np.nonzero(trls==i_t)[0]
        assert(np.all(np.diff(bin_num[ix]) == 1))
        test_trls.append(X[np.ix_(ix, keep_ix)])
        test_ix[i_f].append(ix)

    ########## Make predictions #########
    pred_Y = [] 
    for trl in test_trls:

        if window_size == 'full': 
            ##### Add data, pop it off again ####
            model.add_data(trl)
            g = model.states_list.pop()
            
            # Smoothed (y_t | y0...yT)
            smoothed_trial = g.smooth()

            # Smoothed (x_t | x0...xT)
            x0 = g.smoothed_mus[0, :] # Time x ndim
            P0 = g.smoothed_sigmas[0, :, :]
        
            # Filtered states (x_t | y0...y_t)
            _, filtered_mus, _ = kalman_filter(
            x0, P0,
            g.A, g.B, g.sigma_states,
            g.C, g.D, g.sigma_obs,
            g.inputs, g.data)
            
            assert(filtered_mus.shape[0] == trl.shape[0])
            
            #### Take filtered data, propagate fwd to get estimated next time step; 
            pred_trl = np.dot(g.C, np.dot(g.A, filtered_mus.T)).T

        else: 
            print('window size pred %d'%window_size)
            ntrl, _ = trl.shape
            pred_trl = []; 

            ## so above, all data in trl0...trl{T-1} used to predict trl1...trlT
            for nt in range(ntrl): 

                ## Not enough window size ###
                if nt < (window_size-1): 
                    pred_trl.append(-1000*np.ones((1, len(keep_ix) )) )
                else: 
                    ### Add data in window from trl{t-ws}...trl{t} used to predict 
                    ix_ = np.arange(nt-window_size+1, nt+1)
                    assert(ix_[-1] == nt)
                    assert(len(ix_) == window_size)

                    model.add_data(trl[ix_, :])

                    ### pop it off 
                    g = model.states_list.pop()
                    _ = g.smooth()

                    ### take the last state and predit forward; 
                    x0 = g.smoothed_mus[-1, :] # time x ndim --> corresponds to estimate of spks[nt, :]
                    pred_trl.append(np.dot(g.C, np.dot(g.A, x0)))

            pred_trl = np.vstack((pred_trl)) # time x ndim 
            
        assert(pred_trl.shape == trl.shape)
        pred_Y.append(pred_trl + spike_mn[np.newaxis, keep_ix])

    #### Stack up ####
    test_ix[i_f] = np.hstack((test_ix[i_f]))
    pred_Y = np.vstack((pred_Y))


    #pred_Y_all = np.zeros((pred_Y.shape[0], X.shape[1]))

    ### Set estimate of all neurons equal to their mean 
    pred_Y_all = np.tile(spike_mn, [pred_Y.shape[0], 1])
    assert(pred_Y_all.shape[0] == pred_Y.shape[0])
    assert(pred_Y_all.shape[1] == X.shape[1])

    ## Update the relevant ones; 
    for i, ix in enumerate(keep_ix):
        pred_Y_all[:, ix] = pred_Y[:, i] 

    return pred_Y_all, test_ix 

def fit_diag(data_temp_dict, nneur):
    """
    Generate linear models for each neuron: yi_t = A_i*yi_{t-1} + b_i
    
    Parameters
    ----------
    data_temp_dict : dict
        Description
    nneur : int
        number of neurons
    """

    model_ = {}

    pred_y = [] 

    for i_n in range(nneur): 

        ### Get the training data; 
        y_train = data_temp_dict['spks'][: , i_n]
        x_train = data_temp_dict['spk_tm1_n%d'%i_n][: , np.newaxis]

        ### Fit the ridge model
        model_2 = Ridge(alpha=0., fit_intercept=True)
        model_2.fit(x_train, y_train)

        ### Get the noise of the prediction; 
        pred_y.append(model_2.predict(x_train)[:, np.newaxis])
        
        ### Get the noise :
        #model_2.w = np.cov(pred_y - y_train)

        ### Add model 2; 
        model_[i_n] = model_2

    pred_y = np.hstack((pred_y))
    dy = (pred_y - data_temp_dict['spks'])
    model_['w'] = np.cov(dy.T)

    return model_

def pred_diag(data_temp_dict_test, model_, nneur):
    """
    Use the model_ from fit_diag to generate predictions
    
    Args:
        data_temp_dict_test (dict): dictionary of test data; 
        model_ (dict): of Ridge models from above
        nneur (int): # of neurons 
    
    Returns:
        predY: np.array: T x N predictions 
    """
    predY = []

    for i_n in range(nneur):

        ### Test data --> x_test ###
        x_test = data_temp_dict_test['spk_tm1_n%d'%i_n][:, np.newaxis]

        ### Model N
        model_n = model_[i_n]

        ### Prediction: 
        predY.append(model_n.predict(x_test)[:, np.newaxis])

    predY = np.hstack((predY))

    return predY

def pred_diag_cond(data_temp_dict_test, model_, nneur, KG):
    """
    Construct the prediction now conditioning on acton; 
    
    Args:
        data_temp_dict_test (TYPE): Description
        model_ (TYPE): Description
        nneur (TYPE): Description
    
    Returns:
        TYPE: Description
    """

    predY = []
    cov = model_['w']

    for i_n in range(nneur):

        ### Model N
        model_n = model_[i_n]

        ### Test data --> x_test ###
        x_test = data_temp_dict_test['spk_tm1_n%d'%i_n][:, np.newaxis]

        ### Prediction: 
        predY.append(model_n.predict(x_test)[:, np.newaxis])

    ### predY --> back together 
    predY = np.hstack((predY))

    cov12 = np.dot(KG, cov).T
    cov21 = np.dot(KG, cov)
    cov22 = np.dot(KG, np.dot(cov, KG.T))
    cov22I = np.linalg.inv(cov22)


    ### Get action from t = 0 ###
    A = data_temp_dict_test['psh']

    assert(A.shape[0] == predY.shape[0])

    T = predY.shape[0]

    ### For each time point ###
    predY_w_cond = []; 

    for i_t in range(T):

        ### Get this prediction (mu 1)
        mu1_i = predY[i_t, :].T
        mu1_i = mu1_i[:, np.newaxis]

        ### Get predicted value of action; 
        mu2_i = np.dot(KG, mu1_i)

        ### Actual action; 
        a_i = A[i_t, :][:, np.newaxis]

        ### Conditon step; 
        mu1_2_i = mu1_i + np.dot(cov12, np.dot(cov22I, a_i - mu2_i))

        ### Make sure it matches; 
        assert(np.allclose(np.dot(KG, mu1_2_i), a_i))

        predY_w_cond.append(np.squeeze(np.array(mu1_2_i)))

    predY_w_cond = np.vstack((predY_w_cond))

    return predY_w_cond

def identity_dyn(data_temp_dict, nneur):
    pred_Y = []
    for i in range(nneur): 
        pred_Y.append(data_temp_dict['spk_tm1_n%d'%(i)])
    return np.vstack((pred_Y)).T

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

        elif include_state == 4:
            ### task-related model / string 
            vel_model_nms = []; 
            pos_model_nms = []; 
            tsk_model_nms, tsk_model_str = ['tsk', '']

        elif include_state == 5:
            vel_model_nms, vel_model_str = generate_models_utils.lag_ix_2_var_nm(model_vars, 'vel')
            pos_model_nms, pos_model_str = generate_models_utils.lag_ix_2_var_nm(model_vars, 'pos')

            ### Also include target_info: 
            tg_model_nms, tg_model_str = generate_models_utils.lag_ix_2_var_nm(model_vars, 'tg')    
            tsk_model_nms, tsk_model_str = ['mov', '']           

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
        #### Unique without resorting the neurons for dynamics models 
        variables_keep = []
        for v in variables:
            if v in variables_keep:
                pass
            else:
                variables_keep.append(v)
    
        if len(variables_keep) > 0:
            variables_keep = np.hstack((variables_keep))

            ############ Double check that neurons are in the right order; ###############
            check_variables_order(variables_keep, nneur)
            variables_list.append(np.hstack((variables_keep)))
        else:
            check_variables_order(variables_keep, nneur)
            variables_list.append(variables_keep)
        
    return variables_list

def plot_sweep_alpha(animal, alphas = None, model_set_number = 1, ndays=None, skip_plots = True, 
    r2_ind_or_pop = 'pop', fit_intercept = True, within_bin_shuffle = False, null = False):

    ##### Removed the list of model_var_list ######
    model_var_list, _, _, _ = generate_models_list.get_model_var_list(model_set_number)
    model_names = [i[1] for i in model_var_list]

    ##### Plotted 
    if animal in ['grom', 'home']:
        if fit_intercept:
            if within_bin_shuffle:
                hdf = tables.openFile(analysis_config.config['%s_pref'%animal] + '%s_sweep_alpha_days_models_set%d_shuff.h5' %(animal, model_set_number))
            else:
                hdf = tables.openFile(analysis_config.config['%s_pref'%animal] + '%s_sweep_alpha_days_models_set%d.h5' %(animal, model_set_number))
        else:
            hdf = tables.openFile(analysis_config.config['%s_pref'%animal] + '%s_sweep_alpha_days_models_set%d_no_intc.h5' %(animal, model_set_number))
        
        if ndays is None:
            if animal == 'grom':
                ndays = 9; 
            elif animal == 'home':
                ndays = 5; 

    elif animal == 'jeev':
        if fit_intercept:
            if within_bin_shuffle:
                hdf = tables.openFile(analysis_config.config['jeev_pref'] + 'jeev_sweep_alpha_days_models_set%d_shuff.h5' %model_set_number)
            else:
                hdf = tables.openFile(analysis_config.config['jeev_pref'] + 'jeev_sweep_alpha_days_models_set%d.h5' %model_set_number)
        else:
            hdf = tables.openFile(analysis_config.config['jeev_pref'] + 'jeev_sweep_alpha_days_models_set%d_no_intc.h5' %model_set_number)
        
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
            if model_nm in ['identity_dyn', 'diagonal_dyn']:
                pass
            else:
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

                            if null and 'psh_2' in model_nm:
                                pass
                            else:
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

def check_variables_order(variables, nneur):
    checked = False 
    for spk in ['spk_tm0_n0', 'spk_tm1_n0', 'spk_tm2_n0', 'spk_tp1_n0', 'spk_tp2_n0']:
        if spk in variables:
            checked = True

            ### Starting neuron ####
            ix = np.nonzero(variables == spk)[0]
            for n in range(nneur):

                ### Make sure variables are ordeered correctly ###
                assert(variables[ix+n] == spk[:-1] + '%d'%(n))
    if checked:
        pass
    elif len(variables) == 0:
        print('Identity dynamcis')
    else:
        print('No spikes in this model')

#### Decoder UTILS ####
def get_KG_decoder_home(day_ix): 
    return get_KG_decoder_grom(day_ix, animal='home')

def get_KG_decoder_grom(day_ix, animal='grom'):
    co_obs_dict = pickle.load(open(analysis_config.config['%s_pref'%animal]+'co_obs_file_dict.pkl'))
    input_type = analysis_config.data_params['%s_input_type'%animal]

    ### First CO task for that day: 
    if animal == 'grom':
        te_num = input_type[day_ix][0][0]
    elif animal == 'home':
        te_num = input_type[day_ix][0]
    dec = co_obs_dict[te_num, 'dec']
    decix = dec.rfind('/')
    decoder = pickle.load(open(analysis_config.config['%s_pref'%animal]+dec[decix:]))
    F, KG = decoder.filt.get_sskf()
    KG_potent = KG[[3, 5], :]; # 2 x N
    KG_null = scipy.linalg.null_space(KG_potent) # N x (N-2)
    KG_null_proj = np.dot(KG_null, KG_null.T)

    ## Get KG potent too; 
    U, S, Vh = scipy.linalg.svd(KG_potent); #[2x2, 2, 44x44]
    Va = np.zeros_like(Vh)
    Va[:2, :] = Vh[:2, :]
    KG_potent_orth = np.dot(Va.T, Va)

    ### Make sure the eigenvalues are 1 
    ev, _ = np.linalg.eig(KG_potent_orth)
    assert(np.sum(np.round(np.abs(ev), 10) == 1.) == 2)
    ev, _ = np.linalg.eig(KG_null_proj)
    assert(np.sum(np.round(np.abs(ev), 10)== 1.) == KG.shape[1] - 2)
    
    if animal == 'grom':
        return KG_potent, KG_null_proj, KG_potent_orth
    elif animal == 'home':
        return KG_potent, KG_null_proj, KG_potent_orth, decoder.mFR, decoder.sdFR

def generate_KG_decoder_jeev():
    KG_approx = dict() 

    ### Task filelist ###
    filelist = file_key.task_filelist
    days = len(filelist)
    binsize_ms = 5.
    #important_neuron_ix = dict()

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

                print((unbinned['start_index_overall'], i1, i1 -  unbinned['start_index_overall']))

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
        print(('KG est: shape: ', KG.shape, ', R2: ', R2_best))
        import pdb; pdb.set_trace()
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
