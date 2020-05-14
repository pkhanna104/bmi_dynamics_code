import numpy as np

def get_model_var_list(model_set_number):
    ####### INCLUDE STATE VARIABLE -- at lags indicated in "lags" variable
    #######    1 -- velocity & position 
    #######    -1 --  position 
    #######    2 -- velocity & position & target
    #######    3 -- velocity & position & target & task
    #######    0 -- none; 
    
    ####### Include action_lags --> whether to include push_{t-1} etc. WILL include the lags if psh_1 is in the model name; 
    ###### Include past_y{t-1} if zero -- nothing. If > 1, include that many past lags

    ######## include y_{t-1} / y_{t+1} is just an INDICATOR. The lags that are inlcuded are the lags in "lags" entry; 

    model_var_list = []
    include_action_lags = True;
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
        include_action_lags = False; ### Only ever want action at t = 0; 
        history_bins_max = 1; 

    # elif model_set_number == 2:
    #     ### Lags, name, include state?, include y_{t-1}, include y_{t+1} ###
    #     #nclude_state, include_past_yt, include_fut_yt

    #     ### Also want only neural, only history of neural, and a_t plus history of neural. 
    #     model_var_list.append([np.array([0]),                 'prespos_0psh_0spksm_1_spksp_0',         0, 1, 0])     #### Model 1: a_{t+1} | y_{t+1} --> should be 100%
    #     model_var_list.append([np.array([-1]),                'hist_1pos_0psh_0spksm_1_spksp_0',       0, 1, 0])     #### Model 1: a_{t+1} | y_{t}
    #     model_var_list.append([np.array([-1]),                'hist_1pos_0psh_1spksm_0_spksp_0',       0, 0, 0])     #### Model 1: a_{t+1} | a_t
    #     model_var_list.append([np.array([-1]),                'hist_1pos_0psh_1spksm_1_spksp_0',       1, 0, 0])     #### Model 2: a_{t+1} | a_t, y_t
    #     model_var_list.append([np.array([-4, -3, -2, -1]),    'hist_4pos_0psh_1spksm_0_spksp_0',       0, 0, 0])     #### Model 3: a_{t+1} | a_t, a_{t-1},...
    #     model_var_list.append([np.array([-4, -3, -2, -1]),    'hist_4pos_0psh_1spksm_1_spksp_0',       0, 1, 0])     #### Model 4: a_{t+1} | a_t, a_{t-1},..., y_t; 
    #     model_var_list.append([np.array([-4, -3, -2, -1]),    'hist_4pos_0psh_1spksm_4_spksp_0',       0, 4, 0])     #### Model 5: a_{t+1} | a_t, a_{t-1},..., y_t, y_{t-1},...;
    #     predict_key = 'psh'
    #     history_bins_max = 4; 

    elif model_set_number == 3:
        ### Model predicting spikes with a) current spikes b) previous spikes, c) previous actions 
        ###                                                                              include state // y_{t-1} // y_{t+1} ###
        model_var_list.append([np.array([0]),                 'prespos_0psh_0spksm_1_spksp_0',         0, 1, 0])     #### Model 1: y_{t+1} | y_{t+1} --> should be 100%
        model_var_list.append([np.array([-1]),                'hist_1pos_0psh_0spksm_1_spksp_0',       0, 1, 0])     #### Model 1: y_{t+1} | y_{t}
        model_var_list.append([np.array([-1]),                'hist_1pos_0psh_1spksm_0_spksp_0',       0, 0, 0])     #### Model 1: y_{t+1} | a_t
        predict_key = 'spks'
        include_action_lags = True; ### For line item (3) we want action at lag 1; 
        history_bins_max = 1;

    # elif model_set_number == 4:
    #     #### Here we only want action at time T, and only state at the given lag, not all lags;
    #     model_var_list.append([np.array([0]), 'prespos_0psh_1spksm_0_spksp_0',       0, 0, 0])     ### t=0, only action
    #     model_var_list.append([np.array([0]), 'prespos_0psh_1spksm_1_spksp_0',       0, 1, 0])     ### t=0, only action & n_t 
    #     model_var_list.append([np.array([0]), 'prespos_1psh_1spksm_0_spksp_0',       1, 0, 0])     ### t=0,  action and state
    #     model_var_list.append([np.array([0]), 'prespos_1psh_1spksm_1_spksp_0',       1, 1, 0])     ### t=0, only action and state and n_t 
    #     model_var_list.append([np.array([-1]), 'hist_1pos_1psh_1spksm_0_spksp_0',     1, 0, 0])     ### t=0,  action and state @ -1
    #     model_var_list.append([np.array([-1]),  'hist_1pos_1psh_1spksm_1_spksp_0',    1, 1, 0])     ### t=0, only action and state t-1 & n_t 
    #     model_var_list.append([np.array([-2]), 'hist_2pos_1psh_1spksm_0_spksp_0',     1, 0, 0])     ### t=0,  action and state @ -1
    #     model_var_list.append([np.array([-2]), 'hist_2pos_1psh_1spksm_1_spksp_0',     1, 1, 0])     ### t=0, only action and state t-2 adn n_t 
    #     predict_key = 'spks'
    #     history_bins_max = 2; 
    #     include_action_lags = False ### only add action at current time point; 
    #     history_bins_max = 2; 

    # elif model_set_number == 5:
    #     model_var_list.append([np.array([-1]), 'hist_1pos_1psh_0spks_0_spksp_0', 0, 0, 0]) ### state only for regression again.  
    #     model_var_list.append([np.array([-1]), 'hist_1pos_0psh_1spks_1_spksp_0', 0, 1, 0]) ### everything except state: (spks / psh)
    #     model_var_list.append([np.array([-1]), 'hist_1pos_0psh_0spks_1_spksp_0', 0, 1, 0]) ### everything excpet state: (spks)
    #     model_var_list.append([np.array([-1]), 'hist_1pos_1psh_0spks_1_spksp_0', 1, 1, 0]) ### Include both for comparison to jointly fitting. 
    #     model_var_list.append([np.array([-1]), 'hist_1pos_1psh_1spks_1_spksp_0', 1, 1, 0]) ### Include both for comparison to jointly fitting. 
    #     predict_key = 'spks'
    #     history_bins_max = 1; 

    # elif model_set_number == 6:
    #     model_var_list.append([np.array([-1]), 'hist_1pos_1psh_0spks_0_spksp_0', 1, 0, 0]) ### Only state; 
    #     predict_key = 'psh'
    #     history_bins_max = 1; 

    elif model_set_number == 7:
        model_var_list.append([np.array([-1]), 'hist_1pos_0psh_0spksm_1_spksp_0', 0, 1, 0]) ### only previous neural activity; 
        model_var_list.append([np.array([-1]), 'hist_1pos_0psh_1spksm_1_spksp_0', 0, 1, 0]) ### only previous neural activity & action; 
        predict_key = 'spks'
        history_bins_max = 1; 
        # Not sure about htis ###
        #include_action_lags = True;

    elif model_set_number == 8:
        ### Dissecting previous state encoding vs. neural encoding. 
        model_var_list.append([np.array([-1]), 'hist_1pos_0psh_0spksm_1_spksp_0', 0, 1, 0]) ### only previous neural activity; 
        model_var_list.append([np.array([-1]), 'hist_1pos_1psh_0spksm_0_spksp_0', 1, 0, 0]) ### previous state
        model_var_list.append([np.array([-1]), 'hist_1pos_3psh_0spksm_0_spksp_0', 3, 0, 0]) #### previous state + targ + task
        predict_key = 'spks'
        history_bins_max = 1; 

    # elif model_set_number == 9:
    #     ### Summary of all models -- R2 instead of just mean diffs ###
    #     model_var_list.append([np.arary([-1]), 'hist_1pos_0psh_0spksm_1_spksp_0', 0, 1, 0])
    #     model_var_list.append([np.arary([-1]), 'hist_1pos_0psh_1spksm_0_spksp_0', 0, 0, 0])
    #     model_var_list.append([np.arary([0]),   'prespos_0psh_1spksm_0_spksp_0',  0, 0, 0])
    #     model_var_list.append([np.arary([0]),   'prespos_1psh_0spksm_0_spksp_0',  1, 0, 0])
    #     model_var_list.append([np.arary([-1]), 'hist_1pos_1psh_0spksm_0_spksp_0', 1, 0, 0])
    #     history_bins_max = 1; 
    
    return model_var_list, predict_key, include_action_lags, history_bins_max
