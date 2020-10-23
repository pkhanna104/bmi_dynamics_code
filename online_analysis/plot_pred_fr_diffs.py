import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors

import analysis_config
from online_analysis import util_fcns, generate_models, plot_generated_models, plot_fr_diffs


def plot_example_neuron_comm_predictions(neuron_ix = 36, mag = 0, ang = 7, animal='grom', day_ix = 0, nshuffs = 1000,
                            min_bin_indices = 0, save=False, model_set_number = 6, model_nm = 'hist_1pos_0psh_2spksm_1_spksp_0'):
    """
    Plot example single neuron and population deviations from global distribution
    along with predictions 
    
    Args:
        neuron_ix (int, optional): which neuron to plot 
        mag (int, optional): mag bin 
        ang (int, optional): ang bin 
        animal (str, optional): Description
        day_ix (int, optional): Description
        nshuffs (int, optional): Description
        min_bin_indices (int, optional): cut off the first N bins and last N bins 
        save (bool, optional): save figures 
    """
    pref_colors = analysis_config.pref_colors
    
    ################################
    ###### Extract real data #######
    ################################
    spks0, push0, tsk0, trg0, bin_num0, rev_bin_num0, move0, dat = util_fcns.get_data_from_shuff(animal, day_ix)
    spks0 = 10*spks0; 

    #### Get subsampled
    tm0, _ = generate_models.get_temp_spks_ix(dat['Data'])

    ### Get subsampled 
    spks_sub = spks0[tm0, :]
    push_sub = push0[tm0, :]
    move_sub = move0[tm0]
    bin_num_sub = bin_num0[tm0]
    rev_bin_num_sub = rev_bin_num0[tm0]

    ### Get number of neurons 
    nneur = spks_sub.shape[1]
    
    ###############################################
    ###### Get predicted spikes from the model ####
    ###############################################
    model_fname = analysis_config.config[animal+'_pref']+'tuning_models_'+animal+'_model_set'+str(model_set_number)+'_.pkl'
    model_dict = pickle.load(open(model_fname, 'rb'))
    pred_spks = model_dict[day_ix, model_nm]
    pred_spks = 10*pred_spks; 

    ### Make sure spks and sub_spks match -- using the same time indices ###
    assert(np.allclose(spks_sub, 10*model_dict[day_ix, 'spks']))
    assert(np.all(bin_num0[tm0] > 0))

    ###############################################
    ###### Get shuffled prediction of  spikes  ####
    ###############################################
    pred_spks_shuffle = plot_generated_models.get_shuffled_data_v2(animal, day_ix, model_nm, nshuffs = nshuffs, 
        testing_mode = False)
    pred_spks_shuffle = 10*pred_spks_shuffle; 
    ##############################################
    ########## SETUP the plots ###################
    ##############################################
    ### Get command bins 
    mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))
    command_bins = util_fcns.commands2bins([push_sub], mag_boundaries, animal, day_ix, 
                                       vel_ix=[3, 5])[0]

    ### 2 plots --> single neuron and vector 
    fsu, axsu = plt.subplots(figsize=(3, 3))
    fvect, axvect = plt.subplots(figsize=(3, 3))

    ### Return indices for the command ### 
    ix_com = plot_fr_diffs.return_command_indices(bin_num_sub, rev_bin_num_sub, push_sub, mag_boundaries, animal=animal, 
                            day_ix=day_ix, mag=mag, ang=ang, min_bin_num=min_bin_indices,
                            min_rev_bin_num=min_bin_indices)

    cnt = 0
    mFR_vect = {};
    mFR = {}

    pred_vect = {}; 
    pred_mFR = {}; 

    shuff_vect = {}; 
    shuff_mFR = {};

    ###############################################
    ########### COllect movements ################
    ###############################################
    ### For all movements --> figure otu which ones to keep in the global distribution ###
    global_comm_indices = {}
    ix_com_global = []

    for mov in np.unique(move_sub[ix_com]):

        ### Movement specific command indices 
        ix_mc = np.nonzero(move_sub[ix_com] == mov)[0]
        
        ### Which global indices used for command/movement 
        ix_mc_all = ix_com[ix_mc] 

        ### If enough of these then proceed; 
        if len(ix_mc) >= 15:    
            global_comm_indices[mov] = ix_mc_all
            ix_com_global.append(ix_mc_all)

    ix_com_global = np.hstack((ix_com_global))

    #### now that have all the relevant movements - proceed 
    for mov in global_comm_indices.keys(): 

        ### FR for neuron ### 
        ix_mc_all = global_comm_indices[mov]

        #### Get true FR ###
        mFR[mov] = np.mean(spks_sub[ix_mc_all, neuron_ix])
        pred_mFR[mov] = np.mean(pred_spks[ix_mc_all, neuron_ix])
        shuff_mFR[mov] = np.mean(pred_spks_shuffle[ix_mc_all, neuron_ix, :], axis=0)

        FR_vect = spks_sub[ix_mc_all, :]
        pred_FR_vect = pred_spks[ix_mc_all, :]
        shuff_vector = pred_spks_shuffle[ix_mc_all, :, :]

        ### Figure out which of the "ix_com" indices can be used for shuffling for this movement 
        ix_ok, niter = plot_fr_diffs.distribution_match_global_mov(push_sub[np.ix_(ix_mc_all, [3, 5])], 
                                                     push_sub[np.ix_(ix_com_global, [3, 5])])
        
        ### which indices we can use in global distribution for this shuffle ----> #### 
        ix_com_global_ok = ix_com_global[ix_ok] 
        global_mean_vect = np.mean(spks_sub[ix_com_global_ok, :], axis=0)
        
        ### Get matching global distribution 
        mFR_vect[mov] = np.linalg.norm(np.mean(FR_vect, axis=0) - global_mean_vect)/nneur
        pred_vect[mov] = np.linalg.norm(np.mean(pred_FR_vect, axis=0) - global_mean_vect)/nneur
        shuff_vect[mov] = np.linalg.norm(np.mean(shuff_vector, axis=0) - global_mean_vect[:, np.newaxis], axis=0)/nneur
        
    ##############################################
    #### Plot by target number ###################
    ##############################################
    keys = np.argsort(np.hstack((mFR.keys())) % 10)
    xlim = [-1, len(keys)]

    #### Sort correctly #####
    for x, mov in enumerate(np.hstack((mFR.keys()))[keys]):

        ### Draw shuffle ###
        colnm = pref_colors[int(mov)%10]
        colrgba = np.array(mpl_colors.to_rgba(colnm))

        ### Set alpha according to task (tasks 0-7 are CO, tasks 10.0 -- 19.1 are OBS) 
        if mov >= 10:
            colrgba[-1] = 0.5
        else:
            colrgba[-1] = 1.0
        
        #### col rgb ######
        colrgb = util_fcns.rgba2rgb(colrgba)

        ###########################################
        ########## PLOT TRUE DATA #################
        ###########################################
        ### Single neuron --> sampling distribution 
        util_fcns.draw_plot(x, shuff_mFR[mov], 'k', np.array([1., 1., 1., 0]), axsu)
        #axsu.plot(x, mFR[mov], '.', color=colrgb, markersize=20)
        axsu.plot(x, pred_mFR[mov], '*', color=colrgb, markersize=20)
        # axsu.hlines(np.mean(spks_sub[ix_com_global, neuron_ix]), xlim[0], xlim[-1], color='gray',
        #     linewidth=1., linestyle='dashed')
        
        ### Population centered by shuffle mean 
        util_fcns.draw_plot(x, shuff_vect[mov], 'k', np.array([1., 1., 1., 0]), axvect)
        #axvect.plot(x, mFR_vect[mov], '.', color=colrgb, markersize=20)
        axvect.plot(x, pred_vect[mov], '*', color=colrgb, markersize=20)

    axsu.set_xlim([-1, 9])
    axvect.set_xlim([-1, 9])

    fsu.tight_layout()
    fvect.tight_layout()

    util_fcns.savefig(fsu, 'fig4_n36_ex')
    util_fcns.savefig(fvect, 'fig4_pop_dist_ex')
        