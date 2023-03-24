import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

import analysis_config
from online_analysis import util_fcns, generate_models, plot_generated_models, plot_fr_diffs, lds_utils
from online_analysis import generalization_plots
from util_fcns import get_color

from sklearn.linear_model import Ridge
import scipy.stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import gc, copy

###### Supp Fig 4 plot #####
def plot_suppfig4_R2_bars(ridge_norm = False, fraction = False, plts=False):

    model_nms = ['cond', 'hist_1pos_1psh_2spksm_0_spksp_0', 'hist_1pos_2psh_2spksm_0_spksp_0',
        'hist_1pos_5psh_2spksm_0_spksp_0', 'hist_1pos_5psh_2spksm_1_spksp_0']
    plot_order = [1, 2, 3, 4, 0]
    colors=dict()
    dblue = np.array([46, 46, 146])/255.
    colors = ['r', dblue, dblue, dblue, analysis_config.blue_rgb]
    mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))
    ferr, axerr = plt.subplots(ncols = 2, figsize = (6, 3))

    for i_a, animal in enumerate(['grom', 'jeev']):
        f, ax = plt.subplots(figsize = (3, 3))    
        pooled_stats = []

        ### Save R2 ####
        R2 = {}
        for m in model_nms:
            R2[m] = []

        ### Get the population plots ### 
        SU_err = dict(hist_1pos_5psh_2spksm_0_spksp_0=[],
                      hist_1pos_5psh_2spksm_1_spksp_0=[],
                      cond = [])

        POP_err = dict(hist_1pos_5psh_2spksm_0_spksp_0=[],
                      hist_1pos_5psh_2spksm_1_spksp_0=[],
                      cond = [])
        
        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):

            su_err_day = dict(hist_1pos_5psh_2spksm_0_spksp_0=[],
                              hist_1pos_5psh_2spksm_1_spksp_0=[],
                              cond = [])

            pop_err_day = dict(hist_1pos_5psh_2spksm_0_spksp_0=[],
                              hist_1pos_5psh_2spksm_1_spksp_0=[],
                              cond = [])
            
            print('Animal %s, Day %d' %(animal, day_ix))
            true_Y, push, _, _, _, _, mov, dat = util_fcns.get_data_from_shuff(animal, day_ix)
            tm0, tm1 = generate_models.get_temp_spks_ix(dat['Data'])
            true_Y = 10*true_Y[tm0, :]; 
            command_bins = util_fcns.commands2bins([push[tm0, :]], mag_boundaries, animal, day_ix, 
                                       vel_ix=[3, 5])[0]
            mov = mov[tm0]

            for i_m, model_nm in enumerate(model_nms):
    
                if model_nm == 'cond':

                    ### Get activity conditioned on action 
                    pred_Y = 10*plot_generated_models.cond_act_on_psh(animal, day_ix)
                
                else:
                    ### Load data object ####
                    if i_m == 1 and day_ix == 0:
                        print('Extracting')
                        dataObj = generalization_plots.DataExtract(animal, day_ix, model_nm = model_nm, 
                            model_set_number = 12, nshuffs=0, ridge_norm = ridge_norm)
                        dataObj.load()
                        
                    pred_Y = 10*dataObj.model_dict[day_ix, model_nm]
                    assert(pred_Y.shape == true_Y.shape)
                    
                ### Get R2
                R2[model_nm].append(util_fcns.get_R2(true_Y, pred_Y))

                ## Command-mov activity error ###
                if model_nm in SU_err.keys(): 
                    if animal == 'grom' and day_ix == 0:
                        fi, axi = plt.subplots()
                        axi.set_title(model_nm)
                        xcnt = 0

                        tru_store = []
                        pop_store = []
                        mv_store = []

                    for mag in range(4):
                        for ang in range(8): 
                            for mv in np.unique(mov):

                                ix_keep = np.where((command_bins[:, 0] == mag) & (command_bins[:, 1] == ang) & (mov==mv))[0]

                                if len(ix_keep) >= 15: 

                                    tru = np.mean(true_Y[ix_keep, :], axis=0)
                                    pred = np.mean(pred_Y[ix_keep, :], axis=0)

                                    SU_err[model_nm].append(np.abs(tru-pred))
                                    POP_err[model_nm].append(np.linalg.norm(tru-pred)/float(len(tru)))

                                    su_err_day[model_nm].append(np.abs(tru-pred))
                                    pop_err_day[model_nm].append(np.linalg.norm(tru-pred)/float(len(tru)))

                                    if animal == 'grom' and day_ix == 0 and mag == 0 and ang == 7:
                                        axi.plot(xcnt, tru[36], '.', color=util_fcns.get_color(int(mv)))
                                        axi.plot(xcnt, pred[36], 's', color=util_fcns.get_color(int(mv)))
                                        xcnt += 1
                                        tru_store.append(tru)
                                        pop_store.append(pred)
                                        mv_store.append(mv)
                    
                    if animal == 'grom' and day_ix == 0:
                        ### Make pouplation plot: 
                        tru_store = np.vstack((tru_store))
                        pop_store = np.vstack((pop_store))
                        mv_store = np.hstack((mv_store))

                        _, pc_model, _ = util_fcns.PCA(tru_store, 2, mean_subtract = True, skip_dim_assertion=True)
                        trans_tru = util_fcns.dat2PC(tru_store, pc_model)
                        trans_pred = util_fcns.dat2PC(pop_store, pc_model)

                        #### Make the pca plot 
                        fpca, axpca = plt.subplots(figsize=(3, 3))

                        for i_m, m in enumerate(mv_store):
                            axpca.plot(trans_tru[i_m, 0], trans_tru[i_m, 1], '.', color=util_fcns.get_color(int(m)))
                            axpca.plot(trans_pred[i_m, 0], trans_pred[i_m, 1], 's', color=util_fcns.get_color(int(m)))

                        axpca.set_ylabel('PC 2')
                        axpca.set_xlabel('PC 1')
                        fpca.tight_layout()

        
            #### Plot error for each day: 
            for ie, err in enumerate([su_err_day, pop_err_day]):
                mn_err0 = np.mean(np.hstack(( err['cond'])))
                mn_err1 = np.mean(np.hstack(( err['hist_1pos_5psh_2spksm_0_spksp_0'])))
                mn_err2 = np.mean(np.hstack(( err['hist_1pos_5psh_2spksm_1_spksp_0'])))
                err = np.array([mn_err0, mn_err1, mn_err2])
                err = (err - mn_err0)/mn_err0
                axerr[ie].plot(np.array([0, 1, 2]) + 3*i_a, err, 'k-', linewidth=0.5)

        if plts: 
            pass
        else:
            for p in range(5):
                if p == 0: 
                    r2_ref = np.mean(R2[model_nms[p]])
                if fraction:
                    ax.bar(p, (np.mean(R2[model_nms[p]]) - r2_ref )/ r2_ref, color=colors[p])
                else:
                    ax.bar(p, np.mean(R2[model_nms[p]]), color=colors[p])
            
            for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):
                x_ = []
                for p in range(5):
                    if p == 0:
                        r2r = R2[model_nms[p]][day_ix]

                    if fraction:
                        x_.append((R2[model_nms[p]][day_ix] - r2r) / r2r)
                    else:
                        x_.append(R2[model_nms[p]][day_ix])
                ax.plot(range(5), x_, 'k-', linewidth=.5)

            if fraction:
                ax.set_ylabel('Frac. Increase in $R^2$')
            else:
                ax.set_ylabel('$R^2$')

            ax.set_xticks([])

            if fraction:    
                print('R2 relative: mn = %.3f, std = %.3f' %(np.mean(R2['cond']), np.std(R2['cond'])))

            #### Individual animal stats ###
            stat, pv = scipy.stats.ttest_rel(R2['hist_1pos_5psh_2spksm_0_spksp_0'], R2['hist_1pos_5psh_2spksm_1_spksp_0'])
            print('Animal %s, N = %d, T = %.3f, pv = '%(animal, len(R2['hist_1pos_5psh_2spksm_0_spksp_0']), stat))
            print(pv)
            #######################

            ### Print SU stats ###
            stat, pv = scipy.stats.ttest_rel(np.hstack((SU_err['hist_1pos_5psh_2spksm_0_spksp_0'])),
                np.hstack((SU_err['hist_1pos_5psh_2spksm_1_spksp_0'])))
            N = len(np.hstack((SU_err['hist_1pos_5psh_2spksm_0_spksp_0'])))
            print('SU error: Animal %s, N = %d, T = %.3f, pv = '%(animal, N, stat))
            print(pv)

            stat, pv = scipy.stats.ttest_rel(np.hstack((POP_err['hist_1pos_5psh_2spksm_0_spksp_0'])),
                np.hstack((POP_err['hist_1pos_5psh_2spksm_1_spksp_0'])))
            N = len(np.hstack((POP_err['hist_1pos_5psh_2spksm_0_spksp_0'])))
            print('POP error: Animal %s, N = %d, T = %.3f, pv = '%(animal, N, stat))
            print(pv)
            
        ##### 
        f.tight_layout()
        util_fcns.savefig(f, '%s_r2_buildup_frac%s'%(animal, str(fraction)))

        for ie, err in enumerate([SU_err, POP_err]):
            mn_err0 = np.mean(np.hstack((err['cond'])))
            mn_err1 = np.mean(np.hstack(( err['hist_1pos_5psh_2spksm_0_spksp_0'])))
            mn_err2 = np.mean(np.hstack(( err['hist_1pos_5psh_2spksm_1_spksp_0'])))
            err1 = np.array([mn_err0, mn_err1, mn_err2])
            err1 = (err1 - mn_err0)/mn_err0
            print('Animal %s, Cond err %.3f' %(animal, mn_err0))
            axerr[ie].bar(3*i_a, err1[0], color='r')
            axerr[ie].bar(1+ 3*i_a, err1[1], color=dblue)
            axerr[ie].bar(2 + 3*i_a, err1[2], color=analysis_config.blue_rgb)
    
    for axe in axerr:
        axe.set_xticks([0.5, 2.5])
        axe.set_xticklabels(['G', 'J'])
    axerr[0].set_ylabel("Activity Error (Hz)")
    axerr[1].set_ylabel("Pop. Dist. Error")
    #axerr[0].set_ylim([0.6, 1.6])
    #axerr[1].set_ylim([.05, .4])
    ferr.tight_layout()
    util_fcns.savefig(ferr, 'SU_POP_err_buildup')

###### Fig 4C r2 plot ########
def plot_fig4c_R2(cond_on_act = True, plot_act = False, nshuffs=10, keep_bin_spk_zsc = False):
    if cond_on_act:
        model_name = 'hist_1pos_0psh_2spksm_1_spksp_0'
    else:
        model_name = 'hist_1pos_0psh_0spksm_1_spksp_0'
    
    if plot_act:
        assert(cond_on_act == False)

    fax_r2, ax_r2 = plt.subplots(figsize = (4, 4))
    for ia, (animal, _) in enumerate(zip(['grom','jeev', 'home'], ['2016','2013', '2021'])):
        pooled_shuff = []

        for i_d in range(analysis_config.data_params['%s_ndays'%animal]):
            
            ### Load data ###
            dataObj = generalization_plots.DataExtract(animal, i_d, model_nm = model_name, 
                model_set_number = 6, nshuffs=nshuffs, keep_bin_spk_zsc=keep_bin_spk_zsc)
            dataObj.load()
            print('starting %s, day %d' %(animal, i_d))
            valid_ix = dataObj.valid_analysis_ix

            spks = dataObj.spks

            ### Condition on push 
            if animal == 'home' and keep_bin_spk_zsc:
                mult = 1.
            else:
                mult = 10.
            cond = mult*plot_generated_models.cond_act_on_psh(animal, i_d, keep_bin_spk_zsc=keep_bin_spk_zsc)
            r2_cond = util_fcns.get_R2(spks[valid_ix, :], cond[valid_ix, :])

            pred = dataObj.pred_spks 
            assert(pred.shape == spks.shape)

            KG = util_fcns.get_decoder(animal, i_d)

            ### Both should be conditioned
            if cond_on_act:
                if animal == 'home' and not keep_bin_spk_zsc:
                    pass
                else:   
                    assert(np.allclose(np.dot(KG, cond.T).T, np.dot(KG, pred.T).T))

            r2_pred = util_fcns.get_R2(spks[valid_ix, :], pred[valid_ix, :])

            #### Get shuffled 
            r2_shuff = []
            for i in range(nshuffs):
                if np.mod(i, 200)==0:
                    print('shuff %d' %i)
                # shuffled = plot_generated_models.get_shuffled_data_v2(animal, i_d, model_name, nshuffs = None,
                #     shuff_num = i)

                ### This has been multiplied by 10 
                #shuffi = plot_generated_models.get_shuffled_data_v2_super_stream(animal, i_d, i)
                #shuff = shuffi[:nT, :]
                #assert(np.all(shuffi[nT:, :] == 0.))
                
                # try:
                #     assert(np.allclose(10*shuffled[:, :, 0], shuff))
                # except:
                #     import pdb; pdb.set_trace()

                shuffled = dataObj.pred_spks_shuffle[:, :, i]
                r2_shuff.append(util_fcns.get_R2(spks[valid_ix, :], shuffled[valid_ix, :]))

            ax_r2.plot(ia*10 + i_d, r2_pred, '.', color=analysis_config.blue_rgb, markersize=10)
            ax_r2.plot(ia*10 + i_d+.125, r2_cond, '^', color='gray', markersize=10)
            util_fcns.draw_plot(ia*10 + i_d - .25, r2_shuff, 'k', np.array([1., 1., 1., 0.]), ax_r2, width=1.)
            ax_r2.vlines(ia*10 + i_d, r2_cond, r2_pred, 'gray', linewidth=.5)   

            ix = np.nonzero(r2_pred <= r2_shuff)[0]
            print('Animal %s, Day %d, pv %.5f, r2_pred %.3f, mn r2_shuff%.3f, 95th r2_shuff%.3f' %(animal, i_d, float(len(ix)) / float(nshuffs), 
                r2_pred, np.mean(r2_shuff), np.percentile(r2_shuff, 95)))
            pooled_shuff.append([r2_pred, r2_shuff])

            gc.collect()

            ### pooled stats
            mn_r2 = np.mean([d[0] for d in pooled_shuff])
            mn_shuff = np.mean(np.vstack(([d[1] for d in pooled_shuff])), axis=0)
            assert(len(mn_shuff) == nshuffs)
            ix = np.nonzero(mn_r2 <= mn_shuff)[0]
            print('POOLED: Animal %s, Day %d, pv %5f, r2_pred %.3f, mn r2_shuff%.3f, 95th r2_shuff%.3f' %(animal, 
                i_d, float(len(ix))/float(nshuffs), mn_r2, np.mean(mn_shuff), np.percentile(mn_shuff, 95)))

    ax_r2.set_ylabel('$R^2$')
    ax_r2.set_xlim([-1, 25])
    fax_r2.tight_layout()
    #util_fcns.savefig(fax_r2, 'ax_r2_dyn_cond_act_shuff_n%d.svg' %(nshuffs))

######## Figure 4 examples and fraction neurons well predicted etc. ###########
def plot_example_neuron_comm_predictions(neuron_ix = 36, mag = 0, ang = 7, animal='grom', 
    day_ix = 0, nshuffs = 1000, min_bin_indices = 0, model_set_number = 6, 
    model_nm = 'hist_1pos_0psh_2spksm_1_spksp_0', 
    mean_sub_PC = False):
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
        mean_sub_PC: option to make it so the PC1 plots are centered around zero (for
        comparison to color plots in supplements)
    """
    pref_colors = analysis_config.pref_colors
    pooled_stats = {}


    KG = util_fcns.get_decoder(animal, day_ix)
    assert(KG.shape[0] == 2)
    KG_null_low = scipy.linalg.null_space(KG) # N x (N-2)
    assert(KG_null_low.shape[1] + 2 == KG_null_low.shape[0])

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

    ##################################################
    ##################################################
    cond_spks = plot_generated_models.cond_act_on_psh(animal, day_ix, KG=None, dat=None)
    cond_spks = cond_spks*10; 
    assert(cond_spks.shape == pred_spks.shape)
    ##############################################
    ########## SETUP the plots ###################
    ##############################################
    ### Get command bins 
    mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))
    command_bins = util_fcns.commands2bins([push_sub], mag_boundaries, animal, day_ix, 
                                       vel_ix=[3, 5])[0]

    ### 2 plots --> single neuron and vector 
    fsu, axsu = plt.subplots(figsize=(6, 3), ncols = 2) ### single unit plot diff; 

    fvect_mc, axvect_distmcfr = plt.subplots(figsize=(4, 4))
    fvect_g, axvect_distgfr = plt.subplots(figsize=(4, 4))

    ### Return indices for the command ### 
    ix_com_global = plot_fr_diffs.return_command_indices(bin_num_sub, rev_bin_num_sub, push_sub, mag_boundaries, animal=animal, 
                            day_ix=day_ix, mag=mag, ang=ang, min_bin_num=min_bin_indices,
                            min_rev_bin_num=min_bin_indices)

    cnt = 0
    mFR_vect_disggfr = {};
    mFR = {}
    mFR_vect = {}

    pred_vect_distgfr = {}; 
    pred_vect_distmcfr = {}; 
    pred_mFR_vect = {}
    pred_mFR = {}; 

    shuff_vect_distgfr = {}; 
    shuff_vect_distmcfr = {}; 
    shuff_mFR_vect = {}
    shuff_mFR = {};


    cond_mFR = {}; 
    cond_pop_FR = {}
    ###############################################
    ########### COllect movements ################
    ###############################################
    ### For all movements --> figure otu which ones to keep in the global distribution ###
    global_comm_indices = {}

    for mov in np.unique(move_sub[ix_com_global]):

        ### Movement specific command indices 
        ix_mc = np.nonzero(move_sub[ix_com_global] == mov)[0]
        
        ### Which global indices used for command/movement 
        ix_mc_all = ix_com_global[ix_mc] 

        ### If enough of these then proceed; 
        if len(ix_mc) >= 15:    
            global_comm_indices[mov] = ix_mc_all

    ############ subsample to match across all movements ################
    # global_comm_indices = plot_fr_diffs.distribution_match_mov_multi(global_comm_indices,
    #     push_sub[:, [3, 5]], psig=0.05)

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

        #### calculate NULL distances 
        ### Distance from true firing rate 
        mn_pred_FR_null = np.squeeze(np.dot(KG_null_low.T, np.mean(pred_FR_vect, axis=0)[:, np.newaxis]))
        mn_true_FR_null= np.squeeze(np.dot(KG_null_low.T, np.mean(FR_vect, axis=0)[:, np.newaxis]))
        shuff_FR_null = np.dot(KG_null_low.T, np.mean(shuff_vector, axis=0)) #  N-2 x shuff

        pred_vect_distmcfr[mov] = np.linalg.norm(mn_pred_FR_null - mn_true_FR_null)
        shuff_vect_distmcfr[mov] = np.linalg.norm(shuff_FR_null - mn_true_FR_null[:, np.newaxis], axis=0)

        ##### Save True for later 
        mFR_vect[mov] = np.mean(FR_vect, axis=0)
        pred_mFR_vect[mov] = np.mean(pred_FR_vect, axis=0)
        shuff_mFR_vect[mov] = np.mean(shuff_vector, axis=0) # neurons x shuffs? 
        cond_mFR[mov] = np.mean(cond_spks[ix_mc_all, neuron_ix], axis=0)
        cond_pop_FR[mov] = np.mean(cond_spks[ix_mc_all, :], axis=0)

        ### Figure out which of the "ix_com" indices can be used for shuffling for this movement 
        ix_mov = np.array([i for i, j in enumerate(ix_com_global) if j in ix_mc_all])
        ix_ok, niter = plot_fr_diffs.distribution_match_global_mov(push_sub[np.ix_(ix_mc_all, [3, 5])], 
                                                     push_sub[np.ix_(ix_com_global, [3, 5])], 
                                                     keep_mov_indices_in_pool = True,
                                                     ix_mov=ix_mov)
        
        ### which indices we can use in global distribution for this shuffle ----> #### 
        ix_com_global_ok = ix_com_global[ix_ok] 
        global_mean_vect = np.mean(spks_sub[ix_com_global_ok, :], axis=0)
        
        ### Get matching global distribution 
        #mFR_vect_disggfr[mov] = np.linalg.norm(np.mean(FR_vect, axis=0) - global_mean_vect)/nneur

        ### distance form global FR 
        #pred_vect_distgfr[mov] = np.linalg.norm(np.mean(pred_FR_vect, axis=0) - global_mean_vect)/nneur
        #shuff_vect_distgfr[mov] = np.linalg.norm(np.mean(shuff_vector, axis=0) - global_mean_vect[:, np.newaxis], axis=0)/nneur
        
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
        util_fcns.draw_plot(x, shuff_mFR[mov], 'k', np.array([1., 1., 1., 0]), axsu[1], whisk_min = 2.5, whisk_max=97.5)
        axsu[0].plot(x, mFR[mov], '.', color=colrgb, markersize=20)
        axsu[1].plot(x, pred_mFR[mov], '*', color=colrgb, markersize=20)
        axsu[1].plot(x, cond_mFR[mov], '^', color='gray')

        print('SU -- Mov %.1f, Pos %d, [%.2f, %.2f], pred = %.2f,  true = %.2f'%(mov, x, np.percentile(shuff_mFR[mov], 2.5), 
            np.percentile(shuff_mFR[mov], 97.5), pred_mFR[mov], mFR[mov]))
        # axsu[0].hlines(np.mean(spks_sub[ix_com_global, neuron_ix]), xlim[0], xlim[-1], color='gray',
        #      linewidth=1., linestyle='dashed')
        # axsu[1].hlines(np.mean(spks_sub[ix_com_global, neuron_ix]), xlim[0], xlim[-1], color='gray',
        #      linewidth=1., linestyle='dashed')     

        ### Population distance from movement-command FR
        util_fcns.draw_plot(x, shuff_vect_distmcfr[mov], 'k', np.array([1., 1., 1., 0]), axvect_distmcfr)
        axvect_distmcfr.plot(x, pred_vect_distmcfr[mov], '*', color=colrgb, markersize=20)

        ### Population distance from global FR
        #util_fcns.draw_plot(x, shuff_vect_distgfr[mov], 'k', np.array([1., 1., 1., 0]), axvect_distgfr)
        #axvect_distgfr.plot(x, pred_vect_distgfr[mov], '*', color=colrgb, markersize=20)
        #axvect_distgfr.plot(x, mFR_vect_disggfr[mov], '.', color=colrgb, markersize=20)


        print('POP -- Mov %.1f, Pos %d, [%.2f, %.2f], pred = %.2f distances'%(mov, x, np.percentile(shuff_vect_distmcfr[mov], 2.5), 
            np.percentile(shuff_vect_distmcfr[mov], 97.5), pred_vect_distmcfr[mov]))


    ### Set the axes
    axsu[0].set_ylim([5, 40])
    axsu[1].set_ylim([17, 31])

    for axi in axsu: 
        axi.set_xlim([-1, 9])
    axvect_distgfr.set_xlim([-1, 9])
    axvect_distmcfr.set_xlim([-1, 9])

    axsu[0].set_ylabel('Activity (Hz)')
    axsu[1].set_ylabel('Predicted Activity (Hz)')
    axvect_distgfr.set_ylabel('Command-Mov Dist. from \n Command Activity (Hz)')
    axvect_distmcfr.set_ylabel('Dist. b/w Pred. and True \nCommand-Mov Activity (Hz)')

    for axi in [axsu[0], axsu[1], axvect_distgfr, axvect_distmcfr]:
        axi.set_xticks([])

    fsu.tight_layout()
    fvect_g.tight_layout()
    fvect_mc.tight_layout()

    util_fcns.savefig(fsu, 'fig4_n36_ex')
    util_fcns.savefig(fvect_g, 'fig4_pop_dist_from_gfr')
    util_fcns.savefig(fvect_mc, 'fig4_pop_dist_from_mcfr')

    ##### PCA Plot ########
    ##### Find 2D PC axes that best capture true mFR_vect difference: 
    #### EDIT 12/17/22 --> NULL SPACE plots instead
    mfr_mc = []; pred_mfr_mc = []; 
    for m in mFR_vect.keys():
        mfr_mc.append(np.squeeze(np.dot(KG_null_low.T, mFR_vect[m][:, np.newaxis])))
        pred_mfr_mc.append(np.squeeze(np.dot(KG_null_low.T, pred_mFR_vect[m][:, np.newaxis])))

    mfr_mc = np.vstack((mfr_mc)) # conditions x neurons - 2 #
    pred_mfr_mc = np.vstack((pred_mfr_mc)) 

    ### Get the pc model 
    _, pc_model, _ = util_fcns.PCA(mfr_mc, 2, mean_subtract = True, skip_dim_assertion=True)

    #### Make the pca plot 
    fpca, axpca = plt.subplots(figsize=(6, 3), ncols = 2)

    #### For true data, plot the coordinates: 
    keys = np.argsort(np.hstack((mFR.keys())) % 10)
    xlim = [-1, len(keys)]

    ### mean_PC1 true 
    if mean_sub_PC:
        trans_true =  util_fcns.dat2PC(mfr_mc, pc_model)
        print('trans true shape')
        print(trans_true.shape)
        mean_PC1 = np.mean(trans_true[:, 0])
        trans_pred = util_fcns.dat2PC(pred_mfr_mc, pc_model)
        mean_pred_PC1 = np.mean(trans_pred[:, 0])
        print('mean PC1 %.2f, N = %d'%(mean_PC1, len(trans_true)))
        print('mean pred PC1 %.2f, N = %d')%(mean_pred_PC1, len(trans_pred))
    else:
        mean_PC1 = 0; 
        mean_pred_PC1 = 0; 

    #### Sort correctly #####
    for x, m in enumerate(np.hstack((mFR.keys()))[keys]):
    #for m in mFR_vect.keys(): 

        ### Project data and plot ###
        trans_true = util_fcns.dat2PC(np.dot(KG_null_low.T, mFR_vect[m][:, np.newaxis]).T, pc_model)
        #axpca[0].plot(trans_true[0, 0], trans_true[0, 1], '.', color=get_color(m), markersize=20)
        axpca[0].plot(x, -1*trans_true[0, 0] - mean_PC1, '.', color=get_color(m), markersize=20)

        ### PLot the predicted data : 
        trans_pred = util_fcns.dat2PC(np.dot(KG_null_low.T, pred_mFR_vect[m][:, np.newaxis]).T, pc_model)
        axpca[1].plot(x, -1*trans_pred[0, 0] - mean_pred_PC1, '*', color=get_color(m), markersize=20)
        
        #axpca[1].plot(trans_pred[0, 0], trans_pred[0, 1], '*', color=get_color(m), markersize=20)
        #axpca[0].plot([trans_true[0, 0], trans_pred[0, 0]], [trans_true[0, 1], trans_pred[0, 1]], '-', 
        #    color=get_color(m), linewidth=1.)

        ## Plot output only: 
        trans_cond = util_fcns.dat2PC(np.dot(KG_null_low.T, cond_pop_FR[m][:, np.newaxis]).T, pc_model)
        axpca[1].plot(x, -1*trans_cond[0, 0] - mean_pred_PC1, '^', color='gray')

        ### PLot the shuffled: Shuffles x 2 
        trans_shuff = util_fcns.dat2PC(np.dot(KG_null_low.T, shuff_mFR_vect[m]).T, pc_model)
        #e = confidence_ellipse(trans_shuff[:, 0], trans_shuff[:, 1], axpca[1], n_std=3.0,
        #    facecolor = get_color(m, alpha=1.0))
        util_fcns.draw_plot(x, -1*trans_shuff[:, 0] - mean_pred_PC1, 'k', np.array([1., 1., 1., 0]), axpca[1])


    ### Add the global FR command ####
    #global_mFR = np.mean(spks_sub[ix_com_global, :], axis=0)
    #trans_global = util_fcns.dat2PC(global_mFR[np.newaxis, :], pc_model)

    ## Dashed output ## 
    #axpca[0].hlines(trans_global[0, 0], -1, len(keys), linestyles='dashed')
    #axpca[0].plot(trans_global[0, 0], trans_global[0, 1], 'k.', markersize=10)
    #axpca[1].plot(trans_global[0, 0], trans_global[0, 1], 'k.', markersize=10)
    
    for axpcai in axpca:
        axpcai.set_ylabel('PC1')
        axpcai.set_xlabel('Condition')
        axpcai.set_xlim(xlim)

    ## UNCOMMENT TEMP ##
    if mean_sub_PC: 
        axpca[0].set_ylim([-25, 20])
        axpca[1].set_ylim([-10, 8])

    else:
        axpca[0].set_ylim([14, 57])
        axpca[1].set_ylim([32, 48])

    fpca.tight_layout()
    util_fcns.savefig(fpca,'fig4_eg_pca')

    ##### 12/17/22 -- deprecated 
    # fdyn, axdyn = plt.subplots(figsize=(6, 3), ncols = 2)
    # ### Make these axes teh dynamics axes 
    # #model_data[i_d, model_nm, i_fold, type_of_model_index, 'model'] 
    # model = model_dict[day_ix, model_nm, 0, 0, 'model']

    # A = model.coef_
    # b = model.intercept_
    # dim0 = 0; 
    # dim1 = 1; 

    # ### High dim axis in dominant dynamics dimensions: 
    # assert(A.shape[0] == A.shape[1])
    
    # ### Get the eigenvalue / eigenvectors: 
    # T, evs = lds_utils.get_sorted_realized_evs(A)
    # T_inv = np.linalg.pinv(T)

    # ### Linear transform of A matrix
    # Za = np.real(np.dot(T_inv, np.dot(A, T)))

    # ### Which eigenvalues are these adn what are their properties? 
    # dt = 0.1
    # td = -1/np.log(np.abs(evs[[dim0, dim1]]))*dt; 
    # hz0 = np.angle(evs[dim0])/(2*np.pi*dt)
    # hz1 = np.angle(evs[dim1])/(2*np.pi*dt)
    # print('TIme decay %s'%(str(td)))
    # print('Hz %.3f, %.3f'%(hz0, hz1))

    # #### For true data, plot the coordinates: 
    # for m in mFR_vect.keys(): 

    #     ### Project data and plot ###
    #     trans_true = np.dot(T_inv, mFR_vect[m][:, np.newaxis])
    #     axdyn[0].plot(trans_true[0, 0], trans_true[1, 0], '.', color=get_color(m), markersize=20)

    #     ### PLot the predicted data : 
    #     trans_pred = np.dot(T_inv, pred_mFR_vect[m][:, np.newaxis])
    #     axdyn[1].plot(trans_pred[0, 0], trans_pred[1, 0], '*', color=get_color(m), markersize=20)
        
    #     ### PLot the shuffled: Shuffles x 2 
    #     trans_shuff = np.dot(T_inv, shuff_mFR_vect[m])
    #     e = confidence_ellipse(trans_shuff[0, :], trans_shuff[1, :], axdyn[1], n_std=3.0,
    #         facecolor = get_color(m, alpha=1.0))
    
    # ### Add the global FR command ####
    # global_mFR = np.mean(spks_sub[ix_com_global, :], axis=0)
    # trans_global = np.dot(T_inv, global_mFR[:, np.newaxis])
    # #axdyn[0].plot(trans_global[0, 0], trans_global[1, 0], 'k.', markersize=10)
    # #axdyn[1].plot(trans_global[0, 0], trans_global[1, 0], 'k.', markersize=10)
    
    # for axi in axdyn:
    #     axi.set_xlabel('Dim 1')
    #     axi.set_ylabel('Dim 2')
    # fdyn.tight_layout()
    # util_fcns.savefig(fdyn,'fig4_eg_dyn')

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      edgecolor=facecolor, facecolor=np.array([1., 1., 1., 0.]), linewidth=0.5, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
def frac_sig_science_compressions(nshuffs = 1000, min_bin_indices = 0, 
    model_set_number = 6, model_nm = 'hist_1pos_0psh_2spksm_1_spksp_0', 
    only_sig_cc=False, min_com_cond = 15, save = True):
    '''
    min_bin_indices --> minimum of bin index to include (default = 0), if more then willl exclude initial parts of trial 
    only_sig_cc --> whether to use only sig CCs to assess sig. neuron distances // model R2 predictions 
    NOTE: 12/17/22 --> single neuron distances are full actiivty distances whereas sig. CC and sig Commands are 
        based on null distances 
        Model r2 predictions are also based on null distances 
    min_com_cond --> how many commadn conditions are needed to analyze a given command's condition
    '''

    #### Each plot ####
    f_fracCC, ax_fracCC = plt.subplots(figsize=(2, 3))
    f_fracCom, ax_fracCom = plt.subplots(figsize=(2, 3))
    f_fracN, ax_fracN = plt.subplots(figsize=(2, 3))
    f_r2, ax_r2 = plt.subplots(figsize=(2, 3))
    
    ylabels = dict()
    ylabels['fracCC'] = 'frac. (command,condition) \nsig. predicted in null'
    ylabels['fracCom']= 'frac. (command) \nsig. predicted in null'
    ylabels['fracN']  = 'frac. (neuron)  \nsig. predicted in full'
    ylabels['r2'] = 'r2 of condition-specific null \ncomponent of command'
    ylabels['r2_shuff'] = 'r2 of condition-specific null \ncomponent of command'

    count = {}
    count['fracCC'] = []
    count['fracCom'] = []
    count['fracN'] = []

    for ia, animal in enumerate(['grom', 'jeev']):
        bar_dict = dict(fracCC=[], fracCom=[], fracN=[], r2=[], r2_shuff=[])
        stats_dict = dict(CC=[], Neur=[], R2=[])
        
        model_fname = analysis_config.config[animal+'_pref']+'tuning_models_'+animal+'_model_set'+str(model_set_number)+'_.pkl'
        model_dict = pickle.load(open(model_fname, 'rb'))

        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):

            day_stats_dict = dict(CC=[])
            
            ### Get KG_null 
            KG = util_fcns.get_decoder(animal, day_ix)
            assert(KG.shape[0] == 2)
            KG_null_low = scipy.linalg.null_space(KG) # N x (N-2)
            assert(KG_null_low.shape[1] + 2 == KG_null_low.shape[0])

            ################################
            ###### Extract real data #######
            ################################
            spks0, push0, tsk0, trg0, bin_num0, rev_bin_num0, move0, dat = util_fcns.get_data_from_shuff(animal, day_ix)
            spks0 = 10*spks0; 

            #### Get subsampled
            tm0, _ = generate_models.get_temp_spks_ix(dat['Data'])

            ### Get subsampled 
            spks_sub = spks0[tm0, :]
            spks_null_sub = np.dot(KG_null_low.T, spks_sub.T).T 
            
            push_sub = push0[tm0, :]
            move_sub = move0[tm0]
            bin_num_sub = bin_num0[tm0]
            rev_bin_num_sub = rev_bin_num0[tm0]

            ### Get number of neurons 
            nneur = spks_sub.shape[1]
    
            ###############################################
            ###### Get predicted spikes from the model ####
            ###############################################
            pred_spks = model_dict[day_ix, model_nm]
            pred_spks = 10*pred_spks; 
            cond_spks = 10*plot_generated_models.cond_act_on_psh(animal, day_ix)

            pred_spks_null = np.dot(KG_null_low.T, pred_spks.T).T
            cond_spks_null = np.dot(KG_null_low.T, cond_spks.T).T

            ### Make sure spks and sub_spks match -- using the same time indices ###
            assert(np.allclose(spks_sub, 10*model_dict[day_ix, 'spks']))
            assert(np.all(bin_num0[tm0] > 0))

            ###############################################
            ###### Get shuffled prediction of  spikes  ####
            ###############################################
            pred_spks_shuffle = plot_generated_models.get_shuffled_data_v2(animal, day_ix, model_nm, nshuffs = nshuffs, 
                testing_mode = False)
            pred_spks_shuffle = 10*pred_spks_shuffle; # time x neurons x shuffle ID 
            
            pred_spks_null_shuffle = []
            for ns in range(nshuffs): 
                pred_spks_null_shuffle.append(np.dot(KG_null_low.T, pred_spks_shuffle[:, :, ns].T).T)
            pred_spks_null_shuffle = np.dstack((pred_spks_null_shuffle))
            assert(pred_spks_null_shuffle.shape[0]==pred_spks_shuffle.shape[0])
            assert(pred_spks_null_shuffle.shape[1]==pred_spks_shuffle.shape[1]-2)
            assert(pred_spks_null_shuffle.shape[2]==pred_spks_shuffle.shape[2])
            
            ##############################################
            ########## SETUP the plots ###################
            ##############################################
            ### Get command bins 
            mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))
            command_bins = util_fcns.commands2bins([push_sub], mag_boundaries, animal, day_ix, 
                                               vel_ix=[3, 5])[0]

            ####### Metrics ##########
            nCC=0; nCC_sig=0; 
            nCom=0; nCom_sig=0;
            nNeur=0; nNeur_sig=0; 
            nneur = pred_spks_shuffle.shape[1]
            r2_ = []
            r2_shuff = []

            #### full distances 
            neur_com_mov = dict(vals = [], shuffs = [])

            ### null distances 
            mFR_for_r2 = []; mFR_for_r2_shuff = []

            for mag in range(4):

                for ang in range(8): 

                    ### Return indices for the command ### 
                    ix_com_global = plot_fr_diffs.return_command_indices(bin_num_sub, rev_bin_num_sub, push_sub, 
                                            mag_boundaries, animal=animal, 
                                            day_ix=day_ix, mag=mag, ang=ang, min_bin_num=min_bin_indices,
                                            min_rev_bin_num=min_bin_indices)

                    ###############################################
                    ########### COllect movements ################
                    ###############################################
                    ### For all movements --> figure otu which ones to keep in the global distribution ###
                    global_comm_indices = {}

                    for mov in np.unique(move_sub[ix_com_global]):

                        ### Movement specific command indices 
                        ix_mc = np.nonzero(move_sub[ix_com_global] == mov)[0]
                        
                        ### Which global indices used for command/movement 
                        ix_mc_all = ix_com_global[ix_mc] 

                        ### If enough of these then proceed; 
                        if len(ix_mc) >= min_com_cond:    
                            global_comm_indices[mov] = ix_mc_all

                    if len(global_comm_indices.keys()) > 0: 

                        #ix_com_global = np.hstack((ix_com_global))
                        #global_mFR = np.mean(spks_sub[ix_com_global, :], axis=0)

                        com_mov = dict(vals=[], shuffs=[])

                        #### now that have all the relevant movements - proceed 
                        for mov in global_comm_indices.keys(): 

                            ### FR for neuron ### 
                            ix_mc_all = global_comm_indices[mov]

                            ### Update as of 12/17/22 --> keep movement indices in the pool: 
                            ix_mov = np.array([i for i, j in enumerate(ix_com_global) if j in ix_mc_all])

                            ### Match distribution --> including keeping movement indices in the pool 
                            ### Figure out which of the "ix_com" indices can be used for shuffling for this movement 
                            ix_ok, niter = plot_fr_diffs.distribution_match_global_mov(push_sub[np.ix_(ix_mc_all, [3, 5])], 
                                                                         push_sub[np.ix_(ix_com_global, [3, 5])], 
                                                                         keep_mov_indices_in_pool = True, 
                                                                         ix_mov = ix_mov)

                            #print('Mov %.1f, # Iters %d to match global'%(mov, niter))
                            
                            ### which indices we can use in global distribution for this shuffle ----> #### 
                            ix_com_global_ok = ix_com_global[ix_ok] 
                            
                            global_mFR_null = np.mean(spks_null_sub[ix_com_global_ok, :], axis=0)

                            #### Get true FR ###
                            mFR_null = np.mean(spks_null_sub[ix_mc_all, :], axis=0) # N x 1
                            pred_mFR_null = np.mean(pred_spks_null[ix_mc_all, :], axis=0) # N x 1
                            
                            pred_shuff_mFR_null = np.mean(pred_spks_null_shuffle[ix_mc_all, :, :], axis=0) # N x nshuffs
                            cond_mFR_null = np.mean(cond_spks_null[ix_mc_all, :], axis=0)

                            mFR = np.mean(spks_sub[ix_mc_all, :], axis=0)
                            pred_mFR = np.mean(pred_spks[ix_mc_all, :], axis=0)
                            pred_shuff_mFR = np.mean(pred_spks_shuffle[ix_mc_all, :, :], axis=0) 

                            ##### many NULL distances --> but dont seem to need thse 
                            #true_dist = np.linalg.norm(mFR_null - global_mFR_null)
                            #pred_dist = np.linalg.norm(pred_mFR_null - global_mFR_null)
                            #shuff_dist = np.linalg.norm(pred_shuff_mFR_null - global_mFR_null[:, np.newaxis], axis=0)
                            #assert(len(shuff_dist) == nshuffs)

                            ##### test sig fo command/movement #####
                            err_pred = np.linalg.norm(mFR_null - pred_mFR_null)
                            err_shuff = np.linalg.norm(mFR_null[:, np.newaxis] - pred_shuff_mFR_null, axis=0)
                            
                            ##### Test p-value of com/mov ####
                            n_err_lte = len(np.nonzero(err_shuff <= err_pred)[0])
                            pv = float(n_err_lte)/float(nshuffs)
                            if pv < 0.05: 
                                nCC_sig += 1
                            nCC += 1

                            #nnr = float(len(mFR))
                            day_stats_dict['CC'].append([err_pred, err_shuff, mag, ang, mov])
                            
                            ### Add values : 
                            add = True 
                            if only_sig_cc and pv >= 0.05: 
                                add = False

                            if add: 
                                ##### Add to list ####
                                com_mov['vals'].append(err_pred)
                                com_mov['shuffs'].append(err_shuff)

                                neur_com_mov['vals'].append(np.abs(mFR - pred_mFR))
                                neur_com_mov['shuffs'].append(np.abs(mFR[:, np.newaxis] - pred_shuff_mFR))

                                ### NULL distances 
                                mFR_for_r2.append([mFR_null-cond_mFR_null, pred_mFR_null-cond_mFR_null])
                                mFR_for_r2_shuff.append(pred_shuff_mFR_null-cond_mFR_null[:, np.newaxis])                        
                            
                        #### p-value of com/cond ####
                        if len(com_mov['vals']) > 0: 
                            val_pool_mv = np.mean(com_mov['vals'])
                            shuf_pool_mv = np.mean(np.vstack((com_mov['shuffs'])), axis=0)
                            
                            assert(len(shuf_pool_mv) == nshuffs)
                            
                            n_err_lte = len(np.nonzero(shuf_pool_mv <= val_pool_mv)[0])
                            pv = float(n_err_lte)/float(nshuffs)
                            if pv < 0.05:
                                nCom_sig += 1
                            nCom += 1
                        
                        else:
                            pass
                            print('PASSING ON COMMAND Mag %d Ang %d '%(mag, ang))
                            print("ANIMAL %s Day %d" %(animal, day_ix))


            ######## Neuron sig --> all full distances #########
            ### Ncom-cond x neurons 
            neur_com_mov_vals = np.vstack((neur_com_mov['vals']))
            assert(neur_com_mov_vals.shape[1] == nneur)

            ### neurons x shuffs x Ncom-cond 
            neur_com_mov_shuffs = np.dstack((neur_com_mov['shuffs']))
            assert(neur_com_mov_shuffs.shape[0] == nneur)
            assert(neur_com_mov_shuffs.shape[1] == nshuffs)
            assert(neur_com_mov_shuffs.shape[2] == neur_com_mov_vals.shape[0])
            
            if only_sig_cc: 
                assert(neur_com_mov_vals.shape[0] == nCC_sig)
            else:
                assert(neur_com_mov_vals.shape[0] == nCC)     

            ######### Assess sig. of individual neurons #######
            VL = []
            SHF = []

            ### For each neuron 
            for n in range(nneur): 

                vls = np.mean(neur_com_mov_vals[:, n])
                shf = np.mean(neur_com_mov_shuffs[n, :, :], axis=1)
                VL.append(vls)
                SHF.append(shf)

                assert(len(shf) == nshuffs)

                n_lte = len(np.nonzero(shf <= vls)[0])
                pv = float(n_lte)/float(nshuffs)
                if pv < 0.05: 
                    nNeur_sig += 1
                nNeur += 1

            #### Plot the dots 
            ax_fracCC.plot(ia, float(nCC_sig)/float(nCC), 'k.')
            bar_dict['fracCC'].append(float(nCC_sig)/float(nCC))
            count['fracCC'].append(float(nCC_sig)/float(nCC))

            ax_fracCom.plot(ia, float(nCom_sig)/float(nCom), 'k.')
            bar_dict['fracCom'].append(float(nCom_sig)/float(nCom))
            count['fracCom'].append(float(nCom_sig)/float(nCom))
    
            ax_fracN.plot(ia, float(nNeur_sig)/float(nNeur), 'k.')
            bar_dict['fracN'].append(float(nNeur_sig)/float(nNeur))
            count['fracN'].append(float(nNeur_sig) / float(nNeur))
            
            ##### do r2 -- condition-varitions for a given command ####
            mfr_true = np.vstack(([m[0] for m in mFR_for_r2])) # True distance (null)
            mfr_pred = np.vstack(([m[1] for m in mFR_for_r2])) # Predicted distance (null)
            r2_ = util_fcns.get_R2(mfr_true, mfr_pred)

            mFR_for_r2_shuff = np.dstack((mFR_for_r2_shuff)) # N x nshuff x nCC
            r2_sh = []
            for s in range(nshuffs):
                r2_sh.append(util_fcns.get_R2(mfr_true, mFR_for_r2_shuff[:, s, :].T))
            
            print('Monk R2 of cond-spec activity (null) %.3f, [%.3f, %.3f] mn | 95th perc' %(r2_, np.mean(r2_sh), np.percentile(r2_sh, 95)))
            stats_dict['R2'].append([r2_, r2_sh])

            ax_r2.plot(ia, r2_, 'k.', markersize=12)
            bar_dict['r2'].append(r2_)
            
            ### Line before white dot###
            ax_r2.plot([ia, ia+.4], [r2_, np.mean(r2_sh)], 'k-', linewidth=.25)

            ax_r2.plot(ia+.4, np.mean(r2_sh), '.', color='w', mec='k', mew=1., markersize=12)
            bar_dict['r2_shuff'].append(np.mean(r2_sh))

            

            ######## Pool over neurons and shuffles #########
            VL = np.mean(VL)
            SHF = np.mean(np.vstack((SHF)), axis=0)
            assert(nshuffs == len(SHF))
            pv = float(len(np.nonzero(SHF <= VL)[0]))/float(len(SHF))
            print('NNeurons Monk %s, Day %d, pv %.5f'%(animal, day_ix, pv))
            stats_dict['Neur'].append([VL, SHF])

            ####### Pool over command-condition  ######
            VL = np.mean(np.array([i[0] for i in day_stats_dict['CC']])) ## null 
            SHF = np.mean(np.vstack(([i[1] for i in day_stats_dict['CC']])), axis=0) #Null 
            assert(nshuffs == len(SHF))
            pv = float(len(np.nonzero(SHF <= VL)[0]))/float(len(SHF))
            print('CCs Monk %s, Day %d, pv %.5f'%(animal, day_ix, pv))
            stats_dict['CC'].append([VL, SHF])


        ####### POOLED STATS over days #########
        vl = np.mean(np.array([i[0] for i in stats_dict['Neur']]))
        shf = np.mean(np.vstack((np.array([i[1] for i in stats_dict['Neur']]))), axis=0)
        assert(len(shf) == nshuffs)
        pv = float(len(np.nonzero(shf <= vl)[0]))/float(len(shf))
        print('NNeurons Monk %s, POOLED, pv %.5f, mean = %.5f, [%.5f, %.5f]'%(animal, pv, vl, np.mean(shf), np.percentile(shf, 5)))
        

        vl = np.mean(np.array([i[0] for i in stats_dict['CC']]))
        shf = np.mean(np.vstack((np.array([i[1] for i in stats_dict['CC']]))), axis=0)
        assert(len(shf) == nshuffs)
        pv = float(len(np.nonzero(shf <= vl)[0]))/float(len(shf))
        print('CC Monk %s, POOLED, pv %.5f, mean = %.5f, [%.5f, %.5f]'%(animal, pv, vl/np.mean(shf), np.mean(shf)/np.mean(shf), 
            np.percentile(shf, 5)/np.mean(shf)))
        
        vl = np.mean(np.array([i[0] for i in stats_dict['R2']]))
        shf = np.mean(np.vstack((np.array([i[1] for i in stats_dict['R2']]))), axis=0)
        assert(len(shf) == nshuffs)
        pv = float(len(np.nonzero(shf > vl)[0]))/float(len(shf))
        print('R2 Monk %s, POOLED, pv %.5f, mean = %.5f, [%.5f, %.5f]'%(animal, pv, vl, np.mean(shf), np.percentile(shf, 95)))
        


        #### Plot bar plots 
        for _, (key, ax, wid, offs, alpha) in enumerate(zip(
            ['fracCC', 'fracCom', 'fracN', 'r2', 'r2_shuff'], 
            [ax_fracCC, ax_fracCom, ax_fracN, ax_r2, ax_r2], 
            [.8, .8, .8, .4, .4, ], 
            [0,   0,  0,  0, .4, ],
            [.2, .2, .2, .2, 0., ])):

            if alpha == 0.:
                ax.bar(ia+offs, np.mean(bar_dict[key]), width=wid, color='white',
                    linewidth=1., edgecolor='k')
            else:
                ax.bar(ia+offs, np.mean(bar_dict[key]), width=wid, alpha=alpha, color='k')
            ax.set_ylabel(ylabels[key], fontsize=8)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['G', 'J'])
            ax.hlines(0.05, -.5, 1.5, 'k', linewidth=.5, linestyle='dashed')
            if 'r2' in key: 
                ax.set_xlim([-.3, 1.7])
            else:
                ax.set_xlim([-1, 2])

            if 'r2' in key:
                pass
            else:
                ax.set_yticks([0., 0.2, .4, .6, .8, 1.0])
                ax.set_ylim([0., 1.05])
            
            #### Remove spines 
            for side in ['right', 'top']:
                spine_side = ax.spines[side]
                spine_side.set_visible(False)

        
    if save: 
        for _, (f, yl) in enumerate(zip([f_fracCC, f_fracCom, f_fracN, f_r2],
            ['pred_fracCC', 'pred_fracCom', 'pred_fracN', 'pred_r2'])):
            
            f.tight_layout()
            util_fcns.savefig(f, yl)

    return count

def perc_sig_neuron_comm_predictions(nshuffs = 10, min_bin_indices = 0, 
    model_set_number = 6, model_nm = 'hist_1pos_0psh_2spksm_1_spksp_0'):
    """
    Plot example single neuron and population deviations from global distribution
    along with predictions 

    Also plot VAF plots
    
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

    ### 2 plots --> single neuron and vector examples 
    fsu, axsu = plt.subplots(figsize=(2, 3))
    fvect, axvect = plt.subplots(figsize=(2, 3))

    fvafsu, axvafsu = plt.subplots(ncols = 2, figsize = (6, 3))
    fvafpop, axvafpop = plt.subplots(ncols = 2, figsize = (6, 3))

    #### VAF plots ######
    fvafsu_all, axvafsu_all = plt.subplots(figsize = (3, 3))
    fvafpop_all, axvafpop_all = plt.subplots(figsize = (3, 3))
    
    faxccsu_all, axccsu_all = plt.subplots(figsize = (3, 3))
    faxccpop_all, axccpop_all = plt.subplots(figsize = (3, 3))

    #### Pairwise comparison scatter plots ###
    faxpwsu, axpwsu = plt.subplots(ncols = 2, figsize = (6, 3))
    faxpwpop, axpwpop = plt.subplots(ncols = 2, figsize = (6, 3))

    ####### Save the VAFs ##########
    vaf_dict = dict(); 
    for t in ['su', 'pop']:
        for d in ['true', 'shuff']:
            vaf_dict[t, d] = []

    vaf_all_dict = dict()
    err_all_dict = dict()
    for animal in ['grom', 'jeev']:
        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):
            vaf_all_dict[animal, day_ix] = {}
            err_all_dict[animal, day_ix] = {}
            for t in ['su', 'pop']:
                vaf_all_dict[animal, day_ix][t, 'true'] = []; 
                err_all_dict[animal, day_ix][t, 'true'] = []; 
                err_all_dict[animal, day_ix][t, 'shuff'] = []; 
                for n in range(nshuffs):
                    vaf_all_dict[animal, day_ix][t, 'shuff', n] = [];
                    

    for ia, animal in enumerate(['grom', 'jeev']):

        perc_sig = []
        perc_sig_vect = []

        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):

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

            #### Significantly closer to real than shuffled is to real? 
            nneur_mov_com = 0; 
            nneur_mov_com_sig = 0; 

            nneur_mov_com_vect = 0; 
            nneur_mov_com_sig_vect = 0; 

            for mag in range(4):

                for ang in range(8): 

                    ### Return indices for the command ### 
                    ix_com = plot_fr_diffs.return_command_indices(bin_num_sub, rev_bin_num_sub, push_sub, 
                                            mag_boundaries, animal=animal, 
                                            day_ix=day_ix, mag=mag, ang=ang, min_bin_num=min_bin_indices,
                                            min_rev_bin_num=min_bin_indices)

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

                    if len(ix_com_global) > 0: 
                        ix_com_global = np.hstack((ix_com_global))
                        global_mFR = np.mean(spks_sub[ix_com_global, :], axis=0)

                        #### now that have all the relevant movements - proceed 
                        for mov in global_comm_indices.keys(): 

                            ### FR for neuron ### 
                            ix_mc_all = global_comm_indices[mov]

                            #### Get true FR ###
                            mFR = np.mean(spks_sub[ix_mc_all, :], axis=0) # N x 1
                            pred_mFR = np.mean(pred_spks[ix_mc_all, :], axis=0) # N x 1
                            shuff_mFR = np.mean(pred_spks_shuffle[ix_mc_all, :, :], axis=0) # N x nshuffs

                            true_dist = np.linalg.norm(mFR - global_mFR)
                            pred_dist = np.linalg.norm(pred_mFR - global_mFR)
                            shuff_dist = np.linalg.norm(shuff_mFR - global_mFR[:, np.newaxis], axis=0)
                            assert(len(shuff_dist) == nshuffs)


                            ### Plot VAF ####
                            if animal == 'grom' and day_ix == 0: 
                                if mag == 0 and ang == 7: 
                                    special_color = get_color(mov)
                                else:
                                    special_color = None
                                vaf_dict = vaf_eg_plot(axvafsu, axvafpop, mFR, pred_mFR, shuff_mFR, global_mFR, vaf_dict,
                                    special_color=special_color)

                            ### Get actual VAF ####
                            if len(global_comm_indices.keys()) > 1:
                                vaf_all_dict[animal, day_ix] = vaf_compute(mFR, pred_mFR, shuff_mFR, global_mFR, vaf_all_dict[animal, day_ix])


                            ##### Single neurons 
                            for i_n in range(nneur):
                                ix = np.nonzero(shuff_mFR[i_n, :] - mFR[i_n]  <= pred_mFR[i_n] - mFR[i_n])[0]
                                if float(len(ix))/float(nshuffs) < 0.05: 
                                    nneur_mov_com_sig += 1
                                nneur_mov_com += 1

                                ### Get estimated error: 
                                err_all_dict[animal, day_ix]['su', 'true'].append( np.abs(mFR[i_n] - pred_mFR[i_n]))
                                err_all_dict[animal, day_ix]['su', 'shuff'].append(np.abs(mFR[i_n] - shuff_mFR[i_n, :]))
                            
                            err_all_dict[animal, day_ix]['pop', 'true'].append(np.linalg.norm(mFR - pred_mFR))
                            err_all_dict[animal, day_ix]['pop', 'shuff'].append(np.linalg.norm(mFR[:, np.newaxis] - shuff_mFR, axis=0))

                            ##### Vectors; 
                            dN = np.linalg.norm(mFR - pred_mFR); 
                            dN_shuff = np.linalg.norm(shuff_mFR - mFR[:, np.newaxis], axis=0);
                            ix = np.nonzero(dN_shuff <= dN)[0]
                            if float(len(ix))/float(nshuffs) < 0.05:
                                nneur_mov_com_sig_vect += 1
                            nneur_mov_com_vect += 1

            frac1 = float(nneur_mov_com_sig) / float(nneur_mov_com)
            frac2 = float(nneur_mov_com_sig_vect) / float(nneur_mov_com_vect)

            axsu.plot(ia, frac1, 'k.')
            axvect.plot(ia, frac2, 'k.')

            perc_sig.append(frac1)
            perc_sig_vect.append(frac2)

            ### For this day, plot the vaf ###
            plot_vaf_dist(axvafsu_all, axvafpop_all, axccsu_all, axccpop_all, animal, day_ix, vaf_all_dict, nshuffs)

        axsu.bar(ia, np.mean(perc_sig), width=0.8, color='k', alpha=.2)
        axvect.bar(ia, np.mean(perc_sig_vect), width=0.8, color='k', alpha=.2)
    
    fsu.tight_layout()
    fvect.tight_layout()

    util_fcns.savefig(fsu, 'fig4_perc_NCM_sig')
    util_fcns.savefig(fvect, 'fig4_perc_CM_sig')

    ### Compute VAF ###
    ax_ = [axvafsu, axvafpop]
    for ti, t in enumerate(['su', 'pop']):
        for i_d, d in enumerate(['true', 'shuff']):
            dat = np.vstack((vaf_dict[t, d]))

            ### How much variance of the total thing is accounted for? 
            r2 = util_fcns.get_VAF_no_mean(dat[:, 0], dat[:, 1])
            #### Get linear regression, no intercept ####
            # model = sm.OLS(dat[:, 1], dat[:, 0])
            # results = model.fit()
            # slp = float(results.params)
            # intc = 0.
            # rv2 = float(results.rsquared)

            slp,intc,rv2,_,_ = scipy.stats.linregress(dat[:, 0], dat[:, 1])
            x_ = np.linspace(np.min([0., dat[:, 0].min()]), dat[:, 0].max(), 100)
            y_ = slp*x_ + intc 
            ax_[ti][i_d].plot(x_, y_, 'k--')
    
            print('Neural %s, Dat %s, Vaf %.5f, Corr Coeff %.5f' %(t, d, r2, rv2))
            #ax_[ti][i_d].set_title('Vaf %.3f, (rv)^2 %.3f, avg err %.3f' %(r2, rv2, avg_error), fontsize=12)
            ax_[ti][i_d].set_title('Vaf %.3f, (rv)^2 %.3f' %(r2, rv2), fontsize=12)

    # ### Su Axes
    for axi in axvafsu:
        axi.set_xlim([-15, 15])
        axi.set_ylim([-15, 15])
        axi.plot([-15, 15], [-15, 15], 'k-')

    for axi in axvafpop:
        axi.plot([0, 1.], [0, 1.], 'k-')
        axi.set_xlim([0, 1.])
        axi.set_ylim([0, 1.])
    
    axvafsu_all.set_xlim([-1, 14])
    axvafsu_all.set_ylabel('VAF, Neuron')

    axvafpop_all.set_xlim([-1, 14])
    axvafpop_all.set_ylabel('VAF, Pop. Dist')
    
    axccpop_all.set_xlim([-1, 14])
    axccpop_all.set_ylabel('Corr. Coeff. Pop. Dist')

    axccsu_all.set_xlim([-1, 14])
    axccsu_all.set_ylabel('Corr. Coeff. Neuron')

    nms = ['vaf_su_all', 'vaf_pop_all', 'cc_su_all', 'cc_pop_all', 'n36_eg', 'pop_eg']
    for i_f, f in enumerate([fvafsu_all, fvafpop_all, faxccsu_all, faxccpop_all, fvafsu, fvafpop]):
        f.tight_layout()
        util_fcns.savefig(f, 'fig4'+nms[i_f])
    
    return err_all_dict

def pw_comparison(nshuffs=1, min_bin_indices = 0, 
    model_set_number = 6, model_nm = 'hist_1pos_0psh_2spksm_1_spksp_0'):
    
    #### Pairwise comparison scatter plots ###
    faxpwsu, axpwsu = plt.subplots(ncols = 2, figsize = (6, 3))
    faxpwpop, axpwpop = plt.subplots(ncols = 2, figsize = (6, 3))

    fax_n, ax_n = plt.subplots(figsize=(3, 3))
    fax_p, ax_p = plt.subplots(figsize=(3, 3))

    fax_n_err, ax_n_err = plt.subplots(figsize=(3, 3))
    fax_p_err, ax_p_err = plt.subplots(figsize=(3, 3))

    fax_n_cc, ax_n_cc = plt.subplots(figsize=(2, 3))
    fax_p_cc, ax_p_cc = plt.subplots(figsize=(2, 3))


    for ia, animal in enumerate(['grom', 'jeev']):
        
        animal_data = dict(su=[], pop=[], su_r = [], pop_r = [])

        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):

            ####### Save the data ##########
            pw_dict = dict(); 
            pw_dict_all = dict(); 
            for t in ['su', 'pop']:
                for d in ['true', 'shuff', 'cond']:
                    pw_dict[t, d] = []
                    
                    ### Setup "all" dict ###
                    if d == 'shuff':
                        for i in range(nshuffs):
                            pw_dict_all[t, d, i] = []
                    else:
                        pw_dict_all[t, d] = []

            ################################
            ###### Extract real data #######
            ################################
            ### Get out null KG decoder 
            KG = util_fcns.get_decoder(animal, day_ix)
            assert(KG.shape[0] == 2)
            KG_null_low = scipy.linalg.null_space(KG) # N x (N-2)
            assert(KG_null_low.shape[1] + 2 == KG_null_low.shape[0])

            spks0, push0, tsk0, trg0, bin_num0, rev_bin_num0, move0, dat = util_fcns.get_data_from_shuff(animal, day_ix)
            spks0 = 10*spks0; 

            spks_est_cond_sub = 10*plot_generated_models.cond_act_on_psh(animal, day_ix)
            spks_est_cond_sub_null = np.dot(KG_null_low.T, spks_est_cond_sub.T).T
            
            #### Get subsampled
            tm0, _ = generate_models.get_temp_spks_ix(dat['Data'])

            ### Get subsampled 
            spks_sub = spks0[tm0, :]
            spks_sub_null = np.dot(KG_null_low.T, spks_sub.T).T
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
            pred_spks_null = np.dot(KG_null_low.T, pred_spks.T).T

            ### Make sure spks and sub_spks match -- using the same time indices ###
            assert(np.allclose(spks_sub, 10*model_dict[day_ix, 'spks']))
            assert(np.all(bin_num0[tm0] > 0))

            ###############################################
            ###### Get shuffled prediction of  spikes  ####
            ###############################################
            pred_spks_shuffle = plot_generated_models.get_shuffled_data_v2(animal, day_ix, model_nm, nshuffs = nshuffs, 
                testing_mode = False)
            pred_spks_shuffle = 10*pred_spks_shuffle; 
            
            pred_spks_shuffle_null = []
            for i in range(nshuffs): 
                pred_spks_shuffle_null.append(np.dot(KG_null_low.T, pred_spks_shuffle[:, :, i].T).T)
            pred_spks_shuffle_null = np.dstack((pred_spks_shuffle_null)) # T x Neurons -2 x shuffles? 
        
            ##############################################
            ########## SETUP the plots ###################
            ##############################################
            ### Get command bins 
            mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))
            command_bins = util_fcns.commands2bins([push_sub], mag_boundaries, animal, day_ix, 
                                               vel_ix=[3, 5])[0]
            for mag in range(4):

                for ang in range(8): 

                    ### Return indices for the command ### 
                    ix_com = plot_fr_diffs.return_command_indices(bin_num_sub, rev_bin_num_sub, push_sub, 
                                            mag_boundaries, animal=animal, 
                                            day_ix=day_ix, mag=mag, ang=ang, min_bin_num=min_bin_indices,
                                            min_rev_bin_num=min_bin_indices)

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

                    if len(ix_com_global) > 0: 
                        ix_com_global = np.hstack((ix_com_global))
                        global_mFR = np.mean(spks_sub[ix_com_global, :], axis=0)

                        ######## For this command, compute the PW diffs and use to plot the scatters #######
                        if animal == 'grom' and day_ix == 0:
                            pw_dict = pw_eg_scatter(spks_sub, push_sub, pred_spks, pred_spks_shuffle, ix_com_global, 
                                global_comm_indices, pw_dict, spks_null_sub = spks_sub_null, pred_spks_null = pred_spks_null,
                                pred_spks_null_shuffle = pred_spks_shuffle_null)

                        pw_dict_all = pw_calc(spks_sub, push_sub, pred_spks, pred_spks_shuffle, ix_com_global, 
                                global_comm_indices, pw_dict_all, pred_cond = spks_est_cond_sub, spks_null = spks_sub_null, 
                                pred_spks_null = pred_spks_null,
                                pred_spks_null_shuffle = pred_spks_shuffle_null, 
                                pred_cond_null = spks_est_cond_sub_null)

                    ######## Example plots ##############
                    if mag == 0 and ang == 7 and animal == 'grom' and day_ix == 0:
                        special_dots = pw_eg_plot(spks_sub, push_sub, pred_spks, pred_spks_shuffle, ix_com_global, 
                            global_comm_indices, pred_cond = spks_est_cond_sub, spks_null = spks_sub_null, 
                                pred_spks_null = pred_spks_null,
                                pred_spks_null_shuffle = pred_spks_shuffle_null, 
                                pred_cond_null = spks_est_cond_sub_null)
            
            if animal == 'grom' and day_ix == 0:             
                
                ######## Pairwise Plot Examples  ########
                ax_ = [axpwsu, axpwpop]
                
                for i_t, t in enumerate(['true']):#, 'shuff']):
                    for i_n, n in enumerate(['su', 'pop']): 

                        axi = ax_[i_n][i_t]
                        dat = np.vstack((pw_dict[n, t]))

                        axi.plot(dat[:, 0], dat[:, 1], 'k.', markersize=2.)
                        
                        print('Day %d, Animal %s, len(data) %s, %s = %d'%(day_ix, animal, t, n, len(dat)))

                        ### Stats on data ###
                        slp,intc,rv,_,_ = scipy.stats.linregress(dat[:, 0], dat[:, 1])
                        err = np.mean(np.abs(dat[:, 0] - dat[:, 1]))
                        #r2 = util_fcns.get_R2(dat[:, 0], dat[:, 1])

                        ### Title 
                        axi.set_title('%s, %s, cc: %.3f, err: %.3f'%(n, t, rv, err), fontsize=8)

                        ### Plot best fit line 
                        x_ = np.linspace(np.min(dat[:, 0]), np.max(dat[:, 0]), 10)
                        y_ = slp*x_ + intc
                        axi.plot(x_, y_, 'k--', linewidth=.5)

                        if n == 'su':
                            dt = special_dots['n']
                        elif n == 'pop':
                            dt = special_dots['p']

                        for i_d, d in enumerate(dt): 
                            if t == 'true':
                                axi.plot(d[0], d[1], '.', color=d[3], markersize=15)
                            elif t == 'shuff':
                                axi.plot(d[0], d[2], '.', color=d[3], markersize=15)

                for axi in axpwsu:
                    axi.set_xlim([-0, 25])
                    axi.set_ylim([-0, 10])
                
                for axi in axpwpop:
                    axi.set_xlim([0, 8.])
                    axi.set_ylim([0, 4.5])

                faxpwsu.tight_layout()
                faxpwpop.tight_layout()

                util_fcns.savefig(faxpwsu, 'neur_pw_scatter', png=False)
                util_fcns.savefig(faxpwpop, 'pop_pw_scatter')

            ###### Plot the CCs for single neurons adn for population ####
            ax_ = [ax_n, ax_p, ax_n_err, ax_p_err]
            for i_n, n in enumerate(['su', 'pop']): 
                mn = np.zeros((2,))
                mx = np.zeros((2,))
                
                axi = ax_[i_n]
                axi2 = ax_[i_n + 2]

                for i_t, t in enumerate(['true', 'cond']): ### removed 'shuff'
                    
                    if t in ['true', 'cond']:
                        dat = np.vstack((pw_dict_all[n, t]))
                        slp,intc,rv,pv,err = scipy.stats.linregress(dat[:, 0], dat[:, 1])
                        
                        if t == 'true':
                            print('Animal %s, Day %d, %s:%s, slp=%.3f, intc=%.3f, rv=%.3f, pv=%.5f' %(
                                animal, day_ix, t, n, slp, intc, rv, pv))
                            animal_data[n].append(dat)

                            if n == 'su':
                                ax_n_cc.plot(ia, rv, 'k.')
                                animal_data['su_r'].append(rv)
                            elif n == 'pop':
                                ax_p_cc.plot(ia, rv, 'k.')
                                animal_data['pop_r'].append(rv)

                        if i_t == 0: 
                            mn[0] = rv; 
                            mx[0] = rv; 
                        else:
                            mn[0] = np.min([mn[0], rv])
                            mx[0] = np.max([mx[0], rv])


                        if t == 'true':
                            axi.plot(ia*10+day_ix, rv, '.', color=analysis_config.blue_rgb, markersize=20)
                            
                            #### true / pred ####
                            mer = mnerr(dat[:, 0], dat[:, 1])
                            axi2.plot(ia*10 + day_ix, mer, '.', color=analysis_config.blue_rgb, markersize=20)

                            mn[1] = mer; 
                            mx[1] = mer; 
                            
                        elif t == 'cond':
                            axi.plot(ia*10+day_ix, rv, '^', color='gray', markersize=8)

                            #### true / pred ####
                            mer = mnerr(dat[:, 0], dat[:, 1])
                            axi2.plot(ia*10 + day_ix, mer, '^', color='gray', markersize=8)
                            mn[1] = np.min([mn[1], mer])
                            mx[1] = np.max([mx[1], mer])
                    
                    elif t == 'shuff': 
                        rv_shuff = []; er_shuff = []
                        for i_shuff in range(nshuffs):
                            dat = np.vstack((pw_dict_all[n, t, i_shuff]))
                            _,_,rv,_,_ = scipy.stats.linregress(dat[:, 0], dat[:, 1])
                            rv_shuff.append(rv)
                            er_shuff.append(mnerr(dat[:, 0], dat[:, 1]))

                        util_fcns.draw_plot(ia*10+day_ix-.1, rv_shuff, 'k', 
                            np.array([1., 1., 1., 0.]), axi)

                        util_fcns.draw_plot(ia*10+day_ix-.1, er_shuff, 'k', 
                            np.array([1., 1., 1., 0.]), axi2)
                        
                        mn[0] = np.min([mn[0], np.mean(rv_shuff)])
                        mx[0] = np.max([mx[0], np.mean(rv_shuff)])

                        mn[1] = np.min([mn[1], np.mean(er_shuff)])
                        mx[1] = np.max([mx[1], np.mean(er_shuff)])
                
                #### draw the line; 
                axi.plot([ia*10+day_ix, ia*10+day_ix], [mn[0], mx[0]], 'k-', linewidth=0.5)
                axi2.plot([ia*10+day_ix, ia*10+day_ix], [mn[1], mx[1]], 'k-', linewidth=0.5)
    

        #### animal data ###
        for n in ['su', 'pop']:
            dat = np.vstack(( animal_data[n] ))
            slp,intc,rv,pv,err = scipy.stats.linregress(dat[:, 0], dat[:, 1])
        
            print('Animal POOLED %s, %s, slp=%.3f, intc=%.3f, rv=%.3f, pv=%.5f, N = %d' %(
                animal, n, slp, intc, rv, pv, dat.shape[0]))

            ax_n_cc.bar(ia, np.mean(animal_data['su_r']), color='k', alpha=.2)
            ax_p_cc.bar(ia, np.mean(animal_data['pop_r']), color='k', alpha=.2)

    ax_ = [ax_n, ax_p, ax_n_err, ax_p_err]
    for axi in ax_:
        axi.set_xlim([-1, 14])
        axi.set_xticks([])

    ax_n.set_ylabel('Corr. Coeff., Neuron Dist.', fontsize=10)
    ax_p.set_ylabel('Corr. Coeff., Pop. Dist.', fontsize=10)
    ax_n_err.set_ylabel('Avg. Err. Neuron Dist.', fontsize=10)
    ax_p_err.set_ylabel('Avg. Err. Pop. Dist.', fontsize=10)

    for axi in [ax_n_cc, ax_p_cc]:
        axi.set_xticks([0, 1])
        axi.set_xticklabels(['G', 'J'])
    ax_n_cc.set_ylabel('Corr Coeff. (Neuron)')
    ax_p_cc.set_ylabel('Corr Coeff. (Population)')

    fax_n_cc.tight_layout()
    fax_p_cc.tight_layout()
    util_fcns.savefig(fax_n_cc, 'cc_bar_plt_n')
    util_fcns.savefig(fax_p_cc, 'cc_bar_plt_p')

    fax_n.tight_layout()
    fax_p.tight_layout()
    fax_n_err.tight_layout()
    fax_p_err.tight_layout()

    util_fcns.savefig(fax_n, 'corr_neuron_diffs')
    util_fcns.savefig(fax_p, 'corr_pop_dist')
    util_fcns.savefig(fax_n_err, 'err_neuron_diffs')
    util_fcns.savefig(fax_p_err, 'err_pop_diffs')
    
####### Fig 4 Eigenvalue plots ########
def get_data_EVs(keep_bin_spk_zsc = False, plot_null_dynamics = False, ridge_dict = None): 
    model_set_number = 6
    model_nm = 'hist_1pos_0psh_2spksm_1_spksp_0'

    ### Want the true data ####
    if plot_null_dynamics: 
        pass
    else: 
        ridge_dict = pickle.load(open(analysis_config.config['grom_pref'] + 'max_alphas_ridge_model_set%d.pkl' %model_set_number, 'rb')); 
    
    fnum, axnum = plt.subplots(figsize = (3, 4))
    ffrac, axfrac = plt.subplots(figsize = (3, 4))
    fhz, axhz = plt.subplots(figsize = (3, 4))


    for ia, animal in enumerate(['grom', 'jeev']):#, 'home']):
        num_eigs_td_gte_bin = []
        frac_eigs_td_gte_bin = []
        avg_freq_eigs_td_gte_bin = []

        if plot_null_dynamics: 
            pass
        else: 
            zstr = ''
            if animal == 'home': 
                if keep_bin_spk_zsc:
                    zstr = 'zsc'
                    ridge_dict = pickle.load(open(analysis_config.config['grom_pref'] + 'max_alphas_ridge_model_set%d_%s.pkl' %(model_set_number, zstr), 'rb')); 
                else:
                    ridge_dict = pickle.load(open(analysis_config.config['grom_pref'] + 'max_alphas_ridge_model_set%d.pkl' %(model_set_number), 'rb')); 
            else:
                ridge_dict = pickle.load(open(analysis_config.config['grom_pref'] + 'max_alphas_ridge_model_set%d.pkl' %model_set_number, 'rb')); 

        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):

            ### Get alpha ### 
            if plot_null_dynamics: 
                model = ridge_dict[animal, day_ix]
            
            else: 
                ### Get the saved shuffle data ####
                if animal == 'home':
                    data = pickle.load(open(analysis_config.config['shuff_fig_dir'] + '%s_%d_%s_shuff_ix.pkl'%(animal, day_ix, zstr), 'rb'))
                else:
                    data = pickle.load(open(analysis_config.config['shuff_fig_dir'] + '%s_%d_shuff_ix.pkl'%(animal, day_ix), 'rb'))

                ### Get X, Xtm1 ###
                Data = data['Data']

                ### Get the indices 
                tm0, tm1 = generate_models.get_temp_spks_ix(Data)

                ### Refit dynamics 
                alpha_spec = ridge_dict[animal][0][day_ix, model_nm]
                #print('%s, %d alpha %.1f' %(animal, day_ix, alpha_spec))
                ### Get Ridge; 
                model = Ridge(alpha=alpha_spec, fit_intercept=True)

                ### Fit the model: args = X, y
                model.fit(Data['spks'][tm1, :], Data['spks'][tm0, :])

            ### Dynamics matrix; 
            A = model.coef_
            hz, decay = get_ang_td(A)

            if animal == 'grom' and day_ix == 0: 
                f, ax = plt.subplots(figsize = (4, 4))
                ax.plot(decay, hz, 'k.') 
                ax.set_xlabel('Time Decay in seconds')
                ax.set_ylabel('Frequency (Hz)')
                ax.set_ylim([-.1,5.05])
                ax.vlines(.1, 0, 5.05, 'k', linestyle='dashed', linewidth=.5)
                f.tight_layout()
                if plot_null_dynamics:
                    util_fcns.savefig(f, '%s_%d_eigs_null'%(animal, day_ix))
                else: 
                    util_fcns.savefig(f, '%s_%d_eigs'%(animal, day_ix))

            #### Get stats; 
            ix_gte_bin = np.nonzero(decay >= 0.1)[0]
            print('Animal %s: Day %d, Num dim %d'%(animal, day_ix, len(ix_gte_bin)))
            print(hz[ix_gte_bin])
            print('----')
            num_eigs_td_gte_bin.append(float(len(ix_gte_bin)))
            frac_eigs_td_gte_bin.append(float(len(ix_gte_bin))/float(len(decay)))
            avg_freq_eigs_td_gte_bin.append(np.mean(hz[ix_gte_bin]))

            axnum.plot(ia+np.random.randn()*.1, float(len(ix_gte_bin)), 'k.')
            axfrac.plot(ia, float(len(ix_gte_bin))/float(len(decay)), 'k.')
            axhz.plot(ia, np.mean(hz[ix_gte_bin]), 'k.')

        axnum.bar(ia, np.mean(num_eigs_td_gte_bin), width=0.8, color='k', alpha=.2)
        axfrac.bar(ia, np.mean(frac_eigs_td_gte_bin), width=0.8, color='k', alpha=.2)
        axhz.bar(ia, np.mean(avg_freq_eigs_td_gte_bin), width=0.8, color='k', alpha=.2)
    
    for axi in [axnum, axfrac, axhz]:
        axi.set_xticks([0, 1])
        axi.set_xticklabels(['G', 'J'])
        axi.set_xlim([-1, 2])

    axnum.set_ylabel('# Eigs with td > 0.1 sec')
    axnum.set_ylim([0., 5.])
    axfrac.set_ylabel('Frac. Eigs with td > 0.1 sec')
    axhz.set_ylabel('Avg. Freq.')
    axhz.set_ylim([-.1, 1.])
    for f in [fnum, ffrac, fhz]:
        f.tight_layout()

    if plot_null_dynamics: 
        null = '_null'
    else:
        null = ''
    util_fcns.savefig(fnum, 'num_eigs_gte_bin%s'%null)
    util_fcns.savefig(ffrac, 'frac_eigs_gte_bin%s'%null)
    util_fcns.savefig(fhz, 'avg_freq_of_eigs_gte_bin%s'%null)

def get_ang_td(A, plt_evs_gte=.99, dt=0.1): 
    ev, evect = np.linalg.eig(A)

    ### Only look at eigenvalues explaining > 
    ix_sort = np.argsort(np.abs(ev))[::-1]
    ev_sort = ev[ix_sort]
    cumsum = np.cumsum(np.abs(ev_sort))/np.sum(np.abs(ev_sort))
    ix_keep = np.nonzero(cumsum>plt_evs_gte)[0]
    ev_sort_truc = ev_sort[:ix_keep[0]+1]

    ### get frequency; 
    angs = np.angle(ev_sort_truc) #np.array([ np.arctan2(np.imag(ev[i]), np.real(ev[i])) for i in range(len(ev))])
    hz = np.abs(angs)/(2*np.pi*dt)
    decay = -1./np.log(np.abs(ev_sort_truc))*dt # Time decay constant in ms
    return hz, decay

###### Fig 4 VAF Plots ##########
def vaf_eg_plot(axsu, axpop, mFR, pred_mFR, shuff_mFR, global_mFR, vaf_dict,
    neur_ix = 36, special_color = None):

    assert(shuff_mFR.shape[0] == len(global_mFR))
    assert(len(global_mFR) == len(pred_mFR) == len(mFR))
    
    if special_color is None:
        axsu[0].plot(mFR[neur_ix] - global_mFR[neur_ix], pred_mFR[neur_ix] - global_mFR[neur_ix], 'k.', markersize=4.)
        axsu[1].plot(mFR[neur_ix] - global_mFR[neur_ix], shuff_mFR[neur_ix, 0] - global_mFR[neur_ix], 'k.', markersize=4.)
    else:
        axsu[0].plot(mFR[neur_ix] - global_mFR[neur_ix], pred_mFR[neur_ix] - global_mFR[neur_ix], '.', color=special_color, markersize=10.)
        axsu[1].plot(mFR[neur_ix] - global_mFR[neur_ix], shuff_mFR[neur_ix, 0] - global_mFR[neur_ix], '.', color=special_color, markersize=10.)        
    
    ### Save the single units ####
    vaf_dict['su', 'true'].append(nby2arr(mFR[neur_ix] - global_mFR[neur_ix], pred_mFR[neur_ix] - global_mFR[neur_ix]))
    vaf_dict['su', 'shuff'].append(nby2arr(mFR[neur_ix] - global_mFR[neur_ix], shuff_mFR[neur_ix, 0] - global_mFR[neur_ix]))

    ### Population 
    nneur = float(len(mFR))

    distTrue, distPred = distGlobal_dirTrue(mFR, global_mFR, pred_mFR)
    distTrue2, distPredShuff = distGlobal_dirTrue(mFR, global_mFR, shuff_mFR[:, 0])
    assert(distTrue == distTrue2)

    axpop[0].plot(distTrue/nneur, distPred/nneur, 'k.', markersize=4.)
    axpop[1].plot(distTrue/nneur, distPredShuff/nneur, 'k.', markersize=4.)

    vaf_dict['pop', 'true'].append(nby2arr(distTrue/nneur, distPred/nneur))
    vaf_dict['pop', 'shuff'].append(nby2arr(distTrue/nneur, distPredShuff/nneur)) 
    return vaf_dict

def vaf_compute(mFR, pred_mFR, shuff_mFR, global_mFR, vaf_all_dict):
    
    nneur = float(len(mFR))
    nshuffs = shuff_mFR.shape[1]
    distTrue, distPred = distGlobal_dirTrue(mFR, global_mFR, pred_mFR)
    vaf_all_dict['pop', 'true'].append(nby2arr(distTrue/nneur, distPred/nneur))
    vaf_all_dict['su', 'true'].append(nby2arr(np.abs(mFR - global_mFR), np.abs(pred_mFR - global_mFR)))

    for i_n2 in range(nshuffs):
        vaf_all_dict['su', 'shuff', i_n2].append(nby2arr(np.abs(mFR - global_mFR), np.abs(shuff_mFR[:, i_n2] - global_mFR)))
        #vaf_all_dict['pop', 'shuff', i_n2].append(nby2arr(np.linalg.norm(mFR - global_mFR)/nneur, np.linalg.norm(shuff_mFR[:, i_n2] - global_mFR)/nneur))
        distTrue, distPredShuff = distGlobal_dirTrue(mFR, global_mFR, shuff_mFR[:, i_n2])
        vaf_all_dict['pop', 'shuff', i_n2].append(nby2arr(distTrue/nneur, distPredShuff/nneur))


    return vaf_all_dict

def plot_vaf_dist(axsus, axpop, axsus_cc, axpop_cc, animal, day_ix, vaf_all_dict, nshuffs):
    if animal == 'grom':
        x_pos = day_ix; 

    elif animal == 'jeev':
        x_pos = 10 + day_ix

    for i_n, (neur, axi) in enumerate(zip(['su', 'pop'], [[axsus, axsus_cc], [axpop, axpop_cc]])): 

        ### Get the shuffles ### 
        r2_shuff = []; cc_shuff = []
        for i_n2 in range(nshuffs):
            dat_shuff = np.vstack((vaf_all_dict[animal, day_ix][neur, 'shuff', i_n2]))
            vaf_shuff = util_fcns.get_VAF_no_mean(dat_shuff[:, 0], dat_shuff[:, 1])
            r2_shuff.append(vaf_shuff)

            _,_,rv,_,_ = scipy.stats.linregress(dat_shuff[:, 0], dat_shuff[:, 1])
            cc_shuff.append(rv)
            #avg_err = get_avg_error(dat_shuff[:, 0], dat_shuff[:, 1])
            #r2_shuff.append(avg_err)

        ### Box plot for shuffl 
        util_fcns.draw_plot(x_pos, r2_shuff, 'k', np.array([1., 1., 1., 0.]), axi[0])
        util_fcns.draw_plot(x_pos, cc_shuff, 'k', np.array([1., 1., 1., 0.]), axi[1])

        dat_true = np.vstack((vaf_all_dict[animal, day_ix][neur, 'true']))

        vaf_true = util_fcns.get_VAF_no_mean(dat_true[:, 0], dat_true[:, 1])
        axi[0].plot(x_pos, vaf_true, '.', color=analysis_config.blue_rgb, markersize=20)

        _,_,rv,_,_ = scipy.stats.linregress(dat_true[:, 0], dat_true[:, 1])
        axi[1].plot(x_pos, rv, '.', color=analysis_config.blue_rgb, markersize=20)
        #avg_error = get_avg_error(dat_true[:, 0], dat_true[:, 1])
        #axi.plot(x_pos, avg_error, '.', color=analysis_config.blue_rgb, markersize=20)

##### Fig 4 pairwise distance plots #######
def pw_eg_scatter(spks_sub, push_sub, pred_spks, pred_spks_shuffle, ix_com_global, 
    global_comm_indices, pw_dict, shuff_ix = 0, neuron_ix = 36, spks_null_sub = None, 
    pred_spks_null = None, pred_spks_null_shuffle = None): 

    movements = np.hstack((global_comm_indices.keys()))
    mov_ix = np.argsort(movements % 10)
    movements = movements[mov_ix]
    #### Pairwise examples ####
    for i_m, mov in enumerate(movements):
        for i_m2, mov2 in enumerate(movements[i_m+1:]):

            #### Match command distributions ####
            ix1 = global_comm_indices[mov]
            ix2 = global_comm_indices[mov2]

            #### Match indices by dropping from larger distribution: 
            # distribution_match_mov_pairwise does this on its own :) 
            ix1_1, ix2_2, niter = plot_fr_diffs.distribution_match_mov_pairwise(push_sub[np.ix_(ix1, [3, 5])], push_sub[np.ix_(ix2, [3, 5])],
                psig = 0.05)

            if len(ix1_1) < 15 or len(ix2_2) < 15:
                print('Skipping Movement PW %.1f, %.1f, Niter = %d' %(mov, mov2, niter)) 
            else:
                ix_mov = ix1[ix1_1]
                ix_mov2 = ix2[ix2_2]

                mFR = np.mean(spks_sub[ix_mov, :], axis=0)
                pred_mFR = np.mean(pred_spks[ix_mov, :], axis=0)
                shuff_mFR = np.mean(pred_spks_shuffle[ix_mov, :, shuff_ix], axis=0)       

                mFR2 = np.mean(spks_sub[ix_mov2, :], axis=0)
                pred_mFR2 = np.mean(pred_spks[ix_mov2, :], axis=0)
                shuff_mFR2 = np.mean(pred_spks_shuffle[ix_mov2, :, shuff_ix], axis=0)

                pw_dict['su', 'true'].append(nby2arr(np.abs(mFR[neuron_ix] - mFR2[neuron_ix]), np.abs(pred_mFR[neuron_ix] - pred_mFR2[neuron_ix])))
                #pw_dict['su', 'shuff'].append(nby2arr(np.abs(mFR[neuron_ix] - mFR2[neuron_ix]),np.abs(shuff_mFR[neuron_ix] - shuff_mFR2[neuron_ix])))

#                pw_dict['su', 'true'].append(nby2arr(np.abs(mFR - mFR2), np.abs(pred_mFR - pred_mFR2)))
#                pw_dict['su', 'shuff'].append(nby2arr(np.abs(mFR - mFR2), np.abs(shuff_mFR - shuff_mFR2)))
                

                ##### population distances now use null activity (12/20/22)
                mFR_n = np.mean(spks_null_sub[ix_mov, :], axis=0)
                mFR2_n = np.mean(spks_null_sub[ix_mov2, :], axis=0)

                pred_mFR_n = np.mean(pred_spks_null[ix_mov, :], axis=0)
                pred_mFR2_n = np.mean(pred_spks_null[ix_mov2, :], axis=0)
                
                shuff_mFR_n = np.mean(pred_spks_null_shuffle[ix_mov, :, shuff_ix], axis=0)
                shuff_mFR2_n = np.mean(pred_spks_null_shuffle[ix_mov2, :, shuff_ix], axis=0)

                pw_dict['pop', 'true'].append(nby2arr(nplanm(mFR_n, mFR2_n), nplanm(pred_mFR_n, pred_mFR2_n)))
                #pw_dict['pop', 'shuff'].append(nby2arr(nplanm(mFR_n, mFR2_n), nplanm(shuff_mFR_n, shuff_mFR2_n)))

    return pw_dict

def pw_calc(spks_sub, push_sub, pred_spks, pred_spks_shuffle, ix_com_global, 
    global_comm_indices, pw_dict, pred_cond = None, 
    spks_null=None, pred_spks_null = None, 
    pred_spks_null_shuffle=None, pred_cond_null = None): 

    movements = np.hstack((global_comm_indices.keys()))
    nshuffs = pred_spks_shuffle.shape[2]

    #### Pairwise examples ####
    for i_m, mov in enumerate(movements):
        for i_m2, mov2 in enumerate(movements[i_m+1:]):

            #### Match command distributions ####
            ix1 = global_comm_indices[mov]
            ix2 = global_comm_indices[mov2]

            #### Match indices 
            ix1_1, ix2_2, niter = plot_fr_diffs.distribution_match_mov_pairwise(push_sub[np.ix_(ix1, [3, 5])], 
                push_sub[np.ix_(ix2, [3, 5])],
                psig = 0.05)

            if len(ix1_1) < 15 or len(ix2_2) < 15:
                pass
            else:
                ix_mov = ix1[ix1_1]
                ix_mov2 = ix2[ix2_2]

                mFR = np.mean(spks_sub[ix_mov, :], axis=0)
                mFR_n = np.mean(spks_null[ix_mov, :], axis=0)

                pred_mFR = np.mean(pred_spks[ix_mov, :], axis=0)
                pred_mFR_n = np.mean(pred_spks_null[ix_mov, :], axis=0) 

                shuff_mFR = np.mean(pred_spks_shuffle[ix_mov, :, :], axis=0) 
                shuff_mFR_n = np.mean(pred_spks_null_shuffle[ix_mov, :, :], axis=0)

                mFR2 = np.mean(spks_sub[ix_mov2, :], axis=0)
                mFR2_n = np.mean(spks_null[ix_mov2, :], axis=0)

                pred_mFR2 = np.mean(pred_spks[ix_mov2, :], axis=0)
                pred_mFR2_n = np.mean(pred_spks_null[ix_mov2, :], axis=0)

                shuff_mFR2 = np.mean(pred_spks_shuffle[ix_mov2, :, :], axis=0)
                shuff_mFR2_n = np.mean(pred_spks_null_shuffle[ix_mov2, :, :], axis=0)

                pw_dict['su', 'true'].append(nby2arr(np.abs(mFR - mFR2), np.abs(pred_mFR - pred_mFR2)))
                
                pw_dict['pop', 'true'].append(nby2arr(nplanm(mFR_n, mFR2_n), nplanm(pred_mFR_n, pred_mFR2_n)))
                
                if pred_cond is not None:
                    cond_mFR = np.mean(pred_cond[ix_mov, :], axis=0) 
                    cond_mFR_n = np.mean(pred_cond_null[ix_mov, :], axis=0)

                    cond_mFR2 = np.mean(pred_cond[ix_mov2, :], axis=0)
                    cond_mFR2_n = np.mean(pred_cond_null[ix_mov2, :], axis=0)

                    pw_dict['su', 'cond'].append(nby2arr(np.abs(mFR-mFR2), np.abs(cond_mFR-cond_mFR2)))
                    pw_dict['pop', 'cond'].append(nby2arr(nplanm(mFR_n, mFR2_n), nplanm(cond_mFR_n, cond_mFR2_n)))
                        
                for shuffix in range(nshuffs):
                    pw_dict['su', 'shuff', shuffix].append(nby2arr(np.abs(mFR - mFR2), np.abs(shuff_mFR[:, shuffix] - shuff_mFR2[:, shuffix])))
                    pw_dict['pop', 'shuff', shuffix].append(nby2arr(nplanm(mFR_n, mFR2_n), nplanm(shuff_mFR_n[:, shuffix], shuff_mFR2_n[:, shuffix])))

    return pw_dict

def pw_eg_plot(spks_sub, push_sub, pred_spks, pred_spks_shuffle, ix_com_global, 
                            global_comm_indices, neuron_ix = 36, pred_cond = None, spks_null = None, 
                                pred_spks_null = None,
                                pred_spks_null_shuffle = None, 
                                pred_cond_null = None):
    '''
    Example plot for PW diffs 
    '''
    movements = np.hstack((global_comm_indices.keys()))

    X_lab = []

    Y_val = []
    Y_pred = []
    Y_cond = []
    Y_shuff = [] 
    nneur = spks_sub.shape[1]

    special_dots = dict(n=[], p=[]) ### true vs. predicted dots, colors 

    for i_m, mov in enumerate(movements):
        for i_m2, mov2 in enumerate(movements[i_m+1:]):
            
            #### Match command distributions ####
            ix1 = global_comm_indices[mov]
            ix2 = global_comm_indices[mov2]

            #### Match indices 
            ix1_1, ix2_2, niter = plot_fr_diffs.distribution_match_mov_pairwise(push_sub[np.ix_(ix1, [3, 5])], 
                push_sub[np.ix_(ix2, [3, 5])],
                psig = .05)

            nneur = spks_sub.shape[1]

            if len(ix1_1) < 15 or len(ix2_2) < 15:
                print('Skipping Movement PW %.1f, %.1f, Niter = %d' %(mov, mov2, niter)) 
            else:
                ix_mov = ix1[ix1_1]
                ix_mov2 = ix2[ix2_2]

                mFR = np.mean(spks_sub[ix_mov, :], axis=0)
                mFR_n = np.mean(spks_null[ix_mov, :], axis=0)

                pred_mFR = np.mean(pred_spks[ix_mov, :], axis=0)
                pred_mFR_n = np.mean(pred_spks_null[ix_mov, :], axis=0)
                
                cond_mFR = np.mean(pred_cond[ix_mov, :], axis=0)
                cond_mFR_n = np.mean(pred_cond_null[ix_mov, :], axis=0)

                shuff_mFR = np.mean(pred_spks_shuffle[ix_mov, :, :], axis=0)       
                shuff_mFR_n = np.mean(pred_spks_null_shuffle[ix_mov, :, :], axis=0)

                mFR2 = np.mean(spks_sub[ix_mov2, :], axis=0)
                mFR2_n = np.mean(spks_null[ix_mov2, :], axis=0)

                pred_mFR2 = np.mean(pred_spks[ix_mov2, :], axis=0)
                pred_mFR2_n = np.mean(pred_spks_null[ix_mov2, :], axis=0)
                
                cond_mFR2 = np.mean(pred_cond[ix_mov2, :], axis=0)
                cond_mFR2_n = np.mean(pred_cond_null[ix_mov2, :], axis=0)

                shuff_mFR2 = np.mean(pred_spks_shuffle[ix_mov2, :, :], axis=0)
                shuff_mFR2_n = np.mean(pred_spks_null_shuffle[ix_mov2, :, :], axis=0)

                X_lab.append([mov, mov2])
                
                Y_val.append([np.abs(mFR[neuron_ix] - mFR2[neuron_ix]), nplanm(mFR_n, mFR2_n)])
                Y_cond.append([np.abs(cond_mFR[neuron_ix] - cond_mFR2[neuron_ix]), nplanm(cond_mFR_n, cond_mFR2_n)])
                Y_pred.append([np.abs(pred_mFR[neuron_ix] - pred_mFR2[neuron_ix]), nplanm(pred_mFR_n, pred_mFR2_n)])
                Y_shuff.append([np.abs(shuff_mFR[neuron_ix, :] - shuff_mFR2[neuron_ix, :]), 
                    np.linalg.norm(shuff_mFR_n - shuff_mFR2_n, axis=0)/np.sqrt(nneur)])

    ################ Sort movement comparisons ################
    Y_val = np.vstack((Y_val)) # number of pw comparisons x [neuron diff and pop diff]
    ix_sort_neur = np.argsort(Y_val[:, 0])[::-1] ## High to low 
    ix_sort_pop = np.argsort(Y_val[:, 1])[::-1] ## High to low 

    ################ Pairwise examples ########################
    fn_eg, axn_eg = plt.subplots(figsize =(5, 7))
    fpop_eg, axpop_eg = plt.subplots(figsize =(5, 7))
    
    axn_eg.tick_params(axis='y', labelcolor='darkblue')
    axpop_eg.tick_params(axis='y', labelcolor='darkblue')

    for iv, (vl_n, vl_p) in enumerate(zip(ix_sort_neur, ix_sort_pop)):
        axn_eg.plot(iv, Y_val[vl_n, 0], '.', color='darkblue')
        
        if iv == 0:
            axn_eg2 = axn_eg.twinx()
            axn_eg2.tick_params(axis='y', labelcolor=analysis_config.blue_rgb)

        axn_eg2.plot(iv, Y_pred[vl_n][0], '*', color=analysis_config.blue_rgb)
        axn_eg2.plot(iv, Y_cond[vl_n][0], '^', color='gray')
        
        axn_eg.plot(iv, -1, '.', color=get_color(X_lab[vl_n][0]), markersize=15)
        axn_eg.plot(iv, -1.5, '.', color=get_color(X_lab[vl_n][1]), markersize=15)

        axpop_eg.plot(iv, Y_val[vl_p, 1], '.', color='darkblue')
        if iv == 0:
            axpop_eg2 = axpop_eg.twinx()
            axpop_eg2.tick_params(axis='y', labelcolor=analysis_config.blue_rgb)
        
        axpop_eg2.plot(iv, Y_pred[vl_p][1], '*', color=analysis_config.blue_rgb)
        #util_fcns.draw_plot(iv, Y_shuff[vl_p][1], 'k', np.array([1., 1., 1., 0]), axpop_eg2)
        axpop_eg2.plot(iv, Y_cond[vl_p][1], '^', color='gray')
        axpop_eg.plot(iv, -.03, '.', color=get_color(X_lab[vl_p][0]), markersize=15)
        axpop_eg.plot(iv, -.15, '.', color=get_color(X_lab[vl_p][1]), markersize=15)

        #### just for double checking keys: 
        # if X_lab[vl_p][0] == 1. and  X_lab[vl_p][1] == 10.1: ###### DEEP PINK --> confirmed on 12/20/22
        #     axpop_eg.plot(iv, -.25, 'k*')
        # elif X_lab[vl_p][0] == 1. and  X_lab[vl_p][1] == 3.0: ######## LIMEGREEEN  --> confirmed on 12/20/22
        #     axpop_eg.plot(iv, -.25, 'g*')
        
        add = None

        ### Changed colors on 12/20/22 to match checks in lines 2213-2217
        if X_lab[vl_n][0] == 1. and X_lab[vl_n][1] == 3.0:
            add = 'limegreen'
        elif  X_lab[vl_n][0] == 1. and X_lab[vl_n][1] == 10.1:
            add = 'deeppink'
        if add is None:
            pass
        else:
            special_dots['n'].append([Y_val[vl_n, 0], Y_pred[vl_n][0], X_lab[vl_n], add])
            axn_eg.plot(iv, 0, 19., color=add)

        add = None
        if X_lab[vl_p][0] == 1. and X_lab[vl_p][1] == 3.0:
            add = 'limegreen'
        elif  X_lab[vl_p][0] == 1. and X_lab[vl_p][1] == 10.1:
            add = 'deeppink'
        if add is None:
            pass
        else:
            special_dots['p'].append([Y_val[vl_p, 1], Y_pred[vl_p][1], X_lab[vl_p], add])
            axpop_eg.plot(iv, 0, .88, color=add)
            
    axn_eg.set_xlim([-1, len(X_lab)])
    axpop_eg.set_xlim([-1, len(X_lab)])

    axn_eg.spines['top'].set_visible(False)
    axn_eg2.spines['top'].set_visible(False)
    axpop_eg.spines['top'].set_visible(False)
    axpop_eg2.spines['top'].set_visible(False)

    axn_eg.set_xticks([]) #
    axpop_eg.set_xticks([]) #

    axn_eg.set_ylim([-2.86, 20.])
    axn_eg2.set_ylim([-1, 7])

    axpop_eg2.set_ylim([-.15, 2.6])
    axpop_eg.set_ylim([-.4, 7.5])

    axn_eg.set_ylabel('Pairwise Neuron Diff. (Hz)', color='darkblue')
    axn_eg2.set_ylabel('Pred. Pairwise Neuron Diff. (Hz)', rotation=90, color=analysis_config.blue_rgb)
    axn_eg.set_xlabel('Movement Pairs', rotation=180)
    
    axpop_eg.set_ylabel('Pairwise Pop. Dist. (Hz)', color='darkblue')
    axpop_eg2.set_ylabel('Pred. Pairwise Pop. Dist. (Hz)', rotation=90, color=analysis_config.blue_rgb)
    axpop_eg.set_xlabel('Movement Pairs', rotation=180)

    for xia, axi in enumerate([axn_eg, axn_eg2, axpop_eg, axpop_eg2]):
        yl = axi.get_yticks()
        #if xia == 3:
        #    yl = np.array([0.0, .1, .2, .3])
        #    axi.set_yticks(yl)
        axi.set_yticklabels(np.round(yl, 1), rotation=90)
    
    fn_eg.tight_layout()
    fpop_eg.tight_layout()

    util_fcns.savefig(fn_eg, 'n%d_example_pw_diffs'%neuron_ix)
    util_fcns.savefig(fpop_eg, 'pop_example_pw_diffs')
    return special_dots

def nby2arr(x,y):
    if type(x) is np.ndarray:
        assert(type(y) is np.ndarray)
        tmp = np.vstack(([x, y])).T
        assert(tmp.shape[1] == 2)
        return tmp
    
    elif isinstance(x, float):
        assert(isinstance(y, float))
        return np.array([[x, y]])

def nplanm(x, y): 
    assert(len(x) == len(y))
    assert(len(x.shape) == len(y.shape))
    return np.linalg.norm(x - y)/np.sqrt(len(x))

def mnerr(true, pred):
    '''
    each dot is a population (or single neuron) distance; 
    each is a pairwise comparison
    '''
    assert(len(pred) == len(true))

    ### Average error in units of pred/true
    err = np.mean(np.abs(np.array(true)-np.array(pred)))

    return err 

#### Fig 4 neural behavior correlations #####
def neuraldiff_vs_behaviordiff_corr_pairwise_predictions(min_bin_indices=0, nshuffs = 1, 
    ncommands_psth = 5, min_commands = 15, min_move_per_command = 6): 

    ##### Overall pooled over days #####
    f, ax = plt.subplots(figsize = (3, 3))
    
    fperc_sig, axperc_sig = plt.subplots(figsize = (2, 3))
    fperc_sig_pred, axperc_sig_pred = plt.subplots(figsize = (2, 3))

    faxsig_dist, axsig_dist = plt.subplots(figsize = (2, 3))
    faxsig_dist_pred, axsig_dist_pred = plt.subplots(figsize = (2, 3))

    ### Open mag boundaries 
    mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))

    for ia, animal in enumerate(['grom', 'jeev']):
        
        perc_sig = []; ### fraction of sig correlations b/w behavior and neural data 
        prec_sig_pred = []; ### fraction of sig correlations b/w behavior and PREDICTED neural data 
        
        cc_sig = []; 
        cc_sig_pred = []; 

        animal_pooled_stats = []

        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):
            
            animal_day_pooled_stats = []

            ###### pooled correlation plot ######
            feg, axeg = plt.subplots(figsize=(3, 3))

            #### Extract decoder
            KG = util_fcns.get_decoder(animal, day_ix)
            assert(KG.shape[0] == 2)
            KG_null_low = scipy.linalg.null_space(KG) # N x (N-2)
            assert(KG_null_low.shape[1] + 2 == KG_null_low.shape[0])

            ################################
            ###### Extract real data #######
            ################################
            spks0, push0, tsk0, trg0, bin_num0, rev_bin_num0, move0, dat = util_fcns.get_data_from_shuff(animal, day_ix)
            spks0 = 10*spks0; 

            #### Get subsampled
            tm0, _ = generate_models.get_temp_spks_ix(dat['Data'])

            ### Get subsampled 
            spks_sub = spks0[tm0, :]
            spks_null = np.dot(KG_null_low.T, spks_sub.T).T

            push_sub = push0[tm0, :]
            move_sub = move0[tm0]
            bin_num_sub = bin_num0[tm0]
            rev_bin_num_sub = rev_bin_num0[tm0]

            ### Get number of neurons 
            nneur = spks_sub.shape[1]
            
            ###############################################
            ###### Get predicted spikes from the model ####
            ###############################################
            model_set_number = 6; 
            model_nm = 'hist_1pos_0psh_2spksm_1_spksp_0'
            model_fname = analysis_config.config[animal+'_pref']+'tuning_models_'+animal+'_model_set'+str(model_set_number)+'_.pkl'
            model_dict = pickle.load(open(model_fname, 'rb'))
            pred_spks = model_dict[day_ix, model_nm]
            pred_spks = 10*pred_spks; 
            pred_spks_null = np.dot(KG_null_low.T, pred_spks.T).T

            ### Make sure spks and sub_spks match -- using the same time indices ###
            assert(np.allclose(spks_sub, 10*model_dict[day_ix, 'spks']))
            assert(np.all(bin_num0[tm0] > 0))

            ###############################################
            ###### Get shuffled prediction of  spikes  ####
            ###############################################
            pred_spks_shuffle = plot_generated_models.get_shuffled_data_v2(animal, day_ix, model_nm, nshuffs = nshuffs, 
                testing_mode = False)
            pred_spks_shuffle = 10*pred_spks_shuffle; 

            ### get shuffle predictions in nullspace 
            pred_spks_null_shuffle = []
            for i in range(nshuffs): 
                pred_spks_null_shuffle.append(np.dot(KG_null_low.T, pred_spks_shuffle[:, :, i].T).T)
            pred_spks_null_shuffle = np.dstack((pred_spks_null_shuffle))
            assert(pred_spks_null_shuffle.shape[0] == pred_spks_shuffle.shape[0])
            assert(pred_spks_null_shuffle.shape[2] == pred_spks_shuffle.shape[2] == nshuffs)

            ##### Data pts to be used in the correlation 
            D_pool = []; 
            D_shuff = {}
            for i in range(nshuffs):
                D_shuff[i] = []
            D_grom0 = []

            Ncommands = 0
            Ncommands_sig = 0 # sig corr b/w beh and true neural distance
            Ncommands_sig_pred = 0 # sig corr b/w beh and pred neural distances 
            DistSigCommands = []
            DistSigCommands_pred = []

            ###############################################
            ### For each command: #########################
            ###############################################
            for mag in range(4):
                
                for ang in range(8): 
            
                    #### Common indices 
                    #### Get the indices for command ####
                    ix_com = plot_fr_diffs.return_command_indices(bin_num_sub, rev_bin_num_sub, push_sub, mag_boundaries, mag=mag, ang=ang,
                                           animal=animal, day_ix=day_ix, min_bin_num=min_bin_indices,
                                           min_rev_bin_num=min_bin_indices)

                    ##### Which movements go to the global? 
                    global_comm_indices = {}

                    if animal == 'grom' and day_ix == 0 and mag == 0 and ang == 7:
                        feg2,  axeg2 = plt.subplots(figsize=(4, 3))
                        feg22, axeg22 = plt.subplots(figsize=(4, 3))
                    D_comm = []
                    D_comm_shuff = {}
                    for i in range(nshuffs):
                        D_comm_shuff[i] = []


                    #### Go through the movements ####
                    for mov in np.unique(move_sub[ix_com]):
                        
                        ### Movement specific command indices 
                        ix_mc = np.nonzero(move_sub[ix_com] == mov)[0]
                        ix_mc_all = ix_com[ix_mc]
                        
                        ### If enough of these then proceed; 
                        if len(ix_mc) >= min_commands:    

                            global_comm_indices[mov] = ix_mc_all

                    if len(global_comm_indices.keys()) >= min_move_per_command:

                        relevant_movs = np.array(global_comm_indices.keys())

                        ##### Get the movements that count; 
                        for imov, mov in enumerate(relevant_movs): 
                            
                            ### MOV specific 
                            ### movements / command 
                            ix_mc_all = global_comm_indices[mov]
                            Nmov = len(ix_mc_all)

                            ### Get movement #2 
                            for imov2, mov2 in enumerate(relevant_movs[imov+1:]):

                                assert(mov != mov2)
                                ix_mc_all2 = global_comm_indices[mov2]
                                Nmov2 = len(ix_mc_all2)

                                #### match to the two distributions #######
                                ### Figure out which of the "ix_com" indices can be used for shuffling for this movement 
                                ix_ok1, ix_ok2, niter = plot_fr_diffs.distribution_match_mov_pairwise(push_sub[np.ix_(ix_mc_all, [3, 5])], 
                                                             push_sub[np.ix_(ix_mc_all2, [3, 5])])

                                if np.logical_and(len(ix_ok1) >= min_commands, len(ix_ok2) >= min_commands):

                                    #######################################
                                    ######### Indices check ###############
                                    #######################################
                                    assert(np.all(np.array([move_sub[i] == mov for i in ix_mc_all[ix_ok1]])))
                                    assert(np.all(np.array([move_sub[i] == mov2 for i in ix_mc_all2[ix_ok2]])))

                                    #### Proceed comparing these guys ##### 
                                    mov_mean_FR1 = np.mean(spks_sub[ix_mc_all[ix_ok1], :], axis=0)
                                    mov_mean_FR1_n = np.mean(spks_null[ix_mc_all[ix_ok1], :], axis=0)

                                    mov_mean_FR2 = np.mean(spks_sub[ix_mc_all2[ix_ok2], :], axis=0)
                                    mov_mean_FR2_n = np.mean(spks_null[ix_mc_all2[ix_ok2], :], axis=0)

                                    #### Proceed comparing these guys ##### 
                                    mov_pred_mean_FR1 = np.mean(pred_spks[ix_mc_all[ix_ok1], :], axis=0)
                                    mov_pred_mean_FR2 = np.mean(pred_spks[ix_mc_all2[ix_ok2], :], axis=0)
                                    mov_pred_mean_FR1_n = np.mean(pred_spks_null[ix_mc_all[ix_ok1], :], axis=0)
                                    mov_pred_mean_FR2_n = np.mean(pred_spks_null[ix_mc_all2[ix_ok2], :], axis=0)

                                    mov_shuf_mean_FR1 = np.mean(pred_spks_shuffle[ix_mc_all[ix_ok1], :, :], axis=0)
                                    mov_shuf_mean_FR2 = np.mean(pred_spks_shuffle[ix_mc_all2[ix_ok2], :, :], axis=0)
                                    mov_shuf_mean_FR1_n = np.mean(pred_spks_null_shuffle[ix_mc_all[ix_ok1], :, :], axis=0)
                                    mov_shuf_mean_FR2_n = np.mean(pred_spks_null_shuffle[ix_mc_all2[ix_ok2], :, :], axis=0)

                                    #### This stays the same -- behavioral PSTH
                                    mov_PSTH1 = plot_fr_diffs.get_PSTH(bin_num_sub, rev_bin_num_sub, push_sub, ix_mc_all[ix_ok1], num_bins=ncommands_psth,
                                        min_bin_set = 1)
                                    
                                    mov_PSTH2 = plot_fr_diffs.get_PSTH(bin_num_sub, rev_bin_num_sub, push_sub, ix_mc_all2[ix_ok2], num_bins=ncommands_psth,
                                        min_bin_set = 1)

                                    if mov_PSTH1 is not None and mov_PSTH2 is not None:
                                        assert(mov_PSTH1.shape[0] == 2*ncommands_psth + 1)
                                        assert(mov_PSTH1.shape[1] == 2)

                                        Nmov1 = len(ix_ok1)
                                        Nmov2 = len(ix_ok2)

                                        #### Matched dN and dB; 
                                        dN = np.linalg.norm(mov_mean_FR1_n -mov_mean_FR2_n)/np.sqrt(nneur)
                                        dN_pred = np.linalg.norm(mov_pred_mean_FR1_n -mov_pred_mean_FR2_n)/np.sqrt(nneur)
                                        
                                        dB = np.linalg.norm(mov_PSTH1 - mov_PSTH2)


                                        for ishuff in range(nshuffs):
                                            dN_pred_shuff = np.linalg.norm(mov_shuf_mean_FR1_n[:, ishuff] - mov_shuf_mean_FR2_n[:, ishuff])/np.sqrt(nneur)
                                            D_shuff[ishuff].append([dB, dN_pred_shuff])
                                            D_comm_shuff[ishuff].append([dB, dN_pred_shuff])

                                        axeg.plot(dB, dN, '.', color='darkblue')
                                        axeg.plot(dB, dN_pred, '.', color=analysis_config.blue_rgb)
                                        
                                        #### Pooled plot ###
                                        D_pool.append([dB, dN, dN_pred, mag*8 + ang + 100*(day_ix) ])
                                        D_comm.append([dB, dN, dN_pred, mov, len(ix_ok1), len(ix_mc_all), mov2, len(ix_ok2), len(ix_mc_all2)])

                                        if animal == 'grom' and day_ix == 0 and mag == 0 and ang == 7:
                                            axeg2.plot(dB, dN, '.', color='darkblue')
                                            axeg22.plot(dB, dN_pred, '.', color=analysis_config.blue_rgb)                                               
                                               
                                    else:
                                        print('Skipping %s, %d, command %d %d mov %1.f mov2 %.1f -- psth fail :(' %(animal, day_ix,
                                            mag, ang, mov, mov2))
                    
                        ####### Tabulate the commands #########
                        ######### For this command see if rv > shuffle ? #####
                        ##### Vertical stack command #####
                        D_comm = np.vstack((D_comm))
                        _,_,rv_true,pv_true,_ = scipy.stats.linregress(D_comm[:, 0], D_comm[:, 1])
                        _,_,rv_pred,pv_pred,_ = scipy.stats.linregress(D_comm[:, 0], D_comm[:, 2])

                        # rv_shuff = []
                        # for ishuff in range(nshuffs):
                        #     D_ = np.vstack((D_comm_shuff[ishuff]))
                        #     _,_,rv_,_,_ = scipy.stats.linregress(D_[:, 0], D_[:, 1])
                        #     rv_shuff.append(rv_)

                        # ix = np.nonzero(rv_shuff>= rv_pred)[0]
                        # pv = float(len(ix)) / float(nshuffs)
                        if pv_true < 0.05: 
                            Ncommands_sig += 1
                            DistSigCommands.append(rv_true)

                        if pv_pred < 0.05: 
                            Ncommands_sig_pred += 1
                            DistSigCommands_pred.append(rv_pred)
                        Ncommands += 1


                    if animal == 'grom' and day_ix == 0 and mag == 0 and ang == 7:
                        D_comm = np.vstack((D_comm))
                        slp1,int1,rv_true_special_eg,_,_ = scipy.stats.linregress(D_comm[:, 0], D_comm[:, 1])
                        slp2,int2,rv_pred_special_eg,_,_ = scipy.stats.linregress(D_comm[:, 0], D_comm[:, 2])
                        #axeg2.set_title('True %.3f, Pred %.3f'%(rv_true, rv_pred_special_eg), fontsize=8)
                        feg2.tight_layout()
                        #axeg2.spines['top'].set_visible(False)
                        #axeg22.spines['top'].set_visible(False)

                        x_ = np.linspace(np.min(D_comm[:, 0]), np.max(D_comm[:, 0]), 30)
                        y_true = slp1*x_ + int1; 
                        y_pred = slp2*x_ + int2; 
                        axeg2.plot(x_, y_true, '-', color='darkblue')
                        axeg22.plot(x_, y_pred, '-', color=analysis_config.blue_rgb)

                        print('Grom day 0, mag = 0, ang = 7')
                        print('True slp %.3f, intc %.3f, rv %.3f, N = %d' %(slp1, int1, rv_true_special_eg, D_comm.shape[0]))
                        print('Pred slp %.3f, intc %.3f, rv %.3f, N = %d' %(slp2, int2, rv_pred_special_eg, D_comm.shape[0]))
                        

                        ##### PASTED FROM confirmed check above #########
                        # if X_lab[vl_p][0] == 1. and  X_lab[vl_p][1] == 10.1: ###### DEEP PINK --> confirmed on 12/20/22
                        #     axpop_eg.plot(iv, -.25, 'k*')
                        # elif X_lab[vl_p][0] == 1. and  X_lab[vl_p][1] == 3.0: ######## LIMEGREEEN  --> confirmed on 12/20/22
                        #     axpop_eg.plot(iv, -.25, 'g*')

                        for i in range(D_comm.shape[0]):
                            if D_comm[i, 3] == 1. and D_comm[i, 6] == 3.:
                                axeg2.plot(D_comm[i, 0], D_comm[i, 1], '.', color='limegreen', markersize=15)
                                axeg22.plot(D_comm[i, 0], D_comm[i, 2], '.', color='limegreen', markersize=15)
                            
                            elif D_comm[i, 3] == 1. and D_comm[i, 6] == 10.1:
                                axeg2.plot(D_comm[i, 0], D_comm[i, 1], '.', color = 'deeppink', markersize=15)
                                axeg22.plot(D_comm[i, 0], D_comm[i, 2], '.', color='deeppink', markersize=15)

                        util_fcns.savefig(feg22, 'grom0_eg_w_pred_dbeh_vs_dneur_corr2')
                        util_fcns.savefig(feg2, 'grom0_eg_w_pred_dbeh_vs_dneur_corr')

            ######## Number of significant commands ####
            axperc_sig.plot(ia, float(Ncommands_sig)/float(Ncommands), 'k.')
            axperc_sig_pred.plot(ia, float(Ncommands_sig_pred)/float(Ncommands), 'k.')

            perc_sig.append(float(Ncommands_sig)/float(Ncommands))
            prec_sig_pred.append(float(Ncommands_sig_pred)/float(Ncommands))

            ##### 
            #util_fcns.draw_plot(10*ia + day_ix, DistSigCommands, 'k', np.array([1., 1., 1., 0]), axsig_dist)
            #util_fcns.draw_plot(10*ia + day_ix, DistSigCommands_pred, 'k', np.array([1., 1., 1., 0]), axsig_dist_pred)
            axsig_dist.plot(ia, np.mean(DistSigCommands), 'k.')
            axsig_dist_pred.plot(ia, np.mean(DistSigCommands_pred), 'k.')

            cc_sig.append(np.mean(DistSigCommands))
            cc_sig_pred.append(np.mean(DistSigCommands_pred))

            # if animal == 'grom' and day_ix == 0:
            #     axsig_dist_pred.plot(10*ia + day_ix, rv_pred_special_eg, '.', color=analysis_config.blue_rgb, markersize=10)
            #     axsig_dist.plot(10*ia + day_ix, rv_true_special_eg, '.', color='darkblue', markersize=10)
            
            ######### Commands for pooled plot  ########
            D_pool = np.vstack((D_pool))
            _,_,rv_eg,_,_ = scipy.stats.linregress(D_pool[:, 0], D_pool[:, 1])
            _,_,rv_pd,_,_ = scipy.stats.linregress(D_pool[:, 0], D_pool[:, 2])
            #axeg.set_title('True %.3f, Pred %.3f'%(rv_eg, rv_pd))
            #axsig_dist.plot(10*ia + day_ix, rv_pd, '.', color='gray', markersize = 10)

            ######### Overall plot of distribution vs. pred vs. true over days #####
            # D_pred = np.vstack((D_pred))
            # _,_,rv_true,_,_ = scipy.stats.linregress(D_pred[:, 0], D_pred[:, 1])
            # _,_,rv_pred,_,_ = scipy.stats.linregress(D_pred[:, 0], D_pred[:, 2])
            # ax.plot(ia*10 + day_ix, rv_true, '.', color='darkblue', markersize=10)
            # ax.plot(ia*10 + day_ix, rv_pred, '*', color=analysis_config.blue_rgb, markersize=10)

            # rv_shuff = []
            # for i_shuff in range(nshuffs):
            #     D = np.vstack((D_shuff[i_shuff]))
            #     _,_,rv,_,_ = scipy.stats.linregress(D[:, 0], D[:, 1])
            #     rv_shuff.append(rv)
            # util_fcns.draw_plot(ia*10 + day_ix, rv_shuff, 'k', np.array([1., 1., 1., 0.]), ax)
        
            ###### linear mixed effect models: 
            lme_dict = {}
            lme_dict['dbeh'] = D_pool[:, 0]
            lme_dict['dneur'] = D_pool[:, 1]
            lme_dict['dgrp'] = D_pool[:, 3] # session and command

            data = pd.DataFrame(lme_dict)
            md = smf.mixedlm("dneur ~ dbeh", data, groups=lme_dict['dgrp'])
            mdf = md.fit()
            print('Animal %s, Day %d TRUE population ' %(animal, day_ix))
            print(mdf.summary())

            lme_dict = {}
            lme_dict['dbeh'] = D_pool[:, 0]
            lme_dict['dneur_pred'] = D_pool[:, 2]
            lme_dict['dgrp'] = D_pool[:, 3] # session and command

            data = pd.DataFrame(lme_dict)
            md = smf.mixedlm("dneur_pred ~ dbeh", data, groups=lme_dict['dgrp'])
            mdf = md.fit()
            print(' ')
            print('Animal %s, Day %d PREDICTED w DYNAMICS ' %(animal, day_ix))
            print(mdf.summary())

            animal_pooled_stats.append(D_pool)

        ######### Bars for frac commands sig ####
        axperc_sig.bar(ia, np.mean(perc_sig), color='k', alpha=0.4)
        axperc_sig_pred.bar(ia, np.mean(prec_sig_pred), color='k', alpha=0.4)
        
        axsig_dist.bar(ia, np.mean(cc_sig), color='k', alpha = 0.4)
        axsig_dist_pred.bar(ia, np.mean(cc_sig_pred), color='k', alpha=0.4)

        ###### linear mixed effect models: 
        print(' ')
        print(' ')
        print(' ')

        print('Pooled TRUE % s' %(animal))
        D_pool = np.vstack((animal_pooled_stats))
        lme_dict = {}
        lme_dict['dbeh'] = D_pool[:, 0]
        lme_dict['dneur'] = D_pool[:, 1]
        lme_dict['dgrp'] = D_pool[:, 3]

        data = pd.DataFrame(lme_dict)
        md = smf.mixedlm("dneur ~ dbeh", data, groups=lme_dict['dgrp'])
        mdf = md.fit()
        print('Animal %s POOLED' %(animal))
        print(mdf.summary())
        print('Pv = %.5f')
        print(mdf.pvalues)


        print('Pooled PREDICTIONS % s' %(animal))
        D_pool = np.vstack((animal_pooled_stats))
        lme_dict = {}
        lme_dict['dbeh'] = D_pool[:, 0]
        lme_dict['dneur_pred'] = D_pool[:, 2]
        lme_dict['dgrp'] = D_pool[:, 3]

        data = pd.DataFrame(lme_dict)
        md = smf.mixedlm("dneur_pred ~ dbeh", data, groups=lme_dict['dgrp'])
        mdf = md.fit()
        print('Animal %s POOLED' %(animal))
        print(mdf.summary())
        print('Pv = %.5f')
        print(mdf.pvalues)

        animal_pooled_stats.append(D_pool)


    # for axi in [axsig_dist, axsig_dist_pred, ax]:
    #     axi.set_xlim([-1, 14])
    #     axi.set_xticks([])
    #     axi.set_ylabel('corr. coeffs of sig. comm-conds')

    ax.set_ylabel('Pooled Corr. Coeff.')
    f.tight_layout()
    feg.tight_layout()
    
    # faxsig_dist.tight_layout()
    # faxsig_dist_pred.tight_layout()

    util_fcns.savefig(f, 'pooled_cc_true_vs_pred_vs_shuff')
    util_fcns.savefig(feg, 'eg_session_cc_true_vs_pred_vs_shuff')
    
    # util_fcns.savefig(faxsig_dist, 'dist_sig_cc')
    # util_fcns.savefig(faxsig_dist_pred, 'dist_sig_cc_pred')

    for axi in [axperc_sig, axperc_sig_pred, axsig_dist, axsig_dist_pred]: 
        axi.set_xticks([0, 1])
        axi.set_xticklabels(['G', 'J'])
    
    for axi in [axperc_sig, axperc_sig_pred, axsig_dist, axsig_dist_pred]: 
        axi.set_yticks([0., .2, .4, .6, .8, 1.0])
        axi.set_yticklabels([0., .2, .4, .6, .8, 1.0])
        axi.set_ylim([0., 1.1])
        

    axperc_sig.set_ylabel('Frac. com -- sig corr')
    axperc_sig_pred.set_ylabel('Frac. com -- sig pred corr')
    axsig_dist.set_ylabel('CC of sig.')
    axsig_dist_pred.set_ylabel('CC of sig. pred')    

    fperc_sig.tight_layout()
    util_fcns.savefig(fperc_sig, 'frac_commands_sig')

    fperc_sig_pred.tight_layout()
    util_fcns.savefig(fperc_sig_pred, 'frac_commands_sig_pred')

    faxsig_dist.tight_layout()
    util_fcns.savefig(faxsig_dist, 'dist_sig_cc')

    faxsig_dist_pred.tight_layout()
    util_fcns.savefig(faxsig_dist_pred, 'dist_sig_cc_pred')

###### Def error functions for each #########
def distGlobal_dirTrue(mFR, global_mFR, pred_mFR):

    g2True = mFR - global_mFR
    distTrue = np.linalg.norm(g2True)
    if distTrue == 0:
        import pdb; pdb.set_trace()
    else:
        g2True_norm = g2True / distTrue

    projPred_g2True = np.dot(pred_mFR - global_mFR, g2True_norm)
    distProjPred = np.linalg.norm(projPred_g2True)

    return distTrue, distProjPred

def get_avg_error(true, pred):
    return np.mean(np.abs(true - pred))

def _test_distGlobal_dirTrue():
    glob = np.array([2, 3])
    mFR = np.array([2, 6])
    pred = np.array([4, 5])

    f, ax = plt.subplots()
    ax.plot(glob[0], glob[1], 'k.')
    ax.plot(mFR[0], mFR[1], 'b.')
    ax.plot(pred[0], pred[1], 'r.')

    g2True = mFR - glob
    ax.plot([0, g2True[0]], [0, g2True[1]], 'b-')

    distTrue = np.linalg.norm(g2True)
    g2True_norm = g2True / distTrue
    g2Pred = pred - glob
    ax.plot([0, g2Pred[0]], [0, g2Pred[1]], 'r-')
    projPred_g2True = np.dot(g2Pred, g2True_norm)
    ax.plot([0, projPred_g2True*g2True_norm[0]], 
            [0, projPred_g2True*g2True_norm[1]], 'r--')
    print('frac %.2f' %(np.linalg.norm(projPred_g2True)/distTrue))

