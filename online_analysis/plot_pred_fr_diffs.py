import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

import analysis_config
from online_analysis import util_fcns, generate_models, plot_generated_models, plot_fr_diffs
from util_fcns import get_color

from sklearn.linear_model import Ridge
import scipy.stats
import statsmodels.api as sm


######## Figure 4 examples and fraction neurons well predicted etc. ###########
def plot_example_neuron_comm_predictions(neuron_ix = 36, mag = 0, ang = 7, animal='grom', 
    day_ix = 0, nshuffs = 1000, min_bin_indices = 0, model_set_number = 6, 
    model_nm = 'hist_1pos_0psh_2spksm_1_spksp_0'):
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
    ix_com = plot_fr_diffs.return_command_indices(bin_num_sub, rev_bin_num_sub, push_sub, mag_boundaries, animal=animal, 
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

    ############ subsample to match across all movements ################
    global_comm_indices = plot_fr_diffs.distribution_match_mov_multi(global_comm_indices,
        push_sub[:, [3, 5]], psig=0.05)

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

        mFR_vect[mov] = np.mean(FR_vect, axis=0)
        pred_mFR_vect[mov] = np.mean(pred_FR_vect, axis=0)
        shuff_mFR_vect[mov] = np.mean(shuff_vector, axis=0)
        cond_mFR[mov] = np.mean(cond_spks[ix_mc_all, neuron_ix], axis=0)

        ### Figure out which of the "ix_com" indices can be used for shuffling for this movement 
        ix_ok, niter = plot_fr_diffs.distribution_match_global_mov(push_sub[np.ix_(ix_mc_all, [3, 5])], 
                                                     push_sub[np.ix_(ix_com_global, [3, 5])])
        
        ### which indices we can use in global distribution for this shuffle ----> #### 
        ix_com_global_ok = ix_com_global[ix_ok] 
        global_mean_vect = np.mean(spks_sub[ix_com_global_ok, :], axis=0)
        
        ### Get matching global distribution 
        mFR_vect_disggfr[mov] = np.linalg.norm(np.mean(FR_vect, axis=0) - global_mean_vect)/nneur
        mean_FR_vect = np.mean(FR_vect, axis=0)
        
        pred_vect_distmcfr[mov] = np.linalg.norm(np.mean(pred_FR_vect, axis=0) - mean_FR_vect)/nneur
        shuff_vect_distmcfr[mov] = np.linalg.norm(np.mean(shuff_vector, axis=0) - mean_FR_vect[:, np.newaxis], axis=0)/nneur
        
        pred_vect_distgfr[mov] = np.linalg.norm(np.mean(pred_FR_vect, axis=0) - global_mean_vect)/nneur
        shuff_vect_distgfr[mov] = np.linalg.norm(np.mean(shuff_vector, axis=0) - global_mean_vect[:, np.newaxis], axis=0)/nneur
        
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
        util_fcns.draw_plot(x, shuff_mFR[mov], 'k', np.array([1., 1., 1., 0]), axsu[1])
        axsu[0].plot(x, mFR[mov], '.', color=colrgb, markersize=20)
        axsu[1].plot(x, pred_mFR[mov], '*', color=colrgb, markersize=20)
        axsu[1].plot(x, cond_mFR[mov], '^', color='gray')
        # axsu[0].hlines(np.mean(spks_sub[ix_com_global, neuron_ix]), xlim[0], xlim[-1], color='gray',
        #      linewidth=1., linestyle='dashed')
        # axsu[1].hlines(np.mean(spks_sub[ix_com_global, neuron_ix]), xlim[0], xlim[-1], color='gray',
        #      linewidth=1., linestyle='dashed')     

        ### Population distance from movement-command FR
        util_fcns.draw_plot(x, shuff_vect_distmcfr[mov], 'k', np.array([1., 1., 1., 0]), axvect_distmcfr)
        axvect_distmcfr.plot(x, pred_vect_distmcfr[mov], '*', color=colrgb, markersize=20)

        ### Population distance from global FR
        util_fcns.draw_plot(x, shuff_vect_distgfr[mov], 'k', np.array([1., 1., 1., 0]), axvect_distgfr)
        axvect_distgfr.plot(x, pred_vect_distgfr[mov], '*', color=colrgb, markersize=20)
        axvect_distgfr.plot(x, mFR_vect_disggfr[mov], '.', color=colrgb, markersize=20)

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
    mfr_mc = []; 
    for m in mFR_vect.keys():
        mfr_mc.append(mFR_vect[m])
    mfr_mc = np.vstack((mfr_mc)) # conditions x neurons #

    ### Get the pc model 
    _, pc_model, _ = util_fcns.PCA(mfr_mc, 2, mean_subtract = True, skip_dim_assertion=True)

    #### Make the pca plot 
    fpca, axpca = plt.subplots(figsize=(6, 3), ncols = 2)

    #### For true data, plot the coordinates: 
    for m in mFR_vect.keys(): 

        ### Project data and plot ###
        trans_true = util_fcns.dat2PC(mFR_vect[m][np.newaxis, :], pc_model)
        axpca[0].plot(trans_true[0, 0], trans_true[0, 1], '.', color=get_color(m), markersize=20)

        ### PLot the predicted data : 
        trans_pred = util_fcns.dat2PC(pred_mFR_vect[m][np.newaxis, :], pc_model)
        axpca[1].plot(trans_pred[0, 0], trans_pred[0, 1], '*', color=get_color(m), markersize=20)
        #axpca[0].plot([trans_true[0, 0], trans_pred[0, 0]], [trans_true[0, 1], trans_pred[0, 1]], '-', 
        #    color=get_color(m), linewidth=1.)

        ### PLot the shuffled: Shuffles x 2 
        trans_shuff = util_fcns.dat2PC(shuff_mFR_vect[m].T, pc_model)
        e = confidence_ellipse(trans_shuff[:, 0], trans_shuff[:, 1], axpca[1], n_std=3.0,
            facecolor = get_color(m, alpha=.2))

    ### Add the global FR command ####
    global_mFR = np.mean(spks_sub[ix_com_global, :], axis=0)
    trans_global = util_fcns.dat2PC(global_mFR[np.newaxis, :], pc_model)
    axpca[0].plot(trans_global[0, 0], trans_global[0, 1], 'k.', markersize=10)
    #axpca[1].plot(trans_global[0, 0], trans_global[0, 1], 'k.', markersize=10)
    
    for axpcai in axpca:
        axpcai.set_xlabel('PC1')
        axpcai.set_ylabel('PC2')
    fpca.tight_layout()
    util_fcns.savefig(fpca,'fig4_eg_pca')

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
                      facecolor=facecolor, **kwargs)

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
    for animal in ['grom', 'jeev']:
        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):
            vaf_all_dict[animal, day_ix] = {}
            for t in ['su', 'pop']:
                vaf_all_dict[animal, day_ix][t, 'true'] = []; 
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

                            ##### Singl neurons 
                            for i_n in range(nneur):
                                ix = np.nonzero(shuff_mFR[i_n, :] - mFR[i_n]  <= pred_mFR[i_n] - mFR[i_n])[0]
                                if float(len(ix))/float(nshuffs) < 0.05: 
                                    nneur_mov_com_sig += 1
                                nneur_mov_com += 1

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
    
def pw_comparison(nshuffs=10, min_bin_indices = 0, 
    model_set_number = 6, model_nm = 'hist_1pos_0psh_2spksm_1_spksp_0'):
    
    #### Pairwise comparison scatter plots ###
    faxpwsu, axpwsu = plt.subplots(ncols = 2, figsize = (6, 3))
    faxpwpop, axpwpop = plt.subplots(ncols = 2, figsize = (6, 3))

    fax_n, ax_n = plt.subplots(figsize=(2, 3))
    fax_p, ax_p = plt.subplots(figsize=(2, 3))


    for ia, animal in enumerate(['grom', 'jeev']):
        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):

            ####### Save the data ##########
            pw_dict = dict(); 
            pw_dict_all = dict(); 
            for t in ['su', 'pop']:
                for d in ['true', 'shuff']:
                    pw_dict[t, d] = []
                    
                    ### Setup "all" dict ###
                    if d == 'true':
                        pw_dict_all[t, d] = []
                    else:
                        for i in range(nshuffs):
                            pw_dict_all[t, d, i] = []

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
                                global_comm_indices, pw_dict)

                        pw_dict_all = pw_calc(spks_sub, push_sub, pred_spks, pred_spks_shuffle, ix_com_global, 
                                global_comm_indices, pw_dict_all)

                    ######## Example plots ##############
                    if mag == 0 and ang == 7 and animal == 'grom' and day_ix == 0:
                        special_dots = pw_eg_plot(spks_sub, push_sub, pred_spks, pred_spks_shuffle, ix_com_global, 
                            global_comm_indices)
            
            if animal == 'grom' and day_ix == 0:             
                ######## Pairwise Plot Examples  ########
                ax_ = [axpwsu, axpwpop]
                dot_colors = [np.array([82, 184, 72])/255., np.array([237, 42, 145])/255.]
                
                for i_t, t in enumerate(['true', 'shuff']):
                    for i_n, n in enumerate(['su', 'pop']): 

                        axi = ax_[i_n][i_t]
                        dat = np.vstack((pw_dict[n, t]))
                        axi.plot(dat[:, 0], dat[:, 1], 'k.', markersize=2.)
                        
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
                                axi.plot(d[0], d[1], '.', color=dot_colors[i_d], markersize=15)
                            elif t == 'shuff':
                                axi.plot(d[0], d[2], '.', color=dot_colors[i_d], markersize=15)

                for axi in axpwsu:
                    axi.set_xlim([-30, 30])
                    axi.set_ylim([-20, 20])
                
                for axi in axpwpop:
                    axi.set_xlim([0, 1])
                    axi.set_ylim([0, .5])

                faxpwsu.tight_layout()
                faxpwpop.tight_layout()

                util_fcns.savefig(faxpwsu, 'neur_pw_scatter', png=True)
                util_fcns.savefig(faxpwpop, 'pop_pw_scatter')

            ###### Plot the CCs for single neurons adn for population ####
            ax_ = [ax_n, ax_p]
            for i_t, t in enumerate(['true', 'shuff']):
                for i_n, n in enumerate(['su', 'pop']): 
                    axi = ax_[i_n]

                    if t == 'true': 
                        dat = np.vstack((pw_dict_all[n, t]))
                        _,_,rv,_,_ = scipy.stats.linregress(dat[:, 0], dat[:, 1])
                        axi.plot(ia*10+day_ix, rv, '.', color=analysis_config.blue_rgb, markersize=10)

                    elif t == 'shuff': 
                        rv_shuff = []
                        for i_shuff in range(nshuffs):
                            dat = np.vstack((pw_dict_all[n, t, i_shuff]))
                            _,_,rv,_,_ = scipy.stats.linregress(dat[:, 0], dat[:, 1])
                            rv_shuff.append(rv)
                        util_fcns.draw_plot(ia*10+day_ix, rv_shuff, 'k', 
                            np.array([1., 1., 1., 0.]), axi)

    ax_ = [ax_n, ax_p]
    for axi in ax_:
        axi.set_xlim([-1, 14])
    ax_n.set_ylabel('Corr. Coeff., Neuron Dist.', fontsize=10)
    ax_p.set_ylabel('Corr. Coeff., Pop. Dist.', fontsize=10)

    fax_n.tight_layout()
    fax_p.tight_layout()

    util_fcns.savefig(fax_n, 'corr_neuron_diffs')
    util_fcns.savefig(fax_p, 'corr_pop_dist')

####### Fig 4 Eigenvalue plots ########
def get_data_EVs(): 
    model_set_number = 6
    model_nm = 'hist_1pos_0psh_2spksm_1_spksp_0'

    ### Want the true data ####
    ridge_dict = pickle.load(open(analysis_config.config['grom_pref'] + 'max_alphas_ridge_model_set%d.pkl' %model_set_number, 'rb')); 
    
    fnum, axnum = plt.subplots(figsize = (3, 4))
    ffrac, axfrac = plt.subplots(figsize = (3, 4))
    fhz, axhz = plt.subplots(figsize = (3, 4))


    for ia, animal in enumerate(['grom', 'jeev']):
        num_eigs_td_gte_bin = []
        frac_eigs_td_gte_bin = []
        avg_freq_eigs_td_gte_bin = []

        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):

            ### Get the saved shuffle data ####
            data = pickle.load(open(analysis_config.config['shuff_fig_dir'] + '%s_%d_shuff_ix.pkl'%(animal, day_ix), 'rb'))

            ### Get X, Xtm1 ###
            Data = data['Data']

            ### Get the indices 
            tm0, tm1 = generate_models.get_temp_spks_ix(Data)

            ### Get alpha ### 
            alpha_spec = ridge_dict[animal][0][day_ix, model_nm]
            print('%s, %d alpha %.1f' %(animal, day_ix, alpha_spec))
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
                util_fcns.savefig(f, '%s_%d_eigs'%(animal, day_ix))

            #### Get stats; 
            ix_gte_bin = np.nonzero(decay >= 0.1)[0]
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
    axfrac.set_ylabel('Frac. Eigs with td > 0.1 sec')
    axhz.set_ylabel('Avg. Freq.')
    axhz.set_ylim([-.1, 1.])
    for f in [fnum, ffrac, fhz]:
        f.tight_layout()

    util_fcns.savefig(fnum, 'num_eigs_gte_bin')
    util_fcns.savefig(ffrac, 'frac_eigs_gte_bin')
    util_fcns.savefig(fhz, 'avg_freq_of_eigs_gte_bin')

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
    vaf_all_dict['su', 'true'].append(nby2arr(mFR - global_mFR, pred_mFR - global_mFR))

    for i_n2 in range(nshuffs):
        vaf_all_dict['su', 'shuff', i_n2].append(nby2arr(mFR - global_mFR, shuff_mFR[:, i_n2] - global_mFR))
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
    global_comm_indices, pw_dict, shuff_ix = 0, neuron_ix = 36): 

    movements = np.hstack((global_comm_indices.keys()))
    mov_ix = np.argsort(movements % 10)
    movements = movements[mov_ix]
    #### Pairwise examples ####
    for i_m, mov in enumerate(movements):
        for i_m2, mov2 in enumerate(movements[i_m+1:]):

            #### Match command distributions ####
            ix1 = global_comm_indices[mov]
            ix2 = global_comm_indices[mov2]

            #### Match indices 
            ix1_1, ix2_2, niter = plot_fr_diffs.distribution_match_mov_pairwise(push_sub[np.ix_(ix1, [3, 5])], push_sub[np.ix_(ix2, [3, 5])],
                psig = 0.2)

            if len(ix1_1) == 0 or len(ix2_2) == 0:
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

                # pw_dict['su', 'true'].append(nby2arr(mFR[neuron_ix] - mFR2[neuron_ix], pred_mFR[neuron_ix] - pred_mFR2[neuron_ix]))
                # pw_dict['su', 'shuff'].append(nby2arr(mFR[neuron_ix] - mFR2[neuron_ix], shuff_mFR[neuron_ix] - shuff_mFR2[neuron_ix]))

                pw_dict['su', 'true'].append(nby2arr(mFR - mFR2, pred_mFR - pred_mFR2))
                pw_dict['su', 'shuff'].append(nby2arr(mFR - mFR2, shuff_mFR - shuff_mFR2))

                pw_dict['pop', 'true'].append(nby2arr(nplanm(mFR, mFR2), nplanm(pred_mFR, pred_mFR2)))
                pw_dict['pop', 'shuff'].append(nby2arr(nplanm(mFR, mFR2), nplanm(shuff_mFR, shuff_mFR2)))

    return pw_dict

def pw_calc(spks_sub, push_sub, pred_spks, pred_spks_shuffle, ix_com_global, 
    global_comm_indices, pw_dict): 

    movements = np.hstack((global_comm_indices.keys()))
    nshuffs = pred_spks_shuffle.shape[2]

    #### Pairwise examples ####
    for i_m, mov in enumerate(movements):
        for i_m2, mov2 in enumerate(movements[i_m+1:]):

            #### Match command distributions ####
            ix1 = global_comm_indices[mov]
            ix2 = global_comm_indices[mov2]

            #### Match indices 
            ix1_1, ix2_2, niter = plot_fr_diffs.distribution_match_mov_pairwise(push_sub[np.ix_(ix1, [3, 5])], push_sub[np.ix_(ix2, [3, 5])],
                psig = 0.05)

            if len(ix1_1) == 0 or len(ix2_2) == 0:
                pass
            else:
                ix_mov = ix1[ix1_1]
                ix_mov2 = ix2[ix2_2]

                mFR = np.mean(spks_sub[ix_mov, :], axis=0)
                pred_mFR = np.mean(pred_spks[ix_mov, :], axis=0)
                shuff_mFR = np.mean(pred_spks_shuffle[ix_mov, :, :], axis=0)       

                mFR2 = np.mean(spks_sub[ix_mov2, :], axis=0)
                pred_mFR2 = np.mean(pred_spks[ix_mov2, :], axis=0)
                shuff_mFR2 = np.mean(pred_spks_shuffle[ix_mov2, :, :], axis=0)

                pw_dict['su', 'true'].append(nby2arr(mFR - mFR2, pred_mFR - pred_mFR2))
                pw_dict['pop', 'true'].append(nby2arr(nplanm(mFR, mFR2), nplanm(pred_mFR, pred_mFR2)))
                
                for shuffix in range(nshuffs):
                    pw_dict['su', 'shuff', shuffix].append(nby2arr(mFR - mFR2, shuff_mFR[:, shuffix] - shuff_mFR2[:, shuffix]))
                    pw_dict['pop', 'shuff', shuffix].append(nby2arr(nplanm(mFR, mFR2), nplanm(shuff_mFR[:, shuffix], shuff_mFR2[:, shuffix])))

    return pw_dict

def pw_eg_plot(spks_sub, push_sub, pred_spks, pred_spks_shuffle, ix_com_global, 
                            global_comm_indices, neuron_ix = 36): 
    '''
    Exampel plot for PW diffs 
    '''
    movements = np.hstack((global_comm_indices.keys()))

    X_lab = []

    Y_val = []
    Y_pred = []
    Y_shuff = [] 
    nneur = spks_sub.shape[1]

    special_dots = dict(n=[], p=[]) ### true vs. predicted dots, colors 

    for i_m, mov in enumerate(movements):
        for i_m2, mov2 in enumerate(movements[i_m+1:]):
            
            #### Match command distributions ####
            ix1 = global_comm_indices[mov]
            ix2 = global_comm_indices[mov2]

            #### Match indices 
            ix1_1, ix2_2, niter = plot_fr_diffs.distribution_match_mov_pairwise(push_sub[np.ix_(ix1, [3, 5])], push_sub[np.ix_(ix2, [3, 5])],
                psig = .2)
            
            if len(ix1_1) == 0 or len(ix2_2) == 0:
                print('Skipping Movement PW %.1f, %.1f, Niter = %d' %(mov, mov2, niter)) 
            else:
                ix_mov = ix1[ix1_1]
                ix_mov2 = ix2[ix2_2]

                mFR = np.mean(spks_sub[ix_mov, :], axis=0)
                pred_mFR = np.mean(pred_spks[ix_mov, :], axis=0)
                shuff_mFR = np.mean(pred_spks_shuffle[ix_mov, :, :], axis=0)       

                mFR2 = np.mean(spks_sub[ix_mov2, :], axis=0)
                pred_mFR2 = np.mean(pred_spks[ix_mov2, :], axis=0)
                shuff_mFR2 = np.mean(pred_spks_shuffle[ix_mov2, :, :], axis=0)

                X_lab.append([mov, mov2])
                Y_val.append([mFR[neuron_ix] - mFR2[neuron_ix], nplanm(mFR, mFR2)])
                Y_pred.append([pred_mFR[neuron_ix] - pred_mFR2[neuron_ix], nplanm(pred_mFR, pred_mFR2)])
                Y_shuff.append([shuff_mFR[neuron_ix, :] - shuff_mFR2[neuron_ix, :], 
                    np.linalg.norm(shuff_mFR - shuff_mFR2, axis=0)/float(nneur)])

    ################ Sort movement comparisons ################
    Y_val = np.vstack((Y_val))
    ix_sort_neur = np.argsort(Y_val[:, 0])[::-1]
    ix_sort_pop = np.argsort(Y_val[:, 1])[::-1]

    ################ Pairwise examples ########################
    fn_eg, axn_eg = plt.subplots(figsize =(5, 5))
    fpop_eg, axpop_eg = plt.subplots(figsize =(5, 5))

    for iv, (vl_n, vl_p) in enumerate(zip(ix_sort_neur, ix_sort_pop)):
        axn_eg.plot(iv, Y_val[vl_n, 0], '.', color='darkblue')
        
        if iv == 0:
            axn_eg2 = axn_eg.twinx()
            axn_eg2.yaxis.label.set_color(np.array([.7, .7, .7]))

        axn_eg2.plot(iv, Y_pred[vl_n][0], '*', color=analysis_config.blue_rgb)
        util_fcns.draw_plot(iv, Y_shuff[vl_n][0], 'k', np.array([1., 1., 1., 0]), axn_eg2)
        axn_eg.plot(iv, -20, '.', color=get_color(X_lab[vl_n][0]), markersize=15)
        axn_eg.plot(iv, -21.5, '.', color=get_color(X_lab[vl_n][1]), markersize=15)

        axpop_eg.plot(iv, Y_val[vl_p, 1], '.', color='darkblue')
        if iv == 0:
            axpop_eg2 = axpop_eg.twinx()
            axpop_eg2.yaxis.label.set_color(np.array([.7, .7, .7]))
        
        axpop_eg2.plot(iv, Y_pred[vl_p][1], '*', color=analysis_config.blue_rgb)
        util_fcns.draw_plot(iv, Y_shuff[vl_p][1], 'k', np.array([1., 1., 1., 0]), axpop_eg2)
        axpop_eg.plot(iv, 0., '.', color=get_color(X_lab[vl_p][0]), markersize=15)
        axpop_eg.plot(iv, -.03, '.', color=get_color(X_lab[vl_p][1]), markersize=15)
        
        if iv == 0 or iv == len(ix_sort_neur) - 1:
            special_dots['n'].append([Y_val[vl_n, 0], Y_pred[vl_n][0], Y_shuff[vl_n][0][0]])
            special_dots['p'].append([Y_val[vl_p, 1], Y_pred[vl_p][1], Y_shuff[vl_p][1][0]])

    
    axn_eg.set_xlim([-1, len(X_lab)])
    axpop_eg.set_xlim([-1, len(X_lab)])

    axn_eg.spines['top'].set_visible(False)
    axn_eg2.spines['top'].set_visible(False)
    axpop_eg.spines['top'].set_visible(False)
    axpop_eg2.spines['top'].set_visible(False)

    axn_eg.set_xticks([]) #
    axpop_eg.set_xticks([]) #
    axpop_eg2.set_ylim([-.05, .4])
    axn_eg2.set_ylim([-8, 8])

    axn_eg.set_ylabel('Pairwise Neuron Diff. (Hz)')
    axn_eg.set_xlabel('Movement Pairs')
    axpop_eg.set_ylabel('Pairwise Pop. Activity Diff. (Hz)')
    axpop_eg.set_xlabel('Movement Pairs')

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
    return np.linalg.norm(x - y)/float(len(x))

#### Fig 4 neural behavior correlations #####
def neuraldiff_vs_behaviordiff_corr_pairwise_predictions(min_bin_indices=0, nshuffs = 1, 
    ncommands_psth = 5, min_commands = 15, min_move_per_command = 6): 

    ##### Overall pooled over days #####
    f, ax = plt.subplots(figsize = (3, 3))
    fperc_sig, axperc_sig = plt.subplots(figsize = (3, 3))
    faxsig_dist, axsig_dist = plt.subplots(figsize = (3, 3))

    ### Open mag boundaries 
    mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))

    for ia, animal in enumerate(['grom', 'jeev']):
        
        perc_sig = [] 

        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):
            
            ###### pooled correlation plot ######
            feg, axeg = plt.subplots(figsize=(3, 3))

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
            model_set_number = 6; 
            model_nm = 'hist_1pos_0psh_2spksm_1_spksp_0'
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

            ##### Data pts to be used in the correlation 
            D_pool = []; 
            D_pred = []; 
            D_shuff = {}
            for i in range(nshuffs):
                D_shuff[i] = []
            D_grom0 = []

            Ncommands = 0
            Ncommands_sig = 0
            DistSigCommands = []

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
                    ix_com_global = []
                    global_comm_indices = {}

                    if animal == 'grom' and day_ix == 0 and mag == 0 and ang == 7:
                        feg2, axeg2 = plt.subplots(figsize=(4, 3))
                        axeg22 = axeg2.twinx()
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
                            ix_com_global.append(ix_mc_all)

                    if len(ix_com_global) > 0:
                        ix_com_global = np.hstack((ix_com_global))
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
                                    mov_mean_FR2 = np.mean(spks_sub[ix_mc_all2[ix_ok2], :], axis=0)

                                    #### Proceed comparing these guys ##### 
                                    mov_pred_mean_FR1 = np.mean(pred_spks[ix_mc_all[ix_ok1], :], axis=0)
                                    mov_pred_mean_FR2 = np.mean(pred_spks[ix_mc_all2[ix_ok2], :], axis=0)

                                    mov_shuf_mean_FR1 = np.mean(pred_spks_shuffle[ix_mc_all[ix_ok1], :, :], axis=0)
                                    mov_shuf_mean_FR2 = np.mean(pred_spks_shuffle[ix_mc_all2[ix_ok2], :, :], axis=0)

                                    #### This stays the same 
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
                                        dN = np.linalg.norm(mov_mean_FR1 -mov_mean_FR2)/nneur
                                        dN_pred = np.linalg.norm(mov_pred_mean_FR1 -mov_pred_mean_FR2)/nneur
                                        
                                        dB = np.linalg.norm(mov_PSTH1 - mov_PSTH2)
                                        D_pred.append([dB, dN, dN_pred])

                                        for ishuff in range(nshuffs):
                                            dN_pred_shuff = np.linalg.norm(mov_shuf_mean_FR1[:, ishuff] - mov_shuf_mean_FR2[:, ishuff])/nneur
                                            D_shuff[ishuff].append([dB, dN_pred_shuff])
                                            D_comm_shuff[ishuff].append([dB, dN_pred_shuff])

                                        axeg.plot(dB, dN, '.', color='darkblue')
                                        axeg.plot(dB, dN_pred, '.', color=analysis_config.blue_rgb)
                                        
                                        #### Pooled plot ###
                                        D_pool.append([dB, dN, dN_pred])
                                        D_comm.append([dB, dN, dN_pred])
                                        if animal == 'grom' and day_ix == 0 and mag == 0 and ang == 7:
                                            axeg2.plot(dB, dN, '.', color='darkblue')
                                            axeg22.plot(dB, dN_pred, '.', color=analysis_config.blue_rgb)
                                               
                                    else:
                                        print('Skipping %s, %d, command %d %d mov %1.f mov2 %.1f -- psth fail :(' %(animal, day_ix,
                                            mag, ang, mov, mov2))
                    
                    if animal == 'grom' and day_ix == 0 and mag == 0 and ang == 7:
                        D_comm = np.vstack((D_comm))
                        _,_,rv_true,_,_ = scipy.stats.linregress(D_comm[:, 0], D_comm[:, 1])
                        _,_,rv_pred_special_eg,_,_ = scipy.stats.linregress(D_comm[:, 0], D_comm[:, 2])
                        axeg2.set_title('True %.3f, Pred %.3f'%(rv_true, rv_pred_special_eg), fontsize=8)
                        feg2.tight_layout()
                        axeg2.spines['top'].set_visible(False)
                        axeg22.spines['top'].set_visible(False)
                        util_fcns.savefig(feg2, 'grom0_eg_w_pred_dbeh_vs_dneur_corr')

                    ######### Tabulate the commands #########
                    ######### For this command see if rv > shuffle ? #####
                    if len(D_comm) >= min_move_per_command: 

                        ##### Vertical stack command #####
                        D_comm = np.vstack((D_comm))
                        _,_,rv_pred,_,_ = scipy.stats.linregress(D_comm[:, 0], D_comm[:, 2])

                        rv_shuff = []
                        for ishuff in range(nshuffs):
                            D_ = np.vstack((D_comm_shuff[ishuff]))
                            _,_,rv_,_,_ = scipy.stats.linregress(D_[:, 0], D_[:, 1])
                            rv_shuff.append(rv_)

                        ix = np.nonzero(rv_shuff>= rv_pred)[0]
                        pv = float(len(ix)) / float(nshuffs)
                        if pv < 0.05: 
                            Ncommands_sig += 1
                            DistSigCommands.append(rv_pred)
                        Ncommands += 1

            ######## Number of significant commands ####
            axperc_sig.plot(ia, float(Ncommands_sig)/float(Ncommands), 'k.')
            perc_sig.append(float(Ncommands_sig)/float(Ncommands))
            util_fcns.draw_plot(10*ia + day_ix, DistSigCommands, 'k', np.array([1., 1., 1., 0]), axsig_dist)

            if animal == 'grom' and day_ix == 0:
                axsig_dist.plot(10*ia + day_ix, rv_pred_special_eg, '.', color=analysis_config.blue_rgb, markersize=10)
            ######### Commands for pooled plot  ########
            D_pool = np.vstack((D_pool))
            _,_,rv_eg,_,_ = scipy.stats.linregress(D_pool[:, 0], D_pool[:, 1])
            _,_,rv_pd,_,_ = scipy.stats.linregress(D_pool[:, 0], D_pool[:, 2])
            axeg.set_title('True %.3f, Pred %.3f'%(rv_eg, rv_pd))
            axsig_dist.plot(10*ia + day_ix, rv_pd, '.', color='gray', markersize = 10)

            ######### Overall plot of distribution vs. pred vs. true over days #####
            D_pred = np.vstack((D_pred))
            _,_,rv_true,_,_ = scipy.stats.linregress(D_pred[:, 0], D_pred[:, 1])
            _,_,rv_pred,_,_ = scipy.stats.linregress(D_pred[:, 0], D_pred[:, 2])
            ax.plot(ia*10 + day_ix, rv_true, '.', color='darkblue', markersize=10)
            ax.plot(ia*10 + day_ix, rv_pred, '*', color=analysis_config.blue_rgb, markersize=10)

            rv_shuff = []
            for i_shuff in range(nshuffs):
                D = np.vstack((D_shuff[i_shuff]))
                _,_,rv,_,_ = scipy.stats.linregress(D[:, 0], D[:, 1])
                rv_shuff.append(rv)
            util_fcns.draw_plot(ia*10 + day_ix, rv_shuff, 'k', np.array([1., 1., 1., 0.]), ax)
        
        ######### Bars for frac commands sig ####
        axperc_sig.bar(ia, np.mean(perc_sig), color='k', width=.8, alpha=0.2)

    for axi in [axsig_dist, ax]:
        axi.set_xlim([-1, 14])
        axi.set_xticks([])

    ax.set_ylabel('Pooled Corr. Coeff.')
    f.tight_layout()
    feg.tight_layout()
    faxsig_dist.tight_layout()

    util_fcns.savefig(f, 'pooled_cc_true_vs_pred_vs_shuff')
    util_fcns.savefig(feg, 'eg_session_cc_true_vs_pred_vs_shuff')
    util_fcns.savefig(faxsig_dist, 'dist_cc_pred')

    axperc_sig.set_xticks([0, 1])
    axperc_sig.set_xticklabels(['G', 'J'])
    axperc_sig.set_ylabel('Frac. Commands\n with Sig. Correlation')
    fperc_sig.tight_layout()
    util_fcns.savefig(fperc_sig, 'frac_commands_sig_pred')

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

