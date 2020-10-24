import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

import analysis_config
from online_analysis import util_fcns, generate_models, plot_generated_models, plot_fr_diffs

from sklearn.linear_model import Ridge

######## Figure 4 ###########
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

        mFR_vect[mov] = np.mean(FR_vect, axis=0)
        pred_mFR_vect[mov] = np.mean(pred_FR_vect, axis=0)
        shuff_mFR_vect[mov] = np.mean(shuff_vector, axis=0)

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

def get_color(mov, alpha=None):
    colnm = analysis_config.pref_colors[int(mov)%10]
    colrgba = np.array(mpl_colors.to_rgba(colnm))

    ### Set alpha according to task (tasks 0-7 are CO, tasks 10.0 -- 19.1 are OBS) 
    if alpha is None:
        if mov >= 10:
            colrgba[-1] = 0.5
        else:
            colrgba[-1] = 1.0
    else:
        colrgba[-1] = alpha
    
    #### col rgb ######
    colrgb = util_fcns.rgba2rgb(colrgba)
    return colrgb

def perc_sig_neuron_comm_predictions(nshuffs = 10, min_bin_indices = 0, 
    model_set_number = 6, model_nm = 'hist_1pos_0psh_2spksm_1_spksp_0'):
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

    ### 2 plots --> single neuron and vector 
    fsu, axsu = plt.subplots(figsize=(2, 3))
    fvect, axvect = plt.subplots(figsize=(2, 3))

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

                        #### now that have all the relevant movements - proceed 
                        for mov in global_comm_indices.keys(): 

                            ### FR for neuron ### 
                            ix_mc_all = global_comm_indices[mov]

                            #### Get true FR ###
                            mFR = np.mean(spks_sub[ix_mc_all, :], axis=0) # N x 1
                            pred_mFR = np.mean(pred_spks[ix_mc_all, :], axis=0) # N x 1
                            shuff_mFR = np.mean(pred_spks_shuffle[ix_mc_all, :, :], axis=0) # N x nshuffs

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

        axsu.bar(ia, np.mean(perc_sig), width=0.8, color='k', alpha=.2)
        axvect.bar(ia, np.mean(perc_sig_vect), width=0.8, color='k', alpha=.2)
    fsu.tight_layout()
    fvect.tight_layout()

    util_fcns.savefig(fsu, 'fig4_perc_NCM_sig')
    util_fcns.savefig(fvect, 'fig4_perc_CM_sig')

####### Eigenvalue plots ########
def get_data_EVs(): 
    model_set_number = 6
    model_nm = 'hist_1pos_0psh_0spksm_1_spksp_0'
    ridge_dict = pickle.load(open(analysis_config.config['grom_pref'] + 'max_alphas_ridge_model_set%d_shuff.pkl' %model_set_number, 'rb')); 
    
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
