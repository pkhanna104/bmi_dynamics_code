############## Methods to plot models generated in 'generate models' ###########

import seaborn
seaborn.set(font='Arial',context='talk',font_scale=1.25, style='white')

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pickle, copy, os, glob

import analysis_config, util_fcns
import generate_models, generate_models_list, generate_models_utils
from dynamics_sims import plot_flow_field_utils

from sklearn.linear_model import Ridge
import scipy
import scipy.io as sio
from collections import defaultdict
import time

fig_dir = analysis_config.config['fig_dir3']

#### so far which models are useful 
'''
model 1 (fig 4)
model 3 -- general extraction of stuff
'''

def plot_alpha_shuff_vs_og(model_set_number = 2):

    ridge_dict_shuff = pickle.load(open(analysis_config.config['grom_pref'] + 'max_alphas_ridge_model_set%d_shuff.pkl' %model_set_number, 'rb')); 
    ridge_dict = pickle.load(open(analysis_config.config['grom_pref'] + 'max_alphas_ridge_model_set%d.pkl' %model_set_number, 'rb')); 

    mvl, _, _, _ = generate_models_list.get_model_var_list(model_set_number)

    model_nms = [mvl[i][1] for i in range(len(mvl))]

    ndays = dict(grom=9, jeev=4)

    ### For each animal: 
    for i_a, animal in enumerate(['grom', 'jeev']):

        #### Plot model names; 
        f, ax = plt.subplots(ncols = len(model_nms), figsize = (12, 4))

        for i_m, mod in enumerate(model_nms):

            for i_d in range(ndays[animal]):
                ax[i_m].bar(i_d, ridge_dict[animal][0][i_d, mod], color='k', width=.3)
                ax[i_m].bar(i_d+.3, ridge_dict_shuff[animal][0][i_d, mod], color='b', width=.3)
            
            ax[i_m].set_title(mod, fontsize = 8)

        ax[0].set_ylabel(animal)
        f.tight_layout()
   
##### Fig 1  ####### Null vs. Potent; #####
def plot_real_mean_diffs_x_null_vs_potent(model_set_number = 3, min_obs = 15, cov = False):
    ### Take real task / target / command / neuron / day comparisons for each neuron in the BMI
    ### Plot within a bar 
    ### Plot only sig. different ones
    ### Plot cov. diffs (mat1 - mat2)
    

    ##### Radial boundaries based on percentages fit in feb 2019; 
    mag_boundaries = pickle.load(open(analysis_config.config['grom_pref']+'radial_boundaries_fit_based_on_perc_feb_2019.pkl'))

    ### Now bar plot that: 
    fsumm, axsumm = plt.subplots(figsize=(4, 4))
    NP = dict()

    for ia, animal in enumerate(['grom','jeev']):
        NP[animal, 'pot'] = []
        NP[animal, 'nul'] = []
        NP[animal, 'tot'] = []

        ### Open this tuning model file (PKL)
        model_dict = pickle.load(open(analysis_config.config[animal+'_pref']+'tuning_models_'+animal+'_model_set%d.pkl' %model_set_number, 'rb'))
        ndays = analysis_config.data_params[animal+'_ndays']

        DX_pot = []; 
        DX_nul = []; 
        days = []; grp = []; mets = [];
        mnz = np.zeros((ndays, 2)) # days x [pot / nul]

        f, ax = plt.subplots(figsize=(ndays/2., 4))

        for i_d in range(ndays):

            diffs_null = []; 
            diffs_pot = []; 
            spks = model_dict[i_d, 'spks']

            ### Get the decoder ###
            ### Get kalman gain etc. 
            if animal == 'grom':
                KG, KG_null_proj, KG_potent_orth = generate_models.get_KG_decoder_grom(i_d)

            elif animal == 'jeev':
                KG, KG_null_proj, KG_potent_orth = generate_models.get_KG_decoder_jeev(i_d)

            ### Get null and potent -- KG_null_proj, KG_potent_orth
            spks_null = np.dot(KG_null_proj, spks.T).T
            spks_pot =  np.dot(KG_potent_orth, spks.T).T

            #### Why is this multiplied by 10 --> conversion to Hz; 
            NP[animal, 'nul'].append(np.var(10*spks_null, axis=0))
            NP[animal, 'pot'].append(np.var(10*spks_pot, axis=0))
            NP[animal, 'tot'].append(np.var(10*spks, axis=0))

            assert np.allclose(spks, spks_null + spks_pot)

            N = spks.shape[1]; 
            tsk  = model_dict[i_d, 'task']
            targ = model_dict[i_d, 'trg']
            push = model_dict[i_d, 'np']

            #### Get the disc of commands 
            commands_disc = util_fcns.commands2bins([push], mag_boundaries, animal, i_d, vel_ix = [0, 1])[0]

            ### Now go through combos and plot 
            for mag_i in range(4):
                for ang_i in range(8):
                    for targi in range(8):

                        ### Get all the CO commands that fit; 
                        ix_co = (commands_disc[:, 0] == mag_i) & (commands_disc[:, 1] == ang_i) & (tsk == 0) & (targ == targi)
                        
                        if len(np.nonzero(ix_co == True)[0]) > min_obs:

                            for targi2 in range(8):

                                #### All the obstacle commands #####
                                ix_ob = (commands_disc[:, 0] == mag_i) & (commands_disc[:, 1] == ang_i) & (tsk == 1) & (targ == targi2)

                                if len(np.nonzero(ix_ob == True)[0]) > min_obs:
                                    
                                    ##### Get the indices; ####
                                    ix_co0 = np.nonzero(ix_co == True)[0]
                                    ix_ob0 = np.nonzero(ix_ob == True)[0]

                                    ii = np.random.permutation(len(ix_co0))
                                    i1 = ii[:int(len(ix_co0)/2.)]
                                    i2 = ii[int(len(ix_co0)/2.):]

                                    jj = np.random.permutation(len(ix_ob0))
                                    j1 = jj[:int(len(ix_ob0)/2.)]
                                    j2 = jj[int(len(ix_ob0)/2.):]

                                    ix_co1 = ix_co0[i1]
                                    ix_co2 = ix_co0[i2]
                                    ix_ob1 = ix_ob0[j1]
                                    ix_ob2 = ix_ob0[j2]

                                    assert np.sum(np.isnan(ix_co1)) == np.sum(np.isnan(ix_co2)) == np.sum(np.isnan(ix_ob1)) == np.sum(np.isnan(ix_ob2)) == 0
                                    assert np.sum(np.isnan(np.mean(spks[ix_co1, :], axis=0))) == 0
                                    assert np.sum(np.isnan(np.mean(spks[ix_co2, :], axis=0))) == 0
                                    assert np.sum(np.isnan(np.mean(spks[ix_ob1, :], axis=0))) == 0
                                    assert np.sum(np.isnan(np.mean(spks[ix_ob2, :], axis=0))) == 0
                                    
                                    ### make the plot: 
                                    if cov: 
                                        diffs_null = util_fcns.get_cov_diffs(ix_co1, ix_ob1, spks_null, diffs_null)
                                        diffs_null = util_fcns.get_cov_diffs(ix_co2, ix_ob2, spks_null, diffs_null)

                                        diffs_pot = util_fcns.get_cov_diffs(ix_co1, ix_ob1, spks_pot, diffs_pot)
                                        diffs_pot = util_fcns.get_cov_diffs(ix_co2, ix_ob2, spks_pot, diffs_pot)

                                    else:
                                        diffs_null.append(np.mean(spks_null[ix_co1, :], axis=0) - np.mean(spks_null[ix_ob1, :], axis=0))
                                        diffs_null.append(np.mean(spks_null[ix_co2, :], axis=0) - np.mean(spks_null[ix_ob2, :], axis=0))

                                        diffs_pot.append(np.mean(spks_pot[ix_co1, :], axis=0) - np.mean(spks_pot[ix_ob1, :], axis=0))
                                        diffs_pot.append(np.mean(spks_pot[ix_co2, :], axis=0) - np.mean(spks_pot[ix_ob2, :], axis=0))

            if cov: 
                mult = 1; 
            else:
                mult = 10; 

            ##### Take absolute value of differences ####
            AD_nul = np.abs(np.hstack((diffs_null)))*mult # to Hz
            AD_pot = np.abs(np.hstack((diffs_pot)))*mult # To hz

            ##### Take non absolute value of differences ####
            DX_nul.append(AD_nul)
            DX_pot.append(AD_pot)

            days.append(np.hstack(( np.zeros_like(AD_nul) + i_d, np.zeros_like(AD_pot) + i_d)))
            grp.append(np.zeros_like(AD_nul))
            grp.append(np.ones_like(AD_pot))
            mets.append(AD_nul)
            mets.append(AD_pot)

            for i, (D, col) in enumerate(zip([AD_nul, AD_pot], ['r', 'k'])):
                ax.bar(i_d + i*.45, np.nanmean(D), color='k', edgecolor=col, width=.4, linewidth=1.0, alpha=.8)
                ax.errorbar(i_d + i*.45, np.nanmean(D), np.nanstd(D)/np.sqrt(len(D)), marker='|', color=col)
                mnz[i_d, i] = np.nanmean(D)
         
        grp = np.hstack((grp))
        mets = np.hstack((mets))
        days = np.hstack((days))

        DX_nul = np.hstack((DX_nul))
        DX_pot = np.hstack((DX_pot))

        ### look at: run_LME(Days, Grp, Metric):
        pv, slp = util_fcns.run_LME(days, grp, mets)

        print 'LME model, fixed effect is day, rand effect is nul vs. pot., N = %d, ndays = %d, pv = %.4f, slp = %.4f' %(len(days), len(np.unique(days)), pv, slp)

        ##
        axsumm.bar(0 + ia, np.mean(DX_nul), color='k', edgecolor='r', width=.4, linewidth=2.0, alpha = .7,)
        axsumm.bar(0.4 + ia, np.mean(DX_pot), color='k', edgecolor='none', width=.4, linewidth=2.0, alpha =.7)

        for i_d in range(ndays):
            axsumm.plot(np.array([0, .4]) + ia, mnz[i_d, :], '-', color='k', linewidth=1.0)

        if pv < 0.001: 
            axsumm.text(0.2+ia, np.max(mnz), '***')
        elif pv < 0.01: 
            axsumm.text(0.2+ia, np.max(mnz), '**')
        elif pv < 0.05: 
            axsumm.text(0.2+ia, np.max(mnz), '*')
        else:
            axsumm.text(0.2+ia, np.max(mnz), 'n.s.')

        ### Need to stats test the differences across populations:    
    axsumm.set_xticks([0.2, 1.2])
    axsumm.set_xticklabels(['G', 'J'])
    if cov:
        # axsumm.set_ylabel(' Abs Cov Diffs ($Hz^2$) ')  
        # axsumm.set_ylim([0, 30])
        axsumm.set_ylim([0., 6.])
        axsumm.set_ylabel(['Main Cov. Overlap'])

    else:
        axsumm.set_ylabel(' Abs Mean Diffs (Hz) ')  
        axsumm.set_ylim([0, 5])
    fsumm.tight_layout()

    #fsumm.savefig(fig_dir+'both_monks_nul_vs_pot_x_task_mean_diffs_cov%s.svg'%(str(cov)))
    ### Figure for nul variance; -- each list in NP[animal, 'nul'] is an Nx1 array of neural variance for that day (null or pot)
    
    fnul, axnul = plt.subplots(figsize = (4, 4))
    for ia, animal in enumerate(['grom', 'jeev']):
        M = []; D = []; O = []; 
        nul = np.hstack((NP[animal, 'nul']))
        pot = np.hstack((NP[animal, 'pot']))

        axnul.bar(ia, np.mean(nul), width=.4, color='k', edgecolor='r', linewidth=1.5, alpha=0.8)
        axnul.bar(ia + .41, np.mean(pot), width=.4, color='k', edgecolor='none', linewidth=1.5, alpha = 0.8)

        ymax = 0; 

        for i_d in range(len(NP[animal, 'nul'])):
            nul = NP[animal, 'nul'][i_d]
            pot = NP[animal, 'pot'][i_d]

            M.append(nul)
            M.append(pot)

            D.append(np.zeros_like(nul) + i_d)
            D.append(np.zeros_like(pot) + i_d)

            O.append(np.zeros_like(nul) + 0)
            O.append(np.zeros_like(pot) + 1)

            axnul.plot([ia, ia+.41], [np.mean(nul), np.mean(pot)], 'k-')
            ymax = np.max([ymax, np.mean(nul)])
            ymax = np.max([ymax, np.mean(pot)])

        M = np.hstack((M))
        D = np.hstack((D))
        O = np.hstack((O))

        ### Plots: 
        pv, slp = util_fcns.run_LME(D, O, M)

        if pv < 0.001:
            axnul.text(ia+.2, ymax, '***')
        elif pv < 0.01:
            axnul.text(ia+.2, ymax, '**')
        if pv < 0.05:
            axnul.text(ia+.2, ymax, '*')

    axnul.set_xticks([.4, 1.4])
    axnul.set_xticklabels(['G', 'J'])
    axnul.set_ylabel('Neural Variance ($Hz^2$)')
    axnul.set_ylim([0., 175.])
    fnul.tight_layout()
    #fnul.savefig(fig_dir+'null_pot_neural_var.svg')

#### Fig 2 ###############
def plot_real_mean_diffplot_r2_bar_tg_spec_7_gens(model_set_number = 3, min_obs = 15, 
    plot_ex = False, plot_disc = False, cov = False, percent_change = False):
    ### Take real task / target / command / neuron / day comparisons for each neuron in the BMI
    ### Plot within a bar 
    ### Plot only sig. different ones

    ### Plot cov. diffs (mat1 - mat2)
    mag_boundaries = pickle.load(open(analysis_config.config['grom_pref']+'radial_boundaries_fit_based_on_perc_feb_2019.pkl'))
    co_obs_cmap = [np.array([0, 103, 56])/255., np.array([46, 48, 146])/255., ]

    for ia, animal in enumerate(['grom','jeev']):
        model_dict = pickle.load(open(analysis_config.config[animal+'_pref']+'tuning_models_'+animal+'_model_set%d.pkl' %model_set_number, 'rb'))
        ndays = analysis_config.data_params[animal+'_ndays']
        
        if animal == 'grom':
            ndays = 9; 
            width = 3.

        elif animal == 'jeev':
            ndays = 4; 
            width = 2.

        ### Now bar plot that: 
        f, ax = plt.subplots(figsize=(width, 4))
        
        for i_d in range(ndays):
            if plot_ex: 
                if i_d == 0:
                    fex, axex = plt.subplots(ncols = 4, figsize = (8, 2.5))
                    axex_cnt = 0; 

            diffs = []; diffs_cov = []; 
            sig_diffs = []; 
            cnt = 0; 

            spks = model_dict[i_d, 'spks']
            N = spks.shape[1]; 
            tsk  = model_dict[i_d, 'task']
            targ = model_dict[i_d, 'trg']
            push = model_dict[i_d, 'np']
            commands_disc = util_fcns.commands2bins([push], mag_boundaries, animal, i_d, vel_ix = [0, 1])[0]

            ##################################
            ######### SNR / DISC THING #######
            #################################
            #min_max = np.zeros((N, 2, 2))
            #snr = np.zeros_like(min_max)

            if plot_disc:
                n_disc = np.zeros((4, 8, 2))
                ### For each neuron, get the min/max mean command for a command/task/target: 

                if animal == 'grom' and i_d == 0: 
                    for tsk_i in range(2): 
                        if tsk_i == 0:
                            targeti = 4; 
                        elif tsk_i == 1:
                            targeti = 5; 

                        for mag_i in range(4):
                            for ang_i in range(8): 
                                ix = (commands_disc[:, 0] == mag_i) & (commands_disc[:, 1] == ang_i) & (tsk == tsk_i) & (targ == targeti)
                                #if len(np.nonzero(ix==True)[0]) > min_obs:
                                ix = np.nonzero(ix == True)[0]
                                
                                ### Which neuron are we analyzing? 
                                n = 38; 
                                if len(ix) > 0:
                                    n_disc[mag_i, ang_i, tsk_i] = np.nanmean(spks[ix, n])

                    ### Make a disc plot: 
                    print('Day 0, Grom, Target 4 CO / Target 5 OBS / neuron 38')
                    disc_plot(n_disc)

            ### Now go through combos and plot 
            for mag_i in range(4):
                for ang_i in range(8):
                    for targi in range(8):
                        ix_co = (commands_disc[:, 0] == mag_i) & (commands_disc[:, 1] == ang_i) & (tsk == 0) & (targ == targi)
                        
                        if len(np.nonzero(ix_co == True)[0]) > min_obs:
                            
                            for targi2 in range(8):
                                ix_ob = (commands_disc[:, 0] == mag_i) & (commands_disc[:, 1] == ang_i) & (tsk == 1) & (targ == targi2)

                                if len(np.nonzero(ix_ob == True)[0]) > min_obs:
                                    ####
                                    ix_co0 = np.nonzero(ix_co == True)[0]
                                    assert(np.allclose(np.mean(spks[ix_co, :], axis=0), np.mean(spks[ix_co0, :], axis=0)))
                                            
                                    ### make the plot: 
                                    if cov:
                                        diffs = util_fcns.get_cov_diffs(ix_co, ix_ob, spks, diffs)
                                    else:
                                        if percent_change:
                                            ix_co0 = np.nonzero(ix_co == True)[0]
                                            ix_ob0 = np.nonzero(ix_ob == True)[0]

                                            tmp_dff = np.mean(spks[ix_co0, :], axis=0) - np.mean(spks[ix_ob0, :], axis=0)
                                            tmp_mn = np.mean(spks[np.hstack((ix_co0, ix_ob0)), :], axis=0)

                                            ### If zero tmp_diff; 
                                            assert(np.all(tmp_dff[tmp_mn == 0.] == 0.))
                                            tmp_mn[tmp_mn == 0.] = 1.

                                            perc_dff = np.array([float(td)/float(tm) for _, (td, tm) in enumerate(zip(tmp_dff, tmp_mn))])
                                            diffs.append(perc_dff)
                                        else:
                                            diffs.append(np.mean(spks[ix_co, :], axis=0) - np.mean(spks[ix_ob, :], axis=0))
                                    #### Get the covariance and plot that: #####

                                    for n in range(N): 
                                        ks, pv = scipy.stats.ks_2samp(spks[ix_co, n], spks[ix_ob, n])
                                        if pv < 0.05:
                                            pass
                                            #sig_diffs.append(np.mean(spks[ix_co, n]) - np.mean(spks[ix_ob, n]))

                                        if plot_ex:
                                            #if np.logical_and(len(ix_co) > 30, len(ix_ob) > 30):
                                            if i_d == 0:
                                                if axex_cnt < 4:
                                                    if np.logical_and(mag_i > 1, n == 38): 
                                                        if np.logical_and(targi == 4, targi2 == 5):
                                                            axi = axex[axex_cnt]# / 9, axex_cnt %9]

                                                            util_fcns.draw_plot(0, 10*spks[ix_co, n], co_obs_cmap[0], 'white', axi)
                                                            util_fcns.draw_plot(1, 10*spks[ix_ob, n], co_obs_cmap[1], 'white', axi)

                                                            print('Animal %s, Day %d, Tg CO %d, Tg OB %d, ang %d, mag %d, mFR Co = %.2f, mFR OB = %.2f'%(animal,
                                                                i_d, targi, targi2, ang_i, mag_i, np.mean(10*spks[ix_co, n]), np.mean(10*spks[ix_ob, n])))

                                                            axi.set_title('A:%d, M:%d, NN%d, \nCOT: %d, OBST: %d, Monk:%s'%(ang_i, mag_i, n, targi, targi2, animal),fontsize=6)
                                                            #axi.set_ylabel('Firing Rate (Hz)')
                                                            axi.set_xlim([-.5, 1.5])
                                                            if ang_i == 1 and mag_i == 3:
                                                                axi.set_ylim([50, 150])
                                                            #axi.set_ylim([-1., 10*(1+np.max(np.hstack(( spks[ix_co, n], spks[ix_ob, n]))))])
                                                            axex_cnt += 1
                                                            if axex_cnt == 4:
                                                                fex.tight_layout()
                                                                fex.savefig(fig_dir + 'monk_%s_ex_mean_diffs.eps'%animal)

            if len(sig_diffs) > 0:
                SD = np.abs(np.hstack((sig_diffs)))*10 # to Hz
            else:
                SD = []; 

            #### Absolute difference in means; 
            if percent_change:
                AD = np.abs(np.hstack((diffs))) ## Fraction, no need to convert to hz
            else:
                AD = np.abs(np.hstack((diffs)))*10 # to Hz
            if cov:
                assert(len(AD) <= 4*8*8*8*N*(N-1))
            else:
                assert(len(AD) <= 4*8*8*8*N)

            for i, (D, col) in enumerate(zip([AD], ['k'])): #, SD, CD], ['k', 'r', 'b'])):
                
                ### Plot this thing; 
                util_fcns.draw_plot(i_d, D, 'k', [1, 1, 1, .2], ax, .8)
                #ax.bar(i_d + i*.3, np.mean(D), color='w', edgecolor='k', width=.8, linewidth=1.)
                #ax.errorbar(i_d + i*.3, np.mean(D), np.std(D)/np.sqrt(len(D)), marker='|', color=col)
                
                ### add all the dots 
                ax.plot(np.random.randn(len(D))*.1 + i*.3 + i_d, D, '.', color='gray', markersize=1.5, alpha = .5)
        
        if percent_change:
            ax.set_ylabel('Abs Fraction Change Across Conditions')
        else:
            ax.set_ylabel('Abs Hz Diff Across Conditions')
        ax.set_xlabel('Days')
        ax.set_xlim([-1, ndays + 1])
        if cov:
            pass
        else:
            if percent_change:
                ax.set_ylim([0., 3.])
            else:
                ax.set_ylim([0., 20.])
        ax.set_title('Monk '+animal[0].capitalize())
        f.tight_layout()
        f.savefig(fig_dir + 'monk_%s_mean_diffs_box_plots_cov_%s_perc_diff%s.eps' %(animal, str(cov), str(percent_change)), transparent = True)

def plot_real_mean_diffs_wi_vs_x(model_set_number = 3, min_obs = 15, cov = False, percent_change = False, skip_within = False):
    ### Take real task / target / command / neuron / day comparisons for each neuron in the BMI
    ### Plot within a bar 
    ### Plot only sig. different ones

    ### Plot cov. diffs (mat1 - mat2)
    mag_boundaries = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/radial_boundaries_fit_based_on_perc_feb_2019.pkl'))

    ### Now bar plot that: 
    fsumm, axsumm = plt.subplots(figsize=(4, 4))

    for ia, animal in enumerate(['grom','jeev']):
        model_dict = pickle.load(open(analysis_config.config[animal+'_pref']+'tuning_models_'+animal+'_model_set%d.pkl' %model_set_number, 'rb'))
        
        if animal == 'grom':
            ndays = 9; 
        elif animal == 'jeev':
            ndays = 4; 

        DWI = []; 
        DX = []; 
        DSH = []; 
        days = []; mets = []; grp = []; 
        mnz = np.zeros((ndays, 3)) # days x [x / wi]

        f, ax = plt.subplots(figsize=(ndays/2., 4))

        for i_d in range(ndays):

            diffs_shuff = []; 
            diffs_wi = []; 
            diffs_x = []; 

            spks = model_dict[i_d, 'spks']
            N = spks.shape[1]; 
            tsk  = model_dict[i_d, 'task']
            targ = model_dict[i_d, 'trg']
            push = model_dict[i_d, 'np']
            commands_disc = util_fcns.commands2bins([push], mag_boundaries, animal, i_d, vel_ix = [0, 1])[0]

            ### Now go through combos and plot 
            spks_shuff = np.zeros_like(spks)
            for mag_i in range(4):
                for ang_i in range(8):
                    
                    #### Get the command-specifci indices; 
                    all_ix = np.nonzero(np.logical_and(commands_disc[:, 0] == mag_i, commands_disc[:, 1] == ang_i))[0]

                    ### Shuffle them ###
                    shuff_ix = np.random.permutation(len(all_ix))

                    ### shuffle up the spks for this particular command ###
                    spks_shuff[all_ix, :] = spks[all_ix[shuff_ix], :]

            assert(np.allclose(np.sum(spks_shuff, axis=0), np.sum(spks, axis=0)))
            assert(np.allclose(np.mean(spks_shuff, axis=0), np.mean(spks, axis=0)))

            for mag_i in range(4):
                for ang_i in range(8):
                    all_ix = np.nonzero(np.logical_and(commands_disc[:, 0] == mag_i, commands_disc[:, 1] == ang_i))[0]
                    for targi in np.unique(targ):
                        ix_co = (commands_disc[:, 0] == mag_i) & (commands_disc[:, 1] == ang_i) & (tsk == 0) & (targ == targi)
                        ix_co0 = np.nonzero(ix_co == True)[0]
                        for ii in ix_co0: assert(ii in all_ix)

                        if len(np.nonzero(ix_co == True)[0]) > min_obs:

                            for targi2 in np.unique(targ):
                                ix_ob = (commands_disc[:, 0] == mag_i) & (commands_disc[:, 1] == ang_i) & (tsk == 1) & (targ == targi2)
                                ix_ob0 = np.nonzero(ix_ob == True)[0]
                                for ii in ix_ob0: assert(ii in all_ix)
                        
                                if len(np.nonzero(ix_ob == True)[0]) > min_obs:
                                    
                                    ii = np.random.permutation(len(ix_co0))
                                    i1 = ii[:int(len(ix_co0)/2.)]
                                    i2 = ii[int(len(ix_co0)/2.):]

                                    jj = np.random.permutation(len(ix_ob0))
                                    j1 = jj[:int(len(ix_ob0)/2.)]
                                    j2 = jj[int(len(ix_ob0)/2.):]

                                    ix_co1 = ix_co0[i1]
                                    ix_co2 = ix_co0[i2]
                                    ix_ob1 = ix_ob0[j1]
                                    ix_ob2 = ix_ob0[j2]

                                    assert np.sum(np.isnan(ix_co1)) == np.sum(np.isnan(ix_co2)) == np.sum(np.isnan(ix_ob1)) == np.sum(np.isnan(ix_ob2)) == 0
                                    assert np.sum(np.isnan(np.mean(spks[ix_co1, :], axis=0))) == 0
                                    assert np.sum(np.isnan(np.mean(spks[ix_co2, :], axis=0))) == 0
                                    assert np.sum(np.isnan(np.mean(spks[ix_ob1, :], axis=0))) == 0
                                    assert np.sum(np.isnan(np.mean(spks[ix_ob2, :], axis=0))) == 0
                                    
                                    if cov:
                                        ### make the plot: 
                                        diffs_wi = util_fcns.get_cov_diffs(ix_co1, ix_co2, spks, diffs_wi)
                                        diffs_wi = util_fcns.get_cov_diffs(ix_ob1, ix_ob2, spks, diffs_wi)

                                        diffs_x = util_fcns.get_cov_diffs(ix_co1, ix_ob1, spks, diffs_x)
                                        diffs_x = util_fcns.get_cov_diffs(ix_co2, ix_ob2, spks, diffs_x)

                                    else:
                                        if percent_change: 
                                            ### make the plot: 
                                            diffs_shuff.append(return_perc_diff(spks_shuff, ix_co1, ix_ob1))
                                            diffs_shuff.append(return_perc_diff(spks_shuff, ix_co2, ix_ob2))

                                            diffs_wi.append(return_perc_diff(spks, ix_co1, ix_co2))
                                            diffs_wi.append(return_perc_diff(spks, ix_ob1, ix_ob2))

                                            diffs_x.append(return_perc_diff(spks, ix_co1, ix_ob1))
                                            diffs_x.append(return_perc_diff(spks, ix_co2, ix_ob2))

                                        else:
                                            ### make the plot: 
                                            diffs_shuff.append(np.mean(spks_shuff[ix_co1, :], axis=0) - np.mean(spks_shuff[ix_ob1, :], axis=0))
                                            diffs_shuff.append(np.mean(spks_shuff[ix_co2, :], axis=0) - np.mean(spks_shuff[ix_ob2, :], axis=0))
                                            
                                            diffs_wi.append(np.mean(spks[ix_co1, :], axis=0) - np.mean(spks[ix_co2, :], axis=0))
                                            diffs_wi.append(np.mean(spks[ix_ob1, :], axis=0) - np.mean(spks[ix_ob2, :], axis=0))

                                            diffs_x.append(np.mean(spks[ix_co1, :], axis=0) - np.mean(spks[ix_ob1, :], axis=0))
                                            diffs_x.append(np.mean(spks[ix_co2, :], axis=0) - np.mean(spks[ix_ob2, :], axis=0))

            if cov:
                mult = 1; 
            else: 
                if percent_change:
                    mult = 1;
                else:
                    mult = 10; 

            AD_sh = np.abs(np.hstack((diffs_shuff)))*mult # to hz; 
            AD_wi = np.abs(np.hstack((diffs_wi)))*mult # to Hz
            AD_x = np.abs(np.hstack((diffs_x)))*mult # To hz

            DSH.append(AD_sh)
            DWI.append(AD_wi)
            DX.append(AD_x)

            days.append(np.hstack(( np.zeros_like(AD_wi) + i_d, np.zeros_like(AD_x) + i_d, np.zeros_like(AD_sh) + i_d)))
            grp.append( np.hstack(( np.zeros_like(AD_wi) + 1,   np.zeros_like(AD_x), np.zeros_like(AD_sh) + 2))) # looking for + slp
            
            mets.append(AD_wi)
            mets.append(AD_x)
            mets.append(AD_sh)

            if skip_within:
                for i, (D, col) in enumerate(zip([AD_x, AD_sh], ['k', 'lightgray'])):
                    ax.bar(i_d + i*.25, np.nanmean(D), color=col, edgecolor='none', width=.25, linewidth=1.0)
                    ax.errorbar(i_d + i*.25, np.nanmean(D), np.nanstd(D)/np.sqrt(len(D)), marker='|', color=col)
                    mnz[i_d, i] = np.nanmean(D)
            else:
                for i, (D, col) in enumerate(zip([AD_x, AD_wi, AD_sh], ['k', 'gray', 'lightgray'])):
                    ax.bar(i_d + i*.25, np.nanmean(D), color=col, edgecolor='none', width=.25, linewidth=1.0)
                    ax.errorbar(i_d + i*.25, np.nanmean(D), np.nanstd(D)/np.sqrt(len(D)), marker='|', color=col)
                    mnz[i_d, i] = np.nanmean(D)
         
        DWI = np.hstack((DWI))
        DX = np.hstack((DX))
        DSH = np.hstack((DSH))

        days = np.hstack((days))
        grp = np.hstack((grp))
        mets = np.hstack((mets))

        assert(len(days) == len(grp) == len(mets))

        ### look at: run_LME(Days, Grp, Metric):
        pv, slp = util_fcns.run_LME(days[grp < 2], grp[grp < 2], mets[grp < 2])
        print 'LME model, fixed effect is day, rand effect is X vs. Wi., N = %d, ndays = %d, pv = %.4f, slp = %.4f' %(len(days[grp < 2]), len(np.unique(days[grp < 2])), pv, slp)
        
        pv1, slp1 = util_fcns.run_LME(days[grp != 1], grp[grp != 1], mets[grp != 1])
        print 'LME model, fixed effect is day, rand effect is X vs. SHUFF., N = %d, ndays = %d, pv = %.4f, slp = %.4f' %(len(days[grp != 1]), len(np.unique(days[grp != 1])), pv1, slp1)

        ###
        axsumm.bar(0 + ia, np.mean(DX), color='k', edgecolor='none', width=.25, linewidth=2.0, alpha = .5)
        if skip_within:
            axsumm.bar(0.25 + ia, np.mean(DSH), color='lightgray', edgecolor='none', width=.25, linewidth=2.0, alpha =.5)
        else:
            axsumm.bar(0.25 + ia, np.mean(DWI), color='gray', edgecolor='none', width=.25, linewidth=2.0, alpha =.5)
            axsumm.bar(0.5 + ia, np.mean(DSH), color='lightgray', edgecolor='none', width=.25, linewidth=2.0, alpha =.5)

        # axsumm.plot(ia + (np.random.randn(len(DX))*.01), DX, 'k.', markersize = 2)
        # axsumm.plot(0.25 + ia + (np.random.randn(len(DWI))*.01), DWI, 'k.', markersize = 2)
        # axsumm.plot(0.50 + ia + (np.random.randn(len(DSH))*.01), DSH, 'k.', markersize = 2)

        # axsumm.violinplot([DX, DWI, DSH], positions = [0+ia, 0.25+ia, 0.5+ia], widths = 0.25,
        #     showmeans = True)

        for i_d in range(ndays):
            if skip_within:
                axsumm.plot(np.array([0, .25 ]) + ia, mnz[i_d, [0, 1]], '-', color='k', linewidth=1.0)
            else:
                axsumm.plot(np.array([0, .25 , .5]) + ia, mnz[i_d, :], '-', color='k', linewidth=1.0)


        axsumm.plot(ia + np.array([0., .25]), [1.02*np.max(mnz), 1.02*np.max(mnz)], 'k-')
        if skip_within:
            pass
        else:
            axsumm.plot(ia + np.array([0., .5]) , [1.1*np.max(mnz), 1.1*np.max(mnz)], 'k-')

        if skip_within:
            PVs = [pv1]
        else:
            PVs = [pv, pv1]
        
        for ip, pvi in enumerate(PVs):
            if ip == 0:
                mult = 1.02
            else:
                mult = 1.1

            if pvi < 0.001: 
                axsumm.text(.125 + ia, mult*np.max(mnz), '***', ha='center')
            elif pvi < 0.01: 
                axsumm.text(.125 + ia, mult*np.max(mnz), '**', ha='center')
            elif pvi < 0.05: 
                axsumm.text(.125 + ia, mult*np.max(mnz), '*', ha='center')
            else:
                axsumm.text(.125 + ia, mult*np.max(mnz), 'n.s.', ha='center')

        # ax.set_ylabel('Difference in Hz')
        # ax.set_xlabel('Days')
        # ax.set_xticks(np.arange(ndays))
        # ax.set_title('Monk '+animal[0].capitalize())
        # f.tight_layout()

        ### Need to stats test the differences across populations:    
    #axsumm.set_xticks([0.25, 1.25])
    if skip_within:
        axsumm.set_xticks([0., .25, 1., 1.25])
        axsumm.set_xticklabels(['Grom\nX Cond','Grom\nX Cond Shuff',
                            'Jeev\nX Cond', 'Jeev\nX Cond Shuff'],
                            rotation = 45)
    else:
        axsumm.set_xticks([0., .25, .5, 1., 1.25, 1.5])
        axsumm.set_xticklabels(['Grom\nX Cond', 'Grom\nW Cond','Grom\nX Cond Shuff',
                            'Jeev\nX Cond', 'Jeev\nW Cond','Jeev\nX Cond Shuff'],
                            rotation = 45, fontsize = 8)
    #axsumm.set_xticklabels(['G', 'J']) 
    if cov:
        axsumm.set_ylim([0, 30])
        axsumm.set_ylabel(' Cov Diffs ($Hz^2$) ') 
        #axsumm.set_ylim([0, .6])
        #axsumm.set_ylabel(' Main Cov. Overlap ') 
    else:
        if percent_change:
            axsumm.set_ylim([0, 1])
            axsumm.set_ylabel(' Fraction change in mFR ') 
        else:
            axsumm.set_ylim([0, 5])
            axsumm.set_ylabel(' Mean Diffs (Hz) ') 
    fsumm.tight_layout()
    fsumm.savefig(fig_dir+'both_monks_w_vs_x_task_mean_diffs_cov%s_perc_change%s.eps'%(str(cov), str(percent_change)))

def return_perc_diff(spks, ix0, ix1):
    tmp_diff = np.mean(spks[ix0, :], axis=0) - np.mean(spks[ix1, :], axis=0)
    #tmp_mn = spks[np.hstack((ix0, ix1)), :]
    #assert(tmp_mn.shape[0] == len(ix0) + len(ix1))
    tmp_mn = np.mean(spks, axis=0)
    assert(np.all(tmp_diff[tmp_mn == 0.] == 0.))
    tmp_mn[tmp_mn==0.] = 1
    perc_diff = np.array([td/tm for _, (td, tm) in enumerate(zip(tmp_diff, tmp_mn))])
    assert(len(perc_diff) == spks.shape[1])
    return perc_diff

def disc_plot(n_disc, polygon = False, wedge = True):
    '''
    n_disc is 4 x 8 x 2 --> mag / angle / task 
    '''
    if polygon:
        bw = np.array([0., 0., 0.])
        co_obs_cmap = [bw, bw]
        co_obs_cmap_cm = []; 

        for _, (c, cnm) in enumerate(zip(co_obs_cmap, ['co', 'obs'])):
            colors = [[1, 1, 1], c]  # white --> color
            cmap_name = 'my_list'
            cm = LinearSegmentedColormap.from_list(
                cnm, colors, N=1000)
            co_obs_cmap_cm.append(cm)


        fig, (ax1, ax2) = plt.subplots(ncols=2, subplot_kw=dict(projection='polar'))

        # Generate some data...
        # Note that all of these are _2D_ arrays, so that we can use meshgrid
        # You'll need to "grid" your data to use pcolormesh if it's un-ordered points
        theta, r = np.mgrid[-np.pi/8.:(2*np.pi-np.pi/8.):9j, 0:4:5j]
        
        im1 = ax1.pcolormesh(theta, r, n_disc[:, :, 0].T, cmap = co_obs_cmap_cm[0], vmin=np.min(n_disc), vmax=np.max(n_disc))
        im2 = ax2.pcolormesh(theta, r, n_disc[:, :, 1].T, cmap = co_obs_cmap_cm[1], vmin=np.min(n_disc), vmax=np.max(n_disc))
        for ax in [ax1, ax2]:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
        fig.savefig(fig_dir+'grom_day0_neur_38_targ4_targ5_dist.eps')


    elif wedge:
        import matplotlib.colors as colors
        from matplotlib import cm
        import matplotlib.patches as mpatches

        cmap = plt.get_cmap('binary')
        cNorm = colors.Normalize(vmin=np.min(n_disc), vmax=np.max(n_disc))
        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)

        f, ax = plt.subplots(ncols = 2)

        angles1 = np.linspace(0., 2*np.pi, 9.) - np.pi/8
        angles2 = np.linspace(0., 2*np.pi, 9.) + np.pi/8
        angles1 = angles1[:-1]
        angles2 = angles2[:-1]
        angles1 = angles1 / np.pi * 180.
        angles2 = angles2 / np.pi * 180.
        mag1 = range(4)
        mag2 = range(1, 5)

        for i_t in range(2):
            axi = ax[i_t]
            axi.axis('square')

            for m in range(4):
                for a in range(8):
                    colorVal = scalarMap.to_rgba(n_disc[m, a, i_t])
                    patch = mpatches.Wedge(center=(0., 0.),
                                           r=mag2[m],
                                           theta1=angles1[a],
                                           theta2=angles2[a],
                                           width=mag2[m]-mag1[m],
                                           fill=True, facecolor=colorVal)
                    axi.add_patch(patch)
            axi.set_yticklabels([])
            axi.set_xticklabels([])
            axi.set_xlim([-5, 5])
            axi.set_ylim([-5, 5])
        f.savefig(fig_dir+'grom_day0_neur_38_targ4_targ5_dist_wedge.eps')

def tiny_wedge_plot(mag, ang):
    import matplotlib.patches as mpatches
    angles1 = np.linspace(0., 2*np.pi, 9.) - np.pi/8
    angles2 = np.linspace(0., 2*np.pi, 9.) + np.pi/8
    angles1 = angles1[:-1]
    angles2 = angles2[:-1]
    angles1 = angles1 / np.pi * 180.
    angles2 = angles2 / np.pi * 180.
    mag1 = range(4)
    mag2 = range(1, 5)

    f, ax = plt.subplots()
    ax.axis('square')
    patch = mpatches.Wedge(center=(0., 0.),
                           r=mag2[mag],
                           theta1=angles1[ang],
                           theta2=angles2[ang],
                           width=mag2[mag]-mag1[mag],
                           fill=False, ec = 'k', lw = 1.)
    ax.add_patch(patch)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    f.savefig(fig_dir+'tiny_wedge_m%d_a%d.eps' %(mag, ang))

#### Fig 4 ----- bar plots of diff models ####
def plot_r2_bar_model_1(min_obs = 15, 
    ndays = None, pt_2 = False, r2_pop = True, 
    perc_increase = True, model_set_number = 1,
    task_spec_ix = None, include_shuffs = None):
    
    '''
    inputs: min_obs -- number of obs need to count as worthy of comparison; 
    perc_increase: get baseline from y_t | a_t and use this to compute the "1" value; 
    ndays -- ? 
    pt_2 more plots -- not totally sure what they are;  
    r2_pop -- assume its population vs. indivdual 
    model_set_number -- which models to plot from which set; 
    include_shuffs -- Number of shuffles to include (will search for range(include_shuffs)). Right now code is just programmed to get index "0"
    '''

    ### For stats each neuron is an observation ##
    mag_boundaries = pickle.load(open(analysis_config.config['grom_pref']+ 'radial_boundaries_fit_based_on_perc_feb_2019.pkl'))
    key = 'spks' 

    ### Now generate plots -- testing w/ 1 day
    if ndays is None:
        ndays = dict(grom=9, jeev=4)
    else:
        ndays = dict(grom=ndays, jeev=ndays)

    for ia, (animal, yr) in enumerate(zip(['grom','jeev'], ['2016','2013'])):
        
        if model_set_number == 1:
            models_to_include = ['prespos_0psh_1spksm_0_spksp_0', 
                                'hist_1pos_-1psh_1spksm_0_spksp_0', 
                                'hist_1pos_1psh_1spksm_0_spksp_0',
                                'hist_1pos_2psh_1spksm_0_spksp_0', 
                                'hist_1pos_3psh_1spksm_0_spksp_0', 
                                'hist_1pos_3psh_1spksm_1_spksp_0',
                                'hist_1pos_0psh_1spksm_1_spksp_0',]
                                
                                #'hist_1pos_-1psh_1spksm_1_spksp_0', 
                                #'hist_1pos_1psh_1spksm_1_spksp_0',
                                #'hist_1pos_2psh_1spksm_1_spksp_0']

            models_to_compare = np.array([0, 4, 5])
            models_colors = [[255., 0., 0.], 
                             [101, 44, 144],
                             [101, 44, 144],
                             [101, 44, 144],
                             [101, 44, 144],
                             [39, 169, 225],
                             [39, 169, 225],]
                             #[39, 169, 225],
                             #[39, 169, 225],
                             #[39, 169, 225]]
            xlab = [
            '$a_{t}$',
            '$a_{t}, p_{t-1}$',
            '$a_{t}, p_{t-1}, v_{t-1}$',
            '$a_{t}, p_{t-1}, v_{t-1}, tg$',
            '$a_{t}, p_{t-1}, v_{t-1}, tg, tsk$',
            '$a_{t}, p_{t-1}, v_{t-1}, tg, tsk, y_{t-1}$',
            '$a_{t}, y_{t-1}$']#,
            #'$a_{t}, y_{t-1}, p_{t-1}$',
            #'$a_{t}, y_{t-1}, p_{t-1}, v_{t-1}$',
            #'$a_{t}, y_{t-1}, p_{t-1}, v_{t-1}, tg$']

        elif model_set_number == 2:
            models_to_include = ['prespos_0psh_1spksm_0_spksp_0', 
                                 'hist_1pos_3psh_1spksm_0_spksp_0', 
                                 'hist_1pos_3psh_1spksm_1_spksp_0',
                                 'hist_1pos_0psh_1spksm_1_spksp_0']
            
            models_to_compare = []#np.array([0, 1, 2, 3])
            xlab = ['$a_{t}$',             
                    '$a_{t}, p_{t-1}, v_{t-1}, tg, tsk$',
                    '$a_{t}, p_{t-1}, v_{t-1}, tg, tsk, y_{t-1}$',
                    '$a_{t}, y_{t-1}$']

            models_colors = [[255., 0., 0.], 
                             [101, 44, 144],
                             [39, 169, 225],
                             [39, 169, 225]]

        elif model_set_number == 3: 
            models_to_include = [#'prespos_0psh_0spksm_1_spksp_0',
                                 #'prespos_0psh_0spksm_1_spksp_0potent',
                                 #'prespos_0psh_0spksm_1_spksp_0null',
                                 'hist_1pos_0psh_0spksm_1_spksp_0',]
                                 #'hist_1pos_0psh_1spksm_0_spksp_0']
                                 #'hist_1pos_0psh_0spksm_1_spksp_0potent',
                                 #'hist_1pos_0psh_0spksm_1_spksp_0null']
            models_colors = [[39, 169, 225]]
            xlab = ['$y_{t-1}$']
            models_to_compare = []

        elif model_set_number == 6:
            models_to_include = ['hist_1pos_0psh_2spksm_1_spksp_0',
                                 'hist_1pos_0psh_1spksm_1_spksp_0',
                                 'identity_dyn',
                                 'hist_1pos_0psh_0spksm_1_spksp_0']

            xlab = ['$y_t = f(y_{t-1} | a_t)$', 
                    '$y_t = f(y_{t-1}, a_t)$', 
                    '$y_t = y_{t-1}$',
                    '$y_t = f(y_{t-1})$']
            models_colors = [[39, 169, 225], 
                             [39, 169, 225], 
                             [39, 169, 225], 
                             [39, 169, 225], ]
            models_to_compare = []


        #### Number of models ######
        M = len(models_to_include)

        #### Make sure match #######
        assert(len(models_colors) == M)

        #### normalize the colors #######
        models_colors = [np.array(m)/256. for m in models_colors]

        ##### fold ? ######
        #fold = ['maroon', 'orangered', 'goldenrod', 'teal', 'blue']
        
        if r2_pop:
            pop_str = 'Population'
        else:
            pop_str = 'Indiv'        

        ### Go through each neuron and plot mean true vs. predicted (i.e R2) for a given command  / target combo: 
        if task_spec_ix is None:
            model_dict = pickle.load(open(analysis_config.config[animal+'_pref']+'tuning_models_'+animal+'_model_set%d.pkl'%model_set_number, 'rb'))
        
        #### Only for task specific ####
        else:
            if os.path.exists(analysis_config.config[animal+'_pref']+'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen_match_tsk_N.pkl'%model_set_number):
                print('Option 1')
                model_dict = pickle.load(open(analysis_config.config[animal+'_pref']+'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen_match_tsk_N.pkl'%model_set_number, 'rb'))
            else:
                print('Option 2')
                model_dict = pickle.load(open(analysis_config.config[animal+'_pref']+'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen.pkl'%model_set_number, 'rb'))
        
        if include_shuffs is not None:
            shuffs = []
            pre3 = analysis_config.config[animal+'_pref']+'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen_match_tsk_N_'%model_set_number
            pre2 = analysis_config.config[animal+'_pref']+'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen_'%model_set_number
            pre1 = analysis_config.config[animal+'_pref']+'tuning_models_'+animal+'_model_set%d__'%model_set_number

            for shuf_ix in range(include_shuffs):
                if os.path.exists('%swithin_bin_shuff%d.pkl'%(pre2, shuf_ix)):
                    print('Shuff option 2')
                    tmp = pickle.load(open('%swithin_bin_shuff%d.pkl'%(pre2, shuf_ix), 'rb'))
                    shuffs.append(copy.deepcopy(tmp))

                    # tmp = pickle.load(open('%sfull_shuff%d.pkl'%(pre1, shuf_ix), 'rb'))
                    # shuffs.append(copy.deepcopy(tmp))

                elif os.path.exists('%swithin_bin_shuff%d.pkl'%(pre3, shuf_ix)):
                    print('Shuff option 3')
                    tmp = pickle.load(open('%swithin_bin_shuff%d.pkl'%(pre3, shuf_ix), 'rb'))
                    shuffs.append(copy.deepcopy(tmp))

                    # tmp = pickle.load(open('%sfull_shuff%d.pkl'%(pre2, shuf_ix), 'rb'))
                    # shuffs.append(copy.deepcopy(tmp))

                elif os.path.exists('%swithin_bin_shuff%d.pkl'%(pre1, shuf_ix)):
                    print('Shuff option 1')
                    tmp = pickle.load(open('%swithin_bin_shuff%d.pkl'%(pre1, shuf_ix), 'rb'))
                    shuffs.append(copy.deepcopy(tmp))

                    # tmp = pickle.load(open('%sfull_shuff%d.pkl'%(pre3, shuf_ix), 'rb'))
                    # shuffs.append(copy.deepcopy(tmp))
                else:
                    raise Exception('No shuffles of index %d identified! '%(shuf_ix))

        #### Setup the plot ####
        f, ax = plt.subplots(figsize=(6, 6))
        if include_shuffs:
            fsh, axsh  = plt.subplots(figsize = (6, 6))
        
        ##### Data holder for either mean of individual neurons or population R2 ####
        ### Will be normalized by baseline if that setting is selected ######
        R2S = dict()

        ####### Holders for LME if not doing population R2 ########
        D = []; # Days; 
        Mod = []; # Model Number 

        ######### R2 for each neuron if not doing population R2  #######
        R2_stats = []; 

        ####### Iterate through each day #########
        for i_d in range(ndays[animal]):

            ###### Get the decoders ###########
            if animal == 'grom': 
                KG, KG_null_proj, KG_potent_orth = generate_models.get_KG_decoder_grom(i_d)
            
            elif animal == 'jeev':
                KG, KG_null_proj, KG_potent_orth = generate_models.get_KG_decoder_jeev(i_d)

            ###### True data #####

            R2s = dict()

            ###### Get the baseline ####
            if perc_increase: 
                
                #### Predicted data ? 
                if task_spec_ix is None:
                    pdata = model_dict[i_d, models_to_include[0]]
                else:
                    print('General task spec baseline, model: %s, %d' %(models_to_include[0], task_spec_ix))
                    pdata = model_dict[i_d, models_to_include[0]][:, :, task_spec_ix]

                tdata = model_dict[i_d, 'spks']
                R2_baseline = util_fcns.get_R2(tdata, pdata, pop = r2_pop)
                print('R2 baseline; %.4f' %(R2_baseline))

            mdz = [] ### This is to keep track of the future x-axis of shuffles plotted; 
            mdz_shuff = []; 

            for i_mod, mod in enumerate(models_to_include):
                
                ###### Predicted data, cross validated ####
                tdata = model_dict[i_d, 'spks']
                if task_spec_ix is None:
                    pdata = model_dict[i_d, mod]
                else:
                    pdata = model_dict[i_d, mod][:, :, task_spec_ix]

                ### Get population R2 ### 
                R2 = util_fcns.get_R2(tdata, pdata, pop = r2_pop)

                ### Only for indiv
                if i_mod in models_to_compare: 
                    if perc_increase:
                        if r2_pop:
                            pass
                        else:
                            assert(len(R2) == len(R2_baseline))
                        R2_stats.append((R2 - R2_baseline)/R2_baseline)
                    else:
                        R2_stats.append(R2)
                    
                    # Remove NaNs 
                    D.append(np.zeros_like(R2) + i_d)
                    Mod.append(np.zeros_like(R2) + i_mod)

                ### Put the R2 in a dictionary: 
                if perc_increase:
                    R2S[i_mod, i_d] = (R2 - R2_baseline)/R2_baseline
                else:
                    R2S[i_mod, i_d] = R2; 
                mdz.append(i_mod)

                #### Add shuffles ####
                if include_shuffs is not None:
                    assert(task_spec_ix is not None)
                    ### For each shuffle: 
                    width = 1./float(len(shuffs))

                    for i_s, shuf in enumerate(shuffs):
                        tdata = shuf[i_d, 'spks']
                        pdata = shuf[i_d, mod][:, :, task_spec_ix]
                        R2 = util_fcns.get_R2(tdata, pdata, pop = r2_pop)

                        if perc_increase:
                            R2_stats.append((R2 - R2_baseline)/R2_baseline)
                            R2S[i_mod + 0.1*(i_s+1)*width, i_d] = (R2 - R2_baseline)/R2_baseline
                        else:
                            R2_stats.append(R2)
                            R2S[i_mod + 0.1*(i_s+1)*width, i_d] = R2

                        D.append(np.zeros_like(R2) + i_d)
                        Mod.append(np.zeros_like(R2) + i_mod + 0.1*(i_s+1)*width)
                        mdz_shuff.append(i_mod + 0.1*(i_s+1)*width)

            ##### Plot this single day #####
            day_trace = []; 
            for i_mod2 in mdz:
                day_trace.append(R2S[i_mod2, i_d])

            #### Line plot of R2 w increasing models #####
            ax.plot(mdz, day_trace, '-', color='gray', linewidth = 1.)

            if include_shuffs:
                shday_trace = []; 
                for i_mod22 in mdz_shuff:
                    shday_trace.append(R2S[i_mod22, i_d])
                axsh.plot(mdz, shday_trace, '-', color='gray', linewidth=1.)

        #### Plots total mean pooled over days ###
        mean_bar = []; std_bar = []; 
        for i_mod3 in mdz:
            tmp2 = []; 
            for i_d in range(ndays[animal]):
                tmp2.append(R2S[i_mod3, i_d])
            tmp2 = np.hstack((tmp2))
            
            ## mean 
            mean_bar.append(np.mean(tmp2))

            ## s.e.m
            std_bar.append(np.std(tmp2)/np.sqrt(len(tmp2)))

        if include_shuffs is None:
            width = 1.
        else:
            sh_mean_bar = []; sh_std_bar = []; 
            for i_mod4 in mdz_shuff:
                tmp2sh = []; 
                for i_d in range(ndays[animal]):
                    tmp2sh.append(R2S[i_mod4, i_d])
                tmp2sh = np.hstack((tmp2sh))

                sh_mean_bar.append(np.mean(tmp2sh))
                sh_std_bar.append(np.std(tmp2sh)/np.sqrt(len(tmp2sh)))

        ### Overal for the full data #####
        for i_m, i_mod in enumerate(mdz):
            ### Plot integers; 
            if int(i_mod) == i_mod:
                color = models_colors[int(i_mod)]
            ax.bar(i_mod, mean_bar[i_m], color = color, edgecolor='k', linewidth = 1., width = width, alpha = 0.5)
            ax.errorbar(i_mod, mean_bar[i_m], yerr=std_bar[i_m], marker='|', color='k')    

            if include_shuffs is not None:
                axsh.bar(i_mod, sh_mean_bar[i_m], color=color, edgecolor='k', linewidth = 1., width = width, alpha = 0.2)
                axsh.errorbar(i_mod, sh_mean_bar[i_m], yerr=sh_std_bar[i_m], marker='|', color='k')    
                ax.set_ylim([-.3, .5])
                axsh.set_ylim([-.3, .5])
        ax.set_ylabel('%s R2, neur, perc_increase R2 %s'%(pop_str, perc_increase))
        ax.set_xticks(np.arange(M))
        ax.set_xticklabels(xlab, rotation=45)#, fontsize=6)
        f.tight_layout()

        if include_shuffs is not None:
            axsh.set_ylabel('%s R2, neur, perc_increase R2 %s'%(pop_str, perc_increase))
            axsh.set_xticks(np.arange(M))
            axsh.set_xticklabels(xlab, rotation=45)#, fontsize=6)
            fsh.tight_layout()

        ### Overall for the SHUFFLED data #####
        if task_spec_ix is None:
            f.savefig(fig_dir + animal + '_%sr2_behav_models_perc_increase%s_model%d.eps'%(pop_str, perc_increase, model_set_number))
            if include_shuffs is not None:
                fsh.savefig(fig_dir + animal + '_%sr2_behav_models_perc_increase%s_model%d_SHUFFLED.eps'%(pop_str, perc_increase, model_set_number))
        else:
            f.savefig(fig_dir + animal + '_%sr2_behav_models_perc_increase%s_model%d_tsk_spec_%d.eps'%(pop_str, perc_increase, model_set_number, task_spec_ix))
            if include_shuffs is not None:
                fsh.savefig(fig_dir + animal + '_%sr2_behav_models_perc_increase%s_model%d_tsk_spec_%d_SHUFFLED.eps'%(pop_str, perc_increase, model_set_number, task_spec_ix))
        
        ##### Print stats ####
        if model_set_number == 1:
            R2_stats = np.hstack((R2_stats))
            ix = ~np.isnan(R2_stats)

            if len(ix) < len(R2_stats):
                print('ERRORR')
                raise Exception
                # import pdb; pdb.set_trace()

            ### Get non-nans: 
            R2_stats = R2_stats[ix]
            D = np.hstack((D))[ix]
            Mod = np.hstack((Mod))[ix]

            ### Run 2 LMEs: [0, 4] and [4, 5]; 
            for j, (m1, m2) in enumerate([[0, 4], [4, 5]]):
                ix = np.nonzero(np.logical_or(Mod == m1, Mod== m2))[0]

                pv, slp = util_fcns.run_LME(D[ix], Mod[ix], R2_stats[ix])
                print '---------------------------'
                print '---------------------------'
                print 'Pop R2 %s, Percent Increase %s'%(r2_pop, perc_increase)
                print 'LME: Animal %s, Model 1: %d, Model 2: %d, Pv: %.4f, Slp: %.4f' %(animal, m1, m2, pv, slp)
                print '---------------------------'
                print '---------------------------'
        
        # if pt_2: 
        #     f, ax = plt.subplots(ncols = 2)
        #     f1, ax1 = plt.subplots(ncols = len(models_to_include), figsize=(12.5, 2.5))
        #     f2, ax2 = plt.subplots()
        #     ### now comput R2 ###
        #     Pred = dict(); Tru = dict(); 
            
        #     for i_d in range(ndays[animal]):

        #         ### Basics -- get the binning for the neural push commands: 
        #         neural_push = model_dict[i_d, 'np']

        #         ### Commands
        #         commands = util_fcns.commands2bins([neural_push], mag_boundaries, animal, i_d, vel_ix = [0, 1])[0]
                
        #         ### Get task / target
        #         tsk = model_dict[i_d, 'task']
        #         targ = model_dict[i_d, 'trg']
        #         bin_num = model_dict[i_d, 'bin_num']

        #         ### Now go through each task targ and assess when there are enough observations: 
        #         y_true = model_dict[i_d, key]
        #         T, N = y_true.shape

        #         ### Neural activity
        #         R2 = dict(); 
        #         for i_m, mod in enumerate(models_to_include):
        #             R2['co', mod] = []; 
        #             R2['obs', mod] = []; 
        #             R2['both', mod] = []; 

        #         for i_mag in range(4):
        #             for i_ang in range(8):
        #                 for i_t in range(2): # Task 
        #                     for targ in range(8): # target: 
        #                         ix0 = (commands[:,0] == i_mag) & (commands[:,1] == i_ang) & (tsk == i_t) & (targ == targ)
        #                         ix0 = np.nonzero(ix0 == True)[0]
        #                         if len(ix0) > min_obs:

        #                             for i_m, model in enumerate(models_to_include):

        #                                 if task_spec_ix is None:
        #                                     y_pred = model_dict[i_d, model]
        #                                 else:
        #                                     y_pred = model_dict[i_d, model][:, :, task_spec_ix]

        #                                 ### Get R2 of this observation: 
        #                                 spk_true = np.mean(y_true[ix0, :], axis=0)
        #                                 spk_pred = np.mean(y_pred[ix0, :], axis=0)

        #                                 if i_t == 0:
        #                                     R2['co', model].append([spk_true, spk_pred])
        #                                 elif i_t == 1:
        #                                     R2['obs', model].append([spk_true, spk_pred])

        #                                 ### Both 
        #                                 R2['both', model].append([spk_true, spk_pred])

        #         tsk_cols = ['b']#,'r','k']
        #         for i_t, tsk in enumerate(['both']):#, 'obs', 'both']):
        #             for i_m, model in enumerate(models_to_include):

        #                 tru_co = np.vstack(( [R[0] for R in R2[tsk, model]] ))
        #                 pre_co = np.vstack(( [R[1] for R in R2[tsk, model]] ))

        #                 SSR = np.sum((tru_co - pre_co)**2)# Not over neruons, axis = 0)
        #                 SST = np.sum((tru_co - np.mean(tru_co, axis=0)[np.newaxis, :])**2)#, axis=0)
        #                 R2_co_pop = 1 - (SSR/SST)
                        
        #                 SSR = np.sum((tru_co - pre_co)**2, axis = 0)
        #                 SST = np.sum((tru_co - np.mean(tru_co, axis=0)[np.newaxis, :])**2, axis=0)
        #                 R2_co_neur = 1 - (SSR/SST)

        #                 ax[0].plot(i_m, R2_co_pop, 'k*')
        #                 ax[0].set_ylabel('R2 -- of mean task/targ/command/neuron (population neuron R2)')
        #                 ax[0].set_ylim([-1, 1.])

        #                 ax[1].plot(i_m, np.nanmean(R2_co_neur), 'k*')
        #                 ax[1].set_ylabel('R2 -- of mean task/targ/command/neuron (individual neuron R2)')
        #                 ax[1].set_ylim([-1, 1.])

        #                 ax1[i_m].plot(tru_co.reshape(-1), pre_co.reshape(-1), 'k.', alpha=.2)
        #                 try:
        #                     Tru[model].append(tru_co.reshape(-1))
        #                     Pred[model].append(pre_co.reshape(-1))
        #                 except:
        #                     Tru[model] = [tru_co.reshape(-1)]
        #                     Pred[model] = [pre_co.reshape(-1)]
                            
        #     for i_m, model in enumerate(models_to_include):
        #         slp,intc,rv,pv,err =scipy.stats.linregress(np.hstack((Tru[model])), np.hstack((Pred[model])))
        #         x_ = np.linspace(np.min(np.hstack((Tru[model]))), np.max(np.hstack((Tru[model]))))
        #         y_ = slp*x_ + intc; 
        #         ax1[i_m].plot(x_, y_, '-', linewidth=.5)
        #         ax1[i_m].set_title('%s, \n pv=%.2f\nrv=%.2f\nslp=%.2f' %(model, pv, rv, slp),fontsize=8)


        #         if model == 'hist_1pos_0psh_0spksm_1_spksp_0':
        #             if task_spec_ix is None:
        #                 y_pred = model_dict[i_d, model]
        #             else:
        #                 y_pred = model_dict[i_d, model][:, :, task_spec_ix]
        #             y_true = model_dict[i_d, key]; 

        #             SSR = np.sum((y_pred - y_true)**2, axis=0)
        #             SST = np.sum((y_true - np.mean(y_true, axis=0)[np.newaxis, :])**2, axis=0)
        #             SST[np.isinf(SST)] = np.nan

        #             r22 = 1 - SSR/SST; 
        #             ax2.plot(r22)

        #             print 'R2 mean day %d, %.2f', (i_d, np.mean(r22))

            #f.tight_layout()
            #f1.tight_layout()

##########################################
##### July 2020 figs 4 / 5         #######
##########################################
def plot_r2_bar_model_dynamics_only(min_obs = 15, 
    r2_pop = True, perc_increase = True, model_dicts = None, 
    cond_on_act = True, plot_act = False, skip_task_spec = False, 
    skip_shuffle = False, fig4_opt = False, nshuffs = 1000):

    '''
    inputs: min_obs -- number of obs need to count as worthy of comparison; 
    perc_increase: get baseline from shuffle
    pt_2 more plots -- not totally sure what they are;  
    r2_pop -- assume its population vs. indivdual 
    model_set_number -- which models to plot from which set; 
    include_shuffs -- Number of shuffles to include (will search for range(include_shuffs)). Right now code is just programmed to get index "0"
    
    fig4_opt -- 
        bar w/ [0, 1, 3, 4] -- 0 = shuff, 1 = R2 dyn, 3 = R2 dyn split by task, 4 = R2 task-spec dyn split by task 
    '''

    def get_dyn(q, col=2):
        if len(q.shape) == 3:
            return q[:,:,col]
        elif len(q.shape) == 2:
            return q

    ### For stats each neuron is an observation ##
    mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))
    key = 'spks' 

    ### Now generate plots -- testing w/ 1 day
    ndays = dict(grom=9, jeev=4)

    if cond_on_act:
        model_name = 'hist_1pos_0psh_2spksm_1_spksp_0'
    else:
        model_name = 'hist_1pos_0psh_0spksm_1_spksp_0'

    if plot_act:
        assert(cond_on_act == False)

    f, ax = plt.subplots(figsize = (5, 5))
    
    fax_frac_sig, ax_frac_sig = plt.subplots(figsize = (3, 4))
    frac_sig = dict(grom=[], jeev=[])

    fax_gte_shuff, ax_gte_shuff = plt.subplots(figsize = (6, 4))

    fax_r2, ax_r2 = plt.subplots(figsize = (4, 4))

    pooled_stats = {}

    for ia, (animal, yr) in enumerate(zip(['grom','jeev'], ['2016','2013'])):

        if fig4_opt:
            xlab = ['shuffled', 
                    'general\ndynamics', 
                    'general\ndynamics by tsk',
                    'task-spec\ndynamics',
                    'identity_dyn']
            xkeys = ['shuffled', 'dyn_gen', 'dyn_gen_by_tsk', 'dyn_tsk_by_tsk', 'identity_dyn']
            nbars = 4

        else:
            xlab = ['shuffled', 
                    'general\ndynamics', 
                    'conditioning']
            xkeys = ['shuffled', 'dyn_gen', 'cond']
            nbars = 3

        if plot_act:
            models_colors = [[100, 100, 100], 
                         [255, 0, 0], 
                         [255, 0, 0], 
                         [100, 100, 100]]
        else:
            models_colors = [[100, 100, 100], 
                         [39, 169, 225],
                         #[39, 169, 225], 
                         [100, 100, 100]]

        if fig4_opt:
            models_colors = [[100, 100, 100], 
                         [39, 169, 225], 
                         [39, 169, 225], 
                         [39, 169, 225], 
                         [100, 100, 100]]            

        models_to_compare = []
        if skip_shuffle:
            nbars -= 1
        if skip_task_spec:
            nbars -= 1

        ax_offset = (nbars + 1)*ia

        #### Number of models ######
        M = len(xlab)

        #### Make sure match #######
        assert(len(models_colors) == M)

        #### normalize the colors #######
        models_colors = [np.array(m)/256. for m in models_colors]

        ##### fold ? ######
        #fold = ['maroon', 'orangered', 'goldenrod', 'teal', 'blue']
        if r2_pop:
            pop_str = 'Population'
        else:
            pop_str = 'Indiv'        

        ### Go through each neuron and plot mean true vs. predicted (i.e R2) for a given command  / target combo: 
        model_set_number = 6
        if model_dicts is None:
            model_dict = pickle.load(open(analysis_config.config[animal+'_pref']+'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen_match_tsk_N.pkl'%model_set_number, 'rb'))
        else:
            print('Using model dicts input')
            model_dict = model_dicts[animal]
        ##### Data holder for either mean of individual neurons or population R2 ####
        ### Will be normalized by baseline if that setting is selected ######
        R2S = dict()

        ####### Holders for LME if not doing population R2 ########
        D = []; # Days; 
        Mod = []; # Model Number 

        ######### R2 for each neuron if not doing population R2  #######
        R2s = dict()
        R2s_task_spec = dict()

        ####### Iterate through each day #########
        for i_d in range(ndays[animal]):

            pooled_stats[animal, i_d] = []
        
            ###### Get the decoders ###########
            KG = util_fcns.get_decoder(animal, i_d)
            
            ###### True data #####
            pred_Y = dict()

            ##### Get neural activity conditioned on action ####
            pred_Y['cond'] = cond_act_on_psh(animal, i_d)

            ######### General dynamics conditioned on action #########
            pred_Y['dyn_gen'] =  get_dyn(model_dict[i_d, model_name])

            ######### Get task specific #########
            if fig4_opt:
                pred_Y['dyn_gen_by_tsk'] = []
                pred_Y['dyn_tsk_by_tsk'] = []

                for tsk in range(2):
                    ix_tsk = np.nonzero(model_dict[i_d, 'task'] == tsk)[0]
                    pred_Y['dyn_gen_by_tsk'].append(get_dyn(model_dict[i_d, model_name][ix_tsk]))
                    pred_Y['dyn_tsk_by_tsk'].append(get_dyn(model_dict[i_d, model_name][ix_tsk],col=tsk))

            else:
                tsk_spec = np.zeros_like(get_dyn(model_dict[i_d, model_name]))
                for tsk in range(2):
                    ix_tsk = np.nonzero(model_dict[i_d, 'task'] == tsk)[0]
                    tsk_spec[ix_tsk, :] = get_dyn(model_dict[i_d, model_name][ix_tsk], col=tsk)
                pred_Y['dyn_tsk'] = tsk_spec.copy()

            ######### Get shuffled ###############
            if skip_shuffle:
                pass
            else:
                t00 = time.time()
                pred_Y['shuffled'] = get_shuffled_data_v2(animal, i_d, model_name, nshuffs = nshuffs)
                print('Time post shuffles %.5f' %(time.time() - t00))

            ##########################################################
            ####### Plot individual day vs. shuffled distribution ####
            ##########################################################
            if animal == 'grom' and i_d == 0:   
                ### Plot both ####
                shuff_vs_gen_dyn_plot(pred_Y, model_dict[i_d, 'spks'], i_d, animal, model_name, by_neuron = True,
                    perc_increase = perc_increase)
                
            ###########################################################################
            ####### For each day plot fraction of significant neurons > shuffle #######
            ###########################################################################
            frac_dayi, r2_true_dayi, r2_shuff_dayi = shuff_vs_gen_frac_sig(pred_Y, model_dict[i_d, 'spks'], i_d, animal, model_name, 
                ax_frac_sig, ax_gte_shuff, ax_r2)

            pooled_stats[animal, i_d].append([r2_true_dayi, r2_shuff_dayi])
            frac_sig[animal].append(frac_dayi)


            ######## Get zero order hold #########
            try:
                pred_Y['identity_dyn'] = get_dyn(model_dict[i_d, 'identity_dyn'])
            except:
                pass

            ######## PLotting #########
            for i_mod, (mod, xl) in enumerate(zip(xkeys, xlab)):
                
                ###### Predicted data, cross validated ####
                if plot_act:
                    tdata = model_dict[i_d, 'np']
                    if mod == 'shuffled':
                        if skip_shuffle:
                            pass
                        else:
                            pdata = []
                            for ns in range(pred_Y['shuffled'].shape[2]):
                                pdata.append(np.dot(KG, pred_Y['shuffled'][:, :, ns].T).T)
                            pdata = np.dstack((pdata))
                            assert(tdata.shape[0] == pdata.shape[0])  
                            assert(tdata.shape[1] == pdata.shape[1])  
                    else:
                        pdata = np.dot(KG, pred_Y[mod].T).T
                        assert(tdata.shape == pdata.shape)       
                
                else:
                    tdata = model_dict[i_d, 'spks']
                    pdata = pred_Y[mod]

                ### Get population R2 ### 
                if mod != 'shuffled':
                    if fig4_opt and 'tsk' in mod: 
                        R2 = []
                        for tsk in range(2):
                            ix_tsk = np.nonzero(model_dict[i_d, 'task'] == tsk)[0]
                            r2 = util_fcns.get_R2(tdata[ix_tsk, :], pred_Y[mod][tsk])
                            R2.append(r2)
                    else:
                        R2 = util_fcns.get_R2(tdata, pdata, pop = r2_pop)
                        if fig4_opt:
                            R2 = [R2]

                else:
                    if skip_shuffle:
                        pass
                    else:
                        R2 = []
                        for shuffi in range(pdata.shape[2]):
                            R2.append(util_fcns.get_R2(tdata, pdata[:, :, shuffi], pop = r2_pop))
                        R2 = np.hstack((R2))
                        R2mn = np.mean(R2)
                
                if fig4_opt:
                    if perc_increase: 
                        R2s[i_d, mod] = (np.hstack((R2)) - R2mn) / R2mn
                    else:
                        R2s[i_d, mod] = np.hstack((R2))

                if skip_shuffle and mod == 'shuffled':
                    pass
                else:
                    if perc_increase:
                        R2s[i_d, mod] = (R2 - R2mn)/R2mn
                    else:
                        R2s[i_d, mod] = R2

        ### Bar plot w/ distribution at beginning ###
        if skip_shuffle:
            pass
        else:
            R2_shuff = [R2s[i_d, 'shuffled'] for i_d in range(ndays[animal])]
            util_fcns.draw_plot(0 + ax_offset, np.hstack((R2_shuff)), models_colors[0], np.array([1., 1., 1., 0.]),
                ax, width = 0.9)
    
        #### Print percent of days > shuffle for general dynamics #####

        ###### Mean R2 values to be aggregated here; 
        day_r2 = np.zeros((ndays[animal], len(xkeys)))

        if fig4_opt:
            day_r2 = np.zeros((ndays[animal], len(xkeys), 2))
        
        ###### Go through all the xkey possibilities ####
        for i_mod, mod in enumerate(xkeys):

            #### Don't plot shuffle, already did that 
            if mod == 'shuffled':
                if skip_shuffle:
                    pass
                else:
                    print('Animal %s' %(animal))
                    print('Avg R2 of Shuffle %.4f' %(np.mean(np.hstack((R2_shuff)))))

            else:
            ####### don't plot identity, looks terrible 
                if mod == 'identity_dyn':
                    r2_tmp = [R2s[i_d, mod] for i_d in range(ndays[animal])]
                    print('Animal %s' %(animal))
                    print('Avg. R2 of Identity dynamics %.4f' %(np.mean(r2_tmp)))

            ###### dont' plot task if requested to skip it #####
                elif skip_task_spec and mod == 'dyn_tsk':
                    pass

            ###### Plot mean ####
                else:
                    r2_tmp = [R2s[i_d, mod] for i_d in range(ndays[animal])]
                    
                    #### Skip the bar plot ####
                    ax.bar(i_mod + ax_offset, np.mean(r2_tmp), width=.9, color = models_colors[i_mod])
                    ax.errorbar(i_mod + ax_offset, np.mean(r2_tmp), np.std(r2_tmp), marker = '|', color = 'k')
                    ax.plot(i_d, R2s[i_d, mod])

            #### For each day ####
            for i_d in range(ndays[animal]):
                if mod == 'shuffled':
                    if skip_shuffle:
                        pass
                    else:
                        day_r2[i_d, i_mod] = np.mean(R2s[i_d, mod])
                elif fig4_opt and 'by_tsk' in mod:
                    day_r2[i_d, i_mod, :] = R2s[i_d, mod]
                else:
                    day_r2[i_d, i_mod] = R2s[i_d, mod]

        ##### Plot the individual lines for each day #####
        shuff_vs_dyn = np.zeros((ndays[animal]))
        
        if skip_task_spec:

            ax.set_xticks(np.arange(len(xlab)))
            ax.set_xticklabels(xlab[:], rotation = 45)
            lme_shuff_vs_gen = dict(day = [], shuf_vs_gen = [], r2 = [])

            for i_d in range(ndays[animal]):
                ax.plot(np.arange(len(xkeys)) + ax_offset, day_r2[i_d, :], '-', color='gray', linewidth=1.)
                shuff_vs_dyn[i_d] = get_perc(R2s[i_d, 'shuffled'], R2s[i_d, 'dyn_gen'])
                lme_shuff_vs_gen['day'].append([i_d, i_d])
                lme_shuff_vs_gen['shuf_vs_gen'].append([0, 1])                
                assert(xkeys[0] == 'shuffled')
                assert(xkeys[1] == 'dyn_gen')
                assert(len(day_r2.shape) == 2)
                lme_shuff_vs_gen['r2'].append(np.squeeze(day_r2[i_d, [0, 1]]))
        
            print('Shuffle vs. General ! Animal %s' %(animal))
            pv1, slp1 = util_fcns.run_LME(lme_shuff_vs_gen['day'], lme_shuff_vs_gen['shuf_vs_gen'],
                lme_shuff_vs_gen['r2'])
            pv1_str = util_fcns.get_pv_str(pv1)
            
            ########### LME for shuff vs. general #######
            mx = np.max(day_r2[:, [0, 1]])
            ax.plot(np.array([0, 1]) + ax_offset, [1.1*mx, 1.1*mx], 'k-')
            ax.text(.5 + ax_offset, 1.15*mx, pv1_str, ha='center')

            print('Individual Day Shuffle vs. Generla Animal %s' %(animal))
            for itmpday, itmp in enumerate(shuff_vs_dyn):
                print('Day %d, perc shuff gte data = %.5f' %(itmpday, itmp))

        elif skip_shuffle:
            lme_gen_vs_tsk = dict(day=[], gen_vs_tsk=[], r2 = [])
            ax.set_xticks(np.arange(1, len(xlab)-1))
            ax.set_xticklabels(xlab[1:-1], rotation = 45)
            for i_d in range(ndays[animal]):
                ax.plot(np.arange(1, len(xkeys)-1) + ax_offset, day_r2[i_d, 1:-1], '-', color='gray', linewidth=1.)
                lme_gen_vs_tsk['day'].append([i_d, i_d])
                lme_gen_vs_tsk['gen_vs_tsk'].append([0, 1])
                lme_gen_vs_tsk['r2'].append([R2s[i_d, 'dyn_gen'], R2s[i_d, 'dyn_tsk']])                
            
            ###### Run LME for tsk spec vs. general #####
            pv0, slp0 = util_fcns.run_LME(lme_gen_vs_tsk['day'], lme_gen_vs_tsk['gen_vs_tsk'],
                lme_gen_vs_tsk['r2'])
            pv0_str = util_fcns.get_pv_str(pv0)
            mx = np.max(day_r2[:, [1, 2]])
            ax.plot(np.array([1, 2]) + ax_offset, [1.1*mx, 1.1*mx], 'k-')
            ax.text(1.5 + ax_offset, 1.15*mx, pv0_str, ha='center')

        ##### Most general #######
        else:
            lme_gen_vs_tsk = dict(day=[], gen_vs_tsk=[], r2 = [])
            lme_shuff_vs_gen = dict(day = [], shuf_vs_gen = [], r2 = [])

            for i_d in range(ndays[animal]):
            
                if fig4_opt:
                    ax.plot(np.array([0, 1]) + ax_offset, day_r2[i_d, [0, 1]], '-', color='gray', linewidth = 1.)
                    lme_shuff_vs_gen['day'].append([i_d, i_d])
                    lme_shuff_vs_gen['shuf_vs_gen'].append([0, 1])
                    lme_shuff_vs_gen['r2'].append(np.squeeze(day_r2[i_d, [0, 1], 0]))

                    ### Centerout and obstacle day line #####
                    ax.plot(np.array([2, 3]) + ax_offset, day_r2[i_d, [2, 3], 0], '-', color='gray', linewidth = 1.)
                    ax.plot(np.array([2, 3]) + ax_offset, day_r2[i_d, [2, 3], 1], '-', color='gray', linewidth = 1.)

                    lme_gen_vs_tsk['day'].append([i_d, i_d])
                    lme_gen_vs_tsk['gen_vs_tsk'].append([0, 1])
                    lme_gen_vs_tsk['r2'].append(day_r2[i_d, [2, 3], 0])
                    
                    lme_gen_vs_tsk['day'].append([i_d, i_d])
                    lme_gen_vs_tsk['gen_vs_tsk'].append([0, 1])
                    lme_gen_vs_tsk['r2'].append(day_r2[i_d, [2, 3], 1])
                    
                    ### Get something about R2s for shuffled vs. dyn gen ####
                    shuff_vs_dyn[i_d] = get_perc(R2s[i_d, 'shuffled'], R2s[i_d, 'dyn_gen'])

                else:

                    ax.plot(np.arange(len(xkeys)-1) + ax_offset, day_r2[i_d, :-1], '-', color='gray', linewidth=1.)

                    #### Compare model 1 to shuffle #####
                    shuff_vs_dyn[i_d] = get_perc(R2s[i_d, 'shuffled'], R2s[i_d, 'dyn_gen'])

                    #### Compare model 1 to model 2 #####
                    lme_gen_vs_tsk['day'].append([i_d, i_d])
                    lme_gen_vs_tsk['gen_vs_tsk'].append([0, 1])
                    lme_gen_vs_tsk['r2'].append([R2s[i_d, 'dyn_gen'], R2s[i_d, 'dyn_tsk']])

                    lme_shuff_vs_gen['day'].append([i_d, i_d])
                    lme_shuff_vs_gen['shuf_vs_gen'].append([0, 1])
                    assert(xkeys[0] == 'shuffled')
                    assert(xkeys[1] == 'dyn_gen')
                    assert(len(day_r2.shape) == 2)
                    lme_shuff_vs_gen['r2'].append(np.squeeze(day_r2[i_d, [0, 1]]))


            ###### Run LME for tsk spec vs. general #####
            print('General vs. Task Specific! Animal %s' %(animal))
            pv0, slp0 = util_fcns.run_LME(lme_gen_vs_tsk['day'], lme_gen_vs_tsk['gen_vs_tsk'],
                lme_gen_vs_tsk['r2'])
            pv0_str = util_fcns.get_pv_str(pv0)

            print('Shuffle vs. General ! Animal %s' %(animal))
            pv1, slp1 = util_fcns.run_LME(lme_shuff_vs_gen['day'], lme_shuff_vs_gen['shuf_vs_gen'],
                lme_shuff_vs_gen['r2'])
            pv1_str = util_fcns.get_pv_str(pv1)
            
            ########### LME for shuff vs. general #######
            mx = np.max(day_r2[:, [0, 1]])
            ax.plot(np.array([0, 1]) + ax_offset, [1.1*mx, 1.1*mx], 'k-')
            ax.text(.5 + ax_offset, 1.15*mx, pv1_str, ha='center')

            print('Individual Day Shuffle vs. Generla Animal %s' %(animal))
            for itmpday, itmp in enumerate(shuff_vs_dyn):
                print('Day %d, perc shuff gte data = %.5f' %(itmpday, itmp))

            if fig4_opt:
                mx = np.max(day_r2[:, [2, 3]])
                ax.plot(np.array([2, 3]) + ax_offset, [1.1*mx, 1.1*mx], 'k-')
                ax.text(2.5 + ax_offset, 1.15*mx, pv0_str, ha='center')
                ax.set_xticks(np.arange(len(xlab)-1))
                ax.set_xticklabels(xlab[:-1], rotation = 45)

            else:
                if skip_task_spec:
                    pass
                else:
                    mx = np.max(day_r2[:, [1, 2]])
                    ax.plot(np.array([1, 2]) + ax_offset, [1.1*mx, 1.1*mx], 'k-')
                    ax.text(1.5 + ax_offset, 1.15*mx, pv0_str, ha='center')
                    ax.set_xticks(np.arange(len(xlab)-1))
                    ax.set_xticklabels(xlab[:-1], rotation = 45)
                
        #### Plot stars ####
        if skip_shuffle:
            ax.set_xlim([.5, 2.6 + ax_offset])
        else:
            if fig4_opt:
                ax.set_xlim([-.6, 3.6])
            else:
                if np.all(shuff_vs_dyn == 1.):
                    mx = np.max(day_r2[:, [0, 1]])
                    ax.plot(np.array([0, 1]) + ax_offset, [1.1*mx, 1.1*mx], 'k-')
                    ax.text(.5 + ax_offset, 1.15*mx, '***', ha='center')
                ax.set_xlim([-.6, 2.6 + ax_offset])
        if perc_increase:
            ax.set_ylabel('Frac. Increase in R2\nabove Shuffle Mean')
        else:
            ax.set_ylabel('R2')
        
        f.tight_layout()
        f.savefig(analysis_config.config['fig_dir']+'%s_fig4_dynR2_percIncrease%s_condAct%s_pltAct%s_skipShuff%s.svg'%(animal, 
            str(perc_increase), 
            str(cond_on_act), 
            str(plot_act),
            str(skip_shuffle)))
    
    ax_frac_sig.set_xlim([-1, 2])
    for ia, animal in enumerate(['grom', 'jeev']): 

        ax_frac_sig.bar(ia, 
            np.mean(frac_sig[animal]), width=0.8, color='k', alpha=.2)
    ax_frac_sig.set_xticks([0, 1])
    ax_frac_sig.set_xticklabels(['G', 'J'])


    ax_gte_shuff.set_xlim([-1, 14])
    ax_r2.set_ylabel('$R^2$')
    ax_r2.set_xlim([-1, 14])

    fax_frac_sig.tight_layout()
    util_fcns.savefig(fax_frac_sig, 'frac_neur_sig_gt_shuff_n%d.svg' %(nshuffs))

    fax_gte_shuff.tight_layout()
    fax_r2.tight_layout()
    util_fcns.savefig(fax_r2, 'ax_r2_dyn_cond_act_shuff_n%d.svg' %(nshuffs))
    return pooled_stats

def cond_act_on_psh(animal, i_d, KG=None, dat=None):
    """
    estimate neural activity y_t | a_t "most likely" 
    
    Parameters
    ----------
    animal : str
        animal name 
    i_d : int 
        day 
    """

    ### Load the data from elements; 
    if dat is None:
        dat = pickle.load(open(analysis_config.config['shuff_fig_dir']+'%s_%d_shuff_ix.pkl' %(animal, i_d), 'rb'))
    
        ### Get push / spks; 
        tm0ix, _ = generate_models.get_temp_spks_ix(dat['Data'])

    else:
        print('in testing mode')
        tm0ix = np.arange(dat['Data']['spks'].shape[0])

    if KG is None: 
        KG = util_fcns.get_decoder(animal, i_d)


    spks = dat['Data']['spks'][tm0ix, :]
    if dat['Data']['push'].shape[1] == 7:
        act  = dat['Data']['push'][np.ix_(tm0ix, [3, 5])]
    else:
        act = dat['Data']['push'][tm0ix, :]

    ### Estiamte covariance; 
    mu = np.mean(spks, axis=0)
    cov = np.cov(spks.T)

    cov12 = np.dot(KG, cov).T
    cov21 = np.dot(KG, cov)
    cov22 = np.dot(KG, np.dot(cov, KG.T))
    cov22I = np.linalg.inv(cov22)

    MU = np.zeros_like(spks)
    MU[:] = mu 
    assert(np.all(np.diff(MU, axis=0) == 0.))

    mu2_i = np.dot(KG, MU.T).T
    pred_w_cond = MU + np.dot(cov12, np.dot(cov22I, (act - mu2_i).T)).T
    assert(np.allclose(np.dot(KG, pred_w_cond.T).T, act))
    return pred_w_cond

def test_cond_act_on_psh(): 
    """
    Method to test conditioning method above
    Plot: 
        green --> decoder line 
        black dots --> data
        red dots --> E(y_t | a_t)
        lines --> connect black dots/red dots; (should be perp. to green)
    """
    spks = np.random.multivariate_normal(np.array([0, 0]),np.array([[1, .8], [.4, 1]]),200)
    KG = np.array([[0, 1]])
    act = np.dot(KG, spks.T).T

    f, ax = plt.subplots()
    ax.axis('square')
    ax.plot(spks[:, 0], spks[:, 1], 'k.')

    dat = dict()
    dat['Data'] = dict(spks=spks, push=act)
    pred = cond_act_on_psh(None, None, KG=KG, dat=dat)
    ax.plot(pred[:, 0], pred[:, 1], 'r.')
    for t in range(pred.shape[0]):
        ax.plot([spks[t, 0], pred[t, 0]], [spks[t,1], pred[t, 1]], 'k-', linewidth=.5)

    ax.plot([0, KG[0, 0]], [0, KG[0, 1]], 'g--')
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])

def shuff_vs_gen_dyn_plot(pred_Y, true_Y, i_d, animal, model_name, by_neuron = False, 
    perc_increase = False, neur_ix_list = [35, 36, 37, 38, 39, 40, 41, 42]):
    
    ###### Get R2 ########
    n_shuff = pred_Y['shuffled'].shape[2]
    assert(pred_Y['shuffled'].shape[0] == true_Y.shape[0] == pred_Y['dyn_gen'].shape[0])
    assert(pred_Y['shuffled'].shape[1] == true_Y.shape[1] == pred_Y['dyn_gen'].shape[1])

    if by_neuron:
        nneur = true_Y.shape[1] 
        r2_shuff = np.zeros((n_shuff, nneur))
        r2_true = np.zeros((nneur))

    else:
        f, ax = plt.subplots()
        r2_shuff = np.zeros((n_shuff, ))
        r2_true = 0.

    for i_shuff in range(pred_Y['shuffled'].shape[2]):
        if by_neuron:
            for i_n in range(nneur):
                r2_shuff[i_shuff, i_n] = util_fcns.get_R2(true_Y[:, i_n], pred_Y['shuffled'][:, i_n, i_shuff])
        else:
            r2_shuff[i_shuff] = util_fcns.get_R2(true_Y, pred_Y['shuffled'][:, :, i_shuff])

    if by_neuron:
        for i_n in range(nneur):
            r2_true[i_n] = util_fcns.get_R2(true_Y[:, i_n], pred_Y['dyn_gen'][:, i_n])
    else:
        r2_true = util_fcns.get_R2(true_Y, pred_Y['dyn_gen'])

    ####### Make plots ########
    if by_neuron:
        for neur_ix in neur_ix_list:

            ################ Plots ####################
            f, ax = plt.subplots(figsize = (4, 4)) #ncols = 4, nrows = 11, figsize = (10, 22))
            ### Plot distribution in boxplot form: 
            r2_shuff_i = r2_shuff[:, neur_ix]
            r2_true_i = r2_true[neur_ix]

            ### Plot the raw values on left y axis; 
            util_fcns.draw_plot(0, r2_shuff_i, 'k', 'w', ax, width = 1)
            ax.plot(0, r2_true_i, '.', markersize=10, color = np.array([39, 169, 225])/255.)

            ### Plot the percent increase on the right y axis: 
            ylim = np.array(ax.get_ylim())

            ### Make a second y axis: 
            ax2 = ax.twinx()
            mean_r2_i = np.mean(r2_shuff_i)
            if perc_increase:
                ax2.set_ylim((ylim - mean_r2_i)/np.abs(mean_r2_i))
                ax2.set_ylabel('Frac. > Shuf-Mn', rotation=270, va='bottom')
            else:
                ax2.set_ylim(ylim - mean_r2_i)
                ax2.set_ylabel('R2 > Shuf-Mn', rotation=270, va='bottom')

            ax.set_ylabel('$R^2$')
            ax.set_title('n%d' %(neur_ix))
            ax.set_xlim([-1, 1])
            f.tight_layout()

    else:
        f, ax = plt.subplots()

        if perc_increase:
            mean_r2 = np.mean(r2_shuff)
            ax.hist((r2_shuff - mean_r2) / mean_r2)
            ax.vlines((r2_true - mean_r2) / mean_r2, 0., 50, 'k')
            ax.set_xlabel('Perc Increase gt mean shuff')
        else:
            ax.hist(r2_shuff)
            ax.vlines(r2_true, 0., 100, 'k')
            ax.set_xlabel('Population R2')

        ax.set_title('%s, Day%d, mod%s'%(animal, i_d, model_name))

def shuff_vs_gen_frac_sig(pred_Y, true_Y, i_d, animal, model_name, 
    ax_frac_sig, ax_gte_shuff, ax_r2, frac_gte_flag = False):
    """
    plot the fraction of sig neurons for this day; 
        input: frac_gte_flag: 
            if True: then plots teh fraction greater than shuffle 
            if False: plots teh difference in R2 compared to shuffle mean 
    """
    ###### Get R2 ########
    n_shuff = pred_Y['shuffled'].shape[2]
    assert(pred_Y['shuffled'].shape[0] == true_Y.shape[0] == pred_Y['dyn_gen'].shape[0])
    assert(pred_Y['shuffled'].shape[1] == true_Y.shape[1] == pred_Y['dyn_gen'].shape[1])

    nneur = true_Y.shape[1] 
    r2_shuff = np.zeros((n_shuff, nneur))
    r2_true = np.zeros((nneur))

    cnt_sig = 0; 
    cnt_tot = 0; 

    frac_gte_shuffle = []

    for i_n in range(nneur):
        for i_shuff in range(n_shuff):    
            r2_shuff[i_shuff, i_n] = util_fcns.get_R2(true_Y[:, i_n], pred_Y['shuffled'][:, i_n, i_shuff])
            r2_true[i_n] = util_fcns.get_R2(true_Y[:, i_n], pred_Y['dyn_gen'][:, i_n])

        pv = float(len(np.nonzero(r2_shuff[:, i_n] >= r2_true[i_n])[0])) / float(n_shuff)
        if pv < 0.05:
            cnt_sig += 1

            ### Add to frac > shuffle ### 
            mn = np.mean(r2_shuff[:, i_n])
            if frac_gte_flag:
                frac_gte = (r2_true[i_n] - mn)/mn
            else:
                frac_gte = r2_true[i_n] - mn 

            if np.isnan(frac_gte):
                pass
            else:
                frac_gte_shuffle.append(frac_gte)

        cnt_tot += 1

    KG = util_fcns.get_decoder(animal, i_d)
    r2_shuff_pop = np.zeros((n_shuff))
    for i_shuff in range(n_shuff):
        r2_shuff_pop[i_shuff] = util_fcns.get_R2(true_Y, pred_Y['shuffled'][:, :, i_shuff])
        assert(np.allclose(np.dot(KG, pred_Y['shuffled'][:, :, i_shuff].T).T, np.dot(KG, pred_Y['dyn_gen'].T).T))

    assert(np.allclose(np.dot(KG, pred_Y['cond'].T).T, np.dot(KG, pred_Y['dyn_gen'].T).T))

    if animal == 'grom':
        assert(np.allclose(np.dot(KG, pred_Y['cond'].T).T, np.dot(KG, true_Y.T).T))
    elif animal == 'jeev':
        assert(generate_models_utils.quick_reg(np.dot(KG, pred_Y['cond'].T).T, np.dot(KG, true_Y.T).T) > .99) 
    
    r2_true_pop = util_fcns.get_R2(true_Y, pred_Y['dyn_gen'])
    r2_true_cond = util_fcns.get_R2(true_Y, pred_Y['cond'])

    if animal == 'grom':
        i_d_plt = 0 
        i_d_plt2 = i_d 
    elif animal == 'jeev':
        i_d_plt = 1 
        i_d_plt2 = 10 + i_d

    ######## Plot dot for fraction of neurons sig > shuffle 
    ax_frac_sig.plot(i_d_plt, float(cnt_sig)/float(cnt_tot), 'k.')

    ######## "Effect size": distribution of frac > shuffle for significant neurons 
    util_fcns.draw_plot(i_d_plt2, frac_gte_shuffle, 'k', 'w', ax_gte_shuff)

    ######## Plot distribution of cond + shuffle distribution of R2 + true data #####
    util_fcns.draw_plot(i_d_plt2 - .25, r2_shuff_pop, 'k', 'w', ax_r2)
    ax_r2.plot(i_d_plt2, r2_true_pop, '.', markersize = 20, color=np.array([39, 169, 225])/256.)
    ax_r2.plot(i_d_plt2 + .125, r2_true_cond, '^', color='gray')
    ax_r2.vlines(i_d_plt2, r2_true_cond, r2_true_pop, 'gray', linewidth=.5)

    ######## Aesthetics ########
    ######## Set Ylabel 
    ax_frac_sig.set_ylabel('Frac Neurons Pred. Sig > Shuff', fontsize=14)

    if frac_gte_flag:
        ax_gte_shuff.set_ylabel('Frac. > Shuf-Mn (sig neur)', fontsize=14)
    else:
        ax_gte_shuff.set_ylabel('R2 > Shuf-Mn (sig neur)', fontsize=14)

    return float(cnt_sig)/float(cnt_tot), r2_true_pop, r2_shuff_pop

def get_shuffled_data_v2_super_stream(animal, day_ix, shuffle_num):
    pref = analysis_config.config['shuff_fig_dir']

    dat = sio.loadmat(os.path.join(pref, '%s_%s_predY_wc_shuff%d.mat'%(animal, day_ix, shuffle_num)))
    return dat['pred']

def get_shuffled_data_v2_super_stream_nocond(animal, day_ix, shuffle_num):
    pref = analysis_config.config['shuff_fig_dir']

    dat = sio.loadmat(os.path.join(pref, '%s_%s_predY_no_cond_shuff%d.mat'%(animal, day_ix, shuffle_num)))
    return dat['pred']

def get_shuffled_data_v2_streamlined_wc(animal, day, spks_tm1, push_tm0, tm0ix, test_ix, shuffle_num,
    KG, former_shuff_ix, now_shuff_ix, t0):
    """Summary
    
    Parameters
    ----------
    animal : string
        'grom' or 'jeev'
    day : int
        session index 
    spks_tm1 : np.array
        full_spks[tm1, :] from non-streamlined fcn
    push_tm0 : np.array
        full_push[tm0, :] from non-streamlined fcn
    tm0ix : np.array
        the actual tm0 indices used to figure otu the right train/test data 
    temp_N : int
        the length of tm0ix
    test_ix : list with 5 entries, each an np.array of the original test indices 
    shuffle_num : shuffle number -- used to indicate which weights to open 
    KG : KalmanGain for this day 
    former_shuff_ix : shuffle indices that were used to make train/test data 
    now_shuff_ix : np.arange(temp_N) --> full data length
    
    Returns
    -------
    TYPE
        Description
    
    Deleted Parameters
    ------------------
    data_temp_dict
        Description
    """
    ### Get model variables 
    #### setup pred_Y to be accurate 
    nneur = KG.shape[1]
    
    ##### This is correct -- but we did everything wrong ###
    #pred_Y = np.zeros((len(tm0ix), nneur))
    pred_Y = np.zeros((len(now_shuff_ix), nneur))

    #### Load the coefficients 
    pref = analysis_config.config['shuff_fig_dir']
    shuff_str = str(shuffle_num)
    shuff_str = shuff_str.zfill(3)
    model_file = sio.loadmat(pref + '%s_%d_shuff%s_%s_models.mat' %(animal, day, shuff_str, 'hist_1pos_0psh_2spksm_1_spksp_0'))

    ##### Deal with the shuffling indices ###
    shuff_x = []
    current_inds = tm0ix.copy()
    slope = float(len(current_inds))/float(len(now_shuff_ix))
    
    ### map b/w actual_test_ix and current inds? 
    actual_test_ix = now_shuff_ix[former_shuff_ix[tm0ix]]
    t2 = time.time()
    actual_test2current_inds = get_ati2inds_map(actual_test_ix, current_inds, slope)
    
    #### For each data fold #####
    for i_fold in range(5): 
        
        t1 = time.time()
        ### Test indices AFTER shuff_ix[tm0ix]

        ### Whta these test inds are on the global scheme: 
        actual_test_ix_fold = list(actual_test_ix[test_ix[i_fold]])

        ### Need to get the location of these in our current scheme: 
        test_ix_fold_current = np.array([actual_test2current_inds[x] for x in actual_test_ix_fold])
        
        if i_fold < 4:
            shuff_x.append(np.hstack((test_ix_fold_current)))

        elif i_fold == 4:
            ### Get stuff in our current list that wasn't in any of the test indices 
            tot = np.hstack(( np.hstack((shuff_x)), np.hstack((test_ix_fold_current))))
            remaining = np.array([i for i, j in enumerate(current_inds) if i not in tot])
            test_ix_fold_current = np.hstack((remaining, np.hstack((test_ix_fold_current))))
            shuff_x.append(test_ix_fold_current)

            assert(len(np.hstack((shuff_x))) == len(tm0ix))
            assert(len(np.unique(np.hstack((shuff_x)))) == len(tm0ix))

    for i_fold in range(5):
        coef = model_file[str((i_fold, 'coef_'))]
        intc = model_file[str((i_fold, 'intc_'))]
        assert(int(model_file['shuff_num']) == shuffle_num)

        #### Get the updated shuffle indices ####
        test_ix_fold_current = shuff_x[i_fold]

        #### Get W ###
        W = model_file[str((i_fold, 'W'))]

        ### Pre-assemble these to avoid the stupid re-assembly in pred_w_cond ####
        kwargs = dict(A_preassembled = push_tm0[np.ix_(test_ix_fold_current, [3, 5])], 
                      X_preassembled = spks_tm1[test_ix_fold_current, :])

        ### Make the predictions 
        pred_wc = pred_w_cond(coef, intc, None,
            None, None, nneur, w_ = W, KG = KG, 
            **kwargs) 

        pred_Y[test_ix_fold_current, :] = pred_wc.copy()

        #### Make sure the push actuall matches intended push: #### 
        assert(np.allclose(np.dot(KG, pred_wc.T).T, push_tm0[np.ix_(test_ix_fold_current, [3, 5])]))
        assert(np.allclose(np.dot(KG, pred_Y[test_ix_fold_current, :].T).T, push_tm0[np.ix_(test_ix_fold_current, [3, 5])]))

    if np.all(pred_Y == 0):
        import pdb; pdb.set_trace()
    return pred_Y

def get_shuffled_data_v2_streamlined_no_cond(animal, day, spks_tm1, tm0ix, test_ix, shuffle_num,
    KG, former_shuff_ix, now_shuff_ix, t0):
    """Summary
    
    Parameters
    ----------
    animal : string
        'grom' or 'jeev'
    day : int
        session index 
    spks_tm1 : np.array
        full_spks[tm1, :] from non-streamlined fcn
    tm0ix : np.array
        the actual tm0 indices used to figure otu the right train/test data 
    temp_N : int
        the length of tm0ix
    test_ix : list with 5 entries, each an np.array of the original test indices 
    shuffle_num : shuffle number -- used to indicate which weights to open 
    KG : KalmanGain for this day 
    former_shuff_ix : shuffle indices that were used to make train/test data 
    now_shuff_ix : np.arange(temp_N) --> full data length
    
    Returns
    -------
    TYPE
        Description
    
    Deleted Parameters
    ------------------
    data_temp_dict
        Description
    """
    ### Get model variables 
    #### setup pred_Y to be accurate 
    nneur = KG.shape[1]
    
    ##### This is correct
    pred_Y = np.zeros((len(tm0ix), nneur))
    
    #### Load the coefficients 
    pref = analysis_config.config['shuff_fig_dir']
    shuff_str = str(shuffle_num)
    shuff_str = shuff_str.zfill(3)
    model_file = sio.loadmat(pref + '%s_%d_shuff%s_%s_models.mat' %(animal, day, shuff_str, 'hist_1pos_0psh_0spksm_1_spksp_0'))

    ##### Deal with the shuffling indices ###
    shuff_x = []
    current_inds = tm0ix.copy()
    slope = float(len(current_inds))/float(len(now_shuff_ix))
    
    ### map b/w actual_test_ix and current inds? 
    actual_test_ix = now_shuff_ix[former_shuff_ix[tm0ix]]
    t2 = time.time()
    actual_test2current_inds = get_ati2inds_map(actual_test_ix, current_inds, slope)
    
    #### For each data fold #####
    for i_fold in range(5): 
        
        t1 = time.time()
        ### Test indices AFTER shuff_ix[tm0ix]

        ### Whta these test inds are on the global scheme: 
        actual_test_ix_fold = list(actual_test_ix[test_ix[i_fold]])

        ### Need to get the location of these in our current scheme: 
        test_ix_fold_current = np.array([actual_test2current_inds[x] for x in actual_test_ix_fold])
        
        if i_fold < 4:
            shuff_x.append(np.hstack((test_ix_fold_current)))

        elif i_fold == 4:
            ### Get stuff in our current list that wasn't in any of the test indices 
            tot = np.hstack(( np.hstack((shuff_x)), np.hstack((test_ix_fold_current))))
            remaining = np.array([i for i, j in enumerate(current_inds) if i not in tot])
            test_ix_fold_current = np.hstack((remaining, np.hstack((test_ix_fold_current))))
            shuff_x.append(test_ix_fold_current)

            assert(len(np.hstack((shuff_x))) == len(tm0ix))
            assert(len(np.unique(np.hstack((shuff_x)))) == len(tm0ix))

    for i_fold in range(5):
        coef = model_file[str((i_fold, 'coef_'))]
        intc = model_file[str((i_fold, 'intc_'))]
        assert(int(model_file['shuff_num']) == shuffle_num)

        #### Get the updated shuffle indices ####
        test_ix_fold_current = shuff_x[i_fold]

        ### Pre-assemble these to avoid the stupid re-assembly in pred_w_cond ####
        kwargs = dict(X_preassembled = spks_tm1[test_ix_fold_current, :])

        ### Make the predictions 
        pred_wc = pred_wo_cond(coef, intc, None,
            None, None, nneur, KG = KG, **kwargs) 

        pred_Y[test_ix_fold_current, :] = pred_wc.copy()

        #### Make sure the push actuall matches intended push: #### 
        
    if np.all(pred_Y == 0):
        import pdb; pdb.set_trace()
    return pred_Y

def get_ati2inds_map(shuff_ix, current_inds, slope):
    CI_dict = {}
    for s in shuff_ix:
        start = int(s*slope) - 10
        search_ix = np.arange(start, int(s*slope) + 10)
        try:
            CI_dict[s] = np.where(current_inds[search_ix] == s)[0]  + start
        except:
            search_ix = search_ix[search_ix >= 0]
            search_ix = search_ix[search_ix < len(current_inds)]
            CI_dict[s] = np.where(current_inds[search_ix] == s)[0]  + search_ix[0]
        if len(CI_dict[s]) > 0: assert(current_inds[int(CI_dict[s])] == s)
    return CI_dict

# shuff_ix = np.random.randint(0, 10, (10))
# current_inds = np.arange(3, 7)
# actual_test2current_inds = get_ati2inds_map(shuff_ix, current_inds)

# shuff_ix_fold = shuff_ix[:5]
# test_ix_fold_current = np.hstack(list(map(lambda x:actual_test2current_inds[x], shuff_ix_fold)))
        
def get_shuffled_data_v2(animal, day, model_name, nshuffs = 10, shuff_num = None, testing_mode = False, 
    mean_maint = False, within_mov = False, test_streamlined = False):
    """
    New method to get shuffled predictions 

    updates --> Jan 8th 2021: now making sure we predict data using the real y_tm1
        -- previously was using shuffle, saving that, training /testing were based on shuffled indices
        -- here, making sure that the actual data is used at each time point: E(y_t | y_{t-1}, A_shuff, b_shuff)
        -- NOT E(y_t | y_shuff_{t-1}, A_shuff, b_shuff)
    """
    #### With intercept #####

    t0 = time.time()
    pref = analysis_config.config['shuff_fig_dir']
           
    if nshuffs is None:
        assert(shuff_num is not None)
        nshuffs = 1
        shuffle_id = [shuff_num]
    else:
        shuffle_id = range(nshuffs)

    #### Open the file #####
    if mean_maint: 
        data_file = pickle.load(open(pref + '%s_%d_shuff_ix_mn_diff_maint.pkl' %(animal, day)))
    elif within_mov:
        data_file = pickle.load(open(pref + '%s_%d_shuff_ix_within_mov_shuff.pkl' %(animal, day)))
    else:
        data_file = pickle.load(open(pref + '%s_%d_shuff_ix.pkl' %(animal, day)))
    
    full_spks = data_file['Data']['spks']
    full_push = data_file['Data']['push']
    full_bin_num = data_file['Data']['bin_num']
    temp_N = data_file['temp_n']

    #### Test indices #####
    test_ix = data_file['test_ix']
    type_of_model = np.zeros((5, ))

    nneur = full_spks.shape[1]

    ### 2 x N decoder 
    KG = util_fcns.get_decoder(animal, day)

    model_var_list, predict_key, include_action_lags, history_bins_max = generate_models_list.get_model_var_list(6)
    models_to_include = [m[1] for m in model_var_list]

    variables_list = generate_models.return_variables_associated_with_model_var(model_var_list, include_action_lags, nneur)
    pred_Y = np.zeros((temp_N, nneur, nshuffs))

    if testing_mode:
        day_mat = analysis_config.data_params['%s_input_type'%animal] 
        order_mat = analysis_config.data_params['%s_ordered_input_type'%animal] 
        test_Data, test_Data_temp, test_Sub_spikes, test_Sub_spk_temp_all, test_Sub_push = generate_models_utils.get_spike_kinematics(animal, day_mat[day], 
                    order_mat[day], 1, within_bin_shuffle = False, day_ix = day, nshuffs = 1)
    
    #### Get teh spike indices ###
    tm0ix, tm1ix = generate_models.get_temp_spks_ix(data_file['Data'])

    ### Shuffle indices: 
    for ishuff, shuffle in enumerate(shuffle_id):
        if shuffle % 100 == 0:
            print('Shuffle %d, %.5f' %(shuffle, time.time() - t0))
        
        ##### Shuffle used to scramble data before fitting the weights
        former_shuff_ix = data_file[shuffle]

        #if testing_mode: 
        ### Keep data in the correct order; 
        ### The effect of the shuffle should be the fitting of the dynamics model 
        now_shuff_ix = np.arange(full_spks.shape[0])

        #### shuffled + subselected ####
        sub_spikes = data_file['Data']['spks'][now_shuff_ix[tm0ix]]
        sub_spikes_tm1 = data_file['Data']['spks'][now_shuff_ix[tm1ix]]

        ### DO NOT shuffle the pushes; 
        sub_push = data_file['Data']['push'][tm0ix]
        
        #sub_spikes, sub_spikes_tm1, sub_push, tm0ix, tm1ix = generate_models.get_temp_spks(data_file['Data'], shuff_ix)
        #print('Time to shuffles and subselect %.5f '%(time.time() - t0))
        if test_streamlined:
            print('start streamline')
            t1 = time.time()
            pred_y_streamlined = get_shuffled_data_v2_streamlined_wc(animal, day, sub_spikes, sub_spikes_tm1, sub_push, 
                tm0ix, test_ix, shuffle, KG, former_shuff_ix, now_shuff_ix)
            print('end streamline %.4f' %(time.time() - t1))

        if testing_mode: 
            assert(np.allclose(sub_spikes, test_Sub_spikes))
            assert(np.allclose(sub_spikes_tm1, test_Sub_spk_temp_all[:, 0, :]))
            assert(np.allclose(sub_push[:, [3, 5]], test_Sub_push))
            
        shuff_str = str(shuffle)
        shuff_str = shuff_str.zfill(3)

        if mean_maint:
            model_file = sio.loadmat(pref + '%s_%d_shuff%s_%s_mn_diff_maint_models.mat' %(animal, day, shuff_str, model_name))
        elif within_mov:
            model_file = sio.loadmat(pref + '%s_%d_shuff%s_%s_within_mov_shuff_models.mat' %(animal, day, shuff_str, model_name))
        else:
            model_file = sio.loadmat(pref + '%s_%d_shuff%s_%s_models.mat' %(animal, day, shuff_str, model_name))
        
        #### Build the data dictionary ####
        data_temp_dict = {}
        for i_n in range(nneur):
            data_temp_dict['spk_tm1_n%d'%i_n] = sub_spikes_tm1[:, i_n]
            data_temp_dict['spk_tm0_n%d'%i_n] = sub_spikes[:, i_n]
        data_temp_dict['pshx_tm0'] = sub_push[:, 3]
        data_temp_dict['pshy_tm0'] = sub_push[:, 5]

        shuff_x = []
        current_inds = now_shuff_ix[tm0ix]
        #### For each data fold #####
        for i_fold in range(5): 
            
            ### Test indices AFTER shuff_ix[tm0ix]
            test_ix_fold = test_ix[i_fold]
            actual_test_ix_fold = now_shuff_ix[former_shuff_ix[tm0ix[test_ix_fold]]]

            test_ix_fold_current = [np.where(current_inds==atif)[0] for atif in actual_test_ix_fold]
            test_ix_fold_current = np.hstack((test_ix_fold_current))
            
            if i_fold < 4:
                shuff_x.append(test_ix_fold_current)

            elif i_fold == 4:
                tot = np.hstack((shuff_x))
                tot = np.hstack((tot, test_ix_fold_current))

                remaining = np.array([i for i,j in enumerate(current_inds) if i not in tot])
                test_ix_fold_current = np.hstack((remaining, test_ix_fold_current))
                shuff_x.append(test_ix_fold_current)

                assert(len(np.hstack((shuff_x))) == len(tm0ix))
                assert(len(np.unique(np.hstack((shuff_x)))) == len(tm0ix))

        for i_fold in range(5):
            coef = model_file[str((i_fold, 'coef_'))]
            intc = model_file[str((i_fold, 'intc_'))]
            assert(int(model_file['shuff_num']) == shuffle)

            test_ix_fold_current = shuff_x[i_fold]

            ### Re-do this now that you're not shuffling; 
            ### Test ix is supposed to apply to data AFTER former_shuff_ix was applied
            if testing_mode:
                coef[:,:] = np.eye(nneur)
                intc[:] = 0.

            #### Get W if conditinoing on action ####
            if 'psh_2' in model_name: 
                W = model_file[str((i_fold, 'W'))]

                if testing_mode:
                    ### No uncertainty/error in dynamics model 
                    W[:,:] = 0.

            for i_m, (model_nm, variables) in enumerate(zip(models_to_include, variables_list)):

                if model_nm == model_name: 
                    if 'psh_2' in model_name: 

                        pred_wc = pred_w_cond(coef, intc, data_temp_dict,
                            test_ix_fold_current, variables, nneur, w_ = W, KG = KG) 

                        pred_Y[test_ix_fold_current, :, ishuff] = pred_wc.copy()

                        if testing_mode:
                            ### Set the "W" equal to zero so wont match actions 
                            ### Want to test for equality to the previous action/spks
                            pass
                        else:
                            #### Make sure the push actuall matches intended push: #### 
                            assert(np.allclose(np.dot(KG, pred_wc.T).T, sub_push[np.ix_(test_ix_fold_current, [3, 5])]))
                            assert(np.allclose(np.dot(KG, pred_Y[test_ix_fold_current, :, ishuff].T).T, sub_push[np.ix_(test_ix_fold_current, [3, 5])]))

                    else:
                        pred_Y[test_ix_fold_current, :, ishuff] = pred_wo_cond(coef, intc, data_temp_dict,
                            test_ix_fold_current, variables, nneur) 
        ### After all folds, if in testing mode, pred_Y should be equal to sub_spikes_tm1; 
        if testing_mode:
            assert(np.allclose(sub_spikes_tm1, pred_Y[:, :, ishuff]))
    if np.all(pred_Y == 0):
        import pdb; pdb.set_trace()

    if test_streamlined:
        assert(np.allclose(pred_y_streamlined, pred_Y[:, :, ishuff]))
    print('Done with big fcn, %.5f' %(time.time() - t0))
    return pred_Y

def get_shuffled_data_pred_null_roll_pot_shuff(animal, day, model_name, nshuffs = 10, testing_mode = False):

    return get_shuffled_data_pred_null_roll(animal, day, model_name, nshuffs=nshuffs, testing_mode=testing_mode,
        pot_shuff = True)

def get_shuffled_data_pred_null_roll(animal, day, model_name, nshuffs = 10, testing_mode = False,
    pot_shuff = False):
    """
    New method to get shuffled predictions 
    """
    #### With intercept #####

    t0 = time.time()
    pref = analysis_config.config['shuff_fig_dir']
           
    #### Open the file #####
    if pot_shuff:
        data_file = pickle.load(open(pref + '%s_%d_shuff_ix_null_roll_pot_beh_maint.pkl' %(animal, day)))
        mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))
    
    else:
        data_file = pickle.load(open(pref + '%s_%d_shuff_ix_null_roll.pkl' %(animal, day)))
        
    full_push = data_file['Data']['push']
    full_bin_num = data_file['Data']['bin_num']
    temp_N = data_file['temp_n']

    #### Test indices #####
    test_ix = data_file['test_ix']
    type_of_model = np.zeros((5, ))

    #### Get # neurons #
    nT, nneur = data_file['Data']['spks'].shape

    ### 2 x N decoder 
    KG = util_fcns.get_decoder(animal, day)

    model_var_list, predict_key, include_action_lags, history_bins_max = generate_models_list.get_model_var_list(6)
    models_to_include = [m[1] for m in model_var_list]

    variables_list = generate_models.return_variables_associated_with_model_var(model_var_list, include_action_lags, nneur)
    pred_Y = np.zeros((temp_N, nneur, nshuffs)) + np.nan
    true_Y = np.zeros((temp_N, nneur, nshuffs)) + np.nan

    if testing_mode:
        day_mat = analysis_config.data_params['%s_input_type'%animal] 
        order_mat = analysis_config.data_params['%s_ordered_input_type'%animal] 
        test_Data, test_Data_temp, test_Sub_spikes, test_Sub_spk_temp_all, test_Sub_push = generate_models_utils.get_spike_kinematics(animal, day_mat[day], 
                    order_mat[day], 1, day_ix = day, nshuffs = 1)
    t0 = time.time()
    
    #### Get teh spike indices ###
    tm0ix, tm1ix = generate_models.get_temp_spks_ix(data_file['Data'])
    assert(len(tm0ix) == len(tm1ix) == temp_N)

    ### Shuffle indices: 
    for shuffle in range(nshuffs):
        if shuffle % 10 == 0:
            print('Shuffle %d, %.5f' %(shuffle, time.time() - t0))
        
        if pot_shuff:
            null_shuff_ix = data_file[shuffle, 'null_roll']
            pot_shuff_ix = data_file[shuffle]

        else:
            null_shuff_ix = data_file[shuffle]

        if testing_mode: 
            null_shuff_ix = 0 # no roll 
            pot_shuff_ix = np.arange(nT)

        if pot_shuff:
            sub_spikes, sub_spikes_tm1, sub_push, tm0, tm1, ix_roll = generate_models.get_temp_spks_null_roll_pot_shuff(data_file['Data'], pot_shuff_ix, null_shuff_ix)
        
            #### Split up #####
            shuff_com_bins_disc = util_fcns.commands2bins([sub_push], mag_boundaries, animal, day, vel_ix = [3, 5], ndiv=8)[0]
            unshuff_com_bins_disc = util_fcns.commands2bins([full_push[tm0, :]], mag_boundaries, animal, day, vel_ix = [3, 5], ndiv=8)[0]
            
            #### Check that these are linked still; 
            assert(np.allclose(shuff_com_bins_disc, unshuff_com_bins_disc))
            load_str = 'null_roll_pot_beh_maint'
            
        else:
            #### get the spikes and previous spikes with roll applied ###
            sub_spikes, sub_spikes_tm1, sub_push, tm0, tm1, ix_roll = generate_models.get_temp_spks_null_pot_roll(data_file['Data'], null_shuff_ix)
        
            ### Makes ure sub_push matches: 
            assert(np.allclose(sub_push, full_push[tm0, :]))
            load_str = 'null_roll'

        #### This is teh true data that is trying to be predicted
        true_Y[:, :, shuffle] = sub_spikes

        if testing_mode: 
            assert(np.allclose(sub_spikes, test_Sub_spikes))
            assert(np.allclose(sub_spikes_tm1, test_Sub_spk_temp_all[:, 0, :]))
            assert(np.allclose(sub_push[:, [3, 5]], test_Sub_push))
            
        shuff_str = str(shuffle)
        shuff_str = shuff_str.zfill(3)
        model_file = sio.loadmat(pref + '%s_%d_shuff%s_%s_%s_models.mat' %(animal, day, shuff_str, model_name, load_str))
        
        #### Build the data dictionary ####
        data_temp_dict = {}
        for i_n in range(nneur):
            data_temp_dict['spk_tm1_n%d'%i_n] = sub_spikes_tm1[:, i_n]
            data_temp_dict['spk_tm0_n%d'%i_n] = sub_spikes[:, i_n]
        data_temp_dict['pshx_tm0'] = sub_push[:, 3]
        data_temp_dict['pshy_tm0'] = sub_push[:, 5]

        test_stack = np.hstack(([test_ix[i_fold] for i_fold in range(5)]))
        assert(np.allclose(np.sort(test_stack), np.arange(temp_N)))

        #### Deal with index re-assignment; 
        ### map b/w actual_test_ix and current inds? 
        now_shuff_ix = np.arange(full_push.shape[0])
        actual_test_ix = now_shuff_ix[pot_shuff_ix[tm0ix]]
        t2 = time.time()
        current_inds = now_shuff_ix[tm0ix]
        slope = float(len(current_inds))/float(len(now_shuff_ix))
        actual_test2current_inds = get_ati2inds_map(actual_test_ix, current_inds, slope)
        
        shuff_x = []
        #### For each data fold #####
        for i_fold in range(5): 
            ### Test indices AFTER shuff_ix[tm0ix]
            ### What these test inds are on the global scheme: 
            actual_test_ix_fold = list(actual_test_ix[test_ix[i_fold]])

            ### Need to get the location of these in our current scheme: 
            test_ix_fold_current = np.array([actual_test2current_inds[x] for x in actual_test_ix_fold])
            
            if i_fold < 4:
                shuff_x.append(np.hstack((test_ix_fold_current)))

            elif i_fold == 4:
                ### Get stuff in our current list that wasn't in any of the test indices 
                tot = np.hstack(( np.hstack((shuff_x)), np.hstack((test_ix_fold_current))))
                remaining = np.array([i for i, j in enumerate(current_inds) if i not in tot])
                test_ix_fold_current = np.hstack((remaining, np.hstack((test_ix_fold_current))))
                shuff_x.append(test_ix_fold_current)

                assert(len(np.hstack((shuff_x))) == len(tm0ix))
                assert(len(np.unique(np.hstack((shuff_x)))) == len(tm0ix))



        #### For each data fold #####
        for i_fold in range(5): 
            
            coef = model_file[str((i_fold, 'coef_'))]
            intc = model_file[str((i_fold, 'intc_'))]
            assert(model_file['shuff_num'] == shuffle)
            test_ix_fold = shuff_x[i_fold]

            ### Make sure the test Ix are in the right units ## 
            assert(np.all(test_ix_fold < temp_N))

            if testing_mode:
                coef[:,:] = np.eye(nneur)
                intc[:] = 0.

            #### Get W if conditinoing on action ####
            # if 'psh_2' in model_name: 
            #     W = model_file[str((i_fold, 'W'))]

            #     if testing_mode:
            #         ### No uncertainty/error in dynamics model 
            #         W[:,:] = 0.

            for i_m, (model_nm, variables) in enumerate(zip(models_to_include, variables_list)):

                # if model_nm == model_name: 
                #     if 'psh_2' in model_name: 

                #         pred_wc = pred_w_cond(coef, intc, data_temp_dict,
                #             test_ix_fold, variables, nneur, w_ = W, KG = KG) 

                #         pred_Y[test_ix_fold, :, shuffle] = pred_wc.copy()

                #         if testing_mode:
                #             ### Set the "W" equal to zero so wont match actions 
                #             ### Want to test for equality to the previous action/spks
                #             pass
                #         else:
                #             #### Make sure the push actuall matches intended push: #### 
                #             assert(np.allclose(np.dot(KG, pred_wc.T).T, sub_push[np.ix_(test_ix_fold, [3, 5])]))
                #             assert(np.allclose(np.dot(KG, pred_Y[test_ix_fold, :, shuffle].T).T, sub_push[np.ix_(test_ix_fold, [3, 5])]))

                #     else:
                    pred_Y[test_ix_fold, :, shuffle] = pred_wo_cond(coef, intc, data_temp_dict,
                        test_ix_fold, variables, nneur)

        ### After all folds, if in testing mode, pred_Y should be equal to sub_spikes_tm1; 
        if testing_mode:
            assert(np.allclose(sub_spikes_tm1, pred_Y[:, :, shuffle]))
    
    assert(np.sum(np.isnan(pred_Y)) == 0)
    assert(np.sum(np.isnan(true_Y)) == 0)
    return pred_Y, true_Y, sub_push, ix_roll

def get_shuffled_mean_maint(animal, day, model_name, nshuffs = 10, testing_mode = False):
    return get_shuffled_data_v2(animal, day, model_name, nshuffs = nshuffs, testing_mode = False, mean_maint = True)

def get_shuffled_within_mov(animal, day, model_name, nshuffs = 10, testing_mode = False):
    return get_shuffled_data_v2(animal, day, model_name, nshuffs = 10, testing_mode = False, within_mov = True)

def pred_wo_cond(coef_, intc_, data_temp_dict, test_ix_fold, variable_names, nneur, **kwargs): 
    """
    Make predictions from saved data and saved Ridge models, NOT conditioned on action 
    
    Parameters
    ----------
    coef_ : np.array (n x n )
        coef matrix from shuffled file (Ridge.coef_)
    intc_ : np.array (1 x n)
        intc array from shuffled file (Ridge.intercept_)
    data_temp_dict : dictionary 
        holds data 
    test_ix_fold : np.array
        indices corresonding to held out data for this model 
    variable_names : list
        list of variable names -- output from generate_models.return_variables_associated_with_model_var
    nneur : int
        number of neurons 
    
    Returns
    -------
    np.array (T x N ) of dynamcis predictions 
    """

    assert(intc_.shape[1] == nneur == coef_.shape[0] == coef_.shape[1])
    model = Ridge()
    model.coef_ = np.squeeze(coef_.copy())
    model.intercept_ = np.squeeze(intc_.copy())



    ### Get variables #### 
    if 'X_preassembled' in kwargs.keys():
        X = kwargs['X_preassembled']

    else:
        generate_models.check_variables_order(variable_names, nneur)
        x_test = [] 
        for vr in variable_names:
            x_test.append(data_temp_dict[vr][test_ix_fold, np.newaxis])
            X = np.mat(np.hstack((x_test)))

    predy = model.predict(X)
    return predy

def pred_w_cond(coef_, intc_, data_temp_dict, test_ix_fold, variable_names, nneur, w_ = None, 
    KG = None, **kwargs):
    """
    Make predictions from saved data and saved Ridge models, conditioned on action 
    
    Parameters
    ----------
    coef_ : np.array (n x n )
        coef matrix from shuffled file (Ridge.coef_)
    intc_ : np.array (1 x n)
        intc array from shuffled file (Ridge.intercept_)
    data_temp_dict : dictionary 
        holds data 
    test_ix_fold : np.array
        indices corresonding to held out data for this model 
    variable_names : list
        list of variable names -- output from generate_models.return_variables_associated_with_model_var
    nneur : int
        number of neurons 
    w_ : None, optional
        np.array (n x n) error covariance of Ridge dynamics model 
    KG : None, optional
        (2 x N) np.array, Kalman gain 

    **kwargs
        Description
    
    Returns
    -------
    np.array (T x N ) of dynamcis predictions 
    """
    assert(intc_.shape[1] == nneur == coef_.shape[0] == coef_.shape[1] == w_.shape[0] == w_.shape[1])
    model = Ridge()
    model.coef_ = np.squeeze(coef_.copy())
    model.intercept_ = np.squeeze(intc_.copy())
    model.W = np.squeeze(w_.copy())

    ##### check that variables iare in the right order ####

    ### Get variables #### 
    if 'X_preassembled' in kwargs.keys():
        X = kwargs['X_preassembled']

    else:
        generate_models.check_variables_order(variable_names, nneur)
        x_test = [] 
        for vr in variable_names:
            x_test.append(data_temp_dict[vr][test_ix_fold, np.newaxis])
            X = np.mat(np.hstack((x_test)))

    #### Make non-action conditioned prediction ###
    pred = np.mat(model.predict(X))

    ### Get action ###
    if 'A_preassembled' in kwargs.keys():
        A = kwargs['A_preassembled']
    else:
        A = []
        for v in ['pshx_tm0', 'pshy_tm0']: 
            A.append(data_temp_dict[v][test_ix_fold, np.newaxis])
        A = np.hstack((A))

    assert(A.shape[0] == X.shape[0])

    ### Condition on action! 
    ### For each datapoint, estimate 
    ## y  ~ N (Ayt+b, W)
    ## a  ~ N (K(Ayt+b), KWK') 
    ## E(y_t | a_t) = (Ayt + b) + WK'()

    cov = model.W; 
    cov12 = np.dot(KG, cov).T
    cov21 = np.dot(KG, cov)
    cov22 = np.dot(KG, np.dot(cov, KG.T))

    try:
        cov22I = np.linalg.inv(cov22)
        testing_mode = False
    except:
        print('Should only be in testing mode --> W is singular')
        cov22I = np.zeros_like(cov22)
        testing_mode = True

    #### Ge prediction: T x N 
    mu2_i = np.dot(KG, pred.T).T                
    pred_w_cond = pred + np.dot(cov12, np.dot(cov22I, (A - mu2_i).T)).T

    if testing_mode:
        pass
    else:
        assert(np.allclose(np.dot(KG, pred_w_cond.T).T, A))
    return pred_w_cond
      
#### Deprecated shuffles #### 
def get_shuffled_data(animal, day, model_nm, get_model = False, with_intercept = True, nshuffs = 100):
    """


    
    Parameters
    ----------
    animal : TYPE
        Description
    day : TYPE
        Description
    model_nm : TYPE
        Description
    get_model : bool, optional
        Description
    with_intercept : bool, optional
        Description
    nshuffs : int, optional
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    if with_intercept:
        pref = analysis_config.config['shuff_fig_dir']
    else:
        pref = analysis_config.config['shuff_fig_dir_nointc']
    if 'latent' in model_nm:
        pref = analysis_config.config['shuff_fig_dir_latentLDS']

    #### Deprecated as of 10/2020 --> from when saved output predictions ####
    data_file = pickle.load(open(pref + '%s_%d_shuff_ix.pkl' %(animal, day)))
    data_temp = data_file['Data_temp']
    Sub_push_all = data_file['Sub_push']
    Sub_spikes_all = data_file['Sub_spikes']

    #### models #### grom_0_shuff009_hist_1pos_0psh_2spksm_1_spksp_0_models
    files = np.sort(glob.glob(pref + '%s_%s_shuff*_%s.pkl'%(animal, day, model_nm)))
    files = files[:nshuffs]

    #### Predictions 
    pred_y = [] 
    for i_f, fl in enumerate(files): 

        #### Get shuffle indices; 
        model_file = pickle.load(open(fl, 'rb'))
        shuff_num = model_file['shuff_num']
        shuffle_indices = data_file[int(shuff_num)]

        ### Shuffled --> matched ####
        sub_push = Sub_push_all[shuffle_indices, :]
        sub_spikes = Sub_spikes_all[shuffle_indices, :]

        for i_fold in range(model_file['model_set'].keys()):

            #### Get the model ######
            ridge_model = model_file['model_set'][i_fold]

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









    files = np.sort(glob.glob(pref + '%s_%d_shuff*_%s.mat' %(animal, day, model_nm)))
    files = files[:nshuffs]
    pred_Y = []; coef = []; intec = []; coef2 = []; keepix = []; 
    
    for i_f, fl in enumerate(files):
        tmp = sio.loadmat(fl)
        if len(tmp['model_data'].shape) == 3:
            pred_Y.append(tmp['model_data'][:, :, 2])
        else:
            pred_Y.append(tmp['model_data'])

        if get_model:
            if 'latent' in model_nm:
                coef.extend(tmp['modelA'])
                coef2.extend(tmp['modelC'])
                keepix.extend(tmp['modelkeepix'])
            else:
                coef.extend(tmp['model_coef'])

            if with_intercept:
                intec.extend(tmp['model_intc'])

    if get_model:
        if 'latent' in model_nm:
            return np.dstack((pred_Y)), coef, coef2, keepix
        else:
            return np.dstack((pred_Y)), coef, intec
        
    else:
        return np.dstack((pred_Y))

def get_perc(dist, value):
    ix = np.nonzero(dist >= value)[0]
    return float(len(ix))/len(dist)

### Bar R2 and correlation plots -- figure 4;
def plot_real_vs_pred(model_set_number = 2, min_obs = 15, cov = False, 
    use_mFR_option = 'cond_spec', include_shuffs = None, scatter_ds = 0,
    plot_diffs = False, day_i = 0):

    '''
    updates -- 6/10/20 -- plotting task specific plot, using slimmer model 2, adding plot for dHz vs. baseline

    use_mFR_option -- 
        -- ultra_pooled: use pooled mFR | day, also plot mFR | command (not condition)
        -- pooled: for both x/y axis used pooled mFR: mFR | day 
        -- command_spec: for both x/y axis used command mFR: mFR | day, command
        -- command_axis_spec: for x axis used mFR | day, command, yaxis used pred mFR | day, command

    '''
    
    mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))
    
    if model_set_number == 2:
        models_to_include = ['prespos_0psh_1spksm_0_spksp_0', 
                             'hist_1pos_3psh_1spksm_0_spksp_0', ### Full state 
                             'hist_1pos_3psh_1spksm_1_spksp_0',### Full state + y_t-1
                             'hist_1pos_0psh_1spksm_1_spksp_0'] ### a_t + y_t-1

        models_to_include_labs = ['y_t | a_t', 
                                  'y_t | a_t, s_{t-1}, s_tFinal, tsk', 
                                  'y_t | a_t, s_{t-1},..., y_{t-1}',
                                  'y_t | a_t, y_{t-1}']
    elif model_set_number == 6:
        models_to_include = ['hist_1pos_0psh_2spksm_1_spksp_0']
        models_to_include_labs = ['y_t = f(y_{t-1} | a_t)']

    for ia, animal in enumerate(['grom','jeev']):

        ### Load the model ####
        model_dict = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen_match_tsk_N.pkl' %model_set_number, 'rb'))
        #model_dict = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d_.pkl' %model_set_number, 'rb'))
        
        if include_shuffs is not None:
            model_dicts = [model_dict]

            pre1 = analysis_config.config[animal+'_pref']+'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen_match_tsk_N_'%model_set_number
            #pre1 = analysis_config.config[animal+'_pref']+'tuning_models_'+animal+'_model_set%d__'%model_set_number
            for shuf_ix in range(include_shuffs):

                if os.path.exists('%swithin_bin_shuff%d.pkl'%(pre1, shuf_ix)):
                    print('Shuff option 1')
                    tmp = pickle.load(open('%swithin_bin_shuff%d.pkl'%(pre1, shuf_ix), 'rb'))
                    model_dicts.append(copy.deepcopy(tmp))

                    # tmp = pickle.load(open('%sfull_shuff%d.pkl'%(pre1, shuf_ix), 'rb'))
                    # model_dicts.append(copy.deepcopy(tmp))

                else:
                    raise Exception('No shuffles of index %d identified! '%(shuf_ix))
        else:
            model_dicts = [model_dict]

        ### Number of days #####
        ndays = analysis_config.data_params[animal + '_ndays']

        ### Keep track fo diffs and sig diffs #####
        diffs = dict(); 
        sig_diffs = dict(); 

        ### For keeping track of stats ####
        MOD = []; DAY = []; VAL = []; VAL2 = []; MDI = []; 

        pred_diffs = dict(); 
        pred_sig_diffs = dict(); 

        R2 = dict(); R22 = dict(); NN = dict(); 
        z_fr = dict(); pred_z_fr = dict(); mn_fr = {}

        fall, axall = plt.subplots(ncols = len(models_to_include), nrows = 1, figsize=(len(models_to_include)*5, 5))
        fall2, axall2 = plt.subplots(ncols = len(models_to_include), nrows = 1, figsize=(len(models_to_include)*5, 5))

        if len(models_to_include) == 1:
            axall = np.array([axall])
            axall2 = np.array([axall2])

        for i_m, model in enumerate(models_to_include):
            for i_d in range(ndays):

                if len(model_dicts) > 1:
                    ### Make sure task / target / command bins match for this 
                    assert(np.all(model_dicts[0][i_d, 'task'] == model_dicts[1][i_d, 'task'] ))
                    assert(np.all(model_dicts[0][i_d, 'trg'] == model_dicts[1][i_d, 'trg'] ))

                    c0 = model_dicts[0][i_d, 'np']
                    c1 = model_dicts[1][i_d, 'np']
                    assert(c0.shape[1] == c1.shape[1] == 2)
                    com0 = util_fcns.commands2bins([c0], mag_boundaries, animal, i_d, vel_ix = [0, 1])[0]
                    com1 = util_fcns.commands2bins([c1], mag_boundaries, animal, i_d, vel_ix = [0, 1])[0]
                    
                    ### Make sure discretized bins match
                    assert(np.all(com0 == com1))

                    ### Not actual np
                    assert(not np.all(c0 == c1))
                    assert(not np.all(model_dicts[0][i_d, 'spks'] == model_dicts[1][i_d, 'spks']))

                ### For all the model dicts; 
                for i_ds, model_dict_i in enumerate(model_dicts): 

                    R2[model, i_d, i_ds] = []; 
                    R22[model, i_d, i_ds] = []

                    ############# For real model ############
                    ### Get the spiking data
                    spks = model_dict_i[i_d, 'spks']

                    ### General dynamics; 
                    pred = model_dict_i[i_d, model][:, :, 2]

                    ### Get the task parameters
                    tsk  = model_dict_i[i_d, 'task']
                    targ = model_dict_i[i_d, 'trg']
                    push = model_dict_i[i_d, 'np']
                
                    assert(spks.shape[0] == pred.shape[0] == len(tsk) == len(targ) == push.shape[0])
                    assert(push.shape[1] == 2)

                    ### Setup the dicitonaries for stats to be tracked ####
                    diffs[model, i_d, i_ds] = []
                    sig_diffs[model, i_d, i_ds] = []

                    pred_diffs[model, i_d, i_ds] = []
                    pred_sig_diffs[model, i_d, i_ds] = []

                    #### Setup the dictionaries for stats to be tracked ####
                    z_fr[model, i_d, i_ds] = []
                    pred_z_fr[model, i_d, i_ds] = []

                    ### Get the discretized commands
                    commands_disc = util_fcns.commands2bins([push], mag_boundaries, animal, i_d, vel_ix = [0, 1])[0]
                    assert(commands_disc.shape == push.shape)

                    #### Get mFR over full: 
                    mFR_pool = np.mean(spks, axis=0)

                    ### Now go through combos and plot 
                    for mag_i in range(4):
                        for ang_i in range(8):
                            
                            ix = np.nonzero(np.logical_and(commands_disc[:, 0] == mag_i, commands_disc[:, 1] == ang_i))[0]

                            if len(ix) > 0:
                                ### mFR for a given command ###
                                command_mn_fr = np.mean(spks[ix, :], axis=0)
                                pred_command_mn_fr = np.mean(pred[ix, :], axis=0)

                                if use_mFR_option == 'pooled':
                                    x_mfr = mFR_pool.copy()
                                    y_mfr = mFR_pool.copy()
                                
                                elif use_mFR_option == 'command_spec':
                                    x_mfr = command_mn_fr.copy()
                                    y_mfr = command_mn_fr.copy()
                                
                                elif use_mFR_option == 'command_axis_spec':
                                    x_mfr = command_mn_fr.copy()
                                    y_mfr = pred_command_mn_fr.copy()
                                
                                if use_mFR_option == 'ultra_pooled':
                                    x_mfr = mFR_pool.copy()
                                    y_mfr = mFR_pool.copy()                            

                                    ### For this option only plot the 
                                    if len(ix) > min_obs:

                                        ### Add this guy only for ultra_pooled 
                                        z_fr[model, i_d, i_ds].append(np.mean(spks[ix, :], axis=0) - x_mfr)
                                        pred_z_fr[model, i_d, i_ds].append(np.mean(pred[ix, :], axis=0) - y_mfr)

                                else:
                                    co_added_targ = []
                                    obs_added_targ = []

                                    for targi in np.unique(targ):
                                        ### Get co / task 
                                        ix_co = (commands_disc[:, 0] == mag_i) & (commands_disc[:, 1] == ang_i) & (tsk == 0) & (targ == targi)
                                        ix_co = np.nonzero(ix_co == True)[0]
                                        for ixtmp in ix_co: assert(ixtmp in ix)
                                        assert(len(ix_co) <= len(ix))
                                        assert(np.all(commands_disc[ix_co, 0] == mag_i))
                                        assert(np.all(commands_disc[ix_co, 1] == ang_i))
                                        assert(np.all(tsk[ix_co] == 0))
                                        assert(np.all(targ[ix_co] == targi))

                                        if len(ix_co) >= min_obs:
                                            #### Keep the mean CO condition specific command
                                            if targi not in co_added_targ:
                                                z_fr[model, i_d, i_ds].append(np.mean(spks[ix_co, :], axis=0) - x_mfr)
                                                pred_z_fr[model, i_d, i_ds].append(np.mean(pred[ix_co, :], axis=0) - y_mfr)
                                                co_added_targ.append(targi)

                                            ### Get info second task: 
                                            for targi2 in np.unique(targ):
                                                ix_ob = (commands_disc[:, 0] == mag_i) & (commands_disc[:, 1] == ang_i) & (tsk == 1) & (targ == targi2)
                                                ix_ob = np.nonzero(ix_ob == True)[0]
                                                
                                                assert(np.all(ixtmp in ix for ixtmp in ix_ob))
                                                assert(len(ix_ob) <= len(ix))
                                                for ixtmp in ix_ob: assert(ixtmp in ix)
                                                assert(np.all(commands_disc[ix_ob, 0] == mag_i))
                                                assert(np.all(commands_disc[ix_ob, 1] == ang_i))
                                                assert(np.all(tsk[ix_ob] == 1))
                                                assert(np.all(targ[ix_ob] == targi2))

                                                if len(ix_ob) >= min_obs:
                                                    if targi2 not in obs_added_targ:
                                                        z_fr[model, i_d, i_ds].append(np.mean(spks[ix_ob, :], axis=0) - x_mfr)
                                                        pred_z_fr[model, i_d, i_ds].append(np.mean(pred[ix_ob, :], axis=0) - y_mfr)
                                                        obs_added_targ.append(targi)

                                                    assert(len(ix_co) >= min_obs)
                                                    assert(len(ix_ob) >= min_obs)
                                                    #print 'adding: mag_i: %d, ang_i: %d, targi: %d, targi2: %d' %(mag_i, ang_i, targi, targi2)
                                                    ### make the plot: 
                                                    if cov: 
                                                        diffs[model, i_d, i_ds] = util_fcns.get_cov_diffs(ix_co, ix_ob, spks, diffs[model, i_d])
                                                        pred_diffs[model, i_d, i_ds] = util_fcns.get_cov_diffs(ix_co, ix_ob, pred, pred_diffs[model, i_d])
                                                    else: 
                                                        diffs[model, i_d, i_ds].append(np.mean(spks[ix_co, :], axis=0) - np.mean(spks[ix_ob, :], axis=0))    
                                                        pred_diffs[model, i_d, i_ds].append(np.mean(pred[ix_co, :], axis=0) - np.mean(pred[ix_ob, :], axis=0))

            ### Now scatter plot all data over all days: 
            #f, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(8, 4))
            ### Make R2 plots to show how much each plot accounts for variace: 
            for i_d in range(ndays):
                for i_ds in range(len(model_dicts)):

                    MDI.append(i_ds)
                    MOD.append(i_m)
                    DAY.append(i_d)

                    if i_ds == scatter_ds:
                        color = 'k'
                    # elif i_ds == 1:
                    #     color = 'b'
                    # elif i_ds == 2:
                    #     color = 'k'

                    if use_mFR_option == 'ultra_pooled':
                        pass
                    else:
                        if plot_diffs:
                            ### T x N

                            x = np.vstack((diffs[model, i_d, i_ds]))*10
                            y = np.vstack((pred_diffs[model, i_d, i_ds]))*10

                            #### For the diffs, want to reshape everything into a single long array 
                            x = x.reshape(-1)
                            y = y.reshape(-1)

                            if i_ds == scatter_ds:
                                if i_d == day_i:
                                    axall[i_m].plot(x, y, color+'.', markersize=2.)
                            axall[i_m].set_xlim([-40, 40])
                            axall[i_m].set_ylim([-40, 40])
                            axall[i_m].plot([-40, 40], [-40, 40], 'k--', linewidth = 1.)
                            axall[i_m].set_ylabel('Pred Mn Diff | Command', fontsize=14)
                            axall[i_m].set_xlabel('Mn Diff | Command', fontsize=14)

                            ### get variance explained -- here, each point is a neuron / command / day / targ1 / targ 2 difference
                            ### the mean for SST is the neuron specific avg. true difference. 
                            VAF = util_fcns.get_R2(x, y, pop = True)

                            ### Old VAF: 
                            #VAF = 1 - np.sum((x-y)**2)/np.sum((x-np.mean(x))**2)
                            R2[model, i_d, i_ds].append(VAF);
                            if i_ds == scatter_ds and i_d == day_i:
                                axall[i_d, i_m].set_title('Day %d, VAF = %.4f\n %s' %(i_d,VAF, '$'+models_to_include_labs[i_m]+'$'), fontsize=14)
                            VAL.append(VAF)

                    ### Always plot 
                    ########### ONLY PLOT TRUE Z FR #########
                    x2 = np.vstack(( z_fr[model, i_d, i_ds]))*10
                    y2 = np.vstack(( pred_z_fr[model, i_d, i_ds]))*10
                    
                    x2 = x2.reshape(-1)
                    y2 = y2.reshape(-1)

                    
                    ### Here, not reshaping, want to keep the mean FR | command, condition comparable to mFR for neuron overall; 
                    if i_ds == scatter_ds:
                        if i_d == day_i:
                            slp,intc,rv,pv,err = scipy.stats.linregress(x2, y2)
                            axall2[i_m].plot([np.min(x2), np.max(x2)], slp*np.array([np.min(x2), np.max(x2)]) + intc, '--', color=color)
                            axall2[i_m].plot(x2, y2, color+'.', markersize=2.)
    
                    axall2[i_m].set_xlim([-40, 40])
                    axall2[i_m].set_ylim([-40, 40])
                    axall2[i_m].plot([-40, 40], [-40, 40], 'k--', linewidth = 1.)

                    if use_mFR_option == 'ultra_pooled':
                        axall2[i_m].set_ylabel('Pred Mn | Command\n - mFR', fontsize=14)
                        axall2[i_m].set_xlabel('Mn FR | Command\n - mFR', fontsize=14)        
                    
                    else:
                        if use_mFR_option == 'command_spec':
                            axall2[i_m].set_ylabel('Pred Mn | Cond, Command\n - mFR|Command', fontsize=14)
                        elif use_mFR_option == 'command_axis_spec':
                            axall2[i_m].set_ylabel('Pred Mn | Cond, Command\n - Pred mFR|Command', fontsize=14)
                        axall2[i_m].set_xlabel('Mn FR | Cond, Command\n - mFR|Command', fontsize=14)
                    
                    VAF2 = util_fcns.get_R2(x2, y2, pop = True)
                    VAL2.append(VAF2)
                    R22[model, i_d, i_ds].append(VAF2)
                    if i_ds == scatter_ds and i_d == day_i:
                        axall2[i_m].set_title('Day %d, DS %d, VAF = %.4f\n %s \n %s' %(i_d, i_ds, VAF2, '$'+models_to_include_labs[i_m]+'$', use_mFR_option), fontsize=14)

        #### Save the second one ########
        fall2.savefig(fig_dir + 'scatter_%s_%s_command_cond_spec_scatterShuff%d.eps'%(animal, use_mFR_option, scatter_ds))

        if use_mFR_option == 'ultra_pooled':
            VAL = None
        else:
            fall.tight_layout()
            if include_shuffs is None:
                fall.savefig(analysis_config.config['fig_dir3'] + 'diff_scatters_%s_R2_unrolled_scatterShuff%d.eps' %(animal, scatter_ds))
            if len(VAL) > 0:
                VAL = np.hstack((VAL))

        fall2.tight_layout()
        if include_shuffs is None:
            fall2.savefig(analysis_config.config['fig_dir3'] + 'cond_spec_scatters_%s_mFR_%s.eps' %(animal, use_mFR_option))

        MOD = np.hstack((MOD))
        MDI = np.hstack((MDI))
        DAY = np.hstack((DAY))
        VAL2 = np.hstack((VAL2))


        for iv, (val, r2_caps, nm) in enumerate(zip([VAL, VAL2], [R2, R22], ['diffs', 'mFR_cond_command'])):

            if val is None:
                pass
            else:
                cont = False
                if iv > 0:
                    cont = True
                else:
                    if iv == 0 and plot_diffs:
                        cont = True

                if cont:
                    #### R2 bar plot ####
                    fbar, axbar = plt.subplots(figsize=(8, 8))

                    if len(model_dicts) > 1:
                        fbarsh, axbarsh = plt.subplots(figsize = (8, 8))
                    
                    if include_shuffs is None:
                        ### Plot indices ###
                        ix0 = np.nonzero(MOD < 2)[0]
                        ix1 = np.nonzero(np.logical_and(MOD > 0, MOD <3))[0]

                        #pv0, slp0 = util_fcns.run_LME(DAY[ix0], MOD[ix0], val[ix0])
                        #pv1, slp1 = util_fcns.run_LME(DAY[ix1], MOD[ix1], val[ix1])

                        #print('Animal %s, Mods %s, pv: %.3f, slp: %.3f, N: %d' %(animal, str(np.unique(MOD[ix0])), pv0, slp0, len(ix0)))
                        #print('Animal %s, Mods %s, pv: %.3f, slp: %.3f, N: %d' %(animal, str(np.unique(MOD[ix1])), pv1, slp1, len(ix1)))

                    ### Plot as bar plot ###
                    all_data = {}
                    for i_m, model in enumerate(models_to_include):
                        for i_ds in range(len(model_dicts)):
                            all_data[model, i_ds] = []; 

                    for i_d in range(ndays):
                        tmp = []; tmp2 = []; tmp_sh = []; tmp2_sh = []; 
                        
                        for i_m, model in enumerate(models_to_include):
                            for i_ds in range(len(model_dicts)):
                                r2 = np.hstack((r2_caps[model, i_d, i_ds]))
                                #r2[np.isinf(r2)] = np.nan
                                if i_ds == 0:
                                    tmp.append(np.mean(r2))
                                    tmp2.append(i_m + 0.25*(i_ds))
                                elif i_ds == 1:
                                    tmp_sh.append(np.mean(r2))
                                    tmp2_sh.append(i_m)
                                all_data[model, i_ds].append(r2)
                        axbar.plot(tmp2, tmp, '-', color='gray')
                        if len(tmp_sh) > 0:
                            axbarsh.plot(tmp2_sh, tmp_sh, '-', color='gray')

                    #### Model colors ###
                    model_cols = [[255, 0, 0], [101, 44, 144], [39, 169, 225], [39, 169, 225]]
                    model_cols = [np.array(m)/255. for m in model_cols]

                    for i_m, model in enumerate(models_to_include):
                        for i_ds in range(len(model_dicts)):
                            tmp3 = np.hstack((all_data[model, i_ds]))
                            if i_ds == 0:
                                axbar.bar(i_m , np.mean(tmp3), color = model_cols[i_m], edgecolor='k', linewidth=2., width = 1)
                                axbar.errorbar(i_m, np.mean(tmp3), yerr=np.std(tmp3)/np.sqrt(len(tmp3)), color = 'k', marker='|')
                            elif i_ds == 1:
                                axbarsh.bar(i_m , np.mean(tmp3), color = model_cols[i_m], edgecolor='k', linewidth=2., width = 1, alpha = .2)
                                axbarsh.errorbar(i_m, np.mean(tmp3), yerr=np.std(tmp3)/np.sqrt(len(tmp3)), color = 'k', marker='|')

                    axbar.set_xticks(np.arange(len(models_to_include)))
                    if include_shuffs:
                        axbarsh.set_xticks(np.arange(len(models_to_include)))
                    models_to_include_labs_tex = ['$' + m + '$' for m in models_to_include_labs]
                    axbar.set_xticklabels(models_to_include_labs_tex, rotation = 45)#, fontsize=10)
                    if include_shuffs:
                        axbarsh.set_xticklabels(models_to_include_labs_tex, rotation = 45)#, fontsize=10)
    
                    axbar.set_ylim([-.1, .7])
                    if include_shuffs:
                        axbarsh.set_ylim([-.1, .7])
                        fbarsh.tight_layout()

                    fbar.tight_layout()
                    
                    if iv == 0:
                        fbar.savefig(analysis_config.config['fig_dir3']+'monk_%s_r2_comparison_%s.eps' %(animal, nm))
                        if include_shuffs:
                            fbarsh.savefig(analysis_config.config['fig_dir3']+'monk_%s_r2_comparison_%s_SHUFFLE.eps' %(animal, nm))
                    else:
                        fbar.savefig(analysis_config.config['fig_dir3'] + 'monk_%s_r2_comparison_%s_mFR%s.eps' %(animal, nm, use_mFR_option))  
                        if include_shuffs:
                            fbarsh.savefig(analysis_config.config['fig_dir3'] + 'monk_%s_r2_comparison_%s_mFR%s_SHUFFLE.eps' %(animal, nm, use_mFR_option))  

def plot_real_vs_pred_bars_w_shuffle(use_mFR_option, model_dicts = None, min_obs = 15):
    '''
    use_mFR_option -- 
        -- ultra_pooled: use pooled mFR | day, also plot mFR | command (not condition)
        -- pooled: for both x/y axis used pooled mFR: mFR | day 
        -- command_spec: for both x/y axis used command mFR: mFR | day, command
        -- command_axis_spec: for x axis used mFR | day, command, yaxis used pred mFR | day, command
    '''

    mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))
    model = 'hist_1pos_0psh_2spksm_1_spksp_0'
    xkeys = ['shuffled', 'dyn_gen', 'identity_dyn']#'dyn_tsk', 'identity_dyn']
    xlab = ['shuffled', 
        'general\ndynamics', 
        'identity\ndynamics']

    models_colors = [[100, 100, 100], 
                 [39, 169, 225], 
                 #[39, 169, 225], 
                 [100, 100, 100]]
    models_colors = [np.array(m)/255. for m in models_colors]
    model_set_number = 6; 

    for ia, animal in enumerate(['jeev', 'grom']):#,'grom']):

        ### Load the model ####
        if model_dicts is None:
            model_dict = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen_match_tsk_N.pkl' %model_set_number, 'rb'))
        else:
            model_dict = model_dicts[animal]

        ### Number of days #####
        ndays = analysis_config.data_params[animal + '_ndays']
        
        z_fr = {}
        pred_z_fr = {}; 

        for i_d in range(ndays):

            ###### True data #####
            pred_Y = dict()

            ######### General dynamics conditioned on action #########
            pred_Y['dyn_gen'] =  model_dict[i_d, 'hist_1pos_0psh_2spksm_1_spksp_0'][:, :, 2]

            ######### Get task specific #########
            tsk_spec = np.zeros_like(model_dict[i_d, 'hist_1pos_0psh_2spksm_1_spksp_0'][:, :, 2])
            for tsk in range(2):
                ix_tsk = np.nonzero(model_dict[i_d, 'task'] == tsk)[0]
                tsk_spec[ix_tsk, :] = model_dict[i_d, 'hist_1pos_0psh_2spksm_1_spksp_0'][ix_tsk, :, tsk]

            pred_Y['dyn_tsk'] = tsk_spec.copy()

            ######### Get shuffled ###############
            pred_Y['shuffled'] = get_shuffled_data(animal, i_d, 'hist_1pos_0psh_2spksm_1_spksp_0')
            nShuff = pred_Y['shuffled'].shape[2]

            ######## Get zero order hold #########
            pred_Y['identity_dyn'] = model_dict[i_d, 'identity_dyn'][:, :, 2]

            ############# For real model ############
            ### Get the spiking data
            spks = model_dict[i_d, 'spks']

            ### Get the task parameters
            tsk  = model_dict[i_d, 'task']
            targ = model_dict[i_d, 'trg']
            push = model_dict[i_d, 'np']
                
            assert(spks.shape[0] == pred_Y['dyn_gen'].shape[0] == len(tsk) == len(targ) == push.shape[0])
            assert(push.shape[1] == 2)

            for i_ds, xk in enumerate(xkeys):
                
                ### Get the correct predictions 
                pred = pred_Y[xk]

                #### Setup the dictionaries for stats to be tracked ####
                z_fr[model, i_d, i_ds] = []
                pred_z_fr[model, i_d, i_ds] = []

                ### Get the discretized commands
                commands_disc = util_fcns.commands2bins([push], mag_boundaries, animal, i_d, vel_ix = [0, 1])[0]
                assert(commands_disc.shape == push.shape)

                #### Get mFR over full: 
                mFR_pool = np.mean(spks, axis=0)

                ### Now go through combos and plot 
                for mag_i in range(4):
                    for ang_i in range(8):
                        ix = np.nonzero(np.logical_and(commands_disc[:, 0] == mag_i, commands_disc[:, 1] == ang_i))[0]

                        if len(ix) > 0:
                            ### mFR for a given command ###
                            command_mn_fr = np.mean(spks[ix, :], axis=0)
                            if xk == 'shuffled':
                                pred_command_mn_fr = []
                                for i in range(nShuff):
                                    tmp = np.mean(pred[ix, :, i], axis=0)
                                    if np.sum(np.isnan(tmp)) > 0:
                                        print("Stopping line 2009")
                                        import pdb; pdb.set_trace()
                                    pred_command_mn_fr.append(tmp)
                            else:
                                pred_command_mn_fr = np.mean(pred[ix, :], axis=0)

                            if use_mFR_option == 'pooled':
                                x_mfr = mFR_pool.copy()
                                y_mfr = mFR_pool.copy()
                            
                            elif use_mFR_option == 'command_spec':
                                x_mfr = command_mn_fr.copy()
                                y_mfr = command_mn_fr.copy()
                            
                            elif use_mFR_option == 'command_axis_spec':
                                x_mfr = command_mn_fr.copy()
                                y_mfr = copy.copy(pred_command_mn_fr)
                            
                            if use_mFR_option == 'ultra_pooled':
                                x_mfr = mFR_pool.copy()
                                y_mfr = mFR_pool.copy()                            

                                ### For this option only plot the 
                                if len(ix) >= min_obs:

                                    ### Add this guy only for ultra_pooled 
                                    z_fr[model, i_d, i_ds].append(np.mean(spks[ix, :], axis=0) - x_mfr)
                                    if xk == 'shuffled':
                                        tmp = []; 
                                        for ns in range(nShuff):
                                            tmp.append(np.mean(pred[ix, :, ns], axis=0) - y_mfr)
                                        pred_z_fr[model, i_d, i_ds].append(np.vstack((tmp)))
                                    
                                    else:
                                        pred_z_fr[model, i_d, i_ds].append(np.mean(pred[ix, :], axis=0) - y_mfr)

                            else:
                                co_added_targ = []
                                obs_added_targ = []

                                for targi in np.unique(targ):
                                    
                                    ### Get co / task 
                                    ix_co = (commands_disc[:, 0] == mag_i) & (commands_disc[:, 1] == ang_i) & (tsk == 0) & (targ == targi)
                                    ix_co = np.nonzero(ix_co == True)[0]
                                    for ixtmp in ix_co: assert(ixtmp in ix)
                                    assert(len(ix_co) <= len(ix))
                                    assert(np.all(commands_disc[ix_co, 0] == mag_i))
                                    assert(np.all(commands_disc[ix_co, 1] == ang_i))
                                    assert(np.all(tsk[ix_co] == 0))
                                    assert(np.all(targ[ix_co] == targi))

                                    if len(ix_co) >= min_obs:

                                        #### Keep the mean CO condition specific command
                                        if targi not in co_added_targ:
                                            z_fr[model, i_d, i_ds].append(np.mean(spks[ix_co, :], axis=0) - x_mfr)

                                            if xk == 'shuffled':
                                                if use_mFR_option == 'command_axis_spec':
                                                    tmp = []; 
                                                    for ns in range(nShuff):
                                                        tmpi = np.mean(pred[ix_co, :, ns], axis=0) - y_mfr[ns]
                                                        if np.sum(np.isnan(tmpi)) > 0: 
                                                            print('Stopping lin 2071')
                                                            import pdb; pdb.set_trace()
                                                        tmp.append(tmpi)
                                                    pred_z_fr[model, i_d, i_ds].append(np.vstack((tmp)))

                                                else:
                                                    tmp = []; 
                                                    for ns in range(nShuff):
                                                        tmp.append(np.mean(pred[ix, :, ns], axis=0) - y_mfr)
                                                    pred_z_fr[model, i_d, i_ds].append(np.vstack((tmp)))
                                            else:
                                                pred_z_fr[model, i_d, i_ds].append(np.mean(pred[ix_co, :], axis=0) - y_mfr)
   
                                            co_added_targ.append(targi)

                                ### Get info second task: 
                                for targi2 in np.unique(targ):
                                    ix_ob = (commands_disc[:, 0] == mag_i) & (commands_disc[:, 1] == ang_i) & (tsk == 1) & (targ == targi2)
                                    ix_ob = np.nonzero(ix_ob == True)[0]
                                    
                                    assert(np.all(ixtmp in ix for ixtmp in ix_ob))
                                    assert(len(ix_ob) <= len(ix))
                                    for ixtmp in ix_ob: assert(ixtmp in ix)
                                    assert(np.all(commands_disc[ix_ob, 0] == mag_i))
                                    assert(np.all(commands_disc[ix_ob, 1] == ang_i))
                                    assert(np.all(tsk[ix_ob] == 1))
                                    assert(np.all(targ[ix_ob] == targi2))

                                    if len(ix_ob) >= min_obs:
                                        if targi2 not in obs_added_targ:
                                            z_fr[model, i_d, i_ds].append(np.mean(spks[ix_ob, :], axis=0) - x_mfr)

                                            if xk == 'shuffled':
                                                if use_mFR_option == 'command_axis_spec':
                                                    tmp = []
                                                    for ns in range(nShuff):
                                                        tmpi = np.mean(pred[ix_ob, :, ns], axis=0) - y_mfr[ns]
                                                        if np.sum(np.isnan(tmpi)) > 0:
                                                            print('Stopping line 2111')
                                                            import pdb; pdb.set_trace()
                                                        tmp.append(tmpi)
                                                    pred_z_fr[model, i_d, i_ds].append(np.vstack((tmp)))

                                                else:
                                                    tmp = []
                                                    for ns in range(nShuff):
                                                        tmp.append(np.mean(pred[ix, :, ns], axis=0) - y_mfr)
                                                    pred_z_fr[model, i_d, i_ds].append(np.vstack((tmp)))
                                            else:
                                                pred_z_fr[model, i_d, i_ds].append(np.mean(pred[ix_ob, :], axis=0) - y_mfr)
    
                                            obs_added_targ.append(targi)


        ### Now scatter plot all data over all days: 
        #f, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(8, 4))
        ### Make R2 plots to show how much each plot accounts for variace: 
        VAF_mat = np.zeros((ndays, len(xkeys)))
        VAF_mat_shuff = []
        lme_gen_vs_identity = dict(day=[], gen_vs_id=[], r2=[])
        lme_shuff_vs_gen    = dict(day=[], shuf_vs_gen=[], r2=[])
        
        shuff_vs_dyn = np.zeros((ndays))
        #### R2 bar plot ####
        fbar, axbar = plt.subplots(figsize=(5, 5))

        for i_ds, xk in enumerate(xkeys):
            for i_d in range(ndays):

                if xk == 'shuffled':
                    vaf_mat = []
                    assert(len(z_fr[model, i_d, i_ds]) == len(pred_z_fr[model, i_d, i_ds]))
                    
                    ####### For loop #######
                    for nsi in range(nShuff):
                        tru = []
                        pre = []

                        for tmpi in range(len(z_fr[model, i_d, i_ds])):
                            zf = z_fr[model, i_d, i_ds][tmpi]
                            pf = pred_z_fr[model, i_d, i_ds][tmpi]
                            assert(pf.shape[0] == nShuff)
                            if np.sum(np.isnan(pf)) > 0:
                                print('Stopping line 2150')
                                import pdb; pdb.set_trace()

                            tru.append(zf)
                            pre.append(pf[nsi, :])
                        
                        tru = np.vstack((tru))*10
                        pre = np.vstack((pre))*10
                        tru = tru.reshape(-1)
                        pre = pre.reshape(-1)
                        vaf_mat.append(util_fcns.get_R2(tru, pre, pop=True))
                    
                    VAF_mat[i_d, i_ds] = np.mean(vaf_mat)
                    VAF_mat_shuff.append(np.mean(vaf_mat))
                    # print('Stopping line 2163')
                    # import pdb; pdb.set_trace()
                else:
                    ########### ONLY PLOT TRUE Z FR #########
                    x2 = np.vstack(( z_fr[model, i_d, i_ds]))*10 ## Spikes 
                    y2 = np.vstack(( pred_z_fr[model, i_d, i_ds]))*10 ## Spikes 
                    
                    x2 = x2.reshape(-1)
                    y2 = y2.reshape(-1)

                    VAF2 = util_fcns.get_R2(x2, y2, pop = True)
                    VAF_mat[i_d, i_ds] = VAF2
                
                #### LME model saving ###
                if xk in ['dyn_gen', 'identity_dyn']:
                    lme_gen_vs_identity['day'].append(i_d)
                    lme_gen_vs_identity['gen_vs_id'].append(i_ds)
                    lme_gen_vs_identity['r2'].append(VAF2)

                if xk in ['dyn_gen', 'shuffled']:
                    lme_shuff_vs_gen['day'].append(i_d)
                    lme_shuff_vs_gen['shuf_vs_gen'].append(i_ds)
                    if xk == 'dyn_gen':
                        lme_shuff_vs_gen['r2'].append(VAF2)
                    elif xk == 'shuffled':
                        lme_shuff_vs_gen['r2'].append(np.mean(vaf_mat))

                if xk in ['dyn_gen']:
                    ix = np.nonzero(vaf_mat <= VAF_mat[i_d, i_ds])[0]
                    assert(len(vaf_mat) == nShuff)
                    shuff_vs_dyn[i_d] = float(len(ix))/float(len(vaf_mat))

                    ##### PRint p-value for shuff vs. true; 
                    print('Fraction of shuff <= data for day %d, animal %s, p = %.5f' %(i_d, animal, shuff_vs_dyn[i_d] ))

            if xk == 'shuffled':
                util_fcns.draw_plot(i_ds, np.hstack((VAF_mat_shuff)), models_colors[i_ds], 'white', axbar, width = 0.9)
            else:
                axbar.bar(i_ds, np.mean(VAF_mat[:, i_ds]), color=models_colors[i_ds], width=.9)
                axbar.errorbar(i_ds, np.mean(VAF_mat[:, i_ds]), np.std(VAF_mat[:, i_ds])/np.sqrt(ndays), marker='|', color='k')
        axbar.set_xlim([-.5, len(xkeys)])

        ###### Plot lines for each day ######
        for i_d in range(ndays):
            axbar.plot(np.arange(len(xlab)), VAF_mat[i_d, :], '-', color='gray', linewidth=1.)

        ###### Run LME for tsk spec vs. general #####
        print('General vs. Identity dynamics : ')
        pv0, slp0 = util_fcns.run_LME(lme_gen_vs_identity['day'], lme_gen_vs_identity['gen_vs_id'],
            lme_gen_vs_identity['r2'])
        pv0_str = util_fcns.get_pv_str(pv0)

        ##### Plot asterix ###########
        mx = np.max(VAF_mat[:, [1, 2]])
        axbar.plot([1, 2], [1.1*mx, 1.1*mx], 'k-')
        axbar.text(1.5, 1.15*mx, pv0_str, ha='center')
        

        print('Shuff vs. General Dynamics: ')
        pv1, slp1 = util_fcns.run_LME(lme_shuff_vs_gen['day'], lme_shuff_vs_gen['shuf_vs_gen'],
            lme_shuff_vs_gen['r2'])
        pv1_str = util_fcns.get_pv_str(pv1)

        ##### Plot asterix ###########
        mx = np.max(VAF_mat[:, [0, 1]])
        axbar.plot([0, 1], [1.1*mx, 1.1*mx], 'k-')
        axbar.text(.5, 1.15*mx, pv1_str, ha='center')


        #### X labels ########
        axbar.set_xticks(np.arange(len(xlab)))
        axbar.set_xticklabels(xlab, rotation = 45)

        #### Plot stars ####
        if np.all(shuff_vs_dyn == 1.):
            mx = np.max(VAF_mat[:, [0, 1]])
            axbar.plot([0, 1], [1.1*mx, 1.1*mx], 'k-')
            axbar.text(.5, 1.15*mx, '***', ha='center')
        else:
            mx = np.max(VAF_mat[:, [0, 1]])
            axbar.plot([0, 1], [1.1*mx, 1.1*mx], 'k-')
            axbar.text(.5, 1.15*mx, 'n.s.', ha='center')            
        
        axbar.set_ylabel('VAF')
        fbar.tight_layout()
        fbar.savefig(analysis_config.config['fig_dir']+'%s_fig4_%s.svg'%(animal, use_mFR_option))

### Fig 5 #### 
### Mean diffs of action at next time step ###
def plot_real_mean_diffs_behavior_next(model_set_number = 6, min_obs = 15):
    ### Take real task / target / command / neuron / day comparisons for each neuron in the BMI
    ### Plot within a bar 
    ### Plot only sig. different ones

    ### Plot cov. diffs (mat1 - mat2)
    mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))
    fsumm, axsumm = plt.subplots(figsize=(4, 4))

    for ia, animal in enumerate(['grom','jeev']):
        model_dict = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen_match_tsk_N.pkl' %model_set_number, 'rb'))

        if animal == 'grom':
            ndays = 9; 
            width = 3.
        
        elif animal == 'jeev':
            ndays = 4; 
            width = 2.

        ### Now bar plot that: 
        f, ax = plt.subplots(figsize=(width, 4))
        
        days = []; DX = []; DWI = []; 
        cat = []; 
        met = []; 
        mnz = np.zeros((ndays, 2))

        for i_d in range(ndays):

            diffs_x = []; diffs_wi = [];
            tsk  = model_dict[i_d, 'task']
            targ = model_dict[i_d, 'trg']
            push = model_dict[i_d, 'np']
            bin_num = model_dict[i_d, 'bin_num']

            ### This depends on how many "history bins" are included in this model. 
            ### If only 1 unit of history, then this will be = 1. If many units of history, then will be greater; 
            min_bin = np.min(bin_num)
            print('min bin: %d'%min_bin)
            commands_disc = util_fcns.commands2bins([push], mag_boundaries, animal, i_d, vel_ix = [0, 1])[0]

            ### Now go through combos and plot 
            for mag_i in range(4):
                for ang_i in range(8):

                    ### For each target 1 
                    for targi in range(8):
                        ix_co = (commands_disc[:, 0] == mag_i) & (commands_disc[:, 1] == ang_i) & (tsk == 0) & (targ == targi)
                        
                        if len(np.nonzero(ix_co == True)[0]) > min_obs:
                            
                            ### For each target 2 
                            for targi2 in range(8):
                                ix_ob = (commands_disc[:, 0] == mag_i) & (commands_disc[:, 1] == ang_i) & (tsk == 1) & (targ == targi2)

                                if len(np.nonzero(ix_ob == True)[0]) > min_obs:
                                    
                                    ix_co0 = np.nonzero(ix_co == True)[0]
                                    ix_ob0 = np.nonzero(ix_ob == True)[0]

                                    ### Get the NEXT COMMAND ###
                                    ix_co0 = ix_co0 + 1; 
                                    ix_ob0 = ix_ob0 + 1; 

                                    ### Get rid of bins that aren't from prev time step

                                    ### Remove bins that are past hte length of the data; 
                                    ix_co0 = ix_co0[ix_co0 < len(push)]
                                    ix_ob0 = ix_ob0[ix_ob0 < len(push)]

                                    ### Only keep bins that have incremented -- 
                                    ### if a bin is equal to the lowest value "1" then it must have cycled over from previous trial 
                                    ix_co_keep = np.nonzero(bin_num[ix_co0] > min_bin)[0]
                                    ix_co0 = ix_co0[ix_co_keep]

                                    ix_ob_keep = np.nonzero(bin_num[ix_ob0] > min_bin)[0]
                                    ix_ob0 = ix_ob0[ix_ob_keep]

                                    ##### Subselect indices #####
                                    ii = np.random.permutation(len(ix_co0))
                                    i1 = ii[:int(len(ix_co0)/2.)]
                                    i2 = ii[int(len(ix_co0)/2.):]

                                    jj = np.random.permutation(len(ix_ob0))
                                    j1 = jj[:int(len(ix_ob0)/2.)]
                                    j2 = jj[int(len(ix_ob0)/2.):]

                                    ix_co1 = ix_co0[i1]
                                    ix_co2 = ix_co0[i2]
                                    ix_ob1 = ix_ob0[j1]
                                    ix_ob2 = ix_ob0[j2]

                                    assert np.sum(np.isnan(ix_co1)) == np.sum(np.isnan(ix_co2)) == np.sum(np.isnan(ix_ob1)) == np.sum(np.isnan(ix_ob2)) == 0
                                    assert(np.all(tsk[ix_co1] == 0))
                                    assert(np.all(tsk[ix_co2] == 0))
                                    assert(np.all(tsk[ix_ob1] == 1))
                                    assert(np.all(tsk[ix_ob2] == 1))
                                    
                                    ### make the plot: 
                                    diffs_wi.append(np.mean(push[ix_co1, :], axis=0) - np.mean(push[ix_co2, :], axis=0))
                                    diffs_wi.append(np.mean(push[ix_ob1, :], axis=0) - np.mean(push[ix_ob2, :], axis=0))

                                    diffs_x.append(np.mean(push[ix_co1, :], axis=0) - np.mean(push[ix_ob1, :], axis=0))
                                    diffs_x.append(np.mean(push[ix_co2, :], axis=0) - np.mean(push[ix_ob2, :], axis=0))

            W = np.abs(np.hstack((diffs_wi))) # to Hz
            X = np.abs(np.hstack((diffs_x))) 
            DX.append(X); 
            DWI.append(W); 
            tmp = np.hstack((W, X))
            
            ### LME ####
            # days.append(np.zeros_like(tmp) + i_d)
            # cat.append(np.zeros_like(W))
            # cat.append(np.zeros_like(X) + 1)
            # met.append(tmp)
            days.append([i_d, i_d])
            cat.append([0, 1])
            met.append([np.mean(W), np.mean(X)])

            for i, (D, col) in enumerate(zip([W, X], ['gray', 'k'])):
                ax.bar(i_d + i*.45, np.nanmean(D), color=col, edgecolor='none', width=.4, linewidth=1.0)
                ax.errorbar(i_d + i*.45, np.nanmean(D), np.nanstd(D)/np.sqrt(len(D)), marker='|', color=col)
                mnz[i_d, i] = np.nanmean(D)
         
        days = np.hstack((days))
        cat = np.hstack((cat))
        met = np.hstack((met))

        ### look at: run_LME(Days, Grp, Metric):
        pv, slp = util_fcns.run_LME(days, cat, met)
        print 'LME model, fixed effect is day, rand effect is X vs. Wi., N = %d, ndays = %d, pv = %.4f, slp = %.4f' %(len(days), len(np.unique(days)), pv, slp)

        ###
        DX = np.hstack((DX))
        DWI = np.hstack((DWI))

        axsumm.bar(0.4 + ia, np.mean(DX), color='k', edgecolor='none', width=.4, linewidth=2.0, alpha = .8)
        axsumm.bar(0. + ia, np.mean(DWI), color='gray', edgecolor='none', width=.4, linewidth=2.0, alpha =.8)

        for i_d in range(ndays):
            axsumm.plot(np.array([0, .4]) + ia, mnz[i_d, :], '-', color='gray', linewidth=1.0)

        if pv < 0.001: 
            axsumm.text(0.2+ia, np.max(mnz), '***')
        elif pv < 0.01: 
            axsumm.text(0.2+ia, np.max(mnz), '**')
        elif pv < 0.05: 
            axsumm.text(0.2+ia, np.max(mnz), '*')
        else:
            axsumm.text(0.2+ia, np.max(mnz), 'n.s.')

        # ax.set_ylabel('Difference in Hz')
        # ax.set_xlabel('Days')
        # ax.set_xticks(np.arange(ndays))
        # ax.set_title('Monk '+animal[0].capitalize())
        # f.tight_layout()

        ### Need to stats test the differences across populations:    
    axsumm.set_xticks([0.2, 1.2])
    axsumm.set_xticklabels(['G', 'J']) 
    #axsumm.set_ylim([0, 5])
    axsumm.set_ylabel(' Mean Diffs (cm/sec) ') 
    fsumm.tight_layout()
    fsumm.savefig(fig_dir+'both_monks_w_vs_x_task_push_mean_diffs.svg')

### Fig 5 -- identity vs. neural dynamcis
def fig_5_neural_dyn(min_obs = 15, r2_pop = True, 
    model_set_number = 3, ndays = None,):
    
    ### For stats each neuron is an observation ##
    mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))

    ### Now generate plots -- testing w/ 1 day
    if ndays is None:
        ndays = dict(grom=9, jeev=4)
    else:
        ndays = dict(grom=ndays, jeev=ndays)

    fsumm, axsumm = plt.subplots(ncols = 2, nrows = 1, figsize = (4, 4))

    for ia, (animal, yr) in enumerate(zip(['grom', 'jeev'], ['2016', '2013'])):
        
        if model_set_number == 3: 
            models_to_include = [#'prespos_0psh_0spksm_1_spksp_0',
                                 #'prespos_0psh_0spksm_1_spksp_0potent',
                                 #'prespos_0psh_0spksm_1_spksp_0null',
                                 'hist_1pos_0psh_0spksm_1_spksp_0',
                                 'identity']
                                 #'hist_1pos_0psh_1spksm_0_spksp_0']
                                 #'hist_1pos_0psh_0spksm_1_spksp_0potent',
                                 #'hist_1pos_0psh_0spksm_1_spksp_0null']
            models_colors = [[39, 169, 225], [0, 0, 0]]
            xlab = [['$y_{t+1} | y_{t}$'], ['$a_{t+1} | y_{t}$']]
        else: 
            raise Exception

        M = len(models_to_include)
        models_colors = [np.array(m)/256. for m in models_colors]
        
        if r2_pop:
            pop_str = 'Population'
        else:
            pop_str = 'Indiv'        

        ### Go through each neuron and plot mean true vs. predicted (i.e R2) for a given command  / target combo: 
        model_dict = pickle.load(open(analysis_config.config[animal + '_pref'] + 'tuning_models_'+animal+'_model_set%d.pkl'%model_set_number, 'rb'))
            
        R2S = dict(); R2_push = dict()
        ### Super basic R2 plot of neural activity and neural push ###

        for i_d in range(ndays[animal]):

            if animal == 'grom': 
                KG, KG_null_proj, KG_potent_orth = generate_models.get_KG_decoder_grom(i_d)
            
            elif animal == 'jeev':
                KG, KG_null_proj, KG_potent_orth = generate_models.get_KG_decoder_jeev(i_d)

            ###### True data #####
            tdata = model_dict[i_d, 'spks']
            np_data = model_dict[i_d, 'np']
            bin_num = model_dict[i_d, 'bin_num']
            min_bin = np.min(bin_num)

            R2s = dict()

            ###### Get the baseline ####
            for i_mod, mod in enumerate(models_to_include):
                
                if mod == 'identity':
                    ### Get all data points greater than minimum bin; 
                    ix = np.nonzero(bin_num>min_bin)[0]

                    ### set E(y_t) = y_{t-1}
                    pdata = np.zeros_like(tdata) + np.nan 
                    pdata[ix, :] = tdata[ix - 1]

                    ### The only data point we're missing is the one in the beginning; 
                    ixnan = np.nonzero(np.isnan(pdata[:, 0]))
                    assert(np.all(bin_num[ixnan] == 1))

                    np_pred = np.zeros_like(np_data) + np.nan
                    np_pred[ix, :] = np_data[ix - 1, :]

                    ### Get population R2 ### 
                    R2 = util_fcns.get_R2(tdata, pdata, pop = r2_pop, ignore_nans = True)

                    ### Get push ####
                    R2_np = util_fcns.get_R2(np_data, np_pred, pop = r2_pop, ignore_nans = True)
                    #import pdb; pdb.set_trace()

                else:

                    #### Predicted spikes; #######
                    pdata = model_dict[i_d, mod]

                    #### Predicted action | KG*predicted spikes ####
                    np_pred = np.squeeze(np.array(np.dot(KG, pdata.T).T))

                    ### Get population R2 ### 
                    R2 = util_fcns.get_R2(tdata, pdata, pop = r2_pop)

                    ### Get push ####
                    R2_np = util_fcns.get_R2(np_data, np_pred, pop = r2_pop)

                    #import pdb; pdb.set_trace()
                ### Put the R2 in a dictionary: 
                R2S[i_mod, i_d, 0] = R2; 
                R2S[i_mod, i_d, 1] = R2_np; 

            ##### Plot this single day #####
            for q in range(2): 
                tmp = []; 
                for i_mod in range(M):
                    try:
                        tmp.append(np.nanmean(np.hstack((R2S[i_mod, i_d, q]))))
                    except:
                        tmp.append(R2S[i_mod, i_d, q])
                axsumm[q].plot(ia*3 + np.arange(M), tmp, '-', color='gray', linewidth = 1.)

        #### Plots total mean ###
        for q in range(2):
            tmp = []; tmp_e = []; 
            
            for i_mod in range(M):
                tmp2 = []; 
                for i_d in range(ndays[animal]):
                    tmp2.append(R2S[i_mod, i_d, q])
                tmp2 = np.hstack((tmp2))
                tmp2 = tmp2[~np.isnan(tmp2)]

                ## mean 
                tmp.append(np.mean(tmp2))

                ## s.e.m
                tmp_e.append(np.std(tmp2)/np.sqrt(len(tmp2)))

            ### Overal mean 
            for i_mod in range(M):
                axsumm[q].bar(ia*3 + i_mod, tmp[i_mod], color = models_colors[i_mod], edgecolor='k', linewidth = 1., )
                axsumm[q].errorbar(ia*3 + i_mod, tmp[i_mod], yerr=tmp_e[i_mod], marker='|', color='k')        
            axsumm[q].set_ylabel('%s R2, neur '%(pop_str), fontsize=8)

            axsumm[q].set_xticks(np.arange(M))
            axsumm[q].set_xticklabels(xlab[q], rotation=45, fontsize=6)

    fsumm.tight_layout()
    #fsumm.savefig(fig_dir + 'both_%sr2_dyn_model_perc_increase%s_model%d.svg'%(pop_str, perc_increase, model_set_number))

### Fig 5 -- ID vs. neural dynamics on next action diffs. 
def fig_5_neural_dyn_mean_pred(min_obs = 15, r2_pop = True,
    model_set_number = 3, ndays = None, 
    ndiv = 16, center_limit = True, center_rad_limit = 100000, 
    jeev_days = [0, 1, 2, 3]):
    
    ### note that previously jeev-days was [0, 2, 3] -- something about how day 1 only had 16 CO trials 
    
    ### For stats each neuron is an observation ##
    mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))

    ### Now generate plots -- testing w/ 1 day
    if ndays is None:
        ndays = dict(grom=np.arange(9), jeev=jeev_days)
    else:
        ndays = dict(grom=ndays, jeev=ndays)

    ## Plot the 3 different things; (y_t+1 | y_t), (y_)
    fsumm, axsumm = plt.subplots(ncols = 3, nrows = 1, figsize = (8, 4))

    for ia, (animal, yr) in enumerate(zip(['grom', 'jeev'], ['2016', '2013'])):
        
        if model_set_number == 3: 
            models_to_include = ['identity',
                                 'hist_1pos_0psh_0spksm_1_spksp_0']

            models_colors = [[0, 0, 0], [39, 169, 225]]
            xlab = [['$y_{t+1} | y_{t}$'], ['$a_{t+1} | y_{t}$'], ['$\Delta a_{t+1} | y_{t}$']]

        elif model_set_number == 8:
            models_to_include = ['identity',
                                 'hist_1pos_1psh_0spksm_0_spksp_0',
                                 'hist_1pos_3psh_0spksm_0_spksp_0',
                                 'hist_1pos_0psh_0spksm_1_spksp_0']

            models_colors = [[0, 0, 0], [101, 44, 144], [101, 44, 144], [39, 169, 225]]
            xlab = [['$y_{t+1} | y_{t}$'], ['$a_{t+1} | y_{t}$'], ['$\Delta a_{t+1} | y_{t}$']]

        M = len(models_to_include)
        models_colors = [np.array(m)/256. for m in models_colors]
        
        if r2_pop:
            pop_str = 'Population'
        else:
            pop_str = 'Indiv'        

        ### Go through each neuron and plot mean true vs. predicted (i.e R2) for a given command  / target combo: 
        model_dict = pickle.load(open(analysis_config.config[animal + '_pref'] + 'tuning_models_'+animal+'_model_set%d.pkl'%model_set_number, 'rb'))
            
        R2S = dict(); R2_push = dict()
        ### Super basic R2 plot of neural activity and neural push ###

        for i_d in ndays[animal]:

            if animal == 'grom': 
                KG, KG_null_proj, KG_potent_orth = generate_models.get_KG_decoder_grom(i_d)
            elif animal == 'jeev':
                KG, KG_null_proj, KG_potent_orth = generate_models.get_KG_decoder_jeev(i_d)

            ###### True data #####
            tdata = model_dict[i_d, 'spks']
            np_data = model_dict[i_d, 'np']
            target = model_dict[i_d, 'trg']
            task = model_dict[i_d, 'task']

            ### Get commands: 
            commands_disc = util_fcns.commands2bins([np_data], mag_boundaries, animal, i_d, vel_ix = [0, 1])[0]

            bin_num = model_dict[i_d, 'bin_num']
            min_bin = np.min(bin_num)

            R2s = dict()

            ###### Get the baseline ####
            for i_mod, mod in enumerate(models_to_include):
                
                if mod == 'identity':
                    ix = np.nonzero(bin_num>min_bin)[0]
                    pdata = np.zeros_like(tdata) + np.nan 
                    pdata[ix, :] = tdata[ix - 1, :]

                    ixnan = np.nonzero(np.isnan(pdata[:, 0]))
                    assert(np.all(bin_num[ixnan] == 1))

                    np_pred = np.zeros_like(np_data) + np.nan
                    np_pred[ix, :] = np_data[ix - 1, :]

                    ### Get population R2 ### 
                    R2 = util_fcns.get_R2(tdata, pdata, pop = r2_pop, ignore_nans = True)

                    ### Get push ####
                    R2_np = util_fcns.get_R2(np_data, np_pred, pop = r2_pop, ignore_nans = True)

                elif mod in ['hist_1pos_1psh_0spksm_0_spksp_0', 'hist_1pos_3psh_0spksm_0_spksp_0']:
                    ix = np.nonzero(bin_num>min_bin)[0]
                    pdata = np.zeros_like(tdata) + np.nan 

                    ### Get states
                    state_preds = model_dict[i_d, mod][ix - 1, :]

                    ### Propagate by the dynamics model: choose first fold: 
                    model_dyn = model_dict[i_d, 'hist_1pos_0psh_0spksm_1_spksp_0', 0, 'model']
                    pdata[ix, :] = model_dyn.predict(state_preds)

                    ixnan = np.nonzero(np.isnan(pdata[:, 0]))
                    assert(np.all(bin_num[ixnan] == 1))

                    np_pred = np.zeros_like(np_data) + np.nan
                    np_pred[ix, :] = np.dot(KG, pdata[ix, :].T).T

                    ### Get population R2 ### 
                    R2 = util_fcns.get_R2(tdata, pdata, pop = r2_pop, ignore_nans = True)

                    ### Get push ####
                    R2_np = util_fcns.get_R2(np_data, np_pred, pop = r2_pop, ignore_nans = True)

                else:
                    pdata = model_dict[i_d, mod]
                    np_pred = np.squeeze(np.array(np.dot(KG, pdata.T).T))

                    ### Get population R2 ### 
                    R2 = util_fcns.get_R2(tdata, pdata, pop = r2_pop)

                    ### Get push ####
                    R2_np = util_fcns.get_R2(np_data, np_pred, pop = r2_pop)
                
                ### Put the R2 in a dictionary: 
                R2S[i_mod, i_d, 0] = R2; 
                R2S[i_mod, i_d, 1] = R2_np; 
                
                ### Find the mean DIFFS 
                ### Go through the targets (1)
                ### Got through targets (2)
                true_diffs_kg = []; 
                pred_diffs_kg = []; 

                KEEP_IXs = {}

                ####### Only look at the center points < center_rad_limit #######
                if center_limit:
                    rad = np.sqrt(np.sum(model_dict[i_d, 'pos']**2, axis=1))
                    keep_ix = np.nonzero(rad <= center_rad_limit)[0]
                    KEEP_IXs[i_d, 'keep_ix_pos', 0] = keep_ix
                    ncats = 1; 
                
                else:
                    #### Get velocities at t=-1 ###
                    vel_tm1 = model_dict[i_d, 'vel_tm1']; 

                    ### Get angles out from this velocity: 
                    vel_disc = util_fcns.commands2bins([vel_tm1], mag_boundaries, animal, i_d, vel_ix = [0, 1], ndiv = ndiv)[0]

                    ### Here, the second column (vel_disc[:, 1]) has info on the angle: 
                    for ang_i in range(int(ndiv)): 
                        ix = np.nonzero(vel_disc[:, 1] == ang_i)[0]
                        KEEP_IXs[i_d, 'keep_ix_pos', ang_i] = ix; 
                    ncats = int(ndiv);

                #### For each cateogry get the neural push command that falls with in this area #####
                for cat in range(ncats):
                    ix_keep = KEEP_IXs[i_d, 'keep_ix_pos', cat]
                    
                    ##### Get the commands from this category #####
                    sub_commands_disc = commands_disc[ix_keep, :]
                    sub_target = target[ix_keep]
                    sub_task = task[ix_keep]

                    for i_ang in range(8):
                        for i_mag in range(4):
                            for itg in range(10):

                                ix_co = (sub_commands_disc[:, 0] == i_mag) & (sub_commands_disc[:, 1] == i_ang) & (sub_target == itg) & (sub_task == 0)
                                ix_co = np.nonzero(ix_co == True)[0]

                                ix_co_big = ix_keep[ix_co]

                                #### Get the next time step #####
                                ix_co_big = ix_co_big + 1;

                                ### Remove anythign bigger than length of data #####
                                ix_co_big = ix_co_big[ix_co_big < len(bin_num)]

                                #### Remove anything smaller than 2 -- would have to have come over from previous trial #####
                                ix_co_big = ix_co_big[bin_num[ix_co_big] > 1]
                                
                                #### If enougth observations ######
                                if len(ix_co_big) >= min_obs: 
        
                                    ##### Go through the targets #####
                                    for itg2 in np.unique(sub_target):

                                        ##### get the index associated with this target ######
                                        ix_ob = (sub_commands_disc[:, 0] == i_mag) & (sub_commands_disc[:, 1] == i_ang) & (sub_target == itg2) & (sub_task == 1)
                                        ix_ob = np.nonzero(ix_ob == True)[0]
                                        ix_ob_big = ix_keep[ix_ob]

                                        ix_ob_big = ix_ob_big + 1; 
                                        ix_ob_big = ix_ob_big[ix_ob_big < len(bin_num)]
                                        ix_ob_big = ix_ob_big[bin_num[ix_ob_big] > 1]  

                                        if len(ix_ob_big) >= min_obs: 

                                            ###### get the true and predicted differences here ######
                                            true_diffs_kg.append( np.nanmean(np_data[ix_co_big, :], axis=0) - np.nanmean(np_data[ix_ob_big, :], axis=0))
                                            pred_diffs_kg.append( np.nanmean(np_pred[ix_co_big, :], axis=0) - np.nanmean(np_pred[ix_ob_big, :], axis=0))

                ### Get R2: 
                #import pdb; pdb.set_trace()
                R2S[i_mod, i_d, 2] = util_fcns.get_R2(np.vstack((true_diffs_kg)), np.vstack((pred_diffs_kg)), pop=r2_pop)


            ##### Plot this single day #####
            for q in range(3): 
                tmp = []; 
                for i_mod in range(M):
                    try:
                        tmp.append(np.nanmean(np.hstack((R2S[i_mod, i_d, q]))))
                    except:
                        tmp.append(R2S[i_mod, i_d, q])
                axsumm[q].plot(ia*5 + np.arange(M), tmp, '-', color='gray', linewidth = 1.)

        #### Plots total mean ###
        for q in range(3):
            tmp = []; tmp_e = []; 
            
            dayz = []; 
            modz = []; 
            metz = []; 

            for i_mod in range(M):
                tmp2 = []; 
                for i_d in ndays[animal]:
                    tmp2.append(R2S[i_mod, i_d, q])

                    dayz.append(i_d)
                    modz.append(R2S[i_mod, i_d, q])
                    metz.append(i_mod)

                tmp2 = np.hstack((tmp2))
                tmp2 = tmp2[~np.isnan(tmp2)]

                ## mean 
                tmp.append(np.mean(tmp2))

                ## s.e.m
                tmp_e.append(np.std(tmp2)/np.sqrt(len(tmp2)))

            dayz = np.hstack((dayz))
            modz = np.hstack((modz))
            metz = np.hstack((metz))

            pv, slp = util_fcns.run_LME(dayz, modz, metz)
            print('Animal: %s, Plot %d, PV: %.2f, SLP: %.2f' %(animal, q, pv, slp))

            ### Overal mean 
            for i_mod in range(M):
                axsumm[q].bar(ia*5 + i_mod, tmp[i_mod], color = models_colors[i_mod], edgecolor='k', linewidth = 1., )
                axsumm[q].errorbar(ia*5 + i_mod, tmp[i_mod], yerr=tmp_e[i_mod], marker='|', color='k')        
            axsumm[q].set_ylabel('%s R2, neur'%(pop_str))

            axsumm[q].set_xticks(np.arange(M))
            axsumm[q].set_xticklabels(xlab[q], rotation=45)

    fsumm.tight_layout()
    #fsumm.savefig(fig_dir + 'both_%sfig_5_neural_dyn_mean_pred_model%d.svg'%(pop_str, model_set_number))
 
def fig_5_cond_spec_next_action_scatter_bars(use_mvel_option = 'command_axis_spec', 
        scat_animal = 'grom', scat_day = 0, model_dicts=None, min_obs = 15,
        gen_vs_tsk_instead_of_shuff = False):
    '''
    copying style from Fig 4 --> can dynamics plot the condition-specific next action well? 
    use_mvel_option -- 
        -- command_axis_spec: for x axis use mean_vel | (day, command)
        --                    for y axis use pred_vel | (day, command)
    '''

    mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))
    
    ### No conditioning, just dynamics #####
    model = 'hist_1pos_0psh_0spksm_1_spksp_0'

    if gen_vs_tsk_instead_of_shuff:
        xkeys = ['dyn_gen', 'dyn_tsk']
        xlab = ['gen. dyn.', 'tsk-spec.\n dyn.']
        models_colors = [[100, 100, 100], 
                     [255, 0, 0]]
    else:
        ### Scatters for these condtions 
        xkeys = ['shuffled', 'dyn_gen']
        xlab = ['shuffled', 
            'general\ndynamics']
    
        models_colors = [[100, 100, 100], 
                     [255, 0, 0]]
    
    models_colors = [np.array(m)/255. for m in models_colors]
    model_set_number = 6; 
    fbar, axbar = plt.subplots(figsize =(4, 5))

    for i_a, animal in enumerate(['grom', 'jeev']):
        ############ Load the model #############
        if model_dicts is None:
            model_dict = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen_match_tsk_N.pkl' %model_set_number, 'rb'))
        else:
            model_dict = model_dicts[animal]

        ############ Number of days ##############
        ndays = analysis_config.data_params[animal + '_ndays']
        
        nbars_offs = 3*i_a

        z_vel = {}
        pred_z_vel = {}; 
        
        ######### Setup plot #########
        VAF_shuff = []
        vaf_shuff_mn = np.zeros((ndays))
        vaf_dyn = dict()
        for k in xkeys:
            vaf_dyn[k] = np.zeros((ndays))

        for i_d in range(ndays):

            ##### Get kalman gain ####
            KG = util_fcns.get_decoder(animal, i_d)

            ###### True data #####
            pred_A = dict()

            ######### General dynamics conditioned on action #########
            pred_A['dyn_gen'] =  np.dot(KG, model_dict[i_d, model][:, :, 2].T).T ### T x 2 
            
            ######## Task specifci ##############
            tsk_spec = np.zeros_like(model_dict[i_d, model][:, :, 2])
            for tsk in range(2):
                ix_tsk = np.nonzero(model_dict[i_d, 'task'] == tsk)[0]
                tsk_spec[ix_tsk, :] = model_dict[i_d, model][ix_tsk, :, tsk]
            pred_A['dyn_tsk'] = np.dot(KG, tsk_spec.T).T ### Tx 2

            ######### Get shuffled ###############
            if gen_vs_tsk_instead_of_shuff:
                pass
            else:
                shuff_Y = get_shuffled_data(animal, i_d, model)
                nShuff = shuff_Y.shape[2]
                shuff_tmp = []

                for ns in range(nShuff):
                    shuff_tmp.append(np.dot(KG, shuff_Y[:, :, ns].T).T)
                pred_A['shuffled'] = np.dstack((shuff_tmp))

            ############# For real model ############
            ### Get the spiking data
            ### Get the task parameters
            tsk  = model_dict[i_d, 'task']
            targ = model_dict[i_d, 'trg']
            push = model_dict[i_d, 'np']
            bin_num = model_dict[i_d, 'bin_num']
            bin_min = np.min(bin_num)
                
            assert(pred_A['dyn_gen'].shape[0] == len(tsk) == len(targ) == push.shape[0])
            assert(push.shape[1] == 2)

            for i_ds, xk in enumerate(xkeys):
                
                ### Get the correct predictions 
                pred = pred_A[xk]

                #### Setup the dictionaries for stats to be tracked ####
                z_vel[model, i_d, i_ds] = []
                pred_z_vel[model, i_d, i_ds] = []

                ### Get the discretized commands
                commands_disc = util_fcns.commands2bins([push], mag_boundaries, animal, i_d, vel_ix = [0, 1])[0]
                assert(commands_disc.shape == push.shape)

                ############ Now go through combos and plot ##################
                for mag_i in range(4):
                    for ang_i in range(8):

                        ############ Make sure ix is clear to compute ix_tp1 #########
                        ix = np.nonzero(np.logical_and(commands_disc[:, 0] == mag_i, commands_disc[:, 1] == ang_i))[0]

                        ########### Remove last command #####################
                        ix = ix[ix < len(commands_disc) - 1]
                        ix_tp1 = ix + 1

                        ############ Remove commmands whose bin_num is 1 #################
                        ix_keep = np.nonzero(bin_num[ix_tp1] != bin_min)[0]
                        ix = ix[ix_keep]
                        ix_tp1 = ix_tp1[ix_keep]
                        assert(np.all(ix_tp1 - 1 == ix))
                        assert(np.all(bin_num[ix_tp1] > bin_min ))

                        if len(ix) > min_obs:

                            ############ mean NEXT vel commandfor a given command ############
                            command_mn_vel = np.mean(push[ix_tp1, :], axis=0)

                            if xk == 'shuffled':
                                pred_command_mn_vel = []

                                ### Next time point predicted command ####
                                for ns in range(nShuff):
                                    tmp = np.mean(pred[ix_tp1, :, ns], axis=0)
                                    if np.sum(np.isnan(tmp)) > 0:
                                        print("Stopping line 2009")
                                        import pdb; pdb.set_trace()
                                    pred_command_mn_vel.append(tmp)
                            else:
                                pred_command_mn_vel = np.mean(pred[ix_tp1, :], axis=0)

                            if use_mvel_option == 'command_axis_spec':
                                x_mvl = command_mn_vel.copy()
                                y_mvl = copy.copy(pred_command_mn_vel)
                            
                            ### For this option only plot the
                            for i_tsk in range(2): 
                                for targi in np.unique(targ):
                                    
                                    ### Get co / task 
                                    ix_tsk = (commands_disc[:, 0] == mag_i) & (commands_disc[:, 1] == ang_i) & (tsk == i_tsk) & (targ == targi)
                                    ix_tsk = np.nonzero(ix_tsk == True)[0]

                                    ####### Get ix_co_tp1 ######
                                    ix_tsk = ix_tsk[ix_tsk < len(commands_disc) - 1]
                                    ix_tsk_tp1 = ix_tsk+1

                                    ix_keep = np.nonzero(bin_num[ix_tsk_tp1] > bin_min)[0]
                                    ix_tsk = ix_tsk[ix_keep]
                                    ix_tsk_tp1 = ix_tsk_tp1[ix_keep]

                                    assert(np.all(ix_tsk_tp1 - 1 == ix_tsk))
                                    assert(np.all(bin_num[ix_tsk_tp1] > bin_min ))


                                    for ixtmp in ix_tsk: assert(ixtmp in ix)
                                    assert(len(ix_tsk) <= len(ix))

                                    assert(np.all(commands_disc[ix_tsk, 0] == mag_i))
                                    assert(np.all(commands_disc[ix_tsk, 1] == ang_i))
                                    assert(np.all(tsk[ix_tsk] == i_tsk))
                                    assert(np.all(targ[ix_tsk] == targi))

                                    
                                    if len(ix_tsk_tp1) > min_obs:

                                        #### Keep the mean CO condition specific command
                                        z_vel[model, i_d, i_ds].append(np.mean(push[ix_tsk_tp1, :], axis=0) - x_mvl)

                                        if xk == 'shuffled':
                                            if use_mvel_option == 'command_axis_spec':
                                                tmp = []; 

                                                ####### only add a single shuffled ########
                                                for ns in range(nShuff):
                                                    tmpi = np.mean(pred[ix_tsk_tp1, :, ns], axis=0) - y_mvl[ns]
                                                    if np.sum(np.isnan(tmpi)) > 0: 
                                                        print('Stopping lin 2071')
                                                        import pdb; pdb.set_trace()
                                                    tmp.append(tmpi)
                                                pred_z_vel[model, i_d, i_ds].append(np.vstack((tmp)))
                                        else:
                                            pred_z_vel[model, i_d, i_ds].append(np.mean(pred[ix_tsk_tp1, :], axis=0) - y_mvl)

            ####### Plot scattter ######
            for i_ds, xk in enumerate(xkeys):
                ##### make sure thse are the same size; 
                assert(len(pred_z_vel[model, i_d, i_ds]) == len(z_vel[model, i_d, i_ds]))
                NT = len(pred_z_vel[model, i_d, i_ds])
                
                x_tru =  np.vstack((z_vel[model, i_d, i_ds]))  
                x_tru = x_tru.reshape(-1)

                if xk == 'shuffled':
                    vaf = []
                    for ns in range(nShuff):
                        x_pred = np.vstack((pred_z_vel[model, i_d, i_ds][i][ns, :] for i in range(NT)))
                        x_pred = x_pred.reshape(-1)
                        assert(x_tru.shape == x_pred.shape)
                        vaf.append(util_fcns.get_R2(x_tru, x_pred, pop = True))
                    VAF_shuff.append(np.hstack((vaf)))
                    vaf_shuff_mn[i_d] = np.mean(np.hstack((vaf)))

                else:
                    x_pred = np.vstack((pred_z_vel[model, i_d, i_ds]))
                    x_pred = x_pred.reshape(-1)
                    assert(x_tru.shape == x_pred.shape)
                    vaf = util_fcns.get_R2(x_tru, x_pred, pop = True)
                    vaf_dyn[xk][i_d] = vaf 

                if animal == scat_animal and i_d == scat_day:
                    if xk == 'shuffled':
                        x_pred = np.vstack((pred_z_vel[model, i_d, i_ds][i][0, :] for i in range(NT)))
                    else: 
                        x_pred = np.vstack((pred_z_vel[model, i_d, i_ds]))
                    x_pred = x_pred.reshape(-1)
                    vaf_scat = util_fcns.get_R2(x_tru, x_pred, pop = True)
                    ##### Figure #######
                    fscat, axscat = plt.subplots(figsize = (4, 4))
                    axscat.axis('square')

                    axscat.plot(x_tru, x_pred, 'k.', markersize=2.)
                    axscat.set_xlim([-1.5, 1.5])
                    axscat.set_ylim([-1.5, 1.5])
                    axscat.plot([-1.5, 1.5], [-1.5, 1.5], '-', color='gray', linewidth=2.)

                    slp,intc,rv,pv,err = scipy.stats.linregress(x_tru, x_pred)
                    x_ = np.linspace(np.min(x_tru), np.max(x_tru), 10)
                    y_ = x_*slp + intc 
                    axscat.plot(x_, y_, 'k--')
                    axscat.set_title('%s, day %d, VAF %.4f\n Mod: %s' %(animal, i_d, vaf_scat, xk))
                    fscat.tight_layout()
                    fscat.savefig(analysis_config.config['fig_dir']+'%s_day%d_fig5_cond_spec_next_time_scatter_day_%s.svg'%(animal, i_d, xk))

        ########### Bar plot ###########
        if gen_vs_tsk_instead_of_shuff:
            axbar.bar(0+nbars_offs, np.mean(vaf_dyn[xkeys[0]]), color=models_colors[1], width = 0.9)
            axbar.errorbar(0+nbars_offs, np.mean(vaf_dyn[xkeys[0]]), np.std(vaf_dyn[xkeys[0]])/np.sqrt(len(vaf_dyn[xkeys[0]])), marker = '|', color = 'k') 
        else:
            util_fcns.draw_plot(0+nbars_offs, np.hstack((VAF_shuff)), models_colors[0], 'white', axbar, width = 0.9)
        axbar.bar(1+nbars_offs, np.mean(vaf_dyn[xkeys[1]]), color=models_colors[1], width = 0.9)
        axbar.errorbar(1+nbars_offs, np.mean(vaf_dyn[xkeys[1]]), np.std(vaf_dyn[xkeys[1]])/np.sqrt(len(vaf_dyn[xkeys[1]])), marker = '|', color = 'k')
        
        lme_dict = dict(day=[], grp = [], val = [])

        for i_d in range(ndays):
            if gen_vs_tsk_instead_of_shuff:
                axbar.plot([0+nbars_offs, 1+nbars_offs], [ vaf_dyn[xkeys[0]][i_d], vaf_dyn[xkeys[1]][i_d]], '-', color='gray', linewidth=1.)
                lme_dict['day'].append([i_d, i_d])
                lme_dict['grp'].append([0, 1])
                lme_dict['val'].append(vaf_dyn[xkeys[0]][i_d])
                lme_dict['val'].append(vaf_dyn[xkeys[1]][i_d])
            else:
                axbar.plot([0+nbars_offs, 1+nbars_offs], [vaf_shuff_mn[i_d], vaf_dyn[xkeys[1]][i_d]], '-', color='gray', linewidth=1.)
        
        axbar.set_xlim([-.5, 4.5])
        axbar.set_xticks([0, 1])
        axbar.set_xticklabels(xlab, rotation=45)

        ####### Plot significance #######
        if gen_vs_tsk_instead_of_shuff:
            pv, slp = util_fcns.run_LME(np.hstack((lme_dict['day'])), np.hstack((lme_dict['grp'])), np.hstack((lme_dict['val'])))
            pv_str = util_fcns.get_pv_str(pv)
            mx = np.hstack((vaf_dyn[xkeys[0]], vaf_dyn[xkeys[1]] )) 
            axbar.plot([0+nbars_offs, 1+nbars_offs], [1.1*np.max(mx), 1.1*np.max(mx)], 'k-')
            axbar.text(0.5+nbars_offs, 1.15*np.max(mx), pv_str, ha='center')
            
        else:
            dyn_gt_shuff = np.zeros((ndays))
            dyn_lme = dict(day = [], grp = [], val = [])

            for i_d in range(ndays):

                ##### For each day #####
                vaf_dyn_day = vaf_dyn['dyn_gen'][i_d]
                vaf_shuff_day = VAF_shuff[i_d]

                dyn_lme['day'].append([i_d, i_d])
                dyn_lme['grp'].append([0, 1])
                dyn_lme['val'].append([np.mean(vaf_shuff_day), vaf_dyn_day])

                if np.all(vaf_shuff_day < vaf_dyn_day):
                    dyn_gt_shuff[i_d] = 1.
                ix = np.nonzero(vaf_shuff_day >= vaf_dyn_day)[0]
                print('Animal %s, Day %d, Perc shuff get vaf_dyn_day %.5f' %(animal, i_d, float(len(ix))/float(len(vaf_shuff_day))))

            print('Animal %s, LME Dyn vs. shuffle' %(animal))
            pv, slp = util_fcns.run_LME(dyn_lme['day'], dyn_lme['grp'], dyn_lme['val'])
            pv_str = util_fcns.get_pv_str(pv)
            mx = np.max(vaf_dyn['dyn_gen'])
            axbar.plot([0+nbars_offs, 1+nbars_offs], [1.1*mx, 1.1*mx], 'k-')
            axbar.text(.5+nbars_offs, 1.15*mx, pv_str, ha ='center')
        
        fbar.tight_layout()

    if gen_vs_tsk_instead_of_shuff:
        fbar.savefig(analysis_config.config['fig_dir']+'fig5_cond_spec_next_time_act_gen_vs_tskspec.svg')
    else:
        fbar.savefig(analysis_config.config['fig_dir']+'fig5_cond_spec_next_time_act.svg')

### Generalization of Neural dynamics across tasks #####
### Generalization of neural dynamics -- population R2 ### -- figure 6(?)
def plot_r2_bar_model_7_gen(model_set_number = 7, ndays = None, use_action = False,
    plot_by_day = False, sep_R2_by_tsk = False, match_task_spec_n = False, 
    fit_intercept = True, skip_across = False):
    
    ''' 
    not presently sure if including action means "current action" or previous action 
        -- remember that including action forced the model to represent potent action accurately. 

    updates 5/18/20 -- made it so that also computed R2 on each day, not for each task separately
    updates 6/3/20 -- added "match_task_spec_n" so that it selects for models where the same amt 
        of data is used for each task spec. model  
    '''
    ########### FIGS ##########
    # fco, axco = plt.subplots(ncols = 2, nrows = 2, figsize = (4, 4))
    # fob, axob = plt.subplots(ncols = 2, nrows = 2, figsize = (4, 4))
    fbth, axbth = plt.subplots(ncols = 2, nrows = 2, figsize = (6, 8))

    if use_action:
        models_to_include = ['hist_1pos_0psh_1spksm_1_spksp_0']
    else:
        models_to_include = ['hist_1pos_0psh_0spksm_1_spksp_0']#, 
                             #'hist_1pos_4psh_0spksm_1_spksp_0']

    model_colors = [[39, 169, 225],[255, 0, 0]]
    model_colors = [np.array(m)/255. for m in model_colors]

    if ndays is None:
        ndays_none = True
    else:
        ndays_none = False

    for i_a, animal in enumerate(['grom', 'jeev']):
        DAYs = []; 

        RX = []; 
        RW = []; 
        RGEN = []; 

        RX_A = []; 
        RW_A = []; 
        RGEN_A = [];

        f, ax = plt.subplots(nrows = 2, ncols =2 )
        colors = ['b','r','k']
        alphas = [1.,1.,1.]

        ###### Using the task specific models and the general models ######
        ###### Here, these models aren't directly comparable because diff tasks can use diff amounts of data for training; ####
        if match_task_spec_n:
            if fit_intercept:
                dat = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen_match_tsk_N.pkl' %(model_set_number), 'rb'))
            else:
                dat = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen_match_tsk_N_no_intc.pkl' %(model_set_number), 'rb'))
        else:
            if fit_intercept:
                dat = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen.pkl' %(model_set_number), 'rb'))
            else:
                dat = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen_fit_intc.pkl' %(model_set_number), 'rb'))
        
        if ndays_none:
            if animal == 'grom':
                ndays_all = np.arange(9);
                ndays = 9; 
            elif animal == 'jeev':
                ndays_all = [0, 1, 2, 3];
                ndays = 4; 

        #####################
        ### Real bar plot ###
        #####################
        ### 3 plots -- one for CO, one for OB, one combined
        ### CO/OBS/ALL data  x Fit CO/OBS/GEN x ndays
        R2s_plot_spks = np.zeros((2, 3, ndays))
        R2s_plot_acts = np.zeros((2, 3, ndays)) 

        #######################################
        ##### Summary X vs. WIN bar plot ######
        #######################################
        ### Here we want to do within task vs. across task R2 -- combined bar plot: 
        pwi = np.zeros((ndays, 2, 2)) # days  x co/obs x spks/act
        px = np.zeros((ndays, 2, 2))
        pall = np.zeros((ndays, 2, 2))

        pwi_comb = np.zeros((ndays, 2)) # days x spks/act
        px_comb = np.zeros((ndays, 2))
        pall_comb = np.zeros((ndays, 2))

        for i_d, nd in enumerate(ndays_all):

            true_spks = []; true_acts = []; 
            w_pred_spks = []; w_pred_acts = []; 
            x_pred_spks = []; x_pred_acts = []; 
            gen_pred_spks = []; gen_pred_acts = []; 

            ### Split by task -- assess R2 separately:  
            for i_t in range(2):

                ### task specific indices: 
                task_ix = np.nonzero(dat[(nd, 'task')] == i_t)[0]
                predict_key = 'spks'

                #### Get info about task / push / spks
                task = dat[(nd, 'task')][task_ix]
                push = dat[(nd, 'np')][task_ix, :]
                spks = dat[(nd, 'spks')][task_ix, :]

                #### Get the true spikes to line up; 
                true_spks.append(spks.copy())
                true_acts.append(push.copy())

                mn_spks = np.mean(dat[(nd, 'spks')], axis=0)
                mn_push = np.mean(dat[(nd, 'np')], axis=0)

                #if predict_key == 'spks':
                truth = spks.copy(); 
                truth_mn = mn_spks.copy()
                
                # elif predict_key == 'psh':
                #     truth = push.copy() 
                #     truth_mn = mn_push.copy() 

                ### Get the KG: 
                if animal == 'grom':
                    KG, _, _ = generate_models.get_KG_decoder_grom(nd)
                
                elif animal == 'jeev':
                    KG, _, _ = generate_models.get_KG_decoder_jeev(nd)
                    KG = np.squeeze(np.array(KG))

                ### convert this to discrete commands
                for i_m, mod in enumerate(models_to_include):

                    ### for different task 
                    data_mod = dat[(nd, mod)][task_ix, :, :] ### T x N x 3 [train on task 0, train on task 1, train on both]

                    ### Put this data away for later assessmetn; 
                    if i_m == 0:
                        w_pred_spks.append(data_mod[:, :, i_t].copy())
                        other_task = np.mod(i_t + 1, 2)
                        x_pred_spks.append(data_mod[:, :, other_task].copy())
                        gen_pred_spks.append(data_mod[:, :, 2].copy())

                        ##### Add the actions ###
                        w_pred_acts.append(np.dot(KG, data_mod[:, :, i_t].T).T)
                        x_pred_acts.append(np.dot(KG, data_mod[:, :, other_task].T).T)
                        gen_pred_acts.append(np.dot(KG, data_mod[:, :, 2].T).T)

                    ### Iterate through the 3 model possibilities; 
                    for trg_ix, (tsk_col, tsk_alph) in enumerate(zip(colors, alphas)):

                        ### Data fit on models from trg_ix -- shouldn't be any nans; 
                        R2 = util_fcns.get_R2(truth, data_mod[:, :, trg_ix], pop = True) #, ignore_nans = True)

                        ### Data from task i_t, fit on trg_ix, on day: 
                        R2s_plot_spks[i_t, trg_ix, i_d] = R2; 

                        if plot_by_day:
                            xaxis = i_d*4 + trg_ix
                        else:
                            xaxis = i_d + 9*trg_ix 

                        ### Data from task i_t, fit on trg_ix, on day: 
                        ax[i_t, 0].bar(xaxis, R2, color=tsk_col)
                        ax[i_t, 0].set_title('R2 Neural Activity, \n %s' %mod)
                        
                        if predict_key == 'spks':
                            ###### Estimated Push ######
                            R2 = util_fcns.get_R2(push, np.dot(KG, data_mod[:, :, trg_ix].T).T, pop = True, ignore_nans = True)
                            R2s_plot_acts[i_t, trg_ix, i_d] = R2;

                            ax[i_t, 1].bar(xaxis, R2, color=tsk_col)
                            ax[i_t, 1].set_title('R2 Push Activity, \n %s' %mod)

                    #### For these 3 possibilities -- figure out which is which; 
                    #### Within task training; 
                    pwi[i_d, i_t, 0] = R2s_plot_spks[i_t, i_t, i_d]
                    pwi[i_d, i_t, 1] = R2s_plot_acts[i_t, i_t, i_d]
                    
                    non_tsk = np.mod(i_t + 1, 2)
                    px[i_d, i_t, 0] = R2s_plot_spks[i_t, non_tsk, i_d]
                    px[i_d, i_t, 1] = R2s_plot_acts[i_t, non_tsk, i_d]

                    pall[i_d, i_t, 0] = R2s_plot_spks[i_t, 2, i_d]
                    pall[i_d, i_t, 1] = R2s_plot_acts[i_t, 2, i_d]

            #### Combine over tasks; ####
            true_spks = np.vstack((true_spks))
            w_pred_spks = np.vstack((w_pred_spks))
            x_pred_spks = np.vstack((x_pred_spks))
            gen_pred_spks = np.vstack((gen_pred_spks))

            true_acts = np.vstack((true_acts))
            w_pred_acts = np.vstack((w_pred_acts))
            x_pred_acts = np.vstack((x_pred_acts))
            gen_pred_acts = np.vstack((gen_pred_acts))

            ### Add this to the overall thing; 
            pwi_comb[i_d, 0] = util_fcns.get_R2(true_spks, w_pred_spks) # days x spks/act
            px_comb[i_d, 0] = util_fcns.get_R2(true_spks, x_pred_spks) 
            pall_comb[i_d, 0] = util_fcns.get_R2(true_spks, gen_pred_spks) 

            pwi_comb[i_d, 1] = util_fcns.get_R2(true_acts, w_pred_acts) # days x spks/act
            px_comb[i_d, 1] = util_fcns.get_R2(true_acts, x_pred_acts) 
            pall_comb[i_d, 1] = util_fcns.get_R2(true_acts, gen_pred_acts) 

        ### Entries here are day x task (of data)
        PWI = [pwi[:, :, 0].reshape(-1), pwi[:, :, 1].reshape(-1), ]
        PX = [px[:, :, 0].reshape(-1), px[:, :, 1].reshape(-1) ]
        PALL = [pall[:, :, 0].reshape(-1), pall[:, :, 1].reshape(-1)]

        #### Plot figure ####
        tis = ['Neural', 'Action']

        ### Z is neural vs. action #####
        for z in range(2):
            pwii = PWI[z]
            pxi = PX[z]
            palli = PALL[z]

            ### CO data figure ###
            # axco[z, i_a].bar(0, np.mean(R2s_plot_spks[0, 0, :]), color = 'w', edgecolor = 'k', width = .4, linewidth = 2.)
            # axco[z, i_a].errorbar(0, np.mean(R2s_plot_spks[0, 0, :]), yerr=np.std(R2s_plot_spks[0, 0, :])/np.sqrt(ndays), color = 'k', marker='|')
            # axco[z, i_a].bar(0.4, np.mean(R2s_plot_spks[0, 1, :]), color = 'grey', edgecolor = 'k', width = .4, linewidth = 2.)
            # axco[z, i_a].errorbar(0.4, np.mean(R2s_plot_spks[0, 1, :]), yerr=np.std(R2s_plot_spks[0, 1, :])/np.sqrt(ndays), color = 'k', marker='|')

            # axob[z, i_a].bar(0., np.mean(R2s_plot_spks[1, 1, :]), color = 'grey', edgecolor = 'k', width = .4, linewidth = 2.)
            # axob[z, i_a].errorbar(0, np.mean(R2s_plot_spks[1, 1, :]), yerr=np.std(R2s_plot_spks[1, 1, :])/np.sqrt(ndays), color = 'k', marker='|')
            # axob[z, i_a].bar(0.4, np.mean(R2s_plot_spks[1, 0, :]), color = 'w', edgecolor = 'k', width = .4, linewidth = 2.)
            # axob[z, i_a].errorbar(0.4, np.mean(R2s_plot_spks[1, 0, :]), yerr=np.std(R2s_plot_spks[1, 0, :])/np.sqrt(ndays), color = 'k', marker='|')

            #### Figure w/ combo ####
            #### GENERAL / WITHIN / ACROSS ####
            axbth[z, i_a].bar(0, np.mean(palli), color= model_colors[z], edgecolor='k', width=.4*.9)
            axbth[z, i_a].errorbar(0, np.mean(palli), np.std(palli)/np.sqrt(len(palli)), marker='|', color = 'k')

            axbth[z, i_a].bar(0.4, np.mean(pwii), color=model_colors[z], edgecolor='k', width=.4*.9)
            axbth[z, i_a].errorbar(0.4, np.mean(pwii), np.std(pwii)/np.sqrt(len(pwii)), marker='|', color = 'k')
            
            if skip_across:
                pass
            else:
                axbth[z, i_a].bar(0.8, np.mean(pxi), color='k', edgecolor='k', width=.4*.9, alpha = 0.5)
            
            # axbth[z, i_a].bar(0, np.mean(pwii), color='w', edgecolor='k', width=.4, linewidth=.2)
            # axbth[z, i_a].bar(0.4, np.mean(pxi), color='grey', edgecolor='k', width=.4, linewidth=.2)
            # axbth[z, i_a].bar(0.8, np.mean(palli), color='k', edgecolor='k', width=.4, linewidth=.2)
            
        
            for i_d in range(ndays):
                # axco[z, i_a].plot([0, .4], [R2s_plot_spks[0, 0, i_d], R2s_plot_spks[0, 1, i_d]], 'k-', linewidth = 1.)
                # axob[z, i_a].plot([0, .4], [R2s_plot_spks[1, 1, i_d], R2s_plot_spks[1, 0, i_d]], 'k-', linewidth = 1.)

                ### For CO vs. OBS data
                for x in range(2):

                    #### PWII/PXI/PALLI are all NDAYS x 2 --> [day0_0, day0_1, day1_0, day1_1, ...]
                    ### here 
                    if x == 0:
                        #axbth[z, i_a].plot([0, .4, .8], [pwii[2*i_d + x], pxi[2*i_d + x], palli[2*i_d + x]], 'k-', linewidth = 1.)
                        if skip_across:
                            axbth[z, i_a].plot([0, .4], [palli[2*i_d + x], pwii[2*i_d + x],  ], '-', color = 'gray', linewidth = 1.)
                        else:
                            axbth[z, i_a].plot([0, .4, .8], [palli[2*i_d + x], pwii[2*i_d + x], pxi[2*i_d + x]], '-', color = 'gray', linewidth = 1.)
                    elif x == 1:
                        #axbth[z, i_a].plot([0, .4, .8], [pwii[2*i_d + x], pxi[2*i_d + x], palli[2*i_d + x]], 'b-', linewidth = 1.)
                        if skip_across:
                            axbth[z, i_a].plot([0, .4], [palli[2*i_d + x], pwii[2*i_d + x],   ], '-', color = 'gray', linewidth = 1.)
                        else:
                            axbth[z, i_a].plot([0, .4, .8], [palli[2*i_d + x], pwii[2*i_d + x], pxi[2*i_d + x]], '-', color = 'gray', linewidth = 1.)
                    #axbth[z, i_a].plot([0, .4,], [pwii[2*i_d + x], pxi[2*i_d + x]], 'k-', linewidth = 1.)

                    DAYs.append(i_d)

                    if z == 0:
                        RX.append(px[i_d, x, 0])
                        RW.append(pwi[i_d, x, 0])
                        RGEN.append(pall[i_d, x, 0])

                    elif z == 1:
                        RX_A.append(px[i_d, x, 1])
                        RW_A.append(pwi[i_d, x, 1])
                        RGEN_A.append(pall[i_d, x, 1])


            #axbth[z, i_a].set_title('%s, Monk %s'%(tis[z], animal),fontsize = 8)
            axbth[z, i_a].set_ylabel('%s R2' %tis[z])
            if skip_across:
                axbth[z, i_a].set_xticks([0., .4])
                axbth[z, i_a].set_xticklabels(['gen. dyn.', 'tsk-spec\ndyn.'], rotation=45)
            else:
                axbth[z, i_a].set_xticks([0., .4, .8])
                axbth[z, i_a].set_xticklabels(['General', 'Within Task', 'Across Task'])
        
        ### stats: 
        if skip_across:
            pass
        else:
            print 'Neural, subj %s, w vs X' %(animal)
            w_vs_x = np.hstack(( np.hstack((RW)), np.hstack((RX)) ))
            grps = np.hstack(( np.zeros_like(RW), np.zeros_like(RX)+1))
            pv, slp = util_fcns.run_LME(DAYs, grps, w_vs_x)
            print('pv: %.2f, slp %.2f'%(pv, slp))

            if pv < 0.001: 
                axbth[0, i_a].text(0.4, np.max(w_vs_x), '***')
            elif pv < 0.01: 
                axbth[0, i_a].text(0.4, np.max(w_vs_x), '**')
            elif pv < 0.05:
                axbth[0, i_a].text(0.4, np.max(w_vs_x), '*')
            else:
                axbth[0, i_a].text(0.4, np.max(w_vs_x), 'n.s\npv=%.3f'%(pv))

        print 'Neural, subj %s, w vs GEN' %(animal)
        w_vs_g = np.hstack(( np.hstack((RW)), np.hstack((RGEN)) ))
        grps = np.hstack(( np.zeros_like(RW), np.zeros_like(RGEN)+1))
        pv, slp = util_fcns.run_LME(DAYs, grps, w_vs_g)
        print('pv: %.2f, slp %.2f'%(pv, slp))

        if pv < 0.001: 
            axbth[0, i_a].text(0.2, np.max(w_vs_g), '***')
        elif pv < 0.01: 
            axbth[0, i_a].text(0.2, np.max(w_vs_g), '**')
        elif pv < 0.05:
            axbth[0, i_a].text(0.2, np.max(w_vs_g), '*')
        else:
            axbth[0, i_a].text(0.2, np.max(w_vs_g), 'n.s\npv=%.3f'%(pv))


        if skip_across:
            pass
        else:
            print 'ACTION, subj %s, w vs x' %(animal)
            w_vs_x = np.hstack(( np.hstack((RW_A)), np.hstack((RX_A)) ))
            grps = np.hstack(( np.zeros_like(RW_A), np.zeros_like(RX_A)+1))
            pv, slp = util_fcns.run_LME(DAYs, grps, w_vs_x)
            print('pv: %.2f, slp %.2f'%(pv, slp))

            if pv < 0.001: 
                axbth[1, i_a].text(0.4, np.max(w_vs_x), '***')
            elif pv < 0.01: 
                axbth[1, i_a].text(0.4, np.max(w_vs_x), '**')
            elif pv < 0.05:
                axbth[1, i_a].text(0.4, np.max(w_vs_x), '*')
            else:
                axbth[1, i_a].text(0.4, np.max(w_vs_x), 'n.s.')

        print 'ACTION, subj %s, w vs gen' %(animal)
        w_vs_g = np.hstack(( np.hstack((RW_A)), np.hstack((RGEN_A)) ))
        grps = np.hstack(( np.zeros_like(RW_A), np.zeros_like(RGEN_A)+1))
        pv, slp = util_fcns.run_LME(DAYs, grps, w_vs_g)
        print('pv: %.2f, slp %.2f'%(pv, slp))

        if pv < 0.001: 
            axbth[1, i_a].text(0.2, np.max(w_vs_g), '***')
        elif pv < 0.01: 
            axbth[1, i_a].text(0.2, np.max(w_vs_g), '**')
        elif pv < 0.05:
            axbth[1, i_a].text(0.2, np.max(w_vs_g), '*')
        else:
            axbth[1, i_a].text(0.2, np.max(w_vs_g), 'n.s\npv=%.3f'%(pv))

        #### Plot combined tasks R2 figure ###
        for i_z, nm in enumerate(['Neural', 'Action']):
            lme_day = np.hstack(( [ndays_all, ndays_all] ))
            lme_cat = np.hstack(( np.zeros((len(ndays_all))), np.ones((len(ndays_all))) ))
            lme_val = np.hstack(( pwi_comb[:, i_z], pall_comb[:, i_z]))
            lme_val2 = np.hstack((px_comb[:, i_z], pall_comb[:, i_z] ))
            
            util_fcns.run_LME(lme_day, lme_cat, lme_val, bar_plot = True, xlabels = ['Within', 'General'], title=nm+', Each data pt = day')
            util_fcns.run_LME(lme_day, lme_cat, lme_val2, bar_plot = True, xlabels = ['Across', 'General'], title=nm+', Each data pt = day')

    #fco.tight_layout()
    #fob.tight_layout()
    fbth.tight_layout()
    if skip_across:
        fbth.savefig(fig_dir + 'general_dynamics_w_vs_x_matchN%s_skip_across.eps' %(str(match_task_spec_n)))
    else:
        fbth.savefig(fig_dir + 'general_dynamics_w_vs_x_matchN%s.eps' %(str(match_task_spec_n)))

def reinventing_model_7(model_set_number = 7, data = None, data_demean = None):
    ### For each model (above x CO/OBS/GEN), want to plot how well it does on each task
    ### Want to plot how well it explains: 
    ### Each day will have a colored line, Co = -, Obs = -- 

    ###     a) overall R2 -- neural 
    ###     b) overall R2 -- action 
    ###     c) command mean diffs 
    ###     d) condition spec means 
    ###     e) % correct in next action predicitons (CW vs. CCW)
    ###     f) % correct in mag direction 

    ###     g) distribution of eigenvalues 
    ###     h) potent dynamics? 


    fr2, axr2 = plt.subplots(nrows = 3, figsize = (8, 15))
    Xr2 = [[], []]
    
    for ia, animal in enumerate(['grom', 'jeev']):
        
        ### Load the model with matching number of samples // load the model WITH intercept 
        if data is None:
            dat = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen_match_tsk_N.pkl' %(model_set_number), 'rb'))
        else:
            dat = data[ia]

        #### Load the demeaned model ###
        if data_demean is None:
            dat_dem = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen_match_tsk_N_task_demean.pkl' %(model_set_number), 'rb'))
        else:
            dat_dem = data_demean[ia]

        ### Plotting; 
        ### CO/OBS data R2 avg. for models fit on CO / OBS / gen applied to ALL data; 
        model_nms = ['hist_1pos_0psh_0spksm_1_spksp_0', 'hist_1pos_4psh_0spksm_1_spksp_0']
        model_label_nms = ['$y_t|y_{t-1}$', '$y_t|y_{t-1}, tsk$']
        
        ### To keep track of bar plots 
        r2_xlab = []
        r2_xtic = []
        
        ### Then after each plot, will have ###
        tmp_lines = defaultdict(list)

        for i_m, model_nm in enumerate(model_nms):

            ### For each day ###
            for i_d in range(analysis_config.data_params[animal+'_ndays']): 

                ### Get decoder ###
                KG = util_fcns.get_decoder(animal, i_d)

                tsk = dat[i_d, 'task']
                ix_co = np.nonzero(tsk == 0)[0]
                ix_ob = np.nonzero(tsk == 1)[0]
                
                spks = dat[i_d, 'spks']
                pred_spks = dat[i_d, model_nm] # T x N x 3
                psh = dat[i_d, 'np']

                pred_psh = [np.dot(KG, pred_spks[:, :, k].T).T for k in range(3)]
                pred_psh = np.dstack((pred_psh))
                assert(psh.shape[0] == pred_psh.shape[0])
                assert(psh.shape[1] == pred_psh.shape[1])

                for mod_fit, mod_fitdat in enumerate(['CO', 'OBS', 'COMB']):
        
                    #### Overall R2 of the models  
                    tmp_lines[0, i_d, 'neur'].append(util_fcns.get_R2(spks[ix_co, :], pred_spks[ix_co, :, mod_fit]))
                    tmp_lines[1, i_d, 'neur'].append(util_fcns.get_R2(spks[ix_ob, :], pred_spks[ix_ob, :, mod_fit]))

                    tmp_lines[0, i_d, 'act'].append(util_fcns.get_R2(psh[ix_co, :], pred_psh[ix_co, :, mod_fit]))
                    tmp_lines[1, i_d, 'act'].append(util_fcns.get_R2(psh[ix_ob, :], pred_psh[ix_ob, :, mod_fit]))

                    if i_d == 0:
                        r2_xlab.append('%s: Mod %s, Fit %s' %(animal, model_label_nms[i_m], mod_fitdat))
                        r2_xtic.append(ia*10 + i_m*4 + mod_fit)

                    #### Fit task-spec diffs + individual model; 
                    if model_nm == 'hist_1pos_0psh_0spksm_1_spksp_0':

                        #### Get task-specific activity 
                        spks_dem  = dat_dem[i_d, 'spks']
                        pred_spks_dem = dat_dem[i_d, model_nm]
                        tsk_dem   = dat_dem[i_d, 'task']
                        ix_co_dem = np.nonzero(tsk == 0)[0]
                        ix_ob_dem = np.nonzero(tsk == 1)[0]

                        #### Get the task related activity #### 
                        tmp_lines[0, i_d, 'neur_dem'].append(util_fcns.get_R2(spks_dem[ix_co_dem, :], pred_spks_dem[ix_co_dem, :, mod_fit]))
                        tmp_lines[1, i_d, 'neur_dem'].append(util_fcns.get_R2(spks_dem[ix_ob_dem, :], pred_spks_dem[ix_ob_dem, :, mod_fit]))

        ### Aggregate for bars 
        model_R2s = defaultdict(list)

        ### Plot individual days 
        for i_d in range(analysis_config.data_params[animal+'_ndays']):
            
            for i_p, pl in enumerate(['neur', 'act', 'neur_dem']):
                if pl in ['neur', 'act']:
                    ### Plot Days# ###
                    axr2[i_p].plot(r2_xtic, np.hstack((tmp_lines[0, i_d, pl])), '.-', color=analysis_config.pref_colors[i_d])
                    axr2[i_p].plot(r2_xtic, np.hstack((tmp_lines[1, i_d, pl])), '.--', color=analysis_config.pref_colors[i_d])
                    axr2[i_p].set_ylabel(pl)

                    for i in range(len(np.hstack((tmp_lines[0, i_d, pl])))):
                        model_R2s[i, i_p].append(tmp_lines[0, i_d, pl][i])
                        model_R2s[i, i_p].append(tmp_lines[1, i_d, pl][i])
                else:
                    axr2[i_p].plot(np.arange(3)+10*ia, np.hstack((tmp_lines[0, i_d, pl])), '.-', color=analysis_config.pref_colors[i_d])
                    axr2[i_p].plot(np.arange(3)+10*ia, np.hstack((tmp_lines[1, i_d, pl])), '.--', color=analysis_config.pref_colors[i_d])
                    for i in range(3):
                        model_R2s[i, i_p].append(tmp_lines[0, i_d, pl][i])
                        model_R2s[i, i_p].append(tmp_lines[1, i_d, pl][i])
                        
        for i, xt in enumerate(r2_xtic):
            for i_p, pl in enumerate(['neur', 'act', 'neur_dem']):
                if tuple([i, i_p]) in model_R2s.keys():
                    axr2[i_p].bar(xt, np.mean(model_R2s[i, i_p]), width = 1., color = 'gray', alpha = .3)

        Xr2[0].append(r2_xlab)
        Xr2[1].append(r2_xtic)

    for i_p in range(2):
        axr2[i_p].set_xticks(np.hstack((Xr2[1])))
        axr2[i_p].set_xticklabels(np.hstack((Xr2[0])), rotation = 45, fontsize=8)
    axr2[i_p].set_title('Neur Demean: CO | OBS | COMB used to fit task demeaned spikes')
    
    #### Preeya 
    axr2[2].set_ylabel('Neural pred w/ demeaning by task')
    fr2.tight_layout()

def reinventing_model_7_w_sig(model_set_number = 7, data = None,
    task_spec_ix = 0):
    '''
    Same as above but comparing CO vs. GEN for models that were fit with same amount of data; 
    '''
    ### Top row = CO data, bottom row = OBS data 
    ### LEft col = neural, left col = action 
    fr2, axr2 = plt.subplots(nrows = 2, ncols = 2, figsize = (8, 8))
    Xr2 = [[], []]
    
    ### LME model fit; ####
    R2_lme = defaultdict(list)
    Modfit_lme =defaultdict(list)
    Day_Tsk_fit_lme = defaultdict(list)
    Neur_act_lme =defaultdict(list)

    mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))

    for ia, animal in enumerate(['grom', 'jeev']):
        
        ### Load the model with matching number of samples // load the model WITH intercept 
        if data is None:
            dat = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen_match_tsk_N.pkl' %(model_set_number), 'rb'))
        else:
            dat = data[ia]

        ### Plotting; 
        ### CO/OBS data R2 avg. for models fit on CO / OBS / gen applied to ALL data; 
        model_nms = ['hist_1pos_0psh_0spksm_1_spksp_0']#, 'hist_1pos_4psh_0spksm_1_spksp_0']
        model_label_nms = ['$y_t|y_{t-1}$']#, '$y_t|y_{t-1}, tsk$']
        
        ### To keep track of bar plots 
        r2_xlab = []
        r2_xtic = []
        
        ### Then after each plot, will have ###
        tmp_lines = defaultdict(list)

        for i_m, model_nm in enumerate(model_nms):

            ### For each day ###
            for i_d in range(analysis_config.data_params[animal+'_ndays']): 

                ### Get decoder ###
                KG = util_fcns.get_decoder(animal, i_d)

                tsk = dat[i_d, 'task']
                trg = dat[i_d, 'targ']
                ix_co = np.nonzero(tsk == 0)[0]
                ix_ob = np.nonzero(tsk == 1)[0]
                
                spks = dat[i_d, 'spks']
                pred_spks = dat[i_d, model_nm] # T x N x 3
                
                psh = dat[i_d, 'np']
                pred_psh = [np.dot(KG, pred_spks[:, :, k].T).T for k in range(3)]
                pred_psh = np.dstack((pred_psh))
                assert(psh.shape[0] == pred_psh.shape[0])
                assert(psh.shape[1] == pred_psh.shape[1])

                for mod_fit, mod_fitdat in enumerate(['CO', 'OBS', 'COMB']):
                    if mod_fit in [task_spec_ix, 2]:        

                        #### Overall R2 of the models  
                        n0 = util_fcns.get_R2(spks[ix_co, :], pred_spks[ix_co, :, mod_fit])
                        n1 = util_fcns.get_R2(spks[ix_ob, :], pred_spks[ix_ob, :, mod_fit])
                        
                        a0 = util_fcns.get_R2(psh[ix_co, :], pred_psh[ix_co, :, mod_fit])
                        a1 = util_fcns.get_R2(psh[ix_ob, :], pred_psh[ix_ob, :, mod_fit])

                        tmp_lines[0, i_d, 'neur'].append(n0)
                        tmp_lines[1, i_d, 'neur'].append(n1)

                        tmp_lines[0, i_d, 'act'].append(a0)
                        tmp_lines[1, i_d, 'act'].append(a1)

                        R2_lme[animal, 0].append(n0)
                        R2_lme[animal, 1].append(n1)
                        Modfit_lme[animal, 0].append(mod_fit)
                        Modfit_lme[animal, 1].append(mod_fit)
                        Day_Tsk_fit_lme[animal, 0].append(i_d*100 + 0)
                        Day_Tsk_fit_lme[animal, 1].append(i_d*100 + 1)
                        Neur_act_lme[animal, 0].append(0)
                        Neur_act_lme[animal, 1].append(0)
                        
                        R2_lme[animal, 0].append(a0)
                        R2_lme[animal, 1].append(a1)
                        Modfit_lme[animal, 0].append(mod_fit)
                        Modfit_lme[animal, 1].append(mod_fit)
                        Day_Tsk_fit_lme[animal, 0].append(i_d*100 + 0)
                        Day_Tsk_fit_lme[animal, 1].append(i_d*100 + 1)
                        Neur_act_lme[animal, 0].append(1)
                        Neur_act_lme[animal, 1].append(1)
                    
                        if i_d == 0:
                            r2_xlab.append('%s: Mod %s, Fit %s' %(animal, model_label_nms[i_m], mod_fitdat))
                            r2_xtic.append(ia*4 + mod_fit)
                
                ############ Assess command means ############
                pred_command_mns = dict()
                true_command_mns = dict()

                pred_command_mns_cond = dict()
                true_command_mns_cond = dict()

                #### Get command bins; 
                command_bins = util_fcns.commands2bins([psh], mag_boundaries, animal, i_d, vel_ix = [0, 1])[0]
                
                for magi in range(4):
                    for angi in range(8):
                        ix = np.nonzero(np.logical_and(command_bins[:, 0] == magi, command_bins[:, 1] == angi))[0]
                        
                        if len(ix) > min_obs:
                            true_command_mns.append(np.mean(spks[ix, :], axis=0))
                            pred_command_mns.append(np.mean(pred_spks[ix, :], axis=0))

                        for targi in range(10):
                            for targi2 in range(10):

                                ix_co_i = (command_bins[:, 0] == magi) & (command_bins[:, 1] == angi) & (tsk == 0) & (trg == targi)



        ### Aggregate for bars 
        model_R2s = defaultdict(list)

        ### Plot individual days 
        for i_d in range(analysis_config.data_params[animal+'_ndays']):
            for i_p, pl in enumerate(['neur', 'act']):
                
                ### Plot Days# ###
                axr2[0, i_p].plot(r2_xtic, np.hstack((tmp_lines[0, i_d, pl])), '.-', color=analysis_config.pref_colors[i_d])
                axr2[1, i_p].plot(r2_xtic, np.hstack((tmp_lines[1, i_d, pl])), '.--', color=analysis_config.pref_colors[i_d])
                axr2[0, i_p].set_title(pl)

                for i in range(len(np.hstack((tmp_lines[0, i_d, pl])))):
                    model_R2s[i, i_p, 0].append(tmp_lines[0, i_d, pl][i])
                    model_R2s[i, i_p, 1].append(tmp_lines[1, i_d, pl][i])

        for i, xt in enumerate(r2_xtic):
            for i_p, pl in enumerate(['neur', 'act']):
            
                axr2[0, i_p].bar(xt, np.mean(model_R2s[i, i_p, 0]), width = 1., color = 'gray', alpha = .3)
                axr2[1, i_p].bar(xt, np.mean(model_R2s[i, i_p, 1]), width = 1., color = 'gray', alpha = .3)
                
        #### get the LME model significance value ####
        ix_co_n = np.nonzero(np.hstack((Neur_act_lme[animal, 0])) == 0)[0]
        ix_co_a = np.nonzero(np.hstack((Neur_act_lme[animal, 0])) == 1)[0]
        ix_obs_n = np.nonzero(np.hstack((Neur_act_lme[animal, 1])) == 0)[0]
        ix_obs_a = np.nonzero(np.hstack((Neur_act_lme[animal, 1])) == 1)[0]

        task_lab = ['CO data', 'Obs Data']
        for na, ixs in enumerate([[ix_co_n, ix_obs_n], [ix_co_a, ix_obs_a]]):
            for co_obs, ix in enumerate(ixs):
                pv, slp = util_fcns.run_LME(np.hstack((Day_Tsk_fit_lme[animal, co_obs]))[ix], np.hstack((Modfit_lme[animal, co_obs]))[ix], np.hstack((R2_lme[animal, co_obs]))[ix], bar_plot = False)
        
                ### PLot this guy; 
                axr2[co_obs, na].plot(np.array([0, 2]) + 3*ia, [.5, .5], 'k-')
                axr2[co_obs, na].text(1 + 4*ia, .48, '%.4f'%pv)
                axr2[co_obs, na].set_ylabel(task_lab[co_obs])

        ####### Test command means ######
        #for 

    
    fr2.tight_layout()

def generic_r2_plotter(model_set_number, model_nms, model_suffx, plot_ix = None, 
    neural_or_action = 'neural', ylabs = None, savedir = None):
    '''
    general method to plot R2s by target, then all together for visualization
    
    model_set_number: list of model sets to plot; 
    model_nms: list of lists -- each entry corresponds to a model_set_number above; within that entry list all the model nms
        desired to be plotted; 
    model_set_suffx: where there's a specific model set suffix to plot; 
    plot_ix: for the data saved from the models, whether there needs to be an index specified for the 3rd dimension (T x N x ?)
        -- for task specific / generic models, the answer is '2' for general, '0' for CO, '1' for OBS; 
    '''

    for i_a, animal in enumerate(['grom', 'jeev']):
        if animal == 'grom':
            ndays_all = np.arange(analysis_config.data_params['grom_ndays'])
            ndays = 9; 

        elif animal == 'jeev':
            ndays_all = [0, 2, 3]
            ndays  = 3; 

        ### How many bars will there be? 
        bars = 0; 
        for i_m in range(len(model_set_number)):
            for i_m2 in range(len(model_nms[i_m])):
                bars += 1; 
        width = 1./(bars + 2.)
        cols = np.linspace(0., 1., bars)
        colors = [plt.cm.viridis(x) for x in cols]

        print('Bars %d, Width = %.2f' %(bars, width))
        
        ### Iterate trhough teh days; 
        for i_d, nd in enumerate(ndays_all):
       
            ### hold onto real / predicted data;
            ### keys are [set, model_nm, 'tsk_'] or [set, model_nm, 'all_']
            data_dict = {}

            ### Get kalman gain;
            if animal == 'grom':
                KG, _, _ = generate_models.get_KG_decoder_grom(nd)
            elif animal == 'jeev':
                KG, _, _ = generate_models.get_KG_decoder_jeev(nd)

            ### Make a plot! 
            fig, ax = plt.subplots(figsize = (12, 4))
            bar = -1; 

            for i_m, (msn, model_nms_msn) in enumerate(zip(model_set_number, model_nms)):

                ### Load teh data; 
                dat = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d%s.pkl' %(msn, model_suffx), 'rb'))

                for i_m2, model_nm in enumerate(model_nms_msn):
                    
                    data_dict[msn, model_nm, 'all'] = []
                    data_dict[msn, model_nm, 'tsk0'] = []
                    data_dict[msn, model_nm, 'tsk1'] = []

                    bar += 1

                    ### Now ensure that 
                    for i_t in range(2):

                        for i_trg in range(10):

                            ### task specific indices: 
                            cond_ix = np.nonzero(np.logical_and(dat[(nd, 'trg')] == i_trg, 
                                dat[(nd, 'task')] == i_t))[0]

                            if len(cond_ix) > 0:
                                #### true value 
                                spks_true = dat[(nd, 'spks')][cond_ix, :]
                                true_val_conv = n2a(spks_true, neural_or_action, KG)

                                if plot_ix is None:
                                    spks_pred = dat[(nd, model_nm)][cond_ix, :]
                                
                                else:
                                    spks_pred = dat[(nd, model_nm)][cond_ix, :, plot_ix]

                                #### predicted value; 
                                pred_conv = n2a(spks_pred, neural_or_action, KG)

                                ### Add to the data; 
                                data_dict[msn, model_nm, 'all'].append([true_val_conv, pred_conv])
                                data_dict[msn, model_nm, 'tsk%d'%i_t].append([true_val_conv, pred_conv])

                                ### Plotting R2; 
                                ax.bar(bar*width + i_t*15 + i_trg, 
                                    util_fcns.get_R2(true_val_conv, pred_conv, pop = True),
                                    width = width, color = colors[bar])

                        ### Task Summary: 
                        pred = np.vstack([p[1] for p in data_dict[msn, model_nm, 'tsk%d'%i_t]])
                        true = np.vstack([p[0] for p in data_dict[msn, model_nm, 'tsk%d'%i_t]])

                        ax.bar(bar*width + 2*15 + i_t*2, util_fcns.get_R2(true, pred, pop=True),
                            width = width, color = colors[bar] )

                    ### Full summary; 
                    pred = np.vstack([a[1] for a in data_dict[msn, model_nm, 'all']])
                    true = np.vstack([a[0] for a in data_dict[msn, model_nm, 'all']])

                    ax.bar(bar*width + 2*15 + 2*2, util_fcns.get_R2(true, pred, pop=True),
                        width = width, color = colors[bar] )

            ax.set_title('Animal %s, Day %d' %(animal, nd))

            if savedir is None:
                pass
            else:
                fig.savefig(savedir + '%s_day%d_%s.png' %(animal, nd, neural_or_action))

### Want to plot target specifc (within) vs. task specific (within) vs. general model; 
def plot_r2_bar_tg_spec_7_gen(neural_or_action = 'neural'):
    '''
    method to plot condition specific vs. task specific vs. general A R2 

    updates 5/18/20 -- made it so that also computed R2 on each day, not for each target/condition separately 
        -- todo eventually figure otu how we want to do stats

    neural_or_action: plots R2 for either neural activity or the potent activity; 
    '''
    ########### FIGS ##########
    ##### Each thing is its own data;
    ##### jeev & grom;  
    model_nm = 'hist_1pos_0psh_0spksm_1_spksp_0'

    for i_a, animal in enumerate(['grom', 'jeev']):
        DAYs = []; 
        R_cond = []; ### Condition specific A  
        RW = [];  ### Within task A 
        RGEN = []; ### General task A; 
        colors = ['r', 'b', 'k']
        alphas = [1.,1.,1.]

        ###### Get model name ......######
        model_set_number = 7; 
        dat_cond = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d_cond_spec.pkl' %(model_set_number), 'rb'))
        dat = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen.pkl' %(model_set_number), 'rb'))

        if animal == 'grom':
            ndays_all = np.arange(9);
            ndays = 9; 

        elif animal == 'jeev':
            ndays_all = [0, 2, 3];
            ndays = 3; 

        #####################
        ### Real bar plot ###
        #####################
        ### 3 plots -- one for CO, one for OB, one combined
        ### CO / OBS data  x Fit CO/OBS/GEN x ndays
        #### CO / OBS target vs. [COND / WIN TSK / GENERAL] vs. ndays
        R2s_plot_spks = np.zeros((20, 3, ndays)) 
        LME_R2 = []
        LME_day = []
        LME_model_type = [] 

        lme_r2_comb = []; 
        lme_day_comb = []; 
        lme_mod_comb = []; 

        for i_d, nd in enumerate(ndays_all):
            
            ### Get kalman gain;
            if animal == 'grom':
                KG, _, _ = generate_models.get_KG_decoder_grom(nd)
            elif animal == 'jeev':
                KG, _, _ = generate_models.get_KG_decoder_jeev(nd)

            true_spks = []; 
            pred_cond = []; 
            pred_tsk = []; 
            pred_gen = []; 

            ### Get out the type of models 
            type_of_models = dat_cond[nd, 'type_of_model']
            
            ### Split by task -- assess R2 separately:  
            for i_t in range(2):

                for i_trg in range(10):

                    ### task specific indices: 
                    task_ix = np.nonzero(dat[(nd, 'task')] == i_t)[0]
                    targ_ix = np.nonzero(dat[(nd, 'trg')] == i_trg)[0]
                    cond_ix = np.nonzero(np.logical_and(dat[(nd, 'trg')] == i_trg, 
                        dat[(nd, 'task')] == i_t))[0]

                    ### Get true neural activity; 
                    spks_true = dat[(nd, 'spks')][cond_ix, :]
                    ### Get predicted from within task: 
                    ### T x N x 3 [train on task 0, train on task 1, train on both]
                    spks_pred_gen = dat[(nd, model_nm)][cond_ix, :, 2]
                    spks_pred_tsk = dat[(nd, model_nm)][cond_ix, :, i_t]
                    
                    ### Get the correct output ###
                    cond_key = i_t*10 + i_trg

                    if len(dat_cond[(nd, model_nm)][cond_key, 'ix']) == 0:
                        pass
                    else:

                        ix = np.hstack((dat_cond[(nd, model_nm)][cond_key, 'ix']))
                        prd = np.vstack((dat_cond[(nd, model_nm)][cond_key, 'pred']))
                   
                        ### Get the ordered "ix"
                        ix_ord = np.argsort(ix)
                        ix_sort = ix[ix_ord]
                        assert(len(ix_ord) == len(cond_ix))
                        assert(np.sum(np.abs(ix[ix_ord] - cond_ix)) == 0)

                        spks_pred_cond = prd[ix_ord, :]

                        #### Save these guys #####
                        true_val_conv = n2a(spks_true, neural_or_action, KG)
                        pred_gen_conv = n2a(spks_pred_gen, neural_or_action, KG)
                        pred_tsk_conv = n2a(spks_pred_tsk, neural_or_action, KG)
                        pred_cond_conv = n2a(spks_pred_cond, neural_or_action, KG)

                        true_spks.append(true_val_conv)
                        pred_gen.append(pred_gen_conv)
                        pred_tsk.append(pred_tsk_conv)
                        pred_cond.append(pred_cond_conv)

                        ### Get R2s: [cond/tsk/gen]
                        R2s_plot_spks[cond_key, 0, i_d]= util_fcns.get_R2(true_val_conv, pred_cond_conv, pop = True)
                        R2s_plot_spks[cond_key, 1, i_d] = util_fcns.get_R2(true_val_conv, pred_tsk_conv, pop = True)
                        R2s_plot_spks[cond_key, 2, i_d] = util_fcns.get_R2(true_val_conv, pred_gen_conv, pop = True)
            
                        LME_R2.append(R2s_plot_spks[cond_key, :, i_d])
                        LME_day.append(np.zeros((3, )) + i_d)
                        LME_model_type.append(np.arange(3)) #[cond/tsk/gen]

            f, axi = plt.subplots()
            
            ### Add info to the lme r2 values ###
            true_spks = np.vstack((true_spks))
            pred_cond = np.vstack((pred_cond))
            pred_tsk = np.vstack((pred_tsk))
            pred_gen = np.vstack((pred_gen))

            lme_r2_comb.append(util_fcns.get_R2(true_spks, pred_cond))
            lme_r2_comb.append(util_fcns.get_R2(true_spks, pred_tsk))
            lme_r2_comb.append(util_fcns.get_R2(true_spks, pred_gen))
            lme_day_comb.append(np.zeros((3,)) + nd)
            lme_mod_comb.append(np.arange(3)) # [cond/tsk/gen]


            for i_cond in range(20):
                # [cond/tsk/gen]
                axi.bar(i_cond, R2s_plot_spks[i_cond, 0, i_d], color='lightblue', width = .25)
                axi.bar(i_cond+.25, R2s_plot_spks[i_cond, 1, i_d], color='blue', width = .25)
                axi.bar(i_cond+.5, R2s_plot_spks[i_cond, 2, i_d], color='darkblue', width = .25)
            axi.set_title('Subj %s, Day %d' %(animal, i_d))

        #### LME Plot for general vs. task-specific and general vs. cond specific; 
        LME_R2 = np.hstack((LME_R2))
        LME_day = np.hstack((LME_day))
        LME_model_type = np.hstack((LME_model_type))

        ix1 = np.hstack(([i for i, j in enumerate(LME_model_type) if j in [0, 2]]))
        ix2 = np.hstack(([i for i, j in enumerate(LME_model_type) if j in [1, 2]]))

        pv, slp = util_fcns.run_LME(LME_day[ix1], LME_model_type[ix1], LME_R2[ix1], bar_plot = True,
            xlabels = ['Cond-Spec', 'General'], title = 'Cond spec R2')
        pv, slp = util_fcns.run_LME(LME_day[ix2], LME_model_type[ix2], LME_R2[ix2], bar_plot = True,
            xlabels = ['Task-Spec', 'General'], title = 'Cond spec R2')

        ##### For combined R2 across tasks ###
        lme_r2_comb = np.hstack((lme_r2_comb))
        lme_day_comb = np.hstack((lme_day_comb))
        lme_mod_comb = np.hstack((lme_mod_comb))

        ix1 = np.hstack(([i for i, j in enumerate(lme_mod_comb) if j in [0, 2]]))
        ix2 = np.hstack(([i for i, j in enumerate(lme_mod_comb) if j in [1, 2]]))

        pv, slp = util_fcns.run_LME(lme_day_comb[ix1], lme_mod_comb[ix1], lme_r2_comb[ix1], bar_plot = True,
            xlabels = ['Cond-Spec', 'General'], title = 'Day R2s')
        pv, slp = util_fcns.run_LME(lme_day_comb[ix2], lme_mod_comb[ix2], lme_r2_comb[ix2], bar_plot = True,
            xlabels = ['Task-Spec', 'General'], title = 'Day R2s')

def n2a(spks, neural_or_action, KG):
    if neural_or_action == 'neural':
        return spks 
    
    elif neural_or_action == 'action':
        assert(KG.shape[0] == 2)
        assert(KG.shape[1] == spks.shape[1])

        return np.squeeze(np.array(np.dot(KG, spks.T).T))

def plot_r2_bar_state_encoding(res_or_total = 'res'):
    ''' method to plot 9 bars for each animal / day , mapping out [gen / tsk / cond ] A x [gen / tsk / cond ] B 

        res_or_total; whether to plot the R2 of the residuals from the A matrix regression, or to plot 
            the R2 of the total --> 
    '''
    R2_dict = {}; 

    for i_a, animal in enumerate(['grom', 'jeev']):
        R2_dict[animal] = {}

        ### load up the big file that has it all; 
        dat = pickle.load(open(analysis_config.config[animal+'_pref'] + 'res_model_fit_state.pkl', 'rb'))

        ### Get out pred vs. rez dicts 
        pred_dict = dat['pred_dict']
        rez_dict = dat['rez_dict']
        colors = ['lightblue', 'blue', 'k']

        ### For each day ####
        for i_d, day in enumerate(pred_dict.keys()):
            R2_dict[animal][i_d] = []
            for task in range(2):
                f, ax = plt.subplots(nrows = 2, ncols = 5, figsize = (12, 6))

                for targ in range(10):

                    ix = np.nonzero(np.logical_and(rez_dict[day]['trg'] == targ, rez_dict[day]['tsk'] == task))[0]
                    if len(ix) > 0:
                        tmp = dict(Bcond=[], Btsk=[], Bgen = []);

                        for ia, A in enumerate(['res_cond', 'res_tsk', 'res_gen']):
                            rez_true = rez_dict[day]['res', A[4:]]
                            
                            ### True spks not just rez ##
                            true = rez_dict[day]['true_spks']
                            pred_A = rez_dict[day]['pred_spks_'+A[4:]]

                            for ib, B in enumerate(['Bcond', 'Btsk', 'Bgen']):
                                pred_rez = pred_dict[day][A][B]

                                if res_or_total == 'res':
                                    ax[targ/5, targ%5].bar(ib*4 + ia, util_fcns.get_R2(rez_true[ix, :], pred_rez[ix, :]),
                                        width = 1., color=colors[ia])
                                    
                                    tmp[B].append(util_fcns.get_R2(rez_true[ix, :], pred_rez[ix, :]))

                                elif res_or_total == 'total':
                                    ax[targ/5, targ%5].bar(ib*4 + ia, util_fcns.get_R2(true[ix, :], pred_rez[ix, :] + pred_A[ix, :]),
                                        width = 1., color=colors[ia])
                                    tmp[B].append(util_fcns.get_R2(true[ix, :], pred_rez[ix, :] + pred_A[ix, :]))
                                
                                ax[targ/5, targ%5].set_title('%s, Day %d\n Task %d, Targ %d' %(animal, day, task, targ), fontsize = 8)
                    
                    ax[targ/5, targ%5].set_ylim([-.1, .3])
                    ax[targ/5, targ%5].set_xlim([-1, 11])
                    R2_dict[animal][i_d].append(tmp)
                    
                f.tight_layout()

        ### For animal, make summary plot ###
        Day = []; 
        Val = []; 
        Cat = []; 

        for i_d in range(len(pred_dict.keys())):
            for j, tmp in enumerate(R2_dict[animal][i_d]):
                Day.append(np.zeros((3, )) + i_d)
                Val.append(tmp['Bcond'])
                Cat.append(np.arange(3))
        Day = np.hstack((Day))
        Val = np.hstack((Val))
        Cat = np.hstack((Cat))
        
        #### Comparisons --
        cols = ['lightblue', 'blue', 'black']
        xlab = ['Cond Spec A', 'Tsk Spec A', 'Gen A']
        for i, (i1, i2, noti) in enumerate([[0, 1, 2], [1, 2, 0], [0, 2, 1]]):
            ix = np.nonzero(Cat != noti)[0]
            util_fcns.run_LME(Day[ix], Cat[ix], Val[ix], bar_plot = True, xlabels = [xlab[i1], xlab[i2]],
                title = res_or_total)

        ### Full summary plot 
        x2lab = ['Bcond', 'Btsk', 'Bgen']
        f, ax = plt.subplots()
        xl = []; xlt = []; 
        for bix, bx in enumerate(x2lab):
            ### For animal, make summary plot ###
            Day = []; 
            Val = []; 
            Cat = []; 

            for i_d in range(len(pred_dict.keys())):
                for j, tmp in enumerate(R2_dict[animal][i_d]):
                    Day.append(np.zeros((3, )) + i_d)
                    Val.append(tmp[bx])
                    Cat.append(np.arange(3))
            Day = np.hstack((Day))
            Val = np.hstack((Val))
            Cat = np.hstack((Cat))

            for i in range(3):
                ix = np.nonzero(Cat == i)[0]
                ax.bar(i + 4*bix, np.mean(Val[ix]), color = cols[i])
                xl.append(xlab[i] + '\n' + bx)
                xlt.append(i + 4*bix)
        ax.set_xticks(xlt)
        ax.set_xticklabels(xl, rotation=45)
        f.tight_layout()

############ Eigenvalue decomposition #########
############ Fig 6 ############################
def eigvalue_plot(dyn_model = 'hist_1pos_0psh_0spksm_1_spksp_0',
    dt = 0.1, plt_evs_gte = .99, dat = None, 
    with_intercept = True, LDS = False):

    if LDS:
        dyn_model = 'hist_1pos_0psh_0spksm_1_spksp_0_latentLDS'
    else:
        dyn_model = 'hist_1pos_0psh_0spksm_1_spksp_0'
    n_folds = 5; 

    #### Get data if needed #####
    if dat is None:
        dat = {}

        ### Iterate through the animals 
        for animal in ['grom', 'jeev']:
            # tuning_models_grom_model_set7_task_spec_pls_gen_match_tsk_N_no_intc_gen_demean.pkl
            if with_intercept:
                dati = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen_match_tsk_N.pkl' %(6), 'rb'))
            else:
                dati = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen_match_tsk_N_no_intc.pkl' %(6), 'rb'))
            dat[animal] = dati; 

    #### Stable points ####
    f_stb, ax_stb = plt.subplots(ncols = 4, nrows = 4, figsize = (12, 12))
    cnt = 0; 

    for i_a, animal in enumerate(['grom']):#, 'jeev']):

        ###### Using the task specific models and the general models ######
        ##### Get the one with data matched #######
        data = dat[animal]

        ##### for each day and task -- 
        for i_d in range(1):#analysis_config.data_params[animal+'_ndays']):
            f, ax = plt.subplots(ncols = 1, figsize = (3, 3))
            ax = [ax]
            stb = []; shuf_stb = []; 
            mn = []; 

            #### Get shuffles ###
            if LDS:
                pass
            else:   
                try:
                    _, coef_shuff, intc_shuff = get_shuffled_data(animal, i_d, dyn_model, get_model = True, with_intercept = with_intercept)
                    shuff_skip = False
                except:
                    print('No shuffs available')
                    shuff_skip = True

            ### For each day, plot the eigenvalues ####
            for i_f in range(1):

                ### CO / OBS / GEN ####
                for i_m in range(1):

            
                    if i_m == 0:
                        N = data[i_d, 'spks'].shape[0]

                        if LDS:
                            pass
                        else:
                            test_ix = data[i_d, dyn_model, n_folds*i_m + i_f, i_m, 'test_ix']
                            train_ix = np.array([i for i in np.arange(N) if i not in test_ix])
                            mn.append(np.mean(data[i_d, 'spks'][train_ix, :], axis=0))

                    #### Get the A matrix; 
                    if LDS:
                        A = np.mat(data[i_d, dyn_model, n_folds*i_m + i_f, i_m, 'modelA']) 
                    else:
                        #### Get the model 
                        model = data[i_d, dyn_model, n_folds*i_m + i_f, i_m, 'model']
                        A = np.mat(model.coef_)
                        if type(model.intercept_) is np.ndarray:

                            if i_m == 2:
                                #### Compute the ImAinv and stable point #####
                                ImAinv = np.linalg.inv(np.eye(A.shape[0]) - A)
                                stb.append(np.squeeze(np.array(np.dot(ImAinv, model.intercept_[:, np.newaxis]))))
                            
                            N = A.shape[0]
                            # ## Add intercept to the A matrix 
                            A = np.vstack((A, np.zeros((1, N))))
                            intc = model.intercept_[:, np.newaxis]
                            intc = np.vstack((intc, [1]))
                            A = np.hstack((A, intc))

                        else:
                            print('No intercept')

                    ### eigenvalues; 
                    ev, evect = np.linalg.eig(A)

                    ### ONly look at eigenvalues explaining > 
                    ix_sort = np.argsort(np.abs(ev))[::-1]
                    ev_sort = ev[ix_sort]
                    cumsum = np.cumsum(np.abs(ev_sort))/np.sum(np.abs(ev_sort))
                    ix_keep = np.nonzero(cumsum>plt_evs_gte)[0]
                    ev_sort_truc = ev_sort[:ix_keep[0]+1]

                    ### get frequency; 
                    angs = np.angle(ev_sort_truc) #np.array([ np.arctan2(np.imag(ev[i]), np.real(ev[i])) for i in range(len(ev))])
                    hz = np.abs(angs)/(2*np.pi*dt)
                    decay = -1./np.log(np.abs(ev_sort_truc))*dt # Time decay constant in ms
                    ax[i_m].plot(decay, hz, 'k.')
                    ax[i_m].set_xlabel('Time Decay in seconds')
                    if i_m == 0:
                        ax[i_m].set_ylabel('Frequency (Hz)')
                    ax[i_m].set_ylim([-.1,5.05])
                    ax[i_m].vlines(.1, 0, 5.05, 'k', linestyle='dashed', linewidth=.5)
                    #ax[i_m].set_xlim([-.1,1.5])

                    #### Plot shuffle ###
                    if i_m == 2:
                        if LDS:
                            pass
                        elif shuff_skip:
                            pass
                        else:
                            for ia, A in enumerate(coef_shuff):
                                ev, _ = np.linalg.eig(A)
                                angs = np.angle(ev)
                                hz = np.abs(angs)/(2*np.pi*dt)
                                decay = -1./np.log(np.abs(ev))*dt
                                ax[i_m].plot(decay, hz, '.', color='gray', alpha=.3)

                                if with_intercept:
                                    I = intc_shuff[ia]
                                    ImAinv = np.linalg.inv(np.eye(A.shape[0]) - A)
                                    shuf_stb.append(np.squeeze(np.array(np.dot(ImAinv, I[:, np.newaxis]))))

            #### bar plots #####
            if len(stb) > 0:
                stb = np.vstack((stb))
                mn = np.vstack((mn))
                nf, nn = stb.shape

                #### Plot Stable Pt ###
                ax_stb[cnt/4, cnt%4].plot(np.mean(stb, axis=0), np.mean(mn, axis=0), 'k.')

                if len(shuf_stb) > 0:
                    shuf_stb = np.vstack((shuf_stb))
                    ax_stb[cnt/4, cnt%4].plot(np.mean(shuf_stb, axis=0), np.mean(mn, axis=0), '.', color='gray')

                ax_stb[cnt/4, cnt%4].set_xlim([0., 4])
                ax_stb[cnt/4, cnt%4].set_ylim([0., 4])
                ax_stb[cnt/4, cnt%4].plot([0, 4], [0, 4], 'k--', alpha=.5)
                ax_stb[cnt/4, cnt%4].set_xlabel('Est. Stable Pt')
                ax_stb[cnt/4, cnt%4].set_ylabel('Est. mFR')
                ax_stb[cnt/4, cnt%4].set_title('Anim %s, Day %d' %(animal, i_d))
                cnt += 1      

        f.tight_layout()
        util_fcns.savefig(f, '%s_%d_eigs'%(animal, i_d))
    f_stb.tight_layout()

def eig_shuffle(fold=0, plt_evs_gte=.99, dt=.1): 
    file = '/Volumes/Elements/shuffle_data/w_intc/grom_0_shuff000_hist_1pos_0psh_0spksm_1_spksp_0_models.mat'
    mat = sio.loadmat(file)
    A = np.mat(mat[str(tuple((fold, 'coef_')))])
    ev, evect = np.linalg.eig(A)

    ### ONly look at eigenvalues explaining > 
    ix_sort = np.argsort(np.abs(ev))[::-1]
    ev_sort = ev[ix_sort]
    cumsum = np.cumsum(np.abs(ev_sort))/np.sum(np.abs(ev_sort))
    ix_keep = np.nonzero(cumsum>plt_evs_gte)[0]
    ev_sort_truc = ev_sort[:ix_keep[0]+1]

    ### get frequency; 
    angs = np.angle(ev_sort_truc) #np.array([ np.arctan2(np.imag(ev[i]), np.real(ev[i])) for i in range(len(ev))])
    hz = np.abs(angs)/(2*np.pi*dt)
    decay = -1./np.log(np.abs(ev_sort_truc))*dt # Time decay constant in ms

    f, ax = plt.subplots(figsize=(3, 3))
    ax.plot(decay, hz, '.', color='gray')
    ax.set_xlabel('Time Decay in seconds')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_ylim([-.1,5.05])
    ax.set_xlim([-.005, .35])
    ax.set_xticks([.1, .2, .3])
    ax.vlines(.1, 0, 5.05, 'k', linestyle='dashed', linewidth=.5)
    f.tight_layout()
    util_fcns.savefig(f, '%s_%d_eigs_shuff0'%('grom', 0))

def eigvalue_plot2(animal, i_d, dt = 0.1, plt_evs_gte = .9999, dat = None, with_intercept = True):
    ##### Number of folds
    n_folds = 5; 
    i_m = 2 ##### General model 

    dyn_model = 'hist_1pos_0psh_0spksm_1_spksp_0'
    f, ax = plt.subplots(ncols = 2, figsize = (6, 3))

    #### Get shuffles 
    _, coef_shuff, intc_shuff = get_shuffled_data(animal, i_d, dyn_model, get_model = True, with_intercept=with_intercept)
    print('len shuff %d'%(len(coef_shuff)))
    ### Plot 1 model ###
    for i_f in range(5):

        ### True A ###
        model = dat[i_d, dyn_model, n_folds*i_m + i_f, i_m, 'model']
        A = np.mat(model.coef_)
        hz, decay = return_ts_hz(A)
        ax[0].plot(decay, hz, 'k.')#, markersize=2)

        ### Shuffle A ###
        A_shuff = coef_shuff[i_f*2]
        assert(A.shape == A_shuff.shape)

        hz, decay = return_ts_hz(A_shuff)
        ax[1].plot(decay, hz, '.', color='gray')#, markersize=2)
    ax[0].set_xlim([0., .3])
    ax[1].set_xlim([0., .3])
    ax[0].set_ylim([-0.1, 5])
    ax[1].set_ylim([-0.1, 5])

    for axi in ax.reshape(-1):
        axi.set_xlabel("Decay Time (sec)")
        axi.set_ylabel("Frequency (Hz)")
        axi.vlines(.1, 0, 5, 'k', linestyle='dashed')
    ax[0].set_title('Data')
    ax[1].set_title('Shuffled')
    f.tight_layout()
    f.savefig(analysis_config.config['fig_dir']+'/fig6_%s_day%s_eg_eigspec.svg'%(animal, i_d))

def eigvalue_2D_hist(animal, dt = 0.1, plt_evs_gte = .99, dat = None, with_intercept = True):
    if animal == 'grom':
        cmax = 15
    elif animal == 'jeev':
        cmax = 4

    ##### Number of folds
    n_folds = 5; 
    i_m = 2 ##### General model 

    dyn_model = 'hist_1pos_0psh_0spksm_1_spksp_0'
    f, ax = plt.subplots(ncols = 2, figsize = (6, 3))

    true_hz = []; true_decay = []; shuff_hz_w_sep = []; 
    shuff_hz = []; shuff_decay = []; shuff_decay_w_sep = []; 

    #### Iterate through days ####
    for i_d in range(analysis_config.data_params[animal+'_ndays']):

        #### Get shuffles 
        _, coef_shuff, intc_shuff = get_shuffled_data(animal, i_d, dyn_model, get_model = True, 
            with_intercept = with_intercept)

        tmp_true_hz = []; tmp_true_decay = []; 
        tmp_shuff_hz = []; tmp_shuff_decay = []; 
        ### Plot 1 model ###
        for i_f in range(n_folds):

            ### True A ###
            model = dat[i_d, dyn_model, n_folds*i_m + i_f, i_m, 'model']
            A = np.mat(model.coef_)
            hz, decay = return_ts_hz(A)
            tmp_true_hz.append(hz); tmp_true_decay.append(decay)

        for A_shuff in coef_shuff:
            assert(A.shape == A_shuff.shape)
            hz, decay = return_ts_hz(A_shuff)
            tmp_shuff_hz.append(hz)
            tmp_shuff_decay.append(decay)

        true_hz.append(np.hstack((tmp_true_hz)))
        shuff_hz.append(np.hstack((tmp_shuff_hz)))
        true_decay.append(np.hstack((tmp_true_decay)))
        shuff_decay.append(np.hstack((tmp_shuff_decay)))

        shuff_decay_w_sep.append(tmp_shuff_decay)
        shuff_hz_w_sep.append(tmp_shuff_hz)

    #### Plot the 2D histogram ##
    if with_intercept:
        td_max = 0.08
    else:
        td_max = 1.
    bins = [np.linspace(0, .3, 20), np.linspace(0., 5., 20)]
    ax[0].hist2d(np.hstack((true_decay)), np.hstack((true_hz)), bins=bins, cmin=0., cmax = cmax, cmap='gray')
    _, _, _, cax = ax[1].hist2d(np.hstack((shuff_decay)), np.hstack((shuff_hz)), bins=bins, cmin = 0., cmax=cmax*10, cmap='gray')
    for axi in ax.reshape(-1):
        axi.set_xlabel("Decay Time (sec)")
        axi.set_ylabel("Frequency (Hz)")
        #axi.vlines(.05, 0, 5, 'b', linestyle='dashed')

    ax[0].set_title('Data')
    ax[1].set_title('Shuffled')
    f.colorbar(cax, ax=ax[1])
    f.tight_layout()
    f.savefig(analysis_config.config['fig_dir']+'/fig6_%s_eg_eigspec_2Dhist.svg'%(animal))

    ########### Non oscillatory time decays ################
    f, ax = plt.subplots(figsize=(4,5))
    bins = np.linspace(0, td_max, 30)
    half_db = 0.5*(bins[1] - bins[0])

    non_osci = np.nonzero(np.hstack((true_hz)) == 0)[0]
    h, _ = np.histogram(np.hstack((true_decay))[non_osci], bins)
    ax.plot(bins[:-1]+half_db, h/float(np.sum(h)), 'k-')

    mx = np.max(h/float(np.sum(h)))
    ax.vlines(np.mean(np.hstack((true_decay))[non_osci]), 0, 1.1*mx, color='k', linestyle='--', linewidth=1.)
    
    non_osci = np.nonzero(np.hstack((shuff_hz)) == 0.)[0]
    h1, _ = np.histogram(np.hstack((shuff_decay))[non_osci], bins)
    ax.plot(bins[:-1]+half_db, h1/float(np.sum(h1)), '-', color = 'gray')
    mx = np.max(h1/float(np.sum(h1)))
    ax.vlines(np.mean(np.hstack((shuff_decay))[non_osci]), 0, 1.1*mx, color='gray', linestyle='--', linewidth=1.)

    ax.set_xlabel('Time Decay (sec)')
    ax.set_ylabel('Frac. of non-oscillatory Eigenvalues')
    f.savefig(analysis_config.config['fig_dir']+'/fig6_%s_eg_nonosci_eigs_dist.svg'%(animal))

    ########### Non oscillatory time decays ################
    # frel , axrel = plt.subplots()
    # fraw, axraw = plt.subplots()

    # real_frac = []; 
    # real_raw = []
    # shuff_mn_sub = []; 
    # shuff_raw = []; 

    assert(len(true_hz) == analysis_config.data_params[animal+'_ndays'])
    assert(len(true_decay) == analysis_config.data_params[animal+'_ndays'])
    assert(len(shuff_hz) == analysis_config.data_params[animal+'_ndays'])
    assert(len(shuff_decay) == analysis_config.data_params[animal+'_ndays'])

    for i_d in range(analysis_config.data_params[animal+'_ndays']):

        ix = np.nonzero(true_hz[i_d] == 0.)[0]
        real_tmp_dec = true_decay[i_d][ix]

        # ix_shuff = np.nonzero(shuff_hz[i_d] == 0.)[0]
        # shuff_tmp_dec = shuff_decay[i_d][ix_shuff]
        # mn_shuff_dec = np.mean(shuff_tmp_dec)
        
        # #### Plot raw stuff ####
        # axraw.plot([0, 1], [mn_shuff_dec, np.mean(real_tmp_dec)], '-', color='gray')
        
        # #### Plot rel stuff ####
        # frac_real = (np.mean(real_tmp_dec) - mn_shuff_dec) / mn_shuff_dec
        # axrel.plot([0, 1], [0, frac_real], '-', color='gray')

        # ###### store stuff ######
        # real_frac.append(frac_real)
        # real_raw.append(np.mean(real_tmp_dec))
        # shuff_mn_sub.append(shuff_tmp_dec - mn_shuff_dec)
        # shuff_raw.append(mn_shuff_dec)

        assert(len(shuff_decay_w_sep[i_d]) == len(shuff_hz_w_sep[i_d]))# == n_folds*100)

        pv = 0; 
        ###### pv ####
        for _, (ts_dec, ts_hz) in enumerate(zip(shuff_decay_w_sep[i_d], shuff_hz_w_sep[i_d])):
            ixhz = np.nonzero(ts_hz==0)[0]
            mn_shuff = np.mean(ts_dec[ixhz])

            if mn_shuff >= np.mean(real_tmp_dec):
                pv += 1; 
        print('Day %d, PV cnt %d' %(i_d, pv))

    # axraw.bar(0, np.mean(shuff_raw), color='gray', width = .9, alpha = .2)
    # axraw.bar(1, np.mean(real_raw), color='k', width = .9, alpha = .2)

    # util_fcns.draw_plot(0, np.hstack((shuff_mn_sub)), 'gray', 'white', axrel)
    # axrel.bar(1, np.mean(real_frac), color='k', width = .9, alpha = .2)
    
    # axraw.set_xlim([-.5, 1.5])
    # axrel.set_xlim([-.5, 1.5])

def eigvalue_perc_gte_binsize(dt = 0.1, dat = None, with_intercept = True):
    dyn_model = 'hist_1pos_0psh_0spksm_1_spksp_0'
    f, ax = plt.subplots(ncols = 3, figsize = (12, 4))
    #ax = [None, None, ax]
    i_m = 2; 
    n_folds = 5

    for i_a, animal in enumerate(['grom', 'jeev']):

        eigs = dict(); eigs_perc = dict(); 
        shuff_eigs = dict(); shuff_eigs_perc = dict(); 

        lme_dict = dict(day = [], grp = [], val = [])

        #### Iterate through days ####
        for i_d in range(analysis_config.data_params[animal+'_ndays']):

            #### Get shuffles 
            _, coef_shuff, intc_shuff = get_shuffled_data(animal, i_d, dyn_model, get_model = True, 
                with_intercept = with_intercept)

            #### List eigenvalues ####
            eigs[i_d] = []
            shuff_eigs[i_d] = []
            eigs_perc[i_d] = []
            shuff_eigs_perc[i_d] = []

            #### Iterate through coefficient shuffles ###
            for c in coef_shuff:

                #### Dynamics ####
                A = np.mat(c)
                _, decay = return_ts_hz(A)

                #### Number of eigs > .1
                shuff_eigs[i_d].append(len(np.nonzero(decay > dt)[0]))
                shuff_eigs_perc[i_d].append(float(len(np.nonzero(decay > dt)[0])) / float(len(decay)))

            #### Iterate through folds ####
            for i_f in range(n_folds):
                model_ = dat[animal][i_d, dyn_model, n_folds*i_m + i_f, i_m, 'model']
                A = np.mat(model_.coef_)
                _, decay = return_ts_hz(A)
                eigs[i_d].append(len(np.nonzero(decay > dt)[0]))
                eigs_perc[i_d].append(float(len(np.nonzero(decay > dt)[0])) / float(len(decay)))

        #### Box plot ####
        shf = [];           egs = []; 
        shf_perc = [];      egs_perc = []; 
        for i_d in range(analysis_config.data_params[animal+'_ndays']):
            shf.append(shuff_eigs[i_d])
            egs.append(eigs[i_d])
            shf_perc.append(shuff_eigs_perc[i_d])
            egs_perc.append(eigs_perc[i_d])

            shuff_day = np.hstack((shuff_eigs[i_d]))
            day = np.hstack((eigs[i_d]))

            shuff_perc = np.hstack((shuff_eigs_perc[i_d]))
            day_perc = np.hstack((eigs_perc[i_d]))

            #### Shuffle Eigs Raw number #####
            ax[0].plot([3*i_a, 3*i_a + 1], [np.mean(shuff_day), np.mean(day)], 
                '-', color = 'gray')

            ### Percent increase in EVs gte shuff #####
            ax[1].plot([3*i_a, 3*i_a + 1], [0, (np.mean(day) - np.mean(shuff_day)) / np.mean(shuff_day)], 
                '-', color = 'gray')

            ### Frac EVs > .1 ####
            ax[2].plot([3*i_a, 3*i_a + 1], [np.mean(shuff_perc), np.mean(day_perc)], 
                '-', color = 'gray')

            ### Sig ####
            lme_dict['day'].append([i_d, i_d])
            lme_dict['grp'].append([0., 1.])
            lme_dict['val'].append([np.mean(shuff_perc), np.mean(day_perc)])

            #### Perc sig for day ###
            ix = np.nonzero(shuff_perc >= np.mean(day_perc))[0]
            print('Animal %s, Day %d, pv = %.5f' %(animal, i_d, float(len(ix))/float(len(shuff_perc))))

        #### Plot all ####
        util_fcns.draw_plot(3*i_a, np.hstack((shf)), 'gray', 'white', ax[0], width = 0.9)
        ax[0].bar(3*i_a + 1, np.hstack((egs)), width = 0.9, color = 'k')          

        ### Plot number of eigs > and fraction of eigs > and fraction increase in eigs > .1; 
        util_fcns.draw_plot(3*i_a, np.hstack((shf)) - np.mean(shf), 'gray', 'white', ax[1], width = 0.9)
        ax[1].bar(3*i_a + 1, (np.hstack((egs)) - np.mean(shf)) /np.mean(shf), width = 0.9, color = 'k')          

        util_fcns.draw_plot(3*i_a, np.hstack((shf_perc)), 'gray', 'white', ax[2], width = 0.9)
        ax[2].bar(3*i_a + 1, np.mean(np.hstack((egs_perc))), width = 0.9, color = 'k')          

        print('Animal %s' %(animal))
        util_fcns.run_LME(lme_dict['day'], lme_dict['grp'], lme_dict['val'])

    ax[0].set_xlim([-.5, 4.5])
    ax[1].set_xlim([-.5, 4.5])
    ax[2].set_xlim([-.5, 4.5])
    ax[0].set_ylabel('# Eigenvalues \n w. Time Decay > 0.1sec')
    ax[1].set_ylabel('% Increase in Eigenvalues \n w. Time Decay > 0.1sec greater than shuffle')
    ax[2].set_ylabel('% Eigenvalues \n w. Time Decay > 0.1sec')
    f.tight_layout()
    #f.savefig(analysis_config.config['fig_dir'] + 'fig6_perc_eigs_gte_binsize.svg')

def plot_eigs_from_LDS():
    n_states = 15; 
    pref = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/'
    ndays = dict(grom=9, jeev=4)

    f, ax = plt.subplots(ncols = 2, figsize = (8, 4))
    fh, axh = plt.subplots(ncols = 2, figsize = (8, 4))

    hist_bins = np.arange(0., 1.6, .1)
    hist_bins_plt = hist_bins[:-1] + .05; 

    for ia, animal in enumerate(['grom', 'jeev']):
        fn = pref + animal + 'LDSmodels_nstates_'+str(n_states)+'_combined_models_w_dyn_inno_norms.pkl'
        dat = pickle.load(open(fn, 'rb'))

        hist_ = np.zeros_like(hist_bins)

        for i_d in range(ndays[animal]):
            model = dat[i_d]
            A = model.A; 
            ev, _ = np.linalg.eig(A)

            angs = np.array([ np.arctan2(np.imag(ev[i]), np.real(ev[i])) for i in range(len(ev))])
            hz = (angs)/(2*np.pi*.1)
            tau = -1/np.log(np.abs(ev))*.1
            ax[ia].plot(tau, hz, '.', color=cmap_list[i_d], label='Day'+str(i_d))
        
            ### histogram: 
            dig_hist = np.digitize(tau, hist_bins)
            dig_hist = dig_hist - 1; 

            for d in dig_hist:
                hist_[d] += 1; 

        ax[ia].legend(fontsize=10)
        ax[ia].set_title('Monk '+animal[0], fontsize=10)
        ax[ia].set_xlabel('Tau: -1/ln(|eigvalue|) * dt', fontsize=10)
        ax[ia].set_ylabel('Hz: angle(eigvalue) / (2*pi*dt)', fontsize=10)

        ax[ia].set_xlim([0., 1.5])
        ax[ia].set_ylim([-5.5, 5.5])

        axh[ia].bar(hist_bins_plt, hist_[:-1]/ float(np.sum(hist_)), width=.1, color='lightgray')
        axh[ia].bar(hist_bins_plt[-1] + .1, hist_[-1] / float(np.sum(hist_)), width=.1,color='lightgray')
        axh[ia].set_xlabel('Eigenvalue Tau (secs)')
        axh[ia].set_ylabel('Proportion of Eigs')
        axh[ia].set_xticks(np.arange(0., 1.6, .2))
        xlab = ['%.1f' %a for a in np.arange(0., 1.6, .2)]
        axh[ia].set_xticklabels(xlab, fontsize=10)
    
    f.tight_layout()
    fh.tight_layout()

def analze_fixed_pt(dat):
    '''
    for each day get mFR, then get estimated fixed point 
    and compute mean vector diff 
    
    Do the same for shuffles 
    '''
    ###### Plot bar plot ####
    frel, axrel = plt.subplots(figsize = (4, 5))
    fraw, axraw = plt.subplots(figsize = (4, 5))    

    dyn_model = 'hist_1pos_0psh_0spksm_1_spksp_0'
    n_folds = 5; 
    i_m = 2; ## General model 

    for i_a, animal in enumerate(['grom', 'jeev']):

        ###### Using the task specific models and the general models ######
        ##### Get the one with data matched #######
        data = dat[animal]

        norm_diff_true = []
        norm_diff_true_frac_shuff = []
        norm_diff_shuff = []
        norm_diff_shuff_mn = []
        norm_diff_shuff_minus_mn = []
        all_sig = True

        lme_dict = dict(day = [], grp = [], val = [])

        ##### for each day and task -- 
        for i_d in range(analysis_config.data_params[animal+'_ndays']):

            #### Get mean firing rate ####
            mFR = 10*np.mean(data[i_d, 'spks'], axis=0)

            shuf_stb = []; ### Shuffled stable pt -- vector norm diff w/ mFR 
            stb = []; ### True stable pt -- vector norm diff w/ mFR 
            
            #### Get shuffles ###
            _, coef_shuff, intc_shuff = get_shuffled_data(animal, i_d, dyn_model, get_model = True)
            
            ### For each day, plot the eigenvalues ####
            for i_f in range(n_folds):

                #### Get the model 
                model = data[i_d, dyn_model, n_folds*i_m + i_f, i_m, 'model']
                A = model.coef_
                #### Compute the ImAinv and stable point #####
                ImAinv = np.linalg.inv(np.eye(A.shape[0]) - A)
                stb_i = np.squeeze(np.array(np.dot(ImAinv, 10*model.intercept_[:, np.newaxis])))
                stb.append(np.linalg.norm(stb_i - mFR))
        
                for _, (A, I) in enumerate(zip(coef_shuff, intc_shuff)):
                    ImAinv = np.linalg.inv(np.eye(A.shape[0]) - A)
                    shuf_stb_i = np.squeeze(np.array(np.dot(ImAinv, 10*I[:, np.newaxis])))
                    shuf_stb.append(np.linalg.norm(shuf_stb_i - mFR))

            ix = np.nonzero(np.mean(stb) <= np.hstack((shuf_stb)))[0]
            print('Animal %s, day %d, pv %.5f' %(animal, i_d, float(len(ix))/float(len(np.hstack((shuf_stb))))))

            #### Computing norm diffs #####
            norm_diff_true.append(stb)
            norm_diff_shuff.append(shuf_stb)
            
            lme_dict['day'].append([i_d, i_d])
            lme_dict['grp'].append([0, 1])
            lme_dict['val'].append([np.mean(np.hstack((shuf_stb))), np.mean(np.hstack((stb)))])

            mn_shuf = np.mean(np.hstack((shuf_stb)))
            norm_diff_shuff_mn.append(mn_shuf)
            norm_diff_shuff_minus_mn.append(np.hstack((shuf_stb)) - mn_shuf)
            norm_diff_true_frac_shuff.append((np.hstack((stb)) - mn_shuf) / mn_shuf)

        util_fcns.draw_plot(3*i_a, np.hstack((norm_diff_shuff_minus_mn)), 'gray', 'white', axrel)
        axrel.bar(3*i_a + 1, np.mean(np.hstack((norm_diff_true_frac_shuff))), color='k', alpha=.8)

        axraw.bar(3*i_a, np.mean(np.hstack((norm_diff_shuff))), color='gray', alpha=.8)
        axraw.bar(3*i_a+1, np.mean(np.hstack((norm_diff_true))), color='k', alpha=.8)

        mx = 0; mx2 = 0;
        for i_d in range(analysis_config.data_params[animal+'_ndays']):
            axrel.plot([3*i_a, 3*i_a + 1], [np.mean(norm_diff_shuff_minus_mn[i_d]), 
                np.mean(norm_diff_true_frac_shuff[i_d])], '-', color='gray')

            axraw.plot( [3*i_a+1, 3*i_a], [np.mean(norm_diff_true[i_d]), 
                np.mean(norm_diff_shuff[i_d])], '-', color='gray')
        
            mx = np.max([mx, np.mean(norm_diff_true[i_d])])
            mx2 = np.max([mx2, np.mean(norm_diff_true_frac_shuff[i_d])])
        
        for axi in [axrel, axraw]:
            axi.set_xlim([-.5, 4.5])
            axi.set_xticks([0, 1, 3, 4])
            axi.set_xticklabels(['shuffled', 'data', 'shuffled', 'data'], rotation=45)

        # if all_sig:
        #     axraw.plot([3*i_a, 3*i_a+1], [1.1*mx, 1.1*mx], 'k-')
        #     axraw.text(3*i_a + 0.5, 1.15*mx, '***', ha='center')
            
        #     axrel.plot([3*i_a, 3*i_a+1], [1.1*mx2, 1.1*mx2], 'k-')
        #     axrel.text(3*i_a + 0.5, 1.15*mx2, '***', ha='center')
        
        axraw.set_ylabel('Norm Diff. Between mFR and Fixed Point')
        axrel.set_ylabel('Frac. Increase in \nNorm Diff. Between mFR and Fixed Point')

        ##### Print signifcnat 
        print('Animal %s' %(animal))
        pv, slp = util_fcns.run_LME(lme_dict['day'], lme_dict['grp'], lme_dict['val'])
        pv_str = util_fcns.get_pv_str(pv)
        axraw.plot([3*i_a, 3*i_a + 1], [1.1*mx, 1.1*mx], 'k-', linewidth=1.)
        axraw.text(3*i_a + 0.5, 1.15*mx, pv_str, ha='center')

    frel.tight_layout()
    fraw.tight_layout()

    frel.savefig(analysis_config.config['fig_dir']+'/fig6_rel_fixedPt_mFR_diff.svg')
    fraw.savefig(analysis_config.config['fig_dir']+'/fig6_raw_fixedPt_mFR_diff.svg')
        
def return_ts_hz(A, dt = 0.1):
    ev, _ = np.linalg.eig(A)
    angs = np.angle(ev) #np.array([ np.arctan2(np.imag(ev[i]), np.real(ev[i])) for i in range(len(ev))])
    hz = np.abs(angs)/(2*np.pi*dt)
    hz[hz==5.] = 0.
    decay = -1./np.log(np.abs(ev))*dt # Time decay constant in
    return hz, decay 

### Check model dictionaries ####
def check_models(model_dicts, day, model_nm, command_bins):
    
    ### Comparing model 0 (true data) to model 1 (within bin shuffle)
    true_spks = model_dicts[0][day, 'spks']
    true_psh = model_dicts[0][day, 'np']

    shuff_spks = model_dicts[1][day, 'spks']
    shuff_psh = model_dicts[1][day, 'np']

    ### Day bin + shuffles #### 
    day_bin = model_dicts[1][0, 'day_bin_ix']
    shuff_day_bin = model_dicts[1][0, 'day_bin_ix_shuff']

    ntrls = np.max(model_dicts[1][day, 'trl'])
    ### First make sure all the spiking and pushes check out
    passes = 0
    for i_t, shuff in enumerate(shuff_day_bin):
        if shuff in day_bin: 
            ix0 = np.nonzero(day_bin == shuff)[0]
            assert(np.all(shuff_spks[i_t, :] == true_spks[ix0, :]))
            assert(np.all(shuff_psh[i_t, :]  ==  true_psh[ix0, :]))
        else: 
            passes += 1
    print('Total passes %d / Max %d' %(passes, 2*ntrls))

    #### Ok, now query for mag / angs, true conditon vs. shuffled conditon 
    ### distribution 
    command_x_cond = np.zeros((32, 20))
    for imag in range(4):
        for iang in range(8):

            shuff = np.zeros((20, 20, 2))

            ### Get out indices ###
            ix = np.nonzero(np.logical_and(command_bins[:, 0] == imag, command_bins[:, 1] == iang))[0]          
            
            ### Get targets 
            tgs = model_dicts[0][day, 'trg'][ix]
            tsk = model_dicts[0][day, 'task'][ix]

            for j, (tg, ts) in enumerate(zip(tgs, tsk)):
                command_x_cond[imag*8 + iang, int(ts*10 + tg)] += 1

                ### For this particular point, what task/target did the shuffle come from?
                sdb = shuff_day_bin[ix[j]]
                if sdb in day_bin: 
                    ix0 = np.nonzero(day_bin == sdb)[0]

                    ### Get the bin from which this piont is shuffled 
                    tg0 = model_dicts[0][day, 'trg'][ix0]
                    ts0 = model_dicts[0][day, 'task'][ix0]

                    ### Label vs true target 
                    shuff[int(ts*10 + tg), int(ts*10 + tg), 0] += 1
                    shuff[int(ts*10 + tg), int(ts0*10 + tg0), 1] += 1

            #### Shuffle normalization 
            shuff[:, :, 1] = shuff[:, :, 1] / np.sum(shuff[:, :, 1], axis = 1)[:, np.newaxis]
            
            #### For each command, where does the shuffle draw from ### ? 
            f, ax = plt.subplots(ncols = 2, figsize = (8, 4))
            ax[0].pcolormesh(shuff[:, :, 0], cmap = 'viridis')
            cax = ax[1].pcolormesh(shuff[:, :, 1], cmap = 'viridis')
            ax[0].set_ylabel('Dat Label')
            ax[1].set_ylabel('Shuff Dat Label')
            ax[0].set_xlabel('True Targ')
            ax[1].set_xlabel('True Targ')
            ax[0].set_title('Day %d, Mag %d, Ang %d' %(day, imag, iang))

    #### Distribution of commands | conditons 
    f, ax = plt.subplots(nrows = 4, figsize = (10, 10))
    for tmp in range(4):
        tmp_c = command_x_cond[tmp*8:(tmp+1)*8, :]
        cax = ax[tmp].pcolormesh(tmp_c, cmap='viridis', vmin=0, vmax=160)
        ax[tmp].set_ylabel('Commands, Mag %d' %(tmp))

    ax[0].set_title('Commands | Conditions, Day %d' %(day))
    ax[-1].set_xlabel('Conditions: CO | OBS')
    f.tight_layout()

    #### Confirm the predictions from the model? 
    ### Go through teh test ix, and models, and confirm that the model predicts the predicted in both cases; 
    N = model_dicts[1][day, 'np'].shape[0]

    for i_fold in range(5):

        ### Shuff model
        shuff_model = model_dicts[1][day, model_nm, i_fold, 0., 'model']
        shuff_test_ix = model_dicts[1][day, model_nm, i_fold, 0., 'test_ix']
        shuff_train_ix = np.array([i for i in range(N) if i not in shuff_test_ix])

        shuff_act = model_dicts[1][day, 'np'][shuff_test_ix, :]
        spks_pred = model_dicts[1][day, model_nm][shuff_test_ix, :]

        ### Test Ix stats; 
        np.unique(model_dicts[1][day, 'task'][shuff_train_ix])
        np.unique(model_dicts[1][day, 'trg'][shuff_train_ix])

        assert(np.allclose(spks_pred, shuff_model.predict(shuff_act)))

        ### mean FR         

### Putative Fig 6 -- how do dynamics help next time step predictions ###
### What movements are dynamics capable of ###
def plot_dyn_examples_for_conditions(dat, animal = 'grom', day = 0, min_obs = 15):

    ##### Open up model ####
    model = 'hist_1pos_0psh_0spksm_1_spksp_0'
    #dat = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen_match_tsk_N.pkl' %(7), 'rb'))
    KG = util_fcns.get_decoder(animal, day)

    #### Get neural push ###
    push = dat[day, 'np']
    spks = dat[day, 'spks']
    pred_spks = dat[day, model][:, :, 2]
    tsk = dat[day, 'task']
    trg = dat[day, 'trg']
    bin_num = dat[day, 'bin_num']
    min_bin = np.min(bin_num)
    T = push.shape[0]

    ### Get mean spikes ###
    mn_spks = np.mean(spks, axis=0)

    ### Bin the push ###
    mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))
    commands = util_fcns.commands2bins([push], mag_boundaries, animal, day, vel_ix = [0, 1])[0]

    for ang in range(8):
        for mag in range(4):
            
            #### Make plot #####
            f, ax = plt.subplots(ncols = 2, nrows = 2, figsize = (8, 8))

            #### Now for each condition, add data ####
            Dat = []; Act = []; PredAct = []; PredDat = []; Cond = []; 
            ix_command = np.nonzero(np.logical_and(commands[:, 0] == mag, commands[:, 1] == ang))[0]

            for i_tsk in range(2):
                for i_trg in range(8):

                    ### Condition ix; 
                    ix = np.nonzero(np.logical_and(tsk[ix_command] == i_tsk, trg[ix_command] == i_trg))[0]
                    ix_fin = ix_command[ix]
                    ix_fin_tp1 = ix_fin + 1

                    ### Remove ix > len 
                    keep_ix2 = np.nonzero(ix_fin_tp1 < T)[0]
                    ix_fin_tp1 = ix_fin_tp1[keep_ix2]
                    ix_fin = ix_fin[keep_ix2]

                    ### Remove ix that are min; 
                    keep_ix = np.nonzero(bin_num[ix_fin_tp1] > min_bin)[0]
                    ix_fin_tp1 = ix_fin_tp1[keep_ix]
                    ix_fin = ix_fin[keep_ix]

                    if len(ix_fin) > min_obs:

                        #### Data / condition ####
                        PredDat.append(pred_spks[ix_fin_tp1])
                        Dat.append(spks[ix_fin, :])
                        Act.append(push[ix_fin, :])
                        PredAct.append(np.dot(KG, pred_spks[ix_fin_tp1, :].T).T)

                        Cond.append(np.zeros((len(ix_fin), )) + i_tsk*10 + i_trg)

                        assert(np.all(commands[ix_fin, 0] == mag))
                        assert(np.all(commands[ix_fin, 1] == ang))
                        assert(np.all(tsk[ix_fin] == i_tsk))
                        assert(np.all(trg[ix_fin] == i_trg))


            ### Now do PCA on your neural axes ###
            Dat = np.vstack((Dat))
            PredDat = np.vstack((PredDat))

            Act = np.vstack((Act))
            PredAct = np.vstack((PredAct))

            Cond = np.hstack((Cond))
            assert(Dat.shape[0] == len(Cond))
            
            ### Do PCA; 
            transf_data, pc_model, evsort = util_fcns.PCA(Dat, 2, mean_subtract = False)
            transf_data_tp1 = util_fcns.dat2PC(PredDat, pc_model)
            ### For each condition, plot in PC space; 
            for c in np.unique(Cond):
                ix = np.nonzero(Cond == c)[0]
                if c >= 10:
                    ci = int(c - 10)
                    marker = 's'
                else:
                    ci = int(c)
                    marker = 'o'

                ### Can we plot dynamics in this PC space, setting all other dimensions to their mean?!
                model_ridge = dat[day, model, 0 + 2*5, 2, 'model']

                ### Plot flow fields ####
                plot_flow_field_utils.plot_dyn_in_PC_space(model_ridge, pc_model, ax[0, 0], cmax = 5., scale = 5.5, width = 0.01, lims = 10,
                   title = '', animal = 'grom')
                plot_flow_field_utils.plot_dyn_in_PC_space(model_ridge, pc_model, ax[0, 1], cmax = 5., scale = 5.5, width = 0.01, lims = 10,
                   title = '', animal = 'grom')

                #### Plot the decoder axes; 
                K = util_fcns.get_decoder('grom', day)

                #### Vectors in neural space; 
                K_trans_dat = util_fcns.dat2PC(K, pc_model)
                K_trans_dat[0, :] = K_trans_dat[0, :] / np.linalg.norm(K_trans_dat[0, :])
                K_trans_dat[1, :] = K_trans_dat[1, :] / np.linalg.norm(K_trans_dat[1, :])
                
                ax[0, 0].plot([0, K_trans_dat[0, 0]], [0, K_trans_dat[0, 1]], 'k--')
                ax[0, 0].plot([0, K_trans_dat[1, 0]], [0, K_trans_dat[1, 1]], 'b--')
                

                ax[0, 0].plot(np.mean(transf_data[ix, 0]), np.mean(transf_data[ix, 1]), marker, 
                    color = analysis_config.pref_colors[ci])
                ax[0, 1].plot(np.mean(transf_data_tp1[ix, 0]), np.mean(transf_data_tp1[ix, 1]), marker, 
                    color = analysis_config.pref_colors[ci])
                ax[1, 0].plot(np.mean(Act[ix, 0]), np.mean(Act[ix, 1]), marker, color = analysis_config.pref_colors[ci])
                ax[1, 1].plot(np.mean(PredAct[ix, 0]), np.mean(PredAct[ix, 1]), marker, color = analysis_config.pref_colors[ci])
                for axi in ax[1,:]:
                    axi.set_xlim([-5, 5])
                    axi.set_ylim([-5, 5])

            ax[0, 0].set_title("Mag %d, Ang %d" %(mag, ang))

def plot_dyn_examples_for_transitions(dat, animal = 'grom', day = 0, min_obs = 15):

    ##### Open up model ####
    model = 'hist_1pos_0psh_0spksm_1_spksp_0'
    #dat = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen_match_tsk_N.pkl' %(7), 'rb'))
    KG = util_fcns.get_decoder(animal, day)

    #### Get neural push ###
    push = dat[day, 'np']
    spks = dat[day, 'spks']
    pred_spks = dat[day, model][:, :, 2]
    tsk = dat[day, 'task']
    trg = dat[day, 'trg']
    bin_num = dat[day, 'bin_num']
    min_bin = np.min(bin_num)
    T = push.shape[0]

    ### Get mean spikes ###
    mn_spks = np.mean(spks, axis=0)

    ### Bin the push ###
    mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))
    commands = util_fcns.commands2bins([push], mag_boundaries, animal, day, vel_ix = [0, 1])[0]

    for ang in range(8):
        for mag in range(4):
            
            #### Now for each condition, add data ####
            Dat = []; Act = []; PredAct = []; PredDat = []; Cond = []; 
            ix_command = np.nonzero(np.logical_and(commands[:, 0] == mag, commands[:, 1] == ang))[0]

            ix_fin_tp1 = ix_command + 1
            keep_ix2 = np.nonzero(ix_fin_tp1 < T)[0]
            ix_fin_tp1 = ix_fin_tp1[keep_ix2]
            ix_command = ix_command[keep_ix2]

            ### Remove ix that are min; 
            keep_ix = np.nonzero(bin_num[ix_fin_tp1] > min_bin)[0]
            ix_fin_tp1 = ix_fin_tp1[keep_ix]
            ix_command = ix_command[keep_ix]

            if len(ix_fin_tp1) > min_obs:

                for ang2 in range(8):
                    for mag2 in range(4):

                        ##### Do these match tp1 ? #####
                        ix_sub = np.nonzero(np.logical_and(commands[ix_fin_tp1, 0] == mag2, 
                            commands[ix_fin_tp1, 1] == ang2))[0]

                        if len(ix_sub) > min_obs:

                            #### Data / condition ####
                            PredDat.append(pred_spks[ix_fin_tp1[ix_sub], :])
                            Dat.append(spks[ix_command[ix_sub], :])
                            Act.append(push[ix_command[ix_sub], :])
                            PredAct.append(np.dot(KG, pred_spks[ix_fin_tp1[ix_sub], :].T).T)

                            Cond.append(np.zeros((len(ix_sub), )) + mag2*10 + ang2)

                            assert(np.all(commands[ix_command[ix_sub], 0] == mag))
                            assert(np.all(commands[ix_command[ix_sub], 1] == ang))
                            assert(np.all(commands[ix_fin_tp1[ix_sub], 0] == mag2))
                            assert(np.all(commands[ix_fin_tp1[ix_sub], 1] == ang2))

            if len(Dat) > 0:
                ### Now do PCA on your neural axes ###
                Dat = np.vstack((Dat))

                if Dat.shape[0] < Dat.shape[1]:
                    pass
                else:
                    #### Make plot #####
                    f, ax = plt.subplots(ncols = 2, nrows = 2, figsize = (8, 8))
                    PredDat = np.vstack((PredDat))

                    Act = np.vstack((Act))
                    PredAct = np.vstack((PredAct))

                    Cond = np.hstack((Cond))
                    assert(Dat.shape[0] == len(Cond))
                    
                    ### Do PCA; 
                    transf_data, pc_model, evsort = util_fcns.PCA(Dat, 2, mean_subtract = False)
                    transf_data_tp1 = util_fcns.dat2PC(PredDat, pc_model)
                    ### For each condition, plot in PC space; 
                    for c in np.unique(Cond):
                        ix = np.nonzero(Cond == c)[0]
                        if c >= 30:
                            marker = 'd'
                        elif c >= 20:
                            marker = 's'
                        elif c >= 10:
                            marker = 'o'
                        else:
                            marker = '.'
                        ci = int(np.mod(c, 10))

                        ### Can we plot dynamics in this PC space, setting all other dimensions to their mean?!
                        model_ridge = dat[day, model, 0 + 2*5, 2, 'model']

                        ### Plot flow fields ####
                        plot_flow_field_utils.plot_dyn_in_PC_space(model_ridge, pc_model, ax[0, 0], cmax = 5., scale = 5.5, width = 0.01, lims = 10,
                           title = '', animal = 'grom')
                        plot_flow_field_utils.plot_dyn_in_PC_space(model_ridge, pc_model, ax[0, 1], cmax = 5., scale = 5.5, width = 0.01, lims = 10,
                           title = '', animal = 'grom')

                        ax[0, 0].plot(np.mean(transf_data[ix, 0]), np.mean(transf_data[ix, 1]), marker, 
                            color = analysis_config.pref_colors[ci])
                        ax[0, 1].plot(np.mean(transf_data_tp1[ix, 0]), np.mean(transf_data_tp1[ix, 1]), marker, 
                            color = analysis_config.pref_colors[ci])
                        ax[1, 0].plot([0, np.mean(Act[ix, 0])], [0, np.mean(Act[ix, 1])], marker+'-', color = analysis_config.pref_colors[ci])
                        ax[1, 1].plot([0, np.mean(PredAct[ix, 0])], [0, np.mean(PredAct[ix, 1])], marker+'-', color = analysis_config.pref_colors[ci])
                        for axi in ax[1,:]:
                            axi.set_xlim([-5, 5])
                            axi.set_ylim([-5, 5])

                    ax[0, 0].set_title("Mag %d, Ang %d" %(mag, ang))


###### GIANT GENERAL PLOTTING THING with red / black dots for different conditions ######
### Use model predictions to generate means -- potent and null options included. 
def mean_diffs_plot(animal = 'grom', min_obs = 15, load_file = 'default', dt = 1, 
    important_neurons = True, skip_ind_plots = True, only_null = False, only_potent = False, 
    model_set_number = 3, ndays = None, next_pt_pred = True, plot_pred_vals = False,):

    '''
    load_file -- default -- is a way to autoload previously fit model 
            -- dt -- a plotting thing
            -- important_neurons -- use previously fit file to designate important neurons; show distribution of 
            -- skip_ind_plots -- dont plot the important neurons mean diff distributions; 
            -- only_null / only_potent -- 
            -- next_pt_pred -- instead of quantifying how models predict CURRENT time point (aligned to action bin),
        quantify instead how well modesl predict NEXT time point 

    plot_pred_vals -- for model plot 
        a) true neur vs. pred neur, 
        b) tru KG vs. pred KG, 
        c) task neur diff vs. pred task neur diff,
        d) task diff KG, vs. pred task diff KG, d) 
    '''
    
    savedir = analysis_config.config['fig_dir']

    ### Magnitude boundaries: 
    mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))
    marks = ['-', '--']

    if important_neurons:
        imp_file = pickle.load(open(analysis_config.config[animal+'_pref'] + animal + '_important_neurons_svd_feb2019_thresh_0.8.pkl', 'rb'))

    ### List the models to analyze
    mvl, key, _, _ = generate_models_list.get_model_var_list(model_set_number)
    models_to_include = [m[1] for m in mvl]
    
    if load_file is None:
        ### get the model predictions: 
        model_individual_cell_tuning_curves(animal=animal, history_bins_max=4, 
            ridge=True, include_action_lags = True, return_models = True, 
            models_to_include = models_to_include)
    
    elif load_file == 'default':
        ### load the models: 
        model_dict = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d.pkl' %model_set_number, 'rb'))
        
    ### Now generate plots -- testing w/ 1 day
    if ndays is None:
        ndays = dict(grom=9, jeev=4)
    else:
        ndays = dict(grom=ndays, jeev=ndays)

    hist_bins = np.arange(15)

    ### Pooled over all days: 
    f_all, ax_all = plt.subplots(ncols = len(models_to_include), nrows = 2, figsize = (len(models_to_include)*3, 6))
    
    TD_all = dict();
    PD_all = dict(); 

    for mod in models_to_include:
        for sig in ['all', 'sig']:
            TD_all[mod, sig] = []; 
            PD_all[mod, sig] = []; 
        
    for i_d in range(ndays[animal]):

        ### Basics -- get the binning for the neural push commands: 
        neural_push = model_dict[i_d, 'np']

        ### Commands
        commands = util_fcns.commands2bins([neural_push], mag_boundaries, animal, i_d, vel_ix = [0, 1])[0]
        
        ### Get task / target
        tsk = model_dict[i_d, 'task']
        targ = model_dict[i_d, 'trg']
        bin_num = model_dict[i_d, 'bin_num']

        ### Now go through each task targ and assess when there are enough observations: 
        y_true = model_dict[i_d, key]

        T, N = y_true.shape

        if important_neurons:
            important_neur = imp_file[(i_d, animal, 'svd')]
        else:
            important_neur = np.arange(N); 

        if skip_ind_plots:
            important_neur = []; 

        f_dict = dict(); 

        ### Get the decoder 
        ### Get the decoder ###
        if animal == 'grom':
            KG, KG_null, KG_pot = generate_models.get_KG_decoder_grom(i_d)

        if animal == 'jeev':
            KG, KG_null, KG_pot = generate_models.get_KG_decoder_jeev(i_d)

        if np.logical_or(only_null, only_potent):
            if np.logical_and(only_null, only_potent):
                raise Exception('Cant have null and potent')
            else:
                y_true = get_decomp_y(KG_null, KG_pot, y_true, only_null = only_null, only_potent=only_potent)

        ### Only plot distribution diffs for important neurons: 
        for n in important_neur:
            f, ax = plt.subplots(ncols=8, nrows = 4, figsize = (12, 6))
            f_dict[n] = [f, ax]; 
        print('Important neruons %d' %len(important_neur))
        
        ### Make the spiking histograms: 
        sig_diff = np.zeros((N, 4, 8))
        index_dictionary = {}

        for i_mag in range(4):
            for i_ang in range(8):

                sig_i_t = 0; 

                for i_t in range(2):
                    ### Select by task / target / mag / ang; 
                    #ix = (neural_push[:,0] == i_mag) & (neural_push[:,1] == i_ang) & (targ == i_trg) & (tsk == i_t)
                    ix = (commands[:,0] == i_mag) & (commands[:,1] == i_ang) & (tsk == i_t)

                    if np.sum(ix) >= min_obs:
                        #print 'Plotting Day %d, Task %d, Mag %d Ang %d' %(i_d, i_t, i_mag, i_ang)
                        sig_i_t += 1
                        
                        ### Plotting: 
                        y_obs = y_true[ix, :]

                        ## Which indices: 
                        I = i_mag*8 + i_ang

                        ### Indexing: 
                        for n in range(N):

                            ### Plot important neurons; 
                            if n in important_neur:

                                ## get the axis
                                axi = f_dict[n][1][i_mag, i_ang]

                                ### histogram
                                h, biz = np.histogram(y_obs[:, n], hist_bins)

                                ### Plot line
                                axi.plot(biz[:-1] + .5*dt, h / float(np.sum(h)), '-',
                                    color = cmap_list[i_t])

                                ## Plot the mean; 
                                axi.vlines(np.mean(y_obs[:, n]), 0, .5, cmap_list[i_t])

                # for each neuron figure if enough observations of the i_mag / i_ang to plot; 
                if sig_i_t == 2:

                    if next_pt_pred:
                        print 'Aligning to mag %d, ang %d, next step ahead'
                        ix0 = (commands[:,0] == i_mag) & (commands[:,1] == i_ang) & (tsk == 0)
                        ix1 = (commands[:,0] == i_mag) & (commands[:,1] == i_ang) & (tsk == 1)

                        ix0 = np.nonzero(ix0 == True)[0]
                        ix0 = ix0 + 1; 
                        ix0 = ix0[ix0 < len(tsk)]
                        ix0 = ix0[bin_num[ix0] > 0]

                        ix1 = np.nonzero(ix1 == True)[0]
                        ix1 = ix1 + 1; 
                        ix1 = ix1[ix1 < len(tsk)]
                        ix1 = ix1[bin_num[ix1] > 0]

                    else:
                        ## Find relevant commands: 
                        ix0 = (commands[:,0] == i_mag) & (commands[:,1] == i_ang) & (tsk == 0)
                        ix1 = (commands[:,0] == i_mag) & (commands[:,1] == i_ang) & (tsk == 1)
                    
                        ix0 = np.nonzero(ix0 == True)[0]
                        ix1 = np.nonzero(ix1 == True)[0]

                    index_dictionary[i_mag, i_ang] = [ix0, ix1]

                    for n in range(N):
                        ### Two-sided test for distribution difference. If pv > 0.05 can't reject hypothesis that 
                        ### two samples come from the same distribution. 
                        _, pv = scipy.stats.ks_2samp(y_true[ix0, n], y_true[ix1, n])

                        sig_diff[n, i_mag, i_ang] = pv < 0.05; 

                        if n in important_neur:

                            axi = f_dict[n][1][i_mag, i_ang]
                            if pv < 0.05:
                                axi.set_title('Mag %d, Ang %d ***, \nCO N %d, OBS N %d' %(i_mag, i_ang, np.sum(ix0), np.sum(ix1)),
                                    fontsize=8)
                            else:
                                axi.set_title('Mag %d, Ang %d n.s., \nCO N %d, OBS N %d' %(i_mag, i_ang, np.sum(ix0), np.sum(ix1)),
                                    fontsize=8)
                    else:
                        if n in important_neur:
                            axi.set_title('Mag %d, Ang %d n.s., \nCO N %d, OBS N %d' %(i_mag, i_ang, np.sum(ix0), np.sum(ix1)),
                                fontsize=8)

        for n in important_neur:
            f_dict[n][0].tight_layout()
            f_dict[n][0].savefig(savedir + animal + '_day_' + str(i_d) + '_n_'+str(n) + '.png')
            plt.close(f_dict[n][0])

        ###################################
        ### Now get the diff models: ######
        print 'Done with day -- -1 --'
        sig = ['k', 'r']
        f, ax = plt.subplots(ncols = len(models_to_include), nrows = 2, figsize = (3*len(models_to_include), 6))

        ### Rows are CO / OBS; 
        f_, ax_ = plt.subplots(ncols = len(models_to_include), nrows = 4, figsize = (3*len(models_to_include), 8))

        for i_m, mod in enumerate(models_to_include):

            #### These are using the null/potent 
            if 'potent' in mod:
                modi = mod[:-6]
                y_pred = model_dict[i_d, modi, 'pot']
                #y_pred = get_decomp_y(KG_null, KG_pot, y_pred, only_null = False, only_potent=True)
            
            elif 'null' in mod:
                modi = mod[:-4]
                y_pred = model_dict[i_d, modi, 'null']
                #y_pred = get_decomp_y(KG_null, KG_pot, y_pred, only_null = False, only_potent=True)                
            else:
                y_pred = model_dict[i_d, mod]

            # ### Get null activity ###
            # if np.logical_or(only_null, only_potent):
            #     y_pred = get_decomp_y(KG_null, KG_pot, y_pred, only_null = only_null, only_potent=only_potent)

            ax[0, i_m].set_title(mod, fontsize=8)
            TD = []; PD = []; TD_s = []; PD_s = [];

            pred = []; obs = []; 
            predkg = []; obskg = [];
            diff_tru = []; diff_trukg = []; 
            diff_pred = []; diff_predkg = []; 

            ### Make a plot
            for i_mag in range(4):
                for i_ang in range(8):

                    if tuple([i_mag, i_ang]) in index_dictionary.keys():
                        ix0, ix1 = index_dictionary[i_mag, i_ang]

                        assert np.logical_and(len(ix0) >= min_obs, len(ix1) >= min_obs)

                        if key == 'spks':
                            #### True difference over all neurons -- CO vs. OBS:
                            tru_co = 10*np.mean(y_true[ix0, :], axis=0)
                            tru_ob = 10*np.mean(y_true[ix1, :], axis=0)

                            pred_co = 10*np.mean(y_pred[ix0, :], axis=0)
                            pred_ob = 10*np.mean(y_pred[ix1, :], axis=0)

                            tru_diff = tru_co - tru_ob
                            pred_diff = pred_co - pred_ob

                        elif key == 'np':
                            ### Mean angle: 
                            # mean_co = math.atan2(np.mean(y_true[ix0, 1]), np.mean(y_true[ix0, 0]))
                            # mean_ob = math.atan2(np.mean(y_true[ix1, 1]), np.mean(y_true[ix1, 0]))
                            
                            # pred_mean_co = math.atan2(sw.mean(y_pred[ix0, 1]), np.mean(y_pred[ix0, 0]))
                            # pred_mean_ob = math.atan2(np.mean(y_pred[ix1, 1]), np.mean(y_pred[ix1, 0]))
                            
                            # ### do an angular difference: 
                            # tru_diff = ang_difference(np.array([mean_co]), np.array([mean_ob]))
                            # pred_diff = ang_difference(np.array([pred_mean_co]), np.array([pred_mean_ob]))

                            tru_co = np.mean(y_true[ix0, :], axis=0)
                            tru_ob = np.mean(y_true[ix1, :], axis=0)

                            pred_co = np.mean(y_pred[ix0, :], axis=0)
                            pred_ob = np.mean(y_pred[ix1, :], axis=0)

                            tru_diff = tru_co - tru_ob
                            pred_diff = pred_co - pred_ob

                        for n, (td, pd) in enumerate(zip(tru_diff, pred_diff)):
                            if sig_diff[n, i_mag, i_ang] == 1:
                                ax[1, i_m].plot(td, pd, '.', color = sig[int(sig_diff[n, i_mag, i_ang])], alpha=1.)
                                ax[0, i_m].plot(td, pd, '.', color = sig[int(sig_diff[n, i_mag, i_ang])], alpha=1.)

                                ax_all[1, i_m].plot(td, pd, '.', color = sig[int(sig_diff[n, i_mag, i_ang])], alpha=1.)
                                ax_all[0, i_m].plot(td, pd, '.', color = sig[int(sig_diff[n, i_mag, i_ang])], alpha=1.)


                                TD_s.append(td);
                                PD_s.append(pd);

                                TD_all[mod, 'sig'].append(td);
                                PD_all[mod, 'sig'].append(pd);

                            ax[0, i_m].plot(td, pd, '.', color = sig[int(sig_diff[n, i_mag, i_ang])], alpha=0.2)
                            ax_all[0, i_m].plot(td, pd, '.', color = sig[int(sig_diff[n, i_mag, i_ang])], alpha=0.2)

                            TD.append(td);
                            PD.append(pd);
                            TD_all[mod, 'all'].append(td);
                            PD_all[mod, 'all'].append(pd);

                        if plot_pred_vals:
                            ax_[0, i_m].plot(tru_co, pred_co, 'b.')
                            ax_[0, i_m].plot(tru_ob, pred_ob, 'r.')
                            ax_[2, i_m].plot(tru_co - tru_ob, pred_co - pred_ob, 'k.')

                            #### Split by X / Y 
                            if key != 'np':
                                ax_[1, i_m].plot(np.dot(KG, tru_co.T)[0], np.dot(KG, pred_co.T)[0], 'b.')
                                ax_[1, i_m].plot(np.dot(KG, tru_co.T)[1], np.dot(KG, pred_co.T)[1], 'b.', alpha = .3)

                                ax_[1, i_m].plot(np.dot(KG, tru_ob.T)[0], np.dot(KG, pred_ob.T)[0], 'r.')
                                ax_[1, i_m].plot(np.dot(KG, tru_ob.T)[1], np.dot(KG, pred_ob.T)[1], 'r.', alpha = .3)
                            
                                ###### PLOT DIFFERENCE ########
                                ax_[3, i_m].plot(np.dot(KG, tru_co.T) - np.dot(KG, tru_ob.T), np.dot(KG, pred_co.T)-np.dot(KG, pred_ob.T),'k.')  
                            
                            diff_tru.append(tru_co - tru_ob)
                            diff_pred.append(pred_co - pred_ob)

                            if key != 'np':
                                diff_trukg.append(np.dot(KG, tru_co.T) - np.dot(KG, tru_ob.T))
                                diff_predkg.append(np.dot(KG, pred_co.T)-np.dot(KG, pred_ob.T))

                            pred.append(pred_co)
                            pred.append(pred_ob)
                            obs.append(tru_co)
                            obs.append(tru_ob)

                            if key != 'np':
                                predkg.append(np.dot(KG, pred_co.T))
                                predkg.append(np.dot(KG, pred_ob.T))

                                obskg.append(np.dot(KG, tru_co.T))
                                obskg.append(np.dot(KG, tru_ob.T))

                            cokg = np.mean(neural_push[ix0, :], axis=0)
                            obkg = np.mean(neural_push[ix1, :], axis=0)

                            if key != 'np':
                                assert np.sum(np.abs(cokg - np.dot(KG, tru_co.T))) < 5e5
                                assert np.sum(np.abs(obkg - np.dot(KG, tru_ob.T))) < 5e5

                            ax_[0, i_m].set_title('True Y_t vs. Pred Y_t\nModel %s' %mod)
                            ax_[1, i_m].set_title('True K*Y_t vs. Pred K*Y_t\nModel %s' %mod)

            if plot_pred_vals:
                pred = np.hstack((pred)).reshape(-1)
                obs = np.hstack((obs)).reshape(-1)

                if key != 'np':
                    predkg = np.hstack((predkg)).reshape(-1)
                    obskg = np.hstack((obskg)).reshape(-1)
                
                diff_tru = np.hstack((diff_tru)).reshape(-1)
                diff_pred = np.hstack((diff_pred)).reshape(-1)

                if key != 'np':
                    diff_trukg = np.hstack((diff_trukg)).reshape(-1)
                    diff_predkg = np.hstack((diff_predkg)).reshape(-1)

                slp,intc,rv,pv,err = scipy.stats.linregress(obs, pred)
                x_ = np.linspace(np.min(obs), np.max(obs), 100)
                y_ = slp*x_ + intc; 
                ax_[0, i_m].plot(x_, y_, '-', color='gray', linewidth = .5)
                ax_[0, i_m].text(np.percentile(x_, 70), np.percentile(y_, 20), 'r = %.4f \npv = %.4f' %(rv, pv))
                #ax_[0, i_m].set_ylim([0, 9])
                
                slp,intc,rv,pv,err = scipy.stats.linregress(diff_tru, diff_pred)
                x_ = np.linspace(np.min(diff_tru), np.max(diff_tru), 100)
                y_ = slp*x_ + intc; 
                ax_[2, i_m].plot(x_, y_, '-', color='gray', linewidth = .5)
                ax_[2, i_m].text(np.percentile(x_, 70), np.percentile(y_, 20), 'r = %.4f \npv = %.4f' %(rv, pv))
                #ax_[2, i_m].set_ylim([-2, 2])

                if key != 'np':
                    slp,intc,rv,pv,err = scipy.stats.linregress(obskg, predkg)
                    x_ = np.linspace(np.min(obskg), np.max(obskg), 100)
                    y_ = slp*x_ + intc; 
                    ax_[1, i_m].plot(x_, y_, '-', color='gray', linewidth = .5)
                    ax_[1, i_m].text(np.percentile(x_, 70), np.percentile(y_, 20), 'r = %.4f \npv = %.4f' %(rv, pv))
                    ax_[1, i_m].set_ylim([-2, 2])
                
                    slp,intc,rv,pv,err = scipy.stats.linregress(diff_trukg, diff_predkg)
                    x_ = np.linspace(np.min(diff_trukg), np.max(diff_trukg), 100)
                    y_ = slp*x_ + intc; 
                    ax_[3, i_m].plot(x_, y_, '-', color='gray', linewidth = .5)
                    ax_[3, i_m].text(np.percentile(x_, 70), np.percentile(y_, 20), 'r = %.4f \npv = %.4f' %(rv, pv))
                    ax_[3, i_m].set_ylim([-.5, .5])

            ### Lets do a linear correlation: 
            slp,intc,rv,pv,err = scipy.stats.linregress(np.hstack((TD)), np.hstack((PD)))
            x_ = np.linspace(np.min(np.hstack((TD))), np.max(np.hstack((TD))), 100)
            y_ = slp*x_ + intc; 
            ax[0, i_m].plot(x_, y_, '-', color='gray', linewidth = .5)
            vaf = util_fcns.get_R2(np.hstack((TD)), np.hstack((PD)))
            ax[0, i_m].text(np.percentile(x_, 70), np.percentile(y_, 20), 'r = %.4f \npv = %.4f\nslp = %.2f\n VAF = %.2f' %(rv, pv, slp, vaf))

            slp,intc,rv,pv,err = scipy.stats.linregress(np.hstack((TD_s)), np.hstack((PD_s)))
            x_ = np.linspace(np.min(np.hstack((TD_s))), np.max(np.hstack((TD_s))), 100)
            y_ = slp*x_ + intc; 
            ax[1, i_m].plot(x_, y_, '-', color='gray', linewidth = .5)
            vaf = util_fcns.get_R2(np.hstack((TD_s)), np.hstack((PD_s)))
            ax[1, i_m].text(np.percentile(x_, 70), np.percentile(y_, 20), 'r = %.4f \npv = %.4f\nslp = %.2f\n VAF = %.2f' %(rv, pv, slp, vaf))
            print('Done with day -- %d --'%i_m)
        
        f.tight_layout()
        #f.savefig(savedir+animal+'_day_'+str(i_d) + '_trudiff_vs_preddiff_xtask_mean_corrs_onlyNull_'+str(only_null)+'onlyPotent'+str(only_potent)+'_model_set'+str(model_set_number)+'.png')

        print 'Done with day -- end --'

    ## Get all the stuff for the ax_all; 
    for i_m, mod in enumerate(models_to_include):
        for sig, sig_nm in enumerate(['all', 'sig']):
            x = np.hstack((TD_all[mod, sig_nm]))
            y = np.hstack((PD_all[mod, sig_nm]))

            slp,intc,rv,pv,err = scipy.stats.linregress(x, y)
            x_ = np.linspace(np.min(x), np.max(x), 100)
            y_ = slp*x_ + intc; 
            ax_all[sig, i_m].plot(x_, y_, '-', color='gray', linewidth = .5)
            ax_all[sig, i_m].text(np.percentile(x_, 70), np.percentile(y_, 20), 'r = %.4f \npv = %.4f' %(rv, pv))
    f_all.tight_layout()
    #f_all.savefig(savedir+animal+'_all_days_trudiff_vs_preddiff_xtask_mean_corrs_onlyNull_'+str(only_null)+'onlyPotent'+str(only_potent)+'_model_set'+str(model_set_number)+'.png')

