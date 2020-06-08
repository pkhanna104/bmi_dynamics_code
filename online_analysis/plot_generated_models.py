############## Methods to plot models generated in 'generate models' ###########

import seaborn
seaborn.set(font='Arial',context='talk',font_scale=1., style='white')

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pickle 

import analysis_config, util_fcns
import generate_models, generate_models_list

import scipy

fig_dir = analysis_config.config['fig_dir']

#### so far which models are useful 
'''
model 1 (fig 4)
model 3 -- general extraction of stuff
'''

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
def plot_real_mean_diffplot_r2_bar_tg_spec_7_gens(model_set_number = 3, min_obs = 15, plot_ex = False, plot_disc = False, cov = False):
    ### Take real task / target / command / neuron / day comparisons for each neuron in the BMI
    ### Plot within a bar 
    ### Plot only sig. different ones

    ### Plot cov. diffs (mat1 - mat2)
    mag_boundaries = pickle.load(open(analysis_config.config['grom_pref']+'radial_boundaries_fit_based_on_perc_feb_2019.pkl'))

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
                fex, axex = plt.subplots(ncols = 10, nrows = 5, figsize = (12, 6))
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
                                    
                                    ### make the plot: 
                                    if cov:
                                        diffs = util_fcns.get_cov_diffs(ix_co, ix_ob, spks, diffs)
                                    else:
                                        diffs.append(np.mean(spks[ix_co, :], axis=0) - np.mean(spks[ix_ob, :], axis=0))
                                    #### Get the covariance and plot that: #####

                                    for n in range(N): 
                                        ks, pv = scipy.stats.ks_2samp(spks[ix_co, n], spks[ix_ob, n])
                                        if pv < 0.05:
                                            sig_diffs.append(np.mean(spks[ix_co, n]) - np.mean(spks[ix_ob, n]))

                                        if plot_ex:
                                            #if np.logical_and(len(ix_co) > 30, len(ix_ob) > 30):
                                            if axex_cnt < 40:
                                                if np.logical_and(mag_i > 1, n == 38): 
                                                    axi = axex[axex_cnt / 9, axex_cnt %9]

                                                    util_fcns.draw_plot(0, 10*spks[ix_co, n], co_obs_cmap[0], 'white', axi)
                                                    util_fcns.draw_plot(1, 10*spks[ix_ob, n], co_obs_cmap[1], 'white', axi)

                                                    axi.set_title('A:%d, M:%d, NN%d, \nCOT: %d, OBST: %d, Monk:%s'%(ang_i, mag_i, n, targi, targi2, animal),fontsize=6)
                                                    axi.set_ylabel('Firing Rate (Hz)')
                                                    axi.set_xlim([-.5, 1.5])
                                                    #axi.set_ylim([-1., 10*(1+np.max(np.hstack(( spks[ix_co, n], spks[ix_ob, n]))))])
                                                    axex_cnt += 1
                                                    if axex_cnt == 40:
                                                        fex.tight_layout()
                                                        fex.savefig(fig_dir + 'monk_%s_ex_mean_diffs.svg'%animal)

            if len(sig_diffs) > 0:
                SD = np.abs(np.hstack((sig_diffs)))*10 # to Hz
            else:
                SD = []; 

            #### Absolute difference in means; 
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
        
        ax.set_ylabel('Abs Hz Diff Across Conditions')
        ax.set_xlabel('Days')
        ax.set_xlim([-1, ndays + 1])
        if cov:
            pass
        else:
            ax.set_ylim([0., 20.])
        ax.set_title('Monk '+animal[0].capitalize())
        f.tight_layout()
        f.savefig(fig_dir + 'monk_%s_mean_diffs_box_plots_cov_%s.png' %(animal, str(cov)), transparent = True)

def plot_real_mean_diffs_wi_vs_x(model_set_number = 3, min_obs = 15, cov = False):
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
        days = []; mets = []; grp = []; 
        mnz = np.zeros((ndays, 2)) # days x [x / wi]

        f, ax = plt.subplots(figsize=(ndays/2., 4))

        for i_d in range(ndays):

            diffs_wi = []; 
            diffs_x = []; 

            spks = model_dict[i_d, 'spks']
            N = spks.shape[1]; 
            tsk  = model_dict[i_d, 'task']
            targ = model_dict[i_d, 'trg']
            push = model_dict[i_d, 'np']

            commands_disc = util_fcns.commands2bins([push], mag_boundaries, animal, i_d, vel_ix = [0, 1])[0]

            ### Now go through combos and plot 
            for mag_i in range(4):
                for ang_i in range(8):
                    for targi in range(8):
                        ix_co = (commands_disc[:, 0] == mag_i) & (commands_disc[:, 1] == ang_i) & (tsk == 0) & (targ == targi)
                        if len(np.nonzero(ix_co == True)[0]) > min_obs:

                            for targi2 in range(8):
                                ix_ob = (commands_disc[:, 0] == mag_i) & (commands_disc[:, 1] == ang_i) & (tsk == 1) & (targ == targi2)

                                if len(np.nonzero(ix_ob == True)[0]) > min_obs:
                                    
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
                                    
                                    if cov:
                                        ### make the plot: 
                                        diffs_wi = util_fcns.get_cov_diffs(ix_co1, ix_co2, spks, diffs_wi)
                                        diffs_wi = util_fcns.get_cov_diffs(ix_ob1, ix_ob2, spks, diffs_wi)

                                        diffs_x = util_fcns.get_cov_diffs(ix_co1, ix_ob1, spks, diffs_x)
                                        diffs_x = util_fcns.get_cov_diffs(ix_co2, ix_ob2, spks, diffs_x)

                                    else:
                                        ### make the plot: 
                                        diffs_wi.append(np.mean(spks[ix_co1, :], axis=0) - np.mean(spks[ix_co2, :], axis=0))
                                        diffs_wi.append(np.mean(spks[ix_ob1, :], axis=0) - np.mean(spks[ix_ob2, :], axis=0))

                                        diffs_x.append(np.mean(spks[ix_co1, :], axis=0) - np.mean(spks[ix_ob1, :], axis=0))
                                        diffs_x.append(np.mean(spks[ix_co2, :], axis=0) - np.mean(spks[ix_ob2, :], axis=0))

            if cov:
                mult = 1; 
            else: 
                mult = 10; 
            AD_wi = np.abs(np.hstack((diffs_wi)))*mult # to Hz
            AD_x = np.abs(np.hstack((diffs_x)))*mult # To hz

            DWI.append(AD_wi)
            DX.append(AD_x)

            days.append(np.hstack(( np.zeros_like(AD_wi) + i_d, np.zeros_like(AD_x) + i_d)))
            grp.append( np.hstack(( np.zeros_like(AD_wi) + 1,   np.zeros_like(AD_x)))) # looking for + slp
            mets.append(AD_wi)
            mets.append(AD_x)

            for i, (D, col) in enumerate(zip([AD_x, AD_wi], ['k', 'gray'])):
                ax.bar(i_d + i*.45, np.nanmean(D), color=col, edgecolor='none', width=.4, linewidth=1.0)
                ax.errorbar(i_d + i*.45, np.nanmean(D), np.nanstd(D)/np.sqrt(len(D)), marker='|', color=col)
                mnz[i_d, i] = np.nanmean(D)
         
        DWI = np.hstack((DWI))
        DX = np.hstack((DX))
        days = np.hstack((days))
        grp = np.hstack((grp))
        mets = np.hstack((mets))

        ### look at: run_LME(Days, Grp, Metric):
        pv, slp = util_fcns.run_LME(days, grp, mets)

        print 'LME model, fixed effect is day, rand effect is X vs. Wi., N = %d, ndays = %d, pv = %.4f, slp = %.4f' %(len(days), len(np.unique(days)), pv, slp)

        ###
        axsumm.bar(0 + ia, np.mean(DX), color='k', edgecolor='none', width=.4, linewidth=2.0, alpha = .8)
        axsumm.bar(0.4 + ia, np.mean(DWI), color='gray', edgecolor='none', width=.4, linewidth=2.0, alpha =.8)

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

        # ax.set_ylabel('Difference in Hz')
        # ax.set_xlabel('Days')
        # ax.set_xticks(np.arange(ndays))
        # ax.set_title('Monk '+animal[0].capitalize())
        # f.tight_layout()

        ### Need to stats test the differences across populations:    
    axsumm.set_xticks([0.2, 1.2])
    axsumm.set_xticklabels(['G', 'J']) 
    if cov:
        axsumm.set_ylim([0, 30])
        axsumm.set_ylabel(' Cov Diffs ($Hz^2$) ') 
        #axsumm.set_ylim([0, .6])
        #axsumm.set_ylabel(' Main Cov. Overlap ') 
    else:
        axsumm.set_ylim([0, 5])
        axsumm.set_ylabel(' Mean Diffs (Hz) ') 
    fsumm.tight_layout()
    fsumm.savefig(fig_dir+'both_monks_w_vs_x_task_mean_diffs_cov%s.svg'%(str(cov)))

def disc_plot(n_disc):

    #co_obs_cmap = [np.array([0, 103, 56])/255., np.array([46, 48, 146])/255., ]
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
    theta, r = np.mgrid[0:2*np.pi:9j, 0:4:5j]
    
    im1 = ax1.pcolormesh(theta, r, n_disc[:, :, 0].T, cmap = co_obs_cmap_cm[0], vmin=np.min(n_disc), vmax=np.max(n_disc))
    im2 = ax2.pcolormesh(theta, r, n_disc[:, :, 1].T, cmap = co_obs_cmap_cm[1], vmin=np.min(n_disc), vmax=np.max(n_disc))

    for ax in [ax1, ax2]:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    fig.savefig(fig_dir+'grom_day0_neur_38_targ4_targ5_dist.svg')

    # divider = make_axes_locatable(ax1)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(im1, cax=cax, orientation='vertical')

    # divider = make_axes_locatable(ax2)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(im2, cax=cax, orientation='vertical');
    # import pdb; pdb.set_trace()

#### Fig 4 ----- bar plots of diff models ####
def plot_r2_bar_model_1(min_obs = 15, 
    ndays = None, pt_2 = False, r2_pop = True, 
    perc_increase = True, model_set_number = 1,
    task_spec_ix = None):
    
    '''
    inputs: min_obs -- number of obs need to count as worthy of comparison; 
    perc_increase: get baseline from y_t | a_t and use this to compute the "1" value; 
    ndays -- ? 
    pt_2 more plots -- not totally sure what they are;  
    r2_pop -- assume its population vs. indivdual 
    model_set_number -- which models to plot from which set; 
    '''

    ### For stats each neuron is an observation ##
    mag_boundaries = pickle.load(open(analysis_config.config['grom_pref']+ 'radial_boundaries_fit_based_on_perc_feb_2019.pkl'))
    key = 'spks' 

    ### Now generate plots -- testing w/ 1 day
    if ndays is None:
        ndays = dict(grom=9, jeev=4)
    else:
        ndays = dict(grom=ndays, jeev=ndays)

    for ia, (animal, yr) in enumerate(zip(['grom', 'jeev'], ['2016', '2013'])):
        
        if model_set_number == 1:
            models_to_include = ['prespos_0psh_1spksm_0_spksp_0', 
                                'hist_1pos_-1psh_1spksm_0_spksp_0', 
                                'hist_1pos_1psh_1spksm_0_spksp_0',
                                'hist_1pos_2psh_1spksm_0_spksp_0', 
                                'hist_1pos_3psh_1spksm_0_spksp_0', 
                                'hist_1pos_3psh_1spksm_1_spksp_0',
                                'hist_1pos_0psh_1spksm_1_spksp_0',
                                
                                'hist_1pos_-1psh_1spksm_1_spksp_0', 
                                'hist_1pos_1psh_1spksm_1_spksp_0',
                                'hist_1pos_2psh_1spksm_1_spksp_0']

            models_to_compare = np.array([0, 4, 5])
            models_colors = [[255., 0., 0.], 
                             [101, 44, 144],
                             [101, 44, 144],
                             [101, 44, 144],
                             [101, 44, 144],
                             [39, 169, 225],
                             [39, 169, 225],
                             [39, 169, 225],
                             [39, 169, 225],
                             [39, 169, 225]]
            xlab = [
            '$a_{t}$',
            '$a_{t}, p_{t-1}$',
            '$a_{t}, p_{t-1}, v_{t-1}$',
            '$a_{t}, p_{t-1}, v_{t-1}, tg$',
            '$a_{t}, p_{t-1}, v_{t-1}, tg, tsk$',
            '$a_{t}, p_{t-1}, v_{t-1}, tg, tsk, y_{t-1}$',
            '$a_{t}, y_{t-1}$',
            '$a_{t}, y_{t-1}, p_{t-1}$',
            '$a_{t}, y_{t-1}, p_{t-1}, v_{t-1}$',
            '$a_{t}, y_{t-1}, p_{t-1}, v_{t-1}, tg$']

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

        #### Number of models ######
        M = len(models_to_include)

        #### Make sure match #######
        assert(len(models_colors) == M)

        #### normalize the colors #######
        models_colors = [np.array(m)/256. for m in models_colors]

        ##### fold ? ######
        fold = ['maroon', 'orangered', 'goldenrod', 'teal', 'blue']
        
        if r2_pop:
            pop_str = 'Population'
        else:
            pop_str = 'Indiv'        

        ### Go through each neuron and plot mean true vs. predicted (i.e R2) for a given command  / target combo: 
        if task_spec_ix is None:
            model_dict = pickle.load(open(analysis_config.config[animal+'_pref']+'tuning_models_'+animal+'_model_set%d.pkl'%model_set_number, 'rb'))
        
        else:
            model_dict = pickle.load(open(analysis_config.config[animal+'_pref']+'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen.pkl'%model_set_number, 'rb'))
        
        #### Setup the plot ####
        f, ax = plt.subplots(figsize=(4, 6))
        
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
            tdata = model_dict[i_d, 'spks']
            R2s = dict()

            ###### Get the baseline ####
            if perc_increase: 
                #### Predicted data ? 
                if task_spec_ix is None:
                    pdata = model_dict[i_d, models_to_include[0]]
                else:
                    pdata = model_dict[i_d, models_to_include[0]][:, :, task_spec_ix]

                R2_baseline = util_fcns.get_R2(tdata, pdata, pop = r2_pop)

            for i_mod, mod in enumerate(models_to_include):
                
                ###### Predicted data, cross validated ####
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

            ##### Plot this single day #####
            tmp = []; 
            for i_mod in range(len(models_to_include)):
                tmp.append(R2S[i_mod, i_d])

            #### Line plot of R2 w increasing models #####
            ax.plot(np.arange(M), tmp, '-', color='gray', linewidth = 1.)

        #### Plots total mean ###
        tmp = []; tmp_e = []; 
        for i_mod in range(M):
            tmp2 = []; 
            for i_d in range(ndays[animal]):
                tmp2.append(R2S[i_mod, i_d])
            tmp2 = np.hstack((tmp2))
            tmp2 = tmp2[~np.isnan(tmp2)]

            ## mean 
            tmp.append(np.mean(tmp2))

            ## s.e.m
            tmp_e.append(np.std(tmp2)/np.sqrt(len(tmp2)))

        ### Overal mean 
        for i_mod in range(M):
            ax.bar(i_mod, tmp[i_mod], color = models_colors[i_mod], edgecolor='k', linewidth = 1., )
            ax.errorbar(i_mod, tmp[i_mod], yerr=tmp_e[i_mod], marker='|', color='k')        
        ax.set_ylabel('%s R2, neur, perc_increase R2 %s'%(pop_str, perc_increase))

        ax.set_xticks(np.arange(M))
        ax.set_xticklabels(xlab, rotation=45)#, fontsize=6)

        f.tight_layout()
        if task_spec_ix is None:
            f.savefig(fig_dir + animal + '_%sr2_behav_models_perc_increase%s_model%d.svg'%(pop_str, perc_increase, model_set_number))
        else:
            f.savefig(fig_dir + animal + '_%sr2_behav_models_perc_increase%s_model%d_tsk_spec_%d.svg'%(pop_str, perc_increase, model_set_number, task_spec_ix))
        
        ##### Print stats ####
        if model_set_number == 1:
            R2_stats = np.hstack((R2_stats))
            ix = ~np.isnan(R2_stats)

            if len(ix) < len(R2_stats):
                import pdb; pdb.set_trace()

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
        
        if pt_2: 
            f, ax = plt.subplots(ncols = 2)
            f1, ax1 = plt.subplots(ncols = len(models_to_include), figsize=(12.5, 2.5))
            f2, ax2 = plt.subplots()
            ### now comput R2 ###
            Pred = dict(); Tru = dict(); 
            
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

                ### Neural activity
                R2 = dict(); 
                for i_m, mod in enumerate(models_to_include):
                    R2['co', mod] = []; 
                    R2['obs', mod] = []; 
                    R2['both', mod] = []; 

                for i_mag in range(4):
                    for i_ang in range(8):
                        for i_t in range(2): # Task 
                            for targ in range(8): # target: 
                                ix0 = (commands[:,0] == i_mag) & (commands[:,1] == i_ang) & (tsk == i_t) & (targ == targ)
                                ix0 = np.nonzero(ix0 == True)[0]
                                if len(ix0) > min_obs:

                                    for i_m, model in enumerate(models_to_include):

                                        if task_spec_ix is None:
                                            y_pred = model_dict[i_d, model]
                                        else:
                                            y_pred = model_dict[i_d, model][:, :, task_spec_ix]

                                        ### Get R2 of this observation: 
                                        spk_true = np.mean(y_true[ix0, :], axis=0)
                                        spk_pred = np.mean(y_pred[ix0, :], axis=0)

                                        if i_t == 0:
                                            R2['co', model].append([spk_true, spk_pred])
                                        elif i_t == 1:
                                            R2['obs', model].append([spk_true, spk_pred])

                                        ### Both 
                                        R2['both', model].append([spk_true, spk_pred])

                tsk_cols = ['b']#,'r','k']
                for i_t, tsk in enumerate(['both']):#, 'obs', 'both']):
                    for i_m, model in enumerate(models_to_include):

                        tru_co = np.vstack(( [R[0] for R in R2[tsk, model]] ))
                        pre_co = np.vstack(( [R[1] for R in R2[tsk, model]] ))

                        SSR = np.sum((tru_co - pre_co)**2)# Not over neruons, axis = 0)
                        SST = np.sum((tru_co - np.mean(tru_co, axis=0)[np.newaxis, :])**2)#, axis=0)
                        R2_co_pop = 1 - (SSR/SST)
                        
                        SSR = np.sum((tru_co - pre_co)**2, axis = 0)
                        SST = np.sum((tru_co - np.mean(tru_co, axis=0)[np.newaxis, :])**2, axis=0)
                        R2_co_neur = 1 - (SSR/SST)

                        ax[0].plot(i_m, R2_co_pop, 'k*')
                        ax[0].set_ylabel('R2 -- of mean task/targ/command/neuron (population neuron R2)')
                        ax[0].set_ylim([-1, 1.])

                        ax[1].plot(i_m, np.nanmean(R2_co_neur), 'k*')
                        ax[1].set_ylabel('R2 -- of mean task/targ/command/neuron (individual neuron R2)')
                        ax[1].set_ylim([-1, 1.])

                        ax1[i_m].plot(tru_co.reshape(-1), pre_co.reshape(-1), 'k.', alpha=.2)
                        try:
                            Tru[model].append(tru_co.reshape(-1))
                            Pred[model].append(pre_co.reshape(-1))
                        except:
                            Tru[model] = [tru_co.reshape(-1)]
                            Pred[model] = [pre_co.reshape(-1)]
                            
            for i_m, model in enumerate(models_to_include):
                slp,intc,rv,pv,err =scipy.stats.linregress(np.hstack((Tru[model])), np.hstack((Pred[model])))
                x_ = np.linspace(np.min(np.hstack((Tru[model]))), np.max(np.hstack((Tru[model]))))
                y_ = slp*x_ + intc; 
                ax1[i_m].plot(x_, y_, '-', linewidth=.5)
                ax1[i_m].set_title('%s, \n pv=%.2f\nrv=%.2f\nslp=%.2f' %(model, pv, rv, slp),fontsize=8)


                if model == 'hist_1pos_0psh_0spksm_1_spksp_0':
                    if task_spec_ix is None:
                        y_pred = model_dict[i_d, model]
                    else:
                        y_pred = model_dict[i_d, model][:, :, task_spec_ix]
                    y_true = model_dict[i_d, key]; 

                    SSR = np.sum((y_pred - y_true)**2, axis=0)
                    SST = np.sum((y_true - np.mean(y_true, axis=0)[np.newaxis, :])**2, axis=0)
                    SST[np.isinf(SST)] = np.nan

                    r22 = 1 - SSR/SST; 
                    ax2.plot(r22)

                    print 'R2 mean day %d, %.2f', (i_d, np.mean(r22))

            #f.tight_layout()
            #f1.tight_layout()

### Bar R2 and correlation plots -- figure 4;
def plot_real_vs_pred(model_set_number = 1, min_obs = 15, cov = True):
    
    mag_boundaries = pickle.load(open(analysis_config.config['grom_pref'] + 'radial_boundaries_fit_based_on_perc_feb_2019.pkl'))

    models_to_include = ['prespos_0psh_1spksm_0_spksp_0', 
                         'hist_1pos_3psh_1spksm_0_spksp_0', 
                         'hist_1pos_3psh_1spksm_1_spksp_0']

    models_to_include_labs = ['y_t | a_t', 
                              'y_t | a_t, s_{t-1}, s_tFinal, tsk', 
                              'y_t | a_t, s_{t-1}, y_{t-1}']


    for ia, animal in enumerate(['grom','jeev']):
        model_dict = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d.pkl' %model_set_number, 'rb'))
        ndays = analysis_config.data_params[animal + '_ndays']

        diffs = dict(); 
        sig_diffs = dict(); 

        MOD = []; DAY = []; VAL = []; 

        pred_diffs = dict(); 
        pred_sig_diffs = dict(); 

        R2 = dict(); NN = dict(); 

        fbar, axbar = plt.subplots(figsize=(3, 4))
        #fall, axall = plt.subplots(ncols = len(models_to_include), nrows = ndays, figsize=(10, 10))

        for i_m, model in enumerate(models_to_include):

            ### For each model, setup in teh diffs 
            diffs[model] = []
            sig_diffs[model] = []
            pred_diffs[model] = []
            pred_sig_diffs[model] = []

            for i_d in range(ndays):
                
                R2[model, i_d] = []; 
                ### Get the spiking data
                spks = model_dict[i_d, 'spks']
                pred = model_dict[i_d, model]

                ### Get the task parameters
                tsk  = model_dict[i_d, 'task']
                targ = model_dict[i_d, 'trg']
                push = model_dict[i_d, 'np']
                
                ### Setup the dicitonaries to be 
                diffs[model, i_d] = []
                sig_diffs[model, i_d] = []
                pred_diffs[model, i_d] = []
                pred_sig_diffs[model, i_d] = []

                ### Get the discretized commands
                commands_disc = util_fcns.commands2bins([push], mag_boundaries, animal, i_d, vel_ix = [0, 1])[0]

                ### Now go through combos and plot 
                for mag_i in range(4):
                    for ang_i in range(8):
                        for targi in range(8):
                            ### Get co / task 
                            ix_co = (commands_disc[:, 0] == mag_i) & (commands_disc[:, 1] == ang_i) & (tsk == 0) & (targ == targi)
                            ix_co = np.nonzero(ix_co == True)[0]

                            if len(ix_co) >= min_obs:
                                
                                ### Get info second task: 
                                for targi2 in range(8):
                                    ix_ob = (commands_disc[:, 0] == mag_i) & (commands_disc[:, 1] == ang_i) & (tsk == 0) & (targ == targi2)
                                    ix_ob = np.nonzero(ix_ob == True)[0]

                                    if len(ix_ob) >= min_obs:

                                        assert(len(ix_co) >= min_obs)
                                        assert(len(ix_ob) >= min_obs)
                                        #print 'adding: mag_i: %d, ang_i: %d, targi: %d, targi2: %d' %(mag_i, ang_i, targi, targi2)
                                        ### make the plot: 
                                        if cov: 
                                            diffs[model, i_d] = util_fcns.get_cov_diffs(ix_co, ix_ob, spks, diffs[model, i_d])
                                            pred_diffs[model, i_d] = util_fcns.get_cov_diffs(ix_co, ix_ob, pred, pred_diffs[model, i_d])
                                        else: 
                                            diffs[model, i_d].append(np.mean(spks[ix_co, :], axis=0) - np.mean(spks[ix_ob, :], axis=0))    
                                            pred_diffs[model, i_d].append(np.mean(pred[ix_co, :], axis=0) - np.mean(pred[ix_ob, :], axis=0))


            ### Now scatter plot all data over all days: 
            #f, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(8, 4))

            ### Make R2 plots to show how much each plot accounts for variace: 
            for i_d in range(ndays):

                ### T x N
                x = np.vstack((diffs[model, i_d]))
                y = np.vstack((pred_diffs[model, i_d]))
                
                #axall[i_d, i_m].plot(x, y, '.', markersize=2.)
                ### get variance explained -- here, each point is a neuron / command / day / targ1 / targ 2 difference
                ### the mean for SST is the neuron specific avg. true difference. 
                VAF = util_fcns.get_R2(x, y, pop = True)

                ### Old VAF: 
                #VAF = 1 - np.sum((x-y)**2)/np.sum((x-np.mean(x))**2)

                R2[model, i_d].append(VAF);
                #axall[i_d, i_m].set_title(VAF, fontsize=6)

                MOD.append(i_m)
                DAY.append(i_d)
                VAL.append(VAF)

        fall.tight_layout()
        MOD = np.hstack((MOD))
        DAY = np.hstack((DAY))
        VAL = np.hstack((VAL))

        ### Plot indices ###
        ix0 = np.nonzero(MOD < 2)[0]
        ix1 = np.nonzero(MOD > 0)[0]

        pv0, slp0 = util_fcns.run_LME(DAY[ix0], MOD[ix0], VAL[ix0])
        pv1, slp1 = util_fcns.run_LME(DAY[ix1], MOD[ix1], VAL[ix1])

        print('Animal %s, Mods %s, pv: %.3f, slp: %.3f, N: %d' %(animal, str(np.unique(MOD[ix0])), pv0, slp0, len(ix0)))
        print('Animal %s, Mods %s, pv: %.3f, slp: %.3f, N: %d' %(animal, str(np.unique(MOD[ix1])), pv1, slp1, len(ix1)))

        ### Plot as bar plot ###
        all_data = {}
        for i_m, model in enumerate(models_to_include):
            all_data[model] = []; 

        for i_d in range(ndays):
            tmp = []; 

            for i_m, model in enumerate(models_to_include):
                r2 = np.hstack((R2[model, i_d]))
                r2[np.isinf(r2)] = np.nan
                tmp.append(np.nanmean(r2))
                all_data[model].append(r2)
            axbar.plot(np.arange(len(models_to_include)), tmp, '-', color='gray')


        #### Model colors ###
        model_cols = [[255, 0, 0], [101, 44, 144], [39, 169, 225],]
        model_cols = [np.array(m)/255. for m in model_cols]

        for i_m, model in enumerate(models_to_include):
            tmp2 = np.hstack((all_data[model]))
            tmp2 = tmp2[~np.isnan(tmp2)]
            axbar.bar(i_m, np.mean(tmp2), color = model_cols[i_m], edgecolor='k', linewidth=2.)
            axbar.errorbar(i_m, np.mean(tmp2), yerr=np.std(tmp2)/np.sqrt(len(tmp2)), color = 'k', marker='|')

        axbar.set_xticks(np.arange(len(models_to_include)))
        models_to_include_labs_tex = ['$' + m + '$' for m in models_to_include_labs]
        axbar.set_xticklabels(models_to_include_labs_tex, rotation = 45)#, fontsize=10)
        fbar.tight_layout()
        #fbar.savefig('/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/BMI_co_obs_paper/figures/data_figs/monk_%s_r2_comparison_mean_diffs_cov%s.svg' %(animal, str(cov)))
        
### Fig 5 #### 
### Mean diffs of action at next time step ###
def plot_real_mean_diffs_behavior_next(model_set_number = 3, min_obs = 15):
    ### Take real task / target / command / neuron / day comparisons for each neuron in the BMI
    ### Plot within a bar 
    ### Plot only sig. different ones

    ### Plot cov. diffs (mat1 - mat2)
    mag_boundaries = pickle.load(open(analysis_config.config['grom_pref'] + 'radial_boundaries_fit_based_on_perc_feb_2019.pkl'))
    fsumm, axsumm = plt.subplots(figsize=(4, 4))

    for ia, animal in enumerate(['grom','jeev']):
        model_dict = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d.pkl' %model_set_number, 'rb'))

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
                    for targi in range(8):
                        ix_co = (commands_disc[:, 0] == mag_i) & (commands_disc[:, 1] == ang_i) & (tsk == 0) & (targ == targi)
                        
                        if len(np.nonzero(ix_co == True)[0]) > min_obs:
                            
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
            days.append(np.zeros_like(tmp) + i_d)
            cat.append(np.zeros_like(W))
            cat.append(np.zeros_like(X) + 1)
            met.append(tmp)

            for i, (D, col) in enumerate(zip([X, W], ['k', 'gray'])):
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

        axsumm.bar(0 + ia, np.mean(DX), color='k', edgecolor='none', width=.4, linewidth=2.0, alpha = .8)
        axsumm.bar(0.4 + ia, np.mean(DWI), color='gray', edgecolor='none', width=.4, linewidth=2.0, alpha =.8)

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
    #fsumm.savefig(fig_dir+'both_monks_w_vs_x_task_push_mean_diffs.svg')

### Fig 5 -- identity vs. neural dynamcis
def fig_5_neural_dyn(min_obs = 15, r2_pop = True, 
    model_set_number = 3, ndays = None,):
    
    ### For stats each neuron is an observation ##
    mag_boundaries = pickle.load(open(analysis_config.config['grom_pref'] + 'radial_boundaries_fit_based_on_perc_feb_2019.pkl'))

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
    mag_boundaries = pickle.load(open(analysis_config.config['grom_pref'] + 'radial_boundaries_fit_based_on_perc_feb_2019.pkl'))

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
 
### Generalization of Neural dynamics across tasks #####
### Generalization of neural dynamics -- population R2 ### -- figure 6(?)
def plot_r2_bar_model_7_gen(model_set_number = 7, ndays = None, use_action = False,
    plot_by_day = False, sep_R2_by_tsk = False, match_task_spec_n = False, 
    fit_intercept = True):
    
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
    fbth, axbth = plt.subplots(ncols = 2, nrows = 2, figsize = (8, 8))

    if use_action:
        models_to_include = ['hist_1pos_0psh_1spksm_1_spksp_0']
    else:
        models_to_include = ['hist_1pos_0psh_0spksm_1_spksp_0']

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
                ndays_all = [0, 2, 3];
                ndays = 3; 

        #####################
        ### Real bar plot ###
        #####################
        ### 3 plots -- one for CO, one for OB, one combined
        ### CO / OBS data  x Fit CO/OBS/GEN x ndays
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
        tis = ['Neural', 'Push']

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
            axbth[z, i_a].bar(0, np.mean(pwii), color='w', edgecolor='k', width=.4, linewidth=.2)
            axbth[z, i_a].bar(0.4, np.mean(pxi), color='grey', edgecolor='k', width=.4, linewidth=.2)
            axbth[z, i_a].bar(0.8, np.mean(palli), color='k', edgecolor='k', width=.4, linewidth=.2)
            
        
            for i_d in range(ndays):
                # axco[z, i_a].plot([0, .4], [R2s_plot_spks[0, 0, i_d], R2s_plot_spks[0, 1, i_d]], 'k-', linewidth = 1.)
                # axob[z, i_a].plot([0, .4], [R2s_plot_spks[1, 1, i_d], R2s_plot_spks[1, 0, i_d]], 'k-', linewidth = 1.)

                ### For CO vs. OBS data
                for x in range(2):

                    #### PWII/PXI/PALLI are all NDAYS x 2 --> [day0_0, day0_1, day1_0, day1_1, ...]
                    ### here 
                    if x == 0:
                        axbth[z, i_a].plot([0, .4, .8], [pwii[2*i_d + x], pxi[2*i_d + x], palli[2*i_d + x]], 'k-', linewidth = 1.)
                    elif x == 1:
                        axbth[z, i_a].plot([0, .4, .8], [pwii[2*i_d + x], pxi[2*i_d + x], palli[2*i_d + x]], 'b-', linewidth = 1.)
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


            axbth[z, i_a].set_title('%s, Monk %s'%(tis[z], animal),fontsize = 8)
            axbth[z, i_a].set_ylabel('R2')
            axbth[z, i_a].set_xticks([0., .4, .8])
            axbth[z, i_a].set_xticklabels(['Within', 'Across', 'Both'])
        
        ### stats: 
        print 'Neural, subj %s, w vs x' %(animal)
        w_vs_x = np.hstack(( np.hstack((RW)), np.hstack((RX)) ))
        grps = np.hstack(( np.zeros_like(RW), np.zeros_like(RX)+1))
        pv, slp = util_fcns.run_LME(DAYs, grps, w_vs_x)
        print('pv: %.2f, slp %.2f'%(pv, slp))

        if pv < 0.001: 
            axbth[0, i_a].text(0.2, np.max(w_vs_x), '***')
        elif pv < 0.01: 
            axbth[0, i_a].text(0.2, np.max(w_vs_x), '**')
        elif pv < 0.05:
            axbth[0, i_a].text(0.2, np.max(w_vs_x), '*')

        print 'ACTION, subj %s, w vs x' %(animal)
        w_vs_x = np.hstack(( np.hstack((RW_A)), np.hstack((RX_A)) ))
        grps = np.hstack(( np.zeros_like(RW_A), np.zeros_like(RX_A)+1))
        pv, slp = util_fcns.run_LME(DAYs, grps, w_vs_x)
        print('pv: %.2f, slp %.2f'%(pv, slp))

        if pv < 0.001: 
            axbth[1, i_a].text(0.2, np.max(w_vs_x), '***')
        elif pv < 0.01: 
            axbth[1, i_a].text(0.2, np.max(w_vs_x), '**')
        elif pv < 0.05:
            axbth[1, i_a].text(0.2, np.max(w_vs_x), '*')
        else:
            axbth[1, i_a].text(0.2, np.max(w_vs_x), 'n.s.')

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
    #fbth.savefig('gen_w_vs_x.svg')

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

#### Eigenvalue decomposition 
def eigvalue_plot(dt = 0.1):

    dyn_model = 'hist_1pos_0psh_0spksm_1_spksp_0'
    n_folds = 5; 

    for i_a, animal in enumerate(['grom', 'jeev']):

        ###### Using the task specific models and the general models ######
        ##### Get the one with data matched #######
        dat = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen_match_tsk_N.pkl' %(7), 'rb'))

        ##### for each day and task -- 
        for i_d in range(analysis_config.data_params[animal+'_ndays']):
            f, ax = plt.subplots(ncols = 3, figsize = (10, 4))

            ### For each day, plot the eigenvalues ####
            for i_f in range(n_folds):

                ### CO / OBS / GEN ####
                for i_m in range(3):

                    #### Get the model 
                    model = dat[i_d, dyn_model, n_folds*i_m + i_f, i_m, 'model']


                    #pred0 = np.mat(X0)*np.mat(model[0].coef_).T + model[0].intercept_[np.newaxis, :]
                    
                    #### Get the A matrix; 
                    A = np.mat(model.coef_)
                    N = A.shape[0]

                    ### Add intercept to the A matrix 
                    A = np.vstack((A, np.zeros((1, N))))
                    intc = model.intercept_[:, np.newaxis]
                    intc = np.vstack((intc, [1]))
                    A = np.hstack((A, intc))

                    ### eigenvalues; 
                    ev, evect = np.linalg.eig(A)

                    ### get frequency; 
                    angs = np.array([ np.arctan2(np.imag(ev[i]), np.real(ev[i])) for i in range(len(ev))])
                    hz = np.abs(angs)/(2*np.pi*dt)
                    decay = -1./np.log(np.abs(ev))*dt # Time decay constant in ms
                    ax[i_m].plot(decay, hz, 'k.')
                    ax[i_m].set_xlim([0., .2])
                    ax[i_m].set_ylim([-.01, 4.9])


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
    mag_boundaries = pickle.load(open(analysis_config.config['grom_pref'] + 'radial_boundaries_fit_based_on_perc_feb_2019.pkl'))
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

