import seaborn
seaborn.set(font='Arial',context='talk',font_scale=1.5, style='white')

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pickle 

import analysis_config, util_fcns
import generate_models

import scipy

fig_dir = analysis_config.config['fig_dir']

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
def plot_real_mean_diffs(model_set_number = 3, min_obs = 15, plot_ex = False, plot_disc = False, cov = False):
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
    perc_increase = True, model_set_number = 1):
    
    '''
    inputs: min_obs -- number of obs need to count as worthy of comparison; 
    perc_increase: get baseline from y_t | a_t and use this to compute the "1" value; 
    ndays ? 
    pt_2 ? 
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
                                'hist_1pos_3psh_1spksm_1_spksp_0']
            models_to_compare = np.array([0, 4, 5])
            models_colors = [[255., 0., 0.], 
                             [101, 44, 144],
                             [101, 44, 144],
                             [101, 44, 144],
                             [101, 44, 144],
                             [39, 169, 225]]
            xlab = [
            '$a_{t}$',
            '$a_{t}, p_{t-1}$',
            '$a_{t}, p_{t-1}, v_{t-1}$',
            '$a_{t}, p_{t-1}, v_{t-1}, tg$',
            '$a_{t}, p_{t-1}, v_{t-1}, tg, tsk$',
            '$a_{t}, p_{t-1}, v_{t-1}, tg, tsk, y_{t-1}$']

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
        model_dict = pickle.load(open(analysis_config.config[animal+'_pref']+'tuning_models_'+animal+'_model_set%d.pkl'%model_set_number, 'rb'))
            
        #### Setup the plot ####
        f, ax = plt.subplots(figsize=(4, 4))
        
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
                pdata = model_dict[i_d, models_to_include[0]]
                R2_baseline = util_fcns.get_R2(tdata, pdata, pop = r2_pop)

            for i_mod, mod in enumerate(models_to_include):
                
                ###### Predicted data, cross validated ####
                pdata = model_dict[i_d, mod]

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
                if perc_increase:
                    tmp.append(R2S[i_mod, i_d])
                else:
                    tmp.append(np.nanmean(np.hstack((R2S[i_mod, i_d]))))

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
        ax.set_ylabel('%s R2, neur, perc_increase R2 %s'%(pop_str, perc_increase), fontsize=8)

        ax.set_xticks(np.arange(M))
        ax.set_xticklabels(xlab, rotation=45, fontsize=6)

        f.tight_layout()
        f.savefig(fig_dir + animal + '_%sr2_behav_models_perc_increase%s_model%d.svg'%(pop_str, perc_increase, model_set_number))
        
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

                                        y_pred = model_dict[i_d, model]

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
                    y_pred = model_dict[i_d, model]
                    y_true = model_dict[i_d, key]; 

                    SSR = np.sum((y_pred - y_true)**2, axis=0)
                    SST = np.sum((y_true - np.mean(y_true, axis=0)[np.newaxis, :])**2, axis=0)
                    SST[np.isinf(SST)] = np.nan

                    r22 = 1 - SSR/SST; 
                    ax2.plot(r22)

                    print 'R2 mean day %d, %.2f', (i_d, np.mean(r22))

            #f.tight_layout()
            #f1.tight_layout()