
import numpy as np 
import scipy.stats
import scipy.io as sio
import matplotlib.pyplot as plt 
import pickle 

import analysis_config
import generate_models_utils, util_fcns, plot_generated_models

def plot_mean_diffs_all(zscore = False, plot_beh_mn_diffs = False, plot_beh_vs_neur = False):
    
    #### Mean diffs plot ####
    if plot_beh_mn_diffs:
        fraw, axraw = plt.subplots(figsize = (4, 5))
        frel, axrel = plt.subplots(figsize = (4, 5))

    if plot_beh_vs_neur:
        agg_dat = dict()
    
    nShuff = 200; 

    for i_a, animal in enumerate(['grom', 'jeev']): 
        d_beh = []
        d_neur = []
        d_abs_neur = []
        d_neur_shuff = []
        d_abs_neur_shuff = []
        days = []

        for i_d in range(analysis_config.data_params[animal+'_ndays']):

            diff_beh, diff_neur, abs_neur, diff_shuff_neur, abs_shuff_neur = extract_mean_diffs(animal=animal, day =i_d,
                zscore = zscore)

            #### first things first plot mean diffs vs. shuffle mean diffs ####
            d_beh.append(diff_beh)
            d_neur.append(diff_neur)
            d_abs_neur.append(abs_neur)
            d_neur_shuff.append(diff_shuff_neur)
            d_abs_neur_shuff.append(abs_shuff_neur)
            days.append(np.zeros((len(diff_beh))) + i_d)

        if plot_beh_mn_diffs:
            #### For this animal, plot the mean diffs vs. shuffle 
            plot_mean_diffs_w_shuff(axraw, axrel, animal, i_a, d_abs_neur, d_abs_neur_shuff, nShuff)

        if plot_beh_vs_neur:
            ###### Correlation between beh_diff and vector mean diffs for each animal and day #######
            agg_slope, agg_rv = plot_beh_diff_vs_neur_vect_diff(animal, d_beh, d_neur, d_neur_shuff, [1], zscore)
            agg_dat[animal, 'slope'] = agg_slope
            agg_dat[animal, 'rv'] = agg_rv

    ##### Labels for mean diffs vs shuffle #####
    if plot_beh_mn_diffs:
        for _, (f, ax, axtype) in enumerate(zip([fraw,frel], [axraw, axrel], ['raw', 'rel'])):
            ax.set_xlim([-.5, 4.5])
            ax.set_xticks([0, 1, 3, 4])
            ax.set_xticklabels(['shuffled', 'data', 'shuffled', 'data'], rotation=45)
            if zscore:
                if axtype == 'raw':
                    ax.set_ylabel('Mean Z-score Differences (s.d)')
                elif axtype == 'rel':
                    ax.set_ylabel('Frac. Change in Mean Z-score \nDifferences (s.d) vs. Shuffle')
            else:
                if axtype == 'raw':
                    ax.set_ylabel('Mean Differences (Hz)')
                elif axtype == 'rel':
                    ax.set_ylabel('Frac. Change in Mean Diff (Hz)\n vs. Shuffle')

            f.tight_layout()
            f.savefig(analysis_config.config['fig_dir']+'/fig3_mean_diffs_w_shuffle_zsc%s_%s.svg' %(str(zscore), axtype))

    ##### Bar plots for mean slope / rv ###
    if plot_beh_vs_neur:
        plot_bars_beh_vs_neural(agg_dat)
        return agg_dat

def extract_mean_diffs(animal='grom', day=0, lags = 1, min_obs = 15, 
    comparison = 'x_conds', zscore = False):
    '''
    option for comparison; 
        to_command_mn -- compute avg beh for command, then plot condition-specific deviations 
        x_conds -- compute avg beh for CO condition-spec command vs. avg beh for OBS condition-spec command
    '''

    mag_boundaries = pickle.load(open(analysis_config.config['grom_pref'] + 'radial_boundaries_fit_based_on_perc_feb_2019.pkl'))
    if animal == 'grom':
        order_dict = analysis_config.data_params['grom_ordered_input_type']
        input_type = analysis_config.data_params['grom_input_type']
    elif animal == 'jeev':
        order_dict = analysis_config.data_params['jeev_ordered_input_type']
        input_type = analysis_config.data_params['jeev_input_type']        
    
    history_bins_max = 1 # Doing this to match the shuffle data

    _, data_temp, spks, _, sub_push_all = generate_models_utils.get_spike_kinematics(animal,
        input_type[day], order_dict[day], history_bins_max, day_ix = day, within_bin_shuffle = False)

    nNeur = spks.shape[1]
    psh = sub_push_all[:, :, 0]
    task = data_temp['tsk']
    targ = data_temp['trg']
    bin_num = data_temp['bin_num']
    min_bin_num = np.min(bin_num)

    command_bins = util_fcns.commands2bins([psh], mag_boundaries, animal, day, vel_ix = [0, 1])[0]
    
    diff_beh = []; 
    diff_neur = []; 
    abs_neur = []; 
    diff_shuff_neur = []; 
    abs_shuff_neur = []

    #### Get the shuffled spikes themselves from the saved ones ####
    spks_shuff = get_shuffled_spikes(spks, command_bins, animal, day, save=False, load=True)

    #### Conversion from spks to hz; 
    if zscore:
        mn_spks = np.mean(spks, axis=0)
        st_spks = np.std(spks, axis=0)

        #### if std = 0, set to 1. 
        zix = np.nonzero(st_spks == 0.)[0]
        assert(np.all(mn_spks[zix] == 0.))
        st_spks[zix] = 1.

        #### non zero guys 
        nonzix = np.array([z for z in range(len(mn_spks)) if z not in zix])

        ### Zscore properly 
        spks = spks - mn_spks[np.newaxis, :]
        spks = spks / st_spks[np.newaxis, :]
        assert(np.allclose(np.mean(spks,axis=0), 0.))
        assert(np.allclose(np.std(spks[:, nonzix],axis=0), 1.))

        ### Zscore shuffled data with same mf / st fr estimation 
        spks_shuff = spks_shuff - mn_spks[np.newaxis, :, np.newaxis]
        spks_shuff = spks_shuff / st_spks[np.newaxis, :, np.newaxis]
        assert(np.allclose(np.mean(spks_shuff,axis=0), 0.))
        assert(np.allclose(np.std(spks_shuff[:, nonzix, :],axis=0), 1.))

    else:
        #### Convert spks to hz
        spks = spks*10; 
        spks_shuff = spks_shuff*10; 

    ######### For each command #########
    for mag in range(4):
        for ang in range(8):

            ######### Command bins #########
            ix_command = np.nonzero(np.logical_and(command_bins[:, 0] == mag, command_bins[:, 1] == ang))[0]
            ix_command = return_valid_ix(ix_command, lags, bin_num)

            ### If enough compute the means 
            if len(ix_command) >= min_obs:

                ## Means 
                mn_beh, mn_neur = get_mean_beh_neur(ix_command, lags, psh, spks)
                mn_beh_shuff, mn_neur_shuff = get_mean_beh_neur(ix_command, lags, psh, spks_shuff, shuff=True)
                
                #### Condition specific ######
                if comparison == 'to_command_mn':
                    tsks = [0, 1]
                elif comparison == 'x_conds':
                    tsks = [0]

                for i_t in tsks: 
                    for i_trg in np.unique(targ):

                        ### Get conditons ###
                        ix_cond = np.nonzero(np.logical_and(task[ix_command] == i_t, targ[ix_command] == i_trg))[0]

                        ### If length ####
                        if len(ix_cond) >= min_obs:
                            
                            #### Get mean neural / mean beh; 
                            cond_mn_beh, cond_mn_neur = get_mean_beh_neur(ix_command[ix_cond], lags, psh, spks)
                            cond_mn_beh_shuff, cond_mn_neur_shuff = get_mean_beh_neur(ix_command[ix_cond], lags, psh, spks_shuff, shuff=True)

                            if comparison == 'to_command_mn':
                                diff_beh.append(np.linalg.norm(cond_mn_beh - mn_beh))
                                diff_neur.append(np.linalg.norm(cond_mn_neur - mn_neur))
                                abs_neur.append(np.abs(cond_mn_neur-mn_neur))

                                diff_shuff_neur.append(np.linalg.norm(cond_mn_neur_shuff - mn_neur_shuff, axis=0))
                                abs_shuff_neur.append(np.abs(cond_mn_neur_shuff - mn_neur_shuff))

                            elif comparison == 'x_conds':

                                #### Get next condition
                                for i_t2 in [1]:
                                    for i_trg2 in np.unique(targ):
                                        
                                        ### Get conditons ###
                                        ix_cond2 = np.nonzero(np.logical_and(task[ix_command] == i_t2, targ[ix_command] == i_trg2))[0]
                                        
                                        if len(ix_cond2) >= min_obs:
                                            #### Get mean neural / mean beh; 
                                            cond_mn_beh2, cond_mn_neur2 = get_mean_beh_neur(ix_command[ix_cond2], lags, psh, spks)
                                            cond_mn_beh2_shuff, cond_mn_neur2_shuff = get_mean_beh_neur(ix_command[ix_cond2], lags, psh, spks_shuff, shuff=True)

                                            ### Get differences; 
                                            diff_beh.append(np.linalg.norm(cond_mn_beh - cond_mn_beh2))
                                            diff_neur.append(np.linalg.norm(cond_mn_neur - cond_mn_neur2))
                                            abs_neur.append(np.abs(cond_mn_neur-cond_mn_neur2))
                                            
                                            diff_shuff_neur.append(np.linalg.norm(cond_mn_neur_shuff - cond_mn_neur2_shuff, axis=0))
                                            abs_shuff_neur.append(np.abs(cond_mn_neur_shuff - cond_mn_neur2_shuff))


    return np.hstack((diff_beh)), np.hstack((diff_neur)), abs_neur, diff_shuff_neur, abs_shuff_neur
    
    # ######## Plot scatter ######
    # f, ax = plt.subplots()
    # ax.plot(diff_beh, diff_neur, 'k.')
    # slp,intc,rv,pv,err = scipy.stats.linregress(diff_beh, diff_neur)
    # ax.set_title('RV %.4f, PV %.4f' %(rv, pv))

    # if comparison == 'x_conds':
    #     ax.set_xlabel(' || Cond-CO $beh_{t=-1:1}$ for command - Cond-OBS $beh_{t=-1:1}$ for command || ')
    #     ax.set_ylabel(' || Cond-CO $neur_{t=0}$ for command - Cond-OBS $neur_{t=0}$ for command || ')
    
    # elif comparison == 'to_command_mn':
    #     ax.set_xlabel(' || Cond-spec $beh_{t=-1:1}$ for command - mean $beh_{t=-1:1}$ for command || ')
    #     ax.set_ylabel(' || Cond-spec $neur_{t=0}$ for command - mean $neur_{t=0}$ for command || ')

    # ####### Distribution of behavior #####
    # f, ax = plt.subplots()
    # ax.hist(diff_beh, 30)
    # ax.vlines(np.percentile(diff_beh, 50), 0, 40, 'k')
    # prctl = np.percentile(diff_beh, 50)
    # ax.set_xlabel('Beh Diffs (50th percentile marked')
    # ax.set_ylabel('Cnt')

    # ###### Get conditions w/ beh diff > 50th percentile ####
    # ix_big_diff = np.nonzero(diff_beh > prctl)[0]

    # ##### Recompute means for big diff behaviors #####
    # f, ax = plt.subplots()
    # mn_diffs_big = []
    # mn_diffs_sma = []

    # for i, (db, dn) in enumerate(zip(diff_beh, abs_neur)):

    #     ##### plotting #####
    #     if db > prctl: 
    #         mn_diffs_big.append(dn)
    #     else:
    #         mn_diffs_sma.append(dn)

    # #### Bar plots of this; 
    # mn_diffs_big = 10*np.hstack((mn_diffs_big)).reshape(-1)
    # mn_diffs_sma = 10*np.hstack((mn_diffs_sma)).reshape(-1)
    # ax.bar(0, np.mean(mn_diffs_big))
    # ax.errorbar(0, np.mean(mn_diffs_big), np.std(mn_diffs_big)/np.sqrt(len(mn_diffs_big)))
    # ax.bar(1, np.mean(mn_diffs_sma))
    # ax.errorbar(1, np.mean(mn_diffs_sma), np.std(mn_diffs_sma)/np.sqrt(len(mn_diffs_sma)))
    # ax.set_ylabel('Hz')
    # ax.set_xticks([0, 1])
    # ax.set_xticklabels(['Mn diffs for \nbeh diff > 50th', 'Mn diffs for \nbeh diff < 50th'])    

def plot_mean_diffs_w_shuff(axraw, axrel, animal, a_off, d_abs_neur, d_abs_neur_shuff, nShuff):
    
    ##### For this animal plot the mean differences  
    tmp = []; tmp_frac = []
    tmp_shuff = []; pvs = []
    tmp_shuff_mn = []

    for i_d in range(analysis_config.data_params[animal+'_ndays']):

        #### Get mean over neuron/cond1/cond2/command
        day_mn = np.mean(np.hstack((d_abs_neur[i_d])))
        tmp.append(np.mean(np.hstack((d_abs_neur[i_d]))))

        #### how many conditon1/condition2/command comparisons are there? 
        N = len(d_abs_neur_shuff[i_d])
        day_shuff = []

        #### For each shuffle gather the mean 
        for ns in range(nShuff):
            tmp_ns = []
            for ni in range(N):
                tmp_ns.append(d_abs_neur_shuff[i_d][ni][:, ns])

            #### For this shuffle the vals of all 
            tmp_ns = np.hstack((tmp_ns)).reshape(-1)

            ### Append to day shuffle 
            day_shuff.append(np.mean(tmp_ns))
        day_shuff = np.hstack((day_shuff))

        ### append all shuffles from this day to the overall thing 
        tmp_shuff.append(day_shuff - np.mean(day_shuff))
        tmp_shuff_mn.append(np.mean(day_shuff))

        ##### Get P-value ####
        assert(len(day_shuff) == nShuff)
        pv_ix = np.nonzero(day_shuff >= day_mn)[0]
        pv = float(len(pv_ix))/float(len(day_shuff))
        print('Day %d, pv = %.4f, tot_shuff %d'%(i_d, pv, len(day_shuff)))
        pvs.append(pv)

        ##### Plot single day lines 
        axraw.plot([3*a_off, 1 + 3*a_off], [np.mean(day_shuff), tmp[i_d]], '-', color='gray', linewidth=1.)

        #### Relative change for this day: 
        rel_change = (tmp[i_d] - np.mean(day_shuff)) / np.mean(day_shuff)
        tmp_frac.append(rel_change)
        axrel.plot([3*a_off, 1 + 3*a_off], [0., rel_change], '-', color='gray', linewidth=1.)

    ##### Mean difference over all neurons -- RELATIVE 
    util_fcns.draw_plot(3*a_off, np.hstack((tmp_shuff)), 'gray', 'w', axrel, width = .9)
    axrel.bar(1 + 3*a_off, np.mean(np.hstack((tmp_frac))), color='k', width=.9, alpha=.5)
    axrel.errorbar(1 + 3*a_off, np.mean(np.hstack((tmp_frac))), np.std(np.hstack((tmp_frac)))/np.sqrt(len(np.hstack((tmp_frac)))),
        color='k', marker='|')

    #### Raw plot with bars ###
    axraw.bar(0 + 3*a_off, np.mean(np.hstack((tmp_shuff_mn))), color='gray', width=.9, alpha=.8)
    axraw.errorbar(0 + 3*a_off, np.mean(np.hstack((tmp_shuff_mn))), np.std(np.hstack((tmp_shuff_mn)))/np.sqrt(len(np.hstack((tmp_shuff_mn)))),
        color='k', marker='|')
    
    axraw.bar(1 + 3*a_off, np.mean(np.hstack((tmp))), color='k', width=.9, alpha=.8)
    axraw.errorbar(1 + 3*a_off, np.mean(np.hstack((tmp))), np.std(np.hstack((tmp)))/np.sqrt(len(np.hstack((tmp)))),
        color='k', marker='|')
    
    ##### Do LME for raw data ####
    tmp_shuff_mn = np.hstack((tmp_shuff_mn))
    tmp = np.hstack((tmp))
    
    vals = np.hstack((tmp_shuff_mn, tmp ))
    grp = np.hstack(( np.zeros((len(tmp_shuff_mn),)), np.ones((len(tmp), )) ))
    day = np.hstack((np.arange(len(tmp_shuff_mn)), np.arange(len(tmp))))

    pv, slp = util_fcns.run_LME(day, grp, vals)
    pv_str = util_fcns.get_pv_str(pv)

    axraw.plot([3*a_off, 3*a_off + 1], [1.1*np.max(tmp), 1.1*np.max(tmp)], 'k-', linewidth=1.)
    axraw.text(3*a_off + .5, 1.15*np.max(tmp), pv_str, ha='center')

    # if np.all(np.hstack((pvs)) < 0.05):
    #     axrel.plot([3*a_off, 1 + 3*a_off], [1.1*np.max(tmp_frac), 1.1*np.max(tmp_frac)], 'k-', linewidth=1.)
    #     axrel.text(3*a_off + .5, 1.15*np.max(tmp_frac), '***', ha='center')
    #     #axrel.set_ylim([0., -1.2*np.min(tmp_frac)])
    #     yax = axrel.get_ylim()
    #     axrel.set_ylim([yax[0],  1.2*np.max(tmp_frac)])
    # else:
    #     import pdb; pdb.set_trace()

def plot_beh_diff_vs_neur_vect_diff(animal, d_beh, d_neur, d_neur_shuff, plot_days, zscore):

    assert(len(d_beh) == len(d_neur))

    agg_slope = dict(true=[], shuff=[]); 
    agg_rv = dict(true=[], shuff=[]); 

    #### Make a plot for each day ####
    for i_d in range(analysis_config.data_params[animal+'_ndays']):

        dbeh_day = d_beh[i_d]
        dneu_day = d_neur[i_d]

        assert(len(dbeh_day) == len(dneu_day))

        if i_d in plot_days:
            f, ax = plt.subplots(figsize = (5, 5))
            ax.plot(dbeh_day, dneu_day, 'k.')

            ###### Viveks beh perc vs. vector diff plot idea ########
            fv, axv = plt.subplots(figsize = (5, 5))

            #### Stack shuffle for easier access ####
            stacked_shuff = np.vstack((d_neur_shuff[i_d])) ### N x 100
            
            #### Go through percnet thresholds; 
            for perc_thresh in np.arange(10, 95, 5):
                thresh = np.percentile(dbeh_day, perc_thresh)
                ix_high = np.nonzero(dbeh_day > thresh)[0]
                axv.plot(perc_thresh, np.mean(dneu_day[ix_high]), 'k.')
                axv.errorbar(perc_thresh, np.mean(dneu_day[ix_high]), np.std(dneu_day[ix_high])/np.sqrt(len(ix_high)),
                    marker = '|', color='k', elinewidth=.5)

                axv.plot(perc_thresh, np.mean(stacked_shuff[ix_high, :]), '.', color='gray')
                axv.errorbar(perc_thresh, np.mean(stacked_shuff[ix_high, :]), np.std(np.mean(stacked_shuff[ix_high, :], axis=1))/np.sqrt(len(ix_high)),
                    marker ='|', color='gray', elinewidth=.5)
                
            #### Plot the shuffle:
            mn_neur_shuff = np.mean(stacked_shuff, axis = 1)
            for i_n, (db, dns) in enumerate(zip(dbeh_day, mn_neur_shuff)):
                
                if i_d in plot_days:
                    #### Plot the distribution ####
                    ax.plot(db, np.mean(dns), '.', color='gray')
                    ax.errorbar(db, np.mean(dns), np.std(dns), marker='|', color='gray', elinewidth=.5)
            
        cols = ['k', 'gray']
        pv = []
        rv = []
        for i, (yax, ylab) in enumerate(zip([dneu_day, d_neur_shuff[i_d]], ['true', 'shuff'])):
            
            if ylab == 'true':
                slp,intc,rvi,pvi,err = scipy.stats.linregress(dbeh_day, yax)
                agg_slope[ylab].append(slp)
                agg_rv[ylab].append(rvi)
                pv.append(pvi)
                rv.append(rvi)
            elif ylab == 'shuff':
                yax_stack = np.vstack((yax))
                tm1 = []; tm2 = []; 
                for ns in range(yax_stack.shape[1]):
                    slp,intc,rvi,pvi,err = scipy.stats.linregress(dbeh_day, yax_stack[:, ns])
                    tm1.append(slp)
                    tm2.append(rvi)
                agg_slope[ylab].append(np.hstack((tm1)))
                agg_rv[ylab].append(np.hstack((tm2)))

                #### Now do the mean; 
                yax = np.mean(yax_stack, axis=1)
                slp,intc,rvi,pvi,err = scipy.stats.linregress(dbeh_day, yax)
                pv.append(pvi)
                rv.append(rvi)
    
            if i_d in plot_days:
                ### Plot the linear regression 
                x_ = np.array([np.min(dbeh_day), np.max(dbeh_day)])
                y_ = slp*x_ + intc 
                ax.plot(x_, y_, '--', color=cols[i])

        if i_d in plot_days:
            ax.set_xlabel('Mean Norm. Action Segment Diff (cm/sec$)^2$')
            ax.set_title('%s, day %d, pv=%.3f, rv=%.3f\npvshuf=%.3f, rvshuf=%.3f' %(animal, i_d, pv[0], rv[0], pv[1], rv[1]),
                fontsize = 8)
            axv.set_xlabel('Behavioral Diffs Percentile')

            if zscore:
                ax.set_ylabel('Mean Norm. ZSc. Neural Diff (s.d$)^2$')
                axv.set_ylabel('Mean Norm. ZSc. Neural Diff (s.d$)^2$\nfor Action Segments with\nBehavioral Diffs > Percentile')
            else:
                ax.set_ylabel('Mean Norm. Neural Diff (Hz$)^2$')
                axv.set_ylabel('Mean Norm. Neural Diff (Hz$)^2$\nfor Action Segments with\nBehavioral Diffs > Percentile')
            
            f.tight_layout()
            fv.tight_layout()
            f.savefig(analysis_config.config['fig_dir']+'/fig3_zsc%s_%s_day%d_beh_vs_neural.svg' %(str(zscore), animal, i_d))
            fv.savefig(analysis_config.config['fig_dir']+'/fig3_zsc%s_%s_day%d_perc_beh_vs_neural_diff.svg' %(str(zscore), animal, i_d))
    return agg_slope, agg_rv

def plot_bars_beh_vs_neural(agg_dat):
    '''
    bar plots for above fcn 
    '''
    f, ax = plt.subplots(ncols = 2)

    mn_shuff = [[], []]

    #### Animal #####
    for i_a, animal in enumerate(['grom', 'jeev']):

        lme_test = dict(day = [], grp = [], val = [])

        ##### For each animal track LME and % of days that are sig; 
        ##### Metrics #####
        for i_m, met in enumerate(['rv']):#, 'rv']): 

            ##### Shuffle / true ####
            for i_x, (dat, col) in enumerate(zip(['shuff', 'true'], ['gray', 'k'])):
    
                #### RAW Bar plot ####
                ax[0].bar(i_a*3 + i_x, np.mean( np.hstack(( agg_dat[animal, met][dat] )) ), width=.9, color=col)

                #### Relative Bar plot 
                if dat == 'shuff':
                    ndays = len(agg_dat[animal, met][dat])
                    shuff = []; shuff_mn = []
                    for i_d in range(ndays):
                        shuff_dist = agg_dat[animal, met][dat][i_d]

                        #### ABS is used bc fraction greater doesnt makes sense for negative values
                        shuff.append(shuff_dist - np.mean(np.abs(shuff_dist)))
                        shuff_mn.append(np.mean(np.abs(shuff_dist)))

                    util_fcns.draw_plot(3*i_a, np.hstack((shuff)), 'gray', 'w', ax[1], width = .9)
                else:
                    tmp = []
                    for i_d in range(ndays):
                        dat_rel = (agg_dat[animal, met][dat][i_d]  - shuff_mn[i_d])/shuff_mn[i_d]
                        ax[1].plot([i_a*3, i_a*3 + 1], [0, dat_rel], '-', color='gray')
                        tmp.append(dat_rel)
                    ax[1].bar(i_a*3 + 1, np.mean(tmp), width=.9, color=col)

                    #### Plot real p-values ####
                    assert(len(agg_dat[animal, met]['true']) == len(agg_dat[animal, met]['shuff']))

                    ### Plot effect ######
                    for i_d, (t, s) in enumerate(zip(agg_dat[animal, met]['true'], agg_dat[animal, met]['shuff'])):
                        assert(len(s) == 200)
                        pv = float(len(np.nonzero(s >= t)[0])) / float(len(s))
                        print('Animal %s, Day %d, pv %.4f' %(animal, i_d, pv))

                        lme_test['day'].append([i_d, i_d])
                        lme_test['grp'].append([0, 1])
                        lme_test['val'].append([np.mean(s), t])

            ##### Iterate through each day #####
            for i_d in range(len(agg_dat[animal, met][dat])):

                ###### Shuffle / true #####
                tmp_sh = np.mean(agg_dat[animal, met]['shuff'][i_d])
                tmp_dt = agg_dat[animal, met]['true'][i_d]
                ax[0].plot(i_a*3 + np.array([0, 1]), np.array([tmp_sh, tmp_dt]), '-', color='gray')
            
            #### Plot ####
            ax[0].set_ylabel('Correlation Coeff. (r) ')
            ax[1].set_ylabel('Frac. Increase in Correlation\n Coeff. (r) vs. Shuffle')
            ax[1].set_xlim([-.5, 4.5])
            ax[0].set_xlim([-.5, 4.5])

        ###### Plot ######
        print('Animal %s' %animal)
        pv, slp = util_fcns.run_LME(lme_test['day'], lme_test['grp'], lme_test['val']) 
        pv_str = util_fcns.get_pv_str(pv)
        vl = np.hstack((lme_test['val']))
        ax[0].plot([i_a*3, i_a*3 + 1], [1.1*np.max(vl), 1.1*np.max(vl)], 'k-')
        ax[0].text(3*i_a + .5, 1.15*np.max(vl), pv_str, ha='center')

    f.tight_layout()
    f.savefig(analysis_config.config['fig_dir']+'/fig3_rv_of_cc.svg')

########### HELPER FCNS #############
def get_mean_beh_neur(ix_com, lags, push, spks, shuff = False):
    beh = []; 

    if shuff:
        neur = np.mean(spks[ix_com, :, :], axis=0)
    else:
        neur = np.mean(spks[ix_com, :], axis=0)

    for i in ix_com:
        snip = np.arange(i - lags, i + lags + 1)
        beh.append(push[snip, :])
    beh = np.mean(np.vstack((beh)), axis=0)
    return beh, neur

def return_valid_ix(ix_com, lags, bin_num):
    ix_keep = []

    for ix in ix_com:
        snip = np.arange(ix - lags, ix + lags + 1)

        ##### Make sure not at the end ####
        if np.all(snip < len(bin_num)):
            snip_bins = bin_num[snip]

            #### Make sure no trial crossover ###
            if np.all(np.diff(snip_bins) > 0): 
                ix_keep.append(ix)
    if len(ix_keep) == 0:
        return ix_keep
    else:
        return np.hstack(( ix_keep ))

def get_shuffled_spikes(spks, command_bins, animal, day, nShuff=200, save=False, load=False):
    
    if load:
        spks_shuff = sio.loadmat(analysis_config.config['shuff_fig_dir']+'%s_%d_shuffall_spks.mat'%(animal, day))
        spks_shuff = spks_shuff['spks_shuff']
    else:
        T, NN = spks.shape
        spks_shuff = np.zeros((T, NN, nShuff))
        for m in range(4):
            for a in range(8):
                ix_command = np.nonzero(np.logical_and(command_bins[:, 0] == m, command_bins[:, 1] == a))[0]

                ##### Get shuffled 
                for ns in range(nShuff):
                    ix_shuff = np.random.permutation(len(ix_command))
                    spks_shuff[ix_command, :, ns] = spks[ix_command[ix_shuff], :]

        if save:
            sio.savemat(analysis_config.config['shuff_fig_dir']+'%s_%d_shuffall_spks.mat' %(animal, day), dict(spks_shuff=spks_shuff))
    
    assert(np.all(np.sum(spks, axis=0)*nShuff == np.sum(np.sum(spks_shuff, axis=2), axis=0)))
    return spks_shuff

