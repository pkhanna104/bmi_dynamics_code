######## OLDER METHODS FOR EXTRACTING TUNING TRENDS ##########
######### Has code with insanely rigorous compariosns of mFR over time to ensure effect is not due to drift ########

# ** ** CO Obs Tuning Curves
# Updated April 2017-Copy_with_custom_fit_by_day_radial_tuning-July 2017-Figure1FG,3G-Copy1

import pickle, os

import co_obs_tuning_matrices, analysis_config, util_fcns
from resim_ppf import file_key as fk

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pickle
import scipy.stats

from matplotlib.colors import LinearSegmentedColormap
colors = [(1, 1, 1), (46/255., 48/255., 146/255.)]  # W-->Blue
n_bin = 100  # Discretizes the interpolation into bins
cmap_name = 'blue_black'
cmb = LinearSegmentedColormap.from_list(
    cmap_name, colors, N=n_bin)

colors = [(1, 1, 1), (0/255., 102/255., 51/255.)]  # W-->B
cmap_name = 'green_black'
cmg = LinearSegmentedColormap.from_list(
    cmap_name, colors, N=n_bin)

lightred = [222/255.,100/255.,108/255.]
lightgrey = [109/255.,110/255.,112/255.]

# This is saved in grom, but has data for grom and jeev: 
#pref = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/'
mag_bins = pickle.load(open(analysis_config.config['grom_pref']+'radial_boundaries_fit_based_on_perc_feb_2019.pkl'))

### First set of plots: 
###########################
### Get metrics : ########
###########################

def plot_all(animals=['jeev', 'grom'], normalize=True):
    
    # Get the sim plots; 
    sim = None
    for animal in animals:
        sim = distrib_of_neural_pushs_by_day(animal=animal, sim_ax=sim)

    sim.set_xticks([1., 2.])
    sim.set_xticklabels(['G', 'J'], fontsize=14.)
    sim.set_ylabel('Similarity Index', fontsize=14.)
    sim.set_yticks(np.arange(0., 1.1, 0.5))
    sim.set_yticklabels(np.arange(0., 1.1, 0.5), fontsize=14.)

    # Mean diff_mfr
    for animal in animals:
        all_days, all_days_dist, norm_params = mean_diff_mFR(animal=animal, normalize_neur = normalize)

        # Plot it: 
        all_bars = plot_x_task_vs_win_task(all_days, all_days_dist, animal, normalize=normalize)

        # Get all_bars for within vs. across task: 
        all_bars = win_near_vs_far(animal, norm_params, min_samples=15, all_bars=all_bars)
        all_bars = xtask_near_vs_far(animal, norm_params, min_sample=15, all_bars=all_bars)

        # Stats on all: 
        dat_ = all_bars_to_df(all_bars)

def generate_hist_tuning(animal='jeev'):
    ''' notes say that we need to run 
            co_obs_tuning_curves_fig3.mean_diff_mFR
        this fcn depends on a file named: "hist_tuning_w_fit_mag_boundaries_jeev_16_trials_dist_day"
        this fcn is to generate that file 
    '''
    ### Gromit ####
    ### fname_pref is the file name to save 
    ### 
    if animal == 'grom':
        input_type = analysis_config.data_params['grom_input_type']
        co_obs_tuning_matrices.get_all_tuning(input_type, None, None, animal = 'grom',
                fname_pref = 'hist_tuning_w_fit_mag_boundaries_grom_16_trials_dist_', radial_binning=True, 
                mag_thresh='/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/radial_boundaries_fit_based_on_perc_feb_2019.pkl',
                sample_n_trials=16, pre_go = 0.)
    
    elif animal == 'jeev':
        input_type = analysis_config.data_params['jeev_input_type']
        co_obs_tuning_matrices.get_all_tuning(input_type, None, None, animal = 'jeev',
                fname_pref = 'hist_tuning_w_fit_mag_boundaries_jeev_16_trials_dist_', radial_binning=True, 
                mag_thresh='/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/radial_boundaries_fit_based_on_perc_feb_2019.pkl',
                sample_n_trials=16, pre_go = 0.)
    
def distrib_of_neural_pushs_by_day(animal = 'jeev', sim_ax = None, vmax = .25):
    input_type = analysis_config.data_params[animal+'_input_type']
    names = analysis_config.data_params[animal+'_names']
    fname_pref = analysis_config.config[animal+'_pref']+'hist_tuning_w_fit_mag_boundaries_'+animal+'_16_trials_dist_day'
    days = analysis_config.data_params[animal+'_ndays']

    if animal == 'jeev':
        x_bar = 2
    
    elif animal == 'grom':
        x_bar = 1
    ############################################################################
    ############# PLOT histogram of how many commands per division #############
    ############################################################################
    ix = 1
    A = np.linspace(0., 2*np.pi, 9) + np.pi/8

    for select_day in range(days):
        
        f, ax = plt.subplots(ncols = 2)
        f.set_figheight(4)
        f.set_figwidth(8)
        
        R = np.array([0]+mag_bins[animal, select_day]+[mag_bins[animal, select_day][2]+1])
        fname = fname_pref + str(select_day)
        dat = sio.loadmat(fname+'.mat')
        dat_dist = sio.loadmat(fname+'dist'+'.mat')

        # Only takes into account the first block for similarity comparison
        key_co = str(('day'+str(select_day), 'tsk0', 'n0'))
        key_ob = str(('day'+str(select_day), 'tsk1', 'n0'))
        
        #
        hist_co = dat[key_co] # bins x bins x neurons x hist
        hist_obs = dat[key_ob] # bins x bins x neurons x hist
        
        # Probability distributions
        pco = np.sum(hist_co[:, :, ix, :], axis=2).T/np.sum(hist_co[:, :, ix, :])
        pco = np.hstack((np.vstack((pco, np.zeros((1, 8)))), np.zeros((5, 1))))
        
        pob = np.sum(hist_obs[:, :, ix, :], axis=2).T/np.sum(hist_obs[:, :, ix, :])
        pob = np.hstack((np.vstack((pob, np.zeros((1, 8)))), np.zeros((5, 1))))
        
        c0 = polar_plot(R, A, pco, ax=ax[0], vmin=0, vmax=vmax, cmap='Greys')
        c = polar_plot(R, A, pob, ax=ax[1], vmin=0., vmax=vmax, cmap='Greys')  

        plt.tight_layout()
        plt.colorbar(c, ax = ax[1],fraction=0.046, pad=0.04, ticks=np.arange(0., .6, .025))

    ##############################
    ###### SIMILARITY PLOT #######
    ##############################
    neuron_ix = 1 # doesnt matter at all

    if sim_ax is None:
        f, sim_ax = plt.subplots()
        f.set_figheight(4)
        f.set_figwidth(2)
        
    bar_y = []
    for select_day in range(days):    
        fname = fname_pref + str(select_day)
        dat = sio.loadmat(fname+'.mat')
        dat_dist = sio.loadmat(fname+'dist'+'.mat')

        co_nblks = len(input_type[select_day][0])
        obs_nblks = len(input_type[select_day][1])
        
        for c in range(co_nblks):
            key_co = str(('day'+str(select_day), 'tsk0', 'n'+str(c)))
            hist_co = dat[key_co] # bins x bins x neurons x hist
            if c==0:
                # How many times does this command show up? 
                pco = np.sum(hist_co[:, :, neuron_ix, :], axis=2).T
                #pcosum = np.sum(hist_co[:, :, neuron_ix, :])            
            else:
                pco = pco + np.sum(hist_co[:, :, neuron_ix, :], axis=2).T
                #pcosum = pcosum + np.sum(hist_co[:, :, neuron_ix, :])
        pco = pco.reshape(-1)/np.linalg.norm(pco.reshape(-1))
        
        for c in range(obs_nblks):
            key_obs = str(('day'+str(select_day), 'tsk1', 'n'+str(c)))
            hist_obs = dat[key_obs] # bins x bins x neurons x hist
            if c==0:
                pobs = np.sum(hist_obs[:, :, neuron_ix, :], axis=2).T
                pobsum = np.sum(hist_obs[:, :, neuron_ix, :])            
            else:
                pobs = pobs + np.sum(hist_obs[:, :, neuron_ix, :], axis=2).T
                pobsum = pobsum + np.sum(hist_obs[:, :, neuron_ix, :])
        pob = pobs.reshape(-1)/np.linalg.norm(pobs.reshape(-1)) 
        bar_y.append(np.dot(pco, pob))

    x_temp = np.random.randn(days)*.1
    sim_ax.plot(x_temp+x_bar, bar_y, '.', color='grey')
    sim_ax.bar(x_bar, np.mean(bar_y), color='white', edgecolor='black', linewidth=4)

    return sim_ax   

def get_important_units(animal, day_index, est_type = 'est'):
    if animal == 'grom': 
        ### Load the thing: 
        imp_file = pickle.load(open('/Users/preeyakhanna/fa_analysis/online_analysis/grom_important_neurons_svd_feb2019.pkl', 'rb'))
    
    elif animal == 'jeev':
        ### Load the thing: 
        imp_file = pickle.load(open('/Users/preeyakhanna/fa_analysis/online_analysis/jeev_important_neurons_svd_feb2019.pkl', 'rb')) 

    return np.sort(imp_file[day_index, animal, est_type])

def mean_diff_mFR(animal='jeev', normalize_neur = True, min_observations=15, only_important_neurons = False):

    ''' method to get mFR diffs for each neuron / each task '''

    if animal == 'jeev':
        fname_pref='/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/hist_tuning_w_fit_mag_boundaries_jeev_16_trials_dist_day'
        days = 4
    elif animal == 'grom':
        fname_pref='/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/hist_tuning_w_fit_mag_boundaries_grom_16_trials_dist_day'
        days = 9
        
    ############################
    ####### Tuning Diffs #######
    ############################

    # Method to compute the mFR of each neuron | neural push velocity commands
    # Also computes the mFR of each neuron | subsets of nerual push velocity commands
    all_days = {}
    all_days_dist = {}
    norm_params = {}

    for select_day in range(days):

        # Load the file: 
        fname = fname_pref + str(select_day)
        dat = sio.loadmat(fname+'.mat')
        dat_dist = sio.loadmat(fname+'dist'+'.mat')

        # Get the keys for the first task for CO and OBS: 
        key_co = str(('day'+str(select_day), 'tsk0', 'n0'))
        key_ob = str(('day'+str(select_day), 'tsk1', 'n0'))

        hist_co = dat[key_co] # bins x bins x neurons x hist
        hist_obs = dat[key_ob] # bins x bins x neurons x hist

        hist_co_dist = dat_dist[key_co]
        hist_obs_dist = dat_dist[key_ob]
        
        # Get the rest of the task blocks for each task (i.e. join multiple TEs)
        for n in range(1, 10):
            try: 
                key_co = str(('day'+str(select_day), 'tsk0', 'n'+str(n)))
                hist_co += dat[key_co]
                hist_co_dist += dat_dist[key_co]
            except:
                pass
            
            try:
                key_ob = str(('day'+str(select_day), 'tsk1', 'n'+str(n)))
                hist_obs += dat[key_obs]
                hist_obs_dist += dat_dist[key_obs]
            except:
                pass
            
        #####################################
        #######  Get Mean Tuning and  #######
        #####################################
        bins_ang, bins_mag, nneurons, nd = hist_co.shape
        
        if only_important_neurons:
            neurons_to_analyze = get_important_units(animal, select_day, est_type = 'est')
            nneurons = len(neurons_to_analyze)
        else:
            neurons_to_analyze = range(nneurons)

        # Creat the discs: 
        DLambda_overall = np.zeros((bins_ang*bins_mag, nneurons, 2)) # bins x Neurons x Task
        DLambda_overall[:] = np.nan
        DLambda_overall_dist = np.zeros((bins_ang*bins_mag, nneurons, 2, 2)) # bins x Neurons x Task x Distribution
        DLambda_overall_dist[:] = np.nan

        for i_n, n in enumerate(neurons_to_analyze):
            
            ###################################
            ### Total mFRs ###
            ###################################
            
            ix_co = hist_co[:, :, n, :].reshape(bins_ang*bins_mag, nd)
            ix_ob = hist_obs[:, :, n, :].reshape(bins_ang*bins_mag, nd)

            # Get mean and std. for neuron: 
            if normalize_neur: 
                # Use all observations regardless of whether it fulfills min_obs:
                co_all = []
                ob_all = []
                for d in range(nd):
                    co_all.append([d]*int(np.sum(hist_co[:, :, n, d])))
                    ob_all.append([d]*int(np.sum(hist_obs[:, :, n, d])))
                all_ = np.hstack(( np.hstack((co_all)), np.hstack((ob_all)) ))

                neural_mean = float(np.mean(all_))
                neural_std = float(np.std(all_))

            else:
                neural_mean = 0.
                neural_std = 1.

            # Only use bins with more than the min number of observations: 
            ix_0 = np.nonzero(np.sum(ix_co, axis=1) >= min_observations)[0]
            ix_1 = np.nonzero(np.sum(ix_ob, axis=1) >= min_observations)[0]
            
            # only keep ix that meet the min requriment for both tasks: 
            ix_keep = np.array([i for i in range(bins_ang*bins_mag) if np.logical_and(i in ix_0, i in ix_1)])
            
            # For each bin: 
            for ix_keep_i in ix_keep:
                # Bins that have enough observations in each task: 
                # Turn the histogram into a list of values: 
                all_co1 = ix_co[ix_keep_i, :]
                all_co = np.hstack(( [[i]*int(all_co1[i]) for i in range(nd)] ))
                
                all_ob1 = ix_ob[ix_keep_i, :]
                all_ob = np.hstack(( [[i]*int(all_ob1[i]) for i in range(nd)] ))
                
                # Get the mean for each neuron:  
                DLambda_overall[ix_keep_i, i_n, 0] = ( np.mean(all_co) - neural_mean ) / neural_std
                DLambda_overall[ix_keep_i, i_n, 1] = ( np.mean(all_ob) - neural_mean ) / neural_std
                         
                ## Also compute MEAN differences:
                ix_c = np.random.permutation(len(all_co))
                ix_o = np.random.permutation(len(all_ob))

                # Two subsets of within / across tasks
                ix_ci = ix_c[:int(np.floor(len(ix_c)/2.))]
                ix_ci2 = ix_c[int(np.floor(len(ix_c)/2.)):]

                ix_oi = ix_o[:int(np.floor(len(ix_o)/2.))]
                ix_oi2 = ix_o[int(np.floor(len(ix_o)/2.)):]
                
                DLambda_overall_dist[ix_keep_i, i_n, 0, 0] = ( np.mean(all_co[ix_ci]) - neural_mean ) / neural_std
                DLambda_overall_dist[ix_keep_i, i_n, 0, 1] = ( np.mean(all_co[ix_ci2]) - neural_mean ) / neural_std
                
                DLambda_overall_dist[ix_keep_i, i_n, 1, 0] = ( np.mean(all_ob[ix_oi]) - neural_mean ) / neural_std
                DLambda_overall_dist[ix_keep_i, i_n, 1, 1] = ( np.mean(all_ob[ix_oi2]) - neural_mean ) / neural_std
        
                all_days[select_day] = DLambda_overall
                all_days_dist[select_day] = DLambda_overall_dist

                norm_params[select_day, i_n] = [neural_mean, neural_std]

    return all_days, all_days_dist, norm_params

def mean_diff_cov(animal='jeev', min_observations = 15, only_important_neurons = False, main_subspace_ov = True):
    ''' 
    Method to get within task and across mean covariance -- similar to mean_diff_mFR
    but for covariance
    '''
    if animal == 'jeev':
        fname_pref='/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/hist_tuning_w_fit_mag_boundaries_jeev_16_trials_dist_day'
        days = 4
    elif animal == 'grom':
        fname_pref='/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/hist_tuning_w_fit_mag_boundaries_grom_16_trials_dist_day'
        days = 9

    COV_all = dict()

    for select_day in range(days):
        # Load the file: 
        fname = fname_pref + str(select_day)
        dat = pickle.load(open(fname+'.pkl'))
        dat_dist = pickle.load(open(fname+'dist'+'.pkl'))

        Data = dict(ang=[], mag=[], binned_spikes=[], task=[])

        ### Iterate through task sections and add-ons and get info
        for task in [0, 1]: 
            for tsk_it in range(10):
                for key in ['binned_spikes', 'ang', 'mag']:
                    k = tuple((('day'+str(select_day), 'tsk'+str(task), 'n'+str(tsk_it)), key))
                    if k in dat.keys():
                        if key in ['ang', 'mag']:
                            Data[key].append(np.hstack(( dat[k])))
                        else:
                            Data[key].append(np.vstack(( dat[k])))
                        add = True
                        if key == 'ang':
                            print('adding animal %s day %d, task %d, block %d' %(animal, select_day, task, tsk_it))
                    
                    else:
                        add = False
                if add: 
                    Data['task'].append(task + np.zeros_like(np.hstack((dat[k])) ))
        
        for k in Data.keys():
            if k == 'binned_spikes':
                Data[k] = np.vstack((Data[k]))
            else:
                Data[k] = np.hstack((Data[k]))

        ### Make a collection of acceptable bins to analyze: 
        accept = dict()
        for task in [0, 1]:
            accept[task] = []
            task_ix = np.nonzero(Data['task'] == task)[0]

            for ang in range(8):
                for mag in range(4):
                    ix1 = np.nonzero(np.logical_and(Data['ang'][task_ix] == ang, Data['mag'][task_ix] == mag))[0]
                    if len(ix1) > min_observations:
                        accept[task].append([ang, mag])

        ### Only important neurons: 
        if only_important_neurons:
            neuron_ix = get_important_units(animal, select_day)
        else:
            neuron_ix = np.arange(Data['binned_spikes'].shape[1])

        cov_ov_win = []
        cov_ov_x = []

        ### Cool now go through task0 and only compute cov matrices for 
        task_ix0 = np.nonzero(Data['task'] == 0)[0]
        task_ix1 = np.nonzero(Data['task'] == 1)[0]
        
        for bin in accept[0]:
            if bin in accept[1]:

                ### Proceed and compute covariance: 
                ix0 = np.nonzero(np.logical_and(Data['ang'][task_ix0] == bin[0], Data['mag'][task_ix0] == bin[1]))[0]
                ix1 = np.nonzero(np.logical_and(Data['ang'][task_ix1] == bin[0], Data['mag'][task_ix1] == bin[1]))[0]

                ### Get binned spike counts: 
                bs0 = Data['binned_spikes'][task_ix0][np.ix_(ix0, neuron_ix)]
                bs1 = Data['binned_spikes'][task_ix1][np.ix_(ix1, neuron_ix)]

                ### Split into two partitions: 
                _ix0 = np.random.permutation(bs0.shape[0])
                ix0_0 = _ix0[:len(_ix0)/2]
                ix0_1 = _ix0[len(_ix0)/2:]

                _ix1 = np.random.permutation(bs1.shape[0])
                ix1_0 = _ix1[:len(_ix1)/2]
                ix1_1 = _ix1[len(_ix1)/2:]

                ### Task 0
                cov0_0 = np.mat(np.cov(bs0[ix0_0].T))
                cov0_1 = np.mat(np.cov(bs0[ix0_1].T))
                
                ### Task 1
                cov1_0 = np.mat(np.cov(bs1[ix1_0].T))
                cov1_1 = np.mat(np.cov(bs1[ix1_1].T))
                
                ### Get the covariance overlaps within / across: 
                win0 = util_fcns.get_overlap(None, None, 
                    first_UUT=cov0_0, second_UUT=cov0_1, main=main_subspace_ov)

                win1 = util_fcns.get_overlap(None, None, 
                    first_UUT=cov1_0, second_UUT=cov1_1, main=main_subspace_ov)
                
                ### Randomly choose which cross-task to do: 
                x0 = util_fcns.get_overlap(None, None, 
                    first_UUT=cov0_0, second_UUT=cov1_1, main=main_subspace_ov)
                
                x1 = util_fcns.get_overlap(None, None, 
                    first_UUT=cov0_1, second_UUT=cov1_0, main=main_subspace_ov)

                cov_ov_win.append(win0)
                cov_ov_win.append(win1)
                cov_ov_x.append(x0)
                cov_ov_x.append(x1)

        COV_all[select_day, 'x'] = cov_ov_x
        COV_all[select_day, 'win'] = cov_ov_win
        
    return COV_all
                
def plot_diff_cov(cov, X = None, WIN = None, actually_mean = False, ylim = None, ystar = None, ylabel = None):

    ### Overlap plot
    f, ax = plt.subplots(figsize=(3, 4))

    ### Principle angles plot
    f2, ax2 = plt.subplots(figsize=(3, 4))

    f3, ax3 = plt.subplots()

    if np.logical_and(X is None, WIN is None):
        X = []; 
        WIN = []; 

        for k in cov.keys():
            if k[1] == 'x':
                X.append(cov[k])
            elif k[1] == 'win':
                WIN.append(cov[k])

    X = np.vstack((X))
    WIN = np.vstack((WIN))

    #### Stats ###
    tv, pv_ttest = scipy.stats.ttest_ind(X[:, 0], WIN[:, 0])

    #ax.plot(np.random.randn(X.shape[0])*.1, X[:, 0], 'k.')
    ax.bar(0, np.mean(X[:, 0]), color='white', edgecolor='black', linewidth=4)
    ax.errorbar(0, np.mean(X[:, 0]), np.std(X[:, 0])/np.sqrt(len(X)), 
        marker = '.', color='k')

    #ax.plot(np.random.randn(WIN.shape[0])*.1 + 1, WIN[:, 0], 'k.')
    ax.bar(1, np.mean(WIN[:, 0]), color='white', edgecolor='black', linewidth=4)
    ax.errorbar(1, np.mean(WIN[:, 0]), np.std(WIN[:, 0])/np.sqrt(len(WIN)), 
        marker = '.', color='k')

    if ylabel is None:
        ax.set_ylabel('Subspace Overlap', fontsize=14)
    else:
        ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Across \nTask', 'Within \nTask'], fontsize=14)
    if ylim is None:
        ax.plot([0.0, 1.0], [.9, .9], 'k-', linewidth=4.)
    else:
        ax.plot([0., 1.], [ylim[1]*.8, ylim[1]*.8], 'k-', linewidth=4.)

    if ystar is None:
        ystar = .925

    if pv_ttest < .001:
        ax.text(0.5, ystar, '***', fontsize=28, horizontalalignment='center')
    elif pv_ttest < .01:
        ax.text(0.5, ystar, '**', fontsize=28, horizontalalignment='center')
    elif pv_ttest < .05:
        ax.text(0.5, ystar, '*', fontsize=28, horizontalalignment='center')

    if actually_mean:
        pass
    else:
        tv, pv_ttest = scipy.stats.ttest_ind(X[:, 1], WIN[:, 1])
        #ax2.plot(np.random.randn(X.shape[0])*.1, X[:, 1], 'k.')
        ax2.bar(0, np.mean(X[:, 1]), color='white', edgecolor='black', linewidth=4)
        ax2.errorbar(0, np.mean(X[:, 1]), np.std(X[:, 1])/np.sqrt(len(X)), 
            marker = '.', color='k')

        #ax2.plot(np.random.randn(WIN.shape[0])*.1 + 1, WIN[:, 1], 'k.')
        ax2.bar(1, np.mean(WIN[:, 1]), color='white', edgecolor='black', linewidth=4)
        ax2.errorbar(1, np.mean(WIN[:, 1]), np.std(WIN[:, 1])/np.sqrt(len(WIN)), 
            marker = '.', color='k')

        ax2.set_ylabel('Min Principle Angle', fontsize=14)
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(['Across \nTask', 'Within \nTask'], fontsize=14)
        ax2.plot([0.0, 1.0], [.5, .5], 'k-', linewidth=4.)
        ystar = .52
        if pv_ttest < .001:
            ax2.text(0.5, ystar, '***', fontsize=28, horizontalalignment='center')
        elif pv_ttest < .01:
            ax2.text(0.5, ystar, '**', fontsize=28, horizontalalignment='center')
        elif pv_ttest < .05:
            ax2.text(0.5, ystar, '*', fontsize=28, horizontalalignment='center')

        ### Upper bound: 
        ax3.plot(X[:, 1], X[:, 0], '.')
        ang_diff = np.arange(0, np.pi, .1)
        upper_bound = np.cos(ang_diff)
        ax3.plot(ang_diff, upper_bound, 'r.-')

###########################
### Mean FR Plots: ########
###########################
def plot_x_task_vs_win_task(all_days, all_days_dist, animal, all_bars=None, normalize=True,
    avg_across_neur = True):

    if animal == 'grom':
        days = 9

    elif animal == 'jeev':
        days = 4

    ############################################
    ###  Compare X task vs. W/IN task ###
    ############################################    
    if all_bars is None:
        all_bars = {}

    xtask = []
    intask = []
    co = []
    obs = []

    # Plot within task vs. across task: 
    f, ax = plt.subplots(figsize=(3, 4))

    x = []
    y = []
    
    for select_day in range(days):

        DLambda = all_days[select_day] # bins x Neurons x Task
        DLambda_dist = all_days_dist[select_day] # bins x Neurons x Task x Distribution

        if avg_across_neur:
            # AVERAGE across NEURONS. Each point is a bin on a day (X2 for w/in vs. across task comparison)
            # Across task: 
            diff_x = np.nanmean( np.dstack(( np.abs(DLambda_dist[:, :, 0, 0] - DLambda_dist[:, :, 1, 0]),
                np.abs(DLambda_dist[:, :, 0, 1] - DLambda_dist[:, :, 1, 1]) )), axis=1)

            # Within task: 
            diff_win = np.nanmean(np.dstack(( np.abs(DLambda_dist[:, :, 0, 0] - DLambda_dist[:, :, 0, 1]),
                np.abs(DLambda_dist[:, :, 1, 0] - DLambda_dist[:, :, 1, 1]) )), axis=1)
        
        else:
            diff_x = np.dstack(( np.abs(DLambda_dist[:, :, 0, 0] - DLambda_dist[:, :, 1, 0]),
                np.abs(DLambda_dist[:, :, 0, 1] - DLambda_dist[:, :, 1, 1]) ))

            # Within task: 
            diff_win = np.dstack(( np.abs(DLambda_dist[:, :, 0, 0] - DLambda_dist[:, :, 0, 1]),
                np.abs(DLambda_dist[:, :, 1, 0] - DLambda_dist[:, :, 1, 1]) ))

        # There will be NaNs in this: 
        xtask.append(diff_x.reshape(-1))
        intask.append(diff_win.reshape(-1))

        # ax.plot(select_day + np.random.randn(len(diff_x.reshape(-1)))*.05, diff_x.reshape(-1), '.', color='pink')
        # ax.bar(select_day, np.nanmean(diff_x.reshape(-1)), color='red', width=0.4)

        # ax.plot(select_day + 0.4 + np.random.randn(len(diff_win.reshape(-1)))*.05, diff_win.reshape(-1), '.', color='gray')
        # ax.bar(select_day+0.4, np.nanmean(diff_win.reshape(-1)), color='black', width=0.4)

        all_bars['xtask_vs_win', select_day] = np.abs(diff_x - diff_win)

    # Each point is a neuron(20)-bin(32)-day(4) comparison
    XT = np.hstack((xtask)).reshape(-1)
    IT = np.hstack((intask)).reshape(-1)

    all_bars['xtask_vs_win', 'xtask'] = XT
    all_bars['xtask_vs_win', 'win'] = IT

    XT = 10*XT[~np.isnan(XT)] # Convert to Hz? 
    IT = 10*IT[~np.isnan(IT)]

    print('Number of points: %d' %len(XT))

    ax.bar(0, np.mean(XT), color='white', edgecolor='black', linewidth=4)
    ax.errorbar(0, np.mean(XT), yerr=np.std(XT)/np.sqrt(len(XT)), marker='.', color='k')
    ax.bar(1, np.mean(IT), color='white', edgecolor='black',linewidth=4)
    ax.errorbar(1, np.mean(IT), yerr=np.std(IT)/np.sqrt(len(IT)), marker='.', color='k')

    
    if animal == 'grom':
        if not normalize:
            ax.plot([0, 1], [1.7, 1.7], 'k-', linewidth=4.)
            ax.text(0.5, 1.9, '***', fontsize=12, horizontalalignment='center')

        else:
            ax.plot([0, 1], [2.1, 2.1], 'k-', linewidth=4.)
            ax.text(0.5, 2.2, '***', fontsize=12, horizontalalignment='center')
    
    elif animal == 'jeev':
        if normalize:
            ax.plot([0, 1], [2.1, 2.1], 'k-', linewidth=4.)
            ax.text(0.5, 2.2, '***', fontsize=12, horizontalalignment='center')

        else:
            ax.plot([0, 1], [2.0, 2.0], 'k-', linewidth=4.)
            ax.text(0.5, 2.0, '***', fontsize=12, horizontalalignment='center')

    u, p = scipy.stats.kruskal(XT, IT)
    print 'kruskal wallis test: ', u, p, 'n = xtask: ', len(XT), ', w/in task: ', len(IT)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Across \nTask', 'Within \nTask'], fontsize=14)
    if normalize:
        ax.set_ylabel('Norm. Mean Firing Rate Difference', fontsize=14)
    else:
        ax.set_ylabel('Mean Firing Rate Difference', fontsize=14)

    if normalize:
        ax.set_yticks(np.arange(0., 2.4, 0.2))
        x = np.arange(0., 2.4, 0.2)
    else:
        ax.set_yticks(np.arange(0., 2.2, 0.2))
        x = np.arange(0., 2.2, 0.2)

    x = [.1*int(np.round(10*i)) for i in x]
    ax.set_yticklabels(x, fontsize=14)
    return all_bars

def win_near_vs_far(animal, norm_params, min_samples=15, all_bars=None, avg_across_neur = True, 
    only_important_neurons = False):

    ##########################################################
    ###  Compare Close in Time vs. Far in Time Comparisons WINTASK ###
    ##########################################################

    _near = {}
    _far = {}

    if animal == 'grom':
        days = 9
        fname_pref = analysis_config.config['grom_pref'] + 'hist_tuning_w_fit_mag_boundaries_grom_16_trials_dist_day'
        close_compare_win = fk.grom_win_close
        far_compare_win = fk.grom_win_far

    if animal == 'jeev':
        days = 4
        fname_pref=analysis_config.config['grom_pref'] + 'hist_tuning_w_fit_mag_boundaries_jeev_16_trials_dist_day'
        close_compare_win = []
        far_compare_win = []

        for i in range(4):
            for it in range(2):
                for ten in range(1):
                    close_compare_win.append([[13],[13]])
                    far_compare_win.append([[13],[13]])

    # For each day: 
    for select_day in range(days):

        # Load data: 
        fname = fname_pref + str(select_day)
        dat_dist = pickle.load(open(fname+'dist'+'.pkl', 'rb'))
        
        # Init variables: 
        other = dict(co=[], obs=[])
        first_third = 0
        third_third = 0
        
        # Iterate through tasks: 
        for it, (tsk, tsknm) in enumerate(zip(range(2), ['co', 'obs'])):

            # For near index and far index for the task: 
            for n, (ni, fi) in enumerate(zip(close_compare_win[select_day][tsk], far_compare_win[select_day][tsk])):
                
                key = tuple(('day'+str(select_day), 'tsk'+str(tsk), 'n'+str(n)))
                
                # Get the "near index"
                # One of these gusy is bins x bins x neurons x task x distribution
                import pdb; pdb.set_trace()
                if ni == 1:
                    first_third = dat_dist[key][:, :, :, :, 0]
                
                elif ni == 3:
                    third_third = dat_dist[key][:, :, :, :, 3]
                
                elif ni == 13:
                    first_third = dat_dist[key][:, :, :, :, 1]
                    third_third = dat_dist[key][:, :, :, :, 2]
        
                # Get the "far index"
                if fi == 1:
                    far_first_third = dat_dist[key][:, :, :, :, 0]
                
                elif fi == 3:
                    far_third_third = dat_dist[key][:, :, :, :, 3]
                
                elif fi == 13:
                    far_first_third = dat_dist[key][:, :, :, :, 0]
                    far_third_third = dat_dist[key][:, :, :, :, 3]

            ##################        
            ### KL DIV fcn ###
            ##################
            # bins x bins x neurons
            _near[select_day, it] = dlambda(first_third, third_third, min_samples, 
                norm_params=norm_params, day=select_day)

            _far[select_day, it] = dlambda(far_first_third, far_third_third, min_samples, 
                norm_params=norm_params, day=select_day)
        
    ### Aggregate! ###
    X = []; Y = []; 
    if all_bars is None:
        all_bars = {}
    f, ax = plt.subplots()

    # For each day
    for select_day in range(days):
        xx = []
        yy = []
        x_day = []; 
        y_day = []; 

        # For each task: 
        for it in range(2):

            # Bins x bins x neurons: 
            bins_ang, bins_mag, nneurons = _near[select_day, it].shape

            if only_important_neurons:
                neuron_ix = get_important_units(animal, select_day)
                nneurons = len(neuron_ix)
            else:
                neuron_ix = np.arange(nneurons)

            if avg_across_neur:
                # AVERAGE across neuron: bins x bins
                x = np.nanmean(_near[select_day,it][:, :, neuron_ix].reshape(bins_ang*bins_mag, nneurons), axis=1)
                y = np.nanmean( _far[select_day,it][:, :, neuron_ix].reshape(bins_ang*bins_mag, nneurons), axis=1)
            else:
                x = _near[select_day,it][:, :, neuron_ix].reshape(-1)
                y =  _far[select_day,it][:, :, neuron_ix].reshape(-1)              
            
            ix0 = np.unique(np.hstack((np.nonzero(np.isnan(x))[0], np.nonzero(np.isnan(y))[0])))
            ix0 = np.array([i for i in range(len(x)) if i not in ix0])
        
            if len(ix0) > 0:
                X.append(x[ix0])
                Y.append(y[ix0])
                
                x_day.append(x[ix0]); 
                y_day.append(y[ix0]); 

                ax.plot(.05*np.random.randn(len(x))+select_day, x, 'g.')#, CLOSE // positions = [select_day])
                ax.plot(.05*np.random.randn(len(x))+select_day+.3, y, 'r.')# FAR
            
                xx.append(np.hstack((x[ix0])))
                yy.append(np.hstack((y[ix0])))

        all_bars['win_far_close', select_day] = np.abs(np.hstack((xx)) - np.hstack((yy)))
        all_bars['win_close', select_day] = np.hstack((xx))
        all_bars['win_far', select_day] = np.hstack((yy))

        x_day = np.hstack((x_day))
        y_day = np.hstack((y_day))

        ax.bar(select_day, np.nanmean(x_day), 0.3, color='gray', alpha = .5)
        ax.bar(select_day+.3, np.nanmean(y_day), 0.3, color='gray', alpha = .5)

    X = np.hstack((X))
    Y = np.hstack((Y))

    print('Shape of X, Y: '+str(X.shape))

    all_bars['win_far_close', 'near'] = X
    all_bars['win_far_close', 'far'] = Y

    return all_bars

def xtask_near_vs_far(animal, norm_params, min_samples=15, all_bars=None, avg_across_neur = True, 
    only_important_neurons = False):
    ##########################################################
    ###  Compare Close in Time vs. Far in Time Comparisons WINTASK ###
    ##########################################################

    _near = {}
    _far = {}

    if animal == 'grom':
        days = 9
        fname_pref = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/hist_tuning_w_fit_mag_boundaries_grom_16_trials_dist_day'
        
        # Across task: 
        x_close_compare = fk.grom_close
        x_far_compare = fk.grom_far

    if animal == 'jeev':
        days = 4
        fname_pref='/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/hist_tuning_w_fit_mag_boundaries_jeev_16_trials_dist_day'
        
        # Across task: 
        x_close_compare = fk.jeev_close
        x_far_compare = fk.jeev_far

    # For each day: 
    for select_day in range(days):

        # Load data: 
        fname = fname_pref + str(select_day)
        dat_dist = sio.loadmat(fname+'dist'+'.mat')
        
        # Init variables: 
        other = dict(co=[], obs=[])
        first_third = 0
        third_third = 0
        
        # Iterate through tasks: 
        for it, (tsk, tsknm) in enumerate(zip(range(2), ['co', 'obs'])):

            # For near index and far index for the task: 
            for n, (ni, fi) in enumerate(zip(x_close_compare[select_day][tsk], x_far_compare[select_day][tsk])):
                
                if animal == 'grom':
                    key = str(('day'+str(select_day), 'tsk'+str(tsk), 'n'+str(n)))
                
                elif animal == 'jeev':
                    key = str(('day'+str(select_day), 'tsk'+str(tsk), 'n'+str(n)))
                
                # Get the "near index"

                # One of these gusy is bins x bins x neurons x task x distribution
                if ni == 1:
                    first_third = dat_dist[key][:, :, :, :, 0]
                    #print('1 3rd')
                elif ni == 3:
                    third_third = dat_dist[key][:, :, :, :, 3]
                    #print('3 3rd')
                # elif ni == 13:
                #     first_third = dat_dist[key][:, :, :, :, 1]
                #     third_third = dat_dist[key][:, :, :, :, 2]
        
                # Get the "far index"
                if fi == 1:
                    far_first_third = dat_dist[key][:, :, :, :, 0]
                    #print('far 1 3rd')
                elif fi == 3:
                    far_third_third = dat_dist[key][:, :, :, :, 3]
                    #print('far 3 3rd')
                # elif fi == 13:
                #     far_first_third = dat_dist[key][:, :, :, :, 0]
                #     far_third_third = dat_dist[key][:, :, :, :, 3]

        ##################        
        ### KL DIV fcn ###
        ##################
        # bins x bins x neurons
        _near[select_day, it] = dlambda(first_third, third_third, min_samples, 
            norm_params=norm_params, day=select_day)

        _far[select_day, it] = dlambda(far_first_third, far_third_third, min_samples, 
            norm_params=norm_params, day=select_day)
        
    ### Aggregate! ###
    X = []; Y = []; 
    
    if all_bars is None:
        all_bars = {}
    
    f, ax = plt.subplots()

    # For each day
    for select_day in range(days):
        xx = []
        yy = []

        # For each task: 
        for it in range(1, 2):

            # Bins x bins x neurons: 
            bins_ang, bins_mag, nneurons = _near[select_day, it].shape

            if only_important_neurons:
                neuron_ix = get_important_units(animal, select_day)
                nneurons = len(neuron_ix)
            else:
                neuron_ix = np.arange(nneurons)

            if avg_across_neur:
                # AVERAGE across neuron: bins x bins
                x = np.nanmean(_near[select_day,it][:, :, neuron_ix].reshape(bins_ang*bins_mag, nneurons), axis=1)
                y = np.nanmean( _far[select_day,it][:, :, neuron_ix].reshape(bins_ang*bins_mag, nneurons), axis=1)
            
            else:
                x = _near[select_day,it][:, :, neuron_ix].reshape(-1)
                y = _far[select_day, it][:, :, neuron_ix].reshape(-1)

            ix0 = np.unique(np.hstack((np.nonzero(np.isnan(x))[0], np.nonzero(np.isnan(y))[0])))
            ix0 = np.array([i for i in range(len(x)) if i not in ix0])
        
            X.append(x[ix0])
            Y.append(y[ix0])
            ax.plot(.05*np.random.randn(len(x))+select_day, x, 'g.')#, positions = [select_day])
            ax.plot(.05*np.random.randn(len(x))+select_day+.3, y, 'r.')#
            
            xx.append(np.hstack((x[ix0])))
            yy.append(np.hstack((y[ix0])))

        all_bars['x_far_close', select_day] = np.abs(np.hstack((xx)) - np.hstack((yy)))
        all_bars['x_close', select_day] = np.hstack((xx))
        all_bars['x_far', select_day] = np.hstack((yy))

        ax.bar(select_day, np.mean(np.hstack((xx))), .3, color='green', alpha=.3)
        ax.bar(select_day+.3, np.mean(np.hstack((yy))), .3, color='red', alpha=.3)

    X = np.hstack((X))
    Y = np.hstack((Y))

    all_bars['x_far_close', 'near'] = X
    all_bars['x_far_close', 'far'] = Y

    return all_bars

def all_bars_to_df(all_bars, animal):
    from numpy.random import normal
    import pandas as pd
    from collections import namedtuple

    P = ["win_far_close","x_far_close", 'xtask_vs_win']
    #P = ["win_far", "win_close", "xtask_far", "xtask_close", "xtask", "win","win_far_close","xtask_far_close", 'xtask_vs_win']

    Sub = namedtuple('Sub', ['day', 'condition', 'dat'])
    dat_ = dict(day=[], cond=[], dat=[])

    if animal == 'jeev':
        days = 4; 
    elif animal == 'grom':
        days = 9

    for subid in xrange(0, days):
        for i, condition in enumerate(P):

            N = len(all_bars[condition, subid])
            dat_['day'].append(np.zeros((N, ))+subid)
            dat_['cond'].append([condition]*N)
            dat_['dat'].append(10*all_bars[condition, subid])
    
    for i in ['day', 'cond', 'dat']:
        dat_[i] = np.hstack((dat_[i]))


    ### Stats for the win-task vs.  x-task vs. xtask_vs_win
    print '############# Kruskal Wallis w/ MannWhitu + Bonferroni Correction ##########'
    import scipy.stats
    x0 = np.nonzero(dat_['cond']=='xtask_vs_win')[0]
    x1 = np.nonzero(dat_['cond']=='xtask_far_close')[0]
    x2 = np.nonzero(dat_['cond']=='win_far_close')[0]
    print scipy.stats.kruskal(dat_['dat'][x0], dat_['dat'][x1], dat_['dat'][x2], )
    print 'bar 1 vs. bar xtaskfarclose' 
    print scipy.stats.mannwhitneyu(dat_['dat'][x0], dat_['dat'][x1]).pvalue * 2
    print 'bar 1 vs. bar win_far_close'
    print scipy.stats.mannwhitneyu(dat_['dat'][x0], dat_['dat'][x2]).pvalue * 2
    print '############# Kruskal Wallis w/ MannWhitu + Bonferroni Correction ##########'


    return dat_

###########################
### Utils: ###############
###########################

def polar_plot(r, phi, data, ax=None, vmin=None, vmax=None, cmap=None):
    """
    Plots a 2D array in polar coordinates.

    :param r: array of length n containing r coordinates
    :param phi: array of length m containing phi coordinates
    :param data: array of shape (n, m) containing the data to be plotted
    """
    
    # Generate the mesh
    phi_grid, r_grid = np.meshgrid(phi, r)
    x, y = r_grid*np.cos(phi_grid), r_grid*np.sin(phi_grid)
    if ax is None:
        im = plt.pcolormesh(x, y, data, vmin=vmin, vmax=vmax, cmap=cmap)
    else:
        im = ax.pcolormesh(x, y, data, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(np.arange(-4, 5, 2))
    ax.set_xticklabels(np.arange(-4, 5, 2), fontsize=10)
    ax.set_yticks(np.arange(-4, 5, 2))
    ax.set_yticklabels(np.arange(-4, 5, 2), fontsize=10)
    return im

def dlambda(x0, x1, min_samp=5, norm_params=None, day=None):
    """
    get the difference between mean spike counts in each bin, 
    only if each bin has more than the min_samp number of min_samples

    """
    I, J, nn = x0.shape[0], x0.shape[1], x0.shape[2]
    DL = np.zeros((I, J, nn))
    for i in range(I):
        for j in range(J):
            for n in range(nn):
                x0_tmp = []
                x1_tmp = []
                for t in range(40):
                    x0_tmp.append([t]*int(x0[i, j, n, t]))
                    x1_tmp.append([t]*int(x1[i, j, n, t]))
                X0 = np.hstack((x0_tmp))
                X1 = np.hstack((x1_tmp))

                if norm_params is not None:
                    # For this neuron: 
                    mn, std = norm_params[day, n]
                    X0 = (X0 - mn) / std
                    X1 = (X1 - mn) / std
                
                if np.logical_and(len(X0) > min_samp, len(X1) > min_samp):
                    DL[i, j, n] = np.abs(np.mean(X0) - np.mean(X1))
                else:
                    DL[i, j, n] = np.nan
                    
    return DL