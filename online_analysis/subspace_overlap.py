from db import dbfunctions as dbfn
import prelim_analysis as pa
import sklearn.decomposition as skdecomp
import os, pickle
import numpy as np
import matplotlib.pyplot as plt
import tables
import scipy.io as sio
from resim_ppf import ppf_pa 
from resim_ppf import file_key
import co_obs_tuning_matrices
import pandas
import scipy.linalg
import math

def get_keep_units(te_list, epoch_size, animal, binsize):
    '''
    Summary: 
    Input param: te_list: [id0, id1, id2, ...]
    Input param: epoch_size: epoch size for analysis (returns units that are zero for entire epoch)
    Input param: animal: 'grom' or 'jeev'
    Input param: binsize: binsize in MS
    Output param: dictionary of indices of units to KEEP (non-zero in all epochs)
    '''
    rm_ix = []
    for it, te in enumerate(te_list):
        if animal == 'grom':
            co_obs_dict = pickle.load(open(co_obs_tuning_matrices.pref+'co_obs_file_dict.pkl'))
            hdf = co_obs_dict[te, 'hdf']
            hdfix = hdf.rfind('/')
            hdf = tables.openFile(co_obs_tuning_matrices.pref+hdf[hdfix:])
            drives_neurons_ix0 = 3
            rew_ix_total = pa.get_trials_per_min(hdf)
            internal_state = hdf.root.task[:]['internal_decoder_state']
            update_bmi_ix = np.nonzero(np.diff(np.squeeze(internal_state[:, drives_neurons_ix0, 0])))[0]+1
            bin_spk, targ_pos, targ_ix, z, zz = pa.extract_trials_all(hdf, rew_ix_total, update_bmi_ix=update_bmi_ix, 
                keep_trials_sep=True, neural_bins=binsize)
        else:
            bin_spk, targ_i_all, targ_ix, trial_ix, curs_kin, unbinned, _ = ppf_pa.get_jeev_trials_from_task_data(te, binsize=binsize/1000.)
        
        ntrials = len(bin_spk)
        for i in range(np.max([1, ntrials/epoch_size])):
            try:
                bs = [bin_spk[j] for j in range(i*epoch_size, (i+1)*epoch_size)]
            except: 
                bs = bin_spk
                print 'epoch_size > len(bin_spk), so just using all bin_spk'
            
            b = np.vstack(bs)
            rm_ix.append(np.nonzero(np.sum(b, axis=0)==0)[0])
    keep_units = np.array(list(set(np.arange(b.shape[1])).difference(set(np.hstack((rm_ix))))))
    print 'len keep_units: ', te, len(keep_units), b.shape[1]
    return keep_units

def get_important_units(animal, day_index, est_type = 'est', thresh = .98):
    
    if animal == 'grom': 
        ### Load the thing: 
        if thresh == 0.8:
            imp_file = pickle.load(open('/Users/preeyakhanna/fa_analysis/online_analysis/grom_important_neurons_svd_feb2019_thresh_0.8.pkl', 'rb'))
        else:
            imp_file = pickle.load(open('/Users/preeyakhanna/fa_analysis/online_analysis/grom_important_neurons_svd_feb2019.pkl', 'rb'))
    
    elif animal == 'jeev':
        ### Load the thing: 
        if thresh == 0.8:
            imp_file = pickle.load(open('/Users/preeyakhanna/fa_analysis/online_analysis/jeev_important_neurons_svd_feb2019_thresh_0.8.pkl', 'rb'))
        else:
            imp_file = pickle.load(open('/Users/preeyakhanna/fa_analysis/online_analysis/jeev_important_neurons_svd_feb2019.pkl', 'rb')) 

    return np.sort(imp_file[day_index, animal, est_type])

def get_indices_of_matching_command_dist(te_list, animal, day_ix, te_task2, bin_tolerance = .01):
    ### want to subselect commands from both tasks so that distribution of commands is matching
    ### In order to do this, pull data from tasks, characterize into angs / magnitude bins
    ### Then count the minimum number in either task
    ### Then report the trial and index number of acceptable bins
    task1 = [te for te in te_list if te not in te_task2]
    task2 = [te for te in te_list if te in te_task2]

    ### Open the master mag: 
    mag_boundaries = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/radial_boundaries_fit_based_on_perc_feb_2019.pkl'))

    ### Build up list of where to find each command: 
    command_bins_task1 = []
    command_bins_task2 = []

    for te in task1: 
        commands = co_obs_tuning_matrices.get_spks(animal, te, keep_trls_sep = True)
        command_bins = commands2bins(commands, mag_boundaries, animal, day_ix)
        command_bins_task1.append(command_bins)

    for te in task2: 
        commands = co_obs_tuning_matrices.get_spks(animal, te, keep_trls_sep = True)

        ### [mag, ang]
        command_bins = commands2bins(commands, mag_boundaries, animal, day_ix)
        command_bins_task2.append(command_bins)

    ### Now get distribution of task #1: 
    commands1 = np.zeros((4, 8))
    for command_bins_task1_i in command_bins_task1:
        for trl in command_bins_task1_i:
            for ts in range(trl.shape[0]): 
                commands1[int(trl[ts, 0]), int(trl[ts, 1])] += 1

    commands2 = np.zeros((4, 8))
    commands2_dict = dict()
    for i in range(4): 
        for j in range(8):
            commands2_dict[i, j] = []
    task2_inclusion = []

    for ic, command_bins_task2_i in enumerate(command_bins_task2):
        task2_inclusion_i = []
        for i_t, trl in enumerate(command_bins_task2_i):
            for ts in range(trl.shape[0]): 
                m, a = trl[ts, :].astype(int)
                commands2[m, a] += 1

                ### Add this time to dict: 
                commands2_dict[m, a].append([ic, i_t, ts])

            ### Setup the task inclusion
            trl_inc = np.zeros((trl.shape[0]))
            task2_inclusion_i.append(trl_inc)
        task2_inclusion.append(task2_inclusion_i)

    commands1 = commands1.astype(float)
    command_distribution_task1 = commands1 / np.sum(commands1)

    ### Now compute how many commands you would need per bin in command 2: 
    command2_total = np.sum(commands2)
    commands2_putative_dist = np.floor(command_distribution_task1 * command2_total)

    command2_total_to_satisfy = command2_total
    
    ### Now go through and see how many commands you actually have in each command2 bin. 
    ### Adjust the total needed to keep distribution the same: 
    for m in range(4):
        for a in range(8):
            if commands2_putative_dist[m, a] > commands2[m, a]:

                ### Get current fraction and desired fraction to match: 
                if command_distribution_task1[m, a] > (commands2[m, a] / command2_total_to_satisfy) + bin_tolerance:
                    print('Ignoring discrepancy of %.2f' %(command_distribution_task1[m, a] - (commands2[m, a] / command2_total_to_satisfy)))
                
                elif commands2[m, a] == 0:
                    print('Ignoring bc command2[m, a] == 0')
                
                elif command_distribution_task1[m, a] <= bin_tolerance:
                    print('Ignoring bc lte bin bin_tolerance')

                else:
                    ### Compute total s.t. number of commands in commands2 can be satifsifed: 
                    ### ignore if command2 is zero:
                    tot_new = commands2[m, a] / (command_distribution_task1[m, a] - bin_tolerance)
                    if tot_new > command2_total_to_satisfy:
                        print('Ignoring new total is gte current total')
                    else:
                        commands2_putative_dist = np.floor(command_distribution_task1*tot_new)
                        print('Updating total from %d to %d' %(command2_total_to_satisfy, tot_new)) 
                        command2_total_to_satisfy = tot_new 

    ### Now that the total is set to keep the distribution about equal, mark which trial to keep: 
    command_distribution_task2 = np.floor(command_distribution_task1 * command2_total_to_satisfy).astype(int)

    ### This tells me how much of each command I get to add: 
    task2_added_so_far = np.zeros((4, 8))

    for m in range(4):
        for a in range(8): 
            for ix_i, ix in enumerate(range(command_distribution_task2[m, a])): 

                if ix_i < commands2[m, a]:
                    
                    ### Randomly select a command from commands2_dict
                    N = len(commands2_dict[m, a])

                    if commands2[m, a] == 0: 
                        print('skipping mag %d ang %d bc zero in task2, %d = prop number wanted from task 1' %(m, a, command_distribution_task2[m, a]))

                    else:
                        ### Randomly select
                        nix = np.random.permutation(N)[0]

                        ### get the key: 
                        key = commands2_dict[m, a][nix]
                        i, j, k = key

                        ### Update the task2_inclusion
                        task2_inclusion[i][j][k] = 1; 

                        ### Remove this guy from the commands2_dict
                        commands2_dict[m, a].remove(key)

    return task2_inclusion, task2

def commands2bins(commands, mag_boundaries, animal, day_ix, vel_ix = [3, 5], ndiv=8):
    mags = mag_boundaries[animal, day_ix]
    rads = np.linspace(0., 2*np.pi, ndiv+1) + np.pi/8
    command_bins = []
    for com in commands: 
        ### For each trial: 
        T = com.shape[0]
        vel = com[:, vel_ix]
        mag = np.linalg.norm(vel, axis=1)
        ang = []
        for t in range(T):
            ang.append(math.atan2(vel[t, 1], vel[t, 0]))
        ang = np.hstack((ang))

        ### Re-map from [-pi, pi] to [0, 2*pi]
        ang[ang < 0] += 2*np.pi

        ### Digitize: 
        ### will bin in b/w [m1, m2, m3] where index of 0 --> <m1, 1 --> [m1, m2]
        mag_bins = np.digitize(mag, mags)

        ###
        ang_bins = np.digitize(ang, rads)
        ang_bins[ang_bins == 8] = 0; 

        command_bins.append(np.hstack((mag_bins[:, np.newaxis], ang_bins[:, np.newaxis])))
    return command_bins

def targ_vs_all_subspace_align(te_list, file_name=None, cycle_FAs=None, epoch_size=50,
    compare_to_orig_te = False, include_mn=False, subset_of_units = None, 
    hdf_list_instead=None, te_mods=None, include_training_data=False, 
    ignore_zero_units = False, use_rest = False, main_shared=True, 
    only_important = False, day_index = None, subselect_commands = False,
    te_task2 = None):

    '''
    Summary: function that takes a task entry list, parses each task entry into epochs, and computes
        a factor analysis model for each epoch using (1:cycle_FAs = number of factors), and then either
        saves or returns the output and a modified task list

    Input param: te_list: list of task entries (not tiered)
    Input param: file_name: name to save overlap data to afterwards, if None, then returns args
    Input param: cycle_FAs: number of factors to cycle through (1:cycle_FAs) -- uses optimal number if None
    Input param: epoch_size: size of epochs to parse task entries into (uses ONLY reward trials)
    Input param: compare_to_orig_te: if True, then also compares all of the compute FAs to the original 
        FA model in the hdf file of the task entry listed in 'compare to original te'
    Input param: include_mn: whether to save mean that was subtracted off
    Input param: subset_of_units: whether to use a subset of units or not. If so, include dict
    Input param: hdf_list_instead: list of hdf files used 
    Input param: use_rest: flag False means that dont use the leftover trial for own epoch even if 
        there are more than epoch_size - 8 
    Input param: main_shared -- whether to compute main shared variance or just shared variance
    Input param: only_imporant -- only use important neurons
    Output param: 
    '''

    if hdf_list_instead is not None:
        te_list = range(len(te_mods))
    else:
        te_list = np.array(te_list)

    te_mod_list = []
    fa_dict = {}
    #assume no more than 100 FA epochs per task entry: these are used to name the different FA models (eg. 4048.1, 4048.11)
    mod_increment = 0.001
    increment_offs = 0.1

    #Usually only called for seeding LDS -- so all in te_list are in a single day -- dont be alarmed by 
    get_units_to_keep = True; 

    if ignore_zero_units:
        keep_units = get_keep_units(te_list, epoch_size, 'grom', 100)
        get_units_to_keep = False
    
    elif only_important: 
        keep_units = get_important_units('grom', day_index)
        get_units_to_keep = False

    if np.logical_and(ignore_zero_units, only_important):
        raise Exception('Cant filter for both zero units and important units')

    if subselect_commands:
        ### want to subselect commands from both tasks so that distribution of commands is matching
        ### In order to do this, pull data from tasks, characterize into angs / magnitude bins
        ### Then count the minimum number in either task
        ### Then report the trial and index number of acceptable bins
        acceptable_indices, te_subselect = get_indices_of_matching_command_dist(te_list, 'grom', day_index, te_task2)

    #For each TE get the right FA model: 
    for it, te in enumerate(te_list):
        if hdf_list_instead is not None:
            hdf = tables.openFile(hdf_list_instead[it])
        else:
            try:
                t = dbfn.TaskEntry(te)
                hdf = t.hdf
            
            except:
                co_obs_dict = pickle.load(open(co_obs_tuning_matrices.pref+'co_obs_file_dict.pkl'))
                hdf = co_obs_dict[te, 'hdf']
                hdfix = hdf.rfind('/')
                hdf = tables.openFile(co_obs_tuning_matrices.pref+hdf[hdfix:])

        if get_units_to_keep:
            keep_units = np.arange(hdf.root.task[0]['spike_counts'].shape[0])
            # Otherwise use the keep_units from above

        #Now get the right FA model: 
        drives_neurons_ix0 = 3
        rew_ix_total = pa.get_trials_per_min(hdf)
        print 'N REW: ', len(rew_ix_total)
        
        if len(rew_ix_total) > 0:
            proceed = True
        else:
            proceed = False

        if proceed:
            # Dont use the remainder of trials leftover 
            use_rest_i = False

            #Epoch rew_ix into epochs of 'epoch_size' 
            internal_state = hdf.root.task[:]['internal_decoder_state']
            update_bmi_ix = np.nonzero(np.diff(np.squeeze(internal_state[:, drives_neurons_ix0, 0])))[0]+1

            more_epochs = 1
            cnt = 0

            #decoder = t.decoder
            #null_units = np.array([i for i, u in enumerate(decoder.units) if np.logical_and(u[0] > 128, u[0]!= 233)])

            while more_epochs:

                #If not enough trials, still use TE (everything in the TE), or if not epoch size specified
                if (epoch_size is None) or (len(rew_ix_total) < epoch_size):

                    bin_spk, targ_pos, targ_ix, z, zz = pa.extract_trials_all(hdf, 
                        rew_ix_total, update_bmi_ix=update_bmi_ix)

                    ### Subselect spikes if needed
                    bin_spk = bin_spk[:, keep_units]

                    print('TE %d, Keeping units: %d of %d' %(te, len(keep_units), bin_spk.shape[1]))
                    more_epochs = 0
                    te_mod = float(te)
                    trl_ix = np.arange(len(rew_ix_total))
                    rew_ix_subselect = rew_ix_total; 

                else:
                    # Deal with other instances: 
                    if (cnt+1)*epoch_size - 8 <= len(rew_ix_total):
                        if use_rest_i:
                            # This uses the rest leftover to make a new epoch: 
                            rew_ix = rew_ix_total[cnt*epoch_size:]
                            trl_ix = np.arange(cnt*epoch_size, len(rew_ix_total))
                            assert len(rew_ix) >= (epoch_size - 8)
                            more_epochs = 0
                            print 'the nub epoch: ', (cnt*epoch_size), len(rew_ix_total)
                        
                        else:
                            # These are normal epochs: 
                            # Make sure rew_ix is the same as epoch_size for normal epochs: 
                            rew_ix = rew_ix_total[cnt*epoch_size:(cnt+1)*epoch_size]
                            trl_ix = np.arange(cnt*epoch_size, (cnt+1)*epoch_size)
                            assert len(rew_ix) == epoch_size

                        #Only use rew indices for that epoch:
                        bin_spk, targ_pos, targ_ix, z, zz = pa.extract_trials_all(hdf, rew_ix, time_cutoff=1000, 
                            update_bmi_ix=update_bmi_ix)
                        bin_spk = bin_spk[:, keep_units]
                        rew_ix_subselect = rew_ix; 

                        #Be done if next epoch won't have enough trials
                        cnt += 1

                        # If the last epoch is within 8 of correct number: 
                        if np.logical_and((cnt+1)*epoch_size > len(rew_ix_total), (cnt+1)*epoch_size -8 <= len(rew_ix_total)):
                            if not use_rest:
                                more_epochs = 0
                                use_rest_i = False
                                print 'skipping the rest of trials: ', cnt*epoch_size, '-', len(rew_ix_total)
                            
                            else:      
                                more_epochs = True                      
                                use_rest_i = True
                                print 'this will be the end_of_epochs: ', (cnt+1)*epoch_size - 8, len(rew_ix_total)
                                
                        # Turn it off: 
                        elif (cnt+1)*epoch_size - 8 > len(rew_ix_total):
                            more_epochs = 0
                        
                        else:
                            use_rest = False


                        if hdf_list_instead is not None:
                            te_mod = te_mods[it]                   
                        else:
                            te_mod = float(te) + (mod_increment*cnt) + increment_offs


                #Add modified te to dictionary
                te_mod_list.append(te_mod)

                ############################################
                ###### SUBSELECT BINNED SPIKE COUNTS #######
                ############################################
                if subselect_commands:
                    te_index = None
                    for tei, te_sub in enumerate(te_subselect):
                        if te_sub == te:
                            te_index = tei; 

                    if te_index is not None: 
                        sub_bin_spks_ix = []
                        subselect = acceptable_indices[te_index]

                        ### Which trials are in this epoch? 
                        sub_trl_ix = []
                        for ir, rew in enumerate(rew_ix_total):
                            if rew in rew_ix_subselect: 
                                sub_trl_ix.append(ir)

                        ### Now go through the trials and get the indices: 
                        for sub_trl in sub_trl_ix:
                            accept = subselect[sub_trl]
                            sub_bin_spks_ix.append(accept)

                        ### Now smoosh: 
                        sub_bin_spks_ix = np.hstack((sub_bin_spks_ix))
                        assert(len(sub_bin_spks_ix) == bin_spk.shape[0])

                        #### sub-select the spikes: 
                        print('Subselection: te %d, bins: %d, keeping: %d'% (te, bin_spk.shape[0], np.sum(sub_bin_spks_ix)))
                        bin_spk = bin_spk[sub_bin_spks_ix.astype(bool), :]

                #Use the bin_spk from above
                zscore_X, mu = pa.zscore_spks(bin_spk)

                if subset_of_units is not None:
                    print te, subset_of_units[te]
                    zscore_X = zscore_X[:, subset_of_units[te]]
            
                n_neurons = zscore_X.shape[1]


                #If no assigned number of factors to cycle through, find optimal
                if cycle_FAs is None:
                    
                    # Fit with maximum of 10 factors 
                    mx = np.min([n_neurons, 10])

                    # Fit it 5 times
                    log_lik, ax = pa.find_k_FA(zscore_X, iters = 5, max_k = mx, plot=False)
                    
                    # Take the mean of the log likelihood
                    mn_log_like = np.zeros((log_lik.shape[1], ))

                    #Np.nanmean -- 
                    for ii in range(log_lik.shape[1]):
                        cnt_ = 0
                        sum_ = 0
                        for jj in range(log_lik.shape[0]):
                            if np.isnan(log_lik[jj, ii]):
                                pass
                            else:                        
                                cnt_ += 1
                                sum_ += log_lik[jj, ii]
                        if cnt_ == 0.:
                            cnt_ = .1
                        mn_log_like[ii] = sum_/float(cnt_)

                    # Take the index with highest log likelihood:
                    ix = np.argmax(mn_log_like)

                    # Add 1 to go from index to # of factors
                    num_factors = ix + 1

                    print 'optimal number of factors: ', num_factors

                    # Fit this full model: 
                    FA_full = skdecomp.FactorAnalysis(n_components = num_factors)
                    FA_full.fit(zscore_X)

                    if include_training_data:
                        FA_full.training_data = zscore_X
                        FA_full.training_data_trl_ix = trl_ix

                    fa_dict[te_mod, 0] = FA_full
                    try:
                        fa_dict['units', te_mod] = t.decoder.units
                    except:
                        print 'didnt add units'
                    if include_mn:
                        fa_dict['mu', te_mod] = mu

                else:
                    for i in np.arange(1, cycle_FAs+1):
                        FA = skdecomp.FactorAnalysis(n_components = i)
                        FA.fit(zscore_X)
                        fa_dict[te_mod, i] = FA
                        if include_mn:
                            fa_dict[te_mod, i, 'mu'] = mu

    
            print 'END OF CYCLE: ', cnt
    
    #Now FA dict is completed:
    print 'now computing overlaps', te_mod_list
    
    if file_name is None:
        d, te_mod = ss_overlap(cycle_FAs, te_mod_list, fa_dict, file_name=file_name, 
            compare_to_orig_te=compare_to_orig_te, main_shared=main_shared)
        d['keep_units'] = keep_units
        return d, te_mod

    else:
        ss_overlap(cycle_FAs, te_mod_list, fa_dict, file_name=file_name, 
            compare_to_orig_te=compare_to_orig_te, main_shared=main_shared)

def targ_vs_all_subspace_align_jeev(te_list, te_nm_list, file_name=None, cycle_FAs=None, epoch_size=50, 
    include_mn=False, trial_type='all', tsk_ix=None, include_training_data=False, main_shared=True,
    only_important = False, use_rest = False, day_index = None, subselect_commands = False,
    te_task2 = None):

    '''
    Summary: function that takes a task entry list, parses each task entry into epochs, and computes
        a factor analysis model for each epoch using (1:cycle_FAs = number of factors), and then either
        saves or returns the output and a modified task list

    Input param: te_list: list of task entries (not tiered)
    Input param: file_name: name to save overlap data to afterwards, if None, then returns args
    Input param: cycle_FAs: number of factors to cycle through (1:cycle_FAs) -- uses optimal number if None
    Input param: epoch_size: size of epochs to parse task entries into (uses ONLY reward trials)
    Input param: trial_type: 'all', 'hard', 'easy'
    Input param: only_important --> only use important neurons as determined by SVD on KG; 
    Input param: use_rest --> use leftofver if epoch size of 16 not available
    Input param: day_index --> which day 
    Output param: 
    '''

    if use_rest == True: 
        raise Exception('Jeev fcn isnt programmed to allow you to use the remainder of a sessions trials')

    te_mod_list = []
    fa_dict = {}
    #assume no more than 10 FA epochs per task entry: these are used to name the different FA models (eg. 4048.1, 4048.11)
    mod_increment = 0.001
    increment_offs = 0.1


    if only_important: 
        get_units_to_keep = False
        keep_ix = get_important_units('jeev', day_index)

    else:
        get_units_to_keep = True


    if subselect_commands: 
        ### want to subselect commands from both tasks so that distribution of commands is matching
        ### In order to do this, pull data from tasks, characterize into angs / magnitude bins
        ### Then count the minimum number in either task
        ### Then report the trial and index number of acceptable bins
        acceptable_indices, te_subselect = get_indices_of_matching_command_dist(te_list, 'jeev', 
            day_index, te_task2)


    #For each TE get the right FA model: 
    for it, (te, te_nm) in enumerate(zip(te_list, te_nm_list)):
        TASK = tsk_ix[it]

        #Now get the right FA model: 
        bin_spk, targ_i_all, targ_ix, trial_ix, curs_kin, unbinned, _ = ppf_pa.get_jeev_trials_from_task_data(te,
            binsize=.1)

        if get_units_to_keep:
            keep_ix = np.arange(bin_spk[0].shape[1])

        if TASK == 1:
            obstrialList_types = file_key.obstrialList_types
            acceptable_ix = obstrialList_types[trial_type]
        else:
            acceptable_ix = np.arange(20)

        acceptable_trls = []

        #For each trial:
        for i, ii in enumerate(np.sort(np.unique(trial_ix))):
            iii = np.nonzero(trial_ix==ii)[0][1]
            if targ_ix[iii] in acceptable_ix:
                acceptable_trls.append(i)

        n_trials = len(acceptable_trls)
        rew_ix_total = np.array(acceptable_trls)

        more_epochs = True
        cnt = 0

        while more_epochs:

            #If not enough trials, still use TE, or if not epoch size specified
            if (epoch_size is None) or (n_trials < epoch_size):
                rew_ix = rew_ix_total.copy()
                more_epochs = 0
                trl_ix = np.arange(len(rew_ix))
                #te_mod = float(te)

            # Enough trials
            elif (cnt+1)*epoch_size <= n_trials:    
                    
                trl_ix = np.arange(cnt*epoch_size, (cnt+1)*epoch_size)
                rew_ix = rew_ix_total[trl_ix]
                assert len(rew_ix) == epoch_size

                cnt += 1

                if (cnt+1)*epoch_size > len(rew_ix_total):
                    #use_rest = True
                    print 'end_of_epochs: ', (cnt+1)*epoch_size, len(rew_ix_total)
                    more_epochs = 0

            te_mod = te_nm[-7:] + '_'+str((mod_increment*cnt) + increment_offs)

            #Add modified te to dictionary
            te_mod_list.append(te_mod)

            sub_bin_spk = []
            for r in rew_ix:
                # Only take the neurons we want: 
                sub_bin_spk.append(bin_spk[r][:, keep_ix])
            bin_spk_epoch = np.vstack((sub_bin_spk)); 
            
            ############################################
            ###### SUBSELECT BINNED SPIKE COUNTS #######
            ############################################
            if subselect_commands:
                te_index = None
                for tei, te_sub in enumerate(te_subselect):
                    if te_sub == te:
                        te_index = tei; 

                if te_index is not None: 
                    sub_bin_spks_ix = []
                    subselect = acceptable_indices[te_index]

                    ### Which trials are in this epoch? 
                    sub_trl_ix = []
                    for ir, rew in enumerate(rew_ix_total):
                        if rew in rew_ix: 
                            ### Here the actual trial number is encoded in rew_ix_total
                            ### Some trials are excluded, so need to
                            sub_trl_ix.append(rew)

                    ### Now go through the trials and get the indices: 
                    for sub_trl in sub_trl_ix:
                        accept = subselect[sub_trl]
                        sub_bin_spks_ix.append(accept)

                    ### Now smoosh: 
                    sub_bin_spks_ix = np.hstack((sub_bin_spks_ix))
                    assert(len(sub_bin_spks_ix) == bin_spk_epoch.shape[0])

                    #### sub-select the spikes: 
                    try:
                        print('Subselection: te %s, bins: %d, keeping: %d'% (te, bin_spk_epoch.shape[0], np.sum(sub_bin_spks_ix)))
                    except:
                        pass
                    bin_spk_epoch = bin_spk_epoch[sub_bin_spks_ix.astype(bool), :]

            #Use the bin_spk from above
            # proc_spks in time x neurons
            # zscore_X in time x neurons
            zscore_X, mu = pa.zscore_spks(bin_spk_epoch)
            n_neurons = zscore_X.shape[1]

            #If no assigned number of factors to cycle through, find optimal
            if cycle_FAs is None:

                # maximum of 10 factors: 
                mx = np.min([n_neurons, 10])
                log_lik, ax = pa.find_k_FA(zscore_X, iters = 5, max_k = mx, plot=False)
                mn_log_like = np.zeros((log_lik.shape[1], ))

                #Np.nanmean
                for ii in range(log_lik.shape[1]):
                    cnt_ = 0
                    sum_ = 0
                    for jj in range(log_lik.shape[0]):
                        if np.isnan(log_lik[jj, ii]):
                            pass
                        else:                        
                            cnt_ += 1
                            sum_ += log_lik[jj, ii]
                    if cnt_ == 0.:
                        cnt_ = .1
                    mn_log_like[ii] = sum_/float(cnt_)

                ix = np.argmax(mn_log_like)
                num_factors = ix + 1

                FA_full = skdecomp.FactorAnalysis(n_components = num_factors)
                FA_full.fit(zscore_X)
                if include_training_data:
                    FA_full.training_data = zscore_X
                    FA_full.training_data_trl_ix = trl_ix
                fa_dict[te_mod, 0] = FA_full

                if include_mn:
                    fa_dict['mu', te_mod] = mu

            else:
                for i in np.arange(1, cycle_FAs+1):
                    FA = skdecomp.FactorAnalysis(n_components = i)
                    FA.fit(zscore_X)
                    fa_dict[te_mod, i] = FA
                    if include_mn:
                        fa_dict[te_mod, i, 'mu'] = mu
    
            print 'END OF CYCLE: ', cnt

    #Now FA dict is completed:
    print 'now computing overlaps', te_mod_list
    
    if file_name is None:
        d, te_mod = ss_overlap(cycle_FAs, te_mod_list, fa_dict, file_name=file_name, 
            compare_to_orig_te=False, main_shared=main_shared)
        return d, te_mod

    else:
        ss_overlap(cycle_FAs, te_mod_list, fa_dict, file_name=file_name, 
            compare_to_orig_te=False, main_shared=main_shared)
    
def ss_overlap(cycle_FAs, te_mod_list, fa_dict, file_name=None, compare_to_orig_te = False, 
    repnum=None, main_shared=True):
    '''
    Summary: 
    Input param: cycle_FAs: number of factors to cycle throuhg
    Input param: te_mod_list: modified list (epoch parsing) from above
    Input param: fa_dict: ditionary with factor analysis estimates
    Input param: file_name: filename to save to (if 'None' then returned)
    Input param: compare_to_orig_te: True / False 

    Output param: 
    '''

    #Now go through each model and get overlap:
    if cycle_FAs is None:
        fact_ix = 2
    else:
        fact_ix = cycle_FAs

    overlab_mat = np.zeros(( len(te_mod_list), len(te_mod_list), fact_ix ))
    princ_ang_mat = np.zeros(( len(te_mod_list), len(te_mod_list)))

    for it, te in enumerate(te_mod_list):
        for it2, te2 in enumerate(te_mod_list[it:]):
            
            #Second index
            j = it+it2

            if cycle_FAs is None:
                #try:
                fab = fa_dict[te2, 0] #Obs
                fab0 = fa_dict[te, 0] #CO
                
                # except:
                #     fab = fa_dict[te2, 0, repnum]
                #     fab0 = fa_dict[te, 0, repnum]
                    
                overlab_mat[it, j, 0], p1 = get_overlap(fab, fab0, main=main_shared) # te --> te2
                overlab_mat[it, j, 1], p2 = get_overlap(fab0, fab, main=main_shared) # te2 --> te

                if np.allclose(p1, p2):
                    princ_ang_mat[it, j] = p1
                elif np.round(p1*100) == np.round(p2*100):
                    princ_ang_mat[it, j] = p1
                elif np.round(p1*10) == np.round(p2*10):
                    princ_ang_mat[it, j] = np.mean([p1, p2])
                else:
                    raise Exception('p1 != p2 in principle_angs')

            else:
                for i in np.arange(1, fact_ix+1):
                    fab = fa_dict[te2,i]

                    fab0 = fa_dict[te, i]

                    overlab_mat[it, j, i-1], _ = get_overlap(fab, fab0, main=main_shared)

    if compare_to_orig_te:
        comparison_mat = compare_to_orig_FA(te_mod_list, fa_dict)
        fa_dict['compare_overlab_mat'] = comparison_mat
    
    fa_dict['principle_angs'] = princ_ang_mat
    fa_dict['overlab_mat'] = overlab_mat
    fa_dict['te_mod'] = te_mod_list

    if file_name is None:
        return fa_dict, te_mod_list
    
    else:
        f = open(file_name, mode='w')
        pickle.dump(fa_dict, f)
        f.close()

def compare_to_orig_FA(te_mod_list, fa_dict):
    from online_analysis import single_te_sot
    comparison_mat = np.zeros(( len(te_mod_list), 1))

    for it, te in enumerate(te_mod_list):

        # Get original te:
        te_orig = int(te)
        if te_orig in fa_dict.keys():
            fab = fa_dict[te_orig]
            nf = fab.n_components
        else:
            task_entry_db = dbfn.TaskEntry(te_orig)
            fab = single_te_sot.FA_faux(task_entry_db.hdf.root.fa_params)
            fa_dict[te_orig] = fab
            nf = fab.n_components

        fab0 = fa_dict[te, nf]
        comparison_mat[it] = get_overlap(fab, fab0)
    return comparison_mat

def get_overlap(fab, fab0, first_mat = None, second_mat=None,
    first_UUT=None, second_UUT=None, main=True, main_thresh = 0.9): 

    # How much of Obs do you get in Co? 
    ''' Get projection of fab into fab0'''
    #Neurons by factors: 
    if first_mat is None and first_UUT is None:
        U_B = np.matrix(fab.components_.T)
        UUT_B = U_B*U_B.T

    elif first_mat is not None:
        U_B = first_mat
        UUT_B = U_B*U_B.T
    
    elif first_UUT is not None:
        UUT_B = first_UUT
    
    else:
        raise Exception

    v, s, vt = np.linalg.svd(UUT_B)
    s_cum = np.cumsum(s)/np.sum(s)

    # How much variance to we care about for precision reasons? 
    # Let's say: 0.01 %
    s_cum = np.round(s_cum*10e4)/float(10e4);
    red_s = np.zeros((v.shape[1], ))
    red_s_inf = np.zeros((v.shape[1], ))

    #Find shared space that occupies > 90% of var:
    if main:
        ix = np.nonzero(s_cum>main_thresh)[0]
        nf = ix[0] + 1
        #print 'main shared space num factors: ', nf
    else:
        ix = np.nonzero(s_cum==1.0)[0]
        nf = ix[-1] + 1
        #print 'not using main, num factors: ', nf

    red_s[:nf] = 1
    red_s_inf[:nf] = s[:nf]

    Pb = np.dot(np.dot(v, np.diag(red_s)), vt) #orthonormal cov (nfact_b x nfact_b)
    B_inf = np.dot(np.dot(v, np.diag(red_s_inf)), vt) # Covariance

    if second_mat is None and second_UUT is None:
        U_A = np.matrix(fab0.components_.T)
        UUT_A = U_A*U_A.T
    elif second_mat is not None:
        U_A = second_mat
        UUT_A = U_A*U_A.T
    elif second_UUT is not None:
        UUT_A = second_UUT
    else:
        raise Exception

    vv, ss, vvt = np.linalg.svd(UUT_A)
    ss_cum = np.cumsum(ss)/np.sum(ss)
    ss_cum = np.round(ss_cum*10e4)/float(10e4);

    if main:
        ix = np.nonzero(ss_cum>main_thresh)[0]
        nnf = ix[0] + 1
        #print 'main shared space num fact A: ', nnf
    else:
        ix = np.nonzero(s_cum==1.0)[0]
        nnf = ix[-1] + 1
        #print 'not using main, num factors: ', nnf

    red_s = np.zeros((vv.shape[1], ))
    red_s[:nnf] = ss[:nnf]
    A_shar = np.dot(np.dot(vv, np.diag(red_s)), vvt)
    A_inf = A_shar.copy()

    #### Get principle angles between reduced subspaces:
    princ_angs = scipy.linalg.subspace_angles(B_inf, A_inf)
    try:
        proj_A_B = Pb*A_shar*Pb.T
        return np.trace(proj_A_B)/float(np.trace(A_shar)), np.min(princ_angs)

    except:
        print moose
        print "cant compute overlap w/ different numbers of neurons"
        return 0.
    
def run_grom(run_data = False, plot_ov = True, plot_pa = True):

    if run_data:

        for subselect_to_match_command_similarity in [True]:#, False]:

            ### Importantneurons
            many_compare_grom(epoch_size=16, use_rest=False, main_shared=True, only_important=True, 
                subselect_commands = subselect_to_match_command_similarity)
            # many_compare_grom(epoch_size=16, use_rest=False, main_shared=False, only_important=True,
            #     subselect_commands = subselect_to_match_command_similarity)

            ## All neurons
            many_compare_grom(epoch_size=16, use_rest=False, main_shared=True, only_important=False,
                subselect_commands = subselect_to_match_command_similarity)
            # many_compare_grom(epoch_size=16, use_rest=False, main_shared=False, only_important=False,
            #     subselect_commands = subselect_to_match_command_similarity)

    pref = '/Users/preeyakhanna/fa_analysis/grom_data/grom2019_june_subspace_ov_CO_vs_Obs_same_day_many_factors_w_epochs16use_rest_False'
    
    fnames = []
    for main in [True]:#, False]:
        for important in [True, False]:
            for subselect in [True]:#, False]: 
                fnames.append('main'+str(main)+'only_important'+str(important)+'subselect'+str(subselect))

    if plot_ov:
        ### Now plotting / stats: 
        ### Make the plots: 
        for f in fnames: 
            print(f)
            X = plot_many(pref + f + '.pkl', co_obs_tuning_matrices.input_type, return_day_list = True)
            overlap_stats_anova(X, 'grom', save_fig = True, save_name='_co_obs_sub_overlap_'+f)
            print('---------------')
            print('---------------')
            print('---------------')

    if plot_pa: 
        ### Do the principle angle plots; 
        for f in fnames: 
            print(f)
            X = plot_many(pref + f + '.pkl', co_obs_tuning_matrices.input_type, return_day_list = True, 
                principle_angs = True)
            overlap_stats_anova(X, 'grom', principle_angs = True, save_fig = True, save_name='_co_obs_princ_ang_'+f)
            print('---------------')
            print('---------------')
            print('---------------')  

def run_jeev(run_data = True, plot_pa = False, plot_ov = False): 

    if run_data:

        for subselect_to_match_command_similarity in [True, False]:
            many_compare_jeev(epoch_size=16, use_rest=False, main_shared=True, only_important=True,
                trial_type_list = ['all'], subselect_commands = subselect_to_match_command_similarity)
            many_compare_jeev(epoch_size=16, use_rest=False, main_shared=False, only_important=True,
                trial_type_list = ['all'], subselect_commands = subselect_to_match_command_similarity)

            ## All neurons
            many_compare_jeev(epoch_size=16, use_rest=False, main_shared=True, only_important=False,
                trial_type_list = ['all'], subselect_commands = subselect_to_match_command_similarity)
            many_compare_jeev(epoch_size=16, use_rest=False, main_shared=False, only_important=False,
                trial_type_list = ['all'], subselect_commands = subselect_to_match_command_similarity)

    ### Now plotting / stats: 
    #pref = '/Users/preeyakhanna/fa_analysis/grom_data/jeev2019_june_subspace_ov_CO_vs_Obs_same_day_many_factors_w_epochs16alluse_rest_False'
    pref =  '/Users/preeyakhanna/fa_analysis/grom_data/jeev2019_june_subspace_ov_CO_vs_Obs_same_day_many_factors_w_epochs_16alluse_rest_False'
    fnames = []
    for main in [True, False]:
        for important in [True, False]:
            for subselect in [True, False]: 
                fnames.append('main'+str(main)+'only_important'+str(important)+'subselect'+str(subselect))

    if plot_ov:

        ### Make the plots: 
        for f in fnames: 
            print(f)
            X = plot_many(pref + f + '.pkl', file_key.task_input_type, return_day_list = True)
            overlap_stats_anova(X, 'jeev', save_fig = True, save_name='_co_obs_sub_overlap_'+f)
            print('---------------')
            print('---------------')
            print('---------------')

    if plot_pa:
        ### Make the plots: 
        for f in fnames: 
            print(f)
            X = plot_many(pref + f + '.pkl', file_key.task_input_type, return_day_list = True,
                principle_angs = True)
            overlap_stats_anova(X, 'jeev', principle_angs = True, save_fig = True, 
                save_name='_co_obs_princ_ang_'+f)
            print('---------------')
            print('---------------')
            print('---------------')        

def many_compare_grom(epoch_size=16, use_rest=False, main_shared=True, only_important=False,
    subselect_commands = False):
    # input_type = [[[4377], [4378, 4382]], [[4395], [4394]], [[4411], [4412, 4415]], [[4499], 
    # [4497, 4498, 4504]], [[4510], [4509, 4514]], [[4523, 4525], [4520, 4522]], [[4536], 
    # [4532, 4533]], [[4553], [4549]], [[4560], [4558]]]
    
    input_type = co_obs_tuning_matrices.input_type
    day_dict = {}
    for i_d, day in enumerate(input_type):
        print i_d, day
        te_list = list(np.hstack((day)))

        #Will cycle through FAs and choose best one! yay
        d, te_mod = targ_vs_all_subspace_align(te_list, cycle_FAs=None,  epoch_size = epoch_size, 
            include_mn=True, include_training_data=True, use_rest=use_rest, 
            main_shared=main_shared, only_important = only_important, day_index = i_d,
            subselect_commands = subselect_commands, te_task2 = day[1])

        day_dict[i_d] = d
        day_dict[i_d, 'te'] = te_mod

    import pickle
    fname = os.path.expandvars('$FA_GROM_DATA/grom2019_june_subspace_ov_CO_vs_Obs_same_day_many_factors_w_epochs'+
        str(epoch_size)+'use_rest_'+str(use_rest)+'main'+str(main_shared)+
        'only_important'+str(only_important)+
        'subselect'+str(subselect_commands)+
        '.pkl')

    print fname
    pickle.dump(day_dict, open(fname, mode='w'))

def many_compare_jeev(epoch_size=16, use_rest=False, trial_type_list = ['easy', 'hard', 'all'], 
    main_shared=True, only_important = False, subselect_commands = False):

    from resim_ppf import file_key as fk
    reload(fk)
    input_type = fk.task_filelist
    input_names = fk.task_input_type
    directory = fk.task_directory

    for trial_type in trial_type_list:
        day_dict = {}
        for i_d, (day, nm) in enumerate(zip(input_type, input_names)):
            print i_d, day

            #List of task indices
            tsk_ix = []
            for i_t, tsk in enumerate(day):
                for i_f, fn in enumerate(tsk):
                    tsk_ix.append(i_t)


            te_list = list(np.hstack((day)))
            te_nm_list = list(np.hstack((nm)))

            #Will cycle through FAs and choose best one
            d, te_mod = targ_vs_all_subspace_align_jeev(te_list, te_nm_list, cycle_FAs=None,  epoch_size = epoch_size, 
                include_mn=True, trial_type=trial_type, tsk_ix=tsk_ix, include_training_data=True, main_shared=main_shared,
                only_important = only_important, use_rest = use_rest, day_index = i_d, subselect_commands=subselect_commands,
                te_task2 = te_list[1])

            day_dict[i_d] = d
            day_dict[i_d, 'te'] = te_mod

        import pickle
        fname = os.path.expandvars('$FA_GROM_DATA/jeev2019_june_subspace_ov_CO_vs_Obs_same_day_many_factors_w_epochs_'+
            str(epoch_size)+trial_type+'use_rest_'+str(use_rest)+'main'+str(main_shared)+'only_important'+str(only_important)+
            'subselect'+str(subselect_commands)+'.pkl')

        print fname
        pickle.dump(day_dict, open(fname, mode='w'))

def many_compare_w_subset():
    input_type = [[[4377], [4378, 4382]], [[4395], [4394]], [[4411], [4412, 4415]], [[4499], 
    [4497, 4498, 4504]], [[4510], [4509, 4514]], [[4523, 4525], [4520, 4522]], [[4536], 
    [4532, 4533]], [[4553], [4549]], [[4560], [4558]]]

    import pickle
    subset = pickle.load(open(os.path.expandvars('$FA_GROM_DATA/co_obs_max_important_units_snr.pkl')))

    day_dict = {}
    for i_d, day in enumerate(input_type):
        print i_d, day
        te_list = list(np.hstack((day)))

        #Will cycle through FAs and choose best one! yay

        d, te_mod = targ_vs_all_subspace_align(te_list, cycle_FAs=None,  epoch_size = 32, include_mn=True, subset_of_units=subset)

        day_dict[i_d] = d
        day_dict[i_d, 'te'] = te_mod

    fname = os.path.expandvars('$FA_GROM_DATA/grom2016_subspace_ov_CO_vs_Obs_same_day_many_factors_w_epochs32_v3_correct_te_trls.pkl')
    pickle.dump(day_dict, open(fname, mode='w'))

def get_SOT_obs_co():
    te_list = np.array([4098, 4102, 4104, 4114, 4119, 4126, 4132, 4138, 4139, 4150, 4157,
       4159, 4163, 4169, 4180, 4186, 4195, 4196, 4215, 4219, 4240, 4247,
       4257, 4267, 4291, 4301, 4308, 4310, 4321, 4367, 4368, 4369, 4370,
       4377, 4378, 4382, 4395, 4394, 4411, 4412, 4415, 4499, 4497, 4498,
       4504, 4510, 4509, 4514, 4523, 4525, 4520, 4522, 4536, 4532, 4533,
       4553, 4549, 4560, 4558])

    day_dict = {}
    for it, te in enumerate(te_list):
        print it, te
        d, te_mod = targ_vs_all_subspace_align([te], cycle_FAs=None, epoch_size = 32)
        day_dict[te] = d
        day_dict[te, 'te'] = te_mod
    import pickle
    fname = os.path.expandvars('$FA_GROM_DATA/grom2016_extended_FA_for_CO_vs_Obs_epochs32trls.pkl')
    pickle.dump(day_dict, open(fname, mode='w'))    

def compare_obs_using_obs_shar():
    te_list_scaled = [4236, 4503, 4505, 4521, 4552]
    te_list_unsc = [4528, 4534, 4550]

    dic = {}
    for i, (fn, te_list) in enumerate(zip(['unscaled','scaled'],[te_list_unsc, te_list_scaled])):
        d, te_mod = targ_vs_all_subspace_align(te_list, cycle_FAs=10, epoch_size = 30, 
            compare_to_orig_te=True)
        dic[fn] = d
        dic[fn, 'te'] = te_mod

    import pickle
    fname = os.path.expandvars('$FA_GROM_DATA/grom2016_subspace_ov_Obs_epochs30trls_vs_original.pkl')
    pickle.dump(dic, open(fname, mode='w'))    

def single_te():
    for i, (t, ctot) in enumerate(zip([4503, 4487], [True, False])):
        fname = os.path.expandvars('$FA_GROM_DATA/grom2016_subspace_ov_'+str(t)+'_epochs30trls.pkl')
        targ_vs_all_subspace_align([t], cycle_FAs=10, epoch_size=30, file_name=fname, compare_to_orig_te=ctot)

def main():
    #Shared vs. all: Does subspace change w/ shared control? 
    hdf_name = os.path.expandvars('$FA_GROM_DATA/grom2016_days_of_shar_vs_all_overlap.pkl')
    te_list = [4016, 4020, 4021, 4026, 4027, 4028, 4029, 4031, 4036, 4038, 4042, 4043, 4044, 4048, 4049]
    targ_vs_all_subspace_align(te_list, hdf_name)

    hdf_name = os.path.expandvars('$FA_GROM_DATA/grom2016_obs_learning_overlap.pkl')
    te_list = [4098, 4100, 4102, 4104, 4114, 4116, 4118, 4119]
    targ_vs_all_subspace_align(te_list, hdf_name)

    hdf_name = os.path.expandvars('$FA_GROM_DATA/grom2016_interleaved_100trials_CO_overlap.pkl')
    te_list = [4126, 4129, 4131, 4132, 4133, 4134, 4135, 4139, 4140, 4141, 4142, 4143, 4144, 4145, 
                4151, 4152, 4153, 4155, 4156, 4157, 4158, 4159, 
                4164, 4165, 4166, 4168, 4169, 4170, 4171]
    targ_vs_all_subspace_align(te_list, hdf_name)

    hdf_name = os.path.expandvars('$FA_GROM_DATA/grom2016_interleaved_100trials_obs_overlap.pkl')
    te_list = [4190, 4195, 4196, 4215, 4247, 4248, 4249, 4250, 4253, 4257, 4259, 4262, 4263, 4262]
    targ_vs_all_subspace_align(te_list, hdf_name)

def train_w_shar_vs_not_co_vs_obs():
    # list_fmt: [[CO, Obs_all], ]
    te_list_w_shar_CO_train = [[4257, 4263], [4291, 4294], [4291, 4297], [4377, 4382]]
    te_list_no_shar_CO_train = [[4247, 4248], [4267, 4268], [4301, 4302], [4310, 4311], [4377, 4378], [4395, 4394], [4411, 4412], [4411, 4413]]
    dic = {}
    for cond_name, cond in zip(['shar_training', 'no_shar_training'], [te_list_w_shar_CO_train, te_list_no_shar_CO_train]):
        for i_d, day in enumerate(cond):
            d, te_mod = targ_vs_all_subspace_align(day, cycle_FAs=10, epoch_size = 96, 
                compare_to_orig_te=False)
            dic[cond_name, i_d] = d
            dic[cond_name, i_d, 'te'] = te_mod

        import pickle
        fname = os.path.expandvars('$FA_GROM_DATA/grom2016_subspace_ov_Obs_and_CO_shartrain_vs_no_sharetrain_epochs96.pkl')
        pickle.dump(dic, open(fname, mode='w'))    

def plot_many(fname, three_tier_te_list, return_day_list=False, principle_angs=False):
    '''
    Summary: 
    Input param: fname: filename to load overlab mat from (string or actual file allowed)
    Input param: three_tier_te_list: 1) list 2) by days 3) by obs vs. co. data from same day
    Input param: return_day_list: return list of co-co, obs-obs. co-obs ordered by day (for later anova)
    Output param: 
        updated 7/2017 to be less stupid
        updated 7/2019 to also return principle angles
    '''

    if type(fname) == str:
        d = pickle.load(open(fname))
    else:
        d = fname

    #2 tiered list --> task entry w/in day
    te_by_day = [np.hstack((i)) for i in three_tier_te_list]
    
    #d_keys = d.keys()
    d_keys = range(len(te_by_day))

    X = {}
    for i in ['co_co', 'obs_obs', 'co_obs']:
        if return_day_list:
            for j in d_keys:
                X[i, j] = []
        else:
            X[i] = []

    xoffs = 0

    #Iterate through the keys: 
    for te_ep in d_keys:
        skip_day = False
        xoffs += 1

        #Dict for that te_list:
        te_list = d[te_ep, 'te']
        if len(te_list) == 0:
            skip_day = True

        d_mini = d[te_ep]

        if type(d_mini) is np.ndarray:
            d_mini = dict(overlab_mat=d_mini)

            if principle_angs: 
                raise Exception('No principle_angs stored here')

        #Find co vs. obs tasks for the te_list
        #Find correct index: 
        if not skip_day:
            IX = -1
            for i_d, day in enumerate(te_by_day):
                try:
                    if int(te_list[0]) in day:
                        IX = i_d
                except:
                    day_trim = 'jeev'+te_list[0][:7]
                    if day_trim in day:
                        IX = i_d
            if IX == -1:
                raise Exception('No list in three tiered arg that has %s' %te_list[0])

            # Put co vs. obs in correct indices:
            tmpd = {}
            co_ix = []
            obs_ix = []
            for i_t, t in enumerate(te_list):
                try:
                    tmp = int(t)
                except:
                    tmp = 'jeev'+t[:7]

                if tmp in three_tier_te_list[IX][0]:
                    tmpd[t] = 'co'
                    co_ix.append(i_t)

                elif tmp in three_tier_te_list[IX][1]:
                    tmpd[t] = 'obs'
                    obs_ix.append(i_t)
                else: 
                    print t, ' not in te_list: ', three_tier_te_list[IX]

            def ov_from_fa_dict(ix, ov_dict, ix2=None):
                ''' Look at variance from ix into ix2 space'''

                assert ov_dict['overlab_mat'].shape[2] == 2

                ov_arr = []; 
                pa_arr = []; 

                proceed = False
                
                if len(ix) >1:
                    proceed = True
                
                if ix2 is not None:
                    if np.logical_and(len(ix)>=1, len(ix2)>=1):
                        proceed = True

                if proceed:
                    for ii, i in enumerate(ix):
                        if ix2 is None:
                            xx = ix[(ii+1):]
                        else:
                            xx = ix2

                        for j in xx:
                            if j > i:
                                # Take all the overlab mat iterations
                                ov_arr.append(ov_dict['overlab_mat'][i, j, :])
                                pa_arr.append(np.rad2deg(ov_dict['principle_angs'][i, j]))
                            elif i > j:
                                ov_arr.append(ov_dict['overlab_mat'][j, i, :])
                                pa_arr.append(np.rad2deg(ov_dict['principle_angs'][j, i]))
                            else:
                                raise Exception
                else:
                    ov_arr = [np.nan]
                    pa_arr = [np.nan]
                return ov_arr, pa_arr
            
            #Find C.O overlap:
            co_ov_arr, co_pa = ov_from_fa_dict(co_ix, d_mini)

            #Find Obs overlap:
            obs_ov_arr, obs_pa = ov_from_fa_dict(obs_ix, d_mini)

            #Find C.O - Obs overlap:
            x_ov_arr, x_pa = ov_from_fa_dict(co_ix, d_mini, ix2=obs_ix)

            if principle_angs:
                X['co_co', te_ep].append(list(np.hstack((co_pa))))
                X['obs_obs', te_ep].append(list(np.hstack((obs_pa))))
                X['co_obs', te_ep].append(list(np.hstack((x_pa))))
            else:
                X['co_co', te_ep].append(list(np.hstack((co_ov_arr))))
                X['obs_obs', te_ep].append(list(np.hstack((obs_ov_arr))))
                X['co_obs', te_ep].append(list(np.hstack((x_ov_arr))))
    return X

def overlap_stats_anova(X, animal, lds=False, save_fig=False, save_name=None,
    principle_angs = False):

    ''' method to do 2-way anova (factors: day, within vs. across task) '''

    dct = dict(day=[], cond=[], ov = [])
    for ik, key in enumerate(X.keys()):
        if key[0] in ['co_co', 'obs_obs']:
            cond = 'win'
        elif key[0] in ['co_obs']:
            cond = 'x'
        else:
            raise Exception

        if len(X[key]) > 0:
            dat = np.hstack((X[key][0]))
            ix = np.nonzero(~np.isnan(dat))[0]
            n = len(dat[ix])

            if n > 0:
                conds = [cond]*n
                day = np.zeros((n, ))+key[1]

                dct['day'].append(day)
                dct['cond'].append(np.hstack((conds)))
                dct['ov'].append(dat[ix])
    
    for ky in dct.keys():
        dct[ky] = np.hstack((dct[ky]))

    df = pandas.DataFrame.from_dict(dct)
    # # Run ANOVA: 
    # from statsmodels.formula.api import ols
    # from statsmodels.stats.anova import anova_lm
    # formula = 'ov ~ day + cond'
    # model = ols(formula, df).fit()
    # aov_table = anova_lm(model, typ=2)
    # print aov_table
    
    ix = np.nonzero(dct['cond']=='win')[0]
    ix2 = np.nonzero(dct['cond']=='x')[0]
    import scipy.stats
    tv, pv_ttest = scipy.stats.ttest_ind(dct['ov'][ix], dct['ov'][ix2])

    print 'nwin: ', len(ix), 'nx: ', len(ix2)
    kv, pv_ktest = scipy.stats.kruskal(dct['ov'][ix], dct['ov'][ix2])
    
    # Plot: 
    f, ax = plt.subplots(figsize=(3,4))
    d = {}
    d[1] = []
    d[0] = []
    
    for i_n, nm in enumerate(['x', 'win']):
        for i_d in np.unique(df['day']):
            ix = np.nonzero(np.logical_and(df['day']==i_d, df['cond']==nm))[0]
            d[i_n].append(df['ov'][ix])

    for i_n in range(2):
        ax.bar(i_n, np.mean(np.hstack((d[i_n]))), width=1.,edgecolor='k', linewidth=4., color='w')
        ax.errorbar(i_n, np.mean(np.hstack((d[i_n]))), np.std(np.hstack((d[i_n])))/np.sqrt(len(np.hstack((d[i_n])))), marker='.', color='k')
        print 'error:', i_n, ' bar: ', np.std(np.hstack((d[i_n])))/np.sqrt(len(np.hstack((d[i_n]))))

    ax.set_xticks(range(2))
    ax.set_xticklabels(['Across-\nTask', 'Within-\nTask'], fontsize=14)
    ax.set_xlim([-.6, 1.6])

    if principle_angs:
        ax.set_ylabel('Angular Difference', fontsize=14)  
        ax.set_yticks(np.arange(0.0, 30, 10))
        x = np.arange(0., 30, 10)
        xlab = [str(np.round(i)) for i in x]
        ax.set_yticklabels(xlab, fontsize=14)
        ax.set_ylim([0.0, 30])
        ax.plot([0.0, 1.0], [25, 25], 'k-', linewidth=4.)
        ystar = 27     
    
    else:
        ax.set_ylabel('Subspace Overlap', fontsize=14)
        ax.set_yticks(np.arange(0.5, 1.1, 0.1))
        x = np.arange(0.5, 1.1, 0.1)
        xlab = [str(np.round(i*10)/10.) for i in x]
        ax.set_yticklabels(xlab, fontsize=14)
        ax.set_ylim([0.5, 1.0])
        ax.plot([0.0, 1.0], [.9, .9], 'k-', linewidth=4.)
        ystar = .925
    
    if pv_ttest < .001:
        ax.text(0.5, ystar, '***', fontsize=28, horizontalalignment='center')
    elif pv_ttest < .01:
        ax.text(0.5, ystar, '**', fontsize=28, horizontalalignment='center')
    elif pv_ttest < .05:
        ax.text(0.5, ystar, '*', fontsize=28, horizontalalignment='center')
    
    if lds:
        if animal == 'grom':
            ax.set_ylim([0.5, .9])
            plt.plot([0., 1.], [.81, .81], 'k-')
            plt.text(.5, .81, '**', fontdict=dict(fontsize=20))
        elif animal == 'jeev':
            ax.set_ylim([0.5, 1.0])
            plt.plot([0., 1.], [.91, .91], 'k-')
            plt.text(.5, .91, '***', fontdict=dict(fontsize=20))            
    
        if save_fig:
            f.tight_layout()
            f.savefig('/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/BMI_co_obs_paper/figures/data_figs/'+animal+'co_obs_overlap_LDS_is'+str(lds)+'.svg', transparent=True)
    if save_fig:
        f.tight_layout()
        f.savefig('/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/BMI_co_obs_paper/figures/data_figs/'+animal+save_name+'.svg', transparent=True)

def plot_many_sims(fname, three_tier_te_list, nreps):
    '''
    Summary: 
    Input param: fname: filename to load overlab mat from (string or actual file allowed)
    Input param: three_tier_te_list: 1) list 2) by days 3) by obs vs. co. data from same day
    Output param: 
    '''
    print 'deprecated -- see plot_many!'

    if type(fname) == str:
        d = pickle.load(open(fname))
    else:
        d = fname

    X = {}
    for i in ['co_co', 'obs_obs', 'co_obs']:
        X[i] = []

    #2 tiered list --> task entry w/in day
    te_by_day = [np.hstack((i)) for i in three_tier_te_list]
    
    #d_keys = d.keys()
    d_keys = range(len(te_by_day))
    xoffs = 0

    #n factors: 
    n_fact = np.arange(d[0,0]['overlab_mat'].shape[2])
    f, ax = plt.subplots(nrows = len(n_fact))
    if len(n_fact) == 1:
        ax = [ax]

    #Iterate through the keys: 
    for te_ep in d_keys:
        for r in range(nreps):

            #If key is an integer
            if type(te_ep) == int:
                xoffs += 1

                #Dict for that te_list:
                te_list = d[te_ep, 'te', r]
                d_mini = d[te_ep, r]

                #Find co vs. obs tasks for the te_list
                #Find correct index: 
                IX = -1

                for i_d, day in enumerate(te_by_day):
                    try:
                        if int(te_list[0]) in day:
                            IX = i_d
                    except:
                        day_trim = 'jeev'+te_list[0][:7]
                        if day_trim in day:
                            IX = i_d
                if IX == -1:
                    raise Exception('No list in three tiered arg that has %s' %te_list[0])

                # Put co vs. obs in correct indices:
                tmpd = {}
                co_ix = []
                obs_ix = []
                for i_t, t in enumerate(te_list):
                    try:
                        tmp = int(t)
                    except:
                        tmp = 'jeev'+t[:7]

                    if tmp in three_tier_te_list[IX][0]:
                        tmpd[t] = 'co'
                        co_ix.append(i_t)

                    elif tmp in three_tier_te_list[IX][1]:
                        tmpd[t] = 'obs'
                        obs_ix.append(i_t)
                    else: 

                        print t, ' not in te_list: ', three_tier_te_list[IX]

                def ov_from_fa_dict(ix, n_fact, ov_dict, ix2=None):
                    ''' Look at variance from ix into ix2 space'''

                    ov_arr = {}
                    for k in n_fact:
                        ov_arr[k] = []
                    proceed = False
                    if len(ix) >1:
                        proceed = True
                    if ix2 is not None:
                        if np.logical_and(len(ix)==1, len(ix2)==1):
                            proceed = True
                        if np.logical_and(len(ix)==1, len(ix2) > 1):
                            proceed = True
                    if proceed:
                        for ii, i in enumerate(ix):
                            if ix2 is None:
                                xx = ix[(ii+1):]
                            else:
                                xx = ix2

                            for j in xx:
                                if i > j:
                                    i2 = j
                                    j2 = i
                                else:
                                    i2 = i
                                    j2 = j

                                for k in n_fact:

                                    try:
                                        ov_arr[k].append(ov_dict['overlab_mat'][i2, j2, k])
                                    except:
                                        ov_arr[k] = ov_dict['overlab_mat'][i2, j2, k]
                    else:
                        for k in n_fact:
                            ov_arr[k] = np.zeros((1, ))
                    return ov_arr
                
                #Find C.O overlap:
                co_ov_arr = ov_from_fa_dict(co_ix, n_fact, d_mini)
                #Find Obs overlap:
                obs_ov_arr = ov_from_fa_dict(obs_ix, n_fact, d_mini)

                #Find C.O - Obs overlap:
                x_ov_arr = ov_from_fa_dict(co_ix, n_fact, d_mini, ix2=obs_ix)

                X['co_co'].append(co_ov_arr[0])
                X['obs_obs'].append(obs_ov_arr[0])
                X['co_obs'].append(x_ov_arr[0])
                            

                for i in n_fact:
                    axi = ax[i]

                    axi.bar(np.array([0])+xoffs, np.mean(co_ov_arr[i]), .3, color='k')
                    #if type(co_ov_arr) is list:
                    axi.errorbar(np.array([.15])+xoffs, np.mean(co_ov_arr[i]), yerr=np.std(co_ov_arr[i])/np.sqrt(len(co_ov_arr[i])), ecolor='k')
                    axi.plot((np.random.randn(len(co_ov_arr[i]))*.05)+.15+xoffs, co_ov_arr[i], 'c.')

                    axi.bar(np.array([.3])+xoffs, np.mean(x_ov_arr[i]), .3, color='r')
                    #if type(x_ov_arr) is list:
                    axi.errorbar(np.array([.45])+xoffs, np.mean(x_ov_arr[i]), yerr=np.std(x_ov_arr[i])/np.sqrt(len(x_ov_arr[i])), ecolor='r')
                    axi.plot((np.random.randn(len(x_ov_arr[i]))*.05)+.45+xoffs, x_ov_arr[i], 'c.')

                    axi.bar(np.array([.6]) + xoffs, np.mean(obs_ov_arr[i]), .3, color='b')
                    #if type(obs_ov_arr) is list:
                    axi.errorbar(np.array([.75])+xoffs, np.mean(obs_ov_arr[i]), yerr=np.std(obs_ov_arr[i])/np.sqrt(len(obs_ov_arr[i])), ecolor='b')
                    axi.plot((np.random.randn(len(obs_ov_arr[i]))*.05)+.75+xoffs, obs_ov_arr[i], 'c.')
    plt.tight_layout()
    return X

def plot_one(fname, three_tier_te_list, factor=6, names=None):
    ''' fcn to box plot overlap over days for a specific factor'''
    print 'deprecated -- see plot_many!'
    if type(fname) == str:
        d = pickle.load(open(fname))
    else:
        d = fname

    #2 tiered list --> task entry w/in day
    te_by_day = [np.hstack((i)) for i in three_tier_te_list]
    
    #d_keys = d.keys()
    d_keys = range(len(te_by_day))

    xoffs = 0

    #n factors: 
    n_fact = np.arange(d[0]['overlab_mat'].shape[2])
    f, ax = plt.subplots()

    #Iterate through the keys: 
    X = dict()
    for i in ['co_co', 'obs_obs', 'co_obs']:
        X[i] = []

    for te_ep in d_keys:

        #If key is an integer
        if type(te_ep) == int:
            xoffs += 1

            #Dict for that te_list:
            te_list = d[te_ep, 'te']
            d_mini = d[te_ep]

            #Find co vs. obs tasks for the te_list
            #Find correct index: 
            IX = -1

            for i_d, day in enumerate(te_by_day):
                if int(te_list[0]) in day:
                    IX = i_d
            if IX == -1:
                raise Exception('No list in three tiered arg that has %s' %te_list[0])

            # Put co vs. obs in correct indices:
            tmpd = {}
            co_ix = []
            obs_ix = []
            for i_t, t in enumerate(te_list):

                if int(t) in three_tier_te_list[IX][0]:
                    tmpd[t] = 'co'
                    co_ix.append(i_t)

                elif int(t) in three_tier_te_list[IX][1]:
                    tmpd[t] = 'obs'
                    obs_ix.append(i_t)
                else: 

                    print t, ' not in te_list: ', three_tier_te_list[IX]

            def ov_from_fa_dict(ix, n_fact, ov_dict, ix2=None):
                ov_arr = {}
                for k in n_fact:
                    ov_arr[k] = []
                if len(ix) > 0:
                    for ii, i in enumerate(ix):
                        if ix2 is None:
                            xx = ix[(ii)+1:]
                        else:
                            xx = ix2

                        for j in xx:
                            for k in n_fact:
                                try:
                                    ov_arr[k].append(ov_dict['overlab_mat'][i, j, k])
                                except:
                                    ov_arr[k] = ov_dict['overlab_mat'][i, j, k]
                else:
                    for k in n_fact:
                        ov_arr[k] = -1


                return ov_arr
            
            #Find C.O overlap:
            co_ov_arr = ov_from_fa_dict(co_ix, n_fact, d_mini)

            #Find Obs overlap:
            obs_ov_arr = ov_from_fa_dict(obs_ix, n_fact, d_mini)

            #Find C.O - Obs overlap:
            x_ov_arr = ov_from_fa_dict(co_ix, n_fact, d_mini, ix2=obs_ix)

            X['co_co'].append(co_ov_arr[0])
            X['obs_obs'].append(obs_ov_arr[0])
            X['co_obs'].append(x_ov_arr[0])
    

            i = factor-1
            axi = ax

            axi.bar(np.array([0])+xoffs, np.mean(co_ov_arr[i]), width=0.3, color='k')
            axi.bar(np.array([0.3])+xoffs, np.mean(x_ov_arr[i]), width=0.3, color='r')
            axi.bar(np.array([0.6])+xoffs, np.mean(obs_ov_arr[i]), width=0.3, color='b')

            axi.errorbar(np.array([0])+xoffs+.15, np.mean(co_ov_arr[i]), yerr=np.std(co_ov_arr[i])/np.sqrt(len(co_ov_arr[i])), linewidth=3, color='k')
            axi.errorbar(np.array([0.3])+xoffs+.15, np.mean(x_ov_arr[i]), yerr=np.std(x_ov_arr[i])/np.sqrt(len(x_ov_arr[i])), linewidth=3, color='r')
            axi.errorbar(np.array([0.6])+xoffs+.15, np.mean(obs_ov_arr[i]), yerr=np.std(obs_ov_arr[i])/np.sqrt(len(obs_ov_arr[i])), linewidth=3, color='b')
    return X
            
    #         axi.boxplot(np.array([0])+xoffs, np.mean(co_ov_arr[i]), .3, color='k')
    #             if type(co_ov_arr) is list:
    #                 axi.errorbar(np.array([0])+xoffs, np.mean(co_ov_arr[i]), yerr=np.std(co_ov_arr[i])/np.sqrt(len(co_ov_arr[i])), ecolor='k')
                
    #             axi.bar(np.array([.3])+xoffs, np.mean(x_ov_arr[i]), .3, color='r')
    #             if type(x_ov_arr) is list:
    #                 axi.errorbar(np.array([0])+xoffs, np.mean(x_ov_arr[i]), yerr=np.std(x_ov_arr[i])/np.sqrt(len(x_ov_arr[i])), ecolor='r')
                
    #             axi.bar(np.array([.6]) + xoffs, np.mean(obs_ov_arr[i]), .3, color='b')
    #             if type(obs_ov_arr) is list:
    #                 axi.errorbar(np.array([0])+xoffs, np.mean(obs_ov_arr[i]), yerr=np.std(obs_ov_arr[i])/np.sqrt(len(obs_ov_arr[i])), ecolor='b')
    # plt.tight_layout()
    # ax.set_xticks(np.arange(1,len(names)+1)+.5)
    # ax.set_xticklabels(names)
    # plt.setp(ax.get_xticklabels(), rotation=45)
    # ax.set_ylim([0, 1])
    # ax.set_ylabel('Subspace Overlap')

def plot_X_var(X):
    co_training_days = [0, 1, 2, 8]
    tasks = ['co_co', 'obs_obs']
    f, ax = plt.subplots(ncols=2)
    #f2, ax2 = plt.subplots()
    
    D = {}
    Dat = {}

    D['co_co'] = [[], []] #CO-CO, co_training obs_training
    D['obs_obs'] = [[], []] #OBS-OBS, co_training obs_training
    
    Dat['co_co'] = [[], []]
    Dat['obs_obs'] = [[], []]

    for j in tasks:
        for ix, xi in enumerate(X[j]):
            if ix in co_training_days:
                D[j][0].append(np.var(xi))
                Dat[j][0].append(np.mean(xi))
            else:
                D[j][1].append(np.var(xi))
                Dat[j][1].append(np.mean(xi))

    for it, tsk in enumerate(tasks):
        axi = ax[it]
        axi.set_title('Task: '+tasks[it])
        
        for trn in range(2):
            axi.boxplot(D[tsk][trn], positions=[trn])
            axi.set_xlim([-1, 2])
            axi.set_xticks([0, 1])
            axi.set_xticklabels(['CO Decoder', 'Obs Decoder'])
            axi.set_ylabel('Inter-Task Subspace Overlap Variability')
    f.tight_layout()

def plot_X_mn(X):
    co_training_days = [0, 1, 2, 8]
    tasks = ['co_co', 'obs_obs']
    #f2, ax2 = plt.subplots()
    
    D = {}
    Dat = {}

    D['co_co'] = [[], []] #CO-CO, co_training obs_training
    D['obs_obs'] = [[], []] #OBS-OBS, co_training obs_training
    
    Dat['co_co'] = [[], []]
    Dat['obs_obs'] = [[], []]

    for j in tasks:
        for ix, xi in enumerate(X[j]):
            if ix in co_training_days:
                D[j][0].append(np.var(xi))
                Dat[j][0].append(np.mean(xi))
            else:
                D[j][1].append(np.var(xi))
                Dat[j][1].append(np.mean(xi))

    f2, ax2 = plt.subplots()
    axi = ax2
    for trn in range(2):
        dtask = np.array(Dat['co_co'][trn]) - np.array(Dat['obs_obs'][trn])
        axi.boxplot(dtask, positions=[trn])
        axi.plot(np.zeros(len(dtask))+trn, dtask, '.')
        axi.set_xlim([-1, 2])
        axi.set_xticks([0, 1])
        axi.set_xticklabels(['CO Decoder', 'Obs Decoder'])

    #plot days 8, 9 special: 
    dtask = np.array(Dat['co_co'][0][-1]) - np.array(Dat['obs_obs'][0][-1])
    dtask2 = np.array(Dat['co_co'][1][-1]) - np.array(Dat['obs_obs'][1][-1])
    axi.plot(0, dtask, 'bd')
    axi.plot(1, dtask2, 'gd')
    axi.set_ylabel('Mean Within-Day, Within-Task Overlap:\n (CO-CO) - (OBS-OBS)')
    axi.plot([-1, 2], [0, 0], 'k--')
    axi.set_xlim([-1, 2])
    f2.tight_layout()
    plt.tight_layout()

def plot_X_mn_v2(X, save_fname=None):
    taskcol = dict()
    taskcol[0] = ['k', 'royalblue']
    taskcol[1] = ['royalblue', 'k']

    import scipy.stats

    co_training_days = [0, 1, 2, 8]
    tasks = ['co_co', 'obs_obs']
    #f2, ax2 = plt.subplots()
    
    D = {}
    Dat = {}

    D['co_co'] = [[], []] #CO-CO, co_training obs_training
    D['obs_obs'] = [[], []] #OBS-OBS, co_training obs_training
    
    Dat['co_co'] = [[], []]
    Dat['obs_obs'] = [[], []]

    for j in tasks:
        for ix, xi in enumerate(X[j]):
            if ix in co_training_days:
                D[j][0].append(np.var(xi))
                Dat[j][0].append(np.mean(xi))
            else:
                D[j][1].append(np.var(xi))
                Dat[j][1].append(np.mean(xi))

    f2, ax2 = plt.subplots()
    axi = ax2
    for trn in range(2):
        #dtask = np.array(Dat['co_co'][trn]) - np.array(Dat['obs_obs'][trn])
        c = Dat['co_co'][trn]
        o = Dat['obs_obs'][trn]


        axi.bar(trn+0, np.mean(c), width=0.4, color=taskcol[0][trn])
        axi.errorbar(trn+0.2, np.mean(c), np.std(c)/np.sqrt(len(c)), fmt='.', color='k')

        axi.bar(trn+0.4, np.mean(o), width=0.4, color=taskcol[1][trn])
        axi.errorbar(trn+0.6, np.mean(o), np.std(o)/np.sqrt(len(o)), fmt='.', color='k')
        print 'task : ', trn
        print scipy.stats.ranksums(c, o)

        # axi.boxplot(dtask, positions=[trn])
        # axi.plot(np.zeros(len(dtask))+trn, dtask, '.')
        axi.set_xlim([-0.2, 2.0])
        axi.set_xticks([0.1, 0.5, 1.1, 1.5])
        axi.set_xticklabels(['CO Dec -\n CO Task', 'Obs Dec -\n CO Task','CO Dec -\n Obs Task', 'Obs Dec -\n Obs Task',])
        plt.setp(axi.get_xticklabels(), rotation=45)
    #plot days 8, 9 special: 
    # dtask = np.array(Dat['co_co'][0][-1]) - np.array(Dat['obs_obs'][0][-1])
    # dtask2 = np.array(Dat['co_co'][1][-1]) - np.array(Dat['obs_obs'][1][-1])
    # axi.plot(0, dtask, 'bd')
    # axi.plot(1, dtask2, 'gd')
    # axi.set_ylabel('Mean Within-Day, Within-Task Overlap:\n (CO-CO) - (OBS-OBS)')
    # axi.plot([-1, 2], [0, 0], 'k--')
    # axi.set_xlim([-1, 2])

    axi.set_ylabel('Within Task Shared Alignment')
    axi.set_ylim([0.6, 0.9])
    f2.tight_layout()
    plt.tight_layout()
    if save_fname is not None:
        f2.savefig(save_fname, bbox_inches='tight', pad_inches=1)

def plot_jeev_subspace_ov_results():

    X = subspace_overlap.plot_many('jeev2016_subspace_ov_CO_vs_Obs_same_day_many_factors_w_epochs16_all.pkl', file_key.task_input_type)
    within = []
    within = np.hstack((np.hstack((X['co_co'])), np.hstack((X['obs_obs']))))
    x = np.hstack((X['co_obs']))
    x = x[x > 0]
    within = within[within > 0]
    f, ax = plt.subplots()
    f.set_figheight(5)
    f.set_figwidth(5)
    ax.bar(0, np.mean(within), width=.8, color='black')
    ax.errorbar(0.4, np.mean(within), np.std(within)/np.sqrt(len(within)), color='black')    
    ax.bar(1, np.mean(x), width=.8, color='black')
    ax.errorbar(1.4, np.mean(x), np.std(x)/np.sqrt(len(x)), color='black')

    X = subspace_overlap.plot_many_sims('jeev_2017_subspace_ov_CO_vs_Obs_same_day_many_factors_w_epochs16_SIMS_w_IK_opt_realistic_jeev_fixed_IK.pkl', file_key.number_names, 1)
    within = []
    within = np.hstack((np.hstack((X['co_co'])), np.hstack((X['obs_obs']))))
    x = np.hstack((X['co_obs']))
    x = x[x > 0]
    within = within[within > 0]

    ax.bar(2, np.mean(within), width=.8, color='gray')
    ax.errorbar(2.4, np.mean(within), np.std(within)/np.sqrt(len(within)), color='gray')    
    ax.bar(3, np.mean(x), width=.8, color='gray')
    ax.errorbar(3.4, np.mean(x), np.std(x)/np.sqrt(len(x)), color='gray')

    ax.set_ylabel('Shared Alignment')
    ax.set_xticks([.2, 1.2, 2.2, 3.2])
    ax.set_xticklabels(['Within Task', 'Across Task', 'Sim Within Task', 'Sim Across Task'])
    ax.set_ylim([.77, .99])
    plt.setp(ax.get_xticklabels(), rotation=45)

    ax.plot([.4, 1.4], [.86, .86], 'k-')
    ax.text(.8, .87, '**')
    ax.plot([2.4, 3.4], [.95, .95], 'k-')
    ax.text(2.7, .96, 'n.s.')
    plt.tight_layout()
    f.savefig('/home/lab/preeya/fa_analysis/cosyne_figs/jeev_subspace_ov_results_fixedIK.pdf', bbox_inches='tight', pad_inches=1)

def plot_jeev_subspace_ov_results_v2():
    Xsim = subspace_overlap.plot_many_sims('/storage/preeya/grom_data/jeev_2017_subspace_ov_CO_vs_Obs_same_day_many_factors_w_epochs64_SIMS_w_IK_opt_realistic_jeev.pkl', file_key.number_names, 20)
    X = subspace_overlap.plot_many('jeev2016_subspace_ov_CO_vs_Obs_same_day_many_factors_w_epochs_64all.pkl', file_key.task_input_type)

    nreps = 20
    X_all = {}
    for d in range(4):
        X_all[d,'sim'] = Xsim['co_obs'][d*nreps:(d+1)*nreps]
        X_all[d,'real'] = X['co_obs'][d]

    f, ax = plt.subplots()
    for d in range(4):
        ax.bar(d, np.mean(np.hstack((X_all[d, 'real']))), width=.5)
        ax.bar(d+.5, np.mean(np.hstack((X_all[d, 'sim']))), width=.5, color='r')

def plot_grom_subspace_ov_results():

    X = subspace_overlap.plot_many('grom2016_subspace_ov_CO_vs_Obs_same_day_many_factors_w_epochs16trls.pkl', co_obs_tuning_matrices.input_type)
    within = []
    within = np.hstack((np.hstack((X['co_co'])), np.hstack((X['obs_obs']))))
    x = np.hstack((X['co_obs']))
    x = x[x > 0]
    within = within[within > 0]
    f, ax = plt.subplots()
    f.set_figheight(5)
    f.set_figwidth(5)
    ax.bar(0, np.mean(within), width=.8, color='black')
    ax.errorbar(0.4, np.mean(within), np.std(within)/np.sqrt(len(within)), color='black')    
    ax.bar(1, np.mean(x), width=.8, color='black')
    ax.errorbar(1.4, np.mean(x), np.std(x)/np.sqrt(len(x)), color='black')

    X = subspace_overlap.plot_many('grom_2017_subspace_ov_CO_vs_Obs_same_day_many_factors_w_epochs16_SIMS_w_IK_opt_realistic_jeev.pkl', co_obs_tuning_matrices.input_type)
    within = []
    within = np.hstack((np.hstack((X['co_co'])), np.hstack((X['obs_obs']))))
    x = np.hstack((X['co_obs']))
    x = x[x > 0]
    within = within[within > 0]

    ax.bar(2, np.mean(within), width=.8, color='gray')
    ax.errorbar(2.4, np.mean(within), np.std(within)/np.sqrt(len(within)), color='gray')    
    ax.bar(3, np.mean(x), width=.8, color='gray')
    ax.errorbar(3.4, np.mean(x), np.std(x)/np.sqrt(len(x)), color='gray')

    ax.set_ylabel('Shared Alignment')
    ax.set_xticks([.2, 1.2, 2.2, 3.2])
    ax.set_xticklabels(['Within Task', 'Across Task', 'Sim Within Task', 'Sim Across Task'])
    ax.set_ylim([.77, .89])
    plt.setp(ax.get_xticklabels(), rotation=45)

    ax.plot([.4, 1.4], [.82, .82], 'k-')
    ax.text(.8, .83, '**')
    ax.plot([2.4, 3.4], [.87, .87], 'k-')
    ax.text(2.7, .88, 'p=0.1')
    plt.tight_layout()
    f.savefig('/home/lab/preeya/fa_analysis/cosyne_figs/grom_subspace_ov_results.pdf', bbox_inches='tight', pad_inches=1)

def plot_grom_subspace_ov_results_v2():

    Xr = subspace_overlap.plot_many('grom2016_subspace_ov_CO_vs_Obs_same_day_many_factors_w_epochs16trls.pkl', co_obs_tuning_matrices.input_type)
    withinr = []
    withinr = np.hstack((np.hstack((Xr['co_co'])), np.hstack((Xr['obs_obs']))))
    xr = np.hstack((Xr['co_obs']))
    xr = xr[xr > 0]
    withinr = withinr[withinr > 0]
    f, ax2 = plt.subplots()
    f.set_figheight(5)
    f.set_figwidth(5)
    ax2.bar(0, np.mean(withinr), width=.8, color='black')
    ax2.errorbar(0.4, np.mean(withinr), np.std(withinr)/np.sqrt(len(withinr)), color='black')    
    ax2.bar(1, np.mean(xr), width=.8, color='black')
    ax2.errorbar(1.4, np.mean(xr), np.std(xr)/np.sqrt(len(xr)), color='black')

    # filelist = ['grom_2017_subspace_ov_CO_vs_Obs_same_day_many_factors_w_epochs16_SIMS_w_IK_opt_realistic_jeev_fixed_IK_[[4377], [4378, 4382]].pkl', 
    #             'grom_2017_subspace_ov_CO_vs_Obs_same_day_many_factors_w_epochs16_SIMS_w_IK_opt_realistic_jeev_fixed_IK_[[4395], [4394]].pkl',
    #             'grom_2017_subspace_ov_CO_vs_Obs_same_day_many_factors_w_epochs16_SIMS_w_IK_opt_realistic_jeev_fixed_IK_[[4411], [4412, 4415]].pkl',
    #             'grom_2017_subspace_ov_CO_vs_Obs_same_day_many_factors_w_epochs16_SIMS_w_IK_opt_realistic_jeev_fixed_IK_[[4499], [4497, 4498, 4504]].pkl',
    #             'grom_2017_subspace_ov_CO_vs_Obs_same_day_many_factors_w_epochs16_SIMS_w_IK_opt_realistic_jeev_fixed_IK_[[4510], [4509, 4514]].pkl',
    #             'grom_2017_subspace_ov_CO_vs_Obs_same_day_many_factors_w_epochs16_SIMS_w_IK_opt_realistic_jeev_fixed_IK_[[4523, 4525], [4520, 4522]].pkl',
    #             'grom_2017_subspace_ov_CO_vs_Obs_same_day_many_factors_w_epochs16_SIMS_w_IK_opt_realistic_jeev_fixed_IK_[[4536], [4532, 4533]].pkl']

    # X = dict(co_co=[], obs_obs=[], co_obs=[])
    # for ii, i_f in enumerate(filelist):
    #     x = subspace_overlap.plot_many_sims('/home/lab/preeya/fa_analysis/grom_data/'+i_f, [co_obs_tuning_matrices.input_type[ii]], 1)
    #     for k in X.keys():
    #         X[k].append(x[k])

    within = []
    within = np.hstack((np.hstack((X['co_co'])), np.hstack((X['obs_obs']))))
    x = np.hstack((X['co_obs']))
    x = x[x > 0]
    within = within[within > 0]

    ax2.bar(2, np.mean(within), width=.8, color='gray')
    ax2.errorbar(2.4, np.mean(within), np.std(within)/np.sqrt(len(within)), color='gray')    
    ax2.bar(3, np.mean(x), width=.8, color='gray')
    ax2.errorbar(3.4, np.mean(x), np.std(x)/np.sqrt(len(x)), color='gray')

    ax2.set_ylabel('Shared Alignment')
    ax2.set_xticks([.2, 1.2, 2.2, 3.2])
    ax2.set_xticklabels(['Within Task', 'Across Task', 'Sim Within Task', 'Sim Across Task'])
    ax2.set_ylim([.77, .93])
    plt.setp(ax2.get_xticklabels(), rotation=45)

    ax2.plot([.4, 1.4], [.82, .82], 'k-')
    ax2.text(.8, .83, '**')
    ax2.plot([2.4, 3.4], [.92, .92], 'k-')
    ax.text(2.7, .88, 'n.s.')
    plt.tight_layout()
    f.savefig('/home/lab/preeya/fa_analysis/cosyne_figs/grom_subspace_ov_results_fixed_IK.pdf', bbox_inches='tight', pad_inches=1)

def plot_all_x(X):
    key_pos = dict(co_co=0, co_obs=.3, obs_obs=.6)
    col_pos = dict(co_co='k', co_obs='r', obs_obs='b')

    for k in X.keys():
        day = k[1]
        offs = key_pos[k[0]]
        if len(X[k]) > 0:
            data = np.hstack(( X[k] ))
            #data = data[data > 0.]
            plt.plot(day+offs+np.zeros_like(data), data, '.', color='k')
            plt.bar(day+offs-.15, np.mean(data), width=.3, color=col_pos[k[0]])
            
def get_sot(FA, by_neuron=False):
    U = np.mat(FA.components_).T
    v, s, vt = np.linalg.svd(U*U.T)
    s_cum = np.cumsum(s)/np.sum(s)
    red_s = np.zeros((v.shape[1], ))
    hd_s = np.zeros((v.shape[1], ))
    #Find shared space that occupies > 90% of var:
    ix = np.nonzero(s_cum>0.9)[0]
    nf = ix[0] + 1
    red_s[:nf] = s[:nf]
    hd_s[nf:] = s[nf:]
    P_shar = v*np.diag(red_s)*vt #
    P_hd = v*np.diag(hd_s)*vt
    Psi = FA.noise_variance_

    if by_neuron:
        SOT = []
        for n in range(len(Psi)):
            SOT.append(P_shar[n, n] / (P_shar[n, n] + P_hd[n, n] + Psi[n]))
    else:
        SOT = np.trace(P_shar) / (np.trace(P_shar)+np.trace(P_hd)+np.sum(Psi))
    return SOT

def get_SOT_in_control_space(FA, K):
    if K.shape[0] == 7:
        K = np.mat(K.T)
    else:
        K = np.mat(K)

    U = np.mat(FA.components_).T
    v, s, vt = np.linalg.svd(U*U.T)
    s_cum = np.cumsum(s)/np.sum(s)
    red_s = np.zeros((v.shape[1], ))
    hd_s = np.zeros((v.shape[1], ))

    #Find shared space that occupies > 90% of var:
    ix = np.nonzero(s_cum>0.9)[0]
    nf = ix[0] + 1
    red_s[:nf] = s[:nf]
    hd_s[nf:] = s[nf:]

    P_shar = v*np.diag(red_s)*vt #
    P_hd = v*np.diag(hd_s)*vt
    Psi = np.diag(FA.noise_variance_)

    #Project all matrices into control space: 
    K_proj = K*K.T
    
    PK_shar = K_proj*P_shar*K_proj.T
    PK_hd = K_proj*P_hd*K_proj.T
    PK_psi = K_proj*Psi*K_proj.T

    return np.trace(PK_shar)/(np.trace(PK_shar)+np.trace(PK_hd)+np.trace(PK_psi))

def get_optimal_n_factors(FA):
    U = np.mat(FA.components_).T
    v, s, vt = np.linalg.svd(U*U.T)
    s_cum = np.cumsum(s)/np.sum(s)
    red_s = np.zeros((v.shape[1], ))
    hd_s = np.zeros((v.shape[1], ))
    #Find shared space that occupies > 90% of var:
    ix = np.nonzero(s_cum>0.9)[0]
    nf = ix[0] + 1
    return nf

