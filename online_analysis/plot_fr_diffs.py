import analysis_config
from online_analysis import util_fcns, generate_models
from matplotlib import colors as mpl_colors

import scipy.stats 
import numpy as np
import matplotlib.pyplot as plt

import pickle

########## Utilities #########
def return_command_indices(bin_num, rev_bin_num, push, mag_boundaries, animal='grom', day_ix=0, 
                   min_bin_num=0, min_rev_bin_num=0, mag=0, ang=7):
    """Summary
    retuns all the indices that meet command criteria and trial bin criteria 
    Args:
        bin_num (np.arary): bin number of all concatenated trials 
        rev_bin_num (np.arary): bin number (counting from the back ) of all concatenated trials 
        push (np.arary): neural push 
        mag_boundaries (np.arary): dictionary of all the animal/day magnitude boundaries
        animal (str, optional): 
        day_ix (int, optional): 
        min_bin_num (int, optional): what is the minimum bin number acceptable (e.g. 5 if cut off first 5 bins)
        min_rev_bin_num (int, optional): what is the maximum bin number acceptable (e.g. 5 if cut off last 5 bins)
        mag (int, optional): selected magnitude
        ang (int, optional): selected angle 
    
    Returns:
        TYPE: Description
    """
    #### Select correct indices --> indices 0-4 inclusive are rejected 
    ix_valid = np.nonzero(np.logical_and(np.hstack((bin_num)) >= min_bin_num, rev_bin_num >= min_rev_bin_num))[0]
    
    command_bins = util_fcns.commands2bins([push], mag_boundaries, animal, day_ix, 
                                       vel_ix=[3, 5])[0]
    ix_command = np.nonzero(np.logical_and(command_bins[:, 0] == mag, command_bins[:, 1] == ang))[0]
    
    ix_command_valid = np.intersect1d(ix_valid, ix_command)

    assert(np.all(command_bins[ix_command_valid, 0] == mag))
    assert(np.all(command_bins[ix_command_valid, 1] == ang))
    assert(np.all(bin_num[ix_command_valid] >= min_bin_num))
    assert(np.all(rev_bin_num[ix_command_valid] >= min_rev_bin_num))

    return ix_command_valid

def distribution_match_global_mov(push_command_mov, push_command, psig=.05, 
                                  perc_drop = 0.05, plot=False):
    """
    Method to match the global command distrubtion to the 
    movement-specific command distribution 
    
    Args:
        push_command_mov (np.array): small dataset to match to --> pushes 
        push_command (np.array): large dataset to subsample from --> neural pushes
        psig (float, optional): do a t-test to say if command distributions are 
            significantly different. NOT sig different if p > psig; 
        perc_drop (float, optional): If there are still sig differences, drop this 
            percent of the bigger dataset ;
        plot (bool, optional): plot the distributions along the way; 
    
    Returns:
        TYPE: Description
    """
    complete = False
    niter = 0
    nVar = push_command_mov.shape[1]
    assert(push_command.shape[1] == nVar)
    
    #### Keep track of PV 
    PVs = []
    pv_recent = np.zeros((nVar, ))
    
    ### Indices to subsample (will remove these slowly using np.delete)
    indices = np.arange(push_command.shape[0])
    
    ### mean to match
    mean_match = np.mean(push_command_mov, axis=0)

    ### Will make cost function scale by variance of the distribution 
    ### we're trying to match. E.g. if varX = 10*varY in the distribution to match
    ### then want to make sure we penalize deviations in Y axis more heavily. 
    var_match = np.std(push_command_mov, axis=0)
    
    while not complete:

        for var in range(nVar):
            t, pv = scipy.stats.ttest_ind(push_command_mov[:, var], push_command[indices, var])
            pv_recent[var] = pv 
        
        if np.all(pv_recent > psig):
            complete = True
            return indices, niter
        
        else:
            niter += 1
            #print('Starting iter %d' %(niter))
            #print(pv_recent)
            
            ### Compute cost of each sample of push_command 
            mean_diff = push_command[indices, :] - mean_match[np.newaxis, :]
            
            ### So dimensions with high variance aren't weighted as strongly 
            cost = mean_diff / var_match[np.newaxis, :]
        
            ### Sum across dimensions 
            cost_sum = np.sum(cost**2, axis=1)
            
            ### How many pts to drop
            Npts = len(cost_sum)
            NptsDrop = int(np.floor(perc_drop * Npts))
            
            ### Sort from low to high 
            ix_sort = np.argsort(cost_sum)
            
            ### Remove the highest cost 
            ix_remove = ix_sort[-NptsDrop:]
            
            ### Update indices to remove cost
            indices = np.delete(indices, ix_remove)

            ### Make sure the cost when down 
            assert(np.sum(np.sum(cost[ix_sort[:-NptsDrop]]**2, axis=1)) < np.sum(cost_sum))
            
            if plot:
                f, ax = plt.subplots(ncols = nVar)
                for i in range(nVar):
                    ax[i].hist(push_command_mov[:, i], alpha=.3)
                    ax[i].vlines(np.mean(push_command_mov[:, i]), 0, 50, 'b')
                    ax[i].hist(push_command[indices, i], alpha=.3)
                    ax[i].vlines(np.mean(push_command[indices, i]), 0, 50, 'g')

def distribution_match_mov_pairwise(push_com1, push_com2, psig=.05, 
                                  perc_drop = 0.05):
    """
    Method to match the two command distrubtions
    
    Args:
        push_com1 (np.array): dataset to match 
        push_com2 (np.array): dataset to match 
        psig (float, optional): do a t-test to say if command distributions are 
            significantly different. NOT sig different if p > psig; 
        perc_drop (float, optional): If there are still sig differences, drop this 
            percent of the bigger dataset ;
        plot (bool, optional): plot the distributions along the way; 
    
    Returns:
        TYPE: Description
    """
    complete = False
    niter = 0
    nVar = push_com1.shape[1]
    assert(push_com2.shape[1] == nVar)
    
    #### Keep track of PV 
    PVs = []
    pv_recent = np.zeros((nVar, ))
    
    ### Indices to subsample (will remove these slowly using np.delete)
    indices1 = np.arange(push_com1.shape[0])
    indices2 = np.arange(push_com2.shape[0])
   

    ### Will make cost function scale by variance of the distribution 
    ### we're trying to match. E.g. if varX = 10*varY in the distribution to match
    ### then want to make sure we penalize deviations in Y axis more heavily. 
    dct = dict(push_com1=push_com1, push_com2=push_com2, indices1=indices1,
        indices2=indices2, mean1=np.mean(push_com1, axis=0), mean2=np.mean(push_com2, axis=0),
        std1=np.std(push_com1, axis=0), std2=np.std(push_com2, axis=0))

    while not complete:

        for var in range(nVar):
            t, pv = scipy.stats.ttest_ind(dct['push_com1'][dct['indices1'], var], dct['push_com2'][dct['indices2'], var])
            pv_recent[var] = pv 
        
        if np.all(pv_recent > psig):
            complete = True
            return dct['indices1'], dct['indices2'], niter
        
        else:
            niter += 1
            #print('Starting iter %d' %(niter))
            #print(pv_recent)
            
            ##### Match to the smaller distribution
            nsamp1 = len(dct['indices1'])
            nsamp2 = len(dct['indices2'])

            if nsamp1 == 0 or nsamp2 == 0: 
                return None, None, niter

            if nsamp1 >= nsamp2:
                ### match to distribution 2
                mean_key = 'mean2'
                std_key = 'std2'

                ### Eliminate indices from distribution 1
                elim_key = 'push_com1'
                elim_ind = 'indices1'
            else:
                mean_key = 'mean1'
                std_key = 'std1'

                elim_key = 'push_com2'
                elim_ind = 'indices2'

            ### Compute cost of each sample of push_command 
            mean_diff = dct[elim_key][dct[elim_ind], :] - dct[mean_key][np.newaxis, :]
            
            ### So dimensions with high variance aren't weighted as strongly 
            cost = mean_diff / dct[std_key][np.newaxis, :]
        
            ### Sum across dimensions 
            cost_sum = np.sum(cost**2, axis=1)
            
            ### How many pts to drop
            Npts = len(cost_sum)
            NptsDrop = int(np.floor(perc_drop * Npts))
            
            ### Sort from low to high 
            ix_sort = np.argsort(cost_sum)
            
            ### Remove the highest cost 
            ix_remove = ix_sort[-NptsDrop:]
            
            ### Update indices to remove cost
            dct[elim_ind] = np.delete(dct[elim_ind], ix_remove)

            ### Make sure the cost when down 
            assert(np.sum(np.sum(cost[ix_sort[:-NptsDrop]]**2, axis=1)) < np.sum(cost_sum))

def plot_example_neuron_comm(neuron_ix = 36, mag = 0, ang = 7, animal='grom', day_ix = 0, nshuffs = 1000,
                            min_bin_indices = 0, save=False):
    """
    Plot example single neuron and population deviations from global distribution
    
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
    
    ###### Extract data #######
    spks, push, tsk, trg, bin_num, rev_bin_num, move, dat = util_fcns.get_data_from_shuff(animal, day_ix)
    
    ### Multiply spks by 10 early on; 
    spks = spks*10

    ### Get number of neurons 
    nneur = spks.shape[1]

    ### Get command bins 
    mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))
    command_bins = util_fcns.commands2bins([push], mag_boundaries, animal, day_ix, 
                                       vel_ix=[3, 5])[0]

    ### 2 plots --> single neuron and vector 
    f, ax = plt.subplots(figsize=(4,4))
    fvect, axvect = plt.subplots(figsize=(4,4))

    ### Return indices for the command ### 
    ix_com = return_command_indices(bin_num, rev_bin_num, push, mag_boundaries, animal=animal, 
                            day_ix=day_ix, mag=mag, ang=ang, min_bin_num=min_bin_indices,
                            min_rev_bin_num=min_bin_indices)

    cnt = 0
    mFR = {}; 
    mFR_vect = {};
    mov_number = []; 
    mFR_shuffle = {}
    mFR_shuffle_vect = {}

    ### For all movements --> figure otu which ones to keep in the global distribution ###
    global_comm_indices = {}
    ix_com_global = []
    for mov in np.unique(move[ix_com]):

        ### Movement specific command indices 
        ix_mc = np.nonzero(move[ix_com] == mov)[0]
        
        ### Which global indices used for command/movement 
        ix_mc_all = ix_com[ix_mc] 

        ### If enought of these then proceed; 
        if len(ix_mc) >= 15:    

            global_comm_indices[mov] = ix_mc_all
            ix_com_global.append(ix_mc_all)

    if len(ix_com_global) > 0:
        ix_com_global = np.hstack((ix_com_global))

    ### Make sure command 
    assert(np.all(np.array([i in ix_com for i in ix_com_global])))

    ### Make sure in the movements we want 
    assert(np.all(np.array([move[i] in global_comm_indices.keys() for i in ix_com_global])))

    ### Only analyze for commands that have > 1 movement 
    if len(global_comm_indices.keys()) > 1:

        #### now that have all the relevant movements - proceed 
        for mov in global_comm_indices.keys(): 

            ### FR for neuron ### 
            ix_mc_all = global_comm_indices[mov]

            FR = spks[ix_mc_all, neuron_ix]
            FR_vect = spks[ix_mc_all, :]

            ### Figure out which of the "ix_com" indices can be used for shuffling for this movement 
            ix_ok, niter = distribution_match_global_mov(push[np.ix_(ix_mc_all, [3, 5])], 
                                                         push[np.ix_(ix_com_global, [3, 5])])
            print('Mov %.1f, # Iters %d to match global'%(mov, niter))
            
            ### which indices we can use in global distribution for this shuffle ----> #### 
            ix_com_global_ok = ix_com_global[ix_ok] 
            global_mean_vect = np.mean(spks[ix_com_global_ok, :], axis=0)
            
            assert(np.all(command_bins[ix_com_global_ok, 0] == mag))
            assert(np.all(command_bins[ix_com_global_ok, 1] == ang))

            ### make sure both movmenets still represneted. 
            assert(len(np.unique(move[ix_com_global_ok])) > 1)
            
            Nglobal = len(ix_com_global_ok)
            Ncommand_mov = len(ix_mc_all)
            assert(Nglobal > Ncommand_mov)
            
            ### Get matching global distribution 
            mFR_shuffle[mov] = []
            mFR_shuffle_vect[mov] = []
            
            for i_shuff in range(nshuffs):
                ix_sub = np.random.permutation(Nglobal)[:Ncommand_mov]
                mFR_shuffle[mov].append(np.mean(spks[ix_com_global_ok[ix_sub], neuron_ix]))
                
                mshuff_vect = np.mean(spks[ix_com_global_ok[ix_sub], :], axis=0)
                mFR_shuffle_vect[mov].append(np.linalg.norm(mshuff_vect - global_mean_vect)/nneur)
            
            mFR[mov] = np.mean(FR)
            mFR_vect[mov] = np.linalg.norm(np.mean(FR_vect, axis=0) - global_mean_vect)/nneur
            mov_number.append(mov)
           
            print('mov %.1f, N = %d, mFR = %.2f'% (mov, len(ix_mc), np.mean(FR)))

        #### Plot by target number ####
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
            
            ### Single neuron 
            util_fcns.draw_plot(x, mFR_shuffle[mov], colrgba, np.array([0., 0., 0., 0.]), ax)
            ax.plot(x, mFR[mov], 'k.')
            ax.hlines(np.mean(spks[ix_com_global, neuron_ix]), xlim[0], xlim[-1], color='gray',
                linewidth=1., linestyle='dashed')
            
            ### Population centered by shuffle mean 
            mn_shuf = np.mean(mFR_shuffle_vect[mov])
            util_fcns.draw_plot(x, mFR_shuffle_vect[mov] - mn_shuf, colrgba, np.array([0., 0., 0., 0.]), axvect)
            axvect.plot(x, mFR_vect[mov] - mn_shuf, 'k.')
        
        for axi in [ax, axvect]:
            axi.set_xlim(xlim)        
            axi.set_xlabel('Movement')
            
        ax.set_ylabel('Activity (Hz)')   
        ax.set_title('Neuron %d' %(neuron_ix))

        if mag == 0 and ang == 7 and animal == 'grom' and day_ix == 0 and neuron_ix == 36:
            ax.set_ylim([5, 35])
    
        f.tight_layout()
        if save:
            util_fcns.savefig(f, 'n%d_mag%d_ang%d_%s_d%d_min_bin%d'%(neuron_ix, mag, ang, animal, day_ix, min_bin_indices))
        
        axvect.set_ylabel('Pop. Activity Dist. \n centered by shuffle mn (Hz) ')    
        fvect.tight_layout()
        if save:
            util_fcns.savefig(fvect, 'POP_mag%d_ang%d_%s_d%d_min_bin%d'%(mag, ang, animal, day_ix, min_bin_indices))

##### TO DO only analyze behaviors with sig. effects from Fig. 2 ########
def perc_neuron_command_move_sig(nshuffs = 1000, min_bin_indices = 0):
    """
    Args:
        nshuffs (int, optional): Description
        min_bin_indices (int, optional): number of bins to remove from beginning AND end of the trial 
    """

    ### Open mag boundaries 
    mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))

    ### Dictionaries to hold the pvalues and effect sizes of the signfiicant and non-sig 
    perc_sig = {}; 
    perc_sig_vect = {}; 

    dtype_su = [('pv', float), ('abs_diff_fr', float), ('frac_diff_fr', float), ('glob_fr', float)]
    dtype_pop = [('pv', float), ('norm_diff_fr', float)]

    niter2match = {}
    
    for ia, animal in enumerate(['grom', 'jeev']):
        
        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):
            
            ### Setup the dictionaries for this animal and date
            perc_sig[animal, day_ix] = {}
            perc_sig_vect[animal, day_ix] = {}
            niter2match[animal, day_ix] = {}
                        
            ### Pull data ### 
            spks, push, tsk, trg, bin_num, rev_bin_num, move, dat = util_fcns.get_data_from_shuff(animal, day_ix)
            spks = spks * 10
            nneur = spks.shape[1]
            command_bins = util_fcns.commands2bins([push], mag_boundaries, animal, day_ix, 
                                       vel_ix=[3, 5])[0]

            #### Add this for easier analysis later 
            perc_sig[animal, day_ix, 'nneur'] = nneur
            
            ### For each command get: ###
            for mag in range(4):
                
                for ang in range(8): 
            
                    #### Common indices 
                    #### Get the indices for command ####
                    ix_com = return_command_indices(bin_num, rev_bin_num, push, mag_boundaries, mag=mag, ang=ang,
                                           animal=animal, day_ix=day_ix, min_bin_num=min_bin_indices,
                                           min_rev_bin_num=min_bin_indices)
                    
                    ix_com_global = []
                    global_comm_indices = {}

                    #### Go through the movements ####
                    for mov in np.unique(move[ix_com]):
                        
                        ### Movement specific command indices 
                        ix_mc = np.nonzero(move[ix_com] == mov)[0]
                        ix_mc_all = ix_com[ix_mc]
                        
                        ### If enough of these then proceed; 
                        if len(ix_mc) >= 15:    

                            global_comm_indices[mov] = ix_mc_all
                            ix_com_global.append(ix_mc_all)

                    if len(ix_com_global) > 0:
                        ix_com_global = np.hstack((ix_com_global))


                        ### ONly command, only movs we want: ####
                        assert(np.all(command_bins[ix_com_global, 0] == mag))
                        assert(np.all(command_bins[ix_com_global, 1] == ang))
                        assert(np.all(np.array([move[i] in global_comm_indices.keys() for i in ix_com_global])))

                        ### Iterate through the moves we want 
                        for mov in global_comm_indices.keys(): 

                            perc_sig[animal, day_ix][mag, ang, mov] = []
                            
                            ### get indices #### 
                            ix_mc_all = global_comm_indices[mov]
                            Ncommand_mov = len(ix_mc_all)
                            
                            ### FR for neuron ### 
                            mov_mean_FR = np.mean(spks[ix_mc_all, :], axis=0)
                            
                            ### Figure out which of the "ix_com" indices can be used for shuffling for this movement 
                            ix_ok, niter = distribution_match_global_mov(push[np.ix_(ix_mc_all, [3, 5])], 
                                                         push[np.ix_(ix_com_global, [3, 5])])
                            niter2match[animal, day_ix][mag, ang, mov] = niter 
                            
                            ### which indices we can use in global distribution for this shuffle ----> #### 
                            ix_com_global_ok = ix_com_global[ix_ok] 
                            assert(np.all(command_bins[ix_com_global_ok, 0] == mag))
                            assert(np.all(command_bins[ix_com_global_ok, 1] == ang))
                            assert(np.all(np.array([move[i] in global_comm_indices.keys() for i in ix_com_global_ok])))
                            Nglobal = len(ix_com_global_ok)

                            ### Use this as global mean for this movement #####
                            global_mean_FR = np.mean(spks[ix_com_global_ok, :], axis=0)
                            
                            ### Get difference now: 
                            dmean_FR = np.abs(mov_mean_FR - global_mean_FR)
                            
                            ### Get shuffled differences saved; 
                            dmFR_shuffle = []; ## Absolute differences from global 
                            mFR_shuffle = [] ### Just shuffled mFR 
                            
                            for i_shuff in range(nshuffs):
                                ix_sub = np.random.permutation(Nglobal)[:Ncommand_mov]
                                mn_tmp = np.mean(spks[ix_com_global_ok[ix_sub], :], axis=0)
                                
                                dmFR_shuffle.append(np.abs(mn_tmp - global_mean_FR))
                                mFR_shuffle.append(mn_tmp)
                                
                            ### Stack ####
                            dmFR_shuffle = np.vstack((dmFR_shuffle))
                            mFR_shuffle = np.vstack((mFR_shuffle))
                            
                            ### For each neuron go through and do the signficiance test ####
                            for i_neur in range(nneur):

                                ### Find differences that are greater than shuffle 
                                n_lte = len(np.nonzero(dmFR_shuffle[:, i_neur] >= dmean_FR[i_neur])[0])
                                pv = float(n_lte) / float(nshuffs)

                                ### find difference from the global mean 
                                dFR = np.abs(mov_mean_FR[i_neur] - global_mean_FR[i_neur])
                                perc_sig[animal, day_ix][mag, ang, mov].append(np.array((pv, dFR, dFR/global_mean_FR[i_neur], global_mean_FR[i_neur]), dtype=dtype_su))
                                
                            #### Vector #####
                            n_lte = len(np.nonzero(np.linalg.norm(dmFR_shuffle, axis=1) >= np.linalg.norm(dmean_FR))[0])
                            pv = float(n_lte) / float(nshuffs)
                            dist = np.linalg.norm(mov_mean_FR - global_mean_FR)
                            perc_sig_vect[animal, day_ix][mag, ang, mov] = np.array((pv, dist/nneur), dtype=dtype_pop)

                                
    return perc_sig, perc_sig_vect, niter2match

def plot_su_pop_stats(perc_sig, perc_sig_vect, plot_sig_mov_comm_grid = False, min_fr_frac_neur_diff = 0.5):

    #### Plot this; 
    #### Single neuron plots ######
    f, ax = plt.subplots(figsize=(4, 4)) ### Percent sig; 
    f2, ax2 = plt.subplots(figsize=(4, 4)) ### Percent neurons with > 0 sig; 
    feff, axeff = plt.subplots(figsize=(6, 4)) ### Effect size for sig. neurons 
    feff2, axeff2 = plt.subplots(figsize=(6, 4)) ### Effect size (frac diff)

    #### Vector plots ######
    fv, axv = plt.subplots(figsize=(4, 4)) ### Perc sig. command/movs
    fv2, axv2 = plt.subplots(figsize=(4, 4)) ### Perc commands with > 0 sig mov

    # Just show abstract vector norm diff ? 
    fveff, axveff = plt.subplots(figsize=(6, 4)) ### Effect size sig. command/mov

    #### For each animal 
    for ia, animal in enumerate(['grom', 'jeev']):

        ### Bar plots ####
        bar_dict = dict(percNCMsig = [], percNgte1CMsig = [], percCMsig=[], percCgte1Msig = [])

        ### For each date ####        
        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):
            
            nneur = perc_sig[animal, day_ix, 'nneur']
            num_sig = np.zeros((nneur, ))

            NCM_sig = 0
            NCM_tot = 0 

            NCM_sig_hz_diff = []
            NCM_sig_frac_diff = []
            
            ###### Single neuron ########
            for i, (k, varr) in enumerate(perc_sig[animal, day_ix].items()):
                
                ### Pvalues ####
                varr = np.vstack((varr))
                v = np.squeeze(varr['pv'])
                
                for j, vi in enumerate(v): 
                    if vi < 0.05:
                        NCM_sig += 1
                        num_sig[j] += 1
                        
                        NCM_sig_hz_diff.append(float(varr['abs_diff_fr'][j]))

                        if float(varr['glob_fr'][j]) >= min_fr_frac_neur_diff:
                            NCM_sig_frac_diff.append(float(varr['frac_diff_fr'][j]))
                        assert(vi == varr['pv'][j])
                        
                    NCM_tot += 1
            
            ####### Single Neuorns ######
            ax.plot(ia, float(NCM_sig)/float(NCM_tot), 'k.')
            bar_dict['percNCMsig'].append(float(NCM_sig)/float(NCM_tot))
            
            sig_n = len(np.nonzero(num_sig > 0)[0])
            bar_dict['percNgte1CMsig'].append(float(sig_n) / float(nneur))
            ax2.plot(ia, float(sig_n) / float(nneur), 'k.')

            ##### Plot the effect size #######
            util_fcns.draw_plot(day_ix + 10*ia, NCM_sig_hz_diff, 'k', 'w', axeff)
            util_fcns.draw_plot(day_ix + 10*ia, NCM_sig_frac_diff, 'k', 'w', axeff2)
            
            ##### Vector ########
            vect_sig = 0 
            vect_tot = 0 
                        
            ####################
            commands_w_sig = np.zeros((4, 8)) ### How many are sig? 
            command_cnt = np.zeros((4, 8)) ### How many movements count? 
            sig_hz_diff_vect = []
            
            if plot_sig_mov_comm_grid:
                f, axsig = plt.subplots(ncols = 4, figsize = (12, 4))
                for ia, axi in enumerate(axsig):
                    axi.set_xlabel('Movements \n (Mag=%d)'%ia)
                    axi.set_ylabel('Angle')
                axsig[0].set_title('%s, Day%d'%(animal, day_ix))

            for i, (k, varr) in enumerate(perc_sig_vect[animal, day_ix].items()):
                v = float(varr['pv'])

                if v < 0.05:
                    vect_sig += 1
                    commands_w_sig[k[0], k[1]] += 1

                    if plot_sig_mov_comm_grid:
                        axsig[k[0]].plot(k[2], k[1], '.', color='orangered')
                    
                    ### For commands / moves that are different what is the np.linalg.norm? 
                    sig_hz_diff_vect.append(float(varr['norm_diff_fr']))
                
                else:
                    if plot_sig_mov_comm_grid:
                        axsig[k[0]].plot(k[2], k[1], '.', color='k')
                
                command_cnt[k[0], k[1]] += 1
                vect_tot += 1
            
            if plot_sig_mov_comm_grid:
                fsig.tight_layout()
            
            ####### Need at least 2 movemetns #####
            ix_consider = np.nonzero(command_cnt.reshape(-1) >= 2)[0]
            vect_tot = np.sum(command_cnt.reshape(-1)[ix_consider])
            vect_sig = np.sum(commands_w_sig.reshape(-1)[ix_consider])
            axv.plot(ia, float(vect_sig)/float(vect_tot), 'k.')
            bar_dict['percCMsig'].append(float(vect_sig)/float(vect_tot))
            
            
            ### Need at least 1 sig 
            ix1 = np.nonzero(commands_w_sig.reshape(-1)[ix_consider] >= 1)[0]
            axv2.plot(ia, float(len(ix1))/float(len(ix_consider)), 'k.')
            bar_dict['percCgte1Msig'].append(float(len(ix1))/float(len(ix_consider)))
            
            ### Distribution of population ###
            util_fcns.draw_plot(ia*10 + day_ix, sig_hz_diff_vect, 'k', 'w', axveff)
            
        ####### Single neuron summary for the day #####
        ax.bar(ia, np.mean(bar_dict['percNCMsig']), width=.8,alpha=0.2, color='k')
        ax2.bar(ia, np.mean(bar_dict['percNgte1CMsig']), width=.8, alpha=0.2, color='k')
        
        ####### Vector  neuron summary for the day #####
        axv.bar(ia, np.mean(bar_dict['percCMsig']), width=.8,alpha=0.2, color='k')
        axv2.bar(ia, np.mean(bar_dict['percCgte1Msig']), width=.8, alpha=0.2, color='k')
        
    for axi in [ax, ax2, axv, axv2]:
        axi.set_xticks([0, 1])
        axi.set_xticklabels(['G', 'J'])
        axi.set_xlim([-1, 2])

    ax.set_ylabel('Frac of neurons/command/mov sig. diff \nfrom global (nshuff=1000)')
    ax2.set_ylabel('Frac of neurons with > 0 sig. diff \nfrom global (nshuff=1000)')

    axv.set_ylabel('Frac of command/mov sig. diff \nfrom global (nshuff=1000)')
    axv2.set_ylabel('Frac of commands with > 0 sig. diff mov\nfrom global (nshuff=1000)')

    f.tight_layout()
    f2.tight_layout()
    fv.tight_layout()
    fv2.tight_layout()

    util_fcns.savefig(f, 'frac_neur_comm_mov_sig_diff_from_global')
    util_fcns.savefig(f2, 'frac_with_gte_0_sig_diff_from_global')
    util_fcns.savefig(fv, 'frac_comm_mov_sig_diff_from_global')
    util_fcns.savefig(fv2, 'frac_comm_with_gte_0_sig_diff_mov_from_global')

    ###### Effect size plots #######
    for ax_ in [axeff, axeff2, axveff]:
        ax_.set_xlim([-1, 14])
    axeff.set_ylim([0, 9.])
    axeff2.set_ylim([0, 3.0])

    axeff.set_ylabel('Activity Diff from Shuffle Mean (Hz)')
    axeff.set_title('Sig. Diff Neurons/Command/Move')

    axeff2.set_ylabel('Frac. Diff Activity from Shuffle Mean')
    axeff2.set_title('Sig. Diff Neurons/Command/Move')

    axveff.set_ylabel('Pop. Activity Diff from Shuffle Mean')
    axveff.set_title('Sig. Diff. Command/Moves')
    ######  #######

######### Behavior vs. neural correlations #########
def neuraldiff_vs_behaviordiff_corr_pairwise(min_bin_indices=0, nshuffs = 10, ncommands_psth = 5): 
    ### Open mag boundaries 
    mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))

    fr, axr = plt.subplots(figsize = (6, 4))
    fs, axs = plt.subplots(figsize = (6, 4))

    for ia, animal in enumerate(['grom', 'jeev']):
        
        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):
            
            ### Pull data ### 
            spks, push, tsk, trg, bin_num, rev_bin_num, move, dat = util_fcns.get_data_from_shuff(animal, day_ix)
            spks = spks * 10
            nneur = spks.shape[1]
            command_bins = util_fcns.commands2bins([push], mag_boundaries, animal, day_ix, 
                                       vel_ix=[3, 5])[0]

            f, ax = plt.subplots()
            D = []
            D_shuff = {}
            for i in range(nshuffs):
                D_shuff[i] = []


            ### For each command: ###
            for mag in range(4):
                
                for ang in range(8): 
            
                    #### Common indices 
                    #### Get the indices for command ####
                    ix_com = return_command_indices(bin_num, rev_bin_num, push, mag_boundaries, mag=mag, ang=ang,
                                           animal=animal, day_ix=day_ix, min_bin_num=min_bin_indices,
                                           min_rev_bin_num=min_bin_indices)

                    ##### Which movements go to the global? 
                    ix_com_global = []
                    global_comm_indices = {}

                    #### Go through the movements ####
                    for mov in np.unique(move[ix_com]):
                        
                        ### Movement specific command indices 
                        ix_mc = np.nonzero(move[ix_com] == mov)[0]
                        ix_mc_all = ix_com[ix_mc]
                        
                        ### If enough of these then proceed; 
                        if len(ix_mc) >= 15:    

                            global_comm_indices[mov] = ix_mc_all
                            ix_com_global.append(ix_mc_all)

                    if len(ix_com_global) > 0:

                        ix_com_global = np.hstack((ix_com_global))
                        relevant_movs = np.array(global_comm_indices.keys())

                        shuffle_mean_FR = {}

                        ##### Get the movements that count; 
                        for imov, mov in enumerate(relevant_movs): 
                            
                            ### MOV specific 
                            ### movements / command 
                            ix_mc_all = global_comm_indices[mov]
                            Nmov = len(ix_mc_all)

                            for imov2, mov2 in enumerate(relevant_movs[imov+1:]):

                                assert(mov != mov2)
                                ix_mc_all2 = global_comm_indices[mov2]
                                Nmov2 = len(ix_mc_all2)

                                #### match to the two distributions #######
                                ### Figure out which of the "ix_com" indices can be used for shuffling for this movement 
                                ix_ok, ix_ok2, niter = distribution_match_global_mov_pairwise(push[np.ix_(ix_mc_all, [3, 5])], 
                                                             push[np.ix_(ix_mc_all2, [3, 5])])
                            
                            ### which indices we can use in global distribution for this shuffle ----> #### 
                            ix_com_global_ok = ix_com_global[ix_ok] 
                            assert(np.all(command_bins[ix_com_global_ok, 0] == mag))
                            assert(np.all(command_bins[ix_com_global_ok, 1] == ang))
                            assert(np.all(np.array([move[i] in global_comm_indices.keys() for i in ix_com_global_ok])))
                            Nglobal = len(ix_com_global_ok)

                            ### Shuffle takes from teh global distribution Nmov number of point adn saves 
                            shuffle_mean_FR[mov] = []
                            for ishuff in range(nshuffs):
                                ix_shuff = np.random.permutation(Nglobal)[:Nmov]

                                ### Get the shuflfe mean FR and shuffle PSTH 
                                shuffle_mean_FR[mov].append(np.mean(spks[ix_com_global_ok[ix_shuff], :], axis=0))
                                
                        #### For each movement get the mean and PSTH 
                        for imov, mov in enumerate(relevant_movs):
                            ix_mc_all = global_comm_indices[mov]
                            mov_mean_FR = np.mean(spks[ix_mc_all, :], axis=0)
                            mov_PSTH = get_PSTH(bin_num, rev_bin_num, push, ix_mc_all, num_bins=ncommands_psth)
                            
                            assert(mov_PSTH.shape[0] == 2*ncommands_psth + 1)
                            assert(mov_PSTH.shape[1] == 2)


                            if mode == 'global':
                                
                                ### Use this as global mean for this movement #####
                                mov2_mean_FR = [np.mean(spks[ix_com_global_ok, :], axis=0)]
                                mov2_PSTH = [get_PSTH(bin_num, rev_bin_num, push, ix_com_global_ok)]

                            elif mode == 'pairwise':

                                ### Clear these ####
                                mov2_mean_FR = []
                                mov2_PSTH = []       

                                ### If not at the end ####
                                if mov != relevant_movs[-1]:

                                    for imov2, mov2 in enumerate(relevant_movs[imov+1:]):
                                        assert(mov2!=mov)

                                        ### Take these indices 
                                        ix_mc_all2 = global_comm_indices[mov2]
                                        mov2_mean_FR.append(np.mean(spks[ix_mc_all2, :], axis=0))
                                        mov2_PSTH.append(get_PSTH(bin_num, rev_bin_num, push, ix_mc_all2))


                            #### Now do the comparisons #####
                            for im2 in range(len(mov2_PSTH)):

                                dN = np.linalg.norm(mov_mean_FR - mov2_mean_FR[im2])/nneur
                                dB = np.linalg.norm(mov_PSTH - mov2_PSTH[im2])

                                ax.plot(dB, dN, 'k.')
                                D.append([dB, dN])

                                if mode == 'global':

                                    ### Get shuffled vs. global ###
                                    for i in range(nshuffs): 
                                        shuff_dN = np.linalg.norm(shuffle_mean_FR[i] - mov2_mean_FR[im2])/nneur
                                        
                                        #if i == 0:
                                        #    ax.plot(dB, shuff_dN, '.', color='gray')
                                        D_shuff[i].append([dB, shuff_dN])

                                elif mode == 'pairwise':

                                    ### Get shuffled vs. global ###
                                    for i in range(nshuffs): 
                                        shuff_dN = np.linalg.norm(shuffle_mean_FR[mov][i] - shuffle_mean_FR[mov2][i])/nneur
                                        
                                        #if i == 0:
                                            #ax.plot(dB, shuff_dN, '.', color='gray')
                                        D_shuff[i].append([dB, shuff_dN])



            D = np.vstack((D))
            slp,_,rv,_,_ = scipy.stats.linregress(D[:, 0], D[:, 1])
            ax.set_title('Compare to %s, rv %.5f, N = %d'%(mode, rv, D.shape[0]))
            ax.set_xlabel('Norm Diff Behav. PSTH (-5:5)')
            ax.set_ylabel('Norm Diff Pop Neur [0]')

            #### Plto shuffled vs. real R; 
            axr.plot(ia*10 + day_ix, rv, 'k.')
            axs.plot(ia*10 + day_ix, slp, 'k.')

            rshuff = []; slpshuff = []
            for i in range(nshuffs):
                d_tmp = np.vstack((D_shuff[i]))
                slp,_,rv,_,_ = scipy.stats.linregress(d_tmp[:, 0], d_tmp[:, 1])
                rshuff.append(rv)
                slpshuff.append(slp)
            util_fcns.draw_plot(ia*10 + day_ix, rshuff, 'k', 'w', axr)
            util_fcns.draw_plot(ia*10 + day_ix, slpshuff, 'k', 'w', axs)

            axr.set_xlim([-1, 14])
            axs.set_xlim([-1, 14])
            
            axr.set_ylabel('r-value')
            axs.set_ylabel('slope')

def get_PSTH(bin_num, rev_bin_num, push, indices, num_bins=5):

    all_push = []
    push_vel = push[:, [3, 5]]
    for ind in indices:

        if bin_num[ind] >= num_bins and rev_bin_num[ind] >= num_bins:
            all_push.append(push_vel[ind-num_bins:ind+num_bins+1, :])
    all_push = np.dstack((all_push))
    return np.mean(all_push, axis=2)

