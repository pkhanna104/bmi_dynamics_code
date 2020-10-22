import analysis_config
from online_analysis import util_fcns, generate_models
from matplotlib import colors as mpl_colors

import scipy.stats 
import numpy as np
import matplotlib.pyplot as plt

import pickle
import seaborn
seaborn.set(font='Arial',context='talk',font_scale=1.5, style='white')

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
                return [], [], niter

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
    fdist, axdist = plt.subplots(figsize=(3, 3))
    f, ax = plt.subplots(figsize=(3, 3))
    fvect, axvect = plt.subplots(figsize=(3.5, 3))

    ### Return indices for the command ### 
    ix_com = return_command_indices(bin_num, rev_bin_num, push, mag_boundaries, animal=animal, 
                            day_ix=day_ix, mag=mag, ang=ang, min_bin_num=min_bin_indices,
                            min_rev_bin_num=min_bin_indices)

    cnt = 0
    mFR = {}; 
    mFR_vect = {};
    mov_number = []; 

    ### Save 
    mFR_shuffle = {}
    mFR_shuffle_vect = {}
    mFR_dist = {}


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
            mFR_dist[mov] = []
            
            for i_shuff in range(nshuffs):
                ix_sub = np.random.permutation(Nglobal)[:Ncommand_mov]
                mFR_shuffle[mov].append(np.mean(spks[ix_com_global_ok[ix_sub], neuron_ix]))
                
                mshuff_vect = np.mean(spks[ix_com_global_ok[ix_sub], :], axis=0)
                mFR_shuffle_vect[mov].append(np.linalg.norm(mshuff_vect - global_mean_vect)/nneur)
            
            ### Save teh distribution ####
            mFR_dist[mov] = FR 

            ### Save teh shuffles 
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
            
            #### col rgb ######
            colrgb = util_fcns.rgba2rgb(colrgba)

            #### SIngle neuron distribution ###
            util_fcns.draw_plot(x, mFR_dist[mov], colrgb, np.array([0., 0., 0., 0.]), axdist,
                skip_median=True)
            axdist.hlines(np.mean(mFR_dist[mov]), x-.25, x + .25, color=colrgb,
                linewidth=3.)
            axdist.hlines(np.mean(spks[ix_com_global, neuron_ix]), xlim[0], xlim[-1], color='gray',
                linewidth=1., linestyle='dashed')


            ### Single neuron --> sampling distribution 
            util_fcns.draw_plot(x, mFR_shuffle[mov], 'gray', np.array([0., 0., 0., 0.]), ax)
            ax.plot(x, mFR[mov], '.', color=colrgb, markersize=20)
            ax.hlines(np.mean(spks[ix_com_global, neuron_ix]), xlim[0], xlim[-1], color='gray',
                linewidth=1., linestyle='dashed')
            
            ### Population centered by shuffle mean 
            mn_shuf = np.mean(mFR_shuffle_vect[mov])
            util_fcns.draw_plot(x, mFR_shuffle_vect[mov] - mn_shuf, 'gray', np.array([0., 0., 0., 0.]), axvect)
            axvect.plot(x, mFR_vect[mov] - mn_shuf, '.', color=colrgb, markersize=20)
        


        for axi in [ax, axvect, axdist]:
            axi.set_xlim(xlim)        
            axi.set_xlabel('Movement')
            axi.set_xticks([])
            
        ax.set_ylabel('Activity (Hz)')   
        axdist.set_ylabel('Activity (Hz)')  
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

        axdist.set_ylim([-2, 52])
        fdist.tight_layout()
        if save:
            util_fcns.savefig(fdist, 'n%d_dist_mag%d_ang%d_%s_d%d_min_bin%d'%(neuron_ix, mag, ang, animal, day_ix, min_bin_indices))

def plot_example_beh_comm(mag = 0, ang = 7, animal='grom', day_ix = 0, nshuffs = 1000, min_bin_indices = 0,
    save = False, center_by_global = False): 
    
    '''
    Method to plot distribution of command-PSTH compared to the global distribution; 
    For each movement make sure you're subsampling the global distribution to match the movement-specific one
    Then compute difference between global mean and subsampled and global mean vs. movement specific and plot as distribuiton 

    center_by_global --> whether to center each point by the global distribution 
    '''
    pref_colors = analysis_config.pref_colors
    
    ###### Extract data #######
    _, push, tsk, trg, bin_num, rev_bin_num, move, dat = util_fcns.get_data_from_shuff(animal, day_ix)
    
    ###### Get magnitude difference s######
    mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))
    command_bins = util_fcns.commands2bins([push], mag_boundaries, animal, day_ix, 
                                       vel_ix=[3, 5])[0]

    f, ax = plt.subplots(figsize=(4,4))
    f2, ax2 = plt.subplots(figsize=(4,4))
    
    ### Return indices for the command ### 
    ix_com = return_command_indices(bin_num, rev_bin_num, push, mag_boundaries, animal=animal, 
                            day_ix=day_ix, mag=mag, ang=ang, min_bin_num=min_bin_indices,
                            min_rev_bin_num=min_bin_indices)


    ### For all movements --> figure otu which ones to keep in the global distribution ###
    global_comm_indices = {}
    beh_shuffle = {}
    beh_diff = {}

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

            ### PSTH for command-movement ### 
            ix_mc_all = global_comm_indices[mov]
            mean_mc_PSTH = get_PSTH(bin_num, rev_bin_num, push, ix_mc_all, num_bins=5)
            mean_mc_PSTH_osa = get_osa_PSTH(bin_num, rev_bin_num, push, ix_mc_all)

            ### Figure out which of the "ix_com" indices can be used for shuffling for this movement 
            ix_ok, niter = distribution_match_global_mov(push[np.ix_(ix_mc_all, [3, 5])], 
                                                         push[np.ix_(ix_com_global, [3, 5])])
            print('Mov %.1f, # Iters %d to match global'%(mov, niter))
            
            ### which indices we can use in global distribution for this shuffle ----> #### 
            ix_com_global_ok = ix_com_global[ix_ok] 
            global_com_PSTH_mov = get_PSTH(bin_num, rev_bin_num, push, ix_com_global_ok, num_bins=5)
            global_com_PSTH_mov_osa = get_osa_PSTH(bin_num, rev_bin_num, push, ix_com_global_ok)
            
            dPSTH = np.linalg.norm(mean_mc_PSTH - global_com_PSTH_mov)
            dPSTH_osa = np.linalg.norm(mean_mc_PSTH_osa - global_com_PSTH_mov_osa)

            beh_diff[mov] = [dPSTH, dPSTH_osa]

            assert(np.all(command_bins[ix_com_global_ok, 0] == mag))
            assert(np.all(command_bins[ix_com_global_ok, 1] == ang))

            ### make sure both movmenets still represneted. 
            assert(len(np.unique(move[ix_com_global_ok])) > 1)
            
            Nglobal = len(ix_com_global_ok)
            Ncommand_mov = len(ix_mc_all)
            assert(Nglobal > Ncommand_mov)
            
            ### Get matching global distribution 
            beh_shuffle[mov] = []
            for i_shuff in range(nshuffs):
                ix_sub = np.random.permutation(Nglobal)[:Ncommand_mov]
                shuff_psth = get_PSTH(bin_num, rev_bin_num, push, ix_com_global_ok[ix_sub], num_bins=5)
                shuff_psth_osa = get_osa_PSTH(bin_num, rev_bin_num, push, ix_com_global_ok[ix_sub])

                dff = np.linalg.norm(shuff_psth - global_com_PSTH_mov)
                dff_osa = np.linalg.norm(shuff_psth_osa - global_com_PSTH_mov_osa)
                beh_shuffle[mov].append([dff, dff_osa])

        ### Now do the plots 
        #### Plot by target number ####
        keys = np.argsort(np.hstack((beh_diff.keys())) % 10)
        xlim = [-1, len(keys)]

        #### Sort correctly #####
        for x, mov in enumerate(np.hstack((beh_diff.keys()))[keys]):

            ### Draw shuffle ###
            colnm = pref_colors[int(mov)%10]
            colrgba = np.array(mpl_colors.to_rgba(colnm))
            
            ### Set alpha according to task (tasks 0-7 are CO, tasks 10.0 -- 19.1 are OBS) 
            if mov >= 10:
                colrgba[-1] = 0.5
            else:
                colrgba[-1] = 1.0
            
            ### Get rgba ####
            colrgb = util_fcns.rgba2rgb(colrgba)

            ### trajectory  
            bs_mov = np.vstack((beh_shuffle[mov]))
            if center_by_global:
                mean_shuff = np.mean(bs_mov[:, 0])
                mean_shuff_osa = np.mean(bs_mov[:, 1])
                
            else:
                mean_shuff = 0. 
                mean_shuff_osa = 0.

            util_fcns.draw_plot(x, bs_mov[:, 0] - mean_shuff, 'gray', np.array([0., 0., 0., 0.]), ax)
            ax.plot(x, beh_diff[mov][0] - mean_shuff, '.', color=colrgb, markersize=20)
            
            util_fcns.draw_plot(x, bs_mov[:, 1] - mean_shuff_osa, 'gray', np.array([0., 0., 0., 0.]), ax2)
            ax2.plot(x, beh_diff[mov][1] - mean_shuff_osa, '.', color=colrgb, markersize=20)

            ### Population centered by shuffle mean 
            for axi in [ax, ax2]:
                axi.set_xlim(xlim)        
                axi.set_xlabel('Movement')
            
        ax.set_ylabel('Move-Specific Command Traj Diff\nfrom Move-Pooled Command Traj')   
        ax2.set_ylabel('Move-Specific Next Command Diff\nfrom Move-Pooled Next Command')   
        
        f.tight_layout()
        f2.tight_layout()
        if save:
            util_fcns.savefig(f, 'Beh_diff_mag%d_ang%d_%s_d%d_min_bin%d'%(mag, ang, animal, day_ix, min_bin_indices))
            util_fcns.savefig(f2, 'Next_command_diff_mag%d_ang%d_%s_d%d'%(mag, ang, animal, day_ix))
    

##### Main plotting functions to use in Figs 2 and 3 ########
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

def plot_su_pop_stats(perc_sig, perc_sig_vect, sig_move_diffs = None, 
    plot_sig_mov_comm_grid = False, min_fr_frac_neur_diff = 0.5):

    #### Plot this; 
    ############ Single neuron plots ##############
    f, ax = plt.subplots(figsize=(2, 3)) ### Percent sig; 
    f2, ax2 = plt.subplots(figsize=(2, 3)) ### Percent neurons with > 0 sig; 
    
    ### Percent sig of sig diff mov-commands SU and Vect; 
    fsig_su, axsig_su = plt.subplots(figsize=(2, 3))

    feff, axeff = plt.subplots(figsize=(3, 3)) ### Effect size for sig. neurons 
    feff2, axeff2 = plt.subplots(figsize=(3, 3)) ### Effect size (frac diff)

    ### Significant onesl 
    feff_bsig, axeff_bsig = plt.subplots(figsize=(3, 3))
    feff_bsig_frac, axeff_bsig_frac = plt.subplots(figsize=(3, 3))

    ############ Vector plots ##############
    fv, axv = plt.subplots(figsize=(2, 3)) ### Perc sig. command/movs
    fv2, axv2 = plt.subplots(figsize=(2, 3)) ### Perc commands with > 0 sig mov

    # Just show abstract vector norm diff ? 
    fveff, axveff = plt.subplots(figsize=(3, 3)) ### Effect size sig. command/mov
    fveff_bsig, axveff_bsig = plt.subplots(figsize=(3, 3)) ###

    ### Percent sig of sig diff mov-commands  Vect; 
    fsig_pop, axsig_pop = plt.subplots(figsize=(2, 3))

    #### For each animal 
    for ia, animal in enumerate(['grom', 'jeev']):

        ### Bar plots ####
        bar_dict = dict(percNCMsig = [], percNgte1CMsig = [], percCMsig=[], percCgte1Msig = [], 
            percNCMsig_behsig = [], percCMsig_behsig = [])

        ### For each date ####        
        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):
            
            nneur = perc_sig[animal, day_ix, 'nneur']
            num_sig = np.zeros((nneur, ))

            NCM_sig = 0
            NCM_tot = 0 

            NCM_sig_beh_sig = 0
            NCM_tot_beh_sig = 0

            NCM_sig_hz_diff = []
            NCM_sig_frac_diff = []

            NCM_behsig_hz_diff = []
            NCM_behsig_frac_diff = []
            
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
                        
                        ### See if behavior is sig: 
                        if sig_move_diffs is not None: 
                            if list(k) in sig_move_diffs[animal, day_ix]: 
                                NCM_sig_beh_sig += 1
                                NCM_behsig_hz_diff.append(float(varr['abs_diff_fr'][j]))
                                
                                #### Make sure global FR > 0.5 Hz; 
                                if float(varr['glob_fr'][j]) >= min_fr_frac_neur_diff:
                                    NCM_behsig_frac_diff.append(float(varr['frac_diff_fr'][j]))

                    NCM_tot += 1
                    if sig_move_diffs is not None: 
                        if list(k) in sig_move_diffs[animal, day_ix]:
                            NCM_tot_beh_sig += 1
            
            ####### Single Neuorns ######
            ax.plot(ia, float(NCM_sig)/float(NCM_tot), 'k.')
            bar_dict['percNCMsig'].append(float(NCM_sig)/float(NCM_tot))
            
            sig_n = len(np.nonzero(num_sig > 0)[0])
            bar_dict['percNgte1CMsig'].append(float(sig_n) / float(nneur))
            ax2.plot(ia, float(sig_n) / float(nneur), 'k.')

            frac_sig_beh_sig = float(NCM_sig_beh_sig)/float(NCM_tot_beh_sig)
            axsig_su.plot(ia, frac_sig_beh_sig, 'k.')
            bar_dict['percNCMsig_behsig'].append(frac_sig_beh_sig)

            ##### Plot the effect size #######
            util_fcns.draw_plot(day_ix + 10*ia, NCM_sig_hz_diff, 'k', 'w', axeff)
            util_fcns.draw_plot(day_ix + 10*ia, NCM_sig_frac_diff, 'k', 'w', axeff2)

            util_fcns.draw_plot(day_ix + 10*ia, NCM_behsig_hz_diff, 'k', 'w', axeff_bsig)
            util_fcns.draw_plot(day_ix + 10*ia, NCM_behsig_frac_diff, 'k', 'w', axeff_bsig_frac)

            
            ##### Vector ########
            vect_sig = 0 
            vect_tot = 0 

            vect_sig_beh_sig = 0
            vect_tot_beh_sig = 0
                        
            ####################
            commands_w_sig = np.zeros((4, 8)) ### How many are sig? 
            command_cnt = np.zeros((4, 8)) ### How many movements count? 
            sig_hz_diff_vect = []
            sig_hz_diff_vect_bsig = []

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

                    if sig_move_diffs is not None: 
                        if list(k) in sig_move_diffs[animal, day_ix]:
                            vect_sig_beh_sig += 1
                            sig_hz_diff_vect_bsig.append(float(varr['norm_diff_fr']))
                
                else:
                    if plot_sig_mov_comm_grid:
                        axsig[k[0]].plot(k[2], k[1], '.', color='k')
                
                command_cnt[k[0], k[1]] += 1
                vect_tot += 1

                if sig_move_diffs is not None: 
                    if list(k) in sig_move_diffs[animal, day_ix]:
                        vect_tot_beh_sig += 1
            
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
            util_fcns.draw_plot(ia*10 + day_ix, sig_hz_diff_vect_bsig, 'k', 'w', axveff_bsig)
            
            ### Plot the perc sig CM with sig diff from glboal 
            tmp_frc = float(vect_sig_beh_sig) / float(vect_tot_beh_sig)
            axsig_pop.plot(ia, tmp_frc, 'k.')
            bar_dict['percCMsig_behsig'].append(tmp_frc)
            print('A %s, D %d, perc sig. tot %d/%d  =%.2f' %(animal, day_ix, vect_sig, vect_tot, float(vect_sig)/float(vect_tot)))
            print('A %s, D %d, perc sig. of beh sig %d/%d  =%.2f' %(animal, day_ix, vect_sig_beh_sig, vect_tot_beh_sig, tmp_frc))


        ####### Single neuron summary for the day #####
        ax.bar(ia, np.mean(bar_dict['percNCMsig']), width=.8,alpha=0.2, color='k')
        ax2.bar(ia, np.mean(bar_dict['percNgte1CMsig']), width=.8, alpha=0.2, color='k')
        axsig_su.bar(ia, np.mean(bar_dict['percNCMsig_behsig']), width=.8,alpha=0.2, color='k')

        ####### Vector  neuron summary for the day #####
        axv.bar(ia, np.mean(bar_dict['percCMsig']), width=.8,alpha=0.2, color='k')
        axv2.bar(ia, np.mean(bar_dict['percCgte1Msig']), width=.8, alpha=0.2, color='k')
        axsig_pop.bar(ia, np.mean(bar_dict['percCMsig_behsig']), width=.8, alpha=0.2, color='k')

    for axi in [ax, ax2, axv, axv2, axsig_su, axsig_pop]:
        axi.set_xticks([0, 1])
        axi.set_xticklabels(['G', 'J'])
        axi.set_xlim([-1, 2])

    ax.set_ylabel('Frac NCM sig. diff \nfrom global (nshuff=1000)')
    ax2.set_ylabel('Frac N with > 0 sig. CM \nfrom global (nshuff=1000)')
    axsig_su.set_ylabel('Frac of NCM sig. diff \n if CM sig. diff from global')
    axsig_pop.set_ylabel('Frac of CM sig. diff \n if Cm sig. diff from global')

    axv.set_ylabel('Frac CM sig. diff \nfrom global (nshuff=1000)')
    axv2.set_ylabel('Frac C > 0 sig. M \nfrom global (nshuff=1000)')

    f.tight_layout()
    f2.tight_layout()
    fv.tight_layout()
    fv2.tight_layout()
    fsig_su.tight_layout()
    fsig_pop.tight_layout()

    util_fcns.savefig(f, 'frac_neur_comm_mov_sig_diff_from_global')
    util_fcns.savefig(f2, 'frac_with_gte_0_sig_diff_from_global')
    
    util_fcns.savefig(fsig_su, 'frac_NCM_sig_diff_from_global_of_beh_diff')
    util_fcns.savefig(fsig_pop, 'frac_CM_sig_diff_from_global_of_beh_diff')
    
    util_fcns.savefig(fv, 'frac_comm_mov_sig_diff_from_global')
    util_fcns.savefig(fv2, 'frac_comm_with_gte_0_sig_diff_mov_from_global')

    ###### Effect size plots #######
    for ax_ in [axeff, axeff2, axveff, axveff_bsig, axeff_bsig, axeff_bsig_frac]:
        ax_.set_xlim([-1, 14])
    axeff.set_ylim([0, 10.])
    axeff_bsig.set_ylim([0., 10.])

    axeff2.set_ylim([0, 3.0])
    axeff_bsig_frac.set_ylim([0., 3.])

    axveff.set_ylim([0, .8])
    axveff_bsig.set_ylim([0, .8])

    axeff.set_ylabel('Activity Diff from Shuffle Mean (Hz)')
    axeff.set_title('Sig. Diff Neurons/Command/Move')

    axeff2.set_ylabel('Frac. Diff Activity from Shuffle Mean')
    axeff2.set_title('Sig. Diff Neurons/Command/Move')

    axveff.set_ylabel('Pop. Activity Diff from Shuffle Mean')
    axveff.set_title('Sig. Diff. Command/Moves')
    axveff_bsig.set_ylabel('Pop. Activity Diff for Beh Sig. ')

    axeff_bsig.set_ylabel('beh sig: Act. Diff from Shuffle Mean (Hz)')
    axeff_bsig_frac.set_ylabel('beh sig: Frac. Diff from Shuffle Mean')
    feff_bsig.tight_layout()
    feff_bsig_frac.tight_layout()

    ######  #######
    lab = ['hz_diff', 'frac_diff', 'hz_diff_bsig', 'frac_diff_bsig', 'vect_diff', 'vect_diff_bsig']
    F=[feff, feff2, feff_bsig, feff_bsig_frac, fveff, fveff_bsig]
    for i_, (f_, l_) in enumerate(zip(F, lab)):
        util_fcns.savefig(f_, l_)


def plot_perc_command_beh_sig_diff_than_global(nshuffs=1000, min_bin_indices=0, save=True):
    
    move_command_sig = {}

    mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))
    
    ######## Percent sig #############
    f, ax = plt.subplots(figsize=(2, 3))
    f_osa, ax_osa = plt.subplots(figsize =(2, 3))

    perc_sig = dict(grom=[], jeev=[])
    perc_sig_one_step = dict(grom=[], jeev=[])

    ######## Effect Size #############
    fe, axe = plt.subplots(figsize = (3, 3))
    fe_osa, axe_osa = plt.subplots(figsize = (3, 3))

    ############ Loop ##############
    for i_a, animal in enumerate(['grom', 'jeev']):

        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):

            move_command_sig[animal, day_ix] = []
            move_command_sig[animal, day_ix, 'osa'] = []

            ###### Extract data #######
            _, push, tsk, trg, bin_num, rev_bin_num, move, dat = util_fcns.get_data_from_shuff(animal, day_ix)
            
            ###### Get magnitude difference s######
            command_bins = util_fcns.commands2bins([push], mag_boundaries, animal, day_ix, 
                                               vel_ix=[3, 5])[0]
            total_cm = 0 ;
            total_sig_com = 0 ;

            total_cm_osa = 0; 
            total_sig_cm_osa = 0; 

            traj_diff = []
            traj_diff_one_step = []

            for mag in range(4):

                for ang in range(8): 
        
                    ### Return indices for the command ### 
                    ix_com = return_command_indices(bin_num, rev_bin_num, push, mag_boundaries, animal=animal, 
                                            day_ix=day_ix, mag=mag, ang=ang, min_bin_num=min_bin_indices,
                                            min_rev_bin_num=min_bin_indices)

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

                                ### PSTH for command-movement ### 
                                ix_mc_all = global_comm_indices[mov]
                                mean_mc_PSTH = get_PSTH(bin_num, rev_bin_num, push, ix_mc_all, num_bins=5)
                                mean_mc_PSTH_osa = get_osa_PSTH(bin_num, rev_bin_num, push, ix_mc_all)

                                ### Figure out which of the "ix_com" indices can be used for shuffling for this movement 
                                ix_ok, niter = distribution_match_global_mov(push[np.ix_(ix_mc_all, [3, 5])], 
                                                                             push[np.ix_(ix_com_global, [3, 5])])
                                #print('Mov %.1f, # Iters %d to match global'%(mov, niter))
                                
                                ### which indices we can use in global distribution for this shuffle ----> #### 
                                ix_com_global_ok = ix_com_global[ix_ok] 
                                global_com_PSTH_mov = get_PSTH(bin_num, rev_bin_num, push, ix_com_global_ok, num_bins=5)
                                glboal_osa_com_PSTH_mov = get_osa_PSTH(bin_num, rev_bin_num, push, ix_com_global_ok)

                                dPSTH = np.linalg.norm(mean_mc_PSTH - global_com_PSTH_mov)
                                dPSTH_osa = np.linalg.norm(mean_mc_PSTH_osa - glboal_osa_com_PSTH_mov)

                                assert(np.all(command_bins[ix_com_global_ok, 0] == mag))
                                assert(np.all(command_bins[ix_com_global_ok, 1] == ang))

                                ### make sure both movmenets still represneted. 
                                assert(len(np.unique(move[ix_com_global_ok])) > 1)
                                
                                Nglobal = len(ix_com_global_ok)
                                Ncommand_mov = len(ix_mc_all)
                                assert(Nglobal > Ncommand_mov)
                                
                                ### Get matching global distribution 
                                beh_shuffle = []; beh_shuffle_osa = []; 

                                for i_shuff in range(nshuffs):
                                    ix_sub = np.random.permutation(Nglobal)[:Ncommand_mov]
                                    shuff_psth = get_PSTH(bin_num, rev_bin_num, push, ix_com_global_ok[ix_sub], num_bins=5)
                                    shuff_psth_osa = get_osa_PSTH(bin_num, rev_bin_num, push, ix_com_global_ok[ix_sub])

                                    beh_shuffle.append(np.linalg.norm(shuff_psth - global_com_PSTH_mov))
                                    beh_shuffle_osa.append(np.linalg.norm(shuff_psth_osa - glboal_osa_com_PSTH_mov))

                                #### Test for sig for full trajectory ####
                                ix_tmp = np.nonzero(beh_shuffle >= dPSTH)[0]
                                pv = float(len(ix_tmp)) / float(nshuffs)
                                if pv < 0.05: 
                                    total_sig_com += 1
                                    traj_diff.append(dPSTH)
                                    move_command_sig[animal, day_ix].append([mag, ang, mov])
                                total_cm += 1

                                #### Test for sig for one-step-ahead (osa) ####
                                ix_tmp_osa = np.nonzero(beh_shuffle_osa >= dPSTH_osa)[0]
                                pv_osa = float(len(ix_tmp_osa)) / float(nshuffs)
                                if pv_osa < 0.05: 
                                    total_sig_cm_osa += 1
                                    traj_diff_one_step.append(dPSTH_osa)
                                    move_command_sig[animal, day_ix, 'osa'].append([mag, ang, mov])
                                total_cm_osa += 1

            ### Compute percent sig: 
            perc_sig[animal].append(float(total_sig_com)/float(total_cm))
            ax.plot(i_a, float(total_sig_com)/float(total_cm), 'k.')

            #### One step ahead 
            perc_sig_one_step[animal].append(float(total_sig_cm_osa)/float(total_cm_osa))
            ax_osa.plot(i_a, float(total_sig_cm_osa)/float(total_cm_osa), 'k.')
        
            #### Plot the distribution of signficiant differences 
            util_fcns.draw_plot(i_a*10 + day_ix, traj_diff, 'k', 'w', axe)
            util_fcns.draw_plot(i_a*10 + day_ix, traj_diff_one_step, 'k', 'w', axe_osa)

        ### Plot the bar; 
        ax.bar(i_a, np.mean(perc_sig[animal]), width=.8, alpha=0.2, color='k')
        ax_osa.bar(i_a, np.mean(perc_sig_one_step[animal]), width=.8, alpha=0.2, color='k')

    for axi in [ax, ax_osa]:
        axi.set_xticks([0, 1])
        axi.set_xticklabels(['G', 'J'])
        axi.set_ylim([0., 1.])
    
    ax.set_ylabel('Frac. Move-Specific Commands \nwith Sig. Diff. Traj')
    ax_osa.set_ylabel('Frac. Move-Specific Commands \nwith Sig. Diff. Next Command')

    for axi in [axe, axe_osa]:
        axi.set_xlim([-1, 14])
    axe.set_ylabel('Command Traj Diff for Sig.\nDiff. Move-Specific Commands')
    axe_osa.set_ylabel('Next Command Diff for Sig. \nDiff. Move-Specific Commands')

    f.tight_layout()
    f_osa.tight_layout()
    fe.tight_layout()
    fe_osa.tight_layout()

    if save:
        util_fcns.savefig(f, 'Beh_diff_perc_sig')
        util_fcns.savefig(f_osa, 'Beh_diff_perc_sig_one_step_ahead')

        util_fcns.savefig(fe, 'Traj_diff_sig_cm')
        util_fcns.savefig(fe_osa, 'Next_command_diff_sig_cm')

    ### Save signifciant move/commands 
    pickle.dump(move_command_sig, open(analysis_config.config['grom_pref'] + 'sig_move_comm.pkl', 'wb'))

def plot_distribution_of_nmov_per_command(): 
    """
    Plot distribution of number of movements per command as boxplot
    """
    mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))
    
    ######## Percent sig #############
    f, ax = plt.subplots(figsize=(3,3 ))

    ############ Loop ##############
    for i_a, animal in enumerate(['grom', 'jeev']):

        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):

            ###### Extract data #######
            _, push, tsk, trg, bin_num, rev_bin_num, move, dat = util_fcns.get_data_from_shuff(animal, day_ix)
            
            ###### Get magnitude difference s######
            command_bins = util_fcns.commands2bins([push], mag_boundaries, animal, day_ix, 
                                               vel_ix=[3, 5])[0]
            mov_per_com = []

            for mag in range(4):

                for ang in range(8): 
        
                    ### Return indices for the command ### 
                    ix_com = return_command_indices(bin_num, rev_bin_num, push, mag_boundaries, animal=animal, 
                                            day_ix=day_ix, mag=mag, ang=ang, min_bin_num=0,
                                            min_rev_bin_num=0)

                    ### For all movements --> figure otu which ones to keep in the global distribution ###
                    movements = 0

                    for mov in np.unique(move[ix_com]):

                        ### Movement specific command indices 
                        ix_mc = np.nonzero(move[ix_com] == mov)[0]
                        
                        ### Which global indices used for command/movement 
                        ix_mc_all = ix_com[ix_mc] 

                        ### If enough of these then proceed; 
                        if len(ix_mc) >= 15:
                            movements += 1

                    ### Add to list: 
                    if movements >= 2:
                        mov_per_com.append(movements)

            ### Plot distribution ###
            util_fcns.draw_plot(i_a*10 + day_ix, mov_per_com, 'k', 'w', ax)

    ax.set_ylabel('# move per command')
    ax.set_xlim([-1, 14])
    ax.set_xticks([])
    ax.set_xticklabels([])
    f.tight_layout()

    util_fcns.savefig(f, 'fig2_numMov_perComm')


######### Behavior vs. neural correlations #########
def neuraldiff_vs_behaviordiff_corr_pairwise(min_bin_indices=0, nshuffs = 10, ncommands_psth = 5): 
    ### Open mag boundaries 
    mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))

    fr, axr = plt.subplots(figsize = (3, 3))
    fs, axs = plt.subplots(figsize = (3, 3))

    for ia, animal in enumerate(['grom', 'jeev']):
        
        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):
            
            ### Pull data ### 
            spks, push, tsk, trg, bin_num, rev_bin_num, move, dat = util_fcns.get_data_from_shuff(animal, day_ix)
            spks = spks * 10
            nneur = spks.shape[1]
            command_bins = util_fcns.commands2bins([push], mag_boundaries, animal, day_ix, 
                                       vel_ix=[3, 5])[0]

            f, ax = plt.subplots(figsize = (3, 3))
            D = []
            D_shuff = {}
            for i in range(nshuffs):
                D_shuff[i] = []

            pairs_analyzed = {}
            plt_top = []
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
                    pairs_analyzed[mag, ang] = []

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

                            ### Get movement #2 
                            for imov2, mov2 in enumerate(relevant_movs[imov+1:]):

                                assert(mov != mov2)
                                ix_mc_all2 = global_comm_indices[mov2]
                                Nmov2 = len(ix_mc_all2)

                                #### match to the two distributions #######
                                ### Figure out which of the "ix_com" indices can be used for shuffling for this movement 
                                ix_ok1, ix_ok2, niter = distribution_match_mov_pairwise(push[np.ix_(ix_mc_all, [3, 5])], 
                                                             push[np.ix_(ix_mc_all2, [3, 5])])

                                if np.logical_and(len(ix_ok1) >= 15, len(ix_ok2) >= 15):

                                    pairs_analyzed[mag, ang].append([mov, mov2])

                                    #######################################
                                    ######### Indices check ###############
                                    #######################################
                                    for ix_test in [ix_mc_all[ix_ok1], ix_mc_all2[ix_ok2]]: 
                                        assert(np.all(command_bins[ix_test, 0] == mag))
                                        assert(np.all(command_bins[ix_test, 1] == ang))

                                    assert(np.all(np.array([move[i] == mov for i in ix_mc_all[ix_ok1]])))
                                    assert(np.all(np.array([move[i] == mov2 for i in ix_mc_all2[ix_ok2]])))

                                    #### Proceed comparing these guys ##### 
                                    mov_mean_FR1 = np.mean(spks[ix_mc_all[ix_ok1], :], axis=0)
                                    mov_mean_FR2 = np.mean(spks[ix_mc_all2[ix_ok2], :], axis=0)

                                    mov_PSTH1 = get_PSTH(bin_num, rev_bin_num, push, ix_mc_all[ix_ok1], num_bins=ncommands_psth)
                                    mov_PSTH2 = get_PSTH(bin_num, rev_bin_num, push, ix_mc_all2[ix_ok2], num_bins=ncommands_psth)

                                    assert(mov_PSTH1.shape[0] == 2*ncommands_psth + 1)
                                    assert(mov_PSTH1.shape[1] == 2)

                                    Nmov1 = len(ix_ok1)
                                    Nmov2 = len(ix_ok2)

                                    #### Matched dN and dB; 
                                    dN = np.linalg.norm(mov_mean_FR1 -mov_mean_FR2)/nneur
                                    dB = np.linalg.norm(mov_PSTH1 - mov_PSTH2)

                                    if mag == 0 and ang == 7:
                                        #### Pair 1 ### (1., 10.1), (10.1, 15.), 
                                        if mov == 1. and mov2 == 10.1:
                                            plt_top.append([dB, dN, 'blue', 15])
                                        elif mov == 10.1 and mov2 == 15.:
                                            plt_top.append([dB, dN, 'limegreen', 15])
                                        else: 
                                            plt_top.append([dB, dN, 'deeppink', 10])
                                    else:
                                        rgb = util_fcns.rgba2rgb(np.array([0., 0., 0., .5]))
                                        ax.plot(dB, dN, '.', color=rgb, markersize=5)
                                    D.append([dB, dN])

                                    ############################################################
                                    ########## Get the global taht matches the subsample #######
                                    ############################################################
                                    ### Shuffle takes from teh global distribution Nmov number of point adn saves 
                                    globix1, niter1 = distribution_match_global_mov(push[np.ix_(ix_mc_all[ix_ok1], [3, 5])], push[np.ix_(ix_com_global, [3, 5])])
                                    globix2, niter2 = distribution_match_global_mov(push[np.ix_(ix_mc_all2[ix_ok2], [3, 5])], push[np.ix_(ix_com_global, [3, 5])])
                                    
                                    Nglobal1 = len(globix1)
                                    Nglobal2 = len(globix2)

                                    ###### Get shuffles for movement 1 / movement 2
                                    shuffle_mean_FR[mov, mov2, 1] = []
                                    shuffle_mean_FR[mov, mov2, 2] = []

                                    for ishuff in range(nshuffs):
                                        #### Movement 1; 
                                        ix_shuff = np.random.permutation(Nglobal1)[:Nmov1]
                                        shuff1 = np.mean(spks[ix_com_global[globix1[ix_shuff]], :], axis=0)

                                        #### Movement 2: 
                                        ix_shuff2 = np.random.permutation(Nglobal2)[:Nmov2]
                                        shuff2 = np.mean(spks[ix_com_global[globix2[ix_shuff2]], :], axis=0)
                                 
                                        ### difference in neural ####
                                        shuff_dN = np.linalg.norm(shuff1 - shuff2)/nneur
                                        
                                        ### Add shuffle 
                                        D_shuff[ishuff].append([dB, shuff_dN])
                            
            ### Plt_top
            if len(plt_top) > 0:
                for _, (xi, yi, col, ms) in enumerate(plt_top):
                    ax.plot(xi, yi, '.', color=col, markersize=ms)
                plt_top = np.vstack((plt_top))
                _,_,rv,_,_ = scipy.stats.linregress(np.array(plt_top[:, 0], dtype=float), np.array(plt_top[:, 1], dtype=float))
                print('Example (pink) r = %.4f'%rv)

            print('#######################')
            print('Pairs analyzed')
            for i, (k, v) in enumerate(pairs_analyzed.items()):
                print('mag %d, ang %d, N = %d' %(k[0], k[1], len(v)))
            print('#######################')
            
            D = np.vstack((D))
            slp,_,rv,_,_ = scipy.stats.linregress(D[:, 0], D[:, 1])
            ax.set_title('Pairwise comparison, rv %.5f, N = %d'%(rv, D.shape[0]))
            ax.set_xlabel('Norm Diff Behav. PSTH (-5:5)')
            ax.set_ylabel('Norm Diff Pop Neur [0]')

            if animal == 'grom' and day_ix == 0:
                util_fcns.savefig(f, 'eg_neural_vs_beh_scatter%s_%d' %(animal, day_ix))

            rshuff = []; slpshuff = []
            for i in range(nshuffs):
                d_tmp = np.vstack((D_shuff[i]))
                slp,_,rv,_,_ = scipy.stats.linregress(d_tmp[:, 0], d_tmp[:, 1])
                rshuff.append(rv)
                slpshuff.append(slp)
            util_fcns.draw_plot(ia*10 + day_ix, rshuff - np.mean(rshuff), 'k', 'w', axr)
            util_fcns.draw_plot(ia*10 + day_ix, slpshuff, 'k', 'w', axs)

            #### Plto shuffled vs. real R; 
            axr.plot(ia*10 + day_ix, rv - np.mean(rshuff), 'k.')
            axs.plot(ia*10 + day_ix, slp, 'k.')

            axr.set_xlim([-1, 14])
            axs.set_xlim([-1, 14])
            
            axr.set_ylabel('r-value')
            axs.set_ylabel('slope')
    fr.tight_layout()
    util_fcns.savefig(fr, 'neur_beh_corr_rv_vs_shuffN%d'%(nshuffs))

def get_PSTH(bin_num, rev_bin_num, push, indices, num_bins=5):

    all_push = []
    push_vel = push[:, [3, 5]]
    for ind in indices:

        if bin_num[ind] >= num_bins and rev_bin_num[ind] >= num_bins:
            all_push.append(push_vel[ind-num_bins:ind+num_bins+1, :])
    all_push = np.dstack((all_push))
    return np.mean(all_push, axis=2)

def get_osa_PSTH(bin_num, rev_bin_num, push, indices):
    '''
    Get one step ahead (osa) PSTH 
    '''
    all_push = []
    push_vel = push[:, [3, 5]]
    for ind in indices:

        ### Just need to have index + 1 be part of the smae trial
        if rev_bin_num[ind] >= 1:
            all_push.append(push_vel[ind+1, :])
    all_push = np.dstack((all_push))
    return np.mean(all_push, axis=2)


 