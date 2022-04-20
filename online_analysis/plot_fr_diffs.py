import analysis_config
from online_analysis import util_fcns, generate_models, generate_models_utils
from matplotlib import colors as mpl_colors

import scipy.stats 
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
import pandas as pd

import pickle
import seaborn
seaborn.set(font='Arial',context='talk',font_scale=1.2, style='white')

import statsmodels.api as sm


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

def distribution_match_mov_multi(global_comm_indices, push, rm_perc = 0.05,
    psig = 0.05):
    
    assert(push.shape[1] == 2)

    data = dict(xy=[], mov =[], ixog = [])
    cnt = 0
    for key in global_comm_indices.keys(): 

        ix = global_comm_indices[key]
        cnt += len(ix)
        ### Mov variable ### 
        mov = np.zeros((len(ix), )) + key 
        
        ### Add this data to the data 
        data['xy'].append(push[ix, :])
        data['mov'].append(mov)
        data['ixog'].append(ix)

    print('End Cnt: %d' %(cnt))
    data['xy'] = np.vstack((data['xy']))
    data['mov'] = np.hstack((data['mov']))
    data['ixog'] = np.hstack((data['ixog']))

    keep_ix = np.arange(len(data['xy']))
    pv_all = np.array([0., 0.])

    while np.any(pv_all < psig): 
        for variable in range(2):
            ##### AnovA ####
            data_mov = []
            for key in global_comm_indices.keys(): 
                ix_ = np.nonzero(data['mov'][keep_ix] == key)[0]
                data_mov.append(data['xy'][keep_ix[ix_], variable])
            _, pv = scipy.stats.f_oneway(*data_mov)
            pv_all[variable] = pv 

        if np.any(pv_all < psig):
            mean_xy = np.mean(data['xy'][keep_ix, :], axis=0)
            cost = np.linalg.norm(data['xy'][keep_ix, :] - mean_xy[np.newaxis, :], axis=1)
            assert(len(cost) == len(keep_ix))

            N = len(cost)
            Nrm = np.max([1, int(np.floor(rm_perc*N))])

            arg_max_cost = np.argsort(cost)[::-1] ### In order of highest to lowest 
            ix_rm = arg_max_cost[:Nrm]

            ### Remove these indices from keep_ix: 
            print('removing')
            keep_ix = np.delete(keep_ix, ix_rm)
    
    print('pv_all')
    print(str(pv_all))
    #### When broken #####
    global_comm_indices2 = dict()
    cnt = 0; 
    for key in global_comm_indices.keys(): 

        #### Movement keeping #####
        mov_keep = np.nonzero(data['mov'][keep_ix] == key)[0]
        global_comm_indices2[key] = data['ixog'][keep_ix[mov_keep]]
        cnt += len(mov_keep)
    print('End Cnt: %d' %(cnt))
    return global_comm_indices2

def test_multi_match(): 
    X = []; Mov = []; 
    for i in range(10): 
        x = np.random.randn(50) + 0.1*i
        X.append(x)
        Mov.append(np.zeros((50, )) + i)
    X = np.hstack((X))
    Mov = np.hstack((Mov))

    f, ax = plt.subplots()
    for i in range(10):
        ix = np.nonzero(Mov == i)[0]
        util_fcns.draw_plot(i, X[ix], 'k', 'w', ax)
    ax.set_xlim([-1, 10])

    pv  = 0. 
    while pv < 0.05: 
        data_pd = pd.DataFrame(dict(Move=Mov, Metric=X))
        md = smf.mixedlm("Metric ~ 1", data_pd, groups=data_pd["Move"])
        mdf = md.fit()
        pv_all[variable] = mdf.pvalues['Group Var']
        print('pv %.3f' %(mdf.pvalues['Group Var']))

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
            spk_mn = np.mean(spks[ix_com_global, neuron_ix])

            util_fcns.draw_plot(x, np.abs(mFR_shuffle[mov] ), 'gray', np.array([0., 0., 0., 0.]), ax)
            ax.plot(x, np.abs(mFR[mov] ), '.', color=colrgb, markersize=20)
            ax.hlines(spk_mn, xlim[0], xlim[-1], color='gray',
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
def perc_neuron_command_move_sig(nshuffs = 1000, min_bin_indices = 0, keep_bin_spk_zsc = False):
    """
    Args:
        nshuffs (int, optional): Description
        min_bin_indices (int, optional): number of bins to remove from beginning AND end of the trial 
    
    Update: 3/31/21 --> now want to use the "pooling" method to assess if commands / neurons are sig. 
        -- report population significance; 
        -- 
    """

    ### Open mag boundaries 
    mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))

    ### Dictionaries to hold the pvalues and effect sizes of the signfiicant and non-sig 
    perc_sig = {}; 
    perc_sig_vect = {}; 

    dtype_su = [('pv', float), ('abs_diff_fr', float), ('frac_diff_fr', float), ('glob_fr', float)]
    dtype_pop = [('pv', float), ('norm_diff_fr', float), ('frac_norm_diff_fr', float)]

    niter2match = {}

    pooled_stats = {}
    
    for ia, animal in enumerate(['grom', 'jeev']): # add homer later; 
        
        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):
            
            ### Setup the dictionaries for this animal and date
            perc_sig[animal, day_ix] = {}
            perc_sig_vect[animal, day_ix] = {}
            niter2match[animal, day_ix] = {}
                        
            pooled_stats[animal, day_ix] = []

            ### Pull data ### 
            spks, push, tsk, trg, bin_num, rev_bin_num, move, dat = util_fcns.get_data_from_shuff(animal, day_ix, keep_bin_spk_zsc=keep_bin_spk_zsc)

            ### Convert spike count (e.g. 2 spks/bin to rate: 2 spks/bin *1 bin/.1 sec = 20 Hz)
            ### For homer spks count should be in z-scored values for bin
            if animal == 'home' and keep_bin_spk_zsc:
                pass
                ### dont multipy by 10 if youve been zscored
            else:
                spks = spks * 10
    
            nneur = spks.shape[1]
            command_bins = util_fcns.commands2bins([push], mag_boundaries, animal, day_ix, 
                                       vel_ix=[3, 5])[0]

            #### Add this for easier analysis later 
            perc_sig[animal, day_ix, 'nneur'] = nneur
            pooled_stats[animal, day_ix, 'nneur'] = nneur
            pooled_stats[animal, day_ix, 'nshuffs'] = nshuffs

            ### For each command get: ###
            mag_cnt = 0
            for mag in range(4):
                
                for ang in range(8): 
            
                    #### Common indices 
                    #### Get the indices for command ####
                    ix_com = return_command_indices(bin_num, rev_bin_num, push, mag_boundaries, mag=mag, ang=ang,
                                           animal=animal, day_ix=day_ix, min_bin_num=min_bin_indices,
                                           min_rev_bin_num=min_bin_indices)
                    mag_cnt += len(ix_com)

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
                            
                            ### Add mag/ang/mov to this so later can pool over commands / conditions etc ####
                            pooled_stats[animal, day_ix].append([dmean_FR, dmFR_shuffle, mag, ang, mov, global_mean_FR])

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
                            
                            ### Make sure this is the same thign ###
                            assert(dist == np.linalg.norm(dmean_FR))

                            ### Fraction difference; 
                            perc_sig_vect[animal, day_ix][mag, ang, mov] = np.array((pv, dist/nneur, dist/np.linalg.norm(global_mean_FR)), dtype=dtype_pop)

                print('Mag Cnt, mag = %d, monk = %s, day = %d, cnt = %d' %(mag, animal, day_ix, mag_cnt))
                mag_cnt = 0
                                  
    return perc_sig, perc_sig_vect, niter2match, pooled_stats

def print_pooled_stats_fig3(pooled_stats, nshuffs = 1000):

    for i_a, animal in enumerate(['grom', 'jeev']):
        NCM = []
        CM = []
        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):

            nneur = float(len(pooled_stats[animal, day_ix][0][0]))

            ncm_diff = np.mean([np.mean(d[0]) for d in pooled_stats[animal, day_ix]])
            shuff_diffs = np.mean(np.vstack(([np.mean(d[1], axis=1) for d in pooled_stats[animal, day_ix]])), axis=0)
            NCM.append([ncm_diff, shuff_diffs])

            ix = np.nonzero(ncm_diff <= shuff_diffs)[0]
            pv_ncm = float(len(ix)) / float(len(shuff_diffs))
            print('pv_ncm = %.5f, mn_ncm = %.3f, mn_shuff = %.3f (95th = %.3f)' %(pv_ncm, ncm_diff, np.mean(shuff_diffs),
                np.percentile(shuff_diffs, 95)))

            ### Population level 
            cm_diff = np.mean([np.linalg.norm(d[0])/nneur for d in pooled_stats[animal, day_ix]])
            cm_shuff_diffs = np.vstack(([np.linalg.norm(d[1], axis=1)/nneur for d in pooled_stats[animal, day_ix]]))
            assert(cm_shuff_diffs.shape[1] == nshuffs)
            cm_shuff_diffs = np.mean(cm_shuff_diffs, axis=0)
            CM.append([cm_diff, cm_shuff_diffs])
            
            cm_ix = np.nonzero(cm_diff <= cm_shuff_diffs)[0]
            pv_cm = float(len(cm_ix)) / float(len(cm_shuff_diffs))
            print('pv_cm = %.5f, mn_cm = %.3f, mn_shuff = %.3f (95th = %.3f)' %(pv_cm, cm_diff, np.mean(cm_shuff_diffs),
                np.percentile(cm_shuff_diffs, 95)))

        ### Pooled pooled; 
        ncm_pool = np.mean([d[0] for d in NCM])
        ncm_shuff_pool = np.vstack(([d[1] for d in NCM]))
        assert(ncm_shuff_pool.shape[1] == nshuffs)
        ix = np.nonzero(ncm_pool <= np.mean(ncm_shuff_pool, axis=0))[0]
        print('POOL %s, pv_ncm = %.5f, mn_ncm = %.3f, mn_shuff = %.3f (%.3f)' %(animal, float(len(ix))/1000., 
            ncm_pool, np.mean(np.mean(ncm_shuff_pool, axis=0)), np.percentile(np.mean(ncm_shuff_pool, axis=0), 95)))

        ### Pooled pooled; 
        cm_pool = np.mean([d[0] for d in CM])
        cm_shuff_pool = np.vstack(([d[1] for d in CM]))
        assert(cm_shuff_pool.shape[1] == 1000)
        ix = np.nonzero(cm_pool <= np.mean(cm_shuff_pool, axis=0))[0]
        print('POOL %s, pv_cm = %.5f, mn_ncm = %.3f, mn_shuff = %.3f (%.3f)' %(animal, float(len(ix))/1000., 
            cm_pool, np.mean(np.mean(cm_shuff_pool, axis=0)), np.percentile(np.mean(cm_shuff_pool, axis=0), 95)))
 
def plot_pooled_stats_fig3_science_compression(pooled_stats):
    '''
    goal: 
        1. fraction of command/conditions sig. diff (population)
        2. fraction of commands           sig. diff (population, pooled over conditions)
        3. fraction of neurons            sig. diff (single neurons, pooled over command/conditions)
        4. fraction distance from condition-pooled (population)
    '''

    #### Each plot ####
    f_fracCC, ax_fracCC = plt.subplots(figsize=(2, 3))
    f_fracCom, ax_fracCom = plt.subplots(figsize=(2, 3))
    f_fracN, ax_fracN = plt.subplots(figsize=(2, 3))
    f_fracdist, ax_fracdist = plt.subplots(figsize=(2, 3))
    f_fracdist_sig, ax_fracdist_sig = plt.subplots(figsize=(2, 3))

    ylabels = dict()
    ylabels['fracCC'] = 'frac. (command,condition) \nw. sig. deviations'
    ylabels['fracCom']= 'frac. (command) \nw. sig. deviations'
    ylabels['fracN']  = 'frac. (neuron) w. sig. deviations \nfor sig. (command, conditions)'
    ylabels['fracdist'] = ''
    ylabels['fracdist_shuff']  = 'norm. pop. dist'
    ylabels['fracdist_sig'] = ''
    ylabels['fracdist_shuff_sig']  = 'norm. pop. dist for sig. (command,condition)'

    for ia, animal in enumerate(['grom', 'jeev']):
        bar_dict = dict(fracCC=[], fracCom=[], fracN=[], fracdist=[], fracdist_shuff=[], fracdist_sig=[], fracdist_shuff_sig=[])
        stats_dict = dict(fracCC=[], fracCom=[], fracN=[], fracdist_sig=[], fracdist_shuff_sig=[])

        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):

            Nneur = pooled_stats[animal, day_ix, 'nneur']
            Nshuffs = pooled_stats[animal, day_ix, 'nshuffs']

            ########## Frac dist effect size plot ######################
            pop_stats = pooled_stats[animal, day_ix] ### dmean_FR, dmFR_shuffle, mag, ang, mov, global_mean_FR
            frac_pop_dist = []; 
            frac_pop_dist_shuff = []; 

            frac_pop_dist_sig = []; 
            frac_pop_dist_shuff_sig = []; 

            #### For each mag/ang/mov #####
            for _, val in enumerate(pop_stats):
                dFR, dFR_shuff, _, _, _, gFR = val
                frac_pop_dist.append(np.linalg.norm(dFR)/np.linalg.norm(gFR))
                
                assert(dFR_shuff.shape[0] == Nshuffs)
                assert(dFR_shuff.shape[1] == Nneur)
                
                frac_pop_dist_shuff.append(np.percentile(np.linalg.norm(dFR_shuff, axis=1)/np.linalg.norm(gFR), 50))

                ### Is this a sig commands/cond? 
                n_lte = len(np.nonzero(np.linalg.norm(dFR_shuff, axis=1) >= np.linalg.norm(dFR))[0])
                pv_cc = float(n_lte) / float(Nshuffs)

                #### If yes, add it to the plot 
                if pv_cc < 0.05:
                    frac_pop_dist_sig.append(np.linalg.norm(dFR)/np.linalg.norm(gFR))
                    frac_pop_dist_shuff_sig.append(np.percentile(np.linalg.norm(dFR_shuff, axis=1)/np.linalg.norm(gFR), 50))
            
            bar_dict['fracdist'].append(np.mean(frac_pop_dist))
            ax_fracdist.plot(ia, np.mean(frac_pop_dist), 'k.')
            
            bar_dict['fracdist_shuff'].append(np.mean(frac_pop_dist_shuff))
            ax_fracdist.plot(ia+0.4, np.mean(frac_pop_dist_shuff), '.', color='gray')

            ax_fracdist.plot([ia, ia+0.4], [np.mean(frac_pop_dist), np.mean(frac_pop_dist_shuff)], 'k-', linewidth=0.5)

            bar_dict['fracdist_sig'].append(np.mean(frac_pop_dist_sig))
            ax_fracdist_sig.plot(ia, np.mean(frac_pop_dist_sig), 'k.', markersize=12)
            
            bar_dict['fracdist_shuff_sig'].append(np.mean(frac_pop_dist_shuff_sig))
            ax_fracdist_sig.plot([ia, ia+0.4], [np.mean(frac_pop_dist_sig), np.mean(frac_pop_dist_shuff_sig)], 'k-', linewidth=1.)
            ax_fracdist_sig.plot(ia+0.4, np.mean(frac_pop_dist_shuff_sig), '.', markerfacecolor='white', 
                markeredgewidth=1.0, markeredgecolor='k', markersize=12)


            ########## # of Cond/com sig. and # of com sig. ###########
            nCC = 0
            nCC_sig = 0

            nCom = 0
            nCom_sig = 0 

            nNeur = 0
            nNeur_sig = 0

            #### starting with command/conditions signifiant 
            stats = pooled_stats[animal, day_ix] ### dmean_FR, dmFR_shuffle, mag, ang, mov

            command_sig = dict(); 
            commands_already = []
            

            neuron_sig = dict()
            for n in range(Nneur): neuron_sig[n] = dict(vals=[], shuffs=[])

            NComCond = len(stats)
            for i_nComCond in range(NComCond): 

                ### unpack 
                dmean_FR, dmFR_shuffle, mag, ang, mov, _ = stats[i_nComCond]

                assert(Nshuffs == dmFR_shuffle.shape[0])
                assert(len(dmean_FR) == dmFR_shuffle.shape[1] == Nneur)

                ### pv for command/conditions ### 
                n_lte = len(np.nonzero(np.linalg.norm(dmFR_shuffle, axis=1) >= np.linalg.norm(dmean_FR))[0])
                pv_cc = float(n_lte) / float(Nshuffs)

                if pv_cc < 0.05:
                    nCC_sig += 1
                nCC+= 1

                #### Add to command sig 
                if [mag, ang] not in commands_already:
                    command_sig[tuple([mag, ang])] = dict(vals=[], shuffs=[])
                    commands_already.append([mag, ang])

                command_sig[tuple([mag, ang])]['vals'].append(np.linalg.norm(dmean_FR))
                command_sig[tuple([mag, ang])]['shuffs'].append(np.linalg.norm(dmFR_shuffle, axis=1))
                    
                ##### Add to neurons 
                if pv_cc < 0.05:
                    for nneur in range(Nneur):
                        neuron_sig[nneur]['vals'].append(np.abs(dmean_FR[nneur]))
                        neuron_sig[nneur]['shuffs'].append(np.abs(dmFR_shuffle[:, nneur]))
                
            ########## Now we can plot frac of command/conditions sig #############
            ax_fracCC.plot(ia, float(nCC_sig)/float(nCC), 'k.')
            bar_dict['fracCC'].append(float(nCC_sig)/float(nCC))

            ########## Frac commands with sig deviations #########################
            for ic, com in enumerate(commands_already): 
                vals = command_sig[tuple(com)]['vals']
                shuf = command_sig[tuple(com)]['shuffs']
                
                assert(len(vals) == len(shuf))
                ### Average over ###

                shuf = np.vstack((shuf))
                assert(shuf.shape[0] == len(vals))
                assert(shuf.shape[1] == Nshuffs)

                shuf_mn = np.mean(shuf, axis=0)
                val_mn = np.mean(vals)

                nshuff_gte = len(np.nonzero(shuf_mn >= val_mn)[0])
                pv_com = float(nshuff_gte)/float(Nshuffs)

                if pv_com < 0.05:
                    nCom_sig += 1
                nCom += 1

            ########## Plot # sig commands over conditions ##################
            ax_fracCom.plot(ia, float(nCom_sig)/float(nCom), 'k.')
            bar_dict['fracCom'].append(float(nCom_sig)/float(nCom))

            ########## Number of neurons ####################################
            for i_n in neuron_sig.keys():
                vals = neuron_sig[i_n]['vals']
                shuf = neuron_sig[i_n]['shuffs']
                assert(len(vals) == len(shuf))

                shuf = np.vstack((shuf))
                assert(shuf.shape[0] == len(vals))
                assert(shuf.shape[1] == Nshuffs)

                ##### use p-value correction method #######
                # pvs = []
                # ncc_ = len(vals)
                # for cc in range(ncc_):
                #     n = len(np.nonzero(shuf[cc, :] >= vals[cc])[0])
                #     pvs.append(float(n)/float(Nshuffs))

                # pv_eff = 0.05/ncc_
                # if np.any(np.array(pvs) < pv_eff):
                #     nNeur_sig += 1
                # nNeur += 1

                ####### use pooling method #############
                shuf_mn = np.mean(shuf, axis=0)
                val_mn = np.mean(vals)

                nshuff_gte = len(np.nonzero(shuf_mn >= val_mn)[0])
                pv_neur = float(nshuff_gte)/float(Nshuffs)

                if pv_neur < 0.05:
                    nNeur_sig += 1
                nNeur += 1

            ############# Plot sig neurons #################################
            ax_fracN.plot(ia, float(nNeur_sig)/float(nNeur), 'k.')
            bar_dict['fracN'].append(float(nNeur_sig)/float(nNeur))


        #### Plot bar plots 
        for _, (key, ax, wid, offs, alpha) in enumerate(zip(
            ['fracCC', 'fracCom', 'fracN', 'fracdist', 'fracdist_shuff', 'fracdist_sig', 'fracdist_shuff_sig'], 
            [ax_fracCC, ax_fracCom, ax_fracN, ax_fracdist, ax_fracdist, ax_fracdist_sig, ax_fracdist_sig], 
            [.8, .8, .8, .4, .4, .4, .4], 
            [0, 0, 0, 0, .4, 0, .4],
            [.2, .2, .2, .2, 0., .2, 0.])):
            if alpha == 0.:
                ax.bar(ia+offs, np.mean(bar_dict[key]), width=wid, color='w', edgecolor='k',
                    linewidth=.5)
            else:
                ax.bar(ia+offs, np.mean(bar_dict[key]), width=wid, alpha=alpha, color='k')
            ax.set_ylabel(ylabels[key], fontsize=8)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['G', 'J'])

            if key == 'fracdist_shuff_sig':
                ax.set_xlim([-.3, 1.7])
            else:
                ax.set_xlim([-1, 2])
            if 'dist' not in key:
                ax.set_yticks([0., 0.2, .4, .6, .8, 1.0])
                ax.set_ylim([0., 1.05])
            
            #### Remove spines 
            for side in ['right', 'top']:
                spine_side = ax.spines[side]
                spine_side.set_visible(False)

        
    for _, (f, yl) in enumerate(zip([f_fracCC, f_fracCom, f_fracN, f_fracdist, f_fracdist_sig],
        ['fracCC', 'fracCom', 'fracN_sig', 'fracdist', 'fracdist_sig'])):
        
        f.tight_layout()
        util_fcns.savefig(f, yl)


def plot_su_pop_stats(perc_sig, perc_sig_vect, sig_move_diffs = None, 
    plot_sig_mov_comm_grid = False, min_fr_frac_neur_diff = 0.5, neur_ix = 36):

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
    fveff2, axveff2 = plt.subplots(figsize=(3, 3)) ### Effect size sig. command/mov
    fveff_bsig, axveff_bsig = plt.subplots(figsize=(3, 3)) ###

    ### Percent sig of sig diff mov-commands  Vect; 
    fsig_pop, axsig_pop = plt.subplots(figsize=(2, 3))

    #### For each animal 
    for ia, animal in enumerate(['grom', 'jeev', 'home']):

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

                        ### PLot the dot if relevant ###
                        if animal == 'grom' and day_ix == 0 and k[0] == 0 and k[1] == 7 and j == neur_ix: 
                            color = np.array(analysis_config.pref_colors_rgb[ int(k[2]) % 10])
                            if k[2] >= 10.:
                                color[-1] = 0.5
                            else:
                                color[-1] = 1.0

                            color_rgb = util_fcns.rgba2rgb(color)
                            print('mov %.2f, color w alpha %s, color wo alpha %s '%(k[2], color_rgb, color))
                            x_ = [day_ix + 10*ia - 1, day_ix + 10*ia - 0.5]
                            y_ = [float(varr['abs_diff_fr'][j]), float(varr['abs_diff_fr'][j])]
                            axeff.plot(x_, y_, '-', color=color_rgb)

                        if float(varr['glob_fr'][j]) >= min_fr_frac_neur_diff:
                            NCM_sig_frac_diff.append(float(varr['frac_diff_fr'][j]))

                            if animal == 'grom' and day_ix == 0 and k[0] == 0 and k[1] == 7 and j == neur_ix: 
                                x_ = [day_ix + 10*ia - 1, day_ix + 10*ia -0.5]
                                y_ = [float(varr['frac_diff_fr'][j]), float(varr['frac_diff_fr'][j])]
                                axeff2.plot(x_, y_, '-', color=color_rgb)
                            

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
                    else:
                        NCM_tot_beh_sig = np.nan
            
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
            util_fcns.draw_plot(day_ix + 10*ia, NCM_sig_hz_diff, 'k', np.array([1., 1., 1., 0.]), axeff)
            util_fcns.draw_plot(day_ix + 10*ia, NCM_sig_frac_diff, 'k', np.array([1., 1., 1., 0.]), axeff2)

            util_fcns.draw_plot(day_ix + 10*ia, NCM_behsig_hz_diff, 'k', np.array([1., 1., 1., 0.]), axeff_bsig)
            util_fcns.draw_plot(day_ix + 10*ia, NCM_behsig_frac_diff, 'k', np.array([1., 1., 1., 0.]), axeff_bsig_frac)

            
            ##### Vector ########
            vect_sig = 0 
            vect_tot = 0 

            vect_sig_beh_sig = 0
            vect_tot_beh_sig = 0
                        
            ####################
            commands_w_sig = np.zeros((4, 8)) ### How many are sig? 
            command_cnt = np.zeros((4, 8)) ### How many movements count? 
            sig_hz_diff_vect = []
            sig_hz_diff_vect_frac = []
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
                    sig_hz_diff_vect_frac.append(float(varr['frac_norm_diff_fr']))

                    if sig_move_diffs is not None: 
                        if list(k) in sig_move_diffs[animal, day_ix]:
                            vect_sig_beh_sig += 1
                            sig_hz_diff_vect_bsig.append(float(varr['norm_diff_fr']))
                    else:
                        vect_sig_beh_sig = np.nan

                    ##### Plot the dots on the population ######
                    if animal == 'grom' and day_ix == 0 and k[0] == 0 and k[1] == 7: 
                        color = np.array(analysis_config.pref_colors_rgb[ int(k[2]) % 10])
                        if k[2] >= 10:
                            color[-1] = 0.5
                        else:
                            color[-1] = 1.0

                        color_rgb = util_fcns.rgba2rgb(color)
                        x_ = [ia*10 + day_ix - 1, ia*10 + day_ix -0.5]
                        y_1 = [float(varr['norm_diff_fr']), float(varr['norm_diff_fr'])]
                        y_2 = [float(varr['frac_norm_diff_fr']), float(varr['frac_norm_diff_fr'])]

                        axveff.plot(x_, y_1,  '-', color=color_rgb)
                        axveff2.plot(x_, y_2, '-', color=color_rgb)
                else:
                    if plot_sig_mov_comm_grid:
                        axsig[k[0]].plot(k[2], k[1], '.', color='k')
                
                command_cnt[k[0], k[1]] += 1
                vect_tot += 1

                if sig_move_diffs is not None: 
                    if list(k) in sig_move_diffs[animal, day_ix]:
                        vect_tot_beh_sig += 1
                else:
                    vect_tot_beh_sig = np.nan
            
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
            util_fcns.draw_plot(ia*10 + day_ix, sig_hz_diff_vect, 'k', np.array([1., 1., 1., 0.]), axveff)
            util_fcns.draw_plot(ia*10 + day_ix, sig_hz_diff_vect_frac, 'k', np.array([1., 1., 1., 0.]), axveff2)
            util_fcns.draw_plot(ia*10 + day_ix, sig_hz_diff_vect_bsig, 'k', np.array([1., 1., 1., 0.]), axveff_bsig)
            
            ### Plot the perc sig CM with sig diff from glboal 
            tmp_frc = float(vect_sig_beh_sig) / float(vect_tot_beh_sig)
            axsig_pop.plot(ia, tmp_frc, 'k.')
            bar_dict['percCMsig_behsig'].append(tmp_frc)
            print('A %s, D %d, perc sig. tot %d/%d  =%.2f' %(animal, day_ix, vect_sig, vect_tot, float(vect_sig)/float(vect_tot)))
            try:
                print('A %s, D %d, perc sig. of beh sig %d/%d  =%.2f' %(animal, day_ix, vect_sig_beh_sig, vect_tot_beh_sig, tmp_frc))
            except:
                pass

        ####### Single neuron summary for the day #####
        ax.bar(ia, np.mean(bar_dict['percNCMsig']), width=.8,alpha=0.2, color='k')
        ax2.bar(ia, np.mean(bar_dict['percNgte1CMsig']), width=.8, alpha=0.2, color='k')
        axsig_su.bar(ia, np.mean(bar_dict['percNCMsig_behsig']), width=.8,alpha=0.2, color='k')

        ####### Vector  neuron summary for the day #####
        axv.bar(ia, np.mean(bar_dict['percCMsig']), width=.8,alpha=0.2, color='k')
        axv2.bar(ia, np.mean(bar_dict['percCgte1Msig']), width=.8, alpha=0.2, color='k')
        axsig_pop.bar(ia, np.mean(bar_dict['percCMsig_behsig']), width=.8, alpha=0.2, color='k')

    for axi in [ax, ax2, axv, axv2, axsig_su, axsig_pop]:
        axi.set_xticks([0, 1, 2])
        axi.set_xticklabels(['G', 'J', 'H'])
        axi.set_xlim([-1, 3])

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
    try:
        fsig_su.tight_layout()
        fsig_pop.tight_layout()
    except:
        pass
    

    util_fcns.savefig(f, 'frac_neur_comm_mov_sig_diff_from_global')
    util_fcns.savefig(f2, 'frac_with_gte_0_sig_diff_from_global')
    
    util_fcns.savefig(fsig_su, 'frac_NCM_sig_diff_from_global_of_beh_diff')
    util_fcns.savefig(fsig_pop, 'frac_CM_sig_diff_from_global_of_beh_diff')
    
    util_fcns.savefig(fv, 'frac_comm_mov_sig_diff_from_global')
    util_fcns.savefig(fv2, 'frac_comm_with_gte_0_sig_diff_mov_from_global')

    ###### Effect size plots #######
    for ax_ in [axeff, axeff2, axveff, axveff2, axveff_bsig, axeff_bsig, axeff_bsig_frac]:
        ax_.set_xlim([-1, 25])
    axeff.set_ylim([0, 10.])
    axeff.set_xlim([-1.5, 25.])
    axeff_bsig.set_ylim([0., 10.])

    axeff2.set_ylim([0, 3.0])
    axeff2.set_xlim([-1.5, 25])
    axeff_bsig_frac.set_ylim([0., 3.])

    axveff.set_ylim([0, .8])
    axveff.set_xlim([-1.5, 25])
    axveff2.set_ylim([0, .71])
    axveff2.set_xlim([-1.5, 25])
    axveff_bsig.set_ylim([0, .8])

    axeff.set_ylabel('Activity Diff from Command  Act. Mean (Hz)')
    axeff.set_title('Sig. Diff Neurons/Command/Move')

    axeff2.set_ylabel('Frac. Diff Activity from Command Act. Mean')
    axeff2.set_title('Sig. Diff Neurons/Command/Move')

    axveff.set_ylabel('Pop. Activity Diff from Command Act. Mean')
    axveff2.set_ylabel('Frac. Diff Pop. Act. from Command Act. Mean')
    #axveff.set_title('Sig. Diff. Command/Moves')
    axveff_bsig.set_ylabel('Pop. Activity Diff for Beh Sig. ')

    axeff_bsig.set_ylabel('beh sig: Act. Diff from Command Act. Mean (Hz)')
    axeff_bsig_frac.set_ylabel('beh sig: Frac. Diff from Command Act. Mean')
    #feff_bsig.tight_layout()
    #feff_bsig_frac.tight_layout()

    ######  #######
    lab = ['hz_diff', 'frac_diff', 'hz_diff_bsig', 'frac_diff_bsig', 'vect_diff', 'frac_vect_diff', 'vect_diff_bsig']
    F=[feff, feff2, feff_bsig, feff_bsig_frac, fveff, fveff2, fveff_bsig]
    for i_, (f_, l_) in enumerate(zip(F, lab)):
        pass
        #util_fcns.savefig(f_, l_)

def plot_perc_command_beh_sig_diff_than_global(nshuffs=1000, min_bin_indices=0, save=True,
    use_saved_move_command_sig = False):
    
    if use_saved_move_command_sig:
        move_command_sig_dict = pickle.load(open(analysis_config.config['grom_pref'] + 'sig_move_comm_%dshuffs.pkl' %nshuffs, 'rb'))
        plot_only_bars = True
        pooled_stats = move_command_sig_dict['pooled_stats']
    else:
        move_command_sig_dict = None
        plot_only_bars = False
        pooled_stats = {}
    pooled_stats_pool = {}
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

        move_command_sig[animal] = []
        move_command_sig[animal, 'osa'] = []
        pooled_stats_pool[animal] = []
        pooled_stats_pool[animal, 'osa'] = []

        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):

            print('Starting animal %s, Day %d' %(animal, day_ix))

            move_command_sig[animal, day_ix] = []
            move_command_sig[animal, day_ix, 'osa'] = []
            pooled_stats[animal, day_ix] = []
            pooled_stats[animal, day_ix, 'osa'] = []

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

            special_dots = []
            special_dots_osa = []

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
                                

                                if plot_only_bars:
                                    total_cm += 1
                                    total_cm_osa += 1
                                    if [mag, ang, mov] in move_command_sig_dict[animal, day_ix]:
                                        total_sig_com += 1
                                    if [mag, ang, mov] in move_command_sig_dict[animal, day_ix, 'osa']:
                                        total_sig_cm_osa += 1

                                else:
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

                                        if animal == 'grom' and day_ix == 0 and mag == 0 and ang == 7: 
                                            special_dots.append([dPSTH, util_fcns.get_color(mov)])

                                    total_cm += 1
                                    pooled_stats[animal, day_ix].append([dPSTH, beh_shuffle])

                                    #### Test for sig for one-step-ahead (osa) ####
                                    ix_tmp_osa = np.nonzero(beh_shuffle_osa >= dPSTH_osa)[0]
                                    pv_osa = float(len(ix_tmp_osa)) / float(nshuffs)
                                    if pv_osa < 0.05: 
                                        total_sig_cm_osa += 1
                                        traj_diff_one_step.append(dPSTH_osa)
                                        move_command_sig[animal, day_ix, 'osa'].append([mag, ang, mov])
                                        if animal == 'grom' and day_ix == 0 and mag == 0 and ang == 7: 
                                            special_dots_osa.append([dPSTH_osa, util_fcns.get_color(mov)])
                                    total_cm_osa += 1
                                    pooled_stats[animal, day_ix, 'osa'].append([dPSTH_osa, beh_shuffle_osa])
            
            ### Compute percent sig: 
            perc_sig[animal].append(float(total_sig_com)/float(total_cm))
            ax.plot(i_a, float(total_sig_com)/float(total_cm), 'k.')

            #### One step ahead 
            perc_sig_one_step[animal].append(float(total_sig_cm_osa)/float(total_cm_osa))
            ax_osa.plot(i_a, float(total_sig_cm_osa)/float(total_cm_osa), 'k.')
        
            if plot_only_bars:
                pass
            else:
                #### Plot the distribution of signficiant differences 
                util_fcns.draw_plot(i_a*10 + day_ix, traj_diff, 'k', np.array([1., 1., 1., 0.]), axe)
                util_fcns.draw_plot(i_a*10 + day_ix, traj_diff_one_step, 'k', np.array([1., 1., 1., 0.]), axe_osa)

                if len(special_dots) > 0: 
                    for _, (d, col) in enumerate(special_dots):
                        x_ = [i_a*10 + day_ix - 1., i_a*10 + day_ix - 0.5]
                        axe.plot(x_, [d, d], '-', color=col)
                
                if len(special_dots_osa) > 0: 
                    for _, (d, col) in enumerate(special_dots_osa):
                        x_ = [i_a*10 + day_ix - 1., i_a*10 + day_ix -0.5]
                        axe_osa.plot(x_, [d, d], '-', color=col)

            #### Do the stats
            pv_day, dpsth, shuff = print_pv_from_pooled_stats(pooled_stats[animal, day_ix])
            print('Pv = %.5f, dpsth = %.3f, mnshuff = %.3f' %(pv_day, dpsth, np.mean(shuff)))
            pooled_stats_pool[animal].append([dpsth, shuff])

            pv_day, dpsth, shuff = print_pv_from_pooled_stats(pooled_stats[animal, day_ix, 'osa'])
            print('Pv osa = %.5f, dpsth = %.3f, mnshuff = %.3f' %(pv_day, dpsth, np.mean(shuff)))
            pooled_stats_pool[animal, 'osa'].append([dpsth, shuff])

        ### Plot the bar; 
        ax.bar(i_a, np.mean(perc_sig[animal]), width=.8, alpha=0.2, color='k')
        ax_osa.bar(i_a, np.mean(perc_sig_one_step[animal]), width=.8, alpha=0.2, color='k')

        ### Pooled stats over days; 
        print('POOLED: ')
        pv_animal, dp, sh = print_pv_from_pooled_stats(pooled_stats_pool[animal])
        print('PV %s: %.5f, dpsth = %.3f, mnshuff = %.3f (%.3f)' %(animal, pv_animal, dp, np.mean(sh),
            np.percentile(sh, 95)))

        pv_animal_osa, dp, sh = print_pv_from_pooled_stats(pooled_stats_pool[animal, 'osa'])
        print('PV OSA %s: %.5f, dpsth = %.3f, mnshuff = %.3f (%.3f)' %(animal, pv_animal_osa, dp, np.mean(sh),
            np.percentile(sh, 95)))
        
    for axi in [ax, ax_osa]:
        axi.set_xticks([0, 1])
        axi.set_xticklabels(['G', 'J'])
        axi.set_ylim([0., 1.])
        axi.set_yticks([0., .2, .4, .6, .8, 1.0])
        axi.set_yticklabels([0., .2, .4, .6, .8, 1.0])

    ax.set_ylabel('Frac. Move-Specific Commands \nwith Sig. Diff. Traj', fontsize=4)
    ax_osa.set_ylabel('Frac. Move-Specific Commands \nwith Sig. Diff. Next Command', fontsize=4)
    f.tight_layout()
    f_osa.tight_layout()
    if save:
        util_fcns.savefig(f, 'Beh_diff_perc_sig')
        util_fcns.savefig(f_osa, 'Beh_diff_perc_sig_one_step_ahead')


    if plot_only_bars:
        pass
    else:

        for axi in [axe, axe_osa]:
            axi.set_xlim([-1.5, 14])
        axe.set_ylabel('Command Traj Diff for Sig.\nDiff. Move-Specific Commands', fontsize=4)
        axe.set_ylim([0, 8])
        axe_osa.set_ylabel('Next Command Diff for Sig. \nDiff. Move-Specific Commands', fontsize=4)
        axe_osa.set_ylim([0, 2.2])

        fe.tight_layout()
        fe_osa.tight_layout()

        if save:
            util_fcns.savefig(fe, 'Traj_diff_sig_cm')
            util_fcns.savefig(fe_osa, 'Next_command_diff_sig_cm')

        move_command_sig['pooled_stats'] = pooled_stats
        move_command_sig['nshuffs'] = nshuffs

        ### Save signifciant move/commands 
        pickle.dump(move_command_sig, open(analysis_config.config['grom_pref'] + 'sig_move_comm_%dshuffs.pkl'%nshuffs, 'wb'))

def print_pv_from_pooled_stats(stats):
    nshuffs = len(stats[0][1])
    dpsth = np.mean(np.array([d[0] for d in stats]))
    shuff = np.vstack(([d[1] for d in stats]))
    assert(shuff.shape[1] == nshuffs)
    shuff = np.mean(shuff, axis=0)
    assert(len(shuff) == nshuffs)
    pv_day = float(len(np.nonzero(shuff >= dpsth)[0]))/float(len(shuff))
    return pv_day, dpsth, shuff

def plot_distribution_of_nmov_per_command(min_obs = 15): 
    """
    Plot distribution of number of movements per command as boxplot
    """
    mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))
    
    ######## Percent sig #############
    #f, ax = plt.subplots(figsize=(3,3 ))
    f, ax = plt.subplots(figsize=(6, 6 ))

    ############ Loop ##############
    for i_a, animal in enumerate(['grom', 'jeev', 'home']):

        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):

            ###### Extract data #######
            _, push, tsk, trg, bin_num, rev_bin_num, move, dat = util_fcns.get_data_from_shuff(animal, day_ix)
            
            ###### Get magnitude difference s######
            command_bins = util_fcns.commands2bins([push], mag_boundaries, animal, day_ix, 
                                               vel_ix=[3, 5])[0]
            mov_per_com = []
            mov_per_com_array = np.zeros((4, 8))
            mag_cnt = 0
            for mag in range(4):

                for ang in range(8): 
        
                    ### Return indices for the command ### 
                    ix_com = return_command_indices(bin_num, rev_bin_num, push, mag_boundaries, animal=animal, 
                                            day_ix=day_ix, mag=mag, ang=ang, min_bin_num=0,
                                            min_rev_bin_num=0)

                    ### For all movements --> figure otu which ones to keep in the global distribution ###
                    movements = 0
                    #print('Mag %d, Ang %d, # = %d' %(mag, ang, len(ix_com)))
                    mag_cnt += len(ix_com)

                    for mov in np.unique(move[ix_com]):

                        ### Movement specific command indices 
                        ix_mc = np.nonzero(move[ix_com] == mov)[0]
                        
                        ### Which global indices used for command/movement 
                        ix_mc_all = ix_com[ix_mc] 

                        ### If enough of these then proceed; 
                        if len(ix_mc) >= min_obs:
                            movements += 1
                            mov_per_com_array[mag, ang] += 1

                    ### Add to list: 
                    if movements >= 2:
                        mov_per_com.append(movements)

                #####

                print('Animal %s, Day %d, MAG %d: %d '%(animal, day_ix, mag, mag_cnt))
                mag_cnt = 0

            ### Plot distribution ###
            util_fcns.draw_plot(i_a*10 + day_ix, mov_per_com, 'k', 'w', ax)

            f2,ax2 = plt.subplots()
            cax=ax2.pcolor(mov_per_com_array)
            f2.colorbar(cax, ax=ax2)
            ax2.set_title('%s, session %d' %(animal, day_ix))
    ax.set_ylabel('# move per command')
    ax.set_xlim([-1, 26])
    ax.set_xticks([])
    ax.set_xticklabels([])
    f.tight_layout()

    #util_fcns.savefig(f, 'fig2_numMov_perComm')


######### Behavior vs. neural correlations #########
def neuraldiff_vs_behaviordiff_corr_pairwise(min_bin_indices=0, nshuffs = 1, ncommands_psth = 5,
    min_commands = 15, min_move_per_command = 5): 

    ### Open mag boundaries 
    mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))

    fr, axr = plt.subplots(figsize = (2, 3))
    frd, axrd = plt.subplots(figsize = (4, 4))
    fs, axs = plt.subplots(figsize = (4, 4))
    fsd, axsd = plt.subplots(figsize = (4, 4))

    pooled_stats_all = {}
    comm_sig = {}

    for ia, animal in enumerate(['grom', 'jeev']):#, 'grom', 'jeev']):
        rv_agg = []

        ### For each pw diff add: (norm diff behav, norm diff pop neur, day, ang, mag)
        pooled_stats = []
        pooled_dtype = np.dtype([('dB', 'f8'), ('dN', 'f8'), ('day_ix', np.int8), ('ang', np.int8), ('mag', np.int8)])

        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):
            
            print('starting %s, day %d' %(animal, day_ix))
            ### Pull data ### 
            spks, push, tsk, trg, bin_num, rev_bin_num, move, dat = util_fcns.get_data_from_shuff(animal, day_ix)
            spks = spks * 10

            ### Do this step just to match the predicted plots you get later ###
            tm0, _ = generate_models.get_temp_spks_ix(dat['Data'])

            ### Get subsampled 
            spks = spks[tm0, :]
            push = push[tm0, :]
            move = move[tm0]
            bin_num = bin_num[tm0]
            rev_bin_num = rev_bin_num[tm0]

            nneur = spks.shape[1]
            command_bins = util_fcns.commands2bins([push], mag_boundaries, animal, day_ix, 
                                       vel_ix=[3, 5])[0]

            #### Testing plot ####
            #f2, ax2 = plt.subplots(figsize = (6, 5), ncols = 2)
            f, ax = plt.subplots(figsize = (6, 5))

            D = []
            D_shuff = {}
            for i in range(nshuffs):
                D_shuff[i] = []

            comm_sig[animal, day_ix] = []

            pairs_analyzed = {}
            plt_top = []

            var_neur = []
            diff_beh = []

            commands_sig = [0., 0.]

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

                    mov_cnt = 0
                    #### Go through the movements ####
                    for mov in np.unique(move[ix_com]):
                        
                        ### Movement specific command indices 
                        ix_mc = np.nonzero(move[ix_com] == mov)[0]
                        ix_mc_all = ix_com[ix_mc]
                        
                        ### If enough of these then proceed; 
                        if len(ix_mc) >= min_commands:    

                            global_comm_indices[mov] = ix_mc_all
                            ix_com_global.append(ix_mc_all)
                            mov_cnt += 1

                    if len(ix_com_global) > 0 and mov_cnt >= min_move_per_command:

                        ix_com_global = np.hstack((ix_com_global))
                        relevant_movs = np.array(global_comm_indices.keys())
                        shuffle_mean_FR = {}

                        dB_vs_dN_command = []

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

                                if np.logical_and(len(ix_ok1) >= min_commands, len(ix_ok2) >= min_commands):

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

                                    mov_PSTH1 = get_PSTH(bin_num, rev_bin_num, push, ix_mc_all[ix_ok1], num_bins=ncommands_psth, min_bin_set = 1)
                                    mov_PSTH2 = get_PSTH(bin_num, rev_bin_num, push, ix_mc_all2[ix_ok2], num_bins=ncommands_psth, min_bin_set = 1)

                                    if mov_PSTH1 is not None and mov_PSTH2 is not None:
                                        assert(mov_PSTH1.shape[0] == 2*ncommands_psth + 1)
                                        assert(mov_PSTH1.shape[1] == 2)
                                        Nmov1 = len(ix_ok1)
                                        Nmov2 = len(ix_ok2)

                                        #### Matched dN and dB; 
                                        dN = np.linalg.norm(mov_mean_FR1 -mov_mean_FR2)/nneur
                                        dB = np.linalg.norm(mov_PSTH1 - mov_PSTH2)

                                        if mag == 0 and ang == 7 and animal == 'grom':
                                            #### Pair 1 ### (1., 10.1), (10.1, 15.), 
                                            # if mov == 1. and mov2 == 10.1:
                                            #     plt_top.append([dB, dN, 'deeppink', 15, mov, len(ix_ok1), len(ix_mc_all), mov2, len(ix_ok2), len(ix_mc_all2)])
                                            # elif mov == 10.1 and mov2 == 15.:
                                            #     plt_top.append([dB, dN, 'limegreen', 15, mov, len(ix_ok1), len(ix_mc_all), mov2, len(ix_ok2), len(ix_mc_all2)])
                                            if mov == 1. and mov2 == 10.1:
                                                plt_top.append([dB, dN, 'deeppink', 15, mov, len(ix_ok1), len(ix_mc_all), mov2, len(ix_ok2), len(ix_mc_all2)])
                                            elif mov == 1. and mov2 == 3.:
                                                plt_top.append([dB, dN, 'limegreen', 15, mov, len(ix_ok1), len(ix_mc_all), mov2, len(ix_ok2), len(ix_mc_all2)])
                                            
                                            else: 
                                                plt_top.append([dB, dN, 'darkblue', 10, mov, len(ix_ok1), len(ix_mc_all), mov2, len(ix_ok2), len(ix_mc_all2)])
                                        else:
                                            rgb = util_fcns.rgba2rgb(np.array([0., 0., 0., .5]))
                                            #ax.plot(dB, dN, '.', color=rgb, markersize=5)
                                            #ax.plot(dB, dN, '.', color=analysis_config.pref_colors_rgb[int(mov)%10], markersize=10)
                                        D.append([dB, dN])
                                        dB_vs_dN_command.append([dB, dN])

                                        ### For each pw diff add: (norm diff behav, norm diff pop neur, day, ang, mag)
                                        pooled_stats.append(np.array((dB, dN, day_ix, ang, mag), dtype=pooled_dtype))

                                        diff_beh.append([dB, mag])
                                        var_neur.append(np.trace(np.cov(spks[ix_com_global, :].T)))

                                        ############################################################
                                        ########## Get the global taht matches the subsample #######
                                        ############################################################
                                        ###### Get shuffles for movement 1 / movement 2
                                        skip = False

                                        ### Shuffle takes from teh global distribution Nmov number of point adn saves 
                                        _, pv3 = scipy.stats.ttest_ind(push[np.ix_(ix_mc_all[ix_ok1], [3])], push[np.ix_(ix_mc_all2[ix_ok2], [3])])
                                        _, pv5 = scipy.stats.ttest_ind(push[np.ix_(ix_mc_all[ix_ok1], [5])], push[np.ix_(ix_mc_all2[ix_ok2], [5])])
                                        assert(pv3 > 0.05)
                                        assert(pv5 > 0.05)
                                        
                                        ### Ok now get a global distribution that match mov1 and mov2
                                        #globix1, niter1 = distribution_match_global_mov(push[np.ix_(ix_mc_all[ix_ok1],  [3, 5])], push[np.ix_(ix_com_global, [3, 5])])
                                        #globix2, niter2 = distribution_match_global_mov(push[np.ix_(ix_mc_all2[ix_ok2], [3, 5])], push[np.ix_(ix_com_global, [3, 5])])
                                        
                                        #Nglobal1 = len(globix1)
                                        #Nglobal2 = len(globix2)
                                        Nglobal = len(ix_com_global)

                                        shuffle_mean_FR[mov, mov2, 1] = []
                                        shuffle_mean_FR[mov, mov2, 2] = []

                                        for ishuff in range(nshuffs):

                                            complete = False
                                            cnt = 0
                                            while not complete: 
                                                #### Movement 1 / Movement 2 ####
                                                ix_shuff1 = np.random.permutation(Nglobal)[:Nmov1]
                                                ix_shuff2 = np.random.permutation(Nglobal)[:Nmov2]

                                                #### But now make sure these distributions match: 
                                                ix_ok1_1, ix_ok2_2, niter = distribution_match_mov_pairwise(push[np.ix_(ix_com_global[ix_shuff1], [3, 5])], 
                                                                     push[np.ix_(ix_com_global[ix_shuff2], [3, 5])])

                                                if len(ix_ok1_1) >= min_commands and len(ix_ok2_2) >= min_commands:
                                                    complete = True
                                                    shuff1 = np.mean(spks[ix_com_global[ix_shuff1[ix_ok1_1]], :], axis=0)
                                                    shuff2 = np.mean(spks[ix_com_global[ix_shuff2[ix_ok2_2]], :], axis=0)
                                             
                                                    ### difference in neural ####
                                                    shuff_dN = np.linalg.norm(shuff1 - shuff2)/nneur
                                                    
                                                    ### Add shuffle 
                                                    D_shuff[ishuff].append([dB, shuff_dN])
                                                else:
                                                    cnt +=1
                                                    if cnt > 50:
                                                        complete = True
                                                        print('Skipping')

                                            #if ishuff == 0:
                                            #    ax.plot(dB, shuff_dN, 'r.')
                        #### Get dB vs. dN --> regression 
                        dB_vs_dN_command = np.vstack((dB_vs_dN_command))
                        _, _, rv_command, pv, _ = scipy.stats.linregress(dB_vs_dN_command[:, 0], dB_vs_dN_command[:, 1])
                        if pv < 0.05:
                            commands_sig[0] += 1
                            comm_sig[animal, day_ix].append(rv_command)
                        commands_sig[1] += 1

            ### Plt_top
            if len(plt_top) > 0:
                for _, xtmp in enumerate(plt_top):
                    xi, yi, col, ms, _, _, _, _, _, _ = xtmp
                    ax.plot(xi, yi, '.', color=col, markersize=20)
                    #ax.plot(xi, yi, '.', color=analysis_config.pref_colors_rgb[0],markersize=10)
                plt_top = np.vstack((plt_top))
                slp,intc,rv,pv,_ = scipy.stats.linregress(np.array(plt_top[:, 0], dtype=float), np.array(plt_top[:, 1], dtype=float))
                print('Example r = %.4f, slp = %.2f, intc = %.2f, pv = %.5f, N = %d'%(rv, slp, intc, pv, plt_top.shape[0]))
                ix_sort = np.argsort(np.vstack((plt_top))[:, 0])
                #print(np.vstack((plt_top))[ix_sort, :])
            ### Plot dB vs. neural var; 
            diff_beh = np.vstack((diff_beh))
            #ax2[0].plot(diff_beh[:, 1], diff_beh[:, 0], 'k.')
            #ax2[0].set_title('mag vs. dB')

            #ax2[1].plot(diff_beh[:, 1]+np.random.randn(diff_beh.shape[0])*.1, var_neur, 'k.')
            #ax2[1].set_title('mag vs. var neur')
            

            # print('#######################')
            # print('Pairs analyzed')
            # for i, (k, v) in enumerate(pairs_analyzed.items()):
            #     print('mag %d, ang %d, N = %d' %(k[0], k[1], len(v)))
            # print('#######################')
            
            D = np.vstack((D))
            slp_REAL,_,rv_REAL,pv_REAL,_ = scipy.stats.linregress(D[:, 0], D[:, 1])
            rv_agg.append(rv_REAL)
            # model = sm.OLS(D[:, 1], D[:, 0])
            # results = model.fit()
            # slp_REAL = float(results.params)

            ax.set_title('PW , rv %.3f, slp %.3f, pv %.5f, N = %d'%(rv_REAL, slp_REAL, pv_REAL, D.shape[0]))
            ax.set_xlabel('Norm Diff Behav. PSTH (-5:5)')
            ax.set_ylabel('Norm Diff Pop Neur [0]')

            if animal == 'grom' and day_ix == 0:
                util_fcns.savefig(f, 'eg_neural_vs_beh_scatter%s_%d' %(animal, day_ix))

            rshuff = []; slpshuff = []
            for i in range(nshuffs):
                if len(D_shuff[i]) > 0:
                    d_tmp = np.vstack((D_shuff[i]))
                    slp, intc_i, rv,_,_ = scipy.stats.linregress(d_tmp[:, 0], d_tmp[:, 1])

                    # model = sm.OLS(d_tmp[:, 1], d_tmp[:, 0])
                    # results = model.fit()
                    #slp = float(results.params)
                    rshuff.append(rv)
                    slpshuff.append(slp)

            #### For Corr. Coeff plot --> no shuffle ###
            util_fcns.draw_plot(ia*10 + day_ix, np.array(rshuff) - np.mean(np.array(rshuff)), 'k', np.array([1.,1.,1.,0.]), axrd)
            #util_fcns.draw_plot(ia*10 + day_ix, np.array(rshuff), 'k', np.array([1.,1.,1.,0.]), axr)
            util_fcns.draw_plot(ia*10 + day_ix, slpshuff, 'k', np.array([1.,1.,1.,0.]), axs)
            util_fcns.draw_plot(ia*10 + day_ix, np.array(slpshuff) - np.mean(np.array(slpshuff)), 'k', np.array([1.,1.,1.,0.]), axsd)

            #### Plto shuffled vs. real R; 
            axrd.plot(ia*10 + day_ix, rv_REAL - np.mean(rshuff), 'k.')
            axr.plot(ia, rv_REAL, 'k.')
            axs.plot(ia*10 + day_ix, slp_REAL, 'k.')
            axsd.plot(ia*10 + day_ix, slp_REAL - np.mean(slpshuff), 'k.')

            axrd.set_xlim([-1, 14])
            axr.set_xlim([-1, 14])
            axs.set_xlim([-1, 14])
            axsd.set_xlim([-1, 14])
            
            axrd.set_ylabel('r-value - shuffle mean')
            axr.set_ylabel('r-value')
            axs.set_ylabel('slope')
            axsd.set_ylabel('slope - shuf mn')
        axr.bar(ia, np.mean(rv_REAL), width=.8,alpha=0.2, color='k')

        #### Compute pooled stats ###
        pooled_stats_all[animal] = pooled_stats

    fs.tight_layout()
    fsd.tight_layout()

    axr.set_xlim([-1, 2])
    axr.set_xticks([0, 1])
    axr.set_xticklabels(['G', 'J'])
    fr.tight_layout()
    util_fcns.savefig(fr, 'neur_beh_corr_rv_vs_shuffN%d'%(nshuffs))

    frd.tight_layout()
    util_fcns.savefig(frd, 'neur_beh_corr_rv_vs_shuffN%d_centbyshuff_mn'%(nshuffs))
    
    fsd.tight_layout()
    util_fcns.savefig(fsd, 'neur_beh_corr_SLP_vs_shuffN%d_centbyshuff_mn'%(nshuffs))

    return pooled_stats_all

def neuraldiff_vs_behdiff_corr_pw_by_command(min_bin_indices=0, min_move_per_command = 5, min_commands=15,
    ncommands_psth = 5):
    
    ### Command-specific CCs -- distribution of sig ccs ####
    fcc, axcc = plt.subplots(figsize = (3, 3))
    
    ### Command-specific CCs -- percent signficant ####
    fps, axps = plt.subplots(figsize = (2, 3))
    
    ### Open mag boundaries 
    mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))

    for ia, animal in enumerate([ 'grom', 'jeev', 'home']):

        perc_sig = []

        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):
            
            ### Pull data ### 
            spks, push, tsk, trg, bin_num, rev_bin_num, move, dat = util_fcns.get_data_from_shuff(animal, day_ix)
            spks = spks * 10
            nneur = spks.shape[1]
            command_bins = util_fcns.commands2bins([push], mag_boundaries, animal, day_ix, 
                                       vel_ix=[3, 5])[0]

            ncomm_tot = 0; 
            ncomm_sig = 0; 
            sig_cc = []
            pool_D = []
            
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
                        if len(ix_mc) >= min_commands:    

                            global_comm_indices[mov] = ix_mc_all
                            ix_com_global.append(ix_mc_all)

                    #### Only analyze if enough movements ####
                    if len(ix_com_global) > 0 and len(global_comm_indices.keys()) >= min_move_per_command: 
                        comm_D = []
                        ix_com_global = np.hstack((ix_com_global))
                        relevant_movs = np.array(global_comm_indices.keys())

                        ##### Get the movements that count; 
                        for imov, mov in enumerate(relevant_movs): 
                            
                            ### MOV specific 
                            ### movements / command 
                            ix_mc_all = global_comm_indices[mov]
                            Nmov1 = len(ix_mc_all)

                            ### Get movement #2 
                            for imov2, mov2 in enumerate(relevant_movs[imov+1:]):

                                assert(mov != mov2)
                                ix_mc_all2 = global_comm_indices[mov2]
                                Nmov2 = len(ix_mc_all2)

                                #### match to the two distributions #######
                                ### Figure out which of the "ix_com" indices can be used for shuffling for this movement 
                                ix_ok1, ix_ok2, niter = distribution_match_mov_pairwise(push[np.ix_(ix_mc_all, [3, 5])], 
                                                             push[np.ix_(ix_mc_all2, [3, 5])])

                                ### Make sure there's enough left; 
                                if np.logical_and(len(ix_ok1) >= min_commands, len(ix_ok2) >= min_commands):

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

                                    #### Matched dN and dB; 
                                    dN = np.linalg.norm(mov_mean_FR1 -mov_mean_FR2)/nneur
                                    dB = np.linalg.norm(mov_PSTH1 - mov_PSTH2)
                                    comm_D.append([dB, dN])
                                    pool_D.append([dB, dN])

                        ### get corr; 
                        comm_D = np.vstack((comm_D))
                        _, _, rv, pv, _ = scipy.stats.linregress(comm_D[:, 0], comm_D[:, 1])
                        if pv < 0.05:  
                            ncomm_sig += 1
                            sig_cc.append(rv)
                        ncomm_tot += 1

                        ### Plto pink 
                        if animal == 'grom' and day_ix == 0 and mag == 0 and ang == 7:
                            axcc.plot(10*ia + day_ix, rv, '.', color='darkblue', markersize=15)
            ### corr over pooled ###
            pool_D = np.vstack((pool_D))
            _, _, rv, pv, _ = scipy.stats.linregress(pool_D[:, 0], pool_D[:, 1])
            #axcc.plot(10*ia + day_ix, rv, '.', color='gray', markersize=15)

            ### Plot the distribution of sig CCs ###
            util_fcns.draw_plot(10*ia + day_ix, sig_cc, 'k', np.array([1., 1., 1., 0.]), axcc)

            frac_sig = float(ncomm_sig)/float(ncomm_tot)
            axps.plot(ia, frac_sig, 'k.')
            perc_sig.append(frac_sig)

        ### Bar sig; 
        axps.bar(ia, np.mean(perc_sig), width=.8,alpha=0.2, color='k')

    axcc.set_xlim([-1, 25])
    axps.set_xlim([-1, 3])
    axps.set_xticks([0, 1, 2])
    axps.set_xticklabels(['G', 'J', 'H'])

    axps.set_ylabel('% Commands with Sig. Corr',fontsize=10)
    axcc.set_ylabel('Dist. of Sig. Corr. Coeff.',fontsize=10)
    axcc.set_xticks([])

    fps.tight_layout()
    fcc.tight_layout() 
    #util_fcns.savefig(fps, 'perc_comm_w_sig_cc')
    #util_fcns.savefig(fcc, 'dist_of_sig_cc')

######## Plot the command PSTH as arrows ###########
def eg_command_PSTH(animal='grom', day_ix = 0, mag = 0, ang = 7, nbins_past_fut = 5,
    arrow_scale = .003, width=.001): 

    ### Mag boundaries ####
    mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))

    ### Get data ####
    spks, push, tsk, trg, bin_num, rev_bin_num, move, dat = util_fcns.get_data_from_shuff(animal, day_ix)
    
    ### Get push PSTH ####
    command_bins = util_fcns.commands2bins([push], mag_boundaries, animal, day_ix, 
                                       vel_ix=[3, 5])[0]

    ### go through movements
    for i_m, mov in enumerate(np.unique(move)):

        ix = np.nonzero((command_bins[:, 0] == mag) & (command_bins[:, 1] == ang) & (move==mov))[0]
        if len(ix) >= 15: 
            f, ax = plt.subplots(figsize=(10, 2))
            ax.axis('square')

            ### Get this PSTH ####
            PSTH = get_PSTH(bin_num, rev_bin_num, push, ix)
            for i in range(PSTH.shape[0]):
                ax.quiver(i*1.5, 2, PSTH[i, 0], PSTH[i, 1],
                    width=arrow_scale*2, color = util_fcns.get_color(mov), 
                    angles='xy', scale=1, scale_units='xy')
            ax.set_ylim([0.5, 3.5])
            ax.set_xlim([-2, 20])

            util_fcns.savefig(f, 'cm_com_psth_mov%.1f'%mov)

    ### Add condition pooled: 
    ix = np.nonzero((command_bins[:, 0] == mag) & (command_bins[:, 1] == ang))[0]
    f, ax = plt.subplots(figsize=(10, 2))
    ax.axis('square')
    PSTH = get_PSTH(bin_num, rev_bin_num, push, ix)
    for i in range(PSTH.shape[0]):
        ax.quiver(i*1.5, 2, PSTH[i, 0], PSTH[i, 1],
                width=arrow_scale*2, color = 'gray', 
                angles='xy', scale=1, scale_units='xy')
    ax.set_ylim([0.5, 3.5])
    ax.set_xlim([-2, 20])

    util_fcns.savefig(f, 'cm_com_psth_pool')


def eg_command_PSTH_pred(animal='grom', day_ix=0, mag=0, ang=7, nbins_past_fut=1,
    arrow_scale = .03, width=.001):

    ### Mag boundaries ####
    mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))

    #### Note that this is NOT conditioned on push ####
    model_nm = 'hist_1pos_0psh_0spksm_1_spksp_0'
    model_set_number = 6
    model_fname = analysis_config.config[animal+'_pref']+'tuning_models_'+animal+'_model_set'+str(model_set_number)+'_.pkl'
    model_dict = pickle.load(open(model_fname, 'rb'))
    KG = util_fcns.get_decoder(animal, day_ix)

    ##### Get the predicted spikes ######
    pred_spks = model_dict[day_ix, model_nm]

    ##### get predicted action ######
    pred_push = np.dot(KG, pred_spks.T).T
    N = pred_push.shape[0]
    pred_push = np.hstack((np.zeros((N, 3)), pred_push[:, 0][:, np.newaxis], 
        np.zeros((N, 1)), pred_push[:, 1][:, np.newaxis], np.zeros((N, 1))))

    #### Get other stuff #####
    spks, push, tsk, trg, bin_num, rev_bin_num, move, dat = util_fcns.get_data_from_shuff(animal, day_ix)

    ### Get tm0/tm1 to match 
    tm0, tm1 = generate_models.get_temp_spks_ix(dat['Data'])
    push_tm0 = push[tm0, :]
    push_tm1 = push[tm1, :]
    if model_nm == 'hist_1pos_0psh_2spksm_1_spksp_0':
        assert(np.allclose(pred_push[:, [3, 5]], push[np.ix_(tm0, [3, 5])]))

    ### Want to know push at tm1 --> so then can get estimate of push at tm0 for "next command"
    command_bins_tm1 = util_fcns.commands2bins([push_tm1], mag_boundaries, animal, day_ix, 
                                       vel_ix=[3, 5])[0]
    bin_num_tm1 = bin_num[tm1]
    rev_bin_num1 = rev_bin_num[tm1]
    move_tm1 = move[tm1]

    f, ax = plt.subplots(figsize=(5, 10))
    cnt = 0; 
    
    movs = np.unique(move)
    ### Sort by target 
    movs_mod = np.mod(movs, 10.)
    ix_sort = np.argsort(movs_mod)


    for i_m, mov in enumerate(movs[ix_sort]):
        ix = np.nonzero((command_bins_tm1[:, 0] == mag) & (command_bins_tm1[:, 1] == ang) & (move_tm1==mov))[0]
        if len(ix) >= 15: 
            
            ax.axis('square')

            ### Plto the command you're centering on ####
            mn_psh_tmp = np.mean(push_tm1[np.ix_(ix, [3, 5])], axis=0)
            mn_psh_tmp = mn_psh_tmp / np.linalg.norm(mn_psh_tmp)
            
            ax.quiver(1, 10-cnt, mn_psh_tmp[0], mn_psh_tmp[1],
                width=arrow_scale*2, color = util_fcns.get_color(mov), 
                angles='xy', scale=1, scale_units='xy')

            mn_push_tmp = np.mean(push_tm0[np.ix_(ix, [3, 5])], axis=0)
            mn_push_tmp = mn_push_tmp / np.linalg.norm(mn_push_tmp)
            ### Plot the actual next command #######
            ax.quiver(3, 10-cnt, mn_push_tmp[0], mn_push_tmp[1], 
                width=arrow_scale*2, color = util_fcns.get_color(mov), 
                angles='xy', scale=1, scale_units='xy')

            mn_push_tmp = np.mean(pred_push[np.ix_(ix, [3, 5])], axis=0)
            mn_push_tmp = mn_push_tmp / np.linalg.norm(mn_push_tmp)
            ### PLot the predicted next command #####
            ax.quiver(5, 10-cnt, mn_push_tmp[0], mn_push_tmp[1], 
                width=arrow_scale*2, color = util_fcns.get_color(mov), 
                angles='xy', scale=1, scale_units='xy')

            cnt += 1
            
    ax.set_ylim([-1, 11])
    ax.set_xlim([-1, 7])

    util_fcns.savefig(f, 'cm_com_pred_psth_movs')





def get_PSTH(bin_num, rev_bin_num, push, indices, num_bins=5, min_bin_set = 0, skip_assert_min = False):

    if skip_assert_min:
        pass
    else:
        assert(np.min(bin_num) == min_bin_set)
        assert(np.min(rev_bin_num) == min_bin_set)
    
    all_push = []
    push_vel = push[:, [3, 5]]
    for ind in indices:

        if bin_num[ind] >= num_bins + min_bin_set and rev_bin_num[ind] >= num_bins + min_bin_set:
            all_push.append(push_vel[ind-num_bins:ind+num_bins+1, :])
    if len(all_push) > 0:
        all_push = np.dstack((all_push))
        return np.mean(all_push, axis=2)
    else:
        return None

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


 