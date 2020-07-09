#### Utils for model generation ######
import analysis_config
import util_fcns
import prelim_analysis as pa 
from resim_ppf import ppf_pa

import numpy as np 
import pandas
import copy
import pickle 
import tables 

import matplotlib.pyplot as plt

import sklearn.linear_model
from sklearn.linear_model import Ridge
import scipy.stats
from collections import defaultdict

pref = analysis_config.config['grom_pref']
n_entries_hdf = 800

class Model_Table(tables.IsDescription):
    param = tables.Float64Col(shape=(n_entries_hdf, ))
    pvalue = tables.Float64Col(shape=(n_entries_hdf, ))
    r2 = tables.Float64Col(shape=(1,))
    r2_pop = tables.Float64Col(shape=(1,))
    aic = tables.Float64Col(shape=(1,))
    bic = tables.Float64Col(shape=(1,))
    day_ix = tables.Float64Col(shape=(1,))

#### Methods to get data needed for spike models ####
def get_spike_kinematics(animal, day, order, history_bins, full_shuffle = False, 
    within_bin_shuffle = False, shuffix = None, nshuffs = 1, **kwargs):
    
    if 'trial_ix' in kwargs:
        trial_ix = kwargs['trial_ix']
    else:
        trial_ix = None

    if within_bin_shuffle or full_shuffle:
        if 'day_ix' not in kwargs.keys():
            raise Exception('Need to include day ix to get mag boundaries for shuffling wihtin bin')
        else:
            day_ix = kwargs['day_ix']

    if within_bin_shuffle and full_shuffle:
        raise Exception('Cant have both shuffles! Choose one!')

    spks = []
    vel = []
    push = []
    pos = []
    tsk = []
    trl = []
    bin_num = []

    ### Not sure what these are 
    trial_ord = {}

    ### Trial order -- which order in the day did this thing happen? 
    trl_order_ix = []; 
    ### Get out the target indices and target position for each trial 
    trg_trl = []; trg_trl_pos = []; 
    trl_off = 0 

    ### For each task: 
    for i_t, tsk_fname in enumerate(day):

        ### Start the trial count at zero 
        tsk_trl_cnt = 0

        ### For each task filename 
        for i_te_num, te_num in enumerate(tsk_fname):



            ################################################
            ####### Spike and Kinematics Extraction ########
            ################################################
            if animal == 'grom':

                ### Open up CO / OBS ###
                co_obs_dict = pickle.load(open(pref+'co_obs_file_dict.pkl'))
                hdf = co_obs_dict[te_num, 'hdf']
                hdfix = hdf.rfind('/')
                hdf = tables.openFile(pref+hdf[hdfix:])

                dec = co_obs_dict[te_num, 'dec']
                decix = dec.rfind('/')
                decoder = pickle.load(open(pref+dec[decix:]))
                F, KG = decoder.filt.get_sskf()

                # Get trials: 
                drives_neurons_ix0 = 3
                key = 'spike_counts'
                rew_ix = np.array([t[1] for it, t in enumerate(hdf.root.task_msgs[:]) if t[0]=='reward'])
                
                ### Get out all the items desired from prelim_analysis ####
                bin_spk, targ_i_all, targ_ix, trial_ix_all, decoder_all = pa.extract_trials_all(hdf, rew_ix, 
                    drives_neurons_ix0=drives_neurons_ix0, hdf_key=key, keep_trials_sep=True,
                    reach_tm_is_hdf_cursor_pos=False, reach_tm_is_kg_vel=True, **dict(kalman_gain=KG))

                ### Make sure targ_i_all changes and targ_ix changes are at the same time; 
                ############################
                ######## TESTING ###########
                ############################
                dTA = np.diff(targ_i_all, axis=0)
                dTI = np.diff(targ_ix, axis=0)

                ixx = np.nonzero(dTA[:, 0])[0]
                ixy = np.nonzero(dTA[:, 1])[0]
                ixt = np.nonzero(dTI)[0]

                assert(np.all(ixx == ixy))
                assert(np.all(ixy == ixt))

                tmp = np.nonzero(targ_ix >= 0)[0]
                
                assert(len(np.unique(targ_ix[tmp])) == 8)
                assert(len(np.vstack((bin_spk))) == len(np.vstack((decoder_all))))
                assert(len(np.vstack((bin_spk))) == len(targ_i_all))

                #Also get position from cursor pos: 
                _, _, _, _, cursor_pos = pa.extract_trials_all(hdf, rew_ix, 
                    drives_neurons_ix0=drives_neurons_ix0, hdf_key=key, keep_trials_sep=True,
                    reach_tm_is_hdf_cursor_pos=True, reach_tm_is_kg_vel=False)  

                _, _, _, _, cursor_state = pa.extract_trials_all(hdf, rew_ix,
                    drives_neurons_ix0=drives_neurons_ix0, hdf_key=key, keep_trials_sep=True,
                    reach_tm_is_hdf_cursor_pos=False, reach_tm_is_hdf_cursor_state=True, 
                    reach_tm_is_kg_vel=False)

                assert(len(np.vstack((cursor_pos))) == len(targ_i_all))
                assert(len(np.vstack((cursor_state)) == len(targ_i_all)))
                
                ###################################
                ######## End of TESTING ###########
                ###################################
                cursor_vel = [curs[:, 2:, 0] for curs in cursor_state]
                
                ##### Trial order ####### 
                #trl_rd.append(order[i_t][i])     
                
            elif animal == 'jeev':
                
                bin_spk, targ_i_all, targ_ix, trial_ix_all, decoder_all, cursor_state, unbinned, exclude = ppf_pa.get_jeev_trials_from_task_data(te_num, include_pos=True, binsize=.1)
                cursor_pos = [curs[:, [0, 2]] for curs in cursor_state]
                cursor_vel = [curs[:, [3, 5]] for curs in cursor_state]

                ####### TESTING ########
                dTA = np.diff(targ_i_all, axis=0)
                dTI = np.diff(targ_ix, axis=0)

                ixx = np.nonzero(dTA[:, 0])[0]
                ixy = np.nonzero(dTA[:, 1])[0]
                ixxy = np.unique(np.hstack((ixx, ixy)))
                ixt = np.nonzero(dTI)[0]
                if len(ixxy) == len(ixt):
                    assert(np.all(ixxy == ixt))
                elif len(ixxy) < len(ixt):
                    assert(np.all(i in ixt for i in ixxy))
                else:
                    raise Exception('Shouldnt be more target changes than trial changes')

                ### Sometimes targets can be repeated so just 
                ### make sure all ixxys are in ixtrl
                ixtrl = np.nonzero(np.diff(trial_ix_all))[0]
                assert(np.all(i in ixtrl for i in ixxy))

                tmp = np.nonzero(targ_ix >= 0)[0]
                if i_t == 1:
                    ### 9 obstacle target ###
                    assert(np.max(np.unique(targ_ix[tmp])) > 8)
                    print('Len Obs Targs %s'%str(np.unique(targ_ix[tmp])))
                else:
                    assert(len(np.unique(targ_ix[tmp])) == 8)

                ###################################
                ######## End of TESTING ###########
                ###################################

                #trl_rd.append(order[i_t][i])
                
                # f, ax = plt.subplots(ncols=3, nrows=3)
                # if i_t == 0:
                #     fall, axll = plt.subplots()

                # targ_ix2 = np.unique(targ_ix)
                # targ_ix2 = targ_ix2[targ_ix2 >= 0]

                # for it, t in enumerate(targ_ix2):
                #     ### get trial with these guys
                #     psth = []; 

                #     trl = np.unique(trial_ix_all[np.nonzero(targ_ix == t)[0]])
                #     for tr in trl:
                #         ax[it/3, it%3].plot(cursor_pos[tr][:, 0], cursor_pos[tr][:, 1], '-', linewidth=1, color = cmap_list[it])
                #         axll.plot(cursor_pos[tr][:, 0], cursor_pos[tr][:, 1], '-', linewidth=1, color = cmap_list[i_t])
                #     ax[it/3, it%3].set_xlim([-.5, 2.5])
                #     ax[it/3, it%3].set_ylim([1.25, 4.25])
                # if i_t ==1:
                #     import pdb; pdb.set_trace()

            ########## Trial Ix: Select only specific indices #########
            if trial_ix is not None:
                print('Getting trial indices: %s' %str(trial_ix))
                ix_ix = np.nonzero(np.logical_and(trial_ix>=trl_ix_cnt, trial_ix<(len(bin_spk)+trl_ix_cnt)))[0]
                ix_mod = trial_ix[ix_ix] - trl_ix_cnt
            else:
                ix_mod = np.arange(len(bin_spk))

            ############ Get the target, same size as cursor/velocity/spks #########
            rm_trls = [] ### Target indices for the trials 
            for ix_ in ix_mod:

                ### Get an index that corresponds to this trial number 
                ix_x = np.nonzero(trial_ix_all == ix_)[0] # choose an index, not all indices

                assert(np.all(trial_ix_all[ix_x] == ix_))
                assert(len(trial_ix_all) == len(targ_ix))

                #### Make sure all the target indices are the same 
                assert(np.all(targ_ix[ix_x] == targ_ix[ix_x[0]]))

                ### Get the target index for this trial -- if it is -1 then remove this trial 
                if targ_ix[ix_x[0]] < 0:
                    rm_trls.append(ix_)
                    print('REMOVING A TRIAL ')
                    #import pdb; pdb.set_trace()

                ### If we want to keep this one; #
                else:
                    ### Keep so that indexing stays consistent
                    trg_trl.append(targ_ix[ix_x[0]]) ## #Get the right target for this trial
                    trg_trl_pos.append(targ_i_all[ix_x[0], :])
                    assert(np.all(targ_i_all[ix_x, 0] == targ_i_all[ix_x[0], 0]))
                    assert(np.all(targ_i_all[ix_x, 1] == targ_i_all[ix_x[0], 1]))
                    trl_order_ix.append(order[i_t][i_te_num])

            ### Remove trials that are -1: 
            ### Keep these trials ####
            ix_mod = np.array([iii for iii in ix_mod if iii not in rm_trls])
           
            # Add to list: 
            ### Stack up the binned spikes; 
            spks.append(np.vstack(([bin_spk[x] for x in ix_mod])))
            bin_num.append(np.arange(bin_spk[x].shape[0]) for x in ix_mod)
            
            # We need the real velocity: 
            vel.append(np.vstack(([cursor_vel[x] for x in ix_mod])))

            # Get the push too? 
            push.append(np.vstack(([decoder_all[x] for x in ix_mod])))

            # Not the Kalman Gain: 
            tsk.append(np.zeros(( np.vstack(([bin_spk[x] for x in ix_mod])).shape[0], )) + i_t)
            
            ### Get the trial INDEX for this new formation ####
            tmp = np.array([np.zeros(( bin_spk[x].shape[0])) + ix + trl_off for ix, x in enumerate(ix_mod)])
            trl_off += len(ix_mod)
            print('new trial offset %d, total trls in this blk %d' %(trl_off, len(ix_mod)))

            ### Make sure there isn't trial overlapping 
            if len(trl) > 0:
                assert(np.min(np.hstack((tmp))) > np.max(np.hstack((trl))))

            ### If passes then safely add; 
            trl.append(np.hstack((tmp)))

            pos.append(np.vstack(([cursor_pos[x] for x in ix_mod])))

    
    ### After everything is done stack up 
    spks = np.vstack((spks))
    bin_num = np.hstack((bin_num))
    vel = np.vstack((vel))
    push = np.vstack((push))
    tsk = np.hstack((tsk))
    trl = np.hstack((trl))
    pos = np.vstack((pos))
    assert(spks.shape[0] == vel.shape[0] == push.shape[0] == tsk.shape[0] == trl.shape[0] == pos.shape[0])
    assert(len(trg_trl) == len(np.unique(trl)) == len(trg_trl_pos) == len(trl_order_ix))
    assert(len(np.unique(tsk) == 2))
    assert(np.all(np.unique(trl) == np.arange(np.max(trl) + 1)))
    
    ########################################################
    ##### Here we can safely add shuffles if we want #######
    ########################################################
    Data_temp = []; Sub_spk0_temp_all = []; Sub_push_all = []; 

    for nsi in range(nshuffs):
        if nsi % 10 == 0:
            print('Staring shuff %d' %(nsi))

        ### These are the things we want to save: 
        temp_tune = {}
        temp_tune['spks'] = [] ### Time window of spikes; 
        temp_tune['spk0'] = [] ### spks at a given time point

        temp_tune['vel'] = []
        temp_tune['pos'] = []
        
        temp_tune['tsk'] = []
        temp_tune['trg'] = [] ### Index 
        temp_tune['trg_pos'] = [] ### position in space 
        temp_tune['trl'] = []
        temp_tune['psh'] = [] ### xy push 
        temp_tune['trl_bin_ix'] = [] ### index from onset 
        temp_tune['day_bin_ix'] = [] ### Which bin number (from the full day) is this? 
        temp_tune['day_bin_ix_shuff'] = []

        ################## INIT the shuffle seed #################
        if shuffix is not None:
            np.random.seed(shuffix)
        else:
            np.random.seed(nsi)

        ################## Run the shuffle ######################
        if full_shuffle:
            shuff_ix = full_shuffling(spks, push, animal, day_ix)

        elif within_bin_shuffle:
            shuff_ix, KG = within_bin_shuffling(spks, push, animal, day_ix)

        else:
            shuff_ix = np.arange(spks.shape[0])

        assert(spks.shape[0] == push.shape[0] == len(shuff_ix))
        spks = spks[shuff_ix, :]
        push = push[shuff_ix, :]

        if animal == 'grom':
            assert(np.allclose(np.dot(KG, spks.T).T, push))

        ########################################
        ########################################
        for t in np.unique(trl):
            assert(int(t) == t)
            t = int(t)

            trl_ix = np.nonzero(trl == t)[0]

            ##### Get all the trial activity ####
            spk_trl = spks[trl_ix, :]
            vel_trl = vel[trl_ix, :]
            push_trl = push[trl_ix, :]
            pos_trl = pos[trl_ix, :]
            tsk_trl = tsk[trl_ix]
            assert(np.all(tsk_trl[0] == tsk_trl))

            nt = len(trl_ix)
            nneurons = spks.shape[1]

            ### Modified on 9/13/19 and 6/16/20
            spk_temp =  np.zeros((nt-(2*history_bins),    (1+(2*history_bins)), nneurons))
            ### EOM
            spk_temp0 = np.zeros((nt-(2*history_bins),                          nneurons))

            velo =      np.zeros((nt-(2*history_bins), 2*(1+(2*history_bins))))
            poso = np.zeros_like(velo)
            pusho = np.zeros_like(velo)
            
            ### EOM
            task = np.zeros((nt-(2*history_bins), )) + tsk_trl[0]
            targ = np.zeros((nt-(2*history_bins), )) + trg_trl[t]
            targ_pos = np.zeros((nt - (2*history_bins), 2))
            targ_pos[:, 0] = trg_trl_pos[t][0]
            targ_pos[:, 1] = trg_trl_pos[t][1]

            ix_ = np.zeros((len(task), 2))# + t
            ix_[:, 0] = t # Full set of trials 
            ix_[:, 1] = trl_order_ix[t] # Order

            XY_div = (2*history_bins)+1

            # # Add to temporal timing model
            # for t in range(len(bin_spk)):
            #     if t in ix_mod:
            #         trl = bin_spk[t]
            #         nt, nneurons = trl.shape

            #         ### Modified on 9/13/19
            #         spk = np.zeros((nt-(2*history_bins), (1+(2*history_bins)), nneurons))
            #         ### EOM
            #         spk0 = np.zeros((nt - (2*history_bins), nneurons))

            #         velo = np.zeros((nt-(2*history_bins), 2*(1+(2*history_bins))))
            #         poso = np.zeros_like(velo)

            #         ### Modified 9/16/19 -- adding lags to push
            #         pusho = np.zeros_like(velo)
                    
            #         ### EOM
            #         task = np.zeros((nt-(2*history_bins), ))+i_t

            #         ### So this needs to 
            #         targ = np.zeros((nt-(2*history_bins), ))+trg_trl[t]
            #         targ_pos = np.zeros((nt - (2*history_bins), 2)) + trg_trl_pos[t]

            #         ix_ = np.zeros((len(task), 2))# + t
            #         ix_[:, 0] = t # Full set of trials 
            #         ix_[:, 1] = order[i_t][i] # Order

            #         XY_div = (2*history_bins)+1

            ### For each time bin that we can actually capture: 
            for tmi, tm in enumerate(np.arange(history_bins, nt-history_bins)):
                
                ### for the individual time bins; 
                for tmpi, tmpii in enumerate(np.arange(tm - history_bins, tm + history_bins + 1)):
                    spk_temp[tmi, tmpi, :] = spk_trl[tmpii, :]

                spk_temp0[tmi, :] = spk_trl[tm, :]
                velo[tmi, :XY_div] = np.squeeze(np.array(vel_trl[tm-history_bins:tm+history_bins+1, 0]))
                velo[tmi, XY_div:] = np.squeeze(np.array(vel_trl[tm-history_bins:tm+history_bins+1, 1]))
                
                poso[tmi, :XY_div] = np.squeeze(np.array(pos_trl[tm-history_bins:tm+history_bins+1, 0]))
                poso[tmi, XY_div:] = np.squeeze(np.array(pos_trl[tm-history_bins:tm+history_bins+1, 1]))
                
                pusho[tmi, :XY_div] = np.squeeze(np.array(push_trl[tm-history_bins:tm+history_bins+1, 3]))
                pusho[tmi, XY_div:] = np.squeeze(np.array(push_trl[tm-history_bins:tm+history_bins+1, 5]))
            
            temp_tune['spk0'].append(spk_temp0)
            temp_tune['spks'].append(spk_temp)
            temp_tune['vel'].append(velo)
            temp_tune['tsk'].append(task)
            temp_tune['pos'].append(poso)
            temp_tune['trl'].append(ix_)
            temp_tune['trg'].append(targ)
            temp_tune['trg_pos'].append(targ_pos)
            temp_tune['psh'].append(pusho)
            temp_tune['trl_bin_ix'].append(np.arange(history_bins, nt-history_bins))
            temp_tune['day_bin_ix'].append(trl_ix[np.arange(history_bins, nt-history_bins)]) ### Which bin number (from the full day) is this? 
            temp_tune['day_bin_ix_shuff'].append(shuff_ix[trl_ix[np.arange(history_bins, nt-history_bins)]])
            
            #import pdb; pdb.set_trace()
        
        ################################################
        #### STACK UP EVERYTHING WITHOUT TIME SHIFTS ###
        ################################################
        # spikes = np.vstack((spks))
        # velocity = np.vstack((vel))
        spikes = spks.copy()
        velocity = vel.copy()

        if velocity.shape[1] == 7:
            velocity = np.array(velocity[:, [3, 5]])
        
        position = pos.copy()
        task_index = tsk.copy()
        
        #################################
        ######## Model Options ##########
        #################################

        sub_spk_temp_all = np.vstack((temp_tune['spks']))
        sub_spk0_temp_all = np.vstack((temp_tune['spk0']))
        nneur = sub_spk_temp_all.shape[2]

        #### Time lags ####
        TL = np.arange(-1*history_bins, history_bins+1)
        i_zero = np.nonzero(TL==0)[0]

        #### Push at time zero and time zero + future ####
        #### Current push X / Y 
        sub_push_all = np.vstack((temp_tune['psh']))[:, [i_zero, i_zero+XY_div]]

        ### Double checking; #####
        tmp_psh = []
        for trltmp in range(len(temp_tune['trl_bin_ix'])):
            trl_ix = np.nonzero(trl == trltmp)[0]
            subtrlix = temp_tune['trl_bin_ix'][trltmp]
            tmp_psh.append(push[np.ix_(trl_ix[subtrlix], [3, 5])])
        assert(np.allclose(np.vstack((tmp_psh)), sub_push_all[:, :, 0]))

        #### Get all the temporally shifted data ###
        psh_temp = np.vstack((temp_tune['psh']))
        vel_temp = np.vstack((temp_tune['vel']))
        tsk_temp = np.hstack((temp_tune['tsk']))
        pos_temp = np.vstack((temp_tune['pos']))
        trg_temp = np.hstack((temp_tune['trg']))

        #### Target positions / trial bin indices / trial number 
        trg_pos_temp = np.vstack((temp_tune['trg_pos']))
        bin_temp = np.hstack((temp_tune['trl_bin_ix']))
        trl_temp = np.vstack((temp_tune['trl']))[:, 0]
        day_bin_ix_temp = np.hstack((temp_tune['day_bin_ix']))
        day_bin_ix_shuff_temp = np.hstack((temp_tune['day_bin_ix_shuff']))

        ### Generate acceleration #####
        accx = np.hstack((np.array([0.]), np.diff(velocity[:, 0])))
        accy = np.hstack((np.array([0.]), np.diff(velocity[:, 1])))

        ########################################################################
        ### Add all the data to the pandas frame with t, VEL, POS, ACC ###
        #######################################################################

        data = pandas.DataFrame({'velx': velocity[:, 0], 'vely': velocity[:, 1],'tsk': task_index, 
            'posx': position[:, 0], 'posy': position[:, 1]})

        ### Add it to the end 
        data['accx'] = pandas.Series(accx, index=data.index)
        data['accy'] = pandas.Series(accy, index=data.index)

        ### Time lags 
        TL = np.arange(-1*history_bins, history_bins+1)

        tmp_dict = {}
        for i, tli in enumerate(TL):
            if tli <= 0:
                nm0 = 'm'
            elif tli > 0:
                nm0 = 'p'

            for vl in ['vel', 'pos', 'psh']:
                for ixy, (nm, xy_ad) in enumerate(zip(['x', 'y'], [0, XY_div])):
                    if vl=='vel':
                        tmp = vel_temp[:, i + xy_ad]
                    elif vl == 'pos':
                        tmp = pos_temp[:, i + xy_ad]
                    elif vl == 'psh':
                        tmp = psh_temp[:, i + xy_ad]

                    tmp_dict[vl+nm+'_t'+nm0+str(np.abs(TL[i]))] = copy.deepcopy(tmp)

            for n in range(nneur):
                tmp_dict['spk_t'+nm0+str(np.abs(TL[i]))+'_n'+str(n)] = sub_spk_temp_all[:, i, n]

        #tmp_dict['trial_ord'] = trl_temp
        tmp_dict['tsk'] = tsk_temp
        tmp_dict['trg'] = trg_temp
        tmp_dict['trg_posx'] = trg_pos_temp[:, 0]
        tmp_dict['trg_posy'] = trg_pos_temp[:, 1]
        tmp_dict['bin_num'] = bin_temp
        tmp_dict['trl'] = trl_temp; 
        tmp_dict['day_bin_ix'] = day_bin_ix_temp
        tmp_dict['day_bin_ix_shuff'] = day_bin_ix_shuff_temp

        ##################################
        ### MODEL with HISTORY +FUTURE ###
        ##################################
        data_temp = pandas.DataFrame(tmp_dict)
        if nsi == 1:
            plot_data_temp(data_temp, animal, True)

        if nshuffs == 1:
            return data, data_temp, sub_spk0_temp_all, sub_spk_temp_all, sub_push_all
        else:
            Data_temp.append(copy.copy(data_temp)); 
            Sub_spk0_temp_all.append(copy.copy(sub_spk0_temp_all));
            Sub_push_all.append(copy.copy(sub_push_all))

    return Data_temp, Sub_spk0_temp_all, Sub_push_all
            
### Confirmation that extracted data looks right ###
def plot_data_temp(data_temp, animal, use_bg = False):
    
    ### COLORS ####
    cmap_list = analysis_config.pref_colors
    co_obs_cmap = [np.array([0, 103, 56])/255., np.array([46, 48, 146])/255., ]

    if animal == 'jeev':
        use_bg = False

    ### for each target new plto: 
    fco, axco = plt.subplots()
    fob, axob = plt.subplots()
    
    for ax in [axco, axob]:
        ax.axis('square')
    tgs = [4, 5]

    ### open target: 
    for i in range(2): 
        tsk_ix = np.nonzero(data_temp['tsk'] == i)[0]
        print('Animal %s, tsk %d, N = %d' %(animal, i, len(tsk_ix)))

        targs = np.unique(data_temp['trg'][tsk_ix])

        for itr, tr in enumerate(targs):
            ## Get the trial numbers: 
            targ_ix = np.nonzero(data_temp['trg'][tsk_ix] == tr)[0]
            trls = np.unique(data_temp['trl'][tsk_ix[targ_ix]])
            print('Tsk %d, Trg %d, N = %d' %(i, int(tr), len(targ_ix)))

            if tgs[i] == itr:
                alpha = 1.0; LW = 2.0
            else:
                alpha = 0.4; LW = 1.0

            if i == 0:
                axi = axco; 
            else:
                axi = axob#[itr/3, itr%3]

            for trl in trls:
                ix = np.nonzero(data_temp['trl'][tsk_ix] == trl)[0]
                if use_bg:
                    axi.plot(data_temp['posx_tm0'][tsk_ix[ix]], data_temp['posy_tm0'][tsk_ix[ix]], '-', color=co_obs_cmap[i], linewidth = LW, alpha=alpha)
                else:
                    axi.plot(data_temp['posx_tm0'][tsk_ix[ix]], data_temp['posy_tm0'][tsk_ix[ix]], '-', color=cmap_list[itr])

    ### Add target info for Grom: 
    for i, a in enumerate(np.linspace(0., 2*np.pi, 9)):
        if i < 8:
            tg = [10*np.cos(a), 10*np.sin(a)]
            circle = plt.Circle(tg, radius=1.7, color = cmap_list[i], alpha=.2)
            axob.add_artist(circle)
            circle = plt.Circle(tg, radius=1.7, color = cmap_list[i], alpha=.2)
            axco.add_artist(circle)

        for ax in [axco, axob]:
            if animal == 'grom':
                ax.set_xlim([-12, 12])
                ax.set_ylim([-12, 12])
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.set_xlim([-.5, 2.5])
                ax.set_ylim([1.2, 5.0])
                ax.set_xticks([])
                ax.set_yticks([])

##### Training / testing sets ######
def get_training_testings(n_folds, data_temp):
    
    ### Get training and testing datasets: 
    N_pts = [];

    ### List which points are from which task: 
    for tsk in range(2):

        ### Get task specific indices; 
        ix = np.nonzero(data_temp['tsk'] == tsk)[0]

        ### Shuffle the task indices
        N_pts.append(ix[np.random.permutation(len(ix))])
    
    train_ix = dict();
    test_ix = dict(); 

    ### Get training and testing data
    for i_f, fold_perc in enumerate(np.arange(0, 1., 1./n_folds)):
        test_ix[i_f] = []
        train_ix[i_f] = []; 

        ### Task -- pull test/train points 
        for tsk in range(2):
            ntmp = len(N_pts[tsk])
            tst = N_pts[tsk][int(fold_perc*ntmp):int((fold_perc+(1./n_folds))*ntmp)]
            trn = np.array([j for i, j in enumerate(N_pts[tsk]) if j not in tst])
            
            test_ix[i_f].append(tst)
            train_ix[i_f].append(trn)

        test_ix[i_f] = np.hstack((test_ix[i_f])).astype(int)
        train_ix[i_f] = np.hstack((train_ix[i_f])).astype(int)

        tmp = np.unique(np.hstack((test_ix[i_f], train_ix[i_f])))

        ### Make sure that unique point is same as all data
        assert len(tmp) == len(data_temp)
    return test_ix, train_ix 

def get_training_testings_generalization_LDS_trial(n_folds, data_temp, 
    match_task_spec_n = False):
    '''
    same as below but match number of trials; 
    '''
    train_ix = dict()
    test_ix = dict()
    type_of_model = np.zeros((3*n_folds))

    ### Get training and testing datasets: 
    N_trls = []; N = []; 
    all_trls = np.unique(data_temp['trl'])

    ### List which points are from which task: 
    for tsk in range(2):

        ### Get task specific indices; 
        ix = np.nonzero(data_temp['tsk'] == tsk)[0]
        Trls = np.unique(data_temp['trl'][ix])

        ### Shuffle the task indices
        N_trls.append(Trls[np.random.permutation(len(Trls))])
        N.append(len(Trls))

    Ncomb = np.min(np.hstack((N)))
    
    ### Amount of training data from each task; 
    Ncomb_train = int(0.8*Ncomb)
    Ntot = len(np.unique(data_temp['trl']))

    ##### For each fold ###
    for i_f, fold_perc in enumerate(np.arange(0., 1., 1./n_folds)):

        ### [1, 2, 3, 4, 5] ###
        train_ix[i_f] = []; 
        test_ix[i_f] = []; 

        #### For train CO:
        for tsk in range(2):    

            if match_task_spec_n:

                ### For this task, whats the amt of data thats needed for testing, if training is Ncomb_train? 
                n_test = N[tsk] - Ncomb_train
                
                ### If this allows for typical testing data, do that: 
                if int(fold_perc*N[tsk]) + n_test < N[tsk]:
                    ### Own task; 
                    test1 = N_trls[tsk][int(fold_perc*N[tsk]):int(fold_perc*N[tsk]) + n_test]
                
                else:
                ### Do something else ###
                    test1 = N_trls[tsk][-n_test:]

                ### Training data is NOT this test data; 
                trn = np.array([j for i, j in enumerate(N_trls[tsk]) if j not in test1])
                assert(np.allclose(np.sort(np.hstack((trn, test1))), np.sort(N_trls[tsk])))

                ### Get test data from the other task; 
                other_task = np.mod(tsk + 1, 2); 
                n_test_other = N[other_task] - Ncomb_train

                if int(fold_perc*N[other_task]) + n_test_other < N[other_task]:
                    test2 = N_trls[other_task][int(fold_perc*N[other_task]):int(fold_perc*N[other_task]) + n_test_other]
                else:
                    test2 = N_trls[other_task][-n_test_other:]

                train_ix[i_f + tsk*n_folds] = trn; 
                test_ix[i_f + tsk*n_folds] = np.hstack((test1, test2))
                type_of_model[i_f + tsk*n_folds] = tsk

            else:
                ### Get all these points to test; 
                tst = N_trls[tsk][int(fold_perc*N[tsk]):int((fold_perc+(1./n_folds))*N[tsk])]
                
                #### Also add 20% of OTHER taks to testing; 
                other_task = np.mod(tsk + 1, 2); 
                tst2 = N_trls[other_task][int(fold_perc*N[other_task]):int((fold_perc+(1./n_folds))*N[other_task])]

                ### Add all non-testing points from this task to training; 
                trn = np.array([j for i, j in enumerate(N_pts[tsk]) if j not in tst])

                train_ix[i_f + tsk*n_folds] = trn.astype(int)
                test_ix[i_f + tsk*n_folds] = np.hstack((tst, tst2)).astype(int)
                type_of_model[i_f + tsk*n_folds] = tsk

        ### Now do the last part -- combined models with equal data from both tasks;  
        ### [6, 7, 8, 9, 10] ###
        test_ix[i_f + 2*n_folds] = []
        train_ix[i_f + 2*n_folds] = []; 

        ### Match this -- 
        if match_task_spec_n:
            Ncomb_train_gen = int(0.5*Ncomb_train)
        else:
            Ncomb_train_gen = Ncomb_train

        ### Task -- pull test/train points 
        for tsk in range(2):

            #### Total amount of testing data 
            Ncomb_test_tsk = N[tsk] - Ncomb_train_gen

            #### How big are the offsets for this to work? 
            if int(fold_perc*N[tsk]) + Ncomb_test_tsk < N[tsk]:
                test_ix[i_f + 2*n_folds].append(N_trls[tsk][int(fold_perc*N[tsk]):int(fold_perc*N[tsk]) + Ncomb_test_tsk])
            else:
                ### Just add the right amount of data from the end; 
                test_ix[i_f + 2*n_folds].append(N_trls[tsk][-Ncomb_test_tsk:])
        
        test_ix[i_f + 2*n_folds] = np.hstack((test_ix[i_f + 2*n_folds])).astype(int)
        train_ix[i_f + 2*n_folds] = np.array([i for i in range(Ntot) if i not in test_ix[i_f + 2*n_folds]]).astype(int)
        type_of_model[i_f + 2*n_folds] = 2

    for i_t in range(3):
        test = []; 
        for i_f in range(n_folds):
            test.append(np.hstack((test_ix[i_f + i_t*n_folds])))
        test = np.sort(np.unique(np.hstack((test))))
        assert(np.all(test == all_trls))

    return test_ix, train_ix, type_of_model.astype(int)    

def get_training_testings_generalization(n_folds, data_temp, 
    match_task_spec_n = False):
    ''' same as above, but now doing this for train CO, train OBS, train both 
        input: 
            match_task_spec_n --> whether to make sure the data used to train the task spec 
                models is same number in both cases
        output: 
    '''

    train_ix = dict();
    test_ix = dict(); 
    type_of_model = np.zeros((3*n_folds))

    ### Get training and testing datasets: 
    N_pts = []; N = []; 

    ### List which points are from which task: 
    for tsk in range(2):

        ### Get task specific indices; 
        ix = np.nonzero(data_temp['tsk'] == tsk)[0]

        ### Shuffle the task indices
        N_pts.append(ix[np.random.permutation(len(ix))])
        N.append(len(ix))

    Ncomb = np.min(np.hstack((N)))
    ### Amount of training data from each task; 
    Ncomb_train = int(0.8*Ncomb)

    Ntot = len(data_temp['tsk'])

    ##### For each fold ###
    for i_f, fold_perc in enumerate(np.arange(0., 1., 1./n_folds)):

        ### [1, 2, 3, 4, 5] ###
        train_ix[i_f] = []; 
        test_ix[i_f] = []; 

        #### For train CO:
        for tsk in range(2):    

            if match_task_spec_n:

                ### For this task, whats the amt of data thats needed for testing, if training is Ncomb_train? 
                n_test = N[tsk] - Ncomb_train
                
                ### If this allows for typical testing data, do that: 
                if int(fold_perc*N[tsk]) + n_test < N[tsk]:
                    ### Own task; 
                    test1 = N_pts[tsk][int(fold_perc*N[tsk]):int(fold_perc*N[tsk]) + n_test]
                else:
                ### Do something else ###
                    test1 = N_pts[tsk][-n_test:]

                ### Training data is NOT this test data; 
                trn = np.array([j for i, j in enumerate(N_pts[tsk]) if j not in test1])

                ### Get test data from the other task; 
                other_task = np.mod(tsk + 1, 2); 
                n_test_other = N[other_task] - Ncomb_train

                if int(fold_perc*N[other_task]) + n_test_other < N[other_task]:
                    test2 = N_pts[other_task][int(fold_perc*N[other_task]):int(fold_perc*N[other_task]) + n_test_other]
                else:
                    test2 = N_pts[other_task][-n_test_other:]

                train_ix[i_f + tsk*n_folds] = trn; 
                test_ix[i_f + tsk*n_folds] = np.hstack((test1, test2))
                type_of_model[i_f + tsk*n_folds] = tsk

            else:
                ### Get all these points to test; 
                tst = N_pts[tsk][int(fold_perc*N[tsk]):int((fold_perc+(1./n_folds))*N[tsk])]
                
                #### Also add 20% of OTHER taks to testing; 
                other_task = np.mod(tsk + 1, 2); 
                tst2 = N_pts[other_task][int(fold_perc*N[other_task]):int((fold_perc+(1./n_folds))*N[other_task])]

                ### Add all non-testing points from this task to training; 
                trn = np.array([j for i, j in enumerate(N_pts[tsk]) if j not in tst])

                train_ix[i_f + tsk*n_folds] = trn.astype(int)
                test_ix[i_f + tsk*n_folds] = np.hstack((tst, tst2)).astype(int)
                type_of_model[i_f + tsk*n_folds] = tsk

        ### Now do the last part -- combined models with equal data from both tasks;  
        ### [6, 7, 8, 9, 10] ###
        test_ix[i_f + 2*n_folds] = []
        train_ix[i_f + 2*n_folds] = []; 

        ### Match this -- 
        if match_task_spec_n:
            Ncomb_train_gen = int(0.5*Ncomb_train)
        else:
            Ncomb_train_gen = Ncomb_train

        ### Task -- pull test/train points 
        for tsk in range(2):

            #### Total amount of testing data 
            Ncomb_test_tsk = N[tsk] - Ncomb_train_gen

            #### How big are the offsets for this to work? 
            if int(fold_perc*N[tsk]) + Ncomb_test_tsk < N[tsk]:
                test_ix[i_f + 2*n_folds].append(N_pts[tsk][int(fold_perc*N[tsk]):int(fold_perc*N[tsk]) + Ncomb_test_tsk])
            else:
                ### Just add the right amount of data from the end; 
                test_ix[i_f + 2*n_folds].append(N_pts[tsk][-Ncomb_test_tsk:])
        
        test_ix[i_f + 2*n_folds] = np.hstack((test_ix[i_f + 2*n_folds])).astype(int)
        train_ix[i_f + 2*n_folds] = np.array([i for i in range(Ntot) if i not in test_ix[i_f + 2*n_folds]]).astype(int)
        type_of_model[i_f + 2*n_folds] = 2

    return test_ix, train_ix, type_of_model.astype(int)

def get_training_testings_condition_spec(n_folds, data_temp):
    '''
    same as above, but with each train_ix for a specific target

    updated 5/18/20 -- removed the outside target testing; 
    '''
    
    train_ix = dict();
    test_ix = dict(); 
    type_of_model = np.zeros((20*n_folds)) - 1

    ### Get training and testing datasets: 
    N_pts = []; N = []; TSK_TG = []

    ### List which points are from which task: 
    for tsk in range(2):

        for targ in range(10):

            ### Get task specific indices; 
            ix = np.nonzero(np.logical_and(data_temp['tsk'] == tsk, data_temp['trg'] == targ))[0]

            if len(ix) > 0:
            
                ### Shuffle the task indices
                N_pts.append(ix[np.random.permutation(len(ix))])
                N.append(len(ix))

                ### Added task / target ###
                TSK_TG.append([tsk, targ])

    ##### For each fold ###
    for i_f, fold_perc in enumerate(np.arange(0., 1., 1./n_folds)):

        ### [1, 2, 3, 4, 5] ###
        train_ix[i_f] = []; 
        test_ix[i_f] = []; 

        #### Corresponds to task / target ####
        for i_t, (tsk, targ) in enumerate(TSK_TG):

            ### Get all these points to test; 
            tst = N_pts[i_t][int(fold_perc*N[i_t]):int((fold_perc+(1./n_folds))*N[i_t])]
            
            #### Also add 20% of OTHER targets to testing; 
            tst2 = []

            ### Skip other target testing; 
            # for i_t2 in range(len(TSK_TG)):
            #     if i_t2 != i_t: 
            #         tst2.append(N_pts[i_t2][int(fold_perc*N[i_t2]):int((fold_perc+(1./n_folds))*N[i_t2])])

            ### Add all non-testing points from this task to training; 
            trn = np.array([j for i, j in enumerate(N_pts[i_t]) if j not in tst])

            train_ix[i_f + n_folds*(tsk*10 + targ)] = trn; 
            test_ix[i_f + n_folds*(tsk*10 + targ)] = np.hstack((tst))#, np.hstack(( tst2)) ))
            type_of_model[i_f + n_folds*(tsk*10 + targ)] = tsk*10 + targ
    
    return test_ix, train_ix, type_of_model.astype(int)

#### GET Variable names from params #######
def lag_ix_2_var_nm(lag_ixs, var_name='vel', nneur=0, include_action_lags=False,
    model_nm = None):
    ### Get variable name
    ### Update 5/13/20 -- remove neur_lag = 0, entry; neural lags will use lag_ixs

    nms = []

    if var_name == 'psh':
        if model_nm is None:
            raise Exception('Cannot figure out whether to add push if no model nm')
        
        if 'psh_1' in model_nm:
     
            if include_action_lags:
                for l in lag_ixs:
                    if l<=0:
                        nms.append(var_name+'x_tm'+str(np.abs(l)))
                        nms.append(var_name+'y_tm'+str(np.abs(l)))
                    else:
                        nms.append(var_name+'x_tp'+str(np.abs(l)))
                        nms.append(var_name+'y_tp'+str(np.abs(l)))                                  
            else:
                nms.append(var_name+'x_tm0')
                nms.append(var_name+'y_tm0')   
        
        elif 'psh_2' in model_nm:
            print('Conditioning on push')
        
        else:
            print('No push')

    elif var_name == 'neur':
        for nl in lag_ixs:
            if nl <= 0:
                t = 'm'
            elif nl > 0:
                t = 'p'
            for n in range(nneur):
                nms.append('spk_t'+t+str(int(np.abs(nl)))+'_n'+str(n))

    elif var_name == 'tg':
        nms.append('trg_posx')
        nms.append('trg_posy')
    
    else:
        for l in lag_ixs:
            if l <=0: 
                nms.append(var_name+'y_tm'+str(np.abs(l)))
                nms.append(var_name+'x_tm'+str(np.abs(l)))
            else:
                nms.append(var_name+'y_tp'+str(np.abs(l)))
                nms.append(var_name+'x_tp'+str(np.abs(l)))            
    
    nm_str = ''
    for s in nms:
        nm_str = nm_str + s + '+'
    return nms, nm_str[:-1]

#### Reconstruct condition specific predictions
def reconst_spks_from_cond_spec_model(data, model_nm, ndays):
    ### Goal is to get out pred_spks
    reconst = {}

    for day in range(ndays):
        true_spks = data[day, 'spks']
        
        ix_container = []
        pred_container = [];

        pot_container = [];
        nul_container = []; 

        day_dat = data[day, model_nm]
        day_dat_null = data[day, model_nm, 'null']
        day_dat_pot = data[day, model_nm, 'pot']

        for tsk in range(2):
            for trg in range(10):
                if tuple((tsk*10 + trg, 'ix')) in day_dat.keys():
                    ix_ = np.hstack(( day_dat[tsk*10 + trg, 'ix']))
                    pred_ = np.vstack((day_dat[tsk*10 + trg, 'pred']))
                    assert(len(ix_) == pred_.shape[0])

                    pot_ix_ = np.hstack(( day_dat_pot[tsk*10 + trg, 'ix']))
                    pot_ = np.vstack((day_dat_pot[tsk*10 + trg, 'pred']))
                    
                    nul_ix_ = np.hstack(( day_dat_null[tsk*10 + trg, 'ix']))
                    nul_ = np.vstack((day_dat_null[tsk*10 + trg, 'pred']))
                    
                    assert(np.allclose(ix_, pot_ix_))
                    assert(np.allclose(ix_, nul_ix_))

                    ix_container.append(ix_)
                    pred_container.append(pred_)
                    pot_container.append(pot_)
                    nul_container.append(nul_)

        ix_container = np.hstack((ix_container))
        assert(len(np.unique(ix_container)) == true_spks.shape[0])
        assert(len(ix_container) == true_spks.shape[0])
        assert(np.allclose(np.sort(ix_container), np.arange(true_spks.shape[0])))

        pred_container = np.vstack((pred_container))
        pot_container = np.vstack((pot_container))
        nul_container = np.vstack((nul_container))
        assert(pred_container.shape == pot_container.shape == nul_container.shape)

        pred_spks = np.zeros_like(true_spks)
        pred_pot_spks = np.zeros_like(true_spks)
        pred_nul_spks = np.zeros_like(true_spks)
        
        pred_spks[ix_container, :] = pred_container.copy()
        pred_pot_spks[ix_container, :] = pot_container.copy()
        pred_nul_spks[ix_container, :] = nul_container.copy()

        reconst[day, model_nm] = pred_spks.copy()
        reconst[day, model_nm, 'pot'] = pred_pot_spks.copy()
        reconst[day, model_nm, 'null'] = pred_nul_spks.copy()

    return reconst

def quick_reg(bin_spk, decoder_all):
    BS = np.vstack((bin_spk))
    DA = np.vstack((decoder_all))
    if DA.shape[1] == 7:
        DA = DA[:, [3, 5]]
    elif DA.shape[1] == 2:
        pass
    else:
        import pdb; pdb.set_trace()
        raise Exception

    clf = Ridge(alpha=0.)
    clf.fit(DA, BS);
    est = clf.predict(DA)
    return util_fcns.get_R2(BS, est) 


###########################################
############### Shuffling #################
###########################################
def full_shuffling(bin_spk, decoder_all, animal, day_ix):
    '''
    update -- 6/12/20 -- also shuffled action with the neural activity; 
    update -- 6/16/20 -- this gets applied to data after being sorted/vstacked; Returns index only
    '''
    nT, nn = bin_spk.shape
    assert(decoder_all.shape[0] == nT)
    shuff_ix = np.random.permutation(nT)
    shuff_ix = shuff_ix.astype(int)
    #shuff_ix = within_bin_shuffling(spikes, push, animal, day_ix)
    
    #### 
    tmp_shuff = bin_spk[shuff_ix, :]
    tmp_dec_shuff = decoder_all[shuff_ix, :]

    assert(np.allclose(np.sum(bin_spk, axis=0), np.sum(tmp_shuff, axis=0)))
    assert(np.allclose(np.sum(decoder_all, axis=0), np.sum(tmp_dec_shuff, axis=0)))

    #### Get KG ###
    if animal == 'grom':
        _, KG = util_fcns.get_grom_decoder(day_ix)
        assert(np.allclose(tmp_dec_shuff, np.dot(KG, tmp_dec_shuff.T).T))

    return shuff_ix.astype(int)

def within_bin_shuffling(bin_spk, decoder_all, animal, day_ix):
    '''
    update -- 6/12/20 -- also shuffled action with the neural activity; 
    update -- 6/16/20 -- gets applied to data after combining across task, returns index only
    '''

    assert(bin_spk.shape[0] == decoder_all.shape[0])

    #### Mag boundaries #######
    mag_boundaries = pickle.load(open(analysis_config.config['grom_pref'] + 'radial_boundaries_fit_based_on_perc_feb_2019.pkl'))
    print('using animal %s, day_ix %d for mag boundaries' %(animal, day_ix))

    command_bins = util_fcns.commands2bins([decoder_all], mag_boundaries, animal, day_ix, vel_ix = [3, 5])[0]

    assert(bin_spk.shape[0] == command_bins.shape[0])

    #### Get KG ###
    if animal == 'grom':
        _, KG = util_fcns.get_grom_decoder(day_ix)
        #KG = KG[[3, 5], :]

    elif animal == 'jeev':
        KG_imp = util_fcns.get_jeev_decoder(day_ix)
        KG = np.zeros((7, KG_imp.shape[1]))
        KG[[3, 5], :] = KG_imp.copy()

    shuff_ix = np.zeros((bin_spk.shape[0])) - 1
    big_ix = []

    for i_m in range(4):
        for i_a in range(8):

            ### Get indices ######
            ix = np.nonzero(np.logical_and(command_bins[:, 0] == i_m, command_bins[:, 1] == i_a))[0]
            
            if len(ix) > 0:
                if len(big_ix) > 0:
                    for i in ix: assert(i not in np.hstack((big_ix)))

                ixi_sh = np.random.permutation(len(ix))

                ### Shuffle within bin; 
                big_ix.append(ix.copy())
                shuff_ix[ix] = ix[ixi_sh].copy()

    assert(np.all(shuff_ix >=0 ))
    shuff_ix = shuff_ix.astype(int)

    big_ix_sort = np.sort(np.hstack((big_ix)))
    assert(np.all(big_ix_sort == np.arange(shuff_ix.shape[0])))

    ### Make sure applying shuffling to command_bins doesn't change the bins; 
    assert(np.all(command_bins == command_bins[shuff_ix, :]))

    return shuff_ix, KG


####### MODEL ADDING ########
######### Call from generate_models.sweep_ridge_alpha #########
# h5file, model_, _ = generate_models_utils.h5_add_model(h5file, model_, i_d, first=i_d==0, model_nm=name, 
#     test_data = data_temp_dict_test, fold = i_fold, xvars = variables, predict_key = predict_key)

def h5_add_model(h5file, model_v, day_ix, first=False, model_nm=None, test_data=None, 
    fold = 0., xvars = None, predict_key='spks', only_potent_predictor = False, 
    KG_pot = None, KG = None, fit_task_specific_model_test_task_spec = False, fit_intercept = True):

    ##### Ridge Models ########
    if type(model_v) is sklearn.linear_model.ridge.Ridge or type(model_v[0]) is sklearn.linear_model.ridge.Ridge:
        # CLF/RIDGE models: 
        model_v, predictions = sklearn_mod_to_ols(model_v, test_data, xvars, predict_key, only_potent_predictor, KG_pot,
            KG, fit_task_specific_model_test_task_spec, fit_intercept = fit_intercept, model_nm = model_nm)
        
        if type(model_v) is list:
            nneurons = model_v[0].nneurons
        else:
            nneurons = model_v.nneurons
    
    else:
        raise Exception('Deprecated -- test OLS / CLS')
        # OLS / CLF models: 
        # nneurons = model_v.predict().shape[1]
        # model_v, predictions = add_params_to_mult(model_v, test_data, predict_key, only_potent_predictor, KG_pot)
    
    if h5file is None:
        pass
    else:
        if first:
            ### Make a new table / column; ####
            tab = h5file.createTable("/", model_nm+'_fold_'+str(int(fold)), Model_Table)
            col = h5file.createGroup(h5file.root, model_nm+'_fold_'+str(int(fold))+'_nms')

            try:
                vrs = np.array(model_v.coef_names, dtype=np.str)
                h5file.createArray(col, 'vars', vrs)
            except:
                print 'skipping adding varaible names'
        else:
            tab = getattr(h5file.root, model_nm+'_fold_'+str(int(fold)))
    
        #print nneurons, model_nm, 'day: ', day_ix

        for n in range(nneurons):
            row = tab.row
        
            #Add params: 
            #vrs = getattr(getattr(h5file.root, model_nm+'_fold_'+str(int(fold))+'_nms'), 'vars')[:]
            param = np.zeros((n_entries_hdf, ))
            pv = np.zeros((n_entries_hdf, ))
            for iv, v in enumerate(xvars):

                ######### RIDGE ########
                if fit_task_specific_model_test_task_spec:
                    param[iv] = model_v[0].coef_[n, iv]
                    pv[iv] = model_v[0].pvalues[n, iv]
                else:
                    param[iv] = model_v.coef_[n, iv]
                    pv[iv] = model_v.pvalues[n, iv]

                ######### OLS ########
                # try:
                #     # OLS: 
                #     param[iv] = model_v.params[n][v]
                #     pv[iv] = model_v.pvalues[iv, n]
                # except:

            row['param'] = param
            row['pvalue'] = pv
            row['day_ix'] = day_ix

            if fit_task_specific_model_test_task_spec:
                row['r2'] = model_v[0].rsquared[n]
                row['r2_pop'] = model_v[0].rsquared_pop
                row['aic'] = model_v[0].aic[n]
                row['bic'] = model_v[0].bic[n]
            else:
                row['r2'] = model_v.rsquared[n]
                row['r2_pop'] = model_v.rsquared_pop
                row['aic'] = model_v.aic[n]
                row['bic'] = model_v.bic[n]
            
            row.append()

    return h5file, model_v, predictions

def sklearn_mod_to_ols(model, test_data=None, x_var_names=None, predict_key='spks', only_potent_predictor=False, 
    KG_pot = None, KG = None, fit_task_specific_model_test_task_spec = False, testY = None, fit_intercept = True,
    model_nm = None):
    
    # Called from h5_add_model: 
    # model_v, predictions = sklearn_mod_to_ols(model_v, test_data, xvars, predict_key, only_potent_predictor, KG_pot,
    #     fit_task_specific_model_test_task_spec)
    
    x_test = [];

    for vr in x_var_names:
        x_test.append(test_data[vr][: , np.newaxis])
    X = np.mat(np.hstack((x_test)))
    
    if testY is None:
        Y = np.mat(test_data[predict_key])
    else:
        Y = np.mat(testY.copy())

    assert(X.shape[0] == Y.shape[0])
    assert(X.shape[1] == len(x_var_names))

    if fit_task_specific_model_test_task_spec:
        ix0 = np.nonzero(test_data['tsk'] == 0)[0]
        ix1 = np.nonzero(test_data['tsk'] == 1)[0]

        X0 = X[ix0, :]; Y0 = Y[ix0, :];
        X1 = X[ix1, :]; Y1 = Y[ix1, :]; 

        #### Get prediction -- why not use the predict method? #####
        if fit_intercept:
            pred0 = np.mat(X0)*np.mat(model[0].coef_).T + model[0].intercept_[np.newaxis, :]
            pred1 = np.mat(X1)*np.mat(model[1].coef_).T + model[1].intercept_[np.newaxis, :]
        else:
            pred0 = np.mat(X0)*np.mat(model[0].coef_).T 
            pred1 = np.mat(X1)*np.mat(model[1].coef_).T

        pred = pred0; 
        model1 = model[1]; 
        model =  model[0]; 

        ### Reduce size of HDF file by removing; 
        #model.X = X0; 
        #model.y = Y0; 

    #mport pdb; pdb.set_trace()

    ### From training; 
    # This is added in add_ridge
    # model.nneurons = model.y.shape[1]
    # model.nobs = model.X.shape[0]

    if only_potent_predictor:
        X = np.dot(KG_pot, X.T).T
        print 'only potent predictor'

    if fit_intercept:
        pred = np.mat(X)*np.mat(model.coef_).T + model.intercept_[np.newaxis, :]
    else:
        pred = np.mat(X)*np.mat(model.coef_).T
    
    pred_ = model.predict(X); 
    assert(np.all(pred == pred_))

    if 'psh_2' in model_nm: 
        
        A = []
        for v in ['pshx_tm0', 'pshy_tm0']: 
            A.append(test_data[v][:, np.newaxis])
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
        cov22I = np.linalg.inv(cov22)

        T = A.shape[0]

        pred_w_cond = []
        for i_t in range(T):

            ### Get this prediction (mu 1)
            mu1_i = pred[i_t, :].T

            ### Get predicted value of action; 
            mu2_i = np.dot(KG, mu1_i)

            ### Actual action; 
            a_i = A[i_t, :][:, np.newaxis]

            ### Conditon step; 
            mu1_2_i = mu1_i + np.dot(cov12, np.dot(cov22I, a_i - mu2_i))

            ### Make sure it matches; 
            assert(np.allclose(np.dot(KG, mu1_2_i), a_i))

            pred_w_cond.append(np.squeeze(np.array(mu1_2_i)))

        pred = np.vstack((pred_w_cond))

    ########## Get statistics ##################
    SSR = np.sum((np.array(pred - Y))**2, axis=0) 
    SSR_pop = np.sum((np.array(pred - Y))**2)

    dof = X.shape[0] - X.shape[1]
    sse = SSR / float(dof)

    ####### Neuron specific variance accounted for #####
    SST = np.sum(np.array( Y - np.mean(Y, axis=0))**2, axis=0 )
    SST_pop = np.sum(np.array( Y - np.mean(Y) )**2)
    
    ####### IF SST == 0 --> then set SST == SSR so that predictino shows up as perfect; 
    if len(np.nonzero(SST == 0)[0]) > 0:
        SST[np.nonzero(SST==0)[0]] = SSR[np.nonzero(SST==0)[0]]
    try:
        X2 = np.linalg.pinv(np.dot(X.T, X))
    except:
        raise Exception('Can do pinv fo X / X.T')

    ######### Not really sure what this is #######
    se = np.array([ np.sqrt(np.diagonal(sse[i] * X2)) for i in range(sse.shape[0]) ])
    model.t_ = np.mat(model.coef_) / se
    model.pvalues = 2 * (1 - scipy.stats.t.cdf(np.abs(model.t_), Y.shape[0] - X.shape[1]))
    
    ######## Vital ########
    model.rsquared = 1 - (SSR/SST)
    model.rsquared_pop = 1 - (SSR_pop/SST_pop)
    
    nobs2=model.nobs/2.0 # decimal point is critical here!
    llf = -np.log(SSR) * nobs2
    llf -= (1+np.log(np.pi/nobs2))*nobs2

    ####### AIC / BIC #########
    model.aic = -2 *llf + 2 * (X.shape[1] + 1)
    model.bic = -2 *llf + np.log(model.nobs) * (X.shape[1] + 1)
    
    if fit_task_specific_model_test_task_spec:
        return [model, model1], [pred0, pred1]
    else:
        return model, pred