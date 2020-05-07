import analysis_config
import prelim_analysis as pa 
from resim_ppf import ppf_pa

import numpy as np 
import pandas
import copy
import pickle 
import tables 

import matplotlib.pyplot as plt

pref = analysis_config.config['grom_pref']

#### Methods to get data needed for spike models ####
def get_spike_kinematics(animal, day, order, history_bins, **kwargs):
    if 'trial_ix' in kwargs:
        trial_ix = kwargs['trial_ix']
    else:
        trial_ix = None

    spks = []
    vel = []
    push = []
    pos = []
    tsk = []

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

    ### Not sure what these are 
    trial_ord = {}

    ### Trial order -- which order in the day did this thing happen? 
    trl_rd = []

    ### For each task: 
    for i_t, tsk_fname in enumerate(day):

        ### Start the trial count at zero 
        tsk_trl_cnt = 0

        ### For each task filename 
        for i, te_num in enumerate(tsk_fname):

            ### Get out the target indices and target position for each trial 
            trg_trl = []; trg_trl_pos = []; 

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
                trl_rd.append(order[i_t][i])     
                
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

                trl_rd.append(order[i_t][i])
                
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
                ix_x = np.nonzero(trial_ix_all == ix_)[0][4] # choose an index, not all indices

                ### Get the target index for this trial -- if it is -1 then remove this trial 
                if targ_ix[ix_x] < 0:
                    rm_trls.append(ix_)

                ### Keep so that indexing stays consistent
                trg_trl.append(targ_ix[ix_x]) ## #Get the right target for this trial
                trg_trl_pos.append(targ_i_all[ix_x, :])

            ### Remove trials that are -1: 
            ### Keep these trials ####
            ix_mod = np.array([iii for iii in ix_mod if iii not in rm_trls])

            # Add to list: 
            ### Stack up the binned spikes; 
            spks.append(np.vstack(([bin_spk[x] for x in ix_mod])))
            
            # We need the real velocity: 
            vel.append(np.vstack(([cursor_vel[x] for x in ix_mod])))

            # Get the push too? 
            push.append(np.vstack(([decoder_all[x] for x in ix_mod])))

            # Not the Kalman Gain: 
            tsk.append(np.zeros(( np.vstack(([bin_spk[x] for x in ix_mod])).shape[0], )) + i_t)
            pos.append(np.vstack(([cursor_pos[x] for x in ix_mod])))

            # Add to temporal timing model
            for t in range(len(bin_spk)):
                if t in ix_mod:
                    trl = bin_spk[t]
                    nt, nneurons = trl.shape

                    ### Modified on 9/13/19
                    spk = np.zeros((nt-(2*history_bins), (1+(2*history_bins)), nneurons))
                    ### EOM
                    spk0 = np.zeros((nt - (2*history_bins), nneurons))

                    velo = np.zeros((nt-(2*history_bins), 2*(1+(2*history_bins))))
                    poso = np.zeros_like(velo)

                    ### Modified 9/16/19 -- adding lags to push
                    pusho = np.zeros_like(velo)
                    ### EOM

                    task = np.zeros((nt-(2*history_bins), ))+i_t

                    ### So this needs to 
                    targ = np.zeros((nt-(2*history_bins), ))+trg_trl[t]
                    targ_pos = np.zeros((nt - (2*history_bins), 2)) + trg_trl_pos[t]

                    ix_ = np.zeros((len(task), 2))# + t
                    ix_[:, 0] = t + tsk_trl_cnt
                    ix_[:, 1] = order[i_t][i] # Order

                    XY_div = (2*history_bins)+1

                    ### For each time bin that we can actually capture: 
                    for tmi, tm in enumerate(np.arange(history_bins, nt-history_bins)):
                        
                        ### for the individual time bins; 
                        for tmpi, tmpii in enumerate(np.arange(tm - history_bins, tm + history_bins + 1)):
                            spk[tmi, tmpi, :] = trl[tmpii, :]

                        spk0[tmi, :] = trl[tm, :]
                        velo[tmi, :XY_div] = np.squeeze(np.array(cursor_vel[t][tm-history_bins:tm+history_bins+1, 0]))
                        velo[tmi, XY_div:] = np.squeeze(np.array(cursor_vel[t][tm-history_bins:tm+history_bins+1, 1]))
                        
                        poso[tmi, :XY_div] = np.squeeze(np.array(cursor_pos[t][tm-history_bins:tm+history_bins+1, 0]))
                        poso[tmi, XY_div:] = np.squeeze(np.array(cursor_pos[t][tm-history_bins:tm+history_bins+1, 1]))
                        
                        pusho[tmi, :XY_div] = np.squeeze(np.array(decoder_all[t][tm-history_bins:tm+history_bins+1, 3]))
                        pusho[tmi, XY_div:] = np.squeeze(np.array(decoder_all[t][tm-history_bins:tm+history_bins+1, 5]))
                    
                        
                    temp_tune['spk0'].append(spk0)
                    temp_tune['spks'].append(spk)
                    temp_tune['vel'].append(velo)
                    temp_tune['tsk'].append(task)
                    temp_tune['pos'].append(poso)
                    temp_tune['trl'].append(ix_)
                    temp_tune['trg'].append(targ)
                    temp_tune['trg_pos'].append(targ_pos)
                    temp_tune['psh'].append(pusho)
                    temp_tune['trl_bin_ix'].append(np.arange(history_bins, nt-history_bins))

            tsk_trl_cnt += len(bin_spk)
    
    ################################################
    #### STACK UP EVERYTHING WITHOUT TIME SHIFTS ###
    ################################################
    spikes = np.vstack((spks))
    velocity = np.vstack((vel))

    if velocity.shape[1] == 7:
        velocity = np.array(velocity[:, [3, 5]])
    
    position = np.vstack((pos))
    task_index = np.hstack((tsk))
    push = np.vstack((push))


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
    
    ##################################
    ### MODEL with HISTORY +FUTURE ###
    ##################################
    data_temp = pandas.DataFrame(tmp_dict)
    plot_data_temp(data_temp, animal, True)

    return data, data_temp, sub_spk0_temp_all, sub_spk_temp_all, sub_push_all


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
        targs = np.unique(data_temp['trg'][tsk_ix])

        for itr, tr in enumerate(targs):
            ## Get the trial numbers: 
            targ_ix = np.nonzero(data_temp['trg'][tsk_ix] == tr)[0]
            trls = np.unique(data_temp['trl'][tsk_ix[targ_ix]])

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
    import pdb; pdb.set_trace()

