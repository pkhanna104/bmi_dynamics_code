import analysis_config
import prelim_analysis as pa 
from resim_ppf import ppf_pa

import numpy as np 
import pandas
import copy
import pickle 
import tables 

import matplotlib.pyplot as plt

import sklearn.linear_model
import scipy.stats

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

        test_ix[i_f] = np.hstack((test_ix[i_f]))
        train_ix[i_f] = np.hstack((train_ix[i_f]))

        tmp = np.unique(np.hstack((test_ix[i_f], train_ix[i_f])))

        ### Make sure that unique point is same as all data
        assert len(tmp) == len(data_temp)
    return test_ix, train_ix 

def get_training_testings_generalization(n_folds, data_temp):
    ''' same as above, but now doing this for train CO, train OBS, train both '''

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
    Ntot = len(data_temp['tsk'])

    ##### For each fold ###
    for i_f, fold_perc in enumerate(np.arange(0., 1., 1./n_folds)):

        ### [1, 2, 3, 4, 5] ###
        train_ix[i_f] = []; 
        test_ix[i_f] = []; 

        #### For train CO:
        for tsk in range(2):    

            ### Get all these points to test; 
            tst = N_pts[tsk][int(fold_perc*N[tsk]):int((fold_perc+(1./n_folds))*N[tsk])]
            
            #### Also add 20% of OTHER taks to testing; 
            other_task = np.mod(tsk + 1, 2); 
            tst2 = N_pts[other_task][int(fold_perc*N[other_task]):int((fold_perc+(1./n_folds))*N[other_task])]

            ### Add all non-testing points from this task to training; 
            trn = np.array([j for i, j in enumerate(N_pts[tsk]) if j not in tst])

            train_ix[i_f + tsk*n_folds] = trn; 
            test_ix[i_f + tsk*n_folds] = np.hstack((tst, tst2))
            type_of_model[i_f + tsk*n_folds] = tsk

        ### Now do the last part -- combined models with equal data from both tasks;  
        ### [6, 7, 8, 9, 10] ###
        test_ix[i_f + 2*n_folds] = []
        train_ix[i_f + 2*n_folds] = []; 

        ### Amount of training data from each task; 
        Ncomb_train = int(0.8*Ncomb)
        
        ### Task -- pull test/train points 
        for tsk in range(2):

            #### Total amount of testing data 
            Ncomb_test_tsk = N[tsk] - Ncomb_train

            #### How big are the offsets for this to work? 
            if int(fold_perc*N[tsk]) + Ncomb_test_tsk < N[tsk]:
                test_ix[i_f + 2*n_folds].append(N_pts[tsk][int(fold_perc*N[tsk]):int(fold_perc*N[tsk]) + Ncomb_test_tsk])
            else:
                ### Just add the right amount of data from the end; 
                test_ix[i_f + 2*n_folds].append(N_pts[tsk][-Ncomb_test_tsk:])
        
        test_ix[i_f + 2*n_folds] = np.hstack((test_ix[i_f + 2*n_folds]))
        train_ix[i_f + 2*n_folds] = np.array([i for i in range(Ntot) if i not in test_ix[i_f + 2*n_folds]])
        type_of_model[i_f + 2*n_folds] = 2

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


####### MODEL ADDING ########
######### Call from generate_models.sweep_ridge_alpha #########
# h5file, model_, _ = generate_models_utils.h5_add_model(h5file, model_, i_d, first=i_d==0, model_nm=name, 
#     test_data = data_temp_dict_test, fold = i_fold, xvars = variables, predict_key = predict_key)

def h5_add_model(h5file, model_v, day_ix, first=False, model_nm=None, test_data=None, 
    fold = 0., xvars = None, predict_key='spks', only_potent_predictor = False, 
    KG_pot = None, fit_task_specific_model_test_task_spec = False):

    ##### Ridge Models ########
    if type(model_v) is sklearn.linear_model.ridge.Ridge or type(model_v[0]) is sklearn.linear_model.ridge.Ridge:
        # CLF/RIDGE models: 
        model_v, predictions = sklearn_mod_to_ols(model_v, test_data, xvars, predict_key, only_potent_predictor, KG_pot,
            fit_task_specific_model_test_task_spec)
        
        if type(model_v) is list:
            nneurons = model_v[0].nneurons
        else:
            nneurons = model_v.nneurons
    
    else:
        raise Exception('Deprecated -- test OLS / CLS')
        # OLS / CLF models: 
        # nneurons = model_v.predict().shape[1]
        # model_v, predictions = add_params_to_mult(model_v, test_data, predict_key, only_potent_predictor, KG_pot)
    
    
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
    KG_pot = None, fit_task_specific_model_test_task_spec = False):
    
    # Called from h5_add_model: 
    # model_v, predictions = sklearn_mod_to_ols(model_v, test_data, xvars, predict_key, only_potent_predictor, KG_pot,
    #     fit_task_specific_model_test_task_spec)
    
    x_test = [];

    for vr in x_var_names:
        x_test.append(test_data[vr][: , np.newaxis])
    X = np.mat(np.hstack((x_test)))
    Y = np.mat(test_data[predict_key])

    assert(X.shape[0] == Y.shape[0])
    assert(X.shape[1] == len(x_var_names))

    if fit_task_specific_model_test_task_spec:
        ix0 = np.nonzero(test_data['tsk'] == 0)[0]
        ix1 = np.nonzero(test_data['tsk'] == 1)[0]

        X0 = X[ix0, :]; Y0 = Y[ix0, :];
        X1 = X[ix1, :]; Y1 = Y[ix1, :]; 

        #### Get prediction -- why not use the predict method? #####
        pred0 = np.mat(X0)*np.mat(model[0].coef_).T + model[0].intercept_[np.newaxis, :]
        pred1 = np.mat(X1)*np.mat(model[1].coef_).T + model[1].intercept_[np.newaxis, :]

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

    pred = np.mat(X)*np.mat(model.coef_).T + model.intercept_[np.newaxis, :]
    pred_ = model.predict(X); 
    assert(np.all(pred == pred_))

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