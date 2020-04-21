''' Methods to extract both the behavioral (velocity command)
and neural PSTHs'''

import co_obs_tuning_matrices
import prelim_analysis as pa
import fcns
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tables
import math
import scipy.stats
from resim_ppf import ppf_pa

cmap = ['maroon', 'red', 'salmon', 'goldenrod', 'khaki']

def get_neural_zscore_by_day(input_type, animal):
    ZSC = dict()

    for i_d, day in enumerate(input_type):
        BS = []

        for i_t, tsk in enumerate(day):
            for t, te_num in enumerate(tsk):

                if animal == 'grom':
                    # elif on preeyas MBP
                    co_obs_dict = pickle.load(open(co_obs_tuning_matrices.pref+'co_obs_file_dict.pkl'))
                    hdf = co_obs_dict[te_num, 'hdf']
                    hdfix = hdf.rfind('/')
                    hdf = tables.openFile(co_obs_tuning_matrices.pref+hdf[hdfix:])

                    dec = co_obs_dict[te_num, 'dec']
                    decix = dec.rfind('/')
                    decoder = pickle.load(open(co_obs_tuning_matrices.pref+dec[decix:]))
                    F, KG = decoder.filt.get_sskf()

                    # Get trials: 
                    drives_neurons_ix0 = 3
                    key = 'spike_counts'
                    
                    rew_ix = np.array([t[1] for it, t in enumerate(hdf.root.task_msgs[:]) if t[0]=='reward'])

                    # decoder_all is now KG*spks
                    # decoder_all units (aka 'neural push') is in cm / sec
                    bin_spk, targ_i_all, targ_ix, trial_ix_all, decoder_all = pa.extract_trials_all(hdf, rew_ix, 
                        drives_neurons_ix0=drives_neurons_ix0, hdf_key=key, keep_trials_sep=True,
                        reach_tm_is_hdf_cursor_pos=False, reach_tm_is_kg_vel=True, **dict(kalman_gain=KG))

                elif animal == 'jeev':
                    bin_spk, targ_i_all, targ_ix, trial_ix_all, decoder_all, unbinned, exclude = ppf_pa.get_jeev_trials_from_task_data(te_num, binsize=.1)

                BS.extend(bin_spk)
        BS = np.vstack((BS))
        ZSC[i_d, 'mFR']= np.mean(BS, axis=0)
        ZSC[i_d, 'sdFR'] = np.std(BS, axis=0)
        ZSC[i_d, 'sdFR'][ZSC[i_d, 'sdFR'] == 0] = 1.

    # Save animal / day mFR / sdFR
    animal_dir = dict(grom='/grom2016/', jeev='/jeev2013/')
    pickle.dump(tsk, open('/Users/preeyakhanna/Dropbox/TimeMachineBackups'+animal_dir[animal]+'ZSC_values_7_16_18.pkl', 'wb'))

def get_all_snips(input_type, mag_thresh = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/radial_boundaries_fit_based_on_perc.pkl', 
    animal='grom', beh_or_neural='beh', save_fnm = None, save_beh_by_day=False, normalize_neur = False):
    
    CO_snips = {}
    OBS_snips = {}
    all_CO_snips = {}
    all_OBS_snips = {}
    
    kw = dict(normalize = normalize_neur)

    for i_d, day in enumerate(input_type):
        if type(mag_thresh) is str:
            mg_thr = pickle.load(open(mag_thresh))
            mt = np.array(mg_thr[animal, i_d])
        elif type(mag_thresh) is np.ndarray:
            mt = mag_thresh
        else:
            mt = mag_thresh

        for i_t, tsk in enumerate(day):
            for t, te_num in enumerate(tsk):
                snips_, mt = get_snips(te_num, mt, animal, beh_or_neural=beh_or_neural, **kw)
                if i_t == 1:
                    OBS_snips = add_snips(snips_, OBS_snips)
                elif i_t == 0:
                    CO_snips = add_snips(snips_, CO_snips)
        if beh_or_neural == 'neural':
            #plot_snips(CO_snips, OBS_snips, beh_or_neural)
            if save_fnm is not None:
                all_CO_snips[i_d] = CO_snips
                all_OBS_snips[i_d] = OBS_snips

            CO_snips = {}
            OBS_snips = {}
        
        elif np.logical_and(beh_or_neural == 'beh', save_beh_by_day):
            all_CO_snips[i_d] = CO_snips
            all_OBS_snips[i_d] = OBS_snips            
            CO_snips = {}
            OBS_snips = {}
    if beh_or_neural == 'beh':       
        plot_snips(all_CO_snips[i_d], all_OBS_snips[i_d], beh_or_neural)
    
    if save_fnm is not None:
        pickle.dump(all_CO_snips, open(save_fnm+'_CO.pkl', 'wb'))
        pickle.dump(all_OBS_snips, open(save_fnm+'_OBS.pkl', 'wb'))
       
def add_snips(snips_, master_snips):
    for k in master_snips.keys():
        if k in snips_.keys():
            master_snips[k].extend(snips_[k])
    for k in snips_.keys():
        if k not in master_snips.keys():
            master_snips[k] = snips_[k]
    return master_snips

def get_snips(te_num, mag_thresh, animal='grom', beh_or_neural='beh', return_ind_ix=False,
    exclude_ix = None, **kwargs):

    if type(mag_thresh) is str:
        mt = pickle.load(open(mag_thresh))
        mag_thresh = np.array(mt[animal, kwargs['day_ix']])

    if animal == 'grom':

        # Get SSKF for animal: 
        try:
            # if on arc: 
            print moose
            te = dbfn.TaskEntry(te_num)
            hdf = te.hdf
            decoder = te.decoder
        
        except:
            # elif on preeyas MBP
            co_obs_dict = pickle.load(open(co_obs_tuning_matrices.pref+'co_obs_file_dict.pkl'))
            hdf = co_obs_dict[te_num, 'hdf']
            hdfix = hdf.rfind('/')
            hdf = tables.openFile(co_obs_tuning_matrices.pref+hdf[hdfix:])

            dec = co_obs_dict[te_num, 'dec']
            decix = dec.rfind('/')
            decoder = pickle.load(open(co_obs_tuning_matrices.pref+dec[decix:]))
            F, KG = decoder.filt.get_sskf()

        # Get trials: 
        drives_neurons_ix0 = 3
        key = 'spike_counts'
        
        rew_ix = np.array([t[1] for it, t in enumerate(hdf.root.task_msgs[:]) if t[0]=='reward'])
        if exclude_ix is not None:
            rew_ix = np.array([rew_ix[i] for i in range(len(rew_ix)) if i not in exclude_ix])

        # decoder_all is now KG*spks
        # decoder_all units (aka 'neural push') is in cm / sec
        bin_spk, targ_i_all, targ_ix, trial_ix_all, decoder_all = pa.extract_trials_all(hdf, rew_ix, 
            drives_neurons_ix0=drives_neurons_ix0, hdf_key=key, keep_trials_sep=True,
            reach_tm_is_hdf_cursor_pos=False, reach_tm_is_kg_vel=True, **dict(kalman_gain=KG))

    elif animal == 'jeev':
        bin_spk, targ_i_all, targ_ix, trial_ix_all, decoder_all, unbinned, exclude = ppf_pa.get_jeev_trials_from_task_data(te_num, binsize=.1)

    ######################################
    ######################################
    # Get angle and magnitude of commands:

    dig_ = []
    snips_ = {}
    key_list = []
    
    if return_ind_ix:
        bin_ix = {} # Keys: (trial), bin, radial_ix, mag_ix

    # Get normalized mFR and sdFR:

    if beh_or_neural == 'neural':
        mFR = np.mean(np.vstack((bin_spk)), axis=0)
        sdFR = np.std(np.vstack((bin_spk)), axis=0)
        sdFR[sdFR == 0] = 1

    for dc, dec in enumerate(decoder_all):
        dec = np.array(dec)
        ang_mag = np.zeros((len(dec), 2))

        mag = np.sqrt(dec[:, 3]**2 + dec[:, 5]**2)
        ang = np.array([math.atan2(yi, xi) for i, (xi, yi) in enumerate(zip(dec[:, 3], dec[:, 5]))])
        ang[ang < 0] = ang[ang < 0] + 2*np.pi

        boundaries = np.linspace(0, 2*np.pi, 9)
        dig = np.digitize(ang, boundaries) - 1
        
        mag_dig = np.digitize(mag, mag_thresh)        
        # mag_dig = np.zeros_like(mag).astype(int)
        # mag_dig[np.logical_and(mag > mag_thresh[0], mag <= mag_thresh[1])] = int(1)
        # mag_dig[np.logical_and(mag > mag_thresh[1], mag <= mag_thresh[2])] = int(2)
        # mag_dig[mag > mag_thresh[2]] = int(3)

        if return_ind_ix:
            bin_ix[dc] = np.hstack((dig[:, np.newaxis], mag_dig[: , np.newaxis]))
                
        for i_d in range(4, len(dec)-5):
            key = str(dig[i_d])+'_'+str(mag_dig[i_d])
            
            if key not in key_list:
                key_list.append(key)
                snips_[key] = []

            if beh_or_neural == 'beh':
                snips_[key].append(dec[i_d-4:i_d+5, [3, 5]])

            elif beh_or_neural == 'neural':
                if kwargs['normalize']:
                    snippet = (bin_spk[dc][i_d-5:i_d+5, :] - mFR[np.newaxis, :]) / sdFR[np.newaxis, :]
                    if np.any(np.isnan(snippet)):
                        import pdb;
                        pdb.set_trace()
                else:
                    snippet = bin_spk[dc][i_d-5:i_d+5, :]

                snips_[key].append(snippet)

    if return_ind_ix:
        return snips_, mag_thresh, bin_ix
    else:
        return snips_, mag_thresh

def plot_snips(snips_1, snips_2, beh_or_neural):
    
    # Plot Stuff: 
    if beh_or_neural == 'neural':
        plot_neural_snips(snips_1, snips_2)

    elif beh_or_neural == 'beh':   
        nvel = 2

        for n in range(nvel):
            f, ax = plt.subplots(nrows=8, ncols=4)
            axi_cnt = 0
            for rad in range(8):
                for mag in range(4):
                    key = str(rad)+'_'+str(mag)
                    #axi = ax[mag+2*(rad/4), rad%4]
                    axi = ax[rad, mag]
                    try:
                        A1 = np.dstack((snips_1[key]))[:, n, :]
                        A2 = np.dstack((snips_2[key]))[:, n, :]

                        fcns.plot_mean_and_sem(range(A1.shape[0]), A1, axi, array_axis=1, color='blue')
                        fcns.plot_mean_and_sem(range(A2.shape[0]), A2, axi, array_axis=1, color='red')
                    #axi.set_ylim([-2., 2.])
                        axi.set_title('rad: '+str(rad)+'_mag: '+str(mag)+'_vel:'+str(n))
                    except:
                        print 'skipping: ', key

def plot_neural_snips(snips_1, snips_2):
    f, ax = plt.subplots(nrows=8, ncols=4)
    for rad in range(8):
        for mag in range(4):
            key = str(rad)+'_'+str(mag)
            axi = ax[rad, mag]
            try:
                A1 = np.mean(np.dstack((snips_1[key])), axis=2)
                A2 = np.mean(np.dstack((snips_2[key])), axis=2)
                dA = A1 - A2
                axi.pcolormesh(dA.T, vmin=-2, vmax=2., cmap='jet')
            except:
                print 'Skipping :', key
    #, ax = plt.subplots(n)

def plot_dB_vs_dN(save_fnm_beh, save_fnm_neural, only_middle_neur=True, animal='grom'):
    # only middle_neur refers to only comparing the neural activity at T = 0
    # confirm this!

    # Do by day
    dat_beh_CO = pickle.load(open(save_fnm_beh+'CO.pkl'))
    dat_beh_OBS = pickle.load(open(save_fnm_beh+'OBS.pkl'))
    dat_neu_CO = pickle.load(open(save_fnm_neural+'CO.pkl'))
    dat_neu_OBS = pickle.load(open(save_fnm_neural+'OBS.pkl'))

    days = dat_beh_CO.keys()
    f, ax = plt.subplots()
    f.set_figheight(5)
    f.set_figwidth(6)

    b = []
    n = []
    for i_d in days:
        db = []
        dn = []
        for rad in range(8):
            for mag in range(4):

                # Each point:
                key = str(rad)+'_'+str(mag)

                try:
                    # dB:
                    A1 = np.mean(np.dstack((dat_beh_CO[i_d][key])), axis=2)
                    A2 = np.mean(np.dstack((dat_beh_OBS[i_d][key])), axis=2)
                    dB = np.sum((A1 - A2)**2)

                    # dN:
                    try:
                        B1 = np.mean(np.dstack((dat_neu_CO[i_d][key])), axis=2)
                        B2 = np.mean(np.dstack((dat_neu_OBS[i_d][key])), axis=2)
                    
                    except:
                        mod1 = [i for i in dat_neu_CO[i_d][key] if i.shape[0] > 0]
                        mod2 = [i for i in dat_neu_OBS[i_d][key] if i.shape[0] > 0]

                        B1 = np.mean(np.dstack((mod1)), axis=2)
                        B2 = np.mean(np.dstack((mod2)), axis=2)
                         
                    if only_middle_neur:
                        # assumes the snippets are 9 bins long with following; 
                        # 0 | 1 | 2 | 3 | align ix (4) | 5 | 6 | 7 | 8
                        dN = np.sum((B1[4,:]-B2[4,:])**2)

                    else:
                        dN = np.sum((B1 - B2)**2)


                    if np.isnan(dN) or np.isnan(dB):
                        import pdb; pdb.set_trace()
                    ax.plot(np.log(dB), np.log(dN), 'k.')
                    db.append(dB)
                    dn.append(dN)
                except:
                    #print moose
                    print 'skipping: ', key
        slp, intc, rv, pv, ste = scipy.stats.linregress(db, dn)
        print ' day ', i_d, pv, slp
        b.extend(db)
        n.extend(dn)
    slp, intc, rv, pv, ste = scipy.stats.linregress(np.log(b), np.log(n))
    print ' all days: ', pv, slp, len(b), len(n)
    ax.set_xlabel('Log. Difference in Beh. PSTH, -400ms:400 ms', fontsize=14)
    ax.set_ylabel('Log. Difference in Mean Neural, 0 ms', fontsize=14)
    if animal == 'jeev':
        xhat = np.linspace(-1, 5, 100)
    elif animal == 'grom':
        xhat = np.linspace(-3, 6)
    yhat = slp*xhat + intc
    plt.plot(xhat, yhat, 'k-')
    ax.set_title(animal + ', p = '+str(pv))
    f.savefig('/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/BMI_co_obs_paper/figures/data_figs/'+animal+'_behPSTH-400:400ms_vs_norm_neuralPSTH_0ms.svg', transparent=True)

def plot_eg_behav_PSTHS(eg_day = 0, eg_key = '2_0', animal='jeev'):
    if animal == 'jeev':
        #co = pickle.load(open('/Volumes/TimeMachineBackups/jeev2013/jeev_9tm_snps_beh_by_day_mag_thresh_by_day__CO.pkl'))
        #obs = pickle.load(open('/Volumes/TimeMachineBackups/jeev2013/jeev_9tm_snps_beh_by_day_mag_thresh_by_day__OBS.pkl'))
        co = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/jeev_9tm_snps_beh_by_day_mag_thresh_by_day__CO.pkl'))
        obs = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/jeev_9tm_snps_beh_by_day_mag_thresh_by_day__OBS.pkl'))

    co_day = np.dstack(( co[eg_day][eg_key] ))
    obs_day = np.dstack(( obs[eg_day][eg_key] ))

    f, ax = plt.subplots()
    f.set_figheight(5)
    f.set_figwidth(6)

    from fa_analysis import fcns
    x1, ax = fcns.plot_mean_and_sem(np.arange(-.4, .5, .100), co_day[:, 0, :], ax, color='green', array_axis=1, label='Centerout, Vx')
    x2, ax = fcns.plot_mean_and_sem(np.arange(-.400, .500, .100), obs_day[:, 0, :], ax, color='blue', array_axis=1, label='Obstacle, Vx')
    ax.legend()
    ax.set_xlabel('Seconds')
    ax.set_ylabel('X - axis Velocity (cm/sec)')
    ax.set_xticks(np.arange(-.4, .5, .2))
    ax.set_xticklabels(np.arange(-.4, .5, .2))
    f.savefig('/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/BMI_co_obs_paper/figures/data_figs/'+animal+'_behPSTH-400:400ms_'+eg_key+'.svg', transparent=True)

# behav_neural_PSTH.get_all_snips(fk.task_filelist, save_fnm='/Volumes/TimeMachineBackups/jeev2013/jeev_9tm_snps_beh_by_day_mag_thresh_by_day_', animal='jeev', save_beh_by_day=True)
# behav_neural_PSTH.get_all_snips(fk.task_filelist, save_fnm='/Volumes/TimeMachineBackups/jeev2013/jeev_9tm_snps_neural_mag_thresh_by_day_', animal='jeev', beh_or_neural='neural', save_beh_by_day=True)
