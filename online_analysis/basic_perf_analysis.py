# plot rew / min, percent correct for CO  / OBS tasks
import analysis_config 
import pickle
import tables
import matplotlib.pyplot as plt
import numpy as np
import prelim_analysis as pa

def plot_metric(input_type, animal='grom'):
    rpm = {}
    pc = {}
    for i_d, day in enumerate(input_type):
        for i_t, tsk in enumerate(day):
            for te_num in tsk:
                if animal == 'grom':
                    # Get SSKF for animal: 
                    try:
                        # if on arc: 
                        te = dbfn.TaskEntry(te_num)
                        hdf = te.hdf
                        decoder = te.decoder
                    
                    except:
                        # elif on preeyas MBP
                        co_obs_dict = pickle.load(open(analysis_config.config['grom_pref']+'co_obs_file_dict.pkl'))
                        hdf = co_obs_dict[te_num, 'hdf']
                        hdfix = hdf.rfind('/')
                        hdf = tables.open_file(analysis_config.config['grom_pref']+hdf[hdfix:])

                        dec = co_obs_dict[te_num, 'dec']
                        decix = dec.rfind('/')
                        decoder = pickle.load(open(analysis_config.config['grom_pref']+dec[decix:]))
                        F, KG = decoder.filt.get_sskf()

                    # Get trials: 
                    drives_neurons_ix0 = 3
                    rew_ix = np.array([t[1] for it, t in enumerate(hdf.root.task_msgs[:]) if t[0]=='reward'])
                    err_ix = np.array([t[1] for it, t in enumerate(hdf.root.task_msgs[:]) if t[0] in ['hold_penalty', 'timeout_penalty']])
                    min_hdf = len(hdf.root.task)

                    rpm_ = len(rew_ix)/(min_hdf/60./60.)
                    perc_corr = len(rew_ix) / float(len(rew_ix)+len(err_ix))
                    try:
                        rpm[i_d, i_t].append(rpm_)
                        pc[i_d, i_t].append(perc_corr)
                    except:
                        rpm[i_d, i_t] = [rpm_]
                        pc[i_d, i_t] = [perc_corr]
    

    f, ax = plt.subplots()
    f2, ax2 = plt.subplots()
    
    for i_d, day in enumerate(input_type):
        
        if i_d == 1:
            l1 = ax.plot(i_d+np.zeros((len(rpm[i_d, 0]))), rpm[i_d, 0], 'ko', markersize=25, label='CO')
            l2 = ax.plot(i_d+np.zeros((len(rpm[i_d, 1]))), rpm[i_d, 1], 'bo', markersize=25, label='Obs')
            l11 = ax2.plot(i_d+np.zeros((len(pc[i_d, 0]))), pc[i_d, 0], 'ko', markersize=25, label='CO')
            l22 = ax2.plot(i_d+np.zeros((len(pc[i_d, 1]))), pc[i_d, 1], 'bo', markersize=25, label='Obs')

        else:
            ax.plot(i_d+np.zeros((len(rpm[i_d, 0])))+.1*np.random.rand(), rpm[i_d, 0], 'ko',markersize=25)
            ax.plot(i_d+np.zeros((len(rpm[i_d, 1])))+.1*np.random.rand(), rpm[i_d, 1], 'bo',markersize=25)
            ax2.plot(i_d+np.zeros((len(pc[i_d, 0])))+.1*np.random.rand(), pc[i_d, 0], 'ko',markersize=25)
            ax2.plot(i_d+np.zeros((len(pc[i_d, 1])))+.1*np.random.rand(), pc[i_d, 1], 'bo',markersize=25)
    for axi in [ax, ax2]:
        axi.legend(loc=4, frameon=True)
        axi.set_xlim([-.2, i_d+.4])
        axi.set_xlabel('Days')
    ax.set_title('Rew Per Min., Grom, Co and Obs Dataset')
    ax.set_ylabel('Rew Per Min.')
    ax2.set_title('Perc. Correct, Grom, Co and Obs Dataset')
    ax2.set_ylabel('Perc. Correct')

def example_tarj(input_type):

    ### Get color from config ###
    cmap_list = analysis_config.pref_colors
    day_list = range(len(input_type))

    for i_d in day_list:
        day = input_type[i_d]
        f, ax = plt.subplots(ncols=2, )
        f.set_figheight(5)
        f.set_figwidth(10)
        
        for i_t, (tsk, nm) in enumerate(zip(day, ['CO', 'Obs'])):

            for _, te_co in enumerate(tsk): 
                #te_co = tsk[0]
        
                ##### This dictionary has a list of HDF files / decoders mapping ########
                co_obs_dict = pickle.load(open(analysis_config.config['grom_pref']+'co_obs_file_dict.pkl'))
                hdf = co_obs_dict[te_co, 'hdf']
                hdfix = hdf.rfind('/')

                hdf = tables.open_file(analysis_config.config['grom_pref']+hdf[hdfix:])
                rew_ix = np.array([t[1] for it, t in enumerate(hdf.root.task_msgs[:]) if t[0]=='reward'])
                drives_neurons_ix0 = 3
                key = 'spike_counts'
                
                print('day %d, tsk %d' %(i_d, te_co))
                print hdf.root.task.attrs.target_radius - hdf.root.task.attrs.cursor_radius, hdf.root.task.attrs.target_radius, hdf.root.task.attrs.cursor_radius
                rad = hdf.root.task.attrs.target_radius - hdf.root.task.attrs.cursor_radius

                ###### Main extraction code ########
                bin_spk, targ_i_all, targ_ix, trial_ix_all, decoder_all = pa.extract_trials_all(hdf, rew_ix, neural_bins = 100.,
                    drives_neurons_ix0=drives_neurons_ix0, hdf_key=key, keep_trials_sep=True,
                    reach_tm_is_hdf_cursor_pos=True, reach_tm_is_kg_vel=False)

                tm = np.linspace(0, 2*np.pi, 1000)
                xtm = rad*np.cos(tm)
                ytm = rad*np.sin(tm)

                cnt = 0
                for ib, bs in enumerate(decoder_all):
                    ### Get target 
                    trlix = np.nonzero(trial_ix_all == ib)[0]

                    ### First target 
                    tg_pos = targ_i_all[trlix, :][4, :]
                    ax[i_t].plot(tg_pos[0] + xtm, tg_pos[1] + ytm, '-', color = cmap_list[int(targ_ix[cnt])])
                    
                    #if ib < 64:
                    ax[i_t].plot(bs[:, 0], bs[:, 1], '-', color=cmap_list[int(targ_ix[cnt])])
                    cnt += len(bs)
                    ax[i_t].set_title(nm)

