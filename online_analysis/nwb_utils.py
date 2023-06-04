## has to be run in nwb37 conda environment
### had to install matplotlib, pytables, statsmodels

### In BMI3D had to: 
### had to comment out: 
### bmi.py --> from riglib.plexon import Spikes 
### kfdecoder.py --> 

# Add 
# from . import bmi
# from . import train

# Comment 
# # import bmi 
# # import train


################### OVERVIEW OF NWB FILE #####################
## File 
## session datetime
## session ID 

## Subject (Subject)
## subject ID
## age 
## description 
## species 
## sex 

###### BMI timeseries 
# behavior_module = nwbfile.create_processing_module(
#     name="behavior", description="processed behavioral data"
# )

# The first dimension in 'data' must be 'time'#
# time_series_with_rate = TimeSeries(
#     name="test_timeseries",
#     data=data,
#     unit="m",
#     starting_time=0.0,
#     rate=1.0,
# )
# behavior_module.add(time_series_with_rate)

## spk_cnts -- t x Nunits 
## decoder_state -- t x 5
## cursor state -- t x 5
## target state -- t x 5 
## obs state -- t x 2
## obs shape -- t x 1 (text -- 2cm_sq, 3cm_sq, )
## bin number (state) -- t x 1 
## te_num -- t x 1 (task entry number)

###### Analyzed trials 
## id, start_time, stop_time

## nwbfile.add_trial_column(
#     name="moveID3",
#     description="which target condition is on",
#     #data = np.array([102.1, 102.2])
# )
# nwbfile.trials.to_dataframe()

######## Decoder matrices --> timeseries w/ 1 timestamp
## A, W, C, Q  
## steady state F, K 

####### Task params 
# Target radius 
# Cursor radius 
# plant_type -- text 
# reward_time

import analysis_config 
from resim_ppf import ppf_pa
from resim_ppf import file_key as fk

import numpy as np
import pickle
import tables
import scipy 
import scipy.io as sio 

from pynwb import NWBFile, TimeSeries, NWBHDF5IO
from pynwb.epoch import TimeIntervals
from pynwb.file import Subject

from datetime import datetime

### From kathy // carmena water log (https://www.dropbox.com/search/personal?path=%2F&preview=id%3AVbS5Qml0S2oAAAAAAAAE0Q&query=water&search_token=1ZdYHO64bES8yd8BZ8Aqg2pHbrl6CVzTLogdyAIefkk%3D&typeahead_session_id=51864925420230343626450059946156)
# mmu 38380 was born on 05/13/2007 in an outdoor colony  (north) -- Gromit 
# mmu 35195 was born on 05/25/2003 in an outdoor colony (north) -- Jeeves 

def get_monkeyG_dates(): 

    ### Get datetimes for start of blocks: (source: google doc, Gromit Behavior)
    monk_g_dt = [];
    days = [2, 4, 7, 15, 16, 17, 18, 19, 19]
    for monkg_blk_i in range(9): 
        dti = datetime(2016, 3, days[monkg_blk_i])
        monk_g_dt.append(dti)
    return monk_g_dt 

def get_monkeyJ_dates(): 

    ### Get datetimes for start of blocks: (source: analysis_config.py)
    monk_j_dt = []; 
    days = [24, 24, 29, 31]
    for monkj_blk_i in range(4): 
        dti = datetime(2013, 8, days[monkj_blk_i])
        monk_j_dt.append(dti)
    return monk_j_dt

def write_NWB_file_monkeyG(test=True): 
    '''
    See above for specifications 
    ''' 

    ### For monkey G: 
    animal = 'grom';
    pkl_kw = dict(encoding='latin1')

    order_dict = analysis_config.data_params['%s_ordered_input_type'%animal]
    input_type = analysis_config.data_params['%s_input_type'%animal]

    pos_ix = np.array([0, 2]) ## Get x/y position 
    state_ix = np.array([0, 2, 3, 5, 6])## Get x/y position and vel and offset 

    hdf_fs = 60.

    monk_g_dt = get_monkeyG_dates()

    ndays = 9
    if test: 
        ndays = 1

    for day_ix in range(ndays): 

        # Get day TEs 
        day = np.hstack((input_type[day_ix]))

        ## Start NWB file 
        nwbfile = NWBFile(
            session_description="Monkey performing 2D cursor BMI",  # required
            identifier="MonkeyG_session%d"%day_ix,  # required
            session_start_time=monk_g_dt[day_ix],  # required
            session_id="session%d"%day_ix,  # optional
            experimenter="Khanna, Preeya",  # optional
            lab="Carmena lab",  # optional
            institution="UC Berkeley",  # optional
            experiment_description='Single unit recordings from chronically implanted microwire \
                electrode array in PMd/M1 used for BMI control using Kalman filter decoder. File includes BMI-unit spike counts, \
                BMI task parameters, and BMI cursor data used for analysis. TimeSeries are reported at 60 hz, BMI update rate was 10 Hz. \
                Raw electrophysiolgy data files are not included',
            keywords=['BMI control', 'kalman filter', 'chronic electrophysiolgy', 'motor cortex', 'premotor cortex', 'microwire arrays', 'monkey']
        )
    
        ### Now add subject information 
        nwbfile.subject = Subject(
            subject_id="monk_g",
            age="P8Y", # Age in march 2016 
            description="mmu 38380",
            species="Macaca mulatta",
            sex="M"
        )
    
        ### Now add a behavior module
        behavior_module = nwbfile.create_processing_module(
            name="behavior", description="BMI spike counts and behavior")
        
        ### Gather data ### 
        spk_cnts = []; 
        update_bmi = []; 
        cursor_pos = []
        decoder_state = []; 
        target_state = []; 
        obs_pos = []; 
        obs_shape = []; 
        te_num = []; 
        hdf_offset = 0; 
        
        trials = []; 
        
        ### Cycle through days #### 
        for te in day: 
            
            ### Open up CO / OBS ###
            pref = analysis_config.config['%s_pref'%animal]
            co_obs_dict = pickle.load(open(pref+'co_obs_file_dict.pkl', 'rb'), **pkl_kw)
            hdf = co_obs_dict[te, 'hdf']
            hdfix = hdf.rfind('/')
            hdf = tables.open_file(pref+hdf[hdfix:])
            
            ### Length of HDF task #####
            len_hdf = len(hdf.root.task)
            
            ### Spike counts 
            spk_cnts.append(hdf.root.task[:]['spike_counts'])
            
            ### Get updated indices 
            internal_state = hdf.root.task[:]['internal_decoder_state']
            drives_neurons_ix0 = 3; 
            update_bmi_ix = np.nonzero(np.diff(np.squeeze(internal_state[:, drives_neurons_ix0, 0])))[0]+1
            update_bmi_i = np.zeros(len_hdf, )
            update_bmi_i[update_bmi_ix] = 1. 
            update_bmi.append(update_bmi_i[:, np.newaxis])
            
            ### Cusror pos
            cursor_pos.append(hdf.root.task[:]['cursor'][:, pos_ix])
            
            ### Decoder state: 
            decoder_state.append(hdf.root.task[:]['internal_decoder_state'][:, :, 0][:, state_ix])
            
            ### Target state: 
            target_state.append(hdf.root.task[:]['target_state'][:, :, 0][:, state_ix])
            
            ### obs state: 
            if 'obstacle_location' in hdf.root.task.colnames: 
                obs_pos.append(hdf.root.task[:]['obstacle_location'][:, pos_ix])

                ### obs shape: 
                shape = hdf.root.task[:]['obstacle_size'][:, 0]
                shape2 = np.array(['%d_cm_sq'%int(s) for s in shape])
                obs_shape.append(shape2[:, np.newaxis])
            
            else: 
                obs_pos.append(np.zeros((len_hdf, 2)) + np.nan)
                
                shape2 = np.array(['na' for i in range(len_hdf)])
                obs_shape.append(shape2[:, np.newaxis])
                
            ## te_num
            te_num.append(np.zeros((len_hdf, 1)) + te)
            
            ### Get trials to analyze 
            reward_ix = np.array([t[1] for it, t in enumerate(hdf.root.task_msgs[:]) if t[0]==b'reward'])
            
            ### Go index 
            go_ix = np.array([hdf.root.task_msgs[it-3][1] for it, t in enumerate(hdf.root.task_msgs[:]) if 
                scipy.logical_and(t[0] == b'reward', t[1] in reward_ix)])
            
            ### Assertions 
            assert(np.all(s[0] == b'target' for s in hdf.root.task_msgs[:] if s[1] in go_ix))
            assert(len([s[0] for s in hdf.root.task_msgs[:] if s[1] in go_ix and s[0] == b'target']) == len(go_ix))
            assert(np.all(s[0] == b'reward' for s in hdf.root.task_msgs[:] if s[1] in rew_ix))
            assert(len([s[0] for s in hdf.root.task_msgs[:] if s[1] in reward_ix and s[0] == b'reward']) == len(reward_ix))

            reward_ix = reward_ix + hdf_offset
            go_ix = go_ix + hdf_offset
            assert(len(reward_ix) == len(go_ix))
            for _, (gix, rix) in enumerate(zip(go_ix, reward_ix)): 
                assert(rix > gix)
                trials.append([gix / hdf_fs, rix / hdf_fs])
            
            ### ADD HDF OFFSET 
            hdf_offset = hdf_offset + len_hdf
            
            #### ADD DECODER and TASK PARAMETERS #####
            dec = co_obs_dict[te, 'dec']
            decix = dec.rfind('/')
            decoder = pickle.load(open(pref+dec[decix:], 'rb'), **pkl_kw)
            F, KG = decoder.filt.get_sskf()
            F = F[np.ix_(state_ix, state_ix)]
            KG = KG[state_ix, :]

            A = decoder.filt.A[np.ix_(state_ix, state_ix)]
            W = decoder.filt.W[np.ix_(state_ix, state_ix)]
            C = decoder.filt.C[:, state_ix]
            Q = decoder.filt.Q
            
            mats = [A[np.newaxis, :, :], W[np.newaxis, :, :], C[np.newaxis, :, :], Q[np.newaxis, :, :], 
            F[np.newaxis, :, :], KG[np.newaxis, :, :], 
                    np.array([hdf.root.task.attrs.target_radius])[np.newaxis, :], 
                    np.array([hdf.root.task.attrs.cursor_radius])[np.newaxis, :], 
                    np.array([hdf.root.task.attrs.plant_type])[np.newaxis, :], 
                    np.array([hdf.root.task.attrs.reward_time])[np.newaxis, :]]
            
            names = ['decoder_A', 'decoder_W', 'decoder_C','decoder_Q', 'decoder_ssF', 'decoder_ssKG',
                   'target_radius', 'cursor_radius', 'plant_type', 'reward time (sec)']
            descs = ['kalman filter decoder A matrix (cursor dynamics, 5 x 5 )', 
                     'kalman filter decoder W matrix (noise of cursor dynamics 5 x 5 )', 
                     'kalman filter deocder C matrix (neural encoding of cursor 5 x Nneurons)', 
                     'kalman filter deocder Q matrix (noise of cusror encoding Nneurons x Nneurons)',
                     'kalman filter steady state F matrix (cursor dynamics 5 x 5)', 
                     'kalman filter steady state K matrix (kalman gain, neural update 5 x Nneurons)', 
                     'target radius (cm)',
                     'cursor radius (cm)', 
                     'plant type (name of plant corresponding to BMI3D code)',
                     'reward time (sec) -- how long juicer was open to deliver rewards']

            ### Commenting to allow later upload 
            # for _, (mat, name, desc) in enumerate(zip(mats, names, descs)): 
                
            #     ### Add decoder and task params to file 
            #     timestamps = [0]
            #     ts_w_timestamps = TimeSeries(
            #         name="teblk_%d_%s"%(te, name),
            #         data=mat,
            #         unit="a.u.",
            #         timestamps=timestamps, 
            #         description = desc)
            #     behavior_module.add(ts_w_timestamps)

        ### now add these to NWB file 
        for _, (met, met_name, met_desc, met_unit) in enumerate(zip([spk_cnts, update_bmi, cursor_pos, decoder_state, 
             target_state, obs_pos, obs_shape, te_num], 

                                                ['spike_counts', 'update_bmi', 
                                                'cursor', 
                                                 'decoder_state', 
                                                 'target_state', 
                                                'obstacle_position',
                                                'obs_details',
                                                'te_num'], 

                                                ['binned spike counts used for BMI control', 
                                                'binary variable where "1" indicates bins which BMI was updated (10 hz, task runs at 60 hz)',
                                                '2D cursor position (x, y) ', 
                                                'decoder state (2d-pos, 2d-vel, offset)', 
                                                'target state (2d-pos, 2d-vel, offset) -- location of target', 
                                                'position of obstacle (center of square, is (0,0) if no obstacle)', 
                                                'description of obstacle shape (square, and side length of square in cm, is "na" if no obstacle', 
                                                'task entry number for trials (corresponds to task ID in bmi3d db.sql file'], 

                                                ['cnt', 'binary indicator', 'cm', 'cm, cm/sec', 'cm', 'cm', 'text','int'])): 
            assert(np.vstack((met)).shape[0] == hdf_offset)
            ts_w_rate = TimeSeries(name = met_name, 
                                    data = np.squeeze(np.vstack((met))), 
                                    unit = met_unit, 
                                    starting_time = 0.0, 
                                    rate = hdf_fs,
                                    description = met_desc)
            ### Add to behavior 
            behavior_module.add(ts_w_rate)

        ### Add trials to NWB file 
        for _, (go_ts, rew_ts) in enumerate(trials): 
            nwbfile.add_trial(start_time=go_ts, stop_time=rew_ts)


        ### Write NWB file 
        io = NWBHDF5IO("/Users/preeyakhanna/bmi_dynamics_code/nwb_notebooks/nwb_files/monkeyG-session%d.nwb"%day_ix, mode="w")
        io.write(nwbfile)
        io.close()

def write_NWB_file_monkeyJ(test=True): 
    
    ### For monkey J: 
    animal = 'jeev';
    pkl_kw = dict(encoding='latin1')

    order_dict = analysis_config.data_params['%s_ordered_input_type'%animal]
    input_type = analysis_config.data_params['%s_input_type'%animal]
    input_type2 = fk.task_input_type

    task_directory_mbp = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/'
    amy_data = sio.loadmat('/Users/preeyakhanna/bmi_dynamics_code/resim_ppf/jeev_obs_positions_from_amy.mat')

    monk_j_dt = get_monkeyJ_dates()

    ndays = 4
    if test: 
        ndays = 1

    for day_ix in range(ndays): 

        # Get day TEs 
        day = np.hstack((input_type[day_ix]))
        day2 = np.hstack((input_type2[day_ix]))

        ## Start NWB file 
        nwbfile = NWBFile(
            session_description="Monkey performing 2D cursor BMI",  # required
            identifier="MonkeyJ_session%d"%day_ix,  # required
            session_start_time=monk_j_dt[day_ix],  # required
            session_id="session%d"%day_ix,  # optional
            experimenter="Orsborn, Amy",  # optional
            lab="Carmena lab",  # optional
            institution="UC Berkeley",  # optional
            experiment_description='Single unit recordings from chronically implanted microwire \
                electrode array in PMd/M1 used for BMI control using Kalman filter decoder. File includes BMI-unit spike counts, \
                BMI task parameters, and BMI cursor data used for analysis. TimeSeries are reported at 200 hz, BMI update rate was 200 Hz. \
                Raw electrophysiolgy data files are not included',
            keywords=['BMI control', 'point process filter', 'chronic electrophysiolgy', 'motor cortex', 'premotor cortex', 'microwire arrays', 'monkey']
        )
    
        ### Now add subject information 
        nwbfile.subject = Subject(
            subject_id="monk_j",
            age="P12Y", # Age in march 2016 
            description="mmu 35195",
            species="Macaca mulatta",
            sex="M"
        )
    
        ### Now add a behavior module
        behavior_module = nwbfile.create_processing_module(
            name="behavior", description="BMI spike counts and behavior")
        
        ### Gather data ### 
        spk_cnts = []; 
        update_bmi = []; 
        cursor_pos = []
        decoder_state = []; 
        target_state = []; 
        obs_pos = []; 
        obs_shape = []; 
        te_num = []; 
        dat_offset = 0; 

        file_fs = 200.; 
        
        trials = []; 

        ### estimated KG 
        pkl_kw = dict(encoding='latin1')
        dat_KG_est= pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/jeev_KG_approx_fit.pkl', 'rb'), **pkl_kw)
        
        ### Cycle through days #### 
        for ite, te in enumerate(day): 
            
            ### open up data file ###
            dat = sio.loadmat(task_directory_mbp + te)

            ### Start indices 
            if 'jeev082413_VFB_PPF_B100_NS5_NU20_Z1_assist_ofc_cont_cont_assist_ofc_fixData' in te:
                start_index_overall = 55003
            else:
                start_index_overall = 0
            
            ### File length 
            dat_len = dat['spike_counts'][:, start_index_overall:].shape[1]

            ### Make strobed 
            strobed = ppf_pa.make_strobed(dat, start_index_overall)

            ### Get trials -- copied from ppf_pa  
            rew_ix = np.nonzero(strobed[:, 1]==9)[0]
            go_ix = rew_ix - 3
            ix = np.nonzero(strobed[go_ix, 1] == 5)[0]
            ix2 = np.nonzero(strobed[go_ix-1, 1] == 15)[0]
            ix_f = np.intersect1d(ix, ix2)

            rew_ix = rew_ix[ix_f]
            go_ix = go_ix[ix_f]

            #### Make sure all 'go indices' 5s. 
            assert np.sum(np.squeeze(strobed[go_ix, 1] - 5)) == 0
            trial_indices_all = list(zip(strobed[go_ix, 0], strobed[rew_ix, 0]))

            # Ensure only indices > start_index_overall: 
            trial_indices = []
            for ii, (go, rew) in enumerate(trial_indices_all):
                if np.logical_and(go > start_index_overall, rew > start_index_overall):
                    trial_indices.append(np.array([go-start_index_overall + dat_offset, rew-start_index_overall + dat_offset])/file_fs)
            trial_indices = np.vstack((trial_indices))
            trials.append(trial_indices)

            
            ### Spike counts 
            spk_cnts.append(dat['spike_counts'][:, start_index_overall:].T) # nneurons x Time --> hence transpose
            spk_counts_dt = float(dat['Delta_PPF'])
            assert spk_counts_dt == 0.005

            ### Cursor pos / velocity 
            
            ### Get updated indices 
            update_bmi.append(np.ones((dat_len, 1)))
            
            ### Cusror pos
            ix_pos = np.ix_([0, 1], np.arange(start_index_overall, dat['spike_counts'].shape[1]))
            cpos = dat['cursor_kin'][ix_pos].T 
            assert(cpos.shape[0] == dat_len)
            cursor_pos.append(dat['cursor_kin'][ix_pos].T)
            
            ### Decoder state: 
            decoder_state.append(np.hstack((dat['cursor_kin'][:, start_index_overall:].T, np.ones((dat_len, 1)))))
            
            ### Target state: 
            start_ix = strobed[go_ix - 3, 1]
            start_ix[start_ix == 400] = 2; # set start indices from 400 --> 2
            start_ix[start_ix == 15] = 2;# set start indices from 400 --> 2
            
            if np.sum(np.squeeze(start_ix - 2)) == 0:
                task = 'co'
                targ = strobed[go_ix - 2, 1]
                targ_ix = targ - 64 # get target indices 

                # Make sure target indices line up with targets 
                assert np.sum(fk.cotrialList[targ_ix] - targ) == 0
                
                # Go through go indices and set targets for all current --> last 
                targets = np.zeros((dat_len, 5))
                targets[:, -1] = 1. 

                for _, (go_, tg_ix) in enumerate(zip(go_ix, targ_ix)): 
                    targets[(go_ - start_index_overall):, [0, 1]] = fk.targetPos[tg_ix, :]

                obs_state = np.zeros((dat_len, 5)) + np.nan

            else:
                task == 'obs'

                ## Target triplet 
                targ = [strobed[g-4:g-1, 1] for i, g in enumerate(go_ix)]
                targ_ix = []
                for i, tg in enumerate(targ):
                    tmp = np.tile(tg[np.newaxis, :], [len(fk.obstrialList), 1])
                    ix = np.nonzero(np.sum(np.abs(fk.obstrialList - tmp), 1) == 0)[0]
                    if len(ix)==1:
                        targ_ix.append(ix[0])
                    else:
                        targ_ix.append(-1)
                targ_ix = np.array(targ_ix)

                ### Now go through the targ ix and get the final target position 
                
                ### x pos // y pos
                targets = np.zeros((dat_len, 5))
                targets[:, -1] = 1.

                ## obs state: xpos // y pos // maj targ rad // min targ rad // orientation
                obs_state = np.zeros((dat_len, 5))

                for _, (go_, tg_ix) in enumerate(zip(go_ix, targ_ix)): 

                    if tg_ix >= 0: 
                        targ_list = fk.obstrialList
                        targ_series = targ_list[tg_ix, :] - 63

                        target_index = int(targ_series[2]) - 1
                        assert(np.sum(np.isnan(amy_data['targPos'][target_index, :])) == 0) 
                        targets[go_:, [0, 1]] = amy_data['targPos'][target_index, :]

                        obs_index = int(targ_series[1]) - 1
                        obs_info = np.hstack(( amy_data['targPos'][obs_index, :], 
                                                       amy_data['targRad_maj'][obs_index, 0],
                                                       amy_data['targRad_min'][obs_index, 0], 
                                                       amy_data['targOrient'][obs_index, 0] ))
                        assert(np.sum(np.isnan(obs_info)) == 0) 
                        obs_state[go_:, :] = obs_info

            target_state.append(targets)
            obs_pos.append(obs_state[:, [0, 1]]) # position 
            obs_shape.append(obs_state) ## overall state 

            ## te_num
            te_num.append(np.hstack(([day2[ite]]*dat_len))[:, np.newaxis])
                        
            ### ADD HDF OFFSET 
            dat_offset = dat_offset + dat_len
            
            #### ADD DECODER and TASK PARAMETERS #####
            A = np.hstack(( dat['A'], np.array([0., 0., 0., 0.])[:, np.newaxis]))
            A = np.vstack(( A, np.array([0., 0., 0., 0., 1.])))
            
            W = np.zeros((5, 5))
            W[2, 2] = dat['W'][0, 0]
            W[3, 3] = dat['W'][0, 0]
            
            Beta = dat['beta'] # [xvel, yvel, offset] x 20 units
            
            #### Get estimated KG / 
            KG_est = dat_KG_est[day_ix]
            KG_est = np.vstack((np.zeros((2, KG_est.shape[1])), KG_est, np.zeros((1, KG_est.shape[1]))))

            mats = [A[np.newaxis, :, :], W[np.newaxis, :, :], Beta[np.newaxis, :, :], KG_est[np.newaxis, :, :], 

                    np.array([1.2])[np.newaxis, :], # target radius 
                    np.array([np.nan])[np.newaxis, :], # cursor radious 
                    np.array([np.nan])[np.newaxis, :], # plant type 
                    np.array([np.nan])[np.newaxis, :]] # reward time 
            
            names = ['decoder_A', 'decoder_W', 'decoder_beta', 'decoder_ssKG',
                   'target_radius', 'cursor_radius', 'plant_type', 'reward time (sec)']
            descs = ['point process filter decoder A matrix (cursor dynamics, 5 x 5 ) designed for meters', 
                     'point process decoder W matrix (noise of cursor dynamics 5 x 5 ) designed for meters', 
                     'point process deocder beta matrix (neural encoding of cursor 3 x Nneurons) designed for meters', 
                     'estimate of kalman filter steady state K matrix (kalman gain, neural update 5 x Nneurons)', 
                     'target radius (cm)',
                     'cursor radius (cm)', 
                     'plant type (name of plant corresponding to BMI3D code)',
                     'reward time (sec) -- how long juicer was open to deliver rewards']

            #### Commenting for better upload 
            # for _, (mat, name, desc) in enumerate(zip(mats, names, descs)): 
                
            #     ### Add decoder and task params to file 
            #     timestamps = [0]
            #     ts_w_timestamps = TimeSeries(
            #         name="teblk_%s_%s"%(day2[ite], name),
            #         data=mat,
            #         unit="a.u.",
            #         timestamps=timestamps, 
            #         description = desc)
            #     behavior_module.add(ts_w_timestamps)

        ### now add these to NWB file 
        for _, (met, met_name, met_desc, met_unit) in enumerate(zip([spk_cnts, update_bmi, cursor_pos, decoder_state, 
             target_state, obs_pos, obs_shape, te_num], 

                                                ['spike_counts', 'update_bmi', 
                                                'cursor', 
                                                 'decoder_state', 
                                                 'target_state', 
                                                'obstacle_position',
                                                'obs_details',
                                                'te_num'], 

                                                ['binned spike counts used for BMI control', 
                                                'binary variable where "1" indicates bins which BMI was updated (200 hz, task runs at 200 hz)',
                                                '2D cursor position (x, y) (meters)', 
                                                'decoder state (2d-pos, 2d-vel, offset) (meters, m/sec)', 
                                                'target state (2d-pos, 2d-vel, offset) -- location of target (cm)', 
                                                'obstacle position (x, y) (cm)', 
                                                'obstacle shape (position (x,y), major/minor radius, and orientation of obstacle) (cm)', 
                                                'task entry number for trials (corresponds to filenames'], 
                                                ['cnt', 'binary indicator', 'm', 'm, m/sec', 'cm', 'cm', 'text','int'])): 
            assert(np.vstack((met)).shape[0] == dat_offset)
            ts_w_rate = TimeSeries(name = met_name, 
                                    data = np.squeeze(np.vstack((met))), 
                                    unit = met_unit, 
                                    starting_time = 0.0, 
                                    rate = file_fs,
                                    description = met_desc)
            ### Add to behavior 
            behavior_module.add(ts_w_rate)

        # ### Add trials to NWB file 
        trials = np.vstack((trials))

        for _, (go_ts, rew_ts) in enumerate(trials): 
            nwbfile.add_trial(start_time=go_ts, stop_time=rew_ts)


        # ### Write NWB file 
        io = NWBHDF5IO("/Users/preeyakhanna/bmi_dynamics_code/nwb_notebooks/nwb_files/monkeyJ-session%d.nwb"%day_ix, mode="w")
        io.write(nwbfile)
        io.close()