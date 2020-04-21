
import numpy as np
import matplotlib.pyplot as plt
import fit_LDS, subspace_overlap, co_obs_tuning_matrices
from resim_ppf import file_key as fk; 

import pickle

n_dim_latent_i = 15; 

mag_boundaries = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/radial_boundaries_fit_based_on_perc_feb_2019.pkl'))

def assess_likelihood_predictions(animal='grom', binsize_ms = 100., pre_go_secs = 1., min_num_obs = 15.): 
    pre_go_bins = int(pre_go_secs * 1000 / binsize_ms)

    if animal == 'grom':
        days = 9;
        tes = co_obs_tuning_matrices.input_type 
    elif animal == 'jeev':
        days = 4; 
        tes = fk.task_filelist

    for i_d in range(days): 
        BS = []; 
        DA = []; 

        for i_t in range(2):

            for i, te_num in enumerate(tes[i_d][i_t]):

                ### Get data: 
                bin_spk_nonz, targ_ix, trial_ix_all, KG, decoder_all, exclude_ix = fit_LDS.pull_data(te_num, animal,
                    pre_go=pre_go_secs, binsize_ms=binsize_ms, keep_units='all')

                for itrl, trl in enumerate(np.unique(trial_ix_all)):
                    if trl not in exclude_ix:
                        trl = int(trl)
                        BS.append(bin_spk_nonz[pre_go_bins:, trl])
                        DA.append(decoder_all[pre_go_bins:, trl])

       ### Fit LDS on portion of data, test on held-out
        R2_smooth, R2_pred, LL, dynamics_norm, innov_norm, model, pred_data, smooth_data, filt_state, pred_state, filt_sigma, kg, R2_filt, smooth_state = fit_LDS.fit_LDS(
            BS, BS, n_dim_latent_i, return_model=True, seed_w_FA=True, nEMiters=30, **dict(seed_pred_x0_with_smoothed_x0= True,
            get_ratio=True, pre_go_bins=pre_go_bins))
        
        ### Get relevant matrices: 
        dyn_A = model.A; 
        dyn_W = model.state_noise; 
        dyn_C = model.C; 


        ### Get stacked bins: 
        stacked_spks = np.vtstack((BS))

        ### Mu / Sigma filtered: 
        stacked_filt_mu = np.vstack((filt_state))
        stacked_filt_sig = np.dstack((filt_sigma))

        ### Mark t=0 so don't use those guys: 
        stacked_ind = []
        for t in range(len(BS)):
            x = np.ones(( BS[t].shape[0] ))
            x[0] = 1;
            stacked_ind.append(x)
        stacked_ind = np.hstack((stacked_ind))

        ### Also get the velocity bins of this day: 
        command_bins = subspace_overlap.commands2bins(DA, mag_boundaries, animal, i_d)
        stacked_bins = np.vstack((command_bins))

        assert(len(stacked_ind) == stacked_filt_mu.shape[0] == stacked_filt_sig.shape[2] == stacked_spks.shape[0] == stacked_bins.shape[0])

        ### For each data point, get mean and cov for command_bins: 
        action_stats = dict(); 

        for mag in range(4):
            ix_mag = np.nonzero(stacked_bins[:, 0] == mag)[0]
            for ang in range(8):
                ix_ang = np.nonzero(stacked_bins[ix_mag, 1] == ang)[0]

                if len(ix_ang) > min_num_obs:
                    action_stats[mag, ang, 'mu'] = np.mean(stacked_spks[ix_mag[ix_ang], :], axis=0)
                    action_stats[mag, ang, 'cov'] = np.cov(stacked_spks[ix_mag[ix_ang], :].T)

        ### Now for each data point, assess 
        LL_dict = dict(dyn=[], act = [])
        for t in range(1, len(stacked_ind)):
            magi, angi = stacked_bins[t, :]

            if tuple([magi, angi]) in action_stats.keys():
                if stacked_ind[t] == 1:

                    ### Proceed: 
                    ### Actual binned spike count at this time: 
                    obs_t = stacked_spks[t, :]

                    ### Dynamics prediction: 
                    filt_mu_tm1 = stacked_filt_mu[t-1, :]
                    filt_sig_tm1 = stacked_filt_sig[:, :, t-1]

                    ### New mu / sig by propagation: 
                    sig_t_prop = np.dot(dyn_A, np.dot(filt_sig_tm1, dyn_A.T)) + dyn_W
                    mu_t_prop = np.dot(dyn_A, filt_mu_tm1)

                    ### Move the observation space: 
                    obs_sig_t_prop = np.dot(dyn_C, np.dot(sig_t_prop, dyn_C.T)) # Should I be adding Q here ?
                    obs_mu_t_prop = np.dot(dyn_C, mu_t_prop)

                    rv = scipy.stats.multivariate_normal(mean=obs_mu_t_prop, cov=obs_sig_t_prop);
                    ll = rv.logpdf(obs_t)
                    LL_dict['dyn'].append(ll)

                                        




