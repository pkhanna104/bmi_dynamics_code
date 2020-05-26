### Method for deciding which neurons are needed for control ###
### If remove unimporant neurons how well can you reconstruct the 
### Vx / Vy trajectory? ###
### Assess if that's true -- use R2? Implies there may need to be a 
### linear transform to get the full extent 

### Updates 2-3-19: 
### Ideally, each neuron would have been zscored, then we could look at the Kalman gain weights

### Now though, we can assess the variance of each neuron within the day (across both tasks)
### Then, we can do SVD on the Kalman Gain matrix (magic). 
### This gives us U (2x2), S (2x2 diag), V (nxn). Here, the first two rows of V will be non-zero. 
### Each of these rows is unit vector in the direction of max variance (i.e. PCs)
### The singular values (diag(S)) are ordered in decreasing size. Trace(diag(S)**2) is the 
### the full variance in the directions parameterized by the rows of V. 
### The columns of U represent the directions of max variance in the output space (velocity space)

### So how to get the imporant neurons? K*y_t is the transform
### first 'whiten' the neurons in y_t. First need to de-mean the y_ts
### then compute covariance (C), and do C^{-1/2}*y_t. 
### Then the new Kalman Gain can be expressed as K*C^{-1/2}

### So, do SVD on the new KalmanGain
### As we drop 2x1 columns of the V matrix (zero out neurons), see how the Trace(diag(S)**2) changes

import pickle
import analysis_config
import numpy as np
import fit_LDS
import prelim_analysis as pa
import tables
import matplotlib.pyplot as plt
import scipy.stats

def main():

    binsize_ms = 100.
    datafiles = pickle.load(open(analysis_config.config['grom_pref'] + 'co_obs_file_dict.pkl'))
    important_neuron_ix = dict()
    animal = 'grom'
    importance_thresh = 0.8

    for day in range(9):

        # Get bin spikes for this task entry:
        bin_spk_all = []
        cursor_KG_all = []
        
        for task in range(2):
            te_nums = analysis_config.data_params['grom_input_type'][day][task]
            
            for te_num in te_nums:
                # bin_spk_nonz, targ_ix, trial_ix_all, KG, _, exclude_ix = fit_LDS.pull_data(te_num, animal,
                #     pre_go=1, binsize_ms=binsize_ms, keep_units='all')

                # Let's get the actual 
                hdf = datafiles[te_num, 'hdf']
                name = hdf.split('/')[-1]
                hdf = tables.openFile(analysis_config.config['grom_pref']+name)

                # decoder: 
                dec = datafiles[te_num, 'dec']
                dec = dec.split('/')[-1]
                decoder = pickle.load(open(analysis_config.config['grom_pref']+dec))
                F, KG = decoder.filt.get_sskf()

                rew_ix = np.array([t[1] for it, t in enumerate(hdf.root.task_msgs[:]) if t[0]=='reward'])

                bin_spk_nonz, _, _, _, cursor_KG = pa.extract_trials_all(hdf, rew_ix, neural_bins = binsize_ms,
                    drives_neurons_ix0=3, hdf_key='spike_counts', keep_trials_sep=True,
                    reach_tm_is_hdf_cursor_pos=False, reach_tm_is_hdf_cursor_state=False, 
                    reach_tm_is_kg_vel=True, include_pre_go= 1, **dict(kalman_gain=KG))
                exclude_ix = []; 
                bin_spk_all.extend([bs for ib, bs in enumerate(bin_spk_nonz) if ib not in exclude_ix])
                cursor_KG_all.extend([kg for ik, kg in enumerate(cursor_KG) if ik not in exclude_ix])

        bin_spk_all = np.vstack((bin_spk_all))
        cursor_KG_all = np.vstack((cursor_KG_all))[:, [3, 5]]

        # Get the covariance of the neurons: 
        sigma = np.cov(bin_spk_all.T)
        N = sigma.shape[0]
        # The the variance of the neurons: 
        #var = np.sqrt(np.var(bin_spk_all, axis=0))

        # Get the Kalman Gain: 
        F, KG = decoder.filt.get_sskf()
        KG = KG[[3, 5], :]

        # Diagonalize this matrix: 
        eigvalues, eigvect = np.linalg.eig(sigma)

        # Make sure eigenvalues work as a decomp
        assert np.allclose(sigma, np.dot(np.dot(eigvect, np.diag(eigvalues)), np.linalg.inv(eigvect)))

        # Get inverse sqrt cov matrix: 
        inv_cov_sqrt = np.dot(np.dot(eigvect, np.diag(eigvalues**(-0.5))), np.linalg.inv(eigvect))
        cov_sqrt = np.dot(np.dot(eigvect, np.diag(eigvalues**(0.5))), np.linalg.inv(eigvect))

        # Make sure alls good
        if np.any(eigvalues == 0):
            pass
        else:
            assert np.allclose(np.dot(inv_cov_sqrt, cov_sqrt), np.eye(N))

        # Test example: 
        # ktest = np.zeros((2, 4))
        # ktest[0, 0] = 1
        # ktest[1, 0] = 1
        # ktest[1, 1 ] = 2
        # ktest[0, 2 ] = 0.
        # ktest[1, 2 ] = 0.
        # ktest[0, 3 ] = 100.
        # Rows of V make sense as the directions in neural space that matter. 
        # So columns of V need to zeroed out to remove all effects of 

        # Do SVD on the KG*inv_cov_sqrt: 
        KG_new = np.dot(KG, cov_sqrt)
        U, S, Vh = np.linalg.svd(KG_new)
        
        #import pdb; pdb.set_trace()
        Smat = np.zeros((2, N))
        for i in range(2):
            Smat[i, i] = S[i]

        total_variance = np.sum(S**2)

        var_ = []

        cKG = np.squeeze(np.array(cursor_KG_all))
        cKG = np.hstack((cKG[:, 0], cKG[:, 1]))
        # How much does each neuron contribute to the output: norm of columns of V: 
        
        tmp_cursor_KG = np.squeeze(np.array(cursor_KG_all))
        tmp_cursor_KG_mn = np.mean(tmp_cursor_KG, axis=0)
        cursor_KG_demean = tmp_cursor_KG - tmp_cursor_KG_mn[np.newaxis, :]

        for i in range(N):
            
            # Old cool method: 
            V_mod = np.zeros_like(Vh)

            # Zero out the columns of Vh
            V_mod[:2, i] = Vh[i, :2]
            kg_temp = np.dot(np.dot(U, Smat), V_mod)
            _, S_new, _ = np.linalg.svd(np.dot(np.dot(U, Smat), V_mod))
            var_.append(np.sum(S_new**2))

            # New useful method: regression with output: 
            #slpx, intcx, rx, pvx, errx = scipy.stats.linregress(bin_spk_all[:, i], np.squeeze(np.array(cursor_KG_all[:, 0])))
            #slpy, intcy, ry, pvy, erry = scipy.stats.linregress(bin_spk_all[:, i], np.squeeze(np.array(cursor_KG_all[:, 0])))

            # pred_x = slpx*bin_spk_all[:, i] + intcx
            # pred_y = slpy*bin_spk_all[:, i] + intcy

            # pred = np.hstack(( pred_x, pred_y ))

            # error: 
            #var_.append(1 - (np.sum((cKG-pred)**2) / (np.sum(cKG**2))))
            #var_.append(np.mean([pvx, pvy]))

        # Order the neurons: 
        ix = np.argsort(var_)
        order = np.arange(N-1, -1, -1)
        decreasing_ix = ix[order]
        
        # Include the first, then keep including until you get up to 90% variance explained: 
        var_ex = 0.
        i = 1
        V_mod = np.zeros_like(Vh)
        f, ax = plt.subplots()
        ax.set_ylabel('Var Exp.')

        keep_going = True
        keep_going2 = True

        while i <= N:
            neurons = decreasing_ix[:i]
            V_mod[:2, neurons] = Vh[neurons, :2].T
            _, S_new, _ = np.linalg.svd(np.dot(np.dot(U, Smat), V_mod))
            var_ex = np.sum(S_new**2) / total_variance

            ax.plot(len(neurons), var_ex, 'k.')

            KG_reduced = np.zeros_like(KG)
            KG_reduced[:, neurons] = KG[:, neurons]

            # Prediction
            pred_output = np.dot(KG_reduced, bin_spk_all.T)

            # PRediction demeaned: 
            pred_output_mean = np.mean(pred_output, axis=1)
            pred_output_demean = pred_output - pred_output_mean[:, np.newaxis]

            err_x = pred_output_demean[0, :] - cursor_KG_demean[:, 0]
            err_y = pred_output_demean[1, :] - cursor_KG_demean[:, 1]

            vx_r2 = 1 - (np.sum(err_x**2) / np.sum(cursor_KG_demean[:, 0]**2))
            vy_r2 = 1 - (np.sum(err_y**2) / np.sum(cursor_KG_demean[:, 1]**2))

            # _, _, vx_r, _ , _ = scipy.stats.linregress(pred_output[0, :],
            #     np.squeeze(np.array(cursor_KG_all[:, 0])))

            # _, _, vy_r, _ , _ = scipy.stats.linregress(pred_output[1, :],
            #     np.squeeze(np.array(cursor_KG_all[:, 1])))

            ax.plot(len(neurons), vx_r2, 'r.')
            ax.plot(len(neurons), vy_r2, 'b.')
            mn = 1 - ((np.sum(err_x**2) + np.sum(err_y**2)) / np.sum(cursor_KG_demean**2))
            ax.plot(len(neurons), mn, 'g.')

            print 'Neurons: ', i, ' out of Total: ', N, mn, var_ex

            if np.logical_and(var_ex > importance_thresh, keep_going):
                keep_going = False
                final_Neurons_svd = neurons.copy()

            if np.logical_and(mn > importance_thresh, keep_going2):
                keep_going2 = False
                final_N = i; 
                final_Neurons_estimated = np.sort(neurons.copy())
            i = i + 1; 

        ax.set_xlabel('# Neurons')
        ax.set_title('Day: '+str(day))
        plt.tight_layout()
        important_neuron_ix[day, 'grom', 'est'] = final_Neurons_estimated
        important_neuron_ix[day, 'grom', 'svd'] = final_Neurons_svd

        f.savefig('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/resim_day_'+str(day)+'_demean_predictions.png')
    pickle.dump(important_neuron_ix, open('grom_important_neurons_svd_feb2019_thresh_'+str(importance_thresh)+'.pkl', 'wb'))







