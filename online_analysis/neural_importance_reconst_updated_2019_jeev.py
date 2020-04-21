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
from online_analysis import co_obs_tuning_matrices, fit_LDS
import numpy as np
import prelim_analysis as pa
import tables
import matplotlib.pyplot as plt
import scipy.stats
import resim_ppf
from resim_ppf import ppf_pa

def get_corr(P_mats, neurons, binned_spike_counts, cursor_KG_all):
    pred_curs = []
    T, N = binned_spike_counts.shape
    for i in range(T):
        P = np.zeros_like(P_mats[:, :, i])
        P[:, neurons] = P_mats[:, neurons, i].copy()
        out = np.dot(P, binned_spike_counts[i, :])
        pred_curs.append(out[[3, 5]])

    pred_output = np.vstack((pred_curs)).T
    err_x = np.squeeze(np.array(pred_output[0, :] - cursor_KG_all[0, :]))
    err_y = np.squeeze(np.array(pred_output[1, :] - cursor_KG_all[1, :]))

    vx_r2 = 1 - (np.sum(err_x**2) / np.sum(cursor_KG_all[0, :]**2))
    vy_r2 = 1 - (np.sum(err_y**2) / np.sum(cursor_KG_all[1, :]**2))
    mn = 1 - ((np.sum(err_x**2) + np.sum(err_y**2)) / np.sum(cursor_KG_all**2))
    print 'Neurons: ', neurons, 'R2: ', mn
    return vx_r2, vy_r2, mn

def main(important_thresh = 0.8):
    # Reconstructed "Kalman Gains"
    # Note --> these dont really converge. 
    # Maybe ordering by SVD of them is good enough though? Try it. 

    filelist = resim_ppf.task_filelist
    days = len(filelist)
    binsize_ms = 5.
    #KGs = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/jeev_KG_approx_feb_2019.pkl'))
    important_neuron_ix = dict()

    for day in range(days):

        # open the thing: 
        Ps = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/jeev_KG_approx_feb_2019_day'+str(day)+'.pkl'))
        
        # Get bin spikes for this task entry:
        bin_spk_all = []
        cursor_KG_all = []
        P_mats = []

        for task in range(2):
            te_nums = filelist[day][task]
            
            for te_num in te_nums:
                bin_spk, targ_i_all, targ_ix, trial_ix_all, decoder_all, unbinned, exclude = ppf_pa.get_jeev_trials_from_task_data(te_num, 
                    binsize=binsize_ms/1000.)

                indices = []
                for j, (i0, i1) in enumerate(unbinned['ixs']):
                    indices.append(np.arange(i0, i1) - unbinned['start_index_overall'])
                
                for j in np.hstack((indices)):
                    P_mats.append(np.array(Ps[te_num][j]))

                print unbinned['start_index_overall'], i1, i1 -  unbinned['start_index_overall']

                bin_spk_all.extend([bs for ib, bs in enumerate(bin_spk) if ib not in exclude])
                cursor_KG_all.extend([kg for ik, kg in enumerate(decoder_all) if ik not in exclude])

        bin_spk_all = np.vstack((bin_spk_all))
        P_mats = np.dstack((P_mats))

        cursor_KG_all = np.vstack((cursor_KG_all))[:, [3, 5]]
        # De-meaned reconstruciton: 
        cursor_KG_all_demean = (cursor_KG_all - np.mean(cursor_KG_all, axis=0)[np.newaxis, :]).T
        
        # Fit a matrix that predicts cursor KG all from binned spike counts
        KG = np.mat(np.linalg.lstsq(bin_spk_all, cursor_KG_all_demean.T)[0]).T
        #KG = np.mat(np.linalg.lstsq(bin_spk_all, cursor_KG_all)[0]).T
        
        print 'KG est: shape: ', KG.shape
        cursor_KG_all_reconst = KG*bin_spk_all.T

        # vx = np.sum(np.array(cursor_KG_all_demean[0, :] - cursor_KG_all_reconst[0, :])**2)
        # vy = np.sum(np.array(cursor_KG_all_demean[1, :] - cursor_KG_all_reconst[1, :])**2)
        # R2_best = 1 - ((vx + vy)/np.sum(cursor_KG_all_demean**2))

        vx = np.sum(np.array(cursor_KG_all[:, 0] - cursor_KG_all_reconst[0, :])**2)
        vy = np.sum(np.array(cursor_KG_all[:, 1] - cursor_KG_all_reconst[1, :])**2)
        R2_best = 1 - ((vx + vy)/np.sum(cursor_KG_all_demean**2))

        # Re-construct: 
        # Get the covariance of the neurons: 
        sigma = np.cov(bin_spk_all.T)
        N = sigma.shape[0]
        
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
        Smat = np.zeros((2, N))
        for i in range(2):
            Smat[i, i] = S[i]

        total_variance = np.sum(S**2)

        var_ = []

        for i in range(N):
            
            # Old cool method: 
            V_mod = np.zeros_like(Vh)

            # Zero out the columns of Vh
            V_mod[:2, i] = np.squeeze(np.array(Vh[i, :2]))[:, np.newaxis]
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
            i = i + 1

            vx_r2, vy_r2, mn = get_corr(P_mats, neurons, bin_spk_all, cursor_KG_all.T)

            ax.plot(len(neurons), vx_r2, 'r.')
            ax.plot(len(neurons), vy_r2, 'b.')
            #mn = 1 - ((np.sum(err_x**2) + np.sum(err_y**2)) / np.sum(cursor_KG_demean**2))
            ax.plot(len(neurons), mn, 'g.')

            # Not sure how to fix this if we can't reconstruct? 
            if np.logical_and(var_ex > important_thresh, keep_going):
                keep_going = False
                final_N = i; 
                final_Neurons_svd = np.sort(neurons.copy())

            if np.logical_and(mn > important_thresh, keep_going2):
                keep_going2 = False
                final_N_estimated = i; 
                final_Neurons_estimated = np.sort(neurons.copy())

        ax.set_xlabel('# Neurons')
        ax.set_title('Day: '+str(day))
        plt.tight_layout()
        important_neuron_ix[day, 'jeev', 'svd'] = final_Neurons_svd
        important_neuron_ix[day, 'jeev', 'est'] = final_Neurons_estimated

        f.savefig('/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/resim_day_'+str(day)+'_demean_predictions.png')
        import gc; gc.collect()
    pickle.dump(important_neuron_ix, open('jeev_important_neurons_svd_feb2019_thresh_'+str(important_thresh)+'.pkl', 'wb'))







