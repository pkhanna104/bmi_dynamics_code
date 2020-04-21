### Objectives -- program in different smoothness metrics: 
    # Likelihoood, R2 of next step, and dynamics ratio
    # Test with weak and strong dynamics, low noise (dynamica) and high noise (not dynamical)
    # Test with only important and both imporant and unimportant neurons
    # Don't do any dim reduction

import fit_LDS
from pylds.models import DefaultLDS
from pylds.states import kalman_filter
import numpy as np
import tables
import pickle
import matplotlib.pyplot as plt
import scipy.stats
import sklearn.decomposition as skdecomp
import prelim_analysis as pa

# Define some A: 
A = np.array([[ 0.79233844,  0.59354848],
    [-0.59354848,  0.79233844]])

cmap_list = ['maroon', 'orangered', 'darkgoldenrod', 'olivedrab', 'teal',
    'steelblue', 'midnightblue', 'darkmagenta']

select_noise = [.001, .1, 10.]
noise_levels = [.001]; #, .01, .1, 1., 10., 100., 1000., 10000.]
decay_list = [0.001, 0.25, .5, .75, .9, .95, .98, .999]

D_obs = 2
D_latent = 2
D_input = 0

pref = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/'

def test_true_model_noise_and_preds(A=A, cmap_list = cmap_list,
    noise_levels = noise_levels, decay_list=decay_list, mu_init_scale = 1000):

    f, ax = plt.subplots() # R2 of prediction
    f2, ax2 = plt.subplots() # Dynamics ratio
    f3, ax3 = plt.subplots() # Likelihood
    f4, ax4 = plt.subplots(ncols = 3, nrows = 3)
    f5, ax5 = plt.subplots(ncols = 3, nrows = 3) # Covariance convergence: 

    for reps in range(3):    
        for si, decay in enumerate(decay_list):

            # Set parameters
            T = 2000

            # What is A, and W? 
            rotate_rads = 20*np.pi/180.

            # Set the eigenvalues: 

            ### PK 7-12-19 --> this isnt quite right: just set a bunch of rotations
            diag = np.ones((D_latent)).astype(complex)

            for i in range(D_latent/2):
                diag[(i*2)]   = np.complex(decay*np.cos(rotate_rads)   , decay*np.sin(rotate_rads))
                diag[(i*2)+1] = np.complex(decay*np.cos(rotate_rads), -1*decay*np.sin(rotate_rads))
            
            # Do eigenvalue decomposition
            eigv, eigvect = np.linalg.eig(A)
            estA =  np.dot(eigvect, np.dot(np.diag(eigv), np.linalg.inv(eigvect))).real
            assert(np.all(np.isclose(estA, A)))

            # now change eigenvalue: 
            eigv[:] = diag.copy()

            # New A
            newA = np.dot(eigvect, np.dot(np.diag(eigv), np.linalg.inv(eigvect))).real
            
            # Now try same model with diff noise levels
            noise_r2 = []; dr = []; ll = []; cov = [];

            for ni, noise_scale in enumerate(noise_levels):

                # Compute W
                W = np.eye(D_latent)*noise_scale

                #### Simulate from pyLDS ####
                # true_model = DefaultLDS(D_obs, D_latent, D_input, sigma_obs=np.eye(D_obs), 
                #     A = newA, C = np.eye(D_latent), sigma_states = W, mu_init = np.ones(D_latent),
                #     sigma_init = np.eye(D_latent))
                
                # Generate data: 
                #data, stateseq = true_model.generate(T, inputs=inputs)
                                
                #### Simulate from own fcn ###
                stateseq, covseq = generate_states(mu_init_scale*np.ones(D_latent),
                    newA, W, T)                
                inputs = np.random.randn(T, D_input)

                obs = np.zeros((D_obs, D_obs))
                obs[0, 0] = .000001
                obs[1, 1] = .000001


                _, filtered_mus, filtered_sigmas = kalman_filter(
                    np.ones(D_latent), np.eye(D_latent), 
                    newA, np.zeros((D_latent, D_input)), W, 
                    np.eye(D_latent), np.zeros((D_obs, D_input)), obs, 
                    inputs, stateseq)

                # Inputs to KF: 
                    # x0, P0,
                    # g.A, g.B, g.sigma_states,
                    # g.C, g.D, g.sigma_obs,
                    # g.inputs, g.data)

                # if si == 0:
                #     print('Noise scale %f' %noise_scale)
                #     print(W)
                if ni == 0:
                    print('Dyanimcs decay %f' %decay)
                    print(covseq[:, :, -1])

                if np.logical_and(reps == 0, noise_scale == .1):
                     x, y = np.mgrid[-20:20:.1, -20:20:.1]
                     pos = np.empty(x.shape + (2,))
                     pos[:, :, 0] = x; pos[:, :, 1] = y
                     rv = scipy.stats.multivariate_normal(np.zeros((2,)), covseq[:, :, -1])
                     #import pdb; pdb.set_trace()
                     ax5[si/3, si%3].contourf(x, y, rv.logpdf(pos), vmin=-1000, vmax=0, cmap='jet')
                     ax5[si/3, si%3].set_title('A Decay %.2f, Noise %.2f' %(decay, noise_scale), fontsize=6)

                cov.append(covseq[0, 0, -1])
                
                # Assess the dynamics ratio: 
                R2 = get_R2(stateseq, newA)
                DR = get_dyn_ratio(stateseq, newA)
                LL = get_likelihood(stateseq, covseq, newA)

                # ### Fit a test model: 
                # test_model = DefaultLDS(D_obs, D_latent, D_input)
                # test_model.add_data(data, inputs=inputs)

                # # Fit the model: 
                # N_samples = 100
                # def update(model):
                #     model.resample_model()
                #     return model.log_likelihood()

                # lls = [update(test_model) for _ in range(N_samples)]

                # # Smoothed states:
                # g = test_model.states_list[0]
                # _ = g.smooth()

                # smooth_mu = g.smoothed_mus.copy()
                # R2 = get_R2_pred(stateseq, smooth_mu)

                noise_r2.append(np.mean(R2))
                dr.append(np.mean(DR))
                ll.append(np.mean(LL))

                # Plot the actual states: 
                if np.logical_and(noise_scale == 0.1, reps == 0):

                    # Subplot:
                    axi = ax4[si/3, si %3]

                    T_to_plot = 100; 

                    # Now try to actually plot states: 
                    axi.plot(stateseq[1:T_to_plot, 0], 'k-')
                    pred_state = np.dot(newA, stateseq[:T_to_plot-1, :].T).T
                    axi.plot(pred_state[:, 0], 'b--')
                    axi.set_title('Noise: %f , Dyn: %f'%(noise_scale, decay), fontsize=6)
                    axi.set_ylim([-3, 3])

            ax.plot(np.log10(np.array(noise_levels)), noise_r2, '.-', color=cmap_list[si])
            ax2.plot(np.log10(np.array(noise_levels)), dr, '.-', color=cmap_list[si])
            ax3.plot(np.log10(np.array(noise_levels)), ll, '.-', color=cmap_list[si])
            
    ax.set_xlabel('Log10 State Noise')
    ax.set_ylabel('Prediction of X_{t+1} | X_{t} with A matrix')
    ax.set_title('mean R2 (pred next state w A) \n for different strengths of eig(A) and W levels')

    ax2.set_xlabel('Log10 State Noise')
    ax2.set_ylabel('Dynamics Ratio')
    ax2.set_title('Mean DR for ')

    f.tight_layout()
    f2.tight_layout()
    f3.tight_layout()
    f4.tight_layout()
    f5.tight_layout()

def test_models_estimates_noise_and_preds(A=A, cmap_list = cmap_list,
    decay_list=decay_list, nEMiters=30,
    pre_go_bins = 10, init_w_FA = True, nreps = 3):
    
    decay_list = [0.25, .5, .75, .9, .95, .98, .999]
    noise_levels = [.001, .01, .1, 1., 10., 100., 1000., 10000.]

    f, ax = plt.subplots() # R2 of prediction
    f2, ax2 = plt.subplots() # Dynamics ratio
    f3, ax3 = plt.subplots() # Likelihood

    # Plot PSTH of our data in all these categories
    # Each column is a diff noise level: 
    f4, ax4 = plt.subplots(nrows = 3, ncols = 3) 

    # For each decay_list: 
    for si, decay in enumerate(decay_list):

        ##################################################
        # What is A, and W? 
        rotate_rads = 20*np.pi/180.

        # Set the eigenvalues: 
        diag = np.ones((D_latent)).astype(complex)

        for i in range(D_latent/2):
            diag[(i*2)]   = np.complex(decay*np.cos(rotate_rads)   , decay*np.sin(rotate_rads))
            diag[(i*2)+1] = np.complex(decay*np.cos(rotate_rads), -1*decay*np.sin(rotate_rads))
        
        # Do eigenvalue decomposition
        eigv, eigvect = np.linalg.eig(A)
        estA =  np.dot(eigvect, np.dot(np.diag(eigv), np.linalg.inv(eigvect))).real
        assert(np.all(np.isclose(estA, A)))

        # now change eigenvalue: 
        eigv[:] = diag.copy()

        # New A
        newA = np.dot(eigvect, np.dot(np.diag(eigv), np.linalg.inv(eigvect))).real
        
        ##################################################
        ll_strength = dict()
        r2_strength = dict()
        dr_strength = dict()

        psth = dict(ll=[], r2=[], dr = [])

        for ni, noise_scale in enumerate(noise_levels):

            ll_strength[noise_scale] = []
            r2_strength[noise_scale] = []
            dr_strength[noise_scale] = []

            for reps in range(nreps):

                # Compute W
                W = np.eye(D_latent)*noise_scale
                         
                ##################################################       
                #### Simulate data from own fcn ###
                # Set parameters
                ntrls_train = 48; 
                ntrls_test = 16; 
                data = [];
                for trl in range(ntrls_train):
                    T = np.random.randint(40, 60)
                    stateseq, _ = generate_states(np.ones(D_latent),
                        newA, W, T)
                    data.append(stateseq)

                data_test = [];
                for trl in range(ntrls_test):
                    T = np.random.randint(40, 60)
                    stateseq, _ = generate_states(np.ones(D_latent),
                        newA, W, T)
                    data_test.append(stateseq)  

                ##################################################
                ##### Now try to fit this data with a new LDS ####
                model = fit_LDS_model(D_obs, D_latent, D_input, data, init_w_FA = False,
                    nEMiters=nEMiters)

                ##################################################
                #### Now predict the held-out states ###
                r2, dr, ll = get_metrics(data_test, model, pre_go_bins)

                ll_strength[noise_scale].append(ll)
                r2_strength[noise_scale].append(r2)
                dr_strength[noise_scale].append(dr)

        ##################################################
        ### Need to plot all before resetting ###
        ylabels = ['R2 Prediction', 'Log likelihood', 'Dynamics Ratio']
        for _, (metric, axis, yl) in enumerate(zip([r2_strength, ll_strength, dr_strength], [ax, ax2, ax3],
            ylabels)):

            mn = []; 
            sem = [];
            for n in noise_levels:
                nt = [];
                for m in metric[n]:
                    nt.append(np.hstack((m)))
                mn.append(np.mean(np.hstack((nt))))
                sem.append(np.std(np.hstack((nt))) / np.sqrt(len(np.hstack((nt)))))

            axis.plot (np.log10(np.array(noise_levels)), mn, '-', color=cmap_list[si])
            axis.errorbar(np.log10(np.array(noise_levels)), mn, yerr=sem, fmt='|', ecolor=cmap_list[si])
            axis.set_ylabel(yl)
            axis.set_xlabel('Log 10 State Noise')

        # Now try to plot a PSTH of all these metrics: 
        max_t = 0.
        for i in range(nreps):
            for noise_scale in noise_levels:
                for j in metric[noise_scale][i]:
                    max_t = np.max([max_t, len(j)])
        max_t = int(max_t)

        for i_m, (metric, axis, yl) in enumerate(zip([r2_strength, ll_strength, dr_strength], [ax, ax2, ax3],
            ylabels)):

            for n in noise_levels:
                if n in select_noise:
                    ix = [i for i, s in enumerate(select_noise) if s == n]
                    assert len(ix) == 1

                    ### Metric is row, noise level is column ###
                    axi = ax4[i_m, ix[0]]

                    ### plot PSTH ###
                    plot_metric_PSTH(metric[n], nreps, cmap_list[si], axi, max_t)

def plot_metric_PSTH(metric, nreps, color, axi, max_t):
    
    ### Iterate through the noise levels ###
    met = []; 

    ### Iterate through the nreps ###
    for i in range(nreps):

        ### Iterate through the trials
        for j in metric[i]:

            tmp = np.zeros((max_t))
            tmp[:len(j)] = j.copy()

            ### Everything is invalid, except first entries of j
            mask = np.ones((max_t))
            mask[:len(j)] = 0; 

            ma = np.ma.masked_array(tmp, mask=mask)

            ### Make a masked array ###
            met.append(ma)

    ### Now we have a bunch of masked arrays: 
    ### Stack them together
    big_array = np.ma.array(met)

    ### Take the mean and sem: 
    mn_t = big_array.mean(axis=0)
    sem_t = big_array.std(axis=0) / np.sqrt(len(met))
    sem_plus = mn_t + sem_t
    sem_minus = mn_t - sem_t

    ### Plot these: 
    t = np.arange(len(mn_t))
    axi.plot(t, mn_t, '-', color=color)
    axi.fill_between(t, sem_plus, sem_minus, color=color, alpha=0.5)

def generate_states(x0, A, W, T):
    ''' Write the generation of states yourself '''
    x = []; p = []; 
    x_t = x0[:, np.newaxis]
    p_t = W.copy()

    for t in range(T): 
        w_t = np.random.multivariate_normal(np.zeros(( len(x0) )), W, 1).T
        x_t = np.dot(A, x_t) + w_t
        x.append(x_t.copy())

        # Covariance: 
        p.append(p_t.copy())

    return np.hstack(( x )).T, np.dstack((p))

def get_R2(states, A):
    # How well does A predict next state
    states_t = states[:-1, :]
    states_t1 = states[1:, :]

    pred_t1 = np.dot(A, states_t.T).T

    # Get R2 of this: 
    T, nS = pred_t1.shape
    R2 = np.zeros((nS))

    for t in range(nS):
        st1 = states_t1[:, t]
        pt1 = pred_t1[:, t]

        _, _, R, _, _ = scipy.stats.linregress(st1, pt1)
        R2[t] = R**2

    return R2

def get_R2_pred(states1, predstates1, return_SS = True):
    T, nS = states1.shape
    R2 = np.zeros((T))
    SSR = np.zeros((T))
    SST = np.zeros((T))

    for t in range(T):
        st1 = states1[t, :]
        pt1 = predstates1[t, :]

        # Variance accounted for: 
        ss_tot = np.sum((st1)**2)
        ss_res = np.sum((st1 - pt1)**2)

        r2 = 1 - (ss_res / ss_tot)
        R2[t] = r2
        SSR[t] = ss_res
        SST[t] = ss_tot
        
    if return_SS:
        return R2, SSR, SST
    else:
        return R2

def get_dyn_ratio(states1, A):
    # Assess the dynamics ratio 
    pred_t1 = np.dot(A, states1[:-1, :].T).T

    # Diff b/w prediction and x_t
    dyn_norm = np.linalg.norm(pred_t1 - states1[:-1, :], axis=1)

    # Diff b/w x_t+1 and x_t
    tot_norm = np.linalg.norm(states1[:-1, :] - states1[1:, :], axis=1)

    # Diff b/w/ total diff and dy
    inn_norm = np.linalg.norm(states1[1:, :] - pred_t1, axis=1)

    dr = []
    for i, (d, inn) in enumerate(zip(dyn_norm, inn_norm)):
        dr.append( d / (d + inn))

        if (d / (d + inn)) > 1.:
            import pdb; pdb.set_trace()

    return np.hstack((dr))

def get_likelihood(mus, sigmas, A, W = None, fit_model = False):

    # If we're fitting a model, then we need to propagate the covariance
    # If we're just generatign likelihoods for a known model, then can just 
        # assume that we know the covariance (since we do)

    T = len(mus) - 1; 
    LL = []; 

    for t in range(T): 

        mu_t = mus[t, :]
        mu_t1 = mus[t+1, :]

        sig_t = sigmas[:, :, t]

        if fit_model:
            # Time step these forward: 
            # E (x_t+1 | x_t ) = Ax_t 
            # E ((x_t+1 | x_t - x_t+1) (x_t+1 | x_t - x_t+1) ') = AP_{t|t}A.T + W
            sig_t_prop = np.dot(A, np.dot(sig_t, A.T)) + W
        else:
            sig_t_prop = sig_t.copy(); 

        mu_t_prop = np.dot(A, mu_t.T).T

        rv = scipy.stats.multivariate_normal(mean=mu_t_prop, cov=sig_t_prop);
        ll = rv.logpdf(mu_t1)
        LL.append(ll)

    return np.hstack((LL))

def fit_LDS_model(D_obs, D_latent, D_input, data, init_w_FA = False, nEMiters = 30):

    model = DefaultLDS(D_obs, D_latent, D_input)
    for _, d in enumerate(data):
        model.add_data(d)

    ##################################################
    #### Initialize the model with FA ###      
    if init_w_FA:    
        FA = skdecomp.FactorAnalysis(n_components=D_latent)
        dat = np.vstack((data))
        FA.fit(dat)
        x_hat = FA.transform(dat)

        # Do main shared variance to solve this issue: 
        A = np.mat(np.linalg.lstsq(x_hat[:-1, :], x_hat[1:, :])[0])
        err = x_hat[1:, :] - np.dot(A, x_hat[:-1, :].T).T
        err_obs = dat - np.dot(FA.components_.T, x_hat.T).T

        model.C = FA.components_.T # Components is nfactors x n obs., but using the true "U" matrix
        model.A = A

        model.sigma_states = np.cov(err.T)
        model.sigma_obs = np.cov(err_obs.T)

    ##################################################
    #### Train the model ####
    def update(model):
        model.EM_step()
        return model.log_likelihood()

    lls = [update(model) for i in range(nEMiters)]
    
    return model    

def get_metric_psth(data_test, model, pre_go_bins = 10, maxT = 1000):

    ll = []; dr = []; ssr = []; sst = []; 

    ssr = np.zeros((len(data_test), maxT)); ssr[:] = np.nan; 
    sst = np.zeros((len(data_test), maxT)); sst[:] = np.nan; 
    dr =  np.zeros((len(data_test), maxT)); dr[:]  = np.nan; 
    ll =  np.zeros((len(data_test), maxT)); ll[:]  = np.nan; 

    for i_d, data_test_i in enumerate(data_test):
        model.add_data(data_test_i)
        test_trl = model.states_list.pop()

        # Smoothed (y_t | y0...yT)
        smoothed_trial = test_trl.smooth()
        x0 = test_trl.smoothed_mus[0, :] # Time x ndim
        P0 = test_trl.smoothed_sigmas[0, :, :]

        # Filtered states (x_t | y0...y_t)
        _, filtered_mus, filtered_sigmas = kalman_filter(
            x0, P0,
            test_trl.A, test_trl.B, test_trl.sigma_states,
            test_trl.C, test_trl.D, test_trl.sigma_obs,
            test_trl.inputs, test_trl.data)

        # Ignore some "pre_go" states
        # Compare this to the real states: 
        predicted_state = []; 
        actual_state = []; 
        actual_state_tm1 = []; 
        actual_scov_tm1 = []; 

        T = data_test_i.shape[0]

        for t in range(pre_go_bins - 1, T-1):
            pred_tm1 = np.dot(test_trl.A, filtered_mus[t, :].T)
            predicted_state.append(pred_tm1)
            actual_state.append(filtered_mus[t+1, :])
            actual_state_tm1.append(filtered_mus[t, :])
            actual_scov_tm1.append(filtered_sigmas[t, :, :])

        # Add the last state
        actual_state_tm1.append(filtered_mus[t+1, :])
        actual_scov_tm1.append(filtered_sigmas[t+1, :, :])

        _, sr, st = get_R2_pred(np.vstack((actual_state)),
            np.vstack((predicted_state)), return_SS = True)

        ssr[i_d, :len(sr)] = sr; 
        sst[i_d, :len(st)] = st; 

        dri = get_dyn_ratio(np.vstack((actual_state_tm1)),
            test_trl.A )
        dr[i_d, :len(dri)] = dri; 

        lli = get_likelihood(np.vstack((actual_state_tm1)),
            np.dstack((actual_scov_tm1)), test_trl.A, test_trl.sigma_states, 
            fit_model = True)
        ll[i_d, :len(lli)] = lli; 

    ssr = np.vstack((ssr))
    sst = np.vstack((sst))
    r2_arr = 1 - np.nansum(ssr, axis =0)/np.nansum(sst, axis=0)
    dr = np.nanmean(np.vstack((dr)), axis=0)
    ll = np.nanmean(np.vstack((ll)), axis=0)

    return r2_arr, dr, ll

def get_metrics(data_test, model, pre_go_bins = 10):

    ll = []; dr = []; r2 = []; 

    for data_test_i in data_test:
        model.add_data(data_test_i)
        test_trl = model.states_list.pop()

        # Smoothed (y_t | y0...yT)
        smoothed_trial = test_trl.smooth()
        x0 = test_trl.smoothed_mus[0, :] # Time x ndim
        P0 = test_trl.smoothed_sigmas[0, :, :]

        # Filtered states (x_t | y0...y_t)
        _, filtered_mus, filtered_sigmas = kalman_filter(
            x0, P0,
            test_trl.A, test_trl.B, test_trl.sigma_states,
            test_trl.C, test_trl.D, test_trl.sigma_obs,
            test_trl.inputs, test_trl.data)

        # Ignore some "pre_go" states
        # Compare this to the real states: 
        predicted_state = []; 
        actual_state = []; 
        actual_state_tm1 = []; 
        actual_scov_tm1 = []; 

        T = data_test_i.shape[0]

        for t in range(pre_go_bins - 1, T-1):
            pred_tm1 = np.dot(test_trl.A, filtered_mus[t, :].T)
            predicted_state.append(pred_tm1)
            actual_state.append(filtered_mus[t+1, :])
            actual_state_tm1.append(filtered_mus[t, :])
            actual_scov_tm1.append(filtered_sigmas[t, :, :])

        # Add the last state
        actual_state_tm1.append(filtered_mus[t+1, :])
        actual_scov_tm1.append(filtered_sigmas[t+1, :, :])

        r2.append(get_R2_pred(np.vstack((actual_state)),
            np.vstack((predicted_state)) ))

        dr.append(get_dyn_ratio(np.vstack((actual_state_tm1)),
            test_trl.A ))

        ll.append(get_likelihood(np.vstack((actual_state_tm1)),
            np.dstack((actual_scov_tm1)), test_trl.A, test_trl.sigma_states, 
            fit_model = True))
    return r2, dr, ll

#### In our own data, get PSTHs of log-likelihood /  ###

def fit_LDS_plot_PSTH_mets(binned_spikes, D_latent = 15, pre_go_bins = 1):

    D_obs = binned_spikes[0].shape[1]
    D_input = 0; 
    #D_latent = 44; 

    # Distribute our data
    data = []; data_test = []; 
    ntrls_train = np.random.permutation(len(binned_spikes))[:int(.75*len(binned_spikes))]
    ntrls_test  = np.random.permutation(len(binned_spikes))[int(.75*len(binned_spikes)):]

    for trl in ntrls_train:
        data.append(binned_spikes[trl])
    for trl in ntrls_test:
        data_test.append(binned_spikes[trl])

    # Now go through and fit LDS model
    model = fit_LDS_model(D_obs, D_latent, D_input, data, init_w_FA = True)

    # Now make predictions: 
    r2, dr, ll = get_metrics(data_test, model, pre_go_bins)
    r2 = [r2]
    dr = [dr]
    ll = [ll]

    # Find max_t
    max_t = 0; 
    for trl in ll[0]:
        max_t = np.max([max_t, len(trl)])

    f, ax = plt.subplots(nrows = 3)

    # Real data: 
    for i_m, metric in enumerate([r2, dr, ll]):
        plot_metric_PSTH(metric, 1, 'k', ax[i_m], max_t)

def extract_own_data(inputs = [[4377], [4378]], pre_go = 1.,):
    co_obs_dict = pickle.load(open(pref+'co_obs_file_dict.pkl'))
    binned_spikes = []
    for task in range(2):
        for _, te_num in enumerate(inputs[task]):
            
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

            bin_spk, targ_i_all, targ_ix, trial_ix_all, decoder_all = pa.extract_trials_all(hdf, rew_ix, 
                drives_neurons_ix0=drives_neurons_ix0, hdf_key=key, keep_trials_sep=True,
                reach_tm_is_hdf_cursor_pos=False, reach_tm_is_kg_vel=True, 
                include_pre_go = pre_go, **dict(kalman_gain=KG))

            _, _, _, _, cursor_state = pa.extract_trials_all(hdf, rew_ix, neural_bins = 100.,
                    drives_neurons_ix0=drives_neurons_ix0, hdf_key=key, keep_trials_sep=True,
                    reach_tm_is_hdf_cursor_pos=False, reach_tm_is_hdf_cursor_state=True, 
                    reach_tm_is_kg_vel=False, include_pre_go= pre_go, **dict(kalman_gain=KG))

            exclude = []
            for i, deci in enumerate(decoder_all):
                if deci.shape[0] == 1:
                    exclude.append(i)

            binned_spikes.extend(bin_spk[i] for i in range(len(bin_spk)) if i not in exclude)
    return binned_spikes

### Tests: 
def eig_test(degs, decay):

    omega = deg / 180. * np.pi # now in radians
    A1 = decay*np.array([[np.cos(omega), -1*np.sin(omega)],
                  [np.sin(omega), np.cos(omega)]])
    A2 = decay*np.array([[np.cos(omega/2.), -1*np.sin(omega/2.)],
                  [np.sin(omega/2.), np.cos(omega/2.)]])

    As = [A1, A2]
    A = scipy.linalg.block_diag(*As)


