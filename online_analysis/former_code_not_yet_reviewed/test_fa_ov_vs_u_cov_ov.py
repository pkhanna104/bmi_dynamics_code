import sklearn.decomposition as skdecomp
import numpy as np
import prelim_analysis as pa
import matplotlib.pyplot as plt


def run_FA_sim():
    # First, generate zscored data: 
    f, ax0 = plt.subplots(nrows=2)
    ax = ax0[0]
    ax2 = ax0[1]

    for i in range(5):
        n_factors = 3
        n_factors_ov = 1
        n_neurons = 40
        T = 500

        ##########################
        ### Random FACTOR ACTIVATIONS ###
        ##########################
        Z = []
        for i in range(n_factors):
            Z.append(np.random.normal(0., 1, T))
        Z = np.mat(np.vstack((Z)))

        ##########################
        ### Define LOADING MATRICES   ###
        ##########################
        U = np.mat(np.random.randn(n_neurons, n_factors))
        U2 = np.zeros_like(U)
        U2[:, :n_factors_ov] = U[:, :n_factors_ov]
        U2[:, n_factors_ov:] = np.mat(np.random.randn(n_neurons, n_factors - n_factors_ov))

        U_mn, U_norm = get_mn_shar(U*U.T)
        U2_mn, U2_norm = get_mn_shar(U2*U2.T)

        # Actual overlap based on actual Us
        proj_0_1 = np.trace(U2_norm*U_mn*U2_norm.T)/float(np.trace(U_mn))
        proj_1_0 = np.trace(U_norm*U2_mn*U_norm.T)/float(np.trace(U2_mn))

        ##########################
        ### NOISE MATRIX       ###
        ##########################
        Psi = .1*np.diag(np.random.rand(n_neurons))

        ###################
        ### DATASET 1   ###
        ###################
        Y1 = []
        for t in range(T):
            Y1.append(U*Z[:, t] + np.mat(np.random.multivariate_normal(np.zeros((n_neurons, )), Psi)).T)
        Y1 = np.hstack((Y1))

        ###################
        ### DATASET 2   ###
        ###################
        Y2 = []
        for t in range(T):
            Y2.append(U2*Z[:, t] + np.mat(np.random.multivariate_normal(np.zeros((n_neurons, )), Psi)).T)
        Y2 = np.hstack((Y2))

        #####################
        ### FA MODELS 1+2 ###
        #####################
        sot, msc, msc_norm = fit_FA(np.array(Y1.T))
        sot2, msc2, msc_norm2 = fit_FA(np.array(Y2.T))

        ##############################
        ### OVERLAP BETWEEN MODELS ###
        ##############################
        # Over lap fit separately 
        proj_1_0_ind = np.trace(msc_norm*msc2*msc_norm.T)/float(np.trace(msc2))
        proj_0_1_ind = np.trace(msc_norm2*msc*msc_norm2.T)/float(np.trace(msc))

        ######################
        ### FA MODELS COMB ###
        ######################
        Y_all = np.hstack((Y1, Y2))
        sot, msc, msc_norm, X_pred, U_fit = fit_FA(np.array(Y_all.T), predict_X=True)

        ##########################
        ### GET U_FIT1, U_FIT2 ###
        ##########################
        # # Now find subspace overlap b/w main shared covariances: 
        X0_pred = X_pred[:T, :]# - np.mean(X_pred[:T, :],axis=0)[np.newaxis, :]
        X1_pred = X_pred[T:, :]# - np.mean(X_pred[T:, :],axis=0)[np.newaxis, :]

        #C0 = np.linalg.lstsq(X0_pred, Y1.T)[0].T
        #C1 = np.linalg.lstsq(X1_pred, Y2.T)[0].T

        #Z_cov_0 = C0*C0.T
        #Z_cov_1 = C1*C1.T

        Z_cov_0 = U_fit*np.cov(X0_pred.T)*U_fit.T
        Z_cov_1 = U_fit*np.cov(X1_pred.T)*U_fit.T

        Z_cov_0_mn, Z_cov_0_norm = get_mn_shar(Z_cov_0)
        Z_cov_1_mn, Z_cov_1_norm = get_mn_shar(Z_cov_1)

        proj_1_0_comb = np.trace(Z_cov_0_norm*Z_cov_1_mn*Z_cov_0_norm.T)/float(np.trace(Z_cov_1_mn))
        proj_0_1_comb = np.trace(Z_cov_1_norm*Z_cov_0_mn*Z_cov_1_norm.T)/float(np.trace(Z_cov_0_mn))

        ### REP OV #####
        from online_analysis import fit_LDS
        a0, a1 = fit_LDS.get_repertoire_ov(X0_pred, X1_pred)

        ax.plot(range(4), [proj_1_0, proj_1_0_ind, proj_1_0_comb, a1], '.-')
        ax2.plot(range(4), [proj_0_1, proj_0_1_ind, proj_0_1_comb, a0], '.-')

    ax.set_title('FA Simulations')
    ax.set_ylabel('Ov: 1--> 0')
    ax2.set_ylabel('Ov: 0--> 1')
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(['True Ov', 'Ind Ov', 'Comb Ov', 'Comb Rep'])

def get_mn_shar(UUT, perc=.9):
    u, s, v = np.linalg.svd(UUT)
    ix = np.nonzero(np.cumsum(s**2)/float(np.sum(s**2))>perc)[0]
    s_red = np.zeros_like(s)
    print 'nf: B: ', ix[0]+1
    s_red[:ix[0]+1] = s[:ix[0]+1]
    s_ones = np.zeros_like(s)
    s_ones[:ix[0]+1] = 1
    U_norm = u*np.diag(s_ones)*v
    U_mn = u*np.diag(s_red)*v
    return U_mn, U_norm

def fit_FA(Y, predict_X=False):
    assert Y.shape[0] > Y.shape[1]
    log_lik, perc_shar_var = pa.find_k_FA(Y, iters=1, max_k = 10,  plot=False)
    nan_ix = np.isnan(log_lik)
    samp = np.sum(nan_ix==False, axis=0)
    ll = np.nansum(log_lik, axis=0)
    LL_new = np.divide(ll, samp)

    num_factors = 1+(np.argmax(LL_new))
    if num_factors < 2:
        num_factors = 2
        print 'forcing min nm_factors to be 2'
    print 'optimal LL factors: ', num_factors

    FA = skdecomp.FactorAnalysis(n_components=num_factors)
    FA.fit(Y)
    
    if predict_X:
        X = FA.transform(Y)

    U = np.mat(FA.components_).T
    i = np.diag_indices(U.shape[0])
    Psi = np.mat(np.zeros((U.shape[0], U.shape[0])))
    Psi[i] = FA.noise_variance_

    shar_cov = U*U.T
    inv_shar_pls_psi = np.linalg.inv(shar_cov + Psi)
    
    # Main shared: 
    u, s, v = np.linalg.svd(shar_cov)
    s_red = np.zeros_like(s)
    s_bin = np.zeros_like(s)
    s_hd = np.zeros_like(s)
    
    ix = np.nonzero(np.cumsum(s**2)/float(np.sum(s**2))>.90)[0]
    if len(ix) > 0:
        n_dim_main_shared = ix[0]+1
    else:
        n_dim_main_shared = len(s)
    if n_dim_main_shared < 2:
        n_dim_main_shared = 2
    print "main shared: n_dim: ", n_dim_main_shared, np.cumsum(s)/float(np.sum(s))

    s_red[:n_dim_main_shared] = s[:n_dim_main_shared]
    s_bin[:n_dim_main_shared] = 1
    s_hd[n_dim_main_shared:] = s[n_dim_main_shared:]
    main_shared_cov = u*np.diag(s_red)*v
    main_hd_cov = u*np.diag(s_hd)*v
    sot = np.trace(main_shared_cov)/(np.trace(main_shared_cov)+np.trace(main_hd_cov)+np.trace(Psi))

    main_shared_cov_norm = u*np.diag(s_bin)*v

    if predict_X:
        return sot, main_shared_cov, main_shared_cov_norm, X, U
    else:
        return sot, main_shared_cov, main_shared_cov_norm

def fit_LDS_sim(Y, nstates=10, predict_X=False):

    kwargs = dict(seed_pred_x0_with_smoothed_x0=True, 
        seed_w_FA=True, nms=['Spks'], plot=False, 
        get_ratio=False, 
        pre_go_bins=0, fa_thresh =.9)

    #nstates = 10

    #################
    ### LDS ###
    #################
    from online_analysis import fit_LDS
    R2_smooth, R2_pred, LL, dynamics_norm, innov_norm, model, pred_data, smooth_data, filt_state, pred_state, filt_sigma, kg = fit_LDS.fit_LDS([Y], 
        [Y], nstates, nEMiters=30, return_model=True, **kwargs)

    U = np.mat(model.C)
    i = np.diag_indices(U.shape[0])    
    shar_cov = U*U.T
    
    # Main shared: 
    u, s, v = np.linalg.svd(shar_cov)
    s_red = np.zeros_like(s)
    s_bin = np.zeros_like(s)
    
    ix = np.nonzero(np.cumsum(s**2)/float(np.sum(s**2))>.90)[0]
    if len(ix) > 0:
        n_dim_main_shared = ix[0]+1
    else:
        n_dim_main_shared = len(s)
    if n_dim_main_shared < 2:
        n_dim_main_shared = 2
    print "main shared: n_dim: ", n_dim_main_shared, np.cumsum(s)/float(np.sum(s))

    s_red[:n_dim_main_shared] = s[:n_dim_main_shared]
    s_bin[:n_dim_main_shared] = 1
    main_shared_cov = u*np.diag(s_red)*v
    main_shared_cov_norm = u*np.diag(s_bin)*v

    if predict_X:
        x_smooth = model.states_list[0].smoothed_mus   
        return main_shared_cov, main_shared_cov_norm, x_smooth, U
    else:
        return main_shared_cov, main_shared_cov_norm

def run_LDS_sim():
    f, ax0 = plt.subplots(nrows=2)
    ax = ax0[0]
    ax2 = ax0[1]
    from pylds.models import DefaultLDS
    # Set parameters
    D_obs = 40
    D_latent = 3
    D_input = 0
    T = 500

    for i in range(5):
        # Simulate from one LDS
        truemodel = DefaultLDS(D_obs, D_latent, D_input)
        data, stateseq = truemodel.generate(T)
        C = truemodel.C

        truemodel2 = DefaultLDS(D_obs, D_latent, D_input)
        truemodel2.C[:, -1] = C[:, -1]
        truemodel2.sigma_obs = truemodel.sigma_obs
        truemodel2.sigma_states = truemodel.sigma_states
        data2 = truemodel2.C*np.mat(stateseq).T + np.mat(np.random.multivariate_normal(np.zeros((D_obs, )), truemodel2.sigma_obs, T)).T

        colors = ['k', 'b','r']
        for i_n, nstates in enumerate(range(5, 20, 6)):
            ###########################
            ### Get actual overlap: ###
            ###########################
            U_mn, U_norm = get_mn_shar(np.mat(C)*C.T)
            U2_mn, U2_norm = get_mn_shar(np.mat(truemodel2.C)*truemodel2.C.T)

            # Actual overlap based on actual Us
            proj_0_1 = np.trace(U2_norm*U_mn*U2_norm.T)/float(np.trace(U_mn))
            proj_1_0 = np.trace(U_norm*U2_mn*U_norm.T)/float(np.trace(U2_mn))

            ###########################
            ### Get separate overlap: ###
            ###########################

            #####################
            ### FA MODELS 1+2 ###
            #####################
            msc, msc_norm = fit_LDS_sim(np.array(data), nstates=nstates)
            msc2, msc_norm2 = fit_LDS_sim(np.array(data2.T), nstates=nstates)

            ##############################
            ### OVERLAP BETWEEN MODELS ###
            ##############################
            # Over lap fit separately 
            proj_1_0_ind = np.trace(msc_norm*msc2*msc_norm.T)/float(np.trace(msc2))
            proj_0_1_ind = np.trace(msc_norm2*msc*msc_norm2.T)/float(np.trace(msc))

            ####################
            ###  MODELS COMB ###
            ####################
            data_all = np.vstack((data, data2.T))
            msc, msc_norm, X_pred, U_fit = fit_LDS_sim(np.array(data_all), predict_X=True, nstates=nstates)

            ##########################
            ### GET U_FIT1, U_FIT2 ###
            ##########################
            # # Now find subspace overlap b/w main shared covariances: 
            X0_pred = X_pred[:T, :]# - np.mean(X_pred[:T, :],axis=0)[np.newaxis, :]
            X1_pred = X_pred[T:, :]# - np.mean(X_pred[T:, :],axis=0)[np.newaxis, :]

            Z_cov_0 = U_fit*np.cov(X0_pred.T)*U_fit.T
            Z_cov_1 = U_fit*np.cov(X1_pred.T)*U_fit.T

            Z_cov_0_mn, Z_cov_0_norm = get_mn_shar(Z_cov_0)
            Z_cov_1_mn, Z_cov_1_norm = get_mn_shar(Z_cov_1)

            proj_1_0_comb = np.trace(Z_cov_0_norm*Z_cov_1_mn*Z_cov_0_norm.T)/float(np.trace(Z_cov_1_mn))
            proj_0_1_comb = np.trace(Z_cov_1_norm*Z_cov_0_mn*Z_cov_1_norm.T)/float(np.trace(Z_cov_0_mn))

            ##############################
            ### GET Repertoire Overlap ###
            ##############################
            from online_analysis import fit_LDS
            a0, a1 = fit_LDS.get_repertoire_ov(X0_pred, X1_pred)
            ax.plot(range(4), [proj_1_0, proj_1_0_ind, proj_1_0_comb, a1], '.-', color=colors[i_n])
            ax2.plot(range(4), [proj_0_1, proj_0_1_ind, proj_0_1_comb, a0], '.-', color=colors[i_n])
    ax.set_title('Blk: n = 5, Blue: n = 11, Red: n = 17')
    ax.set_ylabel('Ov: 1--> 0')
    ax2.set_ylabel('Ov: 0--> 1')
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(['True Ov', 'Ind Ov', 'Comb Ov', 'Comb Rep'])
