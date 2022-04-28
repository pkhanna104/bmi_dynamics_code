import numpy as np 
import ssm
from online_analysis import generate_models


def fit_slds(data_temp_dict, x_var_names, trl_train_ix, num_states): 
    '''
    method to fit recurrance-only rSLDS model (similar to nascar example)

    latent_dimensionality is full dimensionality of data (or closest to full possible)
    '''

    ### Aggregate the variable name
    x = [] #### So this is X_{t-1}, so t=0:end_trial(-1)
    for vr in x_var_names:
        x.append(data_temp_dict[vr][: , np.newaxis])
    X = np.hstack((x))

    assert(X.shape[1] == len(x_var_names))
    numN = X.shape[1]

    ### Only neurons with FR > 0.5 ### 
    keep_ix = np.nonzero(10*np.sum(X, axis=0)/float(X.shape[0]) > 0.5)[0]

    #### Retrospectively check variable order ####
    generate_models.check_variables_order(x_var_names, numN)

    #### Make trials #####
    trls = data_temp_dict['trl'].to_numpy()
    bin_num = data_temp_dict['bin_num'].to_numpy()

    ##### Append to list ####
    train_trls = []
    for i_t in trl_train_ix: 
        ix = np.nonzero(trls==i_t)[0]
        assert(np.all(np.diff(bin_num[ix]) == 1))
        train_trls.append(X[np.ix_(ix, keep_ix)])

    #### Now get LDS based on these trials ###
    D_obs = len(keep_ix)
    D_input = 0; 
    n_dim_latent = D_obs
    try_dims = np.arange(n_dim_latent, 15, -1)

    for n_dim_latent_try in try_dims:
        print(('starting to try to fit Dim %d' %(n_dim_latent_try)))
        
        rslds_lem = ssm.SLDS(D_obs, num_states, n_dim_latent_try,
                     transitions="recurrent_only",
                     dynamics="gaussian",
                     emissions="gaussian",
                     single_subspace=True)

        # Initialize the model with the observed data.  It is important
        # to call this before constructing the variational posterior since
        # the posterior constructor initialization looks at the rSLDS parameters.
        rslds_lem.initialize(train_trls)

        #############
        # Train sLDS #
        ############# 
        try:
            q_elbos_lem, q_lem = rslds_lem.fit(train_trls, method="laplace_em",
                                   variational_posterior="structured_meanfield",
                                   initialize=False, num_iters=1000, alpha=0.0)

            ####################
            ### #get out LL? ###
            z_hat = q_lem.discrete_expectations
            x_hat = q_lem.mean_continuous_states

            neg_log_prob = 0; 
            for trl in range(len(train_trls)): 

                ### Expected z ###
                Ez, Ezzp1, _ = z_hat[trl] ### Expected 
            
                ### negative expected log something (lower is better)
                neg_log_prob += rslds._laplace_neg_expected_log_joint(train_trls[trl],
                                                 np.zeros((train_trls[trl].shape[0], 0)),
                                                 np.ones_like(train_trls[trl]), 
                                                 None, 
                                                 x_hat[trl],
                                                 Ez,
                                                 Ezzp1)
            return -1*neg_log_prob
        except: 
            pass

    ### If you've gotten here, you've failed to fit any 
    raise Exception('Cant fit any models above dim 15')

