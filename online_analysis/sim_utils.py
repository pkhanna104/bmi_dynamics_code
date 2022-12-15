import pickle 
import numpy as np 
import scipy.stats
import matplotlib.pyplot as plt 
import copy

from sklearn.linear_model import LinearRegression

def return_lqr_data(model = 'n_do', simulation='df_lqr_n', dyn='full', add_noise_no = False): 
    '''
    options to get out data from vivek's DF
    inputs: 
        model: 'n_do' or 'n_o' --> dynamics +offset or just offset
        simulation: 'df_lqr_n' or 'df_lqr_nl' --> noisy or noiseless
        dyn: 'full', 'null'
        add_noise_no: whether or not to add extra null noise to the n_o simulation
    '''
    ## Fixed point simulations -- 
    if dyn == 'null':   
        df = pickle.load(open('/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/BMI_co_obs_paper/data/lqr_init_fp_decoder_null_noisy.pkl'))
    elif dyn == 'full': 
        df = pickle.load(open('/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/BMI_co_obs_paper/data/lqr_init_fp_full_noisy.pkl'))
    
    N = df['num_neurons']
    cols = ['n_%d'%n for n in range(N)]
    
    ### activity 
    ### "'df_lqr_n', it's the data frame storing the noisy lqr simulation"
    ### @Preeya fyi, the variable 'df_lqr_nl' in the data I sent you is the noiseless simulation, and the dataframe has the same structure as 'df_lqr_n'
    act = np.array(df[simulation].loc[:, cols])
    
    ### Get command ID \n",
    ### u_v_mag_bin, u_v_angle_bin are the bins for magnitude and angle of command
    mag = np.array(df[simulation].loc[:, 'u_v_mag_bin'])
    ang = np.array(df[simulation].loc[:, 'u_v_angle_bin'])
    
    command = mag*8 + ang; # calculate command (if mag == 4, command will be 32-39)
    assert(np.all(command[mag == 4] >= 32))
    
    ## Get condition 
    target = np.array(df[simulation].loc[:, 'target'])
    targ_rot = np.array(df[simulation].loc[:, 'task_rot'])
    targ_rot[targ_rot == 1.1] = 1.
    targ_rot[targ_rot == 1.2] = 2.
    condition = targ_rot*8 + target
    
    ### make sure only 24 conditions (co*8, obs-cw*8, obs-ccw*8)
    assert(len(np.unique(condition)) == 24)
    assert(np.allclose(np.unique(condition), np.arange(24)))

    #'model' is 'n_do' = dynamics+offset, 'n_o' = offset only, dynamics=0 
    model_all = np.array(df[simulation].loc[:, 'model'])
    ix_ = np.nonzero(model_all == model)[0]
    
    #the variable 'Kn' is the decoder
    KG = np.array(df['Kn'])
    KG_vel = KG[[2, 3], :] # 0,1 == pos || 2, 3 == vel || 4 == offset
    potent = np.dot(KG_vel, act.T).T
    
    ### Get decompositiions of activity 
    KG_null, KG_pot, KG_nullNtoNmin2 = get_null_pot_decomp(KG_vel)
    
    null_act = np.dot(KG_null, act.T).T
    pot_act = np.dot(KG_pot, act.T).T
    
    ### check all null data is zero 
    assert(np.allclose(np.dot(KG_vel, null_act.T), 0.))
    
    ### check all potent data is equal to potent
    assert(np.allclose(np.dot(KG_vel, pot_act.T).T, potent))

    #### Check that null and potent activity are not correlated; 
    potent2D = potent[ix_, :] # N x 2 
    nullNmin2D = np.dot(KG_nullNtoNmin2.T, act[ix_, :].T).T 
    assert(potent2D.shape[0] == nullNmin2D.shape[0])
    assert(potent2D.shape[1] == 2)
    assert(nullNmin2D.shape[1] == act.shape[1] - 2)

    ### linear correlation b/w potent and null noise; 
    ### x / y --> samples x features
    n_folds = 2; 
    Ntot = nullNmin2D.shape[0]
    Nperfold = int(np.floor(Ntot/float(n_folds)))
    
    #print('Total pts : %d, Pts per fold %d'%(Ntot, Nperfold))
    ix_all = np.arange(Ntot)
    scores = []
    
    for nf in range(n_folds-1): 
        train_ix_sub = ix_all[(nf*Nperfold):((nf+1)*Nperfold)]
        test_ix_sub = ix_all[((nf+1)*Nperfold):((nf+2)*Nperfold)]
        #print('Fold %d, start %d, end %d'%(nf, ix_sub[0], ix_sub[-1]))

        y_mn = np.mean(potent2D[train_ix_sub, :], axis=0)
        y_demean_train = potent2D[train_ix_sub, :] - y_mn[np.newaxis, :]
        assert(np.allclose(np.mean(y_demean_train, axis=0), 0.))

        ## Regression 
        reg = LinearRegression().fit(nullNmin2D[train_ix_sub, :], y_demean_train)
        
        ### Scores on held-out data 
        scores.append(reg.score(nullNmin2D[test_ix_sub, :], potent2D[test_ix_sub, :] - y_mn[np.newaxis, :]))
        print(reg.coef_)
        print(reg.intercept_)
        
        print('Fold %d, Model %s, score %.5f'%(nf, model, scores[nf]))
    print('MEAN: Model %s, score %.5f'%( model, np.mean(scores)))

    print('Model %s, index size: %d'%(model, Ntot))

    if add_noise_no:# and model == 'n_o': 
        nNeurons = act.shape[1]
        noise = 10*np.random.randn(len(ix_), nNeurons)

        ## Project to nullspace -- KG_null is n x N 
        null_noise = np.dot(KG_null, noise.T).T

        assert(null_noise.shape[1] == nNeurons)
        assert(null_noise.shape[0] == len(ix_))

        ### add to activity 
        act[ix_, :] = act[ix_, :] + null_noise 

        ## add to null : 
        print(model)
        print(' before adding noise: var null : %.4f'%(np.sum(np.var(null_act[ix_, :], axis=0))))
        null_act[ix_, :] = null_act[ix_, :] + null_noise
        print(' after adding noise: var null : %.4f'%(np.sum(np.var(null_act[ix_, :], axis=0))))

    return act[ix_, :], command[ix_], condition[ix_], potent[ix_], null_act[ix_], pot_act[ix_], scores

def get_null_pot_decomp(KG_potent): 
    '''
    assume KG is 2 x N
    '''
    #F, KG = decoder.filt.get_sskf()
    #KG_potent = KG[[3, 5], :]; # 2 x N
    
    ## #get null space from potent 
    KG_null = scipy.linalg.null_space(KG_potent) # N x (N-2)
    KG_null_proj = np.dot(KG_null, KG_null.T) # N x N 

    ## Get KG potent too; 
    U, S, Vh = scipy.linalg.svd(KG_potent); #[2x2, 2, 44x44]
    Va = np.zeros_like(Vh) # N x N
    Va[:2, :] = Vh[:2, :] # N x N 
    KG_potent_orth = np.dot(Va.T, Va) # N x N 
    return KG_null_proj, KG_potent_orth, KG_null

def get_cond_com_pls_matched_inds(commands, condition, activity, potent, nshuff=10, pv=.5, 
    min_pool_factor=1., max_pool_factor = 10000., min_commands = 15, max_commands = 1000): 
    '''
    method that takes in command numbers // condition number // activity 
    '''
    assert(len(commands) == len(condition) == activity.shape[0])
    cc_dist = {}
    shuff_cc_dist = {}
    
    cc_var = {}
    pool_var = {}

    for com in np.arange(32): # dont' include commands for 4*8 onwards 
        assert(com < 32)
        
        for cond in np.unique(condition): # all conditions 

            ## command-condition 
            ix_cc = np.nonzero(np.logical_and(commands == com, condition == cond))[0]
            
            if len(ix_cc) > min_commands and len(ix_cc) < max_commands: 
                
                ## Find condition pool that matches this 
                ix_com = np.nonzero(commands == com)[0]

                ## Return indices that match 
                #print('Starting match cond: %d, com %d'%(cond, com))
                ix_com = match_ix(ix_cc, ix_com, potent, pv=pv)
                
                if ix_com is not None:

                    pool_factor = float(len(ix_com))/float(len(ix_cc))

                    if np.logical_and(pool_factor > min_pool_factor, pool_factor < max_pool_factor): 
                        #print('Command %d Cond %d, ix_cc N=%d, ix_com N=%d'%(com, cond, len(ix_cc), len(ix_com)))

                        shuff_cc_dist[com, cond] = []
                        pool_var[com, cond] = np.sum(np.var(activity[ix_com, :], axis=0))

                        # Global pool
                        pool_mn = np.mean(activity[ix_com, :], axis=0)

                        ### ix_cc is maybe not in ix_com? 
                        ix_cc2 = np.array([i for i in ix_cc if i in ix_com])

                        assert(len(ix_cc2) == len(ix_cc))

                        if len(ix_cc2) > 0: 
        
                            # Shuffle_dist
                            tmp_mn = []; 
                            for i_shuff in range(nshuff): 

                               ## which indices 
                                ix_ = np.random.permutation(len(ix_com))[:len(ix_cc2)]
                                mn_ = np.mean(activity[ix_com[ix_], :], axis=0)
                                tmp_mn.append(mn_)
                                shuff_cc_dist[com,cond].append(np.linalg.norm(pool_mn - mn_))

                            cc_dist[com, cond] = np.linalg.norm(pool_mn - np.mean(activity[ix_cc2, :], axis=0))
                            cc_var[com, cond] = np.sum(np.var(activity[ix_cc2, :], axis=0))

                            mn = np.mean(shuff_cc_dist[com,cond])
                            std = np.std(shuff_cc_dist[com,cond])
                            z_dat = (cc_dist[com, cond] - mn) / std

                            if z_dat <-10 : 
                                pot_cc = potent[ix_cc, :]

                                ix_com_og = np.nonzero(commands == com)[0]
                                pool_cc_og = potent[ix_com_og, :]

                                pool_cc = potent[ix_com, :]

                                f, ax= plt.subplots()
                                ax.set_title('K=Pool og, B=cond, R=new pool\n com%d cond %.1f'%(com, cond))
                                ax.plot(pool_cc_og[:, 0], pool_cc_og[:, 1], 'k.')
                                ax.plot(pool_cc[:, 0], pool_cc[:, 1], 'r.')
                                ax.plot(pot_cc[:, 0], pot_cc[:, 1], 'b.', alpha=.5)

                                pot_mn = np.mean(pool_cc, axis=0)
                                pot_mn2 = np.mean(pot_cc, axis=0)

                                ax.plot(pot_mn[0], pot_mn[1], 'r*') ### pool; 
                                ax.plot(pot_mn2[0], pot_mn2[1], 'b*') ### command-cond 

                                ax.plot([pot_mn[0], pot_mn2[0]], [pot_mn[1], pot_mn2[1]], 'b-')
                                ax.set_xlabel('X vel')
                                ax.set_ylabel('Y vel')
                        
                        
                        #z = (cc_dist[com, cond] - np.mean(shuff_cc_dist[com,cond])) / np.std(shuff_cc_dist[com,cond])
                        # if z > 8: 
                        #     f, ax = plt.subplots()
                        #     DF = []
                        #     for i in range(nshuff): 
                        #         DF.append(tmp_mn[i] - pool_mn)
                        #     DF = np.vstack((DF))
                        #     DF2 = []
                        #     for i in range(len(pool_mn)): DF2.append(DF[:, i])

                        #     ax.violinplot(DF2, positions=np.arange(len(pool_mn)))
                        #     df_tru = np.mean(activity[ix_cc2, :], axis=0) - pool_mn
                        #     ax.plot(df_tru, '-*')
                        #     ax.set_title('Com %d, Cond %d' %(com,cond))

    return cc_dist, shuff_cc_dist, cc_var, pool_var

def match_ix(ix0, ix1, val, pv=.5): 
    '''
    method to do matching
    '''
    assert(val.shape[1] == 2) # 2D command 

    ix1_keep = np.arange(len(ix1))
    last_ix1_keep_len = len(ix1_keep)

    _, px = scipy.stats.ttest_ind(val[ix0, 0], val[ix1[ix1_keep], 0])
    _, py = scipy.stats.ttest_ind(val[ix0, 1], val[ix1[ix1_keep], 1])

    ### mean / std 
    mn0 = copy.copy(np.mean(val[ix0, :], axis=0))
    st0 = copy.copy(np.std(val[ix0, :], axis=0))

    ### z-score of value 1; 
    #zsc = (val1[ix1_keep, :] - mn0[np.newaxis, :])/st0[np.newaxis, :]
    #cost_og = np.sum(np.square(zsc), axis=1)

    #f, ax= plt.subplots()
    #ax.plot(mn0[0], mn0[1], 'k*', markersize=20)
    #ax.scatter(val1[:, 0], val1[:, 1], s=None, c=cost_og, cmap='viridis')

    while np.logical_or(px < pv, py < pv): 

        #### index by ix1_keep; 
        zsc = (val[ix1[ix1_keep], :] - mn0[np.newaxis, :])/st0[np.newaxis, :]

        #Return the element-wise square of the input
        cost = np.sum(np.square(zsc), axis=1)

        ### make cost of the command-condition indices zero 
        ix_ix0 = np.array([i for i, j in enumerate(ix1[ix1_keep]) if j in ix0])
        assert(len(ix_ix0) == len(ix0))
        cost[ix_ix0] = 0.

        assert(len(cost) == len(ix1_keep))

        # Drop last 5%: 
        N = len(ix1_keep)
        keep_N = int(np.round(.9*N))

        ### get indices that you want to subselect 
        ix1_keep_sub = np.argsort(cost)[:keep_N]

        ### update ix1_keep with the ones to keep 
        ix1_keep = ix1_keep[ix1_keep_sub]

        ##This is in indices of ix1[ix]
        ix1_keep = np.sort(ix1_keep)
        #print('Len ix1 %d'%len(ix1_keep))

        if len(ix1_keep) == last_ix1_keep_len: 
            break
        else: 
            last_ix1_keep_len = len(ix1_keep)
            _, px = scipy.stats.ttest_ind(val[ix0, 0], val[ix1[ix1_keep], 0])
            _, py = scipy.stats.ttest_ind(val[ix0, 1], val[ix1[ix1_keep], 1])
        
        if len(ix1_keep) <= len(ix0): 
            break

    if np.logical_and(px >= pv, py >= pv):
        #print('Success ix1 %d'%len(ix1_keep))
        return ix1[ix1_keep]

    else:
        #print('Failure')
        return None

def plot2(cc_dist, shuff_cc_dist, x, axi):
    Z = []; 
    for i_k, k in enumerate(cc_dist.keys()):
        mn_shuff = np.mean(shuff_cc_dist[k])
        st_shuff = np.std(shuff_cc_dist[k])
        dat = cc_dist[k]
        z_dat = (dat - mn_shuff) / st_shuff
        
        # if z_dat > 4:
        #     print('Dropping %s, z_dat=%.2f'%(str(k), z_dat))
        # else: 
        axi.plot(x+np.random.randn()*.1, z_dat, 'b.')
        Z.append(z_dat)
    
    axi.bar(x, np.mean(Z), color='b', alpha=.5)
    mn = np.mean(Z) - 2*np.std(Z)
    mx = np.mean(Z) + 2*np.std(Z)
    axi.plot([x, x], [mn, mx], 'k-')
    print('mn %.3f, max %.3f'%(mn, mx))
    return axi

def plot(cc_dist, shuff_cc_dist, title):

    f, ax = plt.subplots(nrows = 8, ncols = 4, figsize = (10, 16))

    for i_k, k in enumerate(cc_dist.keys()):    
        com, cond = k; 
        axi = ax[ com % 8, int(np.floor(com/8))]
        axi.plot(cond, cc_dist[k], 'k.', alpha=.5)
        
        axi.violinplot(shuff_cc_dist[k], positions=[cond])
        #axi.plot([cond, cond], [mn_shuff - 2*st_shuff, mn_shuff+2*st_shuff], 'r-')
        axi.set_title('Com: %d'%com, fontsize=8)
        axi.set_xlim([0., 24.])
        axi.set_ylim([0., 1.0])
	#ax.set_title(title)
	f.tight_layout()