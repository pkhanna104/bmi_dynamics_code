#################################
####### View Tuning Diffs #######
#################################

# Method to compute the mFR of each neuron | bin of the neural push velocity commands
# Also computes the mFR of each neuron | subsets of nerual push velocity commands

import scipy.io as sio
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tables
import prelim_analysis as pa
import math
import fit_LDS
import sklearn.decomposition as skdecomp

fname_pref = {}
fname_pref['jeev'] = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/testing_jeev_16dayday'
fname_pref['grom'] = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/testing_grom_16trls_by_targday'
fname_pref['grom_prego'] = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/testing_grom_16trls_by_targ_prego1day'



important = {}
important['jeev'] = '/Users/preeyakhanna/fa_analysis/online_analysis/jeev_important_neurons_svd_feb2019.pkl'
important['grom'] = '/Users/preeyakhanna/fa_analysis/online_analysis/grom_important_neurons_svd_feb2019.pkl'

days = {}
days['grom'] = 9
days['jeev'] = 4

mag_bins = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/radial_boundaries_fit_based_on_perc_feb_2019.pkl'))
cmap_list = ['maroon', 'orangered', 'darkgoldenrod', 'olivedrab', 'teal', 'steelblue', 'midnightblue', 'darkmagenta']
pref = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/'


def get_tuning_diffs(zscore_neurons = True, important_neurons = False,
    animal = 'grom', min_observations = 30, obs_by_targ = True, 
    main_ov = True, obs_targs_to_process = [0.], dayz = None):
    
    KL_all = {}
    KL_all['important_neurons'] = important_neurons
    KL_all['bins_used'] = np.zeros((8, 4, 2))

    if dayz is None:
        days_a = np.arange(days[animal])
    else:
        days_a = dayz

    KL_all['days'] = days_a

    for select_day in days_a:

        print 'day: ', select_day

        ### Load the distribution ###
        fname = fname_pref[animal] + str(select_day)
        #dat = sio.loadmat(fname+'.mat')
        #dat_dist = sio.loadmat(fname+'dist'+'.mat')
        dat = pickle.load(open(fname+'.pkl'))
        dat_dist = pickle.load(open(fname+'dist'+'.pkl'))

        # Get the first task key: 
        key_co = ('day'+str(select_day), 'tsk0', 'n0')
        key_ob = ('day'+str(select_day), 'tsk1', 'n0')

        hist_co = [dat[key_co]] # bins x bins x neurons x hist
        hist_obs = [dat[key_ob]] # bins x bins x neurons x hist

        hist_co_dist = [dat_dist[key_co]]
        hist_obs_dist = [dat_dist[key_ob]]  
        
        # Aggregate the other blocks: 
        for n in range(1, 100):
            try: 
                key_co = ('day'+str(select_day), 'tsk0', 'n'+str(n))
                for i in range(8):
                    hist_co[i].append(dat[key_co][i])
                    hist_co_dist[i].append(dat[key_co][i])
            except:
                pass
            
            try:
                key_ob = ('day'+str(select_day), 'tsk1', 'n'+str(n))
                for i in range(8):
                    hist_obs[i].append(dat[key_obs][i])
                    hist_obs_dist[i].append(dat[key_obs][i])
            except:
                pass

        #########################################
        ######### Get mean firing rate ##########
        #########################################
        co_nbins = [dat[(('day'+str(select_day), 'tsk0', 'n0'), 'binned_spikes')]]
        obs_nbins = [dat[(('day'+str(select_day), 'tsk1', 'n0'), 'binned_spikes')]]
        
        # Used later for covariance estimation: 
        co_angs = [dat[(('day'+str(select_day), 'tsk0', 'n0'), 'ang')]]
        co_mags = [dat[(('day'+str(select_day), 'tsk0', 'n0'), 'mag')]]
        co_targs = [dat[(('day'+str(select_day), 'tsk0', 'n0'), 'targ_ix')]]

        obs_angs = [dat[(('day'+str(select_day), 'tsk1', 'n0'), 'ang')]]
        obs_mags = [dat[(('day'+str(select_day), 'tsk1', 'n0'), 'mag')]]
        obs_targs = [dat[(('day'+str(select_day), 'tsk1', 'n0'), 'targ_ix')]]

        for i in range(1, 100):
            try:
                co_nbins.append(dat[(('day'+str(select_day), 'tsk0', 'n'+str(i)), 'binned_spikes')])
                co_angs.append(dat[(('day'+str(select_day), 'tsk0', 'n'+str(i)), 'ang')])
                co_mags.append(dat[(('day'+str(select_day), 'tsk0', 'n'+str(i)), 'mag')])
                co_targs.append(dat[(('day'+str(select_day), 'tsk0', 'n'+str(i)), 'targ_ix')])
            except:
                pass
            
            try:
                obs_nbins.append(dat[(('day'+str(select_day), 'tsk1', 'n'+str(i)), 'binned_spikes')])  
                obs_angs.append(dat[(('day'+str(select_day), 'tsk1', 'n'+str(i)), 'ang')])
                obs_mags.append(dat[(('day'+str(select_day), 'tsk1', 'n'+str(i)), 'mag')])
                obs_targs.append(dat[(('day'+str(select_day), 'tsk1', 'n'+str(i)), 'targ_ix')])
            except:
                pass

        co_nbins = np.vstack((co_nbins))
        obs_nbins = np.vstack((obs_nbins))

        co_angs = np.hstack((np.hstack((co_angs))))
        co_mags = np.hstack((np.hstack((co_mags)) ))
        co_targs = np.hstack((np.hstack((co_targs)) ))

        obs_angs = np.hstack((np.hstack((obs_angs)) ))
        obs_mags = np.hstack((np.hstack((obs_mags))))
        obs_targs = np.hstack((np.hstack((obs_targs)) ))

        all_nbins = np.vstack((co_nbins, obs_nbins))
        all_mean = np.mean(all_nbins, axis=0)
        all_std = np.std(all_nbins, axis=0)

        ###########################################
        #######  Get Mean Tuning and KL Div #######
        ###########################################
        #bins_ang, bins_mag, _, _ = hist_co.shape
        bins_ang = 8
        bins_mag = 4
        nd = 40

        important_n = pickle.load(open(important[animal]))
        if important_neurons:
            neurons = np.sort(important_n[select_day, animal, 'est'])
        else:
            neurons = np.arange(co_nbins[0].shape[0])
        nneurons = len(neurons)

        if obs_by_targ:
            if obs_targs_to_process is None:
                n_targs = np.arange(8.)
            else:
                n_targs = obs_targs_to_process
        else:
            n_targs = [-1]

        KL_all['n_targs'] = n_targs

        for targ in n_targs:
            ftarg, axtarg = plt.subplots(ncols=2)
            axtarg[0].set_title('Day: '+str(select_day)+', Targ: '+str(targ))

            print 'target nunm: ', targ
            
            if targ == -1:
                targ_spec_obs_ix = np.arange(len(obs_targs))
                targ_spec_co_ix = np.arange(len(obs_targs))
                
            else:
                targ_spec_obs_ix = np.nonzero(obs_targs == targ)[0]
                targ_spec_co_ix = np.nonzero(co_targs == targ)[0]
                
            # Task means
            co_mn  = np.zeros((bins_ang, bins_mag, nneurons))
            obs_mn = np.zeros_like(co_mn)
            
            Lambda = np.zeros((bins_ang, bins_mag, nneurons, 3 )) # Mean CO, Mean Obs, p-value diff

            DLambda = np.zeros((bins_ang, bins_mag, nneurons, 2))
            DLambda_Dist = np.zeros((bins_ang, bins_mag, nneurons, 2))
            
            DLambda_overall = np.zeros((nneurons, 2))
            DLambda_overall_dist = np.zeros((nneurons, 2, 2))

            DCov = np.zeros(( bins_ang, bins_mag, 2)) # across
            DCov_dist = np.zeros(( bins_ang, bins_mag, 2)) # within

            DCov[:, :, :] = np.nan
            DCov_dist[:, :, :] = np.nan

            Distb = np.zeros(( 9, 5, 2 )) # Distribution fo pushes, CO then OBS

            #nb0, nb1, nn, nd = hist_co.shape # Number of bins, number of bins, number of nuerons, number of distributions
            nb0 = bins_ang; nb1 = bins_mag; nn = nneurons;

            for in_, n in enumerate(neurons):

                print 'neurons: ', n
                ###################################     
                ## Also compute KS differences: ###
                ###################################
                dict_lists = {}
                        
                ###################################
                ### Total mFRs ###
                ###################################
                
                # Bins x FR dist
                #ix_co = hist_co[:, :, n, :].reshape(nb0*nb1, nd)
                #ix_ob = hist_obs[:, :, n, :].reshape(nb0*nb1, nd)

                # BIN 
                ix_co_og = get_hist(co_nbins, 
                    co_angs, co_mags, n)

                ix_co_og_targ = get_hist(co_nbins[targ_spec_co_ix, :],
                    co_angs[targ_spec_co_ix], co_mags[targ_spec_co_ix], n)

                ix_obs_og = get_hist(obs_nbins[targ_spec_obs_ix, :], 
                    obs_angs[targ_spec_obs_ix], obs_mags[targ_spec_obs_ix], n)
                
                ix_co = ix_co_og.reshape(nb0*nb1, nd)
                ix_ob = ix_obs_og.reshape(nb0*nb1, nd)
            
                # Indices of bin combos that are greater than minimum observations
                ix_0 = np.nonzero(np.sum(ix_co, axis=1) >= min_observations)[0]
                ix_1 = np.nonzero(np.sum(ix_ob, axis=1) >= min_observations)[0]
                
                # only keep ix that are in both: 
                ix_keep = np.array([i for i in range(nb0*nb1) if np.logical_and(i in ix_0, i in ix_1)])
                
                for bi in range(bins_ang):
                    for bj in range(bins_mag):
                        
                        # Counts for specific bins: 
                        tmp = ix_co_og[bi, bj, :]
                        tmp1 = ix_co_og_targ[bi, bj, :]
                        tmp2 = ix_obs_og[bi, bj, :]
                        
                        if in_ == 0:
                            Distb[bi, bj, 0] += np.sum(tmp1)
                            Distb[bi, bj, 1] += np.sum(tmp2)

                        smp = []
                        smp2 = []
                        
                        for i, (t0, t1) in enumerate(zip(tmp, tmp2)):
                            
                            # Expand out the bins
                            smp.append([i]*int(t0))
                            smp2.append([i]*int(t1))
                            
                        #### COMPUTE DIFFERENCES IN MEANS & KS #####
                        ## Only compute mean for cases with at least min_observations ##
                        if np.logical_and(len(np.hstack((smp))) > min_observations, len(np.hstack((smp2))) > min_observations):
                            
                            if in_ == 0:
                                KL_all['bins_used'][bi, bj, 0] += 1 

                            C = np.hstack((smp))
                            O = np.hstack((smp2))
                            
                            ## Also compute MEAN differences:
                            ix_c = np.random.permutation(len(C))
                            ix_o = np.random.permutation(len(O))
                            
                            # Two subsets
                            ix_ci = ix_c[:int(np.floor(len(ix_c)/2.))]
                            ix_ci2 = ix_c[int(np.floor(len(ix_c)/2.)):]
                            
                            ix_oi = ix_o[:int(np.floor(len(ix_o)/2.))]
                            ix_oi2 = ix_o[int(np.floor(len(ix_o)/2.)):]
                            
                            ########################
                            ### Within task Mean ###
                            ########################
                            
                            dC = np.abs(np.mean(C[ix_ci]) - np.mean(C[ix_ci2]))
                            dO = np.abs(np.mean(O[ix_oi]) - np.mean(O[ix_oi2]))
                            
                            dx11 = np.abs(np.mean(C[ix_ci]) - np.mean(O[ix_oi]))
                            dx12 = np.abs(np.mean(C[ix_ci]) - np.mean(O[ix_oi2]))
                            dx21 = np.abs(np.mean(C[ix_ci2]) - np.mean(O[ix_oi]))
                            dx22 = np.abs(np.mean(C[ix_ci2]) - np.mean(O[ix_oi2]))
                                                
                            # Randomly chosen -- x task of the same subset
                            DLambda[bi, bj, in_, :] = np.array([dx11, dx22]) #np.mean(np.array([dx11, dx12, dx21, dx22]))
                           
                            # Within task (CO)
                            DLambda_Dist[bi, bj, in_, 0] = dC

                            # Within task (Obs)
                            DLambda_Dist[bi, bj, in_, 1] = dO

                            Lambda[bi, bj, in_, 0] = np.mean(C)
                            Lambda[bi, bj, in_, 1] = np.mean(O)

                            t, p = scipy.stats.ttest_ind(C, O)
                            Lambda[bi, bj, in_, 2] = p
                        
                        else:
                            DLambda[bi, bj, in_] = np.nan
                            DLambda_Dist[bi, bj, in_, :] = np.nan
                            Lambda[bi, bj, in_, :] = np.nan
                # Plot the distrubiton
            A = np.linspace(0., 2*np.pi, 9) - np.pi/8.
            R = np.array([0]+mag_bins[animal, select_day]+[mag_bins[animal, select_day][2]+1])
            Distb[:, :, 0] = Distb[:, :, 0] / float(np.sum(Distb[:, :, 0]))
            Distb[:, :, 1] = Distb[:, :, 1] / float(np.sum(Distb[:, :, 1]))

            im_0 = polar_plot(R, A, Distb[:, :, 0].T, ax = axtarg[0], vmin=0., vmax=0.2)
            im_1 = polar_plot(R, A, Distb[:, :, 1].T, ax = axtarg[1], vmin=0., vmax=0.2)
            plt.colorbar(im_1, ax=axtarg[1], fraction=0.046, pad=0.04, ticks=np.arange(0., .25, .05))
            axtarg[0].axis('square')
            axtarg[1].axis('square')

            # Sorted by bin                
            KL_all[str(select_day), targ, 'Lambda'] = Lambda
            KL_all[str(select_day), targ, 'DLambda'] = DLambda
            KL_all[str(select_day), targ, 'DLambda_Dist'] = DLambda_Dist
            
            # Not sorted by bin
            KL_all[str(select_day), targ, 'DLambda_overall'] = DLambda_overall
            KL_all[str(select_day), targ, 'DLambda_overall_dist'] = DLambda_overall_dist

            ##### COVARIANCE #######
            # For this day, go through all tasks: 

            for bi in range(bins_ang):
                ix_bi_co = np.nonzero(co_angs == bi)[0]
                ix_bi_obs = np.nonzero(obs_angs[targ_spec_obs_ix] == bi)[0]
                 
                for bj in range(bins_mag):
                    ix_bj_co = np.nonzero(co_mags[ix_bi_co] == bj)[0]
                    ix_bj_obs = np.nonzero(obs_mags[targ_spec_obs_ix[ix_bi_obs]] == bj)[0]

                    if np.logical_and(len(ix_bj_co) > min_observations, len(ix_bj_obs) > min_observations):
                        KL_all['bins_used'][bi, bj, 1] += 1 

                        co_cnts = co_nbins[ix_bi_co[ix_bj_co]]
                        obs_cnts = obs_nbins[targ_spec_obs_ix[ix_bi_obs[ix_bj_obs]]]

                        # Subselect 1/2 and 1/2
                        ix_co_i = np.random.permutation(co_cnts.shape[0])
                        ix_co_1 = ix_co_i[:(len(ix_co_i)/2)]
                        ix_co_2 = ix_co_i[(len(ix_co_i)/2):]

                        ix_obs_i = np.random.permutation(obs_cnts.shape[0])
                        ix_obs_1 = ix_obs_i[:(len(ix_obs_i)/2)]
                        ix_obs_2 = ix_obs_i[(len(ix_obs_i)/2):]

                        # Estimate covariance
                        co_cov1 = np.cov(co_cnts[np.ix_(ix_co_1, neurons)].T)
                        co_cov2 = np.cov(co_cnts[np.ix_(ix_co_2, neurons)].T)
                        
                        ob_cov1 = np.cov(obs_cnts[np.ix_(ix_obs_1, neurons)].T)
                        ob_cov2 = np.cov(obs_cnts[np.ix_(ix_obs_2, neurons)].T)                    
                        
                        # Overall within vs. across
                        a1, _ = subspace_overlap(co_cov1, co_cov2)
                        a2, _ = subspace_overlap(co_cov2, co_cov1)

                        b1, _ = subspace_overlap(ob_cov1, ob_cov2)
                        b2, _ = subspace_overlap(ob_cov2, ob_cov1)

                        x1, _ = subspace_overlap(co_cov1, ob_cov1)
                        x2, _ = subspace_overlap(ob_cov1, co_cov1)

                        y1, _ = subspace_overlap(co_cov2, ob_cov2)
                        y2, _ = subspace_overlap(ob_cov2, co_cov2)

                        DCov[bi, bj, 0] = np.mean([x1, x2])
                        DCov[bi, bj, 1] = np.mean([y1, y2])

                        DCov_dist[bi, bj, 0] = np.mean([a1, a2])
                        DCov_dist[bi, bj, 1] = np.mean([b1, b2])
            
            KL_all[str(select_day), targ, 'DCov'] = DCov
            KL_all[str(select_day), targ, 'DCov_dist'] = DCov_dist

    f, ax = plt.subplots(ncols = 2)
    f.set_figwidth(7)
    f.set_figheight(3)
    A = np.linspace(0., 2*np.pi, 9) - np.pi/8.
    R = np.array([0]+mag_bins[animal, select_day]+[mag_bins[animal, select_day][2]+1])

    K = KL_all['bins_used']
    K1 = np.hstack((np.vstack(( K[:, :, 0].T, np.zeros((1, 8)))), np.zeros((5, 1))))
    K2 = np.hstack((np.vstack(( K[:, :, 1].T, np.zeros((1, 8)))), np.zeros((5, 1))))

    polar_plot(R, A, K1, ax = ax[0], cmap='Blues', vmax=1., vmin=0.)
    im1 = polar_plot(R, A, K2, ax = ax[1], cmap='Blues', vmax=1., vmin=0.)
    plt.tight_layout()
    plt.colorbar(im1, ax = ax[1], fraction=0.046, pad=0.04, ticks=np.arange(0., 1., .25))

    return KL_all

def plot_KL_all(KL_all, animal = 'grom', save = False):
    X = []
    WI = []

    CovX = []
    CovWI = []

    n_targs = KL_all['n_targs']

    for select_day in range(days[animal]):
        x = []
        wi = []

        for targ in n_targs:
            diff = KL_all[str(select_day), targ, 'DLambda']
            same = KL_all[str(select_day), targ, 'DLambda_Dist']
            bins_ang, bins_mag, nneurons, _ = diff.shape
            #f, ax = plt.subplots()

            cov_diff = KL_all[str(select_day), targ, 'DCov']
            cov_same = KL_all[str(select_day), targ, 'DCov_dist']

            for i in range(8):
                for j in range(4):
                    for n in range(nneurons):
                        x.append(diff[i, j, n, :])
                        wi.append(same[i, j, n, :])
                    CovX.append(cov_diff[i, j, :])
                    CovWI.append(cov_same[i, j, :])
                    

    x = np.hstack((x))
    wi = np.hstack((wi))

    print 'mean nans cross: ', np.sum(~np.isnan(x))
    print 'mean nans within: ', np.sum(~np.isnan(wi))

    x = x[~np.isnan(x)]
    wi = wi[~np.isnan(wi)]

    CovX = np.hstack((CovX))
    CovWI = np.hstack((CovWI))
    print 'cov nans cross: ', np.sum(~np.isnan(CovX))
    print 'cov nans within: ', np.sum(~np.isnan(CovWI))

    CovX = CovX[~np.isnan(CovX)]
    CovWI = CovWI[~np.isnan(CovWI)]

    # Summary plot
    f, ax = plt.subplots()
    f.set_figheight(6)
    f.set_figwidth(6)

    bins = np.linspace(0, 1.5, 40.)
    bins_plot = bins[1:] + 0.5*(bins[1] - bins[0])
    xc, _  = np.histogram(x, bins)
    wic, _  = np.histogram(wi, bins)
    k, p = scipy.stats.kruskal(x, wi)

    ia = ax.plot(bins_plot, xc/float(np.sum(xc)), 'k-', label='across task')
    ib = ax.plot(bins_plot, wic/float(np.sum(wic)), '--', color='gray', label='within task')
    ax.vlines(np.mean(x), 0, .3, color='k')
    ax.vlines(np.mean(wi), 0, .3, color='gray')

    ax.set_xlabel('Mean FR Differences')
    ax.set_ylabel('Frequency')
    ax.set_title(animal+', Only Imp: '+str(KL_all['important_neurons'])+', p = ' + str(np.round(1000*p)/1000.))
    plt.legend()

    f2, ax = plt.subplots()
    f2.set_figheight(6)
    f2.set_figwidth(6)

    bins = np.linspace(0, 1., 50.)
    bins_plot = bins[1:] + 0.5*(bins[1] - bins[0])
    xc, _  = np.histogram(CovX, bins)
    wic, _  = np.histogram(CovWI, bins)
    k, p = scipy.stats.kruskal(CovX, CovWI)

    ia = ax.plot(bins_plot, xc/float(np.sum(xc)), 'k-', label='ov across task')
    ib = ax.plot(bins_plot, wic/float(np.sum(wic)), '--', color='gray', label='ov within task')
    ax.vlines(np.mean(CovX), 0, .3, color='k')
    ax.vlines(np.mean(CovWI), 0, .3, color='gray')

    ax.set_xlabel('Subspace Ov')
    ax.set_ylabel('Frequency')
    ax.set_title(animal+', Only Imp: '+str(KL_all['important_neurons'])+', p = ' + str(np.round(1000*p)/1000.))
    plt.legend()

    if save:
        f.savefig('Mn_diff_'+animal+'_important_neurons_only_'+str(KL_all['important_neurons'])+'_sep_obs.png')
    #f.legend([ia, ib], ['Across Task', 'Within Task'])
       
    # ix0 = np.unique(np.hstack((np.nonzero(np.isnan(KLmn))[0], np.nonzero(np.isnan(KLmn_dist))[0])))
    # ix0 = np.array([i for i in range(len(KLmn)) if i not in ix0])
    # print ix0.shape
    # #ax.plot(.2*np.random.rand(np.prod(KLmn.shape))+select_day, 10*KLmn.reshape(-1), 'k.')#, positions=[select_day])
    # ax.bar(select_day, 10*np.mean(KLmn[ix0]), width =.4, color='k')
    # ax.errorbar(select_day+.2, 10*np.mean(KLmn[ix0]), yerr = np.std(KLmn[ix0])/np.sqrt(len(ix0)) , color='k', marker='.')
    # xtask.append(KLmn[ix0])
    
    # #ax.plot(.2*np.random.rand(np.prod(KLmn_dist.shape))+select_day+.4, 10*KLmn_dist.reshape(-1), 'b.')#, positions=[select_day+.4]
    # ax.bar(select_day+.4, 10*np.mean(KLmn_dist[ix0]), width =.4, color='b')
    # ax.errorbar(select_day+.6, 10*np.mean(KLmn_dist[ix0]), yerr = np.std(KLmn_dist[ix0])/np.sqrt(len(ix0)) , color='b', marker='.')
    # intask.append(KLmn_dist[ix0])
    
    # ### DIfferences: 
    # all_bars['xtask_vs_win', select_day] = np.abs(KLmn[ix0] - KLmn_dist[ix0])
    # all_bars['xtask', select_day] = KLmn[ix0]
    # all_bars['win', select_day] = KLmn_dist[ix0]
    
    # KW, KWp = scipy.stats.ttest_rel(KLmn[ix0], KLmn_dist[ix0])
    # print select_day, KWp, 'n = ', len(ix0)
    # xlabpos.append(select_day)
    # xlabpos.append(select_day+.4)

def subspace_overlap(UUT_A, UUT_B, main_thresh = 0.9, use_main = True):
    ''' overlab of A projected to B space'''
    
    assert UUT_A.shape[0] == UUT_B.shape[0]
    
    if use_main:

        # SVD on subspace matrix B
        v, s, vt = np.linalg.svd(UUT_B)
        
        # This is not s**2 because you've given SVD a covaranice matrix. If you gave it a data matrix: 
        # e.g. A, in R_D x N
        # you'd get back U S V.T = A
        # Then cov(A) = A*A.T/(n-1) = U S V.T (U S V.T).T / (n-1) = V S U.T U S V.T / (n-1) = V (S^2/(n-1)) V.T
        # so SVD on cov(A) = V S' V.T where S' = (S^2/(n-1))
        # Thus, the 'singular values' of a data matrix are the squareroots of the eigenvalues. 
        
        s_cum = np.cumsum(s)/float(np.sum(s))
        red_s = np.zeros((v.shape[1], ))

        # Get main shared space of B
        ix = np.nonzero(s_cum > main_thresh)[0]
        nf = ix[0] + 1
        #print 'main shared space num factors in subspace B: ', nf
        red_s[:nf] = 1.

        # Compute projection matrix
        Pb = np.mat(v)*np.mat(np.diag(red_s))*np.mat(vt)    

        # Compute SVD on cov matrix A 
        vv, ss, vvt = np.linalg.svd(UUT_A)
        ss_cum = np.cumsum(ss)/float(np.sum(ss))

        # Get main shared space on cov matrix A
        ix = np.nonzero(ss_cum > main_thresh)[0]
        nnf = ix[0] + 1
        #print 'main shared space num fact in cov A: ', nnf

        red_s2 = np.zeros((vv.shape[1], ))
        red_s2[:nnf] = ss[:nnf]

        A_shar = np.mat(vv) * np.mat(np.diag(red_s2))*np.mat(vvt)

        # Ratio of variance
        proj_A_B = float(np.trace(Pb*A_shar*Pb.T)) / float(np.trace(A_shar))
    
    else:
        # Projection matrix
        v, s, vt = np.linalg.svd(UUT_B)
        Pb = np.mat(v)*np.mat(vt)
        nf = np.linalg.matrix_rank(UUT_B)
        nnf = np.linalg.matrix_rank(UUT_A)
        # Variance: 
        proj_A_B = float(np.trace(Pb*UUT_A*Pb.T))/ float(np.trace(UUT_A))
    
    return proj_A_B, np.min([nf, nnf])
 
def get_hist(co_nbins, co_angs, co_mags, n):
    hist = np.zeros(( 8, 4, 40))

    for ic, cnt in enumerate(co_nbins[:, n]):
        hist[co_angs[ic], co_mags[ic], int(cnt)] += 1
    return hist

def polar_plot(r, phi, data, ax=None, vmin=None, vmax=None, cmap=None):
    """
    Plots a 2D array in polar coordinates.

    :param r: array of length n containing r coordinates
    :param phi: array of length m containing phi coordinates
    :param data: array of shape (n, m) containing the data to be plotted
    """
    
    # Generate the mesh
    phi_grid, r_grid = np.meshgrid(phi, r)
    x, y = r_grid*np.cos(phi_grid), r_grid*np.sin(phi_grid)
    if ax is None:
        im = plt.pcolormesh(x, y, data, vmin=vmin, vmax=vmax, cmap=cmap)
    else:
        im = ax.pcolormesh(x, y, data, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(np.arange(-4, 5, 2))
    ax.set_xticklabels(np.arange(-4, 5, 2), fontsize=24)
    ax.set_yticks(np.arange(-4, 5, 2))
    ax.set_yticklabels(np.arange(-4, 5, 2), fontsize=24)
    return im

def bar_plot_means(KL_all, animal = 'grom'):
    ''' 
    Visualization for individual Obs targets -- which 
    bins are used and which have high perc of neurons sig. 

    Also plots bar plot for diff
    '''
    
    #f, ax = plt.subplots() # Main bar plot
    targ = dict()
    targ[0] = [2, 0]
    targ[1] = [2, 1]
    targ[2] = [2, 2]
    targ[3] = [1, 2]
    targ[4] = [0, 2]
    targ[5] = [0, 1]
    targ[6] = [0, 0]
    targ[7] = [1, 0]

    n_targs = KL_all['n_targs']
    dayz = KL_all['days']

    # Mean FR plots: 
    f2, ax2 = plt.subplots(ncols = 3, nrows = 3) #

    # Mean across task subspace OV plots: 
    f3, ax3 = plt.subplots(ncols = 3, nrows = 3) #

    A = np.linspace(0., 2*np.pi, 9) - np.pi/8.

    for select_day in dayz:
        R = np.array([0]+mag_bins[animal, select_day]+[mag_bins[animal, select_day][2]+1])

        for targi in n_targs:
            perc_sig = np.zeros((9, 5, 2))
            x_overlap = np.zeros((9, 5))
            diffs = KL_all[str(select_day), targi, 'Lambda']
            nneurons = diffs.shape[2]

            for i in range(8):
                for j in range(4):
                    x_overlap[i, j] = -1*np.mean(KL_all[str(select_day), targi, 'DCov'][i, j, :])
                    for n in range(nneurons):
                        abs_d = np.abs(diffs[i, j, n, 0] - diffs[i, j, n, 1])
                        if ~np.isnan(abs_d):
                            if diffs[i, j, n, 2] < .05:
                                color = 'r'
                                perc_sig[i, j, 0] += 1
                            else:
                                color = 'gray'
                                perc_sig[i, j, 1] += 1
                            #ax.plot(np.random.randn(), abs_d, '.', color=color)

            # Plot radial dist of sig bins: 
            axi = ax2[tuple(targ[targi])]
            axi3 = ax3[tuple(targ[targi])]

            perc_sig_targ = perc_sig[:, :, 0] / (perc_sig[:, :, 0] + perc_sig[:, :, 1])
            total_per_targ = perc_sig[:, :, 0] + perc_sig[:, :, 1]

            im = polar_plot(R, A, perc_sig_targ.T, ax=axi, vmin=0., vmax=0.5, cmap='Blues')
            im2 = polar_plot(R, A, x_overlap.T, ax=axi3, vmin=-1.0, vmax=-0.5, cmap='Blues')
            
    plt.colorbar(im, ax = axi,fraction=0.046, pad=0.04, ticks=np.arange(0., 1.0, .25))
    plt.colorbar(im2, ax = axi3,fraction=0.046, pad=0.04, ticks=np.arange(-1.0, -0.5, .25))

    f2.tight_layout()
    f3.tight_layout()

def plot_traj(select_day = 0, animal = 'grom'):
    fname = fname_pref[animal] + str(select_day)
    dat = pickle.load(open(fname+'.pkl'))

    f, ax = plt.subplots(ncols = 2)

    for task, task_name in enumerate(['co', 'obs']):
        keep_going = True
        block = 0
        while keep_going:
            key = tuple(('day'+str(select_day), 'tsk'+str(task), 'n'+str(block)))
            
            if key in dat.keys():
                cursor_state = dat[(key, 'cursor_state')]
                targ = dat[(key, 'targ_ix')]
                ix_start = 0
                n_trls = len(cursor_state)
                
                for trl in range(n_trls): 
                    targ_num =  targ[ix_start]
                    pos = cursor_state[trl]
                    ax[task].plot(pos[:, 0, 0], pos[:, 1, 0], '.-', color=cmap_list[int(targ_num)])
                    ix_start += len(pos)
                block += 1
            else:
                keep_going = False
        ax[task].axis('square')

def align_beh_to_bin(bin_angle, bin_mag, obs_target, select_day = 0,
    animal = 'grom'):
    ''' 
    Everytime a nerual push in this bin angle and magnitude shows up
    for a specific target, add the position, velocity, and neural push trajectory 
    to a list
    '''

    fname = fname_pref[animal] + str(select_day)
    dat = pickle.load(open(fname+'.pkl'))

    state = dict(co=[], obs=[])
    push = dict(co=[], obs=[])

    for task, task_name in enumerate(['co', 'obs']):
        has_key = True
        block = 0

        while has_key:
            key = tuple(('day'+str(select_day), 'tsk'+str(task), 'n'+str(block)))
            if key in dat.keys():
                angs = dat[(key, 'ang')]
                mags = dat[(key, 'mag')]
                targ = dat[(key, 'targ_ix')]
                cursor_state = dat[(key, 'cursor_state')]
                neural_push = dat[(key, 'neural_push')]

                n_trls = len(angs)
                ix_start = 0
                
                for trl in range(n_trls):
                    if np.logical_or(task_name == 'co', 
                        np.logical_and(task_name == 'obs', targ[ix_start] == obs_target)):
                        
                        ang = angs[trl]
                        mag = mags[trl]
                    
                        ix_a = np.nonzero(ang == bin_angle)[0]
                        ix_m = np.nonzero(mag == bin_mag)[0]

                        ix_common = np.array(list(set(ix_a).intersection(ix_m)))

                        if len(ix_common) > 0:
                            for ixc in ix_common:
                                if np.logical_and(ixc >= 4, ixc + 4 < len(cursor_state[trl])):
                                    state[task_name].append(cursor_state[trl][ixc - 4:ixc+4, :, 0])
                                    push[task_name].append(np.squeeze(np.array(neural_push[trl][ixc - 4:ixc+4, :])))
                                else:
                                    print 'not enough history: ', task_name, trl
                    ix_start += len(angs[trl])
                block += 1
            else:
                has_key = False

    ### Plot it in behav relevant space ###
    f, ax = plt.subplots()
    n_co =  len(state['co'])
    n_obs = len(state['obs'])

    colors = ['black', 'green']
    x_mark = '-'
    y_mark = '--'

    for i_t, (N, task_name) in enumerate(zip([n_co, n_obs], ['co', 'obs'])):
        pos = state[task_name]
        for n in range(N):
            pos_trl = pos[n]
            try:
                ax.plot(pos_trl[:, 0], pos_trl[:, 1], '-', color=colors[i_t])
                ax.plot(pos_trl[4, 0], pos_trl[4, 1], '.', color=colors[i_t], markersize=10)
            except:
                print 'skipping trial, insuff history: ', task_name, 'trl: ', n
    ax.set_title('Positions of Command: Angle:'+str(bin_angle)+', Mag: '+str(bin_mag)+', Obs Targ: '+str(obs_target))

    ### Now plot PSTH of velocity and push
    f, ax = plt.subplots(ncols = 2)
    f.set_figheight(3)
    f.set_figwidth(7)
    
    T_axis = np.arange(-.400, .400, .100)
    for i_t, (N, task_name) in enumerate(zip([n_co, n_obs], ['co', 'obs'])):
        pos = np.dstack(( state[task_name] ))
        pu = np.dstack(( push[task_name] ))

        axii = plot_mean_and_sem(T_axis, pos[:, 2, :], ax[0], array_axis = 1, label = task_name+' vel x', color=colors[i_t], 
            marker_type = x_mark)
        axii = plot_mean_and_sem(T_axis, pos[:, 3, :], axii, array_axis = 1, label = task_name+' vel y', color=colors[i_t], 
            marker_type = y_mark)
        
        axii = plot_mean_and_sem(T_axis, pu[:, 3, :], ax[1], array_axis = 1, label = task_name+' push x', color=colors[i_t], 
            marker_type = x_mark)
        axii = plot_mean_and_sem(T_axis, pu[:, 5, :], axii, array_axis = 1, label = task_name+' push y', color=colors[i_t], 
            marker_type = y_mark)

    ax[0].set_title('Curs Vel')
    ax[1].set_title('Push')
    ax[1].legend(fontsize=8)

def plot_mean_and_sem(x, array, ax, color='b', array_axis=1, label='0', marker_type ='-'):
    
    mean = np.nanmean(array, axis=array_axis)
    sem_plus = mean + scipy.stats.sem(array, axis=array_axis, nan_policy='omit')
    sem_minus = mean - scipy.stats.sem(array, axis=array_axis, nan_policy='omit')
    
    ax.fill_between(x, sem_plus, sem_minus, color=color, alpha=0.5)
    x = ax.plot(x, mean,marker_type,color=color,label=label)

    return ax

def correlate_change_dr_change_beh_bef_and_aft(select_day = 0, pre_go = 1., 
    animal='grom', n_steps = 4):

    ''' Load data from trials from day 1 '''
    # dat = pickle.load(open(fname_pref[animal+'_prego']+str(select_day)+'.pkl'))
    
    inputs = [[4377], [4378, 4382]]

    # Aggregate spike counts: 
    binned_spikes = []
    angs = []
    mags = []
    task_list = []
    targ = []
    push = []

    co_obs_dict = pickle.load(open(pref+'co_obs_file_dict.pkl'))
    decoder = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/grom20160302_02_RMLC03021529.pkl'))

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
            push_i = [decoder_all[i] for i in range(len(decoder_all)) if i not in exclude]
            push.extend(push_i)
            angi, magi = tuning_diffs.compute_bins(push_i, select_day, animal)
            angs.extend(angi)
            mags.extend(magi)
            task_list.extend([task] for i in range(len(bin_spk)) if i not in exclude)

            sub_targ_ix = []
            for trl in np.unique(trial_ix_all):
                if trl not in exclude:
                    ix = np.nonzero(trial_ix_all==trl)[0]
                    if pre_go is None:
                        sub_targ_ix.append(targ_ix[ix[0]])
                    else:
                        sub_targ_ix.append(targ_ix[ ix[0] + int(pre_go*10) + 2])
            targ.extend(np.hstack((sub_targ_ix)))

    assert len(binned_spikes) == len(push) == len(angs) == len(mags) == len(task_list) == len(targ)
    n_dim_latent = 15
    pre_go_bins = 10

    # Get all binned data to fit an LDS: 
    R2_smooth, R2_pred, LL, dynamics_norm, innov_norm, model, pred_data, smooth_data, filt_state, pred_state, filt_sigma, kg, R2_filt, smooth_state = fit_LDS.fit_LDS(
    binned_spikes, binned_spikes, n_dim_latent, return_model=True, seed_w_FA=True, nEMiters=30, **dict(seed_pred_x0_with_smoothed_x0= True, get_ratio=True, 
        pre_go_bins=pre_go_bins))

    ### Now plot everything to everything: 
    N_trls = len(angs)
    start_bin = int(pre_go*10)
    points = {}
    states = {}
    obs = {}

    task_name = ['co', 'obs']
    co_targs = [np.arange(8)]
    obs_targs = [[i] for i in np.arange(8)]

    trls_by_targ = {}
    for _, (tsk, targ_set) in enumerate(zip(task_name, [co_targs, obs_targs])):
        for it, tg in enumerate(targ_set):
            trls_by_targ[tsk, it] = []
    for trl in np.arange(N_trls):
        tg = targ[trl]
        tsk = task_name[task_list[trl][0]]
        if tsk == 'co':
            trls_by_targ[tsk, 0].append(trl)
        else:
            trls_by_targ[tsk, tg].append(trl)

    targ_comparisons = dict(co=co_targs, obs = obs_targs)

    for i in range(8):
        for j in range(4):
            for task in task_name:
                for itarg, targs in enumerate(targ_comparisons[task]):
                    points[i, j, task, itarg, 'wi'] = []
                    states[i, j, task, itarg, 'wi'] = []
                    obs[i, j, task, itarg, 'wi'] = []

                    if task == 'obs':
                        # Only need to list obstacle target
                        points[i, j, itarg, 'x'] = []
                        states[i, j, itarg, 'x'] = []
                        obs[i, j, itarg, 'x'] = []

            # k = tuple([i, j, 'wi'])
            # k2 = tuple([i, j, 'x'])

            # beh_storage2[k, 'co'] = []
            # beh_storage2[k, 'obs'] = []
            # beh_storage2[k2, 'co'] = []
            # beh_storage2[k2, 'obs'] = []

    # Subselect trials:
    #sub_trls = np.hstack(( np.arange(63)[::2], np.arange(63, 285)[::4] ))
    sub_trls = np.arange(N_trls)
    
    # For each trial
    for trl in sub_trls:
        trl_task = task_list[trl][0]
        trl_task_name = task_name[trl_task]
        trl_non_task_name = task_name[np.mod(trl_task + 1, 2)]
        self_target = int(targ[trl])

        self_targs = [t for t in targ_comparisons[trl_task_name] if self_target in t]
        other_targs = targ_comparisons[trl_non_task_name]

        # Within and across task trials. Still don't know what target I am and what task I am 
        ixx = np.nonzero( np.hstack((task_list)) != trl_task)[0]
        ixx2 = np.nonzero( np.hstack((task_list)) == trl_task)[0]

        for bini in range(8):
            #print 'starting: bini: ', bini
            ii = np.nonzero(angs[trl][start_bin:] == bini)[0]
            
            for binj in range(4):
                ij = np.nonzero(mags[trl][start_bin:] == binj)[0]

                # Don't compare within trial
                combo = np.array(list(set(ii).intersection(ij)))
                combo = combo[combo >= n_steps]
                combo = combo[combo < len(angs[trl]) - start_bin - n_steps]

                if len(combo) > 0:
                    #print trl, bini, binj
                    combo = np.array([combo[np.random.permutation(len(combo))[0]]])

                    # Select the correct target for comparison:
                    # Self targs and other targs 
                    # for comp_i, (index, targ_sets, comp_name) in enumerate(zip([ixx, ixx2], [other_targs, self_targs], ['x', 'wi'])):
                        
                    #     # For each set of within trial vs. across trial indices, select the correct target? 
                    #     for i_targ, target_t in enumerate(targ_sets):
                    #         ixx_t = np.array([ixx_ for ixx_ in index if targ[ixx_] in target_t])

                    #         # All within vs. across trials that meet the self vs. other target expectations
                    #         tmp = np.arange(N_trls)[ixx_t]
                    #         tmp = np.array([t for t in tmp if t != trl])
                    for comp_i, (comp_name, targ_sets) in enumerate(zip(['x', 'wi'], [other_targs, self_targs])):
                        for targ_i, target_t in enumerate(targ_sets):
                            if comp_name == 'x':
                                tmp = np.array(trls_by_targ[trl_non_task_name, targ_i])
                            
                            elif comp_name == 'wi':
                                tmp = np.array(trls_by_targ[trl_task_name, targ_i])

                            # Subselect 8 trial? 
                            if len(tmp) > 4:
                                tmp = tmp[np.random.permutation(len(tmp))[:4]]

                            for trl2 in tmp:
                                ki = np.nonzero(angs[trl2][start_bin:] == bini)[0]
                                kj = np.nonzero(mags[trl2][start_bin:] == binj)[0]
                                
                                combo2 = np.array(list(set(ki).intersection(kj)))
                                combo2 = combo2[combo2 >= n_steps]
                                combo2 = combo2[combo2 < len(angs[trl2]) - start_bin - n_steps]

                                if len(combo2) > 0:
                                    #print len(combo2), len(combo1), len(combo2)*len(combo1)
                                    combo2 = np.array([combo2[np.random.permutation(len(combo2))[0]]])

                                    # For each of the original items: 
                                    for ic1 in combo:

                                        # Get the dynamics norm / innovation norm (OG)
                                        dyn_norm = dynamics_norm[trl][ic1]
                                        inn_norm = innov_norm[trl][ic1]

                                        # Previous filtered state
                                        x_t_1_filt = filt_state[trl][ic1 - 1]
                                        x_t_filt = filt_state[trl][ic1]

                                        # Next behavior state: 
                                        true_beh_t_pls_1 = np.squeeze(np.array(push[trl][start_bin + ic1 + 1]))

                                        kg_ = kg[trl][ic1]

                                        pred_ = np.dot(kg_, np.dot(model.C, np.dot(model.A, x_t_1_filt)))
                                        pred_norm = np.dot(kg_, binned_spikes[trl][ic1 + start_bin])
                                        inn_norm_confirm = np.linalg.norm(pred_ - pred_norm)
                                        dyn_norm_confirm = np.linalg.norm(x_t_1_filt - np.dot(model.A, x_t_1_filt))

                                        # Confirm the dyn norm and inn norm are good: 
                                        assert np.allclose(dyn_norm, dyn_norm_confirm)
                                        assert np.allclose(inn_norm, inn_norm_confirm)

                                        # Original dynamics ratio: 
                                        dr_og = dyn_norm / (dyn_norm + inn_norm)

                                        # Original behavior: 
                                        beh_og = push[trl][start_bin + ic1 - n_steps : start_bin + ic1 + n_steps + 1, [3, 5]]

                                        # Next Behavior predicted by the orignal state + dynamics
                                        F, KG_dec = decoder.filt.get_sskf()
                                        
                                        beh_pred_from_state_t = np.dot(KG_dec, np.dot(model.C, np.dot(model.A, x_t_filt)))
                                        beh_pred_from_state_t = np.linalg.norm(beh_pred_from_state_t[[3, 5]] - true_beh_t_pls_1[[3, 5]])

                                        for ic2 in combo2:

                                            # Now do the swap
                                            pred_norm_new = np.dot(kg_, binned_spikes[trl2][ic2 + start_bin])
                                            inn_norm_new = np.linalg.norm(pred_ - pred_norm_new)
                                            dr_new = dyn_norm / (dyn_norm + inn_norm_new)

                                            # Predicted new state: 
                                            pred_new_state = np.dot((np.eye(15) - np.dot(kg_, model.C)), x_t_1_filt) + np.dot(kg_, binned_spikes[trl2][ic2 + start_bin])

                                            # Now take the behavioral difference norm: 
                                            beh_new = push[trl2][start_bin + ic2 - n_steps : start_bin + ic2 + n_steps + 1, [3, 5]]
                                            diff_beh = np.linalg.norm(beh_og - beh_new)
                                            diff_beh_hist = np.linalg.norm(beh_og[:n_steps, :] - beh_new[:n_steps, :])

                                            x_t_filt_2 = filt_state[trl2][ic2]
                                            beh_pred_from_state_t2 = np.squeeze(np.array(np.dot(KG_dec, np.dot(model.C, np.dot(model.A, x_t_filt_2)))))
                                            beh_pred_from_state_t2 = np.linalg.norm(beh_pred_from_state_t2[[3, 5]] - true_beh_t_pls_1[[3, 5]])

                                            diff_beh_pred = beh_pred_from_state_t - beh_pred_from_state_t2 # Prediction with real vs. prediction with other
                                            diff_state = np.linalg.norm(smooth_state[trl][ic1, :] - smooth_state[trl2][ic2, :])
                                            diff_obs = np.linalg.norm(binned_spikes[trl][ic1 + start_bin, :] - binned_spikes[trl2][ic2 + start_bin, :])

                                            # Now, where to store these observations: 
                                            # If you are across task: 
                                            if comp_name == 'x':
                                                # If we are the obs task, then use own target 
                                                if trl_task_name == 'obs':
                                                    key = [bini, binj, self_target, 'x']

                                                # If other is the obs task, then use the other target
                                                elif trl_task_name == 'co':
                                                    key = [bini, binj, target_t[0], 'x']

                                            elif comp_name == 'wi':
                                                # Keep all as part of the keys; 
                                                key = [bini, binj, trl_task_name, targ_i, 'wi']

                                            points[tuple(key)].append([dr_og, dr_new, diff_beh, diff_beh_hist, diff_beh_pred])
                                            states[tuple(key)].append(diff_state)
                                            obs[tuple(key)].append(diff_obs)

                                            # if binj == 3:
                                            #     if np.logical_or(bini == 2, bini == 3):
                                            #         beh_storage2[(bini, binj, 'x'), trl_task_name].append([beh_og, 
                                            #             beh_new, 
                                            #             filt_state[trl][ic1-n_steps:ic1+n_steps, :],
                                            #             filt_state[trl2][ic2-n_steps:ic2+n_steps, :], 
                                            #             binned_spikes[trl][ic1 + start_bin - n_steps:ic1 + start_bin + n_steps, :],
                                            #             binned_spikes[trl2][ic2 + start_bin - n_steps:ic2 + start_bin + n_steps, :],
                                            #             np.dot(model.A, x_t_1_filt), # Predicted state forward (dynamics step)
                                            #             pred_new_state, # Predicted state w/ new innovation
                                            #             ])

                                            # Swap the other way
                                            inn_norm_2 = innov_norm[trl2][ic2]
                                            dyn_norm_2 = dynamics_norm[trl2][ic2]
                                            dr_og2 = dyn_norm_2 / (dyn_norm_2 + inn_norm_2)
                                            x_t_1_filt_2 = filt_state[trl2][ic2 - 1]
                                            kg_2 = kg[trl2][ic2]

                                            pred_2 = np.dot(kg_2, np.dot(model.C, np.dot(model.A, x_t_1_filt_2)))
                                            pred_norm_2 = np.dot(kg_2, binned_spikes[trl][ic1 + start_bin])
                                            inn_norm_2 = np.linalg.norm(pred_2 - pred_norm_2)
                                            dr_new_2 = dyn_norm_2 / (dyn_norm_2 + inn_norm_2)
                                            #points.append([dr_og2, dr_new_2, diff_beh])
                                            # norm difference in the mean state: 

                                            points[tuple(key)].append([dr_og2, dr_new_2, diff_beh, diff_beh_hist, diff_beh_pred, true_beh_t_pls_1])
                                            states[tuple(key)].append(diff_state)
                                            obs[tuple(key)].append(diff_obs)
        print 'done w/ trial ', trl

    dic = dict(points = points, states = states, obs = obs)
    pickle.dump(dic, open('points_trial_subs.pkl', 'wb'))
    print 'done saving subs'
    return points

def plot_predictions(fname):
    data = pickle.load(open(fname))
    points = data['points']
    states = data['states']
    obs = data['obs']

    # plot state vs. behavior, trial avg: 
    for _, (index, xlab) in enumerate(zip([3, 4], ['diff_in_beh(t-1)', 'diff_in_pred(t+1)'])):
        f, ax = plt.subplots()

        targs = dict(co=[0], obs=range(8))
        no_info = 0
        info = 0

        X = []
        Y = []

        # Plot across and within 
        for i in range(8):
            for j in range(4):
                for task in ['co', 'obs']:
                    for targ in targs[task]:
                        key = [i, j, task, targ, 'wi']

                        try:
                            p = np.array([ip[4] for ip in points[tuple(key)]])  #np.vstack((points[tuple(key)]))
                            if len(p) > 0:
                                c = True
                            else:
                                c = False
                            info += 1
                            #print ' info: ', key, p.shape[0]

                        except:
                            c = False
                            no_info += 1

                        if c:
                            #diff_dr = np.mean(p[:, 1] - p[:, 0])
                            diff_state = np.mean(np.vstack(( states[tuple(key)] )))
                            diff_beh_hist = np.mean(p)

                            ax.plot(diff_beh_hist, diff_state, '.', color = 'grey')
                            X.append(diff_beh_hist)
                            Y.append(diff_state)
                # Across: 
                for targ in range(8):
                    key = [i, j, targ, 'x']
                    try:
                        p = np.array([ip[4] for ip in points[tuple(key)]])
                        #p = np.vstack((points[tuple(key)]))
                        if len(p) > 0:
                            c = True
                        else:
                            c = False
                        info += 1
                        #print 'x info: ', key, p.shape[0]
                    except:
                        c = False
                        no_info += 1
                    if c:
                        #diff_dr = np.mean(p[:, 1] - p[:, 0])
                        diff_state = np.mean(np.vstack(( states[tuple(key)] )))
                        diff_beh_hist = np.mean(p)
                        ax.plot(diff_beh_hist, diff_state, '.', color = 'red')
                        X.append(diff_beh_hist)
                        Y.append(diff_state)
        print 'scipy.stats: '
        Y = np.array(Y)[np.array(X) > -1.5]
        X = np.array(X)[np.array(X) > -1.5]

        slp, intc, rv, pv, err = scipy.stats.linregress(X, Y)
        x = np.linspace(np.percentile(X, 10), np.percentile(X, 90), 30); 
        y = slp*x + intc; 
        ax.plot(x, y, '--', color='k')
        ax.set_xlabel(xlab)
        ax.set_ylabel('diff in dynamics ratio at t')
        ax.set_title('R: '+str(1./1000.*np.round(rv*1000.)))


    # For individual bins -- predictions:  
    for i in range(8):
        for j in range(4):
            pass



    # for _, (index, xlab) in enumerate(zip([3, 4], ['diff_in_beh(t-1)', 'diff_in_pred(t+1)'])):
    #     f, ax = plt.subplots()

    #     targs = dict(co=[0], obs=range(8))
    #     no_info = 0
    #     info = 0

    #     X = []
    #     Y = []

    #     # Plot across and within 
    #     for i in range(8):
    #         for j in range(4):
    #             for task in ['co', 'obs']:
    #                 for targ in targs[task]:
    #                     key = [i, j, task, targ, 'wi']

    #                     try:
    #                         p = np.vstack((points[tuple(key)]))
    #                         c = True
    #                         info += 1
    #                         print ' info: ', key, p.shape[0]
    #                     except:
    #                         c = False
    #                         no_info += 1
    #                     if c:
    #                         diff_dr = np.mean(p[:, 1] - p[:, 0])
    #                         diff_beh_hist = np.mean(p[:, index])
    #                         ax.plot(diff_beh_hist, diff_dr, '.', color = 'grey')
    #                         X.append(diff_beh_hist)
    #                         Y.append(diff_dr)
    #             # Across: 
    #             for targ in range(8):
    #                 key = [i, j, targ, 'x']
    #                 try:
    #                     p = np.vstack((points[tuple(key)]))
    #                     c = True
    #                     info += 1
    #                     print 'x info: ', key, p.shape[0]
    #                 except:
    #                     c = False
    #                     no_info += 1
    #                 if c:
    #                     diff_dr = np.mean(p[:, 1] - p[:, 0])
    #                     diff_beh_hist = np.mean(p[:, index])
    #                     ax.plot(diff_beh_hist, diff_dr, '.', color = 'red')
    #                     X.append(diff_beh_hist)
    #                     Y.append(diff_dr)
    #     print 'scipy.stats: '
    #     slp, intc, rv, pv, err = scipy.stats.linregress(X, Y)
    #     x = np.linspace(np.percentile(X, 10), np.percentile(X, 90), 30); 
    #     y = slp*x + intc; 
    #     ax.plot(x, y, '--', color='k')
    #     ax.set_xlabel(xlab)
    #     ax.set_ylabel('change in dyn. ratio of t')
    #     ax.set_title('R: '+str(1./1000.*np.round(rv*1000.)))

def traj_plot(beh_storage, bin_ang = 3, bin_mag = 3, across_or_win = 'x'):
    ### Single trials and average for within and across ###
    ### First do a trial average plot of 15D state for within ###
    f, ax = plt.subplots(ncols = 5, nrows = 3)
    key = ((bin_ang, bin_mag, across_or_win), 'co')
    beh_key_co = beh_storage2[key]
    
    ### get individual trials, filtered ### 
    state_traj_co = np.dstack(( [b[2] for b in beh_key_co] )) # time x states x trials
    state_traj_co_2 = np.dstack(( [b[3] for b in beh_key_co] )) # time x states x trials

    ### Individual traj ###
    for i, task_arr in enumerate([state_traj_co, state_traj_co_2]):
        for j, numb in enumerate(task_arr):
            # Plot state trajectories for these guys: 
            for k in range(10):
                for state in range(15):
                    axi = ax[state/5, state%5]

                    # The filtered state: 
                    axi.plot(numb[:, state, k], 'b-')
                    axi.plot(4, numb[4, state, k], 'b.')

                    # Now plot the predicted state
                    prev_st = numb[3, state, k]
                    dyn_pred = np.dot(model.A, numb[4-1, :, k])
                    d_dyn_pred = dyn_pred[state] - prev_st
                    axi.arrow(3, prev_st, 0.5, d_dyn_pred,
                        width = .05, facecolor='k')

                    # Now plot how the innovations change the state.
                    innov_pred_st = beh_storage2[key][k][6][state]

                    # So plot from dynamics pred to new state
                    d_innov_act = numb[4, state, k] - dyn_pred[state]
                    axi.arrow(3.5, dyn_pred[state], 0.5, d_innov_act, 
                        width = .05, facecolor='g')

                    # Plot the swapped dynamics
                    axi.arrow(3.5, dyn_pred[state], 0.5, innov_pred_st, 
                        width = .05, facecolor='r')                    

def get_mean_trajs(min_observations = 30.):
    #    len(binned_spikes) == len(push) == len(angs) == len(mags) == len(task_list) == len(targ)

    #    R2_smooth, R2_pred, LL, dynamics_norm, innov_norm, model, pred_data, smooth_data, filt_state, pred_state, filt_sigma, kg, R2_filt, smooth_state = fit_LDS.fit_LDS(
    #        binned_spikes, binned_spikes, n_dim_latent, return_model=True, seed_w_FA=True, nEMiters=30, **dict(seed_pred_x0_with_smoothed_x0= True, get_ratio=True, 
    #        pre_go_bins=pre_go_bins))

    beh_reconst = {}
    for angle in range(8):
        for mag in range(4):

            # State PSTHs by Task, Targ, Vel Bin: 
            smooth_state_dict = {}
            beh_dict = {}
            obs_state_dict = {}
            decoder = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/grom20160302_02_RMLC03021529.pkl'))
            _, decoder_KG = decoder.filt.get_sskf()

            for task in range(2): 
                task_ix = np.nonzero(np.hstack((task_list)) == task)[0]
                
                for target in range(8):
                    targ_ix = np.nonzero(np.array(targ) == target)[0]
                    targs_to_analyze = np.array(list(set(task_ix).intersection(targ_ix)))
                    smooth_state_dict[task, target, angle, mag] = []
                    obs_state_dict[task, target, angle, mag] = []
                    beh_dict[task, target, angle, mag] = []
                    for trl in targs_to_analyze:

                        # Get smoothed state: 
                        data = smooth_state[trl]
                        data_angs = angs[trl][start_bin:]
                        data_mags = mags[trl][start_bin:]

                        ang_ix = np.nonzero(data_angs == angle)[0]
                        mag_ix = np.nonzero(data_mags == mag)[0]

                        ixs = np.array(list(set(ang_ix).intersection(mag_ix)))
                        ixs = ixs[ixs >= 4]
                        ixs = ixs[ixs < len(data) - 4]
                        
                        if len(ixs) > 0:
                            for i in ixs:
                                smooth_state_dict[task, target, angle, mag].append(smooth_state[trl][i-4:i+4, :])
                                beh_dict[task, target, angle, mag].append(np.array(push[trl][start_bin + i - 4 : start_bin + i + 4, :]))
                                obs_state_dict[task, target, angle, mag].append(binned_spikes[trl][start_bin + i - 4: start_bin + i + 4, :])
                    
                    if len(beh_dict[task, target, angle, mag]) > min_observations:
                        mean_state = np.mean(np.dstack((smooth_state_dict[task, target, angle, mag])), axis=2)

                        # Now for each task/bin/target, task the average state trajectory
                        # Then multiply by K*C*state to get an estimate of the mean behavior and see if that
                        # Matches that mean behavior
                        pred_beh = (np.dot(decoder_KG, np.dot(model.C, mean_state.T))[[3, 5], :]).T
                        pred_beh_inflate = np.dot(decoder_KG, np.dot(model.C, mean_state.T)).T
                        pred_angs, pred_mags = tuning_diffs.compute_bins([pred_beh_inflate], 0, 'grom')

                        mean_beh = np.mean(np.dstack((beh_dict[task, target, angle, mag])), axis=2)[:, [3, 5]]
                        mean_beh_inflate = np.mean(np.dstack((beh_dict[task, target, angle, mag])), axis=2)
                        mean_angs, mean_mags = tuning_diffs.compute_bins([mean_beh_inflate], 0, 'grom')
                        
                        ### Sanity check --> observations ###

                        # mean_obs = np.mean(np.dstack((obs_state_dict[task, target, angle, mag])), axis=2)
                        # pred_beh = (np.dot(decoder_KG, mean_obs.T)[[3, 5], :]).T
                        # Correlation b/w mean beh and pred_beh: 
                        slp, intc, rv, pv, err = scipy.stats.linregress(pred_beh.reshape(-1),
                            mean_beh.reshape(-1))

                        reconstruction_error = 1/8.*np.sum((pred_beh.reshape(-1) - mean_beh.reshape(-1))**2)
                        beh_reconst[task, target, angle, mag] = [slp, intc, rv, pv, err, reconstruction_error]

                        # Number of bin diffs: 
                        bin_diffs = 0
                        middle_same = True
                        for iii in range(2, 7):
                            if np.logical_or(pred_angs[0][iii]!=mean_angs[0][iii], pred_mags[0][iii]!=mean_mags[0][iii]):
                                bin_diffs += 1
                                if iii == 4:
                                    middle_same = False
                                


                        reconstruction_error = 1/8.*np.sum((pred_beh.reshape(-1) - mean_beh.reshape(-1))**2)
                        beh_reconst[task, target, angle, mag] = [slp, intc, rv, pv, err, reconstruction_error, bin_diffs, middle_same]
                        
                        # if reconstruction_error < .5:
                        #     if mag == 3:
                        #         f, ax = plt.subplots(ncols=2)
                        #         ax[0].plot(pred_beh[:, 0], '-')
                        #         ax[1].plot(pred_beh[:, 1], '-')

                        #         ax[0].plot(mean_beh[:, 0], '--')
                        #         ax[1].plot(mean_beh[:, 1], '--')
                        #         ax[0].set_title('task: '+str(task) + ', target: '+str(target)+', angle'+str(angle)+', mag:'+str(mag))

    # Ok now what are the dynamics ratios of these mean trajectories? 
    cnt = -1
    for k in beh_reconst.keys():
        if np.logical_and(beh_reconst[k][6]  == 0, beh_reconst[k][7] == True):
            print k
            cnt += 1
    print 'cnt: ', cnt

    err = [beh_reconst[k][6] for k in beh_reconst.keys()]

def plot_x_wi_points(points):
    f, ax = plt.subplots(ncols = 4)
    f.set_figwidth(20)
    f.set_figheight(5)
    x = [];
    y = [];
    y0 = [];
    y00 = [];
    y1 = [];
    y2 = [];

    # Which index combos show the highest beh diff and change in dynamics? 

    colors = dict(x='r', wi='gray')
    for i in range(8):
        for j in range(4):
            for task_type in ['x', 'wi']:
                tmp = np.vstack(( points[i, j, task_type] ))
                
                dr = np.mean(tmp[:, 1])
                dr_norm = np.mean(tmp[:, 1] - tmp[:, 0])
                dr_norm2 = np.mean(tmp[:, 1]/tmp[:, 0])
                beh = np.mean(tmp[:, 2])

                state_diff = np.mean(np.hstack((states[i, j, task_type])))
                obs_diff = np.mean(np.hstack((obs[i, j, task_type])))

                x.append(beh)
                y.append(dr)
                y0.append(dr_norm)
                y00.append(dr_norm2)
                y1.append(state_diff)
                y2.append(obs_diff)

                ax[0].plot(beh, dr, '.', color = colors[task_type])
                ax[1].plot(beh, dr_norm, '.', color=colors[task_type])
                ax[2].plot(beh, state_diff, '.', color = colors[task_type])
                ax[3].plot(beh, obs_diff, '.', color=colors[task_type])

                if beh > 7 : 
                    print 'gt 7: angle: ', i, ', magnitude: ', j, ', task type: ', task_type


    for i, (yi_, tit) in enumerate(zip([y, y0, y1, y2], ['Diff Dyn. Ratio',
        'Subtact Norm diff dyn. ratio', 
        'Norm Diff State', 'Norm Diff Obs'])):

        slp, intc, rv0, pv0, err0 = scipy.stats.linregress(x, yi_)
        x_ = np.arange(4.5, 7.5, .5)
        y_ = slp*x_ + intc
        ax[i].plot(x_, y_, '--', color='gray')
        ax[i].set_title(tit+' p: '+str(np.round(1000*pv0)/1000.) + ', r: '+str(np.round(1000*rv0)/1000.),fontsize=9)

    plt.savefig('comparison_x_w_labs.png')

    _, _, rv0, pv0, err0 = scipy.stats.linregress(x, y0)
    _, _, rv1, pv1, err1 = scipy.stats.linregress(x, y1)
    _, _, rv2, pv2, err2 = scipy.stats.linregress(x, y2)
    
    
            #slp, intc, rv, pv, err = scipy.stats.linregress(tmp[:, 2], tmp[:, 1])
            #print 'rv: ', rv, i, j
            #import pdb; pdb.set_trace()
            #plt.close('all')

    #         X.append(np.mean(tmp[:, 1]))
    #         X_sub_norm.append(np.mean(tmp[:, 1] - tmp[:, 0]))
    #         X_div_norm.append(np.mean(tmp[:, 1] / tmp[:, 0]))

    #         X_all.append(tmp[:, 1])
    #         Y_all.append(tmp[:, 2])
    #         X_og.append(tmp[:, 0])
    #         Y.append(np.mean(tmp[:, 2]))
    # X = np.hstack((X))
    # X_all = np.hstack((X_all))
    # Y_all = np.hstack((Y_all))

def compute_bins(push, select_day, animal):
    angi = []
    magi = []
    for ip, push_trl in enumerate(push):
    
        push_trl = np.squeeze(np.array(push_trl))
        mag = np.sqrt(push_trl[:, 3]**2 + push_trl[:, 5]**2)
        ang = np.array([math.atan2(yi, xi) for i, (xi, yi) in enumerate(zip(push_trl[:, 3], push_trl[:, 5]))])
        ang[ang < 0] = ang[ang < 0] + 2*np.pi

        ### THIS CHANGED AS OF 2-7-19 --> NOW RADIAL TUNING IS [[-22.5, 22.5], [22.5, 45.], [45, 67.5], [67.5, 90], etc.]
        boundaries = np.linspace(0, 2*np.pi, 9) - np.pi/8.

        ### Then subtract minus np.pi
        ang[ang > boundaries[-1]] = ang[ang > boundaries[-1]] - (2*np.pi)
        
        dig = np.digitize(ang, boundaries) - 1 # will vary from 0 - 7

        mag_thresh = mag_bins[(animal, select_day)]
        mag_dig = np.digitize(mag, mag_thresh)

        angi.append(dig)
        magi.append(mag_dig)

    print 'compute bins: ', len(angi), len(magi)
    return angi, magi

def lock_to_high_low_dr():
    ''' Method to plot trajectories sorted by dyn ratio '''
    #    len(binned_spikes) == len(push) == len(angs) == len(mags) == len(task_list) == len(targ)

    #    R2_smooth, R2_pred, LL, dynamics_norm, innov_norm, model, pred_data, smooth_data, filt_state, pred_state, filt_sigma, kg, R2_filt, smooth_state = fit_LDS.fit_LDS(
    #        binned_spikes, binned_spikes, n_dim_latent, return_model=True, seed_w_FA=True, nEMiters=30, **dict(seed_pred_x0_with_smoothed_x0= True, get_ratio=True, 
    #        pre_go_bins=pre_go_bins))

    # Filter by task / target / neural push bin
    # Get DR at each of those bins and beh trajectory nearby

    data_ix = {}

    # For each trial, bin

    for task in range(2):
        for target in range(8):
            for angle in range(8):
                for mag in range(4):
                    data_ix[task, target, angle, mag] = []

    for trl in range(285):

        # Get behavior
        pushes = push[trl]
        ang_trl = angs[trl]
        mag_trl = mags[trl]
        task_trl = task_list[trl][0]
        target = targ[trl]

        for it, (psh, a, m) in enumerate(zip(pushes[10:, :], ang_trl[10:], mag_trl[10:])):
            if np.logical_and(it >= 4, it < len(pushes) - 10 - 4):
                dyn = dynamics_norm[trl][it]; 
                inn = innov_norm[trl][it]; 
                dr = dyn / (dyn + inn)

                data_ix[task, target, a, m].append([np.array(pushes[10 + it - 4: 10 + it + 4, [3, 5]]), dr])

    # For each velocity command #
    T = np.arange(-4, 4, 1)
    colors = ['orangered', 'green', 'blue']
    tasks = ['co', 'obs']
    dr_max = {}
    for task in range(2):
        for target in range(8):
            for angle in range(8):
                for mag in range(4):
                    y = data_ix[task, target, angle, mag]
                    
                    # Pushes and dynamics
                    if len(y) > 30:
                        z = np.dstack(( tmp[0] for tmp in y ))
                        d = np.array([tmp[1] for tmp in y])

                        low_dr = np.percentile(d, 20)
                        high_dr = np.percentile(d, 80)

                        low_ix = np.nonzero(d < low_dr)[0]
                        high_ix = np.nonzero(d > high_dr)[0]
                        mid_ix = np.nonzero(np.logical_and(d >= low_dr, d <= high_dr))[0]
                        dr_max[task, target, angle, mag] = high_dr

                        if high_dr > 0.65:
                            f, ax = plt.subplots(ncols = 2)
                            for ia, axis in enumerate(ax):
                                axi = plot_mean_and_var(T, z[:, ia, high_ix], array_axis = 1, color=colors[0], ax=axis)
                                axi = plot_mean_and_var(T, z[:, ia, mid_ix], array_axis = 1, color=colors[1], ax=axis)
                                axi = plot_mean_and_var(T, z[:, ia, low_ix], array_axis = 1, color=colors[2], ax=axis)
                                axi.axis('square')
                            axis.set_title('Task: '+tasks[task]+', Targ:'+str(target)+', Ang:'+str(angle)+', Mag:'+str(mag))

def plot_mean_and_var(x, array, ax, color='b', array_axis=1, label='0', marker_type ='-'):
    
    mean = np.nanmean(array, axis=array_axis)
    sem_plus = mean + np.nanvar(array, axis=array_axis)
    sem_minus = mean - np.nanvar(array, axis=array_axis)
    
    ax.fill_between(x, sem_plus, sem_minus, color=color, alpha=0.5)
    x = ax.plot(x, mean,marker_type,color=color,label=label)

    return ax

def plot_mean_and_sem(x , array, ax, color='b', array_axis=1, label='0', marker_type ='-'):
    
    mean = np.nanmean(array, axis=array_axis)
    sem_plus = mean + scipy.stats.sem(array, axis=array_axis, nan_policy='omit')
    sem_minus = mean - scipy.stats.sem(array, axis=array_axis, nan_policy='omit')
    
    ax.fill_between(x, sem_plus, sem_minus, color=color, alpha=0.5)
    x = ax.plot(x, mean, marker_type, color=color,label=label)
    ymin = np.percentile(mean, 10)
    ymax = np.percentile(mean, 90)
    yz = [ymin, ymax]
    return x, ax, yz

def shuffle_LDS(pre_go = 1., select_day = 0, animal='grom', reuse_real_A = True, nreps = 1):
    inputs = [[4377], [4378]]#, 4382]]

    # Aggregate spike counts: 
    binned_spikes = []
    angs = []
    mags = []
    task_list = []
    targ = []
    push = []

    co_obs_dict = pickle.load(open(pref+'co_obs_file_dict.pkl'))
    
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
            push_i = [decoder_all[i] for i in range(len(decoder_all)) if i not in exclude]
            angi, magi = compute_bins(push_i, select_day, animal)

            push.extend(push_i)
            angs.extend(angi)
            mags.extend(magi)
            
            task_list.extend([task] for i in range(len(bin_spk)) if i not in exclude)

            sub_targ_ix = []
            for trl in np.unique(trial_ix_all):
                if trl not in exclude:
                    ix = np.nonzero(trial_ix_all==trl)[0]
                    if pre_go is None:
                        sub_targ_ix.append(targ_ix[ix[0]])
                    else:
                        sub_targ_ix.append(targ_ix[ ix[0] + int(pre_go*10) + 2])
            targ.extend(np.hstack((sub_targ_ix)))

    assert len(binned_spikes) == len(push) == len(angs) == len(mags) == len(task_list) == len(targ)
    n_dim_latent = 15
    pre_go_bins = 10

    R2_smooth, R2_pred, LL, dynamics_norm, innov_norm, model, pred_data, smooth_data, filt_state, pred_state, filt_sigma, kg, R2_filt, smooth_state = fit_LDS.fit_LDS(
    binned_spikes, binned_spikes, n_dim_latent, return_model=True, seed_w_FA=True, nEMiters=30, **dict(seed_pred_x0_with_smoothed_x0= True, get_ratio=True, 
        pre_go_bins=pre_go_bins, include_pre_go_data = True))

    if reuse_real_A:
        model_to_use = model
    else:
        model_to_use = None

    # Plot PSTH of trials: 
    # This is for filtered data: 
    f, ax = plt.subplots(ncols = 3, nrows = 3, figsize=(12, 12))
    for i in range(3):
        plot_dyn_ratio(ax[0, i], innov_norm, dynamics_norm, filt_state, 'k')
        plot_mag(ax[1, i], dynamics_norm, 'k')
        plot_mag(ax[2, i], innov_norm, 'k')

    cols  = ['maroon', 'orangered', 'darkgoldenrod', 'olivedrab', 
    'teal', 'steelblue', 'midnightblue', 'darkmagenta']

    # Shuffle over everything
    for i in range(8):
        binned_spikes_shuff = shuff_bins_all(binned_spikes)
        R2_smooth, R2_pred, LL, dynamics_norm, innov_norm, model, pred_data, smooth_data, filt_state, pred_state, filt_sigma, kg, R2_filt, smooth_state = fit_LDS.fit_LDS(
        binned_spikes_shuff, binned_spikes_shuff, n_dim_latent, return_model=True, seed_w_FA=True, nEMiters=30, **dict(seed_pred_x0_with_smoothed_x0= True, get_ratio=True, 
            pre_go_bins=pre_go_bins, eval_model = model_to_use, include_pre_go_data = True))
        plot_dyn_ratio(ax[0, 0], innov_norm, dynamics_norm, filt_state, cols[i])
        plot_mag(ax[1, 0], dynamics_norm, cols[i])
        plot_mag(ax[2, 0], innov_norm, cols[i])



    ax[0, 0].set_title('Real vs. Shuffle ALL')


    # Shuffle within trial
    for i in range(8):
        binned_spikes_shuff = shuff_bins(binned_spikes)
        R2_smooth, R2_pred, LL, dynamics_norm, innov_norm, model, pred_data, smooth_data, filt_state, pred_state, filt_sigma, kg, R2_filt, smooth_state = fit_LDS.fit_LDS(
        binned_spikes_shuff, binned_spikes_shuff, n_dim_latent, return_model=True, seed_w_FA=True, nEMiters=30, **dict(seed_pred_x0_with_smoothed_x0= True, get_ratio=True, 
            pre_go_bins=pre_go_bins, eval_model = model_to_use, include_pre_go_data = True))
        plot_dyn_ratio(ax[0, 1], innov_norm, dynamics_norm, filt_state, cols[i])
        plot_mag(ax[1, 1], dynamics_norm, cols[i])
        plot_mag(ax[2, 1], innov_norm, cols[i])
    ax[0, 1].set_title('Real vs. Shuffle within Trial')

    # Sample with FA: 
    FA = skdecomp.FactorAnalysis(n_components=n_dim_latent)
    dat = np.vstack((binned_spikes))
    FA.fit(dat)

    U = FA.components_.T
    Psi = np.diag(FA.noise_variance_)
    mu = FA.mean_; 

    nfactors = n_dim_latent

    for i in range(8):
        fa_bins = []
        for b in binned_spikes:
            n = b.shape[0]
            neur = b.shape[1]
            factors = np.random.normal(size=(nfactors, n))
            private = np.random.multivariate_normal(np.zeros(len(mu)), Psi, (n))
            obs = np.dot(U, factors).T + private + mu[np.newaxis, :]
            fa_bins.append(obs)
        
        R2_smooth, R2_pred, LL, dynamics_norm, innov_norm, model, pred_data, smooth_data, filt_state, pred_state, filt_sigma, kg, R2_filt, smooth_state = fit_LDS.fit_LDS(
        fa_bins, fa_bins, n_dim_latent, return_model=True, seed_w_FA=True, nEMiters=30, 
        **dict(seed_pred_x0_with_smoothed_x0=  True, get_ratio=True, pre_go_bins=pre_go_bins,
            eval_model = model_to_use, include_pre_go_data = True))

        plot_dyn_ratio(ax[0, 2], innov_norm, dynamics_norm, cols[i], marker='-')
        plot_mag(ax[1, 2], dynamics_norm, cols[i])
        plot_mag(ax[2, 2], innov_norm, cols[i])
    ax[0, 2].set_title('Real vs. Sample w/ FA')

    for il, lab in enumerate(['Dyn Ratio', 'Dyn Norm', 'Inn Norm']):
        ax[il, 0].set_ylabel(lab)

    f.tight_layout()
    f.savefig('shuffle_data_reuseA_'+str(reuse_real_A)+'.png')

def shuff_bins(binned_spikes):
    tmp = []
    for b in binned_spikes:
        ix = np.random.permutation(b.shape[0])
        tmp.append(b[ix, :])
    return tmp

def shuff_bins_all(binned_spikes):
    tmp = []
    T = []
    for b in binned_spikes:
        T.append(b.shape[0])
        tmp.append(b)
    tmp = np.vstack((tmp))
    ix = np.random.permutation(tmp.shape[0])

    tmp2 = []
    for t in T:
        ix2 = ix[:t]
        tmp2.append(tmp[ix2, :])
        ix = ix[t:]
    return tmp2

def plot_dyn_ratio(ax, innov_norm, dynamics_norm, filt_state, color, marker='-'):
    trl = []

    for i, (inn, dyn, flt) in enumerate(zip(innov_norm, dynamics_norm, filt_state)):

        tmp = np.array(inn) + dyn
        tmp2 = np.zeros((300))
        tmp2[:] = np.nan; 
        tmp2[:len(dyn)] = dyn / tmp
        trl.append( tmp2 )

    trl = np.vstack((trl))
    T = np.arange(300)*0.1
    plot_mean_and_sem(T, trl, ax, color=color, array_axis=0, label='0',
        marker_type=marker)
    ax.set_xlim([-1, 5])
    
    # trl2 = []
    # for data in filt_state: 
    #     dyn_ratio, _, _, _ = kf_data2dyn_ratio(dict(mus=data), model.A, prefix_len = 1)
    #     trl2.append(dyn_ratio)

    # Now shuffle: 

def plot_mag(ax, val, color):
    trl = [] 
    for i, v in enumerate(val):
        tmp2 = np.zeros((300))
        tmp2[:] = np.nan; 
        tmp2[:len(v)] = v
        trl.append( tmp2 )

    trl = np.vstack((trl))
    T = np.arange(300)*0.1
    plot_mean_and_sem(T, trl, ax, color=color, array_axis=0, label='0',
        marker_type='-')
    ax.set_xlim([-1, 5])

def kf_data2dyn_ratio(filt_state, A, prefix_len=1):
    #kf_data fields: {'mus', 'sigmas'}
    #mus: num_samples X num_dim
    #sigmas: num_samples X num_dim X num_dim
    #A, dynamics matrix: num_dim X num_dim
    #prefix_len: num of prefix samples in the trial data, 
    #to help set initial state.  
    #These will not be analyzed in dynamics ratio.
    #prefix_len must be greater than 0.  

    num_dim = A.shape[0] #A should be square

    tmp = np.dot(A, filt_state.T)
    #num_dim x num_samples
    
    dyn_pred = tmp[:,:-1].T
    #dyn_pred, evolves the kalman filtered states by the state transition matrix
    #the last sample is dropped, because we can't compare it to anything

    #num_samples-1 X num_dim
    dyn_delta = dyn_pred - filt_state[:-1,:]
    innov = filt_state[prefix_len:,:] - dyn_pred[(prefix_len-1):,:]

    dyn_delta = dyn_delta[prefix_len-1:,:]
    dyn_pred = dyn_pred[(prefix_len-1):,:]
    #num_samples-prefix_len X num_dim

    numerator = np.linalg.norm(dyn_delta, axis=1)
    denominator = numerator + np.linalg.norm(innov, axis=1)
    dyn_ratio = numerator/denominator
    return dyn_ratio, numerator, np.linalg.norm(innov, axis=1)


