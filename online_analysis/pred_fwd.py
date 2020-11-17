
from generalization_plots import DataExtract
import analysis_config
import util_fcns
import plot_fr_diffs, generate_models, generalization_plots, generate_models_utils
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import pickle


def plot_R2_model(model_nm = 'hist_1pos_0psh_0spksm_1_spksp_0', model_set_number = 6,
    nshuffs = 20, plot_action = False):
    
    f, ax = plt.subplots(figsize=(3, 3))
    col = dict()
    col[True] = 'maroon'
    col[False] = analysis_config.blue_rgb; 

    for i_a, animal in enumerate(['grom', 'jeev']):

        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):

            ### Load data ###
            dataObj = DataExtract(animal, day_ix, model_nm = model_nm, 
                model_set_number = model_set_number, nshuffs=nshuffs)
            dataObj.load()
            dataObj.load_null_roll()

            if plot_action:
                KG = util_fcns.get_decoder(animal, day_ix)

                r2_true = util_fcns.get_R2(np.dot(KG, dataObj.spks.T).T, 
                    np.dot(KG, dataObj.pred_spks.T).T)
                
                r2_shuff = []
                for n in range(nshuffs):
                    r2_shuff.append(util_fcns.get_R2(np.dot(KG, dataObj.spks.T).T, 
                        np.dot(KG, dataObj.pred_spks_shuffle[:, :, n].T).T))

                r2_null_roll = []
                for n in range(nshuffs):
                    if animal == 'grom':
                        assert(np.allclose(0.1*np.dot(KG, dataObj.null_roll_true[:, :, n].T).T, dataObj.push[:, [3, 5]]))
                    r2_null_roll.append(util_fcns.get_R2(np.dot(KG, dataObj.null_roll_true[:, :, n].T).T, 
                        np.dot(KG, dataObj.null_roll_pred[:, :, n].T).T))
            else:
                ### get predictions ###
                r2_true = util_fcns.get_R2(dataObj.spks, dataObj.pred_spks)

                r2_shuff = []
                for n in range(nshuffs):
                    r2_shuff.append(util_fcns.get_R2(dataObj.spks, dataObj.pred_spks_shuffle[:, :, n]))

                r2_null_roll = []
                for n in range(nshuffs):
                    r2_null_roll.append(util_fcns.get_R2(dataObj.null_roll_true[:, :, n], dataObj.null_roll_pred[:, :, n]))

            ### Plot the r2 ###
            xpos = i_a*10 + day_ix
            ax.plot(xpos, r2_true, '.', markersize=15, color=col[plot_action])
            #print('r2 act %s %s %d = %.3f' %(str(plot_action), animal, day_ix, r2_true))

            util_fcns.draw_plot(xpos, r2_shuff, 'k', np.array([1., 1., 1., 0.]), ax)
            util_fcns.draw_plot(xpos, r2_null_roll, 'deeppink', np.array([1., 1., 1., 0.]), ax)
            #print('pink shuffle mean r2 act %s %s %d = %.3f' %(str(plot_action), animal, day_ix, np.mean(r2_null_roll)))
            ix = np.nonzero(np.hstack((r2_null_roll)) >= r2_true)[0]
            pv = float(len(ix))/float(len(r2_null_roll))
            print('%s, %d, pv = %.5f' %(animal, day_ix, pv))
            
            ax.plot([xpos, xpos], [np.mean(r2_shuff), r2_true], 'k-', linewidth=.5)

    ax.set_xlim([-1, 14])
    f.tight_layout()
    util_fcns.savefig(f, 'fwd_pred_action%s'%(str(plot_action)))

def frac_next_com_mov_sig(model_nm = 'hist_1pos_0psh_0spksm_1_spksp_0', model_set_number = 6,
    nshuffs = 1000):

    f, ax = plt.subplots(figsize=(2, 3))
    for i_a, animal in enumerate(['grom', 'jeev']):
        frac_sig_animal = []

        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):

            ### Load data ###
            dataObj = DataExtract(animal, day_ix, model_nm = model_nm, 
                model_set_number = model_set_number, nshuffs=nshuffs)
            dataObj.load()
            KG = util_fcns.get_decoder(animal, day_ix)
            sig_mc = 0; 
            all_mc = 0; 

            ### For each move / command is pred closer than shuffle 
            for mag in range(4):
                for ang in range(8): 
                    for mov in np.unique(dataObj.move):
                        ix = np.nonzero((dataObj.command_bins_tm1[:, 0] == mag) & (dataObj.command_bins_tm1[:, 1] == ang) & (dataObj.move_tm1 == mov))[0]

                        if len(ix) >= 15: 

                            #### Get true next MC; 
                            true_next_action = np.mean(np.dot(KG, dataObj.spks[ix, :].T).T, axis=0)
                            pred_next_action = np.mean(np.dot(KG, dataObj.pred_spks[ix, :].T).T, axis=0)

                            pred_dist = np.linalg.norm(pred_next_action - true_next_action)
                            shuff_next_action_dist = []
                            for i in range(nshuffs): 
                                tmp_act = np.mean(np.dot(KG, dataObj.pred_spks_shuffle[ix, :, i].T).T, axis=0)
                                shuff_next_action_dist.append(np.linalg.norm(tmp_act - true_next_action))

                            p = np.nonzero(shuff_next_action_dist <= pred_dist)[0]
                            pv = float(len(p))/float(nshuffs)

                            if pv < 0.05:
                                sig_mc += 1
                            all_mc += 1
            frac_sig = float(sig_mc)/float(all_mc)
            frac_sig_animal.append(frac_sig)
            ax.plot(i_a, frac_sig, 'k.')

        ax.bar(i_a, np.mean(frac_sig_animal), width=0.8, color='k', alpha=0.2)
    ax.set_xticks([0, 1])
    ax.set_ylim([0., 1.])
    ax.set_xticklabels(['G', 'J'])
    ax.set_ylabel('Frac Mov-Commands with\nSig. Next Command Prediction', fontsize=10)
    f.tight_layout()
    util_fcns.savefig(f, 'frac_mc_w_sig_next_comm_pred')

def pred_vs_true_next_command(model_nm = 'hist_1pos_0psh_0spksm_1_spksp_0', model_set_number = 6, nshuffs = 2,
    mag_eg=0, ang_eg=7):

    f_cc, ax_cc = plt.subplots(figsize=(3, 3))
    f_er, ax_er = plt.subplots(figsize=(3, 3))

    f_pw, ax_pw = plt.subplots(figsize=(5, 5))
    ax_pw2 = ax_pw.twinx()
    PW_eg = []; PW_shuff = []; PW_shuff_rol = []


    for i_a, animal in enumerate(['grom', 'jeev']):
        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):


            if animal in ['grom', 'jeev'] and day_ix in range(10):
                f_scat, ax_scatter = plt.subplots(figsize=(3, 3))
                f_scat_shuff, ax_scatter_shuff = plt.subplots(figsize=(3, 3))
                for axi in [ax_scatter, ax_scatter_shuff]:
                    axi.set_ylabel('Subj %s Day %d' %(animal, day_ix))


            ### Load data ###
            dataObj = DataExtract(animal, day_ix, model_nm = model_nm, 
                model_set_number = model_set_number, nshuffs=nshuffs)
            dataObj.load()
            dataObj.load_null_roll()

            KG = util_fcns.get_decoder(animal, day_ix)

            #### multipy by 0.1 so that pred_push is correct magnitude
            pred_push = np.dot(KG, 0.1*dataObj.pred_spks.T).T
            X = [];

            X_shuff = {}; 
            pred_push_shuff = {}; 
            pred_push_shuff_roll = {}
            for i in range(nshuffs):
                X_shuff[i] = []
                pred_push_shuff[i] = np.dot(KG, 0.1*dataObj.pred_spks_shuffle[:, :, i].T).T
                pred_push_shuff_roll[i] = np.dot(KG, 0.1*dataObj.null_roll_pred[:, :, i].T).T
                assert(not np.allclose(pred_push_shuff[i], pred_push_shuff_roll[i]))

            ### For each move / command is pred closer than shuffle 
            for mag in range(4):
                for ang in range(8): 

                    ##### Get all the relevant commands 
                    mov_ix = {}
                    for mov in np.unique(dataObj.move):
                        ix = np.nonzero((dataObj.command_bins_tm1[:, 0] == mag) & (dataObj.command_bins_tm1[:, 1] == ang) & (dataObj.move_tm1 == mov))[0]
                        if len(ix) >= 15: 
                            mov_ix[mov] = ix.copy()
                            
                    ##### Global;
                    movs = np.array(mov_ix.keys()) 
                    for i_m1, mov1 in enumerate(movs):

                        ix1 = mov_ix[mov1]

                        #### Match the global to the move-specific and take the distance between glboal and move spec + 
                        for i_m2, mov2 in enumerate(movs[i_m1+1:]):

                            ix2 = mov_ix[mov2]

                            ix11, ix22, nt = plot_fr_diffs.distribution_match_mov_pairwise(dataObj.push_tm1[np.ix_(ix1, [3, 5])], 
                                dataObj.push_tm1[np.ix_(ix2, [3, 5])], psig=.05, perc_drop = 0.05)

                            if np.logical_and(len(ix11) >= 15, len(ix22)>= 15):
                                mov11 = np.mean(dataObj.push[np.ix_(ix1[ix11], [3, 5])], axis=0)
                                mov22 = np.mean(dataObj.push[np.ix_(ix2[ix22], [3, 5])], axis=0)
                                true_pw_dist = np.linalg.norm(mov11 - mov22)

                                pmv1 = np.mean(pred_push[ix1[ix11], :], axis=0)
                                pmv2 = np.mean(pred_push[ix2[ix22], :], axis=0)

                                pred_pw_dist = np.linalg.norm(pmv1 - pmv2)
                                X.append([true_pw_dist, pred_pw_dist])

                                # import pdb; pdb.set_trace()

                                # if np.isnan(true_pw_dist):
                                #     import pdb; pdb.set_trace()
                                # if np.isnan(pred_pw_dist):
                                #     import pdb; pdb.set_trace()
                                tmp = []; tmp2 = []; 
                                for i in range(nshuffs):
                                    shuf_mn1 = np.mean(pred_push_shuff[i][ix1[ix11], :], axis=0) 
                                    shuf_mn2 = np.mean(pred_push_shuff[i][ix2[ix22], :], axis=0) 
                                    pred_pw_dist_shuff = np.linalg.norm(shuf_mn1 - shuf_mn2)
                                    
                                    shuf_rol1 = np.mean(pred_push_shuff_roll[i][ix1[ix11], :], axis=0)
                                    shuf_rol2 = np.mean(pred_push_shuff_roll[i][ix2[ix22], :], axis=0)
                                    pred_pw_dist_shuff_rol = np.linalg.norm(shuf_rol1 - shuf_rol2)

                                    X_shuff[i].append([true_pw_dist, pred_pw_dist_shuff, pred_pw_dist_shuff_rol])
                                    tmp.append(pred_pw_dist_shuff)
                                    tmp2.append(pred_pw_dist_shuff_rol)

                                    if animal in ['grom', 'jeev'] and day_ix in range(10) and i == 0:
                                        ax_scatter_shuff.plot(true_pw_dist, pred_pw_dist_shuff, 'k.', markersize=5.)
                                        ax_scatter_shuff.plot(true_pw_dist, pred_pw_dist_shuff_rol, '.', color='deeppink', markersize=5.)

                                if animal in ['grom', 'jeev'] and day_ix in range(10):
                                    ax_scatter.plot(true_pw_dist, pred_pw_dist, 'k.', markersize=5.)

                                    if mag == mag_eg and ang == ang_eg:
                                        PW_eg.append([mov1, mov2, true_pw_dist, pred_pw_dist])
                                        PW_shuff.append(tmp)
                                        PW_shuff_rol.append(tmp2)


            X = np.vstack((X))
            _,_,rv,_,_ = scipy.stats.linregress(X[:, 0], X[:, 1])
            ax_cc.plot(10*i_a + day_ix, rv, '.', color='maroon', markersize=15)

            ### Compute error 
            err = np.mean(np.abs(X[:, 0] - X[:, 1]))
            ax_er.plot(10*i_a + day_ix, err, '.', color='maroon', markersize=15)

            if animal in ['grom', 'jeev'] and day_ix in range(10):
                ax_scatter.set_title(' r %.3f' %(rv), fontsize=10)
            rv_shuff = []; er_shuff = []; rv_shuff_rol = []; er_shuff_rol = []; 

            for i in range(nshuffs):
                X_shuff[i] = np.vstack((X_shuff[i])) 
                _,_,rv_shf,_,_ = scipy.stats.linregress(X_shuff[i][:, 0], X_shuff[i][:, 1])
                rv_shuff.append(rv_shf)
                er_shuff.append(np.mean(np.abs(X_shuff[i][:, 0] - X_shuff[i][:, 1])))

                _,_,rv_shf_rol,_,_ = scipy.stats.linregress(X_shuff[i][:, 0], X_shuff[i][:, 2])
                rv_shuff_rol.append(rv_shf_rol)
                er_shuff_rol.append(np.mean(np.abs(X_shuff[i][:, 0] - X_shuff[i][:, 2])))

                if animal in ['grom', 'jeev'] and day_ix in range(10) and i == 0: 
                    ax_scatter_shuff.set_title('r shuff %.3f, rol %.3f' %(rv_shf, rv_shf_rol), fontsize=10)

            util_fcns.draw_plot(10*i_a + day_ix, rv_shuff, 'k', np.array([1.,1., 1., 0.]), ax_cc)
            util_fcns.draw_plot(10*i_a + day_ix, rv_shuff_rol, 'deeppink', np.array([1.,1., 1., 0.]), ax_cc)
            ax_cc.plot([10*i_a + day_ix, 10*i_a + day_ix], [np.mean(rv_shuff), rv], 'k-', linewidth=0.5)

            util_fcns.draw_plot(10*i_a + day_ix, er_shuff, 'k', np.array([1.,1., 1., 0.]), ax_er)
            util_fcns.draw_plot(10*i_a + day_ix, er_shuff_rol, 'deeppink', np.array([1.,1., 1., 0.]), ax_er)
            ax_er.plot([10*i_a + day_ix, 10*i_a + day_ix], [np.mean(er_shuff), err], 'k-', linewidth=0.5)
            

            ax_scatter.set_xlim([0., 2.])
            ax_scatter.set_ylim([0., 1.])
            f_scat.tight_layout()
    
            ax_scatter_shuff.set_xlim([0., 2.])
            ax_scatter_shuff.set_ylim([0., 1.])
            f_scat_shuff.tight_layout()
    
    ax_cc.set_xlim([-1, 14])
    ax_cc.set_xticks([])
    f_cc.tight_layout()

    ax_er.set_xlim([-1, 14])
    ax_er.set_xticks([])
    f_er.tight_layout()

    #util_fcns.savefig(f_scat, 'scatter_action_dist_grom_0')
    #util_fcns.savefig(f_scat_shuff, 'scatter_shuff_action_dist_grom_0')
    util_fcns.savefig(f_cc, 'true_v_pred_next_act_cc')
    util_fcns.savefig(f_er, 'true_v_pred_next_act_err')

    PW_eg = np.vstack((PW_eg))
    ix_sort = np.argsort(PW_eg[:, 2])[::-1]

    for ii, i_s in enumerate(ix_sort):
        ax_pw.plot(ii, PW_eg[i_s, 2], 'k.')
        ax_pw2.plot(ii, PW_eg[i_s, 3], '.', color='maroon')
        util_fcns.draw_plot(ii, PW_shuff[i_s], 'k', np.array([1., 1., 1., 0.]), ax_pw2)
        util_fcns.draw_plot(ii, PW_shuff_rol[i_s], 'deeppink', np.array([1.,1.,1.,0.]), ax_pw2)

        ax_pw.plot(ii, 0, '.', markersize=20, color=util_fcns.get_color(PW_eg[i_s, 0]))
        ax_pw.plot(ii, -0.1, '.', markersize=20, color=util_fcns.get_color(PW_eg[i_s, 1]))
    ax_pw.set_xlim([-1, 36])
    ax_pw2.set_xlim([-1, 36])
    ax_pw2.set_ylim([-.15, .55])
    ax_pw.set_xticks([])
    ax_pw2.set_xticks([])
    f_pw.tight_layout()

    util_fcns.savefig(f_pw, 'pw_plot_next_action')

def plot_pw_next_action_eg(model_nm = 'hist_1pos_0psh_0spksm_1_spksp_0', model_set_number = 6,
    animal = 'grom', day_ix = 0, mag = 0, ang = 0):

    ### Load data ###
    KG = util_fcns.get_decoder(animal, day_ix)
    
    dataObj = DataExtract(animal, day_ix, model_nm = model_nm, 
        model_set_number = model_set_number, nshuffs=nshuffs)
    
    dataObj.load()

    ### plot pairwise differences ###

##### work out issues with the null roll ###
def plot_test_roll(): 
    
    ### Load data ###
    dataObj = DataExtract('grom', 0, model_nm = 'hist_1pos_0psh_0spksm_1_spksp_0', 
        model_set_number = 6, nshuffs=0)


def test_shuffles(model_nm='hist_1pos_0psh_0spksm_1_spksp_0'):
    f, ax = plt.subplots()
    f1, ax1 = plt.subplots()

    alpha_dict = pickle.load(open(analysis_config.config['grom_pref'] + 'max_alphas_ridge_model_set6.pkl', 'rb'))
    mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))

    for i_a, animal in enumerate(['grom']):#, 'jeev']):
        for day_ix in [2, 4]:#range(analysis_config.data_params['%s_ndays'%animal]):
            
            #### Alpha ####
            alpha = alpha_dict[animal][0][day_ix, model_nm]
            alpha = alpha/1000.
            #### Get true data ####
            dataObj = DataExtract(animal, day_ix, model_nm = model_nm, 
                    model_set_number = 6, nshuffs=0)
            dataObj.load()
            
            if animal == 'grom':
                KG, KG_null_proj, KG_potent_orth = generate_models.get_KG_decoder_grom(day_ix)
            
            elif animal == 'jeev':
                KG, KG_null_proj, KG_potent_orth = generate_models.get_KG_decoder_jeev(day_ix)
                
            #### Decompose activity (tm0 and tm1) into null / potent 
            #### maek sure spks and push are aligned? 
            if animal == 'grom':
                assert(np.allclose(np.dot(KG, 0.1*dataObj.spks.T).T, dataObj.push[:, [3, 5]]))
            
            elif animal == 'jeev': 
                assert(generate_models_utils.quick_reg(np.array(np.dot(KG, 0.1*dataObj.spks.T).T), 
                                dataObj.push[:, [3, 5]]) > .98)
                
                #### Make sure null proj is zero ###
                assert(np.allclose(np.dot(KG, np.dot(KG_null_proj, 0.1*dataObj.spks.T)).T, 
                                   0.))
                
                ### Make sure pot proj works: 
                assert(np.allclose(np.array(np.dot(KG, 0.1*dataObj.spks.T).T), 
                                np.dot(KG, np.dot(KG_potent_orth, 0.1*dataObj.spks.T)).T))
            
            ##### Get the null / potent activity ###
            null, pot = generate_models.decompose_null_pot(0.1*dataObj.spks, dataObj.push, KG, KG_null_proj, KG_potent_orth)
            null_tm1, pot_tm1 = generate_models.decompose_null_pot(0.1*dataObj.spks_tm1, dataObj.push_tm1, KG, 
                                                                   KG_null_proj, KG_potent_orth)
            
            ### Push at time - 1; 
            command_bins_tm1 = util_fcns.commands2bins([dataObj.push_tm1], mag_boundaries, animal, day_ix, vel_ix = [3, 5])[0]
            
            #### get command bins disk ####
            assert(np.allclose(np.dot(KG, null.T).T, 0.))
            assert(np.allclose(np.dot(KG, null_tm1.T).T, 0.))
            
            ## Dynamics R2 of null / pot / combined ####
            T = dataObj.spks.shape[0]
            spks_pred, _, _ = generalization_plots.train_and_pred(0.1*dataObj.spks_tm1, 0.1*dataObj.spks, 
                                                                  dataObj.push[:, [3, 5]], 
                                                                  np.arange(T), np.arange(T), alpha, KG, add_mean = None, 
                                                                  skip_cond = True)
            spks_r2 = util_fcns.get_R2(0.1*dataObj.spks, spks_pred)
            ax.plot(day_ix + 10*i_a, spks_r2, 'k*')

            act_r2 = util_fcns.get_R2(dataObj.push[:, [3, 5]], np.dot(KG, spks_pred.T).T)
            ax1.plot(day_ix + 10*i_a, act_r2, 'k*')
     
            shuff = dict(); 
            shuff_act = dict();
            for i in range(2):
                shuff[i] = []
                shuff_act[i] = []
                
            #### Now do some rolling; 
            for i in range(10): 
                
                if i == 0: 
                    feg, axeg = plt.subplots(nrows = 2, ncols = 2)

                ### Shuffle within bin ###
                shuff_ix, _ = generate_models_utils.within_bin_shuffling(pot_tm1, dataObj.push_tm1[:, [3, 5]], 
                                                                           animal, day_ix)
                ######################################################################
                ##################### Shuffled pot + null ############################
                ######################################################################
                #### Make srue the same command bins 
                command_bins_tm1_shuff = util_fcns.commands2bins([dataObj.push_tm1[np.ix_(shuff_ix, [3, 5])]], 
                                                                 mag_boundaries, animal, day_ix, 
                                                                 vel_ix = [0, 1])[0]
                assert(np.allclose(command_bins_tm1_shuff, command_bins_tm1))

                ##### get the full activity shuffled ######
                spks_tm1_shuff1 = 0.1*dataObj.spks_tm1[shuff_ix, :]
                spks_0 = null + pot; 

                spks_pred_i, _, _ = generalization_plots.train_and_pred(spks_tm1_shuff1, spks_0, 
                                                          dataObj.push[:, [3, 5]], 
                                                          np.arange(T), np.arange(T), alpha, KG, add_mean = None, 
                                                          skip_cond = True)
                shuff[0].append(util_fcns.get_R2(spks_0, spks_pred_i))
                shuff_act[0].append(util_fcns.get_R2(dataObj.push[:, [3, 5]], np.dot(KG, spks_pred_i.T).T))

                if i == 0:
                    for ixy in range(2):
                        axeg[ixy, 0].plot(dataObj.push[:, [3, 5]][:, ixy], np.dot(KG, spks_pred_i.T).T[:, ixy], 'k.',
                            alpha=.5)
                        slp,intc,rv,_,_ = scipy.stats.linregress(dataObj.push[:, [3, 5]][:, ixy], 
                            np.dot(KG, spks_pred_i.T).T[:, ixy])
                        r2i=util_fcns.get_R2(dataObj.push[:, [3, 5]][:, ixy], np.dot(KG, spks_pred_i.T).T[:, ixy])
                        axeg[ixy, 0].set_title('slp %.2f, intc %.2f, rv %.2f, r2: %.3f' %(slp, intc, rv, r2i))
                        axeg[ixy, 0].set_ylabel('pred dim %d, beh maint'%ixy)
                        if day_ix == 2:
                            axeg[ixy, 0].plot([-10, 10], [-10, 10], 'k--')
                        elif day_ix == 4:
                            axeg[ixy, 0].plot([-4, 4], [-4, 4], 'k--')
                        
                ######################################################################
                ##################### Shuffled pot + rolled null #####################
                ######################################################################

                #### Roll this guy ; 
                ix = np.arange(T)
                roll_ix = np.roll(ix, np.random.randint(100, T))

                ### Rolled null, un changed potent
                spks_0 = null[roll_ix, :] + pot; 

                if animal == 'grom':
                    assert(np.allclose(np.dot(KG, spks_0.T).T, dataObj.push[:, [3, 5]]))
                elif animal == 'jeev':
                    assert(np.allclose(np.dot(KG, spks_0.T).T, np.dot(KG, 0.1*dataObj.spks.T).T))
                
                #### Shuffled null 
                pot_tm1_shuff = pot_tm1[shuff_ix, :]
                spks_tm1_shuff2 = null_tm1[roll_ix, :] + pot_tm1_shuff; 
                spks_pred_i, _, _ = generalization_plots.train_and_pred(spks_tm1_shuff2, spks_0, 
                                                          dataObj.push[:, [3, 5]], 
                                                          np.arange(T), np.arange(T), alpha, KG, add_mean = None, 
                                                          skip_cond = True)
                
                shuff[1].append(util_fcns.get_R2(spks_0, spks_pred_i))
                shuff_act[1].append(util_fcns.get_R2(dataObj.push[:, [3, 5]], np.dot(KG, spks_pred_i.T).T))
                if i == 0:
                    for ixy in range(2):
                        axeg[ixy, 1].plot(dataObj.push[:, [3, 5]][:, ixy], np.dot(KG, spks_pred_i.T).T[:, ixy], 'b.',
                            alpha=.5)
                        slp,intc,rv,_,_ = scipy.stats.linregress(dataObj.push[:, [3, 5]][:, ixy], 
                            np.dot(KG, spks_pred_i.T).T[:, ixy])
                        r2i=util_fcns.get_R2(dataObj.push[:, [3, 5]][:, ixy], np.dot(KG, spks_pred_i.T).T[:, ixy])
                        axeg[ixy, 1].set_title('slp %.2f, intc %.2f, rv %.2f, r2: %.3f' %(slp, intc, rv, r2i))
                        axeg[ixy, 1].set_ylabel('pred dim %d, null roll + beh maint'%ixy)
                        if day_ix == 2:
                            axeg[ixy, 1].plot([-10, 10], [-10, 10], 'k--')
                        elif day_ix == 4:
                            axeg[ixy, 1].plot([-4, 4], [-4, 4], 'k--')

            cols = ['k','royalblue']
            for i_s in range(2):
                util_fcns.draw_plot(10*i_a + day_ix + i_s*.2, shuff[i_s], cols[i_s],     
                    np.array([1.,1.,1.,0.]), ax)
                
                util_fcns.draw_plot(10*i_a + day_ix + i_s*.2, shuff_act[i_s], cols[i_s], 
                    np.array([1.,1.,1.,0.]), ax1)
                    
    ax.set_xlim([-1, 14])
    ax.set_ylabel('R2 of next activity')
    f.tight_layout()

    ax1.set_xlim([-1, 14])
    ax1.set_ylabel('R2 of next command')
    f1.tight_layout()