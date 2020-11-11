
from generalization_plots import DataExtract
import analysis_config
import util_fcns
import plot_fr_diffs
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

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

            if plot_action:
                KG = util_fcns.get_decoder(animal, day_ix)

                r2_true = util_fcns.get_R2(np.dot(KG, dataObj.spks.T).T, 
                    np.dot(KG, dataObj.pred_spks.T).T)
                r2_shuff = []
                for n in range(nshuffs):
                    r2_shuff.append(util_fcns.get_R2(np.dot(KG, dataObj.spks.T).T, 
                        np.dot(KG, dataObj.pred_spks_shuffle[:, :, n].T).T))

            else:
                ### get predictions ###
                r2_true = util_fcns.get_R2(dataObj.spks, dataObj.pred_spks)

                r2_shuff = []
                for n in range(nshuffs):
                    r2_shuff.append(util_fcns.get_R2(dataObj.spks, dataObj.pred_spks_shuffle[:, :, n]))

            ### Plot the r2 ###
            xpos = i_a*10 + day_ix
            ax.plot(xpos, r2_true, '.', markersize=15, color=col[plot_action])
            util_fcns.draw_plot(xpos, r2_shuff, 'k', np.array([1., 1., 1., 0.]), ax)
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

    f_cc, ax_cc = plt.subplots(figsize=(2, 3))
    f_er, ax_er = plt.subplots(figsize=(2, 3))

    f_scat, ax_scatter = plt.subplots(figsize=(3, 3))
    f_scat_shuff, ax_scatter_shuff = plt.subplots(figsize=(3, 3))
    
    f_pw, ax_pw = plt.subplots(figsize=(5, 5))
    ax_pw2 = ax_pw.twinx()
    PW_eg = []; PW_shuff = [] 


    for i_a, animal in enumerate(['grom']):#, 'jeev']):
        for day_ix in range(1):#analysis_config.data_params['%s_ndays'%animal]):

            ### Load data ###
            dataObj = DataExtract(animal, day_ix, model_nm = model_nm, 
                model_set_number = model_set_number, nshuffs=nshuffs)
            dataObj.load()
            KG = util_fcns.get_decoder(animal, day_ix)

            #### multipy by 0.1 so that pred_push is correct magnitude
            pred_push = np.dot(KG, 0.1*dataObj.pred_spks.T).T
            X = [];

            X_shuff = {}; 
            pred_push_shuff = {}; 
            for i in range(nshuffs):
                X_shuff[i] = []
                pred_push_shuff[i] = np.dot(KG, 0.1*dataObj.pred_spks_shuffle[:, :, i].T).T


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
                                tmp = []
                                for i in range(nshuffs):
                                    shuf_mn1 = np.mean(pred_push_shuff[i][ix1[ix11], :], axis=0) 
                                    shuf_mn2 = np.mean(pred_push_shuff[i][ix2[ix22], :], axis=0) 
                                    
                                    pred_pw_dist_shuff = np.linalg.norm(shuf_mn1 - shuf_mn2)
                                    X_shuff[i].append([true_pw_dist, pred_pw_dist_shuff])
                                    tmp.append(pred_pw_dist_shuff)

                                    if animal == 'grom' and day_ix == 0 and i == 0:
                                        ax_scatter_shuff.plot(true_pw_dist, pred_pw_dist_shuff, 'k.', markersize=5.)

                                if animal == 'grom' and day_ix == 0:
                                    ax_scatter.plot(true_pw_dist, pred_pw_dist, 'k.', markersize=5.)

                                    if mag == mag_eg and ang == ang_eg:
                                        PW_eg.append([mov1, mov2, true_pw_dist, pred_pw_dist])
                                        PW_shuff.append(tmp)


            X = np.vstack((X))
            _,_,rv,_,_ = scipy.stats.linregress(X[:, 0], X[:, 1])
            ax_cc.plot(10*i_a + day_ix, rv, '.', color='maroon', markersize=15)

            ### Compute error 
            err = np.mean(np.abs(X[:, 0] - X[:, 1]))
            ax_er.plot(10*i_a + day_ix, err, '.', color='maroon', markersize=15)

            if animal == 'grom' and day_ix == 0:
                ax_scatter.set_title(' r %.3f' %(rv), fontsize=10)
            rv_shuff = []; er_shuff = []; 
            for i in range(nshuffs):
                X_shuff[i] = np.vstack((X_shuff[i])) 
                _,_,rv_shf,_,_ = scipy.stats.linregress(X_shuff[i][:, 0], X_shuff[i][:, 1])
                rv_shuff.append(rv_shf)
                er_shuff.append(np.mean(np.abs(X_shuff[i][:, 0] - X_shuff[i][:, 1])))

                if animal == 'grom' and day_ix == 0 and i == 0: 
                    ax_scatter_shuff.set_title('r shuff %.3f' %(rv_shf), fontsize=10)

            util_fcns.draw_plot(10*i_a + day_ix, rv_shuff, 'k', np.array([1.,1., 1., 0.]), ax_cc)
            ax_cc.plot([10*i_a + day_ix, 10*i_a + day_ix], [np.mean(rv_shuff), rv], 'k-', linewidth=0.5)

            util_fcns.draw_plot(10*i_a + day_ix, er_shuff, 'k', np.array([1.,1., 1., 0.]), ax_er)
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

    util_fcns.savefig(f_scat, 'scatter_action_dist_grom_0')
    util_fcns.savefig(f_scat_shuff, 'scatter_shuff_action_dist_grom_0')
    util_fcns.savefig(f_cc, 'true_v_pred_next_act_cc')
    util_fcns.savefig(f_er, 'true_v_pred_next_act_err')

    PW_eg = np.vstack((PW_eg))
    ix_sort = np.argsort(PW_eg[:, 2])[::-1]

    for ii, i_s in enumerate(ix_sort):
        ax_pw.plot(ii, PW_eg[i_s, 2], 'k.')
        ax_pw2.plot(ii, PW_eg[i_s, 3], '.', color='maroon')
        util_fcns.draw_plot(ii, PW_shuff[i_s], 'gray', np.array([1., 1., 1., 0.]), ax_pw2)

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

