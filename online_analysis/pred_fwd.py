
from generalization_plots import DataExtract
import analysis_config
import util_fcns
import plot_fr_diffs, generate_models, generate_models_utils
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import pickle
from online_analysis import plot_actions

from collections import defaultdict
import seaborn
seaborn.set(font='Arial',context='talk',font_scale=1.5, style='white')

def plot_R2_model(model_nm = 'hist_1pos_0psh_0spksm_1_spksp_0', model_set_number = 6,
    nshuffs = 20, plot_action = False, nshuffs_roll = 100, keep_bin_spk_zsc = False):

    mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))

    f, ax = plt.subplots(figsize=(3, 3))
    col = dict()
    col[True] = 'red'
    col[False] = analysis_config.blue_rgb; 

    for i_a, animal in enumerate(['home']):#grom', 'jeev']):

        pooled = dict(r2 = [], r2_shuff = [], r2_shuff_roll=[])

        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):

            ### Load data ###
            dataObj = DataExtract(animal, day_ix, model_nm = model_nm, 
                model_set_number = model_set_number, nshuffs=nshuffs, nshuffs_roll=nshuffs_roll,
                keep_bin_spk_zsc = keep_bin_spk_zsc)
            dataObj.load()

            if nshuffs_roll > 0:
                dataObj.load_null_roll_pot_shuff()
    
            valid_ix = dataObj.valid_analysis_ix
            print('ix valid: %d' %(len(valid_ix)))

            ### Get analysis for the push --> eliminate transitions to and from command bin 4; 
            if nshuffs_roll > 0:
                com_rolled = dataObj.rolled_push_comm_bins
                com_rolled_tm1 = dataObj.rolled_push_comm_bins_tm1

                ### Valid ix 
                valid_ix_rolled = np.nonzero(np.logical_and(com_rolled[:, 0] < 4, com_rolled_tm1[:, 0] < 4))[0]
                print('ix valid: %d' %(len(valid_ix_rolled)))
            
                ### Intersecton of these: 
                valid_ix_rolled = np.array([i for i in valid_ix_rolled if i in valid_ix])
                print('ix valid: %d' %(len(valid_ix_rolled)))
            

            if plot_action:
                if animal == 'home':
                    KG, _, _ = util_fcns.get_decoder(animal, day_ix)
                else:
                    KG = util_fcns.get_decoder(animal, day_ix)

                r2_true = util_fcns.get_R2(np.dot(KG, dataObj.spks[valid_ix, :].T).T, 
                    np.dot(KG, dataObj.pred_spks[valid_ix, :].T).T)
                
                r2_shuff = []
                for n in range(nshuffs):
                    r2_shuff.append(util_fcns.get_R2(np.dot(KG, dataObj.spks[valid_ix, :].T).T, 
                        np.dot(KG, dataObj.pred_spks_shuffle[valid_ix, :, n].T).T))

                if nshuffs_roll > 0:
                    r2_null_roll = []
                    true_command_bins = util_fcns.commands2bins([dataObj.push[np.ix_(valid_ix, [3, 5])]], mag_boundaries, animal, day_ix, 
                        vel_ix = [0, 1], ndiv=8)[0]

                    for n in range(nshuffs_roll):
                        shuff_command_bins = util_fcns.commands2bins([0.1*np.dot(KG, dataObj.null_roll_pot_beh_true[valid_ix, :, n].T).T],
                            mag_boundaries, animal, day_ix, vel_ix = [0, 1], ndiv=8)[0]

                        if animal == 'grom':
                            assert(np.allclose(true_command_bins, shuff_command_bins))

                        r2_null_roll.append(util_fcns.get_R2(np.dot(KG, dataObj.null_roll_pot_beh_true[valid_ix_rolled, :, n].T).T, 
                            np.dot(KG, dataObj.null_roll_pot_beh_pred[valid_ix_rolled, :, n].T).T))
            else:
                ### get predictions ###
                r2_true = util_fcns.get_R2(dataObj.spks[valid_ix, :], dataObj.pred_spks[valid_ix, :])

                r2_shuff = []
                for n in range(nshuffs):
                    r2_shuff.append(util_fcns.get_R2(dataObj.spks[valid_ix, :], 
                        dataObj.pred_spks_shuffle[valid_ix, :, n]))

                if nshuffs_roll > 0:
                    r2_null_roll = []
                    for n in range(nshuffs_roll):
                        r2_null_roll.append(util_fcns.get_R2(dataObj.null_roll_pot_beh_true[valid_ix_rolled, :, n], 
                            dataObj.null_roll_pot_beh_pred[valid_ix_rolled, :, n]))

            ### Plot the r2 ###
            xpos = i_a*10 + day_ix

            ax.plot([xpos, xpos], [np.mean(r2_shuff), r2_true], 'k-', linewidth=.5)
            ax.plot(xpos, r2_true, '.', markersize=15, color=col[plot_action])
            #print('r2 act %s %s %d = %.3f' %(str(plot_action), animal, day_ix, r2_true))

            beige = np.array([196, 154, 108])/255.
            util_fcns.draw_plot(xpos, r2_shuff, 'k', np.array([1., 1., 1., 0.]), ax)
        
            if nshuffs_roll > 0:
                util_fcns.draw_plot(xpos, r2_null_roll, 'deeppink', np.array([1., 1., 1., 0.]), ax)
                #print('pink shuffle mean r2 act %s %s %d = %.3f' %(str(plot_action), animal, day_ix, np.mean(r2_null_roll)))

                _, pv = scipy.stats.ks_2samp(r2_shuff, r2_null_roll)
                print('KS test distributions: mn shuff %.3f, mn roll %.3f, pv ks test: %.5f' %(np.mean(r2_shuff), 
                    np.mean(r2_null_roll), pv))

                pooled['r2_shuff_roll'].append(r2_null_roll)

                for i_r, (r2shuffi, shuffnm) in enumerate(zip([r2_shuff, r2_null_roll], ['std','roll'])):
                    ix = np.nonzero(np.hstack((r2shuffi)) >= r2_true)[0]
                    pv = float(len(ix))/float(len(r2shuffi))
                    print('%s, %d: shuffled: %s, pv = %.5f, r2 = %.3f, shuff=[%.3f,%3f]' %(animal, 
                        day_ix, shuffnm, pv, r2_true, np.mean(r2shuffi), np.percentile(r2shuffi, 95)))
            
            pooled['r2'].append(r2_true)
            pooled['r2_shuff'].append(r2_shuff)               

        #### Pooled ####
        for i_r, (key, shuffnm, nsh) in enumerate(zip(['r2_shuff', 'r2_shuff_roll'], ['std','roll'], [nshuffs, nshuffs_roll])):
            
            mean_r2 = np.mean(pooled['r2'])

            if shuffnm == 'roll' and nshuffs_roll > 0:
                mean_shuff = np.mean(np.vstack((pooled[key])), axis=0)
                assert(len(mean_shuff) == nsh)

                ix = np.nonzero(np.hstack((mean_shuff)) >= mean_r2)[0]
                pv = float(len(ix))/float(len(mean_shuff))
                print('POOLED %s: shuffled: %s, pv = %.5f, r2 = %.3f, shuff=[%.3f,%3f]' %(animal, 
                    shuffnm, pv, mean_r2, np.mean(mean_shuff), np.percentile(mean_shuff, 95)))
        

    ax.set_xlim([-1, 14])
    f.tight_layout()
    #util_fcns.savefig(f, 'fwd_pred_action%s'%(str(plot_action)))

def plot_R2_model_fig4_mn_maint_shuff(model_nm = 'hist_1pos_0psh_2spksm_1_spksp_0', model_set_number = 6,
    nshuffs = 10):

    mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))

    f, ax = plt.subplots(figsize=(3, 3))

    for i_a, animal in enumerate(['grom', 'jeev']):

        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):

            ### Load data ###
            dataObj = DataExtract(animal, day_ix, model_nm = model_nm, 
                model_set_number = model_set_number, nshuffs=nshuffs)
            dataObj.load()
            dataObj.load_mn_maint()
            dataObj.load_win_mov_shuff() #self.within_mov_shuff = 10*pred; load_win_mov_shuff

            #import pdb; pdb.set_trace()
            #self.mn_maint_shuff = 10*pred; load_mn_maint

        
            ### get predictions ###
            r2_true = util_fcns.get_R2(dataObj.spks, dataObj.pred_spks)

            r2_shuff = []
            for n in range(nshuffs):
                r2_shuff.append(util_fcns.get_R2(dataObj.spks, dataObj.pred_spks_shuffle[:, :, n]))

            r2_mn_maint = []
            for n in range(nshuffs):
                r2_mn_maint.append(util_fcns.get_R2(dataObj.spks, dataObj.mn_maint_shuff[:, :, n]))

            r2_within_mov = []
            for n in range(nshuffs):
                r2_within_mov.append(util_fcns.get_R2(dataObj.spks, dataObj.within_mov_shuff[:, :, n]))

            ### Plot the r2 ###
            xpos = i_a*10 + day_ix
            ax.plot(xpos, r2_true, '.', markersize=15, color=analysis_config.blue_rgb)
            
            util_fcns.draw_plot(xpos, r2_shuff, 'k', np.array([1., 1., 1., 0.]), ax)
            util_fcns.draw_plot(xpos, r2_mn_maint, 'green', np.array([1., 1., 1., 0.]), ax)
            util_fcns.draw_plot(xpos, r2_within_mov, 'purple', np.array([1., 1.,1., 0.]), ax)

            ix = np.nonzero(np.hstack((r2_mn_maint)) >= r2_true)[0]
            pv = float(len(ix))/float(len(r2_mn_maint))
            print('%s, %d, pv = %.5f' %(animal, day_ix, pv))
            ax.plot([xpos, xpos], [np.mean(r2_shuff), r2_true], 'k-', linewidth=.5)

    ax.set_xlim([-1, 14])
    f.tight_layout()
    util_fcns.savefig(f, 'mean_maint_r2')

def frac_next_com_mov_sig(model_nm = 'hist_1pos_0psh_0spksm_1_spksp_0', model_set_number = 6,
    nshuffs = 1000):

    fcc, axcc = plt.subplots(figsize=(2, 3))
    fcom, axcom = plt.subplots(figsize=(2, 3))
    fccw, axccw= plt.subplots()
    fccw2, axccw2= plt.subplots(figsize=(2, 3))

    for i_a, animal in enumerate(['grom', 'jeev']):
        frac_sig_animal_cc = []
        frac_sig_animal_com = []
        frac_sig_dir = []

        pooled_data = dict(err = [], shuff_err = [])

        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):

            day_data = dict(err = [], shuff_err = [])

            ### Load data ###
            dataObj = DataExtract(animal, day_ix, model_nm = model_nm, 
                model_set_number = model_set_number, nshuffs=nshuffs)
            dataObj.load()
            KG = util_fcns.get_decoder(animal, day_ix)

            sig_mc = 0; 
            all_mc = 0; 

            sig_com = 0; 
            all_com = 0; 

            cw_ccw_corr = 0; 
            cw_ccw_all =0

            shuff_cw_ccw = {}
            for n in range(nshuffs):
                shuff_cw_ccw[n] = [0, 0]

            ### For each move / command is pred closer than shuffle 
            for mag in range(4):
                for ang in range(8): 

                    com_data = dict(err = [], shuff_err = [])
                    assess_command = False
                    
                    for mov in np.unique(dataObj.move):
                        ix = np.nonzero((dataObj.command_bins_tm1[:, 0] == mag) & (dataObj.command_bins_tm1[:, 1] == ang) & (dataObj.move_tm1 == mov))[0]

                        if len(ix) >= 15: 
                            assess_command = True

                            #### Get true next MC; 
                            true_next_action = np.mean(np.dot(KG, dataObj.spks[ix, :].T).T, axis=0)
                            pred_next_action = np.mean(np.dot(KG, dataObj.pred_spks[ix, :].T).T, axis=0)

                            pred_dist = np.linalg.norm(pred_next_action - true_next_action)
                            shuff_next_action_dist = []
                            for i in range(nshuffs): 
                                tmp_act = np.mean(np.dot(KG, dataObj.pred_spks_shuffle[ix, :, i].T).T, axis=0)
                                shuff_next_action_dist.append(np.linalg.norm(tmp_act - true_next_action))

                            ### Keep data; 
                            day_data['err'].append(pred_dist)
                            day_data['shuff_err'].append(np.hstack((shuff_next_action_dist)))

                            com_data['err'].append(pred_dist)
                            com_data['shuff_err'].append(np.hstack((shuff_next_action_dist)))

                            p = np.nonzero(shuff_next_action_dist <= pred_dist)[0]
                            pv = float(len(p))/float(nshuffs)

                            if pv < 0.05:
                                sig_mc += 1
                            all_mc += 1


                            ######## CW/CCW assessment for significant command/conditions ############
                            if pv < 0.05: 
                                current_command = np.mean(np.dot(KG, dataObj.spks_tm1[ix, :].T).T, axis=0)
                                cp_true = np.sign(np.cross(current_command, true_next_action))
                                cp_pred = np.sign(np.cross(current_command, pred_next_action))

                                if cp_true == cp_pred: 
                                    cw_ccw_corr += 1
                                cw_ccw_all += 1

                                for i in range(nshuffs): 
                                    tmp_act = np.mean(np.dot(KG, dataObj.pred_spks_shuffle[ix, :, i].T).T, axis=0)
                                    cp_pred = np.sign(np.cross(current_command, tmp_act))
                                    if cp_true == cp_pred: 
                                        shuff_cw_ccw[i][0] += 1
                                    shuff_cw_ccw[i][1] += 1

                    if assess_command: 
                        #### test if command is sig; 
                        mn_dist = np.mean(com_data['err'])
                        mn_shuff = np.mean(np.vstack((com_data['shuff_err'])), axis=0)
                        assert(len(mn_shuff) == nshuffs)
                        p = np.nonzero(mn_shuff <= mn_dist)[0]
                        pv = float(len(p))/float(nshuffs)
                        if pv < 0.05: 
                            sig_com += 1
                        all_com += 1

            ##### Sig comm-doncs 
            frac_sig = float(sig_mc)/float(all_mc)
            frac_sig_animal_cc.append(frac_sig)
            axcc.plot(i_a, frac_sig, 'k.')

            frac_sig = float(sig_com)/float(all_com)
            frac_sig_animal_com.append(frac_sig)
            axcom.plot(i_a, frac_sig, 'k.')

            frac_corr = float(cw_ccw_corr)/float(cw_ccw_all)
            shuff_frac_corr = []
            for i in range(nshuffs):
                shuff_frac_corr.append(float(shuff_cw_ccw[i][0])/float(shuff_cw_ccw[i][1]))
            
            axccw.plot(i_a*10 + day_ix, frac_corr, 'r.')
            util_fcns.draw_plot(i_a*10 + day_ix, shuff_frac_corr, 'k', np.array([1., 1., 1., 0.]), axccw)

            axccw2.plot(i_a, frac_corr, 'k.')
            frac_sig_dir.append(frac_corr)

            ### stats pooled 
            mn_err = np.mean(day_data['err'])
            mn_shuf = np.mean(np.vstack((day_data['shuff_err'])), axis=0)
            assert(len(mn_shuf) == nshuffs)
            ix = np.nonzero(mn_shuf <= mn_err)[0]
            pv = float(len(ix))/float(len(mn_shuf))
            print('Animal %s, Day %d, pv = %.5f, mn_err = %.3f, mn_shuf = [%.3f, %.3f]' %(animal, 
                day_ix, pv, mn_err, np.mean(mn_shuf), np.percentile(mn_shuf, 5)))

            pooled_data['err'].append(mn_err)
            pooled_data['shuff_err'].append(mn_shuf)

        axcc.bar(i_a, np.mean(frac_sig_animal_cc), width=0.8, color='k', alpha=0.2)
        axcom.bar(i_a, np.mean(frac_sig_animal_com), width=0.8, color='k', alpha=0.2)
        axccw2.bar(i_a, np.mean(frac_sig_dir), width=0.8, color='k', alpha=0.2)

        mn_err = np.mean(pooled_data['err'])
        mn_shuf = np.mean(np.vstack((pooled_data['shuff_err'])), axis=0)
        assert(len(mn_shuf) == nshuffs)
        ix = np.nonzero(mn_shuf <= mn_err)[0]
        pv = float(len(ix))/float(len(mn_shuf))
        print('Animal %s, POOLED: pv = %.5f, mn_err = %.3f, mn_shuf = [%.3f, %.3f]' %(animal, 
            pv, mn_err, np.mean(mn_shuf), np.percentile(mn_shuf, 5)))

    for _, (ax, f, ylab, lab) in enumerate(zip([axcc, axcom, axccw2], [fcc, fcom, fccw2], 
        ['Frac (command, condition)\n with sig. pred. next command','Frac (command) with\nsig. pred. next command', ' (Com-cond) w sig. pred. next command\nfrac. corr. dir'], 
        ['com_cond', 'com', 'sig_com_cond_frac_corr_dir'])): 
        ax.set_xticks([0, 1])
        ax.set_ylim([0., 1.05])
        ax.set_xticklabels(['G', 'J'])
        ax.set_yticks([0., .25, .50, .75, 1.0])
        ax.set_yticklabels([0., .25, .50, .75, 1.0])
        ax.set_ylabel(ylab, fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        f.tight_layout()
        util_fcns.savefig(f, 'frac_%s_w_sig_next_comm_pred'%lab)

    axccw.set_xlim([-1, 14])
    axccw.set_ylabel('Frac. sig. command-movs predicting \nnext command in correct direction ')
    fccw.tight_layout()

def pred_vs_true_next_command(model_nm = 'hist_1pos_0psh_0spksm_1_spksp_0', model_set_number = 6, nshuffs = 2,
    mag_eg=0, ang_eg=7):

    f_cc, ax_cc = plt.subplots(figsize=(2, 3))
    f_er, ax_er = plt.subplots(figsize=(2, 3))

    f_pw, ax_pw = plt.subplots(figsize=(4, 5))
    ax_pw2 = ax_pw.twinx()
    PW_eg = []; PW_shuff = []; PW_shuff_rol = []

    special_dots = []

    for i_a, animal in enumerate(['grom']):#, 'jeev']):
        cc_avg = []
        animal_data = []

        for day_ix in range(1):#analysis_config.data_params['%s_ndays'%animal]):


            if animal in ['grom'] and day_ix in range(1):
                f_scat, ax_scatter = plt.subplots(figsize=(5, 5))
                f_scat_shuff, ax_scatter_shuff = plt.subplots(figsize=(3, 3))
                for axi in [ax_scatter, ax_scatter_shuff]:
                    axi.set_ylabel('Subj %s Day %d' %(animal, day_ix))


            ### Load data ###
            dataObj = DataExtract(animal, day_ix, model_nm = model_nm, 
                model_set_number = model_set_number, nshuffs=nshuffs)
            dataObj.load()
            #dataObj.load_null_roll_pot_shuff()

            KG = util_fcns.get_decoder(animal, day_ix)

            #### multipy by 0.1 so that pred_push is correct magnitude
            pred_push = np.dot(KG, 0.1*dataObj.pred_spks.T).T
            X = [];

            # X_shuff = {}; 
            # pred_push_shuff = {}; 
            # pred_push_shuff_roll = {}
            # for i in range(nshuffs):
            #     X_shuff[i] = []
            #     pred_push_shuff[i] = np.dot(KG, 0.1*dataObj.pred_spks_shuffle[:, :, i].T).T
            #     pred_push_shuff_roll[i] = np.dot(KG, 0.1*dataObj.null_roll_pot_beh_pred[:, :, i].T).T
            #     assert(not np.allclose(pred_push_shuff[i], pred_push_shuff_roll[i]))

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
                    movs = np.sort(np.array(mov_ix.keys()))
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
                                animal_data.append([true_pw_dist, pred_pw_dist])

                                # import pdb; pdb.set_trace()

                                # if np.isnan(true_pw_dist):
                                #     import pdb; pdb.set_trace()
                                # if np.isnan(pred_pw_dist):
                                #     import pdb; pdb.set_trace()
                                # tmp = []; tmp2 = []; 
                                # for i in range(nshuffs):
                                #     shuf_mn1 = np.mean(pred_push_shuff[i][ix1[ix11], :], axis=0) 
                                #     shuf_mn2 = np.mean(pred_push_shuff[i][ix2[ix22], :], axis=0) 
                                #     pred_pw_dist_shuff = np.linalg.norm(shuf_mn1 - shuf_mn2)
                                    
                                #     shuf_rol1 = np.mean(pred_push_shuff_roll[i][ix1[ix11], :], axis=0)
                                #     shuf_rol2 = np.mean(pred_push_shuff_roll[i][ix2[ix22], :], axis=0)
                                #     pred_pw_dist_shuff_rol = np.linalg.norm(shuf_rol1 - shuf_rol2)

                                #     X_shuff[i].append([true_pw_dist, pred_pw_dist_shuff, pred_pw_dist_shuff_rol])
                                #     tmp.append(pred_pw_dist_shuff)
                                #     tmp2.append(pred_pw_dist_shuff_rol)

                                #     if animal in ['grom', 'jeev'] and day_ix in range(10) and i == 0:
                                #         ax_scatter_shuff.plot(true_pw_dist, pred_pw_dist_shuff, 'k.', markersize=5.)
                                #         ax_scatter_shuff.plot(true_pw_dist, pred_pw_dist_shuff_rol, '.', color='deeppink', markersize=5.)

                                if animal in ['grom'] and day_ix in range(1):
                                    ax_scatter.plot(true_pw_dist, pred_pw_dist, 'k.', markersize=5.)

                                    if mag == 0 and ang == 7:
                                        if mov1 == 1. and mov2 == 3.:
                                            ax_scatter.plot(true_pw_dist, pred_pw_dist, '.', color = 'deeppink',
                                                markersize=7.)

                                        elif mov1 == 1. and mov2 == 10.1:
                                            ax_scatter.plot(true_pw_dist, pred_pw_dist, '.', color = 'limegreen',
                                                markersize=7.)

                                    if animal == 'grom' and day_ix == 0 and mag == mag_eg and ang == ang_eg:
                                        PW_eg.append([mov1, mov2, true_pw_dist, pred_pw_dist])
                                        #PW_shuff.append(tmp)
                                        #PW_shuff_rol.append(tmp2)


            X = np.vstack((X))
            _,_,rv,pv,_ = scipy.stats.linregress(X[:, 0], X[:, 1])
            print('%s, %d, pv = %.5f' %(animal, day_ix, pv))
            ax_cc.plot(i_a, rv, '.', color='k')
            cc_avg.append(rv)

            ### Compute error 
            #err = np.mean(np.abs(X[:, 0] - X[:, 1]))
            #ax_er.plot(10*i_a + day_ix, err, '.', color='maroon', markersize=15)

            if animal in ['grom'] and day_ix in range(1):
                ax_scatter.set_title(' r %.3f' %(rv), fontsize=10)
                ax_scatter.set_xlabel('True Command Diffs.')
                ax_scatter.set_ylabel('Pred Command Diffs.')
            # # rv_shuff = []; er_shuff = []; rv_shuff_rol = []; er_shuff_rol = []; 

            # # for i in range(nshuffs):
            # #     X_shuff[i] = np.vstack((X_shuff[i])) 
            # #     _,_,rv_shf,_,_ = scipy.stats.linregress(X_shuff[i][:, 0], X_shuff[i][:, 1])
            # #     rv_shuff.append(rv_shf)
            # #     er_shuff.append(np.mean(np.abs(X_shuff[i][:, 0] - X_shuff[i][:, 1])))

            # #     _,_,rv_shf_rol,_,_ = scipy.stats.linregress(X_shuff[i][:, 0], X_shuff[i][:, 2])
            # #     rv_shuff_rol.append(rv_shf_rol)
            # #     er_shuff_rol.append(np.mean(np.abs(X_shuff[i][:, 0] - X_shuff[i][:, 2])))

            #     if animal in ['grom', 'jeev'] and day_ix in range(10) and i == 0: 
            #         ax_scatter_shuff.set_title('r shuff %.3f, rol %.3f' %(rv_shf, rv_shf_rol), fontsize=10)

            # util_fcns.draw_plot(10*i_a + day_ix, rv_shuff, 'k', np.array([1.,1., 1., 0.]), ax_cc)
            # util_fcns.draw_plot(10*i_a + day_ix, rv_shuff_rol, 'deeppink', np.array([1.,1., 1., 0.]), ax_cc)
            # ax_cc.plot([10*i_a + day_ix, 10*i_a + day_ix], [np.mean(rv_shuff), rv], 'k-', linewidth=0.5)

            # util_fcns.draw_plot(10*i_a + day_ix, er_shuff, 'k', np.array([1.,1., 1., 0.]), ax_er)
            # util_fcns.draw_plot(10*i_a + day_ix, er_shuff_rol, 'deeppink', np.array([1.,1., 1., 0.]), ax_er)
            # ax_er.plot([10*i_a + day_ix, 10*i_a + day_ix], [np.mean(er_shuff), err], 'k-', linewidth=0.5)
            

            ax_scatter.set_xlim([0., 2.])
            ax_scatter.set_ylim([0., 1.])
            f_scat.tight_layout()
    
            # ax_scatter_shuff.set_xlim([0., 2.])
            # ax_scatter_shuff.set_ylim([0., 1.])
            # f_scat_shuff.tight_layout()

        cc_avg_mn = np.mean(cc_avg)
        ax_cc.bar(i_a, cc_avg_mn, color='k', alpha=0.5)
    
        animal_data = np.vstack((animal_data))
        slp,intc,rv,pv,_ = scipy.stats.linregress(animal_data[:, 0], animal_data[:, 1])
        print('POOLED STATS %s, pv = %.5f, rv = %.2f, slp=%.3f'%(
            animal, pv, rv, slp))
            

    #ax_cc.set_xlim([-1, 14])
    ax_cc.set_xticks([0, 1])
    ax_cc.set_xticklabels(['G', 'J'])
    f_cc.tight_layout()

    # ax_er.set_xlim([-1, 14])
    # ax_er.set_xticks([])
    # f_er.tight_layout()

    util_fcns.savefig(f_scat, 'scatter_action_dist_grom_0')
    #util_fcns.savefig(f_scat_shuff, 'scatter_shuff_action_dist_grom_0')
    util_fcns.savefig(f_cc, 'true_v_pred_next_act_cc')
    # util_fcns.savefig(f_er, 'true_v_pred_next_act_err')

    PW_eg = np.vstack((PW_eg))
    ix_sort = np.argsort(PW_eg[:, 2])[::-1]

    for ii, i_s in enumerate(ix_sort):
        ax_pw.plot(ii, PW_eg[i_s, 2], '.', color='k')
        ax_pw2.plot(ii, PW_eg[i_s, 3], '.', color='r')
        ax_pw.plot(ii, 0., '.', markersize=12, color=util_fcns.get_color(PW_eg[i_s, 0]))
        ax_pw.plot(ii, -0.1, '.', markersize=12, color=util_fcns.get_color(PW_eg[i_s, 1]))

    #     util_fcns.draw_plot(ii, PW_shuff[i_s], 'k', np.array([1., 1., 1., 0.]), ax_pw2)
    #     util_fcns.draw_plot(ii, PW_shuff_rol[i_s], 'deeppink', np.array([1.,1.,1.,0.]), ax_pw2)

    ax_pw.set_xlim([-1, 36])
    ax_pw2.set_xlim([-1, 36])
    
    ax_pw.set_ylim([-.12, 1.35])
    ax_pw2.set_ylim([-.05, .55])
    ax_pw2.tick_params(axis='y', labelcolor='r')
    ax_pw.set_xticks([])
    ax_pw2.set_xticks([])
    f_pw.tight_layout()
    for xia, axi in enumerate([ax_pw, ax_pw2]):
        yl = axi.get_yticks()
        axi.set_yticklabels(np.round(yl, 1), rotation=90)

    ax_pw.spines['top'].set_visible(False)
    ax_pw2.spines['top'].set_visible(False)
    
    ax_pw.set_ylabel('Pairwise Next Command Diff.')
    ax_pw2.set_ylabel('Pred. Pairwise Next Command Diff.', rotation=90, color='r')
    ax_pw.set_xlabel('Movement Pairs', rotation=180)
    
    util_fcns.savefig(f_pw, 'pw_plot_next_action')

def perc_corr_pred_next_command_mov_com2com(model_nm = 'hist_1pos_0psh_0spksm_1_spksp_0', model_set_number = 6, nshuffs = 2, 
    min_obs = 15, min_obs_next_command = 8, only_pairwise_com = False, exclude_same_dir = False):

    perc_corr = {}
    shuf_corr = {}
    plot_RED = np.array([237, 28, 36])/256.
    
    fang, axang = plt.subplots()
    fmag, axmag = plt.subplots()


    for i_a, animal in enumerate(['grom', 'jeev']):
        bar = dict(pc_ang = [], pc_mag = [])

        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):
            perc_corr[animal, day_ix] = dict(corr=[], incorr=[], corr_mag=[], incorr_mag=[])
            
            for n in range(nshuffs):
                shuf_corr[animal, day_ix, n] = dict(corr=[], incorr=[], corr_mag=[], incorr_mag=[])

            ### Load data ###
            dataObj = DataExtract(animal, day_ix, model_nm = model_nm, 
                model_set_number = model_set_number, nshuffs=nshuffs)
            dataObj.load()

            KG = util_fcns.get_decoder(animal, day_ix)

            #### multipy by 0.1 so that pred_push is correct magnitude
            pred_push = np.dot(KG, 0.1*dataObj.pred_spks.T).T
            
            ### For each move / command is pred closer than shuffle 
            for mag in range(4):

                for ang in range(8): 

                    ##### Get all the relevant commands 
                    if only_pairwise_com:
                        mv = [np.nan]
                    else:
                        mv =  np.unique(dataObj.move)
                    for mov in mv:

                        if only_pairwise_com:
                            ix = np.nonzero((dataObj.command_bins_tm1[:, 0] == mag) & (dataObj.command_bins_tm1[:, 1] == ang))[0]
                        else:
                            ix = np.nonzero((dataObj.command_bins_tm1[:, 0] == mag) & (dataObj.command_bins_tm1[:, 1] == ang) & (dataObj.move_tm1 == mov))[0]
                        
                        if len(ix) >= 15: 

                            ##### What command does this movement transiton to? #######
                            if exclude_same_dir:
                                ang_ok = np.array([x for x in range(8) if x!=ang])
                            else:
                                ang_ok = np.arange(8)
                            
                            for mag2 in range(4):
                                for ang2 in ang_ok: 
                                    for mov2 in mv: 
                                        if only_pairwise_com:
                                            ix2 = np.nonzero((dataObj.command_bins[ix, 0] == mag2) & (dataObj.command_bins[ix, 1] == ang2))[0]
                                        else:
                                            ix2 = np.nonzero((dataObj.command_bins[ix, 0] == mag2) & (dataObj.command_bins[ix, 1] == ang2) & (dataObj.move[ix] == mov2))[0]
                                        
                                        if len(ix2) >= min_obs_next_command: 
                                            if only_pairwise_com:
                                                pass
                                            else:
                                                assert(mov == mov2)
                                            current_command = np.mean(dataObj.push_tm1[ix[ix2]], axis=0)[[3, 5]]
                                            tru_command = np.mean(dataObj.push    [ix[ix2]], axis=0)[[3, 5]]
                                            pred_command   = np.mean(np.dot(KG, 0.1*dataObj.pred_spks[ix[ix2]].T).T, axis=0)

                                            perc_corr[animal, day_ix] = assign_to_dict(current_command, tru_command, pred_command, perc_corr[animal, day_ix])

                                            ps = []
                                            for n in range(nshuffs):
                                                ps.append(np.mean(np.dot(KG, 0.1*dataObj.pred_spks_shuffle[ix[ix2], :, n].T).T, axis=0))

                                            pred_commands_shuff = np.vstack((ps))
                                            assert(pred_commands_shuff.shape[0] == nshuffs)
                                            assert(pred_commands_shuff.shape[1] == 2)

                                            for n in range(nshuffs):
                                                shuf_corr[animal, day_ix, n] = assign_to_dict(current_command, tru_command, pred_commands_shuff[n, :], 
                                                    shuf_corr[animal, day_ix, n])

            ################################################
            pcang = float(len(perc_corr[animal, day_ix]['corr']))/(float(len(perc_corr[animal, day_ix]['corr'])) + float(len(perc_corr[animal, day_ix]['incorr'])))
            pcmag = float(len(perc_corr[animal, day_ix]['corr_mag']))/(float(len(perc_corr[animal, day_ix]['corr_mag'])) + float(len(perc_corr[animal, day_ix]['incorr_mag'])))
            
            axang.plot(i_a*10 + day_ix, pcang, '.', color=plot_RED)
            axmag.plot(i_a*10 + day_ix, pcmag, '.', color=plot_RED)


            pcang_shuff = []
            pcmag_shuff = []
            for n in range(nshuffs):
                pcang_shuff.append(float(len(shuf_corr[animal, day_ix, n]['corr']))/(float(len(shuf_corr[animal, day_ix, n]['corr'])) + float(len(shuf_corr[animal, day_ix, n]['incorr']))))
                pcmag_shuff.append(float(len(shuf_corr[animal, day_ix, n]['corr_mag']))/(float(len(shuf_corr[animal, day_ix, n]['corr_mag'])) + float(len(shuf_corr[animal, day_ix, n]['incorr_mag']))))

            util_fcns.draw_plot(i_a*10 + day_ix, pcang_shuff, 'k', np.array([1., 1., 1., 0.]), axang)
            util_fcns.draw_plot(i_a*10 + day_ix, pcmag_shuff, 'k', np.array([1., 1., 1., 0.]), axmag)

            ############################################# Scatters 
            fangscat, axangscat = plt.subplots()
            fmagscat, axmagscat = plt.subplots()

            ang_corr = np.vstack(( perc_corr[animal, day_ix]['corr'] ))
            ang_incorr = np.vstack(( perc_corr[animal, day_ix]['incorr'] ))
            axangscat.plot(ang_corr[:, 0], ang_corr[:, 1], 'b.', markersize=2.)
            axangscat.plot(ang_incorr[:, 0], ang_incorr[:, 1], 'r.', markersize=2.)
            x = np.vstack((ang_corr, ang_incorr))
            slp,intc,rv,pv,_ = scipy.stats.linregress(x[:, 0], x[:, 1])
            axangscat.plot(x[:, 0], slp*x[:, 0] + intc, 'b-')

            mag_corr = np.vstack(( perc_corr[animal, day_ix]['corr_mag'] ))
            mag_incorr = np.vstack(( perc_corr[animal, day_ix]['incorr_mag'] ))
            axmagscat.plot(mag_corr[:, 0], mag_corr[:, 1], 'b.', markersize=2.)
            axmagscat.plot(mag_incorr[:, 0], mag_incorr[:, 1], 'r.', markersize=2.)
            x = np.vstack((mag_corr, mag_incorr))
            slp,intc,rv,pv,_ = scipy.stats.linregress(x[:, 0], x[:, 1])
            axmagscat.plot(x[:, 0], slp*x[:, 0] + intc, 'b-')

            ####### hsuffles 
            ang_shuff = np.vstack(( shuf_corr[animal, day_ix, 0]['corr'] ))
            ang_shuff_in = np.vstack(( shuf_corr[animal, day_ix, 0]['incorr'] ))
            axangscat.plot(ang_shuff[:, 0], ang_shuff[:, 1], 'k.', alpha=0.5, markersize=2.)
            axangscat.plot(ang_shuff_in[:, 0], ang_shuff_in[:, 1], 'k.', alpha=0.5, markersize=2.)
            x = np.vstack((ang_shuff, ang_shuff_in))
            slp,intc,rv,pv,_ = scipy.stats.linregress(x[:, 0], x[:, 1])
            axangscat.plot(x[:, 0], slp*x[:, 0] + intc, 'k-')
            
            mag_shuff =  np.vstack(( shuf_corr[animal, day_ix, 0]['corr_mag'] ))
            mag_shuff_in = np.vstack(( shuf_corr[animal, day_ix, 0]['incorr_mag'] ))
            axmagscat.plot(mag_shuff[:, 0], mag_shuff[:, 1], 'k.', alpha=0.5, markersize=2.)
            axmagscat.plot(mag_shuff_in[:, 0], mag_shuff_in[:, 1], 'k.', alpha=0.5, markersize=2.)
            x = np.vstack((mag_shuff, mag_shuff_in))
            slp,intc,rv,pv,_ = scipy.stats.linregress(x[:, 0], x[:, 1])
            axmagscat.plot(x[:, 0], slp*x[:, 0] + intc, 'k-')
            
            axangscat.set_title('ang scat')
            axmagscat.set_title('mag scat')
            fangscat.tight_layout()
            fmagscat.tight_layout()

    axang.set_xlim([-1, 14])
    axmag.set_xlim([-1, 14])
    axang.set_ylabel('Percent correct, Angle dir')
    axmag.set_ylabel('Percent correct, Mag. dir')

    fang.tight_layout()
    fmag.tight_layout()

def perc_corr_ang_pw(model_nm = 'hist_1pos_0psh_0spksm_1_spksp_0', model_set_number = 6, nshuffs = 2, 
    min_obs_next_command = 5):

    perc_corr = {}
    shuf_corr = {}
    plot_RED = np.array([237, 28, 36])/256.
    
    fang, axang = plt.subplots()


    for i_a, animal in enumerate(['grom', 'jeev']):
        bar = dict(pc_ang = [], pc_mag = [])

        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):
            
            pw_ang = dict(corr = 0, N = 0)
            pw_ang_shuff = {}
            for n in range(nshuffs):
                pw_ang_shuff[n] = dict(corr=0, N=0)

            ### Load data ###
            dataObj = DataExtract(animal, day_ix, model_nm = model_nm, 
                model_set_number = model_set_number, nshuffs=nshuffs)
            dataObj.load()

            KG = util_fcns.get_decoder(animal, day_ix)

            ### For each move / command is pred closer than shuffle 
            for mag in range(4):

                for ang in range(8): 

                    ##### Get all the relevant commands 
                    ix = np.nonzero((dataObj.command_bins_tm1[:, 0] == mag) & (dataObj.command_bins_tm1[:, 1] == ang))[0]
                       
                    mag_ang_map = defaultdict(list)
                    mag_ang_map_shuff = {}
                    for n in range(nshuffs):
                        mag_ang_map_shuff[n] = {}

                    for mag2 in range(4):
                        for ang2 in range(8):

                            ### Same mag, not same angle ####
                            ix2 = np.nonzero((dataObj.command_bins[ix, 0] == mag2) & (dataObj.command_bins[ix, 1] == ang2))[0]
                            
                            if len(ix2) >= min_obs_next_command:
                                pred_command   = np.mean(np.dot(KG, 0.1*dataObj.pred_spks[ix[ix2]].T).T, axis=0)
                                mag_ang_map[mag2, ang2] = pred_command

                                for n in range(nshuffs):
                                    mag_ang_map_shuff[n][mag2, ang2] = np.mean(np.dot(KG, 0.1*dataObj.pred_spks_shuffle[ix[ix2], :, n].T).T, axis=0)
                    
                    corr, N = test_pw(mag_ang_map)
                    pw_ang['corr'] += corr; 
                    pw_ang['N'] += N 

                    for n in range(nshuffs):
                        corr, N = test_pw(mag_ang_map_shuff[n])
                        pw_ang_shuff[n]['corr']+= corr; 
                        pw_ang_shuff[n]['N']+= N; 

            ### Plot 
            pc = float(pw_ang['corr']) / float(pw_ang['N'])
            axang.plot(10*i_a + day_ix, pc, 'r.')
            pc_shuff = []
            for n in range(nshuffs):
                pc_shuff.append(float(pw_ang_shuff[n]['corr']) / float(pw_ang_shuff[n]['N']))
            util_fcns.draw_plot(10*i_a + day_ix, pc_shuff, 'k', np.array([1., 1., 1., 0.]), axang)
    
    axang.set_xlim([-1, 14])
    axang.set_ylim([0., 1.])


                                

def test_pw(mag_ang_map):
    angs = np.linspace(0, 2*np.pi, 9)

    corr = 0; N = 0; 
    for mag in range(4): 
        mg = []
        for _, (k, pred) in enumerate(mag_ang_map.items()):
            if k[0] == mag: 
                mg.append([k[1], pred])

        if len(mg) > 1: 
            ### Pairwise ###
            for im in range(len(mg)): 
                ang1 = np.array([np.cos(mg[im][0]), np.sin(mg[im][0])])
                pred1 = mg[im][1]

                for im2 in range(im+1, len(mg)):
                    assert(im!=im2)
                    ang2 = np.array([np.cos(mg[im2][0]), np.sin(mg[im2][0])])
                    pred2 = mg[im2][1]

                    true_cp = np.sign(np.cross(ang1, ang2))
                    act_cp = np.sign(np.cross(pred1, pred2))    

                    if true_cp == act_cp: 
                        corr += 1
                    N += 1
    return corr, N




def assign_to_dict(current_command, tru_command, pred_command, dict_track):

    #### Get differences between true current and true next commadn 
    dA_true, dM_true = plot_actions.get_diffs(tru_command, current_command)
    
    ### Sign the angle; 
    ### As convention, we'll call CW (-) and CCW (+)
    cp = np.sign(np.cross(current_command, tru_command))
    dA_true = cp*dA_true


    #### Get differences between true current and predicted next commadn 
    dA, dM = plot_actions.get_diffs(pred_command, current_command)
    
    ### Sign the angle; 
    ### As convention, we'll call CW (-) and CCW (+)
    cp = np.sign(np.cross(current_command, pred_command))
    dA = cp*dA

    if np.sign(dA) == np.sign(dA_true): 
        dict_track['corr'].append([ dA_true, dA])
    else:
        dict_track['incorr'].append([dA_true, dA])

    if np.sign(dM_true) == np.sign(dM): 
        dict_track['corr_mag'].append([ dM_true, dM])
    else:
        dict_track['incorr_mag'].append([ dM_true, dM])

    return dict_track

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